"""Render the reimported STEP and exact strap sections without geometry smoothing."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from build123d import Plane, import_step, section
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_OUT

from deep_frame.topology_geometry import region_shape
from deep_frame.topology_surface_validation import _mesh, _settings


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def face_colors(normals, elevation, azimuth):
    """Flat Lambert/Blinn shading of existing triangles, without vertex averaging."""
    elev, azim = np.deg2rad([elevation, azimuth])
    view = np.asarray([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    light = view + np.asarray([-0.4, -0.3, 0.5 if elevation >= 0 else -0.5])
    light /= np.linalg.norm(light)
    fill = np.asarray([-0.5, 0.7, 0.25 if elevation >= 0 else -0.25])
    fill /= np.linalg.norm(fill)
    halfway = light + view
    halfway /= np.linalg.norm(halfway)
    intensity = 0.27 + 0.62 * np.maximum(normals @ light, 0) + 0.14 * np.maximum(normals @ fill, 0)
    highlight = 0.20 * np.maximum(normals @ halfway, 0) ** 24
    base = np.asarray([0.24, 0.62, 0.76])
    return np.clip(intensity[:, None] * base + highlight[:, None], 0, 1)


def draw_view(axis, mesh, lower, upper, elevation, azimuth, title):
    axis.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=face_colors(mesh.face_normals, elevation, azimuth),
                                         edgecolors="none", linewidths=0, antialiased=False, zsort="average"))
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
             xlabel="x / mm", ylabel="y / mm", zlabel="z / mm", title=title)
    axis.set_box_aspect(upper - lower, zoom=0.9)
    axis.set_proj_type("ortho")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xticks(np.arange(-60, 61, 20))
    axis.set_yticks(np.arange(-60, 61, 20))
    axis.set_zticks([0, 8, 16, 24, 32])
    axis.tick_params(labelsize=8)
    axis.grid(True, alpha=0.15)
    for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
        dimension.pane.set_facecolor((0.96, 0.97, 0.98, 1))
        dimension._axinfo["grid"]["color"] = (0.4, 0.45, 0.5, 0.13)
    if abs(elevation) == 90:
        axis.set_zticks([])
        axis.set_zlabel("")


def section_paths(shape, plane, coordinates):
    sliced = section(shape, section_by=plane)
    paths = [np.asarray([tuple(point) for point in edge.positions(deflection=0.01)])[:, coordinates]
             for edge in sliced.edges()]
    return sliced, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("C:/clones/Deep_Frame/exports/topology/workstation_20260930/density_study/grid8over3_iter150/inputs.json"))
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    record = {"status": "running", "step": str(args.step.resolve()), "step_sha256": digest(args.step),
              "inputs_sha256": digest(args.inputs), "renderer_sha256": digest(Path(__file__)),
              "rendering": "Fresh STEP import, whole-CAD adaptive tessellation, flat per-triangle normal shading, orthographic projection and metric axes; no geometry smoothing, decimation, hole filling, vertex-normal interpolation or remodeling",
              "section_method": "Exact CAD intersection with each named plane; section faces filled from their tessellation and CAD edges sampled to 0.01 mm deflection", "images": {}, "sections": []}
    def journal(stage):
        record.update(stage=stage, elapsed_s=perf_counter()-started)
        (args.output / "render_manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        print(stage, flush=True)
    journal("STEP import")
    solid = import_step(args.step)
    journal("CAD tessellation")
    mesh = _mesh(solid, _settings({}))
    outward = []
    for body in solid.solids():
        classifier = BRepClass3d_SolidClassifier(body.wrapped)
        classifier.PerformInfinitePoint(1e-7)
        outward.append(classifier.State() == TopAbs_OUT)
    record["topology"] = {"solid_count": len(solid.solids()), "cad_valid": bool(solid.is_valid),
                          "outward_oriented": bool(outward) and all(outward), "mesh_watertight": bool(mesh.is_watertight),
                          "mesh_body_count": int(mesh.body_count), "cad_volume_mm3": float(solid.volume),
                          "mesh_signed_volume_mm3": float(mesh.volume), "triangle_count": len(mesh.faces),
                          "bounds_mm": mesh.bounds.tolist(), "tessellation": mesh.metadata["adaptive_tessellation"]}
    valid = len(solid.solids()) == mesh.body_count == 1 and solid.is_valid and mesh.is_watertight and all(outward) and mesh.volume > 0
    label = "DIAGNOSE – Geometrie- und Mechanikfreigabe offen"
    if not valid:
        label = "DIAGNOSE – Topologieprüfung fehlgeschlagen"
    if args.validation:
        proof = json.loads(args.validation.read_text(encoding="utf-8"))
        expected = proof.get("sha256") or proof.get("provenance", {}).get("candidate_sha256")
        if expected != record["step_sha256"]:
            raise ValueError("Validation record must identify the rendered STEP by matching SHA256")
        validation = proof.get("validation", proof)
        record["validation"] = {"path": str(args.validation), "sha256": digest(args.validation),
                                "passed": validation.get("passed", False), "violations": validation.get("violations", [])}
        if validation.get("passed", False) and valid:
            label = "Geometrie-Gates bestanden – mechanische Freigabe separat prüfen"
        else:
            label = "DIAGNOSE – Geometrie-Gates nicht bestanden"
    record["display_status"] = label
    domain = json.loads(args.inputs.read_text(encoding="utf-8"))["domain"]
    lower = np.asarray(domain["grid"]["origin_mm"], dtype=float)
    upper = lower + np.asarray(domain["grid"]["shape"]) * np.asarray(domain["grid"]["spacing_mm"])
    lower, upper = np.minimum(lower, mesh.bounds[0]), np.maximum(upper, mesh.bounds[1])
    footer = "Reimportierter STEP | SHA256 " + record["step_sha256"][:16] + "… | CAD-Dreiecke: " + str(len(mesh.faces))
    def save(figure, name):
        path = args.output / (name + ".png")
        figure.savefig(path, dpi=200, facecolor="white")
        plt.close(figure)
        record["images"][name] = {"path": path.name, "sha256": digest(path)}
        journal("saved " + name)
    views = [(28, -48, "Isometrie", "isometric"), (90, -90, "Draufsicht (+z)", "top"), (-90, -90, "Unterseite (−z)", "bottom")]
    for elev, azim, title, name in views:
        figure = plt.figure(figsize=(9, 8.2), layout="constrained")
        axis = figure.add_subplot(projection="3d")
        draw_view(axis, mesh, lower, upper, elev, azim, title)
        figure.suptitle(label, fontsize=13, color="#9b392b" if label.startswith("DIAGNOSE") else "#275e3c")
        figure.text(0.5, 0.01, footer, ha="center", fontsize=8)
        save(figure, name)
    journal("exact CAD strap sections")
    strap_voids = [region for region in domain["regions"] if region["role"] == "forbidden" and region["name"].startswith("strap_access_")]
    strap_preserves = [region for region in domain["regions"] if region["role"] == "preserve" and region["name"].startswith("strap_contact_")]
    centers = np.asarray([(np.asarray(region["min_mm"]) + region["max_mm"]) / 2 for region in strap_voids])
    zmid = float(np.mean([(region["min_mm"][2] + region["max_mm"][2]) / 2 for region in strap_preserves]))
    specs = [(f"Quer: y = {y:g} mm", Plane(origin=(0, y, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0)), [0, 2], [-24, 24], [20, 32]) for y in np.unique(centers[:, 1])]
    specs += [(f"Längs: x = {x:g} mm", Plane(origin=(x, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)), [1, 2], [-25, 25], [20, 32]) for x in np.unique(centers[:, 0])]
    specs += [(f"Horizontal: z = {zmid:g} mm", Plane(origin=(0, 0, zmid)), [0, 1], [-23, 23], [-25, 25])]
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    for axis, (title, plane, coordinates, xlim, ylim) in zip(axes.flat, specs):
        journal("section " + title)
        sliced, paths = section_paths(solid, plane, coordinates)
        vertices, triangles = sliced.tessellate(0.01, 0.08)
        if triangles:
            array = np.asarray([tuple(point) for point in vertices])
            axis.add_collection(PolyCollection(array[np.asarray(triangles)][:, :, coordinates], facecolors="#4b9bae", edgecolors="none", antialiased=False))
        axis.add_collection(LineCollection(paths, colors="#194653", linewidths=0.65))
        for region in strap_preserves:
            prescribed = region_shape(region).cut(*[region_shape(cut) for cut in strap_voids])
            _, outlines = section_paths(prescribed, plane, coordinates)
            axis.add_collection(LineCollection(outlines, colors="#46904b", linewidths=1.25, linestyles="--"))
        for region in strap_voids:
            _, outlines = section_paths(region_shape(region), plane, coordinates)
            axis.add_collection(LineCollection(outlines, colors="#c6553c", linewidths=1.2, linestyles=":"))
        axis.set(xlim=xlim, ylim=ylim, title=title, xlabel="xyz"[coordinates[0]] + " / mm", ylabel="xyz"[coordinates[1]] + " / mm")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        record["sections"].append({"title": title, "plane_origin_mm": list(plane.origin), "plane_normal": list(plane.z_dir), "cad_area_mm2": float(sliced.area), "cad_edge_count": len(paths), "view_limits_mm": [xlim, ylim]})
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[Line2D([0], [0], color="#4b9bae", linewidth=8, label="Tatsächliches STEP-Material"),
                                 Line2D([0], [0], color="#46904b", linestyle="--", label="Pflicht-Preserve abzüglich Gurtzugang"),
                                 Line2D([0], [0], color="#c6553c", linestyle=":", label="Freizuhaltender Gurtzugang")], loc="center", fontsize=11)
    figure.suptitle(label + "\nExakte CAD-Schnitte an allen vier Gurtdurchlässen", fontsize=14)
    figure.text(0.5, 0.006, footer, ha="center", fontsize=8)
    save(figure, "strap_sections")
    record["status"] = "complete"
    journal("complete")


if __name__ == "__main__":
    main()
