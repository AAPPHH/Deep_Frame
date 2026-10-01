"""Render actual v0, handoff and selected workstation STL files at one scale."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh

from deep_frame.topology_visualization import _plot_modules


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/workstation_geometry_comparison.png")
    args = parser.parse_args()
    base = ROOT / "exports/topology/workstation_20260930"
    mesh_dir = base / "mesh_study"
    study_dir = base / "candidate_study/grid4_iter300"
    preparation = read(mesh_dir / "preparation.json")
    selection = read(study_dir / "manifest.json")
    selected_id = selection["selected_id"]
    record_path = study_dir / "candidates" / selected_id / "record.json"
    selected = read(record_path)
    if selected["status"] != "ok" or not selected["comparison"]["passed"]:
        raise ValueError("Selected workstation candidate must have passed geometry and independent FEA")
    old_fea = read(ROOT / "docs/validation/topology_phase1/candidate_fea.json")
    baseline_fea = read(ROOT / "docs/validation/topology_phase1/baseline_fea.json")
    sources = [
        ("v0", mesh_dir, preparation["geometries"]["baseline"]["stl"], baseline_fea["frame_mass_g"], "Parametrische v0", "#7994ab"),
        ("handoff_45", mesh_dir, preparation["geometries"]["candidate"]["stl"], old_fea["frame_mass_g"], "Freie Referenz: 45 Updates, t = 0.20", "#b07849"),
        ("workstation_300", study_dir, selected["artifacts"]["stl"], selected["fea"]["frame_mass_g"], f"Workstation: 300 Updates, t = {selected['density_threshold']:.2f}", "#248c87"),
    ]
    grid = read(ROOT / "docs/validation/topology_phase1/inputs.json")["domain"]["grid"]
    lower = np.asarray(grid["origin_mm"], dtype=float)
    size = np.asarray(grid["shape"]) * np.asarray(grid["spacing_mm"])
    upper = lower + size
    plt, collection = _plot_modules()
    figure = plt.figure(figsize=(16, 10), layout="constrained")
    manifest = {
        "schema_version": "deep-frame-workstation-geometry-visualization-v1",
        "rendering": "Direct original STL triangles, Matplotlib orthographic shaded projections, identical physical limits and scale in all panels; no geometry smoothing or remodeling",
        "grid": grid,
        "axis_limits_mm": [lower.tolist(), upper.tolist()],
        "views": [{"elevation_deg": 28, "azimuth_deg": -45}, {"elevation_deg": 90, "azimuth_deg": -90}],
        "packages": {name: version(name) for name in ("numpy", "matplotlib", "trimesh")},
        "renderer_sha256": digest(Path(__file__)),
        "selected_id": selected_id,
        "selected_manifest_sha256": digest(study_dir / "manifest.json"),
        "selected_record_sha256": digest(record_path),
        "sources": [],
    }
    for index, (name, directory, expected, mass, title, color) in enumerate(sources):
        path = directory / expected["path"]
        if digest(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError("STL source failed its recorded artifact hash: " + name)
        raw_mesh = trimesh.load_mesh(path, process=False)
        triangles = np.asarray(raw_mesh.triangles)
        if not np.all(np.isfinite(triangles)):
            raise ValueError("Non-finite STL coordinates")
        checked = raw_mesh.copy()
        checked.process(validate=False)
        if not checked.is_watertight or not checked.is_volume or checked.body_count != 1:
            raise ValueError("Expected one closed STL body: " + name)
        if np.any(checked.bounds[0] < lower - 1e-5) or np.any(checked.bounds[1] > upper + 1e-5):
            raise ValueError("The shared physical plot bounds would clip geometry")
        for row in (0, 1):
            axis = figure.add_subplot(2, 3, index + 1 + row * 3, projection="3d")
            axis.add_collection3d(collection(triangles, facecolors=color, linewidths=0, shade=True))
            axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]), xlabel="x / mm", ylabel="y / mm", zlabel="z / mm")
            axis.set_box_aspect(size)
            axis.set_proj_type("ortho")
            axis.view_init(elev=28 if row == 0 else 90, azim=-45 if row == 0 else -90)
            axis.grid(False)
            axis.set_xticks([-60, -30, 0, 30, 60])
            axis.set_yticks([-60, -30, 0, 30, 60])
            axis.set_zticks([0, 16, 32] if row == 0 else [])
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(f"{title}\nFrame-Masse {mass:.3f} g", fontsize=12)
            else:
                axis.set_zlabel("")
                axis.set_title("Draufsicht", fontsize=11)
        manifest["sources"].append({
            "name": name, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": expected["sha256"], "size_bytes": expected["size_bytes"],
            "triangle_count": len(triangles), "bounds_mm": checked.bounds.tolist(),
            "stl_volume_mm3": float(checked.volume), "cad_frame_mass_g": mass,
            "watertight": bool(checked.is_watertight), "body_count": int(checked.body_count),
            "topology_check_note": "Vertex merging used only for closed-body checks; rendering uses original STL triangles",
        })
    figure.suptitle("Deep_Frame | Workstation: drei Strukturen im gleichen Maßstab\n4-mm-Designraster der freien Strukturen; identische orthografische Ansichten", fontsize=15)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    manifest["image"] = {"path": str(output.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(output), "size_bytes": output.stat().st_size}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"image": str(output), "manifest": str(output.with_suffix('.json')), "sources": len(sources)}))


if __name__ == "__main__":
    main()
