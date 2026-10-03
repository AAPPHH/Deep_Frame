import json
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

from deep_frame.config import DESIGN_RECONSTRUCTION_CONFIG, DESIGN_RECONSTRUCTION_KINDS, IMPLICIT_CONFIG, command_line, configure
from deep_frame.topology_reconstruction import load_paths, reconstruct, reference_body

RUN_CONFIG = {**DESIGN_RECONSTRUCTION_CONFIG, "study": {}, "geometry": None, "panels": None, "labels": None, "fea_surface_targets_mm": [0.5, 0.6], "fea_volume_targets_mm": [1.5, 1.2, 1.0], "fea_feature_degs": [40.0, 60.0, 89.0]}
RUN_KINDS = {**DESIGN_RECONSTRUCTION_KINDS, "study": "object", "geometry": "path", "panels": ["path"], "labels": ["text"], "fea_surface_targets_mm": ["float"], "fea_volume_targets_mm": ["float"], "fea_feature_degs": ["float"]}
FEA_RELAXED = {"surface_deviation_mm": 0.7, "fea_remesh_targets_mm": [1.5], "tet_attempts": ["remesh_hxt", "remesh_delaunay"]}
def _camera(direction, up):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    right = np.cross(d, up)
    right /= np.linalg.norm(right)
    return d, right, np.cross(right, d)

def _raster(screen, depth, faces, width, height, chunk=60000):
    zbuf = np.full(width * height, np.inf)
    fid = np.full(width * height, -1)
    bary = np.zeros((width * height, 2))
    for start in range(0, len(faces), chunk):
        index = np.arange(start, min(start + chunk, len(faces)))
        tri = screen[faces[index]]
        z = depth[faces[index]]
        low = np.clip(np.floor(tri.min(1)).astype(int), 0, [width - 1, height - 1])
        high = np.clip(np.ceil(tri.max(1)).astype(int), 0, [width - 1, height - 1])
        size = high - low + 1
        count = size[:, 0] * size[:, 1]
        owner = np.repeat(np.arange(len(index)), count)
        local = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        px = low[owner, 0] + local % size[owner, 0] + 0.5
        py = low[owner, 1] + local // size[owner, 0] + 0.5
        a, b, c = tri[owner, 0], tri[owner, 1], tri[owner, 2]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        w1 = ((px - a[:, 0]) * (c[:, 1] - a[:, 1]) - (py - a[:, 1]) * (c[:, 0] - a[:, 0])) / np.where(area == 0, 1, area)
        w2 = ((b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (b[:, 1] - a[:, 1]) * (px - a[:, 0])) / np.where(area == 0, 1, area)
        keep = (area != 0) & (w1 >= -1e-9) & (w2 >= -1e-9) & (w1 + w2 <= 1 + 1e-9)
        owner, w1, w2 = owner[keep], w1[keep], w2[keep]
        pix = (py[keep] - 0.5).astype(int) * width + (px[keep] - 0.5).astype(int)
        zz = z[owner, 0] * (1 - w1 - w2) + z[owner, 1] * w1 + z[owner, 2] * w2
        order = np.lexsort((zz, pix))
        pix, zz, owner, w1, w2 = pix[order], zz[order], owner[order], w1[order], w2[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, zz, owner, w1, w2 = pix[first], zz[first], owner[first], w1[first], w2[first]
        better = zz < zbuf[pix]
        pix = pix[better]
        zbuf[pix], fid[pix], bary[pix, 0], bary[pix, 1] = zz[better], index[owner[better]], w1[better], w2[better]
    return zbuf, fid, bary

def _project(points, camera, center, scale, width, height):
    d, right, up = camera
    rel = points - center
    return np.stack([width / 2 + rel @ right * scale, height / 2 - rel @ up * scale], 1), rel @ d

def render(mesh, direction, up=(0, 0, 1), size=(1600, 1200), supersample=2, light=(-0.45, 0.35, -1.0), ground=True):
    width, height = size[0] * supersample, size[1] * supersample
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals)
    camera = _camera(direction, up)
    center = (vertices.min(0) + vertices.max(0)) / 2
    extent = np.stack([_project(vertices, camera, center, 1, 0, 0)[0][:, i] for i in (0, 1)], 1)
    scale = 0.86 * min(width / np.ptp(extent[:, 0]), height / np.ptp(extent[:, 1]))
    screen, depth = _project(vertices, camera, center, scale, width, height)
    zbuf, fid, bary = _raster(screen, depth, faces, width, height)
    lcam = _camera(light, (0, 0, 1) if abs(np.asarray(light)[2]) < 0.99 * np.linalg.norm(light) else (0, 1, 0))
    lsize = 2048
    lext = _project(vertices, lcam, center, 1, 0, 0)[0]
    lscale = 0.9 * lsize / max(np.ptp(lext[:, 0]), np.ptp(lext[:, 1])) / 1.6
    lscreen, ldepth = _project(vertices, lcam, center, lscale, lsize, lsize)
    lz = _raster(lscreen, ldepth, faces, lsize, lsize)[0]
    def lit(points, bias):
        s, z = _project(points, lcam, center, lscale, lsize, lsize)
        total = np.zeros(len(points))
        for ox in (-1.5, 0, 1.5):
            for oy in (-1.5, 0, 1.5):
                ix = np.clip((s[:, 0] + ox).astype(int), 0, lsize - 1)
                iy = np.clip((s[:, 1] + oy).astype(int), 0, lsize - 1)
                total += z <= lz[iy * lsize + ix] + bias
        return total / 9
    image = np.ones((width * height, 3))
    hit = fid >= 0
    f = faces[fid[hit]]
    w = np.stack([1 - bary[hit].sum(1), bary[hit, 0], bary[hit, 1]], 1)
    n = np.einsum("pk,pkj->pj", w, normals[f])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    p = np.einsum("pk,pkj->pj", w, vertices[f])
    view = -camera[0]
    n = np.where((n @ view)[:, None] < 0, -n, n)
    tolight = -lcam[0]
    shadow = lit(p, 0.6 / 1.0)
    key = np.clip(n @ tolight, 0, 1) * (0.25 + 0.75 * shadow)
    fill = np.clip(n @ np.array([0.6, -0.5, 0.4]) / np.linalg.norm([0.6, -0.5, 0.4]), 0, 1)
    sky = 0.5 + 0.5 * n[:, 2]
    half = tolight + view
    half /= np.linalg.norm(half)
    spec = np.clip(n @ half, 0, 1) ** 40 * shadow
    rim = (1 - np.clip(n @ view, 0, 1)) ** 3
    base = np.array([0.88, 0.88, 0.90])
    shade = 0.42 * sky[:, None] + 0.55 * key[:, None] + 0.18 * fill[:, None]
    image[hit] = np.clip(base * shade + 0.18 * spec[:, None] + 0.05 * rim[:, None], 0, 1)
    if ground and abs(camera[0][2]) > 0.2:
        d, right, up_v = camera
        miss = np.flatnonzero(~hit)
        sx = (miss % width + 0.5 - width / 2) / scale
        sy = -(miss // width + 0.5 - height / 2) / scale
        origin = center + sx[:, None] * right + sy[:, None] * up_v
        t = (vertices[:, 2].min() - 0.05 - origin[:, 2]) / d[2]
        g = origin + t[:, None] * d
        occluded = 1 - lit(g, 0.6)
        mask = np.zeros(width * height)
        mask[miss] = occluded
        mask = gaussian_filter(mask.reshape(height, width), 3 * supersample).ravel()
        image[miss] = 1 - 0.28 * mask[miss, None]
    image = image.reshape(height, width, 3)
    image = image.reshape(size[1], supersample, size[0], supersample, 3).mean((1, 3))
    return Image.fromarray((np.clip(image, 0, 1) ** (1 / 1.1) * 255).astype(np.uint8))

VIEWS = {"iso": ((0.55, -0.85, -0.62), (0, 0, 1)), "top": ((0, 0, -1), (0, 1, 0)), "side": ((-1, 0, 0), (0, 0, 1))}

def render_views(mesh, out, views=VIEWS):
    shaded = trimesh.graph.smooth_shade(mesh, angle=np.radians(35), facet_minarea=None)
    for name, (direction, up) in views.items():
        render(shaded, direction, up).save(out / f"{name}.png")

def full_domain(shape, study=None):
    from tools.neural_study import R2Domain, configure as study_config
    return R2Domain(study_config(study or {})).build(shape)[0]

def _widen(region, band):
    if region.get("kind") != "box":
        return region
    low, high = np.asarray(region["min_mm"], dtype=float), np.asarray(region["max_mm"], dtype=float)
    flat = high-low < 0.1
    middle = (low+high)/2
    return {**region, "min_mm": np.where(flat, middle-band, low).tolist(), "max_mm": np.where(flat, middle+band, high).tolist()}

def fea_cases(domain, band):
    cases = deepcopy(domain["comparison_load_cases"])
    for case in cases:
        case["fixed_regions"] = [_widen(region, band) for region in case["fixed_regions"]]
        for load in case.get("loads", []):
            load["region"] = _widen(load["region"], band)
    masses = deepcopy(domain["point_masses"])
    for mass in masses:
        mass["attachment_region"] = _widen(mass["attachment_region"], band)
    return cases, masses

def _count(mesh, region):
    vertices = mesh.vertices
    return int(np.all((vertices >= np.asarray(region["min_mm"])) & (vertices <= np.asarray(region["max_mm"])), axis=1).sum())

def fea_surface(mesh, target, taubin):
    import pymeshlab
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    meshes.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(target), iterations=6, featuredeg=40)
    if taubin:
        meshes.apply_coord_taubin_smoothing(stepsmoothnum=taubin)
    return trimesh.Trimesh(meshes.current_mesh().vertex_matrix(), meshes.current_mesh().face_matrix())

def robust_surface(mesh, config, settings):
    from deep_frame.fea import _prepare_surface
    trials = []
    for volume_target in config["fea_volume_targets_mm"]:
        for feature in config["fea_feature_degs"]:
            for target in config["fea_surface_targets_mm"]:
                surface = fea_surface(mesh, target, config["fea_surface_taubin"])
                row = {"surface_mm": target, "taubin": config["fea_surface_taubin"], "volume_mm": volume_target, "feature_deg": feature}
                try:
                    _prepare_surface(surface, {**settings, "fea_remesh_feature_deg": feature}, False, volume_target)
                    trials.append({**row, "passed": True})
                    return surface, trials, {"fea_remesh_targets_mm": [volume_target], "fea_remesh_feature_deg": feature}
                except ValueError as error:
                    trials.append({**row, "passed": False, "diagnostic": str(error)[-300:]})
    return fea_surface(mesh, config["fea_surface_targets_mm"][0], config["fea_surface_taubin"]), trials, {}

def fea_main(overrides):
    from deep_frame.fea import MESH_KEYS, evaluate
    config = configure(RUN_CONFIG, RUN_KINDS, overrides, ("geometry", "output"))
    config["output"].mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load_mesh(config["geometry"], process=True)
    domain = full_domain(config["fine_shape"], config["study"])
    settings = {**domain["fea_settings"], **{key: IMPLICIT_CONFIG[key] for key in MESH_KEYS}, "work_dir": str(config["output"]/"fea"), "mesh_timeout_s": 900.0, "solver_timeout_s": 1800.0, "threads": 8, "mesh_threads": 4, "fea_memory_budget_mb": 6000.0, "mesh_minimum_sicn": 0.005, **FEA_RELAXED}
    surface_trials, chosen = [], {}
    if config["fea_surface_mm"] > 0:
        mesh, surface_trials, chosen = robust_surface(mesh, config, {**IMPLICIT_CONFIG, **settings})
        settings.update(chosen)
    cases, masses = fea_cases(domain, config["selector_half_band_mm"])
    preflight = {case["name"]: {"fixed": [_count(mesh, r) for r in case["fixed_regions"]], "loads": [_count(mesh, l["region"]) for l in case.get("loads", [])]} for case in cases}
    preflight["battery_attachment"] = [_count(mesh, m["attachment_region"]) for m in masses]
    print(json.dumps(preflight), flush=True)
    started = perf_counter()
    result = evaluate(mesh, domain["material"], masses, cases, settings)
    result.update(runtime_s=perf_counter()-started, relaxed_settings=FEA_RELAXED, fea_surface_trials=surface_trials, fea_mesh_choice=chosen, fea_surface_taubin=config["fea_surface_taubin"], fea_surface_volume_mm3=float(mesh.volume), selector_half_band_mm=config["selector_half_band_mm"], preflight_vertex_counts=preflight, geometry=str(config["geometry"]))
    (config["output"]/"fea_result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(summary(result), indent=1), flush=True)

def summary(result):
    cases = result.get("load_cases", {})
    return {"status": result["status"], "diagnostics": result.get("diagnostics"), "frame_mass_g": result.get("frame_mass_g"), "f1_hz": (result.get("eigenfrequencies_hz") or [None])[0], "arm_tip_n_per_mm": result.get("stiffness_n_per_mm"),
            "crash_front_mm": cases.get("crash_front", {}).get("max_displacement_mm"), "crash_arm_mm": cases.get("crash_arm", {}).get("max_displacement_mm"), "runtime_s": result.get("runtime_s")}

def build_main(overrides):
    config = configure(RUN_CONFIG, RUN_KINDS, overrides, ("source", "output"))
    config["output"].mkdir(parents=True, exist_ok=True)
    domain = full_domain(config["fine_shape"], config["study"])
    density = np.load(config["source"])["density"]
    reference, report_reference = reference_body(domain, density, config)
    reference.export(config["output"]/"reference.stl")
    if config["geometry"]:
        report_reference["render_stl_mm3"] = float(trimesh.load_mesh(config["geometry"]).volume)
    mesh, graph, report = reconstruct(domain, density, config, config["target_volume_mm3"] or report_reference["volume_mm3"])
    report["reference"] = report_reference
    report["load_paths"], report_reference["load_paths"] = load_paths(mesh, domain, config), load_paths(reference, domain, config)
    mesh.export(config["output"]/"geometry.stl")
    report["mass_g"] = report["volume_mm3"]*domain["material"]["density_g_cm3"]/1000
    (config["output"]/"reconstruction.json").write_text(json.dumps({**report, "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}}, indent=1, default=float))
    render_views(mesh, config["output"])
    print(json.dumps({k: report[k] for k in ("members", "shells", "nodes", "continuity", "joint_sections", "load_paths", "volume_mm3", "target_volume_mm3", "global_scale", "mass_budget", "reference", "bodies", "watertight", "runtime_s")}, default=float), flush=True)

def render_main(overrides):
    config = configure(RUN_CONFIG, RUN_KINDS, overrides, ("geometry", "output"))
    config["output"].mkdir(parents=True, exist_ok=True)
    render_views(trimesh.load_mesh(config["geometry"], process=False), config["output"])

def compose_main(overrides):
    config = configure(RUN_CONFIG, RUN_KINDS, overrides, ("panels", "output"))
    images = [Image.open(path).convert("RGB") for path in config["panels"]]
    images = [image.crop(image.point(lambda v: 255-v).getbbox() or (0, 0, *image.size)) for image in images]
    height = max(image.height for image in images)
    images = [image.resize((round(image.width*height/image.height), height), Image.LANCZOS) for image in images]
    canvas = Image.new("RGB", (sum(image.width for image in images)+40*(len(images)+1), height+120), "white")
    from PIL import ImageDraw, ImageFont
    draw, x = ImageDraw.Draw(canvas), 40
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    for index, image in enumerate(images):
        canvas.paste(image, (x, 100))
        if config["labels"]:
            draw.text((x, 30), config["labels"][index], fill="black", font=font)
        x += image.width+40
    canvas.save(config["output"])

def main(argv=None):
    return command_line({"build": build_main, "fea": fea_main, "render": render_main, "compose": compose_main}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
