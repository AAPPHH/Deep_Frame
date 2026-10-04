import json
import shutil
import subprocess
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

from deep_frame.config import CABLES, CABLES_KINDS, DESIGN_RECONSTRUCTION_CONFIG, DESIGN_RECONSTRUCTION_KINDS, IMPLICIT_CONFIG, RUN_SETTINGS, STAGES, SPLINE_RECONSTRUCTION_CONFIG, SPLINE_RECONSTRUCTION_KINDS, command_line, configure
from deep_frame.frame_run import FrameRun, _git
from deep_frame.topology_cables import CableChannels
from deep_frame.topology_reconstruction import body_weights, bumps, core_domain, load_paths, reconstruct, reconstruct_splines, reference_body, spline_graph, stored_domain

RUN_CONFIG = {**DESIGN_RECONSTRUCTION_CONFIG, "domain": None, "study": {}, "geometry": None, "panels": None, "labels": None, "fea_surface_targets_mm": [0.5, 0.6], "fea_volume_targets_mm": [1.5, 1.2, 1.0], "fea_feature_degs": [40.0, 60.0, 89.0]}
RUN_KINDS = {**DESIGN_RECONSTRUCTION_KINDS, "domain": "path", "study": "object", "geometry": "path", "panels": ["path"], "labels": ["text"], "fea_surface_targets_mm": ["float"], "fea_volume_targets_mm": ["float"], "fea_feature_degs": ["float"]}
SPLINE_RUN_CONFIG = {**RUN_CONFIG, **SPLINE_RECONSTRUCTION_CONFIG, "compare_bodies": [], "compare_labels": []}
SPLINE_RUN_KINDS = {**RUN_KINDS, **SPLINE_RECONSTRUCTION_KINDS, "compare_bodies": ["path"], "compare_labels": ["text"]}
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
CABLE_VIEWS = {**VIEWS, "front": ((0, -1, 0), (0, 0, 1)), "bottom": ((0, 0, 1), (0, 1, 0)), "bottom_iso": ((0.55, -0.85, 0.62), (0, 0, 1))}
CABLE_RUN_CONFIG = {**SPLINE_RUN_CONFIG, "body": None, "cables": {}, "section_paths": ["front_left", "rear_right", "camera"]}
CABLE_RUN_KINDS = {**SPLINE_RUN_KINDS, "body": "path", "cables": "object", "section_paths": ["text"]}

def render_views(mesh, out, views=VIEWS):
    shaded = trimesh.graph.smooth_shade(mesh, angle=np.radians(35), facet_minarea=None)
    for name, (direction, up) in views.items():
        render(shaded, direction, up).save(out / f"{name}.png")

def full_domain(config):
    if config["domain"]:
        return stored_domain(config["domain"])
    from tools.neural_study import R2Domain, configure as study_config
    return R2Domain(study_config(config["study"])).build(config["fine_shape"])[0]

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

def fea_main(overrides):
    from deep_frame.fea import MESH_KEYS, evaluate, robust_surface
    config = configure(RUN_CONFIG, RUN_KINDS, overrides, ("geometry", "output"))
    config["output"].mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load_mesh(config["geometry"], process=True)
    domain = full_domain(config)
    settings = {**domain["fea_settings"], **{key: IMPLICIT_CONFIG[key] for key in MESH_KEYS}, "work_dir": str(config["output"]/"fea"), "mesh_timeout_s": 900.0, "solver_timeout_s": 1800.0, "threads": 8, "mesh_threads": 4, "fea_memory_budget_mb": 6000.0, "mesh_minimum_sicn": 0.005, **FEA_RELAXED}
    surface_trials, chosen = [], {}
    if config["fea_surface_mm"] > 0:
        mesh, surface_trials, chosen = robust_surface(mesh, {**settings, "fea_remesh_targets_mm": config["fea_volume_targets_mm"]}, {"targets_mm": config["fea_surface_targets_mm"], "taubin": config["fea_surface_taubin"], "feature_degs": config["fea_feature_degs"]})
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
    domain = full_domain(config)
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

def bump_summary(rows):
    return {"count": len(rows), "classes": {kind: sum(1 for row in rows if row["class"] == kind) for kind in sorted({row["class"] for row in rows})}, "maximum_excess_mm": max([row["excess_mm"] for row in rows], default=0.0)}

def splines_main(overrides):
    import psutil
    config = configure(SPLINE_RUN_CONFIG, SPLINE_RUN_KINDS, overrides, ("source", "output"))
    config["output"].mkdir(parents=True, exist_ok=True)
    domain = full_domain(config)
    density = np.load(config["source"])["density"]
    source = config["section_body"] or config["source"].with_name("geometry.stl")
    body = trimesh.load_mesh(source, process=True) if Path(source).is_file() else None
    mesh, graph, rods, report = reconstruct_splines(domain, density, config, body)
    mesh.export(config["output"]/"geometry.stl")
    report["load_paths"] = load_paths(mesh, domain, config)
    report["mass_g"] = report["volume_mm3"]*domain["material"]["density_g_cm3"]/1000
    report["section_body"] = str(source) if body is not None else None
    bodies = {"recon_v3": mesh, **{label: trimesh.load_mesh(path, process=True) for label, path in zip(config["compare_labels"], config["compare_bodies"])}}
    report["bumps_by_body"] = {"raw": report["bumps"], **{label: bumps(graph, domain, rods, graph.areas(body_weights(item, domain["grid"], config["body_subdivisions"])), config) for label, item in bodies.items()}}
    report["bump_summary"] = {label: bump_summary(rows) for label, rows in report["bumps_by_body"].items()}
    report["peak_rss_gb"] = psutil.Process().memory_info().peak_wset/2**30 if hasattr(psutil.Process().memory_info(), "peak_wset") else None
    (config["output"]/"reconstruction.json").write_text(json.dumps({**report, "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}}, indent=1, default=lambda value: value.tolist() if hasattr(value, "tolist") else float(value) if isinstance(value, np.floating) else str(value)), encoding="utf-8")
    render_views(mesh, config["output"])
    print(json.dumps({k: report[k] for k in ("members", "shells", "nodes", "control_points", "continuity", "joint_sections", "load_paths", "volume_mm3", "mass_g", "mass_budget", "bodies", "watertight", "bump_summary", "peak_rss_gb", "runtime_s")}, default=float), flush=True)

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

RECON3_STUDY = {
    "root": "exports/recon3",
    "viewer": "C:/clones/Deep_Frame-neural/exports",
    "manafly_renders": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_opt/manafly/renders",
    "manafly_stl": "C:/clones/Deep_Frame-neural/exports/manafly_ref/manafly3_repaired.stl",
    "python": "C:/clones/Deep_Frame/.venv/Scripts/python.exe",
    "steps": ["run", "evaluate", "compose", "figure", "table"],
    "only": [],
    "fallback": {"fea_settings": {"fea_memory_budget_mb": 16384.0}, "compute": "reconstruction"},
    "cases": {
        "simp_mma": {"label": "SIMP-MMA", "raw": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_raw_1", "recon": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_recon_1", "result": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_opt/simp_mma"},
        "neural_v06_f1": {"label": "Neural 6 % f1", "raw": "C:/clones/Deep_Frame-r4/exports/runs/r4_neural_v06_f1_1_raw", "recon": "C:/clones/Deep_Frame-r4/exports/runs/r4_neural_v06_f1_recon11_1", "result": "C:/clones/Deep_Frame-r4/exports/runs/r4_neural_v06_f1_1/optimization/r4_neural_v06_f1"},
    },
}
RECON3_KINDS = {"fallback": "object", "root": "path", "viewer": "path", "manafly_renders": "path", "manafly_stl": "path", "python": "text", "steps": ["text"], "only": ["text"], "cases": "object"}
BODIES = ("raw", "recon_1to1", "recon_v3")

class SplineRun(FrameRun):
    def __init__(self, request, case, stages):
        self.case = case
        super().__init__(request, stages)
        self.grid = {**self.grid, "reconstruction": {**self.grid["reconstruction"], "compare_bodies": [str(Path(case["recon"])/"frame.stl")], "compare_labels": ["recon_1to1"]}}

    def available(self, stage):
        return stage == "optimization" or super().available(stage)

    def optimization(self, domain):
        self.manifest["stages"]["optimization"] = {"status": "ran", "source": self.case["result"], "method": "existing optimizer result; reconstruction v3 (member splines)", **_git(str(ROOT))}
        self.save()
        return Path(self.case["result"])

def _stage(path, request):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=1), encoding="utf-8")
    return path

def spline_run(cfg, name, case):
    stages = {key: {**spec, "worktree": ROOT.as_posix()} for key, spec in STAGES.items()}
    stages["reconstruction"] = {**stages["reconstruction"], "argv": ["splines"], "compute": "geometry"}
    request = json.loads((Path(case["raw"])/"config.json").read_text(encoding="utf-8"))
    manifest = SplineRun({**request, "name": f"recon3_{name}", "reconstruction": True}, case, stages).run()
    run = ROOT/RUN_SETTINGS["root"]/manifest["name"]
    target = cfg["viewer"]/f"recon3_{name}"/"geometry.stl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run/"frame.stl", target)
    _stage(cfg["root"]/name/"run.json", {"run": str(run), "status": manifest["status"], "stages": {key: entry["status"] for key, entry in manifest["stages"].items()}})

def evaluate_bodies(cfg, name, case):
    jobs = []
    for label in ("raw", "recon_1to1"):
        run, out = Path(case["raw" if label == "raw" else "recon"]), cfg["root"]/name/label
        spec = json.loads((run/"requests"/"evaluation_frame.json").read_text(encoding="utf-8"))
        spec.update(name=f"recon3_{name}_{label}", stl=str(run/"frame.stl"), output=str(out/"evaluation"))
        spec.pop("datasheet", None)
        frame = _stage(out/"frame.json", spec)
        views = {key: list(value) for key, value in RUN_SETTINGS["views"].items()}
        render = _stage(out/"render.json", {"root": ROOT.as_posix(), "tool": "tools/neural_study.py", "action": "call", "patch": {}, "function": "render_views", "kwargs": {"mesh": str(run/"frame.stl"), "out": str(out/"renders"), "views": views}})
        (out/"renders").mkdir(parents=True, exist_ok=True)
        jobs.append(subprocess.Popen([cfg["python"], str(ROOT/"tools"/"evaluate_frame.py"), "run", str(frame)], cwd=ROOT, stdout=(out/"evaluation.log").open("w", encoding="utf-8"), stderr=subprocess.STDOUT))
        jobs.append(subprocess.Popen([cfg["python"], RUN_SETTINGS["compute"], "render", "--cwd", str(ROOT), "--", cfg["python"], str(ROOT/"run.py"), "stage", str(render)], cwd=ROOT, stdout=(out/"render.log").open("w", encoding="utf-8"), stderr=subprocess.STDOUT))
    return jobs

def paths(cfg, name):
    run = Path(json.loads((cfg["root"]/name/"run.json").read_text(encoding="utf-8"))["run"])
    return {"raw": cfg["root"]/name/"raw", "recon_1to1": cfg["root"]/name/"recon_1to1", "recon_v3": run}

def compose_grid(rows, labels, output):
    from PIL import ImageDraw, ImageFont
    images = [[Image.open(path).convert("RGB") for path in row] for row in rows]
    images = [[image.crop(image.point(lambda v: 255-v).getbbox() or (0, 0, *image.size)) for image in row] for row in images]
    cell = (max(image.width for row in images for image in row), max(image.height for row in images for image in row))
    scale = min(700/cell[0], 420/cell[1])
    width, height = round(cell[0]*scale), round(cell[1]*scale)
    canvas = Image.new("RGB", (len(images[0])*(width+30)+30, len(images)*(height+80)+20), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for r, row in enumerate(images):
        for c, image in enumerate(row):
            factor = min(width/image.width, height/image.height)
            image = image.resize((max(round(image.width*factor), 1), max(round(image.height*factor), 1)), Image.LANCZOS)
            x, y = 30+c*(width+30), 70+r*(height+80)
            canvas.paste(image, (x+(width-image.width)//2, y+(height-image.height)//2))
            draw.text((x, y-50), labels[r][c], fill="black", font=font)
    canvas.save(output)

def compose_case(cfg, name, case):
    import trimesh as mesh_io
    from tools.neural_study import render_views as shaded_views
    views = list(RUN_SETTINGS["views"])
    manafly = cfg["root"]/"manafly"
    manafly.mkdir(parents=True, exist_ok=True)
    missing = {key: (tuple(direction), tuple(up)) for key, (direction, up) in RUN_SETTINGS["views"].items() if not (cfg["manafly_renders"]/f"{key}.png").is_file()}
    for key in set(views)-set(missing):
        shutil.copy2(cfg["manafly_renders"]/f"{key}.png", manafly/f"{key}.png")
    if missing:
        shaded_views(mesh_io.load_mesh(cfg["manafly_stl"], process=True), manafly, missing)
    found = paths(cfg, name)
    rows = [[found["raw"]/"renders"/f"{key}.png" for key in views], [found["recon_1to1"]/"renders"/f"{key}.png" for key in views], [found["recon_v3"]/"renders"/f"{key}.png" for key in views], [manafly/f"{key}.png" for key in views]]
    titles = [f"{case['label']} raw", f"{case['label']} recon 1:1", f"{case['label']} recon v3", "ManaFly"]
    compose_grid(rows, [[f"{title} ({key})" for key in views] for title in titles], cfg["root"]/f"{name}_4views.png")

def body_row(evaluation, reconstruction):
    fallback = Path(evaluation).parent.parent/"evaluation_fallback"/"fea.json"
    fallback = json.loads(fallback.read_text(encoding="utf-8")) if fallback.is_file() else {}
    evaluation = json.loads(Path(evaluation).read_text(encoding="utf-8")) if Path(evaluation).is_file() else {}
    fea, geometry, walls = evaluation.get("fea") or {}, evaluation.get("geometry") or {}, evaluation.get("walls") or {}
    recon = json.loads(Path(reconstruction).read_text(encoding="utf-8")) if reconstruction and Path(reconstruction).is_file() else {}
    return {"mass_g": (geometry.get("mass") or {}).get("frame_mass_g"), "fea_mass_g": fea.get("frame_mass_g"), "arm_tip_n_per_mm": fea.get("stiffness_n_per_mm"), "f1_hz": (fea.get("eigenfrequencies_hz") or [None])[0],
            "fea_status": fea.get("status"), "fea_diagnostics": fea.get("diagnostics"), "fea_mesh": (fea.get("fea_surface") or {}).get("choice"), "bodies": (geometry.get("form") or {}).get("mesh_bodies"),
            "wall_deep_fraction": walls.get("deep_fraction"), "wall_deep_components": walls.get("deep_components"), "wall_largest_deep_mm3": walls.get("largest_deep_mm3"),
            "fallback_status": fallback.get("status"), "fallback_mass_g": fallback.get("frame_mass_g"), "fallback_arm_tip_n_per_mm": fallback.get("stiffness_n_per_mm"), "fallback_f1_hz": (fallback.get("eigenfrequencies_hz") or [None])[0],
            "members": recon.get("members"), "shells": recon.get("shells"), "nodes": recon.get("nodes"), "missed": (evaluation.get("assessment") or {}).get("missed"), "line": evaluation.get("line")}

def table_case(cfg, name, case):
    found = paths(cfg, name)
    v3 = json.loads((found["recon_v3"]/"reconstruction"/"reconstruction.json").read_text(encoding="utf-8"))
    rows = {"raw": body_row(found["raw"]/"evaluation"/"evaluation.json", None), "recon_1to1": body_row(found["recon_1to1"]/"evaluation"/"evaluation.json", Path(case["recon"])/"reconstruction"/"reconstruction.json"),
            "recon_v3": body_row(found["recon_v3"]/"evaluation"/"evaluation.json", found["recon_v3"]/"reconstruction"/"reconstruction.json")}
    rows["raw"].update(members=v3["members"], shells=v3["shells"], nodes=v3["nodes"])
    base = rows["raw"]
    for row in rows.values():
        row["relative_to_raw"] = {key: None if row[key] is None or not base[key] else row[key]/base[key]-1 for key in ("mass_g", "arm_tip_n_per_mm", "f1_hz", "fallback_arm_tip_n_per_mm", "fallback_f1_hz")}
    record = {"case": name, "label": case["label"], "bodies": rows, "within_10_percent": {"standard": all(rows["recon_v3"]["relative_to_raw"][key] is not None and abs(rows["recon_v3"]["relative_to_raw"][key]) <= 0.10 for key in ("mass_g", "arm_tip_n_per_mm", "f1_hz")),
                                    "fallback": all(rows["recon_v3"]["relative_to_raw"][key] is not None and abs(rows["recon_v3"]["relative_to_raw"][key]) <= 0.10 for key in ("mass_g", "fallback_arm_tip_n_per_mm", "fallback_f1_hz"))}, "fallback": cfg["fallback"],
              "wall_rule_not_worse_than_1to1": all(rows["recon_v3"][key] is not None and rows["recon_1to1"][key] is not None and rows["recon_v3"][key] <= rows["recon_1to1"][key] for key in ("wall_deep_fraction", "wall_largest_deep_mm3")),
              "bumps": v3["bump_summary"], "bump_rows": v3["bumps_by_body"], "splines": {key: v3[key] for key in ("control_points", "transition_radius_mm", "joint_sections", "mass_budget", "section_source", "continuity", "load_paths")}, "image": str(cfg["root"]/f"{name}_4views.png"),
              "stl": str(cfg["viewer"]/f"recon3_{name}"/"geometry.stl")}
    _stage(cfg["root"]/f"{name}_summary.json", record)
    return record

BUMP_STYLE = {"prescribed": ("#2a78d6", "s"), "load_point": ("#eb6834", "D"), "junction": ("#1baf7a", "o"), "grid_artefact": ("#eda100", "^"), "optimizer_feature": ("#e87ba4", "v")}

def bump_figure(cfg, name, case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    found = paths(cfg, name)
    record = json.loads((found["recon_v3"]/"reconstruction"/"reconstruction.json").read_text(encoding="utf-8"))
    meshes = {"raw": Path(case["raw"])/"frame.stl", "recon_1to1": Path(case["recon"])/"frame.stl", "recon_v3": found["recon_v3"]/"frame.stl"}
    titles = {"raw": "raw", "recon_1to1": "recon 1:1", "recon_v3": "recon v3"}
    figure, axes = plt.subplots(2, 3, figsize=(18, 9.5), gridspec_kw={"height_ratios": [3, 1]})
    for column, (label, path) in enumerate(meshes.items()):
        mesh = trimesh.load_mesh(path, process=False)
        for row, (i, j) in enumerate(((0, 1), (0, 2))):
            ax = axes[row, column]
            ax.add_collection(PolyCollection(mesh.triangles[:, :, [i, j]], facecolors="#d4d3cc", edgecolors="none"))
            for kind, (color, marker) in BUMP_STYLE.items():
                rows = [bump for bump in record["bumps_by_body"][label] if bump["class"] == kind]
                if rows:
                    points = np.array([bump["point_mm"] for bump in rows])
                    ax.scatter(points[:, i], points[:, j], s=[40+120*bump["excess_mm"] for bump in rows], c=color, marker=marker, edgecolors="white", linewidths=1.0, label=f"{kind.replace('_', ' ')} ({len(rows)})", zorder=3)
            ax.set_xlim(mesh.bounds[0][i]-3, mesh.bounds[1][i]+3)
            ax.set_ylim(mesh.bounds[0][j]-3, mesh.bounds[1][j]+3)
            ax.set_aspect("equal")
            ax.tick_params(colors="#55554f", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#c3c2b7")
            ax.set_xlabel("x mm", color="#55554f", fontsize=9)
            ax.set_ylabel(("y" if row == 0 else "z")+" mm", color="#55554f", fontsize=9)
            if row == 0:
                ax.set_title(f"{case['label']} {titles[label]}: section bumps > {record['config']['bump_minimum_mm']} mm", fontsize=11, color="#1a1a19", loc="left")
                ax.legend(loc="lower left", fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(cfg["root"]/f"{name}_bumps.png", dpi=110)
    plt.close(figure)


def refea(cfg, name):
    found = paths(cfg, name)
    for label, folder in found.items():
        output = folder/"evaluation"
        if (json.loads((output/"evaluation.json").read_text(encoding="utf-8")).get("fea") or {}).get("status") == "ok":
            continue
        frame = str(output/"frame.json")
        with (output/"fea_rerun.log").open("w", encoding="utf-8") as log:
            subprocess.call([cfg["python"], RUN_SETTINGS["compute"], "fea_modal", "--cwd", str(ROOT), "--", cfg["python"], str(ROOT/"tools"/"evaluate_frame.py"), "fea", frame], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            subprocess.call([cfg["python"], str(ROOT/"tools"/"evaluate_frame.py"), "report", frame], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if label == "recon_v3":
            shutil.copy2(output/"evaluation.json", folder/"evaluation.json")

def fallback_fea(cfg, name):
    for label, folder in paths(cfg, name).items():
        output = folder/"evaluation_fallback"
        if (output/"fea.json").is_file() and json.loads((output/"fea.json").read_text(encoding="utf-8")).get("status") == "ok":
            continue
        spec = json.loads((folder/"evaluation"/"frame.json").read_text(encoding="utf-8"))
        spec.update(output=str(output), fea_settings={**spec.get("fea_settings", {}), **cfg["fallback"]["fea_settings"]})
        spec.pop("datasheet", None)
        frame = _stage(output/"frame.json", spec)
        with (output/"fea.log").open("w", encoding="utf-8") as log:
            subprocess.call([cfg["python"], RUN_SETTINGS["compute"], cfg["fallback"]["compute"], "--cwd", str(ROOT), "--", cfg["python"], str(ROOT/"tools"/"evaluate_frame.py"), "fea", str(frame)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

def study_main(overrides):
    cfg = {key: Path(value) if RECON3_KINDS[key] == "path" else value for key, value in configure(RECON3_STUDY, RECON3_KINDS, overrides).items()}
    cfg["root"] = cfg["root"] if cfg["root"].is_absolute() else ROOT/cfg["root"]
    cases = {name: case for name, case in cfg["cases"].items() if not cfg["only"] or name in cfg["only"]}
    if "run" in cfg["steps"] or "evaluate" in cfg["steps"]:
        jobs = [job for name, case in cases.items() for job in (evaluate_bodies(cfg, name, case) if "evaluate" in cfg["steps"] else [])]
        for name, case in cases.items():
            if "run" in cfg["steps"]:
                spline_run(cfg, name, case)
        failed = [job.args for job in jobs if job.wait()]
        if failed:
            print(json.dumps({"failed": [str(args) for args in failed]}), flush=True)
    for name, case in cases.items():
        if "refea" in cfg["steps"]:
            refea(cfg, name)
        if "fallback" in cfg["steps"]:
            fallback_fea(cfg, name)
        if "compose" in cfg["steps"]:
            compose_case(cfg, name, case)
        if "figure" in cfg["steps"]:
            bump_figure(cfg, name, case)
        if "table" in cfg["steps"]:
            record = table_case(cfg, name, case)
            print(json.dumps({name: {body: {key: row[key] for key in ("mass_g", "arm_tip_n_per_mm", "f1_hz", "fallback_arm_tip_n_per_mm", "fallback_f1_hz", "bodies", "wall_deep_fraction", "wall_deep_components", "wall_largest_deep_mm3", "members", "nodes", "relative_to_raw")} for body, row in record["bodies"].items()}}, default=float), flush=True)

def cable_section(mesh, before, path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    free = np.flatnonzero(~path["inside"] & ~path["guide"])
    i = int(free[len(free)//2])
    origin, normal, u, v, profile = path["points"][i], path["tangent"][i], path["u"][i], path["v"][i], path["profiles"][i]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for body, style, label in ((before, dict(color="0.6", lw=1.2, ls="--"), "ohne Kanal"), (mesh, dict(color="k", lw=1.6), "mit Kanal")):
        segments = trimesh.intersections.mesh_plane(body, normal, origin)
        local = np.stack([(segments-origin)@u, (segments-origin)@v], axis=-1)
        local = local[np.all(np.abs(local) < 9, axis=(1, 2))]
        for k, segment in enumerate(local):
            ax.plot(segment[:, 0], segment[:, 1], **style, label=label if k == 0 else None)
    for polygon, color, label in ((profile.cavity(), "tab:blue", "Innenraum (Tropfen)"), (profile.slot_cut(path["slot_depth"][i]), "tab:red", "Klemmschlitz"), (profile.shell(), "tab:green", "Kanalschale")):
        closed = np.vstack([polygon, polygon[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=0.8, ls=":", label=label)
    ax.add_patch(plt.Circle((0, 0), profile.bundle/2, color="tab:orange", alpha=0.35, label=f"Bündel {profile.bundle:.2f} mm"))
    size = profile.size
    ax.set_title(f"{path['name']} bei s = {i*CABLES['sample_mm']:.0f} mm, Glied {int(path['owner'][i])}\ninnen {size['inner_mm']:.2f} / Schlitz {size['slot_mm']:.2f} / Lippe {size['lip_mm']:.2f} / Schale {size['outer_width_mm']:.1f} x {size['outer_height_mm']:.1f} mm", fontsize=9)
    ax.set_xlabel("quer [mm]")
    ax.set_ylabel("oben = Druckachse/Propseite [mm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def cables_main(overrides):
    import pickle
    import psutil
    config = configure(CABLE_RUN_CONFIG, CABLE_RUN_KINDS, overrides, ("source", "output", "body"))
    cables = configure(CABLES, CABLES_KINDS, config["cables"])
    out = config["output"]
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    density = np.load(config["source"])["density"]
    domain = stored_domain(config["domain"], density.shape) if config["domain"] else full_domain(config)
    cache = out/"graph.pkl"
    if cache.is_file():
        graph, rods = pickle.loads(cache.read_bytes())
    else:
        source = config["section_body"] or config["source"].with_name("geometry.stl")
        graph, rods = spline_graph(domain, density, config, trimesh.load_mesh(source, process=True) if Path(source).is_file() else None)[:2]
        cache.write_bytes(pickle.dumps((graph, rods)))
    graph_peak = getattr(psutil.Process().memory_info(), "peak_wset", 0)/2**30
    body = trimesh.load_mesh(config["body"], process=True)
    channels = CableChannels(graph, rods, domain, cables)
    peak = lambda: round(getattr(psutil.Process().memory_info(), "peak_wset", 0)/2**30, 2)
    peaks = {"graph": graph_peak}
    channels.plan(body)
    peaks["plan"] = peak()
    mesh, applied = channels.apply(body, core_domain(domain, config))
    peaks["apply"] = peak()
    report = {**channels.report(mesh, body), "apply": applied, "body": str(config["body"]), "mass_before_g": float(body.volume)*domain["material"]["density_g_cm3"]/1000, "mass_after_g": float(mesh.volume)*domain["material"]["density_g_cm3"]/1000}
    mesh.export(out/"frame.stl")
    for path in channels.paths:
        if path["routed"] and path["name"] in config["section_paths"]:
            cable_section(mesh, body, path, out/f"section_{path['name']}.png")
    peaks["report"] = peak()
    render_views(mesh, out, CABLE_VIEWS)
    peaks["render"] = peak()
    report["peak_gb"] = peaks
    report["runtime_s"], report["peak_rss_gb"] = perf_counter()-started, peak()
    (out/"cables.json").write_text(json.dumps({**report, "config": {"cables": cables, "body": str(config["body"]), "source": str(config["source"])}}, indent=1, default=lambda value: value.tolist() if hasattr(value, "tolist") else float(value) if isinstance(value, np.floating) else str(value)), encoding="utf-8")
    print(json.dumps({"steckbrief": report["steckbrief"], "printability": report["printability"], "mass_g": [report["mass_before_g"], report["mass_after_g"]], "apply": applied, "peak_rss_gb": report["peak_rss_gb"], "peak_gb": peaks}, default=float), flush=True)

def main(argv=None):
    return command_line({"build": build_main, "splines": splines_main, "cables": cables_main, "study": study_main, "fea": fea_main, "render": render_main, "compose": compose_main}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
