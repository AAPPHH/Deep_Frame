import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes
from skimage.morphology import convex_hull_image, skeletonize

from deep_frame.config import CRASH_DIRECTIONS, command_line, configure
from deep_frame.topology_geometry import build_design_domain, symmetric_domains
from deep_frame.topology_neural import _sigmoid, cell_centers, member_widths, neural_settings, optimize_neural, volume_shift
from tools.topology_study import study_parameters

SHAPE, FINE_SHAPE = [102, 96, 24], [204, 192, 48]
NEURAL = {"max_frequency_per_mm": 0.16, "max_iterations": 110, "minimum_iterations": 40, "sharpness_iterations": 60, "sharpness_final": 8.0, "max_width_penalty": 0.0}
RENDER = {"sigma_cells": 1.0, "threshold": 0.5, "taubin": 12, "carve_bores": True, "min_component_mm3": 30}
RUN_CONFIG = {"name": "neural_mc_v05", "output": None, "volume_fraction": 0.05, "crash_directions": CRASH_DIRECTIONS, "viewer": None}
RUN_KINDS = {"name": "text", "output": "path", "volume_fraction": "float", "crash_directions": [tuple(CRASH_DIRECTIONS)], "viewer": "path"}
COMPARE_CONFIG = {"inputs": None, "labels": None, "output": None}
COMPARE_KINDS = {"inputs": ["path"], "labels": ["text"], "output": "path"}
def carve(field, grid, regions):
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    axes = [np.asarray(grid["origin_mm"])[i] + (np.arange(grid["shape"][i]) + 0.5) * spacing[i] for i in range(3)]
    for region in regions:
        if region["kind"] != "cylinder" or region.get("rasterize", True) or region.get("role", "forbidden") != "forbidden":
            continue
        axis = {"x": 0, "y": 1, "z": 2}[region["axis"]]
        radial = [i for i in range(3) if i != axis]
        center = np.asarray(region["center_mm"], dtype=float)
        a, b = np.meshgrid(axes[radial[0]] - center[radial[0]], axes[radial[1]] - center[radial[1]], indexing="ij")
        inside = np.abs(axes[axis] - center[axis]) <= region["height_mm"] / 2
        ramp = np.clip(0.5 + (np.hypot(a, b) - region["radius_mm"]) / spacing[radial[0]], 0, 1)
        index = [slice(None)] * 3
        index[axis] = inside
        view = np.moveaxis(field[tuple(index)], axis, 2)
        view = np.minimum(view, ramp[:, :, None])
        field[tuple(index)] = np.moveaxis(view, 2, axis)
    return field

def surface(density, grid, cfg, regions=()):
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    field = gaussian_filter(np.asarray(density, dtype=np.float32), cfg.get("sigma_cells", 0.5)) if cfg.get("sigma_cells", 0.5) > 0 else np.asarray(density, dtype=np.float32)
    field = carve(field, grid, regions) if cfg.get("carve_bores") else field
    field = np.pad(field, 1)
    vertices, faces, _, _ = marching_cubes(field, cfg.get("threshold", 0.5), spacing=tuple(spacing), allow_degenerate=False)
    vertices = vertices + np.asarray(grid["origin_mm"]) - spacing / 2
    mesh = trimesh.Trimesh(vertices, faces[:, ::-1], process=True)
    dropped = {"count": 0, "volume_mm3": 0.0}
    if cfg.get("min_component_mm3", 0) > 0:
        parts = mesh.split(only_watertight=False)
        small = [part for part in parts if abs(part.volume) < cfg["min_component_mm3"]]
        dropped = {"count": len(small), "volume_mm3": float(sum(abs(part.volume) for part in small))}
        mesh = trimesh.util.concatenate([part for part in parts if abs(part.volume) >= cfg["min_component_mm3"]])
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=cfg.get("taubin", 8))
    if mesh.volume < 0:
        mesh.invert()
    return mesh, dropped

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
    for name, (direction, up) in views.items():
        render(mesh, direction, up).save(out / f"{name}.png")

def domains(shape, crash_directions=()):
    parameters = study_parameters(shape)
    parameters["integration"]["crash_directions"] = list(crash_directions)
    domain = build_design_domain(parameters)
    return symmetric_domains(domain)

def base_settings(domain, overrides):
    keys = ("interface_node_policy", "case_weights", "penalization", "min_stiffness_ratio")
    return neural_settings({**{key: domain["optimizer_settings"][key] for key in keys}, "linear_solver": "cuda_cudss", **overrides})
def sample_full(mapping, fine, sharpness, fraction):
    allowed, preserve = fine["allowed"], fine["preserve"]
    free = allowed & ~preserve
    points = cell_centers(fine["grid"]).reshape(*fine["grid"]["shape"], 3)[free]
    logits = np.concatenate([mapping.field.forward(mapping.field.features(chunk))[0] for chunk in np.array_split(points, max(1, len(points) // 200000))])
    density = preserve.astype(float)
    density[free] = _sigmoid(sharpness * (logits + volume_shift(logits, sharpness, fraction * np.count_nonzero(allowed) - np.count_nonzero(preserve))))
    return density

def finish(out, density, fine_full, render_cfg):
    mesh, dropped = surface(density, fine_full["grid"], render_cfg, fine_full["regions"])
    mesh.export(out / "geometry.stl")
    render_views(mesh, out)
    allowed_volume = np.count_nonzero(fine_full["allowed"]) * np.prod(fine_full["grid"]["spacing_mm"])
    return mesh, {"mass_g": float(mesh.volume * 1.09 / 1000), "mesh_volume_mm3": float(mesh.volume), "volume_fraction_mesh": float(mesh.volume / allowed_volume),
                  "watertight": bool(mesh.is_watertight), "bodies": len(mesh.split(only_watertight=False)), "components_dropped": dropped}

def run(config):
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    full, half = domains(SHAPE, config["crash_directions"])
    settings = base_settings(half, {**NEURAL, "volume_fraction": config["volume_fraction"]})
    log = (out / "iterations.jsonl").open("w")
    holder = {}
    import deep_frame.topology_neural as tn
    original = tn.NeuralDensity
    class Capture(original):
        def __init__(self, *args):
            super().__init__(*args)
            holder["mapping"] = self
    tn.NeuralDensity = Capture
    try:
        result = optimize_neural(half, settings, progress_callback=lambda e: (log.write(json.dumps({k: e.get(k) for k in ("iteration", "objective", "volume_fraction", "sharpness", "elapsed_s")}) + "\n"), log.flush()))
    finally:
        tn.NeuralDensity = original
    log.close()
    optimized = perf_counter() - started
    if result["status"] != "ok":
        print(json.dumps({"variant": config["name"], "status": result["status"], "diagnostics": result["diagnostics"]}), flush=True)
        return 1
    summary = result["summary"]
    (out / "solver.json").write_text(json.dumps({key: summary[key] for key in ("system", "normalization_compliances_n_mm", "normalized_case_weights", "static_surrogate_metrics", "stop_reason")}, indent=1, default=str))
    np.savez_compressed(out / "density_half.npz", density=result["density"])
    fine_full, _ = domains(FINE_SHAPE, config["crash_directions"])
    density = sample_full(holder["mapping"], fine_full, summary["sharpness_final"], settings["volume_fraction"])
    np.savez_compressed(out / "density_fine.npz", density=density.astype(np.float32))
    mesh, info_mesh = finish(out, density, fine_full, RENDER)
    if config["viewer"] is not None:
        Path(config["viewer"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "geometry.stl", config["viewer"])
    info = {"variant": config["name"], "method": "neural", "crash_directions": list(config["crash_directions"]), "volume_fraction_target": settings["volume_fraction"], "volume_fraction_opt": summary["volume_fraction"],
            "iterations": summary["iterations"], "stop_reason": summary["stop_reason"], "optimize_runtime_s": optimized, "total_runtime_s": perf_counter() - started, **info_mesh,
            "grid_opt_half": half["grid"], "grid_render_full": fine_full["grid"], "neural": {k: settings[k] for k in ("max_frequency_per_mm", "frequencies", "hidden", "learning_rate", "sharpness_final", "sharpness_iterations", "max_iterations", "max_width_penalty")},
            "render": RENDER, "objective_final": summary["objective_final"], "factorization_groups": summary["system"]["factorization_groups"]}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("variant", "iterations", "stop_reason", "optimize_runtime_s", "mass_g", "bodies")}), flush=True)
    return 0

def cross_sections(field, domain):
    spacing = np.asarray(domain["grid"]["spacing_mm"], dtype=float)
    solid = field >= 0.5
    skeleton = (skeletonize(solid) > 0) & ~binary_dilation(domain["preserve"] | domain["forbidden"], iterations=2)
    width = np.stack([2 * distance_transform_edt(solid[:, :, k], sampling=spacing[:2]) for k in range(solid.shape[2])], 2)[skeleton] - spacing[0]
    run = np.zeros(solid.shape, dtype=int)
    for k in range(solid.shape[2]):
        run[:, :, k] = (run[:, :, k - 1] + 1) * solid[:, :, k] if k else solid[:, :, k]
    total = run.copy()
    for k in range(solid.shape[2] - 2, -1, -1):
        total[:, :, k] = np.where(solid[:, :, k] & solid[:, :, k + 1], total[:, :, k + 1], total[:, :, k])
    height = total[skeleton] * spacing[2]
    ratio = width / height
    footprint = solid.any(2)
    percentiles = lambda values: {f"p{q}": float(np.percentile(values, q)) for q in (10, 50, 90)}
    return {"skeleton_points": int(skeleton.sum()), "horizontal_width_mm": percentiles(width), "vertical_height_mm": percentiles(height), "width_height_ratio": percentiles(ratio),
            "ribbon_fraction_ratio_above_2": float(np.mean(ratio > 2)), "web_fraction_ratio_below_half": float(np.mean(ratio < 0.5)), "median_abs_log2_ratio": float(np.median(np.abs(np.log2(ratio)))),
            "member_width_3d": member_widths(field, spacing, domain["preserve"] | domain["forbidden"]),
            "top_openness_envelope": float(1 - footprint.sum() / domain["allowed"].any(2).sum()), "top_openness_hull": float(1 - footprint.sum() / convex_hull_image(footprint).sum())}

def timing(path):
    elapsed = [json.loads(line)["elapsed_s"] for line in Path(path).read_text().splitlines() if line.strip()]
    return {"first_iteration_s": elapsed[0], "per_iteration_s": (elapsed[-2] - elapsed[0]) / (len(elapsed) - 2), "iterations": len(elapsed) - 1}

def compare(config):
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    fine_full, _ = domains(FINE_SHAPE)
    rows = {}
    for label, directory in zip(config["labels"], config["inputs"]):
        info = json.loads((directory / "info.json").read_text())
        field = gaussian_filter(np.load(directory / "density_fine.npz")["density"].astype(np.float32), RENDER["sigma_cells"])
        rows[label] = {"source": str(directory), "mass_g": info["mass_g"], "bodies": info["bodies"], "iterations": info["iterations"], "stop_reason": info["stop_reason"], "optimize_runtime_s": info["optimize_runtime_s"],
                       **timing(directory / "iterations.jsonl"), **cross_sections(field, fine_full)}
    (out / "metrics.json").write_text(json.dumps(rows, indent=1))
    stitch(config["inputs"], config["labels"], out)
    print(json.dumps({label: {key: row[key] for key in ("mass_g", "bodies", "per_iteration_s", "width_height_ratio", "top_openness_envelope")} for label, row in rows.items()}), flush=True)
    return 0

def stitch(inputs, labels, output):
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=44)
    tiles = []
    for view in ("iso", "top", "side"):
        images = [Image.open(directory / f"{view}.png").convert("RGB") for directory in inputs]
        row = Image.new("RGB", (sum(image.width for image in images), images[0].height), "white")
        for index, (image, label) in enumerate(zip(images, labels)):
            row.paste(image, (index * image.width, 0))
            ImageDraw.Draw(row).text((index * image.width + 30, 24), f"{label} - {view}", fill="black", font=font)
        tiles.append(row)
    sheet = Image.new("RGB", (tiles[0].width, sum(tile.height for tile in tiles)), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, index * tile.height))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(output / "side_by_side.png")

def main(argv=None):
    return command_line({"run": lambda overrides: run(configure(RUN_CONFIG, RUN_KINDS, overrides, ("output",))),
                         "compare": lambda overrides: compare(configure(COMPARE_CONFIG, COMPARE_KINDS, overrides, ("inputs", "labels", "output")))}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
