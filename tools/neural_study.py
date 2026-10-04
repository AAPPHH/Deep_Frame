import json
from copy import deepcopy
import shutil
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter, label, zoom
from skimage.measure import marching_cubes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deep_frame.topology_neural as topology_neural
from deep_frame.config import CRASH_DIRECTIONS, command_line
from deep_frame.frame import motor_positions, prop_plane_z
from deep_frame.topology_geometry import _merge, build_design_domain, grid_centers, mirror_field, region_contains, symmetric_domains
from deep_frame.topology_neural import member_widths, neural_settings, optimize_neural
from deep_frame.topology_optimization import HexElasticity, ModalConstraint, StiffnessConstraint, _settings, optimize_topology
from deep_frame.topology_reconstruction import hoop_paths
from tools.multi_crash_study import cross_sections, stitch
from tools.topology_study import study_parameters

STUDY = {
    "root": "exports/r3",
    "viewer_root": "C:/clones/Deep_Frame-neural/exports",
    "viewer_prefix": "r3_",
    "crash_directions": list(CRASH_DIRECTIONS),
    "inertia_relief": {"cases": ["arm_tip", "thrust_all", "crash_"], "attachments": {"battery": "battery_rail_", "aio15": "aio_contact_", "camera": "camera_mount_", "motor_": "_motor_contact", "prop_": "_motor_contact"}, "frame_mass": "target"},
    "verify_shape": [68, 64, 24],
    "chord": {"half_width_mm": 20.0, "z_max_mm": 10.0, "y_span_mm": [-25.0, 25.0]},
    "compare": {"labels": ["neural_v05", "mcrash_v05", "r2_v05", "r3_v05", "ManaFly"], "inputs": ["C:/clones/Deep_Frame-neural/exports/fast/neural_v05", "C:/clones/Deep_Frame-mcrash/exports/mcrash/neural_v05", "C:/clones/Deep_Frame-nr2/exports/r2/neural_r2_v05", "exports/r3/neural_r3_v05", "C:/clones/Deep_Frame-neural/exports/fast/_manafly_same_renderer"], "output": "exports/r3/compare"},
    "shape": [102, 96, 24],
    "fine_shape": [204, 192, 48],
    "pad": {"top_mm": 28 / 3, "thickness_mm": 8 / 3, "support_half_mm": 5.0, "bore_margin_mm": 0.5},
    "hoop": {"x_mm": 12.0, "radius_mm": 1.6, "path_yz_mm": [[24, 27.5], [34, 26], [42, 23.5], [46.5, 18], [47.5, 11], [45.5, 5], [41, 2], [33, 1.5]],
             "load_y_min_mm": 44.0, "load_z_mm": [4.0, 22.0], "case_weight": 1.0},
    "neural": {"max_frequency_per_mm": 0.2, "max_iterations": 110, "minimum_iterations": 40, "sharpness_iterations": 80, "sharpness_final": 8.0, "max_width_penalty": 0.0, "max_runtime_s": 1500.0},
    "render": {"sigma_cells": 1.0, "threshold": 0.5, "taubin": 12, "carve_bores": True, "keep": "motor_pads", "min_body_mm3": 1.0, "sample_sharpness": 32.0, "flatten": ["battery_rail_", "battery_contact"]},
    "prop_discs": {"mode": None, "weight": 3.0, "length_mm": 10.0, "corridor_half_width_mm": 4.0, "hub_margin_mm": 2.0},
    "modal": {"f1_min_hz": None, "case": "modes", "modes": 4, "tracked": 2, "initial_iterations": 30, "warm_iterations": 2, "penalty": 10.0, "ks": 40.0, "mass_cutoff": 0.1, "multiplier_interval": 5, "start_iteration": 1},
    "stiffness": {"min_n_per_mm": None, "case": "stiffness_arm_tip", "source_case": "arm_tip", "calibration": 1.0, "penalty": 10.0, "multiplier_interval": 5, "feasibility_tolerance": 0.02},
    "method": "neural",
    "calibration": {"inputs": [], "output": "exports/r4/modal_calibration.json", "linear_solver": "cpu_superlu"},
    "simp": {"filter_radius_mm": 4.0, "projection": "single", "beta_schedule": [1.0, 2.0, 4.0, 8.0], "beta_interval": 35, "max_iterations": 140, "minimum_iterations": 20, "move_limit": 0.1, "max_runtime_s": 1500.0},
    "variants": [{"name": "neural_r3_v05", "neural": {"volume_fraction": 0.05}}],
}
VARIANT_KEYS = ("prop_discs", "modal", "stiffness", "method", "simp")

def configure(overrides):
    unknown = sorted(set(overrides) - set(STUDY))
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(unknown))
    return _merge(STUDY, overrides)

def polyline_distance(points, path):
    best = np.full(points.shape[:-1], np.inf)
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        t = np.clip(((points - start) @ segment) / (segment @ segment), 0, 1)
        best = np.minimum(best, np.linalg.norm(points - start - t[..., None] * segment, axis=-1))
    return best

def bounds(region):
    if region["kind"] == "box":
        return {"kind": "box", "min_mm": list(region["min_mm"]), "max_mm": list(region["max_mm"])}
    half = np.full(3, region["radius_mm"])
    half[{"x": 0, "y": 1, "z": 2}[region.get("axis", "z")]] = region["height_mm"] / 2
    return {"kind": "box", "min_mm": (np.asarray(region["center_mm"]) - half).tolist(), "max_mm": (np.asarray(region["center_mm"]) + half).tolist()}

def lower_chord(density, grid, cfg, sigma=1.0, threshold=0.5):
    solid = gaussian_filter(np.asarray(density, dtype=np.float32), sigma) > threshold
    centers = grid_centers(grid)
    low, high = cfg["y_span_mm"]
    box = (np.abs(centers[..., 0]) <= cfg["half_width_mm"]) & (centers[..., 2] <= cfg["z_max_mm"]) & (centers[..., 1] >= low) & (centers[..., 1] <= high)
    labels, _ = label(solid & box, np.ones((3, 3, 3), dtype=bool))
    y = centers[..., 1]
    spanning = (set(np.unique(labels[box & (y <= low + 2)])) & set(np.unique(labels[box & (y >= high - 2)]))) - {0}
    slices = np.unique(np.round(y[box], 6))
    covered = [bool((solid & box & (np.abs(y - value) < 1e-6)).any()) for value in slices]
    return {"box": cfg, "spanning_components": len(spanning), "lower_chord": bool(spanning), "y_slice_coverage": float(np.mean(covered)),
            "solid_volume_mm3": float(np.count_nonzero(solid & box) * np.prod(grid["spacing_mm"]))}

class R2Domain:
    def __init__(self, cfg):
        self.pad, self.hoop, self.crash, self.relief, self.discs, self.stiffness = cfg["pad"], cfg["hoop"], cfg["crash_directions"], cfg["inertia_relief"], cfg["prop_discs"], cfg["stiffness"]
    def hoop_paths(self):
        return hoop_paths(self.hoop)
    def masses(self, domain, fraction):
        components, regions = domain["metadata"]["components"], {region["name"]: region for region in domain["regions"]}
        items = []
        for name, component in components.items():
            prefix = next((key for key in self.relief["attachments"] if name.startswith(key)), None)
            if prefix is None or component["mass_g"] <= 0:
                continue
            suffix = self.relief["attachments"][prefix]
            targets = [name.split("_", 1)[1] + suffix] if suffix.startswith("_") else [key for key in regions if key.startswith(suffix)]
            items += [{"name": name, "region": bounds(regions[key]), "mass_g": component["mass_g"] / len(targets)} for key in targets]
        frame = fraction * domain["metadata"]["allowed_volume_mm3"] * domain["material"]["density_g_cm3"] / 1000 if self.relief["frame_mass"] == "target" else 0.0
        return {"point_masses": items, "preserve_mass_g": frame}
    def prop_discs(self, parameters):
        motors = np.array(list(motor_positions(parameters).values()))
        return {**self.discs, "motors_mm": motors.tolist(), "radius_mm": parameters["components"]["prop"]["diameter_mm"] / 2, "plane_mm": prop_plane_z(parameters),
                "pad_radius_mm": parameters["components"]["motor"]["diameter_mm"] / 2 + self.discs["hub_margin_mm"]}
    def keep_out(self, centers, discs):
        blocked = np.zeros(centers.shape[:-1], dtype=bool)
        for x, y in discs["motors_mm"]:
            offset = centers[..., :2] - [x, y]
            radial = np.linalg.norm(offset, axis=-1)
            direction = -np.asarray([x, y]) / np.hypot(x, y)
            along = offset @ direction
            across = np.abs(offset @ [-direction[1], direction[0]])
            corridor = (along >= 0) & (across <= discs["corridor_half_width_mm"])
            blocked |= (radial <= discs["radius_mm"]) & (radial > discs["pad_radius_mm"]) & ~corridor & (centers[..., 2] <= discs["plane_mm"])
        return blocked
    def build(self, shape, fraction=0.05):
        parameters = study_parameters(shape)
        parameters["frame"]["arm_height_mm"] = self.pad["top_mm"]
        parameters["integration"]["crash_directions"] = list(self.crash)
        domain = build_design_domain(parameters)
        grid = domain["grid"]
        centers = grid_centers(grid)
        spacing = np.asarray(grid["spacing_mm"])
        bottom = self.pad["top_mm"] - self.pad["thickness_mm"]
        allowed, preserve = domain["allowed"].copy(), domain["preserve"].copy()
        for region in domain["regions"]:
            if region["name"].endswith("_motor_contact"):
                preserve &= ~(region_contains(centers, region) & (centers[..., 2] < bottom))
                region["center_mm"][2], region["height_mm"] = bottom + self.pad["thickness_mm"] / 2, self.pad["thickness_mm"]
            if "_motor_screw_" in region["name"] or region["name"].endswith("_shaft_clearance"):
                floor = grid["origin_mm"][2]
                below = {**region, "center_mm": [*region["center_mm"][:2], (floor - 1 + bottom) / 2], "height_mm": bottom - floor + 1, "radius_mm": region["radius_mm"] + self.pad["bore_margin_mm"]}
                allowed &= ~region_contains(centers, below, [spacing[0] / 2, spacing[1] / 2, 0])
        tube = np.zeros(allowed.shape, dtype=bool)
        for path in self.hoop_paths():
            tube |= polyline_distance(centers, path) <= self.hoop["radius_mm"]
        preserve = (preserve | tube) & allowed
        discs = self.prop_discs(parameters)
        unblocked = int(np.count_nonzero(allowed))
        if discs["mode"] == "hard":
            allowed &= ~(self.keep_out(centers, discs) & ~preserve)
            labels, _ = label(allowed)
            allowed &= np.isin(labels, np.unique(labels[preserve & allowed]))
        domain.update(allowed=allowed, preserve=preserve, forbidden=~allowed)
        if label(allowed)[1] != 1:
            raise ValueError("Round-2 domain is not face-connected")
        motors = np.array(list(motor_positions(parameters).values()))
        tolerance = parameters["integration"]["selection_tolerance_mm"]
        for case in domain["load_cases"]:
            for box in case.get("fixed_regions", []) + [load["region"] for load in case.get("loads", [])]:
                low, high = np.asarray(box["min_mm"], float), np.asarray(box["max_mm"], float)
                center = (low + high) / 2
                if np.min(np.linalg.norm(motors - center[:2], axis=1)) < 1 and high[2] - low[2] < 3 * tolerance:
                    z = bottom if abs(center[2]) < 1 else center[2]
                    box["min_mm"] = [center[0] - self.pad["support_half_mm"], center[1] - self.pad["support_half_mm"], z - tolerance]
                    box["max_mm"] = [center[0] + self.pad["support_half_mm"], center[1] + self.pad["support_half_mm"], z + tolerance]
        front = next(case for case in domain["load_cases"] if case["name"] == "crash_front")
        force = float(np.linalg.norm(front["loads"][0]["force_n"]))
        r, y_max = self.hoop["radius_mm"] + spacing[0], max(y for y, _ in self.hoop["path_yz_mm"]) + self.hoop["radius_mm"] + spacing[1]
        loads = [{"region": {"kind": "box", "min_mm": [sign * self.hoop["x_mm"] - r, self.hoop["load_y_min_mm"], self.hoop["load_z_mm"][0]], "max_mm": [sign * self.hoop["x_mm"] + r, y_max, self.hoop["load_z_mm"][1]]}, "force_n": [0.0, -force / 2, 0.0]} for sign in (-1, 1)]
        domain["load_cases"].append({"name": "crash_hoop", "analysis": "static", "fixed_regions": [dict(box) for box in front["fixed_regions"]], "loads": loads, "purpose": "Frontal crash on the camera hoop fronts"})
        domain["optimizer_settings"]["case_weights"]["crash_hoop"] = self.hoop["case_weight"]
        if self.stiffness["min_n_per_mm"]:
            source = next(case for case in domain["load_cases"] if case["name"] == self.stiffness["source_case"])
            domain["load_cases"].append({**deepcopy(source), "name": self.stiffness["case"], "purpose": "arm-tip stiffness constraint, evaluator support (centre mount undersides fixed)"})
        if self.relief:
            relief = self.masses(domain, fraction)
            for case in domain["load_cases"]:
                if case["analysis"] == "static" and any(case["name"].startswith(key) for key in self.relief["cases"]):
                    case.update(fixed_regions=[], inertia_relief=relief)
        domain["metadata"]["round4"] = {"prop_discs": discs, "allowed_cells": int(np.count_nonzero(allowed)), "unblocked_cells": unblocked}
        domain["metadata"]["round2"] = {"pad": self.pad, "hoop": self.hoop, "hoop_cells": int(np.count_nonzero(tube & allowed)), "motors_mm": motors.tolist(), "crash_directions": list(self.crash),
                                        "inertia_relief_cases": [case["name"] for case in domain["load_cases"] if "inertia_relief" in case], "inertia_relief_masses": relief if self.relief else None}
        return symmetric_domains(domain)

def base_settings(domain, overrides):
    keys = ("interface_node_policy", "case_weights", "penalization", "min_stiffness_ratio")
    return neural_settings({**{key: domain["optimizer_settings"][key] for key in keys}, "linear_solver": "cuda_cudss", **overrides})

def verify(cfg):
    _, half = R2Domain(cfg).build(cfg["verify_shape"])
    system = HexElasticity(half, interface_node_policy=half["optimizer_settings"]["interface_node_policy"])
    density = np.where(half["allowed"], 0.3, 0.0).ravel()
    density[half["preserve"].ravel()] = 1
    rows = {case["name"]: {**case["inertia_relief"], "reactions": system.support_reactions(density, case["name"])} for case in system.cases if "inertia_relief" in case}
    out = Path(cfg["root"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "inertia_relief_check.json").write_text(json.dumps({"shape": cfg["verify_shape"], "masses": half["metadata"]["round2"]["inertia_relief_masses"], "cases": rows}, indent=1))
    print(json.dumps({name: max(entry["max_reaction_n"] / entry["nodal_force_sum_n"] for entry in row["reactions"]) for name, row in rows.items()}), flush=True)

def time_solve(cfg):
    started = perf_counter()
    _, half = R2Domain(cfg).build(cfg["shape"])
    built = perf_counter()
    system = HexElasticity(half, interface_node_policy=half["optimizer_settings"]["interface_node_policy"], linear_solver="cuda_cudss")
    density = np.where(half["allowed"], 0.1, 0.0).ravel()
    density[half["preserve"].ravel()] = 1
    times = []
    for _ in range(2):
        start = perf_counter()
        system.solve(density, 3.0, 1e-6)
        times.append(perf_counter() - start)
    system.close()
    print(json.dumps({"shape": cfg["shape"], "half": half["grid"]["shape"], "cases": len(half["load_cases"]), "domain_s": built - started, "solve_s": times}), flush=True)

def upsample(half_density, fine):
    coarse = mirror_field(np.asarray(half_density, dtype=np.float32))
    density = np.clip(zoom(coarse, np.asarray(fine["grid"]["shape"]) / np.asarray(coarse.shape), order=1), 0, 1)
    return np.where(fine["preserve"], 1.0, np.where(fine["allowed"], density, 0.0))

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

def keep_connected(field, grid, threshold, anchor):
    labels, count = label(field > threshold, np.ones((3, 3, 3), dtype=bool))
    kept = sorted(set(np.unique(labels[anchor]).tolist()) - {0})
    dropped = np.isin(labels, kept, invert=True) & (labels > 0)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    cell = float(np.prod(grid["spacing_mm"]))
    field = np.where(dropped, 0.0, field)
    return field, {"components_raw": int(count), "anchor_components": len(kept), "dropped_count": int(count - len(kept)),
                   "dropped_volume_mm3": float(np.count_nonzero(dropped) * cell), "largest_dropped_mm3": float(max([sizes[i] for i in range(1, count + 1) if i not in kept] or [0]) * cell)}

def surface(density, grid, cfg, regions=(), anchor=None, preserve=None):
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    field = gaussian_filter(np.asarray(density, dtype=np.float32), cfg["sigma_cells"]) if cfg["sigma_cells"] > 0 else np.asarray(density, dtype=np.float32)
    field = field if preserve is None else np.maximum(field, preserve)
    field = carve(field, grid, regions) if cfg["carve_bores"] else field
    connectivity = None
    if anchor is not None:
        field, connectivity = keep_connected(field, grid, cfg["threshold"], anchor)
    field = np.pad(field, 1)
    vertices, faces, _, _ = marching_cubes(field, cfg["threshold"], spacing=tuple(spacing), allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices + np.asarray(grid["origin_mm"]) - spacing / 2, faces[:, ::-1], process=True)
    parts = mesh.split(only_watertight=False)
    slivers = [part for part in parts if abs(part.volume) < cfg["min_body_mm3"]]
    if slivers:
        mesh = trimesh.util.concatenate([part for part in parts if abs(part.volume) >= cfg["min_body_mm3"]])
        connectivity = {**(connectivity or {}), "mesh_slivers_dropped": len(slivers), "mesh_slivers_mm3": float(sum(abs(part.volume) for part in slivers))}
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=cfg["taubin"])
    vertices = mesh.vertices.copy()
    for region in regions:
        if region["kind"] == "box" and region["name"].startswith(tuple(cfg["flatten"])):
            low, high = np.asarray(region["min_mm"]), np.asarray(region["max_mm"])
            top = np.all((vertices[:, :2] >= low[:2]) & (vertices[:, :2] <= high[:2]), axis=1) & (vertices[:, 2] >= high[2] - 2 * spacing[2]) & (vertices[:, 2] <= high[2] + spacing[2])
            vertices[top, 2] = high[2]
    mesh.vertices = vertices
    if mesh.volume < 0:
        mesh.invert()
    return mesh, connectivity

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

def anchors(domain):
    pads = np.zeros(domain["preserve"].shape, dtype=bool)
    centers = grid_centers(domain["grid"])
    for region in domain["regions"]:
        if region["name"].endswith("_motor_contact"):
            pads |= region_contains(centers, region)
    return pads & domain["preserve"]

def finish(out, density, fine_full, cfg, viewer):
    sections = cross_sections(gaussian_filter(np.asarray(density, dtype=np.float32), cfg["render"]["sigma_cells"]), fine_full)
    mesh, connectivity = surface(density, fine_full["grid"], cfg["render"], fine_full["regions"], anchors(fine_full) if cfg["render"]["keep"] == "motor_pads" else None, fine_full["preserve"])
    mesh.export(out / "geometry.stl")
    if viewer is not None:
        viewer.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "geometry.stl", viewer / "geometry.stl")
    render_views(mesh, out)
    allowed_volume = np.count_nonzero(fine_full["allowed"]) * np.prod(fine_full["grid"]["spacing_mm"])
    solid = gaussian_filter(np.asarray(density, dtype=np.float32), cfg["render"]["sigma_cells"]) > cfg["render"]["threshold"]
    discs = fine_full["metadata"]["round4"]["prop_discs"]
    centers = grid_centers(fine_full["grid"])
    footprint = np.zeros(solid.shape, dtype=bool)
    for x, y in discs["motors_mm"]:
        radial = np.hypot(centers[..., 0] - x, centers[..., 1] - y)
        footprint |= (radial <= discs["radius_mm"]) & (radial > discs["pad_radius_mm"])
    return {"prop_disc_share": float(np.count_nonzero(solid & footprint) / max(np.count_nonzero(solid), 1)), "prop_disc_top_share": float(np.count_nonzero((solid & footprint).any(2)) / np.count_nonzero(footprint.any(2))),
            "mass_g": float(mesh.volume * 1.09 / 1000), "mesh_volume_mm3": float(mesh.volume), "volume_fraction_mesh": float(mesh.volume / allowed_volume),
            "watertight": bool(mesh.is_watertight), "bodies": len(mesh.split(only_watertight=False)), "connectivity": connectivity,
            "member_width": member_widths(solid, fine_full["grid"]["spacing_mm"], fine_full["preserve"] | fine_full["forbidden"]),
            "width_height_ratio": sections["width_height_ratio"], "chord": lower_chord(density, fine_full["grid"], cfg["chord"], cfg["render"]["sigma_cells"], cfg["render"]["threshold"])}

def run_variant(cfg, variant):
    neural, render_cfg = {**cfg["neural"], **variant.get("neural", {})}, {**cfg["render"], **variant.get("render", {})}
    cfg = {**_merge(cfg, {key: variant[key] for key in VARIANT_KEYS if key in variant}), "render": render_cfg}
    out = Path(cfg["root"]) / variant["name"]
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    builder = R2Domain(cfg)
    _, half = builder.build(cfg["shape"], neural["volume_fraction"])
    discs = half["metadata"]["round4"]["prop_discs"]
    neural["volume_fraction"] *= half["metadata"]["round4"]["unblocked_cells"] / half["metadata"]["round4"]["allowed_cells"]
    extra = {"prop_discs": discs if discs["mode"] == "soft" else None, "modal": cfg["modal"] if cfg["modal"]["f1_min_hz"] else None}
    simp = cfg["method"] == "simp"
    if cfg["stiffness"]["min_n_per_mm"]:
        if simp:
            raise ValueError("The arm-tip stiffness constraint is implemented for the neural method only")
        extra.update(stiffness=cfg["stiffness"], feasibility_tolerance=cfg["stiffness"]["feasibility_tolerance"])
    keys = ("interface_node_policy", "case_weights", "penalization", "min_stiffness_ratio")
    settings = _settings({**{key: half["optimizer_settings"][key] for key in keys}, "linear_solver": "cuda_cudss", **cfg["simp"], "volume_fraction": neural["volume_fraction"], **extra}) if simp else base_settings(half, {**neural, **extra})
    holder = {}
    original = topology_neural.NeuralDensity
    class Capture(original):
        def __init__(self, *args):
            super().__init__(*args)
            holder["mapping"] = self
    topology_neural.NeuralDensity = Capture
    try:
        with (out / "iterations.jsonl").open("w") as log:
            progress = lambda e: (log.write(json.dumps({k: e.get(k) for k in ("iteration", "objective", "volume_fraction", "sharpness", "projection_beta", "elapsed_s", "f1_hz", "modal_penalty", "stiffness_n_per_mm", "stiffness_penalty")}) + "\n"), log.flush())
            result = optimize_topology(half, settings, progress_callback=progress) if simp else optimize_neural(half, settings, progress_callback=progress)
    finally:
        topology_neural.NeuralDensity = original
    optimized = perf_counter() - started
    if result["status"] != "ok":
        print(json.dumps({"variant": variant["name"], "status": result["status"], "diagnostics": result["diagnostics"]}), flush=True)
        return
    summary = result["summary"]
    np.savez_compressed(out / "density_half.npz", density=result["density"])
    fine_full, _ = builder.build(cfg["fine_shape"], neural["volume_fraction"])
    density = upsample(result["density"], fine_full) if simp else holder["mapping"].sample(fine_full, render_cfg.get("sample_sharpness", summary["sharpness_final"]), settings["volume_fraction"])
    np.savez_compressed(out / "density_fine.npz", density=density.astype(np.float32))
    viewer = Path(cfg["viewer_root"]) / (cfg["viewer_prefix"] + variant["name"]) if cfg["viewer_root"] else None
    info = {"variant": variant["name"], "method": ("SIMP" if simp else "neural") + (" round 4" if cfg["inertia_relief"] else " round 2"),
            "inertia_relief": [{key: value for key, value in case.items() if key in ("name", "inertia_relief")} for case in summary["system"]["cases"] if "inertia_relief" in case] or None, "volume_fraction_target": settings["volume_fraction"], "volume_fraction_opt": summary["volume_fraction"],
            "iterations": summary["iterations"], "stop_reason": summary["stop_reason"], "optimize_runtime_s": optimized,
            **finish(out, density, fine_full, cfg, viewer), "total_runtime_s": perf_counter() - started,
            "grid_opt_half": half["grid"], "grid_render_full": fine_full["grid"], "optimizer": {k: settings[k] for k in (("filter_radius_mm", "projection", "beta_schedule", "max_iterations", "move_limit") if simp else ("max_frequency_per_mm", "frequencies", "hidden", "learning_rate", "sharpness_final", "sharpness_iterations", "max_iterations", "max_width_penalty"))},
            "round2": half["metadata"]["round2"], "round4": {**half["metadata"]["round4"], "modal": cfg["modal"], "stiffness": cfg["stiffness"], "summary_stiffness": summary.get("stiffness"), "volume_weighted": settings["prop_discs"] is not None, "summary_modal": summary.get("modal")}, "render": render_cfg, "objective_final": summary["objective_final"],
            "static_surrogate_metrics": summary["static_surrogate_metrics"], "load_cases": [case["name"] for case in half["load_cases"]]}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("variant", "iterations", "optimize_runtime_s", "total_runtime_s", "mass_g", "bodies", "connectivity")}), flush=True)

def rerender(cfg, variant):
    render_cfg = {**cfg["render"], **variant.get("render", {})}
    cfg = {**cfg, "render": render_cfg}
    out = Path(cfg["root"]) / variant["name"]
    fine_full, _ = R2Domain(cfg).build(cfg["fine_shape"])
    info = json.loads((out / "info.json").read_text())
    info.update(finish(out, np.load(out / "density_fine.npz")["density"], fine_full, cfg, Path(cfg["viewer_root"]) / (cfg["viewer_prefix"] + variant["name"]) if cfg["viewer_root"] else None), render=render_cfg)
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info.get(k) for k in ("variant", "mass_g", "bodies", "connectivity")}), flush=True)

def calibrate(cfg):
    _, half = R2Domain(cfg).build(cfg["shape"])
    system = HexElasticity(half, interface_node_policy=half["optimizer_settings"]["interface_node_policy"], linear_solver=cfg["calibration"]["linear_solver"])
    settings = {**cfg["modal"], "f1_min_hz": cfg["modal"]["f1_min_hz"] or 300.0}
    rows = {}
    for path in cfg["calibration"]["inputs"]:
        started = perf_counter()
        density = np.load(path)["density"].ravel()
        info = ModalConstraint(system, settings)(density, half["optimizer_settings"]["penalization"], half["optimizer_settings"]["min_stiffness_ratio"], settings["initial_iterations"])[2]
        if cfg["stiffness"]["min_n_per_mm"]:
            solution = system.solve(density, half["optimizer_settings"]["penalization"], half["optimizer_settings"]["min_stiffness_ratio"], metrics=True)[cfg["stiffness"]["case"]]
            info.update(stiffness_n_per_mm=solution["stiffness_n_per_mm"], compliance_stiffness_n_per_mm=StiffnessConstraint(system, cfg["stiffness"]).force ** 2 / solution["compliance_n_mm"])
        rows[path] = {**info, "runtime_s": perf_counter() - started}
    system.close()
    Path(cfg["calibration"]["output"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["calibration"]["output"]).write_text(json.dumps(rows, indent=1))
    print(json.dumps({path: {key: row.get(key) for key in ("f1_hz", "stiffness_n_per_mm", "compliance_stiffness_n_per_mm")} for path, row in rows.items()}), flush=True)

def run_main(overrides):
    cfg = configure(overrides)
    for variant in cfg["variants"]:
        run_variant(cfg, variant)

def render_main(overrides):
    cfg = configure(overrides)
    for variant in cfg["variants"]:
        rerender(cfg, variant)

def main(argv=None):
    return command_line({"run": run_main, "render": render_main, "time": lambda overrides: time_solve(configure(overrides)), "calibrate": lambda overrides: calibrate(configure(overrides)), "verify": lambda overrides: verify(configure(overrides)),
                         "compare": lambda overrides: stitch(**{key: [Path(path) for path in value] if key == "inputs" else Path(value) if key == "output" else value for key, value in configure(overrides)["compare"].items()})}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
