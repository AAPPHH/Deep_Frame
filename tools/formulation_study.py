import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deep_frame.config as config
from deep_frame.config import RUN_SETTINGS, STAGES, command_line
from deep_frame.frame_run import ROOT, FrameRun, _git
from deep_frame.topology_geometry import _merge, embed_field
from deep_frame.topology_neural import cell_centers
from deep_frame.topology_optimization import HexElasticity
from deep_frame.topology_stability import STAND_CASES, FOUR_FEET, StandStability, stand_design, stand_domain, voxel_reserve
from deep_frame.topology_problem import ARM_TIP, BATTERY_SUPPORT, CANTILEVER_COVARIANCE, COVARIANCE, LOAD_COVARIANCE, MMA, MMAOptimizer, PROBLEM, TopologyProblem, cantilever_domain, cantilever_dual, cantilever_problem, covariance_cantilever, format_report, orthotropic_material, prolongate, shadow_thickness

RUN = "C:/clones/Deep_Frame-r4/exports/runs/r4_neural_v06_f1_1"
FORMULATION = {
    "request": RUN + "/requests/optimization.json",
    "reference_density": RUN + "/optimization/r4_neural_v06_f1/density_half.npz",
    "reference_info": RUN + "/optimization/r4_neural_v06_f1/info.json",
    "reference_stl": RUN + "/frame.stl",
    "reference_fraction": 0.06,
    "manafly": {"frame": "docs/validation/frame_evaluation_manafly3_frame.json", "voxel_mm": 0.5},
    "shape": [102, 96, 24],
    "coarse_shape": [68, 64, 24],
    "linear_solver": "cpu_superlu",
    "output": "docs/validation/formulation_reference.json",
    "loads": {"thrust_n": 2.02, "safety_factor": 2.0, "thrust_source": "HQProp T2.5X2X3V2S estimate: GTS V3 1203 8000KV table, HQ T65R, 7.4 V, 100 %: 206 g = 2.02 N",
              "electrical_power_w": 52.5, "kv": 8000.0, "voltage_v": 7.4, "loaded_speed_fraction": 0.75, "motor_efficiency": 0.75,
              "torque_source": "ESTIMATE: same table row 7.1 A / 52.5 W; rpm not in the data sheet, assumed 75 % of KV x V under load, motor efficiency 75 %",
              "crash_mass_g": 125.0, "crash_velocity_m_s": 5.0, "crash_stop_distance_mm": 50.0,
              "crash_source": "ASSUMPTION: all-up mass 125 g (INTEGRATION_CONFIG), impact speed 5 m/s, 50 mm combined stopping distance (props, battery, frame); E = m v^2 / 2, F = E / d per direction",
              "rotation": {"front_left": 1.0, "rear_right": 1.0, "front_right": 0.0, "rear_left": 0.0}},
    "cases": ["stiffness_arm_tip", "modes", "thrust_all"],
    "layout": None,
    "compare": {"bodies": {}, "output": "exports/layout/battery_vs_rails.png", "summary": "exports/layout/battery_vs_rails.json", "gap_mm": 25.0, "size": [1400, 900]},
    "camera_limits": {"spacing_mm": 4 / 3, "subdivision": 4, "stack_height_mm": 5.0, "tilts_deg": [20.0, 0.0], "linear_solver": "cpu_superlu", "output": "docs/validation/camera_limits_manafly.json"},
    "camera_fd": {"shape": [68, 64, 24], "step": 1e-5, "seed": 7, "low": 0.3, "high": 0.9, "sparse": 0.02, "battery_support": "free", "output": "docs/validation/camera_support_fd.json", "limits": {}},
    "battery_fd": {"shape": [68, 64, 24], "step": 1e-5, "seed": 7, "low": 0.3, "high": 0.9, "output": "docs/validation/battery_support_fd.json"},
    "stand_fd": {"shape": [68, 64, 24], "steps": [1e-4, 1e-5, 1e-6], "seed": 7, "low": 0.3, "high": 0.9, "sparse": 0.05, "output": "docs/validation/stand_stability_fd.json", "exact": "docs/validation/stand_stability.json",
                 "fields": {"rail_v3b": ["C:/clones/Deep_Frame-cov/exports/runs/simp_mma_cov3_opt/fine/density_half.npz", "C:/clones/Deep_Frame-cov/exports/runs/simp_mma_cov3_v3_2/domain.json"],
                            "battery_free": ["C:/clones/Deep_Frame-layout/exports/runs/battery_free_opt/fine/density_half.npz", "C:/clones/Deep_Frame-layout/exports/runs/battery_free_v3_2/domain.json"]}},
    "mma": {"settings": {}, "cantilever_start": 0.5, "dual_volume": 0.3, "fd_step": 1e-5, "fd_seed": 7, "linear_solver": "auto", "stage_solvers": {}, "until": "export", "coarse": True, "fine_start_level": 3, "start": None, "calibration": None, "memory_s": None,
            "root": "exports/runs/simp_mma_opt", "variant": "simp_mma", "resume": False, "gray": [0.05, 0.95], "method": "simp_mma", "agreement": 0.15,
            "viewer": "C:/clones/Deep_Frame-neural/exports", "manafly_renders": "C:/clones/Deep_Frame-neural/exports/fast/_manafly_same_renderer", "evaluation_python": "C:/clones/Deep_Frame/.venv/Scripts/python.exe",
            "bodies": ["raw", "recon"], "body_start": {"battery": {"density": 0.5, "cells": 2}, "camera": {"density": 0.5, "cells": 2, "zone": True}}, "viewer_names": {"raw": "{method}_final", "recon": "{method}_final_recon", "v3": "{method}_v3"},
            "figures": [{"output": "{method}_4views.png", "panels": ["raw", "recon", "manafly"], "labels": ["SIMP-MMA raw", "SIMP-MMA recon", "ManaFly"]}]},
    "v3": {"worktree": "C:/clones/Deep_Frame-recon3", "ref": "HEAD", "copy": "C:/Users/jfham/AppData/Local/Temp/claude/c--clones-Deep-Frame/2bec171b-ba58-44fe-ab0f-61ff45688b18/scratchpad/recon3_copy",
           "argv": ["splines"], "compute": "geometry", "compare": "recon"},
    "comparison": {"output": "exports/cov/comparison.json", "old": {"raw": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_raw_1", "recon": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_recon_1"},
                   "old_fine": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_opt/fine/result.json", "previous": {}, "previous_fine": None, "gap": {}, "manafly_sigma": "exports/cov/eval/manafly3/sigma.json", "aether4_sigma": "exports/cov/eval/aether4/sigma.json",
                   "twist": {"motor_front_left": 1.0, "motor_rear_right": 1.0, "motor_front_right": -1.0, "motor_rear_left": -1.0}, "twist_dof": "Fz"},
    "covariance": {"variant": "mean", "limit_factor": 4.0, "start": 0.5, "fd_step": 1e-5, "fd_seed": 11, "ks_fd": 5.0, "settings": {},
                   "frame_density": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_opt/fine/density_half.npz", "frame_solver": "auto"},
    "solver_memory": {"run": "C:/clones/Deep_Frame-cov/exports/runs/simp_mma_cov3_opt", "grids": ["coarse", "fine"], "evaluations": 3, "mma_iterations": 5, "sample_s": 0.5,
                      "output": "exports/solver_memory/baseline", "reference": None, "start": None, "multigrid": None, "share_static": True, "modal": {}},
    "setup_memory": {"shape": None, "start": "C:/clones/Deep_Frame-cov/exports/runs/simp_mma_cov3_opt/fine", "evaluate": True, "sample_s": 0.02, "min_free_gb": 5.0, "output": "exports/setup_memory/probe"},
}

def configure(overrides):
    unknown = sorted(set(overrides) - set(FORMULATION))
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(unknown))
    return _merge(FORMULATION, overrides)

def load_numbers(loads):
    speed = loads["kv"] * loads["voltage_v"] * loads["loaded_speed_fraction"] / 60
    torque = loads["electrical_power_w"] * loads["motor_efficiency"] / (2 * np.pi * speed) * 1000
    energy = loads["crash_mass_g"] / 1000 * loads["crash_velocity_m_s"] ** 2 / 2
    return {"thrust_per_motor_n": loads["thrust_n"] * loads["safety_factor"], "yaw_torque_per_motor_n_mm": torque * loads["safety_factor"], "yaw_torque_unfactored_n_mm": torque,
            "rotor_speed_rps": speed, "crash_energy_j": energy, "crash_force_n": energy / (loads["crash_stop_distance_mm"] / 1000),
            "twist": "diagonal differential thrust (front_left, rear_right full; front_right, rear_left idle) = mean + saddle; saddle part +-T SF / 2 per motor, self-equilibrated",
            "yaw": "reaction torque of the full-thrust pair as tangential force couples on the two halves of each pad, other pair idle; inertia relief balances the net yaw moment"}

def patched_settings(cfg):
    from run import _update
    from deep_frame.frame_run import FrameLayout
    from tools.neural_study import configure as study
    request = json.loads(Path(cfg["request"]).read_text(encoding="utf-8"))
    overrides = deepcopy(request["overrides"])
    patches = [request["patch"]]
    if cfg["layout"]:
        layout = FrameLayout(cfg["layout"])
        patches.append(layout.patch())
        overrides["hoop"] = layout.hoop()
    for patch in patches:
        for name, values in patch.items():
            _update(getattr(config, name), values)
    if free_camera_mode():
        overrides["hoop"] = None
    settings = study(overrides)
    if free_battery():
        settings["inertia_relief"]["attachments"].pop("battery")
    if free_camera_mode():
        settings["inertia_relief"]["attachments"].pop("camera")
    return settings

def free_battery():
    return config.TOPOLOGY_CONFIG.get("battery_support", "rails") == "free"

def patched_builder(cfg, stiffness=True):
    from tools.neural_study import R2Domain
    settings = patched_settings(cfg)
    if stiffness:
        settings["stiffness"] = {**settings["stiffness"], "min_n_per_mm": ARM_TIP["min_n_per_mm"]}
    return R2Domain(settings)

def free_support(half):
    regions = {region["name"]: region for region in half["regions"]}
    part, clearance = config.COMPONENT_DEFAULTS["battery"], config.TOPOLOGY_CONFIG["component_clearance_mm"]
    keep_out = regions["battery_insertion"]
    low = np.asarray(keep_out["min_mm"]) + [clearance, clearance, 0.0]
    size = np.array([part["width_mm"], part["length_mm"], part["height_mm"]])
    center = low + size / 2
    mass = part["mass_g"]
    inertia = mass / 12 * np.diag([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2])
    half["battery"] = {"keep_out": keep_out, "reference_mm": [0.0, float(center[1]), float(low[2])], "center_mm": center.tolist(), "size_mm": size.tolist(), "mass_g": mass, "inertia_g_mm2": inertia.tolist()}
    deck = half["interfaces"]["battery"]["regions"][0]
    for case in half["load_cases"]:
        if "inertia_relief" not in case:
            continue
        applied = sum((np.asarray(load["force_n"], dtype=float) for load in case["loads"] if load["region"] == deck), np.zeros(3))
        case["loads"] = [load for load in case["loads"] if load["region"] != deck]
        case["inertia_relief"] = {**deepcopy(case["inertia_relief"]), "bodies": [{"name": "battery", "mass_g": mass, "position_mm": center.tolist(), "inertia_g_mm2": inertia.tolist(), "force_n": applied.tolist()}]}
    half["point_masses"] = [item for item in half["point_masses"] if item["name"] != "battery"]
    half["metadata"]["formulation"]["battery_support"] = {"mode": "free", **{key: half["battery"][key] for key in ("reference_mm", "center_mm", "size_mm", "mass_g")},
                                                          "statement": "battery rigid body on density-dependent contact springs (BATTERY_SUPPORT); its inertia and the crash_back deck load act on the body, not on frame nodes"}
    return half

def free_camera_mode():
    return config.TOPOLOGY_CONFIG.get("camera_support", "prescribed") == "free"

def camera_frame(part):
    tilt = np.radians(part["tilt_deg"])
    axis, up = np.array([0.0, np.cos(tilt), np.sin(tilt)]), np.array([0.0, -np.sin(tilt), np.cos(tilt)])
    rotation = np.column_stack([[1.0, 0.0, 0.0], axis, up])
    size = np.array([part["width_mm"], part["length_mm"], part["height_mm"]])
    inertia = rotation @ np.diag(part["mass_g"] / 12 * np.array([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2])) @ rotation.T
    return axis, up, size, inertia

def fov_cells(grid, front, axis, up, spec):
    h = np.asarray(grid["spacing_mm"], dtype=float)
    centers = cell_centers(grid)
    half_angles = np.tan(np.radians(np.asarray(spec["fov_deg"]) / 2))
    blocked = np.zeros(len(centers), dtype=bool)
    for corner in np.indices((2, 2, 2)).reshape(3, -1).T:
        offset = centers + (corner - 0.5) * h - front
        depth, across, height = offset @ axis, offset[:, 0], offset @ up
        reach = np.maximum(depth, 0.0)
        blocked |= (depth >= -spec["fov_clearance_mm"]) & (np.abs(across) <= spec["aperture_mm"] + spec["fov_clearance_mm"] + reach * half_angles[0]) & (np.abs(height) <= spec["aperture_mm"] + spec["fov_clearance_mm"] + reach * half_angles[1])
    return blocked.reshape(tuple(grid["shape"]))

def free_camera(half):
    spec, part = config.CAMERA_SUPPORT, {**config.COMPONENT_DEFAULTS["camera"]}
    regions = {region["name"]: region for region in half["regions"]}
    keep_out, screw = regions["camera_envelope"], regions["camera_screw_axis"]
    grid = half["grid"]
    upper = np.asarray(grid["origin_mm"]) + np.asarray(grid["spacing_mm"]) * np.asarray(grid["shape"])
    low, high = np.asarray(keep_out["min_mm"]), np.asarray(keep_out["max_mm"])
    center = (low + high) / 2
    center[0] = 0.0
    axis, up, size, inertia = camera_frame(part)
    front = center + axis * size[1] / 2
    blocked = fov_cells(grid, front, axis, up, spec)
    lost = int(np.count_nonzero(blocked & half["preserve"]))
    half["allowed"] = half["allowed"] & ~blocked
    half["preserve"] &= half["allowed"]
    half["forbidden"] = ~half["allowed"]
    zone = {"kind": "box", "min_mm": [-high[0] - spec["zone_mm"]["side"], float(center[1]), float(center[2])], "max_mm": [high[0] + spec["zone_mm"]["side"], float(upper[1]), float(min(high[2] + spec["zone_mm"]["top"], upper[2]))]}
    cases = {case["name"]: case for case in half["load_cases"]}
    force = float(np.linalg.norm(cases["crash_front"]["loads"][0]["force_n"]))
    angle = np.radians(spec["oblique_deg"])
    zones = {name: (force * np.asarray(direction)).tolist() for name, direction in spec["zone_cases"].items() if name in cases}
    zones["crash_camera_oblique"] = (force * np.array([np.sin(angle), -np.cos(angle), 0.0])).tolist()
    oblique = {**deepcopy(cases["crash_front"]), "name": "crash_camera_oblique", "purpose": "oblique camera crash on the impact zone"}
    half["load_cases"].append(oblique)
    body = {"name": "camera", "mass_g": part["mass_g"], "position_mm": center.tolist(), "inertia_g_mm2": inertia.tolist(), "force_n": [0.0, 0.0, 0.0]}
    for case in half["load_cases"]:
        if "inertia_relief" in case:
            case["inertia_relief"] = {**deepcopy(case["inertia_relief"]), "bodies": case["inertia_relief"].get("bodies", []) + [{**deepcopy(body), "force_n": zones.get(case["name"], body["force_n"])}]}
        if case["name"] in zones:
            case["loads"] = []
    half["point_masses"] = [item for item in half["point_masses"] if item["name"] != "camera"]
    step = float(np.min(grid["spacing_mm"])) * spec["step_fraction"]
    window = [[-size[0] / 2 - spec["window_mm"]["side"], size[0] / 2 + spec["window_mm"]["side"]], [-size[2] / 2, size[2] / 2 + spec["window_mm"]["top"]]]
    half["camera"] = {"keep_out": deepcopy(keep_out), "reference_mm": [0.0, *screw["center_mm"][1:]], "center_mm": center.tolist(), "size_mm": size.tolist(), "mass_g": part["mass_g"], "inertia_g_mm2": inertia.tolist(),
                      "screw_axis_yz_mm": list(screw["center_mm"][1:]), "zone": zone, "zones": zones, "displacement_cases": [name for name in spec["displacement_cases"] if name in zones],
                      "reference_regions": [deepcopy(region) for name, region in regions.items() if name.startswith("aio_contact_")],
                      "coverage": {"axes": [axis.tolist(), [1.0, 0.0, 0.0], up.tolist()], "front_mm": front.tolist(), "window_mm": window, "silhouette_mm": [[-size[0] / 2, size[0] / 2], [-size[2] / 2, size[2] / 2]], "step_mm": step, "depth_mm": list(spec["protection_depth_mm"]), "frontal_area_mm2": float(size[0] * size[2])},
                      "fov_cells": int(np.count_nonzero(blocked)), "fov_preserve_cells_removed": lost, "tilt_deg": part["tilt_deg"]}
    half["metadata"]["formulation"]["camera_support"] = {"mode": "free", **{key: half["camera"][key] for key in ("reference_mm", "center_mm", "zone", "zones", "fov_cells", "fov_preserve_cells_removed")},
                                                         "statement": "camera rigid body on density-dependent screw springs (CAMERA_SUPPORT), no hoops or lugs; 4:3 field of view as keep-out; frontal, below and oblique crash as design-dependent loads on the impact zone"}
    return half

def motor_name(center):
    return ("front_" if center[1] > 0 else "rear_") + ("left" if center[0] < 0 else "right")

def frame_domain(cfg, shape):
    _, half = patched_builder(cfg).build(shape, cfg["reference_fraction"])
    numbers = load_numbers(cfg["loads"])
    cases = {case["name"]: case for case in half["load_cases"]}
    thrust = cases["thrust_all"]
    pads = {motor_name((np.asarray(load["region"]["min_mm"]) + load["region"]["max_mm"]) / 2): load["region"] for load in thrust["loads"]}
    for load in thrust["loads"]:
        load["force_n"] = [0.0, 0.0, numbers["thrust_per_motor_n"]]
    regions = {region["name"]: region for region in half["regions"]}
    camera = np.asarray(regions["camera_screw_axis"]["center_mm"], dtype=float)
    centre = lambda box: ((np.asarray(box["min_mm"]) + box["max_mm"]) / 2).tolist()
    deck, view = cases["crash_back"]["loads"][0]["region"], cases["crash_front"]["loads"][0]["region"]
    half["interfaces"] = {**{"motor_" + name: {"regions": [deepcopy(region)], "reference_mm": centre(region)} for name, region in pads.items()},
                          "battery": {"regions": [deepcopy(deck)], "reference_mm": [0.0, *centre(deck)[1:]]}, "camera": {"regions": [deepcopy(view)], "reference_mm": [0.0, *camera[1:].tolist()]}}
    rotation = cfg["loads"]["rotation"]
    twist = {**deepcopy(thrust), "name": "twist", "purpose": "diagonal differential thrust, saddle part", "loads": [{"region": deepcopy(region), "force_n": [0.0, 0.0, (1 if rotation[name] else -1) * numbers["thrust_per_motor_n"] / 2]} for name, region in pads.items()]}
    couples = []
    for name, region in pads.items():
        if not rotation[name]:
            continue
        low, high = np.asarray(region["min_mm"]), np.asarray(region["max_mm"])
        middle, arm = (low[0] + high[0]) / 2, (high[0] - low[0]) / 4
        force = numbers["yaw_torque_per_motor_n_mm"] / (2 * arm)
        for sign, (a, b) in ((-1.0, (low[0], middle)), (1.0, (middle, high[0]))):
            couples.append({"region": {"kind": "box", "min_mm": [a, low[1], low[2]], "max_mm": [b, high[1], high[2]]}, "force_n": [0.0, sign * force, 0.0]})
    torsion = {**deepcopy(thrust), "name": "torsion_yaw", "purpose": "motor reaction torque of the full-thrust pair", "loads": couples}
    for name in PROBLEM["crash"]["cases"]:
        total = sum(np.linalg.norm(load["force_n"]) for load in cases[name]["loads"])
        for load in cases[name]["loads"]:
            load["force_n"] = (np.asarray(load["force_n"]) * numbers["crash_force_n"] / total).tolist()
    relief = thrust["inertia_relief"]
    keep = set(cfg["cases"]) | set(PROBLEM["crash"]["cases"])
    half["load_cases"] = [case for case in half["load_cases"] if case["name"] in keep] + [twist, torsion]
    half["point_masses"] = [{"name": item["name"], "mass_g": item["mass_g"], "attachment_region": item["region"]} for item in relief["point_masses"] if item["name"] in PROBLEM["modal"]["point_masses"]]
    half["material"] = orthotropic_material(half["material"])
    discs = half["metadata"]["round4"]["prop_discs"]
    half["metadata"]["formulation"] = {"loads": numbers, "load_parameters": cfg["loads"], "shadow": {"motors_mm": discs["motors_mm"], "radius_mm": discs["radius_mm"]},
                                       "inertia_relief_frame_mass_g": relief["preserve_mass_g"], "inertia_relief": "point masses + frame mass at the reference fraction on the preserves, design-independent (keeps compliance self-adjoint)"}
    half = free_support(half) if free_battery() else half
    return free_camera(half) if free_camera_mode() else half

def frame_problem(half, references=None):
    problem = deepcopy(PROBLEM)
    if "battery" in half:
        problem["battery"] = deepcopy(BATTERY_SUPPORT)
    if "camera" in half:
        problem["camera"] = deepcopy(config.CAMERA_SUPPORT)
    problem["shadow"].update(half["metadata"]["formulation"]["shadow"])
    if config.STAND_STABILITY["enabled"]:
        problem["stability"] = deepcopy(config.STAND_STABILITY)
    if references:
        problem["crash"].update(reference=references["crash_compliance_n_mm"], reference_source=references["source"])
        problem["shadow"].update(limit_mm=references["manafly_shadow_mm"], source=references["source"])
    return problem

def frame_setup(cfg=FORMULATION, shape=None):
    half = frame_domain(cfg, shape or cfg["shape"])
    problem = frame_problem(half, json.loads(Path(cfg["output"]).read_text(encoding="utf-8")))
    if cfg["mma"]["calibration"]:
        problem["covariance"]["limits"]["calibration"] = dict(cfg["mma"]["calibration"])
    return half, problem

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

def manafly_shadow(cfg):
    import trimesh
    from deep_frame.frame_evaluation import voxel_grid
    spec = json.loads(Path(cfg["manafly"]["frame"]).read_text(encoding="utf-8"))
    mesh = trimesh.load_mesh(spec["stl"], process=True)
    h = cfg["manafly"]["voxel_mm"]
    solid, lower, _ = voxel_grid(mesh, h)
    shadow = {**PROBLEM["shadow"], "motors_mm": [xyz[:2] for xyz in spec["motors"].values()], "radius_mm": spec["prop_diameter_mm"] / 2}
    return {"value_mm": shadow_thickness(solid, lower, h, shadow), "stl": spec["stl"], "stl_sha256_16": digest(spec["stl"]), "voxel_mm": h, "motors_mm": shadow["motors_mm"], "radius_mm": shadow["radius_mm"],
            "hub_radius_mm": shadow["hub_radius_mm"], "exponent": shadow["exponent"], "volume_mm3": float(np.count_nonzero(solid) * h ** 3)}

def own_shadow(cfg, half):
    import trimesh
    from deep_frame.frame_evaluation import voxel_grid
    h = cfg["manafly"]["voxel_mm"]
    solid, lower, _ = voxel_grid(trimesh.load_mesh(cfg["reference_stl"], process=True), h)
    return shadow_thickness(solid, lower, h, {**PROBLEM["shadow"], **half["metadata"]["formulation"]["shadow"]})

def plumbing(cfg):
    _, half = patched_builder(cfg, stiffness=False).build(cfg["shape"], cfg["reference_fraction"])
    system = HexElasticity(half, interface_node_policy=half["optimizer_settings"]["interface_node_policy"], linear_solver=cfg["linear_solver"])
    density = embed_field(np.load(cfg["reference_density"])["density"], half["grid"]).ravel()
    names = ["arm_tip", "thrust_all"] + PROBLEM["crash"]["cases"]
    solved = system.solve(density, 3.0, 1e-6)
    system.close()
    stored = json.loads(Path(cfg["reference_info"]).read_text(encoding="utf-8"))["static_surrogate_metrics"]
    return {name: {"recomputed": solved[name]["compliance_n_mm"], "stored": stored[name]["compliance_n_mm"], "relative": solved[name]["compliance_n_mm"] / stored[name]["compliance_n_mm"] - 1} for name in names}

def git_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()

def references(cfg):
    started = perf_counter()
    check = plumbing(cfg)
    half = frame_domain(cfg, cfg["shape"])
    problem = TopologyProblem(half, frame_problem(half), linear_solver=cfg["linear_solver"])
    density = embed_field(np.load(cfg["reference_density"])["density"], half["grid"]).ravel()
    solved = problem.compliances(density)
    crash = {name: solved[name]["compliance_n_mm"] for name in PROBLEM["crash"]["cases"]}
    problem.close()
    manafly = manafly_shadow(cfg)
    result = {"source": cfg["output"], "sha": git_sha(), "reference_density": cfg["reference_density"], "reference_density_sha256_16": digest(cfg["reference_density"]),
              "model": "half domain %s, 4/3 mm, orthotropic PA6-CF (print axis = grid Z), SIMP p=3, reference density used directly as the physical field (no filter/erosion)" % half["grid"]["shape"],
              "crash_compliance_n_mm": crash, "crash_limit_n_mm": {name: PROBLEM["crash"]["ratio"] * value for name, value in crash.items()},
              "manafly_shadow_mm": manafly["value_mm"], "manafly": manafly, "loads": half["metadata"]["formulation"]["loads"], "load_parameters": cfg["loads"],
              "plumbing_old_model": check, "reference_stl_shadow_mm": own_shadow(cfg, half)}
    problem = TopologyProblem(half, frame_problem(half, result), linear_solver=cfg["linear_solver"])
    report = problem.physical_report(density)
    problem.close()
    result.update(reference_report=report, reference_report_table=format_report(report["rows"]), filter_radius_mm=problem.radius, runtime_s=perf_counter() - started,
                  point_masses=half["point_masses"], monitored={row["name"]: row["value"] for row in report["rows"] if row["status"] == "monitored"})
    Path(cfg["output"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]).write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("crash_compliance_n_mm", "manafly_shadow_mm", "reference_stl_shadow_mm", "loads")}, default=float), flush=True)
    print(result["reference_report_table"], flush=True)

def modal_split(cfg):
    base = frame_domain(cfg, cfg["shape"])
    density = embed_field(np.load(cfg["reference_density"])["density"], base["grid"]).ravel()
    rows = {}
    for name, masses, material in (("battery_isotropic", ["battery"], "isotropic"), ("battery_orthotropic", ["battery"], "orthotropic"), ("three_isotropic", PROBLEM["modal"]["point_masses"], "isotropic"), ("three_orthotropic", PROBLEM["modal"]["point_masses"], "orthotropic")):
        half = deepcopy(base)
        half["point_masses"] = [item for item in half["point_masses"] if item["name"] in masses]
        if material == "isotropic":
            half["material"] = {key: value for key, value in half["material"].items() if key != "orthotropic"}
        problem = deepcopy(PROBLEM)
        problem.update(crash=None, shadow=None, monitor=[])
        solver = TopologyProblem(half, problem, linear_solver=cfg["linear_solver"])
        rows[name] = solver.modal.measure(density, solver.penalization, solver.min_stiffness_ratio, PROBLEM["modal"]["initial_iterations"])[2]
        solver.close()
    target = Path(cfg["output"]).with_name("formulation_modal_split.json")
    target.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(json.dumps({name: row["f1_hz"] for name, row in rows.items()}), flush=True)

def cantilever(cfg):
    result = cantilever_dual()
    Path(cfg["output"]).with_name("formulation_cantilever.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps(result, default=float), flush=True)

def finite_differences(problem, design, step, seed):
    direction = np.random.default_rng(seed).standard_normal(problem.map.n) * problem.map.free
    result, plus, minus = problem.evaluate(design), problem.evaluate(design + step * direction), problem.evaluate(design - step * direction)
    rows = {"objective": [float(result["objective_gradient"] @ direction), float((plus["objective"] - minus["objective"]) / (2 * step))]}
    rows.update({name: [float(gradient @ direction), float((high - low) / (2 * step))] for name, gradient, high, low in zip(result["names"], result["constraint_gradients"], plus["constraints"], minus["constraints"])})
    return {name: {"analytic": a, "finite_difference": f, "relative_error": abs(a - f) / max(abs(f), 1e-30)} for name, (a, f) in rows.items()}

def cantilever_mma(cfg):
    dual = json.loads(Path(cfg["output"]).with_name("formulation_cantilever.json").read_text(encoding="utf-8"))
    domain = cantilever_domain()
    problem = TopologyProblem(domain, cantilever_problem(dual["stiffness_n_per_mm"]))
    random = np.random.default_rng(cfg["mma"]["fd_seed"]).uniform(0.2, 0.8, problem.map.n)
    checks = finite_differences(problem, random, cfg["mma"]["fd_step"], cfg["mma"]["fd_seed"])
    result = MMAOptimizer(problem, cfg["mma"]["settings"]).run(np.full(problem.map.n, cfg["mma"]["cantilever_start"]))
    rows = {row["name"]: row for row in result["result"]["rows"]}
    problem.close()
    problem = TopologyProblem(domain, cantilever_problem(1.0, dual_volume := cfg["mma"]["dual_volume"]))
    compliance = MMAOptimizer(problem, {**cfg["mma"]["settings"], "objective": "tip_stiffness"}).run(np.full(problem.map.n, dual_volume))
    reverse = {row["name"]: row for row in compliance["result"]["rows"]}
    problem.close()
    problem = TopologyProblem(domain, cantilever_problem(reverse["tip_stiffness"]["value"]))
    closing = MMAOptimizer(problem, cfg["mma"]["settings"]).run(np.full(problem.map.n, cfg["mma"]["cantilever_start"]))
    closed = {row["name"]: row for row in closing["result"]["rows"]}
    problem.close()
    duality = {"statement": "MMA min-compliance at V gives k_V; MMA min-mass subject to k >= k_V must return volume ~ V with the stiffness constraint active", "volume_max": dual_volume,
               "min_compliance": {"status": compliance["status"], "iterations": compliance["iterations"], "stiffness_n_per_mm": reverse["tip_stiffness"]["value"], "volume_fraction_intermediate": reverse["volume"]["value"], "volume_status": reverse["volume"]["status"]},
               "min_mass": {"status": closing["status"], "iterations": closing["iterations"], "stiffness_n_per_mm": closed["tip_stiffness"]["value"], "stiffness_status": closed["tip_stiffness"]["status"], "volume_fraction_intermediate": closed["volume"]["value"]},
               "volume_deviation": closed["volume"]["value"] / reverse["volume"]["value"] - 1}
    record = {"dual": dual, "mma_duality": duality, "status": result["status"], "iterations": result["iterations"], "runtime_s": result["runtime_s"], "seconds_per_iteration": result["seconds_per_iteration"], "levels": result["levels"],
              "volume_fraction_intermediate": rows["volume"]["value"], "volume_deviation": rows["volume"]["value"] / dual["volume_fraction_intermediate"] - 1, "stiffness_n_per_mm": rows["tip_stiffness"]["value"],
              "stiffness_status": rows["tip_stiffness"]["status"], "rows": result["result"]["rows"], "table": format_report(result["result"]["rows"]), "finite_differences": checks, "mma": result["settings"], "sha": git_sha()}
    Path(cfg["output"]).with_name("formulation_cantilever_mma.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("status", "iterations", "volume_fraction_intermediate", "volume_deviation", "stiffness_n_per_mm", "stiffness_status", "mma_duality")}, default=float), flush=True)
    print(record["table"], flush=True)

def rows_of(problem, design):
    return {row["name"]: row for row in problem.evaluate(design)["rows"]}

def dense_flexibility(problem, density):
    from scipy.sparse.linalg import splu
    system, model = problem.system, problem.covariance
    stiffness = system.matrix(system.young * (1e-6 + (1 - 1e-6) * density ** 3)).tocsc()
    free = next(case for case in system.cases if case["name"] == model.settings["support"])["free"]
    loads = np.zeros((system.ndof, len(model.labels)))
    for column, (name, dof) in enumerate(model.labels):
        direct, values, _, _ = model.wrenches(problem.domain["interfaces"][name])[LOAD_COVARIANCE["dofs"].index(dof)]
        np.add.at(loads[:, column], direct, values)
    displacement = np.zeros_like(loads)
    displacement[free] = splu(stiffness[free][:, free]).solve(loads[free])
    return loads.T @ displacement

def covariance_cantilever_check(cfg):
    from scipy.linalg import sqrtm
    settings, shape = cfg["covariance"], (32, 6, 12)
    density = np.random.default_rng(settings["fd_seed"]).uniform(0.3, 1.0, shape)
    half, full = TopologyProblem(*covariance_cantilever(shape)), TopologyProblem(*covariance_cantilever(shape, full=True))
    mirrored = np.concatenate([density[:, ::-1, :], density], axis=1).ravel()
    flexibility, sigma = dense_flexibility(full, mirrored), full.covariance.sigma
    root = np.real(sqrtm(sigma))
    dense = {"load_mean": float(np.trace(flexibility @ sigma)), "load_worst": float(np.linalg.eigvalsh(root @ flexibility @ root).max())}
    measured = {name: {row["name"]: row["value"] for part in problem.physics(field)[:2] for row in part if row["name"] in dense} for name, problem, field in (("half", half, density.ravel()), ("full", full, mirrored))}
    solid = dense_flexibility(full, np.ones_like(mirrored))
    length, height, width = shape[0], shape[2], 2 * shape[1]
    beam = {"tip_Fz_n_mm_per_n": length ** 3 / (3 * full.system.young * width * height ** 3 / 12), "tip_Fz_dense": float(solid[0, 0])}
    half.close()
    full.close()
    limits = {"mean_n_mm": 1.2 * dense["load_mean"], "worst_n_mm": 1.2 * dense["load_worst"]}
    domain, problem = covariance_cantilever(shape, limits=limits)
    problem["covariance"]["ks"] = settings["ks_fd"]
    tested = TopologyProblem(domain, problem)
    design = np.random.default_rng(settings["fd_seed"]).uniform(0.2, 0.8, tested.map.n)
    checks = finite_differences(tested, design, settings["fd_step"], settings["fd_seed"])
    tested.close()
    rank_one = {"labels": [["tip", "Fz"], ["mid", "Fz"], ["tip", "Fy"]], "std": [1.0, 1.0, 0.5], "correlation": 1.0}
    domain, problem = covariance_cantilever(shape, settings=rank_one)
    problem["covariance"]["sigma"] = np.outer(rank_one["std"], rank_one["std"]).tolist()
    single = TopologyProblem(domain, problem)
    values = {row["name"]: row["value"] for part in single.physics(density.ravel())[:2] for row in part if row["name"] in dense}
    single.close()
    record = {"statement": "cantilever 32 x 12 x 12 mm (half model 32 x 6 x 12 cells), x = 0 clamped; " + CANTILEVER_COVARIANCE["statement"], "sigma": sigma.tolist(), "labels": [list(label) for label in full.covariance.labels],
              "density": "uniform random [0.3, 1] physical field, SIMP p = 3", "dense_full_model": dense, "optimizer": measured,
              "relative_error": {name: {key: abs(value[key] / dense[key] - 1) for key in dense} for name, value in measured.items()},
              "dense_flexibility": flexibility.tolist(), "rank_one": {**values, "relative_gap": abs(values["load_worst"] / values["load_mean"] - 1)}, "beam_theory_info": beam,
              "finite_differences": {"ks": settings["ks_fd"], "limits": limits, "rows": checks}, "sha": git_sha()}
    Path(cfg["output"]).with_name("formulation_covariance_cantilever.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("dense_full_model", "relative_error", "rank_one")}, default=float), flush=True)
    print(json.dumps({name: row["relative_error"] for name, row in checks.items()}), flush=True)

def covariance_cantilever_mma(cfg):
    settings, shape = cfg["covariance"], (32, 6, 12)
    variant, other = settings["variant"], {"mean": "worst", "worst": "mean"}[settings["variant"]]
    domain, problem = covariance_cantilever(shape)
    reference = TopologyProblem(domain, problem)
    solid = rows_of(reference, np.ones(reference.map.n))
    reference.close()
    limit = settings["limit_factor"] * solid["load_" + variant]["value"]
    domain, problem = covariance_cantilever(shape, limits={variant + "_n_mm": limit, "source": f"{settings['limit_factor']} x solid-block value"})
    tp = TopologyProblem(domain, problem)
    started = perf_counter()
    result = MMAOptimizer(tp, settings["settings"]).run(np.full(tp.map.n, settings["start"]))
    rows = {row["name"]: row for row in result["result"]["rows"]}
    tp.close()
    record = {"variant": variant, "statement": f"min mass subject to load_{variant} <= {settings['limit_factor']} x solid value (load_{other} monitored); the constraint must end active", "solid": {name: row["value"] for name, row in solid.items()},
              "limit_n_mm": limit, "status": result["status"], "iterations": result["iterations"], "runtime_s": perf_counter() - started, "levels": result["levels"], "volume_fraction_intermediate": rows["volume"]["value"],
              "constraint": rows["load_" + variant], "monitored": rows["load_" + other], "rows": result["result"]["rows"], "table": format_report(result["result"]["rows"]), "sha": git_sha()}
    target = Path(cfg["output"]).with_name(f"formulation_covariance_mma_{variant}.json")
    target.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    np.savez_compressed(target.with_suffix(".npz"), design=result["design"])
    print(json.dumps({key: record[key] for key in ("variant", "status", "iterations", "volume_fraction_intermediate", "limit_n_mm")}, default=float), flush=True)
    print(record["table"], flush=True)

def covariance_frame(cfg):
    import psutil
    started = perf_counter()
    half, problem = frame_setup(cfg, cfg["shape"])
    tp = TopologyProblem(half, problem, linear_solver=cfg["covariance"]["frame_solver"])
    built = perf_counter() - started
    density = np.load(cfg["covariance"]["frame_density"])["density"].ravel().astype(float)
    clock = perf_counter()
    report = tp.physical_report(density)
    evaluated = perf_counter() - clock
    record = {"density": cfg["covariance"]["frame_density"], "density_sha256_16": digest(cfg["covariance"]["frame_density"]), "grid": half["grid"]["shape"], "rank": tp.covariance.rank,
              "interfaces": half["interfaces"], "cases": len(tp.system.cases), "factorization_groups": len(tp.system.groups), "build_s": built, "evaluate_s": evaluated,
              "peak_rss_gb": psutil.Process().memory_info().peak_wset / 2 ** 30 if hasattr(psutil.Process().memory_info(), "peak_wset") else None,
              "mass_g": report["mass_g"], "rows": report["rows"], "table": format_report(report["rows"]), "sha": git_sha()}
    tp.close()
    Path(cfg["output"]).with_name("formulation_covariance_frame.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("rank", "build_s", "evaluate_s", "peak_rss_gb", "mass_g")}, default=float), flush=True)
    print(record["table"], flush=True)

def battery_fd(cfg):
    spec = cfg["battery_fd"]
    half, problem = frame_setup(cfg, spec["shape"])
    tp = TopologyProblem(half, problem, linear_solver=cfg["linear_solver"])
    design = np.random.default_rng(spec["seed"]).uniform(spec["low"], spec["high"], tp.map.n)
    result = tp.evaluate(design)
    checks = finite_differences(tp, design, spec["step"], spec["seed"])
    record = {"statement": "central finite differences along one random direction on a uniform random design; all rows incl. the design-dependent battery load (springs), crash active sets frozen", "grid": half["grid"],
              "battery": tp.battery.summary, "battery_body": half["battery"], "rows": result["rows"], "table": format_report(result["rows"]), "finite_differences": checks, "settings": problem["battery"], "sha": git_sha()}
    tp.close()
    Path(spec["output"]).write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(record["table"], flush=True)
    print(json.dumps({name: [row["analytic"], row["finite_difference"], row["relative_error"]] for name, row in checks.items()}, indent=0, default=float), flush=True)

def geometry_differences(problem, design, steps, seed):
    direction = np.random.default_rng(seed).standard_normal(problem.map.n) * problem.map.free
    def rows(x):
        fields, _ = problem.map.fields(x)
        field = fields[problem.problem["fields"]["volume"]]
        return {row["name"]: row for row in problem.geometry(field[0]) if row["name"].startswith("stand_")}, field[1]
    base, slope = rows(design)
    key = lambda row: "value" if row["g"] is None else "g"
    analytic = {name: float(problem.map.pullback(row["gradient"], slope) @ direction) for name, row in base.items()}
    checks = {name: {"analytic": value, "finite_difference": {}, "relative_error": {}} for name, value in analytic.items()}
    for step in steps:
        plus, minus = rows(design + step * direction)[0], rows(design - step * direction)[0]
        for name, row in base.items():
            difference = float((plus[name][key(row)] - minus[name][key(row)]) / (2 * step))
            checks[name]["finite_difference"][f"{step:g}"], checks[name]["relative_error"][f"{step:g}"] = difference, abs(analytic[name] - difference) / max(abs(difference), 1e-30)
    return checks, {name: {"value": row["value"], "g": row["g"]} for name, row in base.items()}

def stand_fd(cfg):
    spec = cfg["stand_fd"]
    cases = {}
    for name, feet in {**STAND_CASES, "four_feet_gray_background_0.3": FOUR_FEET}.items():
        for half in (False, True):
            domain = stand_domain(half)
            design = stand_design(domain, feet, 0.3 if "gray" in name else 0.02)
            measure = StandStability(domain, {"enabled": True}).measure(design)
            exact = voxel_reserve(domain, design, measure["center_of_gravity_xy_mm"])
            cases[f"{name}_{'half' if half else 'full'}"] = {"smooth_mm": measure["reserve_mm"], "min_over_directions_mm": float(measure["reserves_mm"].min()), "smooth_ground_z_mm": measure["ground_z_mm"], **exact,
                                                             "error_vs_cell_centres_mm": measure["reserve_mm"] - exact["cell_centres_mm"], "error_vs_cell_faces_mm": measure["reserve_mm"] - exact["cell_faces_mm"]}
    config.STAND_STABILITY["enabled"] = True
    half, problem = frame_setup(cfg, spec["shape"])
    tp = TopologyProblem(half, problem, linear_solver=cfg["linear_solver"])
    random = np.random.default_rng(spec["seed"])
    preserve = np.asarray(half["preserve"]).ravel()
    bottoms = cell_centers(half["grid"])[:, 2] - half["grid"]["spacing_mm"][2] / 2
    designs = {"uniform": random.uniform(spec["low"], spec["high"], tp.map.n), "sparse_preserve": np.where(preserve, 1.0, random.uniform(spec["sparse"] / 5, spec["sparse"], tp.map.n))}
    frame = {"grid": half["grid"], "floor_z_mm": half["grid"]["origin_mm"][2], "lowest_preserve_z_mm": float(bottoms[preserve].min()), "components": half["metadata"]["components"], "designs": {}}
    for label, design in designs.items():
        checks, rows = geometry_differences(tp, design, spec["steps"], spec["seed"])
        frame["designs"][label] = {"rows": rows, "finite_differences": checks}
        print(label, json.dumps(rows, default=float), json.dumps(checks, default=float), flush=True)
    tp.close()
    exact = json.loads(Path(spec["exact"]).read_text(encoding="utf-8"))
    fields = {}
    for name, (density, path) in spec["fields"].items():
        stored = json.loads(Path(path).read_text(encoding="utf-8"))
        physical = np.load(density)["density"].astype(float)
        grid = {**stored["grid"], "origin_mm": [0.0, *stored["grid"]["origin_mm"][1:]], "shape": list(physical.shape)}
        domain = {"grid": grid, "allowed": np.ones(physical.shape, dtype=bool), "symmetry": {"axis": 0, "plane_mm": 0.0}, "material": half["material"], "metadata": {"components": stored["components"]}}
        rows = {row["name"]: row["value"] for row in StandStability(domain, {"enabled": True}).rows(physical.ravel())}
        fields[name] = {"density": density, "smooth": rows, "voxel_exact": voxel_reserve(domain, physical, StandStability(domain, {"enabled": True}).measure(physical.ravel())["center_of_gravity_xy_mm"]),
                        "body_exact": {key: exact[name][key] for key in ("reserve_mm", "ground_z_mm", "prop_clearance_mm", "support_area_mm2")}}
        print(name, json.dumps(fields[name], default=float), flush=True)
    record = {"statement": "stand stability rows (STAND_STABILITY): approximation error of the smooth reserve against the exact convex hull of the lowest solid cell layer (cell centres and cell faces) on voxel test cases, full and half (mirror) domains; "
                           "central finite differences of the stand rows through the density map pullback (intermediate field) along one random direction on the coarse frame half domain; smooth rows on the converged fine fields against the exact evaluator measure of their v3 bodies",
              "settings": config.STAND_STABILITY, "cases": cases, "frame": frame, "fields": fields, "sha": git_sha()}
    Path(spec["output"]).write_text(json.dumps(record, indent=1, default=lambda value: value.tolist() if hasattr(value, "tolist") else float(value)), encoding="utf-8")
    for name, case in cases.items():
        print(f"{name}: smooth {case['smooth_mm']:.3f} centres {case['cell_centres_mm']} faces {case['cell_faces_mm']}", flush=True)

def manafly_camera_domain(cfg, tilt):
    import trimesh
    from scipy.ndimage import label
    from deep_frame.frame_evaluation import voxel_grid
    spec, frame = cfg["camera_limits"], json.loads(Path(cfg["manafly"]["frame"]).read_text(encoding="utf-8"))
    mesh = trimesh.load_mesh(frame["stl"], process=True)
    h, fine = spec["spacing_mm"], spec["spacing_mm"] / spec["subdivision"]
    solid, lower, _ = voxel_grid(mesh, fine)
    points = lower + (np.argwhere(solid) + 0.5) * fine
    origin = np.array([0.0, np.floor(mesh.bounds[0][1] / h) * h, 0.0])
    shape = np.ceil((mesh.bounds[1] - origin) / h).astype(int) + 1
    index = np.floor((points[points[:, 0] >= 0] - origin) / h).astype(int)
    counts = np.zeros(shape)
    np.add.at(counts, tuple(index[np.all((index >= 0) & (index < shape), axis=1)].T), 1.0)
    allowed = counts / spec["subdivision"] ** 3 >= 0.5
    grid = {"origin_mm": origin.tolist(), "spacing_mm": [h, h, h], "shape": shape.tolist(), "axis_order": "xyz", "order": "C"}
    part = {**config.COMPONENT_DEFAULTS["camera"], "tilt_deg": tilt}
    axis, up, size, inertia = camera_frame(part)
    clearance = config.TOPOLOGY_CONFIG["component_clearance_mm"]
    center = np.asarray(next(item["center_mm"] for item in frame["components"] if item["type"] == "camera"), dtype=float)
    extent = np.abs(np.column_stack([[1.0, 0.0, 0.0], axis, up])) @ size / 2 + clearance
    centers = cell_centers(grid)
    patch = (np.hypot(centers[:, 1] - center[1], centers[:, 2] - center[2]) <= config.CAMERA_SUPPORT["patch_radius_mm"]) & (centers[:, 0] > extent[0]) & allowed.ravel()
    face = float(np.min(centers[patch, 0]) - h / 2)
    keep_out = {"kind": "box", "min_mm": (center - [face, extent[1], extent[2]]).tolist(), "max_mm": (center + [face, extent[1], extent[2]]).tolist()}
    cut = np.all((centers + h / 2 > np.asarray(keep_out["min_mm"]) + 1e-9) & (centers - h / 2 < np.asarray(keep_out["max_mm"]) - 1e-9), axis=1) & allowed.ravel()
    allowed &= ~cut.reshape(allowed.shape)
    labels, _ = label(allowed)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    dropped = int(allowed.sum() - sizes.max())
    allowed = labels == np.argmax(sizes)
    selectors, top = frame["selectors"], np.asarray(mesh.bounds[1])
    raise_box = lambda box, low, high: {"kind": "box", "min_mm": [*box["min_mm"][:2], low], "max_mm": [*box["max_mm"][:2], high]}
    battery = next(item for item in frame["keep_outs"] if item["name"] == "battery_envelope")
    c = config.COMPONENT_DEFAULTS
    relief = {"point_masses": [{"name": "frame", "region": {"kind": "box", "min_mm": origin.tolist(), "max_mm": (origin + shape * h).tolist()}, "mass_g": float(mesh.volume * config.PRINT_MATERIAL["density_g_cm3"] / 1000)},
                               {"name": "aio15", "region": raise_box(selectors["center_fixtures"][0], -h, spec["stack_height_mm"]), "mass_g": c["aio15"]["mass_g"] / 4}]
                              + [{"name": "aio15", "region": raise_box(box, -h, spec["stack_height_mm"]), "mass_g": c["aio15"]["mass_g"] / 4} for box in selectors["center_fixtures"][1:]]
                              + [{"name": "motor", "region": box, "mass_g": c["motor"]["mass_g"] + c["prop"]["mass_g"]} for box in selectors["motor_fixtures"]]
                              + [{"name": "battery", "region": raise_box(battery, battery["min_mm"][2] - 2 * h, battery["min_mm"][2] + h), "mass_g": c["battery"]["mass_g"]}], "preserve_mass_g": 0.0}
    force = load_numbers(cfg["loads"])["crash_force_n"]
    angle = np.radians(config.CAMERA_SUPPORT["oblique_deg"])
    zones = {"crash_front": [0.0, -force, 0.0], "crash_camera_oblique": (force * np.array([np.sin(angle), -np.cos(angle), 0.0])).tolist()}
    body = {"name": "camera", "mass_g": part["mass_g"], "position_mm": center.tolist(), "inertia_g_mm2": inertia.tolist()}
    thrust = load_numbers(cfg["loads"])["thrust_per_motor_n"]
    cases = [{"name": "thrust_all", "analysis": "static", "fixed_regions": [], "loads": [{"region": box, "force_n": [0.0, 0.0, thrust]} for box in selectors["motor_fixtures"]], "inertia_relief": {**relief, "bodies": [{**body, "force_n": [0.0, 0.0, 0.0]}]}}]
    cases += [{"name": name, "analysis": "static", "fixed_regions": [], "loads": [], "inertia_relief": {**relief, "bodies": [{**body, "force_n": vector}]}} for name, vector in zones.items()]
    s = config.CAMERA_SUPPORT
    zone = {"kind": "box", "min_mm": [-extent[0] - s["zone_mm"]["side"], float(center[1]), float(center[2])], "max_mm": [extent[0] + s["zone_mm"]["side"], float(top[1] + h), float(center[2] + extent[2] + s["zone_mm"]["top"])]}
    front = center + axis * size[1] / 2
    window = [[-size[0] / 2 - s["window_mm"]["side"], size[0] / 2 + s["window_mm"]["side"]], [-size[2] / 2, size[2] / 2 + s["window_mm"]["top"]]]
    domain = {"grid": grid, "allowed": allowed, "preserve": np.zeros_like(allowed), "forbidden": ~allowed, "symmetry": {"axis": 0, "plane_mm": 0.0}, "material": orthotropic_material(config.PRINT_MATERIAL), "load_cases": cases,
              "point_masses": [], "optimizer_settings": {"interface_node_policy": "allowed_adjacent"},
              "camera": {"keep_out": keep_out, "reference_mm": center.tolist(), "center_mm": center.tolist(), "mass_g": part["mass_g"], "screw_axis_yz_mm": center[1:].tolist(), "zone": zone, "zones": zones,
                         "displacement_cases": list(zones), "reference_regions": [raise_box(box, 0.0, spec["stack_height_mm"]) for box in selectors["center_fixtures"]],
                         "coverage": {"axes": [axis.tolist(), [1.0, 0.0, 0.0], up.tolist()], "front_mm": front.tolist(), "window_mm": window, "silhouette_mm": [[-size[0] / 2, size[0] / 2], [-size[2] / 2, size[2] / 2]], "step_mm": h * s["step_fraction"], "depth_mm": list(s["protection_depth_mm"]), "frontal_area_mm2": float(size[0] * size[2])}}}
    return domain, {"plate_face_x_mm": face, "camera_cells_cut": int(cut.sum()), "island_cells_dropped": dropped, "solid_cells_half": int(allowed.sum()), "frame_mass_g": relief["point_masses"][0]["mass_g"], "grid": grid, "tilt_deg": tilt, "zone": zone}

def camera_measure(domain, linear_solver):
    problem = {**deepcopy(PROBLEM), "stiffness": None, "covariance": None, "modal": None, "shadow": None, "monitor": [], "crash": {"cases": ["crash_front"], "ratio": 1.5, "reference": {}}, "camera": deepcopy(config.CAMERA_SUPPORT)}
    tp = TopologyProblem(domain, problem, linear_solver=linear_solver)
    physical = np.asarray(domain["allowed"], dtype=float).ravel()
    report = tp.physical_report(physical)
    patch = float(np.sum(tp.camera.area))
    tp.close()
    rows = {row["name"]: row["value"] for row in report["rows"]}
    return {**rows, "mount_patch_area_mm2": patch, "mount_fraction": rows["camera_mount_area"] / patch, "table": format_report(report["rows"]), "spring_nodes": tp.camera.summary}

def camera_limits(cfg):
    spec = cfg["camera_limits"]
    results = {}
    for tilt in spec["tilts_deg"]:
        domain, info = manafly_camera_domain(cfg, tilt)
        results[f"tilt_{tilt:g}"] = {**info, **camera_measure(domain, spec["linear_solver"])}
        print(results[f"tilt_{tilt:g}"]["table"], flush=True)
    config.TOPOLOGY_CONFIG["camera_support"] = "free"
    half = frame_domain(cfg, cfg["shape"])
    tp = TopologyProblem(half, {**frame_problem(half), "crash": {**PROBLEM["crash"], "cases": []}}, linear_solver=spec["linear_solver"])
    density = embed_field(np.load(cfg["reference_density"])["density"], half["grid"]).ravel()
    tp.body_update(density)
    solved = tp.compliances(density)
    tp.close()
    reference, shielding = {name: solved[name]["compliance_n_mm"] for name in tp.zones}, {name: zone.shielding(density) for name, zone in tp.zones.items()}
    record = {"statement": "ManaFly camera cage measured with the CAMERA_SUPPORT definitions (half model, binary 4/3 mm voxels at >= 50 % volume fraction, inertia relief over frame, AIO, motors, battery and camera), springs on the inner plate faces (gap camera side to plate bridged rigidly by the screw/spacer); "
              "crash references for the zone cases = compliance of the formulation reference density in the free-camera domain under the zone loads", "manafly": results, "crash_reference_n_mm": reference, "reference_shielding": shielding, "reference_density": cfg["reference_density"], "sha": git_sha()}
    Path(spec["output"]).write_text(json.dumps(record, indent=1, default=lambda value: value.tolist() if hasattr(value, "tolist") else float(value)), encoding="utf-8")
    print(json.dumps({key: {name: value[name] for name in ("camera_shift_crash_front", "camera_shift_crash_camera_oblique", "camera_coverage", "camera_mount_area", "mount_fraction", "plate_face_x_mm", "camera_cells_cut")} for key, value in results.items()}, default=float), flush=True)
    print(json.dumps({"crash_reference_n_mm": reference, "reference_shielding": shielding}), flush=True)

def camera_fd(cfg):
    spec = cfg["camera_fd"]
    config.TOPOLOGY_CONFIG.update(camera_support="free", battery_support=spec["battery_support"])
    half, problem = frame_setup(cfg, spec["shape"])
    problem["camera"].update(spec["limits"])
    tp = TopologyProblem(half, problem, linear_solver=cfg["linear_solver"])
    random = np.random.default_rng(spec["seed"])
    record = {"statement": "central finite differences along one random direction; all rows incl. the design-dependent zone loads with rebuilt inertia relief, camera springs, camera shift adjoints, coverage and the free battery; second design with the impact zone near the weight floor",
              "grid": half["grid"], "camera": tp.camera.summary, "camera_body": {key: half["camera"][key] for key in ("center_mm", "reference_mm", "zone", "zones", "fov_cells")}, "limits": spec["limits"], "designs": {}, "sha": git_sha()}
    for label, low, high in (("uniform", spec["low"], spec["high"]), ("sparse_zone", spec["low"], spec["high"])):
        design = random.uniform(low, high, tp.map.n)
        if label == "sparse_zone":
            design[tp.zones["crash_front"].elements] = spec["sparse"]
        result = tp.evaluate(design)
        record["designs"][label] = {"rows": result["rows"], "table": format_report(result["rows"]), "finite_differences": finite_differences(tp, design, spec["step"], spec["seed"])}
        print(record["designs"][label]["table"], flush=True)
        print(json.dumps({name: [row["analytic"], row["finite_difference"], row["relative_error"]] for name, row in record["designs"][label]["finite_differences"].items()}, indent=0, default=float), flush=True)
    tp.close()
    Path(spec["output"]).write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")

def body_properties(stl, layout, density=config.PRINT_MATERIAL["density_g_cm3"]):
    import trimesh
    from deep_frame.frame_run import LayoutModel
    mesh = trimesh.load_mesh(stl, process=True)
    frame = {"mass_g": mesh.volume * density / 1000, "center_mm": np.asarray(mesh.center_mass).tolist(), "inertia_g_mm2": (np.asarray(mesh.moment_inertia) * density / 1000).tolist()}
    agility = layout.agility()
    craft = LayoutModel({**agility["setup"], "frame": frame}).evaluate(agility["layout"])
    return mesh, {"stl": str(stl), "sha256_16": digest(stl), "frame": frame, "craft": {key: craft[key] for key in ("mass_g", "center_of_mass_mm", "cg_above_rotor_plane_mm", "inertia_g_mm2", "alpha_rad_s2", "alpha_min_rad_s2", "limiting_axis", "layout", "violated")}}

def battery_compare(cfg):
    import trimesh
    from PIL import Image, ImageDraw, ImageFont
    from deep_frame.frame_run import FrameLayout
    from tools.neural_study import render, _camera
    spec = cfg["compare"]
    meshes, summary = {}, {}
    for label, body in spec["bodies"].items():
        meshes[label], summary[label] = body_properties(body["stl"], FrameLayout(body["request"]))
        run = Path(body["stl"]).parent
        summary[label].update(note=body.get("note"), evaluation=body_summary(str(run), None, cfg["comparison"]) if (run / "evaluation.json").is_file() else None)
    labels = [label for label, body in spec["bodies"].items() if body.get("figure", True)]
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    rows = []
    for name, (direction, up) in config.RUN_SETTINGS["views"].items():
        right = _camera(direction, up)[1]
        width = max(float(np.ptp(mesh.vertices @ right)) for mesh in meshes.values())
        parts, offsets = [], []
        for index, label in enumerate(labels):
            mesh = meshes[label].copy()
            mesh.apply_translation(-mesh.bounds.mean(axis=0) + right * index * (width + spec["gap_mm"]))
            parts.append(mesh)
        image = render(trimesh.util.concatenate(parts), tuple(direction), tuple(up), size=tuple(spec["size"]))
        low, top, high, bottom = image.convert("L").point(lambda v: 255 if v < 250 else 0).getbbox() or (0, 0, *image.size)
        image = image.crop((0, max(top - 20, 0), image.width, min(bottom + 20, image.height)))
        canvas = Image.new("RGB", (image.width, image.height + 60), "white")
        canvas.paste(image, (0, 60))
        draw = ImageDraw.Draw(canvas)
        for index, label in enumerate(labels):
            draw.text((int((index + 0.5) * image.width / len(labels)) - 220, 10), f"{label} ({name})", fill="black", font=font)
        rows.append(canvas)
    figure = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    for index, row in enumerate(rows):
        figure.paste(row, (0, sum(previous.height for previous in rows[:index])))
    Path(spec["output"]).parent.mkdir(parents=True, exist_ok=True)
    figure.save(spec["output"])
    Path(spec["summary"]).write_text(json.dumps({"bodies": summary, "renderer": "tools.neural_study.render, both bodies in one scene per view (same camera, same px/mm), side by side along the screen axis"}, indent=1, default=float), encoding="utf-8")
    print(json.dumps({label: {"mass_g": entry["frame"]["mass_g"], "alpha": entry["craft"]["alpha_rad_s2"]} for label, entry in summary.items()}, default=float), flush=True)

class MemoryProbe:
    def __init__(self, interval, log=None):
        import cupy
        import psutil
        self.cp, self.process, self.interval, self.samples, self.phases, self.running = cupy, psutil.Process(), interval, [], [], True
        self.log, self.started = (open(log, "a") if log else None), perf_counter()
        self.cp.cuda.runtime.memGetInfo()
        self.cp.zeros(1)
        paths = ",".join(f"'\\GPU Process Memory(pid_{os.getpid()}_*)\\{name} Usage'" for name in ("Dedicated", "Shared"))
        script = (f"Get-Counter -Counter {paths} -SampleInterval 1 -Continuous -ErrorAction SilentlyContinue | ForEach-Object {{ $c = $_.CounterSamples; "
                  "$d = ($c | Where-Object Path -like '*dedicated*' | Measure-Object CookedValue -Sum).Sum; $s = ($c | Where-Object Path -like '*shared*' | Measure-Object CookedValue -Sum).Sum; "
                  "[Console]::Out.WriteLine(\"$d $s\"); [Console]::Out.Flush() }")
        self.readers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for command in
                        (["powershell", "-NoProfile", "-Command", script], ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", str(int(interval * 1000))])]
        self.threads = [threading.Thread(target=self.read, args=(reader, kind), daemon=True) for reader, kind in zip(self.readers, ("counter", "smi"))] + [threading.Thread(target=self.poll, daemon=True)]
        for thread in self.threads:
            thread.start()
    def read(self, reader, kind):
        for line in reader.stdout:
            try:
                values = [float(item) for item in line.split()]
            except ValueError:
                continue
            if kind == "counter" and len(values) == 2:
                self.record({"dedicated_bytes": values[0], "shared_bytes": values[1]})
            elif kind == "smi" and len(values) == 1:
                self.record({"smi_used_bytes": values[0] * 2 ** 20})
    def record(self, values):
        clock = perf_counter()
        self.samples.append((clock, values))
        if self.log:
            self.log.write(json.dumps({"t": round(clock - self.started, 2), **{key.replace("_bytes", "_gb"): round(value / 2 ** 30, 4) for key, value in values.items()}}) + "\n")
            self.log.flush()
    def poll(self):
        pool = self.cp.get_default_memory_pool()
        while self.running:
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.record({"rss_bytes": self.process.memory_info().rss, "pool_used_bytes": pool.used_bytes(), "pool_total_bytes": pool.total_bytes(), "device_used_bytes": total - free})
            threading.Event().wait(self.interval)
    def phase(self, name):
        probe = self
        class Phase:
            def __enter__(self):
                self.started = perf_counter()
                return self
            def __exit__(self, *error):
                threading.Event().wait(1.5)
                probe.phases.append((name, self.started, perf_counter()))
        return Phase()
    def peaks(self, name, reduce=max):
        _, started, ended = next(phase for phase in self.phases if phase[0] == name)
        found = {}
        for clock, values in list(self.samples):
            if started <= clock <= ended:
                for key, value in values.items():
                    found.setdefault(key, []).append(value)
        return {key.replace("_bytes", "_gb"): float(reduce(value)) / 2 ** 30 for key, value in found.items()}
    def summary(self):
        names = [phase[0] for phase in self.phases]
        return {"interval_s": self.interval, "idle_mean": self.peaks(names[0], np.mean), "peaks": {name: self.peaks(name) for name in names},
                "last": {name: self.peaks(name, lambda value: value[-1]) for name in names}, "device_total_gb": self.cp.cuda.runtime.memGetInfo()[1] / 2 ** 30}
    def close(self):
        self.running = False
        for reader in self.readers:
            reader.kill()
        if self.log:
            self.log.close()

def factorizations(system):
    return sum(len(solver.timings) for solver in system.gpu_solvers.values()) + sum(len(entry["solves"]) for entry in system.gpu_solver_history)

def multigrid_state(system):
    if system.multigrid is None:
        return {}
    torch, mg = system.multigrid.torch, system.multigrid
    return {"multigrid_solves": len(mg.statistics), "multigrid_coarsest_factorizations": sum(len(solver.timings) for solver in mg.coarsest_solvers.values()), "torch_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}

def multigrid_summary(statistics):
    return [{"columns": len(row["iterations"]), "batches": row["batches"], "floating": row["floating"], "levels": row["levels"], "coarsest_dofs": row["coarsest_dofs"], "iterations_max": max(row["iterations"]), "iterations_mean": float(np.mean(row["iterations"])),
             "relative_residual_max": max(row["relative_residual"]), "seconds": row["seconds"], "setup_s": row["setup_s"], "rhs_kernel_component_max": max(row["rhs_kernel_component"] or [0.0])} if "batches" in row else
            {"lobpcg_iterations": row["iterations"], "converged": row["converged"], "levels": row["levels"], "coarsest_dofs": row["coarsest_dofs"], "relative_residual_max": max(row["relative_residual"]), "seconds": row["seconds"], "setup_s": row["setup_s"]} for row in statistics]

def cudss_factors(system):
    rows = [{"owner": "formulation", "key": str(key)[:40], "dofs": solver.shape[0]} for key, solver in system.gpu_solvers.items()]
    if system.multigrid is not None:
        rows += [{"owner": "multigrid_coarsest", "key": str(key)[:40], "dofs": solver.shape[0]} for key, solver in system.multigrid.coarsest_solvers.items()]
    return rows

def factor_inventory(tp):
    system, rows = tp.system, []
    for key, members in system.groups.items():
        part = members[0][1]
        rows.append({"solver": "static", "sign": part["sign"], "cases": [case["name"] for case, _ in members], "rhs": len(members), "fixed_dofs": len(part["fixed"]), "free_dofs": len(part["free"]), "key": key,
                     **({"factorized_with": [case["name"] for case, _ in system.groups[system.shared[key]["host"]]][:1], "constraint_columns": len(system.shared[key]["constrained"])} if key in (system.shared or {}) else {})})
    for part in tp.modal.parts if tp.modal is not None else []:
        rows.append({"solver": "modal", "sign": part["sign"], "cases": [tp.problem["modal"]["case"]], "rhs": tp.problem["modal"]["modes"], "fixed_dofs": len(system.active_dofs) - len(part["free"]), "free_dofs": len(part["free"]), "key": part["key"]})
    for row in rows:
        solver = system.gpu_solvers.get(row.pop("key"))
        if solver is not None:
            row.update(nnz=len(solver.indices), analysis_s=solver.analysis_s, **(solver.memory_estimates or {}))
    cases = [{"name": case["name"], "analysis": case["analysis"], "support": "inertia_relief" if "inertia_relief" in case else "fixed", "fixed_dofs": len(case["fixed"]), "parts": len(case["parts"])} for case in system.cases]
    return rows, cases

def evaluation_arrays(result):
    values = {row["name"]: row["value"] for row in result["rows"]}
    return {"objective": np.float64(result["objective"]), "objective_gradient": result["objective_gradient"], "constraints": result["constraints"], "constraint_gradients": result["constraint_gradients"],
            "names": np.array(result["names"]), "values": np.array([values[name] for name in result["names"]]), "monitored": np.array([row["name"] for row in result["rows"] if row["g"] is None and isinstance(row["value"], float)]),
            "monitored_values": np.array([row["value"] for row in result["rows"] if row["g"] is None and isinstance(row["value"], float)]), "f1": np.float64(values["f1"]), "mass_g": np.float64(result["mass_g"]), "beta": np.float64(result["beta"])}

def deviation(a, b):
    scale = lambda x: np.maximum(np.abs(x), 1e-30)
    rows = [np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30) for x, y in zip(a["constraint_gradients"], b["constraint_gradients"])]
    return {"objective_rel": float(abs(a["objective"] - b["objective"]) / scale(b["objective"])), "constraints_g_abs": float(np.max(np.abs(a["constraints"] - b["constraints"]))),
            "values_rel": float(np.max(np.abs(a["values"] - b["values"]) / scale(b["values"]))), "values_by_name": {str(name): {"value": float(y), "abs": float(abs(x - y)), "rel": float(abs(x - y) / scale(y))} for name, x, y in zip(b["names"], a["values"], b["values"])}, "f1_rel": float(abs(a["f1"] - b["f1"]) / b["f1"]),
            "gradients_rel_norm": float(max(rows)), "gradients_rel_norm_by_name": dict(zip([str(name) for name in b["names"]], map(float, rows))),
            "objective_gradient_rel_norm": float(np.linalg.norm(a["objective_gradient"] - b["objective_gradient"]) / np.linalg.norm(b["objective_gradient"])),
            "monitored_by_name": {str(name): {"value": float(y), "abs": float(abs(x - y)), "rel": float(abs(x - y) / scale(y))} for name, x, y in zip(b.get("monitored", []), a.get("monitored_values", []), b.get("monitored_values", []))}}

def solver_grid(cfg, spec, grid, probe, out):
    half, problem = frame_setup(cfg, cfg["shape"] if grid == "fine" else cfg["coarse_shape"])
    problem = {**problem, "multigrid": spec["multigrid"], "share_static": spec["share_static"], "modal": {**problem["modal"], **spec.get("modal", {})}}
    free, total = probe.cp.cuda.runtime.memGetInfo()
    with probe.phase(grid + "_build"):
        clock = perf_counter()
        tp = TopologyProblem(half, problem, linear_solver=cfg["mma"]["linear_solver"])
        built = perf_counter() - clock
    while tp.advance():
        pass
    source = Path(spec["start"] or Path(spec["run"]) / grid) / "design.npz"
    design = np.load(source)["design"]
    if spec["start"]:
        design = prolongate(design, read(source.parent / "result.json")["grid"], half)
    evaluations, arrays = [], []
    for index in range(spec["evaluations"]):
        if index == 1:
            for part in tp.modal.parts:
                part["vectors"] = None
        counted = factorizations(tp.system)
        with probe.phase(f"{grid}_evaluation_{index}"):
            clock = perf_counter()
            result = tp.evaluate(design)
            seconds = perf_counter() - clock
        arrays.append(evaluation_arrays(result))
        evaluations.append({"index": index, "modal": "cold" if index < 2 else "warm", "seconds": seconds, "factorizations": factorizations(tp.system) - counted, **multigrid_state(tp.system), **probe.peaks(f"{grid}_evaluation_{index}")})
    inventory, cases = factor_inventory(tp)
    factors = cudss_factors(tp.system)
    np.savez_compressed(out / f"{grid}.npz", **arrays[0])
    optimizer, iterations = MMAOptimizer(tp, cfg["mma"]["settings"]), []
    x = optimizer.start(design)
    state = optimizer.fresh(x)
    with probe.phase(grid + "_mma"):
        for index in range(spec["mma_iterations"]):
            clock, counted = perf_counter(), factorizations(tp.system)
            x = optimizer.step(x, tp.evaluate(x), state)
            iterations.append({"seconds": perf_counter() - clock, "factorizations": factorizations(tp.system) - counted})
    multigrid = {**multigrid_state(tp.system), "settings": tp.system.multigrid.settings, "solves": multigrid_summary(tp.system.multigrid.statistics)} if tp.system.multigrid is not None else None
    tp.close()
    probe.cp.get_default_memory_pool().free_all_blocks()
    if multigrid is not None:
        tp.system.multigrid.torch.cuda.empty_cache()
        tp.system.multigrid.torch.cuda.reset_peak_memory_stats()
    record = {"multigrid": multigrid, "grid": half["grid"], "design": str(source), "design_sha256_16": digest(source), "dofs": tp.system.ndof, "active_dofs": len(tp.system.active_dofs), "build_s": built,
              "gpu_before_build": {"device_free_gb": free / 2 ** 30, "device_total_gb": total / 2 ** 30}, "build": probe.peaks(grid + "_build"), "evaluations": evaluations,
              "noise_floor": deviation(arrays[1], arrays[0]), "mma": {"iterations": iterations, "seconds_per_iteration": float(np.mean([row["seconds"] for row in iterations])), **probe.peaks(grid + "_mma")},
              "factors": inventory, "cudss_factors": factors, "linear_solver": tp.system.linear_solver, "cases": cases, "names": result["names"], "f1_hz": float(arrays[0]["f1"]), "mass_g": float(arrays[0]["mass_g"])}
    if spec["reference"]:
        reference = np.load(Path(spec["reference"]) / f"{grid}.npz")
        record["deviation"] = deviation(arrays[0], {key: reference[key] for key in reference.files})
    return record

def solver_memory(cfg):
    spec = cfg["solver_memory"]
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    probe = MemoryProbe(spec["sample_s"])
    record = {"sha": git_sha(), "config": spec, "grids": {}}
    try:
        for grid in spec["grids"]:
            record["grids"][grid] = solver_grid(cfg, spec, grid, probe, out)
    finally:
        probe.close()
    record["peak_wset_gb"] = probe.process.memory_info().peak_wset / 2 ** 30 if hasattr(probe.process.memory_info(), "peak_wset") else None
    lines = ["| grid | factorizations / evaluation | s / evaluation (cold, warm) | s / MMA iteration | dedicated peak GB | shared peak GB | nvidia-smi total GB | CuPy pool GB | host RSS GB |", "|---|---|---|---|---|---|---|---|---|"]
    for grid, item in record["grids"].items():
        phases = item["evaluations"] + [item["mma"]]
        peak = lambda key: max(phase.get(key, 0.0) for phase in phases)
        lines.append(f"| {grid} | {item['evaluations'][-1]['factorizations']} | {item['evaluations'][0]['seconds']:.1f}, {item['evaluations'][-1]['seconds']:.1f} | {item['mma']['seconds_per_iteration']:.1f} | "
                     f"{peak('dedicated_gb'):.2f} | {peak('shared_gb'):.2f} | {peak('smi_used_gb'):.2f} | {peak('pool_total_gb'):.2f} | {peak('rss_gb'):.2f} |")
    record["table"] = "\n".join(lines)
    (out / "result.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(record["table"], flush=True)
    for grid, item in record["grids"].items():
        print(json.dumps({grid: {"noise_floor": item["noise_floor"], "deviation": item.get("deviation")}}, default=float), flush=True)

SETUP_STEPS = [("deep_frame.topology_optimization", "DensityMap", "__init__", "filter"), ("deep_frame.topology_optimization", "DensityMap", "fields", "projections"),
               ("deep_frame.topology_optimization", "HexElasticity", "__init__", "elasticity"), ("deep_frame.topology_optimization", None, "regular_grid", "elasticity.grid"),
               ("deep_frame.topology_optimization", "HexElasticity", "assembly", "elasticity.assembly_indices"), ("deep_frame.topology_optimization", "HexElasticity", "_register", "load_cases"),
               ("deep_frame.topology_problem", "InterfaceCovariance", "__init__", "covariance"), ("deep_frame.topology_optimization", "HexElasticity", "modal_constraint", "modal"),
               ("deep_frame.topology_multigrid", "GeometricMultigrid", "__init__", "multigrid"), ("deep_frame.topology_multigrid", "GeometricMultigrid", "hierarchy", "multigrid.hierarchy"),
               ("deep_frame.topology_multigrid", "GeometricMultigrid", "_coarsest_solver", "multigrid.coarsest_cudss"), ("deep_frame.topology_optimization", "CudaDirectSolver", "__init__", "cudss_factor"),
               ("deep_frame.topology_problem", "TopologyProblem", "evaluate", "evaluation"), ("deep_frame.topology_optimization", "HexElasticity", "solve", "evaluation.static"),
               ("deep_frame.topology_optimization", "HexElasticity", "_share_plan", "evaluation.static.share_plan"), ("deep_frame.topology_optimization", "HexElasticity", "_collect", "evaluation.static.collect"),
               ("deep_frame.topology_multigrid", "GeometricMultigrid", "solve", "evaluation.static.pcg"), ("deep_frame.topology_problem", "InterfaceCovariance", "measure", "evaluation.covariance"),
               ("deep_frame.topology_multigrid", "MultigridModal", "measure", "evaluation.modal"), ("deep_frame.topology_optimization", "ModalConstraint", "measure", "evaluation.modal"),
               ("deep_frame.topology_optimization", "DensityMap", "pullback", "evaluation.pullback")]

class SetupTrace:
    def __init__(self, spec, log):
        import psutil
        self.psutil, self.process, self.spec, self.log = psutil, psutil.Process(), spec, open(log, "w")
        self.samples, self.events, self.depth, self.running, self.started = [], [], 0, True, perf_counter()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
    def rss(self):
        return self.process.memory_info().rss / 2 ** 30
    def poll(self):
        while self.running:
            self.samples.append((perf_counter(), self.rss()))
            if self.psutil.virtual_memory().available / 2 ** 30 < self.spec["min_free_gb"]:
                self.log.write(json.dumps({"aborted": "free host RAM below " + str(self.spec["min_free_gb"]) + " GB", "t": perf_counter() - self.started, "rss_gb": self.samples[-1][1]}) + "\n")
                self.log.close()
                os._exit(3)
            threading.Event().wait(self.spec["sample_s"])
    def step(self, label, function):
        trace = self
        def wrapped(*args, **kwargs):
            clock, before, trace.depth = perf_counter(), trace.rss(), trace.depth + 1
            try:
                return function(*args, **kwargs)
            finally:
                trace.depth -= 1
                end, after = perf_counter(), trace.rss()
                peak = max([value for moment, value in list(trace.samples) if clock <= moment <= end] + [before, after])
                event = {"step": label, "depth": trace.depth, "t": clock - trace.started, "seconds": end - clock, "rss_before_gb": before, "rss_after_gb": after, "rss_peak_gb": peak}
                trace.events.append(event)
                trace.log.write(json.dumps(event) + "\n")
                trace.log.flush()
        return wrapped
    def install(self, steps=SETUP_STEPS):
        import importlib
        for module, owner, name, label in steps:
            target = importlib.import_module(module)
            target = getattr(target, owner) if owner else target
            original = target.__dict__.get(name)
            if isinstance(original, property):
                setattr(target, name, property(self.step(label, original.fget)))
            elif original is not None:
                setattr(target, name, self.step(label, original))
    def summary(self):
        steps = {}
        for event in self.events:
            row = steps.setdefault(event["step"], {"calls": 0, "seconds": 0.0, "rss_added_gb": 0.0, "rss_peak_gb": 0.0, "rss_after_first_gb": event["rss_after_gb"]})
            row["calls"] += 1
            row["seconds"] += event["seconds"]
            row["rss_added_gb"] += event["rss_after_gb"] - event["rss_before_gb"]
            row["rss_peak_gb"] = max(row["rss_peak_gb"], event["rss_peak_gb"])
        return steps
    def close(self):
        self.running = False
        self.thread.join()
        self.log.close()

def gpu_state():
    import cupy
    free, total = cupy.cuda.runtime.memGetInfo()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
    torch = sys.modules.get("torch")
    live = torch is not None and torch.cuda.is_initialized()
    return {"device_used_gb": (total - free) / 2 ** 30, "smi_used_gb": float(smi[0]) / 1024 if smi else None, "cupy_pool_gb": cupy.get_default_memory_pool().total_bytes() / 2 ** 30,
            "torch_reserved_gb": torch.cuda.memory_reserved() / 2 ** 30 if live else 0.0, "torch_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30 if live else 0.0}

def setup_memory(cfg):
    spec = cfg["setup_memory"]
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    idle = gpu_state()
    trace = SetupTrace(spec, out / "steps.jsonl")
    trace.install()
    record = {"sha": git_sha(), "config": spec, "linear_solver": cfg["mma"]["linear_solver"], "rss_idle_gb": trace.rss(), "gpu_idle": idle}
    try:
        half, problem = trace.step("domain", frame_setup)(cfg, spec["shape"] or cfg["shape"])
        tp = trace.step("problem", TopologyProblem)(half, problem, linear_solver=cfg["mma"]["linear_solver"])
        record.update(grid=half["grid"], dofs=tp.system.ndof, active_elements=int(np.count_nonzero(tp.system.active_elements)), chosen_solver=tp.system.linear_solver, rss_after_problem_gb=trace.rss(), gpu_after_problem=gpu_state())
        while tp.advance():
            pass
        if spec["evaluate"]:
            source = Path(spec["start"])
            design = prolongate(np.load(source / "design.npz")["design"], read(source / "result.json")["grid"], half)
            result = tp.evaluate(design)
            record.update(mass_g=result["mass_g"], max_violation=result["max_violation"], rss_after_evaluation_gb=trace.rss(), gpu_after_evaluation=gpu_state())
        tp.close()
    finally:
        trace.close()
    record.update(steps=trace.summary(), rss_peak_gb=max(value for _, value in trace.samples), events=trace.events)
    lines = ["| step | calls | s | RSS added GB | RSS peak GB |", "|---|---|---|---|---|"] + [f"| {name} | {row['calls']} | {row['seconds']:.1f} | {row['rss_added_gb']:+.2f} | {row['rss_peak_gb']:.2f} |" for name, row in record["steps"].items()]
    record["table"] = "\n".join(lines)
    (out / "result.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(record["table"], flush=True)
    print(json.dumps({key: record.get(key) for key in ("grid", "dofs", "chosen_solver", "rss_after_problem_gb", "rss_after_evaluation_gb", "rss_peak_gb", "gpu_after_problem", "gpu_after_evaluation")}, default=float), flush=True)

def checkpoint(path):
    def save(x, history, levels):
        np.savez_compressed(path, design=x, level=history[-1]["level"], iteration=len(history))
    return save

def optimize_stage(cfg, half, problem, design, out, start_level):
    out.mkdir(parents=True, exist_ok=True)
    tp = TopologyProblem(half, problem, linear_solver=cfg["mma"]["stage_solvers"].get(out.name, cfg["mma"]["linear_solver"]))
    if cfg["mma"]["resume"] and (out / "checkpoint.npz").is_file():
        saved = np.load(out / "checkpoint.npz")
        design, start_level = saved["design"], int(saved["level"])
    with (out / "iterations.jsonl").open("a") as log:
        progress = lambda row: (log.write(json.dumps(row, default=float) + "\n"), log.flush())
        result = MMAOptimizer(tp, {**cfg["mma"]["settings"], "start_level": start_level}).run(design, progress, checkpoint(out / "checkpoint.npz"))
    report = tp.report(result["design"])
    fields, _ = tp.map.fields(result["design"])
    physical = fields["intermediate"][0]
    free = tp.map.free
    gray = float(np.mean((physical[free] > cfg["mma"]["gray"][0]) & (physical[free] < cfg["mma"]["gray"][1])))
    np.savez_compressed(out / "design.npz", design=result["design"])
    np.savez_compressed(out / "density_half.npz", density=physical.reshape(half["grid"]["shape"]).astype(np.float32))
    record = {"status": result["status"], "iterations": result["iterations"], "runtime_s": result["runtime_s"], "seconds_per_iteration": result["seconds_per_iteration"], "levels": result["levels"],
              "start_level": start_level, "grid": half["grid"], "filter_radius_mm": tp.radius, "free_cells": int(np.count_nonzero(free)), "gray_fraction": gray, "mass_g": report["mass_g"],
              "mass_by_field_g": report["mass_by_field_g"], "rows": report["rows"], "table": format_report(report["rows"]), "max_violation": report["max_violation"], "mma": result["settings"], "linear_solver": tp.system.linear_solver}
    tp.close()
    (out / "result.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("status", "iterations", "seconds_per_iteration", "mass_g", "max_violation", "gray_fraction")}, default=float), flush=True)
    print(record["table"], flush=True)
    return result["design"], record

def export_body(cfg, physical, out):
    from tools.neural_study import R2Domain, finish, upsample
    settings = patched_settings(cfg)
    fine_full, _ = R2Domain(settings).build(settings["fine_shape"], cfg["reference_fraction"])
    density = upsample(physical, fine_full)
    np.savez_compressed(out / "density_fine.npz", density=density.astype(np.float32))
    return finish(out, density, fine_full, settings, None)

def body_start(cfg, design, half):
    from deep_frame.topology_geometry import region_contains
    design, centers, allowed = design.copy(), cell_centers(half["grid"]), np.asarray(half["allowed"]).ravel()
    for name, spec in cfg["mma"]["body_start"].items():
        if name not in half or not spec:
            continue
        near = region_contains(centers, half[name]["keep_out"], spec["cells"] * np.asarray(half["grid"]["spacing_mm"]))
        if spec.get("zone"):
            near |= region_contains(centers, half[name]["zone"])
        design[near & allowed] = np.maximum(design[near & allowed], spec["density"])
    return design

def frame_mma(cfg):
    started = perf_counter()
    root = Path(cfg["mma"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    probe = MemoryProbe(cfg["mma"]["memory_s"], root / "memory.jsonl") if cfg["mma"]["memory_s"] else None
    phase = lambda name: probe.phase(name) if probe else nullcontext()
    try:
        with phase("idle"):
            threading.Event().wait(10.0 if probe else 0.0)
        with phase("fine_setup"):
            fine, problem = frame_setup(cfg, cfg["shape"])
        reference = embed_field(np.load(cfg["reference_density"])["density"], fine["grid"]).ravel()
        design, stages, level = reference, {}, 0
        if cfg["mma"]["start"]:
            source = Path(cfg["mma"]["start"])
            design, level = prolongate(np.load(source / "design.npz")["design"], read(source / "result.json")["grid"], fine), cfg["mma"]["fine_start_level"]
        elif cfg["mma"]["coarse"]:
            with phase("coarse"):
                coarse, coarse_problem = frame_setup(cfg, cfg["coarse_shape"])
                start = body_start(cfg, prolongate(reference, fine["grid"], coarse), coarse)
                result, stages["coarse"] = optimize_stage(cfg, coarse, coarse_problem, start, root / "coarse", 0)
            design, level = prolongate(result, coarse["grid"], fine), cfg["mma"]["fine_start_level"]
        if cfg["mma"]["until"] == "coarse":
            return
        with phase("fine"):
            design, stages["fine"] = optimize_stage(cfg, fine, problem, design, root / "fine", level)
        if cfg["mma"]["until"] == "fine":
            return
        out = root / cfg["mma"]["variant"]
        out.mkdir(parents=True, exist_ok=True)
        physical = np.load(root / "fine" / "density_half.npz")["density"]
        np.savez_compressed(out / "density_half.npz", density=physical)
        with phase("export"):
            body = export_body(cfg, physical, out)
    finally:
        if probe:
            probe.close()
            (root / "memory.json").write_text(json.dumps(probe.summary(), indent=1, default=float), encoding="utf-8")
    info = {"variant": cfg["mma"]["variant"], "method": "SIMP-MMA (mmapy 0.3.1), shared formulation " + git_sha(), "stages": stages, "body": body, "total_runtime_s": perf_counter() - started,
            "iterations": sum(stage["iterations"] for stage in stages.values()), "reference": cfg["reference_density"], "start": cfg["mma"]["start"], "memory": probe.summary() if probe else None}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"iterations": info["iterations"], "total_runtime_s": info["total_runtime_s"], "mass_g_body": body["mass_g"], "bodies": body["bodies"], "watertight": body["watertight"]}, default=float), flush=True)

class ResultRun(FrameRun):
    def __init__(self, request, result, stages):
        self.result = Path(result)
        super().__init__(request, stages)
    def available(self, stage):
        return stage == "optimization" or super().available(stage)
    def optimization(self, domain):
        self.manifest["stages"]["optimization"] = {"status": "ran", "source": str(self.result), "method": "SIMP-MMA on the shared formulation", **_git(str(ROOT))}
        self.save()
        return self.result

class V3Run(ResultRun):
    def __init__(self, request, result, stages, compare, sha):
        self.sha = sha
        super().__init__(request, result, stages)
        self.grid = {**self.grid, "compute": {**self.grid["compute"], "reconstruction": stages["reconstruction"]["compute"]},
                     "reconstruction": {**self.grid["reconstruction"], **({"compare_bodies": [str(compare)], "compare_labels": ["recon_1to1"]} if compare else {})}}
    def reconstruction(self, density):
        mesh = super().reconstruction(density)
        self.manifest["stages"]["reconstruction"].update(sha=self.sha, branch="feature/recon-v3 (git archive copy)", method="reconstruction v3 (member splines)")
        self.save()
        return mesh

def v3_source(cfg):
    spec = cfg["v3"]
    sha = subprocess.check_output(["git", "-C", spec["worktree"], "rev-parse", spec["ref"]], text=True).strip()
    target = Path(spec["copy"]) / sha[:10]
    if not (target / "run.py").is_file():
        target.mkdir(parents=True, exist_ok=True)
        tarfile.open(fileobj=io.BytesIO(subprocess.run(["git", "-C", spec["worktree"], "archive", sha], check=True, capture_output=True).stdout)).extractall(target)
    return target, sha

def body_run(cfg, suffix, request, stages, runs):
    root, method = Path(cfg["mma"]["root"]).resolve(), cfg["mma"]["method"]
    named = {**request, "name": f"{method}_{suffix}", "reconstruction": suffix != "raw"}
    if suffix == "v3":
        source, sha = v3_source(cfg)
        stages = {**stages, "reconstruction": {**stages["reconstruction"], "worktree": source.as_posix(), "argv": cfg["v3"]["argv"], "compute": cfg["v3"]["compute"]}}
        compare = runs.get(cfg["v3"]["compare"])
        run = V3Run(named, root / cfg["mma"]["variant"], stages, compare and Path(compare) / "frame.stl", sha)
    else:
        run = ResultRun(named, root / cfg["mma"]["variant"], stages)
    manifest = run.run()
    path = (ROOT / RUN_SETTINGS["root"] / manifest["name"]).resolve()
    target = Path(cfg["mma"]["viewer"]) / cfg["mma"]["viewer_names"][suffix].format(method=method) / "geometry.stl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path / "frame.stl", target)
    print(json.dumps({"run": manifest["name"], "status": manifest["status"], "stages": {name: entry["status"] for name, entry in manifest["stages"].items()}}), flush=True)
    return suffix, str(path)

def frame_runs(cfg):
    root = Path(cfg["mma"]["root"]).resolve()
    stages = {name: {**spec, "worktree": ROOT.as_posix()} for name, spec in STAGES.items()}
    request = deepcopy(cfg["layout"]) if cfg["layout"] else json.loads((Path(RUN) / "config.json").read_text(encoding="utf-8"))
    runs = json.loads((root / "runs.json").read_text(encoding="utf-8")) if (root / "runs.json").is_file() else {}
    todo = [suffix for suffix in cfg["mma"]["bodies"] if suffix not in runs]
    with ThreadPoolExecutor(max(len(todo), 1)) as pool:
        runs.update(pool.map(lambda suffix: body_run(cfg, suffix, request, stages, runs), todo))
    (root / "runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")

def manafly_check(cfg):
    spec = json.loads(Path(cfg["manafly"]["frame"]).read_text(encoding="utf-8"))
    out = Path(cfg["mma"]["root"]).resolve() / "manafly"
    out.mkdir(parents=True, exist_ok=True)
    frame = out / "frame.json"
    frame.write_text(json.dumps({**spec, "output": str(out / "evaluation")}, indent=1), encoding="utf-8")
    subprocess.check_call([cfg["mma"]["evaluation_python"], str(ROOT / "tools" / "evaluate_frame.py"), "run", str(frame)], cwd=ROOT)

def compose(cfg):
    import trimesh
    from PIL import Image
    from tools.neural_study import render_views
    from tools.reconstruction_study import compose_main
    root = Path(cfg["mma"]["root"]).resolve()
    runs = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    manafly = root / "manafly" / "renders"
    manafly.mkdir(parents=True, exist_ok=True)
    views = {name: (tuple(direction), tuple(up)) for name, (direction, up) in RUN_SETTINGS["views"].items()}
    source = Path(cfg["mma"]["manafly_renders"])
    missing = {name: view for name, view in views.items() if not (source / f"{name}.png").is_file()}
    for name in set(views) - set(missing):
        shutil.copy2(source / f"{name}.png", manafly / f"{name}.png")
    if missing:
        render_views(trimesh.load_mesh(json.loads(Path(cfg["manafly"]["frame"]).read_text(encoding="utf-8"))["stl"], process=True), manafly, missing)
    folders = {**{f"previous_{suffix}": Path(path) / "renders" for suffix, path in cfg["comparison"]["previous"].items()}, **{suffix: Path(path) / "renders" for suffix, path in runs.items()}, "manafly": manafly}
    for figure in cfg["mma"]["figures"]:
        output = Path(figure["output"].format(method=cfg["mma"]["method"]))
        output = output if output.is_absolute() else root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for name in views:
            row = output.with_name(f"{output.stem}_{name}.png")
            compose_main({"panels": [str(folders[panel] / f"{name}.png") for panel in figure["panels"]], "labels": [f"{label} ({name})" for label in figure["labels"]], "output": str(row)})
            rows.append(Image.open(row))
        canvas = Image.new("RGB", (max(image.width for image in rows), sum(image.height for image in rows)), "white")
        for index, image in enumerate(rows):
            canvas.paste(image, (0, sum(previous.height for previous in rows[:index])))
        canvas.save(output)

def fea_values(path):
    fea = (json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}).get("fea") or {}
    return {"arm_tip_n_per_mm": fea.get("stiffness_n_per_mm"), "f1_hz": (fea.get("eigenfrequencies_hz") or [None])[0], "frame_mass_g": fea.get("frame_mass_g")}

def agreement(cfg):
    root = Path(cfg["mma"]["root"]).resolve()
    runs, fine = json.loads((root / "runs.json").read_text(encoding="utf-8")), json.loads((root / "fine" / "result.json").read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in fine["rows"]}
    optimizer = {"arm_tip_n_per_mm": rows["arm_tip_stiffness"]["value"], "f1_hz_eroded": rows["f1"]["value"], "f1_hz_intermediate": rows["f1_intermediate"]["value"]}
    evaluation = {}
    for suffix, path in runs.items():
        found = fea_values(Path(path) / "evaluation.json")
        ratio = lambda value, base: None if value is None else value / base - 1
        found["deviation"] = {"arm_tip": ratio(found["arm_tip_n_per_mm"], optimizer["arm_tip_n_per_mm"]), "f1_vs_eroded": ratio(found["f1_hz"], optimizer["f1_hz_eroded"]), "f1_vs_intermediate": ratio(found["f1_hz"], optimizer["f1_hz_intermediate"])}
        found["within"] = {key: value is not None and abs(value) <= cfg["mma"]["agreement"] for key, value in found["deviation"].items()}
        evaluation[suffix] = found
    manafly = root / "manafly" / "evaluation" / "evaluation.json"
    record = {"optimizer": optimizer, "evaluation": evaluation, "tolerance": cfg["mma"]["agreement"], "optimizer_table": fine["table"], "mass_g": fine["mass_g"], "mass_by_field_g": fine["mass_by_field_g"],
              "manafly": {**fea_values(manafly), "line": json.loads(manafly.read_text(encoding="utf-8")).get("line") if manafly.is_file() else None,
                          "shadow_mm": json.loads(Path(cfg["output"]).read_text(encoding="utf-8"))["manafly_shadow_mm"]}}
    (root / "agreement.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(json.dumps(record, indent=1, default=float), flush=True)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).is_file() else {}

def twist(sigma, spec):
    labels = ["/".join(label) for label in sigma["labels"]]
    w = np.zeros(len(labels))
    for name, sign in spec["twist"].items():
        w[labels.index(f"{name}/{spec['twist_dof']}")] = sign
    compliance = float(w @ np.asarray(sigma["flexibility"]) @ w)
    return {"compliance_n_mm": compliance, "stiffness_n_per_mm": float(w @ w / compliance)}

def body_summary(run, sigma_path, spec):
    evaluation = read(Path(run) / "evaluation.json") if run else {}
    sigma = evaluation.get("sigma") or read(sigma_path)
    fea, geometry = evaluation.get("fea") or {}, evaluation.get("geometry") or {}
    stiffness = sigma.get("stiffness") or {}
    return {"run": run, "mass_g": (geometry.get("mass") or {}).get("frame_mass_g"), "arm_tip_n_per_mm": fea.get("stiffness_n_per_mm"), "f1_hz": (fea.get("eigenfrequencies_hz") or [None])[0],
            "sigma_full": [sigma.get("mean_compliance_n_mm"), sigma.get("worst_case_compliance_n_mm")], "sigma_diagonal": [(sigma.get("diagonal") or {}).get("mean_compliance_n_mm"), (sigma.get("diagonal") or {}).get("worst_case_compliance_n_mm")],
            "worst_load": sigma.get("worst_case_load"), "stiffness": {label: stiffness.get(label) for label in stiffness}, "twist": twist(sigma, spec) if sigma.get("flexibility") else None,
            "missed": (evaluation.get("assessment") or {}).get("missed"), "warnings": (evaluation.get("assessment") or {}).get("warnings"), "line": evaluation.get("line"),
            "datasheet": str(Path(run) / "datasheet.md") if run else None, "sigma_peak_memory_gb": sigma.get("peak_memory_gb")}

def stiffness_table(bodies, base="manafly3"):
    names = [name for name in bodies if bodies[name]["stiffness"]]
    labels = list(bodies[base]["stiffness"]) if base in names else []
    others = [name for name in names if name != base]
    lines = ["| interface/dof | " + " | ".join(names) + " | " + " | ".join(f"{name}/{base}" for name in others) + " |", "|---" * (1 + len(names) + len(others)) + "|"]
    for label in labels:
        values = {name: bodies[name]["stiffness"].get(label) for name in names}
        lines.append(f"| {label} | " + " | ".join(f"{values[name]:.4g}" for name in names) + " | " + " | ".join(f"{values[name] / values[base]:.2f}" for name in others) + " |")
    return "\n".join(lines)

def cov_compare(cfg):
    spec, root = cfg["comparison"], Path(cfg["mma"]["root"]).resolve()
    runs, fine, old_fine = read(root / "runs.json"), read(root / "fine" / "result.json"), read(spec["old_fine"])
    bodies = {"old_raw": body_summary(spec["old"]["raw"], "exports/cov/eval/simp_mma_raw_1/sigma.json", spec), "old_recon_1to1": body_summary(spec["old"]["recon"], "exports/cov/eval/simp_mma_recon_1/sigma.json", spec),
              **{f"previous_{suffix}": body_summary(path, None, spec) for suffix, path in spec["previous"].items()}, **{f"new_{suffix}": body_summary(path, None, spec) for suffix, path in runs.items()},
              "manafly3": body_summary(None, spec["manafly_sigma"], spec), "aether4": body_summary(None, spec["aether4_sigma"], spec)}
    stages = {"old": old_fine, "previous": read(spec["previous_fine"]), "new": fine}
    values = lambda stage, name: ({row["name"]: row for row in stage.get("rows", [])}.get(name) or {}).get("value")
    limits = {**COVARIANCE["limits"], "calibration": COVARIANCE["limits"]["calibration"]["free" if free_battery() else "rails"]}
    record = {"optimizer": {**{key: {"mass_g": stage.get("mass_g"), "status": stage.get("status"), "iterations": stage.get("iterations"), "table": stage.get("table"),
                                     "load_worst_info": ({row["name"]: row for row in stage.get("rows", [])}.get("load_worst") or {}).get("info")} for key, stage in stages.items()},
                            "monitored": {name: [values(stage, name) for stage in stages.values()] for name in ("thrust_all", "torsion_yaw", "twist", "f1_intermediate", "arm_tip_stiffness")}},
              "limits": {"evaluator_full": [limits["mean_n_mm"], limits["worst_n_mm"]], "source": limits["source"], "calibration": limits["calibration"],
                         "optimizer": [limits["mean_n_mm"] * limits["calibration"]["mean_n_mm"], limits["worst_n_mm"] * limits["calibration"]["worst_n_mm"]],
                         "evaluator_full_aether4_scaled": [read(spec["aether4_sigma"]).get("scaled", {}).get("mean_compliance_n_mm"), read(spec["aether4_sigma"]).get("scaled", {}).get("worst_case_compliance_n_mm")]},
              "bodies": bodies, "stiffness_table": stiffness_table(bodies)}
    Path(spec["output"]).parent.mkdir(parents=True, exist_ok=True)
    Path(spec["output"]).write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(json.dumps({name: {key: body[key] for key in ("mass_g", "arm_tip_n_per_mm", "f1_hz", "sigma_full", "sigma_diagonal", "twist", "missed")} for name, body in bodies.items()}, indent=1, default=float), flush=True)

def main(argv=None):
    return command_line({"references": lambda overrides: references(configure(overrides)), "modal_split": lambda overrides: modal_split(configure(overrides)), "cantilever": lambda overrides: cantilever(configure(overrides)),
                         "cantilever_mma": lambda overrides: cantilever_mma(configure(overrides)), "frame_mma": lambda overrides: frame_mma(configure(overrides)),
                         "frame_runs": lambda overrides: frame_runs(configure(overrides)), "manafly_check": lambda overrides: manafly_check(configure(overrides)), "compose": lambda overrides: compose(configure(overrides)),
                         "agreement": lambda overrides: agreement(configure(overrides)), "covariance_cantilever": lambda overrides: covariance_cantilever_check(configure(overrides)),
                         "covariance_mma": lambda overrides: covariance_cantilever_mma(configure(overrides)), "covariance_frame": lambda overrides: covariance_frame(configure(overrides)),
                         "cov_compare": lambda overrides: cov_compare(configure(overrides)), "battery_fd": lambda overrides: battery_fd(configure(overrides)),
                         "battery_compare": lambda overrides: battery_compare(configure(overrides)), "camera_limits": lambda overrides: camera_limits(configure(overrides)), "camera_fd": lambda overrides: camera_fd(configure(overrides)),
                         "solver_memory": lambda overrides: solver_memory(configure(overrides)), "setup_memory": lambda overrides: setup_memory(configure(overrides)), "stand_fd": lambda overrides: stand_fd(configure(overrides))}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
