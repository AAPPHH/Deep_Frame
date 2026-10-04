import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deep_frame.config as config
from deep_frame.config import RUN_SETTINGS, STAGES, command_line
from deep_frame.frame_run import ROOT, FrameRun, _git
from deep_frame.topology_geometry import _merge
from deep_frame.topology_neural import cell_centers
from deep_frame.topology_optimization import HexElasticity
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
    "battery_fd": {"shape": [68, 64, 24], "step": 1e-5, "seed": 7, "low": 0.3, "high": 0.9, "output": "docs/validation/battery_support_fd.json"},
    "mma": {"settings": {}, "cantilever_start": 0.5, "dual_volume": 0.3, "fd_step": 1e-5, "fd_seed": 7, "linear_solver": "cuda_cudss", "coarse": True, "fine_start_level": 3,
            "root": "exports/runs/simp_mma_opt", "variant": "simp_mma", "resume": False, "gray": [0.05, 0.95], "method": "simp_mma", "agreement": 0.15,
            "viewer": "C:/clones/Deep_Frame-neural/exports", "manafly_renders": "C:/clones/Deep_Frame-neural/exports/fast/_manafly_same_renderer", "evaluation_python": "C:/clones/Deep_Frame/.venv/Scripts/python.exe",
            "bodies": ["raw", "recon"], "battery_start": {"density": 0.5, "cells": 2}, "viewer_names": {"raw": "{method}_final", "recon": "{method}_final_recon", "v3": "{method}_v3"},
            "figures": [{"output": "{method}_4views.png", "panels": ["raw", "recon", "manafly"], "labels": ["SIMP-MMA raw", "SIMP-MMA recon", "ManaFly"]}]},
    "v3": {"worktree": "C:/clones/Deep_Frame-recon3", "ref": "HEAD", "copy": "C:/Users/jfham/AppData/Local/Temp/claude/c--clones-Deep-Frame/2bec171b-ba58-44fe-ab0f-61ff45688b18/scratchpad/recon3_copy",
           "argv": ["splines"], "compute": "geometry", "compare": "recon"},
    "comparison": {"output": "exports/cov/comparison.json", "old": {"raw": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_raw_1", "recon": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_recon_1"},
                   "old_fine": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_opt/fine/result.json", "previous": {}, "previous_fine": None, "gap": {}, "manafly_sigma": "exports/cov/eval/manafly3/sigma.json", "aether4_sigma": "exports/cov/eval/aether4/sigma.json",
                   "twist": {"motor_front_left": 1.0, "motor_rear_right": 1.0, "motor_front_right": -1.0, "motor_rear_left": -1.0}, "twist_dof": "Fz"},
    "covariance": {"variant": "mean", "limit_factor": 4.0, "start": 0.5, "fd_step": 1e-5, "fd_seed": 11, "ks_fd": 5.0, "settings": {},
                   "frame_density": "C:/clones/Deep_Frame-mma/exports/runs/simp_mma_opt/fine/density_half.npz", "frame_solver": "cuda_cudss"},
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
                      "coverage": {"axes": [axis.tolist(), [1.0, 0.0, 0.0], up.tolist()], "front_mm": front.tolist(), "window_mm": window, "step_mm": step, "length_mm": float(np.linalg.norm(upper - np.asarray(grid["origin_mm"]))), "frontal_area_mm2": float(size[0] * size[2])},
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
    if references:
        problem["crash"].update(reference=references["crash_compliance_n_mm"], reference_source=references["source"])
        problem["shadow"].update(limit_mm=references["manafly_shadow_mm"], source=references["source"])
    return problem

def frame_setup(cfg=FORMULATION, shape=None):
    half = frame_domain(cfg, shape or cfg["shape"])
    return half, frame_problem(half, json.loads(Path(cfg["output"]).read_text(encoding="utf-8")))

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
    density = np.load(cfg["reference_density"])["density"].ravel()
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
    density = np.load(cfg["reference_density"])["density"].ravel()
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
    density = np.load(cfg["reference_density"])["density"].ravel()
    base = frame_domain(cfg, cfg["shape"])
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

def checkpoint(path):
    def save(x, history, levels):
        np.savez_compressed(path, design=x, level=history[-1]["level"], iteration=len(history))
    return save

def optimize_stage(cfg, half, problem, design, out, start_level):
    out.mkdir(parents=True, exist_ok=True)
    tp = TopologyProblem(half, problem, linear_solver=cfg["mma"]["linear_solver"])
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
              "mass_by_field_g": report["mass_by_field_g"], "rows": report["rows"], "table": format_report(report["rows"]), "max_violation": report["max_violation"], "mma": result["settings"]}
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

def battery_start(cfg, design, half):
    from deep_frame.topology_geometry import region_contains
    spec = cfg["mma"]["battery_start"]
    if "battery" not in half or not spec:
        return design
    near = region_contains(cell_centers(half["grid"]), half["battery"]["keep_out"], spec["cells"] * np.asarray(half["grid"]["spacing_mm"])) & np.asarray(half["allowed"]).ravel()
    design = design.copy()
    design[near] = np.maximum(design[near], spec["density"])
    return design

def frame_mma(cfg):
    started = perf_counter()
    root = Path(cfg["mma"]["root"])
    reference = np.load(cfg["reference_density"])["density"].ravel()
    fine, problem = frame_setup(cfg, cfg["shape"])
    design, stages, level = reference, {}, 0
    if cfg["mma"]["coarse"]:
        coarse, coarse_problem = frame_setup(cfg, cfg["coarse_shape"])
        start = battery_start(cfg, prolongate(reference, fine["grid"], coarse), coarse)
        result, stages["coarse"] = optimize_stage(cfg, coarse, coarse_problem, start, root / "coarse", 0)
        design, level = prolongate(result, coarse["grid"], fine), cfg["mma"]["fine_start_level"]
    design, stages["fine"] = optimize_stage(cfg, fine, problem, design, root / "fine", level)
    out = root / cfg["mma"]["variant"]
    out.mkdir(parents=True, exist_ok=True)
    physical = np.load(root / "fine" / "density_half.npz")["density"]
    np.savez_compressed(out / "density_half.npz", density=physical)
    body = export_body(cfg, physical, out)
    info = {"variant": cfg["mma"]["variant"], "method": "SIMP-MMA (mmapy 0.3.1), shared formulation " + git_sha(), "stages": stages, "body": body, "total_runtime_s": perf_counter() - started,
            "iterations": sum(stage["iterations"] for stage in stages.values()), "reference": cfg["reference_density"]}
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
    folders = {**{suffix: Path(path) / "renders" for suffix, path in runs.items()}, "manafly": manafly}
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
    limits = COVARIANCE["limits"]
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
                         "battery_compare": lambda overrides: battery_compare(configure(overrides))}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
