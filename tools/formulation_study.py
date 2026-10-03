import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deep_frame.config as config
from deep_frame.config import command_line
from deep_frame.topology_geometry import _merge
from deep_frame.topology_optimization import HexElasticity
from deep_frame.topology_problem import PROBLEM, TopologyProblem, cantilever_dual, format_report, orthotropic_material, shadow_thickness

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

def patched_builder(cfg, stiffness=True):
    from run import _update
    request = json.loads(Path(cfg["request"]).read_text(encoding="utf-8"))
    for name, values in request["patch"].items():
        _update(getattr(config, name), values)
    from tools.neural_study import R2Domain, configure as study
    overrides = deepcopy(request["overrides"])
    if stiffness:
        overrides["stiffness"] = {"min_n_per_mm": PROBLEM["stiffness"]["min_n_per_mm"]}
    return R2Domain(study(overrides))

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
    return half

def frame_problem(half, references=None):
    problem = deepcopy(PROBLEM)
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

def main(argv=None):
    return command_line({"references": lambda overrides: references(configure(overrides)), "modal_split": lambda overrides: modal_split(configure(overrides)), "cantilever": lambda overrides: cantilever(configure(overrides))}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
