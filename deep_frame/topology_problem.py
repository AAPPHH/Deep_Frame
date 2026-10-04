from copy import deepcopy
from time import perf_counter
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from deep_frame.config import COMPONENT_LIBRARY, CRASH_DIRECTIONS, DEFAULT_SELECTION, INTEGRATION_CONFIG, LOAD_COVARIANCE_LIMITS, PRINT_MATERIAL
from deep_frame.topology_neural import cell_centers
from deep_frame.topology_optimization import SOLVER_CHOICE, DensityMap, HexElasticity, StiffnessConstraint, _settings

ARM_TIP = {"name": "arm_tip", "case": "stiffness_arm_tip", "min_n_per_mm": 10.0, "calibration": 1.0, "penalty": 1.0, "multiplier_interval": 1,
           "definition": "evaluator arm_tip: centre mount undersides fixed, uniform pad load on the front-left motor seat, k = |F| / mean pad displacement along F = |F|^2 / compliance"}
COVARIANCE = {
    "support": "stiffness_arm_tip", "interfaces": ["motor_front_left", "motor_front_right", "motor_rear_left", "motor_rear_right", "battery", "camera"],
    "sigma": None, "labels": None, "model": None, "prefix": "sigma_", "ks": 50.0, "ks_cutoff": 1e-9,
    "limits": {"mean_n_mm": LOAD_COVARIANCE_LIMITS["mean_compliance_n_mm"], "worst_n_mm": LOAD_COVARIANCE_LIMITS["worst_case_compliance_n_mm"], "source": LOAD_COVARIANCE_LIMITS["source"],
               "calibration": {"mean_n_mm": 1.384, "worst_n_mm": 1.374},
               "calibration_source": "stable after iteration 3 (run simp_mma_cov3 with this factor: 18.2 g field / 15.7 g raw body, optimizer 0.94671 / 0.30808 N mm / evaluator full 0.68695 / 0.21967 N mm = 1.378 / 1.402, -0.4 % / +2.1 %, below 10 %, factor kept); "
                                     "iteration 2 on the 18.5 g load-model field / 15.9 g raw body without accessory seats (run simp_mma_cov2 with factor 1.247 / 1.235): optimizer 0.85631 / 0.25736 N mm / evaluator full 0.61892 / 0.18733 N mm = 1.384 / 1.374 (+11.0 % / +11.2 % against iteration 1); "
                                     "iteration 1 on the 26.6 g load-model frame (baff3e1): optimizer measure of its fine eroded field (exports/runs/simp_mma_cov_opt/fine/result.json: tr 0.23219, lambda_max 0.058508 N mm) / evaluator full-coupled measure of its raw body (simp_mma_cov_raw_1: 0.18621 / 0.047386 N mm); "
                                     "iteration 0 was optimizer / evaluator diagonal on the 17.2 g SIMP-MMA design (0.5952 / 0.3928, as full 0.735 / 0.685) and did not transfer; per key because the worst-case modes differ"},
    "definition": {
        "limits": "read from config LOAD_COVARIANCE_LIMITS (evaluator on the references); the optimizer uses limit x calibration per key (optimizer measure / evaluator measure on the same design)",
        "support": "stack mount undersides fixed in all translations (fixed regions of stiffness_arm_tip = evaluator arm_tip / gap finder group 'stack fixed'); the stack wrench is reacted by the fixture, so its 6 rows are dropped from Sigma (36 x 36 block of motors, battery, camera)",
        "interfaces": "motor pads: thrust_all pad patches (pad top); battery: deck band of crash_back (deck top); camera: camera patch of crash_front; reference points as LOAD_COVARIANCE reference_points (pad top on the motor axis, deck top at the band centre on x = 0, midpoint of the camera side-screw axes)",
        "wrench": "unit wrench (F, M about the reference point) -> minimum-norm nodal forces over the patch nodes: f = B^T (B B^T)^-1 w with B_i = [I; skew(r_i - r_ref)]",
        "flexibility": "F = P^T K^-1 P over all 36 interface DOF including the cross-interface blocks (Sigma correlates interfaces, so the off-diagonal blocks enter); work-conjugate generalised displacements",
        "mean": "tr(F Sigma) = sum_k d_k^T F d_k over the principal directions d_k (Sigma = D D^T) = expected compliance E[f^T u], N mm",
        "worst": "lambda_max(Sigma^1/2 F Sigma^1/2) = lambda_max(D^T F D), N mm; the constraint uses KS_s over all eigenvalues of D^T F D / limit (s = ks; upper bound, overestimate <= ln(rank)/s of the limit)",
        "sensitivity": "self-adjoint: d tr / d rho_e = -sum_k u_k^T dK_e u_k; d lambda_i = -(U v_i)^T dK_e (U v_i); KS weights softmax(s lambda_i / limit); the KS of the matrix stays smooth for repeated eigenvalues"},
}
PROBLEM = {
    "objective": "mass",
    "volume_max": 0.10,
    "stiffness": None,
    "covariance": COVARIANCE,
    "crash": {"cases": ["crash_" + name for name in CRASH_DIRECTIONS], "ratio": 1.5, "reference": {}, "reference_source": None},
    "modal": {"f1_min_hz": 300.0, "case": "modes", "modes": 6, "tracked": 3, "ks": 40.0, "mass_cutoff": 0.1, "initial_iterations": 30, "warm_iterations": 3, "penalty": 1.0, "multiplier_interval": 1,
              "point_masses": ["battery", "aio15", "camera"]},
    "shadow": {"limit_mm": None, "exponent": 2.0, "hub_radius_mm": 7.88, "motors_mm": [], "radius_mm": None, "source": None,
               "definition": "w = ((r - r_hub) / (R - r_hub))^n inside the prop cylinder (full height), 0 inside the hub; t_w = sum(w rho V) / sum_discs(int w dA) = radially weighted equivalent material thickness under the props in mm"},
    "monitor": ["thrust_all", "torsion_yaw", "twist"],
    "width": {"minimum_mm": 2.5, "eta": 0.5, "delta": 0.25, "minimum_cells": 1.5},
    "interpolation": {"penalization": 3.0, "min_stiffness_ratio": 1e-6, "stiffness": "SIMP p=3, E_min = 1e-6 E", "mass": "linear, rho^6/c^5 below c = 0.1 (modal only, against spurious low-density modes)"},
    "fields": {"stiffness": "eroded", "mass": "intermediate", "volume": "intermediate", "shadow": "intermediate"},
    "continuation": {"beta_schedule": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]},
    "termination": {"mass_change": 1e-3, "violation": 1e-3, "window": 5, "active": 0.01},
    "material": "orthotropic",
    "solver_choice": SOLVER_CHOICE,
}

def length_scale_ratio(eta_eroded, samples=2001):
    kernel = lambda x: np.clip(1 - np.abs(x), 0, None) ** 2 * (1 + 2 * np.abs(x))
    unit = np.linspace(-1, 1, samples)
    norm = np.trapezoid(kernel(unit), unit)
    x = np.linspace(0, 1.5, 1501)
    def filtered(width, at):
        t = np.linspace(-width / 2, width / 2, samples)
        return np.trapezoid(kernel(np.asarray(at)[:, None] - t[None, :]), t, axis=1) / norm
    low, high = 0.0, 2.0
    for _ in range(50):
        middle = (low + high) / 2
        low, high = (middle, high) if filtered(middle, [0.0])[0] < eta_eroded else (low, middle)
    return float(2 * x[filtered(high, x) >= 0.5].max())

def filter_radius(width, spacing):
    return max(width["minimum_mm"] / length_scale_ratio(width["eta"] + width["delta"]), width["minimum_cells"] * float(np.max(spacing)))

def radial_weight(points, shadow):
    weights = np.zeros(len(points))
    span = shadow["radius_mm"] - shadow["hub_radius_mm"]
    for x, y in shadow["motors_mm"]:
        radial = np.hypot(points[:, 0] - x, points[:, 1] - y)
        inside = (radial > shadow["hub_radius_mm"]) & (radial <= shadow["radius_mm"])
        weights[inside] = np.maximum(weights[inside], ((radial[inside] - shadow["hub_radius_mm"]) / span) ** shadow["exponent"])
    return weights

def weighted_disc_area(shadow):
    span, hub, n = shadow["radius_mm"] - shadow["hub_radius_mm"], shadow["hub_radius_mm"], shadow["exponent"]
    return len(shadow["motors_mm"]) * 2 * np.pi * (span ** 2 / (n + 2) + hub * span / (n + 1))

def shadow_thickness(solid, lower, h, shadow):
    solid = np.asarray(solid, dtype=float)
    centers = np.asarray(lower)[:2] + (np.indices(solid.shape[:2]).reshape(2, -1).T + 0.5) * h
    column = solid.sum(axis=2).ravel() * h ** 3
    return float(np.dot(radial_weight(centers, shadow), column) / weighted_disc_area(shadow))

def constraint_report(rows, termination=PROBLEM["termination"]):
    report = []
    for row in rows:
        g = row["g"]
        status = "monitored" if g is None else "violated" if g > termination["violation"] else "active" if g >= -termination["active"] else "satisfied"
        report.append({**{key: row[key] for key in ("name", "value", "limit", "unit", "sense", "field")}, **({"info": row["info"]} if "info" in row else {}), "g": g, "status": status, "margin": None if g is None else -g})
    return report

def format_report(report):
    lines = ["| constraint | value | limit | unit | status | margin |", "|---|---|---|---|---|---|"]
    for row in report:
        limit = "" if row["limit"] is None else f"{row['sense']} {row['limit']:.4g}"
        margin = "" if row["margin"] is None else f"{100 * row['margin']:+.1f} %"
        lines.append(f"| {row['name']} | {row['value']:.4g} | {limit} | {row['unit']} | {row['status']} | {margin} |")
    return "\n".join(lines)

class Termination:
    def __init__(self, settings=PROBLEM["termination"]):
        self.settings, self.masses, self.violations = settings, [], []
    def __call__(self, mass, violation, final_level=True):
        self.masses.append(float(mass))
        self.violations.append(float(violation))
        window = self.masses[-self.settings["window"]:]
        if len(window) < self.settings["window"] or not final_level:
            return False
        return (max(window) - min(window)) / max(min(window), 1e-30) < self.settings["mass_change"] and self.violations[-1] <= self.settings["violation"]

def prolongate(design, coarse_grid, fine_domain):
    shape = tuple(coarse_grid["shape"])
    axes = [coarse_grid["origin_mm"][i] + (np.arange(shape[i]) + 0.5) * coarse_grid["spacing_mm"][i] for i in range(3)]
    interpolator = RegularGridInterpolator(axes, np.asarray(design, dtype=float).reshape(shape), bounds_error=False, fill_value=None)
    fine = np.clip(interpolator(cell_centers(fine_domain["grid"])), 0, 1)
    fine[np.asarray(fine_domain["preserve"]).ravel()] = 1
    fine[~np.asarray(fine_domain["allowed"]).ravel()] = 0
    return fine

class TopologyProblem:
    def __init__(self, domain, problem=PROBLEM, linear_solver="cpu_superlu"):
        self.problem = problem = deepcopy(problem)
        self.domain = domain
        interpolation = problem["interpolation"]
        self.penalization, self.min_stiffness_ratio = interpolation["penalization"], interpolation["min_stiffness_ratio"]
        width = problem["width"]
        self.radius = filter_radius(width, domain["grid"]["spacing_mm"])
        self.map = DensityMap(domain, _settings({"projection": "robust", "projection_eta": width["eta"], "robust_delta": width["delta"], "filter_radius_mm": self.radius,
                                                 "beta_schedule": problem["continuation"]["beta_schedule"]}))
        self.level = 0
        self.system = HexElasticity(domain, interface_node_policy=domain.get("optimizer_settings", {}).get("interface_node_policy", "allowed_adjacent"), linear_solver=linear_solver, share_static=problem.get("share_static", True), multigrid=problem.get("multigrid"), solver_choice=problem.get("solver_choice"))
        self.factor = 1.0 if self.system.symmetry is None else 2.0
        self.cell = float(np.prod(self.system.spacing))
        self.allowed = int(np.count_nonzero(self.map.allowed))
        self.stiffness = StiffnessConstraint(self.system, problem["stiffness"]) if problem.get("stiffness") else None
        self.covariance = None
        if problem.get("covariance"):
            if "interfaces" not in domain:
                raise ValueError("The load covariance constraint needs domain interfaces")
            self.covariance = InterfaceCovariance(self.system, problem["covariance"], domain["interfaces"])
        self.modal = self.system.modal_constraint(problem["modal"]) if problem.get("modal") else None
        shadow = problem.get("shadow")
        self.shadow = None
        if shadow and shadow["motors_mm"]:
            self.shadow = radial_weight(cell_centers(domain["grid"]), shadow) * np.asarray(domain["allowed"]).ravel() * self.cell * self.factor / weighted_disc_area(shadow)
        names = [case["name"] for case in self.system.cases]
        self.crash = [name for name in (problem.get("crash") or {}).get("cases", []) if name in names]
        self.monitor = [name for name in problem.get("monitor", []) if name in names]
        missing = [name for name in (problem.get("crash") or {}).get("cases", []) if name not in names]
        if missing:
            raise ValueError("Crash cases missing from the domain: " + ", ".join(missing))
    @property
    def beta(self):
        return self.map.beta
    @property
    def final_level(self):
        return self.level == len(self.problem["continuation"]["beta_schedule"]) - 1
    def advance(self):
        if self.final_level:
            return False
        self.level += 1
        self.map.beta = self.problem["continuation"]["beta_schedule"][self.level]
        return True
    def mass_g(self, physical):
        return float(np.sum(physical) * self.cell * self.factor * self.domain["material"]["density_g_cm3"] / 1000)
    def compliances(self, physical):
        return self.system.solve(np.asarray(physical, dtype=float).ravel(), self.penalization, self.min_stiffness_ratio)
    def physics(self, physical, iterations=None):
        physical = np.asarray(physical, dtype=float).ravel()
        solutions = self.compliances(physical)
        rows, monitor = [], []
        if self.stiffness is not None:
            g, slope, info = self.stiffness.measure(solutions[self.stiffness.case])
            rows.append({"name": self.problem["stiffness"]["name"] + "_stiffness", "g": float(g), "gradient": slope, "value": info["stiffness_n_per_mm"], "limit": info["target_n_per_mm"], "unit": "N/mm", "sense": ">="})
        if self.covariance is not None:
            for row in self.covariance.measure(solutions):
                (rows if row["g"] is not None else monitor).append(row)
        for name in self.crash:
            limit = self.problem["crash"]["ratio"] * self.problem["crash"]["reference"][name]
            rows.append({"name": name, "g": solutions[name]["compliance_n_mm"] / limit - 1, "gradient": solutions[name]["derivative"] / limit, "value": solutions[name]["compliance_n_mm"], "limit": limit, "unit": "N mm", "sense": "<="})
        if self.modal is not None:
            g, slope, info = self.modal.measure(physical, self.penalization, self.min_stiffness_ratio, iterations)
            rows.append({"name": "f1", "g": float(g), "gradient": slope, "value": info["f1_hz"], "limit": self.problem["modal"]["f1_min_hz"], "unit": "Hz", "sense": ">=", "info": info})
        monitor += [{"name": name, "g": None, "gradient": None, "value": solutions[name]["compliance_n_mm"], "limit": None, "unit": "N mm", "sense": "", "field": self.problem["fields"]["stiffness"]} for name in self.monitor]
        for row in rows + monitor:
            row["field"] = self.problem["fields"]["stiffness"]
        return rows, monitor, solutions
    def geometry(self, physical):
        physical = np.asarray(physical, dtype=float).ravel()
        rows = [{"name": "volume", "g": float(np.sum(physical[self.map.allowed]) / (self.problem["volume_max"] * self.allowed) - 1), "gradient": self.map.allowed / (self.problem["volume_max"] * self.allowed),
                 "value": float(np.sum(physical[self.map.allowed]) / self.allowed), "limit": self.problem["volume_max"], "unit": "-", "sense": "<="}]
        if self.shadow is not None:
            limit = self.problem["shadow"]["limit_mm"]
            value = float(np.dot(self.shadow, physical))
            rows.append({"name": "shadow", "g": value / limit - 1, "gradient": self.shadow / limit, "value": value, "limit": limit, "unit": "mm", "sense": "<="})
        for row in rows:
            row["field"] = self.problem["fields"]["volume"]
        return rows
    def evaluate(self, design, iterations=None):
        fields, _ = self.map.fields(design)
        eroded, intermediate = fields[self.problem["fields"]["stiffness"]], fields[self.problem["fields"]["mass"]]
        rows, monitor, _ = self.physics(eroded[0], iterations)
        geometric = self.geometry(intermediate[0])
        mass = self.mass_g(intermediate[0])
        scale = self.mass_g(np.full(self.map.n, self.problem["volume_max"]) * self.map.allowed)
        gradients = [self.map.pullback(row["gradient"], eroded[1]) for row in rows] + [self.map.pullback(row["gradient"], intermediate[1]) for row in geometric]
        rows += geometric
        return {"objective": mass / scale, "objective_gradient": self.map.pullback(np.full(self.map.n, self.mass_g(np.ones(1)) / scale), intermediate[1]),
                "mass_g": mass, "mass_scale_g": scale, "constraints": np.array([row["g"] for row in rows]), "constraint_gradients": np.array(gradients), "names": [row["name"] for row in rows],
                "rows": constraint_report(rows + monitor, self.problem["termination"]), "beta": self.beta, "max_violation": float(max(row["g"] for row in rows))}
    def report(self, design):
        result = self.evaluate(design, self.problem["modal"]["initial_iterations"] if self.modal is not None else None)
        fields, _ = self.map.fields(design)
        if self.modal is not None:
            info = self.modal.measure(fields["intermediate"][0], self.penalization, self.min_stiffness_ratio, self.problem["modal"]["initial_iterations"])[2]
            result["rows"].append({"name": "f1_intermediate", "value": info["f1_hz"], "limit": None, "unit": "Hz", "sense": "", "field": "intermediate", "g": None, "status": "monitored", "margin": None})
        result["mass_by_field_g"] = {name: self.mass_g(value[0]) for name, value in fields.items()}
        return result
    def physical_report(self, physical):
        rows, monitor, _ = self.physics(physical, self.problem["modal"]["initial_iterations"] if self.modal is not None else None)
        return {"mass_g": self.mass_g(physical), "rows": constraint_report(rows + monitor + self.geometry(physical), self.problem["termination"])}
    def close(self):
        return self.system.close()

def orthotropic_material(material):
    return {**material, "name": PRINT_MATERIAL["name"], "orthotropic": dict(PRINT_MATERIAL["orthotropic"]), "young_modulus_mpa": PRINT_MATERIAL["orthotropic"]["e_xy_mpa"],
            "source": PRINT_MATERIAL["source"], "assumptions": PRINT_MATERIAL["assumptions"]}

def cantilever_domain(shape=(32, 6, 12), spacing=1.0, force_n=1.0, material=None):
    shape = tuple(shape)
    length, half, height = (np.asarray(shape) * spacing).tolist()
    allowed = np.ones(shape, dtype=bool)
    tolerance = 1e-6
    return {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [spacing] * 3, "shape": list(shape), "axis_order": "xyz", "order": "C"},
            "allowed": allowed, "preserve": np.zeros(shape, dtype=bool), "forbidden": ~allowed, "symmetry": {"axis": 1, "plane_mm": 0.0},
            "material": material or orthotropic_material({"density_g_cm3": PRINT_MATERIAL["density_g_cm3"], "poisson_ratio": PRINT_MATERIAL["poisson_ratio"]}),
            "load_cases": [{"name": "tip", "analysis": "static", "fixed_regions": [{"kind": "box", "min_mm": [-tolerance, -tolerance, -tolerance], "max_mm": [tolerance, half + tolerance, height + tolerance]}],
                            "loads": [{"region": {"kind": "box", "min_mm": [length - tolerance, -tolerance, -tolerance], "max_mm": [length + tolerance, spacing + tolerance, spacing + tolerance]}, "force_n": [0.0, 0.0, -force_n]}]}],
            "point_masses": [], "optimizer_settings": {"interface_node_policy": "allowed_adjacent"},
            "metadata": {"fixture": "3D cantilever L:W:H = %g:%g:%g mm, half model about y = 0, x = 0 face clamped, tip load on the bottom edge at mid width" % (length, 2 * half, height)}}

def cantilever_problem(min_n_per_mm, volume_max=1.0):
    return {**deepcopy(PROBLEM), "volume_max": volume_max, "crash": None, "modal": None, "shadow": None, "monitor": [], "covariance": None,
            "stiffness": {**ARM_TIP, "name": "tip", "case": "tip", "min_n_per_mm": min_n_per_mm}}

CANTILEVER_COVARIANCE = {"labels": [["tip", "Fz"], ["mid", "Fz"], ["tip", "Fy"]], "std": [1.0, 1.0, 0.5], "correlation": 0.7,
                         "statement": "two correlated vertical loads (tip and mid-span top, rho = 0.7) plus an independent lateral tip load (antisymmetric about the y = 0 mirror plane, exercises the antisymmetric half solve)"}

def covariance_cantilever(shape=(32, 6, 12), spacing=1.0, limits=None, full=False, settings=CANTILEVER_COVARIANCE):
    shape = list(shape)
    length, half, height = (np.asarray(shape) * spacing).tolist()
    tolerance = 1e-6
    if full:
        domain = cantilever_domain((shape[0], 2 * shape[1], shape[2]), spacing)
        domain.update(symmetry=None)
        domain["grid"]["origin_mm"][1] = -half
        domain["load_cases"][0]["fixed_regions"][0]["min_mm"][1] = -half - tolerance
    else:
        domain = cantilever_domain(shape, spacing)
    low = -half - tolerance
    domain["interfaces"] = {"tip": {"regions": [{"kind": "box", "min_mm": [length - tolerance, low, -tolerance], "max_mm": [length + tolerance, half + tolerance, height + tolerance]}], "reference_mm": [length, 0.0, height / 2]},
                            "mid": {"regions": [{"kind": "box", "min_mm": [length / 2 - spacing - tolerance, low, height - tolerance], "max_mm": [length / 2 + spacing + tolerance, half + tolerance, height + tolerance]}], "reference_mm": [length / 2, 0.0, height]}}
    std = np.asarray(settings["std"], dtype=float)
    correlation = np.eye(len(std))
    correlation[0, 1] = correlation[1, 0] = settings["correlation"]
    problem = {**cantilever_problem(1.0), "stiffness": None,
               "covariance": {**deepcopy(COVARIANCE), "support": "tip", "sigma": (std[:, None] * correlation * std[None, :]).tolist(), "labels": settings["labels"],
                              "limits": {"mean_n_mm": None, "worst_n_mm": None, "source": "cantilever proof", **(limits or {})}}}
    return domain, problem

def cantilever_dual(volume_fraction=0.3, domain=None, problem=None):
    from deep_frame.topology_optimization import optimize_topology
    domain = domain or cantilever_domain()
    width = (problem or PROBLEM)["width"]
    settings = {"volume_fraction": volume_fraction, "filter_radius_mm": filter_radius(width, domain["grid"]["spacing_mm"]), "projection": "robust", "projection_eta": width["eta"], "robust_delta": width["delta"],
                "beta_schedule": (problem or PROBLEM)["continuation"]["beta_schedule"], "beta_interval": 40, "max_iterations": 400, "move_limit": 0.1, "volume_target_relaxation": 1.0}
    result = optimize_topology(domain, settings)
    check = TopologyProblem(domain, cantilever_problem(1.0))
    check.map.beta = settings["beta_schedule"][-1]
    rows = {row["name"]: row for row in check.evaluate(result["design_density"].ravel())["rows"]}
    check.close()
    return {"status": result["status"], "converged": result["summary"].get("converged"), "volume_fraction_intermediate": rows["volume"]["value"], "stiffness_n_per_mm": rows["tip_stiffness"]["value"],
            "statement": "min-compliance at volume V* (OC, same robust filter) gives stiffness k*; min-mass subject to k >= k* must return volume ~ V* with the stiffness constraint active"}

MMA = {"package": "mmapy==0.3.1 (Deetman, Python port of Svanberg's MMA)", "move": 0.1, "scale": 100.0, "asyinit": 0.5, "asydecr": 0.7, "asyincr": 1.2, "raa0": 1e-5, "c": 1e4, "d": 1.0,
       "start_level": 0, "level_window": 5, "level_mass_change": 1e-2, "level_violation": 1e-2, "level_min_iterations": 10, "level_max_iterations": 80, "final_max_iterations": 300,
       "stall_window": 40, "stall_improvement": 1e-3, "checkpoint_interval": 10, "objective": None}

class MMAOptimizer:
    def __init__(self, problem, settings=MMA):
        from mmapy import mmasub
        self.problem, self.settings, self.subproblem = problem, {**MMA, **settings}, mmasub
        self.free = problem.map.free
    def start(self, design):
        x = np.clip(np.asarray(design, dtype=float).ravel(), 0, 1)
        x[self.problem.map.preserve] = 1
        x[~self.problem.map.allowed] = 0
        return x
    def terms(self, result):
        name = self.settings.get("objective")
        if not name:
            return result["objective"], result["objective_gradient"], result["constraints"], result["constraint_gradients"]
        index = result["names"].index(name)
        keep = np.arange(len(result["names"])) != index
        return result["constraints"][index] + 1, result["constraint_gradients"][index], result["constraints"][keep], result["constraint_gradients"][keep]
    def step(self, x, result, state):
        settings, free = self.settings, self.free
        f0, df0, g, dg = self.terms(result)
        n, m = int(np.count_nonzero(free)), len(g)
        value = x[free][:, None]
        zeros, ones = np.zeros((n, 1)), np.ones((n, 1))
        moved = self.subproblem(m, n, state["iteration"], value, zeros, ones, state["old1"], state["old2"], settings["scale"] * f0, settings["scale"] * df0[free][:, None],
                                settings["scale"] * g[:, None], settings["scale"] * dg[:, free], state["low"], state["upp"], 1.0, np.zeros((m, 1)), np.full((m, 1), settings["c"]), np.full((m, 1), settings["d"]),
                                move=settings["move"], asyinit=settings["asyinit"], asydecr=settings["asydecr"], asyincr=settings["asyincr"], raa0=settings["raa0"])
        state.update(old2=state["old1"], old1=value.copy(), low=moved[9], upp=moved[10], iteration=state["iteration"] + 1)
        x = x.copy()
        x[free] = np.clip(moved[0].ravel(), 0, 1)
        return x
    def fresh(self, x):
        value = x[self.free][:, None]
        return {"iteration": 1, "old1": value.copy(), "old2": value.copy(), "low": np.zeros_like(value), "upp": np.ones_like(value)}
    def stalled(self, history):
        window = self.settings["stall_window"]
        if len(history) < window:
            return False
        masses, violations = [row["f0"] for row in history[-window:]], [row["max_violation"] for row in history[-window:]]
        return (max(masses) - min(masses)) / max(min(masses), 1e-30) < self.settings["level_mass_change"] and violations[0] - min(violations) < self.settings["stall_improvement"]
    def advance_ready(self, level):
        settings = self.settings
        if len(level) >= settings["level_max_iterations"]:
            return "iteration_cap"
        if len(level) < max(settings["level_min_iterations"], settings["level_window"]):
            return None
        masses = [row["f0"] for row in level[-settings["level_window"]:]]
        if (max(masses) - min(masses)) / max(min(masses), 1e-30) < settings["level_mass_change"] and level[-1]["max_violation"] <= settings["level_violation"]:
            return "converged"
        return "stalled" if self.stalled(level) else None
    def run(self, design, progress=None, checkpoint=None):
        problem, settings = self.problem, self.settings
        while problem.level < settings["start_level"] and problem.advance():
            pass
        x, termination, history, levels, level = self.start(design), Termination(problem.problem["termination"]), [], [], []
        state, initial = self.fresh(x), problem.problem["modal"]["initial_iterations"] if problem.modal is not None else None
        started = perf_counter()
        while True:
            clock = perf_counter()
            result = problem.evaluate(x, initial if not level else None)
            f0, _, g, _ = self.terms(result)
            row = {"iteration": len(history), "level": problem.level, "beta": problem.beta, "objective": result["objective"], "f0": float(f0), "mass_g": result["mass_g"], "max_violation": float(np.max(g)),
                   "constraints": dict(zip(result["names"], result["constraints"].tolist()))}
            history.append(row)
            level.append(row)
            converged = termination(row["f0"], row["max_violation"], problem.final_level)
            reason = "converged" if converged else None
            if problem.final_level and not converged:
                reason = "iteration_cap" if len(level) >= settings["final_max_iterations"] else "stalled" if self.stalled(level) else None
            elif not problem.final_level:
                ready = self.advance_ready(level)
                if ready:
                    levels.append({"beta": problem.beta, "iterations": len(level), "reason": ready, "mass_g": row["mass_g"], "max_violation": row["max_violation"]})
                    problem.advance()
                    level, state = [], self.fresh(x)
            if reason:
                levels.append({"beta": problem.beta, "iterations": len(level), "reason": reason, "mass_g": row["mass_g"], "max_violation": row["max_violation"]})
                row["seconds"] = perf_counter() - clock
                break
            if level:
                x = self.step(x, result, state)
            row["seconds"] = perf_counter() - clock
            if progress:
                progress(row)
            if checkpoint and len(history) % settings["checkpoint_interval"] == 0:
                checkpoint(x, history, levels)
        if progress:
            progress(row)
        runtime = perf_counter() - started
        status = "converged" if reason == "converged" else "not_converged_" + reason
        return {"design": x, "status": status, "iterations": len(history), "runtime_s": runtime, "seconds_per_iteration": runtime / len(history), "history": history, "levels": levels, "result": result,
                "settings": settings}

_PARTS = {key: COMPONENT_LIBRARY[DEFAULT_SELECTION[key]] for key in ("motor", "aio15", "camera", "battery")}
_PROP = COMPONENT_LIBRARY["HQProp T2.5X2X3V2S"]
_ROTOR_RAD_S = 2 * np.pi * 8000 * 7.4 * 0.75 / 60
LOAD_COVARIANCE = {
    "interfaces": ["motor_front_left", "motor_front_right", "motor_rear_left", "motor_rear_right", "stack", "battery", "camera"],
    "dofs": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
    "units": {"force": "N", "moment": "N mm", "sigma": "second moment E[f f^T] = C + mu mu^T (N^2, N^2 mm, N^2 mm^2)", "frame": "x right, y forward, z up; loads act on the frame, moments about the interface reference point"},
    "reference_points": {"motor": "pad top centre on the motor axis", "stack": "AIO grommet seat plane on the stack axis", "battery": "deck top (rail plane) under the battery centre", "camera": "midpoint of the two side screw axes"},
    "all_up_mass_g": INTEGRATION_CONFIG["all_up_mass_g"],
    "g_m_s2": INTEGRATION_CONFIG["standard_gravity_m_s2"],
    "rotation": {"motor_front_left": 1.0, "motor_front_right": -1.0, "motor_rear_left": -1.0, "motor_rear_right": 1.0},
    "points": {"motor": [[_PARTS["motor"]["mass_g"], _PARTS["motor"]["dimensions_mm"]["height"] / 2], [_PROP["mass_g"], _PARTS["motor"]["dimensions_mm"]["height"] + _PROP["dimensions_mm"]["hub_height"] / 2]],
               "stack": [[_PARTS["aio15"]["mass_g"], _PARTS["aio15"]["dimensions_mm"]["grommet_height"] + _PARTS["aio15"]["dimensions_mm"]["stack_height"] / 2]],
               "battery": [[_PARTS["battery"]["mass_g"], _PARTS["battery"]["dimensions_mm"]["height"] / 2]], "camera": [[_PARTS["camera"]["mass_g"], 0.0]]},
    "prop_height_mm": _PARTS["motor"]["dimensions_mm"]["height"] + _PROP["dimensions_mm"]["hub_height"] / 2,
    "torque_per_thrust_mm": 52.5 * 0.75 / _ROTOR_RAD_S * 1000 / _PROP["data"]["thrust_n_estimate"],
    "rotor_momentum_n_mm_s": 5.9e-7 * _ROTOR_RAD_S * np.sqrt(0.5) * 1000,
    "factors": [
        {"name": "collective", "kind": "thrust", "pattern": [1, 1, 1, 1], "mean": 1.01, "std": 0.583, "unit": "N je Motor",
         "source": "Schub je Motor gleichverteilt auf [0; 2,02 N] (HQProp-Schaetzung 2,02 N): Mittel 1,01 N, Streuung 2,02/sqrt(12); Fz am Pad, Reaktionsmoment Mz, Traegheit aller Massen mit a_z = 4 T / m_ges"},
        {"name": "diagonal_fl_rr", "kind": "thrust", "pattern": [1, 0, 0, -1], "std": 0.35, "unit": "N",
         "source": "ANNAHME: Steuerung um die Diagonalachse FR-RL (Roll+Nick kombiniert): FL +, RR - (Twist); 0,35 N ~ 17 % Maximalschub, 3 sigma ~ halber Schub"},
        {"name": "diagonal_fr_rl", "kind": "thrust", "pattern": [0, 1, -1, 0], "std": 0.35, "unit": "N", "source": "ANNAHME: wie diagonal_fl_rr fuer die andere Diagonale: FR +, RL -"},
        {"name": "yaw_saddle", "kind": "thrust", "pattern": [1, -1, -1, 1], "std": 0.35, "unit": "N", "source": "ANNAHME: Gier-Kommando, gleichsinnige Diagonalpaare gemeinsam +/- (Sattel); koppelt ueber k_Q in alle vier Mz"},
        {"name": "thrust_scatter", "kind": "motor", "axis": "Fz", "at": "prop", "std": 0.10, "unit": "N", "source": "ANNAHME: unabhaengige Streuung je Motor (Unwucht, Turbulenz, ESC), 5 % Maximalschub"},
        {"name": "h_force_x", "kind": "motor", "axis": "Fx", "at": "prop", "std": 0.10, "unit": "N", "source": "ANNAHME: Rotor-H-Kraft/Blattschlag ~5 % Maximalschub in der Propebene, je Motor unabhaengig"},
        {"name": "h_force_y", "kind": "motor", "axis": "Fy", "at": "prop", "std": 0.10, "unit": "N", "source": "ANNAHME: wie h_force_x in y"},
        {"name": "torque_transient", "kind": "motor", "axis": "Mz", "at": "pad", "std": 4.0, "unit": "N mm", "source": "ANNAHME: Hochlauf-Reaktionsmoment je Motor; Grenze kt I = 1,19 N mm/A x ~10 A ~ 12 N mm = 3 sigma"},
        {"name": "body_rate_p", "kind": "gyro", "axis": 0, "std": 6.0, "unit": "rad/s", "source": "ANNAHME: Rollrate RMS 6 rad/s (Spitzen ~17 rad/s = 1000 deg/s); Kreiselmoment H x Omega am Pad"},
        {"name": "body_rate_q", "kind": "gyro", "axis": 1, "std": 6.0, "unit": "rad/s", "source": "ANNAHME: wie body_rate_p fuer die Nickrate"},
        {"name": "body_accel_x", "kind": "body", "axis": 0, "std": 1.0, "unit": "g", "source": "ANNAHME: seitliche spezifische Kraft im Koerpersystem (Luftwiderstand, Boeen) 1 g RMS, gemeinsam fuer alle Massen"},
        {"name": "body_accel_y", "kind": "body", "axis": 1, "std": 1.0, "unit": "g", "source": "ANNAHME: wie body_accel_x in y"},
        *[{"name": "battery_" + "xyz"[axis], "kind": "component", "interface": "battery", "axis": axis, "std": 7.5, "unit": "g",
           "source": "Nutzervorgabe Akkumasse x 5-10 g in alle Richtungen: Mitte 7,5 g als RMS je Achse, 37 g -> 2,72 N am Akkuschwerpunkt, unabhaengig je Achse; z zusaetzlich zum Schubanteil aus collective"} for axis in range(3)],
        *[{"name": part + "_" + "xyz"[axis], "kind": "component", "interface": part, "axis": axis, "std": 2.0, "unit": "g",
           "source": "Nutzervorgabe Masse x Flugbeschleunigung: lokale Manoever-/Vibrationsbeschleunigung 2 g RMS je Achse (ANNAHME), zusaetzlich zu collective und body_accel"} for part in ("stack", "camera") for axis in range(3)],
    ],
    "sources": {"rotation": "Betaflight-Standard props in: FL und RR CW, FR und RL CCW (von oben); Reaktionsmoment auf den Rahmen +z bei CW; props out kehrt nur die Vorzeichen Mz/Fz um",
                "torque_per_thrust_mm": "Schaetzung wie FORMULATION loads: 52,5 W x 0,75 / (2 pi x 0,75 x KV x 7,4 V / 60) = 8,47 N mm bei 2,02 N -> k_Q = 4,19 mm",
                "rotor_momentum_n_mm_s": "J ~ 5,9e-7 kg m^2 (Prop 1,2 g als Stab 5..31,75 mm + Glocke ~2 g bei 7,5 mm), omega = 0,75 KV 7,4 V sqrt(0,5) bei Mittelschub",
                "points": "Hebel aus COMPONENT_LIBRARY: Motor-SP 9,9/2 ueber Pad, Prop 9,9 + 5/2, AIO 3 + 6/2 ueber Grommetsitz, Akku 11/2 ueber Deck; Kamera-SP auf der Seitenschraubenachse (ANNAHME)",
                "excluded": "Crash-Richtungen sind NICHT in Sigma; sie bleiben Crash-Nebenbedingungen"},
}

class LoadCovariance:
    def __init__(self, config=LOAD_COVARIANCE):
        self.config = c = deepcopy(config)
        self.labels = [(name, dof) for name in c["interfaces"] for dof in c["dofs"]]
        self.motors = [name for name in c["interfaces"] if name.startswith("motor_")]
        self.mean = np.zeros(len(self.labels))
        columns, self.names = [], []
        for factor in c["factors"]:
            for name, column in self.columns(factor):
                columns.append(column * factor["std"])
                self.names.append(name)
                self.mean += column * factor.get("mean", 0.0)
        self.scatter = np.column_stack(columns)
        self.covariance = self.scatter @ self.scatter.T
        self.sigma = self.covariance + np.outer(self.mean, self.mean)
        values, vectors = np.linalg.eigh(self.sigma)
        if values[0] < -1e-9 * values[-1] or not np.allclose(self.sigma, self.sigma.T):
            raise ValueError("Load covariance is not symmetric positive semidefinite")
        self.eigenvalues, vectors = np.clip(values[::-1], 0.0, None), vectors[:, ::-1]
        self.rank = int(np.sum(self.eigenvalues > 1e-10 * self.eigenvalues[0]))
        self.directions = vectors[:, :self.rank] * np.sqrt(self.eigenvalues[:self.rank])
    def wrench(self, name, force, height=0.0, moment=(0.0, 0.0, 0.0)):
        column = np.zeros(len(self.labels))
        start = self.labels.index((name, "Fx"))
        column[start:start + 3] = force
        column[start + 3:start + 6] = np.cross([0.0, 0.0, height], force) + np.asarray(moment)
        return column
    def inertia(self, name, accel):
        return sum(self.wrench(name, -mass / 1000 * np.asarray(accel), height) for mass, height in self.config["points"]["motor" if name in self.motors else name])
    def columns(self, factor):
        c, kind = self.config, factor["kind"]
        if kind == "thrust":
            pattern = np.asarray(factor["pattern"], dtype=float)
            column = sum(self.inertia(name, [0.0, 0.0, pattern.sum() / (c["all_up_mass_g"] / 1000)]) for name in c["interfaces"])
            for name, weight in zip(self.motors, pattern):
                column = column + self.wrench(name, [0.0, 0.0, weight], c["prop_height_mm"], [0.0, 0.0, c["rotation"][name] * c["torque_per_thrust_mm"] * weight])
            return [(factor["name"], column)]
        if kind == "motor":
            unit = np.eye(6)[c["dofs"].index(factor["axis"])]
            return [(factor["name"] + "_" + name, self.wrench(name, unit[:3], c["prop_height_mm"] if factor["at"] == "prop" else 0.0, unit[3:])) for name in self.motors]
        if kind == "gyro":
            rate = np.eye(3)[factor["axis"]]
            return [(factor["name"], sum(self.wrench(name, np.zeros(3), 0.0, np.cross([0.0, 0.0, -c["rotation"][name] * c["rotor_momentum_n_mm_s"]], rate)) for name in self.motors))]
        if kind == "body":
            return [(factor["name"], sum(self.inertia(name, np.eye(3)[factor["axis"]] * c["g_m_s2"]) for name in c["interfaces"]))]
        if kind == "component":
            return [(factor["name"], self.inertia(factor["interface"], np.eye(3)[factor["axis"]] * c["g_m_s2"]))]
        raise ValueError("Unknown load factor kind: " + kind)
    def label(self, index):
        return "/".join(self.labels[index])
    def correlation(self, a, b, centered=False):
        matrix = self.covariance if centered else self.sigma
        i, j = self.labels.index(a), self.labels.index(b)
        return float(matrix[i, j] / np.sqrt(matrix[i, i] * matrix[j, j]))
    def dominant(self, count=6, entries=4):
        rows = []
        for k in range(min(count, self.rank)):
            vector = self.directions[:, k] / np.sqrt(self.eigenvalues[k])
            top = np.argsort(np.abs(vector))[::-1][:entries]
            rows.append({"eigenvalue": float(self.eigenvalues[k]), "share": float(self.eigenvalues[k] / self.eigenvalues.sum()), "entries": [(self.label(i), round(float(vector[i]), 3)) for i in top]})
        return rows
    def markdown(self):
        lines = ["| Faktor | Wert | Einheit | Mittel | Begruendung / Quelle |", "|---|---|---|---|---|"]
        lines += [f"| {f['name']} | {f['std']:g} | {f['unit']} | {f.get('mean', 0.0):g} | {f['source']} |" for f in self.config["factors"]]
        lines += ["", "| Schnittstelle | DOF | Mittel | Streuung | RMS sqrt(Sigma_ii) |", "|---|---|---|---|---|"]
        lines += [f"| {name} | {dof} | {self.mean[i]:.3g} | {np.sqrt(self.covariance[i, i]):.3g} | {np.sqrt(self.sigma[i, i]):.3g} |" for i, (name, dof) in enumerate(self.labels)]
        lines += ["", "| Eigenwert | Anteil | dominante Eintraege |", "|---|---|---|"]
        lines += [f"| {row['eigenvalue']:.4g} | {row['share']:.1%} | " + ", ".join(f"{label} {value:+.2f}" for label, value in row["entries"]) + " |" for row in self.dominant(10)]
        return "\n".join(lines)

def load_covariance(config=LOAD_COVARIANCE):
    model = LoadCovariance(config)
    return {"sigma": model.sigma, "mean": model.mean, "covariance": model.covariance, "directions": model.directions, "eigenvalues": model.eigenvalues, "rank": model.rank, "labels": model.labels, "model": model}

class InterfaceCovariance:
    def __init__(self, system, settings, interfaces):
        self.system, self.settings = system, settings
        if settings.get("sigma") is None:
            model = LoadCovariance(settings.get("model") or LOAD_COVARIANCE)
            keep = [index for index, (name, _) in enumerate(model.labels) if name in settings["interfaces"]]
            sigma, labels = model.sigma[np.ix_(keep, keep)], [model.labels[index] for index in keep]
        else:
            sigma, labels = np.asarray(settings["sigma"], dtype=float), [tuple(label) for label in settings["labels"]]
        missing = sorted({name for name, _ in labels} - set(interfaces))
        if missing:
            raise ValueError("Covariance interfaces missing from the domain: " + ", ".join(missing))
        values, vectors = np.linalg.eigh((sigma + sigma.T) / 2)
        values, vectors = values[::-1], vectors[:, ::-1]
        if values[-1] < -1e-9 * values[0]:
            raise ValueError("Load covariance is not positive semidefinite")
        self.sigma, self.labels = sigma, labels
        self.rank = int(np.sum(values > 1e-10 * values[0]))
        self.directions = vectors[:, :self.rank] * np.sqrt(values[:self.rank])
        self.factor = 1.0 if system.symmetry is None else 2.0
        units = {name: self.wrenches(interfaces[name]) for name in dict.fromkeys(name for name, _ in labels)}
        columns = [units[name][LOAD_COVARIANCE["dofs"].index(dof)] for name, dof in labels]
        self.cases, self.forces = [], []
        for k in range(self.rank):
            force, mirrored = np.zeros(system.ndof), np.zeros(system.ndof)
            for weight, (direct, direct_values, mirror, mirror_values) in zip(self.directions[:, k], columns):
                np.add.at(force, direct, weight * direct_values)
                np.add.at(mirrored, mirror, weight * mirror_values)
            case = system.add_case(settings["prefix"] + str(k), settings["support"], force, mirrored)
            case.pop("force")
            self.cases.append(case["name"])
            self.forces.append({part["sign"]: part["force"] for part in case["parts"]})
    def wrenches(self, interface):
        system = self.system
        selected = [system._select_both(region, "covariance", "load") for region in interface["regions"]]
        direct, mirror = (np.unique(np.concatenate([item[side] for item in selected])) for side in (0, 1))
        flip = np.ones(3) if system.symmetry is None else system._flip()
        arms = np.concatenate([system.points[direct], system.points[mirror] * flip]) - np.asarray(interface["reference_mm"], dtype=float)
        if np.linalg.matrix_rank(arms - arms[0], tol=1e-8) < 2:
            raise ValueError("An interface wrench needs at least three non-collinear nodes")
        blocks = np.zeros((6, 3 * len(arms)))
        for index, (x, y, z) in enumerate(arms):
            blocks[:3, 3 * index:3 * index + 3] = np.eye(3)
            blocks[3:, 3 * index:3 * index + 3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        nodal = (blocks.T @ np.linalg.solve(blocks @ blocks.T, np.eye(6))).reshape(len(arms), 3, 6)
        dofs = lambda nodes: (3 * nodes[:, None] + np.arange(3)).ravel()
        return [(dofs(direct), nodal[:len(direct), :, j].ravel(), dofs(mirror), (nodal[len(direct):, :, j] * flip).ravel()) for j in range(6)]
    def matrix(self, solutions):
        fields = [solutions[name]["fields"] for name in self.cases]
        zero = np.zeros(self.system.ndof)
        flexibility, stacked = np.zeros((self.rank, self.rank)), {}
        for sign in sorted({sign for field in fields for sign in field}):
            stacked[sign] = np.column_stack([field.get(sign, zero) for field in fields])
            flexibility += self.factor * np.column_stack([force.get(sign, zero) for force in self.forces]).T @ stacked[sign]
        return (flexibility + flexibility.T) / 2, stacked
    def energy(self, stacked, vector, derivative):
        total = np.zeros(self.system.nelem)
        for field in stacked.values():
            element = (field @ vector)[self.system.dofs]
            total += np.einsum("ei,ij,ej->e", element, self.system.ke, element, optimize=True)
        return -self.factor * derivative * total
    def measure(self, solutions):
        calibration, s = self.settings["limits"].get("calibration", 1.0), self.settings["ks"]
        factor = lambda key: calibration[key] if isinstance(calibration, dict) else calibration
        limits = {key: self.settings["limits"][key] and self.settings["limits"][key] * factor(key) for key in ("mean_n_mm", "worst_n_mm")}
        flexibility, stacked = self.matrix(solutions)
        mean = float(np.trace(flexibility))
        values, vectors = np.linalg.eigh(flexibility)
        values, vectors = values[::-1], vectors[:, ::-1]
        worst = float(values[0])
        scale = limits["worst_n_mm"] or worst
        weights = np.exp(s * (values - worst) / scale)
        ks = worst + scale * np.log(weights.sum()) / s
        weights /= weights.sum()
        wrench = self.directions @ vectors[:, 0]
        top = np.argsort(np.abs(wrench))[::-1][:6]
        info = {"eigenvalues_n_mm": values[:6].tolist(), "ks_n_mm": float(ks), "ks": s, "rank": self.rank, "worst_wrench": [["/".join(self.labels[i]), float(wrench[i])] for i in top]}
        rows = [{"name": "load_mean", "value": mean, "limit": limits["mean_n_mm"], "unit": "N mm", "sense": "<=", "g": None, "gradient": None},
                {"name": "load_worst", "value": worst, "limit": limits["worst_n_mm"], "unit": "N mm", "sense": "<=", "g": None, "gradient": None, "info": info}]
        if limits["mean_n_mm"]:
            rows[0].update(g=mean / limits["mean_n_mm"] - 1, gradient=sum(solutions[name]["derivative"] for name in self.cases) / limits["mean_n_mm"])
        if limits["worst_n_mm"]:
            derivative = solutions[self.cases[0]]["modulus_derivative"]
            gradient = sum(weight * self.energy(stacked, vectors[:, i], derivative) for i, weight in enumerate(weights) if weight > self.settings["ks_cutoff"])
            rows[1].update(g=float(ks / limits["worst_n_mm"] - 1), gradient=gradient / limits["worst_n_mm"])
        return rows
