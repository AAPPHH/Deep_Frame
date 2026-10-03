from copy import deepcopy
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from deep_frame.config import CRASH_DIRECTIONS, PRINT_MATERIAL
from deep_frame.topology_neural import cell_centers
from deep_frame.topology_optimization import DensityMap, HexElasticity, ModalConstraint, StiffnessConstraint, _settings

PROBLEM = {
    "objective": "mass",
    "volume_max": 0.10,
    "stiffness": {"name": "arm_tip", "case": "stiffness_arm_tip", "min_n_per_mm": 10.0, "calibration": 1.0, "penalty": 1.0, "multiplier_interval": 1,
                  "definition": "evaluator arm_tip: centre mount undersides fixed, uniform pad load on the front-left motor seat, k = |F| / mean pad displacement along F = |F|^2 / compliance"},
    "crash": {"cases": ["crash_" + name for name in CRASH_DIRECTIONS], "ratio": 1.5, "reference": {}, "reference_source": None},
    "modal": {"f1_min_hz": 300.0, "case": "modes", "modes": 6, "tracked": 3, "ks": 40.0, "mass_cutoff": 0.1, "initial_iterations": 30, "warm_iterations": 3, "penalty": 1.0, "multiplier_interval": 1,
              "point_masses": ["battery", "aio15", "camera"]},
    "shadow": {"limit_mm": None, "exponent": 2.0, "hub_radius_mm": 7.88, "motors_mm": [], "radius_mm": None, "source": None,
               "definition": "w = ((r - r_hub) / (R - r_hub))^n inside the prop cylinder (full height), 0 inside the hub; t_w = sum(w rho V) / sum_discs(int w dA) = radially weighted equivalent material thickness under the props in mm"},
    "monitor": ["thrust_all", "torsion_yaw", "twist"],
    "width": {"minimum_mm": 2.0, "eta": 0.5, "delta": 0.25, "minimum_cells": 1.5},
    "interpolation": {"penalization": 3.0, "min_stiffness_ratio": 1e-6, "stiffness": "SIMP p=3, E_min = 1e-6 E", "mass": "linear, rho^6/c^5 below c = 0.1 (modal only, against spurious low-density modes)"},
    "fields": {"stiffness": "eroded", "mass": "intermediate", "volume": "intermediate", "shadow": "intermediate"},
    "continuation": {"beta_schedule": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]},
    "termination": {"mass_change": 1e-3, "violation": 1e-3, "window": 5, "active": 0.01},
    "material": "orthotropic",
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
        report.append({**{key: row[key] for key in ("name", "value", "limit", "unit", "sense", "field")}, "g": g, "status": status, "margin": None if g is None else -g})
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
        self.system = HexElasticity(domain, interface_node_policy=domain.get("optimizer_settings", {}).get("interface_node_policy", "allowed_adjacent"), linear_solver=linear_solver)
        self.factor = 1.0 if self.system.symmetry is None else 2.0
        self.cell = float(np.prod(self.system.spacing))
        self.allowed = int(np.count_nonzero(self.map.allowed))
        self.stiffness = StiffnessConstraint(self.system, problem["stiffness"]) if problem.get("stiffness") else None
        self.modal = ModalConstraint(self.system, problem["modal"]) if problem.get("modal") else None
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
        rows = []
        if self.stiffness is not None:
            g, slope, info = self.stiffness.measure(solutions[self.stiffness.case])
            rows.append({"name": self.problem["stiffness"]["name"] + "_stiffness", "g": float(g), "gradient": slope, "value": info["stiffness_n_per_mm"], "limit": info["target_n_per_mm"], "unit": "N/mm", "sense": ">="})
        for name in self.crash:
            limit = self.problem["crash"]["ratio"] * self.problem["crash"]["reference"][name]
            rows.append({"name": name, "g": solutions[name]["compliance_n_mm"] / limit - 1, "gradient": solutions[name]["derivative"] / limit, "value": solutions[name]["compliance_n_mm"], "limit": limit, "unit": "N mm", "sense": "<="})
        if self.modal is not None:
            g, slope, info = self.modal.measure(physical, self.penalization, self.min_stiffness_ratio, iterations)
            rows.append({"name": "f1", "g": float(g), "gradient": slope, "value": info["f1_hz"], "limit": self.problem["modal"]["f1_min_hz"], "unit": "Hz", "sense": ">=", "info": info})
        monitor = [{"name": name, "g": None, "gradient": None, "value": solutions[name]["compliance_n_mm"], "limit": None, "unit": "N mm", "sense": "", "field": self.problem["fields"]["stiffness"]} for name in self.monitor]
        for row in rows:
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
    return {**deepcopy(PROBLEM), "volume_max": volume_max, "crash": None, "modal": None, "shadow": None, "monitor": [],
            "stiffness": {**PROBLEM["stiffness"], "name": "tip", "case": "tip", "min_n_per_mm": min_n_per_mm}}
