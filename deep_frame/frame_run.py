import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from math import ceil, cos, hypot, radians, sin, tan
from pathlib import Path
from time import perf_counter

import numpy as np

from deep_frame.config import (COMPONENT_DEFAULTS, COMPONENT_LIBRARY, DURABILITY, FRAME_COMPONENT_KINDS, FRAME_DEFAULTS, FRAME_LAYOUT_KINDS, FRAME_PRINT_KINDS, FRAME_REQUEST,
                               FRAME_REQUEST_KINDS, IMPLICIT_CONFIG, LAYOUT_DEFAULT, LAYOUT_OPTIMIZATION, LAYOUT_OVERRIDES, LAYOUT_REFERENCES, LAYOUT_RULES, LIBRARY_FIELDS, MATERIALS, MOUNTING_TYPES, RUN_GRIDS, RUN_SETTINGS, STAGES, STYLES, TOPOLOGY_CONFIG, component_spec,
                               configure, prop_spec)
from deep_frame.frame import camera_mount_z, motor_positions, prop_plane_z
from deep_frame.frame_evaluation import component_inertia, rigid_assembly, stack_pattern

ROOT = Path(__file__).resolve().parents[1]
MOTORS = ("front_left", "front_right", "rear_left", "rear_right")

def _lookup(entry, path):
    for key in path.split("."):
        if not isinstance(entry, dict) or key not in entry:
            return None
        entry = entry[key]
    return entry

def required_fields(kind):
    return LIBRARY_FIELDS["common"] + LIBRARY_FIELDS.get(kind, [])

def library_part(role, name, library=COMPONENT_LIBRARY):
    if name not in library:
        raise ValueError(f"Component '{name}' for '{role}' is not in COMPONENT_LIBRARY; add an entry with the fields: " + ", ".join(required_fields(role)))
    entry = library[name]
    missing = [field for field in required_fields(role) if field not in ("hole_pattern",) and _lookup(entry, field) is None]
    if entry.get("type") != role or missing or entry["mounting"] not in MOUNTING_TYPES:
        raise ValueError(f"Library entry '{name}' is not a valid '{role}': missing or invalid fields " + ", ".join(missing or ["type/mounting"]) + "; required: " + ", ".join(required_fields(role)))
    return deepcopy(entry)

def validate_request(request):
    config = configure(FRAME_REQUEST, FRAME_REQUEST_KINDS, request)
    config["layout"] = configure(FRAME_REQUEST["layout"], FRAME_LAYOUT_KINDS, config["layout"])
    config["print"] = configure(FRAME_REQUEST["print"], FRAME_PRINT_KINDS, config["print"])
    config["components"] = configure(FRAME_REQUEST["components"], FRAME_COMPONENT_KINDS, config["components"])
    unknown = sorted(set(config["overrides"]) - set(LAYOUT_OVERRIDES))
    if unknown:
        raise ValueError("Unknown override groups: " + ", ".join(unknown) + "; allowed: " + ", ".join(LAYOUT_OVERRIDES))
    config["overrides"] = {group: configure(dict.fromkeys(kinds), kinds, config["overrides"][group]) for group, kinds in LAYOUT_OVERRIDES.items() if group in config["overrides"]}
    config["overrides"] = {group: {key: value for key, value in values.items() if value is not None} for group, values in config["overrides"].items()}
    if not 1.5 <= config["prop_size_in"] <= 5.0:
        raise ValueError("prop_size_in must lie between 1.5 and 5 inch")
    nozzle, layer = config["print"]["nozzle_mm"], config["print"]["layer_mm"]
    if not 0.2 <= nozzle <= 1.0 or not 0.05 <= layer <= 0.8 * nozzle:
        raise ValueError("print needs 0.2 <= nozzle_mm <= 1.0 and 0.05 <= layer_mm <= 0.8 x nozzle_mm")
    config["name"] = config["name"] or f"{config['style']}_{config['durability']}_{config['prop_size_in']:g}in"
    return config

AXES = ("roll", "pitch", "yaw")
ROTATION_AXES = (1, 0, 2)

class LayoutModel:
    def __init__(self, setup, settings=LAYOUT_OPTIMIZATION):
        self.setup, self.settings, s = setup, settings, setup
        self.names = list(settings["variables"])
        self.bounds = [tuple(s.get("bounds", {}).get(name, settings["variables"][name])) for name in self.names]
        motors = np.asarray(s["motors_xy"], dtype=float)
        self.hub = motors.mean(axis=0)
        thrust, torque = s["thrust_n"], s["torque_per_thrust_mm"]
        self.torque_n_mm = np.array([2 * thrust * np.abs(motors[:, 0] - self.hub[0]).mean(), 2 * thrust * np.abs(motors[:, 1] - self.hub[1]).mean(), 2 * torque * thrust])
        motor, prop = s["motor"], s["prop"]
        self.prop_bottom = s["pad_z_mm"] + motor["height_mm"]
        self.rotor_z = self.prop_bottom + prop["thickness_mm"] / 2
        self.prop_top = self.prop_bottom + prop["thickness_mm"]
        frame = s["frame"]
        self.fixed = [(frame["mass_g"], frame["center_mm"], frame["inertia_g_mm2"])]
        for x, y in motors:
            self.fixed.append((motor["mass_g"], [x, y, s["pad_z_mm"] + motor["height_mm"] / 2], component_inertia({"mass_g": motor["mass_g"], "size_mm": [motor["diameter_mm"]] * 2 + [motor["height_mm"]], "shape": "box"})))
            self.fixed.append((prop["mass_g"], [x, y, self.rotor_z], component_inertia({"mass_g": prop["mass_g"], "size_mm": [prop["diameter_mm"]] * 2 + [prop["thickness_mm"]], "shape": "disc"})))
        radial, angular = settings["fov_samples"]
        radius, angle = np.meshgrid(np.linspace(prop.get("hub_diameter_mm", 0.0) / 2, prop["diameter_mm"] / 2, radial), np.linspace(0, 2 * np.pi, angular, endpoint=False))
        ring = np.stack([radius.ravel() * np.cos(angle.ravel()), radius.ravel() * np.sin(angle.ravel())], axis=1)
        self.prop_points = np.asarray([[x + dx, y + dy, z] for x, y in motors for dx, dy in ring for z in (self.prop_bottom, self.prop_top)])
        camera = s["camera"]
        tilt = radians(camera["tilt_deg"])
        self.axes = np.array([[1.0, 0.0, 0.0], [0.0, -sin(tilt), cos(tilt)], [0.0, cos(tilt), sin(tilt)]])
        self.camera_extent = (camera["length_mm"] * cos(tilt) + camera["height_mm"] * sin(tilt), camera["height_mm"] * cos(tilt) + camera["length_mm"] * sin(tilt))

    def vector(self, layout):
        return np.asarray([layout[name] for name in self.names], dtype=float)

    def layout(self, vector):
        return {name: float(value) for name, value in zip(self.names, vector)}

    def placement(self, vector):
        s, v = self.setup, self.layout(vector)
        battery, aio, camera = s["battery"], s["aio"], s["camera"]
        camera_z = s["base_mm"] + v["camera_bottom_clearance_mm"] + self.camera_extent[1] / 2
        return {"battery": [0.0, v["battery_y_mm"], v["deck_top_mm"] + battery["height_mm"] / 2], "aio": [0.0, s.get("aio_y_mm", 0.0), s["base_mm"] + v["aio_standoff_mm"] + aio["stack_height_mm"] / 2],
                "camera": [0.0, v["camera_y_mm"], camera_z], "camera_top_mm": camera_z + self.camera_extent[1] / 2, "stack_top_mm": s["base_mm"] + v["aio_standoff_mm"] + aio["stack_height_mm"] + aio["elrs_mm"],
                "lens_mm": [0.0, v["camera_y_mm"] + camera["length_mm"] / 2 * self.axes[2][1], camera_z + camera["length_mm"] / 2 * self.axes[2][2]]}

    def properties(self, vector):
        s, place = self.setup, self.placement(vector)
        sizes = {"battery": ("width_mm", "length_mm", "height_mm"), "aio": ("width_mm", "length_mm", "stack_height_mm"), "camera": ("width_mm", "length_mm", "height_mm")}
        bodies = self.fixed + [(s[name]["mass_g"], place[name], component_inertia({"mass_g": s[name]["mass_g"], "size_mm": [s[name][key] for key in keys], "shape": "box"})) for name, keys in sizes.items()]
        total, center, inertia = rigid_assembly(bodies)
        alpha = self.torque_n_mm / np.diag(inertia)[list(ROTATION_AXES)] * 1e6
        return {"mass_g": float(total), "center_of_mass_mm": center.tolist(), "cg_above_rotor_plane_mm": float(center[2] - self.rotor_z), "inertia_g_mm2": inertia.tolist(),
                "torque_n_mm": dict(zip(AXES, self.torque_n_mm.tolist())), "alpha_rad_s2": dict(zip(AXES, alpha.tolist())), "alpha_min_rad_s2": float(alpha.min()), "limiting_axis": AXES[int(np.argmin(alpha))]}

    def fov_margin(self, vector):
        local = (self.prop_points - np.asarray(self.placement(vector)["lens_mm"])) @ self.axes.T
        off = np.arccos(np.clip(local[:, 2] / np.linalg.norm(local, axis=1), -1.0, 1.0))
        azimuth = np.arctan2(local[:, 1], local[:, 0])
        half = np.radians(self.settings["fov_deg"]) / 2
        return float(np.degrees(np.min(np.maximum(np.abs(off * np.cos(azimuth)) - half[0], np.abs(off * np.sin(azimuth)) - half[1]))))

    def constraints(self, vector):
        s, v, place, settings = self.setup, self.layout(vector), self.placement(vector), self.settings
        center = np.asarray(self.properties(vector)["center_of_mass_mm"])
        battery = s["battery"]
        deck_bottom = v["deck_top_mm"] - s["deck_thickness_mm"]
        low, high = settings["cg_band_mm"]
        corners = np.asarray([[sx * battery["width_mm"] / 2, v["battery_y_mm"] + sy * battery["length_mm"] / 2] for sx in (-1, 1) for sy in (-1, 1)])
        plan = min(float(np.hypot(*(np.clip(m, corners.min(0), corners.max(0)) - m))) for m in np.asarray(s["motors_xy"], dtype=float)) - s["prop"]["diameter_mm"] / 2
        clearance = s["battery_prop_clearance_mm"]
        return {"cg_x": settings["cg_horizontal_mm"] - abs(center[0] - self.hub[0]), "cg_y": settings["cg_horizontal_mm"] - abs(center[1] - self.hub[1]),
                "cg_above_rotor_plane": center[2] - self.rotor_z - low, "cg_band_top": high - (center[2] - self.rotor_z),
                "battery_over_stack": deck_bottom - place["stack_top_mm"] - settings["battery_gap_mm"],
                "battery_over_camera": max(deck_bottom - place["camera_top_mm"] - settings["battery_gap_mm"], abs(v["battery_y_mm"] - v["camera_y_mm"]) - (battery["length_mm"] + self.camera_extent[0]) / 2),
                "camera_sees_no_prop": self.fov_margin(vector), "camera_stack_gap": v["camera_y_mm"] - self.camera_extent[0] / 2 - s.get("aio_y_mm", 0.0) - s["aio"]["length_mm"] / 2 - settings["mount_gap_mm"],
                "camera_in_envelope": s["envelope"]["half_y_mm"] - v["camera_y_mm"] - max(self.camera_extent[0] / 2, s["hoop"]["front_mm"]), "camera_under_hoop": s["hoop"]["top_mm"] - place["camera_top_mm"],
                "battery_clear_of_props": max(plan - clearance, v["deck_top_mm"] - self.prop_top - clearance), "battery_in_envelope": s["envelope"]["half_y_mm"] - abs(v["battery_y_mm"]) - battery["length_mm"] / 2,
                "deck_in_envelope": s["envelope"]["top_mm"] - v["deck_top_mm"]}

    def evaluate(self, layout):
        vector = self.vector(layout)
        constraints = {name: float(value) for name, value in self.constraints(vector).items()}
        return {"layout": self.layout(vector), **self.properties(vector), "constraints": constraints, "violated": [name for name, value in constraints.items() if value < -1e-6], "feasible": min(constraints.values()) >= -1e-6}

    def alpha(self, vector):
        return self.torque_n_mm / np.diag(self.properties(vector)["inertia_g_mm2"])[list(ROTATION_AXES)] * 1e6

    def margins(self, vector, names):
        values = self.constraints(vector)
        return np.asarray([values[name] for name in names])

    def reachable(self):
        from scipy.optimize import differential_evolution
        solver = self.settings["reach_solver"]
        names = list(self.constraints(np.mean(self.bounds, axis=1)))
        return {name: float(-differential_evolution(lambda vector, name=name: -self.constraints(vector)[name], self.bounds, popsize=solver["popsize"], maxiter=solver["maxiter"], seed=solver["seed"], polish=True).fun) for name in names}

    def optimize(self):
        from scipy.optimize import NonlinearConstraint, differential_evolution
        solver, tolerance, held, stages, best = self.settings["solver"], self.settings["rank_tolerance"], {}, [], None
        reach = self.reachable()
        active = [name for name, value in reach.items() if value >= -1e-6]
        limits = [NonlinearConstraint(lambda vector: self.margins(vector, active), 0.0, np.inf)]
        for _ in AXES:
            free = [index for index in range(3) if index not in held]
            if held:
                fixed = dict(held)
                limits = limits[:1] + [NonlinearConstraint(lambda vector, fixed=fixed: self.alpha(vector)[list(fixed)] - np.asarray(list(fixed.values())), 0.0, np.inf)]
            result = differential_evolution(lambda vector, free=free: -self.alpha(vector)[free].min(), self.bounds, constraints=limits, popsize=solver["popsize"], maxiter=solver["maxiter"], tol=solver["tol"],
                                            seed=solver["seed"], polish=True, x0=best)
            values = self.alpha(result.x)
            axis = free[int(np.argmin(values[free]))]
            held[axis] = values[axis] * (1 - tolerance)
            stages.append({"maximised": AXES[axis], "alpha_rad_s2": float(values[axis]), "evaluations": int(result.nfev), "feasible": bool(self.margins(result.x, active).min(initial=0.0) >= -1e-6), "message": str(result.message)})
            best = result.x
        return {**self.evaluate(self.layout(best)), "stages": stages, "reachable_margin": reach, "unsatisfiable": [name for name in reach if name not in active],
                "constraints_used": active}

    def rounded(self, layout):
        from itertools import product
        step, candidates = self.settings["round_mm"], []
        for picks in product((np.floor, np.ceil), repeat=len(self.names)):
            result = self.evaluate({name: round(float(pick(layout[name] / step) * step), 6) for name, pick in zip(self.names, picks)})
            if result["feasible"]:
                candidates.append(result)
        return max(candidates, key=lambda result: (result["alpha_min_rad_s2"], sorted(result["alpha_rad_s2"].values()))) if candidates else None

def layout_setup(name, references=LAYOUT_REFERENCES, settings=LAYOUT_OPTIMIZATION):
    from deep_frame.topology_problem import LOAD_COVARIANCE
    setup = deepcopy(references[name])
    setup.setdefault("thrust_n", settings["thrust_n"])
    setup.setdefault("torque_per_thrust_mm", LOAD_COVARIANCE["torque_per_thrust_mm"])
    if "parts" in setup["frame"]:
        total, center, inertia = rigid_assembly([(part["mass_g"], part["center_mm"], component_inertia(part)) for part in setup["frame"]["parts"]])
        setup["frame"] = {**setup["frame"], "mass_g": float(total), "center_mm": center.tolist(), "inertia_g_mm2": inertia.tolist()}
    return setup

def layout_study(output=None, settings=LAYOUT_OPTIMIZATION):
    layout = FrameLayout({})
    result = {"settings": settings, "frames": {}}
    for name, setup in [("ours", {**layout.setup(settings), "own": settings["frame_share"]["layout"]}), *[(name, layout_setup(name, settings=settings)) for name in LAYOUT_REFERENCES]]:
        model = LayoutModel(setup, settings)
        optimum = model.optimize()
        entry = {"setup": setup, "optimum": optimum, "rounded": model.rounded(optimum["layout"])}
        if setup.get("own"):
            entry["own"] = model.evaluate(setup["own"])
            entry["own_vs_optimum"] = {"alpha_min": entry["own"]["alpha_min_rad_s2"] / optimum["alpha_min_rad_s2"], **{axis: entry["own"]["alpha_rad_s2"][axis] / optimum["alpha_rad_s2"][axis] for axis in AXES}}
        result["frames"][name] = entry
    target = Path(output) if output else ROOT / "exports" / "layout" / "layout_optimization.json"
    _save(target, result)
    print(json.dumps({name: {"optimum": {key: round(value, 2) for key, value in entry["optimum"]["layout"].items()}, "alpha_min": round(entry["optimum"]["alpha_min_rad_s2"], 1), "feasible": entry["optimum"]["feasible"],
                             "violated": entry["optimum"]["violated"], "unsatisfiable": entry["optimum"]["unsatisfiable"], "own_vs_optimum": entry.get("own_vs_optimum")} for name, entry in result["frames"].items()}, indent=1))
    return result

class FrameLayout:
    def __init__(self, request):
        self.request = validate_request(request)
        self.parts = {role: library_part(role, name) for role, name in self.request["components"].items()}
        self.overrides = self.request["overrides"]
        self.free_camera = self._override("camera", "support", TOPOLOGY_CONFIG["camera_support"]) == "free"
        self.style, self.durability = {**STYLES[self.request["style"]], "hoops": STYLES[self.request["style"]]["hoops"] and not self.free_camera}, DURABILITY[self.request["durability"]]
        self.material = MATERIALS[self.request["material"]]
        self.notes = []
        self.components = self._components()
        self.frame = self._frame()
        self.checks = self._checks()

    def _override(self, group, key, value):
        return self.overrides.get(group, {}).get(key, value)

    def _components(self):
        battery = self.parts["battery"]
        connectors = {"xt30": battery["data"]["power_connector"], "balancer": battery["data"]["balance_connector"]}
        specs = {"aio15": component_spec(self.parts["aio"]), "camera": component_spec(self.parts["camera"]), "battery": component_spec(battery), "motor": component_spec(self.parts["motor"]),
                 **{name: component_spec(library_part("connector", part)) for name, part in connectors.items()}, "prop": prop_spec(self.request["prop_size_in"], self.parts["prop"])}
        if self.parts["prop"]["data"]["size_in"] != self.request["prop_size_in"]:
            self.notes.append(f"prop {self.request['components']['prop']} is {self.parts['prop']['data']['size_in']:g} inch; {self.request['prop_size_in']:g} inch uses the provisional swept-disk rule")
        specs["camera"]["tilt_deg"] = self._override("camera", "tilt_deg", COMPONENT_DEFAULTS["camera"]["tilt_deg"])
        specs["camera"].setdefault("parameter_sources", {})["tilt_deg"] = "override" if "tilt_deg" in self.overrides.get("camera", {}) else "design: adjustable initial camera tilt"
        return specs

    def _frame(self):
        rules, c = LAYOUT_RULES, self.components
        angle = self._override("motors", "arm_angle_deg", rules["x_types"][self.request["layout"]["x_type"]]["arm_angle_deg"])
        if not 20.0 <= angle <= 70.0:
            raise ValueError("motors.arm_angle_deg must lie between 20 and 70 degrees")
        spread = min(sin(radians(angle)), cos(radians(angle)))
        minimum = (c["prop"]["diameter_mm"] + rules["prop_tip_gap_mm"]) / spread
        wheelbase = self._override("motors", "wheelbase_mm", ceil(minimum / rules["wheelbase_step_mm"] - 1e-9) * rules["wheelbase_step_mm"])
        if wheelbase < minimum - 1e-9:
            raise ValueError(f"motors.wheelbase_mm {wheelbase} leaves less than {rules['prop_tip_gap_mm']} mm between neighbouring {c['prop']['diameter_mm']} mm props; minimum {minimum:.1f} mm")
        aio, camera = c["aio15"], c["camera"]
        mount = self.request["layout"]["battery_mount"]
        optimum = self._optimum()
        default = optimum or {"camera_y_mm": aio["length_mm"] / 2 + rules["camera"]["stack_gap_mm"] + camera["length_mm"] / 2, "aio_standoff_mm": FRAME_DEFAULTS["aio_standoff_mm"],
                              "camera_bottom_clearance_mm": FRAME_DEFAULTS["camera_bottom_clearance_mm"], "battery_y_mm": FRAME_DEFAULTS["battery_y_mm"], "deck_top_mm": rules["battery_mounts"]["top"]["deck_top_mm"]}
        if optimum is None:
            self.notes.append("no layout optimum for this request (LAYOUT_DEFAULT covers " + json.dumps(LAYOUT_DEFAULT["request"]) + "); rule layout used, run 'run.py layout' for this request")
        frame = {"wheelbase_mm": wheelbase, "lateral_longitudinal_ratio": tan(radians(angle)), "arm_height_mm": rules["pad"]["top_mm"],
                 "camera_y_mm": self._override("camera", "y_mm", default["camera_y_mm"]), "aio_standoff_mm": self._override("stack", "standoff_mm", default["aio_standoff_mm"]),
                 "camera_bottom_clearance_mm": self._override("camera", "bottom_clearance_mm", default["camera_bottom_clearance_mm"]), "battery_y_mm": self._override("battery", "y_mm", default["battery_y_mm"]),
                 "camera_screw_diameter_mm": camera.get("screw_clearance_mm", FRAME_DEFAULTS["camera_screw_diameter_mm"])}
        if mount == "top":
            frame["deck_top_mm"] = self._override("battery", "deck_top_mm", default["deck_top_mm"])
        else:
            need = c["battery"]["height_mm"] + rules["battery_mounts"]["bottom"]["gap_mm"]
            floor = rules["envelope"]["origin_mm"][2]
            if floor > -need:
                raise ValueError(f"battery_mount bottom needs {need:.1f} mm below the base plate, but the design envelope starts at z = {floor:.1f} mm; choose battery_mount top")
        return frame

    def _optimum(self):
        key = LAYOUT_DEFAULT["request"]
        request = {"x_type": self.request["layout"]["x_type"], "battery_mount": self.request["layout"]["battery_mount"], "prop_size_in": self.request["prop_size_in"], "components": self.request["components"],
                   "tilt_deg": self.components["camera"]["tilt_deg"], "motors": self.overrides.get("motors", {})}
        return {name: value for name, value in LAYOUT_DEFAULT["layout"].items()} if request == key else None

    def setup(self, settings=LAYOUT_OPTIMIZATION, frame=None):
        from deep_frame.topology_problem import LOAD_COVARIANCE
        c, f, rules = self.components, {**FRAME_DEFAULTS, **self.frame}, LAYOUT_RULES
        pick = lambda part, keys: {key: c[part][key] for key in keys if key in c[part]}
        envelope = rules["envelope"]
        path = np.asarray(rules["hoop"]["path_yz_mm"]) if self.style["hoops"] else np.zeros((1, 2))
        return {"motors_xy": list(self.motors().values()), "pad_z_mm": f["arm_height_mm"], "motor": pick("motor", ("mass_g", "diameter_mm", "height_mm")),
                "prop": pick("prop", ("mass_g", "diameter_mm", "thickness_mm", "hub_diameter_mm")), "battery": pick("battery", ("mass_g", "width_mm", "length_mm", "height_mm")),
                "aio": {**pick("aio15", ("mass_g", "width_mm", "length_mm", "stack_height_mm")), "elrs_mm": c["aio15"]["elrs_antenna_clearance_mm"]}, "camera": pick("camera", ("mass_g", "width_mm", "length_mm", "height_mm", "tilt_deg")),
                "base_mm": f["base_thickness_mm"], "deck_thickness_mm": f["deck_thickness_mm"], "battery_prop_clearance_mm": rules["battery_prop_clearance_mm"],
                "envelope": {"half_y_mm": min(-envelope["origin_mm"][1], envelope["origin_mm"][1] + envelope["size_mm"][1]), "top_mm": envelope["origin_mm"][2] + envelope["size_mm"][2]},
                "hoop": {"front_mm": float(path[:, 0].max()) + (rules["hoop"]["radius_mm"] if self.style["hoops"] else 0.0), "top_mm": float(path[:, 1].max()) if self.style["hoops"] else envelope["origin_mm"][2] + envelope["size_mm"][2]},
                "frame": {key: value for key, value in (frame or settings["frame_share"]).items() if key in ("mass_g", "center_mm", "inertia_g_mm2", "source")}, "thrust_n": settings["thrust_n"],
                "torque_per_thrust_mm": LOAD_COVARIANCE["torque_per_thrust_mm"], "own": {name: f[name] for name in settings["variables"]}}

    def agility(self, settings=LAYOUT_OPTIMIZATION):
        setup = self.setup(settings)
        return {"setup": setup, **LayoutModel(setup, settings).evaluate(setup["own"]), "source": LAYOUT_DEFAULT["source"] if self._optimum() else "rule layout (no optimum for this request)"}

    def parameters(self):
        frame = {**FRAME_DEFAULTS, **self.frame}
        return {"frame": frame, "components": self.components}

    def motors(self):
        return {name: list(xy) for name, xy in motor_positions(self.parameters()).items()}

    def hoop(self):
        rules, camera_y = LAYOUT_RULES["hoop"], self.frame["camera_y_mm"]
        return {"x_mm": self.components["camera"]["width_mm"] / 2 + rules["side_gap_mm"], "radius_mm": rules["radius_mm"], "path_yz_mm": [[y + camera_y, z] for y, z in rules["path_yz_mm"]],
                "load_y_min_mm": rules["load_y_min_mm"] + camera_y, "load_z_mm": list(rules["load_z_mm"]), "case_weight": self.durability["crash_weight"]}

    def _checks(self):
        rules, c, f = LAYOUT_RULES, self.components, {**FRAME_DEFAULTS, **self.frame}
        low = np.asarray(rules["envelope"]["origin_mm"][:2])
        high = low + np.asarray(rules["envelope"]["size_mm"][:2])
        pad = c["motor"]["diameter_mm"] / 2 + self.parts["motor"]["keep_out"]["clearance_mm"]
        motors = np.asarray(list(self.motors().values()))
        if np.any(motors - pad < low - 1e-9) or np.any(motors + pad > high + 1e-9):
            raise ValueError(f"Motors (radius {pad:.2f} mm with keep-out) leave the design envelope {low.tolist()}..{high.tolist()} at wheelbase {f['wheelbase_mm']} mm; use a smaller prop or another x_type")
        tilt = radians(c["camera"]["tilt_deg"])
        extent_y = abs(c["camera"]["length_mm"] * cos(tilt)) + abs(c["camera"]["height_mm"] * sin(tilt))
        camera_top = camera_mount_z(self.parameters()) + (abs(c["camera"]["height_mm"] * cos(tilt)) + abs(c["camera"]["length_mm"] * sin(tilt))) / 2
        hoop = self.hoop()
        front = max(f["camera_y_mm"] + extent_y / 2, max(y for y, _ in hoop["path_yz_mm"]) + hoop["radius_mm"])
        if front > high[1] or f["camera_y_mm"] - extent_y / 2 < c["aio15"]["length_mm"] / 2:
            raise ValueError(f"Camera at y = {f['camera_y_mm']:.1f} mm with tilt {c['camera']['tilt_deg']} deg does not fit between the stack and the envelope front {high[1]} mm")
        stack_top = f["base_thickness_mm"] + f["aio_standoff_mm"] + c["aio15"]["stack_height_mm"] + c["aio15"]["elrs_antenna_clearance_mm"]
        deck_bottom = f["deck_top_mm"] - f["deck_thickness_mm"]
        over_camera = max(f["deck_top_mm"] - camera_top - rules["camera"]["top_clearance_mm"], abs(f["battery_y_mm"] - f["camera_y_mm"]) - (c["battery"]["length_mm"] + extent_y) / 2)
        if self.request["layout"]["battery_mount"] == "top" and min(over_camera, deck_bottom - stack_top - rules["camera"]["top_clearance_mm"]) < -1e-9:
            raise ValueError(f"Battery underside z = {f['deck_top_mm']:.1f} mm or deck bottom z = {deck_bottom:.1f} mm leaves less than {rules['camera']['top_clearance_mm']} mm above camera top {camera_top:.1f} mm (camera not ahead of the battery) or stack top {stack_top:.1f} mm")
        prop_plane = prop_plane_z(self.parameters())
        corners = np.asarray([[sx * c["battery"]["width_mm"] / 2, f["battery_y_mm"] + sy * c["battery"]["length_mm"] / 2] for sx in (-1, 1) for sy in (-1, 1)])
        plan = min(float(np.min(np.hypot(*(np.clip(m, corners.min(0), corners.max(0)) - m)))) for m in motors) - c["prop"]["diameter_mm"] / 2
        if plan < rules["battery_prop_clearance_mm"] and f["deck_top_mm"] < prop_plane + rules["battery_prop_clearance_mm"]:
            raise ValueError("Battery overlaps the prop disks in plan and does not sit above the prop plane")
        masses = [(c["motor"]["mass_g"] + c["prop"]["mass_g"], *m) for m in motors] + [(c["aio15"]["mass_g"], 0.0, 0.0), (c["camera"]["mass_g"], 0.0, f["camera_y_mm"]), (c["battery"]["mass_g"], 0.0, f["battery_y_mm"])]
        total = sum(m for m, _, _ in masses)
        cg = [sum(m * x for m, x, _ in masses) / total, sum(m * y for m, _, y in masses) / total]
        if hypot(*cg) > rules["cg_tolerance_mm"]:
            raise ValueError(f"Component centre of gravity {cg} is more than {rules['cg_tolerance_mm']} mm from the battery and stack axis")
        return {"motor_pad_envelope": True, "camera_fit": True, "camera_top_mm": camera_top, "stack_top_mm": stack_top, "battery_plan_clearance_mm": plan, "prop_plane_mm": prop_plane,
                "component_cg_mm": cg, "component_mass_g": total}

    def summary(self):
        f = {**FRAME_DEFAULTS, **self.frame}
        return {"request": self.request, "frame": self.frame, "motors_mm": self.motors(), "hoop": self.hoop() if self.style["hoops"] else None, "checks": self.checks,
                "battery": {"mount": self.request["layout"]["battery_mount"], "support": self.support(), "deck_top_mm": f["deck_top_mm"], "position_mm": [0.0, f["battery_y_mm"], f["deck_top_mm"]]},
                "stack": {"position_mm": [0.0, 0.0, f["base_thickness_mm"] + f["aio_standoff_mm"]], "posts": {**self.post(), "standoff_mm": f["aio_standoff_mm"], "height_mm": max(f["aio_standoff_mm"], self.post()["bore_depth_mm"] + self.post()["bore_floor_mm"])}}, "camera": {"y_mm": f["camera_y_mm"], "bottom_clearance_mm": f["camera_bottom_clearance_mm"], "tilt_deg": self.components["camera"]["tilt_deg"]}, "agility": self.agility(),
                "style": self.style, "durability": self.durability, "material": self.material, "parts": {role: part["source"] for role, part in self.parts.items()}, "notes": self.notes}

    def support(self):
        return self._override("battery", "support", TOPOLOGY_CONFIG["battery_support"])

    def post(self):
        post = TOPOLOGY_CONFIG["stack_post"]
        fastening = self._override("stack", "fastening", post["fastening"])
        bore = post["bores"][fastening]
        return {"diameter_mm": post["diameter_mm"], "fastening": fastening, "bore_diameter_mm": bore["diameter_mm"], "bore_depth_mm": bore["depth_mm"], "bore_floor_mm": post["bore_floor_mm"]}

    def patch(self, crash_cases=()):
        material = {key: self.material[key] for key in ("density_g_cm3", "young_modulus_mpa", "poisson_ratio")}
        weights = {name: self.durability["crash_weight"] for name in crash_cases}
        patch = {"FRAME_DEFAULTS": self.frame, "COMPONENT_DEFAULTS": self.components, "FEA_CONFIG": {"material": material},
                 "TOPOLOGY_CONFIG": {"manufacturing": {"nozzle_width_mm": self.request["print"]["nozzle_mm"], "minimum_feature_mm": self.durability["minimum_width_mm"]},
                                     "battery_support": self.support(), "camera_support": "free" if self.free_camera else "prescribed", "stack_post": {"fastening": self.post()["fastening"]}},
                 "INTEGRATION_CONFIG": {"crash_directions": list(self.style["crash_directions"])}}
        if weights:
            patch["TOPOLOGY_CONFIG"]["optimizer"] = {"case_weights": weights}
        return patch

def _git(worktree):
    def run(*args):
        result = subprocess.run(["git", "-C", str(worktree), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"sha": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None

def _save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)), encoding="utf-8")

def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None

class FrameRun:
    def __init__(self, request, stages=STAGES, settings=RUN_SETTINGS):
        self.layout = FrameLayout(request)
        self.request, self.stages, self.settings = self.layout.request, stages, settings
        self.grid = RUN_GRIDS[self.request["grid"]]
        root = Path(settings["root"]) if Path(settings["root"]).is_absolute() else ROOT / settings["root"]
        index = 1 + max([int(path.name.rsplit("_", 1)[1]) for path in root.glob(self.request["name"] + "_*") if path.name.rsplit("_", 1)[1].isdigit()] + [0])
        self.dir = root / f"{self.request['name']}_{index}"
        self.dir.mkdir(parents=True)
        self.manifest = {"name": self.dir.name, "status": "running", "run_interface": _git(ROOT), "grid": self.request["grid"], "stages": {}, "notes": list(self.layout.notes)}
        _save(self.dir / "config.json", self.request)
        _save(self.dir / "layout.json", self.layout.summary())
        self.save()

    def save(self):
        _save(self.dir / "manifest.json", self.manifest)

    def available(self, stage):
        spec = self.stages[stage]
        return Path(spec["worktree"]).is_dir() and (Path(spec["worktree"]) / spec["tool"]).is_file()

    def command(self, stage, request, action, compute=None, label=None):
        spec = self.stages[stage]
        path = self.dir / "requests" / f"{label or stage}.json"
        _save(path, {"root": spec["worktree"], "tool": spec["tool"], "action": action, **request})
        inner = [spec["python"], str(ROOT / "run.py"), "stage", str(path)]
        kind = compute if compute is not None else self.grid["compute"].get(stage, spec["compute"])
        return [sys.executable, self.settings["compute"], kind, "--cwd", spec["worktree"], "--", *inner] if kind else inner

    def execute(self, stage, command, cwd, outputs):
        spec = self.stages[stage]
        entry = {"status": "running", "worktree": spec["worktree"], "tool": spec["tool"], "compute": self.grid["compute"].get(stage, spec["compute"]), **_git(spec["worktree"]), "command": subprocess.list2cmdline(command)}
        if spec["tool"].startswith("exports/"):
            entry["tool_sha256"] = _digest(Path(spec["worktree"]) / spec["tool"])
        self.manifest["stages"][stage] = entry
        self.save()
        started = perf_counter()
        with (self.dir / "logs" / f"{stage}.log").open("w", encoding="utf-8") as log:
            code = subprocess.call(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        missing = [str(path) for path in outputs if not Path(path).exists()]
        entry.update(runtime_s=round(perf_counter() - started, 1), exit_code=code, status="ran" if code == 0 and not missing else "failed", missing_outputs=missing)
        self.save()
        return entry["status"] == "ran"

    def mark(self, stage, status, reason):
        spec = self.stages.get(stage, {})
        self.manifest["stages"][stage] = {"status": status, "reason": reason, "worktree": spec.get("worktree"), "tool": spec.get("tool"), **(_git(spec["worktree"]) if spec and Path(spec["worktree"]).is_dir() else {})}
        self.save()

    def domain(self):
        stage = self.settings["domain_stage"]
        out = self.dir / "domain.json"
        command = self.command(stage, {"patch": self.layout.patch(), "shape": self.grid["shape"], "output": str(out)}, "domain", self.settings["domain_compute"], "layout")
        spec = self.stages[stage]
        self.manifest["stages"]["layout"] = {"status": "running"}
        started = perf_counter()
        with (self.dir / "logs" / "layout.log").open("w", encoding="utf-8") as log:
            code = subprocess.call(command, cwd=spec["worktree"], stdout=log, stderr=subprocess.STDOUT)
        self.manifest["stages"]["layout"] = {"status": "ran" if code == 0 and out.is_file() else "failed", "runtime_s": round(perf_counter() - started, 1), "worktree": spec["worktree"], **_git(spec["worktree"])}
        self.save()
        return _load(out)

    def optimization(self, domain):
        crash = [case["name"] for case in domain["load_cases"] if case["name"].startswith("crash_")]
        variant = self.request["name"]
        out = self.dir / "optimization"
        options = self.layout.overrides.get("optimizer", {})
        frequency = 1 / (2 * LAYOUT_RULES["neural"]["half_wavelength_per_width"] * self.layout.durability["minimum_width_mm"])
        neural = {"max_frequency_per_mm": options.get("max_frequency_per_mm", frequency), **self.grid["neural"], **({"max_runtime_s": options["max_runtime_s"]} if options.get("max_runtime_s") else {})}
        overrides = {"root": str(out), "viewer_root": "", "shape": self.grid["shape"], "fine_shape": self.grid["fine_shape"], "neural": neural,
                     "pad": {**LAYOUT_RULES["pad"], "support_half_mm": 5.0, "bore_margin_mm": 0.5}, "variants": [{"name": variant, "neural": {"volume_fraction": options.get("volume_fraction", self.layout.durability["volume_fraction"])}}]}
        overrides.update({key: value for key, value in (("prop_discs", {"mode": options.get("prop_discs")}), ("modal", {"f1_min_hz": options.get("f1_min_hz")}), ("method", options.get("method"))) if value not in (None, {"mode": None}, {"f1_min_hz": None})})
        if options.get("arm_tip_stiffness_min_n_per_mm"):
            overrides["stiffness"] = {"min_n_per_mm": options["arm_tip_stiffness_min_n_per_mm"], **({"calibration": options["stiffness_calibration"]} if options.get("stiffness_calibration") else {})}
        if self.layout.style["hoops"] or self.layout.free_camera:
            overrides["hoop"] = self.layout.hoop() if self.layout.style["hoops"] else None
        source = (Path(self.stages["optimization"]["worktree"]) / self.stages["optimization"]["tool"]).read_text(encoding="utf-8")
        if '"crash_directions"' in source:
            overrides["crash_directions"] = list(self.layout.style["crash_directions"])
        else:
            self.manifest["notes"].append("optimization stage has no multi-direction crash cases; freestyle crash set reduced to " + ", ".join(crash + (["crash_hoop"] if self.layout.style["hoops"] else [])))
        if self.layout.style["torsion"] and not any("torsion" in case["name"] for case in domain["load_cases"]):
            self.manifest["notes"].append("torsion load case pending: no stage load-case builder provides it yet")
        result = out / variant
        command = self.command("optimization", {"patch": self.layout.patch(crash), "argv": self.stages["optimization"]["argv"], "overrides": overrides}, "cli")
        self.execute("optimization", command, self.stages["optimization"]["worktree"], [result / "geometry.stl", result / "density_fine.npz"])
        return result

    def reconstruction(self, density):
        out = self.dir / "reconstruction"
        study = {"pad": {**LAYOUT_RULES["pad"], "support_half_mm": 5.0, "bore_margin_mm": 0.5}, **({"hoop": self.layout.hoop() if self.layout.style["hoops"] else None} if self.layout.style["hoops"] or self.layout.free_camera else {})}
        overrides = {"source": str(density), "output": str(out), "fine_shape": self.grid["fine_shape"], "study": study, **self.grid["reconstruction"]}
        command = self.command("reconstruction", {"patch": self.layout.patch(), "argv": self.stages["reconstruction"]["argv"], "overrides": overrides}, "cli")
        self.execute("reconstruction", command, self.stages["reconstruction"]["worktree"], [out / "geometry.stl"])
        return out / "geometry.stl"

    def geometry(self, mesh, domain):
        out = self.dir / "geometry" / "walls.json"
        regions = [region for region in domain["regions"] if region.get("role") == "preserve" and "motor" in region["name"]]
        command = self.command("geometry", {"patch": {}, "function": self.stages["geometry"]["argv"][0], "kwargs": {"mesh": str(mesh), "regions": regions}, "result": str(out)}, "call")
        self.execute("geometry", command, self.stages["geometry"]["worktree"], [out])
        return _load(out)

    def evaluation_spec(self, domain, mesh):
        regions = {region["name"]: region for region in domain["regions"]}
        placements = domain["components"]
        cases = {case["name"]: case for case in domain["comparison_load_cases"]}
        battery = domain["point_masses"][0]
        motor = self.layout.components["motor"]
        motors = {name: list(placements["motor_" + name]["position_mm"]) for name in MOTORS}
        patterns = [{"name": f"motor_{name}", "center_mm": xyz[:2], "z_mm": xyz[2] - 1.0, "radius_mm": motor["mount_pitch_mm"] / 2, "count": 4, "hole_diameter_mm": [1.6, 4.0],
                     "screw_diameter_mm": motor["screw_diameter_mm"], "tool_direction": [0, 0, -1]} for name, xyz in motors.items()]
        patterns.append(stack_pattern(regions))
        components = [{"type": entry.get("prototype", name.split("_")[0]), "name": name, "center_mm": entry["center_of_mass_mm"]} for name, entry in placements.items() if entry["mass_g"] > 0]
        rails = [region["max_mm"][2] for name, region in regions.items() if name.startswith("battery_rail_")]
        deck = {**battery["attachment_region"], "min_mm": [*battery["attachment_region"]["min_mm"][:2], min(rails) - self.settings["deck_band_mm"]], "max_mm": [*battery["attachment_region"]["max_mm"][:2], max(rails) + 0.01]} if rails else battery["attachment_region"]
        band, pad = self.settings["fixture_band_mm"], LAYOUT_RULES["pad"]["top_mm"] - LAYOUT_RULES["pad"]["thickness_mm"]
        layer = lambda box, low, high: {**box, "min_mm": [*box["min_mm"][:2], low], "max_mm": [*box["max_mm"][:2], high]}
        tip = cases["arm_tip"]["loads"][0]["region"]
        selectors = {"center_fixtures": [layer(box, box["min_mm"][2], box["max_mm"][2] + band) for box in cases["arm_tip"]["fixed_regions"]],
                     "motor_fixtures": [layer(box, pad - 0.01, pad + band) for box in cases["modes"]["fixed_regions"]], "arm_tip": layer(tip, tip["min_mm"][2] - band, tip["max_mm"][2]), "arm_motor": "front_left",
                     "camera": cases["camera_side"]["loads"][0]["region"], "deck": deck, "battery_center_mm": battery["position_mm"], "battery_mass_g": battery["mass_g"]}
        out = self.dir / "evaluation"
        return {"name": self.dir.name, "stl": str(mesh), "output": str(out), "prop_diameter_mm": self.layout.components["prop"]["diameter_mm"], "motors": motors, "mount_patterns": patterns,
                "components": components, "connectors": [{"name": name, "position_mm": placements[name]["center_of_mass_mm"], "direction": [0, 0, 1]} for name in ("xt30", "balancer") if name + "_contact" in regions],
                "keep_outs": [region for region in domain["regions"] if region["role"] == "forbidden"], "selectors": selectors,
                "loads": {"safety_factor": self.layout.durability["safety_factor"]}, "python": self.stages["evaluation"]["python"], "compute": self.settings["compute"], **self.grid["evaluation"]}

    def evaluation(self, domain, mesh):
        spec = self.stages["evaluation"]
        request = self.dir / "requests" / "evaluation_frame.json"
        _save(request, self.evaluation_spec(domain, mesh))
        out = self.dir / "evaluation" / "evaluation.json"
        self.execute("evaluation", [spec["python"], str(Path(spec["worktree"]) / spec["tool"]), *spec["argv"], str(request)], spec["worktree"], [out])
        return _load(out)

    def renders(self, mesh):
        out = self.dir / "renders"
        command = self.command("renders", {"patch": {}, "function": self.stages["renders"]["argv"][0], "kwargs": {"mesh": str(mesh), "out": str(out), "views": self.settings["views"]}}, "call")
        self.execute("renders", command, self.stages["renders"]["worktree"], [out / f"{view}.png" for view in self.settings["views"]])

    def datasheet(self):
        spec = self.stages["datasheet"]
        command = [sys.executable, self.settings["compute"], spec["compute"], "--cwd", str(ROOT), "--", spec["python"], str(ROOT / spec["tool"]), *spec["argv"], str(self.dir / "manifest.json")]
        self.execute("datasheet", command, ROOT, [self.dir / "datasheet.md"])

    def run(self):
        (self.dir / "logs").mkdir(exist_ok=True)
        domain = self.domain()
        if domain is None:
            self.manifest["status"] = "failed"
            self.save()
            return self.manifest
        mesh = None
        if not self.available("optimization"):
            self.mark("optimization", "pending", "stage tool not found")
        else:
            result = self.optimization(domain)
            if self.manifest["stages"]["optimization"]["status"] == "ran":
                mesh = result / "geometry.stl"
                self.manifest["raw_geometry"] = str(mesh)
                if not self.request["reconstruction"]:
                    self.mark("reconstruction", "skipped", "reconstruction disabled in the request; raw threshold body evaluated")
                elif not self.available("reconstruction"):
                    self.mark("reconstruction", "pending", "stage tool not found")
                else:
                    rebuilt = self.reconstruction(result / "density_fine.npz")
                    mesh = rebuilt if self.manifest["stages"]["reconstruction"]["status"] == "ran" else mesh
        if mesh is None:
            for stage in ("reconstruction", "geometry", "evaluation", "renders"):
                self.manifest["stages"].setdefault(stage, {"status": "skipped", "reason": "no geometry from the optimization stage"})
        else:
            shutil.copy2(mesh, self.dir / "frame.stl")
            self.manifest["frame_stl"] = {"path": str(self.dir / "frame.stl"), "source": str(mesh), "sha256": _digest(self.dir / "frame.stl")}
            walls = self.geometry(self.dir / "frame.stl", domain) if self.available("geometry") else self.mark("geometry", "pending", "stage tool not found")
            self.manifest["wall_rule_passed"] = None if walls is None else bool(walls.get("passed"))
            evaluation = None
            if self.available("evaluation"):
                evaluation = self.evaluation(domain, self.dir / "frame.stl")
            else:
                self.mark("evaluation", "pending", "evaluation tool not found; wired to run as soon as tools/evaluate_frame.py exists")
            if evaluation:
                self.manifest["evaluation_gates"] = {"fea_solved": "fea_solved" not in evaluation.get("assessment", {}).get("missed", []), "missed": evaluation.get("assessment", {}).get("missed")}
            _save(self.dir / "evaluation.json", evaluation or {"status": self.manifest["stages"]["evaluation"]["status"], "reason": self.manifest["stages"]["evaluation"].get("reason", "see logs/evaluation.log")})
            if self.available("renders"):
                self.renders(self.dir / "frame.stl")
            else:
                self.mark("renders", "pending", "renderer not found")
        self.datasheet()
        statuses = [entry["status"] for entry in self.manifest["stages"].values()]
        self.manifest["status"] = "complete" if all(status == "ran" for status in statuses) else "partial"
        self.manifest["print_ready"] = bool(self.manifest.get("wall_rule_passed")) and not (self.manifest.get("evaluation_gates") or {}).get("missed")
        self.manifest["outputs"] = {path.name: _digest(path) for path in sorted(self.dir.iterdir()) if path.is_file()}
        self.save()
        return self.manifest

def _voxels(mesh, pitch):
    grid = mesh.voxelized(pitch).fill()
    matrix = grid.matrix.astype(bool)
    origin = np.asarray(grid.translation) - pitch / 2
    return matrix, origin

def measure(mesh, layout, pitch):
    from scipy.ndimage import binary_fill_holes, distance_transform_edt, label
    from skimage.morphology import skeletonize
    solid, origin = _voxels(mesh, pitch)
    axes = [origin[i] + (np.arange(solid.shape[i]) + 0.5) * pitch for i in range(3)]
    z, y = axes[2], axes[1]
    top = solid.any(axis=2)
    filled = binary_fill_holes(top)
    holes, count = label(filled & ~top)
    sizes = np.bincount(holes.ravel())[1:] * pitch ** 2
    skeleton = skeletonize(solid)
    widths = 2 * distance_transform_edt(solid)[skeleton] * pitch
    middle = (np.abs(y) <= 10)[None, :, None]
    lower = bool((solid & middle & (z <= 5)[None, None, :]).any())
    upper = bool((solid & middle & (z >= 20)[None, None, :]).any())
    heights = {f"{level:g}": float(solid[:, :, np.argmin(np.abs(z - level))].sum() * pitch ** 2) for level in (1.5, 9.5, 17.5, 27.5) if z.min() <= level <= z.max()}
    hoop = layout.get("hoop")
    hoop_cells = 0
    if hoop:
        points = np.argwhere(solid)
        coordinates = origin + (points + 0.5) * pitch
        for sign in (-1, 1):
            path = np.asarray([[sign * hoop["x_mm"], yy, zz] for yy, zz in hoop["path_yz_mm"]])
            near = np.min(np.linalg.norm(coordinates[:, None, :] - path[None, :, :], axis=2), axis=1) <= hoop["radius_mm"] + 1.0
            hoop_cells += int(near.sum())
    normals, areas = mesh.face_normals, mesh.area_faces
    centroids = mesh.triangles_center
    overhang = float(areas[(normals[:, 2] < -cos(radians(45))) & (centroids[:, 2] > mesh.bounds[0][2] + 0.2)].sum())
    return {"bounds_mm": mesh.bounds.tolist(), "volume_mm3": float(mesh.volume), "watertight": bool(mesh.is_watertight), "bodies": len(mesh.split(only_watertight=False)),
            "top_fraction": float(top.sum() / top.size), "openings_100": int((sizes >= 100).sum()), "openings_20_100": int(((sizes >= 20) & (sizes < 100)).sum()),
            "largest_opening_mm2": float(sizes.max()) if count else 0.0, "strut_p10_p50_p90_mm": np.percentile(widths, [10, 50, 90]).tolist() if widths.size else None,
            "strut_below_2mm": float(np.mean(widths < 2.0)) if widths.size else None, "lower_chord": lower, "upper_chord": upper, "area_by_height_mm2": heights,
            "hoop_volume_mm3": hoop_cells * pitch ** 3, "overhang_mm2": overhang, "overhang_fraction": overhang / float(mesh.area)}

def _number(value, digits=1):
    return "–" if value is None else f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")

def _agility_row(agility, mesh, density):
    if not agility:
        return ("12 Dynamik", "kein Layoutmodell in layout.json", "–")
    frame = {"mass_g": mesh.volume * density / 1000, "center_mm": np.asarray(mesh.center_mass).tolist(), "inertia_g_mm2": (np.asarray(mesh.moment_inertia) * density / 1000).tolist()} if mesh is not None else None
    result = LayoutModel({**agility["setup"], "frame": frame}).evaluate(agility["layout"]) if frame else agility
    alpha, cg, inertia, v = result["alpha_rad_s2"], result["center_of_mass_mm"], np.diag(result["inertia_g_mm2"]) * 1e-3, result["layout"]
    names = {"roll": "Rollen", "pitch": "Nicken", "yaw": "Gieren"}
    text = (f"α Rollen/Nicken/Gieren {' / '.join(_number(alpha[axis], 0) for axis in AXES)} rad/s² (Minimum {names[result['limiting_axis']]}); Schwerpunkt ({'; '.join(_number(value, 1) for value in cg)}) mm, "
            f"{_number(result['cg_above_rotor_plane_mm'], 1)} mm über der Rotorebene; Ixx/Iyy/Izz {' / '.join(_number(value, 1) for value in inertia)} kg·mm², Gesamtmasse {_number(result['mass_g'])} g; "
            f"Layout: Akku y {_number(v['battery_y_mm'])} mm, Akku-Unterkante z {_number(v['deck_top_mm'])} mm, Kamera y {_number(v['camera_y_mm'])} mm, Kamera-Bodenabstand {_number(v['camera_bottom_clearance_mm'])} mm, Stack-Abstand {_number(v['aio_standoff_mm'])} mm"
            + (f"; verletzt: {', '.join(result['violated'])}" if result["violated"] else ""))
    source = ("Layoutmodell (frame_run.LayoutModel): Rahmenanteil gemessen aus frame.stl, " if frame else "Layoutmodell: Rahmenanteil " + agility["setup"]["frame"].get("source", "aus der Konfiguration") + ", ") + "Komponenten als Quader/Scheiben, τ aus ΔT_max und festem Hebel, " + agility.get("source", "")
    return ("12 Dynamik", text, source)

def _posts(post):
    if not post:
        return ""
    kind = {"heat_set": "Heat-Set-Einsatz", "self_tapping": "selbstschneidend"}[post["fastening"]]
    longer = f", für die Bohrung {_number(post['height_mm'] - post['standoff_mm'])} mm unter die Grundebene verlängert" if post["height_mm"] > post["standoff_mm"] + 1e-9 else ""
    return (f" auf 4 Pfosten Ø {_number(post['diameter_mm'])} mm (Whoop-Prinzip, Oberkante = Tüllensitz, Höhe = Stack-Abstand {_number(post['standoff_mm'])} mm{longer}), M2 von oben, {kind} (Sackbohrung Ø {_number(post['bore_diameter_mm'])} × {_number(post['bore_depth_mm'])} mm), "
            "Werkzeugfreiraum oben, keine Muttern, kein Zugang von unten, Anbindung durch den Optimierer")

def datasheet(manifest_path):
    import trimesh
    run_dir = Path(manifest_path).parent
    manifest, layout = _load(manifest_path), _load(run_dir / "layout.json")
    evaluation = _load(run_dir / "evaluation.json") or {}
    stl = run_dir / "frame.stl"
    request, frame, material = layout["request"], layout["frame"], layout["material"]
    motors = layout["motors_mm"]
    lateral, longitudinal = 2 * abs(motors["front_right"][0]), 2 * abs(motors["front_right"][1])
    mesh = trimesh.load_mesh(stl, process=True) if stl.is_file() else None
    m = measure(mesh, layout, RUN_SETTINGS["datasheet_voxel_mm"]) if mesh is not None else None
    mass = None if m is None else m["volume_mm3"] * material["density_g_cm3"] / 1000
    missing = "nicht gemessen (keine Geometrie)"
    def measured(text):
        return missing if m is None else text
    widths = (m or {}).get("strut_p10_p50_p90_mm")
    chords = None if m is None else ("Raumfachwerk: Untergurt und Obergurt in der Rumpfmitte" if m["lower_chord"] and m["upper_chord"] else "nur Obergurt in der Rumpfmitte, kein Untergurt" if m["upper_chord"] else "nur Untergurt in der Rumpfmitte" if m["lower_chord"] else "keine Gurte in der Rumpfmitte (|y| <= 10 mm)")
    rows = [
        ("1 Klasse", f"{request['layout']['x_type']}, Motoren bei (±{lateral / 2:.1f}; ±{longitudinal / 2:.1f}) → {lateral:.0f} × {longitudinal:.0f} mm, Diagonale {frame['wheelbase_mm']:.1f} mm, Props {request['prop_size_in']:g}\" ({layout['checks']['prop_plane_mm']:.1f} mm Propebene), Motor {request['components']['motor']}", "Layoutregeln (layout.json)"),
        ("2 Tragwerkstyp", measured(chords), "gemessen (Voxel, Rumpfmitte z ≤ 5 / z ≥ 20 mm)"),
        ("3 Bauhöhe des Tragwerks", measured(f"z {m['bounds_mm'][0][2]:.1f}–{m['bounds_mm'][1][2]:.1f} mm; Fläche je Höhe: " + ", ".join(f"z {k} → {v:.0f} mm²" for k, v in m["area_by_height_mm2"].items())) if m else missing, "gemessen (Voxelschnitte)"),
        ("4 Streben", measured(f"Breite p10/p50/p90 = {widths[0]:.1f} / {widths[1]:.1f} / {widths[2]:.1f} mm, {100 * m['strut_below_2mm']:.0f} % unter 2 mm") if widths else missing, f"gemessen (2 × EDT auf 3D-Skelett, Voxel {RUN_SETTINGS['datasheet_voxel_mm']} mm)"),
        ("5 Offenheit", measured(f"Material in Draufsicht {100 * m['top_fraction']:.0f} % der Bounding-Box, {m['openings_100']} Öffnungen ≥ 100 mm² (größte {m['largest_opening_mm2']:.0f} mm²), {m['openings_20_100']} mit 20–100 mm²") if m else missing, "gemessen (Draufsicht-Projektion)"),
        ("6 Arme", f"{request['layout']['x_type']} mit Armwinkel {np.degrees(np.arctan(frame['lateral_longitudinal_ratio'])):.1f}° zur Längsachse, Motorpads Oberkante z = {frame['arm_height_mm']:.2f} mm (Vorgabe), Armform aus der Optimierung", "Layoutregeln; Form siehe Renders"),
        ("7 Kamera und Schutz", f"Kamera {request['components']['camera']} bei y = {layout['camera']['y_mm']:.1f} mm, Neigung {layout['camera']['tilt_deg']:g}°; Bügel " + (measured(f"vorgegeben, Material im Bügelkanal {m['hoop_volume_mm3']:.0f} mm³") if layout["hoop"] else "nicht vorgegeben"), "Layoutregeln + gemessen"),
        ("8 Akku und Stack", f"Akku {request['components']['battery']} {request['layout']['battery_mount']}, Deckoberkante z = {layout['battery']['deck_top_mm']:.1f} mm, Akkuauflage {'frei (ohne Vorgabegeometrie)' if layout['battery'].get('support') == 'free' else 'Schienen'}, Stack {request['components']['aio']} zentriert bei z = {layout['stack']['position_mm'][2]:.1f} mm{_posts(layout['stack'].get('posts'))}, Antennen {request['components']['antennas']}, XT30 und Balancer ohne vorgeschriebenen Sitz (Gummiband, frei platziert)", "Layoutregeln"),
        ("9 Masse", measured(f"{_number(mass)} g = {m['volume_mm3'] / 1000:.2f} cm³ × {material['density_g_cm3']} g/cm³ ({request['material']}); Komponenten {layout['checks']['component_mass_g']:.1f} g") if m else missing, "gemessen (STL)"),
        ("10 Druckbarkeit", measured(f"{'wasserdicht' if m['watertight'] else 'NICHT wasserdicht'}, {m['bodies']} Körper, Überhang > 45°: {m['overhang_mm2']:.0f} mm² = {100 * m['overhang_fraction']:.0f} % der Oberfläche; Wandregel: {({True: 'bestanden', False: 'nicht bestanden', None: 'nicht geprüft'})[manifest.get('wall_rule_passed')]}; Düse {request['print']['nozzle_mm']} mm, Schicht {request['print']['layer_mm']} mm") if m else missing, "gemessen (STL, Normalen) + Wandregel (int)"),
    ]
    rows.append(_agility_row(layout.get("agility"), mesh, material["density_g_cm3"]))
    trials = [trial for trial in ((evaluation.get("fea") or {}).get("fea_surface") or {}).get("trials", []) if trial.get("passed")]
    limit = IMPLICIT_CONFIG["surface_deviation_mm"]
    surface = (f"Ersatzoberfläche weicht {_number(trials[-1]['deviation_mm'], 2)} mm vom STL ab (Grenze {_number(limit, 2)} mm): " + ("innerhalb" if trials[-1]["deviation_mm"] <= limit else "WARNUNG, überschritten; betrifft nur das FEA-Rechenmodell, nicht die Druckgeometrie (Nutzerentscheid: Warnung statt Abbruch)")) if trials else "FEA direkt auf dem STL, keine Ersatzoberfläche" if (evaluation.get("fea") or {}).get("status") else "keine FEA"
    rows.append(("11 FEA-Oberfläche", surface, "evaluation.json fea.fea_surface"))
    stand = (((evaluation.get("geometry") or {}).get("mass") or {}).get("standing") or {}).get("stability")
    verdict = lambda passed: {True: "erfüllt", False: "NICHT erfüllt", None: "ohne Vorgabe"}[passed]
    rows.append(("12 Standsicherheit", (f"Standfläche {_number(stand['support_area_mm2'], 0)} mm² (konvexe Hülle der tiefsten Materialstellen, ≤ {_number(stand['contact_tolerance_mm'], 1)} mm über dem tiefsten Punkt), Schwerpunkt {_number(stand['reserve_mm'])} mm innerhalb "
                 f"(Vorgabe ≥ {_number(stand['reserve_min_mm'], 0)} mm: {verdict(stand['reserve_passed'])}), Props {_number(stand['prop_clearance_mm'])} mm über Boden (Vorgabe ≥ {_number(stand['prop_clearance_min_mm'])} mm: {verdict(stand['prop_clearance_passed'])})")
                 if stand and stand.get("reserve_mm") is not None else "nicht gemessen" if not stand else "keine Standfläche (weniger als drei nicht kollineare Kontaktpunkte)", "evaluation.json geometry.mass.standing.stability"))
    renders = [path for path in ("iso", "top", "side", "front") if (run_dir / "renders" / f"{path}.png").is_file() or manifest["stages"].get("renders", {}).get("status") == "running"]
    stages = ", ".join(f"{name}: {entry['status']}" for name, entry in manifest["stages"].items() if name != "datasheet")
    line = evaluation.get("line") or f"Neun-Kriterien-Bewertung: {evaluation.get('status', 'pending')} ({evaluation.get('reason', 'siehe evaluation.json')})"
    text = [f"# {manifest['name']} – Datenblatt", "", f"Automatisch erzeugt von run.py aus `config.json`. Geometrie: `frame.stl` (SHA256 {(manifest.get('frame_stl') or {}).get('sha256', '–')[:16]}). Stufen: {stages}.", "",
            "| Feld | Wert | Quelle |", "|---|---|---|", *[f"| {name} | {value} | {source} |" for name, value, source in rows], "", "## Neun Kriterien", "", line, ""]
    text += ["## Ansichten", "", *[f"![{view}](renders/{view}.png)" for view in renders], ""]
    if manifest.get("notes"):
        text += ["## Hinweise", "", *[f"- {note}" for note in manifest["notes"]], ""]
    (run_dir / "datasheet.md").write_text("\n".join(text), encoding="utf-8")
    return 0
