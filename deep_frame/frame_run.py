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
                               FRAME_REQUEST_KINDS, LAYOUT_OVERRIDES, LAYOUT_RULES, LIBRARY_FIELDS, MATERIALS, MOUNTING_TYPES, RUN_GRIDS, RUN_SETTINGS, STAGES, STYLES, component_spec,
                               configure, prop_spec)
from deep_frame.frame import camera_mount_z, motor_positions

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

class FrameLayout:
    def __init__(self, request):
        self.request = validate_request(request)
        self.parts = {role: library_part(role, name) for role, name in self.request["components"].items()}
        self.style, self.durability = STYLES[self.request["style"]], DURABILITY[self.request["durability"]]
        self.material = MATERIALS[self.request["material"]]
        self.overrides = self.request["overrides"]
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
                 **{name: component_spec(library_part("connector", part)) for name, part in connectors.items()}, "prop": prop_spec(self.request["prop_size_in"])}
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
        aio, camera, xt30 = c["aio15"], c["camera"], c["xt30"]
        antenna = self.parts["antennas"]
        antenna_y = self._override("antennas", "y_mm", -(rules["envelope"]["size_mm"][1] / 2 - rules["antennas"]["eyelet_radius_mm"] - rules["antennas"]["envelope_margin_mm"]))
        if antenna_y >= 0:
            raise ValueError("antennas.y_mm must be negative: the antennas sit behind the stack")
        connector_y = -antenna_y - rules["antennas"]["eyelet_radius_mm"] - rules["antennas"]["connector_clearance_mm"] - xt30["length_mm"] / 2
        mount = self.request["layout"]["battery_mount"]
        frame = {"wheelbase_mm": wheelbase, "lateral_longitudinal_ratio": tan(radians(angle)), "arm_height_mm": rules["pad"]["top_mm"],
                 "camera_y_mm": self._override("camera", "y_mm", aio["length_mm"] / 2 + rules["camera"]["stack_gap_mm"] + camera["length_mm"] / 2),
                 "antenna_y_mm": -antenna_y, "antenna_bore_mm": antenna["dimensions_mm"]["bore"], "antenna_holder_height_mm": antenna["dimensions_mm"]["holder_height"],
                 "connector_y_mm": connector_y, "aio_standoff_mm": self._override("stack", "standoff_mm", FRAME_DEFAULTS["aio_standoff_mm"]),
                 "camera_screw_diameter_mm": camera.get("screw_clearance_mm", FRAME_DEFAULTS["camera_screw_diameter_mm"])}
        if mount == "top":
            frame["deck_top_mm"] = self._override("battery", "deck_top_mm", rules["battery_mounts"]["top"]["deck_top_mm"])
        else:
            need = c["battery"]["height_mm"] + rules["battery_mounts"]["bottom"]["gap_mm"]
            floor = rules["envelope"]["origin_mm"][2]
            if floor > -need:
                raise ValueError(f"battery_mount bottom needs {need:.1f} mm below the base plate, but the design envelope starts at z = {floor:.1f} mm; choose battery_mount top")
        return frame

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
        if f["antenna_y_mm"] + f["antenna_bore_mm"] / 2 > -low[1] or f["connector_y_mm"] - c["xt30"]["length_mm"] / 2 < c["aio15"]["length_mm"] / 2:
            raise ValueError("Antenna eyelet and connectors do not fit behind the stack inside the envelope")
        stack_top = f["base_thickness_mm"] + f["aio_standoff_mm"] + c["aio15"]["stack_height_mm"] + c["aio15"]["elrs_antenna_clearance_mm"]
        deck_bottom = f["deck_top_mm"] - f["deck_thickness_mm"]
        if self.request["layout"]["battery_mount"] == "top" and min(deck_bottom - camera_top, deck_bottom - stack_top) < rules["camera"]["top_clearance_mm"] - 1e-9:
            raise ValueError(f"Battery deck bottom z = {deck_bottom:.1f} mm leaves less than {rules['camera']['top_clearance_mm']} mm above camera top {camera_top:.1f} mm or stack top {stack_top:.1f} mm")
        prop_plane = f["arm_height_mm"] + c["motor"]["height_mm"] + f["prop_motor_gap_mm"] + c["prop"]["thickness_mm"]
        corners = np.asarray([[sx * c["battery"]["width_mm"] / 2, sy * c["battery"]["length_mm"] / 2] for sx in (-1, 1) for sy in (-1, 1)])
        plan = min(float(np.min(np.hypot(*(np.clip(m, corners.min(0), corners.max(0)) - m)))) for m in motors) - c["prop"]["diameter_mm"] / 2
        if plan < rules["battery_prop_clearance_mm"] and f["deck_top_mm"] < prop_plane + rules["battery_prop_clearance_mm"]:
            raise ValueError("Battery overlaps the prop disks in plan and does not sit above the prop plane")
        masses = [(c["motor"]["mass_g"] + c["prop"]["mass_g"], *m) for m in motors] + [(c["aio15"]["mass_g"], 0.0, 0.0), (c["camera"]["mass_g"], 0.0, f["camera_y_mm"]), (c["battery"]["mass_g"], 0.0, 0.0)]
        total = sum(m for m, _, _ in masses)
        cg = [sum(m * x for m, x, _ in masses) / total, sum(m * y for m, _, y in masses) / total]
        if hypot(*cg) > rules["cg_tolerance_mm"]:
            raise ValueError(f"Component centre of gravity {cg} is more than {rules['cg_tolerance_mm']} mm from the battery and stack axis")
        if "angle_deg" in self.overrides.get("antennas", {}):
            self.notes.append("antennas.angle_deg is recorded but the stage domain has no antenna angle parameter")
        return {"motor_pad_envelope": True, "camera_fit": True, "camera_top_mm": camera_top, "stack_top_mm": stack_top, "battery_plan_clearance_mm": plan, "prop_plane_mm": prop_plane,
                "component_cg_mm": cg, "component_mass_g": total}

    def summary(self):
        f = {**FRAME_DEFAULTS, **self.frame}
        return {"request": self.request, "frame": self.frame, "motors_mm": self.motors(), "hoop": self.hoop() if self.style["hoops"] else None, "checks": self.checks,
                "battery": {"mount": self.request["layout"]["battery_mount"], "deck_top_mm": f["deck_top_mm"], "position_mm": [0.0, 0.0, f["deck_top_mm"]]},
                "stack": {"position_mm": [0.0, 0.0, f["base_thickness_mm"] + f["aio_standoff_mm"]]}, "camera": {"y_mm": f["camera_y_mm"], "tilt_deg": self.components["camera"]["tilt_deg"]},
                "antennas": {"y_mm": -f["antenna_y_mm"], "angle_deg": self._override("antennas", "angle_deg", LAYOUT_RULES["antennas"]["angle_deg"])},
                "style": self.style, "durability": self.durability, "material": self.material, "parts": {role: part["source"] for role, part in self.parts.items()}, "notes": self.notes}

    def patch(self, crash_cases=()):
        material = {key: self.material[key] for key in ("density_g_cm3", "young_modulus_mpa", "poisson_ratio")}
        weights = {name: self.durability["crash_weight"] for name in crash_cases}
        patch = {"FRAME_DEFAULTS": self.frame, "COMPONENT_DEFAULTS": self.components, "FEA_CONFIG": {"material": material},
                 "TOPOLOGY_CONFIG": {"manufacturing": {"nozzle_width_mm": self.request["print"]["nozzle_mm"], "minimum_feature_mm": self.durability["minimum_width_mm"]}},
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

    def command(self, stage, request, action, compute=None):
        spec = self.stages[stage]
        path = self.dir / "requests" / f"{stage}.json"
        _save(path, {"root": spec["worktree"], "tool": spec["tool"], "action": action, **request})
        inner = [spec["python"], str(ROOT / "run.py"), "stage", str(path)]
        kind = compute if compute is not None else spec["compute"]
        return [sys.executable, self.settings["compute"], kind, "--cwd", spec["worktree"], "--", *inner] if kind else inner

    def execute(self, stage, command, cwd, outputs):
        spec = self.stages[stage]
        entry = {"status": "running", "worktree": spec["worktree"], "tool": spec["tool"], "compute": spec["compute"], **_git(spec["worktree"]), "command": subprocess.list2cmdline(command)}
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
        command = self.command(stage, {"patch": self.layout.patch(), "shape": self.grid["shape"], "output": str(out)}, "domain", self.settings["domain_compute"])
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
        neural = {"max_frequency_per_mm": LAYOUT_RULES["neural"]["max_frequency_per_mm"] * LAYOUT_RULES["neural"]["reference_width_mm"] / self.layout.durability["minimum_width_mm"], **self.grid["neural"]}
        overrides = {"root": str(out), "viewer_root": "", "shape": self.grid["shape"], "fine_shape": self.grid["fine_shape"], "neural": neural,
                     "pad": {**LAYOUT_RULES["pad"], "support_half_mm": 5.0, "bore_margin_mm": 0.5}, "variants": [{"name": variant, "neural": {"volume_fraction": self.layout.durability["volume_fraction"]}}]}
        if self.layout.style["hoops"]:
            overrides["hoop"] = self.layout.hoop()
        source = (Path(self.stages["optimization"]["worktree"]) / self.stages["optimization"]["tool"]).read_text(encoding="utf-8")
        if '"crash_directions"' in source:
            overrides["crash_directions"] = list(self.layout.style["crash_directions"])
        else:
            self.manifest["notes"].append("optimization stage has no multi-direction crash cases; freestyle crash set reduced to " + ", ".join(crash + (["crash_hoop"] if self.layout.style["hoops"] else [])))
        if self.layout.style["torsion"] and not any("torsion" in case["name"] for case in domain["load_cases"]):
            self.manifest["notes"].append("torsion load case pending: no stage load-case builder provides it yet")
        result = out / variant
        command = self.command("optimization", {"patch": self.layout.patch(crash), "argv": ["run"], "overrides": overrides}, "cli")
        self.execute("optimization", command, self.stages["optimization"]["worktree"], [result / "geometry.stl", result / "density_fine.npz"])
        return result

    def reconstruction(self, density):
        out = self.dir / "reconstruction"
        overrides = {"source": str(density), "output": str(out), "fine_shape": self.grid["fine_shape"], **self.grid["reconstruction"]}
        command = self.command("reconstruction", {"patch": self.layout.patch(), "argv": ["build"], "overrides": overrides}, "cli")
        self.execute("reconstruction", command, self.stages["reconstruction"]["worktree"], [out / "geometry.stl"])
        return out / "geometry.stl"

    def geometry(self, mesh, domain):
        out = self.dir / "geometry" / "walls.json"
        regions = [region for region in domain["regions"] if region.get("role") == "preserve" and "motor" in region["name"]]
        command = self.command("geometry", {"patch": {}, "function": "wall_rule", "kwargs": {"mesh": str(mesh), "regions": regions}, "result": str(out)}, "call")
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
        aio = np.asarray([regions[f"aio_screw_{index}"]["center_mm"] for index in range(4)])
        center = aio.mean(axis=0)
        patterns.append({"name": "stack", "center_mm": center[:2].tolist(), "z_mm": 1.5, "radius_mm": float(np.hypot(*(aio[0] - center)[:2])), "count": 4, "hole_diameter_mm": [1.6, 4.0],
                         "screw_diameter_mm": self.layout.components["aio15"]["screw_diameter_mm"], "tool_direction": [0, 0, -1]})
        components = [{"type": entry.get("prototype", name.split("_")[0]), "name": name, "center_mm": entry["center_of_mass_mm"]} for name, entry in placements.items() if entry["mass_g"] > 0]
        selectors = {"center_fixtures": cases["arm_tip"]["fixed_regions"], "motor_fixtures": cases["modes"]["fixed_regions"], "arm_tip": cases["arm_tip"]["loads"][0]["region"], "arm_motor": "front_left",
                     "camera": cases["camera_side"]["loads"][0]["region"], "deck": battery["attachment_region"], "battery_center_mm": battery["position_mm"], "battery_mass_g": battery["mass_g"]}
        out = self.dir / "evaluation"
        return {"name": self.dir.name, "stl": str(mesh), "output": str(out), "prop_diameter_mm": self.layout.components["prop"]["diameter_mm"], "motors": motors, "mount_patterns": patterns,
                "components": components, "connectors": [{"name": name, "position_mm": placements[name]["center_of_mass_mm"], "direction": [0, 0, 1]} for name in ("xt30", "balancer") if name in placements],
                "keep_outs": [region for region in domain["regions"] if region["role"] == "forbidden"], "selectors": selectors,
                "wall_zones": [region for region in domain["regions"] if region["role"] == "preserve" and "motor" in region["name"]],
                "loads": {"safety_factor": self.layout.durability["safety_factor"]}, "python": self.stages["evaluation"]["python"], "compute": self.settings["compute"], **self.grid["evaluation"]}

    def evaluation(self, domain, mesh):
        spec = self.stages["evaluation"]
        request = self.dir / "requests" / "evaluation_frame.json"
        _save(request, self.evaluation_spec(domain, mesh))
        out = self.dir / "evaluation" / "evaluation.json"
        self.execute("evaluation", [spec["python"], str(Path(spec["worktree"]) / spec["tool"]), "run", str(request)], spec["worktree"], [out])
        return _load(out)

    def renders(self, mesh):
        out = self.dir / "renders"
        command = self.command("renders", {"patch": {}, "function": "render_views", "kwargs": {"mesh": str(mesh), "out": str(out), "views": self.settings["views"]}}, "call")
        self.execute("renders", command, self.stages["renders"]["worktree"], [out / f"{view}.png" for view in self.settings["views"]])

    def datasheet(self):
        spec = self.stages["datasheet"]
        command = [sys.executable, self.settings["compute"], spec["compute"], "--cwd", str(ROOT), "--", spec["python"], str(ROOT / "run.py"), "datasheet", str(self.dir / "manifest.json")]
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
                if not self.available("reconstruction"):
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
            _save(self.dir / "evaluation.json", evaluation or {"status": self.manifest["stages"]["evaluation"]["status"], "reason": self.manifest["stages"]["evaluation"].get("reason", "see logs/evaluation.log")})
            if self.available("renders"):
                self.renders(self.dir / "frame.stl")
            else:
                self.mark("renders", "pending", "renderer not found")
        self.datasheet()
        statuses = [entry["status"] for entry in self.manifest["stages"].values()]
        self.manifest["status"] = "complete" if all(status == "ran" for status in statuses) else "partial"
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

def datasheet(manifest_path):
    import trimesh
    run_dir = Path(manifest_path).parent
    manifest, layout = _load(manifest_path), _load(run_dir / "layout.json")
    evaluation = _load(run_dir / "evaluation.json") or {}
    stl = run_dir / "frame.stl"
    request, frame, material = layout["request"], layout["frame"], layout["material"]
    motors = layout["motors_mm"]
    lateral, longitudinal = 2 * abs(motors["front_right"][0]), 2 * abs(motors["front_right"][1])
    m = measure(trimesh.load_mesh(stl, process=True), layout, RUN_SETTINGS["datasheet_voxel_mm"]) if stl.is_file() else None
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
        ("8 Akku und Stack", f"Akku {request['components']['battery']} {request['layout']['battery_mount']}, Deckoberkante z = {layout['battery']['deck_top_mm']:.1f} mm, Stack {request['components']['aio']} zentriert bei z = {layout['stack']['position_mm'][2]:.1f} mm, Antennen {request['components']['antennas']} bei y = {layout['antennas']['y_mm']:.1f} mm", "Layoutregeln"),
        ("9 Masse", measured(f"{_number(mass)} g = {m['volume_mm3'] / 1000:.2f} cm³ × {material['density_g_cm3']} g/cm³ ({request['material']}); Komponenten {layout['checks']['component_mass_g']:.1f} g") if m else missing, "gemessen (STL)"),
        ("10 Druckbarkeit", measured(f"{'wasserdicht' if m['watertight'] else 'NICHT wasserdicht'}, {m['bodies']} Körper, Überhang > 45°: {m['overhang_mm2']:.0f} mm² = {100 * m['overhang_fraction']:.0f} % der Oberfläche; Wandregel: {({True: 'bestanden', False: 'nicht bestanden', None: 'nicht geprüft'})[manifest.get('wall_rule_passed')]}; Düse {request['print']['nozzle_mm']} mm, Schicht {request['print']['layer_mm']} mm") if m else missing, "gemessen (STL, Normalen) + Wandregel (int)"),
    ]
    renders = [path for path in ("iso", "top", "side", "front") if (run_dir / "renders" / f"{path}.png").is_file() or manifest["stages"].get("renders", {}).get("status") == "running"]
    stages = ", ".join(f"{name}: {entry['status']}" for name, entry in manifest["stages"].items())
    line = evaluation.get("line") or f"Neun-Kriterien-Bewertung: {evaluation.get('status', 'pending')} ({evaluation.get('reason', 'siehe evaluation.json')})"
    text = [f"# {manifest['name']} – Datenblatt", "", f"Automatisch erzeugt von run.py aus `config.json`. Geometrie: `frame.stl` (SHA256 {(manifest.get('frame_stl') or {}).get('sha256', '–')[:16]}). Stufen: {stages}.", "",
            "| Feld | Wert | Quelle |", "|---|---|---|", *[f"| {name} | {value} | {source} |" for name, value, source in rows], "", "## Neun Kriterien", "", line, ""]
    text += ["## Ansichten", "", *[f"![{view}](renders/{view}.png)" for view in renders], ""]
    if manifest.get("notes"):
        text += ["## Hinweise", "", *[f"- {note}" for note in manifest["notes"]], ""]
    (run_dir / "datasheet.md").write_text("\n".join(text), encoding="utf-8")
    return 0
