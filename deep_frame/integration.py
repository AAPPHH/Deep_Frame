import hashlib
import json
import math
import os
import re
import subprocess
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from deep_frame.components import build_components
from deep_frame.fea import evaluate, resolve_solver
from deep_frame.frame import assembly_placements, motor_positions
from deep_frame.geometry import build_geometry


def _box(minimum, maximum):
    return {"kind": "box", "min_mm": list(minimum), "max_mm": list(maximum)}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def solver_identity(settings):
    executable = resolve_solver(settings)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run([executable, "-v"], capture_output=True, text=True, timeout=10, creationflags=flags)
    match = re.search(r"Version\s+([\d.]+)", completed.stdout + completed.stderr)
    if not match:
        raise RuntimeError("CalculiX version could not be identified")
    return {
        "solver_path": executable,
        "calculix_version": match.group(1),
        "calculix_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
        "gmsh_version": version("gmsh"),
        "build123d_version": version("build123d"),
        "ocp_version": version("cadquery-ocp-novtk"),
    }


def prepare_frame_case(parameters, fea_config=None, integration_config=None):
    fea = deepcopy(fea_config if fea_config is not None else parameters["fea"])
    integration = deepcopy(integration_config if integration_config is not None else parameters["integration"])
    frame = parameters["frame"]
    if parameters["material"]["density_g_cm3"] != fea["material"]["density_g_cm3"]:
        raise ValueError("Geometry and FEA material densities must agree")
    for key, value in integration.items():
        if isinstance(value, (int, float)) and (not math.isfinite(value) or (value <= 0 and key != "battery_attachment_y_mm")):
            raise ValueError(f"Integration setting {key} must be finite and positive")
    for key in ("central_fixture_fraction", "camera_upper_height_fraction", "camera_length_fraction"):
        if integration[key] > 1:
            raise ValueError(f"Integration setting {key} must not exceed one")
    tolerance = integration["selection_tolerance_mm"]
    motors = motor_positions(parameters)
    pad_half = frame["motor_pad_radius_mm"] + integration["motor_pad_margin_mm"]
    motor_fixtures = [_box((x - pad_half, y - pad_half, -tolerance), (x + pad_half, y + pad_half, tolerance)) for x, y in motors.values()]
    fraction = integration["central_fixture_fraction"]
    central_fixture = _box(
        (-frame["body_width_mm"] * fraction / 2, -frame["body_length_mm"] * fraction / 2, -tolerance),
        (frame["body_width_mm"] * fraction / 2, frame["body_length_mm"] * fraction / 2, tolerance),
    )
    x, y = motors[integration["arm_tip_motor"]]
    arm_region = _box((x - pad_half, y - pad_half, frame["arm_height_mm"] - tolerance), (x + pad_half, y + pad_half, frame["arm_height_mm"] + tolerance))
    deck_half_width = frame["deck_width_mm"] / 2 + integration["battery_attachment_margin_mm"]
    band_half_width = integration["battery_attachment_band_width_mm"] / 2
    band_y = integration["battery_attachment_y_mm"]
    deck_region = _box(
        (-deck_half_width, band_y - band_half_width, frame["deck_top_mm"] - tolerance),
        (deck_half_width, band_y + band_half_width, frame["deck_top_mm"] + tolerance),
    )
    camera_half_width = (parameters["components"]["camera"]["width_mm"] + 2 * frame["camera_side_clearance_mm"] + 2 * frame["minimum_wall_mm"]) / 2
    camera_half_length = frame["cage_length_mm"] * integration["camera_length_fraction"] / 2
    camera_region = _box(
        (-camera_half_width - tolerance, frame["camera_y_mm"] - camera_half_length, frame["cage_height_mm"] * (1 - integration["camera_upper_height_fraction"])),
        (camera_half_width + tolerance, frame["camera_y_mm"] + camera_half_length, frame["cage_height_mm"] + tolerance),
    )
    battery = build_components(parameters, assembly_placements(parameters))["battery"]
    mass = {
        "name": "battery",
        "mass_g": battery["mass_g"],
        "position_mm": list(battery["center_of_mass_mm"]),
        "attachment_region": deck_region,
    }
    impact = mass["mass_g"] / 1000 * integration["standard_gravity_m_s2"] * integration["battery_impact_g_factor"]
    load_cases = [
        {"name": "arm_tip", "analysis": "static", "fixed_regions": [central_fixture], "loads": [{"region": arm_region, "force_n": [0.0, 0.0, -integration["arm_tip_force_n"]]}]},
        {"name": "battery_impact", "analysis": "static", "fixed_regions": motor_fixtures, "loads": [{"region": deck_region, "force_n": [0.0, 0.0, -impact]}]},
        {"name": "camera_side", "analysis": "static", "fixed_regions": motor_fixtures, "loads": [{"region": camera_region, "force_n": [integration["camera_side_force_n"], 0.0, 0.0]}]},
        {"name": "modes", "analysis": "modal", "fixed_regions": motor_fixtures},
    ]
    fea["settings"]["stiffness_load_case"] = "arm_tip"
    result = {
        "material": fea["material"],
        "point_masses": [mass],
        "load_cases": load_cases,
        "settings": fea["settings"],
        "integration": integration,
        "mass_scope": "frame plus battery point mass; motors, props, camera and AIO excluded from FEA",
        "fixture_model": "arm: central base bottom rim fixed; other cases: four motor pad bottom surfaces fixed",
        "coupling_model": "battery COM rigidly coupled to a narrow transverse band of the deck rails; local patch deformation suppressed; no battery rotational inertia",
        "impact_model": "equivalent static force, not a transient impact or crash strength prediction",
    }
    return json.loads(json.dumps(result, allow_nan=False))


class FrameEvaluator:
    def __init__(self, reference_parameters, fea_config=None, integration_config=None):
        self.fea_config = deepcopy(fea_config if fea_config is not None else reference_parameters["fea"])
        self.integration_config = deepcopy(integration_config if integration_config is not None else reference_parameters["integration"])
        self.toolchain = solver_identity(self.fea_config["settings"])
        self.fea_config["settings"]["solver_path"] = self.toolchain["solver_path"]
        reference_case = prepare_frame_case(reference_parameters, self.fea_config, self.integration_config)
        physical_settings = {key: value for key, value in self.fea_config["settings"].items() if key not in ("work_dir", "solver_path")}
        directory = Path(__file__).resolve().parent
        sources = {name: hashlib.sha256((directory / name).read_text(encoding="utf-8").encode()).hexdigest() for name in ("integration.py", "fea.py", "fea_mesh.py", "frame.py", "components.py", "geometry.py", "checks.py")}
        self.evaluation_contract = {
            "model_version": self.integration_config["model_version"],
            "material": self.fea_config["material"],
            "integration": self.integration_config,
            "settings": physical_settings,
            "reference_point_masses": reference_case["point_masses"],
            "reference_load_cases": reference_case["load_cases"],
            "toolchain": {key: value for key, value in self.toolchain.items() if key != "solver_path"},
            "source_sha256": sources,
        }
        self.evaluation_id = self.integration_config["model_version"] + ":" + _digest(self.evaluation_contract)

    def __call__(self, parameters):
        model = prepare_frame_case(parameters, self.fea_config, self.integration_config)
        solid = build_geometry(parameters)
        result = evaluate(solid, model["material"], model["point_masses"], model["load_cases"], model["settings"])
        result["evaluation_id"] = self.evaluation_id
        result["model_inputs"] = model
        result["evaluation_contract"] = deepcopy(self.evaluation_contract)
        return json.loads(json.dumps(result, allow_nan=False))
