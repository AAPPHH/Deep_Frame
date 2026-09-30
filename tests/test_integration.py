import json
from copy import deepcopy

import pytest
from build123d import Align, Box, Pos, Solid

from deep_frame.checks import intersection_shape
from deep_frame.components import build_components
from deep_frame.config import CONFIG
from deep_frame.frame import assembly_placements, motor_positions
from deep_frame.geometry import build_geometry, reference_parameters
from deep_frame.integration import FrameEvaluator, prepare_frame_case


@pytest.fixture(scope="module")
def frame():
    return build_geometry(reference_parameters())


def selector_shape(selector):
    minimum, maximum = selector["min_mm"], selector["max_mm"]
    return Pos(*minimum) * Box(*(high - low for low, high in zip(minimum, maximum)), align=(Align.MIN, Align.MIN, Align.MIN))


def test_reference_case_uses_actual_battery_center_and_physical_force():
    parameters = reference_parameters()
    original = deepcopy(parameters)
    model = prepare_frame_case(parameters)
    battery = build_components(parameters, assembly_placements(parameters))["battery"]
    assert model["point_masses"][0]["mass_g"] == 37.0
    assert model["point_masses"][0]["position_mm"] == pytest.approx(battery["center_of_mass_mm"])
    assert model["point_masses"][0]["position_mm"] == pytest.approx([0.0, 0.0, 34.5])
    cases = {case["name"]: case for case in model["load_cases"]}
    assert cases["battery_impact"]["loads"][0]["force_n"] == pytest.approx([0, 0, -3.6284605])
    assert cases["arm_tip"]["loads"][0]["force_n"] == [0.0, 0.0, -1.0]
    assert cases["camera_side"]["loads"][0]["force_n"] == [5.0, 0.0, 0.0]
    assert len(cases["modes"]["fixed_regions"]) == 4
    assert model["settings"]["stiffness_load_case"] == "arm_tip"
    assert model["material"]["density_g_cm3"] == parameters["material"]["density_g_cm3"]
    assert parameters == original
    json.dumps(model, allow_nan=False)


def test_every_fixture_load_and_mass_selector_meets_real_frame(frame):
    model = prepare_frame_case(reference_parameters())
    regions = [model["point_masses"][0]["attachment_region"]]
    for case in model["load_cases"]:
        regions.extend(case["fixed_regions"])
        regions.extend(load["region"] for load in case.get("loads", []))
    for region in regions:
        intersection = intersection_shape(frame, selector_shape(region))
        assert intersection is not None and intersection.volume > 0
    patch = model["point_masses"][0]["attachment_region"]
    assert patch["max_mm"][1] - patch["min_mm"][1] == 4.0
    assert patch["min_mm"][0] == -20.1
    assert patch["max_mm"][0] == 20.1
    for case in model["load_cases"]:
        for fixture in case["fixed_regions"]:
            assert fixture["max_mm"][2] < patch["min_mm"][2]


def test_selectors_follow_changed_candidate_and_battery():
    parameters = reference_parameters()
    parameters["frame"].update(arm_height_mm=4.2, wheelbase_mm=140.0, deck_top_mm=30.0)
    parameters["components"]["battery"].update(mass_g=40.0, height_mm=12.0)
    model = prepare_frame_case(parameters)
    cases = {case["name"]: case for case in model["load_cases"]}
    tip = cases["arm_tip"]["loads"][0]["region"]
    x, y = motor_positions(parameters)["front_left"]
    assert (tip["min_mm"][0] + tip["max_mm"][0]) / 2 == pytest.approx(x)
    assert (tip["min_mm"][1] + tip["max_mm"][1]) / 2 == pytest.approx(y)
    assert (tip["min_mm"][2] + tip["max_mm"][2]) / 2 == pytest.approx(4.2)
    assert model["point_masses"][0]["position_mm"] == pytest.approx([0, 0, 36])
    assert cases["battery_impact"]["loads"][0]["force_n"][2] == pytest.approx(-0.040 * 9.80665 * 10)


@pytest.fixture
def toolchain(monkeypatch):
    identity = {"solver_path": "C:/test/ccx.exe", "calculix_version": "2.22", "calculix_sha256": "test-sha", "gmsh_version": "4.15.2", "build123d_version": "0.13.0", "ocp_version": "8.0.1.0.0"}
    monkeypatch.setattr("deep_frame.integration.solver_identity", lambda settings: deepcopy(identity))
    return identity


def test_evaluator_callback_builds_candidate_and_passes_dict_model(monkeypatch, toolchain, frame):
    parameters = reference_parameters()
    calls = []
    builds = []

    def geometry(candidate):
        builds.append(deepcopy(candidate))
        return frame

    def solver(solid, material, masses, cases, settings):
        assert isinstance(solid, Solid)
        calls.append(deepcopy((material, masses, cases, settings)))
        return {"status": "failed", "mass_g": None, "eigenfrequencies_hz": [], "max_displacement_mm": None, "max_von_mises_mpa": None, "stiffness_n_per_mm": None, "load_cases": {}, "diagnostics": ["callback test"], "artifacts": {}}

    monkeypatch.setattr("deep_frame.integration.build_geometry", geometry)
    monkeypatch.setattr("deep_frame.integration.evaluate", solver)
    evaluator = FrameEvaluator(parameters)
    parameters["frame"]["arm_height_mm"] = 4.2
    result = evaluator(parameters)
    assert builds[0]["frame"]["arm_height_mm"] == 4.2
    assert calls[0][1][0]["mass_g"] == 37.0
    assert calls[0][3]["solver_path"] == toolchain["solver_path"]
    assert result["status"] == "failed"
    assert result["diagnostics"] == ["callback test"]
    assert result["evaluation_id"] == evaluator.evaluation_id
    assert result["model_inputs"]["load_cases"][0]["loads"][0]["region"]["max_mm"][2] == pytest.approx(4.21)
    json.dumps(result, allow_nan=False)


def test_evaluation_contract_binds_physics_and_toolchain_but_not_workdir(toolchain):
    parameters = reference_parameters()
    first = FrameEvaluator(parameters)
    changed = deepcopy(parameters["fea"])
    changed["settings"]["work_dir"] = "another/worktree/exports/fea"
    changed["settings"]["solver_path"] = "another/ccx.exe"
    assert FrameEvaluator(parameters, changed).evaluation_id == first.evaluation_id
    changed["settings"]["mesh_size_mm"] = 2.5
    assert FrameEvaluator(parameters, changed).evaluation_id != first.evaluation_id
    changed = deepcopy(parameters["integration"])
    changed["battery_impact_g_factor"] = 11.0
    assert FrameEvaluator(parameters, integration_config=changed).evaluation_id != first.evaluation_id
    toolchain["calculix_version"] = "2.23"
    assert FrameEvaluator(parameters).evaluation_id != first.evaluation_id


def test_central_config_exposes_all_subsystems_without_shapes_or_paths():
    parameters = reference_parameters()
    assert parameters["material"] == parameters["fea"]["material"]
    assert set(parameters["optimization_search_space"]) == {"frame.arm_height_mm", "frame.arm_width_mm"}
    assert CONFIG["material"] is CONFIG["fea"]["material"]
    assert parameters["optimization"]["initial_candidates"] == [
        {"frame.arm_height_mm": 4.2, "frame.arm_width_mm": 6.5},
        {"frame.arm_height_mm": 4.0, "frame.arm_width_mm": 6.7},
    ]
    json.dumps(parameters, allow_nan=False)


def test_density_mismatch_cannot_silently_change_mass_scope():
    parameters = reference_parameters()
    changed = deepcopy(parameters["fea"])
    changed["material"]["density_g_cm3"] = 1.20
    with pytest.raises(ValueError, match="densities must agree"):
        prepare_frame_case(parameters, changed)
