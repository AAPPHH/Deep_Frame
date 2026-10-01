import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from build123d import Align, Box, Pos, Solid

from deep_frame.config import CONFIG, FEA_CONFIG
from deep_frame.fea import FrameEvaluator, _run, evaluate, prepare_frame_case
from deep_frame.frame import assembly_placements, build_components, intersection_shape, motor_positions, reference_parameters
from tests.test_frame import frame

def beam_inputs(directory):
    solid = Box(80, 8, 4, align=(Align.MIN, Align.CENTER, Align.CENTER)).solid()
    fixed = {"kind": "box", "min_mm": [-0.01, -5, -3], "max_mm": [0.01, 5, 3]}
    tip = {"kind": "box", "min_mm": [79.99, -5, -3], "max_mm": [80.01, 5, 3]}
    settings = deepcopy(FEA_CONFIG["settings"])
    settings.update(mesh_size_mm=2.0, work_dir=str(directory), stiffness_load_case="tip", num_modes=4)
    cases = [
        {"name": "tip", "analysis": "static", "fixed_regions": [fixed], "loads": [{"region": tip, "force_n": [0, 0, -1]}]},
        {"name": "modes", "analysis": "modal", "fixed_regions": [fixed]},
    ]
    return solid, deepcopy(FEA_CONFIG["material"]), cases, settings, tip

@pytest.fixture(scope="module")
def beam_results(tmp_path_factory):
    solid, material, cases, settings, tip = beam_inputs(tmp_path_factory.mktemp("beam_fea"))
    original = deepcopy((material, cases, settings))
    unloaded = evaluate(solid, material, [], cases, settings)
    masses = [{"name": "tip_mass", "mass_g": 1.5, "position_mm": [80, 0, 0], "attachment_region": tip}]
    loaded = evaluate(solid, material, masses, cases, settings)
    assert (material, cases, settings) == original
    assert unloaded["status"] == "ok", unloaded["diagnostics"]
    assert loaded["status"] == "ok", loaded["diagnostics"]
    return unloaded, loaded

def analytical_beam(tip_mass_kg=0):
    length, width, height = 0.080, 0.008, 0.004
    young, density = 4430e6, 1090.0
    area = width * height
    inertia = width * height ** 3 / 12
    mass = density * area * length
    ratio = tip_mass_kg / mass
    def characteristic(beta):
        return 1 + math.cos(beta) * math.cosh(beta) + ratio * beta * (math.cos(beta) * math.sinh(beta) - math.sin(beta) * math.cosh(beta))
    lower, upper = 0.01, 2.0
    for _ in range(60):
        middle = (lower + upper) / 2
        if characteristic(middle) > 0:
            lower = middle
        else:
            upper = middle
    frequency = middle ** 2 / (2 * math.pi) * math.sqrt(young * inertia / (density * area * length ** 4))
    return {"deflection_mm": length ** 3 / (3 * young * inertia) * 1000, "frequency_hz": frequency, "mass_g": mass * 1000}

def test_real_gmsh_calculix_beam_matches_analytical_solution(beam_results):
    measured, _ = beam_results
    expected = analytical_beam()
    displacement = measured["load_cases"]["tip"]["loads"][0]["directional_displacement_mm"]
    assert displacement == pytest.approx(expected["deflection_mm"], rel=0.03)
    assert measured["eigenfrequencies_hz"][0] == pytest.approx(expected["frequency_hz"], rel=0.03)
    assert measured["mass_g"] == pytest.approx(expected["mass_g"], rel=1e-10)
    assert measured["stiffness_n_per_mm"] == pytest.approx(1 / displacement, rel=1e-10)
    assert measured["max_von_mises_mpa"] > 0
    assert measured["mesh"]["minimum_jacobian_mm3"] > 0
    assert measured["mesh"]["element_type"] == "C3D10"
    assert measured["load_cases"]["modes"]["discarded_rigid_modes"] == 0
    json.dumps(measured, allow_nan=False)

def test_real_point_mass_matches_tip_mass_beam_and_mass_sum(beam_results):
    bare, loaded = beam_results
    expected = analytical_beam(0.0015)
    assert loaded["eigenfrequencies_hz"][0] == pytest.approx(expected["frequency_hz"], rel=0.03)
    assert loaded["eigenfrequencies_hz"][0] < bare["eigenfrequencies_hz"][0]
    assert loaded["mass_g"] == pytest.approx(bare["mass_g"] + 1.5)
    assert loaded["stiffness_n_per_mm"] == pytest.approx(bare["stiffness_n_per_mm"], rel=0.005)
    assert loaded["point_mass_coupling"][0]["position_mm"] == [80, 0, 0]

def test_missing_solver_returns_failure_not_zero_measurements(tmp_path):
    solid, material, cases, settings, _ = beam_inputs(tmp_path)
    settings["solver_path"] = str(tmp_path / "missing_calculix_executable")
    result = evaluate(solid, material, [], cases, settings)
    assert result["status"] == "failed"
    assert result["mass_g"] is None
    assert result["max_displacement_mm"] is None
    assert result["eigenfrequencies_hz"] == []
    assert "not found" in result["diagnostics"][0]
    json.dumps(result, allow_nan=False)

def test_empty_load_selector_rejected_before_solver(tmp_path):
    solid, material, cases, settings, _ = beam_inputs(tmp_path)
    cases[0]["loads"][0]["region"] = {"kind": "box", "min_mm": [900, 0, 0], "max_mm": [901, 1, 1]}
    result = evaluate(solid, material, [], cases, settings)
    assert result["status"] == "invalid"
    assert "Empty node selector" in result["diagnostics"][0]
    assert not list(tmp_path.rglob("case_*.inp"))

def test_unrestrained_fixture_is_rejected(tmp_path):
    solid, material, cases, settings, _ = beam_inputs(tmp_path)
    cases[1]["fixed_regions"] = []
    result = evaluate(solid, material, [], cases, settings)
    assert result["status"] == "invalid"
    assert "Fixture" in result["diagnostics"][0]

def test_invalid_density_returns_json_safe_invalid_result(tmp_path):
    solid, material, cases, settings, _ = beam_inputs(tmp_path)
    material["density_g_cm3"] = float("nan")
    result = evaluate(solid, material, [], cases, settings)
    assert result["status"] == "invalid"
    assert result["mass_g"] is None
    json.dumps(result, allow_nan=False)

def test_rigid_point_mass_patch_must_not_intersect_fixture(tmp_path):
    solid, material, cases, settings, _ = beam_inputs(tmp_path)
    masses = [{"name": "bad_mass", "mass_g": 1.5, "position_mm": [0, 0, 0], "attachment_region": cases[0]["fixed_regions"][0]}]
    result = evaluate(solid, material, masses, cases, settings)
    assert result["status"] == "invalid"
    assert "overlaps a fixed fixture" in result["diagnostics"][0]

@pytest.mark.parametrize("options", [{"mesh_second_order_linear": True}, {"mesh_high_order_optimize": 2}])
def test_opt_in_mesh_options_retain_real_beam_physics(tmp_path, options):
    solid, material, cases, settings, _ = beam_inputs(tmp_path)
    settings.update(options)
    result = evaluate(solid, material, [], cases, settings)
    expected = analytical_beam()
    assert result["status"] == "ok", result["diagnostics"]
    assert result["max_displacement_mm"] == pytest.approx(expected["deflection_mm"], rel=0.03)
    assert result["eigenfrequencies_hz"][0] == pytest.approx(expected["frequency_hz"], rel=0.03)
    assert result["mesh"]["minimum_jacobian_mm3"] > 0
    assert result["mesh"]["second_order_linear"] == options.get("mesh_second_order_linear", False)
    assert result["mesh"]["high_order_optimize"] == options.get("mesh_high_order_optimize", 0)

@pytest.mark.parametrize("backend", ["SPOOLES", "PASTIX"])
def test_explicit_backend_preserves_real_point_mass_beam_physics(tmp_path, backend, monkeypatch):
    solid, material, cases, settings, tip = beam_inputs(tmp_path)
    settings.update(linear_solver=backend, threads=1)
    masses = [{"name": "tip_mass", "mass_g": 1.5, "position_mm": [80, 0, 0], "attachment_region": tip}]
    monkeypatch.setenv("CCX_NPROC_STIFFNESS", "7")
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "7")
    result = evaluate(solid, material, masses, cases, settings)
    expected = analytical_beam(0.0015)
    assert result["status"] == "ok", result["diagnostics"]
    assert result["linear_solver"] == backend
    assert result["solver_threads"] == 1
    assert result["eigenfrequencies_hz"][0] == pytest.approx(expected["frequency_hz"], rel=0.03)
    assert result["stiffness_n_per_mm"] == pytest.approx(1 / expected["deflection_mm"], rel=0.03)
    assert result["mass_g"] == pytest.approx(expected["mass_g"] + 1.5)
    for name, keyword in (("tip", "STATIC"), ("modes", "FREQUENCY")):
        text = Path(result["artifacts"][name]["input"]).read_text()
        assert f"*{keyword},SOLVER={backend}\n" in text
        log = Path(result["artifacts"][name]["log"]).read_text()
        assert "Using up to 7 cpu(s)" not in log
        assert "Using up to 1 cpu(s)" in log
    json.dumps(result, allow_nan=False)

@pytest.mark.parametrize("backend", ["AUTO", "spooles", "PARDISO", "SPOOLES\n*STEP", float("nan"), float("inf")])
def test_invalid_backend_is_rejected_before_mesh_or_solver(tmp_path, backend):
    directory = tmp_path / "must_not_be_created"
    solid, material, cases, settings, _ = beam_inputs(directory)
    settings["linear_solver"] = backend
    result = evaluate(solid, material, [], cases, settings)
    assert result["status"] == "invalid"
    assert result["artifacts"] == {}
    assert "linear_solver" in result["diagnostics"][0]
    assert not directory.exists()
    json.dumps(result, allow_nan=False)

@pytest.mark.parametrize("threads", [0, -1, True, 1.5, float("nan"), "1"])
def test_invalid_thread_count_is_json_safe_before_mesh_or_solver(tmp_path, threads):
    directory = tmp_path / "must_not_be_created"
    solid, material, cases, settings, _ = beam_inputs(directory)
    settings.update(linear_solver="SPOOLES", threads=threads)
    result = evaluate(solid, material, [], cases, settings)
    assert result["status"] == "invalid"
    assert result["artifacts"] == {}
    assert "threads" in result["diagnostics"][0]
    assert not directory.exists()
    json.dumps(result, allow_nan=False)

def test_requested_thread_count_overrides_inherited_native_thread_environment(tmp_path, monkeypatch):
    keys = ("OMP_NUM_THREADS", "CCX_NPROC_RESULTS", "CCX_NPROC_STIFFNESS",
            "CCX_NPROC_EQUATION_SOLVER", "NUMBER_OF_CPUS")
    for key in keys:
        monkeypatch.setenv(key, "7")
    script = "import os,json; print(json.dumps({key:os.environ[key] for key in " + repr(keys) + "}))"
    content = _run([sys.executable, "-c", script], tmp_path, 10, 1, tmp_path / "environment.log")
    assert json.loads(content) == {key: "1" for key in keys}

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
    monkeypatch.setattr("deep_frame.fea.solver_identity", lambda settings: deepcopy(identity))
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
    monkeypatch.setattr("deep_frame.fea.build_geometry", geometry)
    monkeypatch.setattr("deep_frame.fea.evaluate", solver)
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
