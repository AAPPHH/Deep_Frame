import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import trimesh
from build123d import Align, Box, Pos, Solid

from deep_frame.config import CONFIG, FEA_CONFIG, IMPLICIT_CONFIG
from deep_frame.fea import MESH_ATTEMPTS, FrameEvaluator, _mesh_settings, _prepare_surface, _read_mesh, _run, _select, _topology, _volume_mesh, evaluate, prepare_frame_case
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

def analytical_beam(tip_mass_kg=0, area=0.008 * 0.004, inertia=0.008 * 0.004 ** 3 / 12, length=0.080):
    young, density = 4430e6, 1090.0
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

def scripted_run(monkeypatch, codes):
    calls = []
    def fake(command, stdout, **kwargs):
        calls.append(command)
        stdout.write(f"attempt {len(calls)}\n")
        return subprocess.CompletedProcess(command, codes[len(calls) - 1])
    monkeypatch.setattr("deep_frame.fea.subprocess.run", fake)
    return calls

def test_native_crash_is_retried_once_and_crash_log_kept(tmp_path, monkeypatch):
    calls = scripted_run(monkeypatch, [0xC0000374, 0])
    content = _run(["ccx"], tmp_path, 10, 1, tmp_path / "solve.log")
    assert len(calls) == 2 and content == "attempt 2\n"
    assert (tmp_path / "solve.crash.log").read_text(encoding="utf-8") == "attempt 1\n"

def test_ordinary_failure_is_not_retried(tmp_path, monkeypatch):
    calls = scripted_run(monkeypatch, [1, 0])
    with pytest.raises(RuntimeError, match=r"failed \(1\)"):
        _run(["ccx"], tmp_path, 10, 1, tmp_path / "solve.log")
    assert len(calls) == 1 and not (tmp_path / "solve.crash.log").exists()

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
    assert cases["arm_tip"]["loads"][0]["force_n"] == [0.0, 0.0, -3.6]
    assert cases["crash_front"]["loads"][0]["force_n"][1] == pytest.approx(-0.125 * 9.80665 * 25)
    assert cases["thrust_all"]["loads"][0]["force_n"] == [0.0, 0.0, 3.6] and len(cases["thrust_all"]["loads"]) == 4
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

def box_beam_mesh():
    return trimesh.creation.box(extents=(80, 8, 4), transform=trimesh.transformations.translation_matrix((40, 0, 0)))

def cylinder_beam_mesh():
    return trimesh.creation.cylinder(radius=4.0, height=100.0, sections=64, transform=trimesh.transformations.translation_matrix((50, 0, 0)) @ trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0)))

def mesh_section(mesh):
    length = mesh.bounds[1, 0] - mesh.bounds[0, 0]
    inertia = mesh.moment_inertia
    return {"area": mesh.volume / length * 1e-6, "inertia": (inertia[0, 0] + inertia[1, 1] - inertia[2, 2]) / 2 / length * 1e-12, "length": length * 1e-3}

def cylinder_cases(cases):
    cases = deepcopy(cases)
    cases[0]["fixed_regions"][0].update(min_mm=[-0.01, -5, -5], max_mm=[0.01, 5, 5])
    cases[0]["loads"][0]["region"].update(min_mm=[99.99, -5, -5], max_mm=[100.01, 5, 5])
    cases[1]["fixed_regions"] = deepcopy(cases[0]["fixed_regions"])
    return cases

@pytest.fixture(scope="module")
def mesh_beam_results(tmp_path_factory):
    _, material, cases, settings, tip = beam_inputs(tmp_path_factory.mktemp("mesh_beam_fea"))
    settings["mesh_second_order_linear"] = True
    masses = [{"name": "tip_mass", "mass_g": 1.5, "position_mm": [80, 0, 0], "attachment_region": tip}]
    results = {"box": evaluate(box_beam_mesh(), material, [], cases, settings), "box_mass": evaluate(box_beam_mesh(), material, masses, cases, settings),
               "cylinder": evaluate(cylinder_beam_mesh(), material, [], cylinder_cases(cases), settings)}
    for result in results.values():
        assert result["status"] == "ok", (result["diagnostics"], result.get("mesh"))
    return results

@pytest.mark.parametrize("name, mesh", [("box", box_beam_mesh), ("cylinder", cylinder_beam_mesh)])
def test_closed_triangle_mesh_beam_matches_analytical_solution(mesh_beam_results, name, mesh):
    measured, solid = mesh_beam_results[name], mesh()
    expected = analytical_beam(**mesh_section(solid))
    displacement = measured["load_cases"]["tip"]["loads"][0]["directional_displacement_mm"]
    assert displacement == pytest.approx(expected["deflection_mm"], rel=0.03)
    assert measured["eigenfrequencies_hz"][0] == pytest.approx(expected["frequency_hz"], rel=0.03)
    assert measured["mass_g"] == pytest.approx(solid.volume * 1.09e-3, rel=1e-10)
    assert measured["stiffness_n_per_mm"] == pytest.approx(1 / displacement, rel=1e-10)
    assert measured["load_cases"]["modes"]["discarded_rigid_modes"] == 0
    assert measured["load_cases"]["tip"]["fixed_node_count"] >= 3 and measured["load_cases"]["tip"]["loads"][0]["node_count"] >= 3
    report = measured["mesh"]
    assert report["element_type"] == "C3D10" and report["minimum_jacobian_mm3"] > 0
    attempts = report["attempts"]
    assert attempts[-1]["status"] == "ok" and report["attempt"] == attempts[-1]["name"] and all(attempt["diagnostic"] for attempt in attempts[:-1])
    assert report["minimum_sicn"] >= 0.01 and report["elements_below_sicn"] == 0
    assert report["boundary_node_deviation_mm"] <= 0.05 and report["boundary_fidelity"]["maximum_sampled_deviation_mm"] <= IMPLICIT_CONFIG["surface_deviation_mm"]
    assert abs(report["boundary_fidelity"]["relative_volume_change"]) <= IMPLICIT_CONFIG["relative_volume_change"] and abs(report["tet_volume_relative_change"]) <= IMPLICIT_CONFIG["relative_volume_change"]
    assert report["tet_volume_relative_change"] == pytest.approx(report["boundary_fidelity"]["relative_volume_change"], abs=1e-9) and report["input_volume_mm3"] == pytest.approx(solid.volume, rel=1e-12)
    assert measured["model_frame_mass_g"] == pytest.approx(report["tet_volume_mm3"] * 1.09e-3, rel=1e-12)
    if report["attempt"].startswith(("remesh", "refine")):
        assert report["prepared_surface"]["topology_passed"] and report["prepared_surface"]["self_intersections_passed"] and report["prepared_surface"]["folded_edges"] == 0
    json.dumps(measured, allow_nan=False)

def test_mesh_point_mass_matches_tip_mass_beam(mesh_beam_results):
    bare, loaded = mesh_beam_results["box"], mesh_beam_results["box_mass"]
    assert loaded["eigenfrequencies_hz"][0] == pytest.approx(analytical_beam(0.0015)["frequency_hz"], rel=0.03)
    assert loaded["mass_g"] == pytest.approx(bare["mass_g"] + 1.5)
    assert loaded["stiffness_n_per_mm"] == pytest.approx(bare["stiffness_n_per_mm"], rel=0.005)

def test_tetrahedral_mesher_failure_is_failure_with_recorded_attempts(tmp_path):
    _, material, cases, settings, _ = beam_inputs(tmp_path)
    settings.update(mesh_minimum_sicn=0.99, fea_remesh_targets_mm=[2.0, 1.5])
    result = evaluate(box_beam_mesh(), material, [], cases, settings)
    assert result["status"] == "failed"
    assert "All tetrahedral meshing attempts failed" in result["diagnostics"][0]
    attempts = result["mesh"]["attempts"]
    assert [(attempt["name"], attempt["target_mm"]) for attempt in attempts] == [("remesh_hxt", 2.0), ("remesh_delaunay", 2.0), ("remesh_hxt", 1.5), ("remesh_delaunay", 1.5), ("refine_hxt", 2.0)] + [(name, None) for name in MESH_ATTEMPTS[3:]]
    assert [attempt["status"] for attempt in attempts] == ["failed"] * (len(attempts) - 1) + ["skipped"]
    assert all(attempt["runtime_s"] >= 0 and not attempt.get("over_budget") for attempt in attempts)
    assert all(attempt["diagnostic"] for attempt in attempts)
    assert any("SICN" in attempt["diagnostic"] for attempt in attempts)
    assert result["mass_g"] is None and result["eigenfrequencies_hz"] == []
    assert not list(tmp_path.rglob("mesh.inp")) and not list(tmp_path.rglob("case_*.inp"))
    json.dumps(result, allow_nan=False)

def test_tetrahedral_body_that_loses_input_volume_is_rejected(tmp_path):
    _, material, cases, settings, _ = beam_inputs(tmp_path)
    settings.update(tet_attempts=["remesh_hxt"], relative_volume_change=1e-6)
    result = evaluate(cylinder_beam_mesh(), material, [], cylinder_cases(cases), settings)
    assert result["status"] == "failed" and result["mass_g"] is None
    attempt = result["mesh"]["attempts"][0]
    assert attempt["status"] == "failed" and "Tetrahedral body deviates" in attempt["diagnostic"]
    assert not list(tmp_path.rglob("case_*.inp"))

def open_mesh():
    mesh = box_beam_mesh()
    return trimesh.Trimesh(mesh.vertices, mesh.faces[1:], process=False)

def inverted_mesh():
    mesh = box_beam_mesh()
    return trimesh.Trimesh(mesh.vertices, mesh.faces[:, ::-1], process=False)

@pytest.mark.parametrize("mesh, override, message", [
    (open_mesh, {}, "closed"), (inverted_mesh, {}, "closed"), (box_beam_mesh, {"tet_attempts": ["fTetWild"]}, "tet_attempts"),
    (box_beam_mesh, {"tet_attempts": []}, "tet_attempts"), (box_beam_mesh, {"mesh_minimum_sicn": 0.0}, "mesh_minimum_sicn"),
    (box_beam_mesh, {"fea_remesh_targets_mm": [2.0, 0.0]}, "fea_remesh_targets_mm"), (box_beam_mesh, {"fea_remesh_targets_mm": []}, "fea_remesh_targets_mm"),
    (box_beam_mesh, {"fea_remesh_targets_mm": [1.0, 1.5]}, "fea_remesh_targets_mm"), (box_beam_mesh, {"fea_memory_per_element_kb": 0.0}, "fea_memory_per_element_kb"), (box_beam_mesh, {"relative_volume_change": 0.0}, "relative_volume_change"), (box_beam_mesh, {"fea_remesh_iterations": 2.5}, "fea_remesh_iterations")])
def test_invalid_triangle_mesh_input_rejected_before_meshing(tmp_path, mesh, override, message):
    directory = tmp_path / "must_not_be_created"
    _, material, cases, settings, _ = beam_inputs(directory)
    settings.update(override)
    result = evaluate(mesh(), material, [], cases, settings)
    assert result["status"] == "invalid"
    assert message in result["diagnostics"][0]
    assert result["artifacts"] == {} and not directory.exists()

def test_attempt_over_the_element_budget_fails_and_finer_targets_are_skipped(tmp_path):
    _, material, cases, settings, _ = beam_inputs(tmp_path)
    settings.update(tet_attempts=["remesh_hxt", "remesh_delaunay", "refine_hxt"], fea_remesh_targets_mm=[2.0, 1.5], fea_memory_per_element_kb=1e12)
    result = evaluate(box_beam_mesh(), material, [], cases, settings)
    assert result["status"] == "failed" and result["mesh"]["memory_budget"]["elements"] == 0
    first, *rest = result["mesh"]["attempts"]
    assert first["status"] == "failed" and first["over_budget"] and first["linear_element_count"] > 0 and "element budget" in first["diagnostic"]
    assert [(attempt["name"], attempt["target_mm"], attempt["status"]) for attempt in rest] == [("remesh_delaunay", 2.0, "skipped"), ("remesh_hxt", 1.5, "skipped"), ("remesh_delaunay", 1.5, "skipped"), ("refine_hxt", 2.0, "failed")]
    assert all("exceeds the budget" in attempt["diagnostic"] for attempt in rest[:-1]) and rest[-1]["over_budget"] and "element budget" in rest[-1]["diagnostic"]
    assert not list(tmp_path.rglob("mesh.inp")) and not list(tmp_path.rglob("case_*.inp"))

def test_folded_coarse_surface_falls_back_to_a_finer_remesh_target(tmp_path):
    settings = _mesh_settings({**beam_inputs(tmp_path)[3], "tet_attempts": ["remesh_hxt", "remesh_delaunay"], "fea_remesh_targets_mm": [4.0, 1.0]})
    result = {}
    nodes, elements = _volume_mesh(trimesh.creation.cylinder(radius=2.0, height=30.0, sections=64), tmp_path, settings, result)
    attempts = result["mesh"]["attempts"]
    assert [(attempt["name"], attempt["target_mm"], attempt["status"]) for attempt in attempts] == [("remesh_hxt", 4.0, "failed"), ("remesh_delaunay", 4.0, "failed"), ("remesh_hxt", 1.0, "ok")]
    assert all("deviates" in attempt["diagnostic"] for attempt in attempts[:2])
    assert result["mesh"]["target_mm"] == 1.0 and result["mesh"]["prepared_surface"]["folded_edges"] == 0 and not result["mesh"]["over_budget"]
    assert result["mesh"]["element_count"] == len(elements) == attempts[2]["linear_element_count"] <= result["mesh"]["memory_budget"]["elements"] and nodes

def thin_web_mesh(thickness):
    from manifold3d import Manifold
    ring = Manifold.cube((20.0, 20.0, 4.0)) - Manifold.cube((14.0, 16.0, 6.0)).translate((2.0, 2.0, -1.0)) - Manifold.cube((4.0 - thickness, 16.0, 6.0)).translate((16.0, 2.0, -1.0))
    mesh = ring.to_mesh()
    return trimesh.Trimesh(mesh.vert_properties[:, :3].astype(float), mesh.tri_verts, process=False)

def test_surface_preparation_never_welds_a_thin_web_silently(tmp_path):
    web = thin_web_mesh(0.2)
    surface, report = _prepare_surface(web, _mesh_settings({}), False, 2.0)
    assert _topology(web) == {"watertight": True, "winding_consistent": True, "body_count": 1, "euler_number": 0}
    assert report["topology"]["prepared"] == report["topology"]["input"] == _topology(surface) == _topology(web) and not report["topology_changed"]
    assert report["passed"] and not report["rejected_steps"] and report["merge_tolerance_mm"] < 1e-6
    welding = _mesh_settings({"fea_merge_relative_tolerance": 0.3 / web.scale})
    surface, report = _prepare_surface(web, welding, False, 2.0)
    assert not report["rejected_steps"].get("merge_close_vertices", {}).get("watertight") and _topology(surface) == _topology(web) and not report["topology_changed"]
    record = tmp_path / "attempt_metadata.json"
    with pytest.raises(ValueError, match="Prepared FEA surface"):
        _prepare_surface(thin_web_mesh(0.1), welding, False, 2.0, record)
    failed = json.loads(record.read_text(encoding="utf-8"))["prepared_surface"]
    assert not failed["passed"] and "merge_close_vertices" in failed["rejected_steps"] and failed["folded_edges"] and failed["topology"]["input"] == _topology(thin_web_mesh(0.1))

def synthetic_frame_mesh(parameters):
    from manifold3d import Manifold, OpType
    frame, motors = parameters["frame"], motor_positions(parameters).values()
    height = frame["arm_height_mm"]
    def cylinder(radius, x, y):
        return Manifold.cylinder(height, radius, circular_segments=64).translate((x, y, 0))
    parts = [cylinder(frame["motor_pad_radius_mm"], x, y) for x, y in motors] + [Manifold.batch_hull([cylinder(4.0, 0, 0), cylinder(4.0, x, y)]) for x, y in motors]
    parts.append(Manifold.cube((frame["deck_width_mm"], frame["body_length_mm"], frame["deck_top_mm"])).translate((-frame["deck_width_mm"] / 2, -frame["body_length_mm"] / 2, 0)))
    parts.append(Manifold.cube((20.0, frame["camera_y_mm"] + 12.0, frame["cage_height_mm"])).translate((-10.0, 0, 0)))
    mesh = Manifold.batch_boolean(parts, OpType.Add).to_mesh64()
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts.astype(int), process=False)

def test_every_frame_selector_meets_tetrahedralized_triangle_mesh_on_exact_planes(tmp_path):
    parameters = reference_parameters()
    model = prepare_frame_case(parameters)
    mesh = synthetic_frame_mesh(parameters)
    assert mesh.is_watertight and mesh.body_count == 1
    settings = {**model["settings"], "work_dir": str(tmp_path), "mesh_size_mm": 4.0, "mesh_second_order_linear": True}
    result = evaluate(mesh, model["material"], model["point_masses"], model["load_cases"], settings)
    assert result["status"] == "ok", (result["diagnostics"], result.get("mesh"))
    assert result["mesh"]["attempt"] in MESH_ATTEMPTS and result["mesh"]["attempts"][-1] == {**result["mesh"]["attempts"][-1], "name": result["mesh"]["attempt"], "status": "ok"}
    assert result["frame_mass_g"] == pytest.approx(mesh.volume * model["material"]["density_g_cm3"] / 1000, rel=1e-12)
    assert set(result["load_cases"]) == {case["name"] for case in model["load_cases"]} and len(result["eigenfrequencies_hz"]) > 0
    assert result["point_mass_coupling"][0]["attachment_nodes"] >= 3
    nodes, _ = _read_mesh(Path(result["artifacts"]["directory"]) / "mesh.inp")
    regions = [model["point_masses"][0]["attachment_region"]]
    for case in model["load_cases"]:
        measured = result["load_cases"][case["name"]]
        assert measured["fixed_node_count"] >= 3 and all(load["node_count"] >= 3 for load in measured.get("loads", []))
        regions.extend(case["fixed_regions"] + [load["region"] for load in case.get("loads", [])])
    for region in regions:
        heights = [nodes[node][2] for node in _select(nodes, region)]
        if region["max_mm"][2] - region["min_mm"][2] < 0.05:
            assert max(abs(z - (region["min_mm"][2] + region["max_mm"][2]) / 2) for z in heights) < 1e-6
    json.dumps(result, allow_nan=False)

def test_prepared_surface_does_not_weld_a_thin_slab():
    slab = trimesh.creation.box(extents=(6.0, 6.0, 0.04))
    surface, report = _prepare_surface(slab, _mesh_settings({}), False, 2.0)
    assert report["passed"] and not report["cleanup_applied"] and not report["topology_changed"] and report["folded_edges"] == 0
    assert surface.is_watertight and surface.volume == pytest.approx(slab.volume, rel=0.01)

def test_fallback_attempts_get_the_short_timeout(tmp_path, monkeypatch):
    import deep_frame.fea as fea
    seen = []
    def run(command, directory, timeout, threads, log_path):
        seen.append((json.loads(Path(command[-1]).read_text(encoding="utf-8"))["attempt"], timeout))
        raise RuntimeError("mesher failed")
    monkeypatch.setattr(fea, "_run", run)
    settings = _mesh_settings({"mesh_timeout_s": 900.0, "tet_attempts": ["remesh_hxt", "classify_hxt", "classify_delaunay"], "fea_remesh_targets_mm": [2.0]})
    with pytest.raises(RuntimeError, match="All tetrahedral meshing attempts failed"):
        _volume_mesh(trimesh.creation.box(extents=(4, 4, 4)), tmp_path, settings, {})
    assert seen == [("remesh_hxt", 900.0), ("classify_hxt", 120.0), ("classify_delaunay", 120.0)]
