import json
import math
from copy import deepcopy

import pytest
from build123d import Align, Box

from deep_frame.fea import evaluate
from deep_frame.fea_config import FEA_CONFIG


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
