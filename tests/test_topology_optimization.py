from copy import deepcopy

import numpy as np
import pytest

from deep_frame.topology_elasticity import HexElasticity, elasticity_matrix, hexahedron_matrices, regular_grid
from deep_frame.topology_optimization import DensityMap, _settings, optimize_topology


def beam_domain(shape=(10, 5, 5), spacing=(2.0, 2.0, 2.0)):
    extent = np.array(shape) * spacing
    fixture = {"kind": "box", "min_mm": [-0.01, -0.01, -0.01], "max_mm": [0.01, extent[1] + 0.01, extent[2] + 0.01]}
    force = {"kind": "box", "min_mm": [extent[0] - 0.01, -0.01, -0.01], "max_mm": (extent + 0.01).tolist()}
    preserve = np.zeros(shape, dtype=bool)
    preserve[0] = True
    preserve[-1] = True
    return {
        "grid": {"shape": list(shape), "spacing_mm": list(spacing), "origin_mm": [0.0, 0.0, 0.0], "order": "C", "axis_order": "xyz"},
        "allowed": np.ones(shape, dtype=bool),
        "preserve": preserve,
        "forbidden": np.zeros(shape, dtype=bool),
        "material": {"young_modulus_mpa": 4430.0, "poisson_ratio": 0.3, "density_g_cm3": 1.09},
        "point_masses": [],
        "load_cases": [
            {"name": "tip", "analysis": "static", "fixed_regions": [fixture], "loads": [{"region": force, "force_n": [0.0, 0.0, -1.0]}]},
            {"name": "side", "analysis": "static", "fixed_regions": [fixture], "loads": [{"region": force, "force_n": [0.0, 0.6, 0.0]}]},
            {"name": "modes", "analysis": "modal", "fixed_regions": [fixture]},
        ],
    }


def test_hex8_rigid_modes_patch_energy_and_consistent_mass():
    spacing = (2.0, 3.0, 4.0)
    stiffness, mass, strain = hexahedron_matrices(spacing, 0.3)
    eigenvalues = np.linalg.eigvalsh(stiffness)
    assert np.max(np.abs(eigenvalues[:6])) < 1e-12
    assert eigenvalues[6] > 0.01
    grid = {"shape": [1, 1, 1], "spacing_mm": spacing, "origin_mm": [0, 0, 0]}
    points, connectivity, _ = regular_grid(grid)
    transform = np.array([[0.01, 0.02, -0.01], [0.03, -0.02, 0.01], [0.04, 0.02, 0.03]])
    displacement = (points[connectivity[0]] @ transform.T).ravel()
    engineering_strain = np.array([0.01, -0.02, 0.03, 0.05, 0.03, 0.03])
    expected = np.prod(spacing) * engineering_strain @ elasticity_matrix(0.3) @ engineering_strain
    assert displacement @ stiffness @ displacement == pytest.approx(expected, rel=1e-12)
    assert np.allclose(strain @ displacement, engineering_strain)
    for axis in range(3):
        rigid = np.zeros(24)
        rigid[axis::3] = 1
        assert rigid @ mass @ rigid == pytest.approx(np.prod(spacing), rel=1e-12)


def test_hex8_cantilever_deflection_and_frequency_against_euler_bernoulli():
    domain = beam_domain((24, 4, 4), (2.0, 1.5, 1.5))
    system = HexElasticity(domain)
    result = system.solve(np.ones(system.nelem), metrics=True)["tip"]
    length, width, height = np.array(domain["grid"]["shape"]) * domain["grid"]["spacing_mm"]
    modulus = domain["material"]["young_modulus_mpa"]
    second_moment = width * height ** 3 / 12
    expected_deflection = length ** 3 / (3 * modulus * second_moment)
    expected_frequency = 1.875104068711961 ** 2 / (2 * np.pi * length ** 2) * np.sqrt(modulus * second_moment / (1.09e-9 * width * height))
    frequency = system.elastic_frequencies(np.ones(system.nelem), "modes", number=1)[0]
    assert result["loads"][0]["directional_displacement_mm"] == pytest.approx(expected_deflection, rel=0.08)
    assert frequency == pytest.approx(expected_frequency, rel=0.05)
    assert system.diagnostics()["independent_fixtures"] == 1


def test_simp_and_filter_sensitivity_matches_finite_difference():
    domain = beam_domain((5, 3, 3), (2.0, 2.0, 2.0))
    settings = _settings({"filter_radius_mm": 3.1, "projection_beta": 2.0})
    mapping = DensityMap(domain, settings)
    system = HexElasticity(domain)
    design = np.full(system.nelem, 0.65)
    design[mapping.preserve] = 1
    physical, projection = mapping.physical(design)
    result = system.solve(physical)["tip"]
    derivative = mapping.pullback(result["derivative"], projection)
    volume_derivative = mapping.pullback(np.ones(system.nelem), projection)
    direction = np.random.default_rng(42).normal(size=system.nelem)
    direction[~mapping.free] = 0
    direction /= np.linalg.norm(direction)
    step = 1e-5
    plus = mapping.physical(design + step * direction)[0]
    minus = mapping.physical(design - step * direction)[0]
    finite_difference = (system.solve(plus)["tip"]["compliance_n_mm"] - system.solve(minus)["tip"]["compliance_n_mm"]) / (2 * step)
    assert derivative @ direction == pytest.approx(finite_difference, rel=2e-5, abs=1e-10)
    assert volume_derivative @ direction == pytest.approx((np.sum(plus) - np.sum(minus)) / (2 * step), rel=1e-7)


def test_free_three_dimensional_optimization_reduces_compliance_and_preserves_masks():
    domain = beam_domain()
    domain["allowed"][4:6, 2, 2] = False
    domain["forbidden"] = ~domain["allowed"]
    original = deepcopy(domain)
    result = optimize_topology(domain, {"volume_fraction": 0.50, "filter_radius_mm": 3.1, "max_iterations": 16, "projection_beta": 1.0})
    assert result["status"] == "ok", result["diagnostics"]
    assert result["summary"]["objective_final"] < 0.9 * result["summary"]["objective_initial"]
    assert result["summary"]["volume_fraction"] <= 0.50000001
    assert np.all(result["density"][domain["preserve"]] == 1)
    assert np.all(result["density"][domain["forbidden"]] == 0)
    assert np.std(result["density"][domain["allowed"] & ~domain["preserve"]]) > 0.15
    assert np.ptp(np.mean(result["density"], axis=(0, 1))) > 0.05
    assert np.ptp(np.mean(result["density"], axis=(0, 2))) > 0.05
    assert np.array_equal(domain["preserve"], original["preserve"])
    assert domain["load_cases"] == original["load_cases"]
    assert all(entry["maximum_relative_residual"] < 1e-4 for entry in result["history"])


def test_optimizer_is_deterministic_and_reports_iteration_limit():
    domain = beam_domain((6, 3, 3))
    settings = {"volume_fraction": 0.7, "filter_radius_mm": 3.0, "max_iterations": 3, "change_tolerance": 1e-12}
    first = optimize_topology(domain, settings)
    second = optimize_topology(domain, settings)
    assert first["status"] == second["status"] == "ok"
    assert np.array_equal(first["density"], second["density"])
    assert not first["summary"]["converged"]
    assert first["summary"]["stop_reason"] == "max_iterations"
    assert first["summary"]["objective_final"] == second["summary"]["objective_final"]


@pytest.mark.parametrize("settings", [{"volume_fraction": 0.1}, {"volume_fraction": float("nan")}, {"penalization": 0}, {"case_weights": {"missing": 1}}, {"max_iterations": 0}])
def test_invalid_settings_and_impossible_preserve_budget_fail_cleanly(settings):
    result = optimize_topology(beam_domain(), settings)
    assert result["status"] == "invalid"
    assert result["diagnostics"]
    assert result["summary"] == {}


def test_invalid_masks_and_empty_selectors_fail_cleanly():
    domain = beam_domain()
    domain["forbidden"][0, 0, 0] = True
    result = optimize_topology(domain, {"volume_fraction": 0.5})
    assert result["status"] == "invalid"
    domain = beam_domain()
    domain["load_cases"][0]["loads"][0]["region"]["min_mm"] = [999, 999, 999]
    domain["load_cases"][0]["loads"][0]["region"]["max_mm"] = [1000, 1000, 1000]
    result = optimize_topology(domain, {"volume_fraction": 0.5})
    assert result["status"] == "invalid"
    assert "Empty topology node selector" in result["diagnostics"][0]


def test_thin_box_expansion_is_bounded_and_reported():
    domain = beam_domain()
    domain["load_cases"][0]["loads"][0]["region"] = {"kind": "box", "min_mm": [19.99, 0, 4.99], "max_mm": [20.01, 10, 5.01]}
    system = HexElasticity(domain)
    expansions = system.diagnostics()["selector_expansions"]
    assert len(expansions) == 1
    assert expansions[0]["expansion_each_side_mm"] == [1.0, 1.0, 1.0]
    assert expansions[0]["case"] == "tip"


def test_modal_point_mass_requires_independent_calculix():
    domain = beam_domain((4, 3, 3))
    domain["point_masses"] = [{"name": "battery", "mass_g": 37}]
    with pytest.raises(ValueError, match="independent CalculiX"):
        HexElasticity(domain).elastic_frequencies(np.ones(36), "modes")
