from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from deep_frame.topology_optimization import DensityMap, HexElasticity, _oc_update, _settings, elasticity_matrix, hexahedron_matrices, optimize_topology, regular_grid

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

def test_progress_callback_reports_iterations_without_mutating_solver_history():
    domain = beam_domain((6, 3, 3))
    settings = {"volume_fraction": 0.7, "filter_radius_mm": 3.0, "max_iterations": 3, "change_tolerance": 1e-12}
    observed = []
    def progress(entry):
        observed.append(deepcopy(entry))
        entry["compliances_n_mm"].clear()
        entry["objective"] = -1
    result = optimize_topology(domain, settings, progress_callback=progress)
    plain = optimize_topology(domain, settings)
    assert result["status"] == "ok"
    assert observed == result["history"]
    assert len(observed) == 4
    assert observed[-1]["final_evaluation"]
    assert observed[-1]["maximum_design_change"] is None
    assert np.array_equal(result["density"], plain["density"])
    assert result["summary"]["objective_final"] == plain["summary"]["objective_final"]

def robust_map(beta, shape=(6, 4, 4), radius=2.6, **settings):
    domain = beam_domain(shape, (1.0, 1.0, 1.0))
    mapping = DensityMap(domain, _settings({"projection": "robust", "filter_radius_mm": radius, **settings}))
    mapping.beta = beta
    return domain, mapping

def random_design(mapping, seed, low=0.2, high=0.8):
    design = np.random.default_rng(seed).uniform(low, high, mapping.n)
    design[mapping.preserve] = 1
    return design

@pytest.mark.parametrize("beta", [0.0, 1.0, 4.0])
def test_single_projection_reproduces_legacy_formula_bitwise(beta):
    domain = beam_domain((6, 4, 4), (1.0, 1.0, 1.0))
    mapping = DensityMap(domain, _settings({"filter_radius_mm": 2.6, "projection_beta": beta, "projection_eta": 0.45}))
    design = random_design(mapping, 3)
    filtered = np.asarray(mapping.filter @ design).ravel() / mapping.sums
    if beta:
        denominator = np.tanh(beta * 0.45) + np.tanh(beta * 0.55)
        expected = (np.tanh(beta * 0.45) + np.tanh(beta * (filtered - 0.45))) / denominator
        derivative = beta * (1 - np.tanh(beta * (filtered - 0.45)) ** 2) / denominator
    else:
        expected, derivative = filtered.copy(), np.ones(mapping.n)
    expected[mapping.preserve] = 1
    derivative[~mapping.free] = 0
    physical, projection = mapping.physical(design)
    assert np.array_equal(physical, np.clip(expected, 0, 1))
    assert np.array_equal(projection, derivative)
    assert mapping.volume(design) == np.sum(physical)
    fields, _ = mapping.fields(design)
    assert list(fields) == ["intermediate"]
    assert np.array_equal(fields["intermediate"][0], physical)

@pytest.mark.parametrize("beta", [1.0, 8.0, 16.0])
def test_robust_projection_derivatives_match_finite_differences(beta):
    domain, mapping = robust_map(beta)
    design = random_design(mapping, 7)
    rng = np.random.default_rng(11)
    direction = rng.normal(size=mapping.n)
    direction[~mapping.free] = 0
    direction /= np.linalg.norm(direction)
    weights = rng.uniform(0.5, 1.5, mapping.n)
    step = 1e-6
    fields, _ = mapping.fields(design)
    plus, _ = mapping.fields(design + step * direction)
    minus, _ = mapping.fields(design - step * direction)
    for name, (_, derivative) in fields.items():
        expected = weights @ (plus[name][0] - minus[name][0]) / (2 * step)
        assert mapping.pullback(weights, derivative) @ direction == pytest.approx(expected, rel=1e-6, abs=1e-9)
    system = HexElasticity(domain)
    design = random_design(mapping, 7, 0.6, 0.95)
    fields, _ = mapping.fields(design)
    gradient = mapping.pullback(system.solve(fields["eroded"][0])["tip"]["derivative"], fields["eroded"][1])
    step = 1e-5
    compliance = [system.solve(mapping.fields(design + sign * step * direction)[0]["eroded"][0])["tip"]["compliance_n_mm"] for sign in (1, -1)]
    assert gradient @ direction == pytest.approx((compliance[0] - compliance[1]) / (2 * step), rel=1e-4, abs=1e-10)

@pytest.mark.parametrize("beta", [1.0, 2.0, 8.0, 16.0, 64.0])
def test_robust_fields_are_ordered_and_respect_masks(beta):
    domain, mapping = robust_map(beta)
    domain["allowed"][2:4, 1, 1] = False
    domain["forbidden"] = ~domain["allowed"]
    mapping = DensityMap(domain, mapping.settings)
    mapping.beta = beta
    for seed in range(3):
        fields, filtered = mapping.fields(random_design(mapping, seed, 0.0, 1.0))
        eroded, intermediate, dilated = (fields[name][0] for name in ("eroded", "intermediate", "dilated"))
        assert np.all(eroded <= intermediate) and np.all(intermediate <= dilated)
        for field, derivative in fields.values():
            assert np.all(field[mapping.preserve] == 1) and np.all(field[mapping.forbidden] == 0)
            assert np.all(derivative[~mapping.free] == 0)
        assert np.all((filtered >= 0) & (filtered <= 1 + 1e-12))

@pytest.mark.parametrize("beta", [32.0, 64.0])
def test_high_beta_volume_derivative_stays_positive_and_oc_accepts_saturation(beta):
    domain, mapping = robust_map(beta)
    design = np.ones(mapping.n)
    fields, filtered = mapping.fields(design)
    legacy = beta * (1 - np.tanh(beta * (filtered - 0.25)) ** 2)
    assert np.any(legacy[mapping.free] == 0)
    assert np.all(fields["dilated"][1][mapping.free] > 0)
    assert np.all(mapping.pullback(np.ones(mapping.n), fields["dilated"][1])[mapping.free] > 0)
    design[mapping.free] = 0.3
    saturated = mapping.pullback(np.ones(mapping.n), mapping.fields(design)[0]["dilated"][1])
    saturated[np.flatnonzero(mapping.free)[::2]] = 0
    objective = -np.ones(mapping.n)
    target = 0.9 * mapping.volume(design)
    candidate = _oc_update(design, objective, saturated, mapping, target, mapping.settings, 0.2)
    assert np.all(np.isfinite(candidate)) and mapping.volume(candidate) <= target
    saturated[np.flatnonzero(mapping.free)[0]] = -1e-9
    with pytest.raises(RuntimeError, match="OC sensitivities"):
        _oc_update(design, objective, saturated, mapping, target, mapping.settings, 0.2)

def test_oc_bisection_hits_dilated_target_and_steps_toward_unreachable_targets():
    _, mapping = robust_map(4.0)
    design = random_design(mapping, 5)
    fields, _ = mapping.fields(design)
    objective = -np.random.default_rng(2).uniform(0.5, 1.5, mapping.n)
    volume = mapping.pullback(np.ones(mapping.n), fields["dilated"][1])
    target = 0.97 * mapping.volume(design)
    candidate = _oc_update(design, objective, volume, mapping, target, mapping.settings, 0.2)
    assert target * (1 - 1e-6) <= mapping.volume(candidate) <= target
    assert np.max(np.abs(candidate - design)) <= 0.2 + 1e-15
    lowest = _oc_update(design, objective, volume, mapping, 0.5 * mapping.volume(design), mapping.settings, 0.05)
    expected = design.copy()
    expected[mapping.free] = np.maximum(mapping.settings["minimum_design_density"], design[mapping.free] - 0.05)
    assert np.array_equal(lowest, expected)

ROBUST_RUN = {"projection": "robust", "volume_fraction": 0.4, "filter_radius_mm": 2.6, "move_limit": 0.12, "beta_schedule": [1, 2, 4],
              "beta_interval": 6, "beta_minimum_iterations": 3, "move_limit_late": 0.03, "move_limit_late_beta": 4,
              "volume_target_relaxation": 1.0, "minimum_iterations": 3, "max_iterations": 30}

@pytest.mark.parametrize("rule", ["change", "interval"])
def test_beta_continuation_switching_rules(rule):
    settings = {**ROBUST_RUN, "beta_change_tolerance": 0.5 if rule == "change" else 1e-12, "change_tolerance": 0.5}
    result = optimize_topology(beam_domain((8, 4, 4), (1.0, 1.0, 1.0)), settings)
    assert result["status"] == "ok", result["diagnostics"]
    updates = [entry for entry in result["history"] if not entry["final_evaluation"]]
    betas = [entry["projection_beta"] for entry in updates]
    level = 3 if rule == "change" else 6
    assert betas == [1.0] * level + [2.0] * level + [4.0] * 3
    assert result["summary"]["converged"] and result["summary"]["stop_reason"] == "change_tolerance"
    assert all(entry["maximum_design_change"] < 0.5 for entry in updates)
    assert [entry["move_limit"] for entry in updates] == [0.12] * (2 * level) + [0.03] * 3
    assert all(entry["maximum_design_change"] <= 0.03 + 1e-15 for entry in updates[-3:])
    assert result["summary"]["continuation"] == {"beta_schedule": [1.0, 2.0, 4.0], "beta_final": 4.0, "final_level_reached": True, "level_iterations_final": 3}

def test_final_beta_level_keeps_the_per_level_minimum():
    settings = {**ROBUST_RUN, "beta_change_tolerance": 0.5, "change_tolerance": 0.5, "minimum_iterations": 1, "beta_minimum_iterations": 4}
    result = optimize_topology(beam_domain((8, 4, 4), (1.0, 1.0, 1.0)), settings)
    betas = [entry["projection_beta"] for entry in result["history"] if not entry["final_evaluation"]]
    assert betas == [1.0] * 4 + [2.0] * 4 + [4.0] * 4 and result["summary"]["stop_reason"] == "change_tolerance"

def test_dilated_target_starts_each_level_at_the_dilated_volume_and_relaxes():
    domain = beam_domain((8, 4, 4), (1.0, 1.0, 1.0))
    settings = {**ROBUST_RUN, "volume_target_relaxation": 0.2, "max_iterations": 16, "change_tolerance": 1e-12, "beta_change_tolerance": 1e-12}
    result = optimize_topology(domain, settings)
    assert result["status"] == "ok", result["diagnostics"]
    total = 0.4 * np.count_nonzero(domain["allowed"])
    previous = None
    for entry in result["history"][:-1]:
        reference = entry["dilated_density_sum"] if previous is None or previous["projection_beta"] != entry["projection_beta"] else previous["dilated_volume_target"]
        expected = reference + 0.2 * (total * entry["dilated_density_sum"] / entry["physical_density_sum"] - reference)
        assert entry["dilated_volume_target"] == pytest.approx(expected, rel=1e-12)
        previous = entry

@pytest.mark.parametrize("iterations", [45, 60])
def test_objective_stall_is_recorded_but_neither_advances_beta_nor_converges(iterations):
    settings = {**ROBUST_RUN, "volume_target_relaxation": 0.2, "objective_window": 3, "change_tolerance": 0.02, "beta_change_tolerance": 0.02, "max_iterations": iterations}
    result = optimize_topology(beam_domain((8, 4, 4), (1.0, 1.0, 1.0)), settings)
    assert result["status"] == "ok", result["diagnostics"]
    updates = [entry for entry in result["history"] if not entry["final_evaluation"]]
    for index, entry in enumerate(updates):
        level = [other for other in updates[:index + 1] if other["projection_beta"] == entry["projection_beta"]][-3:]
        objectives = [other["objective"] for other in level]
        assert entry["objective_stall"] == (None if len(level) < 3 else pytest.approx((max(objectives) - min(objectives)) / min(objectives), rel=1e-12))
    assert [entry["projection_beta"] for entry in updates[:13]] == [1.0] * 6 + [2.0] * 6 + [4.0]
    assert sum(entry["objective_stall"] is not None and entry["objective_stall"] < 0.02 <= entry["maximum_design_change"] for entry in updates) >= 5
    if iterations == 45:
        assert not result["summary"]["converged"] and result["summary"]["stop_reason"] == "max_iterations" and len(updates) == 45
    else:
        assert result["summary"]["converged"] and result["summary"]["stop_reason"] == "change_tolerance" and updates[-1]["maximum_design_change"] < 0.02 <= updates[-2]["maximum_design_change"]

def test_robust_optimization_is_deterministic_and_reports_three_fields():
    domain = beam_domain((8, 4, 4), (1.0, 1.0, 1.0))
    settings = {**ROBUST_RUN, "max_iterations": 16, "change_tolerance": 1e-12}
    first, second = optimize_topology(deepcopy(domain), settings), optimize_topology(deepcopy(domain), settings)
    assert first["status"] == second["status"] == "ok", first["diagnostics"]
    for key in ("density", "design_density", "eroded_density", "dilated_density", "filtered_density"):
        assert np.array_equal(first[key], second[key])
    assert [{key: value for key, value in entry.items() if key != "elapsed_s"} for entry in first["history"]] == [
        {key: value for key, value in entry.items() if key != "elapsed_s"} for entry in second["history"]]
    assert np.all(first["eroded_density"] <= first["density"]) and np.all(first["density"] <= first["dilated_density"])
    for key in ("density", "eroded_density", "dilated_density"):
        assert np.all(first[key][domain["preserve"]] == 1)
    robust = first["summary"]["robust"]
    assert robust["thresholds"] == {"eroded": 0.75, "intermediate": 0.5, "dilated": 0.25}
    assert robust["eroded_volume_fraction"] < first["summary"]["volume_fraction"] < robust["dilated_volume_fraction"]
    assert first["summary"]["volume_fraction"] == pytest.approx(0.4, rel=0.01)
    eroded = HexElasticity(domain).solve(first["eroded_density"])
    for name, value in first["history"][-1]["compliances_n_mm"].items():
        assert value == pytest.approx(eroded[name]["compliance_n_mm"], rel=1e-12)

@pytest.mark.parametrize("settings", [{"volume_fraction": 0.1}, {"volume_fraction": float("nan")}, {"penalization": 0}, {"case_weights": {"missing": 1}}, {"max_iterations": 0},
                                      {"projection": "bogus"}, {"projection": "robust", "robust_delta": 0.5}, {"beta_schedule": [2, 1]}, {"beta_schedule": []},
                                      {"move_limit_late": 0.0}, {"beta_interval": 0}, {"volume_target_relaxation": 0.0}, {"volume_target_relaxation": 1.5}, {"objective_window": 1},
                                      {"gpu_solver_residency": "swap"}])
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

def test_forbidden_only_nodes_receive_no_load_and_no_ghost_stiffness():
    domain = beam_domain((6, 4, 4))
    domain["allowed"][:, 2:, :] = False
    domain["preserve"] &= domain["allowed"]
    domain["forbidden"] = ~domain["allowed"]
    system = HexElasticity(domain)
    inactive = np.setdiff1d(np.arange(system.ndof), system.active_dofs)
    for case in system.cases:
        if case["analysis"] == "static":
            assert np.all(case["force"][inactive] == 0)
    stiffness = system.matrix(np.full(system.nelem, 4430.0))
    assert stiffness[inactive].nnz == 0
    assert system.diagnostics()["selector_filtering"]
    assert system.solve(domain["allowed"].astype(float))["tip"]["compliance_n_mm"] > 0

def test_preserved_interface_policy_excludes_unattached_free_load_nodes():
    domain = beam_domain((6, 5, 5))
    domain["preserve"][-1] = False
    domain["preserve"][-1, 2, 2] = True
    system = HexElasticity(domain, interface_node_policy="preserve_adjacent")
    force = system.cases[0]["force"].reshape(-1, 3)
    loaded = np.flatnonzero(np.linalg.norm(force, axis=1) > 0)
    assert len(loaded) == 4
    assert np.all(np.isin(loaded, system.interface_nodes))
    assert system.diagnostics()["selector_filtering"][0]["removed_nonpreserve_interface_nodes"] == 32
    assert np.sum(force, axis=0) == pytest.approx([0.0, 0.0, -1.0])

@pytest.fixture
def cuda_solver():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device is unavailable")
    pytest.importorskip("nvidia.cu12")
    from deep_frame.topology_optimization import CudaDirectSolver
    return CudaDirectSolver

def test_linear_backend_is_explicit_and_cpu_is_default():
    assert _settings({})["linear_solver"] == "cpu_superlu"
    with pytest.raises(ValueError, match="linear_solver"):
        _settings({"linear_solver": "fake_gpu"})
    result = optimize_topology(beam_domain(), {"linear_solver": "fake_gpu"})
    assert result["status"] == "invalid"

@pytest.mark.parametrize("failure", [RuntimeError, MemoryError])
def test_cuda_unavailable_reports_failure_without_cpu_fallback(monkeypatch, failure):
    from deep_frame import topology_optimization as topology_gpu
    def unavailable(*args, **kwargs):
        raise failure("CUDA unavailable in controlled test")
    monkeypatch.setattr(topology_gpu, "CudaDirectSolver", unavailable)
    result = optimize_topology(beam_domain((5, 3, 3)), {"volume_fraction": 0.65,
                               "filter_radius_mm": 3.1, "linear_solver": "cuda_cudss"})
    assert result["status"] == "failed"
    assert result["diagnostics"] == ["CUDA unavailable in controlled test"]
    assert result["history"] == []

def test_cleanup_diagnostic_preserves_original_solver_error(monkeypatch):
    from deep_frame import topology_optimization as topology_gpu
    def unavailable(*args, **kwargs):
        raise RuntimeError("primary failure")
    monkeypatch.setattr(topology_gpu, "CudaDirectSolver", unavailable)
    monkeypatch.setattr(HexElasticity, "close", lambda self: ["secondary cleanup failure"])
    result = optimize_topology(beam_domain((5, 3, 3)), {"volume_fraction": 0.65,
                               "filter_radius_mm": 3.1, "linear_solver": "cuda_cudss"})
    assert result["status"] == "failed"
    assert result["diagnostics"] == ["primary failure", "secondary cleanup failure"]

def test_cleanup_checks_statuses_after_failed_synchronization():
    import ctypes
    from types import SimpleNamespace
    from deep_frame.topology_optimization import CudaDirectSolver
    def failed_sync():
        raise RuntimeError("device sync failure")
    calls = []
    def destroy(handle):
        calls.append(handle.value)
        return 6
    solver = object.__new__(CudaDirectSolver)
    solver.closed = False
    solver.cleanup_errors = []
    solver.dll_directories = []
    solver.stream = SimpleNamespace(synchronize=failed_sync)
    solver.handles = {"matrix": ctypes.c_void_p(123)}
    solver.library = SimpleNamespace(cudssMatrixDestroy=destroy)
    errors = solver.close()
    assert errors == ["CUDA cleanup synchronization: device sync failure", "cudssMatrixDestroy failed with cuDSS status 6"]
    assert solver.close() == errors
    assert calls == [123]

def test_gpu_structural_zero_storage_preserves_exact_cpu_matrix():
    domain = beam_domain((6, 3, 3))
    domain["allowed"][2:4, 1, 1] = False
    domain["forbidden"] = ~domain["allowed"]
    cpu = HexElasticity(domain)
    gpu = HexElasticity(domain, linear_solver="cuda_cudss")
    patterns = []
    for density in (np.ones(cpu.nelem), np.random.default_rng(17).uniform(0.001, 1, cpu.nelem)):
        moduli = cpu.young * (1e-6 + (1 - 1e-6) * density ** 3)
        reference, actual = cpu.matrix(moduli), gpu.matrix(moduli)
        assert (reference != actual).nnz == 0
        patterns.append((actual.indptr, actual.indices))
    assert np.array_equal(patterns[0][0], patterns[1][0])
    assert np.array_equal(patterns[0][1], patterns[1][1])

def test_cuda_direct_multi_rhs_updates_and_rejects_changed_structure(cuda_solver):
    matrix = csr_matrix([[4., 1.], [1., 3.]])
    rhs = np.array([[1., 2.], [2., 1.]])
    solver = cuda_solver(matrix, rhs)
    try:
        assert np.allclose(solver.solve(matrix, rhs), np.linalg.solve(matrix.toarray(), rhs), rtol=1e-12)
        assert np.allclose(solver.solve(matrix * 2, rhs), np.linalg.solve(matrix.toarray() * 2, rhs), rtol=1e-12)
        with pytest.raises(ValueError, match="sparse structure"):
            solver.solve(csr_matrix(np.eye(2)), rhs)
        assert len(solver.timings) == 2
    finally:
        solver.close()
    solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.solve(matrix, rhs)

def test_cuda_device_factorization_error_is_not_accepted(cuda_solver):
    matrix = csr_matrix([[1., 2.], [2., 1.]])
    solver = cuda_solver(matrix, np.ones((2, 1)))
    try:
        with pytest.raises(RuntimeError, match="cuDSS"):
            solver.solve(matrix, np.ones((2, 1)))
    finally:
        solver.close()

def test_cuda_gradient_filter_and_three_updates_match_cpu(cuda_solver):
    domain = beam_domain((5, 3, 3))
    settings = _settings({"volume_fraction": 0.65, "filter_radius_mm": 3.1,
                          "max_iterations": 3, "change_tolerance": 1e-12})
    mapping = DensityMap(domain, settings)
    cpu, gpu = HexElasticity(domain), HexElasticity(domain, linear_solver="cuda_cudss")
    design = np.full(cpu.nelem, 0.65)
    design[mapping.preserve] = 1
    physical, projection = mapping.physical(design)
    try:
        reference, actual = cpu.solve(physical), gpu.solve(physical)
        for key in reference:
            assert actual[key]["relative_residual"] < 1e-8
            assert actual[key]["compliance_n_mm"] == pytest.approx(reference[key]["compliance_n_mm"], rel=1e-8)
            assert np.allclose(actual[key]["derivative"], reference[key]["derivative"], rtol=1e-7, atol=1e-12)
        derivative = mapping.pullback(actual["tip"]["derivative"], projection)
        direction = np.random.default_rng(42).normal(size=cpu.nelem)
        direction[~mapping.free] = 0
        direction /= np.linalg.norm(direction)
        step = 1e-5
        plus = gpu.solve(mapping.physical(design + step * direction)[0])["tip"]["compliance_n_mm"]
        minus = gpu.solve(mapping.physical(design - step * direction)[0])["tip"]["compliance_n_mm"]
        assert derivative @ direction == pytest.approx((plus - minus) / (2 * step), rel=2e-5, abs=1e-10)
        assert gpu.gpu_reanalyses == 0
    finally:
        gpu.close()
    reference = optimize_topology(deepcopy(domain), settings)
    actual = optimize_topology(deepcopy(domain), {**settings, "linear_solver": "cuda_cudss"})
    assert actual["status"] == reference["status"] == "ok"
    assert np.allclose(actual["density"], reference["density"], rtol=0, atol=1e-7)
    assert actual["summary"]["objective_final"] == pytest.approx(reference["summary"]["objective_final"], rel=1e-8)

def test_cuda_robust_continuation_matches_cpu(cuda_solver):
    domain = beam_domain((8, 4, 4), (1.0, 1.0, 1.0))
    settings = {**ROBUST_RUN, "volume_fraction": 0.5, "beta_schedule": [1, 4], "beta_interval": 2, "beta_minimum_iterations": 2,
                "max_iterations": 4, "change_tolerance": 1e-12, "move_limit_late_beta": 4}
    reference = optimize_topology(deepcopy(domain), settings)
    actual = optimize_topology(deepcopy(domain), {**settings, "linear_solver": "cuda_cudss"})
    assert actual["status"] == reference["status"] == "ok", actual["diagnostics"]
    assert [entry["projection_beta"] for entry in actual["history"]] == [1.0, 1.0, 4.0, 4.0, 4.0]
    for key in ("density", "design_density", "eroded_density", "dilated_density"):
        assert np.max(np.abs(actual[key] - reference[key])) < 1e-7
    for cpu, gpu in zip(reference["history"], actual["history"], strict=True):
        assert max(abs(gpu["compliances_n_mm"][name] / value - 1) for name, value in cpu["compliances_n_mm"].items()) < 1e-6
    assert abs(actual["summary"]["objective_final"] / reference["summary"]["objective_final"] - 1) < 1e-6

def test_cuda_transient_residency_releases_factors_and_matches_resident(cuda_solver):
    domain = beam_domain((8, 4, 4), (1.0, 1.0, 1.0))
    settings = {**ROBUST_RUN, "volume_fraction": 0.5, "beta_schedule": [1, 4], "beta_interval": 2, "beta_minimum_iterations": 2,
                "max_iterations": 4, "change_tolerance": 1e-12, "move_limit_late_beta": 4, "linear_solver": "cuda_cudss"}
    resident = optimize_topology(deepcopy(domain), settings)
    transient = optimize_topology(deepcopy(domain), {**settings, "gpu_solver_residency": "transient"})
    assert resident["status"] == transient["status"] == "ok", transient["diagnostics"]
    for key in ("density", "design_density", "eroded_density", "dilated_density"):
        assert np.allclose(transient[key], resident[key], rtol=0, atol=1e-12)
    system = transient["summary"]["system"]
    evaluations = len(transient["history"]) * system["independent_fixtures"]
    assert system["gpu_transient_releases"] == evaluations == len(system["gpu_solver_details"])
    assert resident["summary"]["system"]["gpu_transient_releases"] == 0
    assert HexElasticity(domain, gpu_solver_residency="transient").solve(np.full(domain["allowed"].size, 0.5))["tip"]["compliance_n_mm"] > 0
    with pytest.raises(ValueError, match="gpu_solver_residency"):
        HexElasticity(domain, gpu_solver_residency="swap")

def test_symmetric_half_domain_matches_full_compliance_and_sensitivities():
    from deep_frame.topology_geometry import mirror_field, symmetric_domains
    shape = (8, 4, 4)
    box = lambda low, high: {"kind": "box", "min_mm": low, "max_mm": high}
    allowed = np.ones(shape, dtype=bool)
    allowed[1, 3, 3] = False
    preserve = np.zeros(shape, dtype=bool)
    preserve[[0, -1]] = True
    fixtures = [box([-8.01, -0.01, -0.01], [-7.99, 8.01, 8.01]), box([7.99, -0.01, -0.01], [8.01, 8.01, 8.01])]
    loads = {"straddle": [{"region": box([-2.01, 1.99, 7.99], [2.01, 6.01, 8.01]), "force_n": [0.7, 0.3, -1.0]}],
             "one_side": [{"region": box([1.99, -0.01, 7.99], [6.01, 4.01, 8.01]), "force_n": [0.2, -0.4, -1.0]}],
             "lateral": [{"region": box([-4.01, 3.99, 3.99], [4.01, 4.01, 4.01]), "force_n": [1.0, 0.0, 0.0]}]}
    domain = {"grid": {"shape": list(shape), "spacing_mm": [2.0, 2.0, 2.0], "origin_mm": [-8.0, 0.0, 0.0], "order": "C", "axis_order": "xyz"},
              "allowed": allowed, "preserve": preserve & allowed, "forbidden": ~allowed,
              "material": {"young_modulus_mpa": 4430.0, "poisson_ratio": 0.3, "density_g_cm3": 1.09}, "point_masses": [],
              "load_cases": [{"name": name, "analysis": "static", "fixed_regions": fixtures, "loads": load} for name, load in loads.items()]}
    full, half = symmetric_domains(domain)
    assert np.array_equal(mirror_field(half["allowed"]), full["allowed"]) and not full["allowed"][-2, 3, 3]
    half_density = np.random.default_rng(3).uniform(0.2, 1.0, half["grid"]["shape"])
    density = mirror_field(half_density)
    reference = HexElasticity(full).solve(density.ravel(), metrics=True)
    symmetric = HexElasticity(half).solve(half_density.ravel(), metrics=True)
    for name in loads:
        assert symmetric[name]["compliance_n_mm"] == pytest.approx(reference[name]["compliance_n_mm"], rel=1e-9)
        derivative = reference[name]["derivative"].reshape(shape)
        assert np.allclose((derivative + np.flip(derivative, 0))[4:].ravel(), symmetric[name]["derivative"], rtol=1e-8, atol=1e-14)
        assert symmetric[name]["loads"][0]["mean_displacement_mm"] == pytest.approx(reference[name]["loads"][0]["mean_displacement_mm"], rel=1e-8, abs=1e-14)
        assert symmetric[name]["max_von_mises_mpa"] == pytest.approx(reference[name]["max_von_mises_mpa"], rel=1e-8)
    assert HexElasticity(half).diagnostics()["factorization_groups"] == 2

@pytest.mark.parametrize("policy", ["allowed_adjacent", "preserve_adjacent"])
def test_inertia_relief_is_balanced_support_free_and_matches_on_half_domain(policy):
    from deep_frame.topology_geometry import mirror_field, symmetric_domains
    shape = (8, 4, 4)
    box = lambda low, high: {"kind": "box", "min_mm": low, "max_mm": high}
    preserve = np.zeros(shape, dtype=bool)
    preserve[[0, -1]] = True
    relief = {"point_masses": [{"region": box([-8.01, -0.01, -0.01], [-5.99, 8.01, 0.01]), "mass_g": 3.0}, {"region": box([3.99, 1.99, 7.99], [8.01, 6.01, 8.01]), "mass_g": 5.0}], "preserve_mass_g": 2.0}
    loads = {"thrust": [{"region": box([5.99, -0.01, 7.99], [8.01, 8.01, 8.01]), "force_n": [0.0, 0.0, 2.0]}, {"region": box([-8.01, -0.01, 7.99], [-5.99, 8.01, 8.01]), "force_n": [0.0, 0.0, 2.0]}],
             "crash": [{"region": box([1.99, 7.99, 3.99], [6.01, 8.01, 8.01]), "force_n": [0.4, -3.0, 0.5]}]}
    domain = {"grid": {"shape": list(shape), "spacing_mm": [2.0, 2.0, 2.0], "origin_mm": [-8.0, 0.0, 0.0], "order": "C", "axis_order": "xyz"},
              "allowed": np.ones(shape, dtype=bool), "preserve": preserve, "forbidden": np.zeros(shape, dtype=bool),
              "material": {"young_modulus_mpa": 4430.0, "poisson_ratio": 0.3, "density_g_cm3": 1.09}, "point_masses": [],
              "load_cases": [{"name": name, "analysis": "static", "fixed_regions": [], "loads": load, "inertia_relief": relief} for name, load in loads.items()]}
    full, half = symmetric_domains(domain)
    half_density = np.random.default_rng(5).uniform(0.2, 1.0, half["grid"]["shape"])
    density = mirror_field(half_density)
    systems = {"full": HexElasticity(full, interface_node_policy=policy), "half": HexElasticity(half, interface_node_policy=policy)}
    results = {"full": systems["full"].solve(density.ravel()), "half": systems["half"].solve(half_density.ravel())}
    for name in loads:
        assert results["half"][name]["compliance_n_mm"] == pytest.approx(results["full"][name]["compliance_n_mm"], rel=1e-8)
        derivative = results["full"][name]["derivative"].reshape(shape)
        assert np.allclose((derivative + np.flip(derivative, 0))[4:].ravel(), results["half"][name]["derivative"], rtol=1e-7, atol=1e-14)
        for label, system in systems.items():
            case = next(case for case in system.cases if case["name"] == name)
            assert case["inertia_relief"]["mass_g"] == pytest.approx(10.0)
            reactions = system.support_reactions(density.ravel() if label == "full" else half_density.ravel(), name)
            assert sum(entry["support_dofs"] for entry in reactions) == (6 if label == "full" else 3 * len(reactions))
            assert all(entry["max_reaction_n"] < 1e-9 * entry["nodal_force_sum_n"] for entry in reactions)

def modal_settings(**changes):
    return {"f1_min_hz": 5000.0, "case": "modes", "modes": 4, "tracked": 2, "initial_iterations": 80, "warm_iterations": 2, "penalty": 10.0, "ks": 40.0, "mass_cutoff": 0.1, "multiplier_interval": 10 ** 6, **changes}

def test_modal_constraint_matches_eigsh_and_finite_difference():
    from deep_frame.topology_optimization import ModalConstraint
    domain = beam_domain((8, 3, 3))
    system = HexElasticity(domain)
    density = np.random.default_rng(1).uniform(0.3, 1.0, system.nelem)
    value, gradient, info = ModalConstraint(system, modal_settings())(density)
    reference = system.elastic_frequencies(density, "modes", number=2)
    assert sorted(frequency for _, frequency in info["frequencies_hz"]) == pytest.approx(reference, rel=1e-6)
    for element in (5, 40):
        step = np.zeros(system.nelem)
        step[element] = 1e-5
        difference = (ModalConstraint(system, modal_settings())(density + step)[0] - ModalConstraint(system, modal_settings())(density - step)[0]) / 2e-5
        assert gradient[element] == pytest.approx(difference, rel=1e-3)

def test_modal_constraint_on_half_domain_matches_full_and_adds_point_masses():
    from deep_frame.topology_geometry import mirror_field, symmetric_domains
    from deep_frame.topology_optimization import ModalConstraint
    shape = (8, 4, 4)
    box = lambda low, high: {"kind": "box", "min_mm": low, "max_mm": high}
    preserve = np.zeros(shape, dtype=bool)
    preserve[[0, -1]] = True
    fixtures = [box([-8.01, -0.01, -0.01], [-7.99, 8.01, 8.01]), box([7.99, -0.01, -0.01], [8.01, 8.01, 8.01])]
    domain = {"grid": {"shape": list(shape), "spacing_mm": [2.0, 2.0, 2.0], "origin_mm": [-8.0, 0.0, 0.0], "order": "C", "axis_order": "xyz"},
              "allowed": np.ones(shape, dtype=bool), "preserve": preserve, "forbidden": np.zeros(shape, dtype=bool),
              "material": {"young_modulus_mpa": 4430.0, "poisson_ratio": 0.3, "density_g_cm3": 1.09}, "point_masses": [],
              "load_cases": [{"name": "push", "analysis": "static", "fixed_regions": fixtures, "loads": [{"region": box([-2.01, 1.99, 7.99], [2.01, 6.01, 8.01]), "force_n": [0.0, 0.0, -1.0]}]},
                             {"name": "modes", "analysis": "modal", "fixed_regions": fixtures}]}
    full, half = symmetric_domains(domain)
    half_density = np.random.default_rng(4).uniform(0.3, 1.0, half["grid"]["shape"])
    reference = ModalConstraint(HexElasticity(full), modal_settings())(mirror_field(half_density).ravel())[2]
    symmetric = ModalConstraint(HexElasticity(half), modal_settings())(half_density.ravel())[2]
    assert symmetric["f1_hz"] == pytest.approx(reference["f1_hz"], rel=1e-6)
    full["point_masses"] = half["point_masses"] = [{"name": "battery", "mass_g": 5.0, "attachment_region": box([-4.01, -0.01, 7.99], [4.01, 8.01, 8.01])}]
    loaded = ModalConstraint(HexElasticity(full), modal_settings())
    assert loaded.lumped.sum() == pytest.approx(3 * 5e-6)
    assert loaded(mirror_field(half_density).ravel())[2]["f1_hz"] < reference["f1_hz"]
    assert ModalConstraint(HexElasticity(half), modal_settings())(half_density.ravel())[2]["f1_hz"] == pytest.approx(loaded(mirror_field(half_density).ravel())[2]["f1_hz"], rel=1e-3)

def test_cuda_modal_constraint_matches_cpu_with_warm_substitution(cuda_solver):
    from deep_frame.topology_optimization import ModalConstraint
    domain = beam_domain((8, 3, 3))
    density = np.random.default_rng(1).uniform(0.3, 1.0, 72)
    cpu = ModalConstraint(HexElasticity(domain), modal_settings())(density)
    system = HexElasticity(domain, linear_solver="cuda_cudss")
    constraint = ModalConstraint(system, modal_settings())
    first = constraint(density)
    warm = constraint(density)
    system.close()
    assert first[2]["f1_hz"] == pytest.approx(cpu[2]["f1_hz"], rel=1e-8) and warm[2]["f1_hz"] == pytest.approx(cpu[2]["f1_hz"], rel=1e-8)
    assert np.allclose(warm[1], cpu[1], rtol=1e-5, atol=1e-9 * np.abs(cpu[1]).max())

@pytest.mark.parametrize("method", ["simp", "neural"])
def test_optimizers_carry_prop_disc_weights_and_the_f1_constraint(method):
    from deep_frame.topology_neural import optimize_neural
    discs = {"mode": "soft", "motors_mm": [[16.0, 5.0]], "radius_mm": 6.0, "plane_mm": 10.0, "weight": 3.0, "length_mm": 4.0}
    common = {"volume_fraction": 0.5, "max_iterations": 3, "minimum_iterations": 3, "prop_discs": discs, "modal": modal_settings()}
    result = optimize_topology(beam_domain(), common) if method == "simp" else optimize_neural(beam_domain(), {**common, "frequencies": 8, "hidden": [6], "mirror_axis": None})
    assert result["status"] == "ok" and result["summary"]["modal"]["final"]["f1_hz"] > 0
    assert all(entry["f1_hz"] > 0 and entry["modal_penalty"] > 0 for entry in result["history"][:-1])

def stiffness_settings(**changes):
    return {"min_n_per_mm": 10.0, "case": "push", "calibration": 1.0, "penalty": 10.0, "multiplier_interval": 10 ** 6, **changes}

def test_stiffness_constraint_matches_pad_stiffness_on_half_domain_and_finite_difference():
    from deep_frame.topology_geometry import mirror_field, symmetric_domains
    from deep_frame.topology_optimization import StiffnessConstraint
    shape = (8, 4, 4)
    box = lambda low, high: {"kind": "box", "min_mm": low, "max_mm": high}
    preserve = np.zeros(shape, dtype=bool)
    preserve[[0, -1]] = True
    fixtures = [box([-8.01, -0.01, -0.01], [8.01, 0.01, 8.01])]
    domain = {"grid": {"shape": list(shape), "spacing_mm": [2.0, 2.0, 2.0], "origin_mm": [-8.0, 0.0, 0.0], "order": "C", "axis_order": "xyz"},
              "allowed": np.ones(shape, dtype=bool), "preserve": preserve, "forbidden": np.zeros(shape, dtype=bool),
              "material": {"young_modulus_mpa": 4430.0, "poisson_ratio": 0.3, "density_g_cm3": 1.09}, "point_masses": [],
              "load_cases": [{"name": "push", "analysis": "static", "fixed_regions": fixtures, "loads": [{"region": box([3.99, 5.99, 7.99], [8.01, 8.01, 8.01]), "force_n": [0.0, 0.0, 3.6]}]}]}
    full, half = symmetric_domains(domain)
    half_density = np.random.default_rng(4).uniform(0.3, 1.0, half["grid"]["shape"])
    values = []
    for model, density in ((full, mirror_field(half_density).ravel()), (half, half_density.ravel())):
        system = HexElasticity(model)
        solution = system.solve(density, metrics=True)["push"]
        constraint = StiffnessConstraint(system, stiffness_settings())
        assert constraint.force ** 2 / solution["compliance_n_mm"] == pytest.approx(solution["stiffness_n_per_mm"], rel=1e-9)
        values.append(solution["stiffness_n_per_mm"])
    assert values[0] == pytest.approx(values[1], rel=1e-8)
    density = half_density.ravel()
    target = 2 * values[1]
    _, gradient, info = StiffnessConstraint(system, stiffness_settings(min_n_per_mm=target))(system.solve(density)["push"])
    assert info["violation"] == pytest.approx(1.0, rel=1e-8) and info["stiffness_n_per_mm"] == pytest.approx(values[1], rel=1e-9)
    for element in (3, 40):
        step = np.zeros(system.nelem)
        step[element] = 1e-6
        penalty = lambda value: StiffnessConstraint(system, stiffness_settings(min_n_per_mm=target))(system.solve(value)["push"])[0]
        assert gradient[element] == pytest.approx((penalty(density + step) - penalty(density - step)) / 2e-6, rel=1e-4)
    assert StiffnessConstraint(system, stiffness_settings(min_n_per_mm=values[1] / 2))(system.solve(density)["push"])[0] == 0.0

def test_neural_stiffness_constraint_stays_out_of_the_objective_and_blocks_infeasible_stall():
    from deep_frame.topology_neural import optimize_neural
    domain = beam_domain()
    domain["load_cases"].append({**deepcopy(domain["load_cases"][0]), "name": "push"})
    settings = {"volume_fraction": 0.5, "max_iterations": 6, "minimum_iterations": 2, "sharpness_iterations": 2, "objective_window": 2, "change_tolerance": 1.0, "frequencies": 8, "hidden": [6], "mirror_axis": None}
    free = optimize_neural(domain, settings)
    assert free["status"] == "ok" and free["summary"]["stop_reason"] == "objective_stall"
    result = optimize_neural(domain, {**settings, "stiffness": stiffness_settings(min_n_per_mm=1e6), "feasibility_tolerance": 0.0})
    summary = result["summary"]
    assert result["status"] == "ok" and "push" not in summary["normalization_compliances_n_mm"] and "push" not in summary["static_surrogate_metrics"]
    assert summary["stop_reason"] == "max_iterations" and summary["stiffness"]["active"] and summary["stiffness"]["case_stiffness_n_per_mm"] == pytest.approx(summary["stiffness"]["compliance_stiffness_n_per_mm"], rel=1e-9)
    assert all(entry["stiffness_penalty"] > 0 and entry["stiffness_n_per_mm"] > 0 for entry in result["history"][:-1])

@pytest.mark.parametrize("spacing,choice,expected", [(2.0, None, "cuda_cudss"), (4 / 3, None, "cuda_cudss"), (1.1, None, "cuda_cudss"), (1.0, None, "multigrid"), (0.75, None, "multigrid"), (1.0, {"multigrid_below_mm": 0.9}, "cuda_cudss")])
def test_auto_linear_solver_by_spacing(spacing, choice, expected):
    from deep_frame.topology_problem import cantilever_domain
    domain = cantilever_domain((8, 2, 3), spacing)
    system = HexElasticity(domain, linear_solver="auto", solver_choice=choice)
    assert system.linear_solver == expected and system.requested_solver == "auto"
    assert HexElasticity(domain, linear_solver="cpu_superlu", solver_choice=choice).linear_solver == "cpu_superlu"
    with pytest.raises(ValueError, match="linear_solver"):
        HexElasticity(domain, linear_solver="unknown")
