from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from deep_frame.topology_optimization import DensityMap, HexElasticity, _settings, elasticity_matrix, hexahedron_matrices, optimize_topology, regular_grid

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
