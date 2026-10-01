from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from deep_frame.topology_elasticity import HexElasticity
from deep_frame.topology_optimization import DensityMap, _settings, optimize_topology
from test_topology_optimization import beam_domain


@pytest.fixture
def cuda_solver():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device is unavailable")
    pytest.importorskip("nvidia.cu12")
    from deep_frame.topology_gpu import CudaDirectSolver

    return CudaDirectSolver


def test_linear_backend_is_explicit_and_cpu_is_default():
    assert _settings({})["linear_solver"] == "cpu_superlu"
    with pytest.raises(ValueError, match="linear_solver"):
        _settings({"linear_solver": "fake_gpu"})
    result = optimize_topology(beam_domain(), {"linear_solver": "fake_gpu"})
    assert result["status"] == "invalid"


@pytest.mark.parametrize("failure", [RuntimeError, MemoryError])
def test_cuda_unavailable_reports_failure_without_cpu_fallback(monkeypatch, failure):
    from deep_frame import topology_gpu

    def unavailable(*args, **kwargs):
        raise failure("CUDA unavailable in controlled test")

    monkeypatch.setattr(topology_gpu, "CudaDirectSolver", unavailable)
    result = optimize_topology(beam_domain((5, 3, 3)), {"volume_fraction": 0.65,
                               "filter_radius_mm": 3.1, "linear_solver": "cuda_cudss"})
    assert result["status"] == "failed"
    assert result["diagnostics"] == ["CUDA unavailable in controlled test"]
    assert result["history"] == []


def test_cleanup_diagnostic_preserves_original_solver_error(monkeypatch):
    from deep_frame import topology_gpu

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
    from deep_frame.topology_gpu import CudaDirectSolver

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
