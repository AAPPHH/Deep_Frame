from copy import deepcopy
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from deep_frame.topology_multigrid import GeometricMultigrid
from deep_frame.topology_optimization import HexElasticity
from deep_frame.topology_problem import cantilever_domain
from tests.test_topology_problem import tiny_domain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="multigrid coarsest grid needs CUDA and cuDSS")

def setup(domain, seed=0, settings=None):
    system = HexElasticity(domain, linear_solver="cuda_cudss" if DEVICE == "cuda" else "cpu_superlu")
    rng = np.random.default_rng(seed)
    density = np.clip(rng.uniform(0.3, 1.2, system.nelem), 0, 1)
    moduli = system.young * (1e-6 + (1 - 1e-6) * density ** 3)
    active = np.where(system.active_elements, moduli, 0)
    mg = GeometricMultigrid(domain["grid"]["shape"], system.spacing, system.ke, {"device": DEVICE, "coarsest_dofs": 200, **(settings or {})})
    mg.update(active)
    return system, moduli, mg

@pytest.mark.parametrize("domain", [cantilever_domain((8, 3, 5), 1.0), tiny_domain()])
def test_operator_matches_assembled_matrix(domain):
    system, moduli, mg = setup(domain)
    matrix = system.matrix(moduli).tocsr()
    vectors = np.random.default_rng(1).normal(size=(system.ndof, 2))
    vectors[np.setdiff1d(np.arange(system.ndof), system.active_dofs)] = 0
    full = mg.flat(mg.operator(mg.grid(vectors))).cpu().numpy()
    assert np.linalg.norm(full - matrix @ vectors) < 1e-12 * np.linalg.norm(matrix @ vectors)
    part = next(iter(system.groups.values()))[0][1]
    free = part["free"]
    masked = np.zeros_like(vectors)
    masked[free] = vectors[free]
    reduced = mg.flat(mg.operator(mg.grid(masked), mask=mg.mask(part["fixed"])[0])).cpu().numpy()
    expected = matrix[free][:, free] @ vectors[free]
    assert np.linalg.norm(reduced[free] - expected) < 1e-12 * np.linalg.norm(expected)

@gpu
def test_galerkin_coarse_operator_equals_rap():
    system, _, mg = setup(cantilever_domain((8, 4, 8), 1.0))
    part = next(iter(system.groups.values()))[0][1]
    hierarchy = mg.hierarchy("check", part["fixed"])
    fine, coarse = hierarchy["levels"][:2]
    vector = torch.rand((1, 3) + coarse.nodes, dtype=torch.float64, device=DEVICE) * coarse.mask64
    prolonged = torch.nn.functional.conv_transpose3d(vector, mg.transfer.double(), stride=2, padding=1, groups=3) * fine.mask64
    rap = torch.nn.functional.conv3d(mg.operator(prolonged, mask=fine.mask64), mg.transfer.double(), stride=2, padding=1, groups=3)
    blocks = mg._gather(vector, torch.float64)
    values = torch.einsum("eij,je->ie", coarse.blocks.double(), blocks.reshape(24, -1)).reshape(blocks.shape)
    direct = mg._scatter(values, torch.float64)
    assert torch.linalg.vector_norm(direct - rap) < 1e-6 * torch.linalg.vector_norm(rap)
    mg.close()

@gpu
@pytest.mark.parametrize("projection", [True, False])
def test_pcg_matches_direct_solution(projection):
    domain = tiny_domain()
    system, moduli, mg = setup(domain, settings={"projection": projection})
    matrix = system.matrix(moduli).tocsr()
    for members in system.groups.values():
        part = members[0][1]
        free = part["free"]
        relief = "inertia_relief" in members[0][0]
        solution, report = mg.solve(part["fixed"].tobytes(), part["fixed"], np.column_stack([p["force"] for _, p in members]), part["support"] if relief else None)
        assert report["converged"]
        from scipy.sparse.linalg import spsolve
        for column, (_, p) in enumerate(members):
            expected = np.zeros(system.ndof)
            expected[free] = spsolve(matrix[free][:, free].tocsc(), p["force"][free])
            assert np.linalg.norm(solution[:, column] - expected) < 1e-6 * np.linalg.norm(expected)
    mg.close()

@gpu
@pytest.mark.parametrize("share,projection", [(True, True), (True, False), (False, True)])
def test_multigrid_linear_solver_matches_direct(share, projection):
    density = np.random.default_rng(3).uniform(0.2, 1.0, HexElasticity(tiny_domain()).nelem)
    reference = HexElasticity(tiny_domain(), share_static=False).solve(density, metrics=True)
    system = HexElasticity(tiny_domain(), linear_solver="multigrid", share_static=share, multigrid={"device": DEVICE, "coarsest_dofs": 200, "batch": 3, "projection": projection})
    result = system.solve(density, metrics=True)
    assert system.diagnostics()["factorization_groups"] == (2 if share else 4)
    for name in reference:
        assert result[name]["compliance_n_mm"] == pytest.approx(reference[name]["compliance_n_mm"], rel=1e-7)
        assert np.linalg.norm(result[name]["derivative"] - reference[name]["derivative"]) <= 1e-7 * np.linalg.norm(reference[name]["derivative"])
        assert result[name]["max_displacement_mm"] == pytest.approx(reference[name]["max_displacement_mm"], rel=1e-7)
    assert not system.close()

@gpu
def test_multigrid_problem_evaluation_matches_cudss():
    from tests.test_topology_problem import tiny_problem
    from deep_frame.topology_problem import TopologyProblem
    problem = {**tiny_problem(), "multigrid": {"device": DEVICE, "coarsest_dofs": 200, "batch": 4}}
    results = []
    for solver in ("cuda_cudss", "multigrid"):
        tp = TopologyProblem(tiny_domain(), problem, linear_solver=solver)
        results.append([tp.evaluate(design) for design in (np.full(tp.map.n, 0.6), np.random.default_rng(5).uniform(0.4, 0.8, tp.map.n))])
        factors = len(tp.system.gpu_solvers)
        tp.close()
    assert type(tp.modal).__name__ == "MultigridModal" and factors == 0 and all(report["converged"] for report in tp.modal.reports) and tp.system._assembly is None
    for direct, multigrid in zip(*results):
        assert multigrid["names"] == direct["names"]
        assert np.allclose(multigrid["constraints"], direct["constraints"], rtol=1e-7, atol=1e-9)
        for a, b in zip(multigrid["constraint_gradients"], direct["constraint_gradients"]):
            assert np.linalg.norm(a - b) <= 1e-7 * max(np.linalg.norm(b), 1e-30)

@gpu
def test_lobpcg_modes_match_shift_invert():
    from scipy.sparse.linalg import eigsh
    from deep_frame.topology_multigrid import MultigridModal
    from deep_frame.topology_optimization import ModalConstraint
    from tools.multigrid_study import modal_cantilever, modal_settings
    domain = modal_cantilever((16, 3, 6), 1.0, 0.5)
    system = HexElasticity(domain, linear_solver="cpu_superlu")
    density = np.clip(np.random.default_rng(3).uniform(0.2, 1.3, system.nelem), 0, 1)
    settings = modal_settings(tracked=2, modes=4)
    reference = ModalConstraint(system, settings)
    stiffness, mass = reference.matrices(density, 3.0, 1e-6)
    modes = []
    for part in reference.parts:
        free = part["free"]
        values, vectors = eigsh(stiffness[free][:, free].tocsc(), k=4, M=mass[free][:, free].tocsc(), sigma=0.0)
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order] / np.sqrt(np.einsum("ij,ij->j", vectors[:, order], mass[free][:, free] @ vectors[:, order]))
        part["values"] = values
        for index in range(2):
            full = np.zeros(system.ndof)
            full[free] = vectors[:, index]
            modes.append((part, values[index], full))
    expected = reference.aggregate(density, modes, 3.0, 1e-6)
    mg = GeometricMultigrid(domain["grid"]["shape"], system.spacing, system.ke, {"device": DEVICE, "coarsest_dofs": 200}, system.me)
    modal = MultigridModal(system, settings, mg)
    result = modal.measure(density, 3.0, 1e-6)
    assert all(report["converged"] and report["levels"] > 1 for report in modal.reports)
    for part, exact in zip(modal.parts, reference.parts):
        assert np.max(np.abs(part["values"] - exact["values"]) / exact["values"]) < 1e-9
    assert abs(result[2]["f1_hz"] / expected[2]["f1_hz"] - 1) < 1e-9
    assert np.linalg.norm(result[1] - expected[1]) < 1e-6 * np.linalg.norm(expected[1])
    mg.close()

@gpu
def test_convolution_filter_matches_sparse_filter():
    from deep_frame.topology_optimization import DensityMap, _settings
    from tests.test_topology_optimization import beam_domain
    shape = (14, 11, 9)
    domain = beam_domain(shape, (0.747, 0.753, 0.667))
    allowed = np.random.default_rng(3).uniform(size=shape) > 0.2
    allowed[0] = allowed[-1] = True
    domain.update(allowed=allowed, forbidden=~allowed)
    sparse, convolution = (DensityMap(domain, _settings({"filter_radius_mm": 2.84, "projection_beta": 4.0, "density_filter": name})) for name in ("sparse", "convolution"))
    rng = np.random.default_rng(5)
    design, sensitivity = rng.uniform(size=allowed.size), rng.normal(size=allowed.size)
    physical, projection = sparse.physical(design)
    assert sparse.filter is not None and convolution.convolution is not None
    assert np.max(np.abs(convolution.sums - sparse.sums) / sparse.sums) < 1e-12
    assert np.max(np.abs(convolution.filtered(design) - sparse.filtered(design))) < 1e-12
    assert np.max(np.abs(convolution.pullback(sensitivity, projection) - sparse.pullback(sensitivity, projection))) < 1e-12 * np.max(np.abs(sparse.pullback(sensitivity, projection)))

@gpu
@pytest.mark.parametrize("builder", ["battery_domain", "camera_domain"])
def test_multigrid_carries_body_springs_and_design_dependent_loads(builder):
    from tests import test_topology_problem
    from deep_frame.topology_problem import TopologyProblem
    domain, problem = getattr(test_topology_problem, builder)()
    design = np.random.default_rng(6).uniform(0.3, 0.7, int(np.prod(domain["grid"]["shape"])))
    results = {}
    for solver in ("cpu_superlu", "multigrid"):
        tested = TopologyProblem(deepcopy(domain), {**deepcopy(problem), "multigrid": {"coarsest_dofs": 200}}, linear_solver=solver)
        results[solver] = tested.evaluate(design)
        if solver == "multigrid":
            assert tested.system.multigrid is not None and tested.system.shared == {} and all(row["levels"] > 1 for row in tested.system.multigrid.statistics)
        tested.close()
    exact, tested = results["cpu_superlu"], results["multigrid"]
    assert exact["names"] == tested["names"]
    assert np.max(np.abs(tested["constraints"] - exact["constraints"])) < 1e-6
    for name, left, right in zip(exact["names"], tested["constraint_gradients"], exact["constraint_gradients"]):
        assert np.linalg.norm(left - right) <= 1e-6 * max(np.linalg.norm(right), 1e-12), name

@gpu
def test_zero_load_columns_skip_pcg():
    system, moduli, mg = setup(tiny_domain())
    part = next(iter(system.groups.values()))[0][1]
    forces = np.zeros((system.ndof, 2))
    forces[part["free"], 0] = part["force"][part["free"]]
    forces[part["free"], 1] = 1e-20 * part["force"][part["free"]]
    flat, report = mg.solve("zero", part["fixed"], forces)
    single, _ = mg.solve("zero", part["fixed"], forces[:, :1])
    assert report["converged"] and report["zero_rhs_columns"] == 1 and report["iterations"][1] == 0
    assert not np.any(flat[:, 1]) and np.linalg.norm(flat[:, 0] - single[:, 0]) <= 1e-12 * np.linalg.norm(single[:, 0])
