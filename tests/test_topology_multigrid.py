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
