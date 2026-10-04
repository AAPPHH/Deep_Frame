from copy import deepcopy
import numpy as np
import pytest

from deep_frame.config import BATTERY_SUPPORT, CAMERA_SUPPORT, COMPONENT_DEFAULTS, PRINT_MATERIAL
from deep_frame.topology_optimization import HexElasticity, elasticity_matrix, hexahedron_matrices, orthotropic_matrix
from deep_frame.topology_stability import FOUR_FEET, StandStability, stand_design, stand_domain
from deep_frame.topology_problem import ARM_TIP, LOAD_COVARIANCE, FrontCoverage, ShieldedImpact, covariance_cantilever, LoadCovariance, MMAOptimizer, PROBLEM, load_covariance, Termination, TopologyProblem, cantilever_domain, cantilever_problem, constraint_report, filter_radius, format_report, length_scale_ratio, orthotropic_material, prolongate, radial_weight, shadow_thickness, weighted_disc_area

def box(low, high):
    return {"kind": "box", "min_mm": list(map(float, low)), "max_mm": list(map(float, high))}

def tiny_domain():
    shape = (4, 8, 4)
    allowed = np.ones(shape, dtype=bool)
    preserve = np.zeros(shape, dtype=bool)
    preserve[0, 3:5, 3] = True
    relief = {"point_masses": [{"name": "battery", "region": box([-1, -1, 3.9], [1, 1, 4.1]), "mass_g": 10.0}], "preserve_mass_g": 0.0}
    fixed = [box([-4.1, -4.01, -0.01], [4.1, -3.99, 4.01])]
    return {"grid": {"origin_mm": [0.0, -4.0, 0.0], "spacing_mm": [1.0, 1.0, 1.0], "shape": list(shape), "axis_order": "xyz", "order": "C"},
            "allowed": allowed, "preserve": preserve, "forbidden": ~allowed, "symmetry": {"axis": 0, "plane_mm": 0.0},
            "material": orthotropic_material({"density_g_cm3": 1.09, "poisson_ratio": 0.3}),
            "load_cases": [{"name": "stiffness_arm_tip", "analysis": "static", "fixed_regions": fixed, "loads": [{"region": box([1.9, 3.99, 2.9], [4.01, 4.01, 4.01]), "force_n": [0.0, 0.0, 1.0]}]},
                           {"name": "crash_front", "analysis": "static", "fixed_regions": [], "inertia_relief": relief, "loads": [{"region": box([-1.01, 3.99, 0.99], [1.01, 4.01, 3.01]), "force_n": [0.0, -2.0, 0.0]}]},
                           {"name": "crash_side_left", "analysis": "static", "fixed_regions": [], "inertia_relief": relief, "loads": [{"region": box([3.99, 1.99, -0.01], [4.01, 4.01, 2.01]), "force_n": [1.5, 0.0, -0.5]}]},
                           {"name": "twist", "analysis": "static", "fixed_regions": [], "inertia_relief": relief, "loads": [{"region": box([2.99, 2.99, 3.99], [4.01, 4.01, 4.01]), "force_n": [0.0, 0.0, 1.0]}, {"region": box([-4.01, -4.01, 3.99], [-2.99, -2.99, 4.01]), "force_n": [0.0, 0.0, 1.0]}, {"region": box([2.99, -4.01, 3.99], [4.01, -2.99, 4.01]), "force_n": [0.0, 0.0, -1.0]}, {"region": box([-4.01, 2.99, 3.99], [-2.99, 4.01, 4.01]), "force_n": [0.0, 0.0, -1.0]}]},
                           {"name": "modes", "analysis": "modal", "fixed_regions": fixed}],
            "point_masses": [{"name": "battery", "mass_g": 5.0, "attachment_region": box([-2.01, 1.99, 3.99], [2.01, 4.01, 4.01])}],
            "optimizer_settings": {"interface_node_policy": "allowed_adjacent"}}

def tiny_problem():
    problem = deepcopy(PROBLEM)
    problem["crash"].update(cases=["crash_front", "crash_side_left"], reference={"crash_front": 0.01, "crash_side_left": 0.01})
    problem["modal"].update(f1_min_hz=2000.0, modes=4, tracked=2, initial_iterations=120, warm_iterations=120)
    problem["shadow"].update(motors_mm=[[3.0, 2.0]], radius_mm=3.0, hub_radius_mm=0.5, limit_mm=0.5)
    problem["width"].update(minimum_mm=1.0)
    problem["continuation"]["beta_schedule"] = [2.0, 8.0]
    problem["monitor"] = ["twist"]
    problem.update(stiffness={**ARM_TIP, "min_n_per_mm": 50.0}, covariance=None)
    return problem

@pytest.fixture(scope="module")
def tiny():
    problem = TopologyProblem(tiny_domain(), tiny_problem())
    yield problem
    problem.close()

def test_orthotropic_reduces_to_isotropic():
    e, nu = 4430.0, 0.3
    iso = {"e_xy_mpa": e, "e_z_mpa": e, "nu_xy": nu, "nu_xz": nu, "g_xy_mpa": e / 2.6, "g_z_mpa": e / 2.6}
    assert np.allclose(orthotropic_matrix(iso) / e, elasticity_matrix(nu))
    stiff = hexahedron_matrices([1, 1, 1], nu, orthotropic_matrix(PRINT_MATERIAL["orthotropic"]) / e)[0]
    assert np.allclose(stiff, stiff.T) and np.linalg.eigvalsh(stiff)[6] > 0

def test_length_scale_and_filter_radius():
    assert 0.85 < length_scale_ratio(0.75) < 0.9
    assert filter_radius(PROBLEM["width"], [4 / 3] * 3) == pytest.approx(2.5 / length_scale_ratio(0.75))

def test_robust_projection_member_width():
    domain = cantilever_domain((48, 2, 2), 0.25)
    problem = cantilever_problem(1.0)
    problem["width"]["minimum_mm"] = 2.0
    problem["continuation"]["beta_schedule"] = [256.0]
    mapping = TopologyProblem(domain, problem).map
    widths = []
    for cells in range(1, 16):
        design = np.zeros((48, 2, 2))
        design[24 - cells // 2:24 - cells // 2 + cells] = 1
        fields, _ = mapping.fields(design.ravel())
        eroded, intermediate = (fields[name][0].reshape(48, 2, 2)[:, 0, 0] for name in ("eroded", "intermediate"))
        if eroded.max() > 0.5:
            widths.append(np.count_nonzero(intermediate > 0.5) * 0.25)
    assert widths and min(widths) >= 2.0 - 0.25

def test_radial_weight_and_disc_area():
    shadow = {"motors_mm": [[0.0, 0.0]], "radius_mm": 10.0, "hub_radius_mm": 2.0, "exponent": 2.0}
    h = 0.05
    axis = np.arange(-10, 10, h) + h / 2
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    assert radial_weight(points, shadow).sum() * h * h == pytest.approx(weighted_disc_area(shadow), rel=2e-3)
    assert radial_weight(np.array([[1.0, 0.0], [10.0, 0.0], [6.0, 0.0]]), shadow).tolist() == pytest.approx([0.0, 1.0, 0.25])
    solid = np.ones((400, 400, 3), dtype=bool)
    assert shadow_thickness(solid, [-10.0, -10.0, 0.0], h, shadow) == pytest.approx(3 * h, rel=2e-3)

def test_gradients_match_finite_differences(tiny):
    random = np.random.default_rng(3)
    design = random.uniform(0.3, 0.7, tiny.map.n)
    direction = random.standard_normal(tiny.map.n) * tiny.map.free
    result = tiny.evaluate(design)
    step = 1e-5
    plus, minus = tiny.evaluate(design + step * direction), tiny.evaluate(design - step * direction)
    assert result["names"] == ["arm_tip_stiffness", "crash_front", "crash_side_left", "f1", "volume", "shadow"]
    fd = (plus["objective"] - minus["objective"]) / (2 * step)
    assert result["objective_gradient"] @ direction == pytest.approx(fd, rel=1e-6)
    for name, g_plus, g_minus, gradient in zip(result["names"], plus["constraints"], minus["constraints"], result["constraint_gradients"]):
        fd = (g_plus - g_minus) / (2 * step)
        assert gradient @ direction == pytest.approx(fd, rel=1e-4, abs=1e-9), name

def test_continuation_report_and_monitor(tiny):
    design = np.full(tiny.map.n, 0.5)
    assert tiny.beta == 2.0 and not tiny.final_level
    assert tiny.advance() and tiny.beta == 8.0 and not tiny.advance()
    result = tiny.report(design)
    rows = {row["name"]: row for row in result["rows"]}
    assert rows["twist"]["status"] == "monitored" and rows["twist"]["value"] > 0
    assert rows["f1_intermediate"]["value"] > 0 and set(result["mass_by_field_g"]) == {"eroded", "intermediate", "dilated"}
    assert result["mass_by_field_g"]["eroded"] <= result["mass_by_field_g"]["intermediate"] <= result["mass_by_field_g"]["dilated"]
    assert "| volume |" in format_report(result["rows"])

def test_report_status():
    rows = [{"name": name, "value": 1.0, "limit": 1.0, "unit": "-", "sense": "<=", "field": "x", "g": g} for name, g in (("a", 0.005), ("b", -0.005), ("c", -0.5), ("d", None))]
    assert [row["status"] for row in constraint_report(rows)] == ["violated", "active", "satisfied", "monitored"]
    assert constraint_report(rows)[2]["margin"] == 0.5

def test_termination_needs_mass_and_violation():
    stop = Termination({"mass_change": 1e-3, "violation": 1e-3, "window": 3, "active": 0.01})
    assert not stop(10.0, 0.0) and not stop(10.0, 0.0)
    assert not stop(10.0, 0.0, final_level=False)
    assert not Termination({"mass_change": 1e-3, "violation": 1e-3, "window": 2, "active": 0.01})(1, 0) and stop(10.0, 0.0)
    late = Termination({"mass_change": 1e-3, "violation": 1e-3, "window": 2, "active": 0.01})
    late(10.0, 0.1)
    assert not late(10.0, 0.1)

def test_prolongate_keeps_masks_and_constants():
    coarse = cantilever_domain((8, 2, 4), 2.0)
    fine = cantilever_domain((16, 4, 8), 1.0)
    fine["preserve"][0, 0, 0] = True
    fine["allowed"][-1, -1, -1] = False
    fine["forbidden"] = ~fine["allowed"]
    result = prolongate(np.full(64, 0.3), coarse["grid"], fine).reshape(16, 4, 8)
    assert result[0, 0, 0] == 1 and result[-1, -1, -1] == 0 and np.allclose(result[1:-1, 1:-1, 1:-1], 0.3)
    ramp = np.indices((8, 2, 4))[0].astype(float).ravel() / 7
    assert np.all(np.diff(prolongate(ramp, coarse["grid"], cantilever_domain((16, 4, 8), 1.0)).reshape(16, 4, 8)[:, 1, 1]) >= 0)

def test_cantilever_fixture_stiffness_constraint():
    domain = cantilever_domain()
    problem = TopologyProblem(domain, cantilever_problem(1.0))
    solid = problem.evaluate(np.ones(problem.map.n))
    rows = {row["name"]: row for row in solid["rows"]}
    assert solid["names"] == ["tip_stiffness", "volume"] and rows["tip_stiffness"]["value"] > 1.0 and rows["volume"]["value"] == pytest.approx(1.0)
    assert solid["mass_g"] == pytest.approx(32 * 12 * 12 * 1.09 / 1000)
    problem.close()

@pytest.mark.parametrize("axis, modulus", [(0, PRINT_MATERIAL["orthotropic"]["e_xy_mpa"]), (1, PRINT_MATERIAL["orthotropic"]["e_xy_mpa"]), (2, PRINT_MATERIAL["orthotropic"]["e_z_mpa"])])
def test_orthotropic_bar_modulus(axis, modulus):
    from deep_frame.topology_optimization import HexElasticity
    shape = [2, 2, 2]
    shape[axis] = 12
    allowed = np.ones(shape, dtype=bool)
    end = np.asarray(shape, dtype=float)
    def face(at):
        low, high = np.full(3, -0.01), end + 0.01
        low[axis], high[axis] = at - 0.01, at + 0.01
        return box(low, high)
    force = np.zeros(3)
    force[axis] = 1.0
    nodes = [face(0.0)]
    domain = {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [1.0, 1.0, 1.0], "shape": shape, "axis_order": "xyz", "order": "C"}, "allowed": allowed, "preserve": np.zeros(shape, dtype=bool), "forbidden": ~allowed,
              "material": orthotropic_material({"density_g_cm3": 1.09, "poisson_ratio": 0.3}), "load_cases": [{"name": "bar", "analysis": "static", "fixed_regions": nodes, "loads": [{"region": face(end[axis]), "force_n": force.tolist()}]}]}
    system = HexElasticity(domain)
    stiffness = system.matrix(np.full(system.nelem, system.young))
    case = system.cases[0]
    clamp = (3 * np.flatnonzero(np.abs(system.points[:, axis]) < 1e-9)[:, None] + np.asarray([index for index in range(3) if index != axis])).ravel()
    free = np.setdiff1d(np.arange(system.ndof), np.union1d(clamp, 3 * np.flatnonzero(np.abs(system.points[:, axis]) < 1e-9) + axis))
    from scipy.sparse.linalg import spsolve
    displacement = np.zeros(system.ndof)
    displacement[free] = spsolve(stiffness[free][:, free].tocsc(), case["force"][free])
    tip = displacement[3 * np.flatnonzero(np.abs(system.points[:, axis] - end[axis]) < 1e-9) + axis].mean()
    assert 1.0 * 12 / (4 * tip) == pytest.approx(modulus, rel=0.04)

def test_cantilever_stiffness_gradient_matches_finite_differences():
    problem = TopologyProblem(cantilever_domain((12, 3, 4), 1.0), cantilever_problem(5.0))
    random = np.random.default_rng(5)
    design, direction = random.uniform(0.3, 0.7, problem.map.n), random.standard_normal(problem.map.n)
    result, step = problem.evaluate(design), 1e-5
    plus, minus = problem.evaluate(design + step * direction), problem.evaluate(design - step * direction)
    for index, name in enumerate(result["names"]):
        assert result["constraint_gradients"][index] @ direction == pytest.approx((plus["constraints"][index] - minus["constraints"][index]) / (2 * step), rel=1e-5), name
    problem.close()

def test_mma_min_mass_cantilever_ends_with_active_stiffness():
    pytest.importorskip("mmapy")
    domain = cantilever_domain((16, 4, 6), 1.0)
    solid = TopologyProblem(domain, cantilever_problem(1.0))
    stiffness = {row["name"]: row for row in solid.evaluate(np.ones(solid.map.n))["rows"]}["tip_stiffness"]["value"]
    solid.close()
    problem = cantilever_problem(0.4 * stiffness)
    problem["continuation"]["beta_schedule"] = [1.0, 4.0]
    optimizer = MMAOptimizer(TopologyProblem(domain, problem), {"level_max_iterations": 60, "final_max_iterations": 200, "move": 0.2})
    result = optimizer.run(np.ones(domain["allowed"].size))
    rows = {row["name"]: row for row in result["result"]["rows"]}
    assert result["status"] == "converged" and rows["tip_stiffness"]["status"] == "active"
    assert rows["volume"]["value"] < 0.8 and result["history"][-1]["max_violation"] <= 1e-3
    optimizer.problem.close()

def test_load_covariance_factorisation_and_physics():
    loads = load_covariance()
    sigma, directions, model = loads["sigma"], loads["directions"], loads["model"]
    assert sigma.shape == (42, 42) and len(loads["labels"]) == 42 and loads["labels"][0] == ("motor_front_left", "Fx") and loads["labels"][-1] == ("camera", "Mz")
    assert np.allclose(sigma, sigma.T) and np.allclose(directions @ directions.T, sigma, atol=1e-9 * np.abs(sigma).max())
    assert loads["rank"] == directions.shape[1] and loads["eigenvalues"][-1] >= 0
    rng = np.random.default_rng(0)
    root = rng.standard_normal((42, 42))
    flexibility = root @ root.T
    samples = model.scatter @ rng.standard_normal((model.scatter.shape[1], 200000)) + model.mean[:, None]
    assert np.isclose(np.trace(flexibility @ sigma), np.mean(np.einsum("ij,ij->j", samples, flexibility @ samples)), rtol=0.02)
    values, vectors = np.linalg.eigh(sigma)
    half = vectors @ np.diag(np.sqrt(np.clip(values, 0, None))) @ vectors.T
    assert np.isclose(np.linalg.eigvalsh(half @ flexibility @ half)[-1], np.linalg.eigvalsh(directions.T @ flexibility @ directions)[-1])
    index = {label: i for i, label in enumerate(loads["labels"])}
    assert loads["mean"][index["motor_front_left", "Mz"]] > 0 > loads["mean"][index["motor_front_right", "Mz"]] and loads["mean"][index["battery", "Fz"]] < 0
    assert model.correlation(("motor_front_left", "Fz"), ("battery", "Fz")) < 0
    assert all(sigma[index[name, dof], index[name, dof]] == 0 for name, dof in (("camera", "Mz"), ("battery", "Mz"), ("stack", "Mz")))

def test_load_covariance_diagonal_factor_is_antisymmetric():
    config = deepcopy(LOAD_COVARIANCE)
    config["factors"] = [factor for factor in config["factors"] if factor["name"] == "diagonal_fl_rr"]
    model = LoadCovariance(config)
    assert model.rank == 1 and np.isclose(model.correlation(("motor_front_left", "Fz"), ("motor_rear_right", "Fz")), -1.0)
    assert np.allclose(model.sigma[12:18], 0) and np.allclose(model.sigma[24:], 0)

def covariance_rows(problem, density):
    return {row["name"]: row for part in problem.physics(density)[:2] for row in part}

def test_covariance_half_model_matches_dense_full_model():
    from scipy.linalg import sqrtm
    from scipy.sparse.linalg import splu
    shape = (16, 3, 6)
    density = np.random.default_rng(3).uniform(0.3, 1.0, shape)
    half = TopologyProblem(*covariance_cantilever(shape))
    full = TopologyProblem(*covariance_cantilever(shape, full=True))
    mirrored = np.concatenate([density[:, ::-1, :], density], axis=1).ravel()
    system, model = full.system, full.covariance
    stiffness = system.matrix(system.young * (1e-6 + (1 - 1e-6) * mirrored ** 3)).tocsc()
    free = next(case for case in system.cases if case["name"] == "tip")["free"]
    loads = np.zeros((system.ndof, len(model.labels)))
    for column, (name, dof) in enumerate(model.labels):
        direct, values, _, _ = model.wrenches(full.domain["interfaces"][name])[LOAD_COVARIANCE["dofs"].index(dof)]
        np.add.at(loads[:, column], direct, values)
        nodal = loads[:, column].reshape(-1, 3)
        arms = system.points - full.domain["interfaces"][name]["reference_mm"]
        assert np.allclose(np.concatenate([nodal.sum(axis=0), np.cross(arms, nodal).sum(axis=0)]), np.eye(6)[LOAD_COVARIANCE["dofs"].index(dof)], atol=1e-10)
    displacement = np.zeros_like(loads)
    displacement[free] = splu(stiffness[free][:, free]).solve(loads[free])
    flexibility, sigma = loads.T @ displacement, model.sigma
    root = np.real(sqrtm(sigma))
    expected = {"load_mean": np.trace(flexibility @ sigma), "load_worst": np.linalg.eigvalsh(root @ flexibility @ root).max()}
    for problem, density in ((half, density.ravel()), (full, mirrored)):
        rows = covariance_rows(problem, density)
        for name, value in expected.items():
            assert rows[name]["value"] == pytest.approx(value, rel=1e-8), name
        assert rows["load_worst"]["info"]["ks_n_mm"] >= rows["load_worst"]["value"]
    half.close()
    full.close()

def test_covariance_rank_one_and_gradients_match_finite_differences():
    shape = (12, 3, 4)
    domain, problem = covariance_cantilever(shape, settings={"labels": [["tip", "Fz"], ["mid", "Fz"], ["tip", "Fy"]], "std": [1.0, 1.0, 0.5], "correlation": 1.0})
    problem["covariance"]["sigma"] = (np.outer([1.0, 1.0, 0.5], [1.0, 1.0, 0.5])).tolist()
    single = TopologyProblem(domain, problem)
    rows = covariance_rows(single, np.full(single.map.n, 0.6))
    assert single.covariance.rank == 1 and rows["load_worst"]["value"] == pytest.approx(rows["load_mean"]["value"], rel=1e-10)
    single.close()
    domain, problem = covariance_cantilever(shape, limits={"mean_n_mm": 0.05, "worst_n_mm": 0.04})
    problem["covariance"]["ks"] = 5.0
    tested = TopologyProblem(domain, problem)
    random = np.random.default_rng(5)
    design, direction = random.uniform(0.3, 0.7, tested.map.n), random.standard_normal(tested.map.n)
    result, step = tested.evaluate(design), 1e-5
    plus, minus = tested.evaluate(design + step * direction), tested.evaluate(design - step * direction)
    assert result["names"] == ["load_mean", "load_worst", "volume"]
    for index, name in enumerate(result["names"]):
        assert result["constraint_gradients"][index] @ direction == pytest.approx((plus["constraints"][index] - minus["constraints"][index]) / (2 * step), rel=1e-5), name
    tested.close()

def test_covariance_calibration_is_keyed_by_battery_support():
    from deep_frame.topology_problem import COVARIANCE
    assert set(COVARIANCE["limits"]["calibration"]) == {"rails", "free"}
    domain, problem = covariance_cantilever((12, 3, 4), limits={"mean_n_mm": 0.05, "worst_n_mm": 0.04, "calibration": {"rails": {"mean_n_mm": 2.0, "worst_n_mm": 3.0}, "free": {"mean_n_mm": 5.0, "worst_n_mm": 7.0}}})
    rails = TopologyProblem(domain, problem)
    rows = {row["name"]: row for row in rails.evaluate(np.full(rails.map.n, 0.5))["rows"]}
    assert rows["load_mean"]["limit"] == pytest.approx(0.1) and rows["load_worst"]["limit"] == pytest.approx(0.12)
    rails.covariance.bodies = {"battery": object()}
    assert rails.covariance.calibration() == {"mean_n_mm": 5.0, "worst_n_mm": 7.0}
    rails.close()

def test_covariance_placeholder_limits_are_monitored():
    domain, problem = covariance_cantilever((12, 3, 4))
    placeholder = TopologyProblem(domain, problem)
    result = placeholder.evaluate(np.full(placeholder.map.n, 0.5))
    rows = {row["name"]: row for row in result["rows"]}
    assert result["names"] == ["volume"] and rows["load_mean"]["status"] == rows["load_worst"]["status"] == "monitored"
    assert PROBLEM["stiffness"] is None and PROBLEM["width"]["minimum_mm"] == 2.5
    with pytest.raises(ValueError, match="interfaces"):
        TopologyProblem(cantilever_domain((12, 3, 4), 1.0), {**problem, "covariance": PROBLEM["covariance"]})
    placeholder.close()

def battery_domain():
    domain = tiny_domain()
    keep = box([-2.0, -2.0, 3.0], [2.0, 2.0, 6.0])
    centers = np.stack(np.meshgrid(np.arange(4) + 0.5, np.arange(8) - 3.5, np.arange(4) + 0.5, indexing="ij"), -1)
    inside = (np.abs(centers[..., 0]) < 2) & (np.abs(centers[..., 1]) < 2) & (centers[..., 2] > 3)
    domain.update(allowed=~inside, preserve=np.zeros(inside.shape, dtype=bool), forbidden=inside, point_masses=[])
    body = {"name": "battery", "mass_g": 10.0, "position_mm": [0.0, 0.0, 4.5], "inertia_g_mm2": (np.eye(3) * 20.0).tolist(), "force_n": [0.0, 0.0, 0.0]}
    for case in domain["load_cases"]:
        if "inertia_relief" in case:
            case["inertia_relief"] = {"point_masses": [], "preserve_mass_g": 0.0, "bodies": [{**body, "force_n": [0.0, 0.0, -1.0] if case["name"] == "crash_side_left" else [0.0, 0.0, 0.0]}]}
    domain["battery"] = {"keep_out": keep, "reference_mm": [0.0, 0.0, 3.0], "center_mm": [0.0, 0.0, 4.5], "mass_g": 10.0}
    problem = tiny_problem()
    problem["battery"] = {**deepcopy(BATTERY_SUPPORT), "pad_normal_n_mm3": 50.0, "pad_shear_n_mm3": 20.0, "min_area_mm2": 4.0, "limit_mm": 0.01, "band_preload_n": 0.5}
    problem["shadow"] = None
    return domain, problem

def test_free_battery_support_equilibrium_and_gradients():
    tested = TopologyProblem(*battery_domain())
    battery = tested.battery
    assert battery.summary["bottom_nodes"] and battery.summary["side_nodes"] and battery.summary["bottom_area_mm2"] == pytest.approx(16.0)
    random = np.random.default_rng(4)
    design, direction = random.uniform(0.3, 0.7, tested.map.n), random.standard_normal(tested.map.n) * tested.map.free
    tested.battery_sigma = np.diag([1.0, 2.0, 1.0, 3.0, 5.0, 1.0]) + 0.3
    result, step = tested.evaluate(design), 1e-5
    wrench = np.array([0.3, -0.2, -1.0, 0.1, 0.2, -0.1])
    state = battery.state(wrench, False)
    nodal = state["stiffness"] * np.einsum("ndi,i->nd", battery.rows, state["motion"])
    assert np.allclose(np.einsum("ndi,nd->i", battery.rows, nodal), wrench)
    assert {"battery_shift_flight", "battery_shift_crash_front", "battery_shift_crash_side_left", "battery_contact_area"} <= set(result["names"])
    plus, minus = tested.evaluate(design + step * direction), tested.evaluate(design - step * direction)
    for index, name in enumerate(result["names"]):
        assert result["constraint_gradients"][index] @ direction == pytest.approx((plus["constraints"][index] - minus["constraints"][index]) / (2 * step), rel=1e-4, abs=1e-9), name
    lumped = tested.modal.lumped - tested.modal.base_lumped
    assert lumped.sum() / 3 == pytest.approx(10.0e-6 / 2)
    tested.close()

def camera_domain():
    domain = tiny_domain()
    keep = box([-1.0, 1.0, 1.0], [1.0, 3.0, 3.0])
    centers = np.stack(np.meshgrid(np.arange(4) + 0.5, np.arange(8) - 3.5, np.arange(4) + 0.5, indexing="ij"), -1)
    inside = (centers[..., 0] < 1) & (np.abs(centers[..., 1] - 2) < 1) & (np.abs(centers[..., 2] - 2) < 1)
    domain.update(allowed=~inside, preserve=np.zeros(inside.shape, dtype=bool), forbidden=inside)
    zone = box([-3.0, 2.0, 2.0], [3.0, 4.0, 4.0])
    zones = {"crash_front": [0.0, -2.0, 0.0], "crash_camera_oblique": [1.4, -1.4, 0.0]}
    cases = {case["name"]: case for case in domain["load_cases"]}
    domain["load_cases"].insert(2, {**deepcopy(cases["crash_front"]), "name": "crash_camera_oblique"})
    body = {"name": "camera", "mass_g": 2.0, "position_mm": [0.0, 2.0, 2.0], "inertia_g_mm2": (np.eye(3) * 5.0).tolist(), "force_n": [0.0, 0.0, 0.0]}
    for case in domain["load_cases"]:
        if "inertia_relief" in case:
            case["inertia_relief"] = {**deepcopy(case["inertia_relief"]), "bodies": [{**deepcopy(body), "force_n": zones.get(case["name"], body["force_n"])}]}
        if case["name"] in zones:
            case["loads"] = []
    tilt = np.radians(20.0)
    axis, up = [0.0, np.cos(tilt), np.sin(tilt)], [0.0, -np.sin(tilt), np.cos(tilt)]
    domain["camera"] = {"keep_out": keep, "reference_mm": [0.0, 2.0, 2.0], "center_mm": [0.0, 2.0, 2.0], "mass_g": 2.0, "screw_axis_yz_mm": [2.0, 2.0], "zone": zone, "zones": zones,
                        "displacement_cases": ["crash_front", "crash_camera_oblique"], "reference_regions": [box([0.0, -4.0, 0.0], [2.0, -2.0, 2.0])],
                        "coverage": {"axes": [axis, [1.0, 0.0, 0.0], up], "front_mm": [0.0, 3.0, 2.0], "window_mm": [[-2.0, 2.0], [-1.0, 2.0]], "silhouette_mm": [[-1.0, 1.0], [-1.0, 1.0]], "step_mm": 0.5, "depth_mm": [0.0, 4.0], "frontal_area_mm2": 4.0}}
    problem = tiny_problem()
    problem["camera"] = {**deepcopy(CAMERA_SUPPORT), "patch_radius_mm": 1.0, "limit_mm": 0.01, "min_area_mm2": 1.0, "min_coverage": 0.2, "crash_reference": {"crash_front": 0.01}}
    problem["shadow"] = None
    return domain, problem

def external_split(zone, physical):
    _, _, caught, passed = zone.split(physical)
    frame = (zone.direct_map @ caught.ravel() + zone.mirror_map @ caught.ravel()).sum() * zone.force
    return frame, passed.sum() * zone.force

def test_shielded_impact_conserves_the_crash_force():
    domain, problem = camera_domain()
    tested = TopologyProblem(domain, problem)
    settings = {**problem["camera"], "window_mm": {"side": 1.0, "top": 1.0}, "protection_depth_mm": [0.0, 1.0]}
    zone = ShieldedImpact(tested.system, tested.zones["crash_front"].case, [0.0, -2.0, 0.0], domain["camera"]["keep_out"], settings, 0.5)
    assert zone.spec["origin_mm"] == pytest.approx([0.0, 3.0, 2.0]) and zone.rays.rays == 32
    empty, full = np.zeros(tested.system.nelem), np.asarray(domain["allowed"], dtype=float).ravel()
    frame, camera = external_split(zone, empty)
    assert zone.shielding(empty) == 0 and np.allclose(frame, 0) and np.allclose(camera, zone.force)
    frame, camera = external_split(zone, full)
    shield = zone.shielding(full)
    assert shield > 0.8 and np.allclose(frame, shield * zone.force) and np.allclose(camera, (1 - shield) * zone.force)
    for physical in (empty, full):
        direct, mirrored, loads = zone.loads(physical)
        total = direct.reshape(-1, 3).sum(axis=0) + (mirrored.reshape(-1, 3) * tested.system._flip()).sum(axis=0) + sum(np.asarray(load["force_n"]) for load in loads)
        assert np.allclose(total, 0.0, atol=1e-9)
    tested.close()

def test_front_coverage_counts_only_material_within_the_protection_depth():
    shape = (8, 20, 8)
    domain = {"grid": {"origin_mm": [-4.0, 0.0, 0.0], "spacing_mm": [1.0, 1.0, 1.0], "shape": list(shape)}, "allowed": np.ones(shape, dtype=bool)}
    spec = {"front_mm": [0.0, 2.0, 4.0], "axes": [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], "window_mm": [[-3.0, 3.0], [-3.0, 3.0]], "silhouette_mm": [[-1.0, 1.0], [-1.0, 1.0]],
            "step_mm": 0.5, "depth_mm": [0.0, 6.0], "frontal_area_mm2": 4.0}
    far, near = np.zeros(shape), np.zeros(shape)
    far[:, 13:15] = 1.0
    near[:, 4:6] = 1.0
    coverage = FrontCoverage(domain, spec)
    assert coverage.measure(far.ravel())[0] == 0 and np.allclose(coverage.measure(far.ravel())[1].reshape(shape)[:, 10:], 0)
    assert coverage.measure(near.ravel())[0] == pytest.approx(32.0 / 4.0)
    assert FrontCoverage(domain, {**spec, "depth_mm": [0.0, 15.0]}).measure(far.ravel())[0] == pytest.approx(32.0 / 4.0)

def test_free_camera_support_equilibrium_and_gradients():
    tested = TopologyProblem(*camera_domain())
    camera, zone = tested.camera, tested.zones["crash_camera_oblique"]
    assert camera.summary["nodes"] == 10 and set(tested.zones) == {"crash_front", "crash_camera_oblique"}
    random = np.random.default_rng(5)
    design, direction = random.uniform(0.3, 0.7, tested.map.n), random.standard_normal(tested.map.n) * tested.map.free
    result, step = tested.evaluate(design), 1e-5
    force, mirrored, loads = zone.loads(tested.physical)
    flip = tested.system._flip()
    total = force.reshape(-1, 3).sum(axis=0) + (mirrored.reshape(-1, 3) * flip).sum(axis=0) + sum(np.asarray(load["force_n"]) for load in loads)
    assert np.allclose(total, 0.0, atol=1e-9)
    assert {"crash_front", "camera_shift_crash_front", "camera_shift_crash_camera_oblique", "camera_mount_area", "camera_coverage"} <= set(result["names"])
    plus, minus = tested.evaluate(design + step * direction), tested.evaluate(design - step * direction)
    for index, name in enumerate(result["names"]):
        assert result["constraint_gradients"][index] @ direction == pytest.approx((plus["constraints"][index] - minus["constraints"][index]) / (2 * step), rel=1e-4, abs=1e-9), name
    lumped = tested.modal.lumped - tested.modal.base_lumped
    assert lumped.sum() / 3 == pytest.approx(2.0e-6 / 2)
    tested.close()

def test_shared_static_factorization_matches_own_supports():
    systems = [HexElasticity(tiny_domain(), share_static=share) for share in (False, True)]
    density = np.random.default_rng(3).uniform(0.2, 1.0, systems[0].nelem)
    own, shared = (system.solve(density, metrics=True) for system in systems)
    assert systems[1].diagnostics()["factorization_groups"] == 2 and systems[0].diagnostics()["factorization_groups"] == 4
    stiffness = systems[1].matrix(np.full(systems[1].nelem, 1.0))
    for sign in (1.0, -1.0):
        modes = systems[1]._rigid_modes(sign)
        assert modes.shape[1] == 3 and np.abs(stiffness @ modes).max() < 1e-9 * np.abs(stiffness).max()
    assert own.keys() == shared.keys()
    for name in own:
        assert shared[name]["compliance_n_mm"] == pytest.approx(own[name]["compliance_n_mm"], rel=1e-10)
        assert np.linalg.norm(shared[name]["derivative"] - own[name]["derivative"]) <= 1e-10 * np.linalg.norm(own[name]["derivative"])
        assert shared[name]["max_displacement_mm"] == pytest.approx(own[name]["max_displacement_mm"], rel=1e-10)

def test_shared_static_falls_back_to_own_factorization_when_inaccurate():
    own, shared = (HexElasticity(tiny_domain(), share_static=share) for share in (False, True))
    rng = np.random.default_rng(3)
    density = rng.uniform(0.2, 1.0, own.nelem)
    shared.shared = shared._share_plan()
    for entry in shared.shared.values():
        entry["modes"] = entry["modes"] * (1 + 1e-3 * rng.standard_normal(entry["modes"].shape))
    exact, tested = own.solve(density), shared.solve(density)
    assert len(shared.unshared) == 2 and shared.shared is None
    for name in exact:
        assert tested[name]["compliance_n_mm"] == pytest.approx(exact[name]["compliance_n_mm"], rel=1e-10)
    shared.solve(density)
    assert shared.diagnostics()["factorization_groups"] == 4 and shared.diagnostics()["unshared_static_groups"] == 2

@pytest.mark.parametrize("half", [False, True])
def test_stand_reserve_box_on_four_feet(half):
    domain = stand_domain(half)
    rows = {row["name"]: row for row in StandStability(domain, {"enabled": True}).rows(stand_design(domain, FOUR_FEET))}
    assert 18.0 < rows["stand_reserve"]["value"] <= 19.5 and rows["stand_reserve"]["g"] < 0
    assert rows["stand_ground_z"]["value"] == pytest.approx(0.0, abs=0.01)
    assert rows["stand_prop_clearance"]["value"] == pytest.approx(20.0 - COMPONENT_DEFAULTS["prop"]["thickness_mm"] / 2, abs=0.01)

def test_stand_reserve_violated_on_one_sided_or_tilted_feet():
    domain = stand_domain(True)
    stability = StandStability(domain, {"enabled": True})
    raised = stability.rows(stand_design(domain, [(sx * 18, 18, 0) for sx in (-1, 1)] + [(sx * 18, -18, 3) for sx in (-1, 1)]))[0]
    assert raised["value"] < -15.0 and raised["g"] > 1
    heavy = StandStability({**domain, "metadata": {"components": {"battery": {"center_of_mass_mm": [0.0, 30.0, 20.0], "mass_g": 200.0}}}}, {"enabled": True})
    shifted = heavy.rows(stand_design(domain, FOUR_FEET))[0]
    assert shifted["value"] < 0 and shifted["g"] > 1 and shifted["info"]["center_of_gravity_xy_mm"][1] > 20

def test_stand_prop_clearance_and_gradients():
    domain = stand_domain(True, components={"prop_a": {"center_of_mass_mm": [10.0, 0.0, 14.5], "mass_g": 1.0}})
    stability = StandStability(domain, {"enabled": True, "prop_clearance_min_mm": 17.0})
    design = stand_design(domain, [(sx * 18, sy * 18, 4) for sx in (-1, 1) for sy in (-1, 1)])
    rows = stability.rows(design)
    assert rows[1]["name"] == "stand_prop_clearance" and rows[1]["value"] == pytest.approx(8.0, abs=0.01) and rows[1]["g"] == pytest.approx(1 - 8.0 / 17.0, abs=1e-3)
    random = np.random.default_rng(3)
    design = np.clip(0.8 * design + random.uniform(0, 0.2, design.size), 0, 1)
    direction, step = random.standard_normal(design.size), 1e-5
    rows, plus, minus = stability.rows(design), stability.rows(design + step * direction), stability.rows(design - step * direction)
    for row, high, low in zip(rows, plus, minus):
        key = "value" if row["g"] is None else "g"
        assert row["gradient"] @ direction == pytest.approx((high[key] - low[key]) / (2 * step), rel=1e-4)

def test_stand_rows_in_the_shared_formulation():
    domain = cantilever_domain((12, 3, 4), 1.0)
    domain["metadata"] = {"components": {"prop_a": {"center_of_mass_mm": [6.0, 0.0, 30.0], "mass_g": 1.0}}}
    problem = cantilever_problem(1.0)
    off = TopologyProblem(domain, problem)
    assert off.stability is None and not any(name.startswith("stand_") for name in off.evaluate(np.full(off.map.n, 0.5))["names"])
    off.close()
    on = TopologyProblem(domain, {**problem, "stability": {"enabled": True, "reserve_min_mm": 1.0, "prop_clearance_min_mm": 17.0}})
    result = on.evaluate(np.full(on.map.n, 0.5))
    assert result["names"][-2:] == ["stand_reserve", "stand_prop_clearance"] and any(row["name"] == "stand_ground_z" and row["status"] == "monitored" for row in result["rows"])
    on.close()
    gated = {row["name"]: row for row in on.geometry(np.full(on.map.n, 0.5)) if row["name"].startswith("stand_")}
    assert on.beta < on.stability.settings["start_beta"] and gated["stand_reserve"]["g"] == -1.0 and not np.any(gated["stand_reserve"]["gradient"]) and gated["stand_ground_z"]["g"] is None
    on.map.beta = on.stability.settings["start_beta"]
    live = {row["name"]: row for row in on.geometry(np.full(on.map.n, 0.5)) if row["name"].startswith("stand_")}
    assert np.any(live["stand_reserve"]["gradient"]) and live["stand_reserve"]["value"] == gated["stand_reserve"]["value"]
    on.close()
