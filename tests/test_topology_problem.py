from copy import deepcopy
import numpy as np
import pytest

from deep_frame.config import PRINT_MATERIAL
from deep_frame.topology_optimization import elasticity_matrix, hexahedron_matrices, orthotropic_matrix
from deep_frame.topology_problem import PROBLEM, Termination, TopologyProblem, cantilever_domain, cantilever_problem, constraint_report, filter_radius, format_report, length_scale_ratio, orthotropic_material, prolongate, radial_weight, shadow_thickness, weighted_disc_area

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
    problem["stiffness"]["min_n_per_mm"] = 50.0
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
    assert filter_radius(PROBLEM["width"], [4 / 3] * 3) == pytest.approx(2.0 / length_scale_ratio(0.75))

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
