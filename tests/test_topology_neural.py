import numpy as np
import pytest

from deep_frame.topology_neural import NeuralAugmentedLagrangian, NeuralDensity, NeuralDesign, neural_al_settings, neural_settings, optimize_neural
from deep_frame.topology_optimization import AugmentedLagrangian, HexElasticity
from deep_frame.topology_problem import TopologyProblem
from tests.test_topology_optimization import beam_domain
from tests.test_topology_problem import tiny_domain, tiny_problem

def holed_beam(shape=(12, 4, 4)):
    domain = beam_domain(shape)
    domain["allowed"][5:7, 1:3, :] = False
    domain["forbidden"] = ~domain["allowed"]
    return domain

@pytest.mark.parametrize("discs", [None, {"mode": "soft", "motors_mm": [[6.0, 3.0]], "radius_mm": 4.0, "plane_mm": 6.0, "weight": 3.0, "length_mm": 2.0}])
def test_network_sensitivities_match_finite_differences(discs):
    domain = beam_domain((6, 3, 3))
    settings = neural_settings({"frequencies": 8, "hidden": [6], "mirror_axis": None, "max_frequency_per_mm": 0.2, "volume_fraction": 0.5, "prop_discs": discs})
    mapping = NeuralDensity(domain, settings)
    system = HexElasticity(domain)
    assert np.dot(mapping.weights, mapping.physical(2.0)[0][mapping.free]) == pytest.approx(mapping.budget, rel=1e-9)
    assert (mapping.weights.max() > 2.0) == (discs is not None)
    def objective():
        physical, cache = mapping.physical(2.0)
        solutions = system.solve(physical)
        return sum(result["compliance_n_mm"] for result in solutions.values()), sum(result["derivative"] for result in solutions.values()), cache
    _, derivative, cache = objective()
    gradients = mapping.gradient(cache, derivative)
    random, scale = np.random.default_rng(3), max(np.max(np.abs(value)) for value in gradients)
    for index in (0, 1, 2, 3):
        value, gradient = mapping.field.parameters[index], gradients[index]
        for flat in random.choice(value.size, min(3, value.size), replace=False):
            position = np.unravel_index(flat, value.shape)
            original = value[position]
            step = 1e-6 * max(1.0, abs(original))
            value[position] = original + step
            upper = objective()[0]
            value[position] = original - step
            lower = objective()[0]
            value[position] = original
            assert np.isclose((upper - lower) / (2 * step), gradient[position], rtol=1e-4, atol=1e-6 * scale)

def test_mirrored_coordinates_give_exactly_symmetric_density():
    domain = beam_domain((8, 4, 4))
    domain["grid"]["origin_mm"] = [-8.0, -3.0, 0.0]
    mirrored = NeuralDensity(domain, neural_settings({"mirror_axis": 0, "volume_fraction": 0.5})).physical(3.0)[0].reshape(8, 4, 4)
    assert np.array_equal(mirrored, mirrored[::-1]) and not np.array_equal(mirrored, mirrored[:, ::-1])
    plain = NeuralDensity(domain, neural_settings({"mirror_axis": None, "volume_fraction": 0.5})).physical(3.0)[0].reshape(8, 4, 4)
    assert not np.allclose(plain, plain[::-1])

def test_masks_hold_every_iteration_and_in_the_fine_sample():
    domain, fine = holed_beam(), holed_beam((24, 8, 8))
    fine["grid"]["spacing_mm"] = [1.0, 1.0, 1.0]
    seen = []
    def progress(entry):
        seen.append(entry["iteration"])
    result = optimize_neural(domain, {"volume_fraction": 0.4, "max_iterations": 6, "minimum_iterations": 2, "sharpness_iterations": 3}, progress_callback=progress, output_domain=fine)
    assert result["status"] == "ok" and len(seen) == 7
    for density, case in ((result["optimization_density"], domain), (result["density"], fine)):
        assert np.all(density[case["preserve"]] == 1) and np.all(density[case["forbidden"]] == 0)

def test_cantilever_reduces_compliance_and_meets_volume():
    domain = beam_domain((16, 4, 4))
    result = optimize_neural(domain, {"volume_fraction": 0.4, "max_iterations": 80, "minimum_iterations": 30, "sharpness_iterations": 30, "learning_rate": 0.02, "mirror_axis": None})
    summary = result["summary"]
    assert result["status"] == "ok"
    assert summary["objective_final"] < 0.3 * summary["objective_initial"] and summary["converged"]
    assert abs(summary["volume_fraction"] - 0.4) < 1e-6
    assert summary["gray_fraction_free"] < 0.1

def test_local_volume_penalty_gradient_and_blob_vs_strut():
    from deep_frame.topology_neural import LocalVolumePenalty
    domain = beam_domain((14, 10, 10))
    domain["preserve"][:] = False
    penalty = LocalVolumePenalty(domain, neural_settings({"max_width_penalty": 1.0, "max_width_window_mm": 5.0, "max_local_fraction": 0.3, "volume_fraction": 0.2}))
    blob, strut = np.zeros((14, 10, 10)), np.zeros((14, 10, 10))
    blob[4:10, 3:7, 3:7], strut[:, 5, 5] = 1, 1
    assert penalty(strut.ravel())[0] == 0 and penalty(blob.ravel())[0] > 0
    density = np.random.default_rng(1).uniform(0.2, 0.9, blob.size)
    value, gradient = penalty(density)
    for index in (17, 700, 1111):
        step = np.zeros_like(density)
        step[index] = 1e-6
        assert np.isclose((penalty(density + step)[0] - penalty(density - step)[0]) / 2e-6, gradient[index], rtol=1e-5, atol=1e-9)

def test_augmented_lagrangian_vector_and_penalty_rule():
    settings = {"penalty": 10.0, "multiplier_interval": 1, "penalty_growth": 2.0, "penalty_progress": 0.25, "penalty_max": 40.0}
    scalar, vector = AugmentedLagrangian({"penalty": 10.0, "multiplier_interval": 1}), AugmentedLagrangian(settings, 2)
    assert scalar.augment(0.3, 2.0) == pytest.approx((10 / 2 * 0.3 ** 2, 10 * 0.3 * 2.0, 0.0)) and scalar.multiplier == pytest.approx(3.0)
    value, gradient, weights = vector.terms([0.3, -0.5], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert value == pytest.approx(0.45) and gradient.tolist() == pytest.approx([3.0, 0.0]) and weights.tolist() == pytest.approx([3.0, 0.0])
    vector.update([0.3, -0.5])
    assert vector.multiplier.tolist() == pytest.approx([3.0, 0.0]) and vector.penalty.tolist() == [10.0, 10.0]
    vector.update([0.2, -0.5])
    assert vector.penalty.tolist() == [20.0, 10.0]
    vector.update([0.01, -0.5])
    vector.update([0.01, -0.5])
    vector.update([0.01, -0.5])
    assert vector.penalty.tolist() == [40.0, 10.0]

def test_neural_al_lagrangian_gradient_matches_finite_differences():
    problem = TopologyProblem(tiny_domain(), tiny_problem())
    design = NeuralDesign(neural_al_settings({"frequencies": 8, "hidden": [6], "max_frequency_per_mm": 0.3})).attach(tiny_domain())
    multipliers = AugmentedLagrangian({"penalty": 5.0, "multiplier_interval": 1}, 6)
    multipliers.multiplier[:] = [0.4, 0.2, 0.1, 0.3, 0.0, 0.2]
    def lagrangian():
        x, cache = design.design()
        result = problem.evaluate(x)
        value, gradient, _ = multipliers.terms(result["constraints"], result["constraint_gradients"])
        return result["objective"] + value, result["objective_gradient"] + gradient, cache
    _, derivative, cache = lagrangian()
    gradients = design.gradient(cache, derivative)
    random = np.random.default_rng(5)
    direction = [random.standard_normal(value.shape) for value in design.field.parameters]
    step = 1e-6
    def shifted(scale):
        for value, delta in zip(design.field.parameters, direction):
            value += scale * step * delta
        result = lagrangian()[0]
        for value, delta in zip(design.field.parameters, direction):
            value -= scale * step * delta
        return result
    upper, lower = shifted(1.0), shifted(-1.0)
    problem.close()
    assert sum(np.sum(g * d) for g, d in zip(gradients, direction)) == pytest.approx((upper - lower) / (2 * step), rel=1e-4)

def test_neural_al_fit_and_guarded_run():
    domain = tiny_domain()
    target = np.where(np.indices(tuple(domain["grid"]["shape"]))[2].ravel() >= 2, 0.9, 0.1)
    optimizer = NeuralAugmentedLagrangian({"frequencies": 8, "hidden": [8], "max_frequency_per_mm": 0.3, "fit": {"iterations": 400, "learning_rate": 0.05}, "max_iterations": 6, "al": {"multiplier_interval": 2},
                                           "level": {"minimum_iterations": 3, "maximum_iterations": 3}})
    result = optimizer.run([(domain, lambda: TopologyProblem(domain, tiny_problem()))], start=target)
    summary = result["summary"]
    assert summary["fit"]["mean_abs_error"] < 0.1 and summary["iterations"] == 6 and summary["stop_reason"] == "guard" and not summary["converged"]
    assert [entry["beta"] for entry in result["history"]] == [2.0] * 3 + [8.0] * 3 and len(summary["level_reports"]) == 1
    assert result["density"].shape == tuple(domain["grid"]["shape"]) and np.all(result["density"][domain["preserve"]] == 1)
    assert any(value > 0 for value in summary["multipliers"]) and len(summary["final"]["rows"]) > 6
