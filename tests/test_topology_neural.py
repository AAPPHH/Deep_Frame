import numpy as np

from deep_frame.topology_neural import NeuralDensity, neural_settings, optimize_neural
from deep_frame.topology_optimization import HexElasticity
from tests.test_topology_optimization import beam_domain

def holed_beam(shape=(12, 4, 4)):
    domain = beam_domain(shape)
    domain["allowed"][5:7, 1:3, :] = False
    domain["forbidden"] = ~domain["allowed"]
    return domain

def test_network_sensitivities_match_finite_differences():
    domain = beam_domain((6, 3, 3))
    settings = neural_settings({"frequencies": 8, "hidden": [6], "mirror_axis": None, "max_frequency_per_mm": 0.2, "volume_fraction": 0.5})
    mapping = NeuralDensity(domain, settings)
    system = HexElasticity(domain)
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
