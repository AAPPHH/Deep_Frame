from copy import deepcopy
from time import perf_counter

import numpy as np

from deep_frame.topology_optimization import HexElasticity, _case_scaling, _history_entry, validate_masks

NEURAL_SETTINGS = {
    "volume_fraction": 0.12,
    "penalization": 3.0,
    "min_stiffness_ratio": 1e-6,
    "max_iterations": 150,
    "minimum_iterations": 40,
    "change_tolerance": 0.003,
    "objective_window": 10,
    "learning_rate": 0.01,
    "frequencies": 96,
    "max_frequency_per_mm": 0.125,
    "hidden": [48, 48],
    "seed": 0,
    "mirror_axis": 0,
    "sharpness_final": 4.0,
    "sharpness_iterations": 80,
    "case_weights": {},
    "interface_node_policy": "allowed_adjacent",
    "linear_solver": "cpu_superlu",
    "gpu_solver_residency": "resident",
    "max_runtime_s": None,
}

def neural_settings(settings):
    result = deepcopy(NEURAL_SETTINGS)
    unknown = set(settings) - set(result)
    if unknown:
        raise ValueError("Unknown neural settings: " + ", ".join(sorted(unknown)))
    result.update(deepcopy(settings))
    if not 0 < result["volume_fraction"] < 1 or result["max_frequency_per_mm"] <= 0 or result["frequencies"] < 1 or result["learning_rate"] <= 0:
        raise ValueError("Invalid neural topology settings")
    if result["mirror_axis"] not in (None, 0, 1, 2):
        raise ValueError("mirror_axis must be None, 0, 1 or 2")
    result["minimum_iterations"] = min(result["minimum_iterations"], result["max_iterations"])
    return result

def cell_centers(grid):
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    return np.asarray(grid.get("origin_mm", [0.0, 0.0, 0.0]), dtype=float) + (np.indices(tuple(grid["shape"])).reshape(3, -1).T + 0.5) * spacing

class FourierField:
    def __init__(self, settings):
        random = np.random.default_rng(settings["seed"])
        directions = random.normal(size=(3, settings["frequencies"]))
        directions /= np.linalg.norm(directions, axis=0)
        self.wavenumbers = 2 * np.pi * directions * random.uniform(0, settings["max_frequency_per_mm"], settings["frequencies"])
        self.mirror_axis = settings["mirror_axis"]
        sizes = [2 * settings["frequencies"], *settings["hidden"], 1]
        self.parameters = [random.normal(0, np.sqrt(2 / rows), (rows, columns)) for rows, columns in zip(sizes[:-1], sizes[1:])]
        self.parameters[-1] *= 0.1
        fraction = settings["volume_fraction"]
        self.parameters += [np.zeros(columns) for columns in sizes[1:]]
        self.parameters[-1][:] = np.log(fraction / (1 - fraction))
    def features(self, points):
        points = np.array(points, dtype=float)
        if self.mirror_axis is not None:
            points[:, self.mirror_axis] = np.abs(points[:, self.mirror_axis])
        phase = points @ self.wavenumbers
        return np.hstack([np.sin(phase), np.cos(phase)])
    def forward(self, features):
        layers = len(self.parameters) // 2
        activations = [features]
        for index in range(layers):
            value = activations[-1] @ self.parameters[index] + self.parameters[layers + index]
            activations.append(np.where(value > 0, value, 0.01 * value) if index < layers - 1 else value)
        return activations[-1][:, 0], activations
    def backward(self, activations, gradient):
        layers = len(self.parameters) // 2
        upstream = np.asarray(gradient)[:, None]
        weights, biases = [None] * layers, [None] * layers
        for index in reversed(range(layers)):
            weights[index] = activations[index].T @ upstream
            biases[index] = upstream.sum(axis=0)
            if index:
                upstream = (upstream @ self.parameters[index].T) * np.where(activations[index] > 0, 1.0, 0.01)
        return weights + biases

class Adam:
    def __init__(self, parameters, rate, beta1=0.9, beta2=0.999):
        self.rate, self.beta1, self.beta2, self.count = rate, beta1, beta2, 0
        self.first = [np.zeros_like(value) for value in parameters]
        self.second = [np.zeros_like(value) for value in parameters]
    def step(self, parameters, gradients):
        self.count += 1
        for value, gradient, first, second in zip(parameters, gradients, self.first, self.second):
            first *= self.beta1
            first += (1 - self.beta1) * gradient
            second *= self.beta2
            second += (1 - self.beta2) * gradient ** 2
            value -= self.rate * first / (1 - self.beta1 ** self.count) / (np.sqrt(second / (1 - self.beta2 ** self.count)) + 1e-8)

def _sigmoid(value):
    return 0.5 * (1 + np.tanh(0.5 * value))

def volume_shift(logits, sharpness, budget):
    lower, upper = -60.0 / sharpness - np.max(logits), 60.0 / sharpness - np.min(logits)
    if not 0 < budget < len(logits):
        raise ValueError("The volume budget must exceed the preserve cells and stay below the allowed cells")
    for _ in range(100):
        middle = (lower + upper) / 2
        if np.sum(_sigmoid(sharpness * (logits + middle))) > budget:
            upper = middle
        else:
            lower = middle
    return (lower + upper) / 2

class NeuralDensity:
    def __init__(self, domain, settings):
        self.field = FourierField(settings)
        self.allowed, self.preserve, self.forbidden = validate_masks(domain)
        self.free = self.allowed & ~self.preserve
        self.budget = settings["volume_fraction"] * np.count_nonzero(self.allowed) - np.count_nonzero(self.preserve)
        self.features = self.field.features(cell_centers(domain["grid"])[self.free])
    def physical(self, sharpness=1.0):
        logits, activations = self.field.forward(self.features)
        density = _sigmoid(sharpness * (logits + volume_shift(logits, sharpness, self.budget)))
        physical = self.preserve.astype(float)
        physical[self.free] = density
        return physical, (activations, sharpness * density * (1 - density))
    def gradient(self, cache, physical_gradient):
        activations, slope = cache
        gradient = np.asarray(physical_gradient)[self.free]
        return self.field.backward(activations, slope * (gradient - np.dot(gradient, slope) / max(np.sum(slope), 1e-300)))
    def sample(self, domain, sharpness, fraction):
        allowed, preserve, _ = validate_masks(domain)
        free = allowed & ~preserve
        logits = self.field.forward(self.field.features(cell_centers(domain["grid"])[free]))[0]
        density = preserve.astype(float)
        density[free] = _sigmoid(sharpness * (logits + volume_shift(logits, sharpness, fraction * np.count_nonzero(allowed) - np.count_nonzero(preserve))))
        return density.reshape(tuple(domain["grid"]["shape"]))

def optimize_neural(domain, settings, *, progress_callback=None, output_domain=None):
    started = perf_counter()
    history, diagnostics, system, outcome = [], [], None, None
    density = np.zeros(tuple(domain.get("grid", {}).get("shape", (0, 0, 0))))
    try:
        settings = neural_settings(settings)
        mapping = NeuralDensity(domain, settings)
        allowed_count = int(np.count_nonzero(mapping.allowed))
        target = settings["volume_fraction"]
        system = HexElasticity(domain, interface_node_policy=settings["interface_node_policy"], linear_solver=settings["linear_solver"], gpu_solver_residency=settings["gpu_solver_residency"])
        optimizer = Adam(mapping.field.parameters, settings["learning_rate"])
        scales = None
        converged, stop_reason = False, "max_iterations"
        for iteration in range(1, settings["max_iterations"] + 1):
            sharpness = 1 + (settings["sharpness_final"] - 1) * min(1.0, (iteration - 1) / max(settings["sharpness_iterations"], 1))
            physical, cache = mapping.physical(sharpness)
            solutions = system.solve(physical, settings["penalization"], settings["min_stiffness_ratio"])
            if scales is None:
                scales, normalization, weights = _case_scaling(solutions, settings)
            objective_gradient = sum(scales[name] * result["derivative"] for name, result in solutions.items())
            gradients = mapping.gradient(cache, objective_gradient)
            previous = [value.copy() for value in mapping.field.parameters]
            optimizer.step(mapping.field.parameters, gradients)
            change = float(max(np.max(np.abs(value - old)) for value, old in zip(mapping.field.parameters, previous)))
            history.append(_history_entry(iteration, physical, solutions, scales, change, perf_counter() - started))
            history[-1].update(volume_fraction=float(np.sum(physical[mapping.allowed]) / allowed_count), sharpness=sharpness)
            density = physical.reshape(tuple(domain["grid"]["shape"]))
            if progress_callback is not None:
                progress_callback(deepcopy(history[-1]))
            recent = [entry["objective"] for entry in history[-settings["objective_window"]:]]
            stall = (max(recent) - min(recent)) / min(recent) if len(recent) == settings["objective_window"] else None
            if iteration >= max(settings["minimum_iterations"], settings["sharpness_iterations"]) and stall is not None and stall < settings["change_tolerance"]:
                converged, stop_reason = True, "objective_stall"
                break
            if settings["max_runtime_s"] is not None and perf_counter() - started >= settings["max_runtime_s"]:
                stop_reason = "max_runtime_s"
                break
        physical, _ = mapping.physical(sharpness)
        density = physical.reshape(tuple(domain["grid"]["shape"]))
        solutions = system.solve(physical, settings["penalization"], settings["min_stiffness_ratio"], metrics=True)
        final_entry = _history_entry(iteration + 1, physical, solutions, scales, None, perf_counter() - started, final=True)
        history.append(final_entry)
        if progress_callback is not None:
            progress_callback(deepcopy(final_entry))
        volume = float(np.sum(physical) * np.prod(system.spacing))
        summary = {
            "method": "Neural reparameterization (TOuNN-style): Fourier-feature MLP density, hard preserve/forbidden masks, Hex8 SIMP compliance with chain-rule sensitivities, Adam, exact volume by logit shift",
            "initialization": "network output near the target fraction everywhere; no frame template",
            "settings": settings,
            "converged": converged,
            "stop_reason": stop_reason,
            "iterations": iteration,
            "volume_fraction": float(np.sum(physical[mapping.allowed]) / allowed_count),
            "target_volume_fraction": target,
            "density_volume_mm3": volume,
            "density_frame_mass_g": volume * system.density / 1000,
            "objective_initial": history[0]["objective"],
            "objective_final": final_entry["objective"],
            "normalization_compliances_n_mm": normalization,
            "normalized_case_weights": weights,
            "static_surrogate_metrics": {name: {key: value for key, value in result.items() if key != "derivative"} for name, result in solutions.items()},
            "gray_fraction_free": float(np.mean((physical[mapping.free] > 0.1) & (physical[mapping.free] < 0.9))),
            "sharpness_final": sharpness,
            "parameter_count": int(sum(value.size for value in mapping.field.parameters)),
            "system": system.diagnostics(),
            "elapsed_s": perf_counter() - started,
        }
        outcome = {"status": "ok", "density": density.copy(), "summary": summary, "history": history, "diagnostics": diagnostics}
        if output_domain is not None:
            outcome["optimization_density"] = outcome["density"]
            outcome["density"] = mapping.sample(output_domain, sharpness, target)
            summary["output_grid"] = output_domain["grid"]
            summary["output_volume_fraction"] = float(np.sum(outcome["density"][output_domain["allowed"]]) / np.count_nonzero(output_domain["allowed"]))
        if not converged:
            diagnostics.append(f"Neural iteration stopped at {stop_reason}; convergence is not claimed")
        return outcome
    except ValueError as error:
        outcome = {"status": "invalid", "density": density, "summary": {}, "history": history, "diagnostics": [str(error)]}
        return outcome
    except (RuntimeError, MemoryError, np.linalg.LinAlgError) as error:
        outcome = {"status": "failed", "density": density, "summary": {}, "history": history, "diagnostics": [str(error)]}
        return outcome
    finally:
        if system is not None:
            cleanup = system.close()
            if cleanup and outcome is not None:
                outcome["diagnostics"].extend(cleanup)
                outcome["status"] = "failed"
