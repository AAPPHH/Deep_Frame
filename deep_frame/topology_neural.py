from copy import deepcopy
from time import perf_counter

import numpy as np

from deep_frame.topology_optimization import HexElasticity, StiffnessConstraint, _case_scaling, _history_entry, validate_masks, volume_weights, continuation_decision

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
    "max_width_penalty": 0.0,
    "max_width_window_mm": 8.0,
    "max_local_fraction": 0.3,
    "prop_discs": None,
    "modal": None,
    "stiffness": None,
    "feasibility_tolerance": None,
    "initial_density": None,
    "initial_fit_iterations": 300,
}

def neural_settings(settings):
    result = deepcopy(NEURAL_SETTINGS)
    unknown = set(settings) - set(result)
    if unknown:
        raise ValueError("Unknown neural settings: " + ", ".join(sorted(unknown)))
    result.update(deepcopy(settings))
    if not 0 < result["volume_fraction"] < 1 or result["max_frequency_per_mm"] <= 0 or result["frequencies"] < 1 or result["learning_rate"] <= 0:
        raise ValueError("Invalid neural topology settings")
    if result["initial_fit_iterations"] < 0:
        raise ValueError("initial_fit_iterations must be non-negative")
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

def volume_shift(logits, sharpness, budget, weights=None):
    lower, upper = -60.0 / sharpness - np.max(logits), 60.0 / sharpness - np.min(logits)
    weights = np.ones(len(logits)) if weights is None else weights
    if not 0 < budget < np.sum(weights):
        raise ValueError("The volume budget must exceed the preserve cells and stay below the allowed cells")
    for _ in range(100):
        middle = (lower + upper) / 2
        if np.dot(weights, _sigmoid(sharpness * (logits + middle))) > budget:
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
        self.discs = settings["prop_discs"]
        points = cell_centers(domain["grid"])[self.free]
        self.weights = volume_weights(points, self.discs)
        self.features = self.field.features(points)
    def physical(self, sharpness=1.0):
        logits, activations = self.field.forward(self.features)
        density = _sigmoid(sharpness * (logits + volume_shift(logits, sharpness, self.budget, self.weights)))
        physical = self.preserve.astype(float)
        physical[self.free] = density
        return physical, (activations, sharpness * density * (1 - density))
    def gradient(self, cache, physical_gradient):
        activations, slope = cache
        gradient = np.asarray(physical_gradient)[self.free]
        return self.field.backward(activations, slope * (gradient - self.weights * np.dot(gradient, slope) / max(np.dot(self.weights, slope), 1e-300)))
    def sample(self, domain, sharpness, fraction, chunk=200000):
        allowed, preserve, _ = validate_masks(domain)
        free = allowed & ~preserve
        points = cell_centers(domain["grid"])[free]
        logits = np.concatenate([self.field.forward(self.field.features(part))[0] for part in np.array_split(points, max(1, len(points) // chunk))])
        density = preserve.astype(float)
        density[free] = _sigmoid(sharpness * (logits + volume_shift(logits, sharpness, fraction * np.count_nonzero(allowed) - np.count_nonzero(preserve), volume_weights(points, self.discs))))
        return density.reshape(tuple(domain["grid"]["shape"]))
    def warm_target(self, path, shape):
        from scipy.ndimage import zoom
        source = np.load(path)["density"]
        target = zoom(np.asarray(source, dtype=float), np.asarray(shape) / np.asarray(source.shape), order=1, grid_mode=True, mode="nearest")
        if target.shape != tuple(shape):
            raise ValueError(f"Warm-start field {source.shape} cannot be resampled onto {tuple(shape)}")
        target = np.clip(target, 0, 1).ravel()
        target[self.preserve], target[~self.allowed] = 1.0, 0.0
        return target, list(source.shape)
    def fit(self, target, sharpness, iterations, rate):
        optimizer = Adam(self.field.parameters, rate)
        count = np.count_nonzero(self.free)
        for _ in range(iterations):
            physical, cache = self.physical(sharpness)
            optimizer.step(self.field.parameters, self.gradient(cache, 2 * (physical - target) / count))
        return float(np.sqrt(np.mean((self.physical(sharpness)[0] - target)[self.allowed] ** 2)))

class LocalVolumePenalty:
    def __init__(self, domain, settings):
        from scipy.ndimage import binary_dilation
        self.shape = tuple(domain["grid"]["shape"])
        self.size = [int(2 * round(settings["max_width_window_mm"] / spacing / 2) + 1) for spacing in domain["grid"]["spacing_mm"]]
        allowed, preserve, _ = validate_masks(domain)
        near = binary_dilation(preserve.reshape(self.shape), np.ones(self.size, dtype=bool))
        self.mask = (allowed & ~preserve).reshape(self.shape) & ~near
        self.weight, self.limit = settings["max_width_penalty"], settings["max_local_fraction"]
        self.scale = self.weight / (settings["volume_fraction"] * np.count_nonzero(allowed) * self.limit ** 2)
    def __call__(self, physical):
        from scipy.ndimage import uniform_filter
        if self.weight <= 0:
            return 0.0, np.zeros(physical.size)
        excess = np.where(self.mask, np.maximum(uniform_filter(physical.reshape(self.shape), self.size, mode="constant") - self.limit, 0.0), 0.0)
        return float(self.scale * np.sum(excess ** 2)), uniform_filter(2 * self.scale * excess, self.size, mode="constant").ravel()

def member_widths(density, spacing, masks=None, threshold=0.5):
    from scipy.ndimage import binary_dilation, distance_transform_edt
    from skimage.morphology import skeletonize
    solid = np.asarray(density) >= threshold
    skeleton = skeletonize(solid) > 0
    if masks is not None:
        skeleton &= ~binary_dilation(masks, iterations=2)
    widths = 2 * distance_transform_edt(solid, sampling=spacing)[skeleton] - min(spacing)
    return {"threshold": threshold, "method": "2 x Euclidean distance on the 3D skeleton minus one cell, skeleton cells within two cells of preserve/forbidden excluded",
            **{f"p{q:02d}_mm": float(np.percentile(widths, q)) for q in (5, 10, 25, 50, 75, 90)}, "fraction_below_2mm": float(np.mean(widths < 2)), "skeleton_cells": int(widths.size)}

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
        penalty = LocalVolumePenalty(domain, settings)
        modal = system.modal_constraint(settings["modal"]) if settings["modal"] else None
        stiffness = StiffnessConstraint(system, settings["stiffness"]) if settings["stiffness"] else None
        modal_log, stiffness_log = [], []
        warm = None
        if settings["initial_density"] is not None:
            fit_started = perf_counter()
            target_density, source_shape = mapping.warm_target(settings["initial_density"], domain["grid"]["shape"])
            initial_rmse = float(np.sqrt(np.mean((mapping.physical(settings["sharpness_final"])[0] - target_density)[mapping.allowed] ** 2)))
            warm = {"path": str(settings["initial_density"]), "source_shape": source_shape, "target_shape": list(domain["grid"]["shape"]), "fit_iterations": settings["initial_fit_iterations"],
                    "fit_sharpness": settings["sharpness_final"], "initial_rmse": initial_rmse, "fit_rmse": mapping.fit(target_density, settings["sharpness_final"], settings["initial_fit_iterations"], settings["learning_rate"]),
                    "target_mean_free": float(np.mean(target_density[mapping.free])), "budget_mean_free": float(mapping.budget / np.count_nonzero(mapping.free)), "fit_runtime_s": perf_counter() - fit_started}
        scales = None
        converged, stop_reason = False, "max_iterations"
        final_iterations, previous_density, previous_sharpness = 0, None, None
        for iteration in range(1, settings["max_iterations"] + 1):
            sharpness = settings["sharpness_final"] if warm else 1 + (settings["sharpness_final"] - 1) * min(1.0, (iteration - 1) / max(settings["sharpness_iterations"], 1))
            physical, cache = mapping.physical(sharpness)
            final_level = sharpness == settings["sharpness_final"]
            final_iterations = final_iterations+1 if final_level else 0
            change = None if previous_density is None or sharpness != previous_sharpness else float(np.max(np.abs(physical[mapping.free]-previous_density)))
            solutions = system.solve(physical, settings["penalization"], settings["min_stiffness_ratio"])
            stiffness_penalty, stiffness_gradient, stiffness_info = stiffness(solutions.pop(stiffness.case)) if stiffness is not None else (0.0, 0.0, {})
            if scales is None:
                scales, normalization, weights = _case_scaling(solutions, settings)
            width_penalty, width_gradient = penalty(physical)
            objective_gradient = sum(scales[name] * result["derivative"] for name, result in solutions.items()) + width_gradient + stiffness_gradient
            if stiffness is not None:
                stiffness_log.append({"iteration": iteration, **stiffness_info})
            modal_penalty, modal_info = 0.0, {}
            if modal is not None:
                modal_penalty, modal_gradient, modal_info = modal(physical, settings["penalization"], settings["min_stiffness_ratio"])
                objective_gradient = objective_gradient + modal_gradient
                modal_log.append({"iteration": iteration, **modal_info})
            gradients = mapping.gradient(cache, objective_gradient)
            history.append(_history_entry(iteration, physical, solutions, scales, change, perf_counter() - started))
            history[-1].update(volume_fraction=float(np.sum(physical[mapping.allowed]) / allowed_count), sharpness=sharpness, width_penalty=width_penalty, modal_penalty=modal_penalty, f1_hz=modal_info.get("f1_hz"),
                               stiffness_penalty=stiffness_penalty, stiffness_n_per_mm=stiffness_info.get("stiffness_n_per_mm"))
            history[-1]["objective"] += width_penalty + modal_penalty + stiffness_penalty
            density = physical.reshape(tuple(domain["grid"]["shape"]))
            recent = [entry["objective"] for entry in history[-settings["objective_window"]:]]
            stall = (max(recent) - min(recent)) / max(abs(min(recent)), 1e-30) if len(recent) == settings["objective_window"] else None
            violations = [info["violation"] for info in (modal_info, stiffness_info) if info]
            violation = max(violations, default=0.0)
            history[-1].update(objective_stall=stall, max_violation=violation, final_level_iterations=final_iterations)
            if progress_callback is not None:
                progress_callback(deepcopy(history[-1]))
            decision = continuation_decision(final_iterations, final_level, change, violation, settings["minimum_iterations"], float("inf"), settings["change_tolerance"], 1e-3 if settings["feasibility_tolerance"] is None else settings["feasibility_tolerance"])
            if decision == "converged":
                converged, stop_reason = True, "change_tolerance"
                break
            if settings["max_runtime_s"] is not None and perf_counter() - started >= settings["max_runtime_s"]:
                stop_reason = "max_runtime_s"
                break
            if iteration == settings["max_iterations"]:
                break
            previous_density, previous_sharpness = physical[mapping.free].copy(), sharpness
            optimizer.step(mapping.field.parameters, gradients)
        physical, _ = mapping.physical(sharpness)
        density = physical.reshape(tuple(domain["grid"]["shape"]))
        solutions = system.solve(physical, settings["penalization"], settings["min_stiffness_ratio"], metrics=True)
        constrained = solutions.pop(stiffness.case) if stiffness is not None else None
        final_entry = _history_entry(iteration + 1, physical, solutions, scales, None, perf_counter() - started, final=True)
        final_entry["width_penalty"] = penalty(physical)[0]
        history.append(final_entry)
        if progress_callback is not None:
            progress_callback(deepcopy(final_entry))
        volume = float(np.sum(physical) * np.prod(system.spacing))
        summary = {
            "method": "Neural reparameterization (TOuNN-style): Fourier-feature MLP density, hard preserve/forbidden masks, Hex8 SIMP compliance with chain-rule sensitivities, Adam, exact volume by logit shift",
            "initialization": "network pre-fitted to the resampled warm-start field (MSE through the volume-preserving density mapping)" if warm else "network output near the target fraction everywhere; no frame template",
            "warm_start": warm,
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
            "modal": None if modal is None else {"settings": settings["modal"], "final": modal(physical, settings["penalization"], settings["min_stiffness_ratio"], settings["modal"]["initial_iterations"])[2], "history": modal_log,
                                                 "model": "voxel surrogate: motor-contact undersides fixed (domain 'modes' case = evaluator fixture), symmetric and antisymmetric half-domain parts, SIMP stiffness, density mass with low-density cutoff, point masses lumped on their attachment nodes"},
            "stiffness": None if stiffness is None else {"settings": settings["stiffness"], "case_stiffness_n_per_mm": constrained.get("stiffness_n_per_mm"), "compliance_stiffness_n_per_mm": stiffness.force ** 2 / constrained["compliance_n_mm"],
                                                         "active": bool(stiffness_log and stiffness_log[-1]["violation"] > -(settings["feasibility_tolerance"] or 0.0)), "history": stiffness_log,
                                                         "model": "voxel surrogate of the evaluator arm_tip case: centre mount undersides fixed, uniform pad load, stiffness = |F| / mean pad displacement along F = |F|^2 / compliance; excluded from the compliance objective"},
            "system": system.diagnostics(),
            "elapsed_s": perf_counter() - started,
        }
        outcome = {"status": "ok", "density": density.copy(), "summary": summary, "history": history, "diagnostics": diagnostics}
        if output_domain is not None:
            outcome["optimization_density"] = outcome["density"]
            outcome["density"] = mapping.sample(output_domain, sharpness, target)
            summary["output_grid"] = output_domain["grid"]
            summary["member_width"] = member_widths(outcome["density"], output_domain["grid"]["spacing_mm"], output_domain["preserve"] | output_domain["forbidden"])
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
