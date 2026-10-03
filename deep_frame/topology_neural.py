from copy import deepcopy
from time import perf_counter

import numpy as np

from deep_frame.topology_optimization import AugmentedLagrangian, HexElasticity, ModalConstraint, StiffnessConstraint, _case_scaling, _history_entry, validate_masks, volume_weights

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
        modal = ModalConstraint(system, settings["modal"]) if settings["modal"] else None
        stiffness = StiffnessConstraint(system, settings["stiffness"]) if settings["stiffness"] else None
        modal_log, stiffness_log = [], []
        scales = None
        converged, stop_reason = False, "max_iterations"
        for iteration in range(1, settings["max_iterations"] + 1):
            sharpness = 1 + (settings["sharpness_final"] - 1) * min(1.0, (iteration - 1) / max(settings["sharpness_iterations"], 1))
            physical, cache = mapping.physical(sharpness)
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
            previous = [value.copy() for value in mapping.field.parameters]
            optimizer.step(mapping.field.parameters, gradients)
            change = float(max(np.max(np.abs(value - old)) for value, old in zip(mapping.field.parameters, previous)))
            history.append(_history_entry(iteration, physical, solutions, scales, change, perf_counter() - started))
            history[-1].update(volume_fraction=float(np.sum(physical[mapping.allowed]) / allowed_count), sharpness=sharpness, width_penalty=width_penalty, modal_penalty=modal_penalty, f1_hz=modal_info.get("f1_hz"),
                               stiffness_penalty=stiffness_penalty, stiffness_n_per_mm=stiffness_info.get("stiffness_n_per_mm"))
            history[-1]["objective"] += width_penalty + modal_penalty + stiffness_penalty
            density = physical.reshape(tuple(domain["grid"]["shape"]))
            if progress_callback is not None:
                progress_callback(deepcopy(history[-1]))
            recent = [entry["objective"] for entry in history[-settings["objective_window"]:]]
            stall = (max(recent) - min(recent)) / min(recent) if len(recent) == settings["objective_window"] else None
            violations = [info["violation"] for info in (modal_info, stiffness_info) if info]
            feasible = settings["feasibility_tolerance"] is None or max(violations, default=0.0) <= settings["feasibility_tolerance"]
            if iteration >= max(settings["minimum_iterations"], settings["sharpness_iterations"]) and stall is not None and stall < settings["change_tolerance"] and feasible:
                converged, stop_reason = True, "objective_stall"
                break
            if settings["max_runtime_s"] is not None and perf_counter() - started >= settings["max_runtime_s"]:
                stop_reason = "max_runtime_s"
                break
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

NEURAL_AL_SETTINGS = {
    "learning_rate": 0.01,
    "frequencies": 96,
    "max_frequency_per_mm": 0.125,
    "hidden": [48, 48],
    "seed": 0,
    "mirror_axis": None,
    "volume_fraction": 0.06,
    "al": {"penalty": 10.0, "multiplier_interval": 5, "penalty_growth": 2.0, "penalty_progress": 0.25, "penalty_max": 1e4},
    "fit": {"iterations": 800, "learning_rate": 0.01},
    "level": {"window": 5, "mass_change": 1e-2, "violation": 0.02, "minimum_iterations": 15, "maximum_iterations": 80},
    "coarse_levels": 0,
    "infeasible_window": 40,
    "infeasible_progress": 1e-3,
    "max_iterations": 1500,
    "max_runtime_s": None,
}
AL_RULE = ("L = m/m10 + sum_i [mu_i/2 max(0, g_i + lambda_i/mu_i)^2 - lambda_i^2/(2 mu_i)], g_i <= 0 dimensionless relative to its limit; every multiplier_interval iterations, between Adam steps: "
           "lambda_i <- max(0, lambda_i + mu_i g_i); V_i = |max(g_i, -lambda_i/mu_i)|; mu_i <- min(penalty_growth mu_i, penalty_max) if V_i > penalty_progress x V_i at the previous update")

def neural_al_settings(settings):
    result = deepcopy(NEURAL_AL_SETTINGS)
    unknown = set(settings) - set(result)
    if unknown:
        raise ValueError("Unknown neural AL settings: " + ", ".join(sorted(unknown)))
    for key, value in deepcopy(settings).items():
        result[key] = {**result[key], **value} if isinstance(result[key], dict) else value
    return result

class NeuralDesign:
    def __init__(self, settings):
        self.field = FourierField(settings)
    def attach(self, domain):
        self.allowed, self.preserve, _ = validate_masks(domain)
        self.free = self.allowed & ~self.preserve
        self.features = self.field.features(cell_centers(domain["grid"])[self.free])
        return self
    def design(self):
        logits, activations = self.field.forward(self.features)
        density = _sigmoid(logits)
        design = self.preserve.astype(float)
        design[self.free] = density
        return design, (activations, density * (1 - density))
    def gradient(self, cache, design_gradient):
        activations, slope = cache
        return self.field.backward(activations, slope * np.asarray(design_gradient)[self.free])
    def fit(self, target, settings):
        target = np.clip(np.asarray(target, dtype=float).ravel()[self.free], 0, 1)
        optimizer = Adam(self.field.parameters, settings["learning_rate"])
        for _ in range(settings["iterations"]):
            logits, activations = self.field.forward(self.features)
            optimizer.step(self.field.parameters, self.field.backward(activations, (_sigmoid(logits) - target) / target.size))
        density = self.design()[0][self.free]
        return {"iterations": settings["iterations"], "mean_abs_error": float(np.mean(np.abs(density - target))), "volume_target": float(target.mean()), "volume_fit": float(density.mean())}

def _compact(report):
    return {key: report[key] for key in ("mass_g", "beta", "max_violation", "rows", "mass_by_field_g") if key in report}

class NeuralAugmentedLagrangian:
    def __init__(self, settings):
        self.settings = neural_al_settings(settings)
        self.design = NeuralDesign(self.settings)
        self.optimizer = Adam(self.design.field.parameters, self.settings["learning_rate"])
        self.history, self.reports, self.multipliers, self.fit = [], [], None, None
    def step(self, problem, grid):
        x, cache = self.design.design()
        tick = perf_counter()
        result = problem.evaluate(x)
        if self.multipliers is None:
            self.multipliers = AugmentedLagrangian(self.settings["al"], len(result["constraints"]))
        value, gradient, _ = self.multipliers.terms(result["constraints"], result["constraint_gradients"])
        self.optimizer.step(self.design.field.parameters, self.design.gradient(cache, result["objective_gradient"] + gradient))
        entry = {"iteration": len(self.history) + 1, "grid": grid, "level": problem.level, "beta": problem.beta, "mass_g": result["mass_g"], "objective": result["objective"], "lagrangian": result["objective"] + value,
                 "max_violation": result["max_violation"], "constraints": dict(zip(result["names"], result["constraints"].tolist())), "multipliers": self.multipliers.multiplier.tolist(),
                 "penalties": self.multipliers.penalty.tolist(), "evaluate_s": perf_counter() - tick, "elapsed_s": perf_counter() - self.started}
        self.history.append(entry)
        return result, entry, x
    def settled(self, steps, violation):
        level = self.settings["level"]
        recent = [row["mass_g"] for row in self.history[-steps:][-level["window"]:]]
        return len(recent) == level["window"] and (max(recent) - min(recent)) / min(recent) < level["mass_change"] and violation <= level["violation"]
    def infeasible(self, problem, result, steps):
        window, tolerance = self.settings["infeasible_window"], problem.problem["termination"]["violation"]
        if not problem.final_level or steps < window or result["max_violation"] <= tolerance:
            return False
        violated = result["constraints"] > tolerance
        return bool(np.all(self.multipliers.penalty[violated] >= self.settings["al"]["penalty_max"]) and self.history[-window]["max_violation"] - result["max_violation"] < self.settings["infeasible_progress"] * window)
    def run(self, stages, start=None, progress_callback=None):
        from deep_frame.topology_problem import Termination
        self.started = perf_counter()
        settings, problem, level, stop_reason = self.settings, None, 0, "guard"
        try:
            for index, (domain, factory) in enumerate(stages):
                final_grid, grid = index == len(stages) - 1, list(domain["grid"]["shape"])
                if problem is not None:
                    problem.close()
                problem = factory()
                while problem.level < level:
                    problem.advance()
                self.design.attach(domain)
                if start is not None and self.fit is None:
                    self.fit = self.design.fit(start, settings["fit"])
                termination, steps, stop_reason = Termination(problem.problem["termination"]), 0, None
                while stop_reason is None:
                    result, entry, x = self.step(problem, grid)
                    steps += 1
                    if progress_callback is not None:
                        progress_callback(entry)
                    if steps % settings["al"]["multiplier_interval"] == 0:
                        self.multipliers.update(result["constraints"])
                    if final_grid and termination(result["mass_g"], result["max_violation"], problem.final_level):
                        stop_reason = "termination"
                    elif not problem.final_level and steps >= settings["level"]["minimum_iterations"] and (self.settled(steps, result["max_violation"]) or steps >= settings["level"]["maximum_iterations"]):
                        self.reports.append({"iteration": entry["iteration"], "grid": grid, "reason": "settled" if self.settled(steps, result["max_violation"]) else "level_iterations", **_compact(problem.report(x))})
                        problem.advance()
                        level, steps = problem.level, 0
                        if not final_grid and level >= settings["coarse_levels"]:
                            stop_reason = "next_grid"
                    elif final_grid and self.infeasible(problem, result, steps):
                        stop_reason = "infeasible_plateau"
                    elif len(self.history) >= settings["max_iterations"] or (settings["max_runtime_s"] and perf_counter() - self.started >= settings["max_runtime_s"]):
                        stop_reason = "guard"
                if stop_reason != "next_grid":
                    break
            x, _ = self.design.design()
            final = problem.report(x)
            fields, _ = problem.map.fields(x)
            shape = tuple(domain["grid"]["shape"])
            grids = {str(row["grid"]) for row in self.history}
            summary = {"method": "Neural reparameterization (Fourier-feature MLP, sigmoid design x) on the shared min-mass formulation TopologyProblem: robust eroded/intermediate/dilated projection of the network output, every constraint an augmented-Lagrangian term, Adam",
                       "al_rule": AL_RULE, "settings": settings, "stop_reason": stop_reason, "converged": stop_reason == "termination", "iterations": len(self.history), "elapsed_s": perf_counter() - self.started,
                       "seconds_per_iteration": {grid: float(np.mean([row["evaluate_s"] for row in self.history if str(row["grid"]) == grid])) for grid in grids},
                       "iterations_per_grid": {grid: sum(str(row["grid"]) == grid for row in self.history) for grid in grids},
                       "fit": self.fit, "final": _compact(final), "level_reports": self.reports, "multipliers": self.multipliers.multiplier.tolist(), "penalties": self.multipliers.penalty.tolist(),
                       "parameter_count": int(sum(value.size for value in self.design.field.parameters))}
            return {"status": "ok", "design": x.reshape(shape), "density": fields["intermediate"][0].reshape(shape), "eroded": fields["eroded"][0].reshape(shape), "dilated": fields["dilated"][0].reshape(shape),
                    "summary": summary, "history": self.history}
        finally:
            if problem is not None:
                problem.close()
