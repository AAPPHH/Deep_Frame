from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from deep_frame.topology_elasticity import HexElasticity


DEFAULT_SETTINGS = {
    "volume_fraction": 0.20,
    "filter_radius_mm": 6.0,
    "penalization": 3.0,
    "min_stiffness_ratio": 1e-6,
    "max_iterations": 60,
    "minimum_iterations": 10,
    "change_tolerance": 0.01,
    "move_limit": 0.15,
    "projection_beta": 1.0,
    "projection_eta": 0.5,
    "minimum_design_density": 1e-3,
    "case_weights": {},
    "max_runtime_s": None,
    "interface_node_policy": "allowed_adjacent",
    "linear_solver": "cpu_superlu",
}


def _settings(settings):
    result = deepcopy(DEFAULT_SETTINGS)
    result.update(deepcopy(settings))
    positive = ("filter_radius_mm", "penalization", "min_stiffness_ratio", "change_tolerance", "move_limit", "minimum_design_density")
    for name in positive:
        if not np.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("max_iterations", "minimum_iterations"):
        if isinstance(result[name], bool) or int(result[name]) != result[name] or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if result["minimum_iterations"] > result["max_iterations"]:
        result["minimum_iterations"] = result["max_iterations"]
    if not np.isfinite(result["volume_fraction"]) or not 0 < result["volume_fraction"] <= 1:
        raise ValueError("volume_fraction must be in (0, 1]")
    if result["penalization"] < 1 or not 0 < result["min_stiffness_ratio"] < 1 or result["minimum_design_density"] >= 1 or result["move_limit"] > 1:
        raise ValueError("Invalid SIMP or OC settings")
    if not np.isfinite(result["projection_beta"]) or result["projection_beta"] < 0 or not 0 < result["projection_eta"] < 1:
        raise ValueError("Invalid density projection settings")
    if result["max_runtime_s"] is not None and (not np.isfinite(result["max_runtime_s"]) or result["max_runtime_s"] <= 0):
        raise ValueError("max_runtime_s must be positive or None")
    if result["interface_node_policy"] not in ("allowed_adjacent", "preserve_adjacent"):
        raise ValueError("Invalid topology interface_node_policy")
    if result["linear_solver"] not in ("cpu_superlu", "cuda_cudss"):
        raise ValueError("Invalid topology linear_solver")
    return result


def validate_masks(domain):
    shape = tuple(domain["grid"]["shape"])
    masks = []
    for name in ("allowed", "preserve", "forbidden"):
        array = np.asarray(domain[name])
        if array.shape != shape or array.dtype != bool:
            raise ValueError(f"{name} must be a boolean array with grid shape")
        masks.append(array.ravel().copy())
    allowed, preserve, forbidden = masks
    if not np.array_equal(forbidden, ~allowed) or np.any(preserve & forbidden):
        raise ValueError("Forbidden must complement allowed; preserve must be allowed")
    if not np.any(allowed & ~preserve):
        raise ValueError("The design domain has no free material cells")
    return allowed, preserve, forbidden


class DensityMap:
    def __init__(self, domain, settings):
        self.settings = settings
        self.allowed, self.preserve, self.forbidden = validate_masks(domain)
        self.free = self.allowed & ~self.preserve
        self.n = self.allowed.size
        shape = tuple(domain["grid"]["shape"])
        spacing = np.asarray(domain["grid"]["spacing_mm"], dtype=float)
        coordinates = np.indices(shape).reshape(3, -1).T * spacing
        active = np.flatnonzero(self.allowed)
        tree = cKDTree(coordinates[active])
        radius = settings["filter_radius_mm"]
        distances = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
        weights = radius - distances.data
        self.filter = coo_matrix((weights, (active[distances.row], active[distances.col])), shape=(self.n, self.n)).tocsr()
        self.filter.eliminate_zeros()
        self.sums = np.asarray(self.filter.sum(axis=1)).ravel()
        self.sums[self.forbidden] = 1
        if np.any(self.sums <= 0):
            raise ValueError("Density filter contains an empty allowed-cell neighborhood")

    def physical(self, design):
        design = np.asarray(design, dtype=float).ravel()
        if design.size != self.n or not np.all(np.isfinite(design)) or np.any(design < 0) or np.any(design > 1):
            raise ValueError("Design densities must be finite and within [0, 1]")
        filtered = np.asarray(self.filter @ design).ravel() / self.sums
        beta = self.settings["projection_beta"]
        eta = self.settings["projection_eta"]
        if beta:
            denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
            physical = (np.tanh(beta * eta) + np.tanh(beta * (filtered - eta))) / denominator
            derivative = beta * (1 - np.tanh(beta * (filtered - eta)) ** 2) / denominator
        else:
            physical = filtered.copy()
            derivative = np.ones(self.n)
        physical[self.preserve] = 1
        physical[self.forbidden] = 0
        derivative[~self.free] = 0
        return np.clip(physical, 0, 1), derivative

    def pullback(self, sensitivity, projection_derivative):
        result = np.asarray(self.filter.T @ (np.asarray(sensitivity).ravel() * projection_derivative / self.sums)).ravel()
        result[~self.free] = 0
        return result

    def initial(self, target):
        design = np.zeros(self.n)
        design[self.preserve] = 1
        lower = self.settings["minimum_design_density"]
        design[self.free] = lower
        if np.sum(self.physical(design)[0]) > target + 1e-8:
            raise ValueError("Volume budget cannot contain preserves and their filtered transition")
        upper = 1.0
        for _ in range(70):
            value = (lower + upper) / 2
            design[self.free] = value
            if np.sum(self.physical(design)[0]) > target:
                upper = value
            else:
                lower = value
        design[self.free] = lower
        return design


def _oc_update(design, objective_derivative, volume_derivative, mapping, target, settings):
    free = mapping.free
    if np.any(volume_derivative[free] <= 0) or not np.all(np.isfinite(objective_derivative[free])):
        raise RuntimeError("Invalid OC sensitivities")
    ratios = np.maximum(1e-30, -objective_derivative[free] / volume_derivative[free])
    lower_density = np.maximum(settings["minimum_design_density"], design[free] - settings["move_limit"])
    upper_density = np.minimum(1.0, design[free] + settings["move_limit"])

    def proposal(multiplier):
        result = design.copy()
        result[free] = np.clip(design[free] * np.sqrt(ratios / max(multiplier, 1e-100)), lower_density, upper_density)
        return result

    upper = max(float(np.max(ratios)), 1e-12)
    for _ in range(100):
        if np.sum(mapping.physical(proposal(upper))[0]) <= target + 1e-9:
            break
        upper *= 2
    else:
        raise RuntimeError("OC could not bracket the volume multiplier")
    lower = 0.0
    candidate = proposal(upper)
    for _ in range(70):
        multiplier = (lower + upper) / 2
        proposed = proposal(multiplier)
        if np.sum(mapping.physical(proposed)[0]) > target:
            lower = multiplier
        else:
            upper = multiplier
            candidate = proposed
        if (upper - lower) / max(upper + lower, 1e-100) < 1e-8:
            break
    return candidate


def _case_scaling(solutions, settings):
    supplied = settings["case_weights"]
    if set(supplied) - set(solutions):
        raise ValueError("case_weights references an unknown static load case")
    weights = {name: float(supplied.get(name, 1.0)) for name in solutions}
    if any(not np.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("Every static load case requires a finite positive weight")
    total = sum(weights.values())
    normalization = {name: result["compliance_n_mm"] for name, result in solutions.items()}
    return {name: weights[name] / total / normalization[name] for name in solutions}, normalization, {name: weight / total for name, weight in weights.items()}


def _history_entry(iteration, physical, solutions, scales, change, elapsed, final=False):
    return {
        "iteration": iteration,
        "final_evaluation": final,
        "objective": float(sum(scales[name] * result["compliance_n_mm"] for name, result in solutions.items())),
        "physical_density_sum": float(np.sum(physical)),
        "compliances_n_mm": {name: result["compliance_n_mm"] for name, result in solutions.items()},
        "maximum_design_change": change,
        "maximum_relative_residual": float(max(result["relative_residual"] for result in solutions.values())),
        "elapsed_s": elapsed,
    }


def optimize_topology(domain, settings, *, progress_callback=None):
    started = perf_counter()
    history = []
    diagnostics = []
    system = None
    density = np.zeros(tuple(domain.get("grid", {}).get("shape", (0, 0, 0))))
    try:
        settings = _settings(settings)
        mapping = DensityMap(domain, settings)
        target = settings["volume_fraction"] * int(np.count_nonzero(mapping.allowed))
        if np.count_nonzero(mapping.preserve) >= target:
            raise ValueError("The volume budget must exceed the preserve-cell volume")
        design = mapping.initial(target)
        system = HexElasticity(domain, interface_node_policy=settings["interface_node_policy"], linear_solver=settings["linear_solver"])
        scales = None
        converged = False
        stop_reason = "max_iterations"
        for iteration in range(1, settings["max_iterations"] + 1):
            physical, projection_derivative = mapping.physical(design)
            density = physical.reshape(tuple(domain["grid"]["shape"]))
            solutions = system.solve(physical, settings["penalization"], settings["min_stiffness_ratio"])
            if scales is None:
                scales, normalization, weights = _case_scaling(solutions, settings)
            physical_gradient = sum(scales[name] * result["derivative"] for name, result in solutions.items())
            gradient = mapping.pullback(physical_gradient, projection_derivative)
            volume_gradient = mapping.pullback(np.ones(mapping.n), projection_derivative)
            candidate = _oc_update(design, gradient, volume_gradient, mapping, target, settings)
            change = float(np.max(np.abs(candidate - design)))
            history.append(_history_entry(iteration, physical, solutions, scales, change, perf_counter() - started))
            if progress_callback is not None:
                progress_callback(deepcopy(history[-1]))
            design = candidate
            if iteration >= settings["minimum_iterations"] and change < settings["change_tolerance"]:
                converged = True
                stop_reason = "change_tolerance"
                break
            if settings["max_runtime_s"] is not None and perf_counter() - started >= settings["max_runtime_s"]:
                stop_reason = "max_runtime_s"
                break
        physical, _ = mapping.physical(design)
        density = physical.reshape(tuple(domain["grid"]["shape"]))
        solutions = system.solve(physical, settings["penalization"], settings["min_stiffness_ratio"], metrics=True)
        final_entry = _history_entry(iteration + 1, physical, solutions, scales, None, perf_counter() - started, final=True)
        history.append(final_entry)
        if progress_callback is not None:
            progress_callback(deepcopy(final_entry))
        volume = float(np.sum(physical) * np.prod(system.spacing))
        static_metrics = {name: {key: value for key, value in result.items() if key != "derivative"} for name, result in solutions.items()}
        summary = {
            "method": "3D Hex8 SIMP, spatial density filter, smooth projection, normalized multi-load compliance, OC volume constraint",
            "initialization": "uniform in every free cell; fixed preserves; no frame template or connectivity seed",
            "settings": settings,
            "converged": converged,
            "stop_reason": stop_reason,
            "iterations": iteration,
            "volume_fraction": float(np.sum(physical) / np.count_nonzero(mapping.allowed)),
            "target_volume_fraction": settings["volume_fraction"],
            "density_volume_mm3": volume,
            "density_frame_mass_g": volume * system.density / 1000,
            "objective_initial": history[0]["objective"],
            "objective_final": final_entry["objective"],
            "normalization_compliances_n_mm": normalization,
            "normalized_case_weights": weights,
            "static_surrogate_metrics": static_metrics,
            "modal_assessment": "independent gmsh/CalculiX candidate verification, including specified point masses; no voxel-frequency substitute",
            "stress_assessment": "Hex8 Gauss-point SIMP stresses are surrogate diagnostics only; exact-solid CalculiX governs candidate stress constraints",
            "manufacturing_assessment": "density filtering regularizes lengths; reconstructed-solid feature, connection, forbidden and manufacturing checks remain mandatory",
            "gray_fraction_free": float(np.mean((physical[mapping.free] > 0.1) & (physical[mapping.free] < 0.9))),
            "element_model": "three displacement DOFs/node, fully integrated trilinear 8-node 3D elasticity, 2x2x2 Gauss integration",
            "units": {"length": "mm", "force": "N", "stress": "MPa", "density": "g/cm3", "internal_modal_mass": "tonne"},
            "system": system.diagnostics(),
            "elapsed_s": perf_counter() - started,
        }
        if not converged:
            diagnostics.append(f"Density iteration stopped at {stop_reason}; convergence is not claimed")
        return {"status": "ok", "density": density.copy(), "design_density": design.reshape(density.shape).copy(), "summary": summary, "history": history, "diagnostics": diagnostics}
    except ValueError as error:
        return {"status": "invalid", "density": density, "summary": {}, "history": history, "diagnostics": [str(error)]}
    except (RuntimeError, np.linalg.LinAlgError) as error:
        return {"status": "failed", "density": density, "summary": {}, "history": history, "diagnostics": [str(error)]}
    finally:
        if system is not None:
            system.close()
