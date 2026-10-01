import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

CONSTRAINT_METRICS = {
    "mass_ratio_max": ("mass_g", "max"),
    "stiffness_ratio_min": ("stiffness_n_per_mm", "min"),
    "frequency_ratio_min": ("first_frequency_hz", "min"),
    "displacement_ratio_max": ("max_displacement_mm", "max"),
    "stress_ratio_max": ("max_von_mises_mpa", "max"),
}

class EvaluationFailure(RuntimeError):
    pass

def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))

def _get(parameters, path):
    value = parameters
    try:
        for key in path.split("."):
            value = value[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing parameter path: {path}") from error
    return value

def _set(parameters, path, value):
    keys = path.split(".")
    parent = parameters
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value

def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return float(value)

def check_printability(parameters, settings):
    nozzle = _number(settings["nozzle_width_mm"], "nozzle_width_mm", True)
    paths = _number(settings["minimum_wall_nozzles"], "minimum_wall_nozzles", True)
    checks = {}
    for path in settings["wall_thickness_paths"]:
        measured = _number(_get(parameters, path), path)
        checks[path] = {
            "measured_mm": measured,
            "minimum_mm": nozzle * paths,
            "passed": measured >= nozzle * paths,
        }
    for section in settings["clamp_sections"]:
        width = _number(_get(parameters, section["width_path"]), section["width_path"])
        height = _number(_get(parameters, section["height_path"]), section["height_path"])
        minimum = _number(section["minimum_area_mm2"], "minimum_area_mm2", True)
        checks[f"section:{section['name']}"] = {
            "measured_mm2": width * height,
            "minimum_mm2": minimum,
            "passed": width * height >= minimum,
        }
    if not settings["wall_thickness_paths"] or not settings["clamp_sections"]:
        raise ValueError("Explicit wall paths and clamp sections are required")
    violations = [name for name, check in checks.items() if not check["passed"]]
    return {"passed": not violations, "violations": violations, "checks": checks}

def _precheck(parameters, validator, printability):
    printed = check_printability(parameters, printability)
    if not printed["passed"]:
        return {"passed": False, "outcome": "rejected_printability", "printability": printed, "geometry": None}
    try:
        geometry = _json_copy(validator(deepcopy(parameters)))
        if not isinstance(geometry.get("passed"), bool):
            raise ValueError("Geometry validator must return a boolean passed flag")
    except Exception as error:
        geometry = {"passed": False, "violations": [f"{type(error).__name__}: {error}"], "checks": {}}
    return {
        "passed": geometry["passed"],
        "outcome": "valid" if geometry["passed"] else "rejected_geometry",
        "printability": printed,
        "geometry": geometry,
    }

def _evaluate(parameters, evaluator):
    result = _json_copy(evaluator(deepcopy(parameters)))
    if result.get("status") != "ok":
        raise EvaluationFailure(f"Evaluator status {result.get('status')}: {result.get('diagnostics', [])}")
    metrics = {
        "mass_g": _number(result["mass_g"], "mass_g", True),
        "stiffness_n_per_mm": _number(result["stiffness_n_per_mm"], "stiffness_n_per_mm", True),
        "max_displacement_mm": _number(result["max_displacement_mm"], "max_displacement_mm"),
        "max_von_mises_mpa": _number(result["max_von_mises_mpa"], "max_von_mises_mpa"),
    }
    frequencies = [_number(value, "eigenfrequency_hz", True) for value in result["eigenfrequencies_hz"]]
    if not frequencies or frequencies != sorted(frequencies):
        raise EvaluationFailure("Sorted positive elastic eigenfrequencies are required")
    metrics["first_frequency_hz"] = frequencies[0]
    return {"result": result, "metrics": metrics}

def _constraints(metrics, baseline, factors):
    checks = {}
    for name, factor in factors.items():
        metric, comparison = CONSTRAINT_METRICS[name]
        reference = baseline[metric]
        measured = metrics[metric]
        limit = reference * factor
        difference = measured - limit if comparison == "max" else limit - measured
        checks[name] = {
            "metric": metric,
            "measured": measured,
            "reference": reference,
            "ratio": measured / reference if reference else None,
            "factor": factor,
            "limit": limit,
            "comparison": comparison,
            "residual": difference / max(abs(reference), 1e-12),
            "passed": difference <= 0,
        }
    return checks

def _objectives(metrics):
    return [metrics["mass_g"], metrics["stiffness_n_per_mm"], metrics["first_frequency_hz"]]

def _dominates(first, second):
    left = [first[0], -first[1], -first[2]]
    right = [second[0], -second[1], -second[2]]
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))

def _pareto(trials):
    valid = [trial for trial in trials if trial["outcome"] == "valid" and trial["state"] == "COMPLETE" and all(check["passed"] for check in trial["constraints"].values())]
    return [trial for trial in valid if not any(_dominates(other["objectives"], trial["objectives"]) for other in valid)]

def _export(result, output_dir):
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "result_json": str(directory / "result.json"),
        "pareto_json": str(directory / "pareto.json"),
        "pareto_csv": str(directory / "pareto.csv"),
        "storage": result["study"]["storage"],
    }
    result["artifacts"] = artifacts
    Path(artifacts["pareto_json"]).write_text(json.dumps(result["pareto_front"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with Path(artifacts["pareto_csv"]).open("w", newline="", encoding="utf-8") as file:
        columns = ["trial", "mass_g", "stiffness_n_per_mm", "first_frequency_hz", "improved_objectives", "parameters", "constraints"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for trial in result["pareto_front"]:
            writer.writerow({
                "trial": trial["number"],
                **{key: trial["metrics"][key] for key in columns[1:4]},
                **{key: json.dumps(trial[key], sort_keys=True, allow_nan=False) for key in columns[4:]},
            })
    Path(artifacts["result_json"]).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return _json_copy(result)

def _prepare_storage(storage):
    if not isinstance(storage, str) or not storage:
        raise ValueError("Persistent storage URL is required")
    if storage.startswith("sqlite:///"):
        filename = storage.removeprefix("sqlite:///")
        if filename == ":memory:" or "?" in filename:
            raise ValueError("Use a persistent SQLite file without URL parameters")
        path = Path(filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return "sqlite:///" + path.as_posix()
    return storage

def _validate_contract(reference_parameters, search_space, settings):
    import optuna
    for path, specification in search_space.items():
        _get(reference_parameters, path)
        if "choices" in specification:
            optuna.distributions.CategoricalDistribution(specification["choices"])
        else:
            optuna.distributions.FloatDistribution(specification["low"], specification["high"], step=specification.get("step"))
    factors = settings["relative_constraints"]
    if set(factors) != set(CONSTRAINT_METRICS):
        raise ValueError(f"Required relative constraints: {sorted(CONSTRAINT_METRICS)}")
    for key, factor in factors.items():
        _number(factor, key, True)
    if not settings.get("evaluation_id"):
        raise ValueError("evaluation_id must identify the evaluator and its physical settings")
    if not isinstance(settings["n_trials"], int) or settings["n_trials"] < 0:
        raise ValueError("n_trials must be a nonnegative integer")
    for candidate in settings.get("initial_candidates", []):
        for path, value in candidate.items():
            if path not in search_space:
                raise ValueError(f"Initial candidate path outside search space: {path}")
            spec = search_space[path]
            if "choices" in spec:
                valid = value in spec["choices"]
            else:
                valid = spec["low"] <= value <= spec["high"]
                if valid and spec.get("step"):
                    steps = (value - spec["low"]) / spec["step"]
                    valid = math.isclose(steps, round(steps), abs_tol=1e-8)
            if not valid:
                raise ValueError(f"Initial candidate outside distribution: {path}={value}")

def _optimize(reference_parameters, search_space, evaluator, validator, settings):
    import optuna
    from optuna.trial import TrialState
    _validate_contract(reference_parameters, search_space, settings)
    contract = _json_copy({
        "version": 1,
        "reference_parameters": reference_parameters,
        "search_space": search_space,
        "relative_constraints": settings["relative_constraints"],
        "printability": settings["printability"],
        "evaluation_id": settings["evaluation_id"],
        "seed": settings["seed"],
        "population_size": settings.get("population_size", 8),
    })
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    storage = _prepare_storage(settings["storage"])
    study = optuna.create_study(
        directions=["minimize", "maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=settings["seed"], population_size=settings.get("population_size", 8)),
        storage=storage,
        study_name=settings["study_name"],
        load_if_exists=True,
    )
    existing = study.user_attrs.get("contract_signature")
    if existing is not None and existing != signature:
        raise ValueError("Existing study uses a different reference, evaluator, search space or constraint contract")
    if existing is None and study.trials:
        raise ValueError("Existing study has trials without a compatible contract")
    study.set_user_attr("contract_signature", signature)
    study.set_user_attr("contract", contract)
    reference = study.user_attrs.get("reference")
    if reference is None:
        precheck = _precheck(reference_parameters, validator, settings["printability"])
        reference = {"parameters": _json_copy(reference_parameters), "precheck": precheck}
        if not precheck["passed"]:
            raise ValueError(f"Reference configuration failed {precheck['outcome']}")
        try:
            reference.update(_evaluate(reference_parameters, evaluator))
        except Exception as error:
            raise EvaluationFailure(f"Reference evaluation failed: {error}") from error
        reference["constraints"] = _constraints(reference["metrics"], reference["metrics"], settings["relative_constraints"])
        study.set_user_attr("reference", reference)
    baseline = reference["metrics"]
    for candidate in settings.get("initial_candidates", []):
        study.enqueue_trial(candidate, user_attrs={"initial_candidate": True}, skip_if_exists=True)
    def objective(trial):
        parameters = deepcopy(reference_parameters)
        for path, specification in search_space.items():
            if "choices" in specification:
                value = trial.suggest_categorical(path, specification["choices"])
            else:
                value = trial.suggest_float(path, specification["low"], specification["high"], step=specification.get("step"))
            _set(parameters, path, value)
        trial.set_user_attr("parameters", _json_copy(parameters))
        try:
            precheck = _precheck(parameters, validator, settings["printability"])
        except Exception as error:
            precheck = {"passed": False, "outcome": "rejected_printability", "diagnostics": [f"{type(error).__name__}: {error}"]}
        trial.set_user_attr("precheck", precheck)
        if not precheck["passed"]:
            trial.set_user_attr("outcome", precheck["outcome"])
            raise optuna.TrialPruned(precheck["outcome"])
        try:
            evaluated = _evaluate(parameters, evaluator)
        except Exception as error:
            trial.set_user_attr("outcome", "evaluation_failed")
            trial.set_user_attr("diagnostics", [f"{type(error).__name__}: {error}"])
            raise EvaluationFailure(str(error)) from error
        constraints = _constraints(evaluated["metrics"], baseline, settings["relative_constraints"])
        for name, check in constraints.items():
            trial.set_constraint(name, check["residual"])
        outcome = "valid" if all(check["passed"] for check in constraints.values()) else "rejected_constraints"
        trial.set_user_attr("outcome", outcome)
        trial.set_user_attr("evaluation", evaluated)
        trial.set_user_attr("constraints", constraints)
        values = _objectives(evaluated["metrics"])
        base_values = _objectives(baseline)
        improved = [name for name, better in zip(
            ["mass_g", "stiffness_n_per_mm", "first_frequency_hz"],
            [values[0] < base_values[0], values[1] > base_values[1], values[2] > base_values[2]],
        ) if better]
        trial.set_user_attr("improved_objectives", improved)
        return values
    study.optimize(objective, n_trials=settings["n_trials"], catch=(EvaluationFailure,), show_progress_bar=False)
    trials = []
    for trial in study.trials:
        attrs = trial.user_attrs
        evaluation = attrs.get("evaluation", {})
        trials.append({
            "number": trial.number,
            "state": trial.state.name,
            "outcome": attrs.get("outcome", trial.state.name.lower()),
            "parameters": attrs.get("parameters"),
            "sampled_parameters": trial.params,
            "initial_candidate": attrs.get("initial_candidate", False),
            "objectives": trial.values,
            "metrics": evaluation.get("metrics"),
            "result": evaluation.get("result"),
            "constraints": attrs.get("constraints", {}),
            "precheck": attrs.get("precheck"),
            "diagnostics": attrs.get("diagnostics", []),
            "improved_objectives": attrs.get("improved_objectives", []),
        })
    counts = {key: sum(trial["outcome"] == key for trial in trials) for key in [
        "valid", "rejected_printability", "rejected_geometry", "rejected_constraints", "evaluation_failed", "waiting", "running",
    ]}
    counts["total"] = len(trials)
    counts["completed"] = sum(trial.state == TrialState.COMPLETE for trial in study.trials)
    front = _pareto(trials)
    result = {
        "status": "ok" if counts["valid"] else "no_valid_design",
        "reference": reference,
        "study": {"name": study.study_name, "storage": storage, "contract_signature": signature, "directions": ["minimize", "maximize", "maximize"]},
        "trial_counts": counts,
        "trials": trials,
        "pareto_front": front,
        "has_improved_design": any(trial["improved_objectives"] for trial in front),
        "constraint_factors": settings["relative_constraints"],
    }
    return _export(result, settings["output_dir"])

def optimize(reference_parameters, search_space, evaluator, validator, settings):
    import optuna
    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    try:
        return _optimize(_json_copy(reference_parameters), _json_copy(search_space), evaluator, validator, _json_copy(settings))
    finally:
        optuna.logging.set_verbosity(previous_verbosity)

def show_candidates(candidates, build_geometry, show=None, viewer_settings=None):
    if show is None:
        from ocp_vscode import show
    if not candidates:
        raise ValueError("At least one Pareto candidate is required")
    shapes = []
    names = []
    for candidate in candidates:
        if candidate.get("outcome") != "valid":
            raise ValueError("Only valid candidates can be displayed")
        shapes.append(build_geometry(deepcopy(candidate["parameters"])))
        names.append(f"Trial {candidate['number']}")
    show(*shapes, names=names, **(viewer_settings or {}))
    return {"displayed_trials": [candidate["number"] for candidate in candidates]}
