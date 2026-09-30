import hashlib
import inspect
import json
import math
import subprocess
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
from build123d import export_step, export_stl

from deep_frame.fea import evaluate
from deep_frame.geometry import build_geometry
from deep_frame.integration import solver_identity
from deep_frame.topology_domain import build_design_domain
from deep_frame.topology_geometry import reconstruct_topology, validate_topology
from deep_frame.topology_optimization import optimize_topology


PIPELINE_CONFIG = {
    "output_dir": "exports/topology/phase1",
    "resume": True,
    "seed": 42,
    "evaluation_id": "phase1-independent-calculix-v1",
    "density_thresholds": [0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    "optimizer_variants": [{"name": "default", "settings": {}}],
    "optimizer_settings": {},
    "reconstruction_settings": {"remove_unanchored_islands": False, "repair_manifold_voxels": True, "maximum_repair_voxels": 40},
    "fea_settings": {"mesh_second_order_linear": True, "mesh_high_order_optimize": 0},
    "relative_constraints": {
        "frame_mass_ratio_max": 2.0,
        "stiffness_ratio_min": 0.5,
        "frequency_ratio_min": 0.7,
        "displacement_ratio_max": 2.0,
        "stress_ratio_max": 2.0,
    },
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _encoded(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _configuration(settings):
    config = _merge(PIPELINE_CONFIG, settings)
    if not config["density_thresholds"] or any(not math.isfinite(value) or not 0 < value <= 1 for value in config["density_thresholds"]):
        raise ValueError("Pipeline density thresholds must be nonempty and in (0,1]")
    if len(set(config["density_thresholds"])) != len(config["density_thresholds"]):
        raise ValueError("Pipeline thresholds must be unique")
    variants = config["optimizer_variants"]
    if not variants or len({variant["name"] for variant in variants}) != len(variants):
        raise ValueError("Optimizer variants need unique names")
    for variant in variants:
        name = variant["name"]
        if not name or not all(character.isalnum() or character in "_-" for character in name):
            raise ValueError("Variant names must contain only letters, digits, underscores and hyphens")
    for name in PIPELINE_CONFIG["relative_constraints"]:
        if not math.isfinite(config["relative_constraints"][name]) or config["relative_constraints"][name] <= 0:
            raise ValueError("Relative constraints must be finite and positive")
    _encoded(config)
    return config


def _callable_identity(function):
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = repr(type(function))
    return {"name": function.__module__ + "." + getattr(function, "__qualname__", type(function).__qualname__), "source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def _provenance(functions, fea_settings, real_evaluator):
    packages = {}
    for name in ("numpy", "scipy", "build123d", "gmsh", "trimesh", "cadquery-ocp-novtk"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source_dir = Path(__file__).parent
    sources = {path.name: _file_digest(path) for path in sorted(source_dir.glob("*.py"))}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_dir, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source_dir, capture_output=True, text=True, check=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    result = {"packages": packages, "source_sha256": sources, "callbacks": {name: _callable_identity(function) for name, function in functions.items()}, "git_revision": revision, "git_tracked_dirty": dirty}
    if real_evaluator:
        result["solver"] = solver_identity(fea_settings)
    return result


def _metrics(result):
    if result.get("status") != "ok":
        raise ValueError("Independent FEA did not complete successfully")
    metrics = {name: result.get(name) for name in ("frame_mass_g", "stiffness_n_per_mm", "max_displacement_mm", "max_von_mises_mpa")}
    metrics["first_frequency_hz"] = min(result.get("eigenfrequencies_hz", []) or [float("nan")])
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in metrics.values()):
        raise ValueError("Independent FEA needs finite positive mass, stiffness, displacement, stress and frequency")
    return metrics


def _verify_cases(result, expected):
    _metrics(result)
    actual = result.get("load_cases", {})
    for case in expected:
        solved = actual.get(case["name"], {})
        if solved.get("analysis") != case["analysis"]:
            raise ValueError("Independent FEA omitted or changed case " + case["name"])
        if case["analysis"] == "modal":
            values = solved.get("eigenfrequencies_hz", [])
        else:
            values = [solved.get("max_displacement_mm"), solved.get("max_von_mises_mpa")]
        if not values or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Independent FEA has missing or invalid metrics in case " + case["name"])


def compare_to_baseline(result, baseline, constraints):
    candidate, reference = _metrics(result), _metrics(baseline)
    ratios = {
        "frame_mass_ratio": candidate["frame_mass_g"] / reference["frame_mass_g"],
        "stiffness_ratio": candidate["stiffness_n_per_mm"] / reference["stiffness_n_per_mm"],
        "frequency_ratio": candidate["first_frequency_hz"] / reference["first_frequency_hz"],
        "displacement_ratio": candidate["max_displacement_mm"] / reference["max_displacement_mm"],
        "stress_ratio": candidate["max_von_mises_mpa"] / reference["max_von_mises_mpa"],
    }
    checks = {
        "frame_mass": ratios["frame_mass_ratio"] <= constraints["frame_mass_ratio_max"],
        "stiffness": ratios["stiffness_ratio"] >= constraints["stiffness_ratio_min"],
        "frequency": ratios["frequency_ratio"] >= constraints["frequency_ratio_min"],
        "displacement": ratios["displacement_ratio"] <= constraints["displacement_ratio_max"],
        "stress": ratios["stress_ratio"] <= constraints["stress_ratio_max"],
    }
    score = ratios["frame_mass_ratio"] + 1 / ratios["stiffness_ratio"] + 1 / ratios["frequency_ratio"] + ratios["displacement_ratio"] + ratios["stress_ratio"]
    return {"passed": all(checks.values()), "checks": checks, "ratios": ratios, "limits": deepcopy(constraints), "candidate": candidate, "reference": reference, "selection_score": score}


def _pareto(candidates):
    valid = [record for record in candidates if record["status"] == "ok"]
    values = {}
    for record in valid:
        metrics = record["comparison"]["candidate"]
        values[record["id"]] = np.array([metrics["frame_mass_g"], -metrics["stiffness_n_per_mm"], -metrics["first_frequency_hz"], metrics["max_displacement_mm"], metrics["max_von_mises_mpa"]])
    return [record["id"] for record in valid if not any(np.all(other <= values[record["id"]]) and np.any(other < values[record["id"]]) for key, other in values.items() if key != record["id"])]


def _artifact(path, directory):
    path = Path(path)
    return {"path": str(path.relative_to(directory)).replace("\\", "/"), "sha256": _file_digest(path), "size_bytes": path.stat().st_size}


def _fea_artifacts(result, directory):
    location = result.get("artifacts", {}).get("directory")
    if not location:
        return {}
    location = Path(location).resolve()
    if not location.is_dir() or not location.is_relative_to(directory):
        raise ValueError("Independent FEA artifacts must remain inside the run directory")
    return {"raw_fea_" + str(path.relative_to(location)).replace("\\", "/"): _artifact(path, directory) for path in sorted(location.rglob("*")) if path.is_file()}


def _valid_artifacts(record, directory):
    for artifact in record.get("artifacts", {}).values():
        if not isinstance(artifact, dict) or "sha256" not in artifact:
            continue
        path = directory / artifact["path"]
        if not path.is_file() or _file_digest(path) != artifact["sha256"]:
            return False
    return True


def _save_fields(path, domain, result):
    arrays = {name: np.asarray(domain[name], dtype=bool) for name in ("allowed", "preserve", "forbidden")}
    arrays["density"] = np.asarray(result["density"], dtype=float)
    if result.get("design_density") is not None:
        arrays["design_density"] = np.asarray(result["design_density"], dtype=float)
    np.savez_compressed(path, **arrays)


def run_topology(parameters, settings=None, *, domain_builder=build_design_domain, generator=optimize_topology, reconstructor=reconstruct_topology, validator=validate_topology, evaluator=evaluate, baseline_builder=build_geometry):
    started = perf_counter()
    parameters = deepcopy(parameters)
    config = _configuration(settings or {})
    domain = domain_builder(deepcopy(parameters))
    domain_json = {key: value for key, value in domain.items() if not isinstance(value, np.ndarray)}
    field_contract = {key: {"shape": list(np.asarray(domain[key]).shape), "dtype": str(np.asarray(domain[key]).dtype), "sha256": hashlib.sha256(np.asarray(domain[key]).tobytes(order="C")).hexdigest()} for key in ("allowed", "preserve", "forbidden")}
    fea_settings = _merge(domain["fea_settings"], config["fea_settings"])
    functions = {"domain_builder": domain_builder, "generator": generator, "reconstructor": reconstructor, "validator": validator, "evaluator": evaluator, "baseline_builder": baseline_builder}
    provenance = _provenance(functions, fea_settings, evaluator is evaluate)
    physical_config = {key: value for key, value in config.items() if key not in ("output_dir", "resume")}
    identity_provenance = {key: value for key, value in provenance.items() if key not in ("git_revision", "git_tracked_dirty")}
    inputs = {"parameters": parameters, "domain": domain_json, "masks": field_contract, "settings": physical_config, "provenance": identity_provenance}
    fingerprint = _digest(inputs)
    directory = Path(config["output_dir"]).resolve() / fingerprint[:16]
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    inputs_path = directory / "inputs.json"
    if inputs_path.is_file():
        if _digest(json.loads(inputs_path.read_text(encoding="utf-8"))) != fingerprint:
            raise ValueError("Immutable topology input record has been changed")
    else:
        _save(inputs_path, inputs)
    if config["resume"] and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_sha256") != fingerprint:
            raise ValueError("Existing topology run has conflicting input identity")
    else:
        manifest = {"schema_version": "deep-frame-topology-run-v1", "status": "running", "input_sha256": fingerprint, "run_dir": str(directory), "parameters": parameters, "domain": domain_json, "array_contract": {"axis_order": "xyz", "order": "C", "masks": field_contract}, "settings": config, "provenance": provenance, "units": {"length": "mm", "force": "N", "stress": "MPa", "mass": "g", "density": "g/cm3", "frequency": "Hz"}, "baseline": None, "optimizations": [], "candidates": [], "pareto_ids": [], "selected_id": None, "diagnostics": [], "artifacts": {}}
        _save(manifest_path, manifest)
    domain_fields_path = directory / "domain_masks.npz"
    if not domain_fields_path.is_file() or not _valid_artifacts({"artifacts": {"domain_masks": manifest.get("artifacts", {}).get("domain_masks", {})}}, directory):
        np.savez_compressed(domain_fields_path, **{name: domain[name] for name in ("allowed", "preserve", "forbidden")})
    manifest["artifacts"].update({"inputs": _artifact(inputs_path, directory), "domain_masks": _artifact(domain_fields_path, directory)})
    comparison_cases = deepcopy(domain.get("comparison_load_cases", domain["load_cases"]))
    if not any(case["analysis"] == "modal" for case in comparison_cases) or not any(case["analysis"] == "static" for case in comparison_cases):
        raise ValueError("Independent topology verification requires static and modal cases")
    manifest["verification_load_cases"] = comparison_cases
    manifest["comparison_policy"] = "Same material, point masses, physical selectors, loads and mesh settings for v0 and every candidate; additional attachment proof loads remain in the SIMP record."
    manifest["constraint_interpretation"] = "Predeclared Phase-1 engineering screens; neither flightworthiness nor real-print crash strength is certified."
    manifest["selection_policy"] = "Feasible five-objective Pareto set; selected candidate minimizes mass_ratio + 1/stiffness_ratio + 1/frequency_ratio + displacement_ratio + stress_ratio."
    baseline_record = manifest.get("baseline")
    if not baseline_record or "fea.json" not in baseline_record.get("artifacts", {}) or not _valid_artifacts(baseline_record, directory):
        baseline_dir = directory / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        baseline_settings = deepcopy(fea_settings)
        baseline_settings["work_dir"] = str(baseline_dir / "fea")
        try:
            baseline_solid = baseline_builder(deepcopy(parameters))
            baseline_result = evaluator(baseline_solid, deepcopy(domain["material"]), deepcopy(domain["point_masses"]), deepcopy(comparison_cases), baseline_settings)
            export_step(baseline_solid, baseline_dir / "geometry.step")
            export_stl(baseline_solid, baseline_dir / "geometry.stl")
        except Exception as error:
            baseline_result = {"status": "failed", "diagnostics": [f"{type(error).__name__}: {error}"]}
        _save(baseline_dir / "fea.json", baseline_result)
        artifacts = {name: _artifact(baseline_dir / name, directory) for name in ("fea.json", "geometry.step", "geometry.stl") if (baseline_dir / name).is_file()}
        artifacts.update(_fea_artifacts(baseline_result, directory))
        baseline_record = {"result": baseline_result, "artifacts": artifacts}
        manifest["baseline"] = baseline_record
        _save(manifest_path, manifest)
    baseline = json.loads((directory / baseline_record["artifacts"]["fea.json"]["path"]).read_text(encoding="utf-8"))
    baseline_record["result"] = baseline
    try:
        _verify_cases(baseline, comparison_cases)
    except ValueError as error:
        manifest.update(status="failed", diagnostics=["Baseline verification: " + str(error)], elapsed_s=perf_counter() - started)
        _save(manifest_path, manifest)
        return manifest
    for variant in config["optimizer_variants"]:
        variant_name = variant["name"]
        optimizer_settings = _merge(_merge(domain["optimizer_settings"], config["optimizer_settings"]), variant.get("settings", {}))
        optimizer_dir = directory / "optimizer" / variant_name
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        record = next((entry for entry in manifest["optimizations"] if entry["name"] == variant_name), None)
        if record and _valid_artifacts(record, directory):
            result = json.loads((directory / record["artifacts"]["result"]["path"]).read_text(encoding="utf-8"))
            with np.load(directory / record["artifacts"]["fields"]["path"], allow_pickle=False) as fields:
                result["density"] = fields["density"].copy()
                if "design_density" in fields:
                    result["design_density"] = fields["design_density"].copy()
        else:
            try:
                result = generator(deepcopy(domain), deepcopy(optimizer_settings))
            except Exception as error:
                result = {"status": "failed", "density": np.zeros(tuple(domain["grid"]["shape"])), "summary": {}, "history": [], "diagnostics": [f"{type(error).__name__}: {error}"]}
            _save_fields(optimizer_dir / "fields.npz", domain, result)
            _save(optimizer_dir / "result.json", {key: value for key, value in result.items() if not isinstance(value, np.ndarray)})
            record = {"name": variant_name, "status": result["status"], "settings": optimizer_settings, "artifacts": {"fields": _artifact(optimizer_dir / "fields.npz", directory), "result": _artifact(optimizer_dir / "result.json", directory)}}
            manifest["optimizations"] = [entry for entry in manifest["optimizations"] if entry["name"] != variant_name] + [record]
            manifest["candidates"] = [entry for entry in manifest["candidates"] if entry["optimizer_variant"] != variant_name]
            _save(manifest_path, manifest)
        if result["status"] != "ok":
            continue
        for index, threshold in enumerate(config["density_thresholds"]):
            candidate_id = variant_name + f"_t{index:02d}"
            previous = next((entry for entry in manifest["candidates"] if entry["id"] == candidate_id), None)
            if previous and "record" in previous.get("artifacts", {}) and _valid_artifacts(previous, directory):
                canonical = json.loads((directory / previous["artifacts"]["record"]["path"]).read_text(encoding="utf-8"))
                canonical["artifacts"]["record"] = previous["artifacts"]["record"]
                manifest["candidates"] = [entry for entry in manifest["candidates"] if entry["id"] != candidate_id] + [canonical]
                continue
            candidate_dir = directory / "candidates" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            reconstruction_settings = _merge(domain["reconstruction_settings"], config["reconstruction_settings"])
            reconstruction_settings["density_threshold"] = threshold
            current = {"id": candidate_id, "optimizer_variant": variant_name, "density_threshold": threshold, "reconstruction_settings": reconstruction_settings, "status": "invalid", "stage": "geometry", "diagnostics": [], "artifacts": {}}
            try:
                solid = reconstructor(deepcopy(domain), result["density"].copy(), deepcopy(reconstruction_settings))
                current["reconstruction"] = deepcopy(getattr(solid, "topology_report", {}))
                current["frame_mass_g"] = float(solid.volume * domain["material"]["density_g_cm3"] / 1000)
                export_step(solid, candidate_dir / "geometry.step")
                export_stl(solid, candidate_dir / "geometry.stl")
                current["artifacts"].update({name: _artifact(candidate_dir / name, directory) for name in ("geometry.step", "geometry.stl")})
                current["checks"] = validator(solid, deepcopy(domain), deepcopy(reconstruction_settings))
                if not current["checks"].get("passed", False):
                    raise ValueError("Geometry/manufacturing validation failed: " + "; ".join(current["checks"].get("violations", [])))
                if current["frame_mass_g"] > config["relative_constraints"]["frame_mass_ratio_max"] * baseline["frame_mass_g"]:
                    current["stage"] = "mass_constraint"
                    raise ValueError("Exact CAD frame mass exceeds the predeclared baseline-relative limit")
                current["stage"] = "independent_fea"
                candidate_fea_settings = deepcopy(fea_settings)
                candidate_fea_settings["work_dir"] = str(candidate_dir / "fea")
                current["fea"] = evaluator(solid, deepcopy(domain["material"]), deepcopy(domain["point_masses"]), deepcopy(comparison_cases), candidate_fea_settings)
                _save(candidate_dir / "fea.json", current["fea"])
                current["artifacts"]["fea"] = _artifact(candidate_dir / "fea.json", directory)
                current["artifacts"].update(_fea_artifacts(current["fea"], directory))
                _verify_cases(current["fea"], comparison_cases)
                current["comparison"] = compare_to_baseline(current["fea"], baseline, config["relative_constraints"])
                current["status"] = "ok" if current["comparison"]["passed"] else "invalid"
                current["stage"] = "complete"
                if not current["comparison"]["passed"]:
                    current["diagnostics"].append("Failed mechanical screens: " + ", ".join(name for name, passed in current["comparison"]["checks"].items() if not passed))
            except (ValueError, KeyError, TypeError) as error:
                current["diagnostics"].append(f"{type(error).__name__}: {error}")
            except Exception as error:
                current["status"] = "failed"
                current["diagnostics"].append(f"{type(error).__name__}: {error}")
            _save(candidate_dir / "record.json", current)
            current["artifacts"]["record"] = _artifact(candidate_dir / "record.json", directory)
            manifest["candidates"] = [entry for entry in manifest["candidates"] if entry["id"] != candidate_id] + [current]
            _save(manifest_path, manifest)
    manifest["pareto_ids"] = _pareto(manifest["candidates"])
    feasible = [entry for entry in manifest["candidates"] if entry["status"] == "ok"]
    manifest["selected_id"] = min(feasible, key=lambda entry: (entry["comparison"]["selection_score"], entry["id"]))["id"] if feasible else None
    manifest["status"] = "ok" if feasible else "invalid"
    manifest["elapsed_s"] = perf_counter() - started
    manifest["diagnostics"] = [] if feasible else ["No automatically generated candidate passed all geometry, manufacturing and independent mechanical screens"]
    pareto = [entry for entry in manifest["candidates"] if entry["id"] in manifest["pareto_ids"]]
    _save(directory / "pareto.json", pareto)
    header = "id,frame_mass_g,stiffness_n_per_mm,first_frequency_hz,max_displacement_mm,max_von_mises_mpa\n"
    rows = [",".join([entry["id"]] + [str(entry["comparison"]["candidate"][key]) for key in ("frame_mass_g", "stiffness_n_per_mm", "first_frequency_hz", "max_displacement_mm", "max_von_mises_mpa")]) for entry in pareto]
    (directory / "pareto.csv").write_text(header + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    manifest["artifacts"].update({name: _artifact(directory / name, directory) for name in ("pareto.json", "pareto.csv")})
    _save(manifest_path, manifest)
    _save(Path(config["output_dir"]).resolve() / "latest.json", {"manifest": str(manifest_path), "input_sha256": fingerprint, "status": manifest["status"], "selected_id": manifest["selected_id"]})
    return manifest
