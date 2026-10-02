from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh
from build123d import export_step, export_stl, import_step

from deep_frame.config import command_line, configure
from deep_frame.fea import evaluate
from deep_frame.frame import build_geometry
from deep_frame.topology_geometry import _merge, reconstruct_topology, validate_topology
from deep_frame.topology_pipeline import (
    _artifact, _digest, _fea_artifacts, _file_digest, _metrics, _pareto,
    _plot_modules, _provenance, _read, _save, _valid_artifacts, _verify_cases, candidate_entry, compare_to_baseline, log_run,
)

EVIDENCE = ROOT / "docs/validation/topology_phase1"
DEFAULT_OUTPUT = ROOT / "exports/topology/workstation_20260930/mesh_study"

SCREEN = {
    "stiffness_n_per_mm": 0.02,
    "max_displacement_mm": 0.02,
    "first_frequency_hz": 0.02,
    "max_von_mises_mpa": 0.05,
}

MESH_CONFIG = {"output": DEFAULT_OUTPUT, "mesh_sizes": [3.0, 2.5, 2.0], "threads": 2, "linear_solver": None, "prepare_only": False}
MESH_KINDS = {"output": "path", "mesh_sizes": ["float"], "threads": "int", "linear_solver": ("SPOOLES", "PASTIX"), "prepare_only": "flag"}
CANDIDATES_CONFIG = {"source": None, "output": None, "baseline_study": None, "geometry_only": False, "linear_solver": None, "threads": 2, "run_log": None}
CANDIDATES_KINDS = {"source": "path", "output": "path", "baseline_study": "path", "geometry_only": "flag",
                    "linear_solver": ("SPOOLES", "PASTIX"), "threads": "int", "run_log": "path"}
RENDER_CONFIG = {"output": ROOT / "docs/validation/workstation_geometry_comparison.png"}
RENDER_KINDS = {"output": "path"}

def geometry_metrics(solid):
    bounds = solid.bounding_box()
    return {
        "volume_mm3": float(solid.volume),
        "bounds_min_mm": list(bounds.min),
        "bounds_max_mm": list(bounds.max),
        "solids": len(solid.solids()),
        "is_valid": solid.is_valid,
    }

def mesh_prepare(output, threads, linear_solver=None):
    previous_path = output / "preparation.json"
    if not previous_path.exists() and any(output.iterdir()):
        raise ValueError("A new study requires an empty output directory; preserve partial files and use a fresh directory")
    acceptance = _read(EVIDENCE / "acceptance.json")
    checks = {}
    for name, expected in acceptance["versioned_artifacts"].items():
        path = EVIDENCE / name
        checks[name] = {
            "sha256": _file_digest(path), "size_bytes": path.stat().st_size,
            "passed": _file_digest(path) == expected["sha256"]
            and path.stat().st_size == expected["size_bytes"],
        }
    if not all(item["passed"] for item in checks.values()):
        raise ValueError("Historical evidence failed its stored SHA256/size check")
    inputs = _read(EVIDENCE / "inputs.json")
    domain = deepcopy(inputs["domain"])
    mask_checks = {}
    with np.load(EVIDENCE / "fields.npz", allow_pickle=False) as fields:
        for name in ("allowed", "preserve", "forbidden"):
            array = fields[name].copy()
            contract = inputs["masks"][name]
            valid = (list(array.shape) == contract["shape"]
                     and str(array.dtype) == contract["dtype"]
                     and hashlib.sha256(array.tobytes(order="C")).hexdigest() == contract["sha256"])
            mask_checks[name] = valid
            if not valid:
                raise ValueError("Invalid archived mask: " + name)
            domain[name] = array
        density = fields["density"].copy()
    reconstruction_settings = _read(EVIDENCE / "candidate_record.json")["reconstruction_settings"]
    print("Reconstructing the archived 45-iteration density", flush=True)
    candidate = reconstruct_topology(domain, density, reconstruction_settings)
    validation = validate_topology(candidate, domain, reconstruction_settings)
    if not validation["passed"]:
        raise ValueError("Reconstructed geometry failed: " + repr(validation["violations"]))
    baseline = build_geometry(deepcopy(inputs["parameters"]))
    solids = {"baseline": baseline, "candidate": candidate}
    geometries = {}
    for name, solid in solids.items():
        metric = geometry_metrics(solid)
        historical_mass = acceptance["comparison"]["reference" if name == "baseline" else "candidate"]["frame_mass_g"]
        expected_volume = historical_mass * 1000 / domain["material"]["density_g_cm3"]
        metric["historical_volume_mm3"] = expected_volume
        metric["volume_difference_mm3"] = metric["volume_mm3"] - expected_volume
        metric["historical_volume_matches"] = abs(metric["volume_difference_mm3"]) <= 1e-5
        if not metric["historical_volume_matches"] or not metric["is_valid"] or metric["solids"] != 1:
            raise ValueError("Historical geometry reconstruction mismatch: " + name)
        directory = output / "geometry" / name
        directory.mkdir(parents=True, exist_ok=True)
        for suffix, exporter in (("step", export_step), ("stl", export_stl)):
            path = directory / ("geometry." + suffix)
            if not path.exists():
                exporter(solid, path)
            metric[suffix] = _artifact(path, output)
        geometries[name] = metric
    settings = deepcopy(domain["fea_settings"])
    settings.update(inputs["settings"]["fea_settings"])
    settings.update(threads=threads, mesh_threads=1, mesh_timeout_s=600.0, solver_timeout_s=900.0)
    if linear_solver is not None:
        settings["linear_solver"] = linear_solver
    provenance = _provenance({"reconstructor": reconstruct_topology, "evaluator": evaluate,
                              "baseline_builder": build_geometry}, settings, True)
    provenance["study_runner_sha256"] = _file_digest(__file__)
    source_match = {name: digest == provenance["source_sha256"].get(name)
                    for name, digest in inputs["provenance"]["source_sha256"].items()}
    record = {
        "schema_version": "deep-frame-workstation-mesh-study-v1",
        "historical_input_sha256": acceptance["input_sha256"],
        "historical_raw_run_present": (ROOT / "exports/topology/phase1/2cd7bcbc15b67fc7/manifest.json").is_file(),
        "historical_archive_present": (ROOT / "exports/topology/handoff/Deep_Frame-phase1-938bbdf-workstation.zip").is_file(),
        "migration_note": "Original ZIP, STEP/STL and raw FEA were not transferred. CAD is regenerated from SHA256-verified archived density and inputs; new files are not claimed byte-identical to the lost raw artifacts.",
        "versioned_artifact_checks": checks,
        "acceptance_sha256": _file_digest(EVIDENCE / "acceptance.json"),
        "mask_checks": mask_checks,
        "historical_source_byte_matches": source_match,
        "provenance": provenance,
        "geometries": geometries,
        "candidate_validation": validation,
        "reconstruction": candidate.topology_report,
        "material": domain["material"],
        "point_masses": domain["point_masses"],
        "load_cases": domain["comparison_load_cases"],
        "fea_settings": settings,
        "relative_constraints": inputs["settings"]["relative_constraints"],
        "mesh_change_screen": SCREEN,
        "screen_interpretation": "Predeclared successive-mesh sensitivity screen; not a proof of asymptotic convergence. Physical selector boxes, point-mass patch, loads, material and CAD stay fixed; node counts and equal nodal force allocation vary with the mesh. Peaks at sharp corners may be singular.",
    }
    identity = {key: record[key] for key in ("historical_input_sha256", "acceptance_sha256", "versioned_artifact_checks", "material", "point_masses", "load_cases", "fea_settings", "relative_constraints", "mesh_change_screen")}
    identity.update(parameters=inputs["parameters"], grid=domain["grid"], reconstruction_settings=reconstruction_settings)
    identity["provenance"] = {key: provenance[key] for key in ("packages", "python_version", "platform", "source_sha256", "solver", "study_runner_sha256")}
    record["study_input_sha256"] = _digest(identity)
    if previous_path.exists():
        previous = _read(previous_path)
        if previous["study_input_sha256"] != record["study_input_sha256"]:
            raise ValueError("Study provenance changed; choose a fresh output directory")
        for name in solids:
            if not _valid_artifacts({"artifacts": {kind: previous["geometries"][name][kind] for kind in ("step", "stl")}}, output):
                raise ValueError("Stored study geometry was modified")
    else:
        _save(previous_path, record)
    print(json.dumps({"prepared": str(output), "volumes_mm3": {name: item["volume_mm3"] for name, item in geometries.items()}, "validation_passed": validation["passed"]}), flush=True)
    return solids, record

def verify_record(record, output, preparation):
    if record["study_input_sha256"] != preparation["study_input_sha256"]:
        raise ValueError("Conflicting mesh result identity")
    if not _valid_artifacts(record, output):
        raise ValueError("Mesh result artifacts failed hash verification")
    result = record["result"]
    if result["status"] != record["status"]:
        raise ValueError("Conflicting record/result status")
    if result.get("artifacts", {}).get("result"):
        raw_result_path = Path(result["artifacts"]["result"])
        if not raw_result_path.resolve().is_relative_to(output):
            raise ValueError("Raw result escapes the study directory")
        if _read(raw_result_path) != result:
            raise ValueError("Recorded FEA differs from its hashed raw result")
    if record["status"] == "ok":
        _verify_cases(result, preparation["load_cases"])

def summarize(output, preparation):
    rows = []
    for path in sorted((output / "results").glob("*/record.json")):
        record = _read(path)
        verify_record(record, output, preparation)
        row = {key: record[key] for key in ("geometry", "mesh_size_mm", "status")}
        row["record"] = str(path.relative_to(output)).replace("\\", "/")
        row["record_sha256"] = _file_digest(path)
        if record["status"] == "ok":
            result = record["result"]
            row.update(metrics=_metrics(result), mesh=result["mesh"], runtime_s=result["runtime_s"], point_mass_coupling=result["point_mass_coupling"])
        else:
            row["diagnostics"] = record["result"].get("diagnostics", [])
        rows.append(row)
    changes = []
    for name in ("baseline", "candidate"):
        complete = sorted([row for row in rows if row["geometry"] == name and row["status"] == "ok"], key=lambda row: -row["mesh_size_mm"])
        for coarse, fine in zip(complete, complete[1:]):
            relative = {metric: abs(fine["metrics"][metric] / coarse["metrics"][metric] - 1) for metric in SCREEN}
            changes.append({"geometry": name, "coarse_mm": coarse["mesh_size_mm"], "fine_mm": fine["mesh_size_mm"], "relative_changes": relative, "screen_passed": all(relative[key] <= SCREEN[key] for key in SCREEN)})
    comparisons = []
    for size in sorted({row["mesh_size_mm"] for row in rows}, reverse=True):
        matching = {row["geometry"]: row for row in rows if row["mesh_size_mm"] == size and row["status"] == "ok"}
        if len(matching) == 2:
            results = {name: _read(output / row["record"])["result"] for name, row in matching.items()}
            comparisons.append({"mesh_size_mm": size, **compare_to_baseline(results["candidate"], results["baseline"], preparation["relative_constraints"])})
    summary = {"schema_version": "deep-frame-workstation-mesh-summary-v1", "study_input_sha256": preparation["study_input_sha256"], "preparation": "preparation.json", "preparation_sha256": _file_digest(output / "preparation.json"), "rows": rows, "successive_mesh_changes": changes, "same_mesh_comparisons": comparisons, "screen": SCREEN, "interpretation": preparation["screen_interpretation"]}
    _save(output / "summary.json", summary)
    return summary

def mesh_main(overrides):
    config = configure(MESH_CONFIG, MESH_KINDS, overrides)
    if config["threads"] < 1 or any(size <= 0 or not np.isfinite(size) for size in config["mesh_sizes"]):
        raise ValueError("Mesh sizes and threads must be positive")
    output = config["output"].resolve()
    output.mkdir(parents=True, exist_ok=True)
    solids, preparation = mesh_prepare(output, config["threads"], config["linear_solver"])
    if config["prepare_only"]:
        return
    for size in config["mesh_sizes"]:
        for name, solid in solids.items():
            directory = output / "results" / (name + "_" + str(size).replace(".", "p") + "mm")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "record.json"
            if path.exists():
                existing = _read(path)
                verify_record(existing, output, preparation)
                if existing["status"] == "ok":
                    print(f"Verified cached {name} mesh={size} mm", flush=True)
                    continue
                raise ValueError("Existing failed/incomplete result; preserve it and choose a fresh output")
            settings = deepcopy(preparation["fea_settings"])
            settings.update(mesh_size_mm=size, work_dir=str(directory / "fea"))
            print(f"FEA start {name} mesh={size} mm", flush=True)
            result = evaluate(solid, deepcopy(preparation["material"]), deepcopy(preparation["point_masses"]), deepcopy(preparation["load_cases"]), settings)
            record = {"study_input_sha256": preparation["study_input_sha256"], "geometry": name, "mesh_size_mm": size, "status": result["status"], "settings": settings, "result": result, "artifacts": _fea_artifacts(result, output)}
            _save(path, record)
            summarize(output, preparation)
            print(json.dumps({"geometry": name, "mesh_size_mm": size, "status": result["status"], "runtime_s": result["runtime_s"], "diagnostics": result["diagnostics"]}), flush=True)
            _verify_cases(result, preparation["load_cases"])
    summary = summarize(output, preparation)
    print(json.dumps({"summary": str(output / "summary.json"), "completed_evaluations": len(summary["rows"])}), flush=True)

def physical_settings(settings):
    return {key: value for key, value in settings.items() if key != "work_dir"}

def load_source(source):
    manifest = _read(source / "manifest.json")
    if manifest["status"] != "ok":
        raise ValueError("Density study must have completed successfully")
    if not {"inputs.json", "result.json", "fields.npz", "domain_masks.npz"}.issubset(manifest["artifacts"]):
        raise ValueError("Density manifest is missing required artifacts")
    for name, expected in manifest["artifacts"].items():
        path = source / name
        if not path.is_file() or _file_digest(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError("Density study artifact mismatch: " + name)
    inputs = _read(source / "inputs.json")
    result = _read(source / "result.json")
    if result["status"] != "ok":
        raise ValueError("Density result must have completed successfully")
    domain = deepcopy(inputs["domain"])
    with np.load(source / "fields.npz", allow_pickle=False) as fields:
        density = fields["density"].copy()
        for name in ("allowed", "preserve", "forbidden"):
            mask = fields[name].copy()
            if hashlib.sha256(mask.tobytes()).hexdigest() != inputs["mask_sha256"][name]:
                raise ValueError("Density mask mismatch: " + name)
            domain[name] = mask
    if density.shape != tuple(domain["grid"]["shape"]) or not np.isfinite(density).all():
        raise ValueError("Invalid stored density field")
    return inputs, result, domain, density

def prepare(source, output, linear_solver=None, threads=2):
    if linear_solver not in (None, "SPOOLES", "PASTIX") or not isinstance(threads, int) or threads < 1:
        raise ValueError("Use linear_solver None/SPOOLES/PASTIX and a positive thread count")
    inputs, result, domain, density = load_source(source)
    acceptance = _read(EVIDENCE / "acceptance.json")
    for name, expected in acceptance["versioned_artifacts"].items():
        path = EVIDENCE / name
        if _file_digest(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError("Historical evidence changed: " + name)
    historical = _read(EVIDENCE / "inputs.json")
    baseline_parameters = {key: value for key, value in inputs["parameters"].items() if key != "topology"}
    historical_parameters = {key: value for key, value in historical["parameters"].items() if key != "topology"}
    if _digest(baseline_parameters) != _digest(historical_parameters):
        raise ValueError("Density study changed the historical baseline geometry parameters")
    for name in ("material", "point_masses", "comparison_load_cases"):
        if _digest(domain[name]) != _digest(historical["domain"][name]):
            raise ValueError("Density study changed the historical physical comparison: " + name)
    settings = _merge(domain["fea_settings"], historical["settings"]["fea_settings"])
    settings.update(mesh_size_mm=3.0, mesh_threads=1, threads=threads,
                    mesh_timeout_s=600.0, solver_timeout_s=900.0)
    if linear_solver is not None:
        settings["linear_solver"] = linear_solver
    provenance = _provenance({"reconstructor": reconstruct_topology,
                              "validator": validate_topology, "evaluator": evaluate,
                              "baseline_builder": build_geometry}, settings, True)
    provenance["runner_sha256"] = _file_digest(__file__)
    identity_provenance = {key: value for key, value in provenance.items()
                           if key not in ("git_revision", "git_tracked_dirty")}
    config = historical["settings"]
    record = {
        "schema_version": "deep-frame-workstation-candidate-study-v1",
        "source_directory": str(source),
        "source_manifest_sha256": _file_digest(source / "manifest.json"),
        "source_inputs_sha256": _file_digest(source / "inputs.json"),
        "source_fields_sha256": _file_digest(source / "fields.npz"),
        "historical_acceptance_sha256": _file_digest(EVIDENCE / "acceptance.json"),
        "historical_inputs_sha256": _file_digest(EVIDENCE / "inputs.json"),
        "parameters": inputs["parameters"],
        "density_summary": result["summary"],
        "domain": {key: value for key, value in domain.items() if not isinstance(value, np.ndarray)},
        "fea_settings": settings,
        "thresholds": config["density_thresholds"],
        "reconstruction_settings": _merge(domain["reconstruction_settings"], config["reconstruction_settings"]),
        "relative_constraints": config["relative_constraints"],
        "baseline_reference_frame_mass_g": acceptance["comparison"]["reference"]["frame_mass_g"],
        "provenance": identity_provenance,
        "comparison_policy": "Unchanged historical material, battery, loads, selectors and acceptance limits; independent C3D10 FEA at 3 mm for baseline and candidates.",
    }
    fingerprint = _digest(record)
    record["input_sha256"] = fingerprint
    path = output / "inputs.json"
    if path.exists():
        if _read(path) != record:
            raise ValueError("Candidate study inputs/provenance changed; use a fresh output")
    else:
        if any(output.iterdir()):
            raise ValueError("A new candidate study requires an empty output directory")
        _save(path, record)
    return record, domain, density

def persist(output, manifest, current):
    path = output / "candidates" / current["id"] / "record.json"
    _save(path, current)
    manifest["candidates"][current["id"]] = _artifact(path, output)
    _save(output / "manifest.json", manifest)

def verify_raw_result(artifacts, result, output):
    artifact = artifacts.get("raw_fea_result.json")
    if artifact is None:
        if result.get("status") == "ok":
            raise ValueError("Successful FEA lacks a hashed raw result")
        return
    if not _valid_artifacts({"artifacts": {"raw_result": artifact}}, output):
        raise ValueError("Raw FEA result failed its hash check")
    if _read(output / artifact["path"]) != result:
        raise ValueError("Recorded FEA differs from the hashed raw result")

def read_candidates(output, manifest, inputs=None, baseline=None):
    records = []
    for name, artifact in manifest["candidates"].items():
        if not _valid_artifacts({"artifacts": {name: artifact}}, output):
            raise ValueError("Stored candidate record changed: " + name)
        record = _read(output / artifact["path"])
        if not _valid_artifacts(record, output):
            raise ValueError("Stored candidate artifact changed: " + name)
        if "fea" in record:
            verify_raw_result(record["artifacts"], record["fea"], output)
        if record["status"] == "ok":
            if inputs is None:
                raise ValueError("Accepted candidate requires explicit physical inputs")
            if not record.get("checks", {}).get("passed"):
                raise ValueError("Accepted candidate lacks geometry/manufacturing validation")
            _verify_cases(record.get("fea", {}), inputs["domain"]["comparison_load_cases"])
        if baseline is not None and "comparison" in record:
            comparison = compare_to_baseline(record["fea"], baseline["result"], inputs["relative_constraints"])
            if comparison != record["comparison"]:
                raise ValueError("Stored candidate comparison changed")
            if (record["status"] == "ok") != comparison["passed"]:
                raise ValueError("Candidate acceptance disagrees with its mechanical comparison")
        records.append(record)
    return records

def baseline_result(output, inputs, baseline_study):
    domain = inputs["domain"]
    if baseline_study is not None:
        preparation_path = baseline_study / "preparation.json"
        preparation = _read(preparation_path)
        if preparation["acceptance_sha256"] != inputs["historical_acceptance_sha256"]:
            raise ValueError("Shared baseline historical acceptance mismatch")
        if preparation["versioned_artifact_checks"]["inputs.json"]["sha256"] != inputs["historical_inputs_sha256"]:
            raise ValueError("Shared baseline geometry/input provenance mismatch")
        for name, key in (("material", "material"), ("point_masses", "point_masses"), ("load_cases", "comparison_load_cases")):
            if _digest(preparation[name]) != _digest(domain[key]):
                raise ValueError("Shared baseline physical contract mismatch: " + name)
        if physical_settings(preparation["fea_settings"]) != physical_settings(inputs["fea_settings"]):
            raise ValueError("Shared baseline FEA settings mismatch")
        for name in ("packages", "python_version", "platform", "source_sha256", "solver"):
            if preparation["provenance"][name] != inputs["provenance"][name]:
                raise ValueError("Shared baseline provenance mismatch: " + name)
        record_path = baseline_study / "results/baseline_3p0mm/record.json"
        record = _read(record_path)
        if record["geometry"] != "baseline" or record["mesh_size_mm"] != 3.0 or record["study_input_sha256"] != preparation["study_input_sha256"]:
            raise ValueError("Shared baseline identity mismatch")
        if physical_settings(record["settings"]) != physical_settings(inputs["fea_settings"]):
            raise ValueError("Shared baseline actual FEA settings mismatch")
        if not record["artifacts"] or not _valid_artifacts(record, baseline_study):
            raise ValueError("Shared baseline raw artifact mismatch")
        raw_result = record["artifacts"].get("raw_fea_result.json")
        if raw_result is None or _read(baseline_study / raw_result["path"]) != record["result"]:
            raise ValueError("Shared baseline record differs from hashed raw result")
        _verify_cases(record["result"], domain["comparison_load_cases"])
        if abs(record["result"]["frame_mass_g"] - inputs["baseline_reference_frame_mass_g"]) > 1e-7:
            raise ValueError("Shared baseline exact CAD mass mismatch")
        return {"result": record["result"], "reuse": {
            "study_directory": str(baseline_study),
            "preparation_path": str(preparation_path), "preparation_sha256": _file_digest(preparation_path),
            "record_path": str(record_path), "record_sha256": _file_digest(record_path),
            "raw_artifacts_verified": len(record["artifacts"]),
        }}
    path = output / "baseline/record.json"
    if path.exists():
        record = _read(path)
        if physical_settings(record["settings"]) != physical_settings(inputs["fea_settings"]):
            raise ValueError("Stored baseline FEA settings changed")
        if not record["artifacts"] or not _valid_artifacts(record, output):
            raise ValueError("Stored baseline raw artifact mismatch")
        verify_raw_result(record["artifacts"], record["result"], output)
        _verify_cases(record["result"], domain["comparison_load_cases"])
        if abs(record["result"]["frame_mass_g"] - inputs["baseline_reference_frame_mass_g"]) > 1e-7:
            raise ValueError("Stored baseline exact CAD mass mismatch")
        return record
    settings = deepcopy(inputs["fea_settings"])
    settings["work_dir"] = str(output / "baseline/fea")
    solid = build_geometry(deepcopy(inputs["parameters"]))
    result = evaluate(solid, deepcopy(domain["material"]), deepcopy(domain["point_masses"]),
                      deepcopy(domain["comparison_load_cases"]), settings)
    record = {"result": result, "artifacts": _fea_artifacts(result, output), "settings": settings}
    _save(path, record)
    _verify_cases(result, domain["comparison_load_cases"])
    return record

def load_saved_baseline(output, manifest, inputs):
    artifact = manifest.get("baseline")
    if artifact is None:
        return None
    if not _valid_artifacts({"artifacts": {"baseline": artifact}}, output):
        raise ValueError("Stored baseline record changed")
    saved = _read(output / artifact["path"])
    study = Path(saved["reuse"]["study_directory"]) if "reuse" in saved else None
    verified = baseline_result(output, inputs, study)
    if verified != saved:
        raise ValueError("Stored baseline evidence changed")
    return saved

def run(source, output, geometry_only=False, baseline_study=None, linear_solver=None, threads=2, run_log=None):
    output.mkdir(parents=True, exist_ok=True)
    clocks = {}
    inputs, domain, density = prepare(source, output, linear_solver, threads)
    path = output / "manifest.json"
    manifest = _read(path) if path.exists() else {
        "schema_version": "deep-frame-workstation-candidate-results-v1",
        "input_sha256": inputs["input_sha256"], "status": "running", "candidates": {},
    }
    if manifest["input_sha256"] != inputs["input_sha256"]:
        raise ValueError("Stored manifest input identity mismatch")
    baseline = load_saved_baseline(output, manifest, inputs)
    records = read_candidates(output, manifest, inputs, baseline)
    if any(record["status"] == "ok" for record in records) and baseline is None:
        raise ValueError("Accepted candidates lack a verified baseline record")
    baseline_solid = build_geometry(deepcopy(inputs["parameters"]))
    baseline_mass = float(baseline_solid.volume * domain["material"]["density_g_cm3"] / 1000)
    if abs(baseline_mass - inputs["baseline_reference_frame_mass_g"]) > 1e-7:
        raise ValueError("Reconstructed baseline CAD mass differs from the historical reference")
    manifest["baseline_cad_frame_mass_g"] = baseline_mass
    for index, threshold in enumerate(inputs["thresholds"]):
        candidate_id = f"density_t{index:02d}"
        if candidate_id in manifest["candidates"]:
            continue
        directory = output / "candidates" / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        settings = {**inputs["reconstruction_settings"], "density_threshold": threshold}
        record = {"id": candidate_id, "density_threshold": threshold, "reconstruction_settings": settings,
                  "stage": "geometry", "status": "invalid", "artifacts": {}, "diagnostics": []}
        print(json.dumps({"event": "reconstruct", "candidate": candidate_id, "threshold": threshold}), flush=True)
        clock = perf_counter()
        try:
            solid = reconstruct_topology(deepcopy(domain), density.copy(), deepcopy(settings))
            record["reconstruction"] = deepcopy(solid.topology_report)
            record["frame_mass_g"] = float(solid.volume * domain["material"]["density_g_cm3"] / 1000)
            for suffix, exporter in (("step", export_step), ("stl", export_stl)):
                geometry_path = directory / ("geometry." + suffix)
                exporter(solid, geometry_path)
                record["artifacts"][suffix] = _artifact(geometry_path, output)
            record["checks"] = validate_topology(solid, deepcopy(domain), deepcopy(settings))
            if not record["checks"]["passed"]:
                raise ValueError("Geometry/manufacturing failed: " + "; ".join(record["checks"]["violations"]))
            if record["frame_mass_g"] > baseline_mass * inputs["relative_constraints"]["frame_mass_ratio_max"]:
                record["stage"] = "mass_constraint"
                raise ValueError("Exact CAD frame mass exceeds historical baseline-relative limit")
            record.update(stage="awaiting_independent_fea", status="pending")
        except ValueError as error:
            record["diagnostics"].append(str(error))
        except Exception as error:
            record.update(status="failed")
            record["diagnostics"].append(type(error).__name__ + ": " + str(error))
        clocks[candidate_id] = perf_counter() - clock
        persist(output, manifest, record)
        print(json.dumps({"event": "geometry_result", "id": candidate_id, "status": record["status"],
                          "diagnostics": record["diagnostics"]}), flush=True)
    records = read_candidates(output, manifest, inputs, baseline)
    if not geometry_only and any(record["status"] == "pending" for record in records):
        baseline = baseline or baseline_result(output, inputs, baseline_study)
        _save(output / "baseline.json", baseline)
        manifest["baseline"] = _artifact(output / "baseline.json", output)
        _save(output / "manifest.json", manifest)
        for record in records:
            if record["status"] != "pending":
                continue
            directory = output / "candidates" / record["id"]
            solid = import_step(output / record["artifacts"]["step"]["path"])
            settings = deepcopy(inputs["fea_settings"])
            settings["work_dir"] = str(directory / "fea")
            print(json.dumps({"event": "fea_start", "candidate": record["id"]}), flush=True)
            clock = perf_counter()
            result = evaluate(solid, deepcopy(domain["material"]), deepcopy(domain["point_masses"]),
                              deepcopy(domain["comparison_load_cases"]), settings)
            record.update(stage="independent_fea", status=result["status"], fea=result, fea_settings=settings)
            record["artifacts"].update(_fea_artifacts(result, output))
            try:
                _verify_cases(result, domain["comparison_load_cases"])
                record["comparison"] = compare_to_baseline(result, baseline["result"], inputs["relative_constraints"])
                record.update(stage="complete", status="ok" if record["comparison"]["passed"] else "invalid")
                if not record["comparison"]["passed"]:
                    record["diagnostics"].append("Failed mechanical screens: " + ", ".join(name for name, passed in record["comparison"]["checks"].items() if not passed))
            except ValueError as error:
                record["status"] = "failed" if result["status"] == "failed" else "invalid"
                record["diagnostics"].append(str(error))
            clocks[record["id"]] = clocks.get(record["id"], 0.0) + perf_counter() - clock
            persist(output, manifest, record)
            print(json.dumps({"event": "fea_result", "id": record["id"], "status": record["status"],
                              "diagnostics": record["diagnostics"]}), flush=True)
    records = read_candidates(output, manifest, inputs, baseline)
    manifest["pareto_ids"] = _pareto(records)
    valid = [record for record in records if record["status"] == "ok"]
    manifest["selected_id"] = min(valid, key=lambda record: record["comparison"]["selection_score"])["id"] if valid else None
    manifest["status"] = "geometry_only" if geometry_only else "ok" if valid else "invalid"
    manifest["summary"] = [{key: record[key] for key in ("id", "density_threshold", "status", "stage", "diagnostics")} for record in records]
    _save(path, manifest)
    for record in records:
        if record["id"] in clocks:
            accepted = record["status"] == "ok"
            log_run(run_log, {**candidate_entry("raster", output, {"id": record["id"], "parameters": {"threshold": record["density_threshold"]}, "status": "accepted" if accepted else record["status"],
                                                                   "failure_stage": None if accepted else record["stage"], "runtime_s": clocks[record["id"]]}, inputs["input_sha256"]), "raster_status": record["status"]})
    print(json.dumps({"event": "finished", "manifest": str(path), "status": manifest["status"], "selected_id": manifest["selected_id"]}), flush=True)
    return manifest

def candidates_main(overrides):
    config = configure(CANDIDATES_CONFIG, CANDIDATES_KINDS, overrides, ("source", "output"))
    run(config["source"].resolve(), config["output"].resolve(), config["geometry_only"],
        config["baseline_study"].resolve() if config["baseline_study"] else None,
        config["linear_solver"], config["threads"], config["run_log"])

def render_main(overrides):
    config = configure(RENDER_CONFIG, RENDER_KINDS, overrides)
    base = ROOT / "exports/topology/workstation_20260930"
    mesh_dir = base / "mesh_study"
    study_dir = base / "candidate_study/grid4_iter300"
    preparation = _read(mesh_dir / "preparation.json")
    selection = _read(study_dir / "manifest.json")
    selected_id = selection["selected_id"]
    record_path = study_dir / "candidates" / selected_id / "record.json"
    selected = _read(record_path)
    if selected["status"] != "ok" or not selected["comparison"]["passed"]:
        raise ValueError("Selected workstation candidate must have passed geometry and independent FEA")
    old_fea = _read(ROOT / "docs/validation/topology_phase1/candidate_fea.json")
    baseline_fea = _read(ROOT / "docs/validation/topology_phase1/baseline_fea.json")
    sources = [
        ("v0", mesh_dir, preparation["geometries"]["baseline"]["stl"], baseline_fea["frame_mass_g"], "Parametrische v0", "#7994ab"),
        ("handoff_45", mesh_dir, preparation["geometries"]["candidate"]["stl"], old_fea["frame_mass_g"], "Freie Referenz: 45 Updates, t = 0.20", "#b07849"),
        ("workstation_300", study_dir, selected["artifacts"]["stl"], selected["fea"]["frame_mass_g"], f"Workstation: 300 Updates, t = {selected['density_threshold']:.2f}", "#248c87"),
    ]
    grid = _read(ROOT / "docs/validation/topology_phase1/inputs.json")["domain"]["grid"]
    lower = np.asarray(grid["origin_mm"], dtype=float)
    size = np.asarray(grid["shape"]) * np.asarray(grid["spacing_mm"])
    upper = lower + size
    plt, collection = _plot_modules()
    figure = plt.figure(figsize=(16, 10), layout="constrained")
    manifest = {
        "schema_version": "deep-frame-workstation-geometry-visualization-v1",
        "rendering": "Direct original STL triangles, Matplotlib orthographic shaded projections, identical physical limits and scale in all panels; no geometry smoothing or remodeling",
        "grid": grid,
        "axis_limits_mm": [lower.tolist(), upper.tolist()],
        "views": [{"elevation_deg": 28, "azimuth_deg": -45}, {"elevation_deg": 90, "azimuth_deg": -90}],
        "packages": {name: version(name) for name in ("numpy", "matplotlib", "trimesh")},
        "renderer_sha256": _file_digest(__file__),
        "selected_id": selected_id,
        "selected_manifest_sha256": _file_digest(study_dir / "manifest.json"),
        "selected_record_sha256": _file_digest(record_path),
        "sources": [],
    }
    for index, (name, directory, expected, mass, title, color) in enumerate(sources):
        path = directory / expected["path"]
        if _file_digest(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError("STL source failed its recorded artifact hash: " + name)
        raw_mesh = trimesh.load_mesh(path, process=False)
        triangles = np.asarray(raw_mesh.triangles)
        if not np.all(np.isfinite(triangles)):
            raise ValueError("Non-finite STL coordinates")
        checked = raw_mesh.copy()
        checked.process(validate=False)
        if not checked.is_watertight or not checked.is_volume or checked.body_count != 1:
            raise ValueError("Expected one closed STL body: " + name)
        if np.any(checked.bounds[0] < lower - 1e-5) or np.any(checked.bounds[1] > upper + 1e-5):
            raise ValueError("The shared physical plot bounds would clip geometry")
        for row in (0, 1):
            axis = figure.add_subplot(2, 3, index + 1 + row * 3, projection="3d")
            axis.add_collection3d(collection(triangles, facecolors=color, linewidths=0, shade=True))
            axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]), xlabel="x / mm", ylabel="y / mm", zlabel="z / mm")
            axis.set_box_aspect(size)
            axis.set_proj_type("ortho")
            axis.view_init(elev=28 if row == 0 else 90, azim=-45 if row == 0 else -90)
            axis.grid(False)
            axis.set_xticks([-60, -30, 0, 30, 60])
            axis.set_yticks([-60, -30, 0, 30, 60])
            axis.set_zticks([0, 16, 32] if row == 0 else [])
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(f"{title}\nFrame-Masse {mass:.3f} g", fontsize=12)
            else:
                axis.set_zlabel("")
                axis.set_title("Draufsicht", fontsize=11)
        manifest["sources"].append({
            "name": name, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": expected["sha256"], "size_bytes": expected["size_bytes"],
            "triangle_count": len(triangles), "bounds_mm": checked.bounds.tolist(),
            "stl_volume_mm3": float(checked.volume), "cad_frame_mass_g": mass,
            "watertight": bool(checked.is_watertight), "body_count": int(checked.body_count),
            "topology_check_note": "Vertex merging used only for closed-body checks; rendering uses original STL triangles",
        })
    figure.suptitle("Deep_Frame | Workstation: drei Strukturen im gleichen Maßstab\n4-mm-Designraster der freien Strukturen; identische orthografische Ansichten", fontsize=15)
    output = config["output"].resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    manifest["image"] = {"path": str(output.relative_to(ROOT)).replace("\\", "/"), "sha256": _file_digest(output), "size_bytes": output.stat().st_size}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"image": str(output), "manifest": str(output.with_suffix('.json')), "sources": len(sources)}))

def main(argv=None):
    return command_line({"mesh": mesh_main, "candidates": candidates_main, "render": render_main}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
