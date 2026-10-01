"""Validate a completed density study against the original Phase-1 screens.

Use --geometry-only first to separate reconstruction from expensive FEA.
Existing evidence is resumed only after input, record and artifact hash checks.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from build123d import export_step, export_stl, import_step

from deep_frame.fea import evaluate
from deep_frame.geometry import build_geometry
from deep_frame.topology_geometry import reconstruct_topology, validate_topology
from deep_frame.topology_pipeline import (
    _artifact, _digest, _fea_artifacts, _file_digest, _merge, _pareto,
    _provenance, _save, _valid_artifacts, _verify_cases, compare_to_baseline,
)

EVIDENCE = ROOT / "docs/validation/topology_phase1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def physical_settings(settings):
    return {key: value for key, value in settings.items() if key != "work_dir"}


def load_source(source):
    manifest = read(source / "manifest.json")
    if manifest["status"] != "ok":
        raise ValueError("Density study must have completed successfully")
    if not {"inputs.json", "result.json", "fields.npz", "domain_masks.npz"}.issubset(manifest["artifacts"]):
        raise ValueError("Density manifest is missing required artifacts")
    for name, expected in manifest["artifacts"].items():
        path = source / name
        if not path.is_file() or _file_digest(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError("Density study artifact mismatch: " + name)
    inputs = read(source / "inputs.json")
    result = read(source / "result.json")
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
    acceptance = read(EVIDENCE / "acceptance.json")
    for name, expected in acceptance["versioned_artifacts"].items():
        path = EVIDENCE / name
        if _file_digest(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError("Historical evidence changed: " + name)
    historical = read(EVIDENCE / "inputs.json")
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
        if read(path) != record:
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
    if read(output / artifact["path"]) != result:
        raise ValueError("Recorded FEA differs from the hashed raw result")


def read_candidates(output, manifest, inputs=None, baseline=None):
    records = []
    for name, artifact in manifest["candidates"].items():
        if not _valid_artifacts({"artifacts": {name: artifact}}, output):
            raise ValueError("Stored candidate record changed: " + name)
        record = read(output / artifact["path"])
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
        preparation = read(preparation_path)
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
        record = read(record_path)
        if record["geometry"] != "baseline" or record["mesh_size_mm"] != 3.0 or record["study_input_sha256"] != preparation["study_input_sha256"]:
            raise ValueError("Shared baseline identity mismatch")
        if physical_settings(record["settings"]) != physical_settings(inputs["fea_settings"]):
            raise ValueError("Shared baseline actual FEA settings mismatch")
        if not record["artifacts"] or not _valid_artifacts(record, baseline_study):
            raise ValueError("Shared baseline raw artifact mismatch")
        raw_result = record["artifacts"].get("raw_fea_result.json")
        if raw_result is None or read(baseline_study / raw_result["path"]) != record["result"]:
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
        record = read(path)
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
    saved = read(output / artifact["path"])
    study = Path(saved["reuse"]["study_directory"]) if "reuse" in saved else None
    verified = baseline_result(output, inputs, study)
    if verified != saved:
        raise ValueError("Stored baseline evidence changed")
    return saved


def run(source, output, geometry_only=False, baseline_study=None, linear_solver=None, threads=2):
    output.mkdir(parents=True, exist_ok=True)
    inputs, domain, density = prepare(source, output, linear_solver, threads)
    path = output / "manifest.json"
    manifest = read(path) if path.exists() else {
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
    print(json.dumps({"event": "finished", "manifest": str(path), "status": manifest["status"], "selected_id": manifest["selected_id"]}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-study", type=Path)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--linear-solver", choices=("SPOOLES", "PASTIX"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.geometry_only,
        args.baseline_study.resolve() if args.baseline_study else None,
        args.linear_solver, args.threads)


if __name__ == "__main__":
    main()
