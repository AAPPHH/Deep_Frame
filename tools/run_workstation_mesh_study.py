"""Reconstruct the archived density and independently refine its FEA mesh.

This does not run topology optimization or alter the historical acceptance files.
Run from the repository with the local venv Python; see --help.
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
from build123d import export_step, export_stl

from deep_frame.fea import evaluate
from deep_frame.geometry import build_geometry
from deep_frame.topology_geometry import reconstruct_topology, validate_topology
from deep_frame.topology_pipeline import (
    _artifact, _digest, _fea_artifacts, _file_digest, _metrics, _provenance,
    _save, _valid_artifacts, _verify_cases, compare_to_baseline,
)

EVIDENCE = ROOT / "docs/validation/topology_phase1"
DEFAULT_OUTPUT = ROOT / "exports/topology/workstation_20260930/mesh_study"
SCREEN = {
    "stiffness_n_per_mm": 0.02,
    "max_displacement_mm": 0.02,
    "first_frequency_hz": 0.02,
    "max_von_mises_mpa": 0.05,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def geometry_metrics(solid):
    bounds = solid.bounding_box()
    return {
        "volume_mm3": float(solid.volume),
        "bounds_min_mm": list(bounds.min),
        "bounds_max_mm": list(bounds.max),
        "solids": len(solid.solids()),
        "is_valid": solid.is_valid,
    }


def prepare(output, threads, linear_solver=None):
    previous_path = output / "preparation.json"
    if not previous_path.exists() and any(output.iterdir()):
        raise ValueError("A new study requires an empty output directory; preserve partial files and use a fresh directory")
    acceptance = read(EVIDENCE / "acceptance.json")
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
    inputs = read(EVIDENCE / "inputs.json")
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
    reconstruction_settings = read(EVIDENCE / "candidate_record.json")["reconstruction_settings"]
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
    provenance["study_runner_sha256"] = _file_digest(Path(__file__))
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
        previous = read(previous_path)
        if previous["study_input_sha256"] != record["study_input_sha256"]:
            raise ValueError("Study provenance changed; choose a fresh --output directory")
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
        if read(raw_result_path) != result:
            raise ValueError("Recorded FEA differs from its hashed raw result")
    if record["status"] == "ok":
        _verify_cases(result, preparation["load_cases"])


def summarize(output, preparation):
    rows = []
    for path in sorted((output / "results").glob("*/record.json")):
        record = read(path)
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
            results = {name: read(output / row["record"])["result"] for name, row in matching.items()}
            comparisons.append({"mesh_size_mm": size, **compare_to_baseline(results["candidate"], results["baseline"], preparation["relative_constraints"])})
    summary = {"schema_version": "deep-frame-workstation-mesh-summary-v1", "study_input_sha256": preparation["study_input_sha256"], "preparation": "preparation.json", "preparation_sha256": _file_digest(output / "preparation.json"), "rows": rows, "successive_mesh_changes": changes, "same_mesh_comparisons": comparisons, "screen": SCREEN, "interpretation": preparation["screen_interpretation"]}
    _save(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh-sizes", nargs="+", type=float, default=[3.0, 2.5, 2.0])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--linear-solver", choices=("SPOOLES", "PASTIX"), default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or any(size <= 0 or not np.isfinite(size) for size in args.mesh_sizes):
        parser.error("Mesh sizes and threads must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    solids, preparation = prepare(output, args.threads, args.linear_solver)
    if args.prepare_only:
        return
    for size in args.mesh_sizes:
        for name, solid in solids.items():
            directory = output / "results" / (name + "_" + str(size).replace(".", "p") + "mm")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "record.json"
            if path.exists():
                existing = read(path)
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


if __name__ == "__main__":
    main()
