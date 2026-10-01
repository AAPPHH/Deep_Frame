"""Reconstruct density surfaces, validate final STEP CAD and mechanically screen candidates."""

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from build123d import export_step, import_step

from deep_frame.fea import evaluate
from deep_frame.geometry import build_geometry
from deep_frame.topology_pipeline import _merge, _provenance, _verify_cases, compare_to_baseline
from deep_frame.topology_surface import reconstruct_surface
from tools.run_workstation_candidate_study import load_source


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    """Atomically replace progress files so observers never read partial JSON."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(21):
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 20:
                raise
            time.sleep(0.05)


def artifacts(directory):
    return {p.relative_to(directory).as_posix(): {"sha256": digest(p), "size_bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()
            and not (p.parent == directory and p.name in ("record.json", "manifest.json", "status.json"))}


def acceptance(records, complete):
    accepted = [record for record in records if record.get("status") == "accepted"
                and record.get("validation", {}).get("passed") is True
                and record.get("validation", {}).get("checks", {}).get("surface_maturity", {}).get("passed") is True
                and record.get("mass_screen", {}).get("passed") is True
                and record.get("fea", {}).get("status") == "ok"
                and record.get("comparison", {}).get("passed") is True]
    selected = min(accepted, key=lambda item: (item["comparison"]["selection_score"], item["id"]), default=None)
    return {"accepted_count": len(accepted), "selected_id": selected["id"] if complete and selected else None,
            "overall_acceptance": bool(complete and accepted)}


def save_geometry_evidence(directory, stage, report, shape=None, mesh=None):
    """Keep exact diagnostic CAD/mesh even when reconstruction cannot return a final solid."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stage)
    result = {"stage": stage, "report": report, "artifacts": {}, "errors": []}
    for suffix, value, exporter in (("step", shape, export_step), ("ply", mesh, lambda value, path: value.export(path))):
        if value is None:
            continue
        path = directory / (stem + "." + suffix)
        try:
            exporter(value, path)
            if not path.is_file() or not path.stat().st_size:
                raise ValueError("Geometry export produced no data")
            result["artifacts"][path.name] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        except Exception as error:
            result["errors"].append(type(error).__name__ + ": " + str(error))
    write(directory / (stem + ".json"), result)
    return result


def export_final_mesh(solid, candidate, validator, settings):
    mesh = validator._mesh(solid, validator._settings(settings))
    mesh.export(candidate / "geometry.stl")
    mesh.export(candidate / "geometry.ply")
    report = {"source": "geometry.step imported final CAD", "source_step_sha256": digest(candidate / "geometry.step"),
              "cad_geometry_modified": False, "tessellation": mesh.metadata["adaptive_tessellation"],
              "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
    write(candidate / "final_mesh.json", report)
    return report


def run_candidate(candidate, record, domain, density, reconstruction, validator, validation_settings,
                  reference, baseline_solid, baseline_mass, baseline, settings, constraints, geometry_only, progress):
    record["status"] = "reconstructing"
    progress({"stage": "reconstructing"})
    solid = reconstruct_surface(domain, density, {**reconstruction, "density_threshold": record["threshold"]}, progress=progress)
    record["reconstruction"] = solid.surface_report
    progress({"stage": "step_export"})
    export_step(solid, candidate / "geometry.step")
    imported = import_step(candidate / "geometry.step")
    change = imported.volume / solid.volume - 1
    record["step_roundtrip"] = {"valid": bool(imported.is_valid), "solids": len(imported.solids()),
                                "original_volume_mm3": solid.volume, "imported_volume_mm3": imported.volume,
                                "relative_volume_change": change}
    if not imported.is_valid or len(imported.solids()) != 1 or abs(change) > 1e-8:
        raise ValueError("STEP roundtrip changed the valid single body or its volume")
    record["status"] = "validating"
    progress({"stage": "validation"})
    record["validation"] = validator.validate_surface(imported, domain, validation_settings,
                                                       reference_solid=reference, progress=progress)
    record["mass_g"] = imported.volume * domain["material"]["density_g_cm3"] / 1000
    ratio = record["mass_g"] / baseline_mass
    record["mass_screen"] = {"ratio_to_v0": ratio, "maximum_ratio": constraints["frame_mass_ratio_max"],
                             "passed": ratio <= constraints["frame_mass_ratio_max"]}
    passed = (record["validation"].get("passed") is True and record["mass_screen"]["passed"]
              and record["validation"].get("checks", {}).get("surface_maturity", {}).get("passed") is True)
    record["status"] = "geometry_valid" if passed else "geometry_invalid"
    progress({"stage": "validation_finished", "passed": passed})
    # Validation evidence must exist before an STL failure can reject the candidate.
    try:
        record["final_mesh"] = export_final_mesh(imported, candidate, validator, validation_settings)
    except Exception as error:
        record["final_mesh"] = {"error": type(error).__name__ + ": " + str(error)}
        if passed:
            raise
        record["diagnostics"].append("Final CAD mesh export: " + str(error))
    if not passed or geometry_only:
        return baseline
    if baseline is None:
        record["status"] = "baseline_fea"
        progress({"stage": "baseline_fea"})
        directory = candidate.parents[1] / "baseline"
        directory.mkdir(exist_ok=True)
        result = evaluate(baseline_solid, domain["material"], domain["point_masses"], domain["comparison_load_cases"],
                          {**settings, "work_dir": str(directory / "fea")})
        write(directory / "raw_fea_result.json", result)
        _verify_cases(result, domain["comparison_load_cases"])
        baseline = result
        write(directory / "record.json", {"status": "ok", "result": baseline, "artifacts": artifacts(directory)})
    record["status"] = "candidate_fea"
    progress({"stage": "candidate_fea"})
    result = evaluate(imported, domain["material"], domain["point_masses"], domain["comparison_load_cases"],
                      {**settings, "work_dir": str(candidate / "fea")})
    write(candidate / "raw_fea_result.json", result)
    record["fea"] = result
    _verify_cases(result, domain["comparison_load_cases"])
    record["comparison"] = compare_to_baseline(result, baseline, constraints)
    record["status"] = "accepted" if record["comparison"]["passed"] else "mechanically_rejected"
    return baseline


def run_study(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("A new empty output directory is required; old evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    manifest = {"schema_version": "deep-frame-continuous-geometry-study-v2", "status": "running", "stage": "preparing",
                "density_source": str(source), "geometry_only": args.geometry_only, "thresholds": args.thresholds,
                "accepted_count": 0, "selected_id": None, "overall_acceptance": False, "candidates": [],
                "acceptance_scope": "Geometry and independent FEA screens; final manufacturing/render review remains required",
                "study_timeout_s": args.study_timeout_s,
                "budget_policy": "Study budget checked at stage/progress boundaries; external orchestrator enforces process timeout"}
    write(output / "manifest.json", manifest)
    write(output / "status.json", {"status": "running", "stage": "preparing"})
    try:
        from deep_frame import topology_surface_validation as validator

        if not args.reference_step.is_file():
            raise ValueError("A reference STEP file is required for surface maturity")
        load_source(source)
        source_manifest = read(source / "manifest.json")
        snapshot = output / "density_source"
        snapshot.mkdir()
        for name in [*source_manifest["artifacts"], "manifest.json"]:
            original = (source / name).resolve()
            if not original.is_relative_to(source):
                raise ValueError("Density artifact path escapes its source directory: " + name)
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        inputs, density_result, domain, density = load_source(snapshot)
        historical = read(ROOT / "docs/validation/topology_phase1/inputs.json")
        for name in ("material", "point_masses", "comparison_load_cases"):
            if domain[name] != historical["domain"][name]:
                raise ValueError("Physical comparison input changed: " + name)
        if {k: v for k, v in inputs["parameters"].items() if k != "topology"} != {k: v for k, v in historical["parameters"].items() if k != "topology"}:
            raise ValueError("Historical v0 geometry parameters changed")
        provenance = output / "provenance"
        for path in list((ROOT / "deep_frame").glob("*.py")) + list(ROOT.glob("requirements*.txt")) + [Path(__file__), ROOT / "tools/run_workstation_candidate_study.py"]:
            target = provenance / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        settings = _merge(domain["fea_settings"], historical["settings"]["fea_settings"])
        settings.update(mesh_size_mm=3.0, mesh_threads=1, threads=1, linear_solver="SPOOLES",
                        mesh_timeout_s=args.mesh_timeout_s, solver_timeout_s=args.solver_timeout_s)
        reconstruction = {"interpolation_subdivisions": args.subdivisions, "interpolation_method": args.interpolation_method,
                          "density_smoothing_sigma_mm": args.density_smoothing_sigma_mm,
                          "manufacturing_opening_radius_mm": args.manufacturing_opening_radius_mm,
                          "manufacturing_opening_method": args.manufacturing_opening_method,
                          "surface_constraint_mode": args.surface_constraint_mode,
                          "free_forbidden_buffer_mm": args.free_forbidden_buffer_mm,
                          "decimation_bounds_mode": args.decimation_bounds_mode,
                          "preserve_fusion_mode": args.preserve_fusion_mode,
                          "boolean_fuzzy_value_mm": args.boolean_fuzzy_mm,
                          "maximum_surface_deviation_mm": args.surface_deviation_mm, "decimation_face_budgets": args.face_budgets}
        validation_settings = validator._settings({"maximum_wall_samples": args.maximum_wall_samples,
                                                    "boolean_fuzzy_mm": args.validation_boolean_fuzzy_mm})
        shutil.copy2(args.reference_step, output / "reference.step")
        reference = import_step(output / "reference.step")
        if not reference.is_valid or len(reference.solids()) != 1 or reference.volume <= 0:
            raise ValueError("Reference STEP must be a valid single positive-volume solid")
        manifest.update(source_manifest_sha256=digest(snapshot / "manifest.json"), density_status=density_result["status"],
                        reconstruction_settings=reconstruction, validation_settings=validation_settings, fea_settings=settings,
                        comparison_reference={"source": str(args.reference_step.resolve()), "sha256": digest(output / "reference.step"), "volume_mm3": reference.volume},
                        material=domain["material"], point_masses=domain["point_masses"], load_cases=domain["comparison_load_cases"],
                        packages={p: importlib.metadata.version(p) for p in ("build123d", "numpy", "scipy", "trimesh", "gmsh", "scikit-image", "fast-simplification", "rtree", "pymeshlab")},
                        source_artifacts=artifacts(provenance),
                        runtime_provenance=_provenance({"reconstructor": reconstruct_surface, "validator": validator.validate_surface,
                                                       "evaluator": evaluate, "baseline_builder": build_geometry}, settings, True),
                        relative_constraints=historical["settings"]["relative_constraints"])
        baseline_solid = build_geometry(deepcopy(inputs["parameters"]))
        baseline_mass = baseline_solid.volume * domain["material"]["density_g_cm3"] / 1000
        manifest["v0_geometry"] = {"volume_mm3": baseline_solid.volume, "frame_mass_g": baseline_mass}
        write(output / "manifest.json", manifest)
        baseline = None
        for index, threshold in enumerate(args.thresholds):
            if time.perf_counter() - started >= args.study_timeout_s:
                raise TimeoutError("Geometry study runtime budget exceeded before next candidate")
            candidate = output / "candidates" / f"t{index:02d}"
            candidate.mkdir(parents=True)
            record = {"id": candidate.name, "threshold": threshold, "status": "reconstructing", "diagnostics": []}
            event_count = 0

            def progress(event):
                nonlocal event_count
                event_count += 1
                event = dict(event)
                shape, mesh = event.pop("shape", None), event.pop("mesh", None)
                if "report" in event:
                    record["reconstruction"] = event["report"]
                if shape is not None or mesh is not None:
                    event["evidence"] = save_geometry_evidence(candidate / "intermediate", f"{event_count:04d}_{event['stage']}", event.get("report", {}), shape, mesh)
                event.update(candidate=record["id"], elapsed_s=time.perf_counter() - started)
                with (candidate / "progress.jsonl").open("a", encoding="utf-8") as journal:
                    journal.write(json.dumps(event, allow_nan=False) + "\n")
                write(candidate / "record.json", record)
                write(output / "status.json", {"status": "running", "candidate_status": record["status"], **event})
                if event["elapsed_s"] >= args.study_timeout_s:
                    raise TimeoutError("Geometry study runtime budget exceeded at " + event["stage"])

            try:
                baseline = run_candidate(candidate, record, domain, density, reconstruction, validator, validation_settings,
                                         reference, baseline_solid, baseline_mass, baseline, settings,
                                         historical["settings"]["relative_constraints"], args.geometry_only, progress)
            except Exception as error:
                record["status"] = "failed"
                record["diagnostics"].append(type(error).__name__ + ": " + str(error))
                if hasattr(error, "report"):
                    record["reconstruction"] = error.report
                    record["failure_evidence"] = save_geometry_evidence(candidate / "failure", "reconstruction", error.report,
                                                                          getattr(error, "shape", None), getattr(error, "mesh", None))
            record["artifacts"] = artifacts(candidate)
            write(candidate / "record.json", record)
            records.append(record)
            manifest["candidates"].append({"id": record["id"], "directory": candidate.relative_to(output).as_posix(),
                                           "record_sha256": digest(candidate / "record.json"), "threshold": threshold, "status": record["status"]})
            manifest.update(acceptance(records, complete=False), stage="candidate_finished")
            write(output / "manifest.json", manifest)
            write(output / "status.json", {"status": "running", "stage": "candidate_finished", "candidate": record["id"], "candidate_status": record["status"]})
            print(json.dumps({"candidate": record["id"], "threshold": threshold, "status": record["status"], "diagnostics": record["diagnostics"]}), flush=True)
        if time.perf_counter() - started >= args.study_timeout_s:
            raise TimeoutError("Geometry study runtime budget exceeded")
        manifest.update(acceptance(records, complete=True), status="complete", stage="finished")
    except Exception as error:
        manifest.update(status="failed", stage="failed", overall_acceptance=False, selected_id=None,
                        error=type(error).__name__ + ": " + str(error))
    manifest.update(runtime_s=time.perf_counter() - started, artifacts=artifacts(output))
    write(output / "manifest.json", manifest)
    write(output / "status.json", {key: manifest[key] for key in ("status", "stage", "accepted_count", "selected_id", "overall_acceptance", "runtime_s")}
          | ({"error": manifest["error"]} if "error" in manifest else {}))
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-step", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    parser.add_argument("--surface-deviation-mm", type=float, default=0.20)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--interpolation-method", choices=("pchip", "cubic"), default="pchip")
    parser.add_argument("--density-smoothing-sigma-mm", type=float, default=0.0)
    parser.add_argument("--manufacturing-opening-radius-mm", type=float, default=0.0)
    parser.add_argument("--manufacturing-opening-method", choices=("grayscale", "distance"), default="grayscale")
    parser.add_argument("--surface-constraint-mode", choices=("embedded", "cad_only", "envelope_only", "envelope_forbidden"), default="embedded")
    parser.add_argument("--free-forbidden-buffer-mm", type=float, default=0.0)
    parser.add_argument("--decimation-bounds-mode", choices=("none", "reference_aabb"), default="none")
    parser.add_argument("--preserve-fusion-mode", choices=("direct", "preunion"), default="direct")
    parser.add_argument("--boolean-fuzzy-mm", type=float, default=1e-7)
    parser.add_argument("--validation-boolean-fuzzy-mm", type=float, default=1e-7)
    parser.add_argument("--maximum-wall-samples", type=int, default=2_000_000)
    parser.add_argument("--face-budgets", nargs="+", type=int, default=[6000, 12000, 24000, 48000])
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--study-timeout-s", type=float, default=14400)
    parser.add_argument("--mesh-timeout-s", type=float, default=900)
    parser.add_argument("--solver-timeout-s", type=float, default=900)
    args = parser.parse_args(argv)
    if not np.isfinite(args.boolean_fuzzy_mm) or args.boolean_fuzzy_mm <= 0:
        parser.error("Boolean fuzzy tolerance must be finite and positive")
    if not np.isfinite(args.validation_boolean_fuzzy_mm) or args.validation_boolean_fuzzy_mm <= 0:
        parser.error("Validation boolean fuzzy tolerance must be finite and positive")
    if args.maximum_wall_samples <= 0:
        parser.error("Maximum wall samples must be a positive integer")
    if any(not np.isfinite(v) or v <= 0 for v in (args.study_timeout_s, args.mesh_timeout_s, args.solver_timeout_s,
                                                  args.surface_deviation_mm, args.subdivisions, *args.face_budgets)):
        parser.error("Use finite positive runtime, tessellation and face budgets")
    if any(not np.isfinite(v) or not 0 < v < 1 for v in args.thresholds):
        parser.error("Thresholds must be finite values in (0,1)")
    if not np.isfinite(args.density_smoothing_sigma_mm) or args.density_smoothing_sigma_mm < 0:
        parser.error("Density smoothing must be finite and nonnegative")
    if not np.isfinite(args.manufacturing_opening_radius_mm) or args.manufacturing_opening_radius_mm < 0:
        parser.error("Manufacturing opening radius must be finite and nonnegative")
    if not np.isfinite(args.free_forbidden_buffer_mm) or args.free_forbidden_buffer_mm < 0:
        parser.error("Free forbidden buffer must be finite and nonnegative")
    if args.free_forbidden_buffer_mm > 0 and (args.surface_constraint_mode != "envelope_forbidden" or args.manufacturing_opening_radius_mm <= 0):
        parser.error("A positive free forbidden buffer requires envelope_forbidden and a positive manufacturing opening radius")
    return args


def main(argv=None):
    return 0 if run_study(parse_args(argv))["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
