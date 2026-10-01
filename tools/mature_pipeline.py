from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from build123d import Plane, export_step, import_step, section
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_OUT

from deep_frame.config import command_line, configure
from deep_frame.fea import evaluate
from deep_frame.frame import build_geometry
from deep_frame.topology_geometry import _merge, region_shape
from deep_frame.topology_pipeline import _file_digest, _plot_modules, _provenance, _read, _verify_cases, compare_to_baseline
from tools.workstation_study import load_source

SURFACE_CONFIG = {"subdivisions": 4, "interpolation_method": "pchip", "thresholds": [0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
                  "face_budgets": [6000, 12000, 24000, 48000], "surface_deviation_mm": 0.20, "density_smoothing_sigma_mm": 0.0,
                  "manufacturing_opening_radius_mm": 0.0, "manufacturing_opening_method": "grayscale",
                  "surface_constraint_mode": "embedded", "free_forbidden_buffer_mm": 0.0, "decimation_bounds_mode": "none",
                  "preserve_fusion_mode": "direct", "boolean_fuzzy_mm": 1e-7, "validation_boolean_fuzzy_mm": 1e-7,
                  "maximum_wall_samples": 2_000_000, "mesh_timeout_s": 900, "solver_timeout_s": 900, "geometry_only": False}
SURFACE_KINDS = {"subdivisions": "int", "interpolation_method": ("pchip", "cubic"), "thresholds": ["float"],
                 "face_budgets": ["int"], "surface_deviation_mm": "float", "density_smoothing_sigma_mm": "float",
                 "manufacturing_opening_radius_mm": "float", "manufacturing_opening_method": ("grayscale", "distance"),
                 "surface_constraint_mode": ("embedded", "cad_only", "envelope_only", "envelope_forbidden"),
                 "free_forbidden_buffer_mm": "float", "decimation_bounds_mode": ("none", "reference_aabb"),
                 "preserve_fusion_mode": ("direct", "preunion"), "boolean_fuzzy_mm": "float", "validation_boolean_fuzzy_mm": "float",
                 "maximum_wall_samples": "int", "mesh_timeout_s": "float", "solver_timeout_s": "float", "geometry_only": "flag"}
RUN_CONFIG = {"output": None, "reference_step": None, "source": None, "density_python": Path(sys.executable),
              "density_workspace": ROOT, "geometry_python": Path(sys.executable), "shape": [51, 48, 12], "max_iterations": 1000,
              "max_runtime_s": 1200, "change_tolerance": 0.005, "density_backend": "cpu_superlu",
              "density_finalization_timeout_s": 600, "geometry_timeout_s": 14400, **SURFACE_CONFIG}
RUN_KINDS = {"output": "path", "reference_step": "path", "source": "path", "density_python": "path",
             "density_workspace": "path", "geometry_python": "path", "shape": ["int"] * 3, "max_iterations": "int",
             "max_runtime_s": "float", "change_tolerance": "float", "density_backend": ("cuda_cudss", "cpu_superlu"),
             "density_finalization_timeout_s": "float", "geometry_timeout_s": "float", **SURFACE_KINDS}
GEOMETRY_CONFIG = {"source": None, "output": None, "reference_step": None, **SURFACE_CONFIG, "study_timeout_s": 14400}
GEOMETRY_KINDS = {"source": "path", "output": "path", "reference_step": "path", **SURFACE_KINDS, "study_timeout_s": "float"}
RENDER_CONFIG = {"step": None, "output": None,
                 "inputs": Path("C:/clones/Deep_Frame/exports/topology/workstation_20260930/density_study/grid8over3_iter150/inputs.json"),
                 "validation": None}
RENDER_KINDS = {"step": "path", "output": "path", "inputs": "path", "validation": "path"}

def execute(command, cwd, logfile, timeout):
    started = time.perf_counter()
    report = {"command": command, "cwd": str(cwd), "timeout_s": timeout, "log": str(logfile)}
    with logfile.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                                   start_new_session=sys.platform != "win32")
        try:
            report.update(returncode=process.wait(timeout=timeout), timed_out=False)
        except subprocess.TimeoutExpired as error:
            report.update(returncode=None, timed_out=True, error=str(error))
            try:
                if sys.platform == "win32":
                    cleanup = subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                                             stdout=log, stderr=subprocess.STDOUT, timeout=15,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                    report["process_tree_cleanup_returncode"] = cleanup.returncode
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                    report["process_tree_cleanup_returncode"] = 0
            except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                report["process_tree_cleanup_error"] = str(cleanup_error)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=15)
    return {**report, "runtime_s": time.perf_counter()-started,
            "log_sha256": _file_digest(logfile)}

def check_surface(config):
    if not math.isfinite(config["boolean_fuzzy_mm"]) or config["boolean_fuzzy_mm"] <= 0:
        raise ValueError("Boolean fuzzy tolerance must be finite and positive")
    if not math.isfinite(config["validation_boolean_fuzzy_mm"]) or config["validation_boolean_fuzzy_mm"] <= 0:
        raise ValueError("Validation boolean fuzzy tolerance must be finite and positive")
    if config["maximum_wall_samples"] <= 0:
        raise ValueError("Maximum wall samples must be a positive integer")
    if not math.isfinite(config["manufacturing_opening_radius_mm"]) or config["manufacturing_opening_radius_mm"] < 0:
        raise ValueError("Manufacturing opening radius must be finite and nonnegative")
    if not math.isfinite(config["free_forbidden_buffer_mm"]) or config["free_forbidden_buffer_mm"] < 0:
        raise ValueError("Free forbidden buffer must be finite and nonnegative")
    if config["free_forbidden_buffer_mm"] > 0 and (config["surface_constraint_mode"] != "envelope_forbidden" or config["manufacturing_opening_radius_mm"] <= 0):
        raise ValueError("A positive free forbidden buffer requires envelope_forbidden and a positive manufacturing opening radius")
    if any(not math.isfinite(v) or v <= 0 for v in (config["mesh_timeout_s"], config["solver_timeout_s"], config["surface_deviation_mm"],
                                                    config["subdivisions"], *config["face_budgets"])):
        raise ValueError("Use finite positive runtime, tessellation and face budgets")
    if any(not math.isfinite(v) or not 0 < v < 1 for v in config["thresholds"]):
        raise ValueError("Thresholds must be finite values in (0,1)")
    if not math.isfinite(config["density_smoothing_sigma_mm"]) or config["density_smoothing_sigma_mm"] < 0:
        raise ValueError("Density smoothing must be finite and nonnegative")

def run_config(overrides):
    config = configure(RUN_CONFIG, RUN_KINDS, overrides, ("output", "reference_step"))
    check_surface(config)
    budgets = (config["max_runtime_s"], config["geometry_timeout_s"], config["density_finalization_timeout_s"])
    if (any(not math.isfinite(v) or v <= 0 for v in budgets) or config["max_iterations"] <= 0
            or min(config["shape"]) <= 1 or not 0 < config["change_tolerance"] < 1):
        raise ValueError("Use finite positive budgets, grid dimensions greater than one and thresholds/tolerance in (0,1)")
    return config

def verify_geometry(directory):
    manifest = _read(directory / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Geometry study did not finish all requested candidates")
    if len(manifest["candidates"]) != len(manifest["thresholds"]):
        raise ValueError("Geometry study candidate count is incomplete")
    records = []
    seen = set()
    for candidate in manifest["candidates"]:
        folder = (directory / candidate["directory"]).resolve()
        if not folder.is_relative_to(directory.resolve()):
            raise ValueError("Candidate directory escapes geometry evidence")
        path = folder / "record.json"
        if _file_digest(path) != candidate["record_sha256"]:
            raise ValueError("Candidate record hash mismatch")
        record = _read(path)
        if (record["id"] in seen or record["id"] != candidate["id"] or record["status"] != candidate["status"]
                or record["threshold"] != candidate["threshold"]):
            raise ValueError("Duplicate or inconsistent candidate identity")
        seen.add(record["id"])
        for name, expected in record["artifacts"].items():
            artifact = (folder / name).resolve()
            if (not artifact.is_relative_to(folder) or not artifact.is_file()
                    or _file_digest(artifact) != expected["sha256"] or artifact.stat().st_size != expected["size_bytes"]):
                raise ValueError("Candidate artifact mismatch: " + name)
        if record["status"] == "accepted":
            required = {"geometry.step", "geometry.stl", "final_mesh.json", "raw_fea_result.json"}
            if not required.issubset(record["artifacts"]) or _read(folder / "raw_fea_result.json") != record.get("fea"):
                raise ValueError("Accepted candidate lacks consistent hashed CAD/FEA evidence")
            baseline_path = directory / "baseline/raw_fea_result.json"
            baseline_artifact = manifest["artifacts"].get("baseline/raw_fea_result.json")
            if (baseline_artifact is None or _file_digest(baseline_path) != baseline_artifact["sha256"]
                    or baseline_path.stat().st_size != baseline_artifact["size_bytes"]):
                raise ValueError("Accepted candidate lacks a verified independent baseline")
            baseline = _read(baseline_path)
            _verify_cases(baseline, manifest["load_cases"])
            _verify_cases(record["fea"], manifest["load_cases"])
            comparison = compare_to_baseline(record["fea"], baseline, manifest["relative_constraints"])
            if comparison != record["comparison"] or not comparison["passed"]:
                raise ValueError("Candidate mechanical comparison cannot be reproduced")
        records.append(record)
    expected = acceptance(records, complete=True)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Geometry acceptance summary disagrees with verified candidate evidence")
    return manifest

def run_main(overrides):
    config = run_config(overrides)
    output = config["output"].resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("A new empty output directory is required")
    output.mkdir(parents=True, exist_ok=True)
    source = config["source"].resolve() if config["source"] else output / "density"
    stages = {"density": None if config["source"] else {
                  "directory": str(source), "shape": config["shape"], "max_iterations": config["max_iterations"],
                  "max_runtime_s": config["max_runtime_s"], "change_tolerance": config["change_tolerance"],
                  "linear_solver": config["density_backend"]},
              "geometry": {"source": str(source), "output": str(output / "geometry"), "reference_step": str(output / "reference.step"),
                           **{key: config[key] for key in SURFACE_CONFIG}, "study_timeout_s": config["geometry_timeout_s"]}}
    configs = {}
    for name, values in stages.items():
        configs[name] = None
        if values is not None:
            path = output / (name + "_config.json")
            write(path, values)
            configs[name] = {"path": str(path), "sha256": _file_digest(path), "values": values}
    density_command = None if config["source"] else [str(config["density_python"].resolve()), "-m", "tools.topology_study", "run",
                                                     configs["density"]["path"]]
    geometry_command = [str(config["geometry_python"].resolve()), str(ROOT/"tools/mature_pipeline.py"), "geometry",
                        configs["geometry"]["path"]]
    report = {"schema_version": "deep-frame-mature-end-to-end-v3", "status": "running", "stage": "preparing",
              "overall_acceptance": False, "accepted_count": 0, "selected_id": None,
              "initial_density": "saved density source; initialization is documented in source inputs" if config["source"] else "fresh uniform free-domain initialization; no restart or prescribed arm seed",
              "source": str(source), "source_mode": "saved" if config["source"] else "fresh_uniform",
              "density_backend": config["density_backend"], "independent_fea": {"linear_solver": "SPOOLES", "threads": 1},
              "surface_constraint_mode": config["surface_constraint_mode"],
              "free_forbidden_buffer_mm": config["free_forbidden_buffer_mm"],
              "decimation_bounds_mode": config["decimation_bounds_mode"],
              "preserve_fusion_mode": config["preserve_fusion_mode"],
              "cad_boolean_settings": {"boolean_fuzzy_value_mm": config["boolean_fuzzy_mm"],
                                       "validator_boolean_fuzzy_value_mm": config["validation_boolean_fuzzy_mm"]},
              "budgets": {"density_updates_s": config["max_runtime_s"], "density_finalization_allowance_s": config["density_finalization_timeout_s"],
                          "maximum_wall_samples": config["maximum_wall_samples"],
                          "geometry_and_fea_process_s": config["geometry_timeout_s"], "mesh_per_run_s": config["mesh_timeout_s"],
                          "solver_per_case_s": config["solver_timeout_s"]},
              "orchestrator_sha256": _file_digest(__file__),
              "commands": {"density": density_command, "geometry": geometry_command}, "configs": configs, "stages": {}}
    write(output/"run.json", report)
    write(output/"status.json", {"status": "running", "stage": "preparing"})
    try:
        for path in (config["geometry_python"], config["reference_step"]):
            if not path.is_file():
                raise ValueError("Required file does not exist: " + str(path))
        shutil.copy2(config["reference_step"], output / "reference.step")
        report["reference_step_sha256"] = _file_digest(output / "reference.step")
        if density_command is not None:
            if not config["density_python"].is_file() or not (config["density_workspace"] / "tools/topology_study.py").is_file():
                raise ValueError("Density Python/workspace does not exist")
            report["stage"] = "density"
            write(output/"run.json", report)
            write(output/"status.json", {"status": "running", "stage": "density"})
            print(json.dumps({"stage": "density", "directory": str(source)}), flush=True)
            report["stages"]["density"] = execute(density_command, config["density_workspace"].resolve(), output/"density.log", config["max_runtime_s"]+config["density_finalization_timeout_s"])
            write(output/"run.json", report)
            if report["stages"]["density"]["returncode"] != 0:
                raise RuntimeError("Density stage failed; inspect density.log and persisted stage evidence")
        else:
            report["stages"]["density"] = {"status": "reused", "directory": str(source)}
        source_inputs, _, _, _ = load_source(source)
        report["source_density_backend"] = source_inputs.get("settings", {}).get("linear_solver", "cpu_superlu")
        report["density_manifest_sha256"] = _file_digest(source / "manifest.json")
        report["source_artifacts_verified"] = True
        report["stage"] = "geometry_and_fea"
        write(output/"run.json", report)
        write(output/"status.json", {"status": "running", "stage": "geometry_and_fea", "detail_status": "geometry/status.json"})
        print(json.dumps({"stage": "geometry_and_fea", "directory": str(output/"geometry")}), flush=True)
        report["stages"]["geometry"] = execute(geometry_command, ROOT, output/"geometry.log", config["geometry_timeout_s"])
        write(output/"run.json", report)
        if report["stages"]["geometry"]["returncode"] != 0:
            raise RuntimeError("Geometry stage failed; inspect geometry.log and persisted candidate evidence")
        geometry = verify_geometry(output / "geometry")
        if (geometry["thresholds"] != config["thresholds"] or geometry["source_manifest_sha256"] != report["density_manifest_sha256"]
                or geometry["comparison_reference"]["sha256"] != report["reference_step_sha256"]):
            raise ValueError("Geometry evidence changed the requested thresholds, density source or reference STEP")
        report.update(status="complete", stage="finished", overall_acceptance=bool(geometry["overall_acceptance"]),
                      selected_id=geometry["selected_id"], accepted_count=geometry["accepted_count"],
                      geometry_manifest_sha256=_file_digest(output/"geometry/manifest.json"))
    except Exception as error:
        report.update(status="failed", stage="failed", overall_acceptance=False, selected_id=None,
                      error=type(error).__name__+": "+str(error))
    write(output/"run.json", report)
    write(output/"status.json", {key: report[key] for key in ("status", "stage", "overall_acceptance", "accepted_count", "selected_id")}
          | ({"error": report["error"]} if "error" in report else {}))
    print(json.dumps(report), flush=True)
    return 0 if report["overall_acceptance"] else (2 if report["status"] == "complete" else 1)

def write(path, data):
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
    return {p.relative_to(directory).as_posix(): {"sha256": _file_digest(p), "size_bytes": p.stat().st_size}
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
            result["artifacts"][path.name] = {"sha256": _file_digest(path), "size_bytes": path.stat().st_size}
        except Exception as error:
            result["errors"].append(type(error).__name__ + ": " + str(error))
    write(directory / (stem + ".json"), result)
    return result

def export_final_mesh(solid, candidate, validator, settings):
    mesh = validator._mesh(solid, validator._settings(settings))
    mesh.export(candidate / "geometry.stl")
    mesh.export(candidate / "geometry.ply")
    report = {"source": "geometry.step imported final CAD", "source_step_sha256": _file_digest(candidate / "geometry.step"),
              "cad_geometry_modified": False, "tessellation": mesh.metadata["adaptive_tessellation"],
              "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
    write(candidate / "final_mesh.json", report)
    return report

def run_candidate(candidate, record, domain, density, reconstruction, validator, validation_settings,
                  reference, baseline_solid, baseline_mass, baseline, settings, constraints, geometry_only, progress):
    from deep_frame.topology_surface import reconstruct_surface
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

def run_study(config):
    from deep_frame.topology_surface import reconstruct_surface
    source, output = config["source"].resolve(), config["output"].resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("A new empty output directory is required; old evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    manifest = {"schema_version": "deep-frame-continuous-geometry-study-v2", "status": "running", "stage": "preparing",
                "density_source": str(source), "geometry_only": config["geometry_only"], "thresholds": config["thresholds"],
                "accepted_count": 0, "selected_id": None, "overall_acceptance": False, "candidates": [],
                "acceptance_scope": "Geometry and independent FEA screens; final manufacturing/render review remains required",
                "study_timeout_s": config["study_timeout_s"],
                "budget_policy": "Study budget checked at stage/progress boundaries; external orchestrator enforces process timeout"}
    write(output / "manifest.json", manifest)
    write(output / "status.json", {"status": "running", "stage": "preparing"})
    try:
        from deep_frame import topology_surface_validation as validator
        if not config["reference_step"].is_file():
            raise ValueError("A reference STEP file is required for surface maturity")
        load_source(source)
        source_manifest = _read(source / "manifest.json")
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
        historical = _read(ROOT / "docs/validation/topology_phase1/inputs.json")
        for name in ("material", "point_masses", "comparison_load_cases"):
            if domain[name] != historical["domain"][name]:
                raise ValueError("Physical comparison input changed: " + name)
        if {k: v for k, v in inputs["parameters"].items() if k != "topology"} != {k: v for k, v in historical["parameters"].items() if k != "topology"}:
            raise ValueError("Historical v0 geometry parameters changed")
        provenance = output / "provenance"
        for path in list((ROOT / "deep_frame").glob("*.py")) + list(ROOT.glob("requirements*.txt")) + [Path(__file__), ROOT / "tools/workstation_study.py"]:
            target = provenance / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        settings = _merge(domain["fea_settings"], historical["settings"]["fea_settings"])
        settings.update(mesh_size_mm=3.0, mesh_threads=1, threads=1, linear_solver="SPOOLES",
                        mesh_timeout_s=config["mesh_timeout_s"], solver_timeout_s=config["solver_timeout_s"])
        reconstruction = {"interpolation_subdivisions": config["subdivisions"], "interpolation_method": config["interpolation_method"],
                          "density_smoothing_sigma_mm": config["density_smoothing_sigma_mm"],
                          "manufacturing_opening_radius_mm": config["manufacturing_opening_radius_mm"],
                          "manufacturing_opening_method": config["manufacturing_opening_method"],
                          "surface_constraint_mode": config["surface_constraint_mode"],
                          "free_forbidden_buffer_mm": config["free_forbidden_buffer_mm"],
                          "decimation_bounds_mode": config["decimation_bounds_mode"],
                          "preserve_fusion_mode": config["preserve_fusion_mode"],
                          "boolean_fuzzy_value_mm": config["boolean_fuzzy_mm"],
                          "maximum_surface_deviation_mm": config["surface_deviation_mm"], "decimation_face_budgets": config["face_budgets"]}
        validation_settings = validator._settings({"maximum_wall_samples": config["maximum_wall_samples"],
                                                    "boolean_fuzzy_mm": config["validation_boolean_fuzzy_mm"]})
        shutil.copy2(config["reference_step"], output / "reference.step")
        reference = import_step(output / "reference.step")
        if not reference.is_valid or len(reference.solids()) != 1 or reference.volume <= 0:
            raise ValueError("Reference STEP must be a valid single positive-volume solid")
        manifest.update(source_manifest_sha256=_file_digest(snapshot / "manifest.json"), density_status=density_result["status"],
                        reconstruction_settings=reconstruction, validation_settings=validation_settings, fea_settings=settings,
                        comparison_reference={"source": str(config["reference_step"].resolve()), "sha256": _file_digest(output / "reference.step"), "volume_mm3": reference.volume},
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
        for index, threshold in enumerate(config["thresholds"]):
            if time.perf_counter() - started >= config["study_timeout_s"]:
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
                if event["elapsed_s"] >= config["study_timeout_s"]:
                    raise TimeoutError("Geometry study runtime budget exceeded at " + event["stage"])
            try:
                baseline = run_candidate(candidate, record, domain, density, reconstruction, validator, validation_settings,
                                         reference, baseline_solid, baseline_mass, baseline, settings,
                                         historical["settings"]["relative_constraints"], config["geometry_only"], progress)
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
                                           "record_sha256": _file_digest(candidate / "record.json"), "threshold": threshold, "status": record["status"]})
            manifest.update(acceptance(records, complete=False), stage="candidate_finished")
            write(output / "manifest.json", manifest)
            write(output / "status.json", {"status": "running", "stage": "candidate_finished", "candidate": record["id"], "candidate_status": record["status"]})
            print(json.dumps({"candidate": record["id"], "threshold": threshold, "status": record["status"], "diagnostics": record["diagnostics"]}), flush=True)
        if time.perf_counter() - started >= config["study_timeout_s"]:
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

def geometry_config(overrides):
    config = configure(GEOMETRY_CONFIG, GEOMETRY_KINDS, overrides, ("source", "output", "reference_step"))
    check_surface(config)
    if not math.isfinite(config["study_timeout_s"]) or config["study_timeout_s"] <= 0:
        raise ValueError("Use finite positive runtime, tessellation and face budgets")
    return config

def geometry_main(overrides):
    return 0 if run_study(geometry_config(overrides))["status"] == "complete" else 1

def face_colors(normals, elevation, azimuth):
    elev, azim = np.deg2rad([elevation, azimuth])
    view = np.asarray([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    light = view + np.asarray([-0.4, -0.3, 0.5 if elevation >= 0 else -0.5])
    light /= np.linalg.norm(light)
    fill = np.asarray([-0.5, 0.7, 0.25 if elevation >= 0 else -0.25])
    fill /= np.linalg.norm(fill)
    halfway = light + view
    halfway /= np.linalg.norm(halfway)
    intensity = 0.27 + 0.62 * np.maximum(normals @ light, 0) + 0.14 * np.maximum(normals @ fill, 0)
    highlight = 0.20 * np.maximum(normals @ halfway, 0) ** 24
    base = np.asarray([0.24, 0.62, 0.76])
    return np.clip(intensity[:, None] * base + highlight[:, None], 0, 1)

def draw_view(axis, mesh, lower, upper, elevation, azimuth, title):
    _, polygons = _plot_modules()
    axis.add_collection3d(polygons(mesh.triangles, facecolors=face_colors(mesh.face_normals, elevation, azimuth),
                                         edgecolors="none", linewidths=0, antialiased=False, zsort="average"))
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
             xlabel="x / mm", ylabel="y / mm", zlabel="z / mm", title=title)
    axis.set_box_aspect(upper - lower, zoom=0.9)
    axis.set_proj_type("ortho")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xticks(np.arange(-60, 61, 20))
    axis.set_yticks(np.arange(-60, 61, 20))
    axis.set_zticks([0, 8, 16, 24, 32])
    axis.tick_params(labelsize=8)
    axis.grid(True, alpha=0.15)
    for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
        dimension.pane.set_facecolor((0.96, 0.97, 0.98, 1))
        dimension._axinfo["grid"]["color"] = (0.4, 0.45, 0.5, 0.13)
    if abs(elevation) == 90:
        axis.set_zticks([])
        axis.set_zlabel("")

def section_paths(shape, plane, coordinates):
    sliced = section(shape, section_by=plane)
    paths = [np.asarray([tuple(point) for point in edge.positions(deflection=0.01)])[:, coordinates]
             for edge in sliced.edges()]
    return sliced, paths

def render_main(overrides):
    plt, _ = _plot_modules()
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from deep_frame.topology_surface_validation import _mesh, _settings
    config = configure(RENDER_CONFIG, RENDER_KINDS, overrides, ("step", "output"))
    config["output"].mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    record = {"status": "running", "step": str(config["step"].resolve()), "step_sha256": _file_digest(config["step"]),
              "inputs_sha256": _file_digest(config["inputs"]), "renderer_sha256": _file_digest(__file__),
              "rendering": "Fresh STEP import, whole-CAD adaptive tessellation, flat per-triangle normal shading, orthographic projection and metric axes; no geometry smoothing, decimation, hole filling, vertex-normal interpolation or remodeling",
              "section_method": "Exact CAD intersection with each named plane; section faces filled from their tessellation and CAD edges sampled to 0.01 mm deflection", "images": {}, "sections": []}
    def journal(stage):
        record.update(stage=stage, elapsed_s=perf_counter()-started)
        (config["output"] / "render_manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    journal("STEP import")
    solid = import_step(config["step"])
    journal("CAD tessellation")
    mesh = _mesh(solid, _settings({}))
    outward = []
    for body in solid.solids():
        classifier = BRepClass3d_SolidClassifier(body.wrapped)
        classifier.PerformInfinitePoint(1e-7)
        outward.append(classifier.State() == TopAbs_OUT)
    record["topology"] = {"solid_count": len(solid.solids()), "cad_valid": bool(solid.is_valid),
                          "outward_oriented": bool(outward) and all(outward), "mesh_watertight": bool(mesh.is_watertight),
                          "mesh_body_count": int(mesh.body_count), "cad_volume_mm3": float(solid.volume),
                          "mesh_signed_volume_mm3": float(mesh.volume), "triangle_count": len(mesh.faces),
                          "bounds_mm": mesh.bounds.tolist(), "tessellation": mesh.metadata["adaptive_tessellation"]}
    valid = len(solid.solids()) == mesh.body_count == 1 and solid.is_valid and mesh.is_watertight and all(outward) and mesh.volume > 0
    label = "DIAGNOSE – Geometrie- und Mechanikfreigabe offen"
    if not valid:
        label = "DIAGNOSE – Topologieprüfung fehlgeschlagen"
    if config["validation"]:
        proof = _read(config["validation"])
        expected = proof.get("sha256") or proof.get("provenance", {}).get("candidate_sha256")
        if expected != record["step_sha256"]:
            raise ValueError("Validation record must identify the rendered STEP by matching SHA256")
        validation = proof.get("validation", proof)
        record["validation"] = {"path": str(config["validation"]), "sha256": _file_digest(config["validation"]),
                                "passed": validation.get("passed", False), "violations": validation.get("violations", [])}
        if validation.get("passed", False) and valid:
            label = "Geometrie-Gates bestanden – mechanische Freigabe separat prüfen"
        else:
            label = "DIAGNOSE – Geometrie-Gates nicht bestanden"
    record["display_status"] = label
    domain = _read(config["inputs"])["domain"]
    lower = np.asarray(domain["grid"]["origin_mm"], dtype=float)
    upper = lower + np.asarray(domain["grid"]["shape"]) * np.asarray(domain["grid"]["spacing_mm"])
    lower, upper = np.minimum(lower, mesh.bounds[0]), np.maximum(upper, mesh.bounds[1])
    footer = "Reimportierter STEP | SHA256 " + record["step_sha256"][:16] + "… | CAD-Dreiecke: " + str(len(mesh.faces))
    def save(figure, name):
        path = config["output"] / (name + ".png")
        figure.savefig(path, dpi=200, facecolor="white")
        plt.close(figure)
        record["images"][name] = {"path": path.name, "sha256": _file_digest(path)}
        journal("saved " + name)
    views = [(28, -48, "Isometrie", "isometric"), (90, -90, "Draufsicht (+z)", "top"), (-90, -90, "Unterseite (−z)", "bottom")]
    for elev, azim, title, name in views:
        figure = plt.figure(figsize=(9, 8.2), layout="constrained")
        axis = figure.add_subplot(projection="3d")
        draw_view(axis, mesh, lower, upper, elev, azim, title)
        figure.suptitle(label, fontsize=13, color="#9b392b" if label.startswith("DIAGNOSE") else "#275e3c")
        figure.text(0.5, 0.01, footer, ha="center", fontsize=8)
        save(figure, name)
    journal("exact CAD strap sections")
    strap_voids = [region for region in domain["regions"] if region["role"] == "forbidden" and region["name"].startswith("strap_access_")]
    strap_preserves = [region for region in domain["regions"] if region["role"] == "preserve" and region["name"].startswith("strap_contact_")]
    centers = np.asarray([(np.asarray(region["min_mm"]) + region["max_mm"]) / 2 for region in strap_voids])
    zmid = float(np.mean([(region["min_mm"][2] + region["max_mm"][2]) / 2 for region in strap_preserves]))
    specs = [(f"Quer: y = {y:g} mm", Plane(origin=(0, y, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0)), [0, 2], [-24, 24], [20, 32]) for y in np.unique(centers[:, 1])]
    specs += [(f"Längs: x = {x:g} mm", Plane(origin=(x, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)), [1, 2], [-25, 25], [20, 32]) for x in np.unique(centers[:, 0])]
    specs += [(f"Horizontal: z = {zmid:g} mm", Plane(origin=(0, 0, zmid)), [0, 1], [-23, 23], [-25, 25])]
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    for axis, (title, plane, coordinates, xlim, ylim) in zip(axes.flat, specs):
        journal("section " + title)
        sliced, paths = section_paths(solid, plane, coordinates)
        vertices, triangles = sliced.tessellate(0.01, 0.08)
        if triangles:
            array = np.asarray([tuple(point) for point in vertices])
            axis.add_collection(PolyCollection(array[np.asarray(triangles)][:, :, coordinates], facecolors="#4b9bae", edgecolors="none", antialiased=False))
        axis.add_collection(LineCollection(paths, colors="#194653", linewidths=0.65))
        for region in strap_preserves:
            prescribed = region_shape(region).cut(*[region_shape(cut) for cut in strap_voids])
            _, outlines = section_paths(prescribed, plane, coordinates)
            axis.add_collection(LineCollection(outlines, colors="#46904b", linewidths=1.25, linestyles="--"))
        for region in strap_voids:
            _, outlines = section_paths(region_shape(region), plane, coordinates)
            axis.add_collection(LineCollection(outlines, colors="#c6553c", linewidths=1.2, linestyles=":"))
        axis.set(xlim=xlim, ylim=ylim, title=title, xlabel="xyz"[coordinates[0]] + " / mm", ylabel="xyz"[coordinates[1]] + " / mm")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        record["sections"].append({"title": title, "plane_origin_mm": list(plane.origin), "plane_normal": list(plane.z_dir), "cad_area_mm2": float(sliced.area), "cad_edge_count": len(paths), "view_limits_mm": [xlim, ylim]})
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[Line2D([0], [0], color="#4b9bae", linewidth=8, label="Tatsächliches STEP-Material"),
                                 Line2D([0], [0], color="#46904b", linestyle="--", label="Pflicht-Preserve abzüglich Gurtzugang"),
                                 Line2D([0], [0], color="#c6553c", linestyle=":", label="Freizuhaltender Gurtzugang")], loc="center", fontsize=11)
    figure.suptitle(label + "\nExakte CAD-Schnitte an allen vier Gurtdurchlässen", fontsize=14)
    figure.text(0.5, 0.006, footer, ha="center", fontsize=8)
    save(figure, "strap_sections")
    record["status"] = "complete"
    journal("complete")

def main(argv=None):
    return command_line({"run": run_main, "geometry": geometry_main, "render": render_main}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
