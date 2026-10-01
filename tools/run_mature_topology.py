"""Run uniform-start density optimization and automatic continuous CAD/FEA selection."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for attempt in range(21):
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 20:
                raise
            time.sleep(0.05)


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
            "log_sha256": hashlib.sha256(logfile.read_bytes()).hexdigest()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-step", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Hash-verified completed density study; skips density optimization")
    parser.add_argument("--density-python", "--gpu-python", dest="density_python", type=Path,
                        default=Path(sys.executable), help="Python interpreter for the chosen CPU or GPU backend")
    parser.add_argument("--density-workspace", type=Path, default=ROOT)
    parser.add_argument("--geometry-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--shape", nargs=3, type=int, default=[51, 48, 12])
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-runtime-s", type=float, default=1200)
    parser.add_argument("--change-tolerance", type=float, default=0.005)
    parser.add_argument("--density-backend", choices=("cuda_cudss", "cpu_superlu"), default="cpu_superlu")
    parser.add_argument("--density-finalization-timeout-s", type=float, default=600,
                        help="Hard timeout allowance beyond the density update budget for final state evaluation")
    parser.add_argument("--geometry-timeout-s", type=float, default=14400)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--interpolation-method", choices=("pchip", "cubic"), default="pchip")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    parser.add_argument("--face-budgets", nargs="+", type=int, default=[6000, 12000, 24000, 48000])
    parser.add_argument("--surface-deviation-mm", type=float, default=0.20)
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
    parser.add_argument("--mesh-timeout-s", type=float, default=900)
    parser.add_argument("--solver-timeout-s", type=float, default=900)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.boolean_fuzzy_mm) or args.boolean_fuzzy_mm <= 0:
        parser.error("Boolean fuzzy tolerance must be finite and positive")
    if not math.isfinite(args.validation_boolean_fuzzy_mm) or args.validation_boolean_fuzzy_mm <= 0:
        parser.error("Validation boolean fuzzy tolerance must be finite and positive")
    if args.maximum_wall_samples <= 0:
        parser.error("Maximum wall samples must be a positive integer")
    if not math.isfinite(args.manufacturing_opening_radius_mm) or args.manufacturing_opening_radius_mm < 0:
        parser.error("Manufacturing opening radius must be finite and nonnegative")
    if not math.isfinite(args.free_forbidden_buffer_mm) or args.free_forbidden_buffer_mm < 0:
        parser.error("Free forbidden buffer must be finite and nonnegative")
    if args.free_forbidden_buffer_mm > 0 and (args.surface_constraint_mode != "envelope_forbidden" or args.manufacturing_opening_radius_mm <= 0):
        parser.error("A positive free forbidden buffer requires envelope_forbidden and a positive manufacturing opening radius")
    budgets = (args.max_runtime_s, args.geometry_timeout_s, args.density_finalization_timeout_s,
               args.mesh_timeout_s, args.solver_timeout_s, args.surface_deviation_mm)
    if (any(not math.isfinite(v) or v <= 0 for v in budgets) or args.max_iterations <= 0
            or min(args.shape) <= 1 or args.subdivisions <= 0 or min(args.face_budgets) <= 0
            or not 0 < args.change_tolerance < 1 or any(not 0 < v < 1 for v in args.thresholds)
            or not math.isfinite(args.density_smoothing_sigma_mm) or args.density_smoothing_sigma_mm < 0):
        parser.error("Use finite positive budgets, grid dimensions greater than one and thresholds/tolerance in (0,1)")
    return args


def verify_geometry(directory):
    from tools.run_mature_geometry_study import acceptance, digest, read
    from deep_frame.topology_pipeline import _verify_cases, compare_to_baseline

    manifest = read(directory / "manifest.json")
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
        if digest(path) != candidate["record_sha256"]:
            raise ValueError("Candidate record hash mismatch")
        record = read(path)
        if (record["id"] in seen or record["id"] != candidate["id"] or record["status"] != candidate["status"]
                or record["threshold"] != candidate["threshold"]):
            raise ValueError("Duplicate or inconsistent candidate identity")
        seen.add(record["id"])
        for name, expected in record["artifacts"].items():
            artifact = (folder / name).resolve()
            if (not artifact.is_relative_to(folder) or not artifact.is_file()
                    or digest(artifact) != expected["sha256"] or artifact.stat().st_size != expected["size_bytes"]):
                raise ValueError("Candidate artifact mismatch: " + name)
        if record["status"] == "accepted":
            required = {"geometry.step", "geometry.stl", "final_mesh.json", "raw_fea_result.json"}
            if not required.issubset(record["artifacts"]) or read(folder / "raw_fea_result.json") != record.get("fea"):
                raise ValueError("Accepted candidate lacks consistent hashed CAD/FEA evidence")
            baseline_path = directory / "baseline/raw_fea_result.json"
            baseline_artifact = manifest["artifacts"].get("baseline/raw_fea_result.json")
            if (baseline_artifact is None or digest(baseline_path) != baseline_artifact["sha256"]
                    or baseline_path.stat().st_size != baseline_artifact["size_bytes"]):
                raise ValueError("Accepted candidate lacks a verified independent baseline")
            baseline = read(baseline_path)
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


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("A new empty output directory is required")
    output.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve() if args.source else output / "density"
    density_command = None if args.source else [str(args.density_python.resolve()), "-m", "tools.run_topology_study", "--directory", str(source),
                       "--shape", *map(str,args.shape), "--max-iterations", str(args.max_iterations),
                       "--max-runtime-s", str(args.max_runtime_s), "--change-tolerance", str(args.change_tolerance),
                       "--linear-solver", args.density_backend]
    geometry_command = [str(args.geometry_python.resolve()), str(ROOT/"tools/run_mature_geometry_study.py"),
                        "--source", str(source), "--output", str(output/"geometry"),
                        "--reference-step", str(output / "reference.step"), "--subdivisions", str(args.subdivisions),
                        "--interpolation-method", args.interpolation_method,
                        "--thresholds", *map(str, args.thresholds), "--face-budgets", *map(str, args.face_budgets),
                        "--surface-deviation-mm", str(args.surface_deviation_mm),
                        "--density-smoothing-sigma-mm", str(args.density_smoothing_sigma_mm),
                        "--manufacturing-opening-radius-mm", str(args.manufacturing_opening_radius_mm),
                        "--manufacturing-opening-method", args.manufacturing_opening_method,
                        "--surface-constraint-mode", args.surface_constraint_mode,
                        "--free-forbidden-buffer-mm", str(args.free_forbidden_buffer_mm),
                        "--decimation-bounds-mode", args.decimation_bounds_mode,
                        "--preserve-fusion-mode", args.preserve_fusion_mode,
                        "--boolean-fuzzy-mm", str(args.boolean_fuzzy_mm),
                        "--validation-boolean-fuzzy-mm", str(args.validation_boolean_fuzzy_mm),
                        "--maximum-wall-samples", str(args.maximum_wall_samples),
                        "--study-timeout-s", str(args.geometry_timeout_s), "--mesh-timeout-s", str(args.mesh_timeout_s),
                        "--solver-timeout-s", str(args.solver_timeout_s)]
    if args.geometry_only:
        geometry_command.append("--geometry-only")
    report = {"schema_version": "deep-frame-mature-end-to-end-v2", "status": "running", "stage": "preparing",
              "overall_acceptance": False, "accepted_count": 0, "selected_id": None,
              "initial_density": "saved density source; initialization is documented in source inputs" if args.source else "fresh uniform free-domain initialization; no restart or prescribed arm seed",
              "source": str(source), "source_mode": "saved" if args.source else "fresh_uniform",
              "density_backend": args.density_backend, "independent_fea": {"linear_solver": "SPOOLES", "threads": 1},
              "surface_constraint_mode": args.surface_constraint_mode,
              "free_forbidden_buffer_mm": args.free_forbidden_buffer_mm,
              "decimation_bounds_mode": args.decimation_bounds_mode,
              "preserve_fusion_mode": args.preserve_fusion_mode,
              "cad_boolean_settings": {"boolean_fuzzy_value_mm": args.boolean_fuzzy_mm,
                                       "validator_boolean_fuzzy_value_mm": args.validation_boolean_fuzzy_mm},
              "budgets": {"density_updates_s": args.max_runtime_s, "density_finalization_allowance_s": args.density_finalization_timeout_s,
                          "maximum_wall_samples": args.maximum_wall_samples,
                          "geometry_and_fea_process_s": args.geometry_timeout_s, "mesh_per_run_s": args.mesh_timeout_s,
                          "solver_per_case_s": args.solver_timeout_s},
              "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "commands": {"density": density_command, "geometry": geometry_command}, "stages": {}}
    save(output/"run.json", report)
    save(output/"status.json", {"status": "running", "stage": "preparing"})
    try:
        from tools.run_workstation_candidate_study import load_source

        for path in (args.geometry_python, args.reference_step):
            if not path.is_file():
                raise ValueError("Required file does not exist: " + str(path))
        shutil.copy2(args.reference_step, output / "reference.step")
        report["reference_step_sha256"] = hashlib.sha256((output / "reference.step").read_bytes()).hexdigest()
        if density_command is not None:
            if not args.density_python.is_file() or not (args.density_workspace / "tools/run_topology_study.py").is_file():
                raise ValueError("Density Python/workspace does not exist")
            report["stage"] = "density"
            save(output/"run.json", report)
            save(output/"status.json", {"status": "running", "stage": "density"})
            print(json.dumps({"stage": "density", "directory": str(source)}), flush=True)
            report["stages"]["density"] = execute(density_command, args.density_workspace.resolve(), output/"density.log", args.max_runtime_s+args.density_finalization_timeout_s)
            save(output/"run.json", report)
            if report["stages"]["density"]["returncode"] != 0:
                raise RuntimeError("Density stage failed; inspect density.log and persisted stage evidence")
        else:
            report["stages"]["density"] = {"status": "reused", "directory": str(source)}
        source_inputs, _, _, _ = load_source(source)
        report["source_density_backend"] = source_inputs.get("settings", {}).get("linear_solver", "cpu_superlu")
        report["density_manifest_sha256"] = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
        report["source_artifacts_verified"] = True
        report["stage"] = "geometry_and_fea"
        save(output/"run.json", report)
        save(output/"status.json", {"status": "running", "stage": "geometry_and_fea", "detail_status": "geometry/status.json"})
        print(json.dumps({"stage": "geometry_and_fea", "directory": str(output/"geometry")}), flush=True)
        report["stages"]["geometry"] = execute(geometry_command, ROOT, output/"geometry.log", args.geometry_timeout_s)
        save(output/"run.json", report)
        if report["stages"]["geometry"]["returncode"] != 0:
            raise RuntimeError("Geometry stage failed; inspect geometry.log and persisted candidate evidence")
        geometry = verify_geometry(output / "geometry")
        if (geometry["thresholds"] != args.thresholds or geometry["source_manifest_sha256"] != report["density_manifest_sha256"]
                or geometry["comparison_reference"]["sha256"] != report["reference_step_sha256"]):
            raise ValueError("Geometry evidence changed the requested thresholds, density source or reference STEP")
        report.update(status="complete", stage="finished", overall_acceptance=bool(geometry["overall_acceptance"]),
                      selected_id=geometry["selected_id"], accepted_count=geometry["accepted_count"],
                      geometry_manifest_sha256=hashlib.sha256((output/"geometry/manifest.json").read_bytes()).hexdigest())
    except Exception as error:
        report.update(status="failed", stage="failed", overall_acceptance=False, selected_id=None,
                      error=type(error).__name__+": "+str(error))
    save(output/"run.json", report)
    save(output/"status.json", {key: report[key] for key in ("status", "stage", "overall_acceptance", "accepted_count", "selected_id")}
         | ({"error": report["error"]} if "error" in report else {}))
    print(json.dumps(report), flush=True)
    return 0 if report["overall_acceptance"] else (2 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
