"""Bounded, observable density experiments; each invocation starts uniformly.

Run from the repository root with ``python -m tools.run_topology_study --help``.
This runner does not reconstruct or mechanically accept any candidate.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import label

from deep_frame.geometry import reference_parameters
from deep_frame.topology_config import TOPOLOGY_CONFIG
from deep_frame.topology_domain import build_design_domain
from deep_frame.topology_optimization import _settings, optimize_topology
from deep_frame.topology_pipeline import _file_digest, _provenance, _save


def study_parameters(shape):
    shape = np.asarray(shape, dtype=int)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError("Grid shape must contain three positive integers")
    grid = TOPOLOGY_CONFIG["grid"]
    extent = np.asarray(grid["shape"]) * np.asarray(grid["spacing_mm"])
    parameters = reference_parameters()
    parameters["topology"] = {"grid": {"shape": shape.tolist(), "spacing_mm": (extent / shape).tolist()}}
    return parameters


def run_study(directory, shape, max_iterations, change_tolerance, max_runtime_s, linear_solver="cpu_superlu"):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    status_path = directory / "status.json"
    _save(status_path, {"status": "building_domain", "pid": os.getpid()})
    try:
        parameters = study_parameters(shape)
        provenance = _provenance({"domain_builder": build_design_domain, "generator": optimize_topology}, {}, False)
        provenance["runner_sha256"] = _file_digest(__file__)
        provenance["thread_environment"] = {key: os.environ.get(key) for key in
                                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        _save(directory / "request.json", {"parameters": parameters, "provenance": provenance,
                                           "requested_settings": {"max_iterations": max_iterations,
                                                                  "change_tolerance": change_tolerance,
                                                                  "max_runtime_s": max_runtime_s, "linear_solver": linear_solver}})
        domain = build_design_domain(parameters)
        settings = _settings({**domain["optimizer_settings"], "max_iterations": max_iterations,
                              "change_tolerance": change_tolerance, "max_runtime_s": max_runtime_s, "linear_solver": linear_solver})
        masks = {name: domain[name] for name in ("allowed", "preserve", "forbidden")}
        np.savez_compressed(directory / "domain_masks.npz", **masks)
        inputs = {"parameters": parameters, "settings": settings,
                  "domain": {key: value for key, value in domain.items() if not isinstance(value, np.ndarray)},
                  "mask_sha256": {name: hashlib.sha256(mask.tobytes()).hexdigest() for name, mask in masks.items()},
                  "provenance": provenance,
                  "scope": "Density iteration and grid study only; no CAD/manufacturing/CalculiX acceptance.",
                  "restart_policy": "Fresh uniform initialization. No partial iteration resume.",
                  "runtime_limit_policy": "Checked between updates; final independent state evaluation can exceed the budget."}
        _save(directory / "inputs.json", inputs)
        _save(status_path, {"status": "optimizing", "pid": os.getpid(), "grid": domain["grid"],
                            "settings": settings, "elapsed_s": perf_counter() - started})
        print(json.dumps({"event": "start", "directory": str(directory), "grid": domain["grid"],
                          "domain_counts": {key: value for key, value in domain["metadata"].items() if key.endswith("_cells")}}), flush=True)
        with (directory / "iterations.jsonl").open("w", encoding="utf-8") as journal:
            def progress(entry):
                journal.write(json.dumps(entry, allow_nan=False) + "\n")
                journal.flush()
                _save(status_path, {"status": "optimizing", "pid": os.getpid(), **entry})
                print(json.dumps({"event": "iteration", **{key: entry[key] for key in
                                 ("iteration", "final_evaluation", "objective", "maximum_design_change", "elapsed_s")}}), flush=True)

            result = optimize_topology(domain, settings, progress_callback=progress)
        arrays = {**masks, "density": result["density"]}
        if "design_density" in result:
            arrays["design_density"] = result["design_density"]
        np.savez_compressed(directory / "fields.npz", **arrays)
        report = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        if result["status"] == "ok":
            connectivity = []
            for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
                occupied = (result["density"] >= threshold) & domain["allowed"]
                components, count = label(occupied)
                preserve_components = np.unique(components[domain["preserve"]])
                connectivity.append({"threshold": threshold, "occupied_cells": int(occupied.sum()),
                                     "face_connected_components": count,
                                     "components_touching_preserves": len(preserve_components[preserve_components > 0])})
            report["threshold_connectivity_before_reconstruction"] = connectivity
        _save(directory / "result.json", report)
        artifacts = {path.name: {"sha256": _file_digest(path), "size_bytes": path.stat().st_size}
                     for path in sorted(directory.iterdir()) if path.is_file() and path.name != "status.json"}
        _save(directory / "manifest.json", {"schema_version": "deep-frame-density-study-v1", "status": result["status"],
                                            "artifacts": artifacts, "elapsed_s": perf_counter() - started})
        _save(status_path, {"status": result["status"], "pid": os.getpid(), "summary": result["summary"],
                            "diagnostics": result["diagnostics"], "elapsed_s": perf_counter() - started})
        print(json.dumps({"event": "finished", "status": result["status"],
                          "iterations": result["summary"].get("iterations"),
                          "stop_reason": result["summary"].get("stop_reason"),
                          "objective_final": result["summary"].get("objective_final"),
                          "diagnostics": result["diagnostics"]}), flush=True)
        return result["status"]
    except Exception as error:
        _save(status_path, {"status": "failed", "pid": os.getpid(), "error": str(error),
                            "elapsed_s": perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="New output directory; refuses existing paths.")
    parser.add_argument("--shape", nargs=3, type=int, default=[34, 32, 8])
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--change-tolerance", type=float, default=0.005)
    parser.add_argument("--max-runtime-s", type=float, default=1800.0)
    parser.add_argument("--linear-solver", choices=("cpu_superlu", "cuda_cudss"), default="cpu_superlu")
    args = parser.parse_args()
    status = run_study(args.directory, args.shape, args.max_iterations, args.change_tolerance, args.max_runtime_s, args.linear_solver)
    raise SystemExit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
