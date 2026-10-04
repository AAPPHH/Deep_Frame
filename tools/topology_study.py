import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import label

from deep_frame.config import TOPOLOGY_CONFIG, command_line, configure
from deep_frame.frame import reference_parameters
from deep_frame.topology_geometry import build_design_domain
from deep_frame.topology_neural import NEURAL_SETTINGS, neural_settings, optimize_neural
from deep_frame.topology_optimization import _settings, optimize_topology
from deep_frame.topology_pipeline import _file_digest, _plot_modules, _provenance, _read, _save, log_run

OPTIMIZER_KINDS = {"filter_radius_mm": "float", "projection": ("single", "robust"), "beta_schedule": ["float"],
                   "beta_interval": "int", "move_limit_late": "float", "volume_target_relaxation": "float",
                   "gpu_solver_residency": ("resident", "transient")}
RUN_CONFIG = {"directory": None, "shape": [34, 32, 8], "max_iterations": 300, "change_tolerance": 0.005,
              "max_runtime_s": 1800.0, "linear_solver": "cpu_superlu", "run_log": None, **dict.fromkeys(OPTIMIZER_KINDS)}
RUN_KINDS = {"directory": "text", "shape": ["int"] * 3, "max_iterations": "int", "change_tolerance": "float",
             "max_runtime_s": "float", "linear_solver": ("cpu_superlu", "cuda_cudss"), "run_log": "path", **OPTIMIZER_KINDS}
SUMMARIZE_CONFIG = {"root": "exports/topology/workstation_20260930/density_study", "runs": None,
                    "output": "docs/validation/workstation_density_study.json"}
PLOT_CONFIG = {"root": "exports/topology/workstation_20260930/density_study", "runs": None,
               "output": "docs/validation/workstation_density_convergence.png"}
STUDIES_KINDS = {"root": "text", "runs": ["text"], "output": "text"}
BENCHMARK_CONFIG = {"source": None, "output": None, "updates": 3}
BENCHMARK_KINDS = {"source": "text", "output": "text", "updates": "int"}
GPU_PLOT_CONFIG = {"evidence": "docs/validation/workstation_gpu", "output": "docs/validation/workstation_gpu_convergence.png"}
GPU_PLOT_KINDS = {"evidence": "text", "output": "text"}
FLOOR_CONFIG = {"run": None, "density": None, "output": "docs/validation/bottom_domain/before", "before": None}
FLOOR_KINDS = {"run": "text", "density": "text", "output": "text", "before": "text"}
NEURAL_KINDS = {"max_iterations": "int", "minimum_iterations": "int", "change_tolerance": "float", "learning_rate": "float", "frequencies": "int",
                "max_frequency_per_mm": "float", "hidden": ["int"], "seed": "int", "mirror_axis": "int", "sharpness_final": "float",
                "sharpness_iterations": "int", "max_runtime_s": "float", "max_width_penalty": "float", "max_width_window_mm": "float", "max_local_fraction": "float", "gpu_solver_residency": ("resident", "transient")}
NEURAL_CONFIG = {"directory": None, "shape": [68, 64, 16], "output_shape": [102, 96, 24], "volume_fractions": [0.12, 0.18, 0.25],
                 "linear_solver": "cuda_cudss", "run_log": None, **{key: NEURAL_SETTINGS[key] for key in NEURAL_KINDS}}
NEURAL_CONFIG_KINDS = {"directory": "text", "shape": ["int"] * 3, "output_shape": ["int"] * 3, "volume_fractions": ["float"],
                       "linear_solver": ("cpu_superlu", "cuda_cudss"), "run_log": "path", **NEURAL_KINDS}

def study_parameters(shape):
    shape = np.asarray(shape, dtype=int)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError("Grid shape must contain three positive integers")
    grid = TOPOLOGY_CONFIG["grid"]
    extent = np.asarray(grid["shape"]) * np.asarray(grid["spacing_mm"])
    parameters = reference_parameters()
    parameters["topology"] = {"grid": {"shape": shape.tolist(), "spacing_mm": (extent / shape).tolist()}}
    return parameters

def density_entry(directory, shape, max_iterations, linear_solver, status, started, result=None, optimizer=None):
    cupy = sys.modules.get("cupy")
    return {"run_dir": str(directory), "kind": "density", "candidate": None, "source_sha256": None,
            "parameters": {"shape": list(shape), "max_iterations": max_iterations, "linear_solver": linear_solver, **(optimizer or {})},
            "success": status == "ok", "status": status, "failure_stage": None if status == "ok" else "optimization",
            "runtime_s": perf_counter() - started, "timings_s": None, "iterations": (result or {}).get("summary", {}).get("iterations"),
            "gpu_pool_mb": cupy.get_default_memory_pool().total_bytes() / 2**20 if cupy else None}

def run_study(directory, shape, max_iterations, change_tolerance, max_runtime_s, linear_solver="cpu_superlu", optimizer=None, run_log=None, generator=optimize_topology, output_shape=None):
    optimizer = optimizer or {}
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    status_path = directory / "status.json"
    _save(status_path, {"status": "building_domain", "pid": os.getpid()})
    try:
        parameters = study_parameters(shape)
        provenance = _provenance({"domain_builder": build_design_domain, "generator": generator}, {}, False)
        provenance["runner_sha256"] = _file_digest(__file__)
        provenance["thread_environment"] = {key: os.environ.get(key) for key in
                                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        _save(directory / "request.json", {"parameters": parameters, "provenance": provenance,
                                           "requested_settings": {"max_iterations": max_iterations,
                                                                  "change_tolerance": change_tolerance,
                                                                  "max_runtime_s": max_runtime_s, "linear_solver": linear_solver, **optimizer}})
        domain = build_design_domain(parameters)
        requested = {"max_iterations": max_iterations, "change_tolerance": change_tolerance, "max_runtime_s": max_runtime_s, "linear_solver": linear_solver, **optimizer}
        if generator is optimize_topology:
            settings = _settings({**domain["optimizer_settings"], **requested})
        else:
            settings = neural_settings({**{key: domain["optimizer_settings"][key] for key in ("interface_node_policy", "case_weights", "penalization", "min_stiffness_ratio")}, **requested})
        output = domain if output_shape is None else build_design_domain(study_parameters(output_shape))
        masks = {name: output[name] for name in ("allowed", "preserve", "forbidden")}
        np.savez_compressed(directory / "domain_masks.npz", **masks)
        inputs = {"parameters": parameters if output_shape is None else study_parameters(output_shape), "settings": settings,
                  "domain": {key: value for key, value in output.items() if not isinstance(value, np.ndarray)},
                  "optimization_grid": domain["grid"],
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
            result = generator(domain, settings, progress_callback=progress, **({} if output_shape is None else {"output_domain": output}))
        arrays = {**masks, **{key: value for key, value in result.items() if isinstance(value, np.ndarray)}}
        np.savez_compressed(directory / "fields.npz", **arrays)
        report = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        if result["status"] == "ok":
            connectivity = []
            for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
                occupied = (result["density"] >= threshold) & output["allowed"]
                components, count = label(occupied)
                preserve_components = np.unique(components[output["preserve"]])
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
        log_run(run_log, density_entry(directory, shape, max_iterations, linear_solver, result["status"], started, result, optimizer))
        print(json.dumps({"event": "finished", "status": result["status"],
                          "iterations": result["summary"].get("iterations"),
                          "stop_reason": result["summary"].get("stop_reason"),
                          "objective_final": result["summary"].get("objective_final"),
                          "diagnostics": result["diagnostics"]}), flush=True)
        return result["status"]
    except Exception as error:
        _save(status_path, {"status": "failed", "pid": os.getpid(), "error": str(error),
                            "elapsed_s": perf_counter() - started})
        log_run(run_log, density_entry(directory, shape, max_iterations, linear_solver, "failed", started, None, optimizer))
        raise

def run_main(overrides):
    config = configure(RUN_CONFIG, RUN_KINDS, overrides, ("directory",))
    status = run_study(config["directory"], config["shape"], config["max_iterations"], config["change_tolerance"],
                       config["max_runtime_s"], config["linear_solver"],
                       {key: config[key] for key in OPTIMIZER_KINDS if config[key] is not None}, config["run_log"])
    return 0 if status == "ok" else 1

def neural_main(overrides):
    config = configure(NEURAL_CONFIG, NEURAL_CONFIG_KINDS, overrides, ("directory",))
    optimizer = {key: config[key] for key in NEURAL_KINDS if key not in ("max_iterations", "change_tolerance", "max_runtime_s")}
    statuses = [run_study(Path(config["directory"]) / f"f{round(100 * fraction):02d}", config["shape"], config["max_iterations"], config["change_tolerance"],
                          config["max_runtime_s"], config["linear_solver"], {**optimizer, "volume_fraction": fraction}, config["run_log"], optimize_neural, config["output_shape"])
                for fraction in config["volume_fractions"]]
    return 0 if all(status == "ok" for status in statuses) else 1

def verify_artifacts(directory, artifacts, required):
    directory = Path(directory).resolve()
    if not set(required).issubset(artifacts):
        raise ValueError("Required evidence artifacts missing from manifest")
    for name, evidence in artifacts.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing or invalid artifact path: {name}")
        if path.stat().st_size != evidence["size_bytes"] or _file_digest(path) != evidence["sha256"]:
            raise ValueError(f"Corrupt evidence artifact: {path}")

def verify_frozen_reference(directory):
    acceptance = _read(Path(directory) / "acceptance.json")
    if acceptance["status"] != "ok":
        raise ValueError("Frozen reference has no successful acceptance")
    verify_artifacts(directory, acceptance["versioned_artifacts"],
                     ("optimization.json", "fields.npz", "run_manifest.json"))

def verify_comparable_settings(reference, studied):
    reference, studied = _settings(reference), _settings(studied)
    permitted = {"max_iterations", "minimum_iterations", "change_tolerance", "max_runtime_s"}
    if {key: value for key, value in reference.items() if key not in permitted} != {
            key: value for key, value in studied.items() if key not in permitted}:
        raise ValueError("Optimization physics/settings changed beyond iteration stopping controls")
    if studied["minimum_iterations"] != min(reference["minimum_iterations"], studied["max_iterations"]):
        raise ValueError("Minimum iteration count changed beyond short benchmark adjustment")

def field_comparison(first, second):
    common_shape = np.lcm(first["density"].shape, second["density"].shape)
    if int(np.prod(common_shape)) > 2_000_000:
        raise ValueError("Common comparison subdivision exceeds the 2 million cell limit")
    def expand(array):
        for axis, count in enumerate(common_shape):
            array = np.repeat(array, int(count // array.shape[axis]), axis=axis)
        return array
    delta = expand(second["density"]) - expand(first["density"])
    first_allowed, second_allowed = expand(first["allowed"]), expand(second["allowed"])
    common_allowed = first_allowed & second_allowed
    common_free = common_allowed & ~expand(first["preserve"]) & ~expand(second["preserve"])
    return {"integration": "Exact piecewise-constant voxel overlap; equal physical envelope and origin.",
            "common_subdivision_shape": common_shape.tolist(),
            "rms_density_full_box": float(np.sqrt(np.mean(delta ** 2))),
            "rms_density_common_allowed": float(np.sqrt(np.mean(delta[common_allowed] ** 2))),
            "rms_density_common_free": float(np.sqrt(np.mean(delta[common_free] ** 2))),
            "maximum_density_difference": float(np.max(np.abs(delta))),
            "allowed_mask_disagreement_fraction_of_box": float(np.mean(first_allowed != second_allowed))}

def summarize(root, names, output):
    reference_dir = Path("docs/validation/topology_phase1")
    verify_frozen_reference(reference_dir)
    reference = _read(reference_dir / "optimization.json")
    reference_manifest = _read(reference_dir / "run_manifest.json")
    reference_fields = dict(np.load(reference_dir / "fields.npz", allow_pickle=False))
    reference_summary = reference["summary"]
    normalization = reference_summary["normalization_compliances_n_mm"]
    weights = reference_summary["normalized_case_weights"]
    if set(normalization) != set(weights):
        raise ValueError("Reference normalization and case weight keys disagree")
    report = {"schema_version": "deep-frame-workstation-density-comparison-v1",
              "analysis_source_sha256": _file_digest(__file__),
              "frozen_reference": {"run_id": "2cd7bcbc15b67fc7", "iterations": reference_summary["iterations"],
                                   "objective_final": reference_summary["objective_final"],
                                   "converged": reference_summary["converged"],
                                   "artifacts": {name: _file_digest(reference_dir / name) for name in
                                                 ("optimization.json", "fields.npz", "run_manifest.json")}},
              "comparability": "Each own-normalized objective has its own initial compliance. Common-reference objectives use the frozen 4 mm initial compliances and weights. Grid refinement changes conservative masks, selected nodes and the absolute volume at a fixed allowed-volume fraction.",
              "studies": []}
    reference_grid = reference_manifest["domain"]["grid"]
    completed_fields = {}
    for name in names:
        directory = Path(root) / name
        manifest = _read(directory / "manifest.json")
        if manifest["status"] != "ok":
            raise ValueError(f"Study {name} has no successful manifest")
        verify_artifacts(directory, manifest["artifacts"],
                         ("inputs.json", "result.json", "fields.npz", "domain_masks.npz", "iterations.jsonl"))
        result = _read(directory / "result.json")
        inputs = _read(directory / "inputs.json")
        if result["status"] != "ok":
            raise ValueError(f"Study {name} did not produce an acceptable numerical result")
        summary, history = result["summary"], result["history"]
        verify_comparable_settings(reference_summary["settings"], summary["settings"])
        if summary["settings"] != inputs["settings"]:
            raise ValueError("Result settings disagree with recorded input settings")
        if any(set(entry["compliances_n_mm"]) != set(normalization) for entry in history):
            raise ValueError("Study compliance case keys disagree with frozen reference")
        grid = inputs["domain"]["grid"]
        for key in ("regions", "load_cases", "comparison_load_cases", "material", "point_masses", "manufacturing"):
            if inputs["domain"][key] != reference_manifest["domain"][key]:
                raise ValueError(f"Study {name} changed physical {key}")
        for key in ("origin_mm", "axis_order", "order"):
            if grid[key] != reference_grid[key]:
                raise ValueError(f"Incompatible grid {key}")
        if not np.allclose(np.asarray(grid["shape"]) * grid["spacing_mm"],
                           np.asarray(reference_grid["shape"]) * reference_grid["spacing_mm"], atol=1e-10, rtol=0):
            raise ValueError("Physical envelope changed")
        fields = dict(np.load(directory / "fields.npz", allow_pickle=False))
        completed_fields[name] = {"fields": fields, "iterations": summary["iterations"], "grid": grid}
        raw = history[-1]["compliances_n_mm"]
        common_objective = sum(weights[case] * value / normalization[case] for case, value in raw.items())
        snapshots = {}
        for updates in (3, 45, 150):
            entry = next((item for item in history if item["iteration"] == updates + 1), None)
            if entry:
                snapshots[str(updates)] = {"objective_after_updates": entry["objective"],
                                          "main_raw_compliances_n_mm": {case: entry["compliances_n_mm"][case] for case in
                                                                        ("arm_tip", "battery_impact", "camera_side")},
                                          "common_reference_objective": sum(weights[case] * value / normalization[case]
                                                                            for case, value in entry["compliances_n_mm"].items())}
        report["studies"].append({"name": name, "directory": str(directory).replace("\\", "/"),
                                  "manifest_sha256": _file_digest(directory / "manifest.json"),
                                  "grid": grid, "all_physical_inputs_match_frozen_reference": True,
                                  "all_optimizer_settings_match_except_stopping_controls": True,
                                  "stopping_controls": {key: summary["settings"][key] for key in
                                                        ("max_iterations", "minimum_iterations", "change_tolerance", "max_runtime_s")},
                                  "summary": {key: summary[key] for key in
                                              ("iterations", "converged", "stop_reason", "objective_final", "elapsed_s",
                                               "volume_fraction", "density_frame_mass_g", "gray_fraction_free")},
                                  "domain_counts": {key: value for key, value in inputs["domain"]["metadata"].items() if key.endswith("_cells")},
                                  "domain_volumes_mm3": {key: value for key, value in inputs["domain"]["metadata"].items() if key.endswith("_volume_mm3")},
                                  "maximum_design_change_last_update": history[-2]["maximum_design_change"],
                                  "maximum_residual_all_evaluations": max(entry["maximum_relative_residual"] for entry in history),
                                  "main_final_raw_compliances_n_mm": {key: raw[key] for key in ("arm_tip", "battery_impact", "camera_side")},
                                  "objective_normalized_to_frozen_reference": common_objective,
                                  "snapshots_after_updates": snapshots,
                                  "relative_objective_change_last_25_updates": abs(history[-1]["objective"] / history[max(0, len(history) - 26)]["objective"] - 1),
                                  "threshold_connectivity_before_reconstruction": result["threshold_connectivity_before_reconstruction"],
                                  "density_vs_frozen_reference": field_comparison(reference_fields, fields)})
    report["pairwise_density_comparisons"] = []
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            first, second = completed_fields[first_name], completed_fields[second_name]
            report["pairwise_density_comparisons"].append({"first": first_name, "second": second_name,
                                                          "same_update_count": first["iterations"] == second["iterations"],
                                                          "same_grid": first["grid"] == second["grid"],
                                                          **field_comparison(first["fields"], second["fields"])})
    _save(output, report)
    return report

def summarize_main(overrides):
    config = configure(SUMMARIZE_CONFIG, STUDIES_KINDS, overrides, ("runs",))
    result = summarize(config["root"], config["runs"], config["output"])
    print(json.dumps({"studies": [study["name"] for study in result["studies"]], "output": config["output"]}))

def plot(root, names, output):
    plt, _ = _plot_modules()
    reference_dir = Path("docs/validation/topology_phase1")
    verify_frozen_reference(reference_dir)
    reference = _read(reference_dir / "optimization.json")["summary"]
    normalization = reference["normalization_compliances_n_mm"]
    weights = reference["normalized_case_weights"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for name in names:
        directory = Path(root) / name
        manifest = _read(directory / "manifest.json")
        if manifest["status"] != "ok":
            raise ValueError(f"Study {name} has no successful manifest")
        verify_artifacts(directory, manifest["artifacts"],
                         ("inputs.json", "result.json", "fields.npz", "domain_masks.npz", "iterations.jsonl"))
        result = _read(directory / "result.json")
        verify_comparable_settings(reference["settings"], result["summary"]["settings"])
        inputs = _read(directory / "inputs.json")
        history = result["history"]
        if any(set(entry["compliances_n_mm"]) != set(normalization) for entry in history):
            raise ValueError("Plot load-case keys disagree with frozen reference")
        grid = inputs["domain"]["grid"]
        label = f"{grid['spacing_mm'][0]:.3g} mm; {result['summary']['iterations']} Updates"
        objective = [sum(weights[case] * value / normalization[case] for case, value in
                         entry["compliances_n_mm"].items()) for entry in history]
        axes[0].semilogy([entry["iteration"] - 1 for entry in history], objective, label=label)
        changes = [entry for entry in history if not entry["final_evaluation"]]
        axes[1].semilogy([entry["iteration"] for entry in changes],
                         [entry["maximum_design_change"] for entry in changes], label=label)
    axes[0].axhline(reference["objective_final"], color="0.45", linestyle=":", label="Referenz nach 45 Updates")
    axes[1].axhline(0.005, color="0.45", linestyle=":", label="Dichteaenderungsgrenze 0.005")
    axes[0].set(title="Gemeinsame Normierung auf 4-mm-Startcompliances", ylabel="Gewichtete normierte Compliance")
    axes[1].set(title="Lokale Designaenderung pro Update", ylabel="Maximale Aenderung der Designdichte")
    for axis in axes:
        axis.set_xlabel("Abgeschlossene Updates")
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Workstation: freie SIMP-Optimierung bei unveraenderten physischen Eingaben", fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

def plot_main(overrides):
    config = configure(PLOT_CONFIG, STUDIES_KINDS, overrides, ("runs",))
    plot(config["root"], config["runs"], config["output"])

def compare(source, output, updates=3):
    source, output = Path(source).resolve(), Path(output).resolve()
    if isinstance(updates, bool) or not 1 <= updates <= 3:
        raise ValueError("Benchmark must use one to three complete updates")
    manifest = _read(source / "manifest.json")
    if manifest["status"] != "ok":
        raise ValueError("Source density study is not successful")
    verify_artifacts(source, manifest["artifacts"], ("inputs.json", "fields.npz", "result.json", "domain_masks.npz"))
    inputs = _read(source / "inputs.json")
    domain = inputs["domain"]
    with np.load(source / "fields.npz", allow_pickle=False) as fields:
        for key in ("allowed", "preserve", "forbidden"):
            domain[key] = fields[key].copy()
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "inputs.json", {"source": str(source), "source_manifest_sha256": _file_digest(source / "manifest.json"),
                                   "source_fields_sha256": _file_digest(source / "fields.npz"), "updates": updates,
                                   "runner_sha256": _file_digest(__file__),
                                   "provenance": _provenance({"generator": optimize_topology}, {}, False),
                                   "thread_environment": {key: os.environ.get(key) for key in
                                                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}})
    results = {}
    try:
        for backend in ("cpu_superlu", "cuda_cudss"):
            settings = {**inputs["settings"], "linear_solver": backend, "max_iterations": updates,
                        "minimum_iterations": updates, "max_runtime_s": 600}
            with (output / (backend + ".jsonl")).open("w", encoding="utf-8") as journal:
                def progress(entry):
                    journal.write(json.dumps(entry, allow_nan=False) + "\n")
                    journal.flush()
                    _save(output / "status.json", {"status": "running", "backend": backend, **entry})
                    print(json.dumps({"backend": backend, "iteration": entry["iteration"],
                                      "elapsed_s": entry["elapsed_s"]}), flush=True)
                started = perf_counter()
                result = generator(domain, settings, progress_callback=progress, **({} if output_shape is None else {"output_domain": output}))
                result["wall_s"] = perf_counter() - started
            arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
            np.savez_compressed(output / (backend + ".npz"), **arrays)
            results[backend] = {key: value for key, value in result.items() if key not in arrays}
            _save(output / (backend + ".json"), results[backend])
            if result["status"] != "ok":
                raise RuntimeError(str(result["diagnostics"]))
        with np.load(output / "cpu_superlu.npz") as cpu, np.load(output / "cuda_cudss.npz") as gpu:
            report = {key + "_max_abs": float(np.max(np.abs(cpu[key] - gpu[key])))
                      for key in ("density", "design_density")}
        pairs = list(zip(results["cpu_superlu"]["history"], results["cuda_cudss"]["history"], strict=True))
        if any(set(cpu["compliances_n_mm"]) != set(gpu["compliances_n_mm"]) for cpu, gpu in pairs):
            raise ValueError("CPU/GPU case sets differ")
        report["history_max_compliance_relative_error"] = max(
            abs(cpu["compliances_n_mm"][key] / gpu["compliances_n_mm"][key] - 1)
            for cpu, gpu in pairs for key in cpu["compliances_n_mm"])
        report["gpu_max_residual"] = max(entry["maximum_relative_residual"] for entry in results["cuda_cudss"]["history"])
        for name, result in results.items():
            report[name + "_wall_s"] = result["wall_s"]
            elapsed = [0] + [entry["elapsed_s"] for entry in result["history"]]
            report[name + "_evaluation_intervals_s"] = np.diff(elapsed).tolist()
        report["speedup"] = report["cpu_superlu_wall_s"] / report["cuda_cudss_wall_s"]
        report["acceptance_limits"] = {"density_max_abs": 1e-7, "design_density_max_abs": 1e-7,
                                       "history_max_compliance_relative_error": 1e-6, "gpu_max_residual": 1e-8}
        report["passed"] = all(report[key] <= limit for key, limit in report["acceptance_limits"].items())
        report["timing_scope"] = "Complete uniform-start optimization including initialization, assembly, transfer, synchronization, OC updates, gradients, final metrics and resource destruction. First interval includes GPU setup; final interval has no OC update."
        _save(output / "comparison.json", report)
        _save(output / "status.json", {"status": "ok" if report["passed"] else "failed"})
        artifacts = {path.name: {"sha256": _file_digest(path), "size_bytes": path.stat().st_size}
                     for path in output.iterdir() if path.is_file()}
        _save(output / "manifest.json", {"status": "ok" if report["passed"] else "failed", "artifacts": artifacts})
        return report
    except Exception as error:
        _save(output / "status.json", {"status": "failed", "error": str(error)})
        raise

def benchmark_main(overrides):
    config = configure(BENCHMARK_CONFIG, BENCHMARK_KINDS, overrides, ("source", "output"))
    result = compare(config["source"], config["output"], config["updates"])
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1

def plot_gpu(evidence, output):
    plt, _ = _plot_modules()
    evidence, output = Path(evidence), Path(output)
    run = evidence / "grid8over3_gpu1000"
    manifest = _read(run / "manifest.json")
    verify_artifacts(run, manifest["artifacts"], ("inputs.json", "result.json", "fields.npz", "iterations.jsonl"))
    result = _read(run / "result.json")
    benchmark = _read(evidence / "three_updates/comparison.json")
    history = [entry for entry in result["history"] if not entry["final_evaluation"]]
    iteration = np.array([entry["iteration"] for entry in history])
    objective = np.array([entry["objective"] for entry in history])
    change = np.array([entry["maximum_design_change"] for entry in history])
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), layout="constrained")
    late = iteration >= 20
    axes[0].plot(iteration[late], objective[late], color="#007f83", linewidth=1.7)
    axes[0].set(xlabel="Update", ylabel="Normalized compliance objective", title="Fixed raster: objective from update 20")
    axes[1].semilogy(iteration, change, color="#007f83", linewidth=1.4)
    axes[1].axhline(0.005, color="#b43b35", linestyle="--", label="Required change < 0.005")
    axes[1].scatter([iteration[-1]], [change[-1]], color="#007f83", zorder=3)
    axes[1].set(xlabel="Update", ylabel="Maximum design-density change", title=f"Criterion reached at update {iteration[-1]}")
    axes[1].legend(fontsize=8)
    times = [benchmark[key] if key in benchmark else benchmark[legacy] for key, legacy in
             (("cpu_superlu_wall_s", "cpu_wall_s"), ("cuda_cudss_wall_s", "gpu_wall_s"))]
    bars = axes[2].bar(["CPU SuperLU", "GPU cuDSS"], times, color=["#66717e", "#007f83"], width=0.6)
    axes[2].bar_label(bars, labels=[f"{value:.3f} s" for value in times], padding=4)
    axes[2].set(ylabel="Complete elapsed time (s)", title=f"3 updates + final evaluation: {benchmark['speedup']:.2f}x")
    axes[2].set_ylim(0, max(times) * 1.16)
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.set_axisbelow(True)
    figure.suptitle("Same Hex8/SIMP physics, 51 x 48 x 12 cells, FP64; geometry and external FEA excluded", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    _save(output.with_suffix(".json"), {"renderer_sha256": _file_digest(__file__),
                                       "result_sha256": _file_digest(run / "result.json"),
                                       "benchmark_sha256": _file_digest(evidence / "three_updates/comparison.json"),
                                       "image_sha256": _file_digest(output),
                                       "interpretation": "Actual stored convergence history and measured three-update wall time. No geometry or FEA acceleration claim."})

def plot_gpu_main(overrides):
    config = configure(GPU_PLOT_CONFIG, GPU_PLOT_KINDS, overrides)
    plot_gpu(config["evidence"], config["output"])

FLOOR_COLORS = ["#ffffff", "#d9d9d9", "#5a5a5a", "#2b6cb0", "#dd6b20"]
FLOOR_LABELS = ["frei, leer", "verboten (Keep-out)", "Optimiererdichte > 0.5", "Vorgabe (Preserve)", "Optimiererdichte in der Bodenschicht z 0..h (vorher abgeschnitten)"]

def floor_check(run, density, output, before=None):
    from tools.formulation_study import configure as formulation, patched_builder
    from deep_frame.topology_geometry import grid_centers, mirror_field, region_contains
    plt, _ = _plot_modules()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cfg = formulation(json.loads(Path(run).read_text(encoding="utf-8")))
    full, _ = patched_builder(cfg).build(cfg["shape"], cfg["reference_fraction"])
    grid, allowed, preserve = full["grid"], full["allowed"], full["preserve"]
    h, origin = np.asarray(grid["spacing_mm"]), np.asarray(grid["origin_mm"])
    field = mirror_field(np.load(density)["density"])
    floor = preserve.shape[2] - field.shape[2]
    field = np.pad(field, ((0, 0), (0, 0), (floor, 0)))
    free = (field > 0.5) & allowed & ~preserve
    centers, cell = grid_centers(grid), float(np.prod(h))
    regions = {region["name"]: region for region in full["regions"]}
    contacts = {}
    for name, region in regions.items():
        if region["role"] == "preserve" and ("aio_contact" in name or "motor_contact" in name):
            cells = region_contains(centers, region) & preserve
            exact = np.pi * region["radius_mm"] ** 2 * region["height_mm"]
            contacts[name] = {"z_mm": [region["center_mm"][2] - region["height_mm"] / 2, region["center_mm"][2] + region["height_mm"] / 2], "radius_mm": region["radius_mm"], "exact_mm3": exact, "raster_mm3": int(cells.sum()) * cell}
    layer = free[:, :, floor]
    labels, count = label(layer)
    blobs = []
    for index in range(1, count + 1):
        cells = np.argwhere(labels == index)
        xy = centers[cells[:, 0], cells[:, 1], floor, :2]
        depth = [int(np.argmin(np.append(free[i, j, floor:], False))) for i, j in cells]
        blobs.append({"cells": len(cells), "x_mm": [float(xy[:, 0].min()), float(xy[:, 0].max())], "y_mm": [float(xy[:, 1].min()), float(xy[:, 1].max())], "height_above_old_floor_mm": [float(min(depth) * h[2]), float(np.median(depth) * h[2]), float(max(depth) * h[2])]})
    fixtures = sorted({(round(box["min_mm"][2], 3), round(box["max_mm"][2], 3)) for case in full["comparison_load_cases"] if case["name"] in ("arm_tip", "crash_front") for box in case.get("fixed_regions", [])})
    report = {"grid": grid, "envelope_z_mm": [float(origin[2]), float(origin[2] + h[2] * grid["shape"][2])], "old_floor_layer": floor, "allowed_mm3": int(allowed.sum()) * cell, "preserve_mm3": int(preserve.sum()) * cell,
              "contacts": contacts, "stack_fixtures_z_mm": fixtures,
              "old_floor_layer_cells": {"preserve": int(preserve[:, :, floor].sum()), "optimizer": int(layer.sum()), "preserve_mm3": int(preserve[:, :, floor].sum()) * cell, "optimizer_mm3": int(layer.sum()) * cell},
              "old_floor_blobs": blobs, "density": density, "run": run}
    if before:
        old = json.loads(Path(before).read_text(encoding="utf-8"))
        report["diff"] = {"allowed_mm3": report["allowed_mm3"] - old["allowed_mm3"], "preserve_mm3": report["preserve_mm3"] - old["preserve_mm3"], "envelope_z_mm": [old["envelope_z_mm"], report["envelope_z_mm"]],
                          "stack_fixtures_z_mm": [old["stack_fixtures_z_mm"], report["stack_fixtures_z_mm"]],
                          "contacts": {name: {"before": old["contacts"].get(name), "after": value} for name, value in contacts.items() if "aio" in name}}
    category = np.where(allowed, 0, 1)
    category[free] = 2
    category[preserve] = 3
    category[:, :, floor][layer] = 4
    aio = regions["aio_contact_3"]["center_mm"]
    i, j = (np.abs(centers[:, 0, 0, 0] - aio[0]).argmin(), np.abs(centers[0, :, 0, 1] - aio[1]).argmin())
    top = origin + h * np.asarray(grid["shape"])
    cmap = ListedColormap(FLOOR_COLORS)
    figure, axes = plt.subplots(3, 1, figsize=(12, 13), gridspec_kw={"height_ratios": [1, 1, 3]})
    views = [(category[i].T, [origin[1], top[1], origin[2], top[2]], f"Schnitt x = {centers[i, 0, 0, 0]:.2f} mm (Seitenansicht durch die rechten Stack-Schrauben)", "y mm"),
             (category[:, j].T, [origin[0], top[0], origin[2], top[2]], f"Schnitt y = {centers[0, j, 0, 1]:.2f} mm (Frontansicht durch die vorderen Stack-Schrauben)", "x mm"),
             (category[:, :, floor].T, [origin[0], top[0], origin[1], top[1]], f"Bodenschicht z {origin[2] + floor * h[2]:.2f}..{origin[2] + (floor + 1) * h[2]:.2f} mm (Draufsicht)", "x mm")]
    for axis, (image, extent, title, xlabel) in zip(axes, views):
        axis.imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("y mm" if image is views[2][0] else "z mm")
        if image is not views[2][0]:
            axis.axhline(0.0, color="#c53030", linestyle="--", linewidth=1)
            axis.set_aspect("equal")
    axes[2].set_aspect("equal")
    figure.legend([Patch(color=color) for color in FLOOR_COLORS], FLOOR_LABELS, loc="lower center", ncol=3, fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / "floor.png", dpi=110)
    (out / "floor.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return report

def floor_main(overrides):
    config = configure(FLOOR_CONFIG, FLOOR_KINDS, overrides, ("run", "density"))
    report = floor_check(config["run"], config["density"], config["output"], config["before"])
    print(json.dumps({key: report[key] for key in ("envelope_z_mm", "old_floor_layer_cells", "stack_fixtures_z_mm")}, default=float))

def main(argv=None):
    return command_line({"run": run_main, "summarize": summarize_main, "plot": plot_main, "benchmark": benchmark_main,
                         "plot-gpu": plot_gpu_main, "neural": neural_main, "floor": floor_main}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
