"""Density study tools: run bounded experiments, summarize verified evidence and plot histories.

Run from the repository root with ``python -m tools.topology_study {run,summarize,plot} --help``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import label

from deep_frame.config import TOPOLOGY_CONFIG
from deep_frame.frame import reference_parameters
from deep_frame.topology_geometry import build_design_domain
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


def run_main(argv=None, prog=None):
    """Bounded, observable density experiments; each invocation starts uniformly.

    Run from the repository root with ``python -m tools.topology_study run --help``.
    This runner does not reconstruct or mechanically accept any candidate.
    """
    parser = argparse.ArgumentParser(prog=prog, description=run_main.__doc__)
    parser.add_argument("--directory", required=True, help="New output directory; refuses existing paths.")
    parser.add_argument("--shape", nargs=3, type=int, default=[34, 32, 8])
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--change-tolerance", type=float, default=0.005)
    parser.add_argument("--max-runtime-s", type=float, default=1800.0)
    parser.add_argument("--linear-solver", choices=("cpu_superlu", "cuda_cudss"), default="cpu_superlu")
    args = parser.parse_args(argv)
    status = run_study(args.directory, args.shape, args.max_iterations, args.change_tolerance, args.max_runtime_s, args.linear_solver)
    return 0 if status == "ok" else 1


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    acceptance = read_json(Path(directory) / "acceptance.json")
    if acceptance["status"] != "ok":
        raise ValueError("Frozen reference has no successful acceptance")
    verify_artifacts(directory, acceptance["versioned_artifacts"],
                     ("optimization.json", "fields.npz", "run_manifest.json"))


def verify_comparable_settings(reference, studied):
    permitted = {"max_iterations", "minimum_iterations", "change_tolerance", "max_runtime_s"}
    if {key: value for key, value in reference.items() if key not in permitted} != {
            key: value for key, value in studied.items() if key not in permitted}:
        raise ValueError("Optimization physics/settings changed beyond iteration stopping controls")
    if studied["minimum_iterations"] != min(reference["minimum_iterations"], studied["max_iterations"]):
        raise ValueError("Minimum iteration count changed beyond short benchmark adjustment")


def field_comparison(first, second):
    """Exact piecewise-constant voxel overlap on the common subdivision."""
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
    reference = read_json(reference_dir / "optimization.json")
    reference_manifest = read_json(reference_dir / "run_manifest.json")
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
        manifest = read_json(directory / "manifest.json")
        if manifest["status"] != "ok":
            raise ValueError(f"Study {name} has no successful manifest")
        verify_artifacts(directory, manifest["artifacts"],
                         ("inputs.json", "result.json", "fields.npz", "domain_masks.npz", "iterations.jsonl"))
        result = read_json(directory / "result.json")
        inputs = read_json(directory / "inputs.json")
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


def summarize_main(argv=None, prog=None):
    """Verify completed density experiments and summarize their numerical evidence."""
    parser = argparse.ArgumentParser(prog=prog, description=summarize_main.__doc__)
    parser.add_argument("--root", default="exports/topology/workstation_20260930/density_study")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="docs/validation/workstation_density_study.json")
    options = parser.parse_args(argv)
    result = summarize(options.root, options.runs, options.output)
    print(json.dumps({"studies": [study["name"] for study in result["studies"]], "output": options.output}))


def plot(root, names, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_dir = Path("docs/validation/topology_phase1")
    verify_frozen_reference(reference_dir)
    reference = read_json(reference_dir / "optimization.json")["summary"]
    normalization = reference["normalization_compliances_n_mm"]
    weights = reference["normalized_case_weights"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for name in names:
        directory = Path(root) / name
        manifest = read_json(directory / "manifest.json")
        if manifest["status"] != "ok":
            raise ValueError(f"Study {name} has no successful manifest")
        verify_artifacts(directory, manifest["artifacts"],
                         ("inputs.json", "result.json", "fields.npz", "domain_masks.npz", "iterations.jsonl"))
        result = read_json(directory / "result.json")
        verify_comparable_settings(reference["settings"], result["summary"]["settings"])
        inputs = read_json(directory / "inputs.json")
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


def plot_main(argv=None, prog=None):
    """Plot stored density histories using the same frozen compliance normalization."""
    parser = argparse.ArgumentParser(prog=prog, description=plot_main.__doc__)
    parser.add_argument("--root", default="exports/topology/workstation_20260930/density_study")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="docs/validation/workstation_density_convergence.png")
    arguments = parser.parse_args(argv)
    plot(arguments.root, arguments.runs, arguments.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = {"run": run_main, "summarize": summarize_main, "plot": plot_main}
    for name, function in handlers.items():
        commands.add_parser(name, help=function.__doc__.splitlines()[0], add_help=False)
    args, arguments = parser.parse_known_args(argv)
    return handlers[args.command](arguments, parser.prog + " " + args.command)


if __name__ == "__main__":
    raise SystemExit(main())
