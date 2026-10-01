"""CPU/GPU density solver tools: benchmark complete optimization updates and plot GPU evidence.

Run from the repository root with ``python -m tools.gpu_benchmark {benchmark,plot} --help``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from deep_frame.topology_optimization import optimize_topology
from deep_frame.topology_pipeline import _file_digest, _provenance, _save
from tools.topology_study import verify_artifacts


def compare(source, output, updates=3):
    source, output = Path(source).resolve(), Path(output).resolve()
    if isinstance(updates, bool) or not 1 <= updates <= 3:
        raise ValueError("Benchmark must use one to three complete updates")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "ok":
        raise ValueError("Source density study is not successful")
    verify_artifacts(source, manifest["artifacts"], ("inputs.json", "fields.npz", "result.json", "domain_masks.npz"))
    inputs = json.loads((source / "inputs.json").read_text())
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
                result = optimize_topology(domain, settings, progress_callback=progress)
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
        artifacts = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
                     for path in output.iterdir() if path.is_file()}
        _save(output / "manifest.json", {"status": "ok" if report["passed"] else "failed", "artifacts": artifacts})
        return report
    except Exception as error:
        _save(output / "status.json", {"status": "failed", "error": str(error)})
        raise


def benchmark_main(argv=None, prog=None):
    """Compare complete CPU/GPU optimization updates from verified physical inputs."""
    parser = argparse.ArgumentParser(prog=prog, description=benchmark_main.__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=3)
    arguments = parser.parse_args(argv)
    result = compare(arguments.source, arguments.output, arguments.updates)
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def render(evidence, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evidence, output = Path(evidence), Path(output)
    run = evidence / "grid8over3_gpu1000"
    manifest = json.loads((run / "manifest.json").read_text())
    verify_artifacts(run, manifest["artifacts"], ("inputs.json", "result.json", "fields.npz", "iterations.jsonl"))
    result = json.loads((run / "result.json").read_text())
    benchmark = json.loads((evidence / "three_updates/comparison.json").read_text())
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
    times = [benchmark["cpu_wall_s"], benchmark["gpu_wall_s"]]
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


def plot_main(argv=None, prog=None):
    """Plot the observed GPU optimization and complete CPU/GPU benchmark."""
    parser = argparse.ArgumentParser(prog=prog, description=plot_main.__doc__)
    parser.add_argument("--evidence", default="docs/validation/workstation_gpu")
    parser.add_argument("--output", default="docs/validation/workstation_gpu_convergence.png")
    arguments = parser.parse_args(argv)
    render(arguments.evidence, arguments.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = {"benchmark": benchmark_main, "plot": plot_main}
    for name, function in handlers.items():
        commands.add_parser(name, help=function.__doc__.splitlines()[0], add_help=False)
    args, arguments = parser.parse_known_args(argv)
    return handlers[args.command](arguments, parser.prog + " " + args.command)


if __name__ == "__main__":
    raise SystemExit(main())
