"""Plot the observed GPU optimization and complete CPU/GPU benchmark."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deep_frame.topology_pipeline import _file_digest, _save
from tools.summarize_topology_study import verify_artifacts


def render(evidence, output):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="docs/validation/workstation_gpu")
    parser.add_argument("--output", default="docs/validation/workstation_gpu_convergence.png")
    arguments = parser.parse_args()
    render(arguments.evidence, arguments.output)
