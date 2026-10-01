"""Plot stored density histories using the same frozen compliance normalization."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.summarize_topology_study import read_json, verify_artifacts, verify_comparable_settings, verify_frozen_reference


def plot(root, names, output):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="exports/topology/workstation_20260930/density_study")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="docs/validation/workstation_density_convergence.png")
    arguments = parser.parse_args()
    plot(arguments.root, arguments.runs, arguments.output)
