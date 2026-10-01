"""Verify completed density experiments and summarize their numerical evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from deep_frame.topology_pipeline import _file_digest, _save


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="exports/topology/workstation_20260930/density_study")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="docs/validation/workstation_density_study.json")
    options = parser.parse_args()
    result = summarize(options.root, options.runs, options.output)
    print(json.dumps({"studies": [study["name"] for study in result["studies"]], "output": options.output}))
