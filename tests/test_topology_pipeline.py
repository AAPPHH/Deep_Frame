import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from build123d import Box

from deep_frame.topology_pipeline import PIPELINE_CONFIG, compare_to_baseline, run_topology


def pipeline_fixture():
    shape = (3, 3, 3)
    region = {"kind": "box", "min_mm": [0, 0, 0], "max_mm": [1, 1, 1]}
    cases = [{"name": "load", "analysis": "static", "fixed_regions": [region], "loads": [{"region": region, "force_n": [0, 0, 1]}]}, {"name": "modes", "analysis": "modal", "fixed_regions": [region]}]
    domain = {
        "grid": {"origin_mm": [0, 0, 0], "shape": list(shape), "spacing_mm": [2, 2, 2], "axis_order": "xyz", "order": "C"},
        "allowed": np.ones(shape, dtype=bool), "preserve": np.zeros(shape, dtype=bool), "forbidden": np.zeros(shape, dtype=bool),
        "material": {"young_modulus_mpa": 4430, "poisson_ratio": 0.3, "density_g_cm3": 1.0},
        "point_masses": [{"name": "battery", "mass_g": 37, "position_mm": [0, 0, 10], "attachment_region": region}],
        "load_cases": cases, "comparison_load_cases": cases, "fea_settings": {}, "optimizer_settings": {}, "reconstruction_settings": {},
        "regions": [], "manufacturing": {}, "metadata": {},
    }
    calls = {"generator": 0, "fea": 0, "reconstructor": 0}

    def domain_builder(parameters):
        return deepcopy(domain)

    def generator(current, settings):
        calls["generator"] += 1
        current["grid"]["origin_mm"][0] = 123
        return {"status": "ok", "density": np.full(shape, 0.4), "design_density": np.full(shape, 0.35), "summary": {"converged": True}, "history": [{"iteration": 1, "objective": 0.5}], "diagnostics": []}

    def reconstructor(current, density, settings):
        calls["reconstructor"] += 1
        if settings["density_threshold"] >= 0.5:
            raise ValueError("Disconnected required interface")
        assert current["grid"]["origin_mm"][0] == 0
        return Box(4, 4, 4)

    def validator(solid, current, settings):
        return {"passed": True, "violations": [], "checks": {"single_solid": True}}

    def evaluator(solid, material, point_masses, load_cases, settings):
        calls["fea"] += 1
        assert load_cases == cases
        assert point_masses[0]["mass_g"] == 37
        baseline = solid.volume > 90
        displacement, stress, stiffness, frequency = (0.1, 1.0, 10.0, 100.0) if baseline else (0.08, 0.8, 12.0, 120.0)
        mass = solid.volume / 1000
        return {"status": "ok", "frame_mass_g": mass, "mass_g": mass + 37, "max_displacement_mm": displacement, "max_von_mises_mpa": stress, "stiffness_n_per_mm": stiffness, "eigenfrequencies_hz": [frequency], "load_cases": {"load": {"analysis": "static", "max_displacement_mm": displacement, "max_von_mises_mpa": stress}, "modes": {"analysis": "modal", "eigenfrequencies_hz": [frequency]}}, "artifacts": {}, "diagnostics": []}

    def baseline_builder(parameters):
        return Box(5, 5, 4)

    return domain, calls, {"domain_builder": domain_builder, "generator": generator, "reconstructor": reconstructor, "validator": validator, "evaluator": evaluator, "baseline_builder": baseline_builder}


def test_pipeline_persists_inputs_fields_failures_pareto_and_resumes(tmp_path):
    domain, calls, callbacks = pipeline_fixture()
    parameters = {"component": {"mass": 37}}
    original = deepcopy(parameters)
    settings = {"output_dir": str(tmp_path), "density_thresholds": [0.2, 0.5]}
    first = run_topology(parameters, settings, **callbacks)
    assert first["status"] == "ok"
    assert first["selected_id"] == "default_t00"
    assert first["pareto_ids"] == ["default_t00"]
    assert first["candidates"][1]["status"] == "invalid"
    assert "Disconnected" in first["candidates"][1]["diagnostics"][0]
    assert calls == {"generator": 1, "fea": 2, "reconstructor": 2}
    directory = Path(first["run_dir"])
    assert json.loads((directory / "inputs.json").read_text())["domain"]["grid"]["origin_mm"] == [0, 0, 0]
    with np.load(directory / "optimizer/default/fields.npz", allow_pickle=False) as fields:
        assert set(fields.files) == {"density", "design_density", "allowed", "preserve", "forbidden"}
        assert np.all(fields["density"] == 0.4)
    assert (directory / "candidates/default_t00/geometry.step").stat().st_size > 0
    assert (directory / "pareto.csv").read_text().count("\n") == 2
    assert parameters == original
    second = run_topology(parameters, settings, **callbacks)
    assert second["input_sha256"] == first["input_sha256"]
    assert second["status"] == "ok"
    assert calls == {"generator": 1, "fea": 2, "reconstructor": 2}


def test_changed_settings_or_inputs_get_new_cache_identity(tmp_path):
    _, calls, callbacks = pipeline_fixture()
    settings = {"output_dir": str(tmp_path), "density_thresholds": [0.2]}
    first = run_topology({"version": 1}, settings, **callbacks)
    second = run_topology({"version": 2}, settings, **callbacks)
    third = run_topology({"version": 2}, {**settings, "optimizer_settings": {"projection_beta": 3.0}}, **callbacks)
    assert len({entry["input_sha256"] for entry in (first, second, third)}) == 3
    assert calls["generator"] == 3
    assert calls["fea"] == 6


def test_corrupted_density_artifact_is_recomputed_and_immutable_inputs_are_checked(tmp_path):
    _, calls, callbacks = pipeline_fixture()
    settings = {"output_dir": str(tmp_path), "density_thresholds": [0.2]}
    first = run_topology({}, settings, **callbacks)
    directory = Path(first["run_dir"])
    (directory / "optimizer/default/fields.npz").write_bytes(b"corrupt")
    second = run_topology({}, settings, **callbacks)
    assert second["status"] == "ok"
    assert calls["generator"] == 2
    assert calls["fea"] == 3
    inputs_path = directory / "inputs.json"
    inputs = json.loads(inputs_path.read_text())
    inputs["settings"]["seed"] = -1
    inputs_path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError, match="Immutable topology input"):
        run_topology({}, settings, **callbacks)


def test_manufacturing_failures_are_rejected_before_independent_fea(tmp_path):
    _, calls, callbacks = pipeline_fixture()

    def validator(solid, domain, settings):
        return {"passed": False, "violations": ["wall below nozzle limit"], "checks": {}}

    callbacks["validator"] = validator
    result = run_topology({}, {"output_dir": str(tmp_path), "density_thresholds": [0.2]}, **callbacks)
    assert result["status"] == "invalid"
    assert result["selected_id"] is None
    assert calls["fea"] == 1
    assert "wall below nozzle limit" in result["candidates"][0]["diagnostics"][0]


def test_missing_modal_case_cannot_pass_mechanical_verification(tmp_path):
    _, _, callbacks = pipeline_fixture()
    original = callbacks["evaluator"]

    def evaluator(solid, material, masses, cases, settings):
        result = original(solid, material, masses, cases, settings)
        if solid.volume < 90:
            result["load_cases"].pop("modes")
        return result

    callbacks["evaluator"] = evaluator
    result = run_topology({}, {"output_dir": str(tmp_path), "density_thresholds": [0.2]}, **callbacks)
    assert result["status"] == "invalid"
    assert "omitted or changed case modes" in result["candidates"][0]["diagnostics"][0]


def test_failed_optimization_is_persisted_without_geometry_or_candidate_fea(tmp_path):
    _, calls, callbacks = pipeline_fixture()

    def generator(domain, settings):
        raise RuntimeError("singular elasticity")

    callbacks["generator"] = generator
    result = run_topology({}, {"output_dir": str(tmp_path), "density_thresholds": [0.2]}, **callbacks)
    assert result["status"] == "invalid"
    assert result["optimizations"][0]["status"] == "failed"
    assert result["candidates"] == []
    assert calls["fea"] == 1
    path = Path(result["run_dir"]) / "optimizer/default/result.json"
    assert "singular elasticity" in json.loads(path.read_text())["diagnostics"][0]


def test_all_five_predeclared_comparisons_are_enforced():
    result = {"status": "ok", "frame_mass_g": 30.0, "stiffness_n_per_mm": 10.0, "max_displacement_mm": 0.1, "max_von_mises_mpa": 1.0, "eigenfrequencies_hz": [100.0]}
    limits = deepcopy(PIPELINE_CONFIG["relative_constraints"])
    assert compare_to_baseline(result, result, limits)["passed"]
    for key, value, expected in (("frame_mass_g", 60.01, "frame_mass"), ("stiffness_n_per_mm", 4.99, "stiffness"), ("max_displacement_mm", 0.201, "displacement"), ("max_von_mises_mpa", 2.001, "stress"), ("eigenfrequencies_hz", [69.99], "frequency")):
        candidate = {**result, key: value}
        comparison = compare_to_baseline(candidate, result, limits)
        assert not comparison["passed"]
        assert not comparison["checks"][expected]
