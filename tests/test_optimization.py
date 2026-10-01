import csv
import json
import math
from copy import deepcopy

import pytest

from deep_frame.optimization import EvaluationFailure, check_printability, optimize, show_candidates

def beam_result(parameters):
    beam = parameters["beam"]
    length = beam["length_mm"] / 1000
    width = beam["width_mm"] / 1000
    height = beam["height_mm"] / 1000
    area = width * height
    inertia = width * height**3 / 12
    modulus = 3.4e9
    density = 1090
    displacement_mm = length**3 / (3 * modulus * inertia) * 1000
    frequency_hz = 1.875104068711961**2 / (2 * math.pi * length**2) * math.sqrt(modulus * inertia / (density * area))
    return {
        "status": "ok",
        "mass_g": density * area * length * 1000,
        "eigenfrequencies_hz": [frequency_hz],
        "max_displacement_mm": displacement_mm,
        "max_von_mises_mpa": 6 * length / (width * height**2) / 1e6,
        "stiffness_n_per_mm": 1 / displacement_mm,
        "load_cases": {},
        "diagnostics": [],
        "artifacts": {},
    }

def valid_geometry(parameters):
    return {"passed": True, "violations": [], "checks": {}}

@pytest.fixture
def reference():
    return {"beam": {"length_mm": 100.0, "width_mm": 10.0, "height_mm": 3.0, "wall_mm": 2.0}}

@pytest.fixture
def settings(tmp_path):
    return {
        "n_trials": 30,
        "seed": 42,
        "population_size": 8,
        "storage": "sqlite:///" + (tmp_path / "study.sqlite3").as_posix(),
        "study_name": "beam-analytical",
        "evaluation_id": "beam-euler-bernoulli-si-v1",
        "output_dir": str(tmp_path / "exports"),
        "relative_constraints": {
            "mass_ratio_max": 1.05,
            "stiffness_ratio_min": 0.95,
            "frequency_ratio_min": 0.95,
            "displacement_ratio_max": 1.05,
            "stress_ratio_max": 1.10,
        },
        "printability": {
            "nozzle_width_mm": 0.4,
            "minimum_wall_nozzles": 4,
            "wall_thickness_paths": ["beam.wall_mm", "beam.height_mm"],
            "clamp_sections": [{"name": "root", "width_path": "beam.width_mm", "height_path": "beam.height_mm", "minimum_area_mm2": 20.0}],
        },
        "initial_candidates": [],
    }

def test_analytical_cantilever_finds_known_optimum(reference, settings):
    original_reference = deepcopy(reference)
    original_settings = deepcopy(settings)
    result = optimize(reference, {"beam.length_mm": {"low": 40.0, "high": 100.0, "step": 10.0}}, beam_result, valid_geometry, settings)
    assert result["status"] == "ok"
    assert result["has_improved_design"]
    assert result["pareto_front"]
    assert all(trial["parameters"]["beam"]["length_mm"] == pytest.approx(40.0, abs=1e-12) for trial in result["pareto_front"])
    winner = result["pareto_front"][0]
    assert winner["metrics"]["mass_g"] / result["reference"]["metrics"]["mass_g"] == pytest.approx(0.4)
    assert winner["metrics"]["stiffness_n_per_mm"] / result["reference"]["metrics"]["stiffness_n_per_mm"] == pytest.approx(15.625)
    assert winner["metrics"]["first_frequency_hz"] / result["reference"]["metrics"]["first_frequency_hz"] == pytest.approx(6.25)
    assert all(check["passed"] for check in winner["constraints"].values())
    assert reference == original_reference
    assert settings == original_settings
    with open(result["artifacts"]["pareto_csv"], newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(result["pareto_front"])
    with open(result["artifacts"]["pareto_json"], encoding="utf-8") as file:
        assert json.load(file) == result["pareto_front"]
    json.dumps(result, allow_nan=False)

@pytest.mark.parametrize("path,value,outcome", [
    ("beam.wall_mm", 1.59, "rejected_printability"),
    ("beam.width_mm", 6.0, "rejected_printability"),
    ("beam.length_mm", 40.0, "rejected_geometry"),
])
def test_invalid_design_never_reaches_evaluator(reference, settings, path, value, outcome):
    evaluations = []
    validations = []
    def evaluator(parameters):
        evaluations.append(deepcopy(parameters))
        return beam_result(parameters)
    def validator(parameters):
        validations.append(deepcopy(parameters))
        passed = parameters["beam"]["length_mm"] >= 50.0
        return {"passed": passed, "violations": [] if passed else ["fixture"], "checks": {}}
    settings["n_trials"] = 1
    result = optimize(reference, {path: {"choices": [value]}}, evaluator, validator, settings)
    assert len(evaluations) == 1
    assert evaluations[0] == reference
    assert len(validations) == (1 if outcome == "rejected_printability" else 2)
    assert result["trial_counts"][outcome] == 1
    assert result["pareto_front"] == []
    assert result["status"] == "no_valid_design"

def test_printability_boundary_and_root_section(reference, settings):
    reference["beam"]["wall_mm"] = 1.6
    assert check_printability(reference, settings["printability"])["passed"]
    reference["beam"]["width_mm"] = 6.0
    check = check_printability(reference, settings["printability"])
    assert not check["passed"]
    assert check["checks"]["section:root"]["measured_mm2"] == 18.0

def test_constraints_preserve_original_reference_and_remove_infeasible_trials(reference, settings):
    settings["n_trials"] = 3
    settings["initial_candidates"] = [{"beam.height_mm": 3.0}, {"beam.height_mm": 4.0}, {"beam.height_mm": 2.0}]
    settings["printability"]["clamp_sections"][0]["minimum_area_mm2"] = 10.0
    result = optimize(reference, {"beam.height_mm": {"choices": [2.0, 3.0, 4.0]}}, beam_result, valid_geometry, settings)
    assert result["trial_counts"]["valid"] == 1
    assert result["trial_counts"]["rejected_constraints"] == 2
    assert [trial["parameters"]["beam"]["height_mm"] for trial in result["pareto_front"]] == [3.0]
    for trial in result["trials"]:
        for check in trial["constraints"].values():
            assert check["reference"] == result["reference"]["metrics"][check["metric"]]
    assert result["trials"][1]["constraints"]["mass_ratio_max"]["passed"] is False
    assert result["trials"][2]["constraints"]["stiffness_ratio_min"]["passed"] is False

@pytest.mark.parametrize("failure", ["status", "exception", "nan", "zero_mode"])
def test_failed_evaluation_never_enters_pareto(reference, settings, failure):
    def evaluator(parameters):
        result = beam_result(parameters)
        if parameters["beam"]["length_mm"] == 40.0:
            if failure == "exception":
                raise RuntimeError("solver timeout")
            if failure == "status":
                result["status"] = "failed"
                result["diagnostics"] = ["solver failed"]
            if failure == "nan":
                result["mass_g"] = float("nan")
            if failure == "zero_mode":
                result["eigenfrequencies_hz"] = [0.0]
        return result
    settings["n_trials"] = 1
    result = optimize(reference, {"beam.length_mm": {"choices": [40.0]}}, evaluator, valid_geometry, settings)
    assert result["trial_counts"]["evaluation_failed"] == 1
    assert result["trials"][0]["state"] == "FAIL"
    assert result["pareto_front"] == []
    assert result["trials"][0]["diagnostics"]

def test_persistent_resume_evaluates_reference_once_and_rejects_contract_change(reference, settings):
    evaluations = []
    def evaluator(parameters):
        evaluations.append(parameters["beam"]["length_mm"])
        return beam_result(parameters)
    settings["n_trials"] = 1
    space = {"beam.length_mm": {"choices": [40.0]}}
    first = optimize(reference, space, evaluator, valid_geometry, settings)
    second = optimize(reference, space, evaluator, valid_geometry, settings)
    assert evaluations == [100.0, 40.0, 40.0]
    assert second["trial_counts"]["total"] == 2
    assert second["reference"] == first["reference"]
    settings["relative_constraints"]["mass_ratio_max"] = 1.20
    with pytest.raises(ValueError, match="different reference"):
        optimize(reference, space, evaluator, valid_geometry, settings)
    assert evaluations == [100.0, 40.0, 40.0]

def test_input_isolation_from_mutating_callbacks(reference, settings):
    original = deepcopy(reference)
    def evaluator(parameters):
        result = beam_result(parameters)
        parameters["beam"]["length_mm"] = -999
        return result
    def validator(parameters):
        parameters["beam"]["height_mm"] = -999
        return valid_geometry(parameters)
    settings["n_trials"] = 1
    result = optimize(reference, {"beam.length_mm": {"choices": [40.0]}}, evaluator, validator, settings)
    assert reference == original
    assert result["pareto_front"][0]["parameters"]["beam"]["height_mm"] == 3.0
    assert result["pareto_front"][0]["parameters"]["beam"]["length_mm"] == 40.0

def test_failed_reference_prevents_trials(reference, settings):
    calls = []
    def evaluator(parameters):
        calls.append(parameters)
        return {"status": "failed", "diagnostics": ["unavailable"]}
    with pytest.raises(EvaluationFailure, match="Reference evaluation failed"):
        optimize(reference, {"beam.length_mm": {"choices": [40.0]}}, evaluator, valid_geometry, settings)
    assert calls == [reference]

def test_viewer_adapter_rebuilds_only_valid_parameters(reference):
    candidate = {"number": 5, "outcome": "valid", "parameters": deepcopy(reference)}
    displayed = []
    def geometry(parameters):
        parameters["beam"]["height_mm"] = 999
        return "shape"
    def show(*shapes, **settings):
        displayed.append((shapes, settings))
    assert show_candidates([candidate], geometry, show, {"port": 3939}) == {"displayed_trials": [5]}
    assert displayed == [(("shape",), {"names": ["Trial 5"], "port": 3939})]
    assert candidate["parameters"] == reference
    candidate["outcome"] = "evaluation_failed"
    with pytest.raises(ValueError, match="Only valid"):
        show_candidates([candidate], geometry, show)
