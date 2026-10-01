from copy import deepcopy

import pytest

from deep_frame.topology_pipeline import _artifact, _save
from tools import run_workstation_candidate_study as study


def small_result():
    return {
        "status": "ok", "frame_mass_g": 1.0, "stiffness_n_per_mm": 2.0,
        "max_displacement_mm": 0.5, "max_von_mises_mpa": 3.0,
        "eigenfrequencies_hz": [100.0],
        "load_cases": {
            "tip": {"analysis": "static", "max_displacement_mm": 0.5, "max_von_mises_mpa": 3.0},
            "modes": {"analysis": "modal", "eigenfrequencies_hz": [100.0]},
        },
    }


def physical_inputs():
    return {
        "domain": {"comparison_load_cases": [
            {"name": "tip", "analysis": "static"}, {"name": "modes", "analysis": "modal"},
        ]},
        "fea_settings": {"mesh_size_mm": 3.0, "threads": 2},
        "baseline_reference_frame_mass_g": 1.0,
    }


def test_density_manifest_cannot_omit_hashes_for_required_inputs(tmp_path):
    _save(tmp_path / "manifest.json", {"status": "ok", "artifacts": {}})
    with pytest.raises(ValueError, match="missing required artifacts"):
        study.load_source(tmp_path)


def test_candidate_resume_rejects_fea_disagreement_with_hashed_raw_result(tmp_path):
    result = small_result()
    _save(tmp_path / "raw_result.json", result)
    corrupted = deepcopy(result)
    corrupted["stiffness_n_per_mm"] = 999.0
    artifacts = {"raw_fea_result.json": _artifact(tmp_path / "raw_result.json", tmp_path)}
    _save(tmp_path / "record.json", {"status": "ok", "fea": corrupted, "artifacts": artifacts})
    manifest = {"candidates": {"test": _artifact(tmp_path / "record.json", tmp_path)}}
    with pytest.raises(ValueError, match="differs from the hashed raw result"):
        study.read_candidates(tmp_path, manifest, physical_inputs())


def test_candidate_resume_rechecks_modal_case_even_when_all_hashes_match(tmp_path):
    result = small_result()
    del result["load_cases"]["modes"]
    _save(tmp_path / "raw_result.json", result)
    artifacts = {"raw_fea_result.json": _artifact(tmp_path / "raw_result.json", tmp_path)}
    _save(tmp_path / "record.json", {"status": "ok", "fea": result, "checks": {"passed": True}, "artifacts": artifacts})
    manifest = {"candidates": {"test": _artifact(tmp_path / "record.json", tmp_path)}}
    with pytest.raises(ValueError, match="omitted or changed case modes"):
        study.read_candidates(tmp_path, manifest, physical_inputs())


def test_intact_own_baseline_resumes_without_running_solver_and_rejects_corruption(tmp_path, monkeypatch):
    result = small_result()
    _save(tmp_path / "baseline/raw_result.json", result)
    artifacts = {"raw_fea_result.json": _artifact(tmp_path / "baseline/raw_result.json", tmp_path)}
    record = {"result": result, "artifacts": artifacts, "settings": physical_inputs()["fea_settings"]}
    _save(tmp_path / "baseline/record.json", record)

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("A verified existing baseline must not start another solver")

    monkeypatch.setattr(study, "evaluate", unexpected_solver)
    assert study.baseline_result(tmp_path, physical_inputs(), None) == record
    (tmp_path / "baseline/raw_result.json").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="raw artifact mismatch"):
        study.baseline_result(tmp_path, physical_inputs(), None)
