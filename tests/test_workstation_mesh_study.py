"""Integrity failures must stop the resumed mesh study before native work."""

from copy import deepcopy
import hashlib
import json

import pytest

from tools import run_workstation_mesh_study as study


def test_modified_archived_field_stops_before_reconstruction(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    original = b"original archived density field"
    (evidence / "fields.npz").write_bytes(b"altered archived density field")
    (evidence / "acceptance.json").write_text(json.dumps({
        "versioned_artifacts": {"fields.npz": {
            "sha256": hashlib.sha256(original).hexdigest(),
            "size_bytes": len(original),
        }},
    }), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(study, "EVIDENCE", evidence)
    with pytest.raises(ValueError, match="SHA256/size"):
        study.prepare(output, 2)


def test_unowned_stale_geometry_is_not_adopted(tmp_path):
    geometry = tmp_path / "geometry/candidate"
    geometry.mkdir(parents=True)
    (geometry / "geometry.step").write_text("stale unrelated CAD", encoding="utf-8")
    with pytest.raises(ValueError, match="empty output directory"):
        study.prepare(tmp_path, 2)


def test_cached_ok_result_still_requires_all_verification_cases(tmp_path):
    result = deepcopy(study.read(study.EVIDENCE / "candidate_fea.json"))
    del result["load_cases"]["modes"]
    result["artifacts"] = {}
    inputs = study.read(study.EVIDENCE / "inputs.json")
    preparation = {"study_input_sha256": "same-input", "load_cases": inputs["domain"]["comparison_load_cases"]}
    record = {"study_input_sha256": "same-input", "status": "ok", "result": result, "artifacts": {}}
    with pytest.raises(ValueError, match="modes"):
        study.verify_record(record, tmp_path, preparation)
