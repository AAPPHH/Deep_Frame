import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from deep_frame.topology_optimization import _settings
from tools.summarize_topology_study import field_comparison, verify_artifacts, verify_comparable_settings, verify_frozen_reference


def test_comparison_rejects_changed_physical_filter_but_allows_iteration_budget():
    reference = _settings({})
    studied = deepcopy(reference)
    studied.update(max_iterations=300, max_runtime_s=1800, change_tolerance=0.005)
    verify_comparable_settings(reference, studied)
    studied["filter_radius_mm"] = 5.0
    with pytest.raises(ValueError, match="physics/settings"):
        verify_comparable_settings(reference, studied)


def test_frozen_reference_verification_rejects_corrupted_fields(tmp_path):
    artifacts = {}
    for name in ("optimization.json", "fields.npz", "run_manifest.json"):
        content = ("test evidence " + name).encode()
        (tmp_path / name).write_bytes(content)
        artifacts[name] = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    (tmp_path / "acceptance.json").write_text(json.dumps({"status": "ok", "versioned_artifacts": artifacts}), encoding="utf-8")
    verify_frozen_reference(tmp_path)
    (tmp_path / "fields.npz").write_bytes(b"corrupt evidence")
    with pytest.raises(ValueError, match="Corrupt evidence"):
        verify_frozen_reference(tmp_path)


def test_evidence_verification_requires_all_critical_artifact_entries(tmp_path):
    with pytest.raises(ValueError, match="Required evidence"):
        verify_artifacts(tmp_path, {}, ("inputs.json", "result.json", "fields.npz", "domain_masks.npz"))


def test_density_comparison_integrates_physical_overlap_between_resolutions():
    def fields(densities):
        density = np.asarray(densities, dtype=float).reshape(1, 1, -1)
        return {"density": density, "allowed": np.ones(density.shape, dtype=bool),
                "preserve": np.zeros(density.shape, dtype=bool)}

    coarse = fields([0, 1])
    assert field_comparison(coarse, fields([0, 0, 1, 1]))["rms_density_full_box"] == 0
    changed_quarter = field_comparison(coarse, fields([0, 1, 1, 1]))
    assert changed_quarter["common_subdivision_shape"] == [1, 1, 4]
    assert changed_quarter["rms_density_full_box"] == pytest.approx(0.5)
