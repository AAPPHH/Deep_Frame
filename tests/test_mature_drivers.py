"""Driver integration gates use small real STEP files and substituted expensive stages."""

from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys
from types import ModuleType

from build123d import Box, export_step
import numpy as np
import pytest
import trimesh

import deep_frame
from deep_frame.topology_surface import SurfaceReconstructionError
from tools import run_mature_geometry_study as geometry
from tools import run_mature_topology as runner


@pytest.fixture
def study(tmp_path, monkeypatch):
    historical = geometry.read(geometry.ROOT / "docs/validation/topology_phase1/inputs.json")
    source = tmp_path / "source"
    source.mkdir()
    shape = (2, 2, 2)
    masks = {"allowed": np.ones(shape, dtype=bool), "preserve": np.zeros(shape, dtype=bool),
             "forbidden": np.zeros(shape, dtype=bool)}
    domain = deepcopy(historical["domain"])
    domain["grid"] = {"shape": list(shape), "origin_mm": [0, 0, 0], "spacing_mm": [1, 1, 1]}
    inputs = {"parameters": historical["parameters"], "domain": domain, "settings": {"linear_solver": "cpu_superlu"},
              "mask_sha256": {name: hashlib.sha256(mask.tobytes()).hexdigest() for name, mask in masks.items()}}
    geometry.write(source / "inputs.json", inputs)
    geometry.write(source / "result.json", {"status": "ok"})
    np.savez_compressed(source / "fields.npz", density=np.full(shape, 0.5), **masks)
    np.savez_compressed(source / "domain_masks.npz", **masks)
    geometry.write(source / "manifest.json", {"status": "ok", "artifacts": geometry.artifacts(source)})
    reference = tmp_path / "reference.step"
    export_step(Box(9, 9, 9), reference)
    args = geometry.parse_args(["--source", str(source), "--output", str(tmp_path / "output"),
                                "--reference-step", str(reference), "--thresholds", "0.3"])
    calls = {"reconstruction": 0, "validation": 0, "fea": 0, "mesh": 0}
    validator = ModuleType("deep_frame.topology_surface_validation")
    validator._settings = lambda settings: {"tessellation_mm": 0.03, **settings}

    def validate(solid, domain, settings, **kwargs):
        calls["validation"] += 1
        return {"passed": True, "checks": {"surface_maturity": {"passed": True}}, "violations": []}

    def mesh(solid, settings):
        calls["mesh"] += 1
        vertices, faces = solid.tessellate(settings["tessellation_mm"])
        result = trimesh.Trimesh(vertices=[tuple(v) for v in vertices], faces=faces)
        result.metadata["adaptive_tessellation"] = {"used_deflection_mm": settings["tessellation_mm"],
                                                    "cad_geometry_modified": False}
        return result

    def reconstruct(domain, density, settings, *, progress):
        calls["reconstruction"] += 1
        solid = Box(8, 8, 8)
        solid.surface_report = {"passed": True}
        progress({"stage": "before_exact_constraints", "report": solid.surface_report, "shape": solid})
        return solid

    def evaluate(solid, material, masses, cases, settings):
        calls["fea"] += 1
        return {"status": "ok", "frame_mass_g": solid.volume * material["density_g_cm3"] / 1000,
                "stiffness_n_per_mm": 10.0, "max_displacement_mm": 0.1, "max_von_mises_mpa": 1.0,
                "eigenfrequencies_hz": [100.0], "load_cases": {case["name"]: {"analysis": case["analysis"],
                 **({"eigenfrequencies_hz": [100.0]} if case["analysis"] == "modal" else
                    {"max_displacement_mm": 0.1, "max_von_mises_mpa": 1.0})} for case in cases}}

    validator.validate_surface = validate
    validator._mesh = mesh
    monkeypatch.setattr(deep_frame, "topology_surface_validation", validator, raising=False)
    monkeypatch.setitem(sys.modules, "deep_frame.topology_surface_validation", validator)
    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(geometry, "build_geometry", lambda parameters: Box(10, 10, 10))
    monkeypatch.setattr(geometry, "evaluate", evaluate)
    monkeypatch.setattr(geometry, "_provenance", lambda *args: {})
    return args, source, validator, calls


def test_invalid_geometry_persists_validation_without_any_fea(study):
    args, source, validator, calls = study
    validator.validate_surface = lambda *a, **k: {"passed": False, "checks": {}, "violations": ["features"]}
    result = geometry.run_study(args)
    record = geometry.read(args.output / "candidates/t00/record.json")
    assert result["status"] == "complete"
    assert result["overall_acceptance"] is False
    assert result["accepted_count"] == 0 and result["selected_id"] is None
    assert record["status"] == "geometry_invalid"
    assert record["validation"]["violations"] == ["features"]
    assert calls["fea"] == 0
    assert (args.output / "candidates/t00/geometry.stl").is_file()
    runner.verify_geometry(args.output)


def test_final_mesh_error_never_hides_validator_rejection(study):
    args, _, validator, calls = study
    validator.validate_surface = lambda *a, **k: {"passed": False, "checks": {}, "violations": ["topology"]}

    def cannot_mesh(*args):
        raise ValueError("Missing CAD triangle coverage")

    validator._mesh = cannot_mesh
    geometry.run_study(args)
    record = geometry.read(args.output / "candidates/t00/record.json")
    assert record["status"] == "geometry_invalid"
    assert record["validation"]["violations"] == ["topology"]
    assert "coverage" in record["final_mesh"]["error"]
    assert calls["fea"] == 0


def test_missing_maturity_check_cannot_trigger_fea_or_acceptance(study):
    args, _, validator, calls = study
    validator.validate_surface = lambda *a, **k: {"passed": True, "checks": {}, "violations": []}
    result = geometry.run_study(args)
    assert result["accepted_count"] == 0 and not result["overall_acceptance"]
    assert calls["fea"] == 0


def test_source_hash_mismatch_stops_geometry_and_persists_failure(study):
    args, source, _, calls = study
    (source / "fields.npz").write_bytes(b"changed density")
    result = geometry.run_study(args)
    assert result["status"] == "failed" and "artifact mismatch" in result["error"]
    assert not result["overall_acceptance"] and result["selected_id"] is None
    assert calls == {"reconstruction": 0, "validation": 0, "fea": 0, "mesh": 0}
    assert geometry.read(args.output / "status.json")["status"] == "failed"


def test_failure_preserves_report_preunion_shape_and_mesh(study, monkeypatch):
    args, _, validator, calls = study
    report = {"before_exact_constraints": {"volume_mm3": 512}, "reason": "Boolean fuse rejected"}

    def fail(domain, density, settings, *, progress):
        solid = Box(8, 8, 8)
        mesh = validator._mesh(solid, validator._settings({}))
        progress({"stage": "before_exact_constraints", "report": report, "shape": solid, "mesh": mesh})
        error = SurfaceReconstructionError("Boolean fuse rejected", report, solid)
        error.mesh = mesh
        raise error

    monkeypatch.setattr(geometry, "reconstruct_surface", fail)
    result = geometry.run_study(args)
    record = geometry.read(args.output / "candidates/t00/record.json")
    assert result["status"] == "complete" and not result["overall_acceptance"]
    assert record["reconstruction"] == report and record["status"] == "failed"
    assert calls["fea"] == 0
    for name in ("failure/reconstruction.step", "failure/reconstruction.ply", "failure/reconstruction.json"):
        assert name in record["artifacts"]
        assert (args.output / "candidates/t00" / name).stat().st_size > 0
    assert list((args.output / "candidates/t00/intermediate").glob("*_before_exact_constraints.step"))


def test_complete_acceptance_only_after_last_candidate_with_final_cad_exports(study, monkeypatch):
    args, _, _, calls = study
    args.thresholds = [0.3, 0.4]
    observed = []
    original = geometry.reconstruct_surface

    def reconstruct(*a, **k):
        observed.append(geometry.read(args.output / "manifest.json"))
        return original(*a, **k)

    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    result = geometry.run_study(args)
    assert all(item["status"] == "running" and not item["overall_acceptance"] and item["selected_id"] is None for item in observed)
    assert observed[1]["accepted_count"] == 1
    assert result["status"] == "complete" and result["overall_acceptance"]
    assert result["accepted_count"] == 2 and result["selected_id"] == "t00"
    assert calls["fea"] == 3  # One baseline, then the two validated candidates.
    mesh_report = geometry.read(args.output / "candidates/t00/final_mesh.json")
    assert mesh_report["source_step_sha256"] == geometry.digest(args.output / "candidates/t00/geometry.step")
    assert mesh_report["tessellation"]["used_deflection_mm"] == 0.03
    runner.verify_geometry(args.output)


def test_geometry_only_does_not_claim_mechanical_acceptance(study):
    args, _, _, calls = study
    args.geometry_only = True
    result = geometry.run_study(args)
    assert result["status"] == "complete" and not result["overall_acceptance"]
    assert result["accepted_count"] == 0 and result["selected_id"] is None and calls["fea"] == 0


def test_changed_baseline_invalidates_stored_acceptance(study):
    args, _, _, _ = study
    result = geometry.run_study(args)
    assert result["overall_acceptance"]
    (args.output / "baseline/raw_fea_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="verified independent baseline"):
        runner.verify_geometry(args.output)


def test_unverified_claims_cannot_be_counted_as_accepted():
    incomplete = {"id": "t00", "status": "accepted", "comparison": {"passed": True, "selection_score": 1}}
    assert geometry.acceptance([incomplete], True) == {"accepted_count": 0, "selected_id": None, "overall_acceptance": False}


def test_orchestrator_saved_source_hash_mismatch_never_launches_subprocess(study, monkeypatch, tmp_path):
    args, source, _, _ = study
    (source / "fields.npz").write_bytes(b"corrupt")
    monkeypatch.setattr(runner, "execute", lambda *a: pytest.fail("No subprocess after source corruption"))
    output = tmp_path / "orchestrator"
    code = runner.main(["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step)])
    report = geometry.read(output / "run.json")
    assert code == 1 and report["status"] == "failed" and not report["overall_acceptance"]
    assert "artifact mismatch" in report["error"] and report["commands"]["density"] is None


@pytest.mark.parametrize("backend", ["cpu_superlu", "cuda_cudss"])
def test_orchestrator_fresh_density_selects_backend_and_verified_geometry(study, monkeypatch, tmp_path, backend):
    args, source, _, _ = study
    output = tmp_path / ("orchestrator_" + backend)
    commands = []

    def execute(command, cwd, logfile, timeout):
        commands.append(command)
        if "tools.run_topology_study" in command:
            shutil.copytree(source, output / "density")
            assert command[command.index("--linear-solver") + 1] == backend
            assert "--source" not in command
        else:
            assert geometry.main(command[2:]) == 0
        return {"returncode": 0}

    monkeypatch.setattr(runner, "execute", execute)
    code = runner.main(["--output", str(output), "--reference-step", str(args.reference_step),
                        "--density-backend", backend, "--thresholds", "0.3"])
    report = geometry.read(output / "run.json")
    assert code == 0 and report["status"] == "complete" and report["overall_acceptance"]
    assert report["source_mode"] == "fresh_uniform" and report["source_artifacts_verified"]
    assert report["accepted_count"] == 1 and report["selected_id"] == "t00" and len(commands) == 2


def test_orchestrator_does_not_trust_false_manifest_acceptance(study, monkeypatch, tmp_path):
    args, source, _, _ = study
    output = tmp_path / "orchestrator"

    def execute(command, cwd, logfile, timeout):
        folder = output / "geometry"
        folder.mkdir()
        geometry.write(folder / "manifest.json", {"status": "complete", "thresholds": [], "candidates": [],
                                                    "overall_acceptance": True, "accepted_count": 1, "selected_id": "fiction"})
        return {"returncode": 0}

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.main(["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step)]) == 1
    report = geometry.read(output / "run.json")
    assert not report["overall_acceptance"] and report["selected_id"] is None
    assert "acceptance summary" in report["error"]


def test_subprocess_timeout_retains_log_hash_and_explicit_budget(tmp_path):
    report = runner.execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, tmp_path / "run.log", 0.1)
    assert report["timed_out"] and report["returncode"] is None and report["timeout_s"] == 0.1
    assert report["log_sha256"] == geometry.digest(tmp_path / "run.log")
    assert report["process_tree_cleanup_returncode"] == 0


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
def test_reference_step_is_required(parse, tmp_path):
    with pytest.raises(SystemExit):
        parse(["--output", str(tmp_path)])


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "-inf"])
def test_manufacturing_opening_radius_rejects_negative_or_nonfinite_values(parse, tmp_path, value):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--manufacturing-opening-radius-mm=" + value])


def test_manufacturing_opening_radius_is_forwarded_to_reconstructor(study, monkeypatch, tmp_path):
    args, source, _, _ = study
    assert args.manufacturing_opening_radius_mm == 0.0
    original = geometry.reconstruct_surface
    seen = []

    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["manufacturing_opening_radius_mm"])
        return original(domain, density, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert command[command.index("--manufacturing-opening-radius-mm") + 1] == "1.0"
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "opening_study"
    assert runner.main(["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step),
                        "--thresholds", "0.3", "--manufacturing-opening-radius-mm", "1.0"]) == 0
    assert seen == [1.0]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["manufacturing_opening_radius_mm"] == 1.0


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nan", "inf"])
def test_wall_sample_budget_must_be_a_positive_integer(parse, tmp_path, value):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--maximum-wall-samples=" + value])


@pytest.mark.parametrize("requested", [None, 1_200_000])
def test_wall_sample_budget_reaches_validator_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, validator, _ = study
    expected = requested if requested is not None else 2_000_000
    assert args.maximum_wall_samples == 2_000_000
    original = validator.validate_surface
    seen = []

    def validate(solid, domain, settings, **kwargs):
        seen.append(settings["maximum_wall_samples"])
        return original(solid, domain, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert command[command.index("--maximum-wall-samples") + 1] == str(expected)
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(validator, "validate_surface", validate)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "wall_budget_study"
    command = ["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step), "--thresholds", "0.3"]
    if requested is not None:
        command.extend(["--maximum-wall-samples", str(requested)])
    assert runner.main(command) == 0
    assert seen == [expected]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["validation_settings"]["maximum_wall_samples"] == expected
    assert geometry.read(output / "run.json")["budgets"]["maximum_wall_samples"] == expected


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
def test_opening_method_rejects_unknown_values(parse, tmp_path):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--manufacturing-opening-method", "unknown"])


@pytest.mark.parametrize("requested", [None, "distance"])
def test_opening_method_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested or "grayscale"
    assert args.manufacturing_opening_method == "grayscale"
    original = geometry.reconstruct_surface
    seen = []

    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["manufacturing_opening_method"])
        return original(domain, density, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert command[command.index("--manufacturing-opening-method") + 1] == expected
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "opening_method_study"
    command = ["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step), "--thresholds", "0.3"]
    if requested is not None:
        command.extend(["--manufacturing-opening-method", requested])
    assert runner.main(command) == 0
    assert seen == [expected]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["manufacturing_opening_method"] == expected


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
@pytest.mark.parametrize("value", ["0", "-1e-7", "nan", "inf", "-inf"])
def test_boolean_fuzzy_tolerance_must_be_finite_and_positive(parse, tmp_path, value):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--boolean-fuzzy-mm=" + value])


@pytest.mark.parametrize("requested", [None, 1e-5])
def test_boolean_fuzzy_tolerance_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested if requested is not None else 1e-7
    assert args.boolean_fuzzy_mm == 1e-7
    original = geometry.reconstruct_surface
    seen = []

    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["boolean_fuzzy_value_mm"])
        return original(domain, density, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert float(command[command.index("--boolean-fuzzy-mm") + 1]) == expected
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "boolean_fuzzy_study"
    command = ["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step), "--thresholds", "0.3"]
    if requested is not None:
        command.extend(["--boolean-fuzzy-mm", str(requested)])
    assert runner.main(command) == 0
    assert seen == [expected]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["boolean_fuzzy_value_mm"] == expected
    wrapper = geometry.read(output / "run.json")
    assert wrapper["cad_boolean_settings"]["boolean_fuzzy_value_mm"] == expected


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
@pytest.mark.parametrize("value", ["0", "-1e-7", "nan", "inf", "-inf"])
def test_validation_boolean_fuzzy_tolerance_must_be_finite_and_positive(parse, tmp_path, value):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--validation-boolean-fuzzy-mm=" + value])


@pytest.mark.parametrize("requested", [None, 1e-5])
def test_validation_fuzzy_tolerance_is_forwarded_separately_and_persisted(study, monkeypatch, tmp_path, requested):
    args, source, validator, _ = study
    expected = requested if requested is not None else 1e-7
    assert args.validation_boolean_fuzzy_mm == 1e-7
    original = validator.validate_surface
    seen = []

    def validate(solid, domain, settings, **kwargs):
        seen.append(settings["boolean_fuzzy_mm"])
        assert "boolean_fuzzy_value_mm" not in settings
        return original(solid, domain, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert float(command[command.index("--validation-boolean-fuzzy-mm") + 1]) == expected
        assert float(command[command.index("--boolean-fuzzy-mm") + 1]) == 2e-6
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(validator, "validate_surface", validate)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "validation_boolean_fuzzy_study"
    command = ["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step),
               "--thresholds", "0.3", "--boolean-fuzzy-mm", "2e-6"]
    if requested is not None:
        command.extend(["--validation-boolean-fuzzy-mm", str(requested)])
    assert runner.main(command) == 0
    assert seen == [expected]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["validation_settings"]["boolean_fuzzy_mm"] == expected
    assert manifest["reconstruction_settings"]["boolean_fuzzy_value_mm"] == 2e-6
    wrapper = geometry.read(output / "run.json")
    assert wrapper["cad_boolean_settings"] == {"boolean_fuzzy_value_mm": 2e-6,
                                               "validator_boolean_fuzzy_value_mm": expected}


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
def test_surface_constraint_mode_rejects_unknown_values(parse, tmp_path):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--surface-constraint-mode", "unknown"])


@pytest.mark.parametrize("requested", [None, "cad_only", "envelope_only", "envelope_forbidden"])
def test_surface_constraint_mode_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested or "embedded"
    assert args.surface_constraint_mode == "embedded"
    original = geometry.reconstruct_surface
    seen = []

    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["surface_constraint_mode"])
        return original(domain, density, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert command[command.index("--surface-constraint-mode") + 1] == expected
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "surface_constraint_mode_study"
    command = ["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step), "--thresholds", "0.3"]
    if requested is not None:
        command.extend(["--surface-constraint-mode", requested])
    assert runner.main(command) == 0
    assert seen == [expected]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["surface_constraint_mode"] == expected
    assert geometry.read(output / "run.json")["surface_constraint_mode"] == expected


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
def test_decimation_bounds_mode_rejects_unknown_values(parse, tmp_path):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--decimation-bounds-mode", "unknown"])


@pytest.mark.parametrize("requested", [None, "reference_aabb"])
def test_decimation_bounds_mode_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested or "none"
    assert args.decimation_bounds_mode == "none"
    original = geometry.reconstruct_surface
    seen = []

    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["decimation_bounds_mode"])
        assert settings["surface_constraint_mode"] == "envelope_only"
        return original(domain, density, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert command[command.index("--decimation-bounds-mode") + 1] == expected
        assert command[command.index("--surface-constraint-mode") + 1] == "envelope_only"
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "decimation_bounds_mode_study"
    command = ["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step),
               "--thresholds", "0.3", "--surface-constraint-mode", "envelope_only"]
    if requested is not None:
        command.extend(["--decimation-bounds-mode", requested])
    assert runner.main(command) == 0
    assert seen == [expected]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["decimation_bounds_mode"] == expected
    assert geometry.read(output / "run.json")["decimation_bounds_mode"] == expected


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
def test_preserve_fusion_mode_rejects_unknown_values(parse, tmp_path):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--preserve-fusion-mode", "unknown"])


@pytest.mark.parametrize("requested", [None, "direct", "preunion"])
def test_preserve_fusion_mode_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested or "direct"
    assert args.preserve_fusion_mode == "direct"
    original = geometry.reconstruct_surface
    seen = []

    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["preserve_fusion_mode"])
        return original(domain, density, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert command[command.index("--preserve-fusion-mode") + 1] == expected
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "preserve_fusion_mode_study"
    command = ["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step),
               "--thresholds", "0.3"]
    if requested is not None:
        command.extend(["--preserve-fusion-mode", requested])
    assert runner.main(command) == 0
    assert seen == [expected]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["preserve_fusion_mode"] == expected
    wrapper = geometry.read(output / "run.json")
    assert wrapper["preserve_fusion_mode"] == expected
    stored_command = wrapper["commands"]["geometry"]
    assert stored_command[stored_command.index("--preserve-fusion-mode") + 1] == expected


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "-inf"])
def test_free_forbidden_buffer_rejects_negative_or_nonfinite_values(parse, tmp_path, value):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--free-forbidden-buffer-mm=" + value])


@pytest.mark.parametrize("parse", [geometry.parse_args, runner.parse_args])
@pytest.mark.parametrize("mode,radius", [("embedded", 1.25), ("cad_only", 1.25), ("envelope_only", 1.25), ("envelope_forbidden", 0)])
def test_positive_free_forbidden_buffer_requires_forbidden_opening(parse, tmp_path, mode, radius):
    with pytest.raises(SystemExit):
        parse(["--source", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
               "--reference-step", str(tmp_path / "reference.step"), "--free-forbidden-buffer-mm", "0.5",
               "--surface-constraint-mode", mode, "--manufacturing-opening-radius-mm", str(radius)])


@pytest.mark.parametrize("requested", [None, 0.0, 0.5])
def test_free_forbidden_buffer_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested if requested is not None else 0.0
    assert args.free_forbidden_buffer_mm == 0.0
    original = geometry.reconstruct_surface
    seen = []

    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["free_forbidden_buffer_mm"])
        if expected > 0:
            assert settings["surface_constraint_mode"] == "envelope_forbidden"
            assert settings["manufacturing_opening_radius_mm"] == 1.25
            assert settings["preserve_fusion_mode"] == "preunion"
        return original(domain, density, settings, **kwargs)

    def execute(command, cwd, logfile, timeout):
        assert command[command.index("--free-forbidden-buffer-mm") + 1] == str(expected)
        return {"returncode": geometry.main(command[2:])}

    monkeypatch.setattr(geometry, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "free_forbidden_buffer_study"
    command = ["--source", str(source), "--output", str(output), "--reference-step", str(args.reference_step),
               "--thresholds", "0.3"]
    if requested is not None:
        command.extend(["--free-forbidden-buffer-mm", str(requested)])
    if expected > 0:
        command.extend(["--surface-constraint-mode", "envelope_forbidden", "--manufacturing-opening-radius-mm", "1.25",
                        "--preserve-fusion-mode", "preunion"])
    assert runner.main(command) == 0
    assert seen == [expected]
    manifest = geometry.read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["free_forbidden_buffer_mm"] == expected
    wrapper = geometry.read(output / "run.json")
    assert wrapper["free_forbidden_buffer_mm"] == expected
    stored_command = wrapper["commands"]["geometry"]
    assert stored_command[stored_command.index("--free-forbidden-buffer-mm") + 1] == str(expected)


@pytest.mark.parametrize("module,method", [(geometry, "write"), (runner, "save")])
@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_atomic_writer_retries_temporary_windows_reader_lock(module, method, winerror, tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text('{"state":"old"}', encoding="utf-8")
    original = Path.replace
    attempts, sleeps = [], []
    error = PermissionError("temporary reader lock")
    error.winerror = winerror

    def replace(temporary, destination):
        attempts.append((temporary, destination))
        assert geometry.read(target) == {"state": "old"}
        assert geometry.read(temporary) == {"state": "new"}
        if len(attempts) < 3:
            raise error
        return original(temporary, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    getattr(module, method)(target, {"state": "new"})
    assert geometry.read(target) == {"state": "new"}
    assert len(attempts) == 3 and sleeps == [0.05, 0.05]
    assert not target.with_name(target.name + ".tmp").exists()


@pytest.mark.parametrize("module,method", [(geometry, "write"), (runner, "save")])
@pytest.mark.parametrize("winerror", [None, 87])
def test_atomic_writer_propagates_unrelated_error_immediately(module, method, winerror, tmp_path, monkeypatch):
    attempts, sleeps = [], []
    error = OSError("unrelated replacement failure")
    if winerror is not None:
        error.winerror = winerror

    def replace(*args):
        attempts.append(args)
        raise error

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    with pytest.raises(OSError) as caught:
        getattr(module, method)(tmp_path / "report.json", {"state": "new"})
    assert caught.value is error and len(attempts) == 1 and sleeps == []


@pytest.mark.parametrize("module,method", [(geometry, "write"), (runner, "save")])
def test_atomic_writer_exhausted_retry_budget_preserves_original_error_and_target(module, method, tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text('{"state":"old"}', encoding="utf-8")
    attempts, sleeps = [], []
    error = PermissionError("persistent reader lock")
    error.winerror = 32

    def replace(*args):
        attempts.append(args)
        raise error

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        getattr(module, method)(target, {"state": "new"})
    assert caught.value is error and len(attempts) == 21 and sleeps == [0.05]*20
    assert geometry.read(target) == {"state": "old"}
    assert geometry.read(target.with_name(target.name + ".tmp")) == {"state": "new"}
