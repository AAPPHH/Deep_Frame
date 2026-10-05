from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

from build123d import Box, export_step
import numpy as np
import pytest
import trimesh

import deep_frame
from deep_frame import topology_pipeline, topology_surface
from deep_frame.config import TOPOLOGY_CONFIG, command_line, configure
from deep_frame.topology_optimization import _settings
from deep_frame.topology_pipeline import _artifact, _file_digest, _read, _save
from deep_frame.topology_surface import SurfaceReconstructionError
from tests.test_topology_implicit_validation import mounted_domain
from tools import compute
from tools import implicit_study as implicit
from tools import mature_pipeline as pipeline
from tools import topology_study
from tools import neural_study
from tools import workstation_study as workstation
from tools.topology_study import field_comparison, plot_gpu, verify_artifacts, verify_comparable_settings, verify_frozen_reference

def test_coldstart_design_enforces_domain_without_saved_density(tmp_path):
    from tools import formulation_study as study
    allowed = np.array([True, False, True, True]).reshape(2, 2, 1)
    preserve = np.array([False, False, True, False]).reshape(2, 2, 1)
    cfg = study.configure({"request": None, "reference_density": None, "mma": {"initial_density": 0.4}})
    half = {"grid": {"shape": [2, 2, 1]}, "allowed": allowed, "preserve": preserve}
    np.testing.assert_array_equal(study.initial_design(cfg, half), [0.4, 0.0, 1.0, 0.4])
    cfg["mma"]["initial_density"] = -0.1
    with pytest.raises(ValueError, match="Initial density"):
        study.initial_design(cfg, half)

def test_coldstart_requires_explicit_layout():
    from tools import formulation_study as study
    with pytest.raises(ValueError, match="explicit layout"):
        study.patched_settings(study.configure({"request": None, "reference_density": None}))

def test_reference_run_disables_stand_and_landing_before_setup(monkeypatch, tmp_path):
    from tools import formulation_study as study
    monkeypatch.setattr(study.config, "STAND_STABILITY", {"enabled": True})
    monkeypatch.setattr(study.config, "LANDING", {"enabled": True})
    monkeypatch.setattr(study.config, "CABLE_WIDTH", {"enabled": True})
    half = {"grid": {"shape": [1, 1, 1]}, "allowed": np.ones((1, 1, 1), bool), "preserve": np.zeros((1, 1, 1), bool)}
    def setup(*args):
        assert study.config.STAND_STABILITY["enabled"] is False
        assert study.config.LANDING["enabled"] is False
        assert study.config.CABLE_WIDTH["enabled"] is False
        return half, {}
    monkeypatch.setattr(study, "frame_setup", setup)
    monkeypatch.setattr(study, "optimize_stage", lambda cfg, half, problem, design, *args: (design, {}))
    cfg = study.configure(json.loads((Path(__file__).parents[1] / "docs/validation/dgx/free_43_coldstart.json").read_text(encoding="utf-8")))
    cfg["mma"].update(root=str(tmp_path), memory_s=None, coarse=False, until="fine")
    study.frame_mma(cfg)

def test_comparison_rejects_changed_physical_filter_but_allows_iteration_budget():
    reference = _settings({})
    studied = deepcopy(reference)
    studied.update(max_iterations=300, max_runtime_s=1800, change_tolerance=0.005)
    verify_comparable_settings(reference, studied)
    studied["filter_radius_mm"] = 5.0
    with pytest.raises(ValueError, match="physics/settings"):
        verify_comparable_settings(reference, studied)

def test_comparison_fills_projection_defaults_of_older_references():
    legacy = {key: value for key, value in _settings({}).items() if key not in
              ("projection", "robust_delta", "beta_schedule", "beta_interval", "beta_minimum_iterations",
               "beta_change_tolerance", "move_limit_late", "move_limit_late_beta", "volume_target_relaxation", "objective_window", "gpu_solver_residency")}
    verify_comparable_settings(legacy, _settings({"max_iterations": 300}))
    with pytest.raises(ValueError, match="physics/settings"):
        verify_comparable_settings(legacy, _settings({"projection": "robust", "beta_schedule": [1, 2, 4]}))

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

@pytest.mark.parametrize("cpu_key,gpu_key", [("cpu_superlu_wall_s", "cuda_cudss_wall_s"), ("cpu_wall_s", "gpu_wall_s")])
def test_gpu_plot_reads_fresh_and_archived_benchmark_timings(tmp_path, cpu_key, gpu_key):
    run = tmp_path / "grid8over3_gpu1000"
    run.mkdir()
    history = [{"iteration": index, "objective": 1 / index, "maximum_design_change": 0.5 / index, "final_evaluation": False}
               for index in range(1, 26)]
    _save(run / "result.json", {"history": history + [{"iteration": 26, "final_evaluation": True}]})
    _save(run / "inputs.json", {})
    np.savez_compressed(run / "fields.npz", density=np.zeros(1))
    (run / "iterations.jsonl").write_text("{}\n", encoding="utf-8")
    artifacts = {path.name: {"sha256": _file_digest(path), "size_bytes": path.stat().st_size} for path in run.iterdir()}
    _save(run / "manifest.json", {"artifacts": artifacts})
    (tmp_path / "three_updates").mkdir()
    _save(tmp_path / "three_updates/comparison.json", {cpu_key: 12.0, gpu_key: 0.5, "speedup": 24.0})
    plot_gpu(tmp_path, tmp_path / "plot.png")
    assert _read(tmp_path / "plot.json")["image_sha256"] == _file_digest(tmp_path / "plot.png")

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
        workstation.load_source(tmp_path)

def test_candidate_resume_rejects_fea_disagreement_with_hashed_raw_result(tmp_path):
    result = small_result()
    _save(tmp_path / "raw_result.json", result)
    corrupted = deepcopy(result)
    corrupted["stiffness_n_per_mm"] = 999.0
    artifacts = {"raw_fea_result.json": _artifact(tmp_path / "raw_result.json", tmp_path)}
    _save(tmp_path / "record.json", {"status": "ok", "fea": corrupted, "artifacts": artifacts})
    manifest = {"candidates": {"test": _artifact(tmp_path / "record.json", tmp_path)}}
    with pytest.raises(ValueError, match="differs from the hashed raw result"):
        workstation.read_candidates(tmp_path, manifest, physical_inputs())

def test_candidate_resume_rechecks_modal_case_even_when_all_hashes_match(tmp_path):
    result = small_result()
    del result["load_cases"]["modes"]
    _save(tmp_path / "raw_result.json", result)
    artifacts = {"raw_fea_result.json": _artifact(tmp_path / "raw_result.json", tmp_path)}
    _save(tmp_path / "record.json", {"status": "ok", "fea": result, "checks": {"passed": True}, "artifacts": artifacts})
    manifest = {"candidates": {"test": _artifact(tmp_path / "record.json", tmp_path)}}
    with pytest.raises(ValueError, match="omitted or changed case modes"):
        workstation.read_candidates(tmp_path, manifest, physical_inputs())

def test_intact_own_baseline_resumes_without_running_solver_and_rejects_corruption(tmp_path, monkeypatch):
    result = small_result()
    _save(tmp_path / "baseline/raw_result.json", result)
    artifacts = {"raw_fea_result.json": _artifact(tmp_path / "baseline/raw_result.json", tmp_path)}
    record = {"result": result, "artifacts": artifacts, "settings": physical_inputs()["fea_settings"]}
    _save(tmp_path / "baseline/record.json", record)
    def unexpected_solver(*args, **kwargs):
        raise AssertionError("A verified existing baseline must not start another solver")
    monkeypatch.setattr(workstation, "evaluate", unexpected_solver)
    assert workstation.baseline_result(tmp_path, physical_inputs(), None) == record
    (tmp_path / "baseline/raw_result.json").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="raw artifact mismatch"):
        workstation.baseline_result(tmp_path, physical_inputs(), None)

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
    monkeypatch.setattr(workstation, "EVIDENCE", evidence)
    with pytest.raises(ValueError, match="SHA256/size"):
        workstation.mesh_prepare(output, 2)

def test_unowned_stale_geometry_is_not_adopted(tmp_path):
    geometry = tmp_path / "geometry/candidate"
    geometry.mkdir(parents=True)
    (geometry / "geometry.step").write_text("stale unrelated CAD", encoding="utf-8")
    with pytest.raises(ValueError, match="empty output directory"):
        workstation.mesh_prepare(tmp_path, 2)

def test_cached_ok_result_still_requires_all_verification_cases(tmp_path):
    result = deepcopy(_read(workstation.EVIDENCE / "candidate_fea.json"))
    del result["load_cases"]["modes"]
    result["artifacts"] = {}
    inputs = _read(workstation.EVIDENCE / "inputs.json")
    preparation = {"study_input_sha256": "same-input", "load_cases": inputs["domain"]["comparison_load_cases"]}
    record = {"study_input_sha256": "same-input", "status": "ok", "result": result, "artifacts": {}}
    with pytest.raises(ValueError, match="modes"):
        workstation.verify_record(record, tmp_path, preparation)

@pytest.fixture(autouse=True)
def isolated_run_log(tmp_path, monkeypatch):
    monkeypatch.setattr(topology_pipeline, "RUN_LOG", tmp_path / "default_run_log.jsonl")

CURRENT = {}

def current_inputs():
    if not CURRENT:
        from deep_frame.frame import reference_parameters
        parameters = reference_parameters()
        built = pipeline.build_design_domain(deepcopy(parameters))
        CURRENT.update(parameters=json.loads(json.dumps(parameters, default=str)), domain=json.loads(json.dumps({key: value for key, value in built.items() if not isinstance(value, np.ndarray)})))
    return deepcopy(CURRENT)

@pytest.fixture
def study(tmp_path, monkeypatch):
    historical = current_inputs()
    source = tmp_path / "source"
    source.mkdir()
    shape = (2, 2, 2)
    masks = {"allowed": np.ones(shape, dtype=bool), "preserve": np.zeros(shape, dtype=bool),
             "forbidden": np.zeros(shape, dtype=bool)}
    domain = deepcopy(historical["domain"])
    domain["grid"] = {"shape": list(shape), "origin_mm": [0, 0, 0], "spacing_mm": [1, 1, 1]}
    inputs = {"parameters": historical["parameters"], "domain": domain, "settings": {"linear_solver": "cpu_superlu"},
              "mask_sha256": {name: hashlib.sha256(mask.tobytes()).hexdigest() for name, mask in masks.items()}}
    pipeline.write(source / "inputs.json", inputs)
    pipeline.write(source / "result.json", {"status": "ok"})
    np.savez_compressed(source / "fields.npz", density=np.full(shape, 0.5), **masks)
    np.savez_compressed(source / "domain_masks.npz", **masks)
    pipeline.write(source / "manifest.json", {"status": "ok", "artifacts": pipeline.artifacts(source)})
    reference = tmp_path / "reference.step"
    export_step(Box(9, 9, 9), reference)
    args = pipeline.geometry_config({"source": str(source), "output": str(tmp_path / "output"),
                                     "reference_step": str(reference), "thresholds": [0.3], "run_log": str(tmp_path / "run_log.jsonl")})
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
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(pipeline, "build_geometry", lambda parameters: Box(10, 10, 10))
    monkeypatch.setattr(pipeline, "evaluate", evaluate)
    monkeypatch.setattr(pipeline, "_provenance", lambda *args: {})
    return args, source, validator, calls

def test_invalid_geometry_persists_validation_without_any_fea(study):
    args, source, validator, calls = study
    validator.validate_surface = lambda *a, **k: {"passed": False, "checks": {}, "violations": ["features"]}
    result = pipeline.run_study(args)
    record = _read(args["output"] / "candidates/t00/record.json")
    assert result["status"] == "complete"
    assert result["overall_acceptance"] is False
    assert result["accepted_count"] == 0 and result["selected_id"] is None
    assert record["status"] == "geometry_invalid"
    assert record["validation"]["violations"] == ["features"]
    assert calls["fea"] == 0
    assert (args["output"] / "candidates/t00/geometry.stl").is_file()
    pipeline.verify_geometry(args["output"])

def test_final_mesh_error_never_hides_validator_rejection(study):
    args, _, validator, calls = study
    validator.validate_surface = lambda *a, **k: {"passed": False, "checks": {}, "violations": ["topology"]}
    def cannot_mesh(*args):
        raise ValueError("Missing CAD triangle coverage")
    validator._mesh = cannot_mesh
    pipeline.run_study(args)
    record = _read(args["output"] / "candidates/t00/record.json")
    assert record["status"] == "geometry_invalid"
    assert record["validation"]["violations"] == ["topology"]
    assert "coverage" in record["final_mesh"]["error"]
    assert calls["fea"] == 0

def test_missing_maturity_check_cannot_trigger_fea_or_acceptance(study):
    args, _, validator, calls = study
    validator.validate_surface = lambda *a, **k: {"passed": True, "checks": {}, "violations": []}
    result = pipeline.run_study(args)
    assert result["accepted_count"] == 0 and not result["overall_acceptance"]
    assert calls["fea"] == 0

def test_source_hash_mismatch_stops_geometry_and_persists_failure(study):
    args, source, _, calls = study
    (source / "fields.npz").write_bytes(b"changed density")
    result = pipeline.run_study(args)
    assert result["status"] == "failed" and "artifact mismatch" in result["error"]
    assert not result["overall_acceptance"] and result["selected_id"] is None
    assert calls == {"reconstruction": 0, "validation": 0, "fea": 0, "mesh": 0}
    assert _read(args["output"] / "status.json")["status"] == "failed"

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
    monkeypatch.setattr(topology_surface, "reconstruct_surface", fail)
    result = pipeline.run_study(args)
    record = _read(args["output"] / "candidates/t00/record.json")
    assert result["status"] == "complete" and not result["overall_acceptance"]
    assert record["reconstruction"] == report and record["status"] == "failed"
    assert calls["fea"] == 0
    for name in ("failure/reconstruction.step", "failure/reconstruction.ply", "failure/reconstruction.json"):
        assert name in record["artifacts"]
        assert (args["output"] / "candidates/t00" / name).stat().st_size > 0
    assert list((args["output"] / "candidates/t00/intermediate").glob("*_before_exact_constraints.step"))

def test_complete_acceptance_only_after_last_candidate_with_final_cad_exports(study, monkeypatch):
    args, _, _, calls = study
    args["thresholds"] = [0.3, 0.4]
    observed = []
    original = topology_surface.reconstruct_surface
    def reconstruct(*a, **k):
        observed.append(_read(args["output"] / "manifest.json"))
        return original(*a, **k)
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    result = pipeline.run_study(args)
    assert all(item["status"] == "running" and not item["overall_acceptance"] and item["selected_id"] is None for item in observed)
    assert observed[1]["accepted_count"] == 1
    assert result["status"] == "complete" and result["overall_acceptance"]
    assert result["accepted_count"] == 2 and result["selected_id"] == "t00"
    assert calls["fea"] == 3
    mesh_report = _read(args["output"] / "candidates/t00/final_mesh.json")
    assert mesh_report["source_step_sha256"] == _file_digest(args["output"] / "candidates/t00/geometry.step")
    assert mesh_report["tessellation"]["used_deflection_mm"] == 0.03
    pipeline.verify_geometry(args["output"])

def test_geometry_only_does_not_claim_mechanical_acceptance(study):
    args, _, _, calls = study
    args["geometry_only"] = True
    result = pipeline.run_study(args)
    assert result["status"] == "complete" and not result["overall_acceptance"]
    assert result["accepted_count"] == 0 and result["selected_id"] is None and calls["fea"] == 0

def test_changed_baseline_invalidates_stored_acceptance(study):
    args, _, _, _ = study
    result = pipeline.run_study(args)
    assert result["overall_acceptance"]
    (args["output"] / "baseline/raw_fea_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="verified independent baseline"):
        pipeline.verify_geometry(args["output"])

def test_unverified_claims_cannot_be_counted_as_accepted():
    incomplete = {"id": "t00", "status": "accepted", "comparison": {"passed": True, "selection_score": 1}}
    assert pipeline.acceptance([incomplete], True) == {"accepted_count": 0, "selected_id": None, "overall_acceptance": False}

def test_orchestrator_saved_source_hash_mismatch_never_launches_subprocess(study, monkeypatch, tmp_path):
    args, source, _, _ = study
    (source / "fields.npz").write_bytes(b"corrupt")
    monkeypatch.setattr(pipeline, "execute", lambda *a: pytest.fail("No subprocess after source corruption"))
    output = tmp_path / "orchestrator"
    code = pipeline.run_main({"source": str(source), "output": str(output), "reference_step": str(args["reference_step"])})
    report = _read(output / "run.json")
    assert code == 1 and report["status"] == "failed" and not report["overall_acceptance"]
    assert "artifact mismatch" in report["error"] and report["commands"]["density"] is None
    assert report["schema_version"] == "deep-frame-mature-end-to-end-v3" and report["configs"]["density"] is None and not (output / "density_config.json").exists()

@pytest.mark.parametrize("backend", ["cpu_superlu", "cuda_cudss"])
def test_orchestrator_fresh_density_selects_backend_and_verified_geometry(study, monkeypatch, tmp_path, backend):
    args, source, _, _ = study
    output = tmp_path / ("orchestrator_" + backend)
    commands = []
    def execute(command, cwd, logfile, timeout):
        commands.append(command)
        if "tools.topology_study" in command:
            shutil.copytree(source, output / "density")
            density = _read(command[-1])
            assert density["linear_solver"] == backend
            assert "source" not in density
        else:
            assert pipeline.main(command[2:]) == 0
        return {"returncode": 0}
    monkeypatch.setattr(pipeline, "execute", execute)
    code = pipeline.run_main({"output": str(output), "reference_step": str(args["reference_step"]),
                              "density_backend": backend, "thresholds": [0.3]})
    report = _read(output / "run.json")
    assert code == 0 and report["status"] == "complete" and report["overall_acceptance"]
    assert report["source_mode"] == "fresh_uniform" and report["source_artifacts_verified"]
    assert report["accepted_count"] == 1 and report["selected_id"] == "t00" and len(commands) == 2
    assert report["schema_version"] == "deep-frame-mature-end-to-end-v3"
    assert all(report["configs"][name]["sha256"] == _file_digest(output / (name + "_config.json")) for name in ("density", "geometry"))

def test_orchestrator_routes_ledger_to_both_stages(study, monkeypatch, tmp_path):
    args, source, _, _ = study
    output, ledger_path = tmp_path / "orchestrator", tmp_path / "routed" / "run_log.jsonl"
    def execute(command, cwd, logfile, timeout):
        if "tools.topology_study" in command:
            shutil.copytree(source, output / "density")
            assert _read(command[-1])["run_log"] == str(ledger_path.resolve())
            topology_pipeline.log_run(_read(command[-1])["run_log"], {"kind": "density", "run_dir": str(output / "density"), "success": True, "status": "ok"})
        else:
            assert _read(command[-1])["run_log"] == str(ledger_path.resolve()) and pipeline.main(command[2:]) == 0
        return {"returncode": 0}
    monkeypatch.setattr(pipeline, "execute", execute)
    assert pipeline.run_main({"output": str(output), "reference_step": str(args["reference_step"]), "thresholds": [0.3], "run_log": str(ledger_path)}) == 0
    assert [(line["kind"], line.get("candidate")) for line in ledger(ledger_path)] == [("density", None), ("cad", "t00")]
    assert _read(output / "run.json")["run_log"] == str(ledger_path.resolve()) and not (tmp_path / "default_run_log.jsonl").exists()

@pytest.mark.parametrize("child_logged", [False, True])
def test_orchestrator_logs_killed_density_stage_once(study, monkeypatch, tmp_path, child_logged):
    args, _, _, _ = study
    output, ledger_path = tmp_path / "orchestrator", tmp_path / "run_log.jsonl"
    def execute(command, cwd, logfile, timeout):
        assert "tools.topology_study" in command
        if child_logged:
            topology_pipeline.log_run(ledger_path, {"kind": "density", "run_dir": str(output / "density"), "success": False, "status": "failed"})
        return {"returncode": None, "timed_out": True, "runtime_s": 5.0}
    monkeypatch.setattr(pipeline, "execute", execute)
    assert pipeline.run_main({"output": str(output), "reference_step": str(args["reference_step"]), "run_log": str(ledger_path)}) == 1
    line, = ledger(ledger_path)
    assert line["kind"] == "density" and not line["success"] and line["run_dir"] == str(output / "density")
    assert line["status"] == ("failed" if child_logged else "timed_out")
    assert child_logged or (line["failure_stage"] == "optimization" and line["runtime_s"] >= 5.0 and line["gpu_pool_mb"] is None)

def test_orchestrator_does_not_trust_false_manifest_acceptance(study, monkeypatch, tmp_path):
    args, source, _, _ = study
    output = tmp_path / "orchestrator"
    def execute(command, cwd, logfile, timeout):
        folder = output / "geometry"
        folder.mkdir()
        pipeline.write(folder / "manifest.json", {"status": "complete", "thresholds": [], "candidates": [],
                                                    "overall_acceptance": True, "accepted_count": 1, "selected_id": "fiction"})
        return {"returncode": 0}
    monkeypatch.setattr(pipeline, "execute", execute)
    assert pipeline.run_main({"source": str(source), "output": str(output), "reference_step": str(args["reference_step"])}) == 1
    report = _read(output / "run.json")
    assert not report["overall_acceptance"] and report["selected_id"] is None
    assert "acceptance summary" in report["error"]

def test_subprocess_timeout_retains_log_hash_and_explicit_budget(tmp_path):
    report = pipeline.execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, tmp_path / "run.log", 0.1)
    assert report["timed_out"] and report["returncode"] is None and report["timeout_s"] == 0.1
    assert report["log_sha256"] == _file_digest(tmp_path / "run.log")
    assert report["process_tree_cleanup_returncode"] == 0

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
def test_reference_step_is_required(parse, tmp_path):
    with pytest.raises(ValueError):
        parse({"output": str(tmp_path)})

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "-inf"])
def test_manufacturing_opening_radius_rejects_negative_or_nonfinite_values(parse, tmp_path, value):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "manufacturing_opening_radius_mm": value})

def test_manufacturing_opening_radius_is_forwarded_to_reconstructor(study, monkeypatch, tmp_path):
    args, source, _, _ = study
    assert args["manufacturing_opening_radius_mm"] == 0.0
    original = topology_surface.reconstruct_surface
    seen = []
    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["manufacturing_opening_radius_mm"])
        return original(domain, density, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        assert str(_read(command[-1])["manufacturing_opening_radius_mm"]) == "1.0"
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "opening_study"
    assert pipeline.run_main({"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]),
                              "thresholds": [0.3], "manufacturing_opening_radius_mm": 1.0}) == 0
    assert seen == [1.0]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["manufacturing_opening_radius_mm"] == 1.0

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nan", "inf"])
def test_wall_sample_budget_must_be_a_positive_integer(parse, tmp_path, value):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "maximum_wall_samples": value})

@pytest.mark.parametrize("requested", [None, 1_200_000])
def test_wall_sample_budget_reaches_validator_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, validator, _ = study
    expected = requested if requested is not None else 2_000_000
    assert args["maximum_wall_samples"] == 2_000_000
    original = validator.validate_surface
    seen = []
    def validate(solid, domain, settings, **kwargs):
        seen.append(settings["maximum_wall_samples"])
        return original(solid, domain, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        assert str(_read(command[-1])["maximum_wall_samples"]) == str(expected)
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(validator, "validate_surface", validate)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "wall_budget_study"
    overrides = {"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]), "thresholds": [0.3]}
    if requested is not None:
        overrides["maximum_wall_samples"] = requested
    assert pipeline.run_main(overrides) == 0
    assert seen == [expected]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["validation_settings"]["maximum_wall_samples"] == expected
    assert _read(output / "run.json")["budgets"]["maximum_wall_samples"] == expected

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
def test_opening_method_rejects_unknown_values(parse, tmp_path):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "manufacturing_opening_method": "unknown"})

@pytest.mark.parametrize("requested", [None, "distance"])
def test_opening_method_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested or "grayscale"
    assert args["manufacturing_opening_method"] == "grayscale"
    original = topology_surface.reconstruct_surface
    seen = []
    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["manufacturing_opening_method"])
        return original(domain, density, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        assert _read(command[-1])["manufacturing_opening_method"] == expected
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "opening_method_study"
    overrides = {"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]), "thresholds": [0.3]}
    if requested is not None:
        overrides["manufacturing_opening_method"] = requested
    assert pipeline.run_main(overrides) == 0
    assert seen == [expected]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["manufacturing_opening_method"] == expected

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
@pytest.mark.parametrize("value", ["0", "-1e-7", "nan", "inf", "-inf"])
def test_boolean_fuzzy_tolerance_must_be_finite_and_positive(parse, tmp_path, value):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "boolean_fuzzy_mm": value})

@pytest.mark.parametrize("requested", [None, 1e-5])
def test_boolean_fuzzy_tolerance_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested if requested is not None else 1e-7
    assert args["boolean_fuzzy_mm"] == 1e-7
    original = topology_surface.reconstruct_surface
    seen = []
    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["boolean_fuzzy_value_mm"])
        return original(domain, density, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        assert float(_read(command[-1])["boolean_fuzzy_mm"]) == expected
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "boolean_fuzzy_study"
    overrides = {"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]), "thresholds": [0.3]}
    if requested is not None:
        overrides["boolean_fuzzy_mm"] = requested
    assert pipeline.run_main(overrides) == 0
    assert seen == [expected]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["boolean_fuzzy_value_mm"] == expected
    wrapper = _read(output / "run.json")
    assert wrapper["cad_boolean_settings"]["boolean_fuzzy_value_mm"] == expected

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
@pytest.mark.parametrize("value", ["0", "-1e-7", "nan", "inf", "-inf"])
def test_validation_boolean_fuzzy_tolerance_must_be_finite_and_positive(parse, tmp_path, value):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "validation_boolean_fuzzy_mm": value})

@pytest.mark.parametrize("requested", [None, 1e-5])
def test_validation_fuzzy_tolerance_is_forwarded_separately_and_persisted(study, monkeypatch, tmp_path, requested):
    args, source, validator, _ = study
    expected = requested if requested is not None else 1e-7
    assert args["validation_boolean_fuzzy_mm"] == 1e-7
    original = validator.validate_surface
    seen = []
    def validate(solid, domain, settings, **kwargs):
        seen.append(settings["boolean_fuzzy_mm"])
        assert "boolean_fuzzy_value_mm" not in settings
        return original(solid, domain, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        geometry = _read(command[-1])
        assert float(geometry["validation_boolean_fuzzy_mm"]) == expected
        assert float(geometry["boolean_fuzzy_mm"]) == 2e-6
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(validator, "validate_surface", validate)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "validation_boolean_fuzzy_study"
    overrides = {"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]),
               "thresholds": [0.3], "boolean_fuzzy_mm": 2e-6}
    if requested is not None:
        overrides["validation_boolean_fuzzy_mm"] = requested
    assert pipeline.run_main(overrides) == 0
    assert seen == [expected]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["validation_settings"]["boolean_fuzzy_mm"] == expected
    assert manifest["reconstruction_settings"]["boolean_fuzzy_value_mm"] == 2e-6
    wrapper = _read(output / "run.json")
    assert wrapper["cad_boolean_settings"] == {"boolean_fuzzy_value_mm": 2e-6,
                                               "validator_boolean_fuzzy_value_mm": expected}

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
def test_surface_constraint_mode_rejects_unknown_values(parse, tmp_path):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "surface_constraint_mode": "unknown"})

@pytest.mark.parametrize("requested", [None, "cad_only", "envelope_only", "envelope_forbidden"])
def test_surface_constraint_mode_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested or "embedded"
    assert args["surface_constraint_mode"] == "embedded"
    original = topology_surface.reconstruct_surface
    seen = []
    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["surface_constraint_mode"])
        return original(domain, density, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        assert _read(command[-1])["surface_constraint_mode"] == expected
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "surface_constraint_mode_study"
    overrides = {"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]), "thresholds": [0.3]}
    if requested is not None:
        overrides["surface_constraint_mode"] = requested
    assert pipeline.run_main(overrides) == 0
    assert seen == [expected]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["surface_constraint_mode"] == expected
    assert _read(output / "run.json")["surface_constraint_mode"] == expected

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
def test_decimation_bounds_mode_rejects_unknown_values(parse, tmp_path):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "decimation_bounds_mode": "unknown"})

@pytest.mark.parametrize("requested", [None, "reference_aabb"])
def test_decimation_bounds_mode_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested or "none"
    assert args["decimation_bounds_mode"] == "none"
    original = topology_surface.reconstruct_surface
    seen = []
    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["decimation_bounds_mode"])
        assert settings["surface_constraint_mode"] == "envelope_only"
        return original(domain, density, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        geometry = _read(command[-1])
        assert geometry["decimation_bounds_mode"] == expected
        assert geometry["surface_constraint_mode"] == "envelope_only"
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "decimation_bounds_mode_study"
    overrides = {"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]),
               "thresholds": [0.3], "surface_constraint_mode": "envelope_only"}
    if requested is not None:
        overrides["decimation_bounds_mode"] = requested
    assert pipeline.run_main(overrides) == 0
    assert seen == [expected]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["decimation_bounds_mode"] == expected
    assert _read(output / "run.json")["decimation_bounds_mode"] == expected

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
def test_preserve_fusion_mode_rejects_unknown_values(parse, tmp_path):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "preserve_fusion_mode": "unknown"})

@pytest.mark.parametrize("requested", [None, "direct", "preunion"])
def test_preserve_fusion_mode_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested or "direct"
    assert args["preserve_fusion_mode"] == "direct"
    original = topology_surface.reconstruct_surface
    seen = []
    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["preserve_fusion_mode"])
        return original(domain, density, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        assert _read(command[-1])["preserve_fusion_mode"] == expected
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "preserve_fusion_mode_study"
    overrides = {"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]), "thresholds": [0.3]}
    if requested is not None:
        overrides["preserve_fusion_mode"] = requested
    assert pipeline.run_main(overrides) == 0
    assert seen == [expected]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["preserve_fusion_mode"] == expected
    wrapper = _read(output / "run.json")
    assert wrapper["preserve_fusion_mode"] == expected
    stored = wrapper["configs"]["geometry"]
    assert wrapper["commands"]["geometry"][-1] == stored["path"] and stored["sha256"] == _file_digest(stored["path"])
    assert stored["values"]["preserve_fusion_mode"] == expected

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "-inf"])
def test_free_forbidden_buffer_rejects_negative_or_nonfinite_values(parse, tmp_path, value):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "free_forbidden_buffer_mm": value})

@pytest.mark.parametrize("parse", [pipeline.geometry_config, pipeline.run_config])
@pytest.mark.parametrize("mode,radius", [("embedded", 1.25), ("cad_only", 1.25), ("envelope_only", 1.25), ("envelope_forbidden", 0)])
def test_positive_free_forbidden_buffer_requires_forbidden_opening(parse, tmp_path, mode, radius):
    with pytest.raises(ValueError):
        parse({"source": str(tmp_path / "source"), "output": str(tmp_path / "output"),
               "reference_step": str(tmp_path / "reference.step"), "free_forbidden_buffer_mm": "0.5",
               "surface_constraint_mode": mode, "manufacturing_opening_radius_mm": str(radius)})

@pytest.mark.parametrize("requested", [None, 0.0, 0.5])
def test_free_forbidden_buffer_reaches_reconstructor_and_is_persisted(study, monkeypatch, tmp_path, requested):
    args, source, _, _ = study
    expected = requested if requested is not None else 0.0
    assert args["free_forbidden_buffer_mm"] == 0.0
    original = topology_surface.reconstruct_surface
    seen = []
    def reconstruct(domain, density, settings, **kwargs):
        seen.append(settings["free_forbidden_buffer_mm"])
        if expected > 0:
            assert settings["surface_constraint_mode"] == "envelope_forbidden"
            assert settings["manufacturing_opening_radius_mm"] == 1.25
            assert settings["preserve_fusion_mode"] == "preunion"
        return original(domain, density, settings, **kwargs)
    def execute(command, cwd, logfile, timeout):
        assert str(_read(command[-1])["free_forbidden_buffer_mm"]) == str(expected)
        return {"returncode": pipeline.main(command[2:])}
    monkeypatch.setattr(topology_surface, "reconstruct_surface", reconstruct)
    monkeypatch.setattr(pipeline, "execute", execute)
    output = tmp_path / "free_forbidden_buffer_study"
    overrides = {"source": str(source), "output": str(output), "reference_step": str(args["reference_step"]), "thresholds": [0.3]}
    if requested is not None:
        overrides["free_forbidden_buffer_mm"] = requested
    if expected > 0:
        overrides.update(surface_constraint_mode="envelope_forbidden", manufacturing_opening_radius_mm=1.25,
                        preserve_fusion_mode="preunion")
    assert pipeline.run_main(overrides) == 0
    assert seen == [expected]
    manifest = _read(output / "geometry/manifest.json")
    assert manifest["reconstruction_settings"]["free_forbidden_buffer_mm"] == expected
    wrapper = _read(output / "run.json")
    assert wrapper["free_forbidden_buffer_mm"] == expected
    stored = wrapper["configs"]["geometry"]
    assert wrapper["commands"]["geometry"][-1] == stored["path"] and stored["sha256"] == _file_digest(stored["path"])
    assert str(stored["values"]["free_forbidden_buffer_mm"]) == str(expected)

@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_atomic_writer_retries_temporary_windows_reader_lock(winerror, tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text('{"state":"old"}', encoding="utf-8")
    original = Path.replace
    attempts, sleeps = [], []
    error = PermissionError("temporary reader lock")
    error.winerror = winerror
    def replace(temporary, destination):
        attempts.append((temporary, destination))
        assert _read(target) == {"state": "old"}
        assert _read(temporary) == {"state": "new"}
        if len(attempts) < 3:
            raise error
        return original(temporary, destination)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    pipeline.write(target, {"state": "new"})
    assert _read(target) == {"state": "new"}
    assert len(attempts) == 3 and sleeps == [0.05, 0.05]
    assert not target.with_name(target.name + ".tmp").exists()

@pytest.mark.parametrize("winerror", [None, 87])
def test_atomic_writer_propagates_unrelated_error_immediately(winerror, tmp_path, monkeypatch):
    attempts, sleeps = [], []
    error = OSError("unrelated replacement failure")
    if winerror is not None:
        error.winerror = winerror
    def replace(*args):
        attempts.append(args)
        raise error
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    with pytest.raises(OSError) as caught:
        pipeline.write(tmp_path / "report.json", {"state": "new"})
    assert caught.value is error and len(attempts) == 1 and sleeps == []

def test_atomic_writer_exhausted_retry_budget_preserves_original_error_and_target(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text('{"state":"old"}', encoding="utf-8")
    attempts, sleeps = [], []
    error = PermissionError("persistent reader lock")
    error.winerror = 32
    def replace(*args):
        attempts.append(args)
        raise error
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        pipeline.write(target, {"state": "new"})
    assert caught.value is error and len(attempts) == 21 and sleeps == [0.05]*20
    assert _read(target) == {"state": "old"}
    assert _read(target.with_name(target.name + ".tmp")) == {"state": "new"}

def ledger(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]

def test_cad_study_logs_one_ledger_line_per_candidate(study):
    args, _, _, _ = study
    args["thresholds"] = [0.3, 0.4]
    result = pipeline.run_study(args)
    lines = ledger(args["run_log"])
    assert [line["candidate"] for line in lines] == ["t00", "t01"] and all(line["kind"] == "cad" and line["success"] and line["failure_stage"] is None for line in lines)
    assert all(line["source_sha256"] == result["source_manifest_sha256"] and line["runtime_s"] > 0 and "git_commit" in line for line in lines)

def test_cad_study_ledger_names_the_failing_stage(study, monkeypatch):
    args, _, _, _ = study
    def fail(*a, **k):
        raise RuntimeError("reconstruction crashed")
    monkeypatch.setattr(topology_surface, "reconstruct_surface", fail)
    pipeline.run_study(args)
    line, = ledger(args["run_log"])
    assert not line["success"] and line["status"] == "failed" and line["failure_stage"] == "reconstructing"

def test_density_study_logs_its_run(tmp_path, monkeypatch):
    def optimize(domain, settings, progress_callback):
        return {"status": "ok", "density": np.full(domain["allowed"].shape, 0.5), "summary": {"iterations": 3}, "diagnostics": []}
    monkeypatch.setattr(topology_study, "optimize_topology", optimize)
    monkeypatch.setitem(topology_study.TOPOLOGY_CONFIG, "camera_support", "prescribed")
    assert topology_study.run_main({"directory": str(tmp_path / "density"), "max_iterations": 3, "run_log": str(tmp_path / "run_log.jsonl")}) == 0
    line, = ledger(tmp_path / "run_log.jsonl")
    assert line["kind"] == "density" and line["success"] and line["iterations"] == 3 and line["run_dir"] == str((tmp_path / "density").resolve())

def test_raster_study_logs_one_ledger_line_per_candidate(tmp_path, monkeypatch):
    inputs = {"input_sha256": "raster-input", "baseline_reference_frame_mass_g": 1.0, "thresholds": [0.1, 0.2], "reconstruction_settings": {}, "parameters": {}, "relative_constraints": {"frame_mass_ratio_max": 2.0}}
    monkeypatch.setattr(workstation, "prepare", lambda *a: (inputs, {"material": {"density_g_cm3": 1.0}}, np.zeros(1)))
    monkeypatch.setattr(workstation, "build_geometry", lambda parameters: SimpleNamespace(volume=1000.0))
    def reconstruct(*a):
        raise ValueError("no solid")
    monkeypatch.setattr(workstation, "reconstruct_topology", reconstruct)
    config = {"source": str(tmp_path / "source"), "output": str(tmp_path / "raster"), "geometry_only": True, "run_log": str(tmp_path / "run_log.jsonl")}
    workstation.candidates_main(config)
    lines = ledger(tmp_path / "run_log.jsonl")
    assert [line["candidate"] for line in lines] == ["density_t00", "density_t01"] and [line["parameters"]["threshold"] for line in lines] == [0.1, 0.2]
    assert all(line["kind"] == "raster" and not line["success"] and line["failure_stage"] == "geometry" and line["source_sha256"] == "raster-input" and line["runtime_s"] >= 0 for line in lines)
    workstation.candidates_main(config)
    assert len(ledger(tmp_path / "run_log.jsonl")) == 2

def fake_fea(calls):
    def evaluate(solid, material, masses, cases, settings):
        calls.append(settings["work_dir"])
        return {"status": "ok", "frame_mass_g": solid.volume * material["density_g_cm3"] / 1000, "stiffness_n_per_mm": 10.0, "max_displacement_mm": 0.1, "max_von_mises_mpa": 1.0,
                "eigenfrequencies_hz": [100.0], "mesh": {"attempts": [{"name": "remesh_hxt", "status": "ok", "runtime_s": 0.5}]},
                "load_cases": {case["name"]: {"analysis": case["analysis"], **({"eigenfrequencies_hz": [100.0]} if case["analysis"] == "modal" else {"max_displacement_mm": 0.1, "max_von_mises_mpa": 1.0})} for case in cases}}
    return evaluate

def implicit_source(tmp_path, connected=True):
    historical = current_inputs()
    domain, density = mounted_domain()
    if not connected:
        density[28:40] = 0.0
    masks = {name: domain.pop(name) for name in ("allowed", "preserve", "forbidden")}
    domain.update({name: historical["domain"][name] for name in ("material", "point_masses", "comparison_load_cases", "fea_settings")})
    source = tmp_path / "source"
    source.mkdir()
    pipeline.write(source / "inputs.json", {"parameters": historical["parameters"], "domain": domain, "settings": {"linear_solver": "cpu_superlu"},
                                            "mask_sha256": {name: hashlib.sha256(mask.tobytes()).hexdigest() for name, mask in masks.items()}})
    pipeline.write(source / "result.json", {"status": "ok"})
    np.savez_compressed(source / "fields.npz", density=density, **masks)
    np.savez_compressed(source / "domain_masks.npz", **masks)
    pipeline.write(source / "manifest.json", {"status": "ok", "artifacts": pipeline.artifacts(source)})
    return source

def implicit_config(tmp_path, source, **changes):
    return {"source": str(source), "output": str(tmp_path / "study"), "reference_step": str(tmp_path / "reference.step"), "run_log": str(tmp_path / "run_log.jsonl"),
            "subdivisions": 4, "thresholds": [0.35], "extensions": ["preserve"], "curvature_samples": 500, **changes}

def test_implicit_study_accepts_connected_mounts_and_logs_success(tmp_path, monkeypatch):
    source, calls = implicit_source(tmp_path), []
    reference = {"source": "reference.step", "sha256": "0" * 64, "volume_mm3": 729.0, "curvature": None,
                 "metrics": {"free_sharp_edge_length_per_area_per_mm": 1e3, "free_axis_normal_area_fraction": 1e3}}
    monkeypatch.setattr(implicit, "reference_geometry", lambda path, domain, config: (trimesh.creation.box(extents=(9, 9, 9)), reference))
    monkeypatch.setattr(implicit, "build_geometry", lambda parameters: Box(20, 20, 20))
    monkeypatch.setattr(implicit, "evaluate", fake_fea(calls))
    monkeypatch.setattr(implicit, "_provenance", lambda *args: {})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(implicit_config(tmp_path, source, section_heights_mm=[7.0, 27.0])), encoding="utf-8")
    assert implicit.main(["run", str(path)]) == 0
    output = tmp_path / "study"
    manifest = _read(output / "manifest.json")
    assert manifest["status"] == "complete" and manifest["overall_acceptance"] and manifest["selected_id"] == "c00"
    assert manifest["success_rate"] == 1.0 and manifest["geometry_success_rate"] == 1.0 and manifest["runtime_per_candidate_s"]["max"] > 0
    assert manifest["candidates"][0]["record_sha256"] == _file_digest(output / "candidates/c00/record.json")
    assert manifest["domain_currency"]["regions_current"] is False
    record = _read(output / "candidates/c00/record.json")
    assert record["status"] == "accepted" and record["success"] and record["validation"]["passed"] and record["comparison"]["passed"]
    assert {"extension", "extraction", "remesh", "booleans", "export", "validation", "metrics", "renders", "fea", "fea_mesh", "fea_solve"} <= set(record["timings_s"])
    assert record["parameters"]["h_mm"] == pytest.approx(0.25) and record["curvature"]["sample_count"] > 0 and record["surface_metrics"]["triangle_count"] > 0
    assert record["final_mesh"]["files"]["geometry.stl"]["sha256"] == _file_digest(output / "candidates/c00/geometry.stl")
    assert sorted(record["renders"]["images"]) == ["section_z27", "section_z7", "view_front", "view_isometric", "view_side", "view_top"]
    assert all(_file_digest(output / "candidates/c00/renders" / image["path"]) == image["sha256"] for image in record["renders"]["images"].values())
    assert record["renders"]["sections"][0]["loops"] > 0 and record["renders"]["sections"][1]["loops"] == 0
    assert len(calls) == 2 and _read(output / "baseline/record.json")["status"] == "ok"
    line, = ledger(tmp_path / "run_log.jsonl")
    assert line["kind"] == "implicit" and line["candidate"] == "c00" and line["success"] and line["failure_stage"] is None
    assert line["source_sha256"] == manifest["source_manifest_sha256"] and line["timings_s"]["renders"] > 0 and line["parameters"]["extension"] == "preserve"

def test_implicit_study_rejects_disconnected_mount_and_summarizes_ledger(tmp_path, monkeypatch):
    source = implicit_source(tmp_path, connected=False)
    export_step(Box(9, 9, 9), tmp_path / "reference.step")
    monkeypatch.setattr(implicit, "build_geometry", lambda parameters: Box(20, 20, 20))
    monkeypatch.setattr(implicit, "evaluate", lambda *a: pytest.fail("No FEA for a rejected candidate"))
    monkeypatch.setattr(implicit, "_provenance", lambda *args: {})
    assert implicit.run_main(implicit_config(tmp_path, source, geometry_only=True)) == 0
    output = tmp_path / "study"
    manifest = _read(output / "manifest.json")
    assert manifest["status"] == "complete" and not manifest["overall_acceptance"] and manifest["success_rate"] == 0.0 and manifest["geometry_success_rate"] == 0.0
    cached = _read(output / "reference_metrics.json")
    assert cached["sha256"] == _file_digest(tmp_path / "reference.step") and cached["metrics"]["free_axis_normal_area_fraction"] == pytest.approx(1.0)
    assert manifest["comparison_reference"]["sha256"] == cached["sha256"]
    record = _read(output / "candidates/c00/record.json")
    assert record["status"] == "mount_disconnected" and record["failure_stage"] == "field" and not (output / "candidates/c00/geometry.stl").exists()
    line, = ledger(tmp_path / "run_log.jsonl")
    assert not line["success"] and line["status"] == "mount_disconnected" and line["failure_stage"] == "field"
    assert implicit.summarize_main({"run_log": str(tmp_path / "run_log.jsonl"), "output": str(tmp_path / "summary.json")}) == 0
    run, = _read(tmp_path / "summary.json")["runs"]
    assert run["kind"] == "implicit" and run["candidates"] == 1 and run["success_rate"] == 0.0 and run["statuses"] == {"mount_disconnected": 1}

def test_comparison_inputs_check_saved_physics_against_the_saved_parameters():
    historical, saved = current_inputs(), _read(pipeline.ROOT / "docs/validation/topology_phase1/inputs.json")
    variant = deepcopy(historical)
    variant["parameters"]["frame"]["wheelbase_mm"] = 131.0
    variant["domain"] = pipeline.build_design_domain(deepcopy(variant["parameters"]))
    assert variant["domain"]["comparison_load_cases"] != historical["domain"]["comparison_load_cases"]
    reference, settings = pipeline.comparison_inputs(variant, variant["domain"], 60.0, 70.0)
    assert reference["settings"]["relative_constraints"] == saved["settings"]["relative_constraints"] and settings["mesh_timeout_s"] == 60.0
    with pytest.raises(ValueError, match="not consistent with the saved parameters: comparison_load_cases"):
        pipeline.comparison_inputs(variant, historical["domain"], 60.0, 70.0)
    tampered = deepcopy(variant["domain"])
    tampered["point_masses"][0]["mass_g"] += 1.0
    with pytest.raises(ValueError, match="point_masses"):
        pipeline.comparison_inputs(variant, tampered, 60.0, 70.0)

def implicit_fakes(monkeypatch, calls):
    reference = {"source": "reference.step", "sha256": "0" * 64, "volume_mm3": 729.0, "curvature": None,
                 "metrics": {"free_sharp_edge_length_per_area_per_mm": 1e3, "free_axis_normal_area_fraction": 1e3}}
    monkeypatch.setattr(implicit, "reference_geometry", lambda path, domain, config: (trimesh.creation.box(extents=(9, 9, 9)), reference))
    monkeypatch.setattr(implicit, "build_geometry", lambda parameters: Box(20, 20, 20))
    monkeypatch.setattr(implicit, "evaluate", fake_fea(calls))
    monkeypatch.setattr(implicit, "_provenance", lambda *args: {})

def test_implicit_study_logs_a_ledger_line_when_preparation_fails(tmp_path, monkeypatch):
    source = implicit_source(tmp_path)
    inputs = _read(source / "inputs.json")
    inputs["domain"]["point_masses"][0]["mass_g"] += 1.0
    pipeline.write(source / "inputs.json", inputs)
    pipeline.write(source / "manifest.json", {"status": "ok", "artifacts": pipeline.artifacts(source)})
    implicit_fakes(monkeypatch, [])
    assert implicit.run_main(implicit_config(tmp_path, source, geometry_only=True)) == 1
    manifest = _read(tmp_path / "study/manifest.json")
    assert manifest["status"] == "failed" and "point_masses" in manifest["error"]
    line, = ledger(tmp_path / "run_log.jsonl")
    assert line["candidate"] is None and line["status"] == "failed" and line["failure_stage"] == "prepare" and not line["success"] and "point_masses" in line["error"]

@pytest.mark.parametrize("enabled", [True, False])
def test_diagnostic_fea_runs_only_when_features_is_the_only_failing_check(tmp_path, monkeypatch, enabled):
    source, calls = implicit_source(tmp_path), []
    implicit_fakes(monkeypatch, calls)
    validate = implicit.validate_implicit
    def features_only(*args):
        result = validate(*args)
        result.update(passed=False, violations=["features"])
        return result
    monkeypatch.setattr(implicit, "validate_implicit", features_only)
    assert implicit.run_main(implicit_config(tmp_path, source, diagnostic_fea=enabled)) == 0
    record = _read(tmp_path / "study/candidates/c00/record.json")
    assert record["status"] == "geometry_invalid" and not record["success"] and record["failure_stage"] == "validation" and not record["geometry_valid"]
    assert len(calls) == (2 if enabled else 0) and record.get("fea_diagnostic_only", False) == enabled and ("comparison" in record) == enabled
    manifest = _read(tmp_path / "study/manifest.json")
    assert not manifest["overall_acceptance"] and manifest["success_rate"] == 0.0
    line, = ledger(tmp_path / "run_log.jsonl")
    assert line["status"] == "geometry_invalid" and not line["success"]

@pytest.fixture
def rails(monkeypatch):
    monkeypatch.setitem(TOPOLOGY_CONFIG, "battery_support", "rails")
    monkeypatch.setitem(TOPOLOGY_CONFIG, "camera_support", "prescribed")

def test_implicit_study_flags_sources_built_on_older_prescribed_geometry(rails):
    from deep_frame.topology_geometry import build_design_domain
    parameters = topology_study.study_parameters([34, 32, 8])
    built = build_design_domain(parameters)
    domain = {**json.loads(json.dumps({key: built[key] for key in ("grid", "regions")})), **{name: built[name].copy() for name in ("allowed", "preserve", "forbidden")}}
    assert implicit.domain_currency(parameters, domain) == {"regions_current": True, "grid_current": True, "mask_cells_changed": {"allowed": 0, "preserve": 0, "forbidden": 0}, "prescribed_clearance_passed": True}
    contact = next(region for region in domain["regions"] if region["name"] == "battery_rail_1")
    contact["max_mm"][2] = 28.0
    domain["preserve"].flat[np.flatnonzero(domain["preserve"])[0]] = False
    stale = implicit.domain_currency(parameters, domain)
    assert not stale["regions_current"] and stale["mask_cells_changed"]["preserve"] == 1

def test_implicit_study_rejects_unknown_keys_and_used_output(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"source": "s", "output": "o", "opening_radius": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration keys: opening_radius"):
        implicit.main(["run", str(path)])
    with pytest.raises(SystemExit):
        implicit.main(["build", str(path)])
    (tmp_path / "used").mkdir()
    (tmp_path / "used/old.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="new empty output"):
        implicit.run_main({"source": str(tmp_path), "output": str(tmp_path / "used")})

DEFAULTS = {"name": None, "count": 3, "scale": 1.0, "flag": False, "path": None, "mode": "a", "shape": [1, 2, 3], "items": ["x"]}
KINDS = {"name": "text", "count": "int", "scale": "float", "flag": "flag", "path": "path", "mode": ("a", "b"),
         "shape": ["int"] * 3, "items": ["text"]}

def test_configure_converts_known_keys_and_keeps_defaults_untouched():
    config = configure(DEFAULTS, KINDS, {"name": "n", "count": "5", "scale": 2, "flag": True, "path": "a/b", "mode": "b",
                                         "shape": (4, 5, 6), "items": ["y", "z"]}, ("name",))
    assert config == {"name": "n", "count": 5, "scale": 2.0, "flag": True, "path": Path("a/b"), "mode": "b",
                      "shape": [4, 5, 6], "items": ["y", "z"]}
    assert configure(DEFAULTS, KINDS, {}) == DEFAULTS and DEFAULTS["shape"] == [1, 2, 3]

@pytest.mark.parametrize("overrides", [[1], "name", None, 3])
def test_configure_rejects_non_object_overrides(overrides):
    with pytest.raises(ValueError, match="JSON object"):
        configure(DEFAULTS, KINDS, overrides)

def test_configure_rejects_unknown_keys_and_missing_required():
    with pytest.raises(ValueError, match="Unknown configuration keys: other, zeta"):
        configure(DEFAULTS, KINDS, {"zeta": 1, "other": 2, "count": 1})
    with pytest.raises(ValueError, match="Missing required configuration: name, path"):
        configure(DEFAULTS, KINDS, {"count": 1}, ("name", "path"))

@pytest.mark.parametrize("key, value", [("count", True), ("count", "5.5"), ("count", 5.5), ("count", None), ("scale", False),
                                        ("scale", "fast"), ("scale", [1.0]), ("name", 3), ("flag", 1), ("flag", "true"),
                                        ("path", 3), ("mode", "c"), ("mode", ["a"]), ("shape", [1, 2]), ("shape", [1, 2, "x"]),
                                        ("shape", 3), ("items", []), ("items", "x"), ("items", [1])])
def test_configure_rejects_wrong_types_and_invalid_choices(key, value):
    with pytest.raises(ValueError, match="Invalid configuration value for " + key):
        configure(DEFAULTS, KINDS, {key: value})

@pytest.mark.parametrize("argv", [[], ["other"], ["run", "a.json", "b.json"]])
def test_command_line_rejects_bad_invocations_with_usage(argv):
    with pytest.raises(SystemExit, match=r"^usage: \{run,plot\} \[config.json\]$"):
        command_line({"run": pytest.fail, "plot": pytest.fail}, argv)

def test_command_line_passes_parsed_json_or_empty_overrides(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"count": 4}', encoding="utf-8-sig")
    assert command_line({"run": lambda overrides: overrides}, ["run", str(path)]) == {"count": 4}
    assert command_line({"run": lambda overrides: overrides}, ["run"]) == {}

def test_compute_request_declares_job_type_and_runs_command_in_cwd(tmp_path):
    spec = compute.request("fea_modal", [getattr(sys, "_base_executable", sys.executable), "-c", "import os,sys; open('out.txt','w').write(os.environ['OMP_NUM_THREADS']+os.environ['CALCULIX_PATH']); sys.exit(3)"],
                           tmp_path, {"CALCULIX_PATH": "ccx", "HOME": "x", "DEEP_FRAME_SEED": "1"}, machine="local")
    need = compute.JOB_TYPES["fea_modal"]
    assert (spec["entrypoint_num_cpus"], spec["entrypoint_memory"], spec["entrypoint_resources"]) == (need["num_cpus"], need["memory_gb"] * 2**30, None)
    assert spec["entrypoint_num_gpus"] is None
    token = spec["entrypoint"].split()[-1]
    assert compute.main(["exec", token]) == 3
    assert (tmp_path / "out.txt").read_text() == str(need["num_cpus"]) + "ccx"
    with pytest.raises(SystemExit):
        compute.main(["unknown", "--", "x"])

def test_compute_local_profile_shares_the_card_by_measured_gpu_gb_plus_margin():
    local = compute.CONFIG["heads"]["local"]
    requests = {kind: compute.request(kind, ["x"], ".", machine="local") for kind in ("density_neural", "density_simp", "density_simp_1mm", "gpu", "fea_static")}
    assert {kind: (spec["entrypoint_num_cpus"], spec["entrypoint_memory"], (spec["entrypoint_resources"] or {}).get("gpu_gb", 0), spec["entrypoint_num_gpus"])
            for kind, spec in requests.items()} == {"density_neural": (4, 24 * 2**30, 14, None), "density_simp": (4, 8 * 2**30, 13.8, None),
                                                    "density_simp_1mm": (8, 28 * 2**30, 14, None), "gpu": (4, 6 * 2**30, 6.6, None), "fea_static": (2, 8 * 2**30, 0, None)}
    assert compute.request("cpu", ["X:/isolated/gpu-venv/Scripts/python.exe"], ".", machine="local")["entrypoint_resources"] == {"gpu_gb": 6.6}
    assert 2 * requests["gpu"]["entrypoint_resources"]["gpu_gb"] <= local["gpu_gb"] and local["throttle"] == {}

def test_compute_dgx_head_fits_four_a100_jobs():
    head, need = compute.CONFIG["heads"]["dgx"], compute.request("gpu_a100", ["x"], ".", machine="dgx")
    assert (need["entrypoint_resources"], need["entrypoint_num_gpus"], compute.request("density_neural", ["x"], ".", machine="dgx")["entrypoint_num_gpus"]) == ({"gpu_gb": 80}, 1, 1)
    assert all(have >= 4 * want for have, want in ((head["num_cpus"], need["entrypoint_num_cpus"]), (head["memory_gb"] * 2**30, need["entrypoint_memory"]),
                                                   (head["gpu_gb"], need["entrypoint_resources"]["gpu_gb"]), (head["num_gpus"], need["entrypoint_num_gpus"])))

def test_compute_node20_21_routes_gpu_jobs_to_the_working_a100():
    gpu = compute.request("gpu_a100", ["x"], ".", machine="node20_21")
    cpu = compute.request("suite", ["x"], ".", {"RAY_ADDRESS": "http://192.168.2.20:8266"}, machine="node20_21")
    assert gpu["entrypoint_num_gpus"] == 1
    assert gpu["entrypoint_resources"] == {"gpu_gb": 80, "node:192.168.2.21": 0.001}
    assert cpu["entrypoint_resources"] == {"node:192.168.2.20": 0.001}
    payload = json.loads(compute.base64.urlsafe_b64decode(cpu["entrypoint"].split()[-1]))
    assert payload["env"]["RAY_ADDRESS"] == "http://192.168.2.20:8266"
    head = compute.head_command("node20_21")
    assert head[head.index("--num-gpus") + 1] == "0" and head[head.index("--dashboard-port") + 1] == "8266"
    assert "--port=6380" in head and "--min-worker-port=22000" in head

def test_compute_head_command_pins_agent_ports_per_profile(monkeypatch):
    monkeypatch.delenv("RAY_PYTHON", raising=False)
    for name, gpus, python in (("local", "1", compute.CONFIG["heads"]["local"]["ray_python"]), ("dgx", "8", compute.CONFIG["heads"]["dgx"]["ray_python"])):
        command = compute.head_command(name)
        assert command[0] == str(Path(python).with_name("ray.exe" if os.name == "nt" else "ray")) and command[1:3] == ["start", "--head"]
        assert command[command.index("--num-gpus") + 1] == gpus and command[command.index("--dashboard-port") + 1] == "8265"
        assert command[-4:] == ["--dashboard-agent-listen-port=53365", "--dashboard-agent-grpc-port=53366", "--metrics-export-port=53367", "--runtime-env-agent-port=53368"]
    monkeypatch.setenv("RAY_PYTHON", "/opt/ray/bin/python")
    assert compute.head_command("dgx")[0] == str(Path("/opt/ray/bin/python").with_name("ray.exe" if os.name == "nt" else "ray"))
    calls = []
    monkeypatch.setattr(compute, "head", lambda *args: calls.append(args) or 0)
    assert compute.main(["head", "dgx"]) == 0 and calls == [("dgx",)]
    with pytest.raises(SystemExit):
        compute.main(["head", "cluster"])

def test_round2_domain_lifts_pads_and_adds_camera_hoops(rails):
    full, half = neural_study.R2Domain(neural_study.STUDY).build([68, 64, 16])
    z = neural_study.grid_centers(full["grid"])[..., 2]
    pads = neural_study.anchors(full)
    pad = neural_study.STUDY["pad"]
    bottom = pad["top_mm"] - pad["thickness_mm"]
    assert pads.any() and z[pads].min() >= bottom and z[pads].max() <= pad["top_mm"]
    assert full["metadata"]["round2"]["hoop_cells"] > 0 and not np.any(full["preserve"] & ~full["allowed"])
    cases = {case["name"]: case for case in full["load_cases"]}
    assert "crash_hoop" in cases and half["optimizer_settings"]["case_weights"]["crash_hoop"] == 1.0
    assert all(box["min_mm"][2] == pytest.approx(bottom - 0.01) for box in cases["battery_impact"]["fixed_regions"])

def test_round3_domain_turns_flight_and_crash_cases_into_inertia_relief(rails):
    full, half = neural_study.R2Domain(neural_study.STUDY).build([68, 64, 16], 0.05)
    cases = {case["name"]: case for case in half["load_cases"]}
    relief = [name for name, case in cases.items() if "inertia_relief" in case]
    assert {"arm_tip", "thrust_all", "crash_hoop", "crash_below"} <= set(relief) and all(not cases[name]["fixed_regions"] for name in relief)
    assert "inertia_relief" not in cases["battery_impact"] and cases["battery_impact"]["fixed_regions"]
    masses = cases["thrust_all"]["inertia_relief"]
    assert sum(item["mass_g"] for item in masses["point_masses"]) == pytest.approx(37.0 + 7.2 + 2.3 + 4 * (4.5 + 1.2))
    assert masses["preserve_mass_g"] == pytest.approx(0.05 * full["metadata"]["allowed_volume_mm3"] * 1.09 / 1000)

def test_round4_hard_prop_keep_out_keeps_corridor_pads_and_connectivity(rails):
    cfg = neural_study.configure({"prop_discs": {"mode": "hard"}})
    full, half = neural_study.R2Domain(cfg).build([68, 64, 16])
    soft, _ = neural_study.R2Domain(neural_study.STUDY).build([68, 64, 16])
    discs, centers = full["metadata"]["round4"]["prop_discs"], neural_study.grid_centers(full["grid"])
    blocked = neural_study.R2Domain(cfg).keep_out(centers, discs)
    x, y = discs["motors_mm"][1]
    corridor = (np.hypot(centers[..., 0] - x * 0.75, centers[..., 1] - y * 0.75) < 1.0) & (centers[..., 2] < 5)
    assert blocked.any() and not np.any(full["allowed"] & blocked & ~full["preserve"]) and full["allowed"][corridor].all()
    assert np.array_equal(full["preserve"], soft["preserve"]) and full["metadata"]["round4"]["unblocked_cells"] == soft["metadata"]["round4"]["allowed_cells"]

@pytest.mark.parametrize("module", ["neural_study", "multi_crash_study"])
def test_threshold_surface_places_cell_faces_at_the_grid_and_keeps_mirror_symmetry(module):
    import importlib
    tool = importlib.import_module(f"tools.{module}")
    grid = {"origin_mm": [-3.0, -2.0, 0.0], "spacing_mm": [0.5, 0.5, 0.5], "shape": [12, 8, 6]}
    density = np.zeros(grid["shape"], dtype=np.float32)
    density[2:10, 2:6, 1:5] = 1.0
    cfg = {"sigma_cells": 0, "carve_bores": False, "threshold": 0.5, "min_body_mm3": 0, "taubin": 0, "flatten": []}
    mesh = tool.surface(density, grid, cfg)[0]
    assert np.allclose(mesh.bounds, [[-2.0, -1.0, 0.5], [2.0, 1.0, 2.5]])

def test_frame_request_accepts_round4_optimizer_options_and_reconstruction_switch():
    from deep_frame.frame_run import validate_request
    request = validate_request({"reconstruction": False, "overrides": {"optimizer": {"volume_fraction": 0.05, "prop_discs": "soft", "f1_min_hz": 300}}})
    assert request["reconstruction"] is False and request["overrides"]["optimizer"] == {"volume_fraction": 0.05, "prop_discs": "soft", "f1_min_hz": 300.0}
    with pytest.raises(ValueError):
        validate_request({"overrides": {"optimizer": {"prop_discs": "corridor"}}})

def test_round4_stiffness_case_keeps_the_evaluator_support_and_request_options(rails):
    from deep_frame.frame_run import validate_request
    cfg = neural_study.configure({"stiffness": {"min_n_per_mm": 10.0}})
    _, half = neural_study.R2Domain(cfg).build([68, 64, 16], 0.08)
    cases = {case["name"]: case for case in half["load_cases"]}
    case = cases["stiffness_arm_tip"]
    assert "inertia_relief" not in case and case["fixed_regions"] and "inertia_relief" in cases["arm_tip"] and case["loads"][0]["region"] == cases["arm_tip"]["loads"][0]["region"]
    _, plain = neural_study.R2Domain(neural_study.STUDY).build([68, 64, 16], 0.08)
    assert "stiffness_arm_tip" not in {case["name"] for case in plain["load_cases"]}
    request = validate_request({"overrides": {"optimizer": {"volume_fraction": 0.08, "f1_min_hz": 300, "arm_tip_stiffness_min_n_per_mm": 10, "stiffness_calibration": 1.2, "max_runtime_s": 14400}}})
    assert request["overrides"]["optimizer"]["arm_tip_stiffness_min_n_per_mm"] == 10.0 and request["overrides"]["optimizer"]["stiffness_calibration"] == 1.2 and request["overrides"]["optimizer"]["max_runtime_s"] == 14400.0

def test_lower_chord_detects_a_continuous_low_member():
    grid = {"origin_mm": [-30.0, -30.0, 0.0], "spacing_mm": [1.0, 1.0, 1.0], "shape": [60, 60, 20]}
    density = np.zeros(grid["shape"])
    density[27:33, 2:58, 2:6] = 1
    cfg = {"half_width_mm": 20.0, "z_max_mm": 10.0, "y_span_mm": [-25.0, 25.0]}
    assert neural_study.lower_chord(density, grid, cfg)["lower_chord"]
    density[:, 28:32] = 0
    report = neural_study.lower_chord(density, grid, cfg)
    assert not report["lower_chord"] and report["y_slice_coverage"] < 1

def test_keep_connected_drops_floating_parts():
    field = np.zeros((20, 10, 10))
    field[1:8, 2:6, 2:6], field[12:18, 2:6, 2:6] = 1, 1
    anchor = np.zeros(field.shape, dtype=bool)
    anchor[2, 3, 3] = True
    kept, report = neural_study.keep_connected(field, {"spacing_mm": [1, 1, 1]}, 0.5, anchor)
    assert report["components_raw"] == 2 and report["dropped_count"] == 1 and report["dropped_volume_mm3"] == 96
    assert kept[3, 3, 3] == 1 and not kept[12:18].any()

def test_interface_couple_and_conjugate_rotation():
    from tools.interface_stiffness import block_mesh, conjugate, nodal_loads
    points = np.random.default_rng(0).uniform(-3, 3, (40, 3))
    for axis in range(3):
        loads = nodal_loads(points, "moment", axis, 2.0)
        r = points - points.mean(axis=0)
        assert np.allclose(loads.sum(axis=0), 0, atol=1e-12)
        assert np.allclose(np.cross(r, loads).sum(axis=0), np.eye(3)[axis] * 2.0)
        rotation = np.cross(np.eye(3)[axis] * 0.01, r) + [0.3, -0.2, 0.1]
        assert conjugate(loads, rotation) / 2.0 == pytest.approx(0.01)
    force = nodal_loads(points, "force", 2, 1.0)
    assert conjugate(force, np.tile([0.0, 0.0, 0.5], (40, 1))) == pytest.approx(0.5)
    nodes, elements = block_mesh([4.0, 2.0, 2.0], 1.0)
    assert len(elements) == 6 * 16
    volume = sum(abs(np.linalg.det(np.array([np.subtract(nodes[n], nodes[e[0]]) for n in e[1:4]]))) / 6 for e in elements.values())
    assert volume == pytest.approx(16.0)
