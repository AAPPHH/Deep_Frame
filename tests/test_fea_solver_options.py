import json
from pathlib import Path
import sys

import pytest

from deep_frame.fea import _run, evaluate
from test_fea import analytical_beam, beam_inputs


@pytest.mark.parametrize("backend", ["SPOOLES", "PASTIX"])
def test_explicit_backend_preserves_real_point_mass_beam_physics(tmp_path, backend, monkeypatch):
    solid, material, cases, settings, tip = beam_inputs(tmp_path)
    settings.update(linear_solver=backend, threads=1)
    masses = [{"name": "tip_mass", "mass_g": 1.5, "position_mm": [80, 0, 0], "attachment_region": tip}]
    monkeypatch.setenv("CCX_NPROC_STIFFNESS", "7")
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "7")
    result = evaluate(solid, material, masses, cases, settings)
    expected = analytical_beam(0.0015)
    assert result["status"] == "ok", result["diagnostics"]
    assert result["linear_solver"] == backend
    assert result["solver_threads"] == 1
    assert result["eigenfrequencies_hz"][0] == pytest.approx(expected["frequency_hz"], rel=0.03)
    assert result["stiffness_n_per_mm"] == pytest.approx(1 / expected["deflection_mm"], rel=0.03)
    assert result["mass_g"] == pytest.approx(expected["mass_g"] + 1.5)
    for name, keyword in (("tip", "STATIC"), ("modes", "FREQUENCY")):
        text = Path(result["artifacts"][name]["input"]).read_text()
        assert f"*{keyword},SOLVER={backend}\n" in text
        log = Path(result["artifacts"][name]["log"]).read_text()
        assert "Using up to 7 cpu(s)" not in log
        assert "Using up to 1 cpu(s)" in log
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("backend", ["AUTO", "spooles", "PARDISO", "SPOOLES\n*STEP", float("nan"), float("inf")])
def test_invalid_backend_is_rejected_before_mesh_or_solver(tmp_path, backend):
    directory = tmp_path / "must_not_be_created"
    solid, material, cases, settings, _ = beam_inputs(directory)
    settings["linear_solver"] = backend
    result = evaluate(solid, material, [], cases, settings)
    assert result["status"] == "invalid"
    assert result["artifacts"] == {}
    assert "linear_solver" in result["diagnostics"][0]
    assert not directory.exists()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, float("nan"), "1"])
def test_invalid_thread_count_is_json_safe_before_mesh_or_solver(tmp_path, threads):
    directory = tmp_path / "must_not_be_created"
    solid, material, cases, settings, _ = beam_inputs(directory)
    settings.update(linear_solver="SPOOLES", threads=threads)
    result = evaluate(solid, material, [], cases, settings)
    assert result["status"] == "invalid"
    assert result["artifacts"] == {}
    assert "threads" in result["diagnostics"][0]
    assert not directory.exists()
    json.dumps(result, allow_nan=False)


def test_requested_thread_count_overrides_inherited_native_thread_environment(tmp_path, monkeypatch):
    keys = ("OMP_NUM_THREADS", "CCX_NPROC_RESULTS", "CCX_NPROC_STIFFNESS",
            "CCX_NPROC_EQUATION_SOLVER", "NUMBER_OF_CPUS")
    for key in keys:
        monkeypatch.setenv(key, "7")
    script = "import os,json; print(json.dumps({key:os.environ[key] for key in " + repr(keys) + "}))"
    content = _run([sys.executable, "-c", script], tmp_path, 10, 1, tmp_path / "environment.log")
    assert json.loads(content) == {key: "1" for key in keys}
