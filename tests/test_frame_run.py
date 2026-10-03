import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import trimesh

import run
from deep_frame.config import COMPONENT_DEFAULTS, COMPONENT_LIBRARY, LIBRARY_FIELDS, RUN_SETTINGS
from deep_frame.frame_run import FrameLayout, datasheet, library_part, validate_request

ROOT = Path(__file__).resolve().parents[1]

def request(**changes):
    return {**deepcopy(run.FRAME), **changes}

def test_component_defaults_stay_derivable_from_the_library():
    assert COMPONENT_DEFAULTS["aio15"]["mount_pitch_mm"] == 25.5 and COMPONENT_DEFAULTS["aio15"]["stack_height_mm"] == 6.0
    assert COMPONENT_DEFAULTS["motor"]["diameter_mm"] == 15.76 and COMPONENT_DEFAULTS["motor"]["mount_layout"] == "bolt_circle" and COMPONENT_DEFAULTS["motor"]["screw_clearance_mm"] == 2.2
    assert COMPONENT_DEFAULTS["camera"]["tilt_deg"] == 20.0 and COMPONENT_DEFAULTS["prop"]["diameter_mm"] == 63.5 and COMPONENT_DEFAULTS["prop"]["thickness_mm"] == 5.0 and COMPONENT_DEFAULTS["prop"]["mass_g"] == 1.2
    assert COMPONENT_DEFAULTS["xt30"]["mass_scope"] == "already_in_battery" and COMPONENT_DEFAULTS["balancer"]["pins"] == 3
    layout = FrameLayout(request())
    motor, aio = layout.components["motor"], layout.components["aio15"]
    assert (motor["diameter_mm"], motor["shaft_diameter_mm"], motor["kv"], motor["thrust_n"], motor["mass_g"]) == (15.76, 1.5, 8000, 1.80, 4.5)
    assert (aio["grommet_height_mm"], aio["elrs_antenna_clearance_mm"]) == (3.0, 3.0)
    assert layout.components["battery"]["cells"] == 2 and layout.components["battery"]["capacity_mah"] == 550

def test_every_library_entry_carries_its_required_fields():
    for name, entry in COMPONENT_LIBRARY.items():
        if name != "GEPRC GR1105":
            assert library_part(entry["type"], name)["type"] == entry["type"]
    with pytest.raises(ValueError, match="data.thrust_n"):
        library_part("motor", "GEPRC GR1105")

def test_missing_part_aborts_with_the_required_field_list():
    with pytest.raises(ValueError, match="not in COMPONENT_LIBRARY") as error:
        FrameLayout(request(components={**run.FRAME["components"], "motor": "XING 1404"}))
    assert all(field in str(error.value) for field in LIBRARY_FIELDS["common"] + LIBRARY_FIELDS["motor"])
    broken = {"Broken": {**deepcopy(COMPONENT_LIBRARY["HDZero Lux"]), "mass_g": None}}
    with pytest.raises(ValueError, match="mass_g"):
        library_part("camera", "Broken", broken)
    with pytest.raises(ValueError, match="not a valid 'battery'"):
        library_part("battery", "HDZero Lux")

@pytest.mark.parametrize("changes, message", [
    ({"colour": "red"}, "Unknown configuration keys: colour"),
    ({"layout": {"x_type": "h_frame"}}, "x_type"),
    ({"layout": {"x_type": "true_x", "deck": 1}}, "Unknown configuration keys: deck"),
    ({"print": {"nozzle_mm": 0.4, "layer_mm": 0.4}}, "layer_mm"),
    ({"style": "racing"}, "style"),
    ({"overrides": {"wings": {"span_mm": 1.0}}}, "Unknown override groups: wings"),
    ({"overrides": {"camera": {"zoom": 2.0}}}, "Unknown configuration keys: zoom"),
    ({"layout": {"x_type": "compressed_x", "battery_mount": "bottom"}}, "battery_mount bottom needs"),
    ({"prop_size_in": 3.0}, "leave the design envelope"),
    ({"overrides": {"battery": {"deck_top_mm": 25.0}}}, "Battery underside"),
    ({"overrides": {"motors": {"wheelbase_mm": 120.0}}}, "neighbouring"),
])
def test_invalid_configurations_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        FrameLayout(request(**changes))

def test_presets_translate_deterministically():
    first, second = FrameLayout(request()), FrameLayout(request())
    assert json.dumps(first.summary(), sort_keys=True) == json.dumps(second.summary(), sort_keys=True)
    assert first.patch(["crash_front", "crash_arm"]) == second.patch(["crash_front", "crash_arm"])
    assert first.frame["wheelbase_mm"] == 132.5 and first.frame["camera_y_mm"] == pytest.approx(35.0) and not {"antenna_y_mm", "connector_y_mm"} & set(first.frame)
    assert "antennas" not in first.summary() and first.motors()["front_right"] == pytest.approx([53.4547, 39.1364], abs=1e-4)
    assert first.hoop()["x_mm"] == 12.0 and first.hoop()["path_yz_mm"][0] == pytest.approx([24.0, 27.5])
    patch = first.patch(["crash_front", "crash_arm"])
    assert patch["TOPOLOGY_CONFIG"]["optimizer"]["case_weights"] == {"crash_front": 2.0, "crash_arm": 2.0}
    assert len(patch["INTEGRATION_CONFIG"]["crash_directions"]) == 9 and patch["TOPOLOGY_CONFIG"]["manufacturing"]["minimum_feature_mm"] == 2.4
    light = FrameLayout(request(durability="light")).patch(["crash_front"])
    assert light["TOPOLOGY_CONFIG"]["optimizer"]["case_weights"]["crash_front"] < 1.0 and light["TOPOLOGY_CONFIG"]["manufacturing"]["minimum_feature_mm"] == 2.0
    stretched = FrameLayout(request(layout={"x_type": "stretched_x", "battery_mount": "top"})).motors()["front_right"]
    assert stretched[1] > stretched[0] and FrameLayout(request(layout={"x_type": "true_x"})).motors()["front_right"][0] == pytest.approx(FrameLayout(request(layout={"x_type": "true_x"})).motors()["front_right"][1])

def test_override_changes_only_the_targeted_value():
    base = FrameLayout(request())
    tilted = FrameLayout(request(overrides={"camera": {"tilt_deg": 25.0}}))
    assert tilted.frame == base.frame and tilted.components["camera"]["tilt_deg"] == 25.0
    assert {key: value for key, value in tilted.components["camera"].items() if key not in ("tilt_deg", "parameter_sources")} == {key: value for key, value in base.components["camera"].items() if key not in ("tilt_deg", "parameter_sources")}
    assert {name: spec for name, spec in tilted.components.items() if name != "camera"} == {name: spec for name, spec in base.components.items() if name != "camera"}
    wide = FrameLayout(request(overrides={"motors": {"wheelbase_mm": 136.0}}))
    assert {key for key in base.frame if base.frame[key] != wide.frame[key]} == {"wheelbase_mm"} and wide.components == base.components
    with pytest.raises(ValueError):
        validate_request(request(overrides={"antennas": {"y_mm": -50.0}}))

def test_datasheet_has_ten_fields_and_the_criteria_line(tmp_path):
    layout = FrameLayout(request())
    trimesh.creation.box(extents=(60, 40, 6), transform=trimesh.transformations.translation_matrix([0, 0, 3])).export(tmp_path / "frame.stl")
    (tmp_path / "layout.json").write_text(json.dumps(layout.summary(), default=str), encoding="utf-8")
    (tmp_path / "evaluation.json").write_text(json.dumps({"line": "| test | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | – |"}), encoding="utf-8")
    manifest = {"name": "test_1", "stages": {"optimization": {"status": "ran"}, "evaluation": {"status": "ran"}}, "wall_rule_passed": True, "notes": ["torsion pending"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert datasheet(tmp_path / "manifest.json") == 0
    text = (tmp_path / "datasheet.md").read_text(encoding="utf-8")
    assert all(f"| {index} " in text for index in range(1, 12)) and "| test | 1 | 2 |" in text and "torsion pending" in text and "15,7 g" in text and "keine FEA" in text
    surface = {"status": "ok", "fea_surface": {"trials": [{"deviation_mm": 0.41, "passed": True}]}}
    (tmp_path / "evaluation.json").write_text(json.dumps({"line": "x", "fea": surface}), encoding="utf-8")
    datasheet(tmp_path / "manifest.json")
    assert "0,41 mm vom STL ab (Grenze 0,20 mm): WARNUNG" in (tmp_path / "datasheet.md").read_text(encoding="utf-8")

def test_stage_shim_patches_the_stage_worktree_config(tmp_path):
    (tmp_path / "deep_frame").mkdir()
    (tmp_path / "deep_frame" / "config.py").write_text("SETTINGS = {'a': {'b': 1, 'c': 2}}\n", encoding="utf-8")
    (tmp_path / "tool.py").write_text("import json, sys\nfrom pathlib import Path\nimport deep_frame.config as c\nPath(sys.argv[2]).with_name('out.json').write_text(json.dumps({'settings': c.SETTINGS, 'argv': sys.argv[1], 'file': c.__file__, 'overrides': json.loads(Path(sys.argv[2]).read_text())}))\n", encoding="utf-8")
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"root": str(tmp_path), "tool": "tool.py", "action": "cli", "patch": {"SETTINGS": {"a": {"b": 5}}}, "argv": ["run"], "overrides": {"x": 1}}), encoding="utf-8")
    assert subprocess.call([sys.executable, str(ROOT / "run.py"), "stage", str(path)], cwd=tmp_path) == 0
    result = json.loads((tmp_path / "out.json").read_text())
    assert result["settings"] == {"a": {"b": 5, "c": 2}} and result["argv"] == "run" and result["overrides"] == {"x": 1} and Path(result["file"]).parent.parent == tmp_path

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DEEP_FRAME_SLOW") != "1", reason="coarse end-to-end run through Ray; set DEEP_FRAME_SLOW=1")
def test_coarse_run_produces_all_outputs(tmp_path):
    settings = {**RUN_SETTINGS, "root": str(tmp_path)}
    from deep_frame.frame_run import FrameRun
    manifest = FrameRun(request(name="slow", grid="coarse"), settings=settings).run()
    directory = tmp_path / manifest["name"]
    assert manifest["stages"]["optimization"]["status"] == "ran"
    for name in ("frame.stl", "datasheet.md", "evaluation.json", "config.json", "manifest.json", "layout.json", *[f"renders/{view}.png" for view in RUN_SETTINGS["views"]]):
        assert (directory / name).is_file(), name
