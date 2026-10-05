import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import trimesh

import run
from deep_frame.config import COMPONENT_DEFAULTS, COMPONENT_LIBRARY, LAYOUT_DEFAULT, LAYOUT_OPTIMIZATION, LAYOUT_RULES, LIBRARY_FIELDS, RUN_SETTINGS
from deep_frame.frame_evaluation import component_inertia, rigid_assembly
from deep_frame.frame_run import FrameLayout, LayoutModel, datasheet, layout_setup, library_part, validate_request

ROOT = Path(__file__).resolve().parents[1]

def request(**changes):
    return {**deepcopy(run.FRAME), **changes}

def test_near_ground_patch_keeps_ideal_press_battery_and_rejects_positioning_aid():
    from deep_frame.config import BATTERY_SUPPORT
    layout = FrameLayout(request(overrides={"battery":{"support":"free"},"camera":{"support":"free","near_ground":True,"flight_pitch_deg":15,"bottom_clearance_mm":-4,"y_mm":52}}))
    patch = layout.patch()
    assert "BATTERY_SUPPORT" not in patch and "battery_guide" not in patch["TOPOLOGY_CONFIG"] and BATTERY_SUPPORT["retention"] == "ideal_press"
    assert patch["TOPOLOGY_CONFIG"]["low_flight"] == {"enabled":True,"pitch_deg":15}
    with pytest.raises(ValueError,match="ideally pressed battery"):
        FrameLayout(request(overrides={"battery":{"support":"rails"},"camera":{"support":"free","near_ground":True}})).patch()
    with pytest.raises(ValueError,match="positioning_aid"):
        validate_request(request(overrides={"battery":{"support":"free","positioning_aid":True}}))

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
    ({"overrides": {"battery": {"deck_top_mm": 19.0}}}, "Battery underside"),
    ({"overrides": {"motors": {"wheelbase_mm": 120.0}}}, "neighbouring"),
])
def test_invalid_configurations_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        FrameLayout(request(**changes))

def test_presets_translate_deterministically():
    first, second = FrameLayout(request()), FrameLayout(request())
    assert json.dumps(first.summary(), sort_keys=True) == json.dumps(second.summary(), sort_keys=True)
    assert first.patch(["crash_front", "crash_arm"]) == second.patch(["crash_front", "crash_arm"])
    assert first.frame["wheelbase_mm"] == 132.5 and first.frame["camera_y_mm"] == LAYOUT_DEFAULT["layout"]["camera_y_mm"] and not {"antenna_y_mm", "connector_y_mm"} & set(first.frame)
    assert "antennas" not in first.summary() and first.motors()["front_right"] == pytest.approx([53.4547, 39.1364], abs=1e-4)
    assert first.hoop()["x_mm"] == 12.0 and first.hoop()["path_yz_mm"][0] == pytest.approx([LAYOUT_RULES["hoop"]["path_yz_mm"][0][0] + LAYOUT_DEFAULT["layout"]["camera_y_mm"], LAYOUT_RULES["hoop"]["path_yz_mm"][0][1]])
    patch = first.patch(["crash_front", "crash_arm"])
    assert patch["TOPOLOGY_CONFIG"]["optimizer"]["case_weights"] == {"crash_front": 2.0, "crash_arm": 2.0}
    assert len(patch["INTEGRATION_CONFIG"]["crash_directions"]) == 9 and patch["TOPOLOGY_CONFIG"]["manufacturing"]["minimum_feature_mm"] == 2.4
    light = FrameLayout(request(durability="light")).patch(["crash_front"])
    assert light["TOPOLOGY_CONFIG"]["optimizer"]["case_weights"]["crash_front"] < 1.0 and light["TOPOLOGY_CONFIG"]["manufacturing"]["minimum_feature_mm"] == 2.0
    assert patch["TOPOLOGY_CONFIG"]["battery_support"] == "free" and FrameLayout(request(overrides={"battery": {"support": "rails"}})).patch()["TOPOLOGY_CONFIG"]["battery_support"] == "rails"
    stretched = FrameLayout(request(layout={"x_type": "stretched_x", "battery_mount": "top"})).motors()["front_right"]
    assert stretched[1] > stretched[0] and FrameLayout(request(layout={"x_type": "true_x"})).motors()["front_right"][0] == pytest.approx(FrameLayout(request(layout={"x_type": "true_x"})).motors()["front_right"][1])

def test_override_changes_only_the_targeted_value():
    base = FrameLayout(request())
    tilted = FrameLayout(request(overrides={"camera": {"tilt_deg": 25.0}}))
    changed = lambda layout: {key for key in base.frame if base.frame[key] != layout.frame[key]}
    assert changed(tilted) <= set(LAYOUT_OPTIMIZATION["variables"]) and any("no layout optimum" in note for note in tilted.notes) and tilted.components["camera"]["tilt_deg"] == 25.0
    assert changed(FrameLayout(request(overrides={"camera": {"y_mm": 45.0}}))) == {"camera_y_mm"}
    assert {key: value for key, value in tilted.components["camera"].items() if key not in ("tilt_deg", "parameter_sources")} == {key: value for key, value in base.components["camera"].items() if key not in ("tilt_deg", "parameter_sources")}
    assert {name: spec for name, spec in tilted.components.items() if name != "camera"} == {name: spec for name, spec in base.components.items() if name != "camera"}
    wide = FrameLayout(request(overrides={"motors": {"wheelbase_mm": 136.0}}))
    assert "wheelbase_mm" in changed(wide) and changed(wide) <= {"wheelbase_mm", *LAYOUT_OPTIMIZATION["variables"]} and wide.components == base.components
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

def point_setup(**changes):
    zero = {"mass_g": 0.0, "width_mm": 0.0, "length_mm": 0.0, "height_mm": 0.0}
    setup = {"motors_xy": [[-40.0, 30.0], [40.0, 30.0], [-40.0, -30.0], [40.0, -30.0]], "pad_z_mm": 5.0, "motor": {"mass_g": 5.0, "diameter_mm": 0.0, "height_mm": 0.0},
             "prop": {"mass_g": 0.0, "diameter_mm": 60.0, "thickness_mm": 0.0}, "battery": zero, "aio": {**zero, "stack_height_mm": 0.0, "elrs_mm": 3.0}, "camera": {**zero, "tilt_deg": 0.0},
             "base_mm": 2.0, "deck_thickness_mm": 2.0, "battery_prop_clearance_mm": 2.0, "envelope": {"half_y_mm": 64.0, "top_mm": 32.0}, "hoop": {"front_mm": 0.0, "top_mm": 32.0},
             "frame": {"mass_g": 0.0, "center_mm": [0.0, 0.0, 0.0], "inertia_g_mm2": np.zeros((3, 3)).tolist()}, "thrust_n": 2.0, "torque_per_thrust_mm": 4.0}
    return {**setup, **changes}

def test_rigid_assembly_matches_point_masses_and_a_box():
    total, center, inertia = rigid_assembly([(1.0, [0.0, 0.0, 0.0], np.zeros((3, 3))), (3.0, [4.0, 0.0, 0.0], np.zeros((3, 3)))])
    assert total == 4.0 and center == pytest.approx([3.0, 0.0, 0.0]) and np.diag(inertia) == pytest.approx([0.0, 12.0, 12.0])
    box = {"mass_g": 12.0, "size_mm": [10.0, 20.0, 30.0], "shape": "box"}
    assert np.diag(component_inertia(box)) == pytest.approx([20 ** 2 + 30 ** 2, 10 ** 2 + 30 ** 2, 10 ** 2 + 20 ** 2])
    mesh = trimesh.creation.box(extents=(10, 20, 30))
    assert np.asarray(mesh.moment_inertia) * 12.0 / mesh.volume == pytest.approx(component_inertia(box))
    _, center, inertia = rigid_assembly([(12.0, [5.0, 0.0, 0.0], component_inertia(box)), (12.0, [-5.0, 0.0, 0.0], component_inertia(box))])
    joined = trimesh.creation.box(extents=(20, 20, 30))
    assert center == pytest.approx([0, 0, 0]) and inertia == pytest.approx(np.asarray(joined.moment_inertia) * 24.0 / joined.volume)

def test_alpha_is_torque_over_inertia_for_point_motors():
    model = LayoutModel(point_setup())
    result = model.evaluate({"battery_y_mm": 0.0, "deck_top_mm": 20.0, "camera_y_mm": 30.0, "camera_bottom_clearance_mm": 2.0, "aio_standoff_mm": 3.0})
    assert result["torque_n_mm"] == pytest.approx({"roll": 2 * 2.0 * 40.0, "pitch": 2 * 2.0 * 30.0, "yaw": 2 * 4.0 * 2.0})
    assert np.diag(result["inertia_g_mm2"]) == pytest.approx([4 * 5.0 * 30.0 ** 2, 4 * 5.0 * 40.0 ** 2, 4 * 5.0 * (30.0 ** 2 + 40.0 ** 2)])
    assert result["alpha_rad_s2"]["roll"] == pytest.approx(160.0 / 32000.0 * 1e6) and result["alpha_rad_s2"]["pitch"] == pytest.approx(120.0 / 18000.0 * 1e6) and result["limiting_axis"] == "yaw" and result["center_of_mass_mm"] == pytest.approx([0.0, 0.0, 5.0])
    shifted = LayoutModel(point_setup(battery={"mass_g": 20.0, "width_mm": 0.0, "length_mm": 0.0, "height_mm": 0.0})).evaluate({**result["layout"], "battery_y_mm": 10.0})
    assert shifted["center_of_mass_mm"][1] == pytest.approx(20.0 * 10.0 / 40.0) and "cg_y" in shifted["violated"]

def test_camera_sees_a_prop_in_its_field_of_view():
    model = LayoutModel(point_setup(camera={"mass_g": 0.0, "width_mm": 10.0, "length_mm": 10.0, "height_mm": 10.0, "tilt_deg": 0.0}, motors_xy=[[-40.0, 60.0], [40.0, 60.0], [-40.0, -60.0], [40.0, -60.0]],
                                    pad_z_mm=10.0, prop={"mass_g": 0.0, "diameter_mm": 20.0, "thickness_mm": 1.0}))
    layout = {"battery_y_mm": 0.0, "deck_top_mm": 20.0, "camera_y_mm": 20.0, "camera_bottom_clearance_mm": 2.0, "aio_standoff_mm": 3.0}
    assert model.fov_margin(model.vector(layout)) < 0
    assert model.fov_margin(model.vector({**layout, "camera_y_mm": 75.0})) > 0

def test_reference_assemblies_reproduce_the_evaluator_inertia():
    ours = FrameLayout(request())
    v3b = LayoutModel(ours.setup()).evaluate(LAYOUT_OPTIMIZATION["frame_share"]["layout"])
    assert np.diag(v3b["inertia_g_mm2"]) == pytest.approx([75564, 93659, 148441], rel=1e-4) and v3b["center_of_mass_mm"][1] == pytest.approx(1.9926, abs=1e-3)
    manafly = layout_setup("manafly3")
    assert np.diag(LayoutModel(manafly).evaluate(manafly["own"])["inertia_g_mm2"])[:2] == pytest.approx([171699, 126652], rel=1e-4)
    agility = ours.summary()["agility"]
    assert agility["layout"] == {name: ours.frame[name] for name in LAYOUT_OPTIMIZATION["variables"]} and set(agility["alpha_rad_s2"]) == {"roll", "pitch", "yaw"}
    assert agility["feasible"] and "layout_optimization.json" in agility["source"] and LayoutModel(ours.setup()).rounded(LAYOUT_DEFAULT["layout"])["layout"] == LAYOUT_DEFAULT["layout"]

def test_stack_posts_follow_the_standoff_and_the_fastening_switch_reaches_domain_and_datasheet(tmp_path):
    layout = FrameLayout(request())
    posts = layout.summary()["stack"]["posts"]
    assert posts["standoff_mm"] == layout.frame["aio_standoff_mm"] and posts["fastening"] == "heat_set" and posts["bore_diameter_mm"] == 3.2
    assert posts["height_mm"] == max(posts["standoff_mm"], posts["bore_depth_mm"] + posts["bore_floor_mm"]) and layout.patch()["TOPOLOGY_CONFIG"]["stack_post"] == {"fastening": "heat_set"}
    tapped = FrameLayout(request(overrides={"stack": {"fastening": "self_tapping", "standoff_mm": 8.0}, "battery": {"deck_top_mm": 26.0}}))
    assert tapped.patch()["TOPOLOGY_CONFIG"]["stack_post"] == {"fastening": "self_tapping"} and tapped.summary()["stack"]["posts"]["height_mm"] == 8.0
    with pytest.raises(ValueError):
        validate_request(request(overrides={"stack": {"fastening": "nut"}}))
    trimesh.creation.box(extents=(60, 40, 6)).export(tmp_path / "frame.stl")
    (tmp_path / "layout.json").write_text(json.dumps(tapped.summary(), default=str), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"name": "t", "stages": {}, "notes": []}), encoding="utf-8")
    datasheet(tmp_path / "manifest.json")
    text = (tmp_path / "datasheet.md").read_text(encoding="utf-8")
    assert "4 Pfosten Ø 4,5 mm" in text and "Höhe = Stack-Abstand 8,0 mm)" in text and "selbstschneidend (Sackbohrung Ø 1,6 × 4,0 mm)" in text and "kein Zugang von unten" in text

def test_git_source_counts_untracked_files_as_dirty(tmp_path):
    from deep_frame.frame_run import _git
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'a.py').write_text('')
    subprocess.run(['git','-C',str(tmp_path),'add','a.py'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=t','-c','user.email=t@t','commit','-qm','a'],check=True)
    assert _git(tmp_path)['dirty'] is False
    (tmp_path/'b.py').write_text('')
    assert _git(tmp_path)['dirty'] is True
    assert _git(tmp_path/'missing')=={'sha':None,'branch':None,'dirty':None}
