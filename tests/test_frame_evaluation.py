import numpy as np
import pytest
import trimesh

from deep_frame.config import EVALUATION_CONFIG
from deep_frame.frame_evaluation import append_datasheet, assess, mass_properties, scaled, top_view, voxel_grid

def test_mass_and_inertia_of_box_with_point_component():
    mesh = trimesh.creation.box(extents=(10, 20, 30))
    component = {"name": "block", "type": "battery", "mass_g": 5.0, "center_mm": [0.0, 0.0, 40.0], "size_mm": [2.0, 4.0, 6.0], "shape": "box"}
    result = mass_properties(mesh, [component], 1.0)
    frame = 6000 * 1e-3
    assert result["frame_mass_g"] == pytest.approx(frame)
    center = 5.0 * 40 / (frame + 5.0)
    assert result["center_of_mass_mm"] == pytest.approx([0, 0, center], abs=1e-9)
    izz = frame / 12 * (10 ** 2 + 20 ** 2) + 5.0 / 12 * (2 ** 2 + 4 ** 2)
    ixx = frame / 12 * (20 ** 2 + 30 ** 2) + frame * center ** 2 + 5.0 / 12 * (4 ** 2 + 6 ** 2) + 5.0 * (40 - center) ** 2
    assert np.diag(result["inertia_g_mm2"]) == pytest.approx([ixx, ixx - frame / 12 * (20 ** 2 - 10 ** 2) - 5.0 / 12 * (4 ** 2 - 2 ** 2), izz], rel=1e-9)

def test_prop_disc_share_of_half_covering_plate():
    plate = trimesh.creation.box(extents=(100, 200, 2), transform=trimesh.transformations.translation_matrix((-50, 0, 1)))
    spec = {"prop_diameter_mm": 40.0, "motors": {name: [x, y, 2.0] for name, x, y in (("front_left", -50, 50), ("front_right", 50, 50), ("rear_left", -50, -50), ("rear_right", 50, -50))}}
    solid, lower, unbalanced = voxel_grid(plate, 0.5)
    result = top_view(solid, lower, 0.5, spec, 0.0)
    assert unbalanced == 0 and not result["discs_overlap"]
    assert result["prop_disc_share"] == pytest.approx(0.5, abs=0.01) and result["prop_radial_share"] == pytest.approx(0.5, abs=0.01)
    assert result["hull_share"] == pytest.approx(1.0, abs=0.02)

def test_unevaluable_conditions_count_as_missed():
    result = assess({"geometry": None, "fea": None, "slicer": None}, EVALUATION_CONFIG)
    assert not result["good"] and {"one_body", "fea_solved", "slicer", "crash_front_strength"} <= set(result["missed"])

def test_targets_decide_while_warnings_only_report():
    result = {"geometry": {"form": {"mesh_bodies": 1, "strut_width_mm": {"p10": 1.0}}}, "fea": None, "slicer": None}
    config = {**EVALUATION_CONFIG, "targets": {"strut_min": ["geometry.form.strut_width_mm.p10", 1.19, None]}, "warnings": {"loops": ["geometry.form.loops.loops", 20, None]}}
    assessment = assess(result, config)
    assert "target:strut_min" in assessment["missed"] and assessment["warnings"] == ["loops"]

def test_scaled_values_use_the_motor_layout_and_datasheet_append_is_idempotent(tmp_path):
    spec = {"motors": {name: [x, y, 4.0] for name, x, y in (("front_left", -30, 40), ("front_right", 30, 40), ("rear_left", -30, -40), ("rear_right", 30, -40))}, "loads": {"arm_tip_force_n": 3.6}}
    values = scaled(spec, {"fea": {"stiffness_n_per_mm": 7.2}, "geometry": {"form": {"symmetry": {"rms_mm": 1.0}}}})
    assert values["wheelbase_mm"] == pytest.approx(100.0) and values["arm_tip_slope"] == pytest.approx(0.01) and values["symmetry_per_wheelbase"] == pytest.approx(0.01) and values["support_per_volume"] is None
    sheet, result = tmp_path / "frame.md", {"name": "probe", "line": "| probe |", "assessment": {"good": False}}
    append_datasheet(sheet, result, "evaluation.json")
    append_datasheet(sheet, result, "evaluation.json")
    assert sheet.read_text(encoding="utf-8").count("| probe |") == 1

@pytest.mark.parametrize("change, missed", [({}, set()), ({"deep_fraction": 0.006}, {"wall_deep_fraction"}), ({"largest_deep_mm3": 5.5}, {"wall_deep_component"}), ({"motor_zone_hits": {"motor_front_left": {"components": 1, "volume_mm3": 0.1}}}, {"wall_motor_zones"}), ({"unbalanced_columns": 3}, {"wall_deep_fraction"})])
def test_calibrated_wall_rule_blocks_on_each_part(change, missed):
    walls = {"part_volume_mm3": 1000.0, "unbalanced_columns": 0, "deep_fraction": 0.004, "largest_deep_mm3": 4.0, "motor_zones": ["motor_front_left"], "motor_zone_hits": {}, **change}
    result = assess({"walls": walls}, EVALUATION_CONFIG)
    assert {name for name in result["missed"] if name.startswith("wall_")} == missed and "wall_opening" not in EVALUATION_CONFIG["targets"]
    assert {name for name in assess({"walls": None}, EVALUATION_CONFIG)["missed"] if name.startswith("wall_")} == {"wall_deep_fraction", "wall_deep_component", "wall_motor_zones"}

def test_datasheet_line_is_replaced_not_duplicated(tmp_path):
    sheet = tmp_path / "frame.md"
    sheet.write_text("# probe\n\n| probe | old |\n\nGut (alle Bedingungen erfüllt): ja.\n", encoding="utf-8")
    append_datasheet(sheet, {"name": "probe", "line": "| probe | new |", "assessment": {"good": False}}, "evaluation.json")
    text = sheet.read_text(encoding="utf-8")
    assert "| probe | new |" in text and "old" not in text and text.count("| probe |") == 1 and "erfüllt): nein" in text

def test_sigma_measure_matches_trace_and_symmetric_root_form():
    from deep_frame.frame_evaluation import InterfaceCompliance
    measure = InterfaceCompliance()
    sigma = measure.model.sigma
    rng = np.random.default_rng(0)
    root = rng.normal(size=(42, 42))
    flexibility = root @ root.T * 1e-3
    result = measure.measure(flexibility)["all_dofs"]
    values, vectors = np.linalg.eigh(sigma)
    half = vectors @ np.diag(np.sqrt(np.clip(values, 0, None))) @ vectors.T
    assert result["mean_compliance_n_mm"] == pytest.approx(np.trace(sigma @ flexibility), rel=1e-10)
    assert result["worst_case_compliance_n_mm"] == pytest.approx(np.linalg.eigvalsh(half @ flexibility @ half)[-1], rel=1e-6)
    assert sum(measure.measure(flexibility)["mean_by_interface_n_mm"].values()) == pytest.approx(result["mean_compliance_n_mm"], rel=1e-10)
    identity = measure.measure(np.eye(42))["all_dofs"]
    assert identity["mean_compliance_n_mm"] == pytest.approx(np.trace(sigma), rel=1e-10)
    assert identity["worst_case_compliance_n_mm"] == pytest.approx(measure.model.eigenvalues[0], rel=1e-10)
    scaled = measure.measure(np.eye(42), arm_mm=measure.config["arm_mm"] * 2)["all_dofs"]
    forces = np.array([dof[0] == "F" for _, dof in measure.labels])
    assert scaled["mean_compliance_n_mm"] == pytest.approx(np.trace(sigma[np.ix_(~forces, ~forces)]) + np.trace(sigma[np.ix_(forces, forces)]) / 2, rel=1e-10)

def test_unit_couple_has_zero_resultant_and_gap_record_maps_to_sigma_order():
    from deep_frame.frame_evaluation import DOFS, InterfaceCompliance, unit_loads
    points = np.random.default_rng(1).uniform(-3, 3, size=(30, 3))
    couple = unit_loads(points, "My")
    assert np.allclose(couple.sum(axis=0), 0, atol=1e-12)
    assert np.allclose(np.cross(points - points.mean(axis=0), couple).sum(axis=0), [0, 1, 0], atol=1e-12)
    assert np.allclose(unit_loads(points, "Fz").sum(axis=0), [0, 0, 1])
    measure = InterfaceCompliance()
    names = {"battery": "battery_rails"}
    record = {"name": "test", "interfaces": {names.get(name, name): {dof: {"stiffness": 1 / (i + 1), "conjugate": i + 1.0, "mean_displacement_mm": [0.5, 0.0, 0.0]} for dof in DOFS}
                                             for i, name in enumerate(dict.fromkeys(name for name, _ in measure.labels))}}
    diagonal, coupled = measure.gap_flexibility(record), measure.gap_flexibility(record, coupled=True)
    assert np.allclose(np.diag(diagonal), np.repeat(np.arange(1, 8), 6))
    assert coupled[0, 1] == pytest.approx(0.25) and coupled[1, 0] == pytest.approx(0.25) and coupled[6, 0] == 0
