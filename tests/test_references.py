import json
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1] / "docs" / "research"


def test_reference_inventory_and_cell_provenance():
    rows = json.loads((RESEARCH / "armattan_frames.json").read_text(encoding="utf-8"))
    ids = {row["id"] for row in rows}
    assert len(rows) == len(ids)
    assert {"tadpole_2_5", "tadpole_3", "tadpole_hd_3", "odonata_40mm", "mrp_130", "cf_355", "nutria_7"} <= ids
    assert any("dji" in key for key in ids)
    required = {"size_class_in", "wheelbase_mm", "layout", "lateral_longitudinal_ratio", "plate_thickness_mm", "stack_height_mm", "deck_height_mm", "mass_g", "motor_mount_mm", "camera_width_mm", "battery_mount", "material"}
    for row in rows:
        assert row["sources"]
        assert all(source.startswith("https://") for source in row["sources"])
        assert required <= row["fields"].keys()
        for field in row["fields"].values():
            if field["value"] is not None:
                assert field["source"] in row["sources"]
            else:
                assert field["note"]


def test_only_scale_independent_principles_are_adopted():
    principles = json.loads((RESEARCH / "principles.json").read_text(encoding="utf-8"))
    rows = json.loads((RESEARCH / "armattan_frames.json").read_text(encoding="utf-8"))
    ids = {row["id"] for row in rows}
    for principle in principles:
        assert principle["rationale"] and principle["sources"]
        assert set(principle["frame_ids"]) <= ids
        assert not principle["adopted"] or principle["scale_independent"]
    assert any(not item["scale_independent"] for item in principles)


def test_defaults_reference_existing_frames_and_adopted_principles():
    defaults = json.loads((RESEARCH / "design_defaults.json").read_text(encoding="utf-8"))
    ids = {row["id"] for row in json.loads((RESEARCH / "armattan_frames.json").read_text(encoding="utf-8"))}
    adopted = {item["id"] for item in json.loads((RESEARCH / "principles.json").read_text(encoding="utf-8")) if item["adopted"] and item["scale_independent"]}
    for default in defaults.values():
        assert set(default["frame_ids"]) <= ids
        assert set(default["principle_ids"]) <= adopted
        assert default["rationale"]
