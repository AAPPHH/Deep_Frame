import json
from copy import deepcopy
from math import cos, pi, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pytest
import trimesh
from build123d import Box, GeomType, Solid
from OCP.BRep import BRep_Tool

from deep_frame.config import COMPONENT_DEFAULTS, CONFIG
from deep_frame.frame import assembly_mass_properties, assembly_scene, battery_prop_overlap, build_component_prototypes, build_components, build_geometry, build_smoke_body, camera_mount_z, collision_and_clearance, export_body, mount_positions, reference_parameters, structural_margins, validate_geometry

@pytest.mark.parametrize("dimensions", [(30.0, 20.0, 3.0), (12.5, 8.0, 1.5)])
def test_body_and_stl(tmp_path, dimensions):
    config = CONFIG | dict(zip(("length_mm", "width_mm", "thickness_mm"), dimensions))
    body = build_smoke_body(config)
    assert body.is_valid
    assert len(body.solids()) == 1
    assert body.volume == pytest.approx(np.prod(dimensions))
    path = export_body(body, tmp_path / "exports" / "smoke.stl")
    mesh = trimesh.load_mesh(path)
    assert mesh.is_volume
    assert mesh.body_count == 1
    np.testing.assert_allclose(mesh.extents, dimensions)
    assert mesh.bounds[0, 2] == pytest.approx(0.0)
    assert mesh.volume == pytest.approx(body.volume)

@pytest.mark.parametrize("key", ["length_mm", "width_mm", "thickness_mm"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_dimensions(key, value):
    with pytest.raises(ValueError, match="Dimensions"):
        build_smoke_body(CONFIG | {key: value})

@pytest.fixture(scope="module")
def frame():
    return build_geometry(reference_parameters())

def test_frame_is_one_closed_solid_and_printable_mesh(frame, tmp_path):
    assert isinstance(frame, Solid)
    assert frame.is_valid and len(frame.solids()) == 1
    assert all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in frame.shells())
    mesh = trimesh.load_mesh(export_body(frame, tmp_path / "frame.stl"))
    assert mesh.is_volume and mesh.body_count == 1
    assert mesh.bounds[0, 2] == pytest.approx(0)

def test_mount_cylinders_have_the_expected_world_axes(frame):
    parameters = reference_parameters()
    expected = [point for group in mount_positions(parameters).values() for point in group]
    faces = [face for face in frame.faces() if face.geom_type == GeomType.CYLINDER and face.radius is not None and abs(face.radius - 1.1) < 1e-7 and abs(face.axis_of_rotation.direction.Z) > 0.999]
    actual = [(face.axis_of_rotation.position.X, face.axis_of_rotation.position.Y) for face in faces]
    assert len(actual) == len(expected) == 20
    for point in expected:
        assert any(np.linalg.norm(np.array(point) - axis) < 1e-6 for axis in actual)
    for name, points in mount_positions(parameters).items():
        if name == "aio15":
            assert abs(points[0][0] - points[-1][0]) == pytest.approx(25.5)
        else:
            assert np.linalg.norm(np.array(points[0]) - points[2]) == pytest.approx(9)

def test_reference_parameters_are_independent_and_json_serializable():
    parameters = reference_parameters()
    json.dumps(parameters, allow_nan=False)
    parameters["frame"]["arm_height_mm"] = 99
    assert reference_parameters()["frame"]["arm_height_mm"] == 4
    for key, value in reference_parameters()["frame"].items():
        assert reference_parameters()["default_sources"][key]["value"] == value

def test_camera_mount_height_follows_tilt():
    parameters = reference_parameters()
    for tilt in [0, 20, 35]:
        parameters["components"]["camera"]["tilt_deg"] = tilt
        assert camera_mount_z(parameters) == pytest.approx(4.5 + 7 * (cos(radians(tilt)) + sin(radians(tilt))))

def test_frame_remains_parametric_for_wheelbase_ratio_and_arm_dimensions():
    parameters = reference_parameters()
    parameters["frame"].update(wheelbase_mm=140, lateral_longitudinal_ratio=1.2, arm_height_mm=4.5, arm_width_mm=7)
    before = deepcopy(parameters)
    shape = build_geometry(parameters)
    assert shape.is_valid and parameters == before
    expected_width = 140 * 1.2 / sqrt(1 + 1.2**2) + 19
    assert shape.bounding_box().size.X == pytest.approx(expected_width)

def test_minimum_wall_rule_includes_shaft_and_mount_window_ligaments():
    parameters = reference_parameters()
    assert min(structural_margins(parameters).values()) >= 2 - 1e-7
    parameters["frame"]["motor_shaft_hole_mm"] = 4.4
    with pytest.raises(ValueError, match="wall"):
        build_geometry(parameters)

@pytest.fixture
def component_config():
    return {"components": deepcopy(COMPONENT_DEFAULTS)}

def test_component_envelopes_and_mass(component_config):
    component_config["components"]["camera"]["tilt_deg"] = 0.0
    parts = build_component_prototypes(component_config)
    expected = {
        "aio15": (31.3, 31.3, 6.0),
        "camera": (16.0, 14.0, 14.0),
        "battery": (30.0, 63.0, 11.0),
        "motor": (14.2, 14.2, 14.6),
        "prop": (65.0, 65.0, 0.8),
        "xt30": (10.2, 12.4, 5.2),
        "balancer": (9.8, 7.5, 5.7),
    }
    for name, dimensions in expected.items():
        part = parts[name]
        shape = part["shape"]
        assert shape.is_valid
        assert len(shape.solids()) == 1
        np.testing.assert_allclose(tuple(shape.bounding_box().size), dimensions, atol=1e-7)
        assert shape.bounding_box().min.Z == pytest.approx(0, abs=1e-7)
        assert part["mass_g"] == component_config["components"][name]["mass_g"]
    assert parts["xt30"]["mass_g"] == parts["balancer"]["mass_g"] == 0
    assert parts["xt30"]["mass_scope"] == parts["balancer"]["mass_scope"] == "already_in_battery"

@pytest.mark.parametrize("name,diameter_key", [("aio15", "screw_diameter_mm"), ("motor", "screw_clearance_mm")])
def test_mount_holes_have_configured_axes_and_diameters(component_config, name, diameter_key):
    spec = component_config["components"][name]
    part = build_component_prototypes(component_config)[name]
    radius = spec[diameter_key] / 2
    surfaces = [face for face in part["shape"].faces() if face.geom_type == GeomType.CYLINDER and abs(face.radius - radius) < 1e-7]
    assert len(surfaces) == 4
    axes = sorted((round(face.axis_of_rotation.position.X, 6), round(face.axis_of_rotation.position.Y, 6)) for face in surfaces)
    half_pitch = spec["mount_pitch_mm"] / 2
    half_side = half_pitch / np.sqrt(2) if spec.get("mount_layout") == "bolt_circle" else half_pitch
    expected = sorted((x, y) for x in (-half_side, half_side) for y in (-half_side, half_side))
    np.testing.assert_allclose(axes, expected, atol=1e-6)
    np.testing.assert_allclose(sorted(tuple(round(value, 6) for value in hole[:2]) for hole in part["mount_holes"]), expected, atol=1e-6)

def test_dimensions_mounts_and_camera_tilt_respond_to_config(component_config):
    specs = component_config["components"]
    specs["aio15"].update(stack_height_mm=9.0, mount_pitch_mm=24.0)
    specs["motor"].update(diameter_mm=20.0, height_mm=15.0, mount_pitch_mm=10.0, mass_g=8.0, mount_layout="square")
    specs["camera"]["tilt_deg"] = 35.0
    specs["prop"]["diameter_mm"] = 72.0
    specs["battery"].update(length_mm=70.0, width_mm=32.0)
    parts = build_component_prototypes(component_config)
    assert parts["aio15"]["shape"].bounding_box().size.Z == pytest.approx(9.0)
    assert {abs(point[0]) for point in parts["aio15"]["mount_holes"]} == {12.0}
    assert parts["motor"]["shape"].bounding_box().size.X == pytest.approx(20.0)
    assert parts["motor"]["shape"].bounding_box().size.Z == pytest.approx(15.0)
    assert parts["motor"]["mass_g"] == 8.0
    assert {abs(point[0]) for point in parts["motor"]["mount_holes"]} == {5.0}
    assert parts["prop"]["shape"].bounding_box().size.X == pytest.approx(72.0)
    assert parts["battery"]["shape"].bounding_box().size.Y == pytest.approx(70.0)
    extent = 14 * (cos(radians(35)) + sin(radians(35)))
    np.testing.assert_allclose(tuple(parts["camera"]["shape"].bounding_box().size), (16.0, extent, extent), atol=1e-6)
    assert parts["camera"]["center_of_mass_mm"][2] == pytest.approx(extent / 2, abs=1e-6)

def test_placement_transforms_geometry_mass_center_and_mount_axes(component_config):
    component_config["components"]["motor"].update(mount_layout="square", diameter_mm=16.0)
    parts = build_components(component_config, {
        "front_motor": {"prototype": "motor", "position": (40, 30, 3), "rotation": (0, 0, 90)},
        "rear_motor": {"prototype": "motor", "position": (-40, -30, 3)},
    })
    np.testing.assert_allclose(parts["front_motor"]["center_of_mass_mm"], (40, 30, 10.3))
    np.testing.assert_allclose(parts["rear_motor"]["center_of_mass_mm"], (-40, -30, 10.3))
    np.testing.assert_allclose(parts["front_motor"]["mount_holes"][0], (44.5, 25.5, 3))
    assert parts["front_motor"]["shape"].distance_to(parts["rear_motor"]["shape"]) > 0

def test_bolt_circle_is_distinct_from_square_mount(component_config):
    spec = component_config["components"]["motor"]
    spec.update(mount_layout="bolt_circle", diameter_mm=14.2)
    motor = build_component_prototypes(component_config)["motor"]
    holes = np.array(motor["mount_holes"])
    np.testing.assert_allclose(np.linalg.norm(holes[:, :2], axis=1), 4.5)
    np.testing.assert_allclose(np.abs(holes[:, :2]), 9 / np.sqrt(8))
    spec["mount_layout"] = "square"
    with pytest.raises(ValueError, match="within the motor"):
        build_component_prototypes(component_config)

@pytest.mark.parametrize("name,key,value", [
    ("aio15", "stack_height_mm", 0),
    ("camera", "tilt_deg", float("nan")),
    ("motor", "diameter_mm", 10),
    ("motor", "screw_clearance_mm", 1.8),
    ("prop", "mass_g", -1),
    ("battery", "length_mm", float("inf")),
])
def test_invalid_component_values_are_rejected(component_config, name, key, value):
    component_config["components"][name][key] = value
    with pytest.raises(ValueError):
        build_component_prototypes(component_config)

@pytest.fixture(scope="module")
def default_result():
    return validate_geometry(reference_parameters())

def test_default_configuration_passes_and_results_are_json_serializable(default_result):
    assert default_result["passed"], default_result["violations"]
    assert default_result["checks"]["clearances"]["collisions"] == []
    assert default_result["checks"]["battery_prop_overlap"]["percent_of_battery_area"] == pytest.approx(0)
    assert default_result["checks"]["mounts"]["passed"]
    json.dumps(default_result, allow_nan=False)
    mass = default_result["checks"]["mass_properties"]
    assert mass["mass_g"] - mass["frame_mass_g"] == pytest.approx(72.9)
    tensor = np.array(mass["inertia_tensor_g_mm2"])
    np.testing.assert_allclose(tensor, tensor.T, atol=1e-7)
    assert min(np.linalg.eigvalsh(tensor)) > 0

def test_small_wheelbase_generates_true_prop_collisions():
    parameters = reference_parameters()
    parameters["frame"]["wheelbase_mm"] = 100
    result = validate_geometry(parameters)
    assert not result["passed"]
    collisions = result["checks"]["clearances"]["collisions"]
    assert any(all(part.startswith("prop_") for part in collision["parts"]) for collision in collisions)
    assert result["checks"]["battery_prop_overlap"]["percent_of_battery_area"] > 0

def test_support_contacts_never_allow_positive_intersection_volume():
    parameters = reference_parameters()
    frame = Box(10, 10, 2)
    touching = {"battery": {"shape": Box(10, 10, 2).translate((0, 0, 2))}}
    overlapping = {"battery": {"shape": Box(10, 10, 2).translate((0, 0, 1))}}
    assert collision_and_clearance(parameters, frame, touching)["passed"]
    result = collision_and_clearance(parameters, frame, overlapping)
    assert not result["passed"]
    assert result["pairs"]["frame:battery"]["intersection_mm3"] == pytest.approx(100)

def test_cad_mass_and_parallel_axis_tensor_match_two_boxes():
    frame = Box(10, 20, 30)
    components = {"box": {"shape": Box(10, 20, 30).translate((40, 0, 0)), "mass_g": 6.0}}
    result = assembly_mass_properties(frame, components, 1.0)
    assert result["mass_g"] == pytest.approx(12)
    np.testing.assert_allclose(result["center_of_mass_mm"], [20, 0, 0], atol=1e-8)
    np.testing.assert_allclose(result["inertia_tensor_g_mm2"], np.diag([1300, 5800, 5300]), atol=1e-7)

def test_projected_battery_overlap_matches_half_circle_area():
    parameters = reference_parameters()
    parameters["components"]["battery"].update(width_mm=65, length_mm=65)
    components = {
        "battery": {"kind": "battery", "center_of_mass_mm": [32.5, 0, 30]},
        "prop": {"kind": "prop", "center_of_mass_mm": [0, 0, 20]},
    }
    result = battery_prop_overlap(parameters, components)
    assert result["area_mm2"] == pytest.approx(pi * 32.5**2 / 2)
    assert result["percent_of_battery_area"] == pytest.approx(100 * pi / 8)

def test_collision_viewer_scene_contains_red_intersection_solids():
    parameters = reference_parameters()
    parameters["frame"]["wheelbase_mm"] = 100
    scene = assembly_scene(parameters)
    assert scene["collisions"]
    red = [(name, shape) for name, color, shape in zip(scene["names"], scene["colors"], scene["shapes"]) if color == "#ff2020"]
    assert len(red) == len(scene["collisions"])
    assert all(name.startswith("COLLISION") and shape.volume > 1e-6 for name, shape in red)

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
