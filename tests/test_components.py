from copy import deepcopy
from math import cos, radians, sin

import numpy as np
import pytest
from build123d import GeomType

from deep_frame.component_defaults import COMPONENT_DEFAULTS
from deep_frame.components import build_component_prototypes, build_components


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
        "motor": (16.0, 16.0, 12.0),
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
    expected = sorted((x, y) for x in (-half_pitch, half_pitch) for y in (-half_pitch, half_pitch))
    np.testing.assert_allclose(axes, expected)
    np.testing.assert_allclose([hole[:2] for hole in part["mount_holes"]], expected)


def test_dimensions_mounts_and_camera_tilt_respond_to_config(component_config):
    specs = component_config["components"]
    specs["aio15"].update(stack_height_mm=9.0, mount_pitch_mm=24.0)
    specs["motor"].update(diameter_mm=20.0, height_mm=15.0, mount_pitch_mm=10.0, mass_g=8.0)
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
    parts = build_components(component_config, {
        "front_motor": {"prototype": "motor", "position": (40, 30, 3), "rotation": (0, 0, 90)},
        "rear_motor": {"prototype": "motor", "position": (-40, -30, 3)},
    })
    np.testing.assert_allclose(parts["front_motor"]["center_of_mass_mm"], (40, 30, 9))
    np.testing.assert_allclose(parts["rear_motor"]["center_of_mass_mm"], (-40, -30, 9))
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
