import json
from copy import deepcopy
from math import cos, radians, sin, sqrt

import numpy as np
import pytest
import trimesh
from build123d import GeomType, Solid
from OCP.BRep import BRep_Tool

from deep_frame.frame import camera_mount_z, mount_positions, structural_margins
from deep_frame.geometry import build_geometry, reference_parameters
from deep_frame.model import export_body


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
