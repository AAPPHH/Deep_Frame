import json
from math import pi

import numpy as np
import pytest
from build123d import Box

from deep_frame.checks import assembly_mass_properties, battery_prop_overlap, collision_and_clearance
from deep_frame.geometry import reference_parameters, validate_geometry
from deep_frame.viewer import assembly_scene


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
