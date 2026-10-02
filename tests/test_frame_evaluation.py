import numpy as np
import pytest
import trimesh

from deep_frame.config import EVALUATION_CONFIG
from deep_frame.frame_evaluation import assess, mass_properties, top_view, voxel_grid

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
    assert result["prop_disc_share"] == pytest.approx(0.5, abs=0.01)
    assert result["hull_share"] == pytest.approx(1.0, abs=0.02)

def test_unevaluable_conditions_count_as_missed():
    result = assess({"geometry": None, "fea": None, "slicer": None}, EVALUATION_CONFIG)
    assert not result["good"] and {"one_body", "fea_solved", "slicer", "crash_front_strength"} <= set(result["missed"])
