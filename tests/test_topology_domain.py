import json
from copy import deepcopy

import numpy as np
import pytest
from scipy.ndimage import label

from deep_frame.geometry import reference_parameters
from deep_frame.topology_domain import build_design_domain, grid_centers, rasterize_regions, region_contains


@pytest.fixture(scope="module")
def domain():
    return build_design_domain(reference_parameters())


def test_domain_is_free_connected_3d_space_with_small_preserve_fraction(domain):
    assert domain["grid"]["shape"] == [34, 32, 8]
    assert domain["grid"]["axis_order"] == "xyz"
    assert domain["grid"]["order"] == "C"
    assert domain["allowed"].dtype == bool
    assert not np.any(domain["preserve"] & domain["forbidden"])
    assert np.array_equal(domain["forbidden"], ~domain["allowed"])
    assert label(domain["allowed"])[1] == 1
    assert domain["metadata"]["free_fraction_of_allowed"] > 0.95
    assert domain["metadata"]["allowed_cells"] > 5000
    assert np.count_nonzero(domain["allowed"].any(axis=(0, 1))) == 8
    assert len([region for region in domain["regions"] if region["role"] == "allowed"]) == 1
    assert domain["metadata"]["preserve_volume_mm3"] == domain["preserve"].sum() * 64.0


def test_no_v0_shape_call_or_shape_parameter_dependency(monkeypatch, domain):
    def fail(*args, **kwargs):
        raise AssertionError("The free domain must not build a parametric frame")

    monkeypatch.setattr("deep_frame.frame.build_frame", fail)
    monkeypatch.setattr("deep_frame.geometry.build_geometry", fail)
    monkeypatch.setattr("deep_frame.integration.build_geometry", fail)
    parameters = reference_parameters()
    parameters["frame"].update(arm_width_mm=0.001, arm_root_mm=80.0, body_width_mm=1.0, body_length_mm=1.0, base_window_mm=999.0, wall_window_height_mm=999.0, deck_window_width_mm=999.0)
    changed = build_design_domain(parameters)
    assert np.array_equal(changed["allowed"], domain["allowed"])
    assert np.array_equal(changed["preserve"], domain["preserve"])


def test_exact_keepouts_include_hardware_props_holes_and_assembly_access(domain):
    names = {region["name"] for region in domain["regions"]}
    assert {"aio15_envelope", "camera_envelope", "battery_envelope", "xt30_envelope", "balancer_envelope", "antenna_bore", "aio_side_assembly_access", "battery_insertion", "camera_front_access", "balance_lead_routing"} <= names
    assert sum(name.endswith("_swept_clearance") for name in names) == 4
    assert sum("motor_screw_" in name for name in names) == 16
    assert sum(name.startswith("aio_screw_") for name in names) == 4
    assert sum(name.startswith("strap_access_") for name in names) == 4
    assert all(not region.get("rasterize", True) for region in domain["regions"] if "motor_screw_" in region["name"] or region["name"].startswith("aio_screw_"))
    assert domain["manufacturing"]["supports_allowed"]


def test_forbidden_overlap_is_conservative_but_support_contact_is_permitted():
    grid = {"origin_mm": [0, 0, 0], "spacing_mm": [4, 4, 4], "shape": [3, 2, 2]}
    regions = [
        {"name": "all", "role": "allowed", "kind": "box", "min_mm": [0, 0, 0], "max_mm": [12, 8, 8]},
        {"name": "component", "role": "forbidden", "kind": "box", "min_mm": [3.9, 0, 4], "max_mm": [4.1, 8, 8]},
    ]
    masks = rasterize_regions(grid, regions)
    assert np.all(masks["allowed"][:, :, 0])
    assert not np.any(masks["allowed"][:2, :, 1])
    assert np.all(masks["allowed"][2, :, 1])


def test_conservative_cylinder_raster_uses_cell_box_distance():
    grid = {"origin_mm": [0, 0, 0], "spacing_mm": [4, 4, 4], "shape": [3, 2, 2]}
    regions = [
        {"name": "all", "role": "allowed", "kind": "box", "min_mm": [0, 0, 0], "max_mm": [12, 8, 8]},
        {"name": "shaft", "role": "forbidden", "kind": "cylinder", "center_mm": [4, 4, 6], "radius_mm": 0.2, "height_mm": 4, "axis": "z"},
    ]
    masks = rasterize_regions(grid, regions)
    assert masks["forbidden"][:2, :, 1].all()
    assert not masks["forbidden"][:, :, 0].any()


def test_every_required_attachment_has_preserved_optimization_cells(domain):
    centers = grid_centers(domain["grid"])
    for region in domain["regions"]:
        if region["role"] == "preserve":
            assert np.any(region_contains(centers, region) & domain["preserve"]), region["name"]
            assert region["attachment_area_min_mm2"] > 0
            assert region["minimum_wall_mm"] >= 2
    assert label(domain["preserve"])[1] > 8


def test_loads_mass_and_fair_local_fixture_contract(domain):
    cases = {case["name"]: case for case in domain["load_cases"]}
    assert set(case["name"] for case in domain["comparison_load_cases"]) == {"arm_tip", "battery_impact", "camera_side", "modes"}
    assert len(cases["arm_tip"]["fixed_regions"]) == 4
    assert cases["arm_tip"]["loads"][0]["force_n"] == [0.0, 0.0, -1.0]
    assert cases["battery_impact"]["loads"][0]["force_n"] == pytest.approx([0, 0, -3.6284605])
    assert domain["point_masses"][0]["mass_g"] == 37
    assert domain["point_masses"][0]["position_mm"] == pytest.approx([0, 0, 34.5])
    assert len([name for name in cases if name.startswith("connection_")]) == 17
    for case in cases.values():
        for fixture in case["fixed_regions"]:
            assert fixture["max_mm"][2] < 0.1
    weights = domain["optimizer_settings"]["case_weights"]
    primary = sum(value for name, value in weights.items() if not name.startswith("connection_"))
    auxiliary = sum(value for name, value in weights.items() if name.startswith("connection_"))
    assert primary / (primary + auxiliary) == pytest.approx(0.9)


def test_parameters_are_copied_and_metadata_is_json_safe(domain):
    parameters = reference_parameters()
    original = deepcopy(parameters)
    built = build_design_domain(parameters)
    assert parameters == original
    built["material"]["density_g_cm3"] = 10
    assert parameters["material"]["density_g_cm3"] != 10
    json.dumps({key: value for key, value in domain.items() if not isinstance(value, np.ndarray)}, allow_nan=False)


def test_additional_forbidden_regions_change_free_material(domain):
    parameters = reference_parameters()
    parameters["topology"] = {"additional_regions": [{"name": "user_keepout", "role": "forbidden", "kind": "box", "min_mm": [-40, -20, 0], "max_mm": [-24, -4, 8], "purpose": "User defined component space"}]}
    changed = build_design_domain(parameters)
    assert changed["allowed"].sum() < domain["allowed"].sum()
    assert np.all(changed["allowed"] <= domain["allowed"])


def test_preserve_cuts_are_explicit_and_new_hardware_overlap_is_rejected(domain):
    declared = domain["metadata"]["declared_preserve_subtractions"]
    assert declared
    by_name = {region["name"]: region for region in domain["regions"]}
    assert all(by_name[pair["forbidden"]]["allow_preserve_subtraction"] for pair in declared)
    parameters = reference_parameters()
    parameters["topology"] = {"additional_regions": [{"name": "bad_component", "role": "forbidden", "kind": "box", "min_mm": [-16, -16, 0], "max_mm": [-10, -10, 4], "purpose": "Conflicting new hardware"}]}
    with pytest.raises(ValueError, match="Undeclared preserve/forbidden overlap"):
        build_design_domain(parameters)


@pytest.mark.parametrize("override", [{"grid": {"spacing_mm": [4, 0, 4]}}, {"grid": {"shape": [3.5, 32, 8]}}, {"grid": {"axis_order": "zyx"}}, {"manufacturing": {"minimum_feature_mm": 1.5}}])
def test_invalid_grid_or_manufacturing_inputs_are_rejected(override):
    parameters = reference_parameters()
    parameters["topology"] = override
    with pytest.raises(ValueError):
        build_design_domain(parameters)
