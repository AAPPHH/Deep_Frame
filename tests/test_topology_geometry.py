import json
from copy import deepcopy

import numpy as np
import pytest
import trimesh
from build123d import export_step, export_stl, import_step
from scipy.ndimage import label

from deep_frame.config import FEA_CONFIG, IMPLICIT_CONFIG, TOPOLOGY_CONFIG
from deep_frame.fea import evaluate
from deep_frame.frame import reference_parameters
from deep_frame.topology_geometry import build_design_domain, embed_field, grid_centers, lowered_grid, region_bounds, prescribed_clearance, rasterize_regions, reconstruct_topology, region_contains, validate_topology, voxel_boxes

@pytest.fixture(scope="module")
def domain():
    return build_design_domain(reference_parameters())

def test_domain_is_free_connected_3d_space_with_small_preserve_fraction(domain):
    assert domain["grid"]["shape"] == [34, 32, 9] and domain["grid"]["origin_mm"] == [-68.0, -64.0, -4.0]
    assert domain["grid"]["axis_order"] == "xyz"
    assert domain["grid"]["order"] == "C"
    assert domain["allowed"].dtype == bool
    assert not np.any(domain["preserve"] & domain["forbidden"])
    assert np.array_equal(domain["forbidden"], ~domain["allowed"])
    assert label(domain["allowed"])[1] == 1
    assert domain["metadata"]["free_fraction_of_allowed"] > 0.95
    assert domain["metadata"]["allowed_cells"] > domain["allowed"].size / 2
    assert np.count_nonzero(domain["allowed"].any(axis=(0, 1))) == 9
    assert len([region for region in domain["regions"] if region["role"] == "allowed"]) == 1
    assert domain["metadata"]["preserve_volume_mm3"] == domain["preserve"].sum() * 64.0

def test_no_v0_shape_call_or_shape_parameter_dependency(monkeypatch, domain):
    def fail(*args, **kwargs):
        raise AssertionError("The free domain must not build a parametric frame")
    monkeypatch.setattr("deep_frame.frame.build_frame", fail)
    monkeypatch.setattr("deep_frame.frame.build_geometry", fail)
    monkeypatch.setattr("deep_frame.fea.build_geometry", fail)
    parameters = reference_parameters()
    parameters["frame"].update(arm_width_mm=0.001, arm_root_mm=80.0, body_width_mm=1.0, body_length_mm=1.0, base_window_mm=999.0, wall_window_height_mm=999.0, deck_window_width_mm=999.0)
    changed = build_design_domain(parameters)
    assert np.array_equal(changed["allowed"], domain["allowed"])
    assert np.array_equal(changed["preserve"], domain["preserve"])

def _y_max(region):
    return region["max_mm"][1] if region["kind"] == "box" else region["center_mm"][1] + (region["height_mm"] / 2 if region["axis"] == "y" else region["radius_mm"])

def test_exact_keepouts_include_hardware_props_holes_and_assembly_access(domain):
    names = {region["name"] for region in domain["regions"]}
    assert {"aio15_envelope", "camera_envelope", "battery_envelope", "aio_side_assembly_access", "battery_insertion", "camera_front_access"} <= names
    assert not names & {"xt30_envelope", "balancer_envelope", "xt30_contact", "balancer_contact", "antenna_contact", "xt30_plug_access", "balancer_plug_access", "antenna_bore", "antenna_insertion_access", "balance_lead_routing"}
    assert {region["name"] for region in domain["regions"] if region["role"] == "preserve" and _y_max(region) < -30} == {"rear_left_motor_contact", "rear_right_motor_contact"}
    assert sum(name.endswith("_swept_clearance") for name in names) == 4
    assert sum("motor_screw_" in name for name in names) == 16
    assert sum(name.startswith("aio_screw_") for name in names) == 4
    assert "elrs_antenna_clearance" in names and not any(name.startswith(("strap_access_", "strap_contact_")) for name in names)
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
    assert set(case["name"] for case in domain["comparison_load_cases"]) == {"arm_tip", "battery_impact", "camera_side", "thrust_all", "crash_front", "crash_arm", "modes"}
    assert len(cases["arm_tip"]["fixed_regions"]) == 4
    assert cases["arm_tip"]["loads"][0]["force_n"] == pytest.approx([0.0, 0.0, -3.6])
    assert all(load["force_n"] == pytest.approx([0.0, 0.0, 3.6]) for load in cases["thrust_all"]["loads"])
    assert cases["battery_impact"]["loads"][0]["force_n"] == pytest.approx([0, 0, -3.6284605])
    assert domain["point_masses"][0]["mass_g"] == 37
    assert domain["point_masses"][0]["position_mm"] == pytest.approx([0, 0, 33.5])
    assert {name for name in cases if name.startswith("connection_")} == {f"connection_aio_contact_{index}" for index in range(4)} | {"connection_battery_rail_-1", "connection_battery_rail_1", "connection_camera_mount_-1", "connection_camera_mount_1"}
    eyes = [[float(region_bounds(eye)[0][2]) - 0.01, float(region_bounds(eye)[1][2]) + 0.01] for eye in _aio_eyes(domain)[1]]
    for case in cases.values():
        for fixture in case["fixed_regions"]:
            assert fixture["max_mm"][2] < 0.1 or [fixture["min_mm"][2], fixture["max_mm"][2]] in eyes
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

def test_tool_access_prevents_camera_voxel_slivers(domain):
    regions = {region["name"]: region for region in domain["regions"]}
    assert regions["camera_screw_axis"]["height_mm"] == pytest.approx(32)
    for sign in (-1, 1):
        lug = regions[f"camera_mount_{sign}"]
        access = regions[f"camera_tool_access_{sign}"]
        lug_outer = sign * lug["center_mm"][0] + lug["height_mm"] / 2
        access_inner = sign * access["center_mm"][0] - access["height_mm"] / 2
        assert lug_outer == access_inner == pytest.approx(16)
        assert access.get("rasterize", True)

def test_field_offsets_stay_inside_the_component_clearance():
    assert IMPLICIT_CONFIG["preserve_inflation_mm"] + IMPLICIT_CONFIG["constraint_offset_mm"] < TOPOLOGY_CONFIG["component_clearance_mm"]

@pytest.mark.parametrize("spacing", [2.0, 4 / 3, 1.0])
def test_battery_rails_and_deck_loads_resolve_on_every_study_grid(spacing):
    from deep_frame.topology_optimization import HexElasticity
    parameters = reference_parameters()
    parameters["topology"] = {"grid": {**TOPOLOGY_CONFIG["grid"], "spacing_mm": [spacing] * 3, "shape": [round(136 / spacing), round(128 / spacing), round(32 / spacing)]}}
    built = build_design_domain(parameters)
    centers = grid_centers(built["grid"])
    for region in (region for region in built["regions"] if region["name"].startswith("battery_rail_")):
        assert np.any(region_contains(centers, region) & built["preserve"])
    model = HexElasticity(built, "preserve_adjacent")
    deck = next(case for case in built["load_cases"] if case["name"] == "battery_impact")["loads"][0]["region"]
    assert deck["min_mm"][2] < parameters["frame"]["deck_top_mm"] < deck["max_mm"][2] and len(model._select(deck, "battery_impact", "load")) > 0
    assert model.selector_expansions == []

def _aio_eyes(domain):
    regions = {region["name"]: region for region in domain["regions"]}
    return regions, [regions[f"aio_contact_{index}"] for index in range(4)]

def _check_eyes(domain, seat, low_z):
    regions, eyes = _aio_eyes(domain)
    floor, tolerance = domain["grid"]["origin_mm"][2], reference_parameters()["integration"]["selection_tolerance_mm"]
    centers = grid_centers(domain["grid"])
    for index, eye in enumerate(eyes):
        low, high = region_bounds(eye)
        assert eye["radius_mm"] == pytest.approx(3.1) and low[2] == pytest.approx(low_z) and high[2] == pytest.approx(seat + TOPOLOGY_CONFIG["flush_overlap_mm"])
        assert np.any(region_contains(centers, eye) & domain["preserve"]) and not (domain["preserve"] & (centers[..., 2] < low[2]) & region_contains(centers, {**eye, "height_mm": 100.0})).any()
        if low_z > floor:
            tool = regions[f"aio_tool_access_{index}"]
            tool_low, tool_high = region_bounds(tool)
            assert tool["role"] == "forbidden" and tool.get("rasterize", True) and tool["radius_mm"] == pytest.approx(TOPOLOGY_CONFIG["aio_tool_radius_mm"])
            assert tool["center_mm"][:2] == eye["center_mm"][:2] and tool_high[2] == pytest.approx(low[2]) and tool_low[2] == pytest.approx(floor - 1)
            assert not (domain["allowed"] & region_contains(centers, tool)).any()
    for case in ("arm_tip", "crash_front"):
        boxes = next(item for item in domain["comparison_load_cases"] if item["name"] == case)["fixed_regions"]
        assert len(boxes) == 4 and all(box["min_mm"][2] == pytest.approx(low_z - tolerance) and box["max_mm"][2] == pytest.approx(seat + TOPOLOGY_CONFIG["flush_overlap_mm"] + tolerance) for box in boxes)
    assert domain["metadata"]["prescribed_clearance"]["passed"]

def test_aio_contacts_are_grid_snapped_seat_eyes_with_tool_access_and_eye_fixtures(domain):
    _check_eyes(domain, 5.5, 0.0)

def test_aio_seat_eyes_on_fine_grids_leave_the_floor_free():
    parameters = reference_parameters()
    parameters["frame"]["aio_standoff_mm"] = 8.8
    parameters["topology"] = {"grid": {"shape": [102, 96, 24], "spacing_mm": [4 / 3] * 3}}
    _check_eyes(build_design_domain(parameters), 11.3, 8.0)

@pytest.mark.parametrize("spacing, cells", [(4.0, 1), (2.0, 2), (4 / 3, 3), (2 / 3, 6), (0.75, 6)])
def test_floor_drop_keeps_spacing_and_lowers_the_floor_by_whole_cells(spacing, cells):
    grid = {"origin_mm": [-68.0, -64.0, 0.0], "spacing_mm": [spacing] * 3, "shape": [10, 10, 12]}
    lowered = lowered_grid(grid, TOPOLOGY_CONFIG["floor_drop_mm"])
    assert lowered["spacing_mm"] == grid["spacing_mm"] and lowered["shape"] == [10, 10, 12 + cells] and grid["shape"] == [10, 10, 12]
    assert lowered["origin_mm"][2] == pytest.approx(-cells * spacing) and -lowered["origin_mm"][2] >= TOPOLOGY_CONFIG["floor_drop_mm"] - 1e-9
    assert lowered["origin_mm"][2] + spacing * lowered["shape"][2] == pytest.approx(12 * spacing)

def test_embed_field_pads_old_fields_below_their_floor():
    field = np.zeros((2, 3, 4))
    field[:, :, 0], field[:, :, 3] = 0.7, 1.0
    embedded = embed_field(field, {"shape": [2, 3, 6]})
    assert embedded.shape == (2, 3, 6) and np.array_equal(embedded[:, :, 2:], field) and np.all(embedded[:, :, :2] == 0.7)
    with pytest.raises(ValueError):
        embed_field(field, {"shape": [2, 3, 3]})

def test_lowered_floor_leaves_room_below_the_old_floor_and_bores_reach_it(domain):
    floor = domain["grid"]["origin_mm"][2]
    centers = grid_centers(domain["grid"])
    below = centers[..., 2] < 0
    assert floor <= -TOPOLOGY_CONFIG["floor_drop_mm"] and domain["allowed"][below].mean() > 0.9 and not domain["preserve"][below].any()
    envelope = next(region for region in domain["regions"] if region["name"] == "design_envelope")
    assert envelope["min_mm"][2] == pytest.approx(floor)
    for region in domain["regions"]:
        if "motor_screw_" in region["name"] or region["name"].endswith("_shaft_clearance") or region["name"].startswith("aio_screw_"):
            assert region_bounds(region)[0][2] == pytest.approx(floor - 1)

def test_prescribed_preserves_keep_two_millimetre_walls_against_keepouts(domain):
    clearance = domain["metadata"]["prescribed_clearance"]
    assert clearance["passed"] and clearance["required_width_mm"] == pytest.approx(2.1)
    regions = {region["name"]: region for region in domain["regions"]}
    assert regions["battery_rail_1"]["max_mm"][2] == pytest.approx(28.5)
    assert regions["front_left_motor_contact"]["radius_mm"] - regions["front_left_motor_screw_0"]["radius_mm"] - 4.5 == pytest.approx(2.1)
    extended = {pair["preserve"] for pair in domain["metadata"]["flush_contact_extensions"]}
    assert {"battery_rail_1", "front_left_motor_contact"} | {f"aio_contact_{index}" for index in range(4)} <= extended and not extended & {"xt30_contact", "balancer_contact", "antenna_contact"}
    assert regions["aio_contact_0"]["height_mm"] == pytest.approx(6.0) and regions["aio_contact_0"]["radius_mm"] == pytest.approx(3.1) and "flush_kept_near_wall" not in clearance
    keepout = {"name": "k", "role": "forbidden", "kind": "box", "min_mm": [0, 0, 4], "max_mm": [10, 10, 8], "purpose": ""}
    pad = {"name": "p", "role": "preserve", "kind": "box", "min_mm": [2, 2, 0], "max_mm": [8, 8, 4], "purpose": ""}
    rim = {"name": "r", "role": "preserve", "kind": "box", "min_mm": [8, 0, 4], "max_mm": [11.9, 10, 6], "purpose": ""}
    near = {"name": "n", "role": "preserve", "kind": "cylinder", "center_mm": [20, 5, 2], "radius_mm": 3, "height_mm": 4, "axis": "z", "purpose": ""}
    other = {"name": "o", "role": "preserve", "kind": "box", "min_mm": [23.5, 0, 0], "max_mm": [30, 10, 4], "purpose": ""}
    violations = prescribed_clearance([keepout, pad, rim, near, other], 2.0, 0.1, 0.3)["violations"]
    assert [violation["rule"] for violation in violations] == ["flush_contact", "rim_width", "preserve_gap"]
    assert violations[1]["width_mm"] == pytest.approx(1.9) and violations[2]["gap_mm"] == pytest.approx(0.5)
    fixed = [keepout, dict(pad, max_mm=[8, 8, 4.5]), dict(rim, max_mm=[12.1, 10, 6]), near, dict(other, min_mm=[26, 0, 0])]
    assert prescribed_clearance(fixed, 2.0, 0.1, 0.3)["passed"]

@pytest.mark.parametrize("override", [{"grid": {"spacing_mm": [4, 0, 4]}}, {"grid": {"shape": [3.5, 32, 8]}}, {"grid": {"axis_order": "zyx"}}, {"manufacturing": {"minimum_feature_mm": 1.5}}])
def test_invalid_grid_or_manufacturing_inputs_are_rejected(override):
    parameters = reference_parameters()
    parameters["topology"] = override
    with pytest.raises(ValueError):
        build_design_domain(parameters)

def spatial_loop():
    mask = np.zeros((16, 16, 10), dtype=bool)
    mask[2:14, 2:5, 1:3] = True
    mask[2:5, 2:14, 1:3] = True
    mask[2:5, 11:14, 1:8] = True
    mask[2:14, 11:14, 6:8] = True
    mask[11:14, 2:14, 6:8] = True
    mask[11:14, 2:5, 1:8] = True
    domain = {
        "grid": {"origin_mm": [0, 0, 0], "spacing_mm": [4, 4, 4], "shape": list(mask.shape), "axis_order": "xyz", "order": "C"},
        "allowed": np.ones(mask.shape, dtype=bool),
        "preserve": np.zeros(mask.shape, dtype=bool),
        "forbidden": np.zeros(mask.shape, dtype=bool),
        "regions": [],
        "manufacturing": {"nozzle_width_mm": 0.4, "minimum_wall_nozzles": 4, "minimum_feature_mm": 4, "minimum_attachment_area_mm2": 20, "supports_allowed": True, "build_direction": [0, 0, 1]},
    }
    return domain, mask.astype(float)

def test_nonplanar_density_reconstructs_exact_volume_and_closed_exports(tmp_path):
    domain, field = spatial_loop()
    solid = reconstruct_topology(domain, field, {})
    assert solid.volume == pytest.approx(float(field.sum()) * 64)
    assert solid.bounding_box().size.Z == pytest.approx(28)
    validation = validate_topology(solid, domain, {})
    assert validation["passed"], validation["violations"]
    assert solid.topology_report["cuboid_count"] == 6
    export_step(solid, tmp_path / "free.step")
    export_stl(solid, tmp_path / "free.stl")
    assert len(import_step(tmp_path / "free.step").solids()) == 1
    mesh = trimesh.load_mesh(tmp_path / "free.stl")
    assert mesh.is_volume and mesh.body_count == 1

def test_exact_preserved_mount_bore_survives_subvoxel_reconstruction():
    domain, field = spatial_loop()
    domain["preserve"][2:5, 2:5, 1:3] = True
    domain["regions"] = [
        {"name": "mount", "kind": "box", "role": "preserve", "purpose": "local bolt mount", "min_mm": [8, 8, 4], "max_mm": [20, 20, 12], "minimum_wall_mm": 2, "attachment_area_min_mm2": 20},
        {"name": "m2_hole", "kind": "cylinder", "role": "forbidden", "purpose": "subvoxel M2 clearance", "center_mm": [14, 14, 8], "radius_mm": 1.1, "height_mm": 10, "axis": "z", "rasterize": False},
    ]
    solid = reconstruct_topology(domain, field, {})
    result = validate_topology(solid, domain, {})
    assert result["passed"], result["violations"]
    assert result["checks"]["forbidden"]["m2_hole"]["intersection_mm3"] < 1e-6
    assert result["checks"]["preserve"]["mount"]["missing_volume_mm3"] < 1e-6
    assert solid.volume == pytest.approx(float(field.sum()) * 64 - np.pi * 1.1**2 * 8)

def test_greedy_cuboids_cover_each_voxel_once():
    rng = np.random.default_rng(18)
    mask = rng.random((7, 8, 9)) > 0.6
    counts = np.zeros(mask.shape, dtype=int)
    for start, stop in voxel_boxes(mask):
        counts[tuple(slice(first, last) for first, last in zip(start, stop))] += 1
    np.testing.assert_array_equal(counts, mask.astype(int))

def test_disconnected_required_mounts_are_rejected_without_bridging():
    domain, field = spatial_loop()
    field[0, 0, 0] = 1
    domain["preserve"][0, 0, 0] = True
    domain["preserve"][2, 2, 1] = True
    with pytest.raises(ValueError, match="Disconnected required"):
        reconstruct_topology(domain, field, {"remove_unanchored_islands": True})

def test_optional_unanchored_island_removal_is_explicit_and_logged():
    domain, field = spatial_loop()
    field[0, 0, 0] = 1
    domain["preserve"][2, 2, 1] = True
    original = field.copy()
    with pytest.raises(ValueError, match="face-connected"):
        reconstruct_topology(domain, field, {})
    solid = reconstruct_topology(domain, field, {"remove_unanchored_islands": True})
    assert solid.topology_report["repairs"] == [{"method": "remove_unanchored_islands", "removed_voxels": 1, "removed_components": 1}]
    np.testing.assert_array_equal(field, original)

def test_unresolved_minimum_feature_and_thin_mount_ligament_are_invalid():
    domain, field = spatial_loop()
    solid = reconstruct_topology(domain, field, {})
    domain["manufacturing"]["minimum_feature_mm"] = 5
    assert not validate_topology(solid, domain, {})["passed"]
    domain["manufacturing"]["minimum_feature_mm"] = 4
    domain["regions"] = [
        {"name": "thin_mount", "kind": "box", "role": "preserve", "purpose": "mount", "min_mm": [8, 8, 4], "max_mm": [20, 20, 12], "minimum_wall_mm": 2},
        {"name": "thin_hole", "kind": "cylinder", "role": "forbidden", "purpose": "bore", "center_mm": [10, 14, 8], "radius_mm": 1.1, "height_mm": 10, "axis": "z", "rasterize": False},
    ]
    solid = reconstruct_topology(domain, field, {})
    result = validate_topology(solid, domain, {})
    assert "preserve:thin_mount" in result["violations"]

def test_nonconservative_hardware_keepout_mask_is_rejected():
    domain, field = spatial_loop()
    domain["regions"] = [{"name": "hardware", "kind": "box", "role": "forbidden", "purpose": "component clearance", "min_mm": [8.1, 8.1, 4.1], "max_mm": [9, 9, 5]}]
    with pytest.raises(ValueError, match="Nonconservative"):
        reconstruct_topology(domain, field, {})

def test_exact_subvoxel_box_cut_cannot_hide_a_half_millimeter_web():
    domain, field = spatial_loop()
    domain["regions"] = [{"name": "adversarial_exact_slot", "kind": "box", "role": "forbidden", "purpose": "thin residual web", "min_mm": [8.5, 8, 3], "max_mm": [19, 20, 13], "rasterize": False}]
    solid = reconstruct_topology(domain, field, {})
    result = validate_topology(solid, domain, {})
    assert not result["passed"]
    rays = result["checks"]["features"]["post_boolean_wall_screen"]
    assert rays["minimum_measured_mm"] < 1
    assert rays["thin_sample_count"] > 0

def test_inaccessible_closed_support_cavity_is_rejected():
    domain, field = spatial_loop()
    field[:] = 1
    field[5:10, 5:10, 3:7] = 0
    solid = reconstruct_topology(domain, field, {})
    result = validate_topology(solid, domain, {})
    assert not result["passed"]
    access = result["checks"]["supports"]["accessibility"]
    assert access["closed_cad_cavities"] == 1
    assert access["trapped_void_voxels"] == 100

def test_nonplanar_reconstruction_reaches_independent_gmsh_and_calculix(tmp_path):
    domain, field = spatial_loop()
    solid = reconstruct_topology(domain, field, {})
    fixed = {"kind": "box", "min_mm": [7.99, 7.99, 3.99], "max_mm": [20.01, 20.01, 4.01]}
    tip = {"kind": "box", "min_mm": [43.99, 43.99, 31.99], "max_mm": [56.01, 56.01, 32.01]}
    cases = [
        {"name": "vertical_tip", "analysis": "static", "fixed_regions": [fixed], "loads": [{"region": tip, "force_n": [0, 0, -1]}]},
        {"name": "modes", "analysis": "modal", "fixed_regions": [fixed]},
    ]
    settings = deepcopy(FEA_CONFIG["settings"])
    settings.update(mesh_second_order_linear=True, work_dir=str(tmp_path), stiffness_load_case="vertical_tip")
    result = evaluate(solid, FEA_CONFIG["material"], [], cases, settings)
    assert result["status"] == "ok", result["diagnostics"]
    assert result["mesh"]["minimum_jacobian_mm3"] > 0
    assert result["max_displacement_mm"] > 0
    assert result["max_von_mises_mpa"] > 0
    assert result["eigenfrequencies_hz"][0] > 0
    assert result["mass_g"] == pytest.approx(21.34656)

def diagonal_contact_field():
    domain, _ = spatial_loop()
    shape = (5, 5, 2)
    domain["grid"]["shape"] = list(shape)
    domain.update(allowed=np.ones(shape, dtype=bool), preserve=np.zeros(shape, dtype=bool), forbidden=np.zeros(shape, dtype=bool))
    field = np.zeros(shape)
    for x, y in [(1, 1), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (2, 2)]:
        field[x, y, :] = 1
    return domain, field

def test_optional_manifold_repair_is_density_guided_and_logged():
    domain, field = diagonal_contact_field()
    field[2, 1, :] = 0.3
    solid = reconstruct_topology(domain, field, {"repair_manifold_voxels": True, "maximum_repair_voxels": 4})
    result = validate_topology(solid, domain, {})
    assert result["passed"], result["violations"]
    repairs = solid.topology_report["repairs"]
    assert repairs and all(item["method"] == "density_guided_local_manifold_fill" for item in repairs)
    assert repairs[0]["cell_xyz"][:2] == [2, 1]
    assert repairs[0]["density_before"] == 0.3
    assert sum(item["added_volume_mm3"] for item in repairs) > 0

def test_manifold_repair_never_fills_forbidden_cells_or_exceeds_limit():
    domain, field = diagonal_contact_field()
    with pytest.raises(ValueError, match="maximum_repair_voxels"):
        reconstruct_topology(domain, field, {"repair_manifold_voxels": True, "maximum_repair_voxels": 0})
    domain["allowed"][1, 2, :] = False
    domain["allowed"][2, 1, :] = False
    domain["forbidden"] = ~domain["allowed"]
    with pytest.raises(ValueError, match="forbidden cells"):
        reconstruct_topology(domain, field, {"repair_manifold_voxels": True, "maximum_repair_voxels": 100})

def test_build_plate_face_alone_does_not_require_support():
    domain, field = spatial_loop()
    field[:] = 1
    domain["manufacturing"]["supports_allowed"] = False
    solid = reconstruct_topology(domain, field, {})
    result = validate_topology(solid, domain, {})
    assert result["passed"], result["violations"]
    assert not result["checks"]["supports"]["supports_required"]
