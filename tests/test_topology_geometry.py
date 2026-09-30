from copy import deepcopy

import numpy as np
import pytest
import trimesh
from build123d import export_step, export_stl, import_step

from deep_frame.topology_geometry import reconstruct_topology, validate_topology, voxel_boxes
from deep_frame.fea import evaluate
from deep_frame.fea_config import FEA_CONFIG


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
