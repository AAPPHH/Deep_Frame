from copy import deepcopy

import numpy as np
import pytest
from build123d import GeomType, export_step, import_step

from deep_frame.topology_geometry import _volume, region_shape
from deep_frame.topology_surface import extract_density_surface, reconstruct_surface

def sphere_domain():
    shape = (14, 14, 14)
    origin = np.array([-7.0, -7.0, -7.0])
    coordinates = np.meshgrid(*(origin[i]+np.arange(shape[i])+0.5 for i in range(3)), indexing="ij")
    density = np.exp(-sum(value**2 for value in coordinates)/16)
    domain = {
        "grid": {"origin_mm": origin.tolist(), "spacing_mm": [1.0, 1.0, 1.0], "shape": list(shape), "axis_order": "xyz", "order": "C"},
        "allowed": np.ones(shape, dtype=bool), "preserve": np.zeros(shape, dtype=bool), "forbidden": np.zeros(shape, dtype=bool),
        "regions": [], "manufacturing": {"minimum_feature_mm": 2.0},
    }
    return domain, density

def test_continuous_sphere_position_volume_fidelity_and_input_immutability():
    domain, density = sphere_domain()
    before = density.copy()
    settings = {"density_threshold": 0.5, "decimation_face_budgets": [500, 1000, 2000]}
    mesh, report = extract_density_surface(domain, density, settings)
    radius = np.sqrt(-16*np.log(0.5))
    assert mesh.volume == pytest.approx(4*np.pi*radius**3/3, rel=0.03)
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), 0, atol=0.04)
    np.testing.assert_array_equal(density, before)
    assert mesh.is_volume and mesh.body_count == 1
    assert report["decimation_attempts"][-1]["passed"]
    assert report["decimation_attempts"][-1]["fidelity"]["maximum_sampled_deviation_mm"] <= 0.2
    assert settings == {"density_threshold": 0.5, "decimation_face_budgets": [500, 1000, 2000]}

@pytest.mark.parametrize("opening_radius, constraint_mode", [(0.0, "embedded"), (0.5, "embedded"), (0.5, "cad_only"), (0.5, "envelope_only"), (0.5, "envelope_forbidden")])
def test_exact_subgrid_bore_preserve_and_step_roundtrip(tmp_path, opening_radius, constraint_mode):
    domain, density = sphere_domain()
    preserve = {"name": "mount", "role": "preserve", "kind": "cylinder", "center_mm": [0, 0, 2.8], "radius_mm": 2.0, "height_mm": 1.5, "axis": "z"}
    bore = {"name": "bore", "role": "forbidden", "kind": "cylinder", "center_mm": [0, 0, 0], "radius_mm": 0.4, "height_mm": 16.0, "axis": "z", "rasterize": False}
    domain["regions"] = [preserve, bore]
    solid = reconstruct_surface(domain, density, {"density_threshold": 0.5, "manufacturing_opening_radius_mm": opening_radius, "surface_constraint_mode": constraint_mode, "decimation_bounds_mode": "reference_aabb" if constraint_mode in ("envelope_only", "envelope_forbidden") else "none", "decimation_face_budgets": [500, 1000, 2000]})
    assert solid.is_valid and len(solid.solids()) == 1
    assert any(face.geom_type == GeomType.CYLINDER for face in solid.faces())
    assert _volume(solid.intersect(region_shape(bore))) < 1e-7
    required = region_shape(preserve).cut(region_shape(bore))
    assert _volume(required.cut(solid)) < 1e-7
    assert not hasattr(solid, "topology_occupied")
    assert not hasattr(solid, "minimum_voxel_feature_mm")
    export_step(solid, tmp_path/"surface.step")
    imported = import_step(tmp_path/"surface.step")
    assert imported.is_valid and len(imported.solids()) == 1
    assert imported.volume == pytest.approx(solid.volume, rel=1e-9)

def test_disconnected_density_is_not_bridged():
    domain, density = sphere_domain()
    density[:] = 0
    density[1:4, 4:7, 4:7] = 1
    density[10:13, 4:7, 4:7] = 1
    with pytest.raises(ValueError, match="not bridged"):
        extract_density_surface(domain, density, {"density_threshold": 0.5})

@pytest.mark.parametrize("fusion_mode", ["direct", "preunion"])
def test_disconnected_exact_preserve_is_not_bridged(fusion_mode):
    domain, density = sphere_domain()
    domain["regions"] = [{"name": "disconnected", "kind": "box", "role": "preserve", "min_mm": [5, 5, 5], "max_mm": [6, 6, 6]}]
    with pytest.raises(ValueError, match="Exact preserve union must yield exactly one valid solid, got 2"):
        reconstruct_surface(domain, density, {"density_threshold": 0.5, "preserve_fusion_mode": fusion_mode, "decimation_face_budgets": [500, 1000, 2000]})

@pytest.mark.parametrize("interface_start", [3.8, 4.0])
@pytest.mark.parametrize("fusion_mode", ["direct", "preunion"])
def test_required_preserve_can_attach_through_another_required_preserve(interface_start, fusion_mode):
    domain, density = sphere_domain()
    domain["regions"] = [
        {"name": "anchored", "role": "preserve", "kind": "box", "min_mm": [2, -0.5, -0.5], "max_mm": [4, 0.5, 0.5]},
        {"name": "indirect", "role": "preserve", "kind": "box", "min_mm": [interface_start, -0.5, -0.5], "max_mm": [5, 0.5, 0.5]},
    ]
    solid = reconstruct_surface(domain, density, {"density_threshold": 0.5, "preserve_fusion_mode": fusion_mode, "decimation_face_budgets": [500, 1000, 2000]})
    assert len(solid.solids()) == 1 and solid.is_valid
    assert all(_volume(region_shape(region).cut(solid)) < 1e-7 for region in domain["regions"])
    assert solid.surface_report["preserve_union"]["mode"] == fusion_mode
    if fusion_mode == "preunion":
        assert solid.surface_report["preserve_preunion"]["status"] == "passed"

def test_reconstruction_uses_no_unprotected_overlap_booleans(monkeypatch):
    from build123d import Solid
    def forbidden_overlap(*args, **kwargs):
        raise AssertionError("Unprotected preliminary solid.intersect called")
    monkeypatch.setattr(Solid, "intersect", forbidden_overlap)
    domain, density = sphere_domain()
    domain["regions"] = [{"name": "mount", "role": "preserve", "kind": "box", "min_mm": [2, -0.5, -0.5], "max_mm": [4, 0.5, 0.5]}]
    solid = reconstruct_surface(domain, density, {"density_threshold": 0.5, "decimation_face_budgets": [500, 1000, 2000]})
    assert solid.is_valid and len(solid.solids()) == 1
    assert _volume(region_shape(domain["regions"][0]).cut(solid)) < 1e-7

def test_existing_required_preserve_can_join_density_components():
    domain, density = sphere_domain()
    coordinates = np.meshgrid(*(np.arange(14)-6.5 for _ in range(3)), indexing="ij")
    density = np.maximum(*[np.exp(-((coordinates[0]-center)**2+coordinates[1]**2+coordinates[2]**2)/3) for center in (-3.0, 3.0)])
    domain["regions"] = [{"name": "required_interface", "role": "preserve", "kind": "box", "min_mm": [-3, -0.5, -0.5], "max_mm": [3, 0.5, 0.5]}]
    solid = reconstruct_surface(domain, density, {"density_threshold": 0.5, "decimation_face_budgets": [500, 1000, 2000]})
    assert solid.surface_report["isosurface"]["bodies"] == 2
    assert solid.surface_report["initial_brep"]["solids"] == 2
    assert len(solid.solids()) == 1 and solid.is_valid
    assert _volume(region_shape(domain["regions"][0]).cut(solid)) < 1e-7

def test_inadequate_decimation_budget_is_rejected():
    domain, density = sphere_domain()
    with pytest.raises(ValueError, match="No decimation budget"):
        extract_density_surface(domain, density, {"density_threshold": 0.5, "decimation_face_budgets": [20], "maximum_surface_deviation_mm": 0.001})

@pytest.mark.parametrize("changes", [{"interpolation_subdivisions": True}, {"maximum_surface_deviation_mm": float("nan")}, {"decimation_face_budgets": [100, 50]}, {"density_threshold": 1}, {"invented_smoothing": True}, {"boolean_fuzzy_value_mm": 0}, {"boolean_fuzzy_value_mm": float("inf")}, {"boolean_strategy": "build123d", "boolean_fuzzy_value_mm": 1e-5}, {"surface_constraint_mode": "invented"}, {"decimation_bounds_mode": "invented"}, {"preserve_fusion_mode": "invented"}, {"preserve_fusion_mode": "preunion", "boolean_strategy": "build123d"}])
def test_bad_settings_fail_before_surface_generation(changes):
    domain, density = sphere_domain()
    with pytest.raises(ValueError):
        extract_density_surface(domain, density, changes)

def test_density_mask_violation_is_rejected():
    domain, density = sphere_domain()
    domain["preserve"][7, 7, 7] = True
    with pytest.raises(ValueError, match="prescribed"):
        extract_density_surface(domain, density, {})

@pytest.mark.parametrize("opening_method", ["distance", "grayscale"])
def test_manufacturing_opening_removes_thin_material_and_restores_only_prescribes(opening_method):
    from deep_frame.topology_surface import _manufacturing_opening, _settings
    spacing = np.full(3, 0.25)
    origin = np.full(3, -4.0)
    axes = np.meshgrid(*[origin[i]+np.arange(33)*spacing[i] for i in range(3)], indexing="ij")
    sphere = 2.8-np.sqrt(sum(value**2 for value in axes))
    fin = np.minimum.reduce([axes[0]-2.0, 3.75-axes[0], 0.25-np.abs(axes[1]), 0.25-np.abs(axes[2])])
    density = np.clip(0.5+np.maximum(sphere, fin), 0, 1)
    preserve = {"name": "required", "role": "preserve", "kind": "box", "min_mm": [1.75, -1, -1], "max_mm": [3.25, 1, 1]}
    cut = {"name": "bore", "role": "forbidden", "kind": "cylinder", "center_mm": [0, 0, 0], "radius_mm": 0.4, "height_mm": 9, "axis": "z"}
    domain = {"grid": {"origin_mm": origin.tolist(), "spacing_mm": spacing.tolist(), "shape": [33, 33, 33]}, "regions": [preserve, cut]}
    before = density.copy()
    opened, report = _manufacturing_opening(density, origin, spacing, domain,
                                           _settings({"density_threshold": 0.5, "manufacturing_opening_radius_mm": 1.0, "manufacturing_opening_method": opening_method}))
    np.testing.assert_array_equal(density, before)
    assert report["opening_removed_sampled_volume_mm3"] > 0
    assert report["opening_added_sampled_volume_mm3"] == report["added_sampled_volume_mm3"] == 0
    assert report["missing_preserve_sampled_volume_mm3"] == 0
    assert opened[30, 16, 16] <= 0
    assert report["preserve_restoration_stage"].startswith("Exact CAD union")
    assert opened[16, 16, 16] < 0

def test_touching_preserves_have_no_artificial_internal_zero_sheet():
    from deep_frame.topology_surface import _primitive_union_distance
    coordinates = np.arange(-3, 3.01, 0.5)
    axes = [coordinates.reshape(tuple(-1 if i == j else 1 for j in range(3))) for i in range(3)]
    boxes = [{"kind": "box", "min_mm": [-2, low, -2], "max_mm": [2, high, 2]} for low, high in ((-2, 0), (0, 2))]
    values, report = _primitive_union_distance(axes, boxes, (13, 13, 13), np.full(3, 0.5))
    assert values[6, 6, 6] > 0.001
    assert values[2, 6, 6] == 0.0
    assert values[1, 6, 6] < 0
    assert report["rectilinear_internal_zero_nodes_corrected"] == 49
    boxes[1]["min_mm"][1] = 0.0001
    gapped, gap_report = _primitive_union_distance(axes, boxes, (13, 13, 13), np.full(3, 0.5))
    assert gapped[6, 6, 6] == 0.0
    assert gap_report["rectilinear_internal_zero_nodes_corrected"] == 0

def test_cad_only_opening_excludes_analytic_imprints_and_keeps_forecast_separate():
    from deep_frame.topology_surface import _manufacturing_opening, _settings
    domain, density = sphere_domain()
    origin, spacing = np.full(3, -6.5), np.ones(3)
    before = density.copy()
    config = _settings({"density_threshold": 0.5, "manufacturing_opening_radius_mm": 1.0, "surface_constraint_mode": "cad_only"})
    unconstrained, _ = _manufacturing_opening(density, origin, spacing, domain, config)
    domain["regions"] = [
        {"name": "outside_mount", "role": "preserve", "kind": "box", "min_mm": [4, -1, -1], "max_mm": [6, 1, 1]},
        {"name": "bore", "role": "forbidden", "kind": "cylinder", "center_mm": [0, 0, 0], "radius_mm": 1.0, "height_mm": 16, "axis": "z", "rasterize": False},
    ]
    opened, report = _manufacturing_opening(density, origin, spacing, domain, config)
    np.testing.assert_array_equal(opened, unconstrained)
    np.testing.assert_array_equal(density, before)
    assert opened[6, 6, 6] > 0
    assert opened[11, 6, 6] < 0
    assert report["preserve_missing_before_opening_sampled_volume_mm3"] > 0
    assert report["diagnostic_constraints_added_to_opened_sampled_volume_mm3"] > 0
    assert report["diagnostic_constraints_removed_from_opened_sampled_volume_mm3"] > 0
    assert report["opening_added_sampled_volume_mm3"] == 0
    assert report["missing_preserve_sampled_volume_mm3"] == 0
    default_config = _settings({"density_threshold": 0.5, "manufacturing_opening_radius_mm": 1.0})
    embedded, embedded_report = _manufacturing_opening(density, origin, spacing, domain, default_config)
    assert embedded_report["surface_constraint_mode"] == "embedded"
    assert embedded[6, 6, 6] < 0
    assert not np.array_equal(embedded, opened)

def test_envelope_only_opening_leaves_hardware_for_cad_and_clips_free_field():
    from deep_frame.topology_surface import _manufacturing_opening, _settings
    domain, density = sphere_domain()
    domain["grid"]["origin_mm"][2] = 0
    origin, spacing = np.full(3, -6.5), np.ones(3)
    config = _settings({"density_threshold": 0.5, "manufacturing_opening_radius_mm": 1, "surface_constraint_mode": "envelope_only"})
    clipped, report = _manufacturing_opening(density, origin, spacing, domain, config)
    domain["regions"] = [{"name": "bore", "role": "forbidden", "kind": "cylinder", "center_mm": [0, 0, 0], "radius_mm": 1.0, "height_mm": 16, "axis": "z", "rasterize": False}]
    with_hardware, _ = _manufacturing_opening(density, origin, spacing, domain, config)
    np.testing.assert_array_equal(clipped, with_hardware)
    assert clipped[6, 6, 6] < 0
    assert clipped[6, 6, 8] > 0
    assert report["surface_constraint_mode"] == "envelope_only"
    assert report["opening_added_sampled_volume_mm3"] == 0
    free, _ = _manufacturing_opening(density, origin, spacing, domain, {**config, "surface_constraint_mode": "cad_only"})
    assert free[6, 6, 6] > 0

@pytest.mark.parametrize("rasterize", [False, True])
def test_envelope_forbidden_opening_removes_cut_generated_free_ligament_without_preserve_imprints(rasterize):
    from deep_frame.topology_surface import _manufacturing_opening, _settings
    spacing, origin = np.full(3, 0.25), np.full(3, -4.0)
    x, y, z = np.meshgrid(*[np.arange(33)*0.25-4]*3, indexing="ij")
    density = np.clip(0.5+np.minimum.reduce([3-np.abs(x), 3-np.abs(y), 1.5-np.abs(z)]), 0, 1)
    before = density.copy()
    cut = {"name": "slot", "role": "forbidden", "kind": "box", "min_mm": [-4, -4, -0.75], "max_mm": [4, 4, 4], "rasterize": rasterize}
    domain = {"grid": {"origin_mm": origin.tolist(), "spacing_mm": spacing.tolist(), "shape": [33, 33, 33]}, "regions": [cut]}
    settings = _settings({"density_threshold": 0.5, "manufacturing_opening_radius_mm": 1.25, "surface_constraint_mode": "envelope_forbidden"})
    opened, report = _manufacturing_opening(density, origin, spacing, domain, settings)
    uncut, _ = _manufacturing_opening(density, origin, spacing, domain, {**settings, "surface_constraint_mode": "envelope_only"})
    assert np.all(opened <= uncut+1e-9)
    assert uncut[16, 16, 12] > 0
    assert opened[16, 16, 12] < 0
    assert np.all(opened[z >= -0.75] <= 0)
    domain["regions"].append({"name": "required", "role": "preserve", "kind": "box", "min_mm": [-1, -1, -3], "max_mm": [1, 1, -2]})
    with_preserve, diagnostic = _manufacturing_opening(density, origin, spacing, domain, settings)
    np.testing.assert_array_equal(with_preserve, opened)
    np.testing.assert_array_equal(density, before)
    assert diagnostic["diagnostic_constraints_added_to_opened_sampled_volume_mm3"] > 0
    assert report["opening_added_sampled_volume_mm3"] == 0
    assert report["opening_removed_sampled_volume_mm3"] > 0

@pytest.mark.parametrize("rasterize", [False, True])
def test_free_forbidden_buffer_is_antiextensive_and_keeps_original_constraints(rasterize):
    from deep_frame.topology_surface import _manufacturing_opening, _settings
    spacing, origin = np.full(3, 0.25), np.full(3, -4.0)
    x, y, z = np.meshgrid(*[np.arange(33)*0.25-4]*3, indexing="ij")
    density = np.clip(0.5+3-np.sqrt(x*x+y*y+z*z), 0, 1)
    domain = {"grid": {"origin_mm": origin.tolist(), "spacing_mm": spacing.tolist(), "shape": [33, 33, 33]},
              "regions": [{"name": "bore", "role": "forbidden", "kind": "cylinder", "center_mm": [0, 0, 0], "radius_mm": 0.5, "height_mm": 9, "axis": "z", "rasterize": rasterize},
                          {"name": "contact", "role": "preserve", "kind": "box", "min_mm": [-1, -1, -1], "max_mm": [1, 1, 1]}]}
    before, domain_before = density.copy(), deepcopy(domain)
    settings = _settings({"density_threshold": 0.5, "manufacturing_opening_radius_mm": 0.5, "surface_constraint_mode": "envelope_forbidden"})
    original, original_report = _manufacturing_opening(density, origin, spacing, domain, settings)
    explicit_zero, _ = _manufacturing_opening(density, origin, spacing, domain, {**settings, "free_forbidden_buffer_mm": 0.0})
    np.testing.assert_array_equal(original, explicit_zero)
    buffered, report = _manufacturing_opening(density, origin, spacing, domain, {**settings, "free_forbidden_buffer_mm": 0.5})
    assert np.all(buffered <= original+1e-12)
    assert np.any((original > 0) & (buffered < 0))
    buffer = report["free_forbidden_buffer"]
    assert buffer["removed_sampled_volume_mm3"] > 0 and buffer["added_sampled_volume_mm3"] == 0
    assert buffer["before_buffer_sampled_volume_mm3"] == original_report["before_sampled_volume_mm3"]
    assert buffer["after_buffer_sampled_volume_mm3"] == report["before_sampled_volume_mm3"]
    assert buffer["forbidden_region_count"] == 1 and buffer["nonrasterized_forbidden_count"] == int(not rasterize)
    assert original_report["free_forbidden_buffer"]["value_mm"] == 0
    assert domain == domain_before
    np.testing.assert_array_equal(density, before)

@pytest.mark.parametrize("changes", [{"free_forbidden_buffer_mm": -0.1}, {"free_forbidden_buffer_mm": float("nan")}, {"free_forbidden_buffer_mm": float("inf")}, {"free_forbidden_buffer_mm": True},
                                     {"free_forbidden_buffer_mm": 0.5, "manufacturing_opening_radius_mm": 0.5, "surface_constraint_mode": "embedded"},
                                     {"free_forbidden_buffer_mm": 0.5, "manufacturing_opening_radius_mm": 0.5, "surface_constraint_mode": "cad_only"},
                                     {"free_forbidden_buffer_mm": 0.5, "manufacturing_opening_radius_mm": 0.5, "surface_constraint_mode": "envelope_only"},
                                     {"free_forbidden_buffer_mm": 0.5, "surface_constraint_mode": "envelope_forbidden"}])
def test_invalid_free_forbidden_buffer_fails_before_extraction(changes):
    from deep_frame.topology_surface import _settings
    with pytest.raises(ValueError, match="free_forbidden_buffer_mm"):
        _settings(changes)

def test_buffer_affects_free_field_but_final_preserve_and_bore_use_original_dimensions():
    domain, density = sphere_domain()
    preserve = {"name": "mount", "role": "preserve", "kind": "cylinder", "center_mm": [0, 0, 2.8], "radius_mm": 2, "height_mm": 1.5, "axis": "z"}
    bore = {"name": "bore", "role": "forbidden", "kind": "cylinder", "center_mm": [0, 0, 0], "radius_mm": 0.4, "height_mm": 16, "axis": "z", "rasterize": False}
    domain["regions"] = [preserve, bore]
    before_regions, before_density = deepcopy(domain["regions"]), density.copy()
    before_masks = {name: domain[name].copy() for name in ("allowed", "preserve", "forbidden")}
    solid = reconstruct_surface(domain, density, {"density_threshold": 0.5, "manufacturing_opening_radius_mm": 0.5, "surface_constraint_mode": "envelope_forbidden", "free_forbidden_buffer_mm": 0.5,
                                                 "preserve_fusion_mode": "preunion", "decimation_bounds_mode": "reference_aabb", "decimation_face_budgets": [500, 1000, 2000]})
    assert solid.is_valid and len(solid.solids()) == 1
    required = region_shape(preserve).cut(region_shape(bore))
    assert _volume(required.cut(solid)) < 1e-7
    assert _volume(solid.intersect(region_shape(bore))) < 1e-7
    assert solid.is_inside((0.65, 0, 3.3))
    assert domain["regions"] == before_regions
    np.testing.assert_array_equal(density, before_density)
    for name, original in before_masks.items():
        np.testing.assert_array_equal(domain[name], original)

def test_reference_bounds_projection_moves_only_vertices_and_preserves_valid_box():
    import trimesh
    from deep_frame.topology_surface import _project_reference_bounds, _mesh_intersections
    reference = trimesh.creation.box(extents=[2, 2, 2])
    expanded = trimesh.creation.box(extents=[2.2, 2.2, 2.2])
    original_vertices, original_faces = expanded.vertices.copy(), expanded.faces.copy()
    projected, report = _project_reference_bounds(expanded, reference)
    np.testing.assert_array_equal(expanded.vertices, original_vertices)
    np.testing.assert_array_equal(projected.faces, original_faces)
    np.testing.assert_array_equal(projected.bounds, reference.bounds)
    assert projected.is_watertight and projected.is_winding_consistent
    assert projected.euler_number == reference.euler_number
    assert _mesh_intersections(projected)["passed"]
    assert report["moved_vertex_count"] == 8
    assert report["maximum_vertex_displacement_mm"] == pytest.approx(np.sqrt(3)*0.1)
    assert report["net_volume_change_mm3"] == pytest.approx(8-2.2**3)
    assert report["degenerate_face_count"] == 0

def test_reference_bounds_projection_rejects_collapsed_triangles_without_removing_them(monkeypatch):
    import trimesh
    import deep_frame.topology_surface as module
    outside = trimesh.creation.box()
    outside.apply_translation([20, 0, 0])
    monkeypatch.setattr(trimesh.Trimesh, "simplify_quadric_decimation", lambda *args, **kwargs: outside.copy())
    domain, density = sphere_domain()
    with pytest.raises(module.SurfaceReconstructionError, match="No decimation budget") as caught:
        extract_density_surface(domain, density, {"decimation_bounds_mode": "reference_aabb", "decimation_face_budgets": [12]})
    attempt = caught.value.report["decimation_attempts"][0]
    assert attempt["reference_bounds_projection"]["degenerate_face_count"] > 0
    assert attempt["reference_bounds_projection"]["face_indices_unchanged"]
    assert attempt["mesh"]["faces"] == len(outside.faces)
    assert "fidelity" not in attempt
    assert "degenerate triangles" in attempt["reason"]

def test_boolean_dropping_the_main_body_is_rejected_even_when_output_is_valid(monkeypatch):
    from build123d import Box
    import deep_frame.topology_surface as module
    report = {"settings": module._settings({})}
    original, operand = Box(10, 10, 10), Box(1, 1, 1)
    monkeypatch.setattr(module, "_boolean", lambda *args: operand)
    with pytest.raises(module.SurfaceReconstructionError, match="volume monotonicity") as raised:
        module._checked_operation(original, [operand], "fuse", "preserve", report, None)
    assert raised.value.shape is original
    assert report["failed_operation"]["volume_monotonicity_passed"] is False

@pytest.mark.parametrize("warning", ["BOPAlgo_AlertAcquiredSelfIntersection", "BOPAlgo_AlertSolidBuilderUnusedFaces", "BOPAlgo_AlertNotSplittableEdge"])
def test_plausible_boolean_volume_does_not_override_native_geometry_warnings(monkeypatch, warning):
    from build123d import Box
    import deep_frame.topology_surface as module
    original, result = Box(10, 10, 10), Box(10, 10, 10)
    result.boolean_warnings = [warning]
    monkeypatch.setattr(module, "_boolean", lambda *args: result)
    report = {"settings": module._settings({})}
    with pytest.raises(module.SurfaceReconstructionError, match="native Boolean geometry warnings"):
        module._checked_operation(original, [Box(1, 1, 1)], "fuse", "preserve", report, None)
    assert report["failed_operation"]["blocking_warnings"] == [warning]
    assert report["exact_operations"][0]["warnings"] == [warning]

@pytest.mark.parametrize("fuzzy_value", [1e-7, 1e-5])
def test_native_boolean_owns_and_captures_warning_report(fuzzy_value):
    from build123d import Box, Pos
    import deep_frame.topology_surface as module
    result = module._boolean(Box(2, 2, 2), [Pos(1, 0, 0)*Box(2, 2, 2)], "fuse", "occt_serial", fuzzy_value)
    assert result.is_valid and len(result.solids()) == 1
    assert result.volume == pytest.approx(12)
    assert result.boolean_warnings == []
    assert result.boolean_fuzzy_value_mm == fuzzy_value

@pytest.mark.parametrize("offset, components, volume", [(1, 1, 12), (4, 2, 16)])
def test_preserve_preunion_keeps_all_overlapping_or_disjoint_components(offset, components, volume):
    from build123d import Box, Pos
    import deep_frame.topology_surface as module
    preserves = [({"name": "first"}, Box(2, 2, 2)), ({"name": "second"}, Pos(offset, 0, 0)*Box(2, 2, 2))]
    report, events = {"settings": module._settings({"preserve_fusion_mode": "preunion"})}, []
    union = module._preunion_preserves(preserves, report, events.append)
    assert len(union.solids()) == components and union.is_valid
    assert union.volume == pytest.approx(volume)
    assert all(item["passed"] and item["missing_volume_mm3"] == 0 for item in report["preserve_preunion"]["coverage"])
    assert report["preserve_preunion"]["result"]["positive_outward_components"]
    assert events[-1]["shape"] is union
    assert events[-1]["stage"] == "Exact preserve preunion"

@pytest.mark.parametrize("count", [0, 1])
def test_preserve_preunion_empty_and_single_are_identities_without_native_booleans(monkeypatch, count):
    from build123d import Box
    import deep_frame.topology_surface as module
    monkeypatch.setattr(module, "_boolean", lambda *args: pytest.fail("An identity must not perform a Boolean"))
    original = Box(2, 2, 2)
    report = {"settings": module._settings({"preserve_fusion_mode": "preunion"})}
    union = module._preunion_preserves([({"name": "single"}, original)] if count else [], report)
    assert union is (original if count else None)
    assert report["preserve_preunion"]["status"] == ("identity_single" if count else "identity_empty")
    assert module._settings({})["preserve_fusion_mode"] == "direct"

@pytest.mark.parametrize("failure", ["error", "warning", "null", "invalid", "volume", "missing", "coverage_warning"])
def test_preserve_preunion_native_failures_cannot_become_success(monkeypatch, failure):
    from build123d import Box, Pos, Face, Wire
    import deep_frame.topology_surface as module
    original = module._boolean
    first, second = Box(2, 2, 2), Pos(4, 0, 0)*Box(2, 2, 2)
    def injected(shape, tools, operation, *settings):
        if operation == "fuse":
            if failure == "error":
                raise RuntimeError("injected native failure")
            if failure == "null":
                return None
            if failure == "invalid":
                return Face(Wire.make_polygon([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
            if failure == "volume":
                return Box(10, 10, 10)
            if failure == "missing":
                return first
        result = original(shape, tools, operation, *settings)
        if failure == "warning" and operation == "fuse" or failure == "coverage_warning" and operation == "cut":
            result.boolean_warnings = ["BOPAlgo_AlertBadPositioning"]
        return result
    monkeypatch.setattr(module, "_boolean", injected)
    report = {"settings": module._settings({"preserve_fusion_mode": "preunion"})}
    with pytest.raises(module.SurfaceReconstructionError, match="Exact preserve preunion"):
        module._preunion_preserves([({"name": "first"}, first), ({"name": "second"}, second)], report)
    assert report["preserve_preunion"]["status"] == "failed"
    assert report["failed_operation"]["name"] == "Exact preserve preunion"
    if failure == "missing":
        assert report["preserve_preunion"]["coverage"][-1]["missing_volume_mm3"] == pytest.approx(8)

def test_preserve_preunion_rejects_inward_original_solid():
    from build123d import Box, Shape
    import deep_frame.topology_surface as module
    inward = Shape.cast(Box(2, 2, 2).wrapped.Reversed())
    report = {"settings": module._settings({"preserve_fusion_mode": "preunion"})}
    with pytest.raises(module.SurfaceReconstructionError, match="positive outward"):
        module._preunion_preserves([({"name": "inward"}, inward)], report)

def test_preserve_preunion_coverage_cannot_cancel_mixed_orientation_components(monkeypatch):
    from build123d import Box, Pos, Shape, Compound
    import deep_frame.topology_surface as module
    original = module._boolean
    positive = Box(2, 2, 2)
    negative = Shape.cast((Pos(10, 0, 0)*Box(2, 2, 2)).wrapped.Reversed())
    mixed = Compound(children=[positive, negative])
    assert mixed.is_valid and mixed.volume == pytest.approx(0, abs=1e-12)
    def injected(shape, tools, operation, *settings):
        return mixed if operation == "cut" else original(shape, tools, operation, *settings)
    monkeypatch.setattr(module, "_boolean", injected)
    report = {"settings": module._settings({"preserve_fusion_mode": "preunion"})}
    with pytest.raises(module.SurfaceReconstructionError, match="coverage failed"):
        module._preunion_preserves([({"name": "first"}, positive), ({"name": "second"}, Pos(1, 0, 0)*Box(2, 2, 2))], report)
    coverage = report["preserve_preunion"]["coverage"][0]
    assert coverage["missing_volume_mm3"] == pytest.approx(0, abs=1e-12)
    assert not coverage["component_orientation_passed"] and not coverage["passed"]

def test_progress_keeps_failed_decimation_evidence():
    domain, density = sphere_domain()
    events = []
    with pytest.raises(ValueError, match="No decimation budget"):
        extract_density_surface(domain, density, {"decimation_face_budgets": [20], "maximum_surface_deviation_mm": 0.001}, progress=events.append)
    assert [event["stage"] for event in events] == ["density_isosurface", "decimation_20"]
    assert events[-1]["mesh"] is not None
    assert events[-1]["report"]["decimation_attempts"][0]["passed"] is False

@pytest.mark.parametrize("example", ["plane", "negative_sphere", "subgrid_negative_cavity", "zero_plateau", "isolated_zero", "zero_filament"])
def test_internal_zero_rule_preserves_real_cavities_and_boundaries(example):
    from deep_frame.topology_surface import _regularize_internal_zero_nodes
    points = np.stack(np.meshgrid(*[np.arange(-4, 5)]*3, indexing="ij"), axis=-1)
    field = np.ones((9, 9, 9))
    if example == "plane":
        field = points[..., 0].astype(float)
    elif example == "negative_sphere":
        field = np.linalg.norm(points, axis=-1)-2
    elif example == "subgrid_negative_cavity":
        field = np.linalg.norm(points, axis=-1)-0.1
    elif example == "zero_plateau":
        field[2:7, 2:7, 2:7] = 0
    elif example == "zero_filament":
        field[2:7, 4, 4] = 0
    else:
        field[4, 4, 4] = 0
    original = field.copy()
    corrected, report = _regularize_internal_zero_nodes(field, 0.001)
    assert report["changed_node_count"] == (1 if example == "isolated_zero" else 0)
    np.testing.assert_array_equal(corrected[original < 0], original[original < 0])
    np.testing.assert_array_equal(field, original)

def test_closed_twisted_box_is_rejected_by_self_intersection_screen():
    import trimesh
    from deep_frame.topology_surface import _mesh_intersections
    mesh = trimesh.creation.box()
    mesh.vertices[mesh.vertices[:, 2] > 0, :2] *= -1
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.body_count == 1
    assert _mesh_intersections(mesh)["passed"] is False

@pytest.mark.parametrize("settings", [{}, {"surface_constraint_mode": "cad_only", "manufacturing_opening_radius_mm": 0.25}, {"surface_constraint_mode": "envelope_forbidden", "manufacturing_opening_radius_mm": 0.25}])
def test_negative_inner_shell_is_not_turned_into_positive_material(settings):
    from deep_frame.topology_surface import SurfaceReconstructionError
    domain, _ = sphere_domain()
    points = np.meshgrid(*[np.arange(14)-6.5]*3, indexing="ij")
    radius = np.sqrt(sum(axis**2 for axis in points))
    density = np.clip(0.5+np.minimum(4-radius, radius-1.25), 0, 1)
    with pytest.raises(SurfaceReconstructionError, match="negative cavity shells") as caught:
        extract_density_surface(domain, density, {"density_threshold": 0.5, **settings})
    volumes = caught.value.report["shell_signed_volumes_mm3"]
    assert len(volumes) == 2 and min(volumes) < 0 < max(volumes)
