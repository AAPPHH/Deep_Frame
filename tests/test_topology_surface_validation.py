from hashlib import sha256

import numpy as np
import pytest
import trimesh
from build123d import Box, Cylinder, GeomType, Pos, Rot, Sphere
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.BOPAlgo import BOPAlgo_BOP, BOPAlgo_AlertTooSmallEdge
from OCP.TopoDS import TopoDS_Shape
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

from deep_frame.topology_surface_validation import SURFACE_VALIDATION_SETTINGS, _CadRayIndex, _accessibility, _attachment_screen, _intersection_volume, _mesh, _prescribed_triangles, _preserve_neighborhood, _self_intersection_screen, _settings, _wall_screen, surface_metrics, validate_surface
from deep_frame import topology_surface_validation as validation

def domain():
    return {"grid": {"origin_mm": [-8, -8, -8], "spacing_mm": [2, 2, 2], "shape": [8, 8, 8]}, "regions": [], "manufacturing": {"nozzle_width_mm": 0.4, "minimum_wall_nozzles": 5, "minimum_feature_mm": 2, "minimum_attachment_area_mm2": 4, "supports_allowed": True, "build_direction": [0, 0, 1]}}

def native_geometry_and_cache_signature(shape):
    digest = sha256()
    digest.update(np.asarray([shape.volume], dtype="<f8").tobytes())
    for collection in (shape.vertices(), shape.edges(), shape.faces()):
        for item in collection:
            digest.update(np.asarray([BRep_Tool.Tolerance_s(item.wrapped)], dtype="<f8").tobytes())
    for vertex in shape.vertices():
        digest.update(np.asarray(BRep_Tool.Pnt_s(vertex.wrapped).Coord(), dtype="<f8").tobytes())
    for face in shape.faces():
        poly = BRep_Tool.Triangulation_s(face.wrapped, TopLoc_Location())
        digest.update(str(face.wrapped.Orientation()).encode())
        if poly is None:
            digest.update(b"no triangulation")
        else:
            digest.update(np.asarray([poly.Deflection()], dtype="<f8").tobytes())
            digest.update(np.asarray([poly.Node(i).Coord() for i in range(1, poly.NbNodes() + 1)], dtype="<f8").tobytes())
            digest.update(np.asarray([triangle.Get() for triangle in poly.Triangles()], dtype="<i8").tobytes())
    return digest.hexdigest()

def maximum_cylinder_chord_deflection(vertices, faces, radius):
    triangles = np.asarray(vertices)[np.asarray(faces)]
    side = triangles[np.ptp(triangles[:, :, 2], axis=1) > 1e-8]
    midpoints = (side + np.roll(side, -1, axis=1)) / 2
    return float(np.max(radius - np.linalg.norm(midpoints[:, :, :2], axis=2)))

@pytest.mark.parametrize("radius", [1, 10, 100])
def test_absolute_cad_deflection_is_scale_independent_for_whole_shape_and_wall_face(radius):
    solid = Cylinder(radius, 5)
    solid.tessellate(0.5, 1.2)
    before = native_geometry_and_cache_signature(solid)
    settings = _settings({"tessellation_mm": 0.03, "tessellation_angle_rad": 1.2})
    mesh = _mesh(solid, settings)
    report = mesh.metadata["adaptive_tessellation"]
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.body_count == 1
    assert 0.005 < maximum_cylinder_chord_deflection(mesh.vertices, mesh.faces, radius) <= 0.03
    assert report["deflection_mode"] == "absolute_mm" and not report["occt_is_relative"]
    assert report["requested_deflection_mm"] == report["used_deflection_mm"] == 0.03
    assert report["native_triangulation_deflection_range_mm"][1] <= 0.03
    assert report["cad_face_count"] == report["triangulated_cad_face_count"] == 3
    assert not report["input_cache_reused"] and not report["input_shape_modified"]
    face = next(item for item in solid.faces() if item.geom_type == GeomType.CYLINDER)
    vertices, triangles, tolerance, failures = validation._face_tessellation(face, settings)
    assert 0.005 < maximum_cylinder_chord_deflection(vertices, triangles, radius) <= 0.03
    assert tolerance == 0.03 and failures == []
    assert native_geometry_and_cache_signature(solid) == before

def test_absolute_tessellation_keeps_uncached_input_and_native_rigid_transform():
    solid = Pos(3, 4, 5) * Rot(20, 35, 15) * Box(2, 4, 6)
    assert all(BRep_Tool.Triangulation_s(face.wrapped, TopLoc_Location()) is None for face in solid.faces())
    before = native_geometry_and_cache_signature(solid)
    native_vertices = np.asarray([BRep_Tool.Pnt_s(vertex.wrapped).Coord() for vertex in solid.vertices()])
    mesh = _mesh(solid, _settings({}))
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(48, abs=1e-10)
    assert np.max(np.min(np.linalg.norm(mesh.vertices[:, None, :] - native_vertices[None, :, :], axis=2), axis=1)) < 1e-12
    assert native_geometry_and_cache_signature(solid) == before

def test_wall_report_records_absolute_face_tessellation_without_changing_physical_sampling():
    result = _wall_screen(Box(4, 4, 4), 2, _settings({}))
    assert result["passed"] and result["complete"]
    assert result["maximum_sample_triangle_edge_mm"] == 1
    assert result["minimum_required_mm"] == 2
    assert result["tessellation"]["deflection_mode"] == "absolute_mm"
    assert result["tessellation"]["used_deflection_face_counts"] == {"0.03": 6}
    assert not result["tessellation"]["input_cache_reused"]

def test_real_native_positive_sliver_is_preserved_closed_with_unit_normal_at_face_zero():
    vertices = np.array([[56.090258081750825, -32.45107194235511, 4.000023782246933], [56.09021101619348, -32.45106162751226, 4.000047563798097], [56.090305147242, -32.45108225749, 4.0], [56.09, -32.45, 5.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    original = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert not original.nondegenerate_faces()[0]
    assert np.array_equal(original.face_normals[0], [0, 0, 0])
    mesh = validation._native_mesh(vertices, faces)
    assert len(mesh.faces) == 4 and mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.euler_number == 2 and mesh.body_count == 1
    assert np.array_equal(mesh.triangles, vertices[faces])
    assert np.allclose(np.linalg.norm(mesh.face_normals, axis=1), 1, rtol=0, atol=1e-12)
    assert mesh.metadata["native_triangle_preservation"]["minimum_triangle_area_mm2"] == pytest.approx(1.861525714847333e-14, rel=1e-6, abs=0)
    assert mesh.metadata["native_triangle_preservation"]["positive_area_faces_removed"] == 0
    assert mesh.metadata["native_triangle_preservation"]["exact_repeated_coordinate_simplices_removed"] == 0

@pytest.mark.parametrize("kind", ["zero", "duplicate", "repeated", "invalid_index", "nonfinite"])
def test_invalid_native_triangles_fail_with_diagnostics_instead_of_deletion(kind):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]])
    if kind == "zero":
        vertices[2] = [0.5, 0, 0]
    elif kind == "duplicate":
        faces = np.array([[0, 1, 2], [2, 1, 0]])
    elif kind == "repeated":
        vertices[2] = vertices[0]
    elif kind == "invalid_index":
        faces[0, 2] = 3
    else:
        vertices[2, 0] = np.nan
    with pytest.raises(validation._InvalidNativeMesh) as caught:
        validation._native_mesh(vertices, faces)
    assert caught.value.report["faces_removed"] == 0
    assert caught.value.report["reason"]

def test_exact_identification_does_not_merge_nearby_distinct_native_vertices():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1e-10, 0.], [0., 0., 0.]])
    faces = np.array([[3, 1, 2]])
    mesh = validation._native_mesh(vertices, faces)
    assert len(mesh.vertices) == 3
    assert np.array_equal(mesh.triangles, vertices[faces])
    assert mesh.face_normals[0] == pytest.approx([0, 0, 1])

def test_zero_area_native_face_fails_validation_without_refinement_or_silent_drop(monkeypatch):
    calls = []
    def zero_tessellation(*args):
        calls.append(args)
        return [(0., 0., 0.), (1., 0., 0.), (0.5, 0., 0.)], [(0, 1, 2)], {"untriangulated_faces": []}
    monkeypatch.setattr(validation, "_absolute_tessellation", zero_tessellation)
    result = validate_surface(Box(4, 4, 4), domain(), {})
    assert not result["passed"] and len(calls) == 1
    assert result["violations"] == ["topology:surface_tessellation_failed"]
    assert result["checks"]["topology"]["native_mesh_diagnostic"]["invalid_triangle_ids_first_100"] == [0]

def test_only_exact_repeated_coordinate_simplices_are_logged_without_hiding_a_hole():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    mesh = validation._native_mesh(vertices, np.array([[0, 1, 2], [0, 3, 1]]), [2])
    assert not mesh.is_watertight
    assert mesh.metadata["native_triangle_preservation"]["removed_zero_simplex_native_triangle_ids"] == [1]
    assert mesh.metadata["native_triangle_preservation"]["positive_area_faces_removed"] == 0
    with pytest.raises(validation._InvalidNativeMesh, match="coverage"):
        validation._native_mesh(vertices, np.array([[0, 1, 2], [0, 3, 1]]), [1, 1])

def test_sphere_pole_zero_simplices_are_logged_and_positive_surface_stays_closed():
    mesh = _mesh(Sphere(4), _settings({}))
    record = mesh.metadata["native_triangle_preservation"]
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.body_count == 1
    assert record["positive_area_faces_removed"] == 0
    assert record["exact_repeated_coordinate_simplices_removed"] > 0
    assert record["retained_triangle_count"] + record["exact_repeated_coordinate_simplices_removed"] == record["native_triangle_count"]

def test_exact_collinearity_cannot_become_positive_area_from_scaled_edge_rounding():
    vertices = np.array([[0., 0., 0.], [3., 7., 13.], [9., 21., 39.]])
    with pytest.raises(validation._InvalidNativeMesh, match="exactly collinear"):
        validation._native_mesh(vertices, [[0, 1, 2]])

def test_zero_simplex_orphan_vertex_does_not_inflate_closed_body_count():
    box = trimesh.creation.box(extents=[4, 4, 4])
    vertices = np.vstack([box.vertices, [[30., 40., 50.]]])
    faces = np.vstack([box.faces, [[0, 0, len(vertices) - 1]]])
    mesh = validation._native_mesh(vertices, faces)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.body_count == 1
    assert mesh.volume == pytest.approx(64)
    assert np.array_equal(mesh.triangles, box.triangles)
    assert mesh.metadata["native_triangle_preservation"]["unreferenced_vertices_compacted"] == 1
    assert mesh.metadata["native_triangle_preservation"]["exact_repeated_coordinate_simplices_removed"] == 1

def disjoint_real_native_pair():
    vertices = np.array([[-18.66864304847, -4.883609670952, 0.001001911567877], [-12.751777026374823, -9.55000049340984, 0.0010338142340638077], [-12.863989121397593, -9.552030881918494, 0.0010332121516310963], [-18.66824315489, -1.747172369733, 0.001001811687905], [-12.527484307643602, -9.557745817348632, 0.00103501807747591], [-12.63956274552688, -9.551906253277675, 0.001034416456265861]])
    return trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False)

def test_real_native_meshlab_false_positive_requires_exact_all_pair_certificate():
    mesh = disjoint_real_native_pair()
    result = _self_intersection_screen(mesh)
    assert result["complete"] and result["passed"]
    assert result["self_intersecting_face_count"] == 2
    assert result["self_intersecting_faces_first_100"] == [0, 1]
    refined = result["exact_selected_pair_refinement"]
    assert refined["complete"] and refined["passed"]
    assert refined["pair_count_required"] == len(refined["certificates"]) == 1
    assert refined["certificates"][0]["faces"] == [0, 1]
    assert refined["certificates"][0]["strictly_disjoint"]
    assert result["input_mesh_unchanged"] and result["native_mesh_unchanged"]

def test_real_crossing_selects_both_faces_and_exact_refinement_cannot_accept():
    mesh = trimesh.Trimesh(vertices=[[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, -0.5, -1], [0, -0.5, 1], [0, 0.5, 0]], faces=[[0, 1, 2], [3, 4, 5]], process=False)
    result = _self_intersection_screen(mesh)
    assert result["complete"] and not result["passed"]
    assert result["self_intersecting_faces_first_100"] == [0, 1]
    assert result["exact_selected_pair_refinement"]["unresolved_pairs"] == [[0, 1]]

@pytest.mark.parametrize("count", [1, 17])
def test_exact_si_refinement_single_selection_and_computation_cap_fail_closed(count):
    triangles = np.repeat(np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]), count, axis=0)
    result = validation._strict_selected_pair_certificates(triangles, np.arange(count))
    assert not result["passed"] and not result["complete"]
    assert not result["certificates"]

def test_exact_si_refinement_must_certify_every_pair_and_cannot_accept_contact():
    first = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    triangles = np.array([first, first + [5, 5, 0], first + [-1, 0, 0]])
    result = validation._strict_selected_pair_certificates(triangles, [0, 1, 2])
    assert result["complete"] and not result["passed"]
    assert result["pair_count_required"] == 3 and len(result["certificates"]) == 2
    assert result["unresolved_pairs"] == [[0, 2]]

def test_exact_si_refinement_nonfinite_input_and_unpinned_library_fail_closed(monkeypatch):
    mesh = disjoint_real_native_pair()
    triangles = mesh.triangles.copy()
    triangles[0, 0, 0] = np.nan
    result = validation._strict_selected_pair_certificates(triangles, [0, 1])
    assert not result["passed"] and not result["complete"]
    actual_version = validation.version
    monkeypatch.setattr(validation, "version", lambda name: "unreviewed" if name == "pymeshlab" else actual_version(name))
    result = _self_intersection_screen(mesh)
    assert result["complete"] and not result["passed"]
    assert result["self_intersecting_face_count"] == 2
    assert not result["exact_selected_pair_refinement"]["complete"]

def test_exact_projection_separation_needs_no_coplanarity_assumption():
    triangles = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[2., 0., -1.], [3., 0., 1.], [2., 1., 2.]]])
    result = validation._strict_selected_pair_certificates(triangles, [0, 1])
    assert result["passed"] and result["complete"]
    assert result["certificates"][0]["projection_axes"] == [0, 1]

def test_exact_si_predicate_error_cannot_turn_selection_into_acceptance(monkeypatch):
    class BrokenExactArithmetic:
        @staticmethod
        def from_float(value):
            raise RuntimeError("injected exact arithmetic failure")
    monkeypatch.setattr(validation, "Fraction", BrokenExactArithmetic)
    result = validation._strict_selected_pair_certificates(disjoint_real_native_pair().triangles, [0, 1])
    assert not result["passed"] and not result["complete"]
    assert "injected" in result["reason"]

@pytest.mark.parametrize("fuzzy_mm", [1e-7, 1e-5])
def test_native_verified_empty_difference_is_distinct_from_csg_failure(fuzzy_mm):
    result = validation._checked_csg("cut", Box(2, 2, 2), Box(4, 4, 4), fuzzy_mm=fuzzy_mm)
    assert result.is_valid
    assert len(result.solids()) == len(result.faces()) == 0
    assert validation._volume(result) == 0

@pytest.mark.parametrize("severity", ["warning", "error"])
def test_native_alert_on_empty_difference_cannot_be_a_zero_volume_pass(monkeypatch, severity):
    class AlertedOperation(BOPAlgo_BOP):
        def Perform(self):
            super().Perform()
            alert = BOPAlgo_AlertTooSmallEdge(TopoDS_Shape())
            if severity == "warning":
                self.AddWarning(alert)
            else:
                self.AddError(alert)
    monkeypatch.setattr(validation, "BOPAlgo_BOP", AlertedOperation)
    events = []
    result = validate_surface(Box(2, 2, 2), domain(), {}, progress=events.append)
    assert not result["passed"]
    assert result["violations"] == ["csg:unresolved"]
    diagnostic = result["checks"]["csg"]
    assert diagnostic["native_alerts"][severity + "s"] == ["BOPAlgo_AlertTooSmallEdge"]
    assert not diagnostic["complete"]
    assert not any("volume" in key for key in diagnostic)
    assert events[-1]["status"] == "unresolved"

def test_null_native_result_without_alert_is_unresolved_not_empty(monkeypatch):
    class NullOperation(BOPAlgo_BOP):
        def Shape(self):
            return TopoDS_Shape()
    monkeypatch.setattr(validation, "BOPAlgo_BOP", NullOperation)
    with pytest.raises(validation._UnresolvedCSG, match="unresolved") as caught:
        validation._checked_csg("cut", Box(2, 2, 2), Box(4, 4, 4))
    assert "null" in caught.value.report["reason"]

def test_explicit_boolean_numerics_are_reported_without_changing_physical_criteria():
    result = validate_surface(Sphere(4), domain(), {"boolean_fuzzy_mm": 1e-5}, reference_solid=Box(10, 10, 10))
    assert result["passed"], result["violations"]
    assert result["numerical_method"]["boolean_fuzzy_mm"] == 1e-5
    assert result["numerical_method"]["default_boolean_fuzzy_mm"] == 1e-7
    assert {key: value for key, value in result["settings"].items() if key != "boolean_fuzzy_mm"} == {key: value for key, value in SURFACE_VALIDATION_SETTINGS.items() if key != "boolean_fuzzy_mm"}

def test_final_bulge_outside_envelope_cannot_hide_behind_voxel_metadata():
    solid = Pos(7, 0, 0) * Box(4, 4, 4)
    solid.topology_report = {"minimum_voxel_feature_mm": 4}
    solid.topology_occupied = np.ones((1, 1, 1), dtype=bool)
    result = validate_surface(solid, domain(), {})
    assert "envelope" in result["violations"]
    assert result["checks"]["envelope"]["outside_grid_box_mm3"] == pytest.approx(16)

def test_final_geometry_preserve_and_keepout_are_checked_without_source_reports():
    specification = domain()
    specification["regions"] = [
        {"name": "mount", "kind": "box", "role": "preserve", "min_mm": [-2, -2, -2], "max_mm": [2, 2, 2]},
        {"name": "clearance", "kind": "box", "role": "forbidden", "min_mm": [2.5, -1, -1], "max_mm": [3.5, 1, 1]},
    ]
    solid = Box(8, 6, 6).cut(Pos(-1.5, 0, 0) * Box(1, 2, 8))
    result = validate_surface(solid, specification, {})
    assert "preserve:mount" in result["violations"]
    assert "forbidden:clearance" in result["violations"]

def test_physical_face_sampling_detects_local_thin_web_on_large_face():
    solid = Box(12, 12, 4).cut(Pos(4, 0, 1.25) * Box(2, 2, 2.5))
    result = _wall_screen(solid, 2, _settings({}))
    assert not result["passed"]
    assert result["minimum_measured_mm"] < 1.6
    assert result["thin_sample_count"] > 0
    assert result["maximum_sample_triangle_edge_mm"] == 1

def test_closed_cavity_detected_from_final_cad_despite_stale_occupancy():
    solid = Box(12, 12, 12).cut(Box(4, 4, 4))
    solid.topology_occupied = np.ones((2, 2, 2), dtype=bool)
    result = _accessibility(solid, domain(), _settings({}))
    assert not result["passed"]
    assert result["closed_cad_cavities"] == 1
    assert result["trapped_void_cells"] == 64

def test_finite_prescribed_boundary_does_not_exclude_protruding_triangle():
    specification = domain()
    specification["regions"] = [{"kind": "box", "role": "preserve", "name": "mount", "min_mm": [-1, -1, 0], "max_mm": [1, 1, 2]}]
    triangles = np.array([[[-4, -0.5, 2], [4, -0.5, 2], [0, 0.5, 2]], [[-0.5, -0.5, 2], [0.5, -0.5, 2], [0, 0.5, 2]]])
    mask = _prescribed_triangles(triangles, np.array([[0, 0, 1], [0, 0, 1]]), specification, _settings({}))
    assert mask.tolist() == [False, True]

def test_coplanar_tessellation_edges_do_not_count_as_roughness():
    mesh = trimesh.creation.box(extents=[4, 4, 4])
    first = surface_metrics(mesh, domain())
    refined = surface_metrics(mesh.subdivide().subdivide(), domain())
    assert first["free_sharp_edge_length_mm"] == pytest.approx(48)
    for key in ("free_surface_area_mm2", "free_sharp_edge_length_mm", "free_axis_normal_area_fraction"):
        assert first[key] == pytest.approx(refined[key])

def test_equal_mass_can_still_have_large_material_relocation():
    reference = Box(4, 4, 4)
    candidate = Pos(1, 0, 0) * Box(4, 4, 4)
    result = validate_surface(candidate, domain(), {}, reference_solid=reference)
    change = result["material_change"]
    assert change["net_volume_change_mm3"] == pytest.approx(0, abs=1e-8)
    assert change["added_volume_mm3"] == pytest.approx(16)
    assert change["removed_volume_mm3"] == pytest.approx(16)
    assert change["symmetric_difference_volume_mm3"] == pytest.approx(32)
    assert "surface_maturity" in result["violations"]

def test_continuous_sphere_passes_surface_and_wall_screens_without_voxel_data():
    reference = Box(10, 10, 10)
    result = validate_surface(Sphere(4), domain(), {}, reference_solid=reference)
    assert result["passed"], result["violations"]
    assert result["checks"]["features"]["minimum_measured_mm"] is None
    assert result["checks"]["features"]["minimum_screened_thickness_lower_bound_mm"] == pytest.approx(2.00001)
    assert result["checks"]["features"]["rays_clear_through_upper_bound"] == result["checks"]["features"]["ray_count"]
    assert result["checks"]["surface_maturity"]["passed"]

def test_tangent_material_difference_with_native_orientation_warning_is_unresolved():
    result = validate_surface(Sphere(4), domain(), {}, reference_solid=Box(8, 8, 8))
    assert result["violations"] == ["csg:unresolved"]
    assert "BOPAlgo_AlertUnableToOrientTheShape" in result["checks"]["csg"]["native_alerts"]["warnings"]
    assert "material_change" not in result

def test_invalid_and_weakened_criteria_fail_before_geometry_work():
    for changes in ({"wall_sample_spacing_mm": float("nan")}, {"maximum_wall_samples": True}, {"maximum_free_axis_fraction_ratio": 0.8}, {"maximum_void_columns": 0.1}, {"misspelled_tolerance": 1}):
        with pytest.raises(ValueError):
            validate_surface(None, domain(), changes)
    assert SURFACE_VALIDATION_SETTINGS["maximum_free_axis_fraction_ratio"] == 0.5

def test_exact_local_crop_contains_outermost_slab_and_preserves_all_sections():
    solid = Box(12, 4, 4).fuse(Box(4, 12, 4), Box(4, 4, 12))
    outer = Box(4, 4, 4)
    settings = _settings({})
    local, inclusion = _preserve_neighborhood(solid, outer, settings)
    global_result = _attachment_screen(solid, outer, {}, domain()["manufacturing"], settings)
    local_result = _attachment_screen(local, outer, {}, domain()["manufacturing"], settings)
    assert np.all(np.asarray(inclusion["crop_min_mm"]) < inclusion["required_min_mm"])
    assert np.all(np.asarray(inclusion["crop_max_mm"]) > inclusion["required_max_mm"])
    for first, second in zip(global_result["levels"], local_result["levels"]):
        assert first["sections_mm2"] == pytest.approx(second["sections_mm2"], abs=1e-8)
    assert global_result["passed"] == local_result["passed"]
    assert _intersection_volume(solid, Pos(20, 0, 0) * Box(4, 4, 4)) == 0

@pytest.mark.parametrize("thickness", [1.5, 1.99998, 1.999995, 2.0, 2.000005, 2.00002])
def test_bounded_exact_cad_index_preserves_thickness_decision_at_tolerance_boundary(thickness):
    solid = Box(4, 4, thickness)
    point, direction = (0, 0, thickness / 2), (0, 0, -1)
    original = IntCurvesFace_ShapeIntersector()
    original.Load(solid.wrapped, 1e-7)
    original.PerformNearest(gp_Lin(gp_Pnt(*point), gp_Dir(*direction)), 1e-5, 20)
    expected = min(original.WParameter(i) for i in range(1, original.NbPnt() + 1))
    actual = _CadRayIndex(solid).intersections(point, direction, 1e-5, 2.00001)
    assert bool(actual and min(actual) < 1.99999) == (expected < 1.99999)
    if expected <= 2.00001:
        assert min(actual) == pytest.approx(expected, abs=1e-8)
    else:
        assert not actual

@pytest.mark.parametrize("shape,point,direction", [
    ("sphere", (0, 0, 1), (np.sqrt(1 - 0.001 ** 2), 0, -0.001)),
    ("sphere", (0, 0, 1), (np.sqrt(1 - 0.99 ** 2), 0, -0.99)),
    ("box", (0, 0, 0.999999), (1, 1e-14, -1e-8)),
    ("box", (2, 2, 1), (0, 0, -1)),
])
def test_cad_index_keeps_grazing_curved_and_face_boundary_intersections(shape, point, direction):
    solid = Sphere(1) if shape == "sphere" else Box(4, 4, 2)
    original = IntCurvesFace_ShapeIntersector()
    original.Load(solid.wrapped, 1e-7)
    original.PerformNearest(gp_Lin(gp_Pnt(*point), gp_Dir(*direction)), 1e-5, 2.00001)
    expected = [original.WParameter(i) for i in range(1, original.NbPnt() + 1) if 1e-5 < original.WParameter(i) <= 2.00001]
    actual = _CadRayIndex(solid).intersections(point, direction, 1e-5, 2.00001)
    assert bool(actual) == bool(expected)
    if expected:
        assert min(actual) == pytest.approx(min(expected), abs=1e-8)

def test_incomplete_bounded_screen_cannot_pass():
    result = _wall_screen(Sphere(3), 2, _settings({"maximum_wall_samples": 2}))
    assert not result["passed"]
    assert not result["complete"]
    assert result["reason"] == "wall sample budget exceeded"

def test_inward_oriented_closed_valid_solid_fails_topology_screen():
    solid = Box(4, 4, 4)
    solid.wrapped.Reverse()
    result = validate_surface(solid, domain(), {})
    topology = result["checks"]["topology"]
    assert topology["valid"] and topology["closed"] and topology["mesh_watertight"]
    assert topology["mesh_winding_consistent"]
    assert not topology["outward_oriented"]
    assert topology["mesh_signed_volume_mm3"] < 0
    assert not result["passed"]
    assert result["violations"] == ["topology:closed_manifold_single_solid"]

def test_spatial_index_retains_all_entry_exit_pairs_for_curves_and_cavities():
    for solid in (Sphere(3), Box(8, 8, 8).cut(Box(4, 4, 4))):
        index = _CadRayIndex(solid)
        original = IntCurvesFace_ShapeIntersector()
        original.Load(solid.wrapped, 1e-7)
        for x in (-3.5, -1.5, 0.5, 3.5):
            point = (x, 0.25, -10)
            original.Perform(gp_Lin(gp_Pnt(*point), gp_Dir(0, 0, 1)), 0, 20)
            expected = sorted(original.WParameter(i) for i in range(1, original.NbPnt() + 1))
            actual = sorted(index.intersections(point, (0, 0, 1), 0, 20, nearest=False))
            assert actual == pytest.approx(expected, abs=1e-8)

def test_micro_sliver_inside_legacy_origin_exclusion_cannot_pass_as_clear():
    solid = Box(4, 4, 1e-6)
    assert solid.is_valid and solid.volume > 1e-5
    result = _wall_screen(solid, 2, _settings({}))
    assert not result["passed"]
    assert result["minimum_measured_mm"] == pytest.approx(1e-6)
    assert result["thin_sample_count"] > 0
    assert result["ray_origin_exclusion_mm"] == 1e-7

def test_origin_guard_rejects_ambiguous_face_edge_even_when_long_ray_is_clear():
    solid = Box(4, 4, 4)
    index = _CadRayIndex(solid)
    source = next(i for i, face in enumerate(index.faces) if face.center().Z == pytest.approx(2))
    assert index.origin_is_resolved((0, 0, 2), (0, 0, -1), source)
    assert not index.origin_is_resolved((2, 0, 2), (-1, 0, 0), source)

def test_closed_consistent_positive_volume_single_body_can_self_intersect():
    mesh = trimesh.creation.box(extents=(4, 4, 4))
    mesh.vertices[7] = [-3, -1, -1]
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.body_count == 1 and mesh.euler_number == 2 and mesh.volume > 0
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    result = _self_intersection_screen(mesh)
    assert result["complete"]
    assert not result["passed"]
    assert result["self_intersecting_face_count"] > 0
    assert result["input_mesh_unchanged"]
    assert np.array_equal(vertices, mesh.vertices) and np.array_equal(faces, mesh.faces)

@pytest.mark.parametrize("kind", ["box", "cylinder"])
def test_valid_cad_box_and_cylinder_pass_self_intersection_screen(kind):
    solid = Box(4, 4, 4) if kind == "box" else Cylinder(3, 4)
    mesh = _mesh(solid, _settings({}))
    result = _self_intersection_screen(mesh)
    assert result["passed"], result
    assert result["complete"] and result["self_intersecting_face_count"] == 0
    assert result["input_mesh_unchanged"]
    assert len(result["mesh_vertices_float64_le_sha256"]) == 64
    assert len(result["mesh_faces_int64_le_sha256"]) == 64
