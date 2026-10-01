import numpy as np
import pytest
import trimesh

from deep_frame.config import TOPOLOGY_CONFIG
from deep_frame.topology_implicit import ImplicitError, ImplicitField, _trimesh, build_implicit, exact_booleans, primitive_distance, region_manifold, remesh
from deep_frame.topology_implicit_validation import VALIDATION_CHECKS, MeshAcceptance, ball_curvature, prescribed_bores, validate_implicit, wall_screen
from deep_frame.topology_surface_validation import _settings as _validation_settings, surface_metrics
from tests.test_topology_implicit import CONFIG, SMALL, box, cylinder, make_domain

SETTINGS = _validation_settings({})

def tilted(mesh, degrees):
    return mesh.copy().apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), [1, 0.3, 0]))

@pytest.mark.parametrize("degrees", [0.0, 30.0])
@pytest.mark.parametrize("thickness, passed", [(1.9, False), (1.99, False), (2.0, True), (2.1, True)])
def test_wall_screen_detects_slabs_thinner_than_minimum(thickness, passed, degrees):
    report = wall_screen(tilted(trimesh.creation.box(extents=(12, 10, thickness)), degrees), 2.0, SETTINGS, CONFIG["remesh_feature_deg"])
    assert report["complete"] and report["passed"] is passed and report["unresolved_sample_count"] == 0
    assert (report["thin_sample_count"] > 0) is not passed and report["ray_count"] > 200
    if thickness <= 2.0:
        assert report["minimum_measured_mm"] == pytest.approx(thickness, abs=1e-9)
    else:
        assert report["minimum_measured_mm"] is None and report["rays_clear_through_upper_bound"] == report["ray_count"]

def test_wall_screen_rejects_thin_rod_and_exhausted_budget():
    rod = wall_screen(trimesh.creation.cylinder(radius=0.95, height=12, sections=48), 2.0, SETTINGS, CONFIG["remesh_feature_deg"])
    assert not rod["passed"] and rod["thin_sample_count"] > 0 and rod["minimum_measured_mm"] < 1.9
    budget = wall_screen(trimesh.creation.box(extents=(12, 10, 3)), 2.0, {**SETTINGS, "maximum_wall_samples": 50}, CONFIG["remesh_feature_deg"])
    assert not budget["passed"] and not budget["complete"]

def test_wall_screen_follows_the_surface_on_remeshed_roundings_but_keeps_creases():
    rounded = ImplicitField.from_function([-4.5, -4.5, -3.5], 0.25, (37, 37, 29), lambda x, y, z: primitive_distance([x, y, z], box("b", "preserve", [-3.7, -3.7, -2.7], [3.7, 3.7, 2.7]))+0.3)
    block = remesh(rounded.extract()[0], CONFIG)
    facets = wall_screen(block, 2.0, SETTINGS, 0.0)
    report = wall_screen(block, 2.0, SETTINGS, CONFIG["remesh_feature_deg"])
    assert facets["complete"] and not facets["passed"] and facets["thin_sample_count"] > 0
    assert report["passed"] and report["thin_sample_count"] == 0 and report["unresolved_sample_count"] == 0
    half = np.tan(np.radians(10.0))*12
    wedge = trimesh.convex.convex_hull([(x, y, z) for x, y in ((0, 0), (12, -half), (12, half)) for z in (0, 10)])
    knife = wall_screen(wedge, 2.0, SETTINGS, CONFIG["remesh_feature_deg"])
    assert not knife["passed"] and knife["thin_sample_count"] > 0 and knife["minimum_measured_mm"] < 1.0

def bore_block(distance, slab=None):
    regions = [cylinder("shaft", "forbidden", [0, 0, 2], 1.4, 6.0, rasterize=False), cylinder("screw", "forbidden", [distance, 0, 2], 1.1, 6.0, rasterize=False)]
    body = region_manifold(box("block", "preserve", [-5, -5, 0], [10, 5, 4]), 0.01)[0]
    for region in regions:
        body = body-region_manifold(region, 0.01, 0.0, 1, 1)[0]
    if slab is not None:
        body = body+region_manifold(box("fin", "preserve", [12, -5, 0], [12+slab, 5, 4]), 0.01)[0]
    return wall_screen(_trimesh(body), 2.0, SETTINGS, CONFIG["remesh_feature_deg"], prescribed_bores(regions, 0.01, 0.0))

def test_prescribed_bore_web_tolerates_only_the_polygon_oversize():
    web = bore_block(4.5)
    allowance = web["prescribed_bore_web"]["maximum_allowance_mm"]
    assert web["passed"] and web["thin_sample_count"] == 0 and web["prescribed_bore_web"]["sample_count"] > 0
    assert 2.0-allowance-1e-5 <= web["prescribed_bore_web"]["minimum_measured_mm"] < 2.0-1e-5 and allowance == pytest.approx(region_manifold(cylinder("a", "forbidden", [0, 0, 0], 1.4, 1.0), 0.01)[1]["oversize_mm"]+region_manifold(cylinder("b", "forbidden", [0, 0, 0], 1.1, 1.0), 0.01)[1]["oversize_mm"])
    narrow = bore_block(4.47)
    assert not narrow["passed"] and narrow["thin_sample_count"] > 0
    free = bore_block(4.5, 1.98)
    assert not free["passed"] and free["minimum_measured_mm"] == pytest.approx(1.98, abs=1e-9) and all(sample["position_mm"][0] >= 12-1e-9 for sample in free["thin_samples"])

@pytest.mark.parametrize("protected", [False, True])
def test_protected_final_opening_removes_shell_fin_beside_an_inset_keepout(protected):
    regions = [box("m1", "preserve", [2, 3, 0], [6, 7, 6]), box("m2", "preserve", [24, 3, 0], [28, 7, 6]), box("cap", "forbidden", [2.2, 2.8, 6], [6.2, 7.2, 8])]
    domain = make_domain((30, 10, 8), regions)
    density = np.where(domain["preserve"], 1.0, 0.0)
    density[6:24, 3:7, 1:5] = 1.0
    density[domain["forbidden"]] = 0.0
    mesh, report, field = build_implicit(domain, density, {**SMALL, "protected_opening": protected})
    walls = wall_screen(mesh, 2.0, SETTINGS, CONFIG["remesh_feature_deg"])
    assert walls["complete"] and walls["unresolved_sample_count"] == 0 and report["final_witness"]["passed"]
    assert (walls["thin_sample_count"] == 0) is protected and (protected or walls["minimum_measured_mm"] < 0.1)
    assert mesh.is_watertight and mesh.body_count == 1 and mesh.bounds[1][2] <= 8.0

def test_capped_mount_leaves_no_shell_fin_beside_its_keepout():
    regions = [box("m1", "preserve", [2, 3, 0], [6, 7, 6]), box("m2", "preserve", [24, 3, 0], [28, 7, 6]), box("cap", "forbidden", [1.8, 2.8, 6], [6.2, 7.2, 8])]
    domain = make_domain((30, 10, 8), regions)
    density = np.where(domain["preserve"], 1.0, 0.0)
    density[6:24, 3:7, 1:5] = 1.0
    density[domain["forbidden"]] = 0.0
    mesh, report, field = build_implicit(domain, density, SMALL)
    assert report["preserve_shell"]["capped"] == {"m1": ["cap"]} and report["final_witness"]["passed"]
    near = (mesh.vertices[:, 0] < 7) & (mesh.vertices[:, 0] > 1)
    assert mesh.vertices[near, 2].max() <= 6.0
    walls = wall_screen(mesh, 2.0, SETTINGS, CONFIG["remesh_feature_deg"])
    assert walls["complete"] and walls["unresolved_sample_count"] == 0 and walls["thin_sample_count"] == len(walls["thin_samples"]) and not [sample for sample in walls["thin_samples"] if sample["position_mm"][2] > 3]

def acceptance(mesh, regions, shape=(20, 12, 8)):
    domain = make_domain(shape, regions)
    domain["manufacturing"] = TOPOLOGY_CONFIG["manufacturing"]
    checker = MeshAcceptance(mesh, domain, CONFIG, SETTINGS)
    checker.topology()
    return checker

BLOCK = trimesh.creation.box(bounds=[[1, 1, 1], [10, 10, 5]])

@pytest.mark.parametrize("region", [box("k", "forbidden", [9.95, 0, 0], [15, 12, 8]), box("k", "forbidden", [9.998, 3, 2], [15, 6, 4]), cylinder("k", "forbidden", [11.99, 5.5, 3], 2.0, 3.0)])
def test_keepout_penetration_fails_volume_and_sampled_distance(region):
    row = acceptance(BLOCK, [region]).forbidden()["k"]
    assert not row["passed"] and row["intersection_mm3"] > SETTINGS["volume_tolerance_mm3"] and row["maximum_penetration_mm"] > CONFIG["penetration_tolerance_mm"]
    assert row["stl_maximum_penetration_mm"] > CONFIG["penetration_tolerance_mm"]

@pytest.mark.parametrize("region", [box("k", "forbidden", [10.01, 0, 0], [15, 12, 8]), box("k", "forbidden", [10, 0, 0], [15, 12, 8]), cylinder("k", "forbidden", [12.01, 5.5, 3], 2.0, 3.0)])
def test_keepout_clearance_or_tangency_passes(region):
    row = acceptance(BLOCK, [region]).forbidden()["k"]
    assert row["passed"] and row["intersection_mm3"] <= SETTINGS["volume_tolerance_mm3"]

def binary32_domain():
    regions = [box("aio", "forbidden", [-16.15, -16.15, -1], [16.15, 16.15, 20]), box("pad", "preserve", [-26.3, -3.3, 0.5], [-18.3, 4.7, 10.5]), cylinder("bore", "forbidden", [-22.3, 0.7, 5.5], 1.4, 30.0, rasterize=False)]
    domain = make_domain((40, 30, 16), regions, origin=(-35.0, -15.0, 0.0))
    domain["manufacturing"] = TOPOLOGY_CONFIG["manufacturing"]
    return domain

def test_keepout_on_a_binary32_inexact_plane_is_checked_on_the_delivered_stl():
    domain = binary32_domain()
    row = MeshAcceptance(trimesh.creation.box(bounds=[[-30, -10, 0.5], [-16.15, 10, 10.5]]), domain, CONFIG, SETTINGS).forbidden()["aio"]
    assert not row["passed"] and row["intersection_by_body_mm3"]["float64"] == 0 and row["intersection_by_body_mm3"]["stl"] > SETTINGS["volume_tolerance_mm3"]
    final, report = exact_booleans(trimesh.creation.box(bounds=[[-30, -10, 0.5], [-10, 10, 10.5]]), domain, CONFIG)
    assert report["passed"] and np.array_equal(final.vertices, final.vertices.astype(np.float32))
    assert final.vertices[:, 0].max() == float(np.nextafter(np.float32(-16.15), np.float32(-np.inf)))
    checker = MeshAcceptance(final, domain, CONFIG, SETTINGS)
    forbidden, preserve = checker.forbidden(), checker.preserve()
    assert all(row["passed"] and max(row["intersection_by_body_mm3"].values()) <= SETTINGS["volume_tolerance_mm3"] for row in forbidden.values())
    assert preserve["pad"]["passed"] and max(preserve["pad"]["missing_by_body_mm3"].values()) <= SETTINGS["volume_tolerance_mm3"]

def test_envelope_detects_material_outside():
    assert acceptance(BLOCK, []).envelope()["passed"]
    report = acceptance(trimesh.creation.box(bounds=[[1, 1, -0.1], [10, 10, 5]]), []).envelope()
    assert not report["passed"] and report["outside_grid_box_mm3"] == pytest.approx(8.1) and report["stl_maximum_vertex_outside_mm"] == pytest.approx(0.1)

def test_cavity_fails_topology_and_accessibility():
    inner = trimesh.creation.box(bounds=[[3, 3, 2], [7, 7, 4]])
    inner.invert()
    hollow = trimesh.Trimesh(np.vstack((BLOCK.vertices, inner.vertices)), np.vstack((BLOCK.faces, inner.faces+8)), process=False)
    checker = acceptance(hollow, [])
    assert not checker.topology()["passed"] and checker.cavities == 1
    accessibility = checker.supports()["accessibility"]
    assert not accessibility["passed"] and accessibility["trapped_void_cells"] > 0

def test_columns_lying_in_a_step_plane_are_reshot_not_left_unresolved():
    from manifold3d import Manifold
    step = _trimesh(Manifold.cube([3.5, 7, 4]).translate([1, 1, 1])+Manifold.cube([3.5, 7, 2]).translate([4.5, 1, 1]))
    accessibility = acceptance(step, []).supports()["accessibility"]
    assert accessibility["passed"] and accessibility["jittered_columns"] > 0 and accessibility["unresolved_columns"] == 0

def mounted_domain():
    regions = [box("m1", "preserve", [1, 5, 4], [8, 12, 10]), box("m2", "preserve", [40, 5, 4], [47, 12, 10]), cylinder("bore", "forbidden", [4.5, 8.5, 7], 1.1, 8.0, rasterize=False), box("keepout", "forbidden", [20, 0, 12], [28, 17, 20])]
    domain = make_domain((48, 17, 14), regions)
    domain["manufacturing"] = TOPOLOGY_CONFIG["manufacturing"]
    density = np.where(domain["preserve"], 1.0, 0.0)
    density[8:40, 6:11, 5:9] = 1.0
    density[domain["forbidden"]] = 0.0
    return domain, density

@pytest.fixture(scope="module")
def mounted():
    domain, density = mounted_domain()
    mesh, report, field = build_implicit(domain, density, SMALL)
    return domain, mesh, report, field

def easy_reference(mesh, domain):
    metrics = surface_metrics(mesh, domain)
    return {key: 4*metrics[key]+0.1 for key in ("free_sharp_edge_length_per_area_per_mm", "free_axis_normal_area_fraction")}

def test_connected_mounts_pass_every_acceptance_check(mounted):
    domain, mesh, report, field = mounted
    result = validate_implicit(mesh, domain, field, report, reference_metrics=easy_reference(mesh, domain))
    assert result["passed"] and not result["violations"] and set(result["checks"]) == set(VALIDATION_CHECKS)
    checks = result["checks"]
    assert all(checks["connectivity"]["witnesses"].values()) and checks["connectivity"]["mesh_body_count"] == 1
    walls = checks["features"]["mesh_wall_screen"]
    assert checks["features"]["field_opening"]["passed"] and (walls["minimum_measured_mm"] is None or walls["minimum_measured_mm"] >= 2.0-SETTINGS["wall_tolerance_mm"])
    assert checks["deviation"]["free_zone"]["candidate_to_reference"]["free_sample_count"] >= CONFIG["free_zone_minimum_samples"]
    assert checks["forbidden"]["bore"]["passed"] and checks["preserve"]["m1"]["missing_volume_mm3"] <= SETTINGS["volume_tolerance_mm3"]
    assert result["settings"]["maximum_wall_samples"] == CONFIG["maximum_wall_samples"]
    assert checks["surface_maturity"]["passed"] and checks["gates"]["values"]["surface_deviation_mm"] == CONFIG["surface_deviation_mm"]
    assert checks["envelope"]["outside_grid_box_by_body_mm3"]["stl"] <= SETTINGS["volume_tolerance_mm3"] and np.array_equal(mesh.vertices, mesh.vertices.astype(np.float32))

def test_surface_maturity_fails_closed(mounted):
    domain, mesh, report, field = mounted
    checker = MeshAcceptance(mesh, domain, CONFIG, SETTINGS)
    assert not checker.maturity(None)["passed"] and "No reference" in checker.maturity(None)["reason"]
    metrics = checker.metrics
    blocky = {key: metrics[key]*1.5 for key in ("free_sharp_edge_length_per_area_per_mm", "free_axis_normal_area_fraction")}
    assert not checker.maturity(blocky)["passed"] and not checker.maturity({key: 0.0 for key in blocky})["passed"]
    assert checker.maturity({key: 4*value+0.1 for key, value in blocky.items()})["passed"]

def test_weakened_report_settings_fail_validation(mounted):
    domain, mesh, report, field = mounted
    result = validate_implicit(mesh, domain, field, {**report, "settings": {**report["settings"], "surface_deviation_mm": 25.0}}, reference_metrics=easy_reference(mesh, domain))
    assert not result["passed"] and "gates" in result["violations"] and "cannot be weakened" in result["checks"]["gates"]["reason"]
    result = validate_implicit(mesh, domain, field, report, settings={"volume_tolerance_mm3": 1.0}, reference_metrics=easy_reference(mesh, domain))
    assert not result["passed"] and "gates" in result["violations"] and result["settings"]["volume_tolerance_mm3"] == SETTINGS["volume_tolerance_mm3"]

def test_unevaluable_checks_fail(mounted):
    domain, mesh, report, field = mounted
    result = validate_implicit(mesh, domain, None, report)
    assert not result["passed"] and {"features", "deviation"} <= set(result["violations"])
    assert result["checks"]["deviation"]["reason"].startswith("Not evaluable")
    sparse = validate_implicit(mesh, domain, field, {**report, "settings": {**report["settings"], "free_zone_minimum_samples": 10**9}}, reference_metrics=easy_reference(mesh, domain))
    assert sparse["violations"] == ["deviation"] and "not evaluable" in sparse["checks"]["deviation"]["free_zone"]["reason"]

def test_raw_density_deviation_is_recorded_but_not_gated(mounted):
    domain, mesh, report, field = mounted
    shifted = ImplicitField(field.origin, field.spacing, field.values)
    shifted.layers = {**field.layers, "reference": np.roll(field.layers["reference"], 2, axis=1)}
    deviation = MeshAcceptance(mesh, domain, {**CONFIG, "free_zone_minimum_samples": 500}, SETTINGS).deviation(shifted, report)
    assert deviation["passed"] and deviation["free_zone"]["maximum_deviation_mm"] <= CONFIG["surface_deviation_mm"]
    assert not deviation["free_zone"]["density_diagnostic"]["gated"] and deviation["free_zone"]["density_diagnostic"]["maximum_deviation_mm"] > 0.4
    assert not deviation["free_zone"]["density_diagnostic"]["within_limit"] and deviation["free_zone"]["density_diagnostic"]["limit_mm"] == CONFIG["surface_deviation_mm"]
    reopened = np.zeros(field.values.shape, dtype=bool)
    reopened[:, : field.values.shape[1]//2] = True
    shifted.layers["reopening_modified"] = reopened
    split = MeshAcceptance(mesh, domain, {**CONFIG, "free_zone_minimum_samples": 1}, SETTINGS).deviation(shifted, report)["free_zone"]
    both, first = split["density_diagnostic"], split["density_diagnostic_first_opening"]
    assert not first["gated"] and first["reopening_modified_sample_fraction"] == pytest.approx(reopened.mean())
    assert first["excluded_sample_fraction"] < both["excluded_sample_fraction"] and first["free_area_fraction"] > both["free_area_fraction"]

def test_extraction_error_against_processed_field_fails_the_gate(mounted):
    domain, mesh, report, field = mounted
    moved = mesh.copy().apply_translation([0.0, 0.0, 0.3])
    deviation = MeshAcceptance(moved, domain, CONFIG, SETTINGS).deviation(field, report)
    assert not deviation["passed"] and not deviation["free_zone"]["passed"] and deviation["free_zone"]["evaluable"]
    assert 0.25 < deviation["free_zone"]["maximum_deviation_mm"] < 0.4

def test_disconnected_mount_never_reaches_validation():
    domain, density = mounted_domain()
    density[30:34] = 0.0
    with pytest.raises(ImplicitError) as error:
        build_implicit(domain, density, SMALL)
    assert error.value.status == "mount_disconnected" and error.value.mesh is None

def open_domain():
    return {"grid": {"origin_mm": [-30.0, -30.0, -30.0], "spacing_mm": [0.7, 0.7, 0.7], "shape": [86, 86, 86]}, "regions": []}

def test_ball_curvature_matches_analytic_mean_curvature():
    sphere = ball_curvature(trimesh.creation.icosphere(subdivisions=5, radius=10), 1.0, 1000)
    cylinder = ball_curvature(trimesh.creation.cylinder(radius=5, height=40, sections=256), 1.0, 2000)
    block = ball_curvature(trimesh.creation.box(extents=(40, 40, 40)), 1.0, 2000)
    assert sphere["sample_count"] >= 900 and sphere["absolute_mean_curvature_per_mm"]["p50"] == pytest.approx(0.1, rel=0.02)
    assert sphere["absolute_mean_curvature_per_mm"]["max"] < 0.105 and sphere["neighbour_difference_rms_per_mm"] < 0.005
    assert cylinder["absolute_mean_curvature_per_mm"]["p50"] == pytest.approx(0.1, rel=0.02)
    assert block["absolute_mean_curvature_per_mm"]["p50"] == pytest.approx(0.0, abs=1e-9) and block["absolute_mean_curvature_per_mm"]["p99"] > 0.3
    assert block["neighbour_difference_rms_per_mm"] > 10*sphere["neighbour_difference_rms_per_mm"]

def test_ball_curvature_is_independent_of_the_triangulation():
    coarse = ball_curvature(trimesh.creation.icosphere(subdivisions=4, radius=10), 1.0, 1000)
    fine = ball_curvature(trimesh.creation.icosphere(subdivisions=6, radius=10), 1.0, 1000)
    block = ball_curvature(trimesh.creation.box(extents=(40, 40, 40)), 1.0, 1000)
    refined = ball_curvature(trimesh.creation.box(extents=(40, 40, 40)).subdivide().subdivide(), 1.0, 1000)
    assert coarse["absolute_mean_curvature_per_mm"]["p50"] == pytest.approx(fine["absolute_mean_curvature_per_mm"]["p50"], rel=0.03)
    assert block["absolute_mean_curvature_per_mm"]["p95"] == pytest.approx(refined["absolute_mean_curvature_per_mm"]["p95"], rel=0.05)
    assert ball_curvature(trimesh.creation.box(extents=(40, 40, 40)), 1.0, 1000) == {**block, "elapsed_s": pytest.approx(block["elapsed_s"], abs=60)}

def test_surface_metrics_separate_smooth_and_axis_aligned_shapes():
    sphere = surface_metrics(trimesh.creation.icosphere(subdivisions=5, radius=10), open_domain())
    block = surface_metrics(trimesh.creation.box(extents=(10, 8, 6)), open_domain())
    assert sphere["prescribed_triangle_count"] == 0 and sphere["free_sharp_edge_length_mm"] == 0.0 and sphere["free_axis_normal_area_fraction"] < 0.01
    assert block["free_axis_normal_area_fraction"] == pytest.approx(1.0)
    assert block["free_sharp_edge_length_per_area_per_mm"] == pytest.approx(4*(10+8+6)/(2*(10*8+10*6+8*6)))
