from fractions import Fraction

import numpy as np
import pytest
import trimesh
from scipy.ndimage import gaussian_filter

from deep_frame.topology_geometry import rasterize_regions, region_contains
from deep_frame.topology_implicit import ImplicitError, ImplicitField, _adjacent_pair, build_field, build_implicit, exact_booleans, export_mesh, extend_density, implicit_settings, mesh_checks, primitive_distance, remesh, segments, surface_fidelity, vertex_manifold

def box(name, role, low, high, **extra):
    return {"name": name, "role": role, "kind": "box", "min_mm": list(low), "max_mm": list(high), **extra}

def cylinder(name, role, center, radius, height, axis="z", **extra):
    return {"name": name, "role": role, "kind": "cylinder", "center_mm": list(center), "radius_mm": radius, "height_mm": height, "axis": axis, **extra}

def make_domain(shape, regions, spacing=1.0, origin=(0.0, 0.0, 0.0)):
    grid = {"origin_mm": list(origin), "spacing_mm": [spacing]*3, "shape": list(shape), "axis_order": "xyz", "order": "C"}
    regions = [box("design_envelope", "allowed", origin, np.asarray(origin)+spacing*np.asarray(shape))]+regions
    return {"grid": grid, **rasterize_regions(grid, regions), "regions": regions}

def brute_distance(points, region):
    if region["kind"] == "sphere":
        return region["radius_mm"]-np.linalg.norm(points-region["center_mm"], axis=1)
    if region["kind"] == "box":
        low, high = np.asarray(region["min_mm"]), np.asarray(region["max_mm"])
        distances = []
        for axis in range(3):
            for plane in (low[axis], high[axis]):
                clamped = np.clip(points, low, high)
                clamped[:, axis] = plane
                distances.append(np.linalg.norm(points-clamped, axis=1))
        distance = np.min(distances, axis=0)
    else:
        axis = "xyz".index(region["axis"])
        delta = points-region["center_mm"]
        radial = np.linalg.norm(np.delete(delta, axis, axis=1), axis=1)
        axial = np.abs(delta[:, axis])-region["height_mm"]/2
        lateral = np.hypot(radial-region["radius_mm"], np.maximum(axial, 0))
        cap = np.hypot(np.maximum(radial-region["radius_mm"], 0), axial)
        distance = np.minimum(lateral, cap)
    return np.where(region_contains(points, region), distance, -distance)

@pytest.mark.parametrize("region", [box("b", "forbidden", [-3, -2, -1], [4, 2.5, 1.5]), cylinder("z", "forbidden", [0.5, -1, 0.2], 2.5, 4.0), cylinder("x", "forbidden", [0.5, -1, 0.2], 1.4, 7.0, axis="x"), {"kind": "sphere", "center_mm": [1, 0, -1], "radius_mm": 3.0}])
def test_primitive_distances_match_brute_force_euclidean_distance(region):
    points = np.random.default_rng(3).uniform(-8, 8, (10000, 3))
    np.testing.assert_allclose(primitive_distance([points[:, i] for i in range(3)], region), brute_distance(points, region), atol=1e-9)

@pytest.mark.parametrize("region, volume", [({"kind": "sphere", "center_mm": [0, 0, 0], "radius_mm": 5.0}, 4*np.pi*125/3), (box("b", "preserve", [-5, -3, -2], [5, 3, 2]), 240.0), (cylinder("c", "preserve", [0, 0, 0], 4.0, 6.0), np.pi*96)])
def test_analytic_field_sample_volume(region, volume):
    field = ImplicitField.from_function([-7.1, -7.1, -7.1], 0.2, (72, 72, 72), lambda x, y, z: primitive_distance([x, y, z], region))
    assert field.volume() == pytest.approx(volume, rel=0.02)

def test_reinitialize_restores_signed_distance_of_perturbed_sphere():
    h = 0.25
    def perturbed(x, y, z):
        exact = 5-np.sqrt(x**2+y**2+z**2)
        noise = 0.5*np.sin(1.3*x)*np.sin(0.7*y+1)*np.cos(z)
        return 1.7*exact+np.where(np.abs(exact) > 4*h, noise, 0)
    field = ImplicitField.from_function([-8.05, -8.05, -8.05], h, (65, 65, 65), perturbed).reinitialize()
    x, y, z = field.axes()
    exact = 5-np.sqrt(x**2+y**2+z**2)
    error = np.abs(field.values-exact)
    assert error[np.abs(exact) <= 2*h].max() <= 0.05
    assert error.max() <= 0.6*h
    assert np.array_equal(field.values > 0, exact > 0)

def rod(x, y, z):
    return primitive_distance([x, y, z], cylinder("rod", "free", [4, 0, 0], 1.5, 14.0, axis="x"))

def crossing(values, coordinates):
    index = np.flatnonzero(np.diff(values > 0))
    return [coordinates[i]+values[i]/(values[i]-values[i+1])*(coordinates[i+1]-coordinates[i]) for i in index]

def test_smooth_union_adds_bounded_volume_and_keeps_distant_rod_radius():
    mount = box("mount", "preserve", [-4, -4, -4], [-2, 4, 4])
    field = ImplicitField.from_function([-6.05, -6.05, -6.05], 0.25, (85, 49, 49), rod)
    mount_values = field.primitives([mount])
    hard = np.maximum(field.values, mount_values)
    field.smooth_union(mount_values, 1.0)
    added = np.count_nonzero((field.values > 0) & ~(hard > 0))
    assert 0 < added <= np.count_nonzero(hard > -1/6)-np.count_nonzero(hard > 0)
    assert np.all(field.values >= hard) and np.all(field.values-hard <= 1/6+1e-6)
    far = ImplicitField.from_function([-6.05, -6.05, -6.05], 0.25, (85, 49, 49), rod).smooth_union(mount_values, 2.0)
    x, y, _ = far.axes()
    column = int(np.argmin(np.abs(x.ravel()-8.0)))
    radii = crossing(far.values[column, :, 24], y.ravel())
    assert np.allclose(np.abs(radii), 1.5, atol=0.01)

def plate_thickness(field):
    _, _, z = field.axes()
    crossings = crossing(field.values[5, 5, :], z.ravel())
    return crossings[-1]-crossings[0] if len(crossings) == 2 else 0.0

@pytest.mark.parametrize("offset", [0.0, 0.1, 0.13])
def test_density_smoothing_keeps_plate_thickness(offset):
    field = ImplicitField.from_function([0, 0, -4.05+offset], 0.25, (11, 11, 33), lambda x, y, z: 1.25-np.abs(z)+0*x*y)
    assert plate_thickness(field.smooth(0.4).reinitialize()) == pytest.approx(2.5, abs=0.02)

@pytest.mark.parametrize("offset", [0.0, 0.05, 0.12])
def test_opening_removes_thin_plate_and_keeps_thick_plate(offset):
    thin = ImplicitField.from_function([0, 0, -4.05+offset], 0.25, (11, 11, 33), lambda x, y, z: 0.95-np.abs(z)+0*x*y)
    report = thin.open(1.35)
    assert not np.any(thin.values > 0) and report["removed_volume_mm3"] == report["before_volume_mm3"] > 0
    for half, shrink in ((1.5, 0.05), (1.75, 0.0)):
        thick = ImplicitField.from_function([0, 0, -4.05+offset], 0.25, (11, 11, 33), lambda x, y, z: half-np.abs(z)+0*x*y)
        report = thick.open(1.35)
        assert 2*half-shrink-0.01 <= plate_thickness(thick) <= 2*half+0.01
        assert report["removed_volume_mm3"] == 0

def chain_domain(gap):
    end = 20 if gap else 21
    mounts = [box("m1", "preserve", [2, 3, 2], [6, 7, 6]), box("m2", "preserve", [12, 3, 2], [16, 7, 6]), box("m3", "preserve", [end+gap, 3, 2], [end+gap+4, 7, 6])]
    domain = make_domain((34, 10, 8), mounts)
    density = np.where(domain["preserve"], 1.0, 0.0)
    density[6:end, 3:7, 2:6] = 1.0
    return domain, density

SMALL = {"subdivisions": 4, "threshold": 0.35, "extension": "preserve"}

@pytest.mark.parametrize("gap", [1, 6])
def test_disconnected_mount_is_rejected_not_bridged(gap):
    domain, density = chain_domain(gap)
    with pytest.raises(ImplicitError) as error:
        build_field(domain, density, SMALL)
    assert error.value.status == "mount_disconnected"
    witness = error.value.report["witness"]
    assert not witness["passed"] and len(witness["mount_components"]) == 2
    assert witness["preserves"]["m1"]["components"] == witness["preserves"]["m2"]["components"] != witness["preserves"]["m3"]["components"]

def sliver_gap(value):
    domain, density = chain_domain(1)
    density[20, 5, 4] = value
    return domain, density

def sliver_neck():
    domain, density = chain_domain(0)
    density[9, 3:7, 2:6] = 0.0
    density[9, 5, 4] = 0.4
    return domain, density

@pytest.mark.parametrize("case", [lambda: sliver_gap(0.36), lambda: sliver_gap(0.4), sliver_neck])
def test_sub_resolution_density_sliver_is_rejected_not_bridged_by_smooth_union(case):
    domain, density = case()
    with pytest.raises(ImplicitError) as error:
        build_implicit(domain, density, SMALL)
    report = error.value.report
    assert error.value.status == "mount_disconnected" and error.value.mesh is None
    assert report["witness"]["passed"] and not report["density_witness"]["passed"] and "opening" not in report
    assert len(report["density_witness"]["mount_components"]) == 2

def test_opening_split_is_rejected_before_ripple_smoothing_can_reconnect_it():
    domain, density = chain_domain(0)
    domain["regions"].append(box("lid", "forbidden", [17, -1, 4.3], [20, 11, 9], rasterize=False))
    with pytest.raises(ImplicitError) as error:
        build_field(domain, density, SMALL)
    report = error.value.report
    assert error.value.status == "mount_disconnected" and report["density_witness"]["passed"]
    assert report["opening"]["components_after"] == 2 and not report["opening_witness"]["passed"] and "final_witness" not in report

def test_touching_mount_chain_builds_one_connected_field():
    domain, density = chain_domain(0)
    field, report = build_field(domain, density, SMALL)
    assert report["status"] == "field_built" and report["witness"]["passed"] and report["density_witness"]["passed"] and report["opening_witness"]["passed"] and report["final_witness"]["passed"]
    assert report["final_witness"]["occupied_components_6"] == 1 and report["extension_guard"]["passed"]
    assert report["opening"]["components_after"] == 1
    assert set(report["timings_s"]) >= {"extension", "upsample", "primitives", "witness", "reinit", "smoothing", "density_witness", "composition", "opening", "restore"}

def test_ambiguous_cube_interior_vertex_is_kept_and_surface_stays_closed():
    cube = np.array([0.03667267, -0.02387328, -0.02498563, 0.03812484, 0.00452967, -0.04602556, -0.04827231, 0.02583875], dtype=np.float32).reshape(2, 2, 2)
    mesh, report = ImplicitField([0, 0, 0], 1.0, np.pad(cube, 1, constant_values=-0.05)).extract()
    assert report["interior_cube_vertices"] == 1 and report["float32_vertex_shift_samples"] < 1e-6 and mesh.is_watertight and mesh.is_winding_consistent

def test_preserve_restore_is_clipped_by_keepout_and_envelope_offsets():
    domain, density = chain_domain(0)
    keepout = box("side", "forbidden", [1, 7.1, 1], [7, 9, 7], rasterize=False)
    domain["regions"].append(keepout)
    field, report = build_field(domain, density, SMALL)
    offset = report["settings"]["constraint_offset_mm"]
    near = (field.primitives([keepout]) > -offset+0.05) | (field.primitives([{"kind": "box", "min_mm": [0, 0, 0], "max_mm": [34, 10, 8]}]) < offset-0.05)
    assert report["final_witness"]["passed"] and field.values[near].max() <= 0

def corner_domain():
    keepout = box("corner", "forbidden", [-1, -1, -1], [5.5, 5.5, 9])
    domain = make_domain((10, 10, 4), [keepout])
    density = np.zeros(domain["allowed"].shape)
    density[0:6, 6:8, :] = 1.0
    density[6:8, 0:6, :] = 1.0
    return domain, density

def test_forbidden_extension_targets_only_partial_cells():
    domain, density = corner_domain()
    extended, report = extend_density(domain, density, "preserve_forbidden")
    assert report["partial_forbidden_cells"] == 11*4 and report["covered_forbidden_cells"] == 25*4
    assert np.all(extended[:5, :5] == 0) and np.all(extended[5, :5] == 1) and np.all(extended[:5, 5] == 1)
    assert np.array_equal(extend_density(domain, density, "preserve")[0], density)

def test_extension_that_closes_a_corner_gap_is_rejected():
    domain, density = corner_domain()
    with pytest.raises(ImplicitError) as error:
        build_field(domain, density, {**SMALL, "threshold": 0.6, "extension": "preserve_forbidden"})
    assert error.value.status == "extension_changed_topology"
    guard = error.value.report["extension_guard"]
    assert guard["original"]["components_6"] == 2 and guard["extended"]["components_6"] == 1

@pytest.mark.parametrize("changes", [{"threshold": 1.2}, {"opening_radius_mm": -1.0}, {"subdivisions": 0}, {"remesh_target_mm": 0.0}, {"unknown": 1}, {"extension": "bridge"}])
def test_invalid_implicit_settings_fail(changes):
    with pytest.raises(ValueError):
        implicit_settings({**SMALL, **changes})

@pytest.mark.parametrize("changes", [{"segment_tolerance_mm": 0.6}, {"surface_deviation_mm": 25.0}, {"relative_volume_change": 0.9}, {"free_zone_minimum_samples": 1}, {"penetration_tolerance_mm": 0.5}, {"penetration_sample_spacing_mm": 2.0},
                                     {"free_zone_preserve_mm": 5.0}, {"free_zone_constraint_mm": 9.0}, {"free_zone_modified_mm": 4.0}, {"free_zone_opening_cells": 0.1}, {"mesh_minimum_sicn": 0.001}, {"mesh_boundary_deviation_mm": 0.5}])
def test_acceptance_gates_cannot_be_weakened(changes):
    with pytest.raises(ValueError, match="cannot be weakened"):
        implicit_settings({**SMALL, **changes})
    domain, density = composite_domain()
    with pytest.raises(ValueError, match="cannot be weakened"):
        build_implicit(domain, density, {**SMALL, **changes})

def test_acceptance_gates_can_be_tightened():
    config = implicit_settings({**SMALL, "surface_deviation_mm": 0.1, "segment_tolerance_mm": 0.005, "free_zone_minimum_samples": 5000, "free_zone_preserve_mm": 2.0, "free_zone_opening_cells": 1.0})
    assert config["surface_deviation_mm"] == 0.1 and segments(1.4, config["segment_tolerance_mm"]) > segments(1.4, 0.01)

def edge_free(points, low, high, h):
    near = (np.abs(points-low) <= h) | (np.abs(points-high) <= h)
    return near.sum(axis=1) == 1

@pytest.mark.parametrize("region, volume, tolerance", [({"kind": "sphere", "center_mm": [0, 0, 0], "radius_mm": 10.0}, 4000*np.pi/3, 0.003), (box("b", "preserve", [-10, -5, -3], [10, 5, 3]), 1200.0, 0.01), (cylinder("c", "preserve", [0, 0, 0], 5.0, 10.0), 250*np.pi, 0.005)])
def test_extraction_volume_distance_and_planes(region, volume, tolerance):
    h = 0.25
    field = ImplicitField.from_function([-12.05, -12.1, -12.15], h, (98, 98, 98), lambda x, y, z: primitive_distance([x, y, z], region))
    mesh, report = field.extract()
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.body_count == 1 and mesh.vertices.dtype == np.float64
    assert abs(mesh.volume/volume-1) <= tolerance and report["float32_vertex_shift_samples"] < 1e-4
    vertices = mesh.vertices
    distance = np.abs(primitive_distance([vertices[:, 0], vertices[:, 1], vertices[:, 2]], region))
    if region["kind"] == "sphere":
        assert distance.max() <= 0.01
    elif region["kind"] == "box":
        low, high = np.asarray(region["min_mm"]), np.asarray(region["max_mm"])
        free = edge_free(vertices, low, high, h)
        assert distance[free].max() <= 0.01
        for axis in range(3):
            for plane in (low[axis], high[axis]):
                face = free & (np.abs(vertices[:, axis]-plane) <= h)
                assert face.any() and np.abs(vertices[face, axis]-plane).max() <= 1e-6
    else:
        radial, axial = np.hypot(vertices[:, 0], vertices[:, 1]), np.abs(vertices[:, 2])
        lateral, cap = axial < 5-h, radial < 5-h
        assert np.abs(radial[lateral]-5).max() <= 0.01 and np.abs(axial[cap]-5).max() <= 1e-6

def stage_checks(mesh):
    checks = mesh_checks(mesh)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.body_count == 1 and mesh.volume > 0
    assert checks["passed"] and checks["self_intersections"]["passed"] and checks["topology"]["manifold3d_status"] == "Error.NoError"

CONFIG = implicit_settings({"threshold": 0.35, "extension": "preserve"})

def random_field(seed):
    noise = gaussian_filter(np.random.default_rng(seed).normal(size=(64, 64, 64)), 6)
    noise *= 2.0/np.abs(noise).max()
    return lambda x, y, z: 6-np.sqrt(x**2+y**2+z**2)+noise

@pytest.mark.parametrize("function", [lambda x, y, z: 6-np.sqrt(x**2+y**2+z**2), lambda x, y, z: primitive_distance([x, y, z], box("b", "preserve", [-6, -4, -3], [6, 4, 3])), lambda x, y, z: primitive_distance([x, y, z], cylinder("c", "preserve", [0, 0, 0], 4.0, 8.0)), random_field(1), random_field(2), random_field(3)])
def test_extract_remesh_and_booleans_stay_closed_and_self_intersection_free(function):
    field = ImplicitField.from_function([-9.05]*3, 0.3, (64, 64, 64), function)
    raw, _ = field.extract()
    stage_checks(raw)
    remeshed = remesh(raw, CONFIG)
    stage_checks(remeshed)
    assert surface_fidelity(raw, remeshed)["maximum_sampled_deviation_mm"] <= 0.2
    domain = {"grid": {"origin_mm": [-8.5, -8.5, -8.5], "spacing_mm": [1.0]*3, "shape": [17, 17, 17]}, "regions": [cylinder("bore", "forbidden", [0.3, -0.2, 0], 1.4, 30.0), box("cut", "forbidden", [-20, 2.5, -20], [20, 20, -1.5])]}
    final, report = exact_booleans(remeshed, domain, CONFIG)
    assert report["passed"] and report["components"] == 1
    stage_checks(final)

def composite_domain():
    regions = [box("m1", "preserve", [2, 3, 0], [6, 7, 6]), box("m2", "preserve", [24, 3, 0], [28, 7, 6]), cylinder("bore", "forbidden", [4, 5, 3], 1.1, 8.0, rasterize=False), box("keepout", "forbidden", [11, 0, 4.5], [17, 10, 9])]
    domain = make_domain((30, 10, 6), regions)
    density = np.where(domain["preserve"], 1.0, 0.0)
    density[6:24, 3:7, 1:5] = 1.0
    density[domain["forbidden"]] = 0.0
    return domain, density

def test_composite_build_has_exact_planes_bore_and_clearance():
    domain, density = composite_domain()
    mesh, report, field = build_implicit(domain, density, SMALL)
    assert report["status"] == "geometry_built" and report["final_mesh"]["passed"] and report["remesh"]["fidelity"]["passed"]
    stage_checks(mesh)
    for plane, sign in ((0.0, -1), (6.0, 1)):
        on_plane = np.all(mesh.triangles[:, :, 2] == plane, axis=1)
        assert on_plane.any() and np.all(np.sign(mesh.face_normals[on_plane, 2]) == sign)
    bore = report["exact_booleans"]["cylinders"][0]
    wall = np.hypot(mesh.vertices[:, 0]-4, mesh.vertices[:, 1]-5) < 1.2
    assert wall.any() and np.allclose(np.hypot(mesh.vertices[wall, 0]-4, mesh.vertices[wall, 1]-5), bore["cut_radius_mm"], atol=1e-5) and bore["oversize_mm"] <= CONFIG["segment_tolerance_mm"]
    assert report["exact_booleans"]["float32"]["passed"] and np.array_equal(mesh.vertices, mesh.vertices.astype(np.float32))
    above = (mesh.vertices[:, 0] > 11) & (mesh.vertices[:, 0] < 17)
    assert mesh.vertices[above, 2].max() <= 4.5
    assert set(report["timings_s"]) >= {"extraction", "remesh", "booleans", "final_checks"}

@pytest.mark.parametrize("radius", [1.1, 1.4, 1.5, 9.5, 34.5])
def test_segment_rule_bounds_circumscribed_oversize(radius):
    count = segments(radius, 0.01)
    assert radius/np.cos(np.pi/count)-radius <= 0.01 and (count == 8 or radius/np.cos(np.pi/(count-1))-radius > 0.01)

def test_default_manifold_segments_would_undersize_a_bore():
    import manifold3d
    count = manifold3d.get_circular_segments(1.4)
    assert count == 8 and 1.4*(1-np.cos(np.pi/count)) > 0.1 and segments(1.4, 0.01) > 20

@pytest.mark.parametrize("region", [cylinder("z11", "forbidden", [0.3, -0.7, 0], 1.1, 20.0), cylinder("z14", "forbidden", [-1.25, 2.0, 0], 1.4, 20.0), cylinder("z15", "forbidden", [1.0, 1.0, 0], 1.5, 20.0), cylinder("x14", "forbidden", [0, 0.4, -0.6], 1.4, 30.0, axis="x"), cylinder("x15", "forbidden", [0, -1.0, 0.5], 1.5, 30.0, axis="x"), box("slot", "forbidden", [-1, -6, -10], [1, 6, 10])])
def test_bores_and_slots_are_dimensionally_exact(region):
    block = trimesh.creation.box(bounds=[[-8, -8, -4], [8, 8, 4]])
    domain = {"grid": {"origin_mm": [-10, -10, -10], "spacing_mm": [1.0]*3, "shape": [20, 20, 20]}, "regions": [region]}
    final, report = exact_booleans(block, domain, CONFIG)
    stage_checks(final)
    if region["kind"] == "box":
        assert block.volume-final.volume == pytest.approx(2*12*8, rel=1e-9)
        return
    row = report["cylinders"][0]
    axis = "xyz".index(region["axis"])
    radial = [i for i in range(3) if i != axis]
    length = 8.0 if axis == 2 else 16.0
    radius = np.linalg.norm((final.vertices-np.asarray(region["center_mm"]))[:, radial], axis=1)
    wall = radius < region["radius_mm"]+0.5
    assert wall.sum() == 2*row["segments"]
    assert np.all((radius[wall] >= region["radius_mm"]-1e-6) & (radius[wall] <= region["radius_mm"]+0.01+1e-6))
    assert np.allclose(radius[wall], row["cut_radius_mm"], atol=1e-5) and row["cut_radius_mm"] > row["circumscribed_radius_mm"]
    area = row["segments"]*row["cut_radius_mm"]**2*np.sin(2*np.pi/row["segments"])/2
    assert block.volume-final.volume == pytest.approx(area*length, rel=1e-6)

def test_exported_ply_is_exact_and_stl_is_written(tmp_path):
    domain, density = composite_domain()
    mesh = exact_booleans(trimesh.creation.box(bounds=[[1, 1, 1], [9.2, 8.1, 5.3]]), domain, CONFIG)[0]
    artifacts = export_mesh(mesh, tmp_path)
    loaded = trimesh.load(tmp_path/"geometry.ply", process=False)
    assert np.array_equal(loaded.vertices, mesh.vertices) and np.array_equal(loaded.faces, mesh.faces)
    assert set(artifacts) == {"geometry.ply", "geometry.stl"} and all(len(row["sha256"]) == 64 for row in artifacts.values())

def exact(points):
    return [[Fraction(value) for value in point] for point in points]

@pytest.mark.parametrize("second, certified", [([[0, 0, 0], [-1, 0, 0], [-1, -1, 0]], True), ([[0, 0, 0], [1, 1, 0], [-1, 1, 0]], False), ([[0, 0, 0], [0.2, 0.2, -1], [0.3, 0.1, 1]], False), ([[0, 0, 0], [-1, 0, 1], [0, -1, 1]], True), ([[0, 0, 0], [-1, 0, 0], [0, -1, 0]], True), ([[0, 0, 0], [2, 0, 0], [0, -1, 0]], False)])
def test_vertex_sharing_pairs_are_certified_only_when_they_meet_in_the_vertex(second, certified):
    assert _adjacent_pair(exact([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), exact(second), [(0, 0)]) is certified

@pytest.mark.parametrize("other, certified", [([0.5, -1, 0], True), ([0.5, 1, 0], False), ([0.5, 0.5, 1], True), ([2, 0.5, 0], False)])
def test_edge_sharing_pairs_reject_coplanar_folds(other, certified):
    assert _adjacent_pair(exact([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]), exact([[0, 0, 0], [1, 0, 0], other]), [(0, 0), (1, 1)]) is certified

def test_real_self_intersection_and_pinched_vertex_are_rejected():
    folded = trimesh.creation.box(extents=(4, 4, 4))
    folded.vertices[7] = [-3, -1, -1]
    checks = mesh_checks(folded)
    assert folded.is_watertight and not checks["passed"] and not checks["self_intersections"]["passed"]
    first = trimesh.creation.box(bounds=[[0, 0, 0], [1, 1, 1]])
    second = trimesh.creation.box(bounds=[[1, 1, 1], [2, 2, 2]])
    pinched = trimesh.Trimesh(np.vstack((first.vertices, second.vertices)), np.vstack((first.faces, second.faces+8)), process=False)
    pinched.merge_vertices()
    assert pinched.is_watertight and len(pinched.vertices) == 15
    report = vertex_manifold(pinched)
    assert not report["passed"] and report["pinched_vertices"] == 1
    assert vertex_manifold(first)["passed"]
