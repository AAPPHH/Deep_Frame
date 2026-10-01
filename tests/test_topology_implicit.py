import numpy as np
import pytest

from deep_frame.topology_geometry import rasterize_regions, region_contains
from deep_frame.topology_implicit import ImplicitError, ImplicitField, build_field, extend_density, implicit_settings, primitive_distance

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
    thick = ImplicitField.from_function([0, 0, -4.05+offset], 0.25, (11, 11, 33), lambda x, y, z: 1.5-np.abs(z)+0*x*y)
    report = thick.open(1.35)
    assert plate_thickness(thick) == pytest.approx(3.0, abs=0.01)
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

def test_touching_mount_chain_builds_one_connected_field():
    domain, density = chain_domain(0)
    field, report = build_field(domain, density, SMALL)
    assert report["status"] == "field_built" and report["witness"]["passed"] and report["final_witness"]["passed"]
    assert report["final_witness"]["occupied_components_6"] == 1 and report["extension_guard"]["passed"]
    assert report["opening"]["components_after"] == 1
    assert set(report["timings_s"]) >= {"extension", "upsample", "primitives", "witness", "reinit", "smoothing", "composition", "opening", "restore"}

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
