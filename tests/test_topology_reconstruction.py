import numpy as np
import pytest
from scipy.ndimage import label

from deep_frame.config import DESIGN_RECONSTRUCTION_CONFIG, SPLINE_RECONSTRUCTION_CONFIG
from deep_frame.topology_implicit import TWENTY_SIX, ImplicitField
from deep_frame.topology_reconstruction import DesignGraph, primitive_distance, Reconstruction, clearance, hoop_paths, load_paths, reconstruct, reconstruct_splines, sections, sweep_values, tube
from tools.reconstruction_study import isotropic_materialization

def test_isotropic_materialization_preserves_physical_coordinates():
    shape, spacing, origin = (3, 3, 4), np.array([2.0, 1.5, 0.75]), np.array([-7.0, 11.0, -3.0])
    centers = origin+(np.indices(shape).transpose(1, 2, 3, 0)+0.5)*spacing
    linear = lambda points: 0.4+(points-origin)@np.array([0.02, 0.03, 0.04])
    density = linear(centers).astype(np.float32)
    domain = {"grid": {"origin_mm": origin.tolist(), "spacing_mm": spacing.tolist(), "shape": list(shape)}, "allowed": np.ones(shape, dtype=bool), "preserve": np.zeros(shape, dtype=bool), "forbidden": np.zeros(shape, dtype=bool), "regions": [{"center_mm": [-3.0, 13.0, -1.0]}]}
    result, sampled = isotropic_materialization(domain, density)
    assert result["grid"] == {"origin_mm": origin.tolist(), "spacing_mm": [0.75]*3, "shape": [8, 6, 4]}
    physical = origin+(np.indices(sampled.shape).transpose(1, 2, 3, 0)+0.5)*0.75
    interior = np.all((physical >= centers[0, 0, 0]) & (physical <= centers[-1, -1, -1]), axis=-1)
    np.testing.assert_allclose(sampled[interior], linear(physical[interior]), atol=1e-7)
    assert result["regions"] == domain["regions"] and domain["grid"]["shape"] == [3, 3, 4]

def test_isotropic_materialization_resamples_masks_and_zeros_outside_extent():
    shape = (3, 2, 2)
    allowed, preserve, forbidden = np.ones(shape, dtype=bool), np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    allowed[0, 1, :] = False
    preserve[1, :, :] = True
    forbidden[1, 1, :] = True
    domain = {"grid": {"origin_mm": [5.0, -2.0, 4.0], "spacing_mm": [1.6, 1.0, 1.0], "shape": list(shape)}, "allowed": allowed, "preserve": preserve, "forbidden": forbidden}
    result, sampled = isotropic_materialization(domain, np.full(shape, 0.4, dtype=np.float32))
    assert result["grid"]["shape"] == [5, 2, 2]
    np.testing.assert_array_equal(result["preserve"][:, 0, 0], [False, False, True, False, False])
    np.testing.assert_array_equal(result["allowed"][:, 1, 0], [False, False, False, True, True])
    assert sampled[0, 0, 0] > 0 and sampled[2, 0, 0] == 1 and np.all(sampled[~result["allowed"]] == 0)
    np.testing.assert_array_equal(result["forbidden"], ~result["allowed"])
    extended = {**domain, "grid": {**domain["grid"], "spacing_mm": [1.4, 1.0, 1.0]}}
    outside, padded = isotropic_materialization(extended, np.full(shape, 0.4, dtype=np.float32))
    assert not outside["allowed"][-1].any() and outside["forbidden"][-1].all() and not padded[-1].any()

@pytest.mark.parametrize("h", [4/3, 2/3])
def test_isotropic_four_thirds_materialization_is_unchanged(h):
    shape = (12, 10, 6)
    density = np.random.default_rng(43).uniform(size=shape).astype(np.float32)
    preserve, allowed = density > 0.8, density > 0.2
    density[preserve], density[~allowed] = 1.0, 0.0
    domain = {"grid": {"origin_mm": [-68.0, -64.0, -4.0], "spacing_mm": [h]*3, "shape": list(shape)}, "allowed": allowed, "preserve": preserve, "forbidden": ~allowed}
    result, sampled = isotropic_materialization(domain, density)
    assert result is domain and sampled is density

H = 0.5
RADIUS = 1.5
ENDS = (np.array([4.0, 4.0, 6.0]), np.array([36.0, 4.0, 6.0]))
HUB = np.array([20.0, 20.0, 6.0])

def _segment_distance(points, start, stop):
    delta = stop-start
    t = np.clip(((points-start)@delta)/(delta@delta), 0, 1)
    return np.linalg.norm(points-start-t[..., None]*delta, axis=-1)

@pytest.fixture(scope="module")
def truss():
    shape = (80, 48, 24)
    centers = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)+0.5)*H
    density = np.zeros(shape, dtype=np.float32)
    for end in ENDS:
        density[_segment_distance(centers, end, HUB) <= RADIUS] = 1.0
    hub = {"name": "hub", "role": "preserve", "kind": "box", "min_mm": (HUB-2.5).tolist(), "max_mm": (HUB+2.5).tolist()}
    preserve = np.all((centers >= HUB-2.5) & (centers <= HUB+2.5), axis=-1)
    density[preserve] = 1.0
    lid = {"name": "lid", "role": "forbidden", "kind": "box", "min_mm": [HUB[0]-4, HUB[1]-4, HUB[2]+2.5], "max_mm": [HUB[0]+4, HUB[1]+4, HUB[2]+5.0]}
    domain = {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [H]*3, "shape": list(shape)}, "regions": [hub, lid], "preserve": preserve}
    config = {**DESIGN_RECONSTRUCTION_CONFIG, "density_sigma_cells": 0.0, "voxel_mm": 0.25, "calibration_steps": 0, "closing_radius_mm": 0.0}
    return domain, density, config

def test_two_rod_truss_reconstructs_to_two_capsules(truss):
    domain, density, config = truss
    graph = DesignGraph(density > 0.5, [0, 0, 0], H, domain["preserve"], config)
    rods = [member for member in graph.members if member["kind"] == "rod"]
    assert len(rods) == 2
    for member in rods:
        assert np.median(np.sqrt(member["a"]*member["b"])) == pytest.approx(RADIUS, rel=0.2)
        assert np.median(member["a"]/member["b"]) < 1.3
        end = min(ENDS, key=lambda e: np.linalg.norm(member["points"][0]-e)+np.linalg.norm(member["points"][-1]-e))
        assert min(np.linalg.norm(member["points"][[0, -1]]-end, axis=1)) < 1.5
        assert np.max(_segment_distance(member["points"], end, HUB)) < 0.5

def test_smooth_union_has_no_thin_neck():
    field = ImplicitField.from_function([-10.0, -10.0, -10.0], 0.25, (81, 81, 81), lambda x, y, z: np.full(np.broadcast_shapes(x.shape, y.shape, z.shape), -5.0, dtype=np.float32))
    axes = field.axes()
    first = sweep_values(axes, np.array([[-8.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.full(2, 1.2), np.full(2, 1.2), np.array([[0.0, 1.0, 0.0]]*2))
    second = sweep_values(axes, np.array([[0.0, 0.0, 0.0], [5.0, 6.0, 0.0]]), np.full(2, 1.2), np.full(2, 1.2), np.array([[0.0, 0.0, 1.0]]*2))
    hard = np.maximum(first, second)
    field.union(first).smooth_union(second, 1.5)
    assert np.all(field.values >= hard-1e-6)
    assert field.volume() > np.count_nonzero(hard > 0)*field.cell_volume
    report = field.copy().open(0.9)
    assert report["components_after"] == 1
    assert report["removed_volume_mm3"] < 0.1*report["before_volume_mm3"]

def test_reconstruction_is_one_watertight_body(truss):
    domain, density, config = truss
    mesh, graph, report = reconstruct(domain, density, config)
    assert mesh.is_watertight and report["bodies"] == 1 and report["exact_booleans"]["passed"]
    assert label(density > 0.5, TWENTY_SIX)[1] == 1
    assert mesh.volume == pytest.approx(float(density.sum())*H**3, rel=0.25)

def test_preserve_is_rounded_but_keeps_exact_mating_plane(truss):
    domain, density, config = truss
    mesh, graph, report = reconstruct(domain, density, config)
    top = HUB[2]+2.5
    assert np.any(np.abs(mesh.vertices[:, 2]-top) < 1e-6)
    assert not mesh.contains([HUB+2.5-0.1])[0] and mesh.contains([[HUB[0], HUB[1], top-0.05]])[0]
    assert report["mass_budget"]["node_spheres_mm3"] <= report["mass_budget"]["assigned_nodes_mm3"]+1e-6

def test_sections_keep_two_millimetre_minimum_and_oval_limit():
    a, b = sections([0.4, 3.0, 9.0], [0.3, 2.0, 1.5], 0.8, DESIGN_RECONSTRUCTION_CONFIG)
    assert np.all(2*b >= 2.0) and np.all(a >= b) and np.all(a <= DESIGN_RECONSTRUCTION_CONFIG["maximum_aspect"]*b+1e-12)

def test_hoop_becomes_one_continuous_tube():
    hoop = {"x_mm": 6.0, "radius_mm": 1.6, "path_yz_mm": [[4.0, 10.0], [10.0, 9.0], [13.0, 5.0], [10.0, 2.0]]}
    domain = {"grid": {"origin_mm": [-10.0, 0.0, 0.0], "spacing_mm": [0.5]*3, "shape": [40, 32, 28]}, "regions": [{"name": "box", "role": "allowed", "kind": "box", "min_mm": [-10.0, 0.0, 0.0], "max_mm": [10.0, 16.0, 14.0]}], "metadata": {"round2": {"hoop": hoop}}}
    builder = Reconstruction(domain, {**DESIGN_RECONSTRUCTION_CONFIG, "voxel_mm": 0.2})
    path = hoop_paths(hoop)[1]
    builder.add_member(tube(path, hoop["radius_mm"]), 0.5)
    mesh = builder.field.extract()[0]
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    assert mesh.body_count == 1 and mesh.volume == pytest.approx(np.pi*1.6**2*length+4/3*np.pi*1.6**3, rel=0.1)

def test_detached_member_fails_continuity_and_load_path(truss):
    domain, density, config = truss
    graph = DesignGraph(density > 0.5, [0, 0, 0], H, domain["preserve"], config)
    assert graph.continuity()["passed"]
    graph.graph.remove_edges_from([(u, v) for u, v, data in graph.graph.edges(data=True) if data.get("anchor") == 1])
    assert not graph.continuity()["passed"] and graph.continuity()["members_off_main"] == 1
    mesh = reconstruct(domain, density, config)[0]
    far = {"name": "far_mount", "role": "preserve", "kind": "box", "min_mm": [30.0, 20.0, 1.0], "max_mm": [32.0, 22.0, 3.0]}
    result = load_paths(mesh, dict(domain, regions=domain["regions"]+[far]), {**config, "load_path_mounts": ["hub", "far"]})
    assert result["r0.5"]["carried"] == 1 and result["r0.5"]["missing"] == ["far_mount"] and not result["passed"]

def test_field_thickness_keeps_rod_radius_and_ignores_mass_target(truss):
    domain, density, config = truss
    config = {**config, "calibration_steps": DESIGN_RECONSTRUCTION_CONFIG["calibration_steps"]}
    target = 0.5*float(density.sum())*H**3
    mesh, graph, report = reconstruct(domain, density, config, target)
    assert report["thickness"] == "field" and not report["volume_match"] and report["global_scale"] == 1.0 and report["calibration"] == []
    middle, axis = (ENDS[0]+HUB)/2, (HUB-ENDS[0])/np.linalg.norm(HUB-ENDS[0])
    grid = np.arange(-2.5, 2.5, 0.05)+0.025
    u, v = np.meshgrid(grid, grid, indexing="ij")
    across = np.cross(axis, [0.0, 0.0, 1.0])
    area = np.count_nonzero(mesh.contains(middle+u.reshape(-1, 1)*across+v.reshape(-1, 1)*np.array([0.0, 0.0, 1.0])))*0.05**2
    length = float(np.linalg.norm(HUB-ENDS[0]))
    centers = (np.argwhere(density > 0.5)+0.5)*H
    along = (centers-ENDS[0])@axis
    slab = (np.abs(along-length/2) < length/4) & (np.linalg.norm(centers-ENDS[0]-along[:, None]*axis, axis=1) < 2*RADIUS)
    field = np.sqrt(np.count_nonzero(slab)*H**3/(length/2)/np.pi)
    assert field == pytest.approx(RADIUS, rel=0.1) and np.sqrt(area/np.pi) == pytest.approx(field, rel=0.03)
    calibrated = reconstruct(domain, density, {**config, "calibrate_volume": True, "volume_match": True}, target)[2]
    assert calibrated["thickness"] == "calibrated" and calibrated["global_scale"] < 1.0

WAVE = {"spacing": 0.25, "a": np.array([4.0, 12.0, 6.0]), "b": np.array([44.0, 12.0, 6.0]), "c": np.array([24.0, 30.0, 6.0]), "radius": 1.75, "taper": (2.2, 1.6), "amplitude": 0.4, "ripple": 0.2, "wavelength": 5.0}

@pytest.fixture(scope="module")
def waves():
    h, shape = WAVE["spacing"], (192, 144, 48)
    centers = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)+0.5)*h
    x, y = centers[..., 0], centers[..., 1]
    phase = lambda s: np.sin(2*np.pi*s/WAVE["wavelength"])
    middle = (WAVE["a"]+WAVE["b"])/2
    along = np.clip(x, WAVE["a"][0], WAVE["b"][0])
    bar = np.hypot(np.hypot(x-along, y-WAVE["a"][1]-WAVE["amplitude"]*phase(along)), centers[..., 2]-WAVE["a"][2]) <= WAVE["radius"]+WAVE["ripple"]*phase(1.3*along)
    t = np.clip((y-middle[1])/(WAVE["c"][1]-middle[1]), 0, 1)
    stem = np.hypot(np.hypot(x-middle[0]-WAVE["amplitude"]*phase(y), y-middle[1]-t*(WAVE["c"][1]-middle[1])), centers[..., 2]-middle[2]) <= WAVE["taper"][0]+t*(WAVE["taper"][1]-WAVE["taper"][0])+WAVE["ripple"]*phase(1.3*y)
    pads = [{"name": f"pad_{name}", "role": "preserve", "kind": "box", "min_mm": (WAVE[name]-[2.5, 2.5, 2.5]).tolist(), "max_mm": (WAVE[name]+[2.5, 2.5, 2.5]).tolist()} for name in ("a", "b", "c")]
    preserve = np.zeros(shape, dtype=bool)
    for pad in pads:
        preserve |= np.all((centers >= pad["min_mm"]) & (centers <= pad["max_mm"]), axis=-1)
    density = (bar | stem | preserve).astype(np.float32)
    domain = {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [h]*3, "shape": list(shape)}, "regions": pads, "preserve": preserve}
    config = {**SPLINE_RECONSTRUCTION_CONFIG, "density_sigma_cells": 0.0, "voxel_mm": 0.25, "preserve_round_mm": 0.0, "transition_radius_mm": 1.0, "preserve_blend_mm": 1.0}
    mesh, graph, rods, report = reconstruct_splines(domain, density, config)
    return domain, density, config, mesh, graph, rods, report

def test_spline_fit_removes_waves_and_keeps_length_and_mass(waves):
    domain, density, config, mesh, graph, rods, report = waves
    built = [rod for rod in rods if rod is not None]
    assert len(built) == 3 and len(graph.nodes) == 1 and all(3 <= len(rod["control"]) <= 5 for rod in built)
    for rod in built:
        points = rod["points"]
        bar = abs(points[-1, 0]-points[0, 0]) > abs(points[-1, 1]-points[0, 1])
        along, lateral = (points[:, 0], points[:, 1]-WAVE["a"][1]) if bar else (points[:, 1], points[:, 0]-WAVE["c"][0])
        inner = np.minimum(np.linalg.norm(points-points[0], axis=1), np.linalg.norm(points-points[-1], axis=1)) > 4.0
        phase = 2*np.pi*along[inner]/WAVE["wavelength"]
        assert np.hypot(2*np.mean(lateral[inner]*np.sin(phase)), 2*np.mean(lateral[inner]*np.cos(phase))) < WAVE["amplitude"]/4
        assert np.abs(lateral[inner]).max() < WAVE["amplitude"]
        assert rod["length"] == pytest.approx(float(np.linalg.norm(points[-1]-points[0])), rel=0.02)
    assert mesh.is_watertight and report["bodies"] == 1
    assert mesh.volume == pytest.approx(float(density.sum())*WAVE["spacing"]**3, rel=0.02)

def test_spline_cross_section_is_monotone_and_smooth(waves):
    domain, density, config, mesh, graph, rods, report = waves
    stem = max((rod for rod in rods if rod is not None), key=lambda rod: abs(rod["points"][-1, 1]-rod["points"][0, 1]))
    radius = np.sqrt(stem["a"]*stem["b"])
    steps = np.diff(radius if stem["points"][-1, 1] > stem["points"][0, 1] else radius[::-1])
    assert np.all(steps <= 1e-9) and radius.max()-radius.min() > 0.2
    assert np.abs(np.diff(steps)).max() < 0.02
    assert np.all(radius >= config["minimum_radius_mm"]-1e-9)

def test_spline_node_transition_has_no_necking(waves):
    domain, density, config, mesh, graph, rods, report = waves
    assert report["joint_sections"]["count"] == 3 and report["joint_sections"]["below_one"] == 0 and report["joint_sections"]["minimum_ratio"] >= 0.98

def test_spline_members_on_the_envelope_floor_are_seated_not_clipped():
    h, shape = 0.25, (176, 64, 40)
    centers = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)+0.5)*h
    start, stop, radius = np.array([6.0, 8.0, 0.75]), np.array([38.0, 8.0, 0.75]), 2.0
    pads = [{"name": f"pad_{i}", "role": "preserve", "kind": "box", "min_mm": (end+[-2.5, -2.5, -0.75]).tolist(), "max_mm": (end+[2.5, 2.5, 4.25]).tolist()} for i, end in enumerate((start, stop))]
    preserve = np.zeros(shape, dtype=bool)
    for pad in pads:
        preserve |= np.all((centers >= pad["min_mm"]) & (centers <= pad["max_mm"]), axis=-1)
    density = ((_segment_distance(centers, start, stop) <= radius) | preserve).astype(np.float32)
    domain = {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [h]*3, "shape": list(shape)}, "regions": pads, "preserve": preserve}
    config = {**SPLINE_RECONSTRUCTION_CONFIG, "density_sigma_cells": 0.0, "voxel_mm": 0.25, "preserve_round_mm": 0.0}
    mesh, graph, rods, report = reconstruct_splines(domain, density, config)
    rod = max((rod for rod in rods if rod is not None), key=lambda rod: rod["length"])
    inner = slice(len(rod["points"])//4, 3*len(rod["points"])//4)
    assert np.all(clearance(domain, rod["points"][inner])[0] >= rod["b"][inner]-config["clearance_tolerance_mm"])
    assert report["seating"]["members_shifted"] >= 1 and report["bodies"] == 1
    assert mesh.volume >= 0.95*float(density.sum())*h**3

def test_spline_anchors_land_inside_the_analytic_prescribed_part():
    h, shape = 0.25, (176, 64, 40)
    centers = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)+0.5)*h
    start, stop, radius = np.array([6.0, 8.0, 5.0]), np.array([38.0, 8.0, 5.0]), 1.5
    pads = [{"name": f"pad_{i}", "role": "preserve", "kind": "box", "min_mm": (end+[-2.5, -2.5, -2.5]).tolist(), "max_mm": (end+[2.5, 2.5, 2.5]).tolist()} for i, end in enumerate((start, stop))]
    preserve = np.zeros(shape, dtype=bool)
    for pad in pads:
        preserve |= np.all((centers >= np.asarray(pad["min_mm"])-[0.0, 0.0, 0.0]) & (centers <= np.asarray(pad["max_mm"])+[1.5, 0.0, 0.0]), axis=-1)
    density = ((_segment_distance(centers, start, stop) <= radius) | preserve).astype(np.float32)
    domain = {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [h]*3, "shape": list(shape)}, "regions": pads, "preserve": preserve}
    config = {**SPLINE_RECONSTRUCTION_CONFIG, "density_sigma_cells": 0.0, "voxel_mm": 0.25, "preserve_round_mm": 0.0}
    mesh, graph, rods, report = reconstruct_splines(domain, density, config)
    rod = max((rod for rod in rods if rod is not None), key=lambda rod: rod["length"])
    inside = lambda point: max(primitive_distance([np.array([point[0]]), np.array([point[1]]), np.array([point[2]])], pad)[0] for pad in pads)
    assert report["seating"]["nodes"]["anchors"]["moved"] >= 1 and report["bodies"] == 1
    assert inside(rod["points"][0]) >= config["anchor_inset_mm"]-config["clearance_tolerance_mm"] and inside(rod["points"][-1]) >= config["anchor_inset_mm"]-config["clearance_tolerance_mm"]


def test_spline_sections_follow_the_raw_section_orientation_along_the_member():
    h, shape = 0.25, (192, 64, 64)
    centers = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)+0.5)*h
    start, stop = np.array([4.0, 8.0, 8.0]), np.array([44.0, 8.0, 8.0])
    s = np.clip((centers[..., 0]-start[0])/(stop[0]-start[0]), 0, 1)
    turn = np.pi/2*np.clip((0.3-np.abs(s-0.5))/0.1, 0, 1)
    y, z = centers[..., 1]-8.0, centers[..., 2]-8.0
    u, v = np.cos(turn)*y+np.sin(turn)*z, -np.sin(turn)*y+np.cos(turn)*z
    pads = [{"name": f"pad_{i}", "role": "preserve", "kind": "box", "min_mm": (end-2.5).tolist(), "max_mm": (end+2.5).tolist()} for i, end in enumerate((start, stop))]
    preserve = np.zeros(shape, dtype=bool)
    for pad in pads:
        preserve |= np.all((centers >= pad["min_mm"]) & (centers <= pad["max_mm"]), axis=-1)
    density = (((u/2.8)**2+(v/1.4)**2 <= 1) | preserve).astype(np.float32)
    domain = {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [h]*3, "shape": list(shape)}, "regions": pads, "preserve": preserve}
    config = {**SPLINE_RECONSTRUCTION_CONFIG, "density_sigma_cells": 0.0, "voxel_mm": 0.25, "preserve_round_mm": 0.0}
    mesh, graph, rods, report = reconstruct_splines(domain, density, config)
    rod = max((rod for rod in rods if rod is not None), key=lambda rod: rod["length"])
    along = (rod["points"][:, 0]-start[0])/(stop[0]-start[0])
    middle, flank = np.argmin(np.abs(along-0.5)), np.argmin(np.abs(along-0.12))
    assert abs(rod["axis"][middle, 2]) > 0.9 and abs(rod["axis"][flank, 1]) > 0.9
    assert rod["a"][middle]/rod["b"][middle] > 1.5 and rod["a"][flank]/rod["b"][flank] > 1.5
    assert mesh.is_watertight and report["bodies"] == 1

def test_fork_web_is_found_where_the_raw_body_fills_between_diverging_members():
    from types import SimpleNamespace
    from deep_frame.topology_reconstruction import fork_webs
    h, shape = 0.5, (80, 64, 24)
    centers = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)+0.5)*h
    node, left, right = np.array([4.0, 16.0, 6.0]), np.array([36.0, 4.0, 6.0]), np.array([36.0, 28.0, 6.0])
    rod = lambda end: {"points": node+np.linspace(0, 1, 40)[:, None]*(end-node), "b": np.full(40, 1.5)}
    rods = [rod(left), rod(right)]
    x, y, z = centers[..., 0], centers[..., 1], centers[..., 2]
    bars = (_segment_distance(centers, node, left) <= 1.5) | (_segment_distance(centers, node, right) <= 1.5)
    web = (x <= 16.0) & (np.abs(y-16.0) <= (x-node[0])*12/32+1.0) & (np.abs(z-6.0) <= 1.5)
    config = {**SPLINE_RECONSTRUCTION_CONFIG, "minimum_radius_mm": 1.25}
    found = fork_webs(SimpleNamespace(weights=(bars | web).astype(np.float32), origin=np.zeros(3), h=h), rods, config)
    assert len(found) == 1 and found[0]["members"] == [0, 1]
    assert 10.0 <= found[0]["length_mm"] <= 15.0 and found[0]["open_mm"] >= 5.0
    assert found[0]["thickness_mm"] == pytest.approx(3.0, abs=0.6)
    assert not fork_webs(SimpleNamespace(weights=bars.astype(np.float32), origin=np.zeros(3), h=h), rods, config)

def test_spline_contact_surface_ends_exactly_on_the_keep_out_plane():
    h, shape, plane = 0.25, (176, 64, 40), 5.0
    centers = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)+0.5)*h
    pads = [{"name": f"pad_{i}", "role": "preserve", "kind": "box", "min_mm": [x-2.5, 5.5, 1.5], "max_mm": [x+2.5, 10.5, 4.5]} for i, x in enumerate((6.0, 38.0))]
    lid = {"name": "battery_envelope", "role": "forbidden", "kind": "box", "min_mm": [10.0, 2.0, plane], "max_mm": [34.0, 14.0, 9.5]}
    contact = {"name": "battery_contact", "role": "allowed", "kind": "box", "min_mm": [10.0, 2.0, plane-3.0], "max_mm": [34.0, 14.0, plane]}
    preserve = np.zeros(shape, dtype=bool)
    for pad in pads:
        preserve |= np.all((centers >= pad["min_mm"]) & (centers <= pad["max_mm"]), axis=-1)
    density = (np.all((centers >= [6.0, 6.0, plane-2.5]) & (centers <= [38.0, 10.0, plane]), axis=-1) | preserve).astype(np.float32)
    domain = {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [h]*3, "shape": list(shape)}, "regions": [*pads, lid, contact], "preserve": preserve}
    tops = {}
    for names in (["battery_contact"], []):
        config = {**SPLINE_RECONSTRUCTION_CONFIG, "density_sigma_cells": 0.0, "voxel_mm": 0.25, "preserve_round_mm": 0.0, "contact_regions": names}
        mesh = reconstruct_splines(domain, density, config)[0]
        under = np.all((mesh.vertices[:, :2] >= [12.0, 2.0]) & (mesh.vertices[:, :2] <= [32.0, 14.0]), axis=1)
        up = (mesh.face_normals[:, 2] > 0.99) & (np.abs(mesh.triangles_center[:, 2]-plane) <= 0.02) & np.all((mesh.triangles_center[:, :2] >= [12.0, 2.0]) & (mesh.triangles_center[:, :2] <= [32.0, 14.0]), axis=1)
        tops[bool(names)] = (float(mesh.vertices[under, 2].max()), float(mesh.area_faces[up].sum()), len(mesh.split(only_watertight=False)))
    assert abs(tops[True][0]-plane) <= 0.02 and tops[True][1] >= 40.0 and tops[True][2] == 1
    assert tops[False][0] <= plane-SPLINE_RECONSTRUCTION_CONFIG["boolean_offset_mm"]+0.02 and tops[False][1] == 0.0

def test_camera_contacts_close_only_the_side_screw_gap_and_keep_the_bore_free():
    from deep_frame.topology_reconstruction import Reconstruction, finalize
    h, shape = 0.5, (40, 40, 40)
    envelope = {"name": "design_envelope", "role": "allowed", "kind": "box", "min_mm": [0, 0, 0], "max_mm": [20, 20, 20]}
    camera = {"name": "camera_envelope", "role": "forbidden", "kind": "box", "min_mm": [7, 5, 5], "max_mm": [13, 15, 15]}
    screw = {"name": "camera_screw_axis", "role": "forbidden", "kind": "cylinder", "axis": "x", "center_mm": [10, 10, 10], "radius_mm": 1.1, "height_mm": 20}
    domain = {"grid": {"origin_mm": [0, 0, 0], "spacing_mm": [h]*3, "shape": list(shape)}, "regions": [envelope, camera, screw], "preserve": np.zeros(shape, dtype=bool)}
    cfg = {**SPLINE_RECONSTRUCTION_CONFIG, "voxel_mm": 0.25}
    builder = Reconstruction(domain, cfg)
    x, y, z = builder.field.axes()
    left = np.minimum.reduce(np.broadcast_arrays(x-3.0, 6.65-x, 4.0-np.hypot(y-10, z-10)))
    right = np.minimum.reduce(np.broadcast_arrays(x-13.35, 17.0-x, 4.0-np.hypot(y-10, z-10)))
    base = np.minimum.reduce(np.broadcast_arrays(x-3.0, 17.0-x, y-6.0, 14.0-y, z-1.0, 7.0-z))
    builder.field.values = np.maximum.reduce([left, right, base]).astype(np.float32)
    mesh = builder.finish()[0]
    for plane, sign in ((7.0, 1), (13.0, -1)):
        cap = (np.abs(mesh.triangles_center[:, 0]-plane) < 0.35) & (sign*mesh.face_normals[:, 0] > 0.9)
        assert mesh.area_faces[cap].sum() > 20.0
    exact, _ = finalize(mesh, domain, cfg)
    for plane, sign in ((7.0, 1), (13.0, -1)):
        cap = (np.abs(exact.triangles_center[:, 0]-plane) < 0.02) & (sign*exact.face_normals[:, 0] > 0.9)
        assert exact.area_faces[cap].sum() > 20.0
    radial = np.hypot(exact.triangles_center[:, 1]-10, exact.triangles_center[:, 2]-10)
    assert not np.any((radial < 1.0) & (np.abs(exact.face_normals[:, 0]) > 0.9))
    assert len(exact.split(only_watertight=False)) == 1
    empty = Reconstruction(domain, cfg)
    before = empty.field.values.copy()
    empty.camera_contacts()
    assert np.array_equal(empty.field.values, before)
