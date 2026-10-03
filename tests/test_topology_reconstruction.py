import numpy as np
import pytest
from scipy.ndimage import label

from deep_frame.config import DESIGN_RECONSTRUCTION_CONFIG
from deep_frame.topology_implicit import TWENTY_SIX, ImplicitField
from deep_frame.topology_reconstruction import DesignGraph, Reconstruction, hoop_paths, load_paths, reconstruct, sections, sweep_values, tube

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
