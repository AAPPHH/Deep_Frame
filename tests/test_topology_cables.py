import networkx as nx
import numpy as np
import pytest

from deep_frame.config import CABLES, COMPONENT_LIBRARY
from deep_frame.topology_cables import CableRouter, ChannelProfile, bundle_diameter, profile_frames

MOTOR = COMPONENT_LIBRARY["GTS V3 1203"]["cable"]["bundle_diameter_mm"]

def test_profile_dimensions_follow_the_spec():
    size = ChannelProfile(MOTOR).size
    assert size["inner_mm"] == pytest.approx(MOTOR+0.2)
    assert 0.3 <= MOTOR-size["slot_mm"] <= 0.4
    assert size["slot_mm"] < size["bundle_mm"]
    assert size["lip_mm"] >= 0.8
    assert size["wall_mm"] == pytest.approx(CABLES["wall_mm"])
    assert size["outer_width_mm"] > size["inner_mm"]+2*0.8

def test_guide_profile_flares_and_opens():
    guide, core = ChannelProfile(MOTOR, guide=True), ChannelProfile(MOTOR)
    assert guide.radius > core.radius
    assert guide.slot == pytest.approx(2*guide.radius)

def _edge_angles(polygon):
    closed = np.vstack([polygon, polygon[:1]])
    edges = np.diff(closed, axis=0)
    upper = (closed[:-1, 1] > 1e-9) & (closed[1:, 1] > 1e-9)
    return np.degrees(np.arctan2(np.abs(edges[upper, 1]), np.abs(edges[upper, 0])))

@pytest.mark.parametrize("bundle", [1.6, MOTOR, 4.4])
def test_teardrop_ceiling_is_printable(bundle):
    profile = ChannelProfile(bundle)
    assert _edge_angles(profile.cavity()).min() >= 90-CABLES["teardrop_deg"]-1e-6
    assert profile.cavity()[:, 1].max() == pytest.approx(profile.tip)
    assert profile.top-profile.tip == pytest.approx(CABLES["wall_mm"])

def test_bundle_model():
    assert bundle_diameter([1.02]*3) == pytest.approx(MOTOR, abs=0.01)
    assert bundle_diameter([MOTOR, MOTOR]) == pytest.approx(2*MOTOR)
    assert bundle_diameter([MOTOR]) == pytest.approx(MOTOR)

def test_profile_frames_keep_the_opening_down():
    points = np.stack([np.linspace(0, 20, 41), np.zeros(41), np.full(41, 5.0)], axis=1)
    tangent, u, v = profile_frames(points)
    assert np.allclose(v, [0, 0, 1])
    assert np.allclose(np.einsum("ij,ij->i", u, tangent), 0)

class Graph:
    def __init__(self, members, centers, anchors):
        self.members, self.nodes, self.graph = members, [{"center": np.asarray(c, dtype=float)} for c in centers], nx.MultiGraph()
        for i, member in enumerate(members):
            self.graph.add_edge(*member["nodes"], member=i)
        for name, node in anchors:
            self.graph.add_edge(name, node, anchor=name)

def _line(*points):
    points = np.asarray(points, dtype=float)
    return np.concatenate([np.linspace(a, b, 21)[:-1] for a, b in zip(points[:-1], points[1:])]+[points[-1:]])

def _router(config):
    preserve = np.zeros((30, 30, 4), dtype=bool)
    preserve[0:2, 0:2, 0:2] = preserve[20:22, 20:22, 0:2] = True
    regions = [{"name": "a_motor_contact", "role": "preserve", "kind": "box", "min_mm": [0, 0, 0], "max_mm": [2, 2, 2]},
               {"name": "aio_contact_0", "role": "preserve", "kind": "box", "min_mm": [20, 20, 0], "max_mm": [22, 22, 2]},
               {"name": "prop_a", "role": "forbidden", "kind": "cylinder", "center_mm": [-40, -40, 5], "radius_mm": 30.0},
               {"name": "motor_a_envelope", "role": "forbidden", "kind": "cylinder", "center_mm": [-40, -40, 5], "radius_mm": 7.0}]
    domain = {"grid": {"origin_mm": [0, 0, 0], "spacing_mm": [1, 1, 1]}, "preserve": preserve, "regions": regions}
    corner, bend, target = [1, 1, 1], [21, 1, 1], [21, 21, 1]
    members = [{"nodes": (1, 2), "points": _line(corner, bend)[1:-1], "anchors": [None, None]},
               {"nodes": (2, 3), "points": _line(bend, target)[1:-1], "anchors": [None, None]},
               {"nodes": (1, 3), "points": _line(corner, [-5, 25, 1], target)[1:-1], "anchors": [None, None]}]
    graph = Graph(members, [corner, bend, target], [("preserve1", 1), ("preserve2", 3)])
    return CableRouter(graph, [None]*3, domain, config)

def test_bend_penalty_picks_the_straighter_route():
    straight = _router({**CABLES, "bend_mm_per_rad": 0.0})
    assert straight.names == {"preserve1": "a_motor_contact", "preserve2": "aio_contact_0"}
    assert straight.route(["preserve1"], ["preserve2"])["members"] == [0, 1]
    bent = _router(CABLES)
    route = bent.route(["preserve1"], ["preserve2"])
    assert route["members"] == [2]
    assert route["cost"] == pytest.approx(bent.costs[2]["cost"])
    assert bent.costs[0]["cost"]+bent.costs[1]["cost"] < route["cost"] < bent.costs[0]["cost"]+bent.costs[1]["cost"]+CABLES["bend_mm_per_rad"]*np.pi/2

def test_prop_proximity_cost():
    router = _router(CABLES)
    inside = router.member_cost(_line([-40, -20, 5], [-40, -12, 5]))
    hub = router.member_cost(_line([-40, -36, 5], [-40, -35, 5]))
    outside = router.member_cost(_line([20, 20, 5], [28, 20, 5]))
    assert inside["prop_mm"] > 0 and hub["prop_mm"] == 0 and outside["prop_mm"] == 0
    assert inside["cost"] == pytest.approx(inside["length_mm"]+CABLES["prop_weight"]*inside["prop_mm"])
    assert inside["cost"] > outside["cost"]
