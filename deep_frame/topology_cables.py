import heapq
from itertools import count
from time import perf_counter

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d, label, maximum_filter1d

from deep_frame.config import CABLES, COMPONENT_LIBRARY, IMPLICIT_CONFIG
from deep_frame.topology_geometry import region_contains
from deep_frame.topology_implicit import TWENTY_SIX, _envelope, _manifold, _round_float32, _trimesh, exact_booleans, float32_margin, region_manifold
from deep_frame.topology_problem import radial_weight
from deep_frame.topology_reconstruction import clearance

def _unit(vectors):
    vectors = np.asarray(vectors, dtype=float)
    return vectors/np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)

def arc_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0

def first_hits(mesh, origins, directions, chunk=CABLES["ray_chunk"]):
    distance = np.full(len(origins), np.inf)
    for start in range(0, len(origins), chunk):
        hits, rays, _ = mesh.ray.intersects_location(origins[start:start+chunk], directions[start:start+chunk], multiple_hits=False)
        distance[start+rays] = np.linalg.norm(hits-origins[start+rays], axis=1)
    return distance

def pinch_points(mesh):
    rounded = np.asarray(mesh.vertices, dtype=np.float32).astype(float)
    _, first, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    pairs = np.column_stack((first[inverse.ravel()], np.arange(len(rounded))))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    edges = {tuple(edge) for edge in np.sort(mesh.edges_unique, axis=1).tolist()}
    loose = np.array([tuple(sorted(pair)) not in edges for pair in pairs.tolist()], dtype=bool)
    return mesh.vertices[pairs[loose if loose.any() else slice(None), 1]]

def folded_edges(mesh, limit_deg=CABLES["fold_deg"]):
    return int(np.sum(mesh.face_adjacency_angles > np.radians(limit_deg)))

def _collapse(mesh, pairs):
    mapping = np.arange(len(mesh.vertices))
    mapping[pairs[:, 0]] = pairs[:, 1]
    faces = mapping[mesh.faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    result = trimesh.Trimesh(mesh.vertices, faces[counts[inverse.ravel()] == 1], process=False)
    result.remove_unreferenced_vertices()
    return result if result.is_watertight and result.is_winding_consistent and result.euler_number == mesh.euler_number and result.body_count == mesh.body_count else None

def _flip_caps(mesh, faces, limit_deg):
    faces, flipped = mesh.faces.copy(), 0
    edges = {tuple(e) for e in np.sort(mesh.edges_unique, axis=1).tolist()}
    owner = {(int(a), int(b)): i for i, row in enumerate(faces) for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0]))}
    done = set()
    for f in _caps(mesh, faces, limit_deg):
        k = int(np.argmax(mesh.face_angles[f]))
        x, u, w = (faces[f][(k+i) % 3] for i in range(3))
        g = owner.get((int(w), int(u)))
        if g is None or {f, g} & done:
            continue
        y = next(int(v) for v in faces[g] if v not in (u, w))
        if tuple(sorted((int(x), y))) in edges:
            continue
        faces[f], faces[g] = (u, y, x), (y, w, x)
        done |= {f, g}
        edges.add(tuple(sorted((int(x), y))))
        flipped += 1
    return trimesh.Trimesh(mesh.vertices, faces, process=False), flipped

def _caps(mesh, faces, limit_deg):
    folded = np.unique(mesh.face_adjacency[mesh.face_adjacency_angles > np.radians(limit_deg)])
    return [int(f) for f in folded if mesh.face_angles[f].max() > np.radians(limit_deg)]

def weld_folds(mesh, keepouts, config=CABLES):
    report = {"method": "knife-edge slivers (face pairs folded by more than limit_deg) removed by collapsing their shortest edge onto an existing vertex when it is shorter than weld_mm (caps with a near-180 deg angle and no short edge: their longest edge flipped, vertices unchanged), at least weld_keepout_mm away from every keep-out and preserve and the result stays closed, oriented, single-body and of equal Euler characteristic",
              "weld_mm": config["weld_mm"], "collapsed_edges": 0, "flipped_edges": 0, "skipped_long": 0, "skipped_keepout": 0, "rejected": 0, "folded_before": folded_edges(mesh, config["fold_deg"])}
    for _ in range(config["weld_rounds"]):
        folded = mesh.face_adjacency_angles > np.radians(config["fold_deg"])
        if not folded.any():
            break
        faces = mesh.faces[np.unique(mesh.face_adjacency[folded])]
        lengths = np.linalg.norm(mesh.vertices[faces]-mesh.vertices[np.roll(faces, -1, axis=1)], axis=2)
        j = np.argmin(lengths, axis=1)
        short = lengths[np.arange(len(j)), j] < config["weld_mm"]
        report["skipped_long"] += int((~short).sum())
        pairs = np.unique(np.sort(np.stack([faces[np.arange(len(j)), j], faces[np.arange(len(j)), (j+1) % 3]], axis=1)[short], axis=1), axis=0)
        near = np.any([region_contains(mesh.vertices[pairs].reshape(-1, 3), r, config["weld_keepout_mm"]).reshape(-1, 2).any(axis=1) for r in keepouts], axis=0) if keepouts and len(pairs) else np.zeros(len(pairs), dtype=bool)
        report["skipped_keepout"] += int(near.sum())
        used, chosen = set(), []
        for a, b in pairs[~near].tolist():
            if a not in used and b not in used:
                used |= {a, b}
                chosen.append((a, b))
        if not chosen:
            flipped, count = _flip_caps(mesh, mesh.faces, config["fold_deg"])
            if not count or folded_edges(flipped, config["fold_deg"]) >= folded.sum() or not flipped.is_watertight or not flipped.is_winding_consistent:
                break
            mesh = flipped
            report["flipped_edges"] += count
            continue
        merged = _collapse(mesh, np.array(chosen))
        if merged is None:
            for pair in chosen:
                single = _collapse(mesh, np.array([pair]))
                report["rejected"] += single is None
                mesh = mesh if single is None else single
        else:
            mesh = merged
        report["collapsed_edges"] += len(chosen)
    report["folded_after"] = folded_edges(mesh, config["fold_deg"])
    return mesh, report

def grown(region, distance):
    if region["kind"] == "box":
        return {**region, "min_mm": (np.asarray(region["min_mm"])-distance).tolist(), "max_mm": (np.asarray(region["max_mm"])+distance).tolist()}
    return {**region, "radius_mm": region["radius_mm"]+distance, "height_mm": region["height_mm"]+2*distance}

def bundle_diameter(diameters, packing=CABLES["packing"]):
    diameters = sorted(diameters, reverse=True)
    return max(float(sum(diameters[:2])), float(diameters[0]*packing[str(min(len(diameters), max(map(int, packing))))])) if diameters else 0.0

class ChannelProfile:
    def __init__(self, bundle, config=CABLES, guide=False):
        self.bundle, self.config, self.guide = float(bundle), config, guide
        self.radius = (self.bundle+config["clearance_mm"])/2+(config["guide_flare_mm"]/2 if guide else 0.0)
        self.slot = 2*self.radius if guide else self.bundle-config["slot_interference_mm"]
        self.half_angle = np.radians(config["teardrop_deg"])
        self.floor = float(np.sqrt(max(self.radius**2-(self.slot/2)**2, 0.0)))
        self.outer = max(self.radius+config["wall_mm"], float(np.hypot(self.floor+config["lip_mm"], self.slot/2)))
        self.tip = self.radius/np.sin(self.half_angle)
        self.top = self.tip+config["wall_mm"]

    @property
    def lip(self):
        return float(np.sqrt(self.outer**2-(self.slot/2)**2)-self.floor)

    @property
    def size(self):
        return {"bundle_mm": self.bundle, "inner_mm": 2*self.radius, "slot_mm": self.slot, "lip_mm": self.lip, "wall_mm": self.outer-self.radius, "outer_width_mm": 2*self.outer, "outer_height_mm": self.outer+self.top, "tip_mm": self.tip, "teardrop_deg": float(np.degrees(self.half_angle))}

    def _arc(self, radius):
        n = self.config["profile_segments"]
        angles = np.linspace(np.pi-self.half_angle, 2*np.pi+self.half_angle, n)
        return np.stack([radius*np.cos(angles), radius*np.sin(angles)], axis=1)

    def cavity(self, depth=0.0):
        if not (self.guide and depth):
            return np.vstack([[[0.0, self.tip]], self._arc(self.radius)])
        angles = np.linspace(0.0, self.half_angle, (self.config["profile_segments"]-2)//2)
        side = self.radius*np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return np.vstack([[[0.0, self.tip]], side[::-1]*[-1, 1], [[-self.radius, -depth], [self.radius, -depth]], side])

    def shell(self):
        arc = self._arc(self.outer)
        apex = self.outer/np.sin(self.half_angle)
        if self.top >= apex:
            return np.vstack([[[0.0, apex]], arc])
        x = arc[-1, 0]*(apex-self.top)/(apex-arc[-1, 1])
        return np.vstack([[[x, self.top], [-x, self.top]], arc])

    def slot_cut(self, depth):
        half = min(self.slot, self.bundle-self.config["slot_interference_mm"])/2
        return np.array([[-half, 0.0], [half, 0.0], [half, -depth], [-half, -depth]])

def profile_frames(points, axis=CABLES["print_axis"]):
    up = _unit(axis)
    tangent = _unit(np.gradient(points, axis=0))
    vertical = tangent@up
    lift = up-vertical[:, None]*tangent
    radial = points-np.outer(points@up, up)
    radial = radial-np.einsum("ij,ij->i", radial, tangent)[:, None]*tangent
    steep = np.clip((np.abs(vertical)-0.8)/0.2, 0.0, 1.0)
    v = _unit(lift+steep[:, None]*_unit(radial))
    v = _unit(gaussian_filter1d(v, 2.0, axis=0, mode="nearest"))
    v = _unit(v-np.einsum("ij,ij->i", v, tangent)[:, None]*tangent)
    return tangent, np.cross(v, tangent), v

def sweep(points, us, vs, polygons, extend=0.0, overlap=CABLES["sweep_overlap_mm"]):
    from manifold3d import Manifold, OpType
    tangent = _unit(np.gradient(points, axis=0))
    shift = np.full(len(points), overlap)
    shift[[0, -1]] += extend
    rings = [p+poly[:, :1]*u+poly[:, 1:]*v for p, u, v, poly in zip(points, us, vs, polygons)]
    return Manifold.batch_boolean([Manifold.hull_points(np.vstack([rings[i]-shift[i]*tangent[i], rings[i+1]+shift[i+1]*tangent[i+1]])) for i in range(len(rings)-1)], OpType.Add)

def anchor_regions(domain):
    labels, total = label(domain["preserve"], TWENTY_SIX)
    origin, spacing = np.asarray(domain["grid"]["origin_mm"], dtype=float), np.asarray(domain["grid"]["spacing_mm"], dtype=float)
    preserves = [r for r in domain["regions"] if r["role"] == "preserve"]
    names = {}
    for k in range(1, total+1):
        centers = origin+(np.argwhere(labels == k)+0.5)*spacing
        share = [float(region_contains(centers, r, 0.7).mean()) for r in preserves]
        names[f"preserve{k}"] = preserves[int(np.argmax(share))]["name"] if share and max(share) > 0.2 else None
    return names

def prop_shadow(domain, config=CABLES):
    discs = [r for r in domain["regions"] if r["name"].startswith("prop_") and r["kind"] == "cylinder"]
    motors = [r for r in domain["regions"] if r["name"].startswith("motor_") and r["name"].endswith("_envelope")]
    return {"motors_mm": [r["center_mm"][:2] for r in discs], "radius_mm": max(r["radius_mm"] for r in discs)+config["prop_margin_mm"], "hub_radius_mm": max(r["radius_mm"] for r in motors),
            "exponent": config["prop_exponent"], "discs": discs}

class CableRouter:
    def __init__(self, graph, rods, domain, config=CABLES):
        self.graph, self.rods, self.domain, self.config = graph, rods, domain, config
        self.names = anchor_regions(domain)
        self.shadow = prop_shadow(domain, config)
        self.lines, self.costs, self.adjacent = {}, {}, {}
        for u, v, data in graph.graph.edges(data=True):
            if "member" in data:
                i = data["member"]
                self.lines[i] = self.polyline(i)
                self.costs[i] = self.member_cost(self.lines[i])
                self.adjacent.setdefault(u, []).append(("member", i, v, False))
                self.adjacent.setdefault(v, []).append(("member", i, u, True))
            else:
                self.adjacent.setdefault(u, []).append(("anchor", data["anchor"], v, False))
                self.adjacent.setdefault(v, []).append(("anchor", data["anchor"], u, False))

    def polyline(self, i):
        member, rod = self.graph.members[i], self.rods[i]
        if rod is not None:
            return np.asarray(rod["points"], dtype=float)
        ends = [[self.graph.nodes[n-1]["center"]] if isinstance(n, int) else [] if member["anchors"][side] is None else [member["anchors"][side]] for side, n in enumerate(member["nodes"])]
        return np.concatenate([np.asarray(ends[0]).reshape(-1, 3), member["points"], np.asarray(ends[1]).reshape(-1, 3)])

    def member_cost(self, line):
        steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
        middle = (line[1:]+line[:-1])/2
        length, shadow = float(steps.sum()), float(np.dot(radial_weight(middle, self.shadow), steps))
        return {"length_mm": length, "prop_mm": shadow, "cost": self.config["length_weight"]*length+self.config["prop_weight"]*shadow}

    def _direction(self, line, reach=2.0):
        steps = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        return _unit(line[min(int(np.searchsorted(steps, reach)), len(line)-1)]-line[0])

    def terminals(self, pattern):
        return sorted(node for node, name in self.names.items() if name is not None and pattern in name and node in self.adjacent)

    def route(self, sources, targets):
        targets, tie = set(targets), count()
        heap = [(0.0, next(tie), node, None, None, [], 0.0) for node in sources]
        seen = set()
        while heap:
            cost, _, node, last, heading, steps, bends = heapq.heappop(heap)
            if node in targets:
                return {"source": steps[0][2] if steps else node, "target": node, "cost": cost, "members": [i for kind, i, _, _ in steps if kind == "member"], "steps": steps, "bend_rad": bends}
            if (node, last) in seen:
                continue
            seen.add((node, last))
            for kind, i, other, backward in self.adjacent.get(node, []):
                if kind == "anchor":
                    if str(other).startswith("preserve") and other not in targets:
                        continue
                    heapq.heappush(heap, (cost, next(tie), other, last, heading, steps+[(kind, i, node, other)], bends))
                    continue
                if i == last:
                    continue
                line = self.lines[i][::-1] if backward else self.lines[i]
                bend = float(np.arccos(np.clip(heading@self._direction(line), -1, 1))) if heading is not None and isinstance(node, (int, np.integer)) else 0.0
                heapq.heappush(heap, (cost+self.costs[i]["cost"]+self.config["bend_mm_per_rad"]*bend, next(tie), other, i, -self._direction(line[::-1]), steps+[(kind, i, node, other)], bends+bend))
        return None

    def path_points(self, route):
        pieces, owners = [], []
        for kind, i, start, _ in route["steps"]:
            if kind != "member":
                continue
            backward = self.graph.members[i]["nodes"][0] != start
            line = self.lines[i][::-1] if backward else self.lines[i]
            line = line if not pieces else line[1:] if np.linalg.norm(line[0]-pieces[-1][-1]) < 1e-9 else line
            pieces.append(line)
            owners.append(np.full(len(line), i))
        return np.concatenate(pieces), np.concatenate(owners)

class CableChannels:
    def __init__(self, graph, rods, domain, config=CABLES):
        self.router = CableRouter(graph, rods, domain, config)
        self.graph, self.rods, self.domain, self.config = graph, rods, domain, config
        self.preserves = {r["name"]: r for r in domain["regions"] if r["role"] == "preserve"}

    def plan(self, body=None):
        router, cfg = self.router, self.config
        stack = router.terminals(cfg["stack_regions"])
        routes = []
        for kind, spec in cfg["cables"].items():
            starts = router.terminals(spec["regions"])
            groups = [[node] for node in starts] if spec["count"] > 1 else [starts]
            for group in groups:
                found = router.route(group, stack)
                name = kind if spec["count"] == 1 else router.names[group[0]].replace(spec["regions"], "")
                routes.append({"name": name, "kind": kind, "bundle_mm": COMPONENT_LIBRARY[spec["part"]]["cable"]["bundle_diameter_mm"], "route": found, "start_regions": [router.names[n] for n in group]})
        shared = {}
        for item in routes:
            for i in (item["route"] or {}).get("members", []):
                shared.setdefault(i, []).append(item["bundle_mm"])
        self.shared = {i: bundle_diameter(values, cfg["packing"]) for i, values in shared.items()}
        self.paths = [self._path(item, body) for item in routes]
        return self.paths

    def _inside(self, points, names, padding=0.0):
        return np.any([region_contains(points, self.preserves[n], padding) for n in names], axis=0) if names else np.zeros(len(points), dtype=bool)

    def _path(self, item, body):
        cfg, route = self.config, item["route"]
        if route is None:
            return {**item, "routed": False}
        raw, owners = self.router.path_points(route)
        ends = [self.router.names[route["source"]], self.router.names[route["target"]]]
        for side in (0, -1):
            direction = _unit(raw[side]-raw[1 if side == 0 else -2])
            raw = np.vstack([raw[:1]+direction*cfg["pad_extension_mm"], raw]) if side == 0 else np.vstack([raw, raw[-1:]+direction*cfg["pad_extension_mm"]])
            owners = np.r_[owners[:1], owners] if side == 0 else np.r_[owners, owners[-1:]]
        steps = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        s = np.linspace(0, steps[-1], max(int(np.ceil(steps[-1]/cfg["sample_mm"])), 2)+1)
        points = np.stack([np.interp(s, steps, raw[:, k]) for k in range(3)], axis=1)
        owner = owners[np.clip(np.searchsorted(steps, s), 0, len(owners)-1)]
        sigma = cfg["smooth_mm"]/cfg["sample_mm"]
        smooth = gaussian_filter1d(points, sigma, axis=0, mode="nearest")
        fade = np.clip(np.minimum(s, s[-1]-s)/(cfg["pad_extension_mm"]+1e-9), 0, 1)[:, None]
        points = fade*smooth+(1-fade)*points
        inside = self._inside(points, ends)
        free = np.flatnonzero(~inside)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        first, last = (free[0], free[-1]) if len(free) else (0, len(points)-1)
        guide = (arc <= arc[first]+cfg["guide_mm"]) | (arc >= arc[last]-cfg["guide_mm"])
        bundles = np.array([self.shared.get(int(i), item["bundle_mm"]) for i in owner])
        profiles = [ChannelProfile(b, cfg, bool(g)) for b, g in zip(bundles, guide)]
        points, seat = self._seat(points, profiles, ~inside, sigma)
        tangent, u, v = profile_frames(points, cfg["print_axis"])
        depth = self._slot_depth(body, points, v) if body is not None else np.zeros(len(points))
        outer = np.array([p.outer for p in profiles])
        ramp = np.clip(np.minimum(arc-arc[first], arc[last]-arc)/cfg["guide_mm"], 0, 1)*~inside
        drop = np.minimum(gaussian_filter1d(maximum_filter1d(np.maximum(depth-outer, 0.0), 7), sigma, mode="nearest"), cfg["underside_drop_max_mm"])*ramp
        if drop.max() > 0:
            points = points-drop[:, None]*v
            tangent, u, v = profile_frames(points, cfg["print_axis"])
            depth = self._slot_depth(body, points, v)
        depth = np.clip(np.where(guide, 0.0, maximum_filter1d(depth, 7)), [p.outer for p in profiles], cfg["slot_depth_max_mm"])+cfg["slot_margin_mm"]
        return {**item, "routed": True, "points": points, "tangent": tangent, "u": u, "v": v, "owner": owner, "inside": inside, "guide": guide, "profiles": profiles, "slot_depth": depth, "end_regions": ends,
                "route_mm": arc_length(raw), "seat_shift_mm": seat, "underside_drop_mm": float(drop.max()), "channel_mm": float(arc[last]-arc[first]) if len(free) else 0.0, "lead_mm": self._lead(item, points)}

    def _seat(self, points, profiles, free, sigma):
        cfg, moved = self.config, np.zeros(len(points))
        for _ in range(cfg["seat_iterations"]):
            tangent, u, v = profile_frames(points, cfg["print_axis"])
            rings = [p+poly[:, :1]*a+poly[:, 1:]*b for p, a, b, poly in zip(points, u, v, (q.shell() for q in profiles))]
            sizes = [len(r) for r in rings]
            value, slope = clearance(self.domain, np.vstack(rings))
            starts = np.r_[0, np.cumsum(sizes)[:-1]]
            worst = np.array([start+int(np.argmin(value[start:start+n])) for start, n in zip(starts, sizes)])
            deficit = np.where(free, np.maximum(cfg["seat_margin_mm"]-value[worst], 0.0), 0.0)
            if deficit.max() <= cfg["seat_tolerance_mm"]:
                break
            shift = gaussian_filter1d(maximum_filter1d(deficit[:, None]*slope[worst], int(2*sigma)+1, axis=0), sigma, axis=0, mode="nearest")
            shift = shift*np.where(free, 1.0, 0.0)[:, None]
            points = points+shift
            moved += np.linalg.norm(shift, axis=1)
        return points, float(moved.max())

    def _lead(self, item, points):
        names = item["start_regions"]
        centers = [np.asarray(self.preserves[n]["center_mm"] if "center_mm" in self.preserves[n] else (np.asarray(self.preserves[n]["min_mm"])+self.preserves[n]["max_mm"])/2, dtype=float) for n in names]
        return float(min(np.linalg.norm(c[:2]-points[0, :2]) for c in centers))

    def _slot_depth(self, body, points, v):
        distance = first_hits(body, points, -v)
        return np.where(np.isfinite(distance) & body.contains(points), distance, 0.0)

    def solids(self):
        shells, cavities, swept = [], [], set()
        for path in (p for p in self.paths if p["routed"]):
            own = ~np.isin(path["owner"], list(swept)) | path["guide"]
            own = maximum_filter1d(own.astype(np.uint8), 2*int(np.ceil(self.config["guide_flare_mm"]/self.config["sample_mm"]))+1).astype(bool)
            path["swept"] = own
            swept |= set(int(i) for i in path["owner"])
            runs = np.split(np.arange(len(own)), np.flatnonzero(np.diff(own.astype(int)))+1)
            outside = maximum_filter1d((~path["inside"]).astype(np.uint8), 3).astype(bool)
            overlap = self.config["sweep_overlap_mm"]
            for run in (r for r in runs if own[r[0]] and len(r) > 1):
                profiles, depth = [path["profiles"][k] for k in run], path["slot_depth"][run]
                args = (path["points"][run], path["u"][run], path["v"][run])
                for part in (r for r in np.split(np.arange(len(run)), np.flatnonzero(np.diff(outside[run].astype(int)))+1) if outside[run][r[0]] and len(r) > 1):
                    shells.append(sweep(*(x[part] for x in args), [profiles[k].shell() for k in part], 0.0, overlap))
                cavities.append(sweep(*args, [p.cavity(d) for p, d in zip(profiles, depth)], self.config["cavity_extend_mm"], 3*overlap))
                cavities.append(sweep(*args, [p.slot_cut(d) for p, d in zip(profiles, depth)], self.config["cavity_extend_mm"], 5*overlap))
        return shells, cavities

    def apply(self, body, core):
        from manifold3d import Error, Manifold, OpType
        started = perf_counter()
        shells, cavities = self.solids()
        margin, tolerance = float32_margin(self.domain), IMPLICIT_CONFIG["segment_tolerance_mm"]
        keep = Manifold.batch_boolean([region_manifold(grown(r, self.config["keep_margin_mm"]), tolerance, margin, 1, 1)[0] for r in self.preserves.values()], OpType.Add)
        forbidden = [region_manifold(r, tolerance, margin, 1, 1)[0] for r in core["regions"] if r["role"] == "forbidden"]
        raw = Manifold.batch_boolean(shells, OpType.Add)
        clear = Manifold.batch_boolean([raw, Manifold.batch_boolean(forbidden, OpType.Add)], OpType.Subtract) if forbidden else raw
        shell = Manifold.batch_boolean([clear, region_manifold(_envelope(core), tolerance, side=-1)[0]], OpType.Intersect)
        cavity = Manifold.batch_boolean([Manifold.batch_boolean(cavities, OpType.Add), keep], OpType.Subtract)
        solid = _manifold(body)
        joined = Manifold.batch_boolean([solid, shell], OpType.Add)
        carved = Manifold.batch_boolean([joined, cavity], OpType.Subtract)
        parts = sorted(carved.decompose(), key=lambda part: -part.volume())
        largest = parts[0]
        keepouts = [r for r in core["regions"] if r["role"] in ("forbidden", "preserve")]
        booleans = {"method": "body already passed exact_booleans; channel shells minus forbidden (cylinders grown by the binary32 margin) and clipped to the envelope before the union, cavities minus the grown preserves; binary32 rounding of the result, then knife-edge welding onto existing binary32 vertices",
                    "status": str(carved.status()), "components": len(parts), "forbidden_removed_mm3": float(raw.volume()-clear.volume()), "envelope_removed_mm3": float(clear.volume()-shell.volume()), "pinch_cuts": []}
        for _ in range(self.config["pinch_rounds"]):
            mesh, booleans["float32_largest"] = _round_float32(_trimesh(largest), margin)
            if booleans["float32_largest"]["passed"]:
                break
            points = pinch_points(_trimesh(largest))
            booleans["pinch_cuts"] += np.round(points, 3).tolist()
            spheres = Manifold.batch_boolean([Manifold.sphere(self.config["pinch_radius_mm"], 24).translate(tuple(point)) for point in points], OpType.Add)
            largest = sorted(Manifold.batch_boolean([largest, Manifold.batch_boolean([spheres, keep], OpType.Subtract)], OpType.Subtract).decompose(), key=lambda part: -part.volume())[0]
        mesh, booleans["weld"] = weld_folds(mesh, keepouts, self.config)
        booleans.update(shell_outside_body_mm3=float(joined.volume()-solid.volume()), removed_from_body_mm3=float(joined.volume()-carved.volume()), float32_passed=bool(carved.status() == Error.NoError and booleans["float32_largest"]["passed"]))
        minor = [{"volume_mm3": float(part.volume()), "bounds_mm": np.round(np.reshape(part.bounding_box(), (2, 3)), 2).tolist()} for part in parts[1:]]
        reference = exact_booleans(body, core, IMPLICIT_CONFIG)[1]
        return mesh, {"shell_mm3": float(shell.volume()), "cavity_mm3": float(cavity.volume()), "bodies": len(parts), "dropped_bodies": minor[:6], "dropped_volume_mm3": float(sum(part["volume_mm3"] for part in minor)), "watertight": bool(mesh.is_watertight), "volume_mm3": float(mesh.volume), "folded_edges": folded_edges(mesh), "booleans": booleans,
                      "body_preserve_deficit_mm3": reference["preserve_added_mm3"], "runtime_s": perf_counter()-started}

    def check(self, mesh):
        cfg = self.config
        reports = []
        for path in self.paths:
            if not path["routed"]:
                reports.append({"name": path["name"], "routed": False, "continuous": False, "closed_to_props": False})
                continue
            step = max(int(round(cfg["check_step_mm"]/cfg["sample_mm"])), 1)
            rows = np.flatnonzero(~self._inside(path["points"], path["end_regions"], max(p.radius for p in path["profiles"])))[::step]
            core = rows[~path["guide"][rows]]
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            ring = [path["points"][rows]+0.45*np.array([p.bundle for p in np.array(path["profiles"])[rows]])[:, None]*(np.cos(a)*path["u"][rows]+np.sin(a)*path["v"][rows]) for a in angles]
            blocked = np.zeros(len(rows), dtype=bool)
            for points in [path["points"][rows]]+ring:
                blocked |= mesh.contains(points)
            blocked_points = np.round(path["points"][rows][blocked], 1).tolist()[:12]
            open_rays, total, open_targets, open_any = self._closure(mesh, path["points"][core])
            fit = self._fit(path)
            reports.append({"name": path["name"], "kind": path["kind"], "routed": True, "start": path["start_regions"], "end": path["end_regions"][-1], "members": path["route"]["members"],
                            "route_mm": round(path["route_mm"], 1), "channel_mm": round(path["channel_mm"], 1), "lead_mm": round(path["lead_mm"], 1), "cable_mm": round(path["route_mm"]+path["lead_mm"], 1),
                            "cost": round(path["route"]["cost"], 1), "bend_deg": round(float(np.degrees(path["route"]["bend_rad"])), 1), "prop_mm": round(sum(self.router.costs[i]["prop_mm"] for i in path["route"]["members"]), 1),
                            "checked_sections": int(len(rows)), "blocked_sections": int(blocked.sum()), "blocked_points_mm": blocked_points, "seat_shift_mm": round(path["seat_shift_mm"], 2), "underside_drop_mm": round(path["underside_drop_mm"], 2), "continuous": bool(len(rows) and not blocked.any()),
                            "closure_rays": total, "open_rays": open_rays, "open_targets": open_targets, "open_any_disc_rays": open_any, "open_points_mm": getattr(self, "open_points", np.zeros((0, 3)))[:12].tolist() if open_rays else [], "closed_to_props": bool(total and not open_rays), "members_below_limit": [row for row in fit if row["below_limit"]],
                            "profile": path["profiles"][len(path["profiles"])//2].size, "centerline_mm": np.round(path["points"], 2).tolist()})
        return reports

    def _closure(self, mesh, points):
        cfg, shadow = self.config, self.router.shadow
        if not len(points):
            return 0, 0, {}, 0
        up = _unit(cfg["print_axis"])
        angles = np.linspace(0, 2*np.pi, cfg["closure_rays"]+1)[:-1]
        origins, ends, labels, spec = [points, points], [points+up*100.0, points+_unit(points-np.outer(points@up, up))*100.0], [np.full(len(points), "up"), np.full(len(points), "outside")], [np.ones(len(points), bool)]*2
        for disc in shadow["discs"]:
            center = np.asarray(disc["center_mm"], dtype=float)
            near = np.hypot(points[:, 0]-center[0], points[:, 1]-center[1]) <= shadow["radius_mm"]
            for radius in (0.5*disc["radius_mm"], disc["radius_mm"]):
                targets = center+radius*np.stack([np.cos(angles), np.sin(angles), 0*angles], axis=1)
                origins.append(np.repeat(points, len(targets), axis=0))
                ends.append(np.tile(targets, (len(points), 1)))
                labels.append(np.full(len(points)*len(targets), disc["name"]))
                spec.append(np.repeat(near, len(targets)))
        origins, ends, labels, spec = np.vstack(origins), np.vstack(ends), np.concatenate(labels), np.concatenate(spec)
        directions = ends-origins
        reach = np.linalg.norm(directions, axis=1)
        blocked = first_hits(mesh, origins, directions/reach[:, None]) < reach
        self.open_points = np.unique(np.round(origins[spec & ~blocked], 1), axis=0)
        found, counts = np.unique(labels[~blocked], return_counts=True)
        return int((spec & ~blocked).sum()), int(spec.sum()), {str(k): int(v) for k, v in zip(found, counts)}, int((~blocked).sum())

    def _fit(self, path):
        rows = []
        for i in dict.fromkeys(path["route"]["members"]):
            rod, member = self.rods[i], self.graph.members[i]
            b = np.asarray(rod["b"] if rod is not None else member["b"], dtype=float)
            a = np.asarray(rod["a"] if rod is not None else member["a"], dtype=float)
            profile = ChannelProfile(self.shared.get(i, path["bundle_mm"]), self.config)
            width = float(2*np.min(b))
            rows.append({"member": int(i), "kind": "rod" if rod is not None else member["kind"], "width_min_mm": round(width, 2), "width_median_mm": round(float(2*np.median(b)), 2), "height_median_mm": round(float(2*np.median(a)), 2),
                         "channel_width_mm": round(2*profile.outer, 2), "channel_height_mm": round(profile.outer+profile.top, 2), "below_limit": bool(width < 2*profile.outer)})
        return rows

    def printability(self, mesh, before):
        from scipy.spatial import cKDTree
        cfg = self.config
        centers = np.vstack([p["points"][~p["inside"]] for p in self.paths if p["routed"]])
        radius = max(p.radius for path in self.paths if path["routed"] for p in path["profiles"])
        distance = cKDTree(centers).query(mesh.triangles_center)[0]
        near = distance <= radius+0.05
        index = np.flatnonzero(near)
        near[index] = np.concatenate([before.nearest.on_surface(mesh.triangles_center[index[k:k+2000]])[1] for k in range(0, len(index), 2000)]) > 0.02
        normals = mesh.face_normals[near]
        down = -normals@_unit(cfg["print_axis"])
        overhang = np.degrees(np.arcsin(np.clip(down, -1, 1)))
        area = mesh.area_faces[near]
        worst = overhang > cfg["teardrop_deg"]+1.0
        spots = np.unique(np.round(mesh.triangles_center[near][worst]/2)*2, axis=0)
        return {"cavity_faces": int(near.sum()), "maximum_overhang_deg": float(overhang.max()) if len(overhang) else 0.0, "area_over_limit_mm2": float(area[worst].sum()), "cavity_area_mm2": float(area.sum()),
                "limit_deg": cfg["teardrop_deg"], "spots_mm": spots[:20].tolist(), "spot_count": int(len(spots)), "definition": "final-mesh faces within inner radius + 0.05 mm of a channel centre line and more than 0.02 mm from the body before the channels; overhang = asin(-n . print_axis)"}

    def report(self, mesh, before):
        checks = self.check(mesh)
        return {"paths": checks, "printability": self.printability(mesh, before), "shared_bundles_mm": {str(k): v for k, v in self.shared.items()},
                "volume_before_mm3": float(before.volume), "volume_after_mm3": float(mesh.volume), "volume_change_mm3": float(mesh.volume-before.volume),
                "steckbrief": {item["name"]: {"cable_mm": item.get("cable_mm"), "channel_mm": item.get("channel_mm"), "continuous": item["continuous"], "closed_to_props": item["closed_to_props"]} for item in checks},
                "bundle_model": {kind: COMPONENT_LIBRARY[spec["part"]]["cable"] for kind, spec in self.config["cables"].items()}}
