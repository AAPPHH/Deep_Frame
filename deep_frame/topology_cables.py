import heapq
from itertools import count
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d, label, maximum_filter1d

from deep_frame.config import CABLES, COMPONENT_LIBRARY, IMPLICIT_CONFIG
from deep_frame.topology_geometry import region_contains
from deep_frame.topology_implicit import TWENTY_SIX, _manifold, _trimesh, exact_booleans, float32_margin, region_manifold
from deep_frame.topology_problem import radial_weight
from deep_frame.topology_reconstruction import clearance

def _unit(vectors):
    vectors = np.asarray(vectors, dtype=float)
    return vectors/np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)

def arc_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0

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

    def cavity(self):
        return np.vstack([[[0.0, self.tip]], self._arc(self.radius)])

    def shell(self):
        arc = self._arc(self.outer)
        apex = self.outer/np.sin(self.half_angle)
        if self.top >= apex:
            return np.vstack([[[0.0, apex]], arc])
        x = arc[-1, 0]*(apex-self.top)/(apex-arc[-1, 1])
        return np.vstack([[[x, self.top], [-x, self.top]], arc])

    def slot_cut(self, depth):
        half = self.slot/2
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

def sweep(points, us, vs, polygons):
    from manifold3d import Manifold, OpType
    rings = [p+poly[:, :1]*u+poly[:, 1:]*v for p, u, v, poly in zip(points, us, vs, polygons)]
    return Manifold.batch_boolean([Manifold.hull_points(np.vstack([rings[i], rings[i+1]])) for i in range(len(rings)-1)], OpType.Add)

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

    def _inside(self, points, names):
        return np.any([region_contains(points, self.preserves[n], 0.0) for n in names], axis=0) if names else np.zeros(len(points), dtype=bool)

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
        depth = np.clip(np.where(guide, 0.0, maximum_filter1d(depth, 7)), [p.outer for p in profiles], cfg["slot_depth_max_mm"])+cfg["slot_margin_mm"]
        return {**item, "routed": True, "points": points, "tangent": tangent, "u": u, "v": v, "owner": owner, "inside": inside, "guide": guide, "profiles": profiles, "slot_depth": depth, "end_regions": ends,
                "route_mm": arc_length(raw), "seat_shift_mm": seat, "channel_mm": float(arc[last]-arc[first]) if len(free) else 0.0, "lead_mm": self._lead(item, points)}

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
        hits, rays, _ = body.ray.intersects_location(points, -v, multiple_hits=False)
        depth = np.zeros(len(points))
        inside = body.contains(points)
        depth[rays] = np.where(inside[rays], np.linalg.norm(hits-points[rays], axis=1), 0.0)
        return depth

    def solids(self):
        shells, cavities = [], []
        for path in (p for p in self.paths if p["routed"]):
            profiles = path["profiles"]
            args = (path["points"], path["u"], path["v"])
            shells.append(sweep(*args, [p.shell() for p in profiles]))
            cavities.append(sweep(*args, [p.cavity() for p in profiles]))
            cavities.append(sweep(*args, [p.slot_cut(d) for p, d in zip(profiles, path["slot_depth"])]))
        return shells, cavities

    def apply(self, body, core):
        from manifold3d import Manifold, OpType
        started = perf_counter()
        shells, cavities = self.solids()
        margin = float32_margin(self.domain)
        keep = Manifold.batch_boolean([region_manifold(r, IMPLICIT_CONFIG["segment_tolerance_mm"], margin, 1, 1)[0] for r in self.preserves.values()], OpType.Add)
        shell = Manifold.batch_boolean(shells, OpType.Add)
        cavity = Manifold.batch_boolean([Manifold.batch_boolean(cavities, OpType.Add), keep], OpType.Subtract)
        result = Manifold.batch_boolean([Manifold.batch_boolean([_manifold(body), shell], OpType.Add), cavity], OpType.Subtract)
        mesh, booleans = exact_booleans(_trimesh(result), core, IMPLICIT_CONFIG)
        parts = mesh.split(only_watertight=False)
        largest = max(parts, key=lambda part: abs(part.volume))
        if len(parts) > 1:
            mesh, booleans = exact_booleans(largest, core, IMPLICIT_CONFIG)
        pieces = sorted(mesh.split(only_watertight=False), key=lambda part: -abs(part.volume))
        return mesh, {"shell_mm3": float(shell.volume()), "cavity_mm3": float(cavity.volume()), "debris": {"count": len(parts)-1, "volume_mm3": float(sum(abs(p.volume) for p in parts if p is not largest))}, "bodies": len(pieces), "minor_bodies": [{"volume_mm3": float(abs(part.volume)), "bounds_mm": np.round(part.bounds, 2).tolist()} for part in pieces[1:6]],
                      "watertight": bool(mesh.is_watertight), "volume_mm3": float(mesh.volume), "exact_booleans": {key: booleans[key] for key in ("status", "passed", "components", "preserve_added_mm3", "forbidden_removed_mm3", "envelope_removed_mm3")}, "runtime_s": perf_counter()-started}

    def check(self, mesh):
        cfg = self.config
        reports = []
        for path in self.paths:
            if not path["routed"]:
                reports.append({"name": path["name"], "routed": False, "continuous": False, "closed_to_props": False})
                continue
            step = max(int(round(cfg["check_step_mm"]/cfg["sample_mm"])), 1)
            rows = np.flatnonzero(~path["inside"])[::step]
            core = rows[~path["guide"][rows]]
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            ring = [path["points"][rows]+0.45*np.array([p.bundle for p in np.array(path["profiles"])[rows]])[:, None]*(np.cos(a)*path["u"][rows]+np.sin(a)*path["v"][rows]) for a in angles]
            blocked = np.zeros(len(rows), dtype=bool)
            for points in [path["points"][rows]]+ring:
                blocked |= mesh.contains(points)
            blocked_points = np.round(path["points"][rows][blocked], 1).tolist()[:12]
            open_rays, total = self._closure(mesh, path["points"][core])
            fit = self._fit(path)
            reports.append({"name": path["name"], "kind": path["kind"], "routed": True, "start": path["start_regions"], "end": path["end_regions"][-1], "members": path["route"]["members"],
                            "route_mm": round(path["route_mm"], 1), "channel_mm": round(path["channel_mm"], 1), "lead_mm": round(path["lead_mm"], 1), "cable_mm": round(path["route_mm"]+path["lead_mm"], 1),
                            "cost": round(path["route"]["cost"], 1), "bend_deg": round(float(np.degrees(path["route"]["bend_rad"])), 1), "prop_mm": round(sum(self.router.costs[i]["prop_mm"] for i in path["route"]["members"]), 1),
                            "checked_sections": int(len(rows)), "blocked_sections": int(blocked.sum()), "blocked_points_mm": blocked_points, "seat_shift_mm": round(path["seat_shift_mm"], 2), "continuous": bool(len(rows) and not blocked.any()),
                            "closure_rays": total, "open_rays": open_rays, "open_points_mm": getattr(self, "open_points", np.zeros((0, 3)))[:12].tolist() if open_rays else [], "closed_to_props": bool(total and not open_rays), "members_below_limit": [row for row in fit if row["below_limit"]],
                            "profile": path["profiles"][len(path["profiles"])//2].size, "centerline_mm": np.round(path["points"], 2).tolist()})
        return reports

    def _closure(self, mesh, points):
        cfg = self.config
        if not len(points):
            return 0, 0
        targets = []
        for disc in self.router.shadow["discs"]:
            center = np.asarray(disc["center_mm"], dtype=float)
            for radius in (0.5*disc["radius_mm"], disc["radius_mm"]):
                angles = np.linspace(0, 2*np.pi, cfg["closure_rays"]+1)[:-1]
                targets.append(center+radius*np.stack([np.cos(angles), np.sin(angles), 0*angles], axis=1))
        targets = np.vstack(targets)
        origins = np.repeat(points, len(targets), axis=0)
        ends = np.tile(targets, (len(points), 1))
        up = points+np.asarray(cfg["print_axis"])*100.0
        origins, ends = np.vstack([origins, points]), np.vstack([ends, up])
        directions = ends-origins
        reach = np.linalg.norm(directions, axis=1)
        hits, rays, _ = mesh.ray.intersects_location(origins, directions/reach[:, None], multiple_hits=False)
        blocked = np.zeros(len(origins), dtype=bool)
        blocked[rays] = np.linalg.norm(hits-origins[rays], axis=1) < reach[rays]
        self.open_points = np.unique(np.round(origins[~blocked], 1), axis=0)
        return int((~blocked).sum()), int(len(origins))

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

    def printability(self, mesh):
        from scipy.spatial import cKDTree
        cfg = self.config
        centers = np.vstack([p["points"][~p["inside"]] for p in self.paths if p["routed"]])
        radius = max(p.radius for path in self.paths if path["routed"] for p in path["profiles"])
        distance = cKDTree(centers).query(mesh.triangles_center)[0]
        near = distance <= radius+0.05
        normals = mesh.face_normals[near]
        down = -normals@_unit(cfg["print_axis"])
        overhang = np.degrees(np.arcsin(np.clip(down, -1, 1)))
        area = mesh.area_faces[near]
        worst = overhang > cfg["teardrop_deg"]+1.0
        spots = np.unique(np.round(mesh.triangles_center[near][worst]/2)*2, axis=0)
        return {"cavity_faces": int(near.sum()), "maximum_overhang_deg": float(overhang.max()) if len(overhang) else 0.0, "area_over_limit_mm2": float(area[worst].sum()), "cavity_area_mm2": float(area.sum()),
                "limit_deg": cfg["teardrop_deg"], "spots_mm": spots[:20].tolist(), "spot_count": int(len(spots)), "definition": "final-mesh faces within inner radius + 0.05 mm of a channel centre line; overhang = asin(-n . print_axis)"}

    def report(self, mesh, before):
        checks = self.check(mesh)
        return {"paths": checks, "printability": self.printability(mesh), "shared_bundles_mm": {str(k): v for k, v in self.shared.items()},
                "volume_before_mm3": float(before.volume), "volume_after_mm3": float(mesh.volume), "volume_change_mm3": float(mesh.volume-before.volume),
                "steckbrief": {item["name"]: {"cable_mm": item.get("cable_mm"), "channel_mm": item.get("channel_mm"), "continuous": item["continuous"], "closed_to_props": item["closed_to_props"]} for item in checks},
                "bundle_model": {kind: COMPONENT_LIBRARY[spec["part"]]["cable"] for kind, spec in self.config["cables"].items()}}
