import json
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt, gaussian_filter, gaussian_filter1d, label, map_coordinates
from skimage.morphology import skeletonize

from deep_frame.config import DESIGN_RECONSTRUCTION_CONFIG, IMPLICIT_CONFIG
from deep_frame.topology_geometry import region_bounds, region_contains
from deep_frame.topology_implicit import TWENTY_SIX, ImplicitField, _envelope, exact_booleans, primitive_distance
from deep_frame.topology_surface import _upsample

CUBE = np.ones((3, 3, 3), dtype=np.uint8)
OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if i or j or k])

def solid_body(density, config=DESIGN_RECONSTRUCTION_CONFIG, cell_volume=1.0):
    field = np.asarray(density, dtype=np.float32)
    field = gaussian_filter(field, config["density_sigma_cells"]) if config["density_sigma_cells"] > 0 else field
    labels, count = label(field > config["threshold"], TWENTY_SIX)
    if not count:
        raise ValueError("Density field has no material above the threshold")
    sizes = np.bincount(labels.ravel())[1:]
    return labels == int(np.argmax(sizes))+1, {"count": int(count-1), "volume_mm3": float((sizes.sum()-sizes.max())*cell_volume)}

def _degree(skeleton):
    return np.where(skeleton, convolve(skeleton.astype(np.uint8), CUBE, mode="constant")-1, 0)

def _groups(labels):
    coords = np.argwhere(labels > 0)
    ids = labels[tuple(coords.T)]
    order = np.argsort(ids, kind="stable")
    coords, ids = coords[order], ids[order]
    return np.split(coords, np.flatnonzero(np.diff(ids))+1) if len(coords) else []

def _order(coords):
    index = {tuple(c): i for i, c in enumerate(coords)}
    neighbours = [[index[key] for key in map(tuple, c+OFFSETS) if key in index] for c in coords]
    start = next((i for i, n in enumerate(neighbours) if len(n) <= 1), 0)
    path, seen = [start], {start}
    while True:
        step = [n for n in neighbours[path[-1]] if n not in seen]
        if not step:
            break
        step = min(step, key=lambda n: np.abs(coords[n]-coords[path[-1]]).sum())
        path.append(step)
        seen.add(step)
    return coords[path]

def _neighbour_label(labels, voxel):
    hits = [labels[tuple(c)] for c in voxel+OFFSETS if np.all(c >= 0) and np.all(c < labels.shape)]
    hits = [h for h in hits if h > 0]
    return int(max(set(hits), key=hits.count)) if hits else 0

class DesignGraph:
    def __init__(self, solid, origin, spacing, preserve, config=DESIGN_RECONSTRUCTION_CONFIG, weights=None):
        self.h, self.origin, self.config = float(spacing), np.asarray(origin, dtype=float), config
        self.solid, self.preserve = solid, preserve
        self.weights = solid.astype(np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        self.radius = distance_transform_edt(solid)*self.h
        skeleton = skeletonize(solid).astype(bool)
        self.spurs = self._prune(skeleton, binary_dilation(preserve, CUBE, 2))
        skeleton &= ~preserve
        self.skeleton = skeleton
        self._split()
        self._sections()
        self._anchor()

    def centers(self, coords):
        return self.origin+(np.asarray(coords)+0.5)*self.h

    def _prune(self, skeleton, protected):
        removed = {"count": 0, "length_mm": 0.0}
        for _ in range(self.config["prune_passes"]):
            degree = _degree(skeleton)
            junction = skeleton & (degree >= 3)
            near = convolve(junction.astype(np.uint8), CUBE, mode="constant") > 0
            changed = False
            for coords in _groups(label(skeleton & ~junction, TWENTY_SIX)[0]):
                cells = tuple(coords.T)
                length = len(coords)*self.h
                if (degree[cells] == 1).any() and near[cells].any() and not protected[cells].any() and length < max(self.config["spur_minimum_mm"], self.config["spur_factor"]*self.radius[cells].max()):
                    skeleton[cells] = False
                    removed["count"] += 1
                    removed["length_mm"] += length
                    changed = True
            if not changed:
                break
        return removed

    def _split(self):
        degree = _degree(self.skeleton)
        junction = self.skeleton & (degree >= 3)
        nodes, _ = label(junction, TWENTY_SIX)
        self.nodes = [{"center": self.centers(c).mean(axis=0), "radius": float(self.radius[tuple(c.T)].max()), "voxels": c} for c in _groups(nodes)]
        self.members, self.graph = [], nx.MultiGraph()
        self.graph.add_nodes_from(range(1, len(self.nodes)+1))
        for coords in _groups(label(self.skeleton & ~junction, TWENTY_SIX)[0]):
            path = _order(coords)
            ends = [_neighbour_label(nodes, path[0]), _neighbour_label(nodes, path[-1])]
            for side, node in enumerate(ends):
                if not node:
                    ends[side] = f"end{len(self.members)}_{side}"
                    self.graph.add_node(ends[side])
            self.graph.add_edge(ends[0], ends[1], member=len(self.members))
            self.members.append({"voxels": path, "nodes": ends})

    def assign(self, weights):
        cells = np.argwhere((weights > 0) & ~self.preserve)
        return cells, self.ids[tuple(self.nearest[(slice(None),)+tuple(cells.T)])], weights[tuple(cells.T)].astype(float)

    def areas(self, weights=None):
        cells, owner, w = self.assign(self.weights if weights is None else weights)
        outward = np.zeros_like(self.tangents)
        for member in self.members:
            outward[member["rows"][0]] -= member["tangents"][0]
            outward[member["rows"][-1]] += member["tangents"][-1]
        beyond = np.einsum("ij,ij->i", (cells-self.nearest[(slice(None),)+tuple(cells.T)].T)*self.h, outward[owner]) > self.h/2
        counts = np.bincount(owner, weights=np.where(beyond, 0.0, w), minlength=len(self.tangents))
        result = []
        for member in self.members:
            step = np.linalg.norm(np.diff(self.centers(member["voxels"]), axis=0), axis=1)
            result.append(counts[member["rows"]]*self.h**3/np.maximum(np.r_[step, 0]/2+np.r_[0, step]/2, self.h/2))
        return result

    def _sections(self):
        h, cfg = self.h, self.config
        ids = np.full(self.skeleton.shape, -1, dtype=np.int64)
        voxels = np.argwhere(self.skeleton)
        ids[tuple(voxels.T)] = np.arange(len(voxels))
        self.ids = ids
        tangents = np.zeros((len(voxels), 3))
        for member in self.members:
            points = gaussian_filter1d(self.centers(member["voxels"]).astype(float), cfg["path_sigma_samples"], axis=0, mode="nearest") if len(member["voxels"]) > 2 else self.centers(member["voxels"])
            member["points"] = points
            tangent = np.gradient(points, axis=0) if len(points) > 1 else np.array([[1.0, 0.0, 0.0]])
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
            member["tangents"] = tangent
            tangents[ids[tuple(member["voxels"].T)]] = tangent
        self.tangents = tangents
        self.nearest = distance_transform_edt(~self.skeleton, return_indices=True)[1]
        cells, owner, weight = self.assign(self.weights)
        owners = self.nearest[(slice(None),)+tuple(cells.T)].T
        offsets = (cells-owners)*h
        along = np.einsum("ij,ij->i", offsets, tangents[owner])
        across = offsets-along[:, None]*tangents[owner]
        moments = np.zeros((len(voxels), 3, 3))
        np.add.at(moments, owner, weight[:, None, None]*across[:, :, None]*across[:, None, :])
        counts = np.bincount(owner, weights=weight, minlength=len(voxels))
        self.assigned = {"owner": owner, "cells": cells, "weight": weight}
        for node in self.nodes:
            node["volume"] = float(counts[ids[tuple(node["voxels"].T)]].sum()*h**3)
        for number, member in enumerate(self.members):
            rows = ids[tuple(member["voxels"].T)]
            n = gaussian_filter1d(counts[rows], cfg["section_sigma_samples"], mode="nearest")
            m = gaussian_filter1d(moments[rows], cfg["section_sigma_samples"], axis=0, mode="nearest")/np.maximum(n, 1e-9)[:, None, None]
            values, vectors = np.linalg.eigh(m)
            a = 2*np.sqrt(np.maximum(values[:, 2], 0)+h*h/12)
            b = 2*np.sqrt(np.maximum(values[:, 1], 0)+h*h/12)
            aspect = a/b
            a = np.minimum(np.maximum(a, b), cfg["maximum_aspect"]*b)
            axis = vectors[:, :, 2]
            for i in range(1, len(axis)):
                if axis[i] @ axis[i-1] < 0:
                    axis[i] = -axis[i]
            axis = gaussian_filter1d(axis, cfg["section_sigma_samples"], axis=0, mode="nearest")
            axis -= np.einsum("ij,ij->i", axis, member["tangents"])[:, None]*member["tangents"]
            axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
            step = np.linalg.norm(np.diff(member["points"], axis=0), axis=1)
            ds = np.maximum(np.r_[step, 0]/2+np.r_[0, step]/2, h if len(step) == 0 else 0)
            target = float(counts[rows].sum()*h**3)
            built = float(np.sum(np.pi*a*b*ds))
            scale = float(np.clip(np.sqrt(target/built), 2/3, 1.5)) if cfg["volume_match"] and built > 0 else 1.0
            member.update(a=a*scale, b=b*scale, axis=axis, target_volume=target, scale=scale, length=float(step.sum()+h), aspect=float(np.median(aspect)), kind="shell" if np.median(aspect) >= cfg["shell_aspect"] else "rod", rows=rows)
            if member["kind"] == "shell":
                member["shell"] = self._shell(np.isin(owner, rows))

    def _anchor(self):
        distance, index = distance_transform_edt(~self.preserve, return_indices=True)
        regions, _ = label(self.preserve, TWENTY_SIX)
        for number, member in enumerate(self.members):
            member["anchors"] = [None, None]
            for side, node in enumerate(member["nodes"]):
                if isinstance(node, int):
                    continue
                cell = tuple(member["voxels"][-side])
                if distance[cell]*self.h > float(member["b"][-side])+self.config["anchor_reach_mm"]:
                    continue
                target = tuple(index[(slice(None),)+cell])
                member["anchors"][side] = self.centers(np.array(target))
                anchor = f"preserve{regions[target]}"
                self.graph.add_edge(node, anchor, anchor=number)

    def _shell(self, mask):
        h, sigma = self.h, self.config["shell_sigma_mm"]
        points, weight = self.centers(self.assigned["cells"][mask]), self.assigned["weight"][mask]
        center = points.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov((points-center).T))
        frame = vectors[:, ::-1]
        local = (points-center)@frame
        low = local[:, :2].min(axis=0)-3*h
        shape = np.ceil((local[:, :2].max(axis=0)+3*h-low)/h).astype(int)+1
        cells = tuple(np.floor((local[:, :2]-low)/h).astype(int).T)
        count = np.zeros(shape)
        height = np.zeros(shape)
        np.add.at(count, cells, weight)
        np.add.at(height, cells, weight*local[:, 2])
        footprint = count > 0
        weight = gaussian_filter(footprint.astype(float), sigma/h)
        surface = gaussian_filter(np.where(footprint, height/np.maximum(count, 1), 0), sigma/h)/np.maximum(weight, 1e-9)
        thickness = gaussian_filter(count*h, sigma/h)/np.maximum(weight, 1e-9)
        outline = gaussian_filter((distance_transform_edt(footprint)-distance_transform_edt(~footprint))*h, self.config["shell_outline_sigma_mm"]/h)
        return {"center": center, "frame": frame, "low": low, "surface": surface, "thickness": thickness, "outline": outline, "bounds": (points.min(axis=0)-2.0, points.max(axis=0)+2.0)}

    def continuity(self):
        anchors = {node for node in self.graph if str(node).startswith("preserve")}
        parts = [set(part) for part in nx.connected_components(self.graph)]
        carrying = [part for part in parts if part & anchors]
        dangling = sum(1 for m in self.members for side, node in enumerate(m["nodes"]) if not isinstance(node, int) and m["anchors"][side] is None)
        main = max(carrying, key=lambda part: len(part & anchors), default=set())
        return {"anchors": len(anchors), "anchor_components": len(carrying), "anchors_in_main": len(main & anchors), "members_off_main": sum(1 for m in self.members if m["nodes"][0] not in main), "dangling_ends": dangling,
                "passed": len(carrying) == 1 and not any(m["nodes"][0] not in main for m in self.members)}

    def report(self):
        rods = [m for m in self.members if m["kind"] == "rod"]
        return {"nodes": len(self.nodes), "members": len(rods), "shells": len(self.members)-len(rods), "skeleton_voxels": int(self.skeleton.sum()), "graph_components": nx.number_connected_components(self.graph), "spurs_dropped": self.spurs, "continuity": self.continuity(),
                "member_length_mm": float(sum(m["length"] for m in rods)), "thin_member_length_mm": float(sum(m["length"]*np.mean(2*m["b"] < 2.0) for m in rods)),
                "median_aspect": float(np.median([m["aspect"] for m in self.members])) if self.members else 0.0, "volume_scale": [float(np.min([m["scale"] for m in self.members])), float(np.max([m["scale"] for m in self.members]))] if self.members else [],
                "node_volume_mm3": float(sum(n["volume"] for n in self.nodes))}

def _box(low, high):
    return {"kind": "box", "min_mm": low, "max_mm": high}

def sweep_values(axes, points, a, b, axis, flat=(False, False)):
    values = np.full(np.broadcast_shapes(*(x.shape for x in axes)), -np.inf, dtype=np.float32)
    x = np.stack(np.broadcast_arrays(*axes), axis=-1)
    last = max(len(points)-1, 1)
    for i in range(last):
        j = min(i+1, len(points)-1)
        start, delta = points[i], points[j]-points[i]
        length = float(delta@delta)
        t = np.clip(((x-start)@delta)/length, 0, 1) if length > 1e-18 else np.zeros(x.shape[:-1])
        tangent = delta/np.sqrt(length) if length > 1e-18 else np.array([1.0, 0.0, 0.0])
        q = x-start-t[..., None]*delta
        major = axis[i]+t[..., None]*(axis[j]-axis[i])
        major = major-(major@tangent)[..., None]*tangent
        major /= np.maximum(np.linalg.norm(major, axis=-1, keepdims=True), 1e-12)
        minor = np.cross(tangent, major)
        ai, bi = a[i]+t*(a[j]-a[i]), b[i]+t*(b[j]-b[i])
        axial = q@tangent
        cut = ((axial < 0) & flat[0] & (i == 0)) | ((axial > 0) & flat[1] & (i == last-1))
        rho = np.sqrt((np.einsum("...k,...k->...", q, major)/ai)**2+(np.einsum("...k,...k->...", q, minor)/bi)**2+np.where(cut, 0, axial/bi)**2)
        np.maximum(values, np.where(cut, np.minimum(bi*(1-rho), -np.abs(axial)), bi*(1-rho)).astype(np.float32), out=values)
    return values

def shell_values(axes, shell):
    x = np.stack(np.broadcast_arrays(*axes), axis=-1)
    local = (x-shell["center"])@shell["frame"]
    grid = [(local[..., i]-shell["low"][i])/shell["h"] for i in range(2)]
    sample = lambda field: map_coordinates(field, grid, order=1, mode="nearest")
    return np.minimum(sample(shell["thickness"])/2-np.abs(local[..., 2]-sample(shell["surface"])), sample(shell["outline"])).astype(np.float32)

def smoothed_density(density, config):
    field = np.asarray(density, dtype=np.float32)
    return gaussian_filter(field, config["density_sigma_cells"]) if config["density_sigma_cells"] > 0 else field

def reference_body(domain, density, config=DESIGN_RECONSTRUCTION_CONFIG):
    started = perf_counter()
    sampled, origin, spacing = _upsample(domain, smoothed_density(density, config), config["reference_subdivisions"], "pchip")
    mesh = ImplicitField(origin, spacing, sampled-config["threshold"]).extract()[0]
    parts = mesh.split(only_watertight=False)
    largest = max(parts, key=lambda part: abs(part.volume))
    mesh, booleans = exact_booleans(largest, domain, IMPLICIT_CONFIG)
    cell = float(np.prod(domain["grid"]["spacing_mm"]))
    return mesh, {"density_integral_mm3": float(np.sum(density)*cell), "threshold_voxels_mm3": float(np.count_nonzero(np.asarray(density) > config["threshold"])*cell), "extracted_mm3": float(largest.volume), "volume_mm3": float(mesh.volume),
                  "debris_mm3": float(sum(abs(p.volume) for p in parts if p is not largest)), "booleans_passed": bool(booleans["passed"]), "bodies": len(mesh.split(only_watertight=False)), "watertight": bool(mesh.is_watertight), "runtime_s": perf_counter()-started,
                  "method": "density smoothed by density_sigma_cells, PCHIP-upsampled, marching cubes at the threshold, largest body, exact forbidden/envelope booleans"}

class Reconstruction:
    def __init__(self, domain, config=DESIGN_RECONSTRUCTION_CONFIG, density=None):
        self.domain, self.config = domain, config
        self.density = None if density is None else smoothed_density(density, config)
        envelope = _envelope(domain)
        h, pad = config["voxel_mm"], 1.0
        low = np.asarray(envelope["min_mm"], dtype=float)-pad+h/2
        shape = tuple(np.ceil((np.asarray(envelope["max_mm"], dtype=float)+pad-low)/h).astype(int)+1)
        self.field = ImplicitField(low, h, np.full(shape, -pad, dtype=np.float32))

    def _blend(self, window, values, radius=0.0):
        part = ImplicitField(self.field.origin, self.field.spacing, self.field.values[window])
        part.smooth_union(values, radius)
        self.field.values[window] = part.values

    def raw_values(self, axes):
        grid = self.domain["grid"]
        spacing, origin = np.asarray(grid["spacing_mm"], dtype=float), np.asarray(grid["origin_mm"], dtype=float)
        x = np.stack(np.broadcast_arrays(*axes), axis=0)
        index = (x-origin.reshape(3, 1, 1, 1))/spacing.reshape(3, 1, 1, 1)-0.5
        value = map_coordinates(self.density, index, order=1, mode="constant", cval=0.0)
        slope = np.sqrt(sum(map_coordinates(g, index, order=1, mode="nearest")**2 for g in np.gradient(self.density, *spacing)))
        return ((value-self.config["threshold"])/np.maximum(slope, self.config["threshold"]/self.config["root_slope_floor_mm"])).astype(np.float32)

    def add_roots(self):
        cfg = self.config
        reach = cfg["root_distance_mm"]+cfg["root_taper_mm"]
        for region in (r for r in self.domain["regions"] if r["role"] == "preserve" and r["name"].endswith(cfg["root_preserves"])):
            window = self.field.window(region, reach)
            if window is None:
                continue
            axes = self.field.axes(window)
            outside = np.maximum(-primitive_distance(axes, region), 0)
            raw = self.raw_values(axes)-np.maximum(outside-cfg["root_distance_mm"], 0)*cfg["root_taper_slope"]
            self._blend(window, np.where(outside <= reach, raw, -np.inf).astype(np.float32))

    def add_bridges(self):
        gap, regions, count = self.config["bridge_gap_mm"], [r for r in self.domain["regions"] if r["role"] == "preserve"], 0
        for i, a in enumerate(regions):
            window = self.field.window(a, gap)
            if window is None:
                continue
            axes = self.field.axes(window)
            near = -primitive_distance(axes, a) <= gap
            zone = np.zeros(near.shape, dtype=bool)
            for b in regions[i+1:]:
                zone |= near & (-primitive_distance(axes, b) <= gap)
            if zone.any():
                count += 1
                self._blend(window, np.where(zone, self.raw_values(axes), -np.inf).astype(np.float32), self.config["transition_radius_mm"])
        self.bridges = count

    def preserve_values(self, regions, pad):
        values = np.full(self.field.values.shape, -np.inf, dtype=np.float32)
        flush, r = self.config["preserve_flush_mm"], self.config["preserve_round_mm"]
        for region in regions:
            window = self.field.window(region, pad)
            if window is None:
                continue
            axes = self.field.axes(window)
            value = primitive_distance(axes, shrunk_region(region, r))+r
            if region["kind"] == "box" or region.get("axis", "z") == "z":
                value = np.minimum(value, region_bounds(region)[1][2]-flush-axes[2])
            np.maximum(values[window], value.astype(np.float32), out=values[window])
        return values

    def _fit(self, axes, values, core, key="raw_margin_mm"):
        margin = self.config.get(key, -1.0)
        return values if self.density is None or margin < 0 else np.maximum(np.minimum(values, self.raw_values(axes)+margin), core)

    def add_member(self, member, h, blend=0.0):
        k, r = self.config["transition_radius_mm"], self.config["minimum_radius_mm"]
        if member["kind"] == "shell":
            shell = dict(member["shell"], h=h)
            window = self.field.window(_box(*shell["bounds"]), k)
            if window is not None:
                axes = self.field.axes(window)
                values = np.maximum(shell_values(axes, shell), sweep_values(axes, member["points"], member["b"], member["b"], member["axis"]))
                self._blend(window, self._fit(axes, values, sweep_values(axes, member["points"], np.full(len(member["b"]), r), np.full(len(member["b"]), r), member["axis"])), blend)
            return
        reach = float(np.max(member["a"]))+2*self.field.spacing[0]
        window = self.field.window(_box(member["points"].min(axis=0), member["points"].max(axis=0)), reach+k)
        if window is None:
            return
        flat = [isinstance(node, int) for node in member["nodes"]]
        def swept(a, b):
            values = np.full(tuple(s.stop-s.start for s in window), -np.inf, dtype=np.float32)
            for i in range(max(len(member["points"])-1, 1)):
                j = min(i+1, len(member["points"])-1)
                pair = member["points"][[i, j]]
                local = self.field.window(_box(pair.min(axis=0), pair.max(axis=0)), max(a[i], a[j])+2*self.field.spacing[0])
                if local is None:
                    continue
                local = tuple(slice(max(l.start, w.start), min(l.stop, w.stop)) for l, w in zip(local, window))
                if any(s.stop <= s.start for s in local):
                    continue
                target = tuple(slice(l.start-w.start, l.stop-w.start) for l, w in zip(local, window))
                np.maximum(values[target], sweep_values(self.field.axes(local), pair, a[[i, j]], b[[i, j]], member["axis"][[i, j]], (i == 0 and flat[0], j == len(member["points"])-1 and flat[1])), out=values[target])
            return values
        values = swept(member["a"], member["b"])
        if self.density is not None and self.config.get("rod_raw_margin_mm", -1.0) >= 0:
            values = self._fit(self.field.axes(window), values, swept(*[np.full(len(member["a"]), r)]*2), "rod_raw_margin_mm")
        self._blend(window, values, blend)

    def add_node(self, node, blend):
        region = {"kind": "sphere", "center_mm": node["center"], "radius_mm": node["radius"]}
        window = self.field.window(region, 2*blend)
        if window is not None:
            axes = self.field.axes(window)
            self._blend(window, self._fit(axes, primitive_distance(axes, region), primitive_distance(axes, dict(region, radius_mm=min(node["radius"], self.config["minimum_radius_mm"])))).astype(np.float32), blend)

    def lift_contacts(self, faces, reach, over, radial=None):
        h = float(self.field.spacing[0])
        for axis, sign, plane, low, high in faces:
            bounds = np.array([low, high])
            bounds[:, axis] = sorted((plane-sign*reach, plane+sign*over))
            window = self.field.window(_box(*bounds), 0.0)
            if window is None:
                continue
            axes, values = self.field.axes(window), self.field.values[window]
            depth = sign*(axes[axis]-plane)
            first, second = [(axes[i] >= low[i]+h) & (axes[i] <= high[i]-h) for i in range(3) if i != axis]
            inside = first & second
            if radial is not None:
                center, radius = radial
                inside &= sum((axes[i]-center[i])**2 for i in range(3) if i != axis) <= radius**2
            peak = np.where((depth >= -reach) & (depth <= 0), values, -np.inf).max(axis=axis, keepdims=True)
            self.field.values[window] = np.where(inside & (depth >= -reach) & (depth <= over) & (peak > 0), np.maximum(values, peak), values)

    def camera_contacts(self):
        radius = self.config.get("camera_contact_radius_mm", 0.0)
        regions = {r["name"]: r for r in self.domain["regions"]}
        if radius <= 0 or "camera_envelope" not in regions or "camera_screw_axis" not in regions:
            return
        low, high = region_bounds(regions["camera_envelope"])
        center = np.asarray(regions["camera_screw_axis"]["center_mm"], dtype=float)
        first, last = np.maximum(low, center-radius-self.field.spacing[0]), np.minimum(high, center+radius+self.field.spacing[0])
        faces = [(0, sign, float(plane), first, last) for sign, plane in ((1, low[0]), (-1, high[0]))]
        reach = float(max(self.domain["grid"]["spacing_mm"]))+self.config["boolean_offset_mm"]
        self.lift_contacts(faces, reach, abs(self.config["preserve_flush_mm"]), (center, radius))

    def finish(self):
        h = self.field.spacing[0]
        preserves = [r for r in self.domain["regions"] if r["role"] == "preserve"]
        forbidden = [r for r in self.domain["regions"] if r["role"] == "forbidden"]
        offset = np.float32(self.config["boolean_offset_mm"])
        if self.config["closing_radius_mm"] > 0:
            gaps = ImplicitField(self.field.origin, self.field.spacing, -self.field.values)
            gaps.open(self.config["closing_radius_mm"])
            self.field.values = -gaps.values
        self.field.smooth(self.config["member_smooth_mm"])
        if self.density is not None and self.config["root_distance_mm"] > 0:
            self.add_roots()
        if self.density is not None and self.config.get("bridge_gap_mm", 0) > 0:
            self.add_bridges()
        clip = lambda margin: self.field.intersect(-self.field.primitives(forbidden, margin+3*h)-margin)
        clip(offset)
        self.lift_contacts(contact_faces(self.domain, self.config.get("contact_regions", [])), self.config.get("contact_reach_mm", 0.0), abs(self.config["preserve_flush_mm"]))
        self.camera_contacts()
        self.field.smooth_union(self.preserve_values(preserves, self.config["preserve_blend_mm"]+3*h), self.config["preserve_blend_mm"])
        clip(np.float32(self.config["preserve_flush_mm"]))
        self.field.intersect(self.field.primitives([_envelope(self.domain)])+offset)
        return self.field.extract()

def contact_faces(domain, names):
    faces = {}
    for region in (r for r in domain["regions"] if r["name"] in names and r["kind"] == "box"):
        low, high = region_bounds(region)
        for other in (r for r in domain["regions"] if r["role"] == "forbidden" and r["kind"] == "box"):
            other_low, other_high = region_bounds(other)
            start, stop = np.maximum(low, other_low), np.minimum(high, other_high)
            for axis in range(3):
                rest = [i for i in range(3) if i != axis]
                for sign, plane, face in ((1, high[axis], other_low[axis]), (-1, low[axis], other_high[axis])):
                    if abs(plane-face) < 1e-6 and np.all(stop[rest] > start[rest]):
                        faces[(axis, sign, round(float(plane), 6), *np.round(start, 6), *np.round(stop, 6))] = (axis, sign, float(plane), start, stop)
    return list(faces.values())

def shrunk_region(region, r):
    if region["kind"] == "box":
        return dict(region, min_mm=(np.asarray(region["min_mm"], dtype=float)+r).tolist(), max_mm=(np.asarray(region["max_mm"], dtype=float)-r).tolist())
    return dict(region, radius_mm=region["radius_mm"]-r, height_mm=region["height_mm"]-2*r)

def core_domain(domain, config):
    return dict(domain, regions=[shrunk_region(r, config["preserve_round_mm"]) if r["role"] == "preserve" else r for r in domain["regions"]])

def node_radius(node):
    return float(min((3*node["volume"]/(4*np.pi))**(1/3), node["radius"]))

def sections(a, b, scale, config):
    b = np.maximum(np.asarray(b, dtype=float)*scale, config["minimum_radius_mm"])
    return np.clip(np.asarray(a, dtype=float)*scale, b, config["maximum_aspect"]*b), b

def stored_domain(path):
    from deep_frame.topology_geometry import grid_centers, rasterize_regions
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    domain = stored.get("domain", stored)
    masks = rasterize_regions(domain["grid"], domain["regions"])
    centers = grid_centers(domain["grid"])
    hoop = domain.get("metadata", {}).get("round2", {}).get("hoop")
    for path in hoop_paths(hoop):
        masks["preserve"] |= sweep_values([centers[..., 0], centers[..., 1], centers[..., 2]], *[tube(path, hoop["radius_mm"])[key] for key in ("points", "a", "b", "axis")]) >= 0
    masks["preserve"] &= masks["allowed"]
    return {**domain, **masks}

def hoop_paths(hoop):
    return [np.array([[sign*hoop["x_mm"], y, z] for y, z in hoop["path_yz_mm"]], dtype=float) for sign in (-1, 1)] if hoop else []

def tube(points, radius):
    tangent = np.gradient(points, axis=0)
    axis = np.cross(tangent, np.where(np.abs(tangent[:, :1]) < 0.9*np.linalg.norm(tangent, axis=1, keepdims=True), [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return {"kind": "rod", "points": points, "a": np.full(len(points), radius), "b": np.full(len(points), radius), "axis": axis, "nodes": ["hoop", "hoop"]}

def extended(member, nodes, scale, config):
    ends = [nodes[n-1]["center"] if isinstance(n, int) else member["anchors"][side] for side, n in enumerate(member["nodes"])]
    pad = lambda values: np.concatenate([values[:1]]*(ends[0] is not None)+[values]+[values[-1:]]*(ends[1] is not None))
    points = np.concatenate([[ends[0]]]*(ends[0] is not None)+[member["points"]]+[[ends[1]]]*(ends[1] is not None))
    a, b = sections(pad(member["a"]), pad(member["b"]), scale, config)
    return dict(member, points=points, a=a, b=b, axis=pad(member["axis"]), nodes=["joined", "joined"])

def joints(graph, rods):
    result = []
    for member, rod in zip(graph.members, rods):
        for side, node in enumerate(member["nodes"]):
            if isinstance(node, int) and rod is not None and len(rod["points"]) > 2:
                end, inner = (0, 1) if side == 0 else (-1, -2)
                result.append({"node": node, "center": rod["points"][end], "direction": rod["points"][inner]-rod["points"][end], "a": float(rod["a"][end]), "b": float(rod["b"][end])})
    return result

def joint_sections(field, items, radii, h):
    rows = []
    for item in items:
        radius = radii[item["node"]-1]
        direction = item["direction"]/max(np.linalg.norm(item["direction"]), 1e-12)
        u = np.cross(direction, [1.0, 0.0, 0.0] if abs(direction[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        step = h/2
        offsets = np.arange(-item["a"]*1.5, item["a"]*1.5+step, step)
        s, t = np.meshgrid(offsets, offsets, indexing="ij")
        disk = s**2+t**2 <= (1.5*item["a"])**2
        points = item["center"]+radius*direction+s[disk][:, None]*u+t[disk][:, None]*v
        inside = map_coordinates(field.values, ((points-field.origin)/field.spacing).T, order=1, mode="constant", cval=-1.0) > 0
        rows.append(float(inside.sum()*step**2/(np.pi*item["a"]*item["b"])))
    worst = [{"node": items[i]["node"], "center_mm": np.round(items[i]["center"], 2).tolist(), "ratio": rows[i]} for i in np.argsort(rows)[:5]]
    rows = np.asarray(rows)
    return {"count": len(rows), "worst": worst, "minimum_ratio": float(rows.min()) if len(rows) else None, "p05_ratio": float(np.percentile(rows, 5)) if len(rows) else None, "below_one": int(np.sum(rows < 0.95)),
            "method": "section area of the final field on the plane through each member at the node body radius (disk 1.5 a) over the incident member section pi a b; below_one counts ratios < 0.95 (sampling at h/2)"}

def build_field(graph, domain, config, scale=1.0, density=None, check=False):
    builder = Reconstruction(domain, config, density)
    rods = [extended(member, graph.nodes, scale, config) if member["kind"] == "rod" else None for member in graph.members]
    for rod in rods:
        if rod is not None:
            builder.add_member(rod, graph.h)
    add_shells(builder, graph, domain, config, scale)
    items = joints(graph, rods)
    radii = [node_radius(dict(node, volume=node["volume"]*scale**2)) for node in graph.nodes]
    blends = [[] for _ in graph.nodes]
    for item in items:
        radii[item["node"]-1] = max(radii[item["node"]-1], item["b"])
        blends[item["node"]-1].append(item["b"])
    for node, radius, blend in zip(graph.nodes, radii, blends):
        builder.add_node(dict(node, radius=radius), config["joint_blend_factor"]*float(np.mean(blend)) if blend else config["transition_radius_mm"])
    mesh, extraction = builder.finish()
    report = {**extraction, "node_spheres_mm3": float(sum(4/3*np.pi*r**3 for r in radii))}
    if check:
        report["joint_sections"] = joint_sections(builder.field, items, radii, builder.field.spacing[0])
    return mesh, report

def _volume(graph, domain, config, scale, density):
    return float(exact_booleans(build_field(graph, domain, {**config, "voxel_mm": config["calibration_voxel_mm"]}, scale, density)[0], core_domain(domain, config), IMPLICIT_CONFIG)[0].volume)

def reconstruct(domain, density, config=DESIGN_RECONSTRUCTION_CONFIG, target=None):
    started, times = perf_counter(), {}
    grid = domain["grid"]
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    solid, fragments = solid_body(density, config, float(np.prod(spacing)))
    graph = DesignGraph(solid, grid["origin_mm"], spacing[0], domain["preserve"], config)
    times["skeleton_s"] = perf_counter()-started
    target = target or config["target_volume_mm3"] or float(solid.sum()*np.prod(spacing))
    low, high = config["minimum_scale"], config["maximum_scale"]
    trials = [[1.0, _volume(graph, domain, config, 1.0, density)]] if config["calibrate_volume"] and config["calibration_steps"] else []
    if trials and abs(trials[0][1]/target-1) > config["calibration_tolerance"]:
        low, high = (low, 1.0) if trials[0][1] > target else (1.0, high)
        for _ in range(config["calibration_steps"]):
            scale = (low+high)/2
            trials.append([scale, _volume(graph, domain, config, scale, density)])
            low, high = (scale, high) if trials[-1][1] < target else (low, scale)
        scale = min(trials, key=lambda trial: abs(trial[1]/target-1))[0]
    else:
        scale = 1.0
    times["calibration_s"] = perf_counter()-started-times["skeleton_s"]
    mesh, extraction = build_field(graph, domain, config, scale, density, True)
    times["field_extract_s"] = perf_counter()-started-times["skeleton_s"]-times["calibration_s"]
    mesh, final = finalize(mesh, domain, config)
    times["booleans_s"] = perf_counter()-started-sum(times.values())
    rods = [m for m in graph.members if m["kind"] == "rod"]
    budget = {"target_mm3": target, "assigned_members_mm3": float(sum(m["target_volume"] for m in graph.members)), "built_members_mm3": float(sum(np.sum(np.pi*m["a"]*m["b"]*scale**2*m["length"]/max(len(m["a"]), 1)) for m in rods)),
              "assigned_nodes_mm3": float(sum(n["volume"] for n in graph.nodes)), "node_spheres_mm3": extraction["node_spheres_mm3"], "preserves_mm3": float(np.count_nonzero(domain["preserve"])*np.prod(spacing)), "result_mm3": float(mesh.volume), "relative_to_target": float(mesh.volume/target-1)}
    report = {**graph.report(), "fragments_dropped": fragments, "target_volume_mm3": target, "global_scale": scale, "thickness": "calibrated" if config["calibrate_volume"] else "field", "volume_match": config["volume_match"], "calibration": trials, "mass_budget": budget, **final, "solid_volume_mm3": float(solid.sum()*np.prod(spacing)),
              "extraction": {key: extraction[key] for key in ("mesh", "runtime_s")}, "joint_sections": extraction["joint_sections"], "voxel_mm": config["voxel_mm"], "transition_radius_mm": config["transition_radius_mm"], "preserve_blend_mm": config["preserve_blend_mm"], "runtime_s": {**times, "total_s": perf_counter()-started}}
    return mesh, graph, report

def add_shells(builder, graph, domain, config, scale=1.0, blend=0.0, extend=False):
    for member in graph.members:
        if member["kind"] == "shell":
            a, b = sections(member["a"], member["b"], scale, config)
            member = dict(extended(member, graph.nodes, scale, config), kind="shell") if extend else dict(member, a=a, b=b)
            builder.add_member(dict(member, shell=dict(member["shell"], thickness=np.maximum(member["shell"]["thickness"]*scale, config.get("minimum_shell_mm", 2*config["minimum_radius_mm"])))), graph.h, blend)
    hoop = domain.get("metadata", {}).get("round2", {}).get("hoop")
    for path in hoop_paths(hoop):
        builder.add_member(tube(path, max(hoop["radius_mm"], config["minimum_radius_mm"])), graph.h, blend)

def finalize(mesh, domain, config):
    parts = mesh.split(only_watertight=False)
    largest = max(parts, key=lambda part: abs(part.volume))
    mesh, booleans = exact_booleans(largest, core_domain(domain, config), IMPLICIT_CONFIG)
    return mesh, {"extraction_debris": {"count": len(parts)-1, "volume_mm3": float(sum(abs(p.volume) for p in parts if p is not largest))}, "volume_mm3": float(mesh.volume), "watertight": bool(mesh.is_watertight), "bodies": len(mesh.split(only_watertight=False)),
                  "exact_booleans": {key: booleans[key] for key in ("status", "components", "passed", "preserve_added_mm3", "forbidden_removed_mm3", "envelope_removed_mm3", "maximum_cylinder_oversize_mm")}}

def body_weights(mesh, grid, subdivisions):
    from deep_frame.topology_implicit_validation import occupancy
    spacing, origin, shape = np.asarray(grid["spacing_mm"], dtype=float), np.asarray(grid["origin_mm"], dtype=float), tuple(grid["shape"])
    h = float(spacing[0])/subdivisions
    solid, lower, _ = occupancy(mesh, h, 2*h)
    index = np.floor((lower+(np.argwhere(solid)+0.5)*h-origin)/spacing).astype(int)
    index = index[np.all((index >= 0) & (index < shape), axis=1)]
    weights = np.zeros(shape, dtype=np.float32)
    np.add.at(weights, tuple(index.T), np.float32(1/subdivisions**3))
    return np.minimum(weights, 1)

def spline_basis(t, count):
    degree = min(3, count-1)
    knots = np.r_[[0.0]*degree, np.linspace(0, 1, count-degree+1), [1.0]*degree]
    return BSpline.design_matrix(np.clip(t, 0, 1), knots, degree).toarray()

def fit_spline(path, count, samples):
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    t = t/max(t[-1], 1e-12)
    basis = spline_basis(t, count)
    inner = np.linalg.lstsq(basis[:, 1:-1], path-basis[:, :1]*path[0]-basis[:, -1:]*path[-1], rcond=None)[0]
    control = np.vstack([path[0], inner, path[-1]])
    return control, spline_basis(np.linspace(0, 1, samples), count)@control, t

def fit_profile(t, values, keep, degree):
    keep = keep if keep.sum() >= 2 else np.ones(len(t), dtype=bool)
    if keep.sum() < 2:
        return np.array([float(np.mean(values))]), (0.0, 1.0)
    return np.polyfit(t[keep], values[keep], min(degree, int(keep.sum())-1)), (float(t[keep].min()), float(t[keep].max()))

def _slope(value, points, eps=0.05):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    slope = np.stack([value(points+e)-value(points-e) for e in np.eye(3)*eps], axis=1)/(2*eps)
    return value(points), slope/np.maximum(np.linalg.norm(slope, axis=1, keepdims=True), 1e-9)

def clearance(domain, points):
    forbidden = [r for r in domain["regions"] if r["role"] == "forbidden"]
    return _slope(lambda x: np.minimum.reduce([primitive_distance([x[:, 0], x[:, 1], x[:, 2]], _envelope(domain))]+[-primitive_distance([x[:, 0], x[:, 1], x[:, 2]], r) for r in forbidden]), points)

def snap_anchors(graph, domain, config):
    preserves = [r for r in domain["regions"] if r["role"] == "preserve"]
    value = lambda x: np.maximum.reduce([primitive_distance([x[:, 0], x[:, 1], x[:, 2]], r) for r in preserves])
    moved = []
    for member in graph.members:
        for side, anchor in enumerate(member["anchors"]):
            if anchor is None or not preserves:
                continue
            point = np.asarray(anchor, dtype=float).copy()
            for _ in range(config["clearance_iterations"]):
                d, n = _slope(value, point)
                if d[0] < -config["anchor_snap_mm"] or d[0] >= config["anchor_inset_mm"]-config["clearance_tolerance_mm"]:
                    break
                point = point+(config["anchor_inset_mm"]-d[0])*n[0]
            moved.append(float(np.linalg.norm(point-anchor)))
            member["anchors"][side] = point
    return {"anchors": len(moved), "moved": sum(m > config["clearance_tolerance_mm"] for m in moved), "maximum_shift_mm": max(moved, default=0.0)}

def seat_nodes(graph, domain, config):
    moved = []
    for node in graph.nodes:
        start = np.asarray(node["center"], dtype=float).copy()
        for _ in range(config["clearance_iterations"]):
            c, n = clearance(domain, node["center"])
            deficit = float(node["radius"]+config["clearance_margin_mm"]-c[0])
            if deficit <= config["clearance_tolerance_mm"]:
                break
            node["center"] = np.asarray(node["center"], dtype=float)+min(deficit, node["radius"])*n[0]
        moved.append(float(np.linalg.norm(node["center"]-start)))
    return {"moved": sum(m > 0 for m in moved), "maximum_shift_mm": max(moved, default=0.0)}

def frame_axes(curve, major):
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    axis = major-np.sum(tangent*major, axis=-1)[:, None]*tangent
    fallback = np.cross(tangent, np.where(np.abs(tangent[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]))
    axis = np.where(np.linalg.norm(axis, axis=1, keepdims=True) > 1e-3, axis, fallback)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return axis, np.cross(tangent, axis)

def seat_curve(domain, control, curve, a, b, major, count, config):
    shift = 0.0
    for _ in range(config["clearance_iterations"]):
        axis, side = frame_axes(curve, major)
        c, n = clearance(domain, curve)
        need = np.sqrt((a*np.einsum("ij,ij->i", axis, n))**2+(b*np.einsum("ij,ij->i", side, n))**2)+config["clearance_margin_mm"]
        deficit = np.minimum(np.maximum(need-c, 0.0), a)
        deficit[[0, -1]] = 0.0
        if deficit.max() <= config["clearance_tolerance_mm"]:
            break
        shift = max(shift, float(deficit.max()))
        control, curve, _ = fit_spline(curve+deficit[:, None]*n, count, len(curve))
    return control, curve, shift

def profile_values(profile, t):
    coefficients, (low, high) = profile
    return np.polyval(coefficients, np.clip(t, low, high))

def plane_sections(weights, origin, h, curve, reach, threshold=0.5):
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    u = np.cross(tangent, np.where(np.abs(tangent[:, 2:]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tangent, u)
    offsets = np.arange(-reach, reach+h/4, h/2)
    s, t = np.meshgrid(offsets, offsets, indexing="ij")
    points = curve[:, None, None]+s[None, ..., None]*u[:, None, None]+t[None, ..., None]*v[:, None, None]
    values = map_coordinates(np.asarray(weights, dtype=np.float32), ((points-origin)/h-0.5).reshape(-1, 3).T, order=1, mode="constant").reshape(points.shape[:3])
    tensors, found = np.zeros((len(curve), 3, 3)), np.zeros(len(curve), dtype=bool)
    for i, value in enumerate(values):
        labels, count = label(value > threshold)
        if not count:
            continue
        w = value*(labels == labels.flat[np.argmin(np.where(labels > 0, s**2+t**2, np.inf))])
        q = s[..., None]*u[i]+t[..., None]*v[i]
        q = q-np.einsum("ij,ijk->k", w, q)/w.sum()
        tensors[i], found[i] = np.einsum("ij,ijk,ijl->kl", w, q, q)/w.sum(), True
    return tensors, found

def section_shape(graph, curve, valid, member, config):
    tensors, found = plane_sections(graph.weights, graph.origin, graph.h, curve, config["section_reach_mm"])
    valid = valid & found
    if not valid.any():
        major = np.mean(member["axis"], axis=0)
        return np.full(len(curve), float(np.clip(member["aspect"], 1.0, config["maximum_aspect"]))), np.tile(major if np.linalg.norm(major) > 1e-6 else [0.0, 0.0, 1.0], (len(curve), 1))
    rows = np.flatnonzero(valid)
    step = max(float(np.linalg.norm(np.diff(curve, axis=0), axis=1).mean()), 1e-9) if len(curve) > 1 else 1.0
    tensors = gaussian_filter1d(tensors[np.clip(np.arange(len(curve)), rows[0], rows[-1])], config["section_smooth_mm"]/step, axis=0, mode="nearest")
    values, vectors = np.linalg.eigh(tensors)
    major = vectors[:, :, 2]
    for i in range(1, len(major)):
        if major[i] @ major[i-1] < 0:
            major[i] = -major[i]
    return np.clip(np.sqrt(np.maximum(values[:, 2], 1e-12)/np.maximum(values[:, 1], 1e-12)), 1.0, config["maximum_aspect"]), major

def spline_member(graph, member, area, config, domain=None):
    ends = [graph.nodes[n-1]["center"] if isinstance(n, int) else member["anchors"][side] for side, n in enumerate(member["nodes"])]
    inner = graph.centers(member["voxels"])
    path = np.concatenate([[ends[0]]]*(ends[0] is not None)+[inner]+[[ends[1]]]*(ends[1] is not None))
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else graph.h
    count = 3+sum(length >= limit for limit in config["spline_lengths_mm"])
    control, curve, t = fit_spline(path, count, max(int(np.ceil(length/config["spline_sample_mm"]))+1, 4))
    t = t[int(ends[0] is not None):len(t)-int(ends[1] is not None)]
    reach = [graph.nodes[n-1]["radius"] if isinstance(n, int) else config["transition_radius_mm"] for n in member["nodes"]]
    keep = (t*length > reach[0]) & ((1-t)*length > reach[1])
    arc = max(float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()), 1e-12)
    area = area*length/arc
    measured = gaussian_filter1d(area, config["profile_sigma_samples"], mode="nearest") if config["profile_sigma_samples"] > 0 and len(area) > 1 else area
    profile = fit_profile(t, measured, keep, config["profile_degree"])
    samples = np.linspace(0, 1, len(curve))
    aspect, major = section_shape(graph, curve, (samples >= profile[1][0]) & (samples <= profile[1][1]), member, config)
    fitted = np.maximum(profile_values(profile, samples), 0.0)
    b = np.sqrt(fitted/(np.pi*aspect))
    a, b = np.maximum(aspect*b, config["minimum_radius_mm"]), np.maximum(b, config["minimum_radius_mm"])
    control, curve, shift = seat_curve(domain, control, curve, a, b, major, count, config) if domain is not None else (control, curve, 0.0)
    return {"kind": "rod", "points": curve, "a": a, "b": b, "axis": frame_axes(curve, major)[0], "nodes": ["joined", "joined"], "control": control, "profile": profile, "t": t, "keep": keep, "length": arc, "path_length": length, "aspect": float(np.median(aspect)), "stretch": length/arc, "seat_shift_mm": shift}

def _from_end(rod, side, step):
    points, b = (rod["points"], rod["b"]) if side == 0 else (rod["points"][::-1], rod["b"][::-1])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    s = np.arange(0, arc[-1], step)
    return np.stack([np.interp(s, arc, points[:, k]) for k in range(3)], axis=1), np.interp(s, arc, b)

def _raw(graph, points):
    return map_coordinates(graph.weights, ((np.atleast_2d(points)-graph.origin)/graph.h-0.5).T, order=1, mode="constant")

def _run(values, centre):
    inside = values > 0.5
    if not inside[centre]:
        return 0
    low, high = centre, centre
    while low > 0 and inside[low-1]:
        low -= 1
    while high < len(inside)-1 and inside[high+1]:
        high += 1
    return high-low+1

def fork_webs(graph, rods, config):
    step, webs = graph.h/2, []
    ends = [(number, side, rod["points"][-side]) for number, rod in enumerate(rods) if rod is not None and len(rod["points"]) > 2 for side in (0, 1)]
    for x, (i, si, ei) in enumerate(ends):
        for j, sj, ej in ends[x+1:]:
            if i == j or np.linalg.norm(ei-ej) > config["web_split_mm"]:
                continue
            (pi, bi), (pj, bj) = _from_end(rods[i], si, step), _from_end(rods[j], sj, step)
            count = min(len(pi), len(pj), int(config["web_length_mm"]/step))
            if count < 3 or np.degrees(np.arccos(np.clip(np.dot(*[(p[2]-p[0])/max(np.linalg.norm(p[2]-p[0]), 1e-9) for p in (pi, pj)]), -1, 1))) > config["web_angle_deg"]:
                continue
            last, open_, thickness = 0, 0, []
            for k in range(count):
                span = pj[k]-pi[k]
                width = float(np.linalg.norm(span))
                gap = width-bi[k]-bj[k]
                if gap > graph.h:
                    direction = span/width
                    inner = pi[k]+direction*np.arange(bi[k], width-bj[k], step)[:, None]
                    if np.mean(_raw(graph, inner) > 0.5) < config["web_fill"]:
                        break
                    open_ += 1
                    middle = (pi[k]+pj[k])/2
                    normal = np.cross(span, (pi[min(k+1, count-1)]+pj[min(k+1, count-1)])/2-(pi[max(k-1, 0)]+pj[max(k-1, 0)])/2)
                    if np.linalg.norm(normal) > 1e-9:
                        line = np.arange(-6.0, 6.0+step/4, step/2)
                        thickness.append(_run(_raw(graph, middle+line[:, None]*normal/np.linalg.norm(normal)), len(line)//2)*step/2)
                last = k
            if open_*step < config["web_minimum_mm"] or not thickness:
                continue
            sheet = np.concatenate([pi[k]+(pj[k]-pi[k])*np.linspace(0, 1, max(int(np.linalg.norm(pj[k]-pi[k])/step), 1)+1)[:, None] for k in range(last+1)])
            webs.append({"members": [i, j], "length_mm": round((last+1)*step, 2), "open_mm": round(open_*step, 2), "thickness_mm": float(max(np.median(thickness), 2*config["minimum_radius_mm"])), "sheet": sheet})
    return webs

def add_webs(builder, webs, blend):
    from scipy.spatial import cKDTree
    core = builder.config["minimum_radius_mm"]
    for web in webs:
        half = web["thickness_mm"]/2+builder.config["web_margin_mm"]
        window = builder.field.window(_box(web["sheet"].min(axis=0), web["sheet"].max(axis=0)), half+blend)
        if window is None:
            continue
        axes = builder.field.axes(window)
        points = np.stack(np.broadcast_arrays(*axes), axis=-1)
        distance = cKDTree(web["sheet"]).query(points.reshape(-1, 3))[0].reshape(points.shape[:3])
        values = half-distance if builder.density is None else np.maximum(np.minimum(builder.raw_values(axes), half-distance), core-distance)
        builder._blend(window, values.astype(np.float32), blend)

def bumps(graph, domain, rods, areas, config):
    loads = [(case["name"], load["region"]) for key in ("load_cases", "comparison_load_cases") for case in domain.get(key, []) for load in case.get("loads", [])]
    preserves = [region for region in domain["regions"] if region["role"] == "preserve"]
    gap = distance_transform_edt(~graph.preserve)*graph.h
    pad, rows = config["bump_pad_mm"], []
    for number, (member, rod, area) in enumerate(zip(graph.members, rods, areas)):
        if rod is None or len(area) < 3:
            continue
        area = area*rod["stretch"]
        measured = gaussian_filter1d(area, config["profile_sigma_samples"], mode="nearest") if config["profile_sigma_samples"] > 0 else area
        fitted = np.sqrt(np.maximum(profile_values(rod["profile"], rod["t"]), 1e-9)/np.pi)
        excess = np.sqrt(measured/np.pi)-fitted
        step = rod["path_length"]/max(len(area), 1)
        for i in np.flatnonzero(excess > config["bump_minimum_mm"]):
            if excess[i] < excess[max(i-2, 0):i+3].max():
                continue
            above = excess > excess[i]/2
            low, high = i, i
            while low > 0 and above[low-1]:
                low -= 1
            while high < len(above)-1 and above[high+1]:
                high += 1
            point = graph.centers(member["voxels"][i])
            extent = (high-low+1)*step
            near = [n for n in member["nodes"] if isinstance(n, int) and np.linalg.norm(point-graph.nodes[n-1]["center"]) < graph.nodes[n-1]["radius"]+pad]
            cases = sorted({name for name, region in loads if region_contains(point[None], region, pad)[0]})
            touching = [region["name"] for region in preserves if region_contains(point[None], region, float(fitted[i])+pad)[0]]
            kind = ("prescribed" if touching or gap[tuple(member["voxels"][i])] <= fitted[i]+pad else "load_point" if cases else "junction" if near or not rod["keep"][i] else "grid_artefact" if extent < rod["length"]/3 else "optimizer_feature")
            rows.append({"member": number, "point_mm": np.round(point, 2).tolist(), "excess_mm": round(float(excess[i]), 3), "fitted_radius_mm": round(float(fitted[i]), 3), "extent_mm": round(float(extent), 2), "member_length_mm": round(rod["length"], 2), "class": kind,
                         "preserve_gap_mm": round(float(gap[tuple(member["voxels"][i])]), 2), "preserves": touching, "load_cases": cases})
    return rows

def reconstruct_splines(domain, density, config, body=None):
    started, times = perf_counter(), {}
    grid = domain["grid"]
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    solid, fragments = solid_body(density, config, float(np.prod(spacing)))
    weights = None if body is None else body_weights(body, grid, config["body_subdivisions"])
    graph = DesignGraph(solid, grid["origin_mm"], spacing[0], domain["preserve"], config, weights)
    areas = graph.areas()
    seating = {**seat_nodes(graph, domain, config), "anchors": snap_anchors(graph, domain, config)}
    rods = [spline_member(graph, member, area, config, domain) if member["kind"] == "rod" else None for member, area in zip(graph.members, areas)]
    rods = [None if rod is not None and member["nodes"][0] == member["nodes"][1] and rod["length"] < config["loop_factor"]*float(np.mean(rod["b"])) else rod for member, rod in zip(graph.members, rods)]
    times["skeleton_s"] = perf_counter()-started
    builder, k = Reconstruction(domain, {**config, **({} if body is not None else {"raw_margin_mm": -1.0, "rod_raw_margin_mm": -1.0})}, (density if body is None else graph.weights) if config["bridge_gap_mm"] > 0 else None), config["transition_radius_mm"]
    for rod in rods:
        if rod is not None:
            builder.add_member(rod, graph.h, k)
    add_shells(builder, graph, domain, config, 1.0, k, True)
    webs = fork_webs(graph, rods, config) if config["web_fill"] <= 1 else []
    add_webs(builder, webs, k)
    items = joints(graph, rods)
    radii = [max([node["radius"]]+[item["b"] for item in items if item["node"] == number+1]) for number, node in enumerate(graph.nodes)]
    for node, radius in zip(graph.nodes, radii):
        builder.add_node(dict(node, radius=radius), k)
    mesh, extraction = builder.finish()
    sections_report = joint_sections(builder.field, items, radii, builder.field.spacing[0])
    times["field_extract_s"] = perf_counter()-started-times["skeleton_s"]
    mesh, final = finalize(mesh, domain, config)
    times["booleans_s"] = perf_counter()-started-sum(times.values())
    built = [rod for rod in rods if rod is not None]
    members = [{"member": number, "length_mm": round(rod["length"], 2), "control_points": len(rod["control"]), "aspect": round(rod["aspect"], 3), "radius_mm": [round(float(np.sqrt(rod["a"]*rod["b"]).min()), 3), round(float(np.sqrt(rod["a"]*rod["b"]).max()), 3)]} for number, rod in enumerate(rods) if rod is not None]
    budget = {"measured_members_mm3": float(sum(m["target_volume"] for m in graph.members)), "built_rods_mm3": float(sum(np.sum(np.pi*rod["a"]*rod["b"])*rod["length"]/len(rod["a"]) for rod in built)),
              "node_bodies_mm3": float(sum(4/3*np.pi*r**3 for r in radii)), "preserves_mm3": float(np.count_nonzero(domain["preserve"])*np.prod(spacing)), "section_source_mm3": float(graph.weights.sum()*np.prod(spacing)), "result_mm3": final["volume_mm3"]}
    report = {**graph.report(), **final, "method": "spline", "fragments_dropped": fragments, "splines": members, "control_points": {str(n): sum(1 for m in members if m["control_points"] == n) for n in (3, 4, 5)}, "mass_budget": budget, "seating": {"nodes": seating, "members_shifted": sum(1 for rod in built if rod["seat_shift_mm"] > 0), "maximum_member_shift_mm": max([rod["seat_shift_mm"] for rod in built], default=0.0)},
              "joint_sections": sections_report, "preserve_bridges": getattr(builder, "bridges", 0), "webs": [{key: web[key] for key in ("members", "length_mm", "open_mm", "thickness_mm")} for web in webs], "bumps": bumps(graph, domain, rods, areas, config), "extraction": {key: extraction[key] for key in ("mesh", "runtime_s")}, "voxel_mm": config["voxel_mm"], "transition_radius_mm": k,
              "section_source": "raw body occupancy" if body is not None else "thresholded density", "runtime_s": {**times, "total_s": perf_counter()-started}}
    return mesh, graph, rods, report

def mount_regions(domain, config):
    regions = [r for r in domain["regions"] if r["role"] == "preserve" and any(key in r["name"] for key in config["load_path_mounts"])]
    hoop = domain.get("metadata", {}).get("round2", {}).get("hoop")
    return regions+[dict(tube(path, hoop["radius_mm"]), name=f"hoop_{side}") for side, path in enumerate(hoop_paths(hoop))]

def _mount_cells(region, lower, h, shape, pad):
    if region["kind"] == "rod":
        low, high = region["points"].min(axis=0)-region["a"].max(), region["points"].max(axis=0)+region["a"].max()
    else:
        low, high = region_bounds(region)
    start = np.maximum(np.floor((np.asarray(low)-pad-lower)/h).astype(int), 0)
    stop = np.minimum(np.ceil((np.asarray(high)+pad-lower)/h).astype(int)+1, shape)
    cells = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(start, stop)], indexing="ij"), axis=-1).reshape(-1, 3)
    points = lower+(cells+0.5)*h
    if region["kind"] == "rod":
        inside = sweep_values([points[:, 0], points[:, 1], points[:, 2]], region["points"], region["a"]+pad, region["b"]+pad, region["axis"]) > 0
    else:
        inside = region_contains(points, region, pad)
    return cells[inside]

def load_paths(mesh, domain, config=DESIGN_RECONSTRUCTION_CONFIG):
    from deep_frame.topology_implicit_validation import occupancy
    h = config["load_path_voxel_mm"]
    solid, lower, _ = occupancy(mesh, h, 2*h)
    depth = distance_transform_edt(solid)*h
    mounts = mount_regions(domain, config)
    result = {"voxel_mm": h, "mounts": len(mounts), "method": "material eroded by r (EDT > r) labelled with 26-connectivity; a mount is carried when the largest core component reaches into the mount region padded by r + h; r = 0.9 mm stands for the 2 mm minimum section"}
    for radius in config["load_path_core_mm"]:
        labels, count = label(depth > radius, TWENTY_SIX)
        main = int(np.argmax(np.bincount(labels.ravel())[1:]))+1 if count else -1
        missing = [region["name"] for region in mounts if not np.any(labels[tuple(_mount_cells(region, lower, h, labels.shape, radius+h).T)] == main)]
        result[f"r{radius:g}"] = {"components": int(count), "carried": len(mounts)-len(missing), "missing": missing}
    result["passed"] = not result[f"r{config['load_path_core_mm'][-1]:g}"]["missing"]
    return result
