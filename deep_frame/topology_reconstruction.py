from time import perf_counter

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt, gaussian_filter, gaussian_filter1d, label, map_coordinates
from skimage.morphology import skeletonize

from deep_frame.config import DESIGN_RECONSTRUCTION_CONFIG, IMPLICIT_CONFIG
from deep_frame.topology_implicit import TWENTY_SIX, ImplicitField, _envelope, exact_booleans, primitive_distance

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
    def __init__(self, solid, origin, spacing, preserve, config=DESIGN_RECONSTRUCTION_CONFIG):
        self.h, self.origin, self.config = float(spacing), np.asarray(origin, dtype=float), config
        self.solid, self.preserve = solid, preserve
        self.radius = distance_transform_edt(solid)*self.h
        skeleton = skeletonize(solid).astype(bool)
        self.spurs = self._prune(skeleton, binary_dilation(preserve, CUBE, 2))
        skeleton &= ~preserve
        self.skeleton = skeleton
        self._split()
        self._sections()

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

    def _sections(self):
        h, cfg = self.h, self.config
        ids = np.full(self.skeleton.shape, -1, dtype=np.int64)
        voxels = np.argwhere(self.skeleton)
        ids[tuple(voxels.T)] = np.arange(len(voxels))
        tangents = np.zeros((len(voxels), 3))
        for member in self.members:
            points = gaussian_filter1d(self.centers(member["voxels"]).astype(float), cfg["path_sigma_samples"], axis=0, mode="nearest") if len(member["voxels"]) > 2 else self.centers(member["voxels"])
            member["points"] = points
            tangent = np.gradient(points, axis=0) if len(points) > 1 else np.array([[1.0, 0.0, 0.0]])
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
            member["tangents"] = tangent
            tangents[ids[tuple(member["voxels"].T)]] = tangent
        _, index = distance_transform_edt(~self.skeleton, return_indices=True)
        cells = np.argwhere(self.solid & ~self.preserve)
        owners = index[(slice(None),)+tuple(cells.T)].T
        owner = ids[tuple(owners.T)]
        offsets = (cells-owners)*h
        along = np.einsum("ij,ij->i", offsets, tangents[owner])
        across = offsets-along[:, None]*tangents[owner]
        moments = np.zeros((len(voxels), 3, 3))
        np.add.at(moments, owner, across[:, :, None]*across[:, None, :])
        counts = np.bincount(owner, minlength=len(voxels)).astype(float)
        self.assigned = {"owner": owner, "cells": cells}
        for node in self.nodes:
            node["volume"] = float(counts[ids[tuple(node["voxels"].T)]].sum()*h**3)
        for number, member in enumerate(self.members):
            rows = ids[tuple(member["voxels"].T)]
            n = gaussian_filter1d(counts[rows], cfg["section_sigma_samples"], mode="nearest")
            m = gaussian_filter1d(moments[rows], cfg["section_sigma_samples"], axis=0, mode="nearest")/np.maximum(n, 1e-9)[:, None, None]
            values, vectors = np.linalg.eigh(m)
            a = 2*np.sqrt(np.maximum(values[:, 2], 0)+h*h/12)
            b = 2*np.sqrt(np.maximum(values[:, 1], 0)+h*h/12)
            b = np.maximum(b, cfg["minimum_radius_mm"])
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

    def _shell(self, mask):
        h, sigma = self.h, self.config["shell_sigma_mm"]
        points = self.centers(self.assigned["cells"][mask])
        center = points.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov((points-center).T))
        frame = vectors[:, ::-1]
        local = (points-center)@frame
        low = local[:, :2].min(axis=0)-3*h
        shape = np.ceil((local[:, :2].max(axis=0)+3*h-low)/h).astype(int)+1
        cells = tuple(np.floor((local[:, :2]-low)/h).astype(int).T)
        count = np.zeros(shape)
        height = np.zeros(shape)
        np.add.at(count, cells, 1.0)
        np.add.at(height, cells, local[:, 2])
        footprint = count > 0
        weight = gaussian_filter(footprint.astype(float), sigma/h)
        surface = gaussian_filter(np.where(footprint, height/np.maximum(count, 1), 0), sigma/h)/np.maximum(weight, 1e-9)
        thickness = gaussian_filter(count*h, sigma/h)/np.maximum(weight, 1e-9)
        outline = gaussian_filter((distance_transform_edt(footprint)-distance_transform_edt(~footprint))*h, sigma/h)
        return {"center": center, "frame": frame, "low": low, "surface": surface, "thickness": thickness, "outline": outline, "bounds": (points.min(axis=0)-2.0, points.max(axis=0)+2.0)}

    def report(self):
        rods = [m for m in self.members if m["kind"] == "rod"]
        return {"nodes": len(self.nodes), "members": len(rods), "shells": len(self.members)-len(rods), "skeleton_voxels": int(self.skeleton.sum()), "graph_components": nx.number_connected_components(self.graph), "spurs_dropped": self.spurs,
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

class Reconstruction:
    def __init__(self, domain, config=DESIGN_RECONSTRUCTION_CONFIG):
        self.domain, self.config = domain, config
        envelope = _envelope(domain)
        h, pad = config["voxel_mm"], 1.0
        low = np.asarray(envelope["min_mm"], dtype=float)-pad+h/2
        shape = tuple(np.ceil((np.asarray(envelope["max_mm"], dtype=float)+pad-low)/h).astype(int)+1)
        self.field = ImplicitField(low, h, np.full(shape, -pad, dtype=np.float32))

    def _blend(self, window, values):
        part = ImplicitField(self.field.origin, self.field.spacing, self.field.values[window])
        part.smooth_union(values, self.config["transition_radius_mm"])
        self.field.values[window] = part.values

    def add_member(self, member, h):
        k = self.config["transition_radius_mm"]
        if member["kind"] == "shell":
            shell = dict(member["shell"], h=h)
            window = self.field.window(_box(*shell["bounds"]), k)
            if window is not None:
                self._blend(window, np.maximum(shell_values(self.field.axes(window), shell), sweep_values(self.field.axes(window), member["points"], member["b"], member["b"], member["axis"])))
            return
        reach = float(np.max(member["a"]))+2*self.field.spacing[0]
        window = self.field.window(_box(member["points"].min(axis=0), member["points"].max(axis=0)), reach+k)
        if window is None:
            return
        values = np.full(tuple(s.stop-s.start for s in window), -np.inf, dtype=np.float32)
        flat = [isinstance(node, int) for node in member["nodes"]]
        for i in range(max(len(member["points"])-1, 1)):
            j = min(i+1, len(member["points"])-1)
            pair = member["points"][[i, j]]
            local = self.field.window(_box(pair.min(axis=0), pair.max(axis=0)), max(member["a"][i], member["a"][j])+2*self.field.spacing[0])
            if local is None:
                continue
            local = tuple(slice(max(l.start, w.start), min(l.stop, w.stop)) for l, w in zip(local, window))
            if any(s.stop <= s.start for s in local):
                continue
            target = tuple(slice(l.start-w.start, l.stop-w.start) for l, w in zip(local, window))
            np.maximum(values[target], sweep_values(self.field.axes(local), member["points"][[i, j]], member["a"][[i, j]], member["b"][[i, j]], member["axis"][[i, j]], (i == 0 and flat[0], j == len(member["points"])-1 and flat[1])), out=values[target])
        self._blend(window, values)

    def add_node(self, node):
        region = {"kind": "sphere", "center_mm": node["center"], "radius_mm": node["radius"]}
        window = self.field.window(region, self.config["transition_radius_mm"])
        if window is not None:
            self._blend(window, primitive_distance(self.field.axes(window), region).astype(np.float32))

    def finish(self):
        k, h = self.config["transition_radius_mm"], self.field.spacing[0]
        preserves = [r for r in self.domain["regions"] if r["role"] == "preserve"]
        forbidden = [r for r in self.domain["regions"] if r["role"] == "forbidden"]
        offset = np.float32(self.config["boolean_offset_mm"])
        if self.config["closing_radius_mm"] > 0:
            gaps = ImplicitField(self.field.origin, self.field.spacing, -self.field.values)
            gaps.open(self.config["closing_radius_mm"])
            self.field.values = -gaps.values
        self.field.smooth_union(self.field.primitives(preserves, k+3*h)-offset, k)
        self.field.intersect(-self.field.primitives(forbidden, offset+3*h)-offset)
        self.field.intersect(self.field.primitives([_envelope(self.domain)])+offset)
        return self.field.extract()

def build_field(graph, domain, config, scale=1.0):
    builder = Reconstruction(domain, config)
    for member in graph.members:
        builder.add_member(dict(member, a=member["a"]*scale, b=member["b"]*scale, **({"shell": dict(member["shell"], thickness=member["shell"]["thickness"]*scale)} if "shell" in member else {})), graph.h)
    for node in graph.nodes:
        builder.add_node(dict(node, radius=node["radius"]*scale))
    return builder.finish()

def reconstruct(domain, density, config=DESIGN_RECONSTRUCTION_CONFIG):
    started, times = perf_counter(), {}
    grid = domain["grid"]
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    solid, fragments = solid_body(density, config, float(np.prod(spacing)))
    graph = DesignGraph(solid, grid["origin_mm"], spacing[0], domain["preserve"], config)
    times["skeleton_s"] = perf_counter()-started
    target = config["target_volume_mm3"] or float(solid.sum()*np.prod(spacing))
    low, high, trials = 0.5, 1.5, []
    for _ in range(config["calibration_steps"]):
        scale = (low+high)/2
        volume = float(exact_booleans(build_field(graph, domain, {**config, "voxel_mm": config["calibration_voxel_mm"]}, scale)[0], domain, IMPLICIT_CONFIG)[0].volume)
        trials.append([scale, volume])
        low, high = (scale, high) if volume < target else (low, scale)
    scale = (low+high)/2 if trials else 1.0
    times["calibration_s"] = perf_counter()-started-times["skeleton_s"]
    mesh, extraction = build_field(graph, domain, config, scale)
    times["field_extract_s"] = perf_counter()-started-times["skeleton_s"]-times["calibration_s"]
    parts = mesh.split(only_watertight=False)
    largest = max(parts, key=lambda part: abs(part.volume))
    debris = {"count": len(parts)-1, "volume_mm3": float(sum(abs(p.volume) for p in parts if p is not largest))}
    mesh, booleans = exact_booleans(largest, domain, IMPLICIT_CONFIG)
    times["booleans_s"] = perf_counter()-started-sum(times.values())
    report = {**graph.report(), "fragments_dropped": fragments, "target_volume_mm3": target, "global_scale": scale, "calibration": trials,"extraction_debris": debris, "solid_volume_mm3": float(solid.sum()*np.prod(spacing)), "volume_mm3": float(mesh.volume),
              "watertight": bool(mesh.is_watertight), "bodies": len(mesh.split(only_watertight=False)), "exact_booleans": {key: booleans[key] for key in ("status", "components", "passed", "preserve_added_mm3", "forbidden_removed_mm3", "envelope_removed_mm3", "maximum_cylinder_oversize_mm")},
              "extraction": {key: extraction[key] for key in ("mesh", "runtime_s")}, "voxel_mm": config["voxel_mm"], "transition_radius_mm": config["transition_radius_mm"], "runtime_s": {**times, "total_s": perf_counter()-started}}
    return mesh, graph, report
