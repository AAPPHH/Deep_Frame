from fractions import Fraction
from io import BytesIO
from functools import reduce
from itertools import combinations, product
from time import perf_counter

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, generate_binary_structure, label, maximum_filter
from skimage.measure import euler_number, marching_cubes

from deep_frame.config import IMPLICIT_CONFIG, IMPLICIT_KINDS, configure
from deep_frame.topology_geometry import _primitive_wall_checks, _trapped_voids, region_bounds, region_contains
from deep_frame.topology_pipeline import _file_digest
from deep_frame.topology_surface import _domain_field, _mesh_summary, _one_mesh, _progress, _statistics, _upsample
from deep_frame.topology_surface_validation import _self_intersection_screen, _settings as _validation_settings, _strict_selected_pair_certificates, _triangle_samples, surface_maturity, surface_metrics

AXES = {"x": 0, "y": 1, "z": 2}
SIX = generate_binary_structure(3, 1)
EXTENSIONS = ("none", "preserve", "preserve_forbidden")
NONNEGATIVE = ("density_sigma_mm", "transition_radius_mm", "preserve_inflation_mm", "constraint_offset_mm", "opening_radius_mm", "ripple_sigma_mm")
PROPAGATION_PASSES = 2
COLUMN_JITTER = 1e-4*np.array([np.sqrt(2), np.sqrt(3)])
GATE_MAXIMA = ("surface_deviation_mm", "relative_volume_change", "segment_tolerance_mm", "penetration_tolerance_mm", "penetration_sample_spacing_mm", "free_zone_preserve_mm", "free_zone_constraint_mm", "free_zone_modified_mm", "mesh_boundary_deviation_mm")
GATE_MINIMA = ("free_zone_minimum_samples", "free_zone_opening_cells", "mesh_minimum_sicn")
BUDGETS = ("maximum_wall_samples", "maximum_void_columns")

class ImplicitError(ValueError):
    def __init__(self, status, message, report, mesh=None):
        super().__init__(message)
        report["status"] = status
        self.status, self.report, self.mesh = status, report, mesh

def implicit_settings(settings):
    config = configure({**IMPLICIT_CONFIG, "threshold": None, "extension": None}, {**IMPLICIT_KINDS, "threshold": "float", "extension": EXTENSIONS}, settings, ("threshold", "extension"))
    for key, value in config.items():
        if isinstance(value, (int, float)) and (not np.isfinite(value) or value < 0 or value == 0 and key not in NONNEGATIVE):
            raise ValueError("Implicit setting must be finite and " + ("nonnegative: " if key in NONNEGATIVE else "positive: ") + key)
    if not 0 < config["threshold"] < 1:
        raise ValueError("Implicit density threshold must lie in (0,1)")
    weakened = [key for key in GATE_MAXIMA if config[key] > IMPLICIT_CONFIG[key]]+[key for key in GATE_MINIMA if config[key] < IMPLICIT_CONFIG[key]]
    if weakened:
        raise ValueError("Frozen implicit acceptance gates cannot be weakened: " + ", ".join(weakened))
    return config

def frozen_gates(settings):
    config = implicit_settings(settings)
    return config, {"passed": True, "values": {key: config[key] for key in GATE_MAXIMA+GATE_MINIMA}, "method": "Build settings re-validated against IMPLICIT_CONFIG; acceptance gates may only be tightened"}

def _bounds(region):
    if region["kind"] == "sphere":
        center = np.asarray(region["center_mm"], dtype=float)
        return center-region["radius_mm"], center+region["radius_mm"]
    return region_bounds(region)

def primitive_distance(axes, region):
    if region["kind"] == "sphere":
        return region["radius_mm"]-np.sqrt(sum((axes[i]-region["center_mm"][i])**2 for i in range(3)))
    low, high = region_bounds(region)
    if region["kind"] == "box":
        q = [np.maximum(low[i]-axes[i], axes[i]-high[i]) for i in range(3)]
    elif region["kind"] == "cylinder":
        axis = AXES[region.get("axis", "z")]
        q = [np.sqrt(sum((axes[i]-region["center_mm"][i])**2 for i in range(3) if i != axis))-region["radius_mm"], np.maximum(low[axis]-axes[axis], axes[axis]-high[axis])]
    else:
        raise ValueError("Unsupported implicit primitive: " + str(region["kind"]))
    return -(np.sqrt(sum(np.maximum(value, 0)**2 for value in q))+np.minimum(reduce(np.maximum, q), 0))

def _slab(axis, start, stop):
    return tuple(slice(start, stop) if i == axis else slice(None) for i in range(3))

class ImplicitField:
    def __init__(self, origin, spacing, values):
        self.origin = np.asarray(origin, dtype=float)
        self.spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.layers = {}

    @classmethod
    def from_function(cls, origin, spacing, shape, function):
        field = cls(origin, spacing, np.empty(shape, dtype=np.float32))
        field.values[...] = function(*field.axes())
        return field

    @classmethod
    def from_density(cls, domain, density, threshold, subdivisions, method):
        sampled, origin, spacing = _upsample(domain, density, subdivisions, method)
        return cls(origin, spacing, sampled-threshold)

    def copy(self):
        return ImplicitField(self.origin, self.spacing, self.values.copy())

    @property
    def cell_volume(self):
        return float(np.prod(self.spacing))

    def volume(self, occupied=None):
        return int(np.count_nonzero(self.values > 0 if occupied is None else occupied))*self.cell_volume

    def axes(self, window=(slice(None),)*3):
        return [(self.origin[i]+np.arange(self.values.shape[i])[window[i]]*self.spacing[i]).reshape(tuple(-1 if j == i else 1 for j in range(3))) for i in range(3)]

    def window(self, region, pad=None):
        if pad is None:
            return (slice(None),)*3
        low, high = _bounds(region)
        start = np.maximum(np.floor((low-pad-self.origin)/self.spacing).astype(int), 0)
        stop = np.minimum(np.ceil((high+pad-self.origin)/self.spacing).astype(int)+1, self.values.shape)
        return None if np.any(stop <= start) else tuple(slice(int(a), int(b)) for a, b in zip(start, stop))

    def primitives(self, regions, pad=None):
        values = np.full(self.values.shape, -np.inf, dtype=np.float32)
        for region in regions:
            window = self.window(region, pad)
            if window is not None:
                np.maximum(values[window], primitive_distance(self.axes(window), region), out=values[window])
        return values

    def _distance(self, window, points):
        total = None
        for i, axis in enumerate(self.axes(window)):
            term = axis.astype(np.float32)-points[i]
            term *= term
            total = term if total is None else np.add(total, term, out=total)
        return np.sqrt(total, out=total)

    def reinitialize(self, band_cells=1.5):
        inside = self.values > 0
        interface = np.zeros(inside.shape, dtype=bool)
        for axis in range(3):
            change = np.diff(inside, axis=axis)
            interface[_slab(axis, None, -1)] |= change
            interface[_slab(axis, 1, None)] |= change
        if not interface.any():
            self.values = np.where(inside, 1, -1).astype(np.float32)*np.float32(np.sum(np.asarray(inside.shape)*self.spacing))
            return self
        gradient = np.gradient(self.values, *self.spacing)
        norm = np.maximum(np.sqrt(sum(value**2 for value in gradient)), np.float32(1e-12))
        limit = band_cells*float(self.spacing.max())
        band = interface | (np.abs(self.values) < limit*norm) & binary_dilation(interface, generate_binary_structure(3, 3), int(np.ceil(band_cells)))
        index = np.nonzero(band)
        norm = norm[index].astype(float)
        local = np.clip(self.values[index]/norm, -limit, limit)
        points = np.column_stack([self.origin[i]+index[i]*self.spacing[i]-local/norm*gradient[i][index] for i in range(3)])
        del gradient
        identifier = np.full(inside.shape, -1, dtype=np.int64)
        identifier[index] = np.arange(len(local))
        identifier = identifier[tuple(distance_transform_edt(~band, sampling=self.spacing, return_distances=False, return_indices=True))]
        closest = [points[:, i].astype(np.float32)[identifier] for i in range(3)]
        del identifier
        best = self._distance((slice(None),)*3, closest)
        for _ in range(PROPAGATION_PASSES):
            for axis, step in product(range(3), (1, -1)):
                target, source = (_slab(axis, 1, None), _slab(axis, None, -1))[::step]
                candidate = [closest[i][source] for i in range(3)]
                distance = self._distance(target, candidate)
                better = distance < best[target]
                for i in range(3):
                    closest[i][target][better] = candidate[i][better]
                best[target][better] = distance[better]
        self.values = np.where(inside, np.maximum(best, np.float32(1e-6)), -best)
        return self

    def smooth(self, sigma):
        if sigma > 0:
            self.values = gaussian_filter(self.values, sigma/self.spacing, mode="nearest")
        return self

    def union(self, values):
        np.maximum(self.values, values, out=self.values)
        return self

    def intersect(self, values):
        np.minimum(self.values, values, out=self.values)
        return self

    def smooth_union(self, values, radius):
        if radius <= 0:
            return self.union(values)
        blend = np.maximum(radius-np.abs(self.values-values), 0)/radius
        self.values = np.maximum(self.values, values)+blend**3*(radius/6)
        return self

    def open(self, radius, band_cells=1.5):
        before = self.values > 0
        eroded = self.values > radius
        report = {"radius_mm": radius, "method": "erosion {phi >= r}; its reinitialized signed distance plus r is the dilation; clipped to phi",
                  "eroded_samples": int(eroded.sum()), "before_volume_mm3": self.volume(before), "components_before": int(label(before, SIX)[1])}
        self.intersect(ImplicitField(self.origin, self.spacing, self.values-np.float32(radius)).reinitialize(band_cells).values+np.float32(radius))
        after = self.values > 0
        report.update(after_volume_mm3=self.volume(after), removed_volume_mm3=self.volume(before & ~after), added_volume_mm3=self.volume(after & ~before), components_after=int(label(after, SIX)[1]))
        return report

    def extract(self):
        started = perf_counter()
        padded = np.pad(self.values, 1, constant_values=-np.float32(self.spacing.max()))
        vertices, faces, _, _ = marching_cubes(padded, 0.0, method="lewiner", allow_degenerate=False, gradient_direction="ascent")
        exact, shift = _edge_vertices(padded, vertices)
        mesh = trimesh.Trimesh(self.origin-self.spacing+exact*self.spacing, faces, process=False)
        return mesh, {"method": "Lewiner marching cubes on the field padded by one negative sample; vertices recomputed in float64 on their grid edges",
                      "float32_vertex_shift_samples": shift, "mesh": _mesh_summary(mesh), "runtime_s": perf_counter()-started}

def _edge_vertices(values, vertices):
    vertices = vertices.astype(float)
    rows = np.arange(len(vertices))
    axis = np.argmax(np.abs(vertices-np.rint(vertices)), axis=1)
    low = np.rint(vertices).astype(np.int64)
    low[rows, axis] = np.floor(vertices[rows, axis])
    high = low.copy()
    high[rows, axis] = np.minimum(high[rows, axis]+1, np.asarray(values.shape)[axis]-1)
    first, second = values[tuple(low.T)].astype(float), values[tuple(high.T)].astype(float)
    exact = low.astype(float)
    exact[rows, axis] += np.divide(first, first-second, out=vertices[rows, axis]-low[rows, axis], where=first != second)
    node = np.abs(vertices-np.rint(vertices)).max(axis=1) <= 1e-6
    exact[node] = np.rint(vertices[node])
    shift = float(np.abs(exact-vertices).max(initial=0))
    if shift > 1e-3:
        raise RuntimeError("Marching cubes vertex is not on its grid edge")
    return exact, shift

def extend_density(domain, density, extension):
    masks = {name: np.asarray(domain[name], dtype=bool) for name in ("allowed", "preserve", "forbidden")}
    free = masks["allowed"] & ~masks["preserve"]
    targets = masks["preserve"] & (extension != "none")
    covered = np.zeros(free.shape, dtype=bool)
    grid = domain["grid"]
    origin, spacing = np.asarray(grid["origin_mm"], dtype=float), np.asarray(grid["spacing_mm"], dtype=float)
    if extension == "preserve_forbidden":
        corners = [np.stack(np.meshgrid(*(origin[a]+(np.arange(free.shape[a])+offset[a])*spacing[a] for a in range(3)), indexing="ij"), axis=-1) for offset in product((0, 1), repeat=3)]
        for region in domain["regions"]:
            if region["role"] == "forbidden" and region.get("rasterize", True):
                covered |= np.all([region_contains(corner, region) for corner in corners], axis=0)
        targets = targets | (masks["forbidden"] & ~covered)
    extended = np.array(density, dtype=float)
    if targets.any():
        nearest = distance_transform_edt(~free, sampling=spacing, return_distances=False, return_indices=True)
        extended[targets] = np.asarray(density, dtype=float)[tuple(index[targets] for index in nearest)]
    return extended, {"extension": extension, "method": "target cells take the density of their nearest free allowed cell; forbidden cells fully inside one exact keep-out stay void",
                      "preserve_cells": int(np.count_nonzero(targets & masks["preserve"])), "partial_forbidden_cells": int(np.count_nonzero(targets & masks["forbidden"])),
                      "covered_forbidden_cells": int(np.count_nonzero(covered & masks["forbidden"]))}

def _composition(values, preserve, forbidden, envelope):
    return ((values >= 0) | (preserve > 0)) & (forbidden < 0) & (envelope > 0)

def _topology(occupied):
    return {"components_6": int(label(occupied, SIX)[1]), "euler_number_6": int(euler_number(occupied, connectivity=1))}

def mount_witness(field, occupied, preserves, forbidden):
    labels, count = label(occupied, SIX)
    rows = {}
    for region in preserves:
        window = field.window(region, 0.0)
        found = np.array([], dtype=int) if window is None else labels[window][(primitive_distance(field.axes(window), region) > 0) & (forbidden[window] < 0)]
        rows[region["name"]] = {"samples": len(found), "components": np.unique(found).tolist()}
    used = sorted(set().union(*(row["components"] for row in rows.values())) - {0})
    passed = all(row["samples"] and 0 not in row["components"] for row in rows.values()) and len(used) <= 1
    return {"passed": passed, "occupied_components_6": count, "mount_components": used, "preserves": rows,
            "method": "6-connected labels of occupied samples; every sample strictly inside an exact preserve and outside all keep-outs must be occupied and all must share one label; nothing is bridged"}

def _connected(field, report, name, occupied, preserves, forbidden):
    witness = mount_witness(field, occupied, preserves, forbidden)
    witness["single_component"] = witness["occupied_components_6"] == 1
    report[name] = witness
    if not witness["passed"]:
        raise ImplicitError("mount_disconnected", "Field separates required mounts at " + name, report)
    if not witness["single_component"]:
        raise ImplicitError("field_disconnected", "Field has detached bodies at " + name + "; they are neither bridged nor removed", report)

def build_field(domain, density, settings, progress=None):
    config = implicit_settings(settings)
    timings = {}
    clock = [perf_counter()]
    def lap(name):
        now = perf_counter()
        timings[name] = timings.get(name, 0.0)+now-clock[0]
        clock[0] = now
    report = {"status": "building_field", "settings": config, "timings_s": timings, "volumes_mm3": {}}
    original, origin, spacing = _domain_field(domain, density)
    extended, report["extension"] = extend_density(domain, original, config["extension"])
    lap("extension")
    reference = ImplicitField.from_density(domain, original, config["threshold"], config["subdivisions"], config["interpolation_method"])
    field = reference.copy() if config["extension"] == "none" else ImplicitField.from_density(domain, extended, config["threshold"], config["subdivisions"], config["interpolation_method"])
    field.layers["extension_changed"] = np.asarray(extended != original)
    lap("upsample")
    report["field"] = {"origin_mm": field.origin.tolist(), "spacing_mm": field.spacing.tolist(), "shape": list(field.values.shape)}
    k, delta, offset = config["transition_radius_mm"], config["preserve_inflation_mm"], config["constraint_offset_mm"]
    pad = max(k, offset)+delta+3*float(field.spacing.max())
    preserves = [region for region in domain["regions"] if region["role"] == "preserve"]
    preserve = field.primitives(preserves, pad)
    forbidden = field.primitives([region for region in domain["regions"] if region["role"] == "forbidden"], pad)
    envelope = field.primitives([{"kind": "box", "min_mm": origin, "max_mm": origin+spacing*np.asarray(domain["grid"]["shape"])}])
    report["primitive_pad_mm"] = pad
    lap("primitives")
    occupied = _composition(reference.values, preserve, forbidden, envelope)
    report["volumes_mm3"]["density_composition"] = field.volume(occupied)
    report["witness"] = mount_witness(field, occupied, preserves, forbidden)
    if report["witness"]["passed"] and config["extension"] != "none":
        guard = {"original": _topology(occupied), "extended": _topology(_composition(field.values, preserve, forbidden, envelope))}
        guard["passed"] = guard["original"] == guard["extended"]
        report["extension_guard"] = guard
    del occupied
    lap("witness")
    if not report["witness"]["passed"]:
        raise ImplicitError("mount_disconnected", "Required mounts are not connected by the density; they are rejected, never bridged", report)
    if not report.get("extension_guard", {"passed": True})["passed"]:
        raise ImplicitError("extension_changed_topology", "Density extension changed components or Euler number of the hard composition", report)
    _progress(progress, "field_witness", report)
    field.reinitialize(config["reinit_band_cells"])
    lap("reinit")
    report["volumes_mm3"]["density"] = field.volume()
    field.smooth(config["density_sigma_mm"]).reinitialize(config["reinit_band_cells"])
    report["volumes_mm3"]["smoothed"] = field.volume()
    lap("smoothing")
    opened = field.copy()
    report["density_opening"] = opened.open(config["opening_radius_mm"], config["reinit_band_cells"]) if config["opening_radius_mm"] else None
    report["density_witness"] = mount_witness(opened, _composition(opened.values, preserve, forbidden, envelope), preserves, forbidden)
    report["density_witness"]["method"] += "; evaluated on the smoothed density opened by the opening radius, before any smooth union, so only connections of structural width count"
    del opened
    lap("density_witness")
    if not report["density_witness"]["passed"]:
        raise ImplicitError("mount_disconnected", "Smoothed and opened density does not connect the required mounts; the smooth union may not create the connection", report)
    preserve += np.float32(delta)
    field.smooth_union(preserve, k).intersect(-forbidden-np.float32(offset)).intersect(envelope-np.float32(offset))
    report["volumes_mm3"]["composed"] = field.volume()
    lap("composition")
    field.reinitialize(config["reinit_band_cells"])
    before = field.values.copy()
    report["opening"] = field.open(config["opening_radius_mm"], config["reinit_band_cells"]) if config["opening_radius_mm"] else None
    field.layers["opening_modified"] = np.abs(field.values-before) > config["free_zone_opening_cells"]*float(field.spacing.max())
    del before
    lap("opening")
    _connected(field, report, "opening_witness", (field.values > 0) | (preserve > 0), preserves, forbidden)
    lap("witness")
    field.smooth(config["ripple_sigma_mm"])
    report["volumes_mm3"]["ripple_smoothed"] = field.volume()
    field.union(preserve)
    report["volumes_mm3"]["final"] = field.volume()
    lap("restore")
    field.layers["reference"] = reference.values
    _connected(field, report, "final_witness", field.values > 0, preserves, forbidden)
    lap("witness")
    report["status"] = "field_built"
    _progress(progress, "field_built", report)
    return field, report

def remesh(mesh, config):
    import pymeshlab
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64), face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    meshes.meshing_isotropic_explicit_remeshing(iterations=config["remesh_iterations"], targetlen=pymeshlab.PureValue(config["remesh_target_mm"]), featuredeg=config["remesh_feature_deg"],
                                                checksurfdist=True, maxsurfdist=pymeshlab.PureValue(config["remesh_max_surface_distance_mm"]))
    meshes.meshing_remove_unreferenced_vertices()
    result = meshes.current_mesh()
    return trimesh.Trimesh(result.vertex_matrix(), result.face_matrix(), process=False)

def _surface_samples(mesh):
    triangles = mesh.triangles
    return np.concatenate((triangles.mean(axis=1), (triangles[:, 0]+triangles[:, 1])/2, (triangles[:, 1]+triangles[:, 2])/2, (triangles[:, 0]+triangles[:, 2])/2, mesh.vertices))

def _point_distances(points, target):
    import pymeshlab
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(points, dtype=np.float64)))
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(target.vertices, dtype=np.float64), face_matrix=np.asarray(target.faces, dtype=np.int32)))
    meshes.compute_scalar_by_distance_from_another_mesh_per_vertex(measuremesh=0, refmesh=1, signeddist=False, maxdist=pymeshlab.PureValue(float(np.linalg.norm(np.ptp(np.vstack((points.min(axis=0), points.max(axis=0), target.bounds)), axis=0)))))
    return meshes.mesh(0).vertex_scalar_array()

def _surface_distances(source, target):
    distances = _point_distances(_surface_samples(source), target)
    return {"sample_count": len(distances), "surface_distance_mm": _statistics(distances)}

def surface_fidelity(reference, approximation):
    started = perf_counter()
    forward, reverse = _surface_distances(reference, approximation), _surface_distances(approximation, reference)
    return {"method": "bidirectional vertices, centroids and all edge midpoints; exact nearest-surface distance by MeshLab per-vertex distance", "reference_to_approximation": forward, "approximation_to_reference": reverse,
            "maximum_sampled_deviation_mm": max(forward["surface_distance_mm"]["max"], reverse["surface_distance_mm"]["max"]),
            "relative_volume_change": float(approximation.volume/reference.volume-1), "runtime_s": perf_counter()-started}

def segments(radius, tolerance):
    if not tolerance > 0:
        raise ValueError("Segment tolerance must exceed the float32 margins")
    count = max(8, int(np.ceil(np.pi/np.arccos(1/(1+tolerance/radius)))))
    while radius/np.cos(np.pi/count)-radius > tolerance:
        count += 1
    return count

def _envelope(domain):
    lower = np.asarray(domain["grid"]["origin_mm"], dtype=float)
    return {"name": "envelope", "role": "envelope", "kind": "box", "min_mm": lower, "max_mm": lower+np.asarray(domain["grid"]["spacing_mm"], dtype=float)*np.asarray(domain["grid"]["shape"])}

def float32_margin(domain):
    envelope = _envelope(domain)
    return 2*float(np.spacing(np.float32(np.abs(np.concatenate((envelope["min_mm"], envelope["max_mm"]))).max())))

def _float32(value, direction):
    rounded = np.float32(value)
    if (float(rounded)-value)*direction < 0:
        rounded = np.nextafter(rounded, np.float32(direction*np.inf))
    return float(rounded)

def realised_bounds(region, side=0):
    low, high = region_bounds(region)
    if side:
        low, high = np.array([_float32(value, -side) for value in low]), np.array([_float32(value, side) for value in high])
    return low, high

def region_manifold(region, tolerance, margin=0.0, grow=0, side=0, inscribed=False):
    from manifold3d import Manifold
    low, high = realised_bounds(region, side)
    if region["kind"] == "box":
        return Manifold.hull_points(np.array(list(product(*zip(low, high))), dtype=float)), None
    if region["kind"] != "cylinder":
        raise ValueError("Exact Booleans support box and cylinder regions only")
    axis = AXES[region.get("axis", "z")]
    radial = [i for i in range(3) if i != axis]
    count = segments(region["radius_mm"], tolerance-2*margin)
    circumscribed = region["radius_mm"]/np.cos(np.pi/count)
    outer = region["radius_mm"] if inscribed else circumscribed+grow*margin
    angles = 2*np.pi*np.arange(count)/count
    points = np.empty((2*count, 3))
    for index, level in enumerate((low[axis], high[axis])):
        rows = slice(index*count, (index+1)*count)
        points[rows, axis] = level
        points[rows, radial[0]] = region["center_mm"][radial[0]]+outer*np.cos(angles)
        points[rows, radial[1]] = region["center_mm"][radial[1]]+outer*np.sin(angles)
    return Manifold.hull_points(points), {"name": region["name"], "role": region["role"], "radius_mm": region["radius_mm"], "segments": count, "circumscribed_radius_mm": circumscribed, "cut_radius_mm": circumscribed+margin,
                                          "reference_radius_mm": circumscribed+2*margin, "float32_margin_mm": margin, "oversize_mm": circumscribed+2*margin-region["radius_mm"]}

def _manifold(mesh):
    from manifold3d import Manifold, Mesh64
    return Manifold(Mesh64(vert_properties=np.array(mesh.vertices, dtype=np.float64, order="C"), tri_verts=np.array(mesh.faces, dtype=np.uint64, order="C")))

def _snap_planes(mesh, regions, tolerance=1e-9):
    planes = [set() for _ in range(3)]
    for region, side in regions:
        low, high = realised_bounds(region, side)
        for axis in range(3):
            if region["kind"] == "box" or axis == AXES[region.get("axis", "z")]:
                planes[axis] |= {float(low[axis]), float(high[axis])}
    vertices = np.array(mesh.vertices)
    moved, largest = 0, 0.0
    for axis, values in enumerate(planes):
        values = np.array(sorted(values))
        index = np.clip(np.searchsorted(values, vertices[:, axis]), 1, len(values)-1)
        nearest = np.where(np.abs(vertices[:, axis]-values[index-1]) < np.abs(vertices[:, axis]-values[index]), values[index-1], values[index])
        snap = (np.abs(vertices[:, axis]-nearest) <= tolerance) & (vertices[:, axis] != nearest)
        moved, largest = moved+int(snap.sum()), max(largest, float(np.abs(vertices[snap, axis]-nearest[snap]).max(initial=0)))
        vertices[snap, axis] = nearest[snap]
    mesh.vertices = vertices
    return {"method": "vertex coordinates within the tolerance of an exact axis-aligned constraint plane are set onto it; Boolean round-off only", "tolerance_mm": tolerance, "moved_coordinates": moved, "maximum_displacement_mm": largest}

def _round_float32(mesh, margin):
    rounded = np.asarray(mesh.vertices, dtype=np.float32).astype(float)
    shift = float(np.abs(rounded-mesh.vertices).max(initial=0))
    merged = len(rounded)-len(np.unique(rounded, axis=0))
    mesh.vertices = rounded
    return {"method": "all vertices rounded to binary32 so the validated mesh equals the delivered STL; constraint planes are binary32 values rounded to the safe side and cut cylinders lie one margin outside the checked polygon, so rounding cannot enter a keep-out",
            "margin_mm": margin, "maximum_rounding_mm": shift, "merged_vertices": merged, "passed": merged == 0 and shift*np.sqrt(3) < margin}

def _trimesh(body):
    mesh = body.to_mesh64()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts, dtype=np.int64), process=False)

def exact_booleans(mesh, domain, config):
    from manifold3d import Error, Manifold, OpType
    started = perf_counter()
    body = _manifold(mesh)
    margin = float32_margin(domain)
    report = {"method": "manifold3d float64: (field mesh + union(preserves) - union(forbidden)) ^ envelope; box planes rounded to binary32 outward (preserve, forbidden) or inward (envelope), cylinders circumscribed polygons grown by the binary32 margin",
              "input_status": str(body.status()), "input_volume_mm3": body.volume(), "cylinders": []}
    operands = {}
    for role in ("preserve", "forbidden"):
        operands[role] = []
        for region in domain["regions"]:
            if region["role"] == role:
                solid, row = region_manifold(region, config["segment_tolerance_mm"], margin, 1, 1)
                operands[role].append(solid)
                if row is not None:
                    report["cylinders"].append(row)
    operands["envelope"] = [region_manifold(_envelope(domain), config["segment_tolerance_mm"], side=-1)[0]]
    for name, role, operation in (("preserve_added_mm3", "preserve", OpType.Add), ("forbidden_removed_mm3", "forbidden", OpType.Subtract), ("envelope_removed_mm3", "envelope", OpType.Intersect)):
        before = body.volume()
        if operands[role]:
            body = Manifold.batch_boolean([body, Manifold.batch_boolean(operands[role], OpType.Add)], operation)
        report[name] = abs(body.volume()-before)
        if body.status() != Error.NoError:
            break
    report.update(status=str(body.status()), components=len(body.decompose()), genus=int(body.genus()), volume_mm3=body.volume(),
                  maximum_cylinder_oversize_mm=max((row["oversize_mm"] for row in report["cylinders"]), default=0.0), runtime_s=perf_counter()-started)
    mesh = _trimesh(body)
    report["plane_snap"] = _snap_planes(mesh, [(region, 1) for region in domain["regions"] if region["role"] in ("preserve", "forbidden")]+[(_envelope(domain), -1)])
    report["float32"] = _round_float32(mesh, margin)
    report["passed"] = body.status() == Error.NoError and report["components"] == 1 and report["maximum_cylinder_oversize_mm"] <= config["segment_tolerance_mm"] and report["float32"]["passed"]
    return mesh, report

def _vector(first, second):
    return [a-b for a, b in zip(first, second)]

def _cross(first, second):
    return [first[1]*second[2]-first[2]*second[1], first[2]*second[0]-first[0]*second[2], first[0]*second[1]-first[1]*second[0]]

def _dot(first, second):
    return sum(a*b for a, b in zip(first, second))

def _origin_in_hull(points):
    edges = [_vector(point, points[0]) for point in points[1:]]
    volume = _dot(_cross(edges[0], edges[1]), edges[2])
    if volume:
        signs = []
        for index in range(4):
            replaced = [[0, 0, 0] if i == index else point for i, point in enumerate(points)]
            edges = [_vector(point, replaced[0]) for point in replaced[1:]]
            signs.append(_dot(_cross(edges[0], edges[1]), edges[2])*volume)
        return all(sign >= 0 for sign in signs)
    normal = next((normal for normal in (_cross(edges[i], edges[j]) for i, j in combinations(range(3), 2)) if any(normal)), None)
    if normal is None:
        direction = next((edge for edge in edges if any(edge)), None)
        if direction is not None and any(_cross(points[0], direction)):
            return False
        raise ValueError("Degenerate adjacent triangle cones")
    if _dot(normal, points[0]):
        return False
    keep = [axis for axis in range(3) if axis != max(range(3), key=lambda axis: abs(normal[axis]))]
    planar = [[point[axis] for axis in keep] for point in points]
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    for triple in combinations(planar, 3):
        area = orient(*triple)
        if area and all(orient(triple[i], triple[(i+1) % 3], [0, 0])*area >= 0 for i in range(3)):
            return True
    return False

def _adjacent_pair(first, second, shared):
    if len(shared) == 1:
        apex = first[shared[0][0]]
        cones = [_vector(point, apex) for i, point in enumerate(first) if i != shared[0][0]]+[_vector(apex, point) for i, point in enumerate(second) if i != shared[0][1]]
        return not _origin_in_hull(cones)
    if len(shared) == 2:
        start, end = first[shared[0][0]], first[shared[1][0]]
        other = [first[next(i for i in range(3) if i not in (shared[0][0], shared[1][0]))], second[next(i for i in range(3) if i not in (shared[0][1], shared[1][1]))]]
        axis = _vector(end, start)
        normal = _cross(axis, _vector(other[0], start))
        if _dot(normal, _vector(other[1], start)):
            return True
        return _dot(normal, _cross(axis, _vector(other[1], start))) < 0
    return False

def adjacent_refinement(mesh, intersections):
    report = {"method": "MeshLab-selected face pairs: index-sharing pairs need exact proof that they meet only in the shared vertex or edge (tangent-cone hull and coplanar fold predicates in binary64 rationals); other pairs need the existing strict exact separation certificate",
              "passed": False, "complete": False, "pairs": 0, "adjacent_certified": 0, "separated_certified": 0, "unresolved_pairs": []}
    if not intersections.get("complete") or not intersections.get("exact_refinement_selection_coverage") or intersections["self_intersecting_face_count"] > 100 or not (intersections["native_mesh_unchanged"] and intersections["input_mesh_unchanged"]):
        report["reason"] = "MeshLab selection is incomplete, unpinned or larger than the recorded face list"
        return report
    selected = intersections["self_intersecting_faces_first_100"]
    try:
        exact = {face: [[Fraction.from_float(float(value)) for value in point] for point in mesh.triangles[face]] for face in selected}
        for first, second in combinations(selected, 2):
            report["pairs"] += 1
            shared = [(i, j) for i, a in enumerate(mesh.faces[first]) for j, b in enumerate(mesh.faces[second]) if a == b]
            if shared:
                certified = _adjacent_pair(exact[first], exact[second], shared)
                report["adjacent_certified"] += certified
            else:
                certified = _strict_selected_pair_certificates(mesh.triangles[[first, second]], [0, 1])["passed"]
                report["separated_certified"] += certified
            if not certified:
                report["unresolved_pairs"].append([int(first), int(second)])
        report["complete"] = True
        report["passed"] = not report["unresolved_pairs"]
    except Exception as error:
        report["reason"] = type(error).__name__+": "+str(error)
    return report

def vertex_manifold(mesh):
    faces = np.asarray(mesh.faces)
    links = np.concatenate([faces[:, [i, (i+1) % 3, (i+2) % 3]] for i in range(3)])
    nodes, inverse = np.unique(np.concatenate((links[:, [0, 1]], links[:, [0, 2]])), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    graph = coo_matrix((np.ones(len(links)), (inverse[:len(links)], inverse[len(links):])), shape=(len(nodes), len(nodes)))
    count, labels = connected_components(graph, directed=False)
    fans = np.bincount(nodes[np.unique(labels, return_index=True)[1], 0], minlength=len(mesh.vertices))
    referenced = np.zeros(len(mesh.vertices), dtype=bool)
    referenced[faces.ravel()] = True
    return {"pinched_vertices": int(np.count_nonzero(fans[referenced] != 1)), "unreferenced_vertices": int(np.count_nonzero(~referenced)), "passed": bool(np.all(fans[referenced] == 1) and referenced.all())}

def mesh_checks(mesh):
    from manifold3d import Error
    started = perf_counter()
    topology = {**_mesh_summary(mesh), "single_closed_oriented_body": _one_mesh(mesh), "manifold3d_status": str(_manifold(mesh).status()), "vertex_manifold": vertex_manifold(mesh)}
    topology["passed"] = topology["single_closed_oriented_body"] and topology["manifold3d_status"] == str(Error.NoError) and topology["vertex_manifold"]["passed"]
    intersections = _self_intersection_screen(mesh)
    if intersections["complete"] and not intersections["passed"]:
        intersections["adjacent_refinement"] = adjacent_refinement(mesh, intersections)
        intersections["passed"] = intersections["adjacent_refinement"]["passed"]
    return {"topology": topology, "self_intersections": intersections, "passed": topology["passed"] and intersections["passed"], "runtime_s": perf_counter()-started}

def export_mesh(mesh, directory):
    vertices = np.zeros(len(mesh.vertices), dtype=[("vertex", "<f8", (3,))])
    vertices["vertex"] = mesh.vertices
    faces = np.zeros(len(mesh.faces), dtype=[("count", "u1"), ("index", "<i4", (3,))])
    faces["count"], faces["index"] = 3, mesh.faces
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}", *(f"property double {axis}" for axis in "xyz"), f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header", ""]
    (directory/"geometry.ply").write_bytes("\n".join(header).encode("ascii")+vertices.tobytes()+faces.tobytes())
    mesh.export(directory/"geometry.stl")
    return {name: {"sha256": _file_digest(directory/name), "size_bytes": (directory/name).stat().st_size} for name in ("geometry.ply", "geometry.stl")}

def build_implicit(domain, density, settings, progress=None):
    field, report = build_field(domain, density, settings, progress)
    config, timings = report["settings"], report["timings_s"]
    def fail(stage, message, mesh):
        report["failure_stage"] = stage
        raise ImplicitError("geometry_invalid", message, report, mesh)
    report["status"] = "extracting"
    started = perf_counter()
    raw, report["extraction"] = field.extract()
    timings["extraction"] = perf_counter()-started
    _progress(progress, "extraction", report, mesh=raw)
    if not _one_mesh(raw):
        fail("extraction", "Marching cubes surface is not one closed oriented body", raw)
    started = perf_counter()
    remeshed = remesh(raw, config)
    timings["remesh"] = perf_counter()-started
    fidelity = surface_fidelity(raw, remeshed)
    fidelity["passed"] = fidelity["maximum_sampled_deviation_mm"] <= config["surface_deviation_mm"] and abs(fidelity["relative_volume_change"]) <= config["relative_volume_change"]
    report["remesh"] = {"checks": mesh_checks(remeshed), "fidelity": fidelity}
    timings["remesh_checks"] = perf_counter()-started-timings["remesh"]
    _progress(progress, "remesh", report, mesh=remeshed)
    if not report["remesh"]["checks"]["passed"] or not fidelity["passed"]:
        fail("remesh", "Remeshed surface failed closure, self-intersection or fidelity gates", remeshed)
    report["status"] = "booleans"
    started = perf_counter()
    final, report["exact_booleans"] = exact_booleans(remeshed, domain, config)
    timings["booleans"] = perf_counter()-started
    if not report["exact_booleans"]["passed"]:
        fail("booleans", "Exact Booleans did not yield one valid manifold", final)
    started = perf_counter()
    report["final_mesh"] = mesh_checks(final)
    timings["final_checks"] = perf_counter()-started
    _progress(progress, "booleans", report, mesh=final)
    if not report["final_mesh"]["passed"]:
        fail("final_mesh", "Final mesh failed closure, orientation or self-intersection checks", final)
    report["status"] = "geometry_built"
    return final, report, field

def segment_hits(mesh, origins, directions, lower, upper, batch=10000):
    triangles, tree, found = mesh.triangles, mesh.triangles_tree, []
    for start in range(0, len(origins), batch):
        origin, direction = origins[start:start+batch], directions[start:start+batch]
        ends = (origin+lower*direction, origin+upper*direction)
        faces, counts = tree.intersection_v(np.minimum(*ends)-1e-6, np.maximum(*ends)+1e-6)
        rays, faces = np.repeat(np.arange(len(origin)), counts.astype(np.int64)), faces.astype(np.int64)
        corner = triangles[faces, 0]
        first, second, ray = triangles[faces, 1]-corner, triangles[faces, 2]-corner, direction[rays]
        across = np.cross(ray, second)
        determinant = np.einsum("ij,ij->i", first, across)
        offset = origin[rays]-corner
        normal = np.cross(offset, first)
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v, t = (np.einsum("ij,ij->i", a, b)/determinant for a, b in ((offset, across), (ray, normal), (second, normal)))
            hit = (np.abs(determinant) > 1e-14*np.linalg.norm(first, axis=1)*np.linalg.norm(second, axis=1)) & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (t >= lower) & (t <= upper)
        found.append((rays[hit]+start, faces[hit], t[hit]))
    return [np.concatenate(values) for values in zip(*found)] if found else [np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)]

def corner_normals(mesh, crease_deg):
    faces, normals, angles = mesh.faces, mesh.face_normals, mesh.face_angles
    star, limit, result = mesh.vertex_faces, np.cos(np.radians(crease_deg)), np.empty((len(faces), 3, 3))
    batch = max(1, 1000000//star.shape[1])
    for start in range(0, len(faces), batch):
        corner = faces[start:start+batch]
        incident = star[corner]
        valid = incident >= 0
        incident = np.where(valid, incident, 0)
        weight = (angles[incident]*(faces[incident] == corner[..., None, None])).sum(axis=-1)
        weight *= valid & (np.einsum("fcdk,fk->fcd", normals[incident], normals[start:start+batch]) >= limit)
        total = np.einsum("fcd,fcdk->fck", weight, normals[incident])
        length = np.linalg.norm(total, axis=-1, keepdims=True)
        result[start:start+batch] = np.where(length > 1e-12, total/np.maximum(length, 1e-300), normals[start:start+batch, None])
    return result

def wall_screen(mesh, minimum, settings, crease_deg):
    started = perf_counter()
    spacing, tolerance, guard = settings["wall_sample_spacing_mm"], settings["wall_tolerance_mm"], 1e-7
    triangles = mesh.triangles
    small = np.linalg.norm(triangles-np.roll(triangles, -1, axis=1), axis=2).max(axis=1) <= spacing
    points, sources = [triangles[small].mean(axis=1)], [np.flatnonzero(small)]
    count = len(sources[0])
    for face in np.flatnonzero(~small):
        samples = list(_triangle_samples(triangles[face], spacing))
        points.append(np.asarray(samples))
        sources.append(np.full(len(samples), face))
        count += len(samples)
        if count > settings["maximum_wall_samples"]:
            break
    if count > settings["maximum_wall_samples"]:
        return {"passed": False, "complete": False, "reason": "wall sample budget exceeded", "maximum_wall_samples": settings["maximum_wall_samples"]}
    points, sources = np.concatenate(points), np.concatenate(sources)
    weights = trimesh.triangles.points_to_barycentric(triangles[sources], points)
    directions = np.einsum("sc,sck->sk", weights, corner_normals(mesh, crease_deg)[sources])
    length = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.where(length > 1e-12, directions/np.maximum(length, 1e-300), mesh.face_normals[sources])
    rays, faces, distances = segment_hits(mesh, points, -directions, -guard, minimum+tolerance)
    other = faces != sources[rays]
    beyond = other & (distances > guard)
    thickness = np.full(len(points), np.inf)
    np.minimum.at(thickness, rays[beyond], distances[beyond])
    thin = thickness < minimum-tolerance
    unresolved = np.zeros(len(points), dtype=bool)
    unresolved[rays[other & (np.abs(distances) <= guard)]] = True
    unresolved &= ~thin
    measured = np.isfinite(thickness)
    lowest = int(np.argmin(thickness)) if measured.any() else None
    return {"passed": bool(len(points) and not thin.any() and not unresolved.any()), "complete": True, "minimum_required_mm": minimum, "query_upper_bound_mm": minimum+tolerance,
            "minimum_measured_mm": float(thickness[lowest]) if lowest is not None else None,
            "minimum_sample": {"face": int(sources[lowest]), "position_mm": points[lowest].tolist(), "thickness_mm": float(thickness[lowest])} if lowest is not None else None,
            "ray_count": len(points), "rays_clear_through_upper_bound": int(np.count_nonzero(~measured)), "thin_sample_count": int(thin.sum()), "unresolved_sample_count": int(unresolved.sum()),
            "thin_samples": [{"face": int(sources[i]), "position_mm": points[i].tolist(), "thickness_mm": float(thickness[i])} for i in np.flatnonzero(thin)[:30]],
            "unresolved_samples": [{"face": int(sources[i]), "position_mm": points[i].tolist()} for i in np.flatnonzero(unresolved)[:30]],
            "ray_origin_exclusion_mm": guard, "maximum_sample_triangle_edge_mm": spacing, "normal_crease_deg": crease_deg, "elapsed_s": perf_counter()-started,
            "method": "Final mesh triangles subdivided until every sample triangle edge meets the spacing limit; float64 Moller-Trumbore along the inward surface normal (barycentric interpolation of angle-weighted corner normals over incident faces within the remeshing feature angle, so sharp creases keep the facet normal) over (1e-7, minimum + tolerance] against rtree-culled triangles; another triangle hit within +/-1e-7 of an otherwise passing origin is unresolved",
            "limitations": "Finite normal-chord screen, not a global minimum-thickness proof; sharp convex wedges can be conservatively rejected"}

def _lattice(triangles, spacing):
    counts = np.maximum(np.ceil(np.linalg.norm(triangles-np.roll(triangles, -1, axis=1), axis=2).max(axis=1)/spacing).astype(int), 1)
    points = []
    for count in np.unique(counts):
        i, j = (value.ravel() for value in np.meshgrid(np.arange(count+1), np.arange(count+1)))
        keep = i+j <= count
        weights = np.column_stack((count-i[keep]-j[keep], i[keep], j[keep]))/count
        points.append(np.einsum("kb,tbd->tkd", weights, triangles[counts == count]).reshape(-1, 3))
    return np.concatenate(points)

def _penetration(mesh, bounds, region, spacing):
    low, high = _bounds(region)
    near = np.all((bounds[0] <= high+1e-6) & (bounds[1] >= low-1e-6), axis=1)
    if not near.any():
        return None
    return max(float(primitive_distance(_lattice(mesh.triangles[near], spacing).T, region).max()), 0.0)

def _dilate(mask, spacing, radius):
    result = np.zeros(mask.shape, dtype=bool)
    if mask.any():
        pad = np.ceil(radius/spacing).astype(int)+1
        window = []
        for axis in range(3):
            present = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
            window.append(slice(max(int(present[0])-pad[axis], 0), int(present[-1])+pad[axis]+1))
        window = tuple(window)
        result[window] = distance_transform_edt(~mask[window], sampling=spacing) <= radius
    return result

def stl_roundtrip(mesh):
    loaded = trimesh.load_mesh(BytesIO(trimesh.exchange.stl.export_stl(mesh)), file_type="stl")
    report = {"checks": mesh_checks(loaded), "vertices_equal": len(loaded.vertices) == len(mesh.vertices), "faces_equal": len(loaded.faces) == len(mesh.faces),
              "maximum_float32_displacement_mm": float(np.abs(np.asarray(mesh.vertices, dtype=np.float32).astype(float)-mesh.vertices).max(initial=0)),
              "method": "Binary STL as written by export_mesh, reloaded with vertex merging; closed, oriented, vertex-manifold, manifold3d and self-intersection checks repeated on its float32 coordinates"}
    report["passed"] = report["checks"]["passed"] and report["vertices_equal"] and report["faces_equal"]
    return loaded, report

def _union(solids):
    from manifold3d import Manifold, OpType
    return Manifold.batch_boolean(solids, OpType.Add) if solids else None

def _cube(low, high):
    from manifold3d import Manifold
    return Manifold.cube((np.asarray(high)-np.asarray(low)).tolist()).translate(np.asarray(low).tolist())

def _column_crossings(mesh, columns, bottom, length):
    origins = np.column_stack((columns, np.full(len(columns), bottom)))
    rays, _, distances = segment_hits(mesh, origins, np.tile([0.0, 0.0, 1.0], (len(origins), 1)), 0.0, length)
    order = np.lexsort((distances, rays))
    rays, heights = rays[order], bottom+distances[order]
    keep = np.ones(len(rays), dtype=bool)
    keep[1:] = (rays[1:] != rays[:-1]) | (np.diff(heights) > 1e-6)
    return rays[keep], heights[keep], np.bincount(rays[keep], minlength=len(columns)) % 2 == 1

class MeshAcceptance:
    def __init__(self, mesh, domain, config, settings):
        self.mesh, self.domain, self.config, self.settings = mesh, domain, config, settings
        self.tolerance = settings["volume_tolerance_mm3"]
        manufacturing = domain["manufacturing"]
        self.minimum = max(manufacturing["nozzle_width_mm"]*manufacturing["minimum_wall_nozzles"], manufacturing["minimum_feature_mm"])
        self.regions = {role: [region for region in domain["regions"] if region["role"] == role] for role in ("allowed", "preserve", "forbidden")}
        self.margin = float32_margin(domain)
        self.stl, self.stl_report = stl_roundtrip(mesh)
        self.bodies = {"float64": _manifold(mesh), "stl": _manifold(self.stl)}
        self.body, self.cavities, self.metrics = self.bodies["float64"], None, None

    def volumes(self, operation):
        values = {name: operation(body).volume() for name, body in self.bodies.items()}
        return max(values.values()), values

    def topology(self):
        checks = mesh_checks(self.mesh)
        self.intersections = checks["self_intersections"]
        bodies = self.mesh.split(only_watertight=False) if self.mesh.body_count > 1 else [self.mesh]
        self.cavities = int(sum(body.volume < 0 for body in bodies))
        topology = {**checks["topology"], "signed_volume_mm3": float(self.mesh.volume), "negative_volume_shells": self.cavities, "stl_roundtrip": self.stl_report}
        topology["passed"] = checks["topology"]["passed"] and topology["signed_volume_mm3"] > self.tolerance and not self.cavities and self.stl_report["passed"]
        return topology

    def envelope(self):
        tolerance = self.config["segment_tolerance_mm"]
        envelope = _envelope(self.domain)
        box = region_manifold(envelope, tolerance)[0]
        outside, by_body = self.volumes(lambda body: body-box)
        allowed = _union([region_manifold(region, tolerance, self.margin, inscribed=True)[0] for region in self.regions["allowed"]])
        outside_allowed, allowed_by_body = self.volumes(lambda body: body-allowed) if allowed is not None else (outside, by_body)
        stl_outside = max(float(-primitive_distance(self.stl.vertices.T, envelope).min()), 0.0)
        return {"outside_grid_box_mm3": outside, "outside_allowed_regions_mm3": outside_allowed, "outside_grid_box_by_body_mm3": by_body, "outside_allowed_regions_by_body_mm3": allowed_by_body, "stl_maximum_vertex_outside_mm": stl_outside,
                "passed": max(outside, outside_allowed) <= self.tolerance and stl_outside <= self.config["penetration_tolerance_mm"],
                "method": "manifold3d differences of the float64 mesh and of the reloaded binary STL against the exact envelope box and the inscribed allowed polyhedra, the larger gated; binary STL vertices against the exact box"}

    def forbidden(self):
        rows, spacing = {}, self.config["penetration_sample_spacing_mm"]
        bounds = [(item, (item.triangles.min(axis=1), item.triangles.max(axis=1))) for item in (self.mesh, self.stl)]
        for region in self.regions["forbidden"]:
            solid, polygon = region_manifold(region, self.config["segment_tolerance_mm"], self.margin)
            volume, by_body = self.volumes(lambda body: body ^ solid)
            penetration, stl_penetration = (_penetration(item, box, region, spacing) for item, box in bounds)
            rows[region["name"]] = {"intersection_mm3": volume, "intersection_by_body_mm3": by_body, "circumscribed": polygon, "maximum_penetration_mm": penetration, "stl_maximum_penetration_mm": stl_penetration,
                                    "passed": volume <= self.tolerance and all(value is None or value <= self.config["penetration_tolerance_mm"] for value in (penetration, stl_penetration))}
        return rows

    def preserve(self):
        rows = {}
        settings, tolerance = self.settings, self.config["segment_tolerance_mm"]
        depth, offsets = settings["attachment_probe_depth_mm"], settings["attachment_offsets_mm"]
        forbidden = _union([region_manifold(region, tolerance, self.margin, 2, 1)[0] for region in self.regions["forbidden"]])
        for region in self.regions["preserve"]:
            solid = region_manifold(region, tolerance, self.margin)[0]
            required = solid-forbidden if forbidden is not None else solid
            low, high = region_bounds(region)
            pad = max(offsets)+depth+tolerance+1e-6
            window = _cube(low-pad, high+pad)
            local = self.bodies["stl"] ^ window
            levels = []
            for offset in offsets:
                sections = {}
                for axis, side in product(range(3), (-1, 1)):
                    start, stop = low.copy(), high.copy()
                    start[axis] = low[axis]-offset-depth if side < 0 else high[axis]+offset
                    stop[axis] = start[axis]+depth
                    sections["xyz"[axis]+("-" if side < 0 else "+")] = (local ^ _cube(start, stop)).volume()/depth
                levels.append({"offset_mm": offset, "sections_mm2": sections, "summed_area_mm2": sum(sections.values())})
            area = region.get("attachment_area_min_mm2", self.domain["manufacturing"]["minimum_attachment_area_mm2"])
            attachment = {"passed": all(level["summed_area_mm2"]+1e-6 >= area for level in levels), "minimum_required_area_mm2": area, "levels": levels,
                          "binding_offset_mm": min(levels, key=lambda level: level["summed_area_mm2"])["offset_mm"],
                          "method": "Summed mean areas of manifold3d intersections of the reloaded binary STL with six slabs outside the exact preserve AABB at all prescribed offsets"}
            walls = _primitive_wall_checks(region, self.regions["forbidden"], max(self.minimum, region.get("minimum_wall_mm", self.minimum)))
            missing, by_body = self.volumes(lambda body: required-(body ^ window))
            rows[region["name"]] = {"required_volume_mm3": required.volume(), "missing_volume_mm3": missing, "missing_by_body_mm3": by_body, "walls": walls, "attachment": attachment,
                                    "required": "circumscribed preserve minus the forbidden polygons grown by two binary32 margins (the cut holes plus rounding); missing volume gated on the float64 mesh and the reloaded binary STL"}
            rows[region["name"]]["passed"] = rows[region["name"]]["required_volume_mm3"] > self.tolerance and rows[region["name"]]["missing_volume_mm3"] <= self.tolerance and walls["passed"] and attachment["passed"]
        return rows

    def features(self, field, report):
        config, h = self.config, float(field.spacing.max())
        required = self.minimum/2+h*np.sqrt(3)/2+config["ripple_sigma_mm"]**2/self.minimum+config["remesh_max_surface_distance_mm"]
        opening = {"opening_radius_mm": config["opening_radius_mm"], "required_radius_mm": required, "applied": report.get("opening") is not None,
                   "method": "Opening radius covers half the minimum wall plus the EDT error h*sqrt(3)/2, the ripple-smoothing shrink sigma^2/minimum and the remeshing surface distance"}
        opening["passed"] = bool(opening["applied"] and config["opening_radius_mm"] >= required)
        rays = wall_screen(self.mesh, self.minimum, self.settings, config["remesh_feature_deg"])
        return {"minimum_wall_mm": self.minimum, "field_opening": opening, "mesh_wall_screen": rays, "passed": opening["passed"] and rays["passed"]}

    def supports(self):
        started = perf_counter()
        settings, mesh, manufacturing = self.settings, self.mesh, self.domain["manufacturing"]
        envelope = _envelope(self.domain)
        lower, upper, spacing = envelope["min_mm"], envelope["max_mm"], settings["void_grid_spacing_mm"]
        shape = np.ceil((upper-lower)/spacing).astype(int)
        needed = bool(np.any((mesh.face_normals[:, 2] < -0.1) & (mesh.triangles_center[:, 2] > mesh.bounds[0, 2]+1e-6)))
        result = {"supports_required": needed, "build_direction": manufacturing["build_direction"], "supports_allowed": bool(manufacturing["supports_allowed"])}
        if shape[0]*shape[1] > settings["maximum_void_columns"]:
            return {**result, "passed": False, "accessibility": {"passed": False, "reason": "void column budget exceeded"}}
        columns = np.stack(np.meshgrid(*(lower[axis]+(np.arange(shape[axis])+0.5)*spacing for axis in range(2)), indexing="ij"), axis=-1).reshape(-1, 2)
        bottom, length = lower[2]-spacing, upper[2]-lower[2]+2*spacing
        rays, heights, odd = _column_crossings(mesh, columns, bottom, length)
        ambiguous = np.flatnonzero(odd)
        if len(ambiguous):
            again, again_heights, still = _column_crossings(mesh, columns[ambiguous]+COLUMN_JITTER*spacing, bottom, length)
            keep = ~odd[rays]
            rays, heights = np.concatenate((rays[keep], ambiguous[again])), np.concatenate((heights[keep], again_heights))
            odd[ambiguous] = still
        toggles = np.zeros((len(columns), shape[2]+1), dtype=np.int64)
        np.add.at(toggles, (rays, np.searchsorted(lower[2]+(np.arange(shape[2])+0.5)*spacing, heights, side="right")), 1)
        occupied = np.cumsum(toggles, axis=1)[:, :shape[2]] % 2 == 1
        occupied[odd] = False
        trapped = _trapped_voids(occupied.reshape(tuple(shape)))
        accessibility = {"passed": not trapped and self.cavities == 0 and not odd.any(), "negative_volume_shells": self.cavities, "trapped_void_cells": trapped, "unresolved_columns": int(odd.sum()), "jittered_columns": len(ambiguous),
                         "grid_spacing_mm": spacing, "grid_shape": shape.tolist(), "occupied_cells": int(occupied.sum()), "elapsed_s": perf_counter()-started,
                         "method": "Vertical float64 segment-triangle crossings per column centre, deduplicated within 1e-6 mm, odd columns (rays grazing an edge or lying in a constraint plane) re-shot once at the fixed offset 1e-4*(sqrt 2, sqrt 3) cells and unresolved if still odd; six-connected exterior void flood fill of the cell-centre occupancy plus the negative-volume shell count"}
        return {**result, "accessibility": accessibility, "passed": manufacturing["build_direction"] == [0, 0, 1] and (manufacturing["supports_allowed"] or not needed) and accessibility["passed"]}

    def deviation(self, field, report):
        config = self.config
        extraction = {key: report["remesh"]["fidelity"][key] for key in ("maximum_sampled_deviation_mm", "relative_volume_change")}
        extraction["passed"] = extraction["maximum_sampled_deviation_mm"] <= config["surface_deviation_mm"] and abs(extraction["relative_volume_change"]) <= config["relative_volume_change"]
        reference = ImplicitField(field.origin, field.spacing, field.layers["reference"]).extract()[0]
        margin = float(np.linalg.norm(field.spacing))/2
        stencil = maximum_filter(np.pad(field.layers["extension_changed"], 2), size=4, origin=-1)[np.ix_(*(np.arange(n)//config["subdivisions"]+1 for n in field.values.shape))]
        excluded = _dilate(field.layers["opening_modified"] | stencil, field.spacing, config["free_zone_modified_mm"]+margin)
        del stencil
        constraint = config["constraint_offset_mm"]+config["free_zone_constraint_mm"]
        pad = max(config["free_zone_preserve_mm"], constraint)+2*margin
        excluded |= field.primitives(self.regions["preserve"], pad) > -config["free_zone_preserve_mm"]-margin
        excluded |= field.primitives(self.regions["forbidden"], pad) > -constraint-margin
        envelope = _envelope(self.domain)
        def free(points):
            index = np.clip(np.rint((points-field.origin)/field.spacing).astype(np.int64), 0, np.asarray(field.values.shape)-1)
            return ~excluded[tuple(index.T)] & (primitive_distance(points.T, envelope) > constraint)
        rows = {}
        for name, source, target in (("candidate_to_density", self.mesh, reference), ("density_to_candidate", reference, self.mesh)):
            points = _surface_samples(source)
            chosen = points[free(points)]
            rows[name] = {"sample_count": len(points), "free_sample_count": len(chosen), "surface_distance_mm": _statistics(_point_distances(chosen, target)) if len(chosen) else None}
        area = self.mesh.area_faces
        enough = all(row["free_sample_count"] >= config["free_zone_minimum_samples"] for row in rows.values())
        largest = max(row["surface_distance_mm"]["max"] for row in rows.values()) if enough else None
        free_zone = {**rows, "maximum_deviation_mm": largest, "free_area_fraction": float(area[free(self.mesh.triangles_center)].sum()/area.sum()), "excluded_sample_fraction": float(excluded.mean()),
                     "reference_mesh": _mesh_summary(reference), "lookup_margin_mm": margin, "passed": enough and largest <= config["surface_deviation_mm"],
                     "method": "Bidirectional exact MeshLab nearest-surface distances between the final mesh and marching cubes of the unextended, unsmoothed density minus threshold, on free-zone samples only: beyond the preserve and constraint distances from every exact primitive and the envelope and outside the opening- or extension-modified samples dilated by the configured distance; grid lookups add half the sample diagonal"}
        if not enough:
            free_zone["reason"] = "Too few free-zone samples; the deviation is not evaluable"
        return {"extraction": extraction, "free_zone": free_zone, "passed": extraction["passed"] and free_zone["passed"]}

    def connectivity(self, report, preserve):
        witnesses = {name: bool(report.get(name, {}).get("passed")) for name in ("witness", "density_witness", "opening_witness", "final_witness")}
        present = isinstance(preserve, dict) and "reason" not in preserve and all(row["missing_volume_mm3"] <= self.tolerance for row in preserve.values())
        result = {"witnesses": witnesses, "final_field_single_component": bool(report.get("final_witness", {}).get("single_component")), "mesh_body_count": int(self.mesh.body_count), "preserves_present": present,
                  "method": "Field witnesses on the hard density composition, the opened density, after the opening and on the final field; the final mesh is one body containing every exact preserve; mounts are never bridged"}
        result["passed"] = all(witnesses.values()) and result["final_field_single_component"] and result["mesh_body_count"] == 1 and present
        return result

    def maturity(self, reference):
        self.metrics = surface_metrics(self.mesh, self.domain, self.settings)
        if reference is None:
            return {"passed": False, "reason": "No reference geometry supplied; geometry screen alone cannot establish maturity"}
        return surface_maturity(self.metrics, reference, self.settings)

    def material_change(self, reference):
        other = _manifold(reference)
        added, removed = (self.body-other).volume(), (other-self.body).volume()
        return {"added_volume_mm3": added, "removed_volume_mm3": removed, "symmetric_difference_volume_mm3": added+removed, "net_volume_change_mm3": self.body.volume()-other.volume(),
                "method": "manifold3d float64 differences in both directions; diagnostic only"}

VALIDATION_CHECKS = ("gates", "topology", "self_intersections", "envelope", "forbidden", "preserve", "features", "supports", "deviation", "connectivity", "surface_maturity")

def validate_implicit(mesh, domain, field, report, settings=None, reference_metrics=None, reference_mesh=None, progress=None):
    started = perf_counter()
    overrides = settings or {}
    settings = _validation_settings({"maximum_wall_samples": report["settings"]["maximum_wall_samples"], **{key: value for key, value in overrides.items() if key in BUDGETS}})
    checks, timings, material = {}, {}, None
    def run(name, function, *arguments):
        clock = perf_counter()
        try:
            checks[name] = function(*arguments)
        except Exception as error:
            checks[name] = {"passed": False, "complete": False, "reason": "Not evaluable: "+type(error).__name__+": "+str(error)}
        timings[name] = perf_counter()-clock
        _progress(progress, "validation_"+name, checks[name])
    acceptance = None
    try:
        if set(overrides)-set(BUDGETS):
            raise ValueError("Only the validation sample budgets may be overridden; frozen gates cannot be weakened: " + ", ".join(sorted(set(overrides)-set(BUDGETS))))
        config, checks["gates"] = frozen_gates(report["settings"])
        acceptance = MeshAcceptance(mesh, domain, config, settings)
    except Exception as error:
        checks["topology" if "gates" in checks else "gates"] = {"passed": False, "reason": "Not evaluable: "+type(error).__name__+": "+str(error)}
    if acceptance is not None:
        run("topology", acceptance.topology)
        checks["self_intersections"] = getattr(acceptance, "intersections", {"passed": False, "reason": "Not evaluable without the topology check"})
        if checks["topology"]["passed"] and checks["self_intersections"]["passed"]:
            run("envelope", acceptance.envelope)
            run("forbidden", acceptance.forbidden)
            run("preserve", acceptance.preserve)
            run("features", acceptance.features, field, report)
            run("supports", acceptance.supports)
            run("deviation", acceptance.deviation, field, report)
            run("surface_maturity", acceptance.maturity, reference_metrics)
            if reference_mesh is not None:
                try:
                    material = acceptance.material_change(reference_mesh)
                except Exception as error:
                    material = {"reason": type(error).__name__+": "+str(error)}
        run("connectivity", acceptance.connectivity, report, checks.get("preserve"))
    violations = []
    for key in VALIDATION_CHECKS:
        value = checks.get(key)
        if value is None:
            violations.append(key+":not_evaluated")
        elif key in ("forbidden", "preserve") and "reason" not in value:
            violations.extend(key+":"+name for name, item in value.items() if not item["passed"])
        elif not value["passed"]:
            violations.append(key)
    return {"passed": not violations, "violations": violations, "checks": checks, "surface_metrics": getattr(acceptance, "metrics", None), "material_change": material, "settings": settings, "timings_s": timings, "runtime_s": perf_counter()-started,
            "acceptance_scope": "Geometry screens and surface maturity on the final mesh and its reloaded binary STL; mechanical verification and render review remain mandatory"}
