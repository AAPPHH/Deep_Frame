from functools import reduce
from itertools import product
from time import perf_counter

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, generate_binary_structure, label
from skimage.measure import euler_number

from deep_frame.config import IMPLICIT_CONFIG, IMPLICIT_KINDS, configure
from deep_frame.topology_geometry import region_bounds, region_contains
from deep_frame.topology_surface import _domain_field, _progress, _upsample

AXES = {"x": 0, "y": 1, "z": 2}
SIX = generate_binary_structure(3, 1)
EXTENSIONS = ("none", "preserve", "preserve_forbidden")
NONNEGATIVE = ("density_sigma_mm", "transition_radius_mm", "preserve_inflation_mm", "constraint_offset_mm", "opening_radius_mm", "ripple_sigma_mm")
PROPAGATION_PASSES = 2

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
    return config

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
    if not report["witness"]["passed"]:
        raise ImplicitError("mount_disconnected", "Required mounts are not connected by the density; they are rejected, never bridged", report)
    if config["extension"] != "none":
        guard = {"original": _topology(occupied), "extended": _topology(_composition(field.values, preserve, forbidden, envelope))}
        guard["passed"] = guard["original"] == guard["extended"]
        report["extension_guard"] = guard
        if not guard["passed"]:
            raise ImplicitError("extension_changed_topology", "Density extension changed components or Euler number of the hard composition", report)
    del occupied
    lap("witness")
    _progress(progress, "field_witness", report)
    field.reinitialize(config["reinit_band_cells"])
    lap("reinit")
    report["volumes_mm3"]["density"] = field.volume()
    field.smooth(config["density_sigma_mm"]).reinitialize(config["reinit_band_cells"])
    report["volumes_mm3"]["smoothed"] = field.volume()
    lap("smoothing")
    preserve += np.float32(delta)
    field.smooth_union(preserve, k).intersect(-forbidden-np.float32(offset)).intersect(envelope-np.float32(offset))
    report["volumes_mm3"]["composed"] = field.volume()
    lap("composition")
    field.reinitialize(config["reinit_band_cells"])
    field.layers["pre_opening"] = field.values.copy()
    report["opening"] = field.open(config["opening_radius_mm"], config["reinit_band_cells"]) if config["opening_radius_mm"] else None
    lap("opening")
    field.smooth(config["ripple_sigma_mm"])
    report["volumes_mm3"]["ripple_smoothed"] = field.volume()
    field.union(preserve)
    report["volumes_mm3"]["final"] = field.volume()
    lap("restore")
    witness = mount_witness(field, field.values > 0, preserves, forbidden)
    witness["single_component"] = witness["occupied_components_6"] == 1
    report["final_witness"] = witness
    field.layers["reference"] = reference.values
    if not witness["passed"]:
        raise ImplicitError("mount_disconnected", "Final field separates required mounts", report)
    if not witness["single_component"]:
        raise ImplicitError("field_disconnected", "Final field has detached bodies; they are neither bridged nor removed", report)
    lap("witness")
    report["status"] = "field_built"
    _progress(progress, "field_built", report)
    return field, report
