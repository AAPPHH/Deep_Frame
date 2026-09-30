from itertools import permutations
from math import isfinite
from time import perf_counter

import numpy as np
from build123d import Align, Box, Compound, Cylinder, Pos, Rot, Solid
from scipy.ndimage import generate_binary_structure, label


def region_shape(region):
    if region["kind"] == "box":
        lower = np.asarray(region["min_mm"], dtype=float)
        upper = np.asarray(region["max_mm"], dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower) or not np.all(np.isfinite([lower, upper])):
            raise ValueError("Region box needs finite ordered corners")
        return Pos(*lower) * Box(*(upper - lower), align=(Align.MIN, Align.MIN, Align.MIN))
    if region["kind"] == "cylinder":
        radius, height = region["radius_mm"], region["height_mm"]
        if not isfinite(radius) or not isfinite(height) or min(radius, height) <= 0:
            raise ValueError("Region cylinder dimensions must be finite and positive")
        rotation = {"x": (0, 90, 0), "y": (90, 0, 0), "z": (0, 0, 0)}[region["axis"]]
        return Pos(*region["center_mm"]) * Rot(*rotation) * Cylinder(radius, height)
    raise ValueError("Unsupported region primitive: " + str(region.get("kind")))


def _volume(shape):
    if shape is None:
        return 0.0
    if isinstance(shape, list):
        return sum(item.volume for item in shape)
    return shape.volume


def _compound(shape):
    return Compound(children=shape) if isinstance(shape, list) else shape


def _greedy_boxes(mask, order=(0, 1, 2)):
    remaining = np.transpose(np.asarray(mask, dtype=bool), order).copy()
    result = []
    while np.any(remaining):
        start = np.argwhere(remaining)[0]
        stop = start + 1
        for axis in (2, 1, 0):
            while stop[axis] < remaining.shape[axis]:
                selection = [slice(start[i], stop[i]) for i in range(3)]
                selection[axis] = slice(stop[axis], stop[axis] + 1)
                if not np.all(remaining[tuple(selection)]):
                    break
                stop[axis] += 1
        remaining[tuple(slice(start[i], stop[i]) for i in range(3))] = False
        reverse = np.argsort(order)
        result.append((start[reverse], stop[reverse]))
    return result


def voxel_boxes(mask):
    alternatives = [_greedy_boxes(mask, order) for order in permutations(range(3))]
    return min(alternatives, key=len)


def threshold_field(domain, density, settings):
    grid = domain["grid"]
    shape = tuple(grid["shape"])
    field = np.asarray(density, dtype=float)
    if field.shape != shape or not np.all(np.isfinite(field)) or np.any((field < 0) | (field > 1)):
        raise ValueError("Density must be finite in [0,1] with exactly grid.shape")
    allowed = np.asarray(domain["allowed"], dtype=bool)
    preserve = np.asarray(domain["preserve"], dtype=bool)
    forbidden = np.asarray(domain["forbidden"], dtype=bool)
    if any(mask.shape != shape for mask in (allowed, preserve, forbidden)) or np.any(preserve & ~allowed) or not np.array_equal(forbidden, ~allowed):
        raise ValueError("Inconsistent allowed/preserve/forbidden domain masks")
    threshold = settings.get("density_threshold", 0.4)
    if not isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("density_threshold must lie in (0,1]")
    if np.any(field[preserve] < 1 - 1e-8) or np.any(field[forbidden] > 1e-8):
        raise ValueError("Density violates prescribed preserve or forbidden masks")
    occupied = ((field >= threshold) & allowed) | preserve
    labels, count = label(occupied, generate_binary_structure(3, 1))
    repairs = []
    if count > 1 and settings.get("remove_unanchored_islands", False):
        anchor_labels = set(np.unique(labels[preserve])) - {0}
        if len(anchor_labels) != 1:
            raise ValueError("Disconnected required preserve regions; no automatic bridges are permitted")
        keep = labels == next(iter(anchor_labels))
        repairs.append({"method": "remove_unanchored_islands", "removed_voxels": int(np.sum(occupied & ~keep)), "removed_components": count - 1})
        occupied = keep
        count = 1
    if count != 1:
        raise ValueError(f"Density threshold produced {count} face-connected components")
    return occupied, {"method": "conservative_voxel_threshold_greedy_cuboids", "density_threshold": threshold, "occupied_voxels": int(np.sum(occupied)), "connectivity": 6, "repairs": repairs}


def reconstruct_topology(domain: dict, density, settings: dict) -> Solid:
    started = perf_counter()
    grid = domain["grid"]
    if grid.get("axis_order") != "xyz" or grid.get("order") != "C":
        raise ValueError("Only xyz axes and C-order fields are supported")
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    origin = np.asarray(grid["origin_mm"], dtype=float)
    if spacing.shape != (3,) or origin.shape != (3,) or not np.all(np.isfinite([spacing, origin])) or np.min(spacing) <= 0:
        raise ValueError("Grid spacing must be finite positive xyz values")
    occupied, report = threshold_field(domain, density, settings)
    boxes = voxel_boxes(occupied)
    shapes = [Pos(*(origin + start * spacing)) * Box(*((stop - start) * spacing), align=(Align.MIN, Align.MIN, Align.MIN)) for start, stop in boxes]
    frame = shapes[0].fuse(*shapes[1:], glue=True) if len(shapes) > 1 else shapes[0]
    report["cuboid_count"] = len(boxes)
    report["voxel_volume_mm3"] = float(np.sum(occupied) * np.prod(spacing))
    clipped_voxels = {}
    for region in domain["regions"]:
        if region["role"] == "forbidden" and region.get("rasterize", True):
            overlap = _volume(frame.intersect(region_shape(region)))
            if overlap > settings.get("volume_tolerance_mm3", 1e-5):
                clipped_voxels[region["name"]] = overlap
    if clipped_voxels:
        raise ValueError("Nonconservative domain would cut subvoxel material features: " + str(clipped_voxels))
    report["raster_forbidden_voxel_overlap_mm3"] = clipped_voxels
    preserves = [region_shape(region) for region in domain["regions"] if region["role"] == "preserve"]
    if preserves:
        frame = frame.fuse(*preserves)
    forbidden = [region_shape(region) for region in domain["regions"] if region["role"] == "forbidden"]
    if forbidden:
        frame = frame.cut(*forbidden)
    bounds = Pos(*origin) * Box(*(np.array(grid["shape"]) * spacing), align=(Align.MIN, Align.MIN, Align.MIN))
    frame = _compound(frame.intersect(bounds))
    if frame is None or not frame.is_valid or len(frame.solids()) != 1:
        count = 0 if frame is None else len(frame.solids())
        raise ValueError(f"Exact region Boolean reconstruction is not one valid solid ({count} solids)")
    solid = frame.solids()[0]
    report.update({"final_volume_mm3": float(solid.volume), "exact_region_volume_change_mm3": float(solid.volume - report["voxel_volume_mm3"]), "runtime_s": perf_counter() - started, "source_grid": grid, "minimum_voxel_feature_mm": float(np.min(spacing)), "preserve_union_count": len(preserves), "forbidden_cut_count": len(forbidden)})
    solid.topology_report = report
    solid.topology_occupied = occupied.copy()
    return solid


def validate_topology(solid, domain: dict, settings: dict) -> dict:
    from deep_frame.topology_validation import validate_topology as validate

    return validate(solid, domain, settings)
