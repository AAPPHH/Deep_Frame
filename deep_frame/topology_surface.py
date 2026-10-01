from copy import deepcopy
from io import BytesIO
from itertools import product
from math import isfinite
from time import perf_counter

import numpy as np
import trimesh
from build123d import Compound, Face, Shape, Shell, Solid, Wire
from OCP.BOPAlgo import BOPAlgo_BOP, BOPAlgo_COMMON, BOPAlgo_CUT, BOPAlgo_FUSE
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_OUT
from OCP.collections import List_TopoDS_Shape
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_opening, label, map_coordinates, minimum_filter
from scipy.interpolate import PchipInterpolator
from skimage.measure import marching_cubes

from .topology_geometry import region_shape
from .topology_domain import region_bounds, region_contains


class SurfaceReconstructionError(ValueError):
    def __init__(self, message, report, shape=None, mesh=None):
        super().__init__(message)
        self.report = report
        self.shape = shape
        self.mesh = mesh


def _progress(callback, stage, report, **artifacts):
    if callback is not None:
        callback({"stage": stage, "report": report, **artifacts})


def _positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(name + " must be finite and " + ("nonnegative" if zero else "positive"))
    return float(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _settings(settings):
    result = {
        "density_threshold": 0.30,
        "interpolation_subdivisions": 4,
        "interpolation_method": "pchip",
        "boolean_strategy": "occt_serial",
        "boolean_fuzzy_value_mm": 1e-7,
        "preserve_fusion_mode": "direct",
        "density_smoothing_sigma_mm": 0.0,
        "manufacturing_opening_radius_mm": 0.0,
        "manufacturing_opening_method": "grayscale",
        "free_forbidden_buffer_mm": 0.0,
        "surface_constraint_mode": "embedded",
        "decimation_bounds_mode": "none",
        "maximum_surface_deviation_mm": 0.20,
        "maximum_relative_volume_change": 0.01,
        "decimation_face_budgets": [6000, 12000, 24000, 48000],
        "fidelity_batch_size": 1000,
        "boolean_tolerance_mm3": 1e-5,
    }
    unknown = set(settings) - set(result)
    if unknown:
        raise ValueError("Unknown continuous surface settings: " + ", ".join(sorted(unknown)))
    result.update(deepcopy(settings))
    if result["interpolation_method"] not in ("cubic", "pchip"):
        raise ValueError("interpolation_method must be cubic or pchip")
    if result["boolean_strategy"] not in ("occt_serial", "build123d"):
        raise ValueError("boolean_strategy must be occt_serial or build123d")
    if result["preserve_fusion_mode"] not in ("direct", "preunion"):
        raise ValueError("preserve_fusion_mode must be direct or preunion")
    if result["preserve_fusion_mode"] == "preunion" and result["boolean_strategy"] != "occt_serial":
        raise ValueError("Preserve preunion requires the guarded occt_serial strategy")
    if result["manufacturing_opening_method"] not in ("grayscale", "distance"):
        raise ValueError("manufacturing_opening_method must be grayscale or distance")
    if result["surface_constraint_mode"] not in ("embedded", "cad_only", "envelope_only", "envelope_forbidden"):
        raise ValueError("surface_constraint_mode must be embedded, cad_only, envelope_only or envelope_forbidden")
    if result["decimation_bounds_mode"] not in ("none", "reference_aabb"):
        raise ValueError("decimation_bounds_mode must be none or reference_aabb")
    for name in ("density_threshold", "maximum_surface_deviation_mm", "maximum_relative_volume_change", "boolean_tolerance_mm3", "boolean_fuzzy_value_mm"):
        result[name] = _positive(result[name], name)
    if result["boolean_strategy"] == "build123d" and result["boolean_fuzzy_value_mm"] != 1e-7:
        raise ValueError("Explicit Boolean fuzzy tolerance requires occt_serial strategy")
    if result["density_threshold"] >= 1 or result["maximum_relative_volume_change"] >= 1:
        raise ValueError("Density threshold and relative volume change must be less than one")
    result["density_smoothing_sigma_mm"] = _positive(result["density_smoothing_sigma_mm"], "density_smoothing_sigma_mm", zero=True)
    result["manufacturing_opening_radius_mm"] = _positive(result["manufacturing_opening_radius_mm"], "manufacturing_opening_radius_mm", zero=True)
    result["free_forbidden_buffer_mm"] = _positive(result["free_forbidden_buffer_mm"], "free_forbidden_buffer_mm", zero=True)
    if result["free_forbidden_buffer_mm"] > 0 and (result["surface_constraint_mode"] != "envelope_forbidden" or result["manufacturing_opening_radius_mm"] <= 0):
        raise ValueError("Positive free_forbidden_buffer_mm requires envelope_forbidden mode and a positive manufacturing opening radius")
    for name in ("interpolation_subdivisions", "fidelity_batch_size"):
        result[name] = _integer(result[name], name)
    budgets = result["decimation_face_budgets"]
    if not isinstance(budgets, (list, tuple)) or not budgets:
        raise ValueError("decimation_face_budgets must be a nonempty increasing list")
    budgets = [_integer(value, "decimation face budget") for value in budgets]
    if budgets != sorted(set(budgets)):
        raise ValueError("decimation_face_budgets must strictly increase")
    result["decimation_face_budgets"] = budgets
    return result


def _domain_field(domain, density):
    grid = domain["grid"]
    if grid.get("axis_order") != "xyz" or grid.get("order") != "C":
        raise ValueError("Only xyz axes and C-order fields are supported")
    shape = grid["shape"]
    if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in shape):
        raise ValueError("Grid shape needs three integer dimensions of at least two")
    origin, spacing = np.asarray(grid["origin_mm"], dtype=float), np.asarray(grid["spacing_mm"], dtype=float)
    if origin.shape != (3,) or spacing.shape != (3,) or not np.all(np.isfinite([origin, spacing])) or np.any(spacing <= 0):
        raise ValueError("Grid origin and positive spacing must be finite triples")
    field = np.asarray(density, dtype=float)
    if field.shape != tuple(shape) or not np.all(np.isfinite(field)) or np.any((field < 0) | (field > 1)):
        raise ValueError("Density must be finite in [0,1] with exactly grid.shape")
    masks = {name: np.asarray(domain[name], dtype=bool) for name in ("allowed", "preserve", "forbidden")}
    if any(mask.shape != field.shape for mask in masks.values()) or not np.array_equal(masks["forbidden"], ~masks["allowed"]) or np.any(masks["preserve"] & ~masks["allowed"]):
        raise ValueError("Inconsistent domain masks")
    if np.any(field[masks["preserve"]] < 1-1e-8) or np.any(field[masks["forbidden"]] > 1e-8):
        raise ValueError("Density violates prescribed preserve or forbidden masks")
    return field, origin, spacing


def _mesh_summary(mesh):
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces), "bodies": int(mesh.body_count),
            "euler_number": int(mesh.euler_number),
            "watertight": bool(mesh.is_watertight), "consistent_winding": bool(mesh.is_winding_consistent),
            "volume_mm3": float(mesh.volume), "area_mm2": float(mesh.area)}


def _one_mesh(mesh):
    return bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.body_count == 1 and mesh.volume > 0)


def _closed_oriented_mesh(mesh):
    return bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0)


def _mesh_intersections(mesh):
    import pymeshlab

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64), face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    mesh_set.apply_filter("compute_selection_by_self_intersections_per_face")
    selected = np.flatnonzero(mesh_set.current_mesh().face_selection_array())
    report = {"method": "MeshLab compute_selection_by_self_intersections_per_face; selection only, no mesh modification",
              "self_intersecting_faces": len(selected), "passed": len(selected) == 0}
    if len(selected):
        points = mesh.triangles[selected].reshape((-1, 3))
        report.update(face_indices=selected.tolist(), bounds_mm=[points.min(axis=0).tolist(), points.max(axis=0).tolist()])
    return report


def _statistics(values):
    return {"max": float(np.max(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "mean": float(np.mean(values))}


def _directed_fidelity(source, target, batch_size):
    triangles = source.triangles
    points = np.concatenate((triangles.mean(axis=1), triangles[:, 0]*0.5+triangles[:, 1]*0.5,
                             triangles[:, 1]*0.5+triangles[:, 2]*0.5, triangles[:, 0]*0.5+triangles[:, 2]*0.5,
                             source.vertices))
    distances, angles = [], []
    for start in range(0, len(points), batch_size):
        _, distance, indices = trimesh.proximity.closest_point(target, points[start:start+batch_size])
        distances.extend(distance.tolist())
        stop = min(start+batch_size, len(source.faces))
        if start < stop:
            cosine = np.sum(source.face_normals[start:stop]*target.face_normals[indices[:stop-start]], axis=1)
            angles.extend(np.degrees(np.arccos(np.clip(cosine, -1, 1))).tolist())
    return {"sample_count": len(points), "surface_distance_mm": _statistics(distances),
            "normal_sample_count": len(source.faces), "normal_angle_deg": _statistics(angles)}


def measure_surface_fidelity(reference, approximation, batch_size=1000):
    started = perf_counter()
    forward = _directed_fidelity(reference, approximation, batch_size)
    reverse = _directed_fidelity(approximation, reference, batch_size)
    return {"method": "bidirectional vertices, centroids and all edge midpoints; nearest triangle distance; centroid normals",
            "limitation": "finite deterministic sampling is not an analytic Hausdorff or continuous-field error bound",
            "reference_to_approximation": forward, "approximation_to_reference": reverse,
            "maximum_sampled_deviation_mm": max(forward["surface_distance_mm"]["max"], reverse["surface_distance_mm"]["max"]),
            "relative_volume_change": float(approximation.volume/reference.volume-1), "runtime_s": perf_counter()-started}


def _primitive_inside_distance(axes, region):
    
    if region["kind"] == "box":
        values = [np.minimum(axes[i]-region["min_mm"][i], region["max_mm"][i]-axes[i]) for i in range(3)]
        return np.minimum(np.minimum(values[0], values[1]), values[2])
    axis = {"x": 0, "y": 1, "z": 2}[region.get("axis", "z")]
    radial = [i for i in range(3) if i != axis]
    radial_distance = np.sqrt(sum((axes[i]-region["center_mm"][i])**2 for i in radial))
    return np.minimum(region["radius_mm"]-radial_distance,
                      region["height_mm"]/2-np.abs(axes[axis]-region["center_mm"][axis]))


def _primitive_union_distance(axes, regions, shape, sample_spacing):
    values = np.full(shape, -np.inf)
    for region in regions:
        np.maximum(values, _primitive_inside_distance(axes, region), out=values)
    
    
    
    boxes = [region for region in regions if region["kind"] == "box"]
    indices = np.argwhere(np.abs(values) < 1e-8) if len(boxes) > 1 else np.empty((0, 3), dtype=int)
    corrected = 0
    epsilon = float(np.min(sample_spacing)*1e-3)
    if len(indices):
        points = np.column_stack([axes[i].ravel()[indices[:, i]] for i in range(3)])
        interior = np.ones(len(points), dtype=bool)
        
        
        for region in boxes:
            for coordinate in range(3):
                for boundary in (region["min_mm"][coordinate], region["max_mm"][coordinate]):
                    distance = np.abs(points[:, coordinate]-boundary)
                    interior &= ~((distance > 1e-8) & (distance <= epsilon))
        for direction in product((-1, 0, 1), repeat=3):
            if direction == (0, 0, 0):
                continue
            tested = points+epsilon*np.asarray(direction)
            covered = np.zeros(len(points), dtype=bool)
            for region in boxes:
                covered |= region_contains(tested, region)
            interior &= covered
        selected = indices[interior]
        corrected = len(selected)
        if corrected:
            inside_distance = distance_transform_edt(values >= -1e-8, sampling=sample_spacing)
            key = tuple(selected.T)
            values[key] = np.maximum(inside_distance[key]-np.linalg.norm(sample_spacing)/2, np.min(sample_spacing)/2)
    return values, {"rectilinear_internal_zero_nodes_corrected": corrected, "interior_probe_half_width_mm": epsilon, "coincident_plane_tolerance_mm": 1e-8,
                    "method": "For zero-valued box-union nodes, all 26 adjacent sign cells are inside the exact primitive union; only these interior scalar values are made positive."}


def _manufacturing_opening(sampled, sample_origin, sample_spacing, domain, config):
    




    radius = config["manufacturing_opening_radius_mm"]
    threshold = config["density_threshold"]
    gradient = np.sqrt(sum(value**2 for value in np.gradient(sampled, *sample_spacing)))
    implicit = (sampled-threshold)/np.maximum(gradient, 1e-6)
    mode = config["surface_constraint_mode"]
    forbidden_buffer = config["free_forbidden_buffer_mm"]
    axes = [sample_origin[i]+np.arange(sampled.shape[i]).reshape(tuple(-1 if j == i else 1 for j in range(3)))*sample_spacing[i] for i in range(3)]
    preserves, preserve_union_report = _primitive_union_distance(axes, [region for region in domain["regions"] if region["role"] == "preserve"], sampled.shape, sample_spacing)
    forbidden, forbidden_union_report = _primitive_union_distance(axes, [region for region in domain["regions"] if region["role"] == "forbidden"], sampled.shape, sample_spacing)
    lower = np.asarray(domain["grid"]["origin_mm"])
    upper = lower+np.asarray(domain["grid"]["spacing_mm"])*np.asarray(domain["grid"]["shape"])
    envelope = _primitive_inside_distance(axes, {"kind": "box", "min_mm": lower, "max_mm": upper})
    if mode == "embedded":
        np.maximum(implicit, preserves, out=implicit)
    unbuffered_occupied = None
    if forbidden_buffer > 0:
        unbuffered_occupied = (implicit > 0) & (forbidden < 0) & (envelope > 0)
    if mode in ("embedded", "envelope_forbidden"):
        np.minimum(implicit, -forbidden-forbidden_buffer if forbidden_buffer else -forbidden, out=implicit)
    if mode in ("embedded", "envelope_only", "envelope_forbidden"):
        np.minimum(implicit, envelope, out=implicit)
    occupied = implicit > 0
    voxel_volume = float(np.prod(sample_spacing))
    before_count = int(occupied.sum())
    if unbuffered_occupied is None:
        unbuffered_occupied = occupied
    buffer_report = {"value_mm": forbidden_buffer, "applied": forbidden_buffer > 0,
                     "method": "Before opening, intersect only the free density implicit with envelope and minus the existing forbidden-union scalar minus the buffer; no analytic preserve imprint.",
                     "metric_limitation": "Offset of the existing primitive-union scalar, not a rounded Euclidean offset or a guaranteed final CAD separation. The QEM deviation gate does not establish clearance.",
                     "before_buffer_sampled_volume_mm3": int(unbuffered_occupied.sum())*voxel_volume,
                     "after_buffer_sampled_volume_mm3": before_count*voxel_volume,
                     "removed_sampled_volume_mm3": int(np.count_nonzero(unbuffered_occupied & ~occupied))*voxel_volume,
                     "added_sampled_volume_mm3": int(np.count_nonzero(occupied & ~unbuffered_occupied))*voxel_volume,
                     "forbidden_region_count": sum(region["role"] == "forbidden" for region in domain["regions"]),
                     "nonrasterized_forbidden_count": sum(region["role"] == "forbidden" and region.get("rasterize") is False for region in domain["regions"]),
                     "original_domain_and_cad_primitives_modified": False}
    if config["manufacturing_opening_method"] == "distance":
        erosion_margin = float(np.linalg.norm(sample_spacing)/2)
        eroded = distance_transform_edt(occupied, sampling=sample_spacing) > radius+erosion_margin
        opened = np.minimum(implicit, radius-distance_transform_edt(~eroded, sampling=sample_spacing))
        method = "conservative sampled Euclidean erosion and dilation, clipped to the opening input implicit material"
        operation_detail = {"erosion_boundary_margin_mm": erosion_margin}
    else:
        extents = np.ceil(radius/sample_spacing).astype(int)
        offsets = np.meshgrid(*[np.arange(-n, n+1)*sample_spacing[i] for i, n in enumerate(extents)], indexing="ij")
        footprint = sum(axis**2 for axis in offsets) <= radius**2+1e-12
        opened = grey_opening(implicit, footprint=footprint, mode="constant", cval=float(implicit.min()-2*radius))
        
        
        if np.any(opened > implicit+1e-9):
            raise ValueError("Grayscale opening unexpectedly added implicit material")
        np.minimum(opened, implicit, out=opened)
        method = "flat grayscale minimum-then-maximum opening with a symmetric physical-space spherical footprint"
        operation_detail = {"footprint_shape": footprint.shape, "footprint_samples": int(footprint.sum()),
                            "maximum_footprint_radius_mm": float(np.sqrt(sum(axis**2 for axis in offsets)[footprint].max())),
                            "erosion_boundary_margin_mm": 0.0}
    opened_occupied = opened > 0
    required = (preserves > 0) & (forbidden < 0) & (envelope > 0)
    report = {"method": method,
              "surface_constraint_mode": mode,
              "opening_input": {"embedded": "density-derived approximate signed distance with analytic preserve/forbidden/envelope imprints", "cad_only": "density-derived approximate signed distance only; no analytic preserve/forbidden/envelope imprints", "envelope_only": "density-derived approximate signed distance intersected with analytic envelope only; preserve and forbidden primitives are applied only in final CAD", "envelope_forbidden": "density-derived approximate signed distance intersected with envelope and with every forbidden primitive subtracted, including rasterize=False; analytical preserves are applied only in final CAD"}[mode],
              "constraint_stages": "Exact CAD preserves, all forbidden primitives including rasterize=False, and envelope are applied after surface decimation. Sampled analytic restoration below is a separate diagnostic forecast, never the returned opening field.",
              "source_density_constraints": "Stored density and its original optimizer masks remain unchanged in every mode.",
              "free_forbidden_buffer": buffer_report,
              "radius_mm": radius, "sample_spacing_mm": sample_spacing.tolist(), **operation_detail,
              "preserve_union_implicit": preserve_union_report, "forbidden_union_implicit": forbidden_union_report,
              "sampling_volume_mm3": voxel_volume, "before_components_6_neighbor": int(label(occupied)[1]),
              "opened_components_6_neighbor": int(label(opened_occupied)[1]),
              "before_sampled_volume_mm3": before_count*voxel_volume,
              "opened_sampled_volume_mm3": int(opened_occupied.sum())*voxel_volume,
              "opening_removed_sampled_volume_mm3": int(np.count_nonzero(occupied & ~opened_occupied))*voxel_volume,
              "opening_added_sampled_volume_mm3": int(np.count_nonzero(opened_occupied & ~occupied))*voxel_volume,
              "preserve_removed_by_opening_sampled_volume_mm3": int(np.count_nonzero(required & occupied & ~opened_occupied))*voxel_volume,
              "preserve_missing_before_opening_sampled_volume_mm3": int(np.count_nonzero(required & ~occupied))*voxel_volume,
              "preserve_missing_after_opening_sampled_volume_mm3": int(np.count_nonzero(required & ~opened_occupied))*voxel_volume,
              "limitation": "Sampled physical-space opening is not an analytic CAD minimum-thickness proof; exact final CAD validation remains mandatory."}
    
    
    
    restored = np.minimum(np.minimum(np.maximum(opened, preserves), -forbidden), envelope)
    after = restored > 0
    report.update(after_components_6_neighbor=int(label(after)[1]),
                  after_sampled_volume_mm3=int(after.sum())*voxel_volume,
                  removed_sampled_volume_mm3=int(np.count_nonzero(occupied & ~after))*voxel_volume,
                  added_sampled_volume_mm3=int(np.count_nonzero(after & ~occupied))*voxel_volume,
                  missing_preserve_sampled_volume_mm3=int(np.count_nonzero(required & ~after))*voxel_volume,
                  diagnostic_constraints_added_to_opened_sampled_volume_mm3=int(np.count_nonzero(after & ~opened_occupied))*voxel_volume,
                  diagnostic_constraints_removed_from_opened_sampled_volume_mm3=int(np.count_nonzero(opened_occupied & ~after))*voxel_volume,
                  preserve_restoration_stage="Exact CAD union after density-surface decimation; after-restoration samples above are diagnostic only")
    return opened, report


def _regularize_internal_zero_nodes(field, level, origin=None, spacing=None, domain=None):
    output = field.copy()
    candidates = np.argwhere((field == 0) & (minimum_filter(field, size=3, mode="constant", cval=-np.inf) >= 0))
    changed = []
    bounds = [] if domain is None else [region_bounds(region) for region in domain["regions"] if region["role"] == "forbidden"]
    if domain is not None:
        lower = np.asarray(domain["grid"]["origin_mm"])
        upper = lower+np.asarray(domain["grid"]["spacing_mm"])*np.asarray(domain["grid"]["shape"])
    target = float(np.nextafter(np.float32(level), np.float32(np.inf)))
    for index in candidates:
        star = field[tuple(slice(i-1, i+2) for i in index)]
        if star.shape != (3, 3, 3) or not all(np.any(star[i:i+2, j:j+2, k:k+2] > 0) for i, j, k in product((0, 1), repeat=3)):
            continue
        if not all(star[neighbor] > level for neighbor in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2))):
            continue
        if domain is not None:
            minimum, maximum = origin+(index-1)*spacing, origin+(index+1)*spacing
            if np.any(minimum <= lower) or np.any(maximum >= upper) or any(np.all(minimum <= high) and np.all(maximum >= low) for low, high in bounds):
                continue
        output[tuple(index)] = target
        changed.append(index.tolist())
    report = {"method": "Exact-zero nodes only; all 26 neighbors nonnegative; all six axis neighbors above the positive contour level; each of eight incident trilinear cells has a positive corner; incident cell box strictly inside envelope and disjoint from forbidden AABBs",
              "changed_node_count": len(changed), "changed_indices": changed, "target_value": target,
              "regular_zero_superlevel_material_volume_change_mm3": 0.0,
              "scope": "Regularizes zero-measure interior zeros of the trilinear zero superlevel; does not remove negative cavities or finite zero plateaus. Final contour and CAD geometry still require independent validation."}
    if origin is not None and spacing is not None:
        report["changed_world_mm"] = [(origin+np.asarray(index)*spacing).tolist() for index in changed]
    return output, report


def _contour_mesh(field, level, sample_spacing, sample_origin):
    vertices, faces, _, _ = marching_cubes(field, level, spacing=sample_spacing, allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=vertices+sample_origin, faces=faces, process=True)
    trimesh.repair.fix_normals(mesh, multibody=False)
    return mesh


def _project_reference_bounds(mesh, reference):
    bounds = np.asarray(reference.bounds)
    before = mesh.vertices.copy()
    output = mesh.copy()
    output.vertices = np.clip(before, bounds[0], bounds[1])
    movement = np.linalg.norm(output.vertices-before, axis=1)
    changed = np.flatnonzero(movement > 0)
    degenerate = np.flatnonzero(~output.nondegenerate_faces())
    report = {"method": "Coordinate projection of QEM vertices to the unmodified reference mesh AABB; unchanged face indices; no welding, face removal, filling or normal repair",
              "reference_bounds_mm": bounds.tolist(), "bounds_before_mm": mesh.bounds.tolist(), "bounds_after_mm": output.bounds.tolist(),
              "moved_vertex_count": len(changed), "moved_vertex_indices": changed.tolist(),
              "maximum_vertex_displacement_mm": float(movement.max(initial=0)),
              "volume_before_mm3": float(mesh.volume), "volume_after_mm3": float(output.volume),
              "net_volume_change_mm3": float(output.volume-mesh.volume),
              "face_indices_unchanged": bool(np.array_equal(output.faces, mesh.faces)),
              "degenerate_face_count_before": int(np.count_nonzero(~mesh.nondegenerate_faces())),
              "degenerate_face_count": len(degenerate), "degenerate_face_indices": degenerate.tolist(),
              "degenerate_screen": "Trimesh nondegenerate_faces with its unchanged default minimum-height criterion; flagged triangles are rejected, never removed",
              "within_reference_aabb": bool(np.all(output.vertices >= bounds[0]) and np.all(output.vertices <= bounds[1])),
              "acceptance_scope": "This operation grants no acceptance; closed orientation, topology, self-intersections and complete fidelity/volume gates must be checked on the projected mesh."}
    return output, report


def extract_density_surface(domain, density, settings, *, progress=None):
    started = perf_counter()
    config = _settings(settings)
    field, origin, spacing = _domain_field(domain, density)
    padded = np.pad(field, 1)
    subdivisions = config["interpolation_subdivisions"]
    if config["interpolation_method"] == "cubic":
        coordinates = np.meshgrid(*[np.arange((n-1)*subdivisions+1)/subdivisions for n in padded.shape], indexing="ij")
        sampled = np.clip(map_coordinates(padded, np.asarray(coordinates), order=3, mode="constant", cval=0), 0, 1)
    else:
        sampled = padded
        for axis in range(3):
            count = padded.shape[axis]
            sampled = PchipInterpolator(np.arange(count), sampled, axis=axis)(np.arange((count-1)*subdivisions+1)/subdivisions)
    smoothing_report = None
    if config["density_smoothing_sigma_mm"]:
        smoothed = gaussian_filter(sampled, config["density_smoothing_sigma_mm"]/(spacing/subdivisions), mode="constant", cval=0)
        before_mask, after_mask = sampled > config["density_threshold"], smoothed > config["density_threshold"]
        cell_volume = float(np.prod(spacing/subdivisions))
        smoothing_report = {"method": "Gaussian filtering after continuous-density interpolation in physical millimetres; stored optimizer field unchanged",
                            "sigma_mm": config["density_smoothing_sigma_mm"], "sample_spacing_mm": (spacing/subdivisions).tolist(),
                            "maximum_density_change": float(np.max(np.abs(smoothed-sampled))),
                            "added_sampled_volume_mm3": int(np.count_nonzero(after_mask & ~before_mask))*cell_volume,
                            "removed_sampled_volume_mm3": int(np.count_nonzero(before_mask & ~after_mask))*cell_volume,
                            "limitation": "Sampled threshold volumes before exact constraints, not exact final CAD symmetric-difference volumes"}
        sampled = smoothed
    if not sampled.min() < config["density_threshold"] < sampled.max():
        raise ValueError("Density threshold has no isosurface")
    opening_report, before_zero_mesh = None, None
    level = config["density_threshold"]
    if config["manufacturing_opening_radius_mm"]:
        sampled, opening_report = _manufacturing_opening(sampled, origin-spacing/2, spacing/subdivisions, domain, config)
        
        
        
        
        level = 0.001
        opening_report["nominal_implicit_contour_level_mm"] = level
        opening_report["contour_level_limitation"] = "Density divided by local gradient only approximates signed distance; the nominal positive level is not a uniform physical offset guarantee."
        corrected, zero_report = _regularize_internal_zero_nodes(sampled, level, origin-spacing/2, spacing/subdivisions, domain)
        if zero_report["changed_node_count"]:
            before_zero_mesh = _contour_mesh(sampled, level, spacing/subdivisions, origin-spacing/2)
            zero_report["positive_contour_before"] = _mesh_summary(before_zero_mesh)
        sampled = corrected
        opening_report["internal_zero_regularization"] = zero_report
        opening_report["internal_zero_regularization_constraint_scope"] = "Unchanged narrow interior-zero rule; analytic envelope and forbidden bounds only veto candidate edits, never imprint hardware surfaces. Actual changed-node count and positive-contour volume change are recorded separately."
    reference = _contour_mesh(sampled, level, spacing/subdivisions, origin-spacing/2)
    report = {"method": config["interpolation_method"]+" density interpolation in fixed xyz axis order, Lewiner isosurface, error-gated quadric decimation, exact CAD hardware booleans",
              "settings": config, "grid": deepcopy(domain["grid"]), "isosurface": _mesh_summary(reference), "decimation_attempts": []}
    report["manufacturing_opening"] = opening_report
    report["density_smoothing"] = smoothing_report
    if before_zero_mesh is not None:
        zero_report["positive_contour_after"] = _mesh_summary(reference)
        zero_report["positive_contour_net_volume_change_mm3"] = reference.volume-before_zero_mesh.volume
        zero_report["positive_contour_volume_scope"] = "Actual triangulated positive-level contour change; distinct from the zero-superlevel regularization statement; does not equate net volume with separate added and removed volumes."
        _progress(progress, "before_internal_zero_regularization", report, mesh=before_zero_mesh)
    _progress(progress, "density_isosurface", report, mesh=reference)
    if not _closed_oriented_mesh(reference):
        raise SurfaceReconstructionError("Continuous density surface must be closed and consistently oriented; no hole filling is permitted", report, mesh=reference)
    report["isosurface_intersections"] = _mesh_intersections(reference)
    if not report["isosurface_intersections"]["passed"]:
        raise SurfaceReconstructionError("Continuous density surface has self-intersections", report, mesh=reference)
    if reference.body_count != 1:
        report["shell_signed_volumes_mm3"] = [piece.volume for piece in reference.split(only_watertight=False)]
        if any(volume <= 0 for volume in report["shell_signed_volumes_mm3"]):
            raise SurfaceReconstructionError("Enclosed negative cavity shells require explicit CAD inner-shell construction; they are not filled or turned into positive solids", report, mesh=reference)
    if reference.body_count != 1 and not any(region["role"] == "preserve" for region in domain["regions"]):
        raise SurfaceReconstructionError("Disconnected density bodies are not bridged without prescribed preserve geometry", report, mesh=reference)
    selected = None
    for budget in config["decimation_face_budgets"]:
        mesh = reference.copy() if budget >= len(reference.faces) else reference.simplify_quadric_decimation(face_count=budget)
        projection_report = None
        if config["decimation_bounds_mode"] == "reference_aabb":
            mesh, projection_report = _project_reference_bounds(mesh, reference)
            envelope_upper = origin+spacing*np.asarray(domain["grid"]["shape"])
            projection_report["reference_aabb_inside_grid_envelope"] = bool(np.all(reference.bounds[0] >= origin) and np.all(reference.bounds[1] <= envelope_upper))
            projection_report["grid_envelope_bounds_mm"] = [origin.tolist(), envelope_upper.tolist()]
        attempt = {"face_budget": budget, "mesh": _mesh_summary(mesh)}
        if projection_report is not None:
            attempt["reference_bounds_projection"] = projection_report
        if projection_report is not None and projection_report["degenerate_face_count"]:
            attempt.update(passed=False, reason="reference-bounds candidate contains degenerate triangles; no triangle removal is permitted")
            attempt["intersections"] = {"passed": False, "complete": False, "reason": "Not run on triangles rejected by the degeneration gate"}
        else:
            attempt["intersections"] = _mesh_intersections(mesh)
        if not attempt["intersections"]["passed"]:
            attempt.update(passed=False, reason=attempt.get("reason", "decimation introduced geometric self-intersections"))
        elif _closed_oriented_mesh(mesh) and mesh.body_count == reference.body_count and mesh.euler_number == reference.euler_number:
            fidelity = measure_surface_fidelity(reference, mesh, config["fidelity_batch_size"])
            attempt["fidelity"] = fidelity
            attempt["passed"] = fidelity["maximum_sampled_deviation_mm"] <= config["maximum_surface_deviation_mm"] and abs(fidelity["relative_volume_change"]) <= config["maximum_relative_volume_change"]
        else:
            attempt.update(passed=False, reason="decimation changed closed oriented component topology")
        report["decimation_attempts"].append(attempt)
        _progress(progress, "decimation_"+str(budget), report, mesh=mesh)
        if attempt["passed"]:
            selected = mesh
            break
    if selected is None:
        raise SurfaceReconstructionError("No decimation budget met the explicit surface deviation, volume and topology limits", report, mesh=reference)
    report["selected_surface"] = _mesh_summary(selected)
    report["surface_extraction_s"] = perf_counter()-started
    return selected, report


def _single_solid(shape, label, report=None, previous=None):
    if isinstance(shape, list):
        shape = Compound(children=shape)
    warnings = [] if shape is None else getattr(shape, "boolean_warnings", [])
    blocking_warnings = [warning for warning in warnings if any(key in warning for key in ("AcquiredSelfIntersection", "SolidBuilderUnusedFaces", "NotSplittableEdge"))]
    if shape is None or len(shape.solids()) != 1 or not shape.is_valid or blocking_warnings:
        count = 0 if shape is None else len(shape.solids())
        if report is not None:
            report["failed_operation"] = {"name": label, "output_solid_count": count,
                                           "output_volumes_mm3": [] if shape is None else [item.volume for item in shape.solids()],
                                           "warnings": warnings, "blocking_warnings": blocking_warnings}
            message = label + f" must yield exactly one valid solid, got {count}"
            if blocking_warnings:
                message += "; native Boolean geometry warnings: " + ", ".join(blocking_warnings)
            raise SurfaceReconstructionError(message, report, previous)
        raise ValueError(label + f" must yield exactly one valid solid, got {count}")
    return shape.solids()[0]


def _boolean(shape, tools, operation_name, strategy, fuzzy_value_mm=1e-7):
    if strategy == "build123d":
        return getattr(shape, operation_name)(*tools)
    operation = BOPAlgo_BOP()
    operation.SetOperation({"fuse": BOPAlgo_FUSE, "cut": BOPAlgo_CUT, "intersect": BOPAlgo_COMMON}[operation_name])
    arguments, operands = List_TopoDS_Shape(), List_TopoDS_Shape()
    arguments.Append(shape.wrapped)
    for tool in tools:
        operands.Append(tool.wrapped)
    operation.SetArguments(arguments)
    operation.SetTools(operands)
    operation.SetRunParallel(False)
    operation.SetNonDestructive(True)
    operation.SetFuzzyValue(fuzzy_value_mm)
    operation.Perform()
    warnings, errors = BytesIO(), BytesIO()
    operation.DumpWarnings(warnings)
    operation.DumpErrors(errors)
    warning_messages = warnings.getvalue().decode(errors="replace").splitlines()
    if operation.HasErrors() or operation.Shape().IsNull():
        raise RuntimeError("OCCT serial exact " + operation_name + " failed: " + errors.getvalue().decode(errors="replace") + "; warnings: " + ", ".join(warning_messages))
    result = Shape.cast(operation.Shape())
    result.boolean_warnings = warning_messages
    result.boolean_fuzzy_value_mm = operation.FuzzyValue()
    return result


def _checked_operation(solid, operands, operation, label, report, progress):
    started = perf_counter()
    before_volume = solid.volume
    try:
        result = _boolean(solid, operands, operation, report["settings"]["boolean_strategy"], report["settings"]["boolean_fuzzy_value_mm"])
    except Exception as error:
        report["failed_operation"] = {"name": label, "operation": operation, "input_volume_mm3": before_volume,
                                       "error": type(error).__name__+": "+str(error), "runtime_s": perf_counter()-started}
        raise SurfaceReconstructionError(label+" failed: "+str(error), report, solid) from error
    row = {"name": label, "operation": operation, "input_volume_mm3": before_volume,
           "output_volume_mm3": result.volume, "solid_count": len(result.solids()), "valid": bool(result.is_valid),
           "warnings": getattr(result, "boolean_warnings", []), "fuzzy_value_mm": getattr(result, "boolean_fuzzy_value_mm", None), "runtime_s": perf_counter()-started}
    report.setdefault("exact_operations", []).append(row)
    output = _single_solid(result, label, report, solid)
    tolerance = report["settings"]["boolean_tolerance_mm3"]
    volume = output.volume
    passed = before_volume-tolerance <= volume <= before_volume+sum(item.volume for item in operands)+tolerance if operation == "fuse" else tolerance < volume <= before_volume+tolerance
    row["volume_monotonicity_passed"] = passed
    if not passed:
        report["failed_operation"] = row
        raise SurfaceReconstructionError(label+" violated Boolean volume monotonicity", report, solid)
    _progress(progress, label, report, shape=output)
    return output


def _solid_set_status(shape):
    solids = list(shape.solids())
    states = []
    for solid in solids:
        classifier = BRepClass3d_SolidClassifier(solid.wrapped)
        classifier.PerformInfinitePoint(1e-7)
        states.append(classifier.State())
    volumes = [float(solid.volume) for solid in solids]
    return {"valid": bool(shape.is_valid), "solid_count": len(solids), "faces": len(shape.faces()),
            "volume_mm3": float(shape.volume), "component_volumes_mm3": volumes,
            "infinite_point_classifications": [str(state) for state in states],
            "positive_outward_components": bool(solids) and all(volume > 0 for volume in volumes) and all(state == TopAbs_OUT for state in states)}


def _preunion_preserves(preserves, report, progress=None):
    started = perf_counter()
    config = report["settings"]
    tolerance = config["boolean_tolerance_mm3"]
    row = {"status": "running", "input_count": len(preserves), "input_names": [region["name"] for region, _ in preserves],
           "method": "Union of all original prescribed regions before density fusion; every resulting component is retained. Original-preserve coverage uses protected native differences. No density-body diagnostic Boolean is performed.",
           "native_input_policy": "OCCT non-destructive operations; no geometry or tolerance repair. The native outer Compound Free flag may change without changing geometry, as recorded in the independent raw-BREP diagnostic.",
           "operations": [], "coverage": []}
    report["preserve_preunion"] = row
    _progress(progress, "Exact preserve preunion started", report)

    def fail(message, shape=None):
        row.update(status="failed", reason=message, runtime_s=perf_counter()-started)
        report["failed_operation"] = {"name": "Exact preserve preunion", "reason": message}
        raise SurfaceReconstructionError("Exact preserve preunion: "+message, report, shape)

    def native(first, operands, operation, name):
        operation_started = perf_counter()
        item = {"name": name, "operation": operation, "fuzzy_value_mm": config["boolean_fuzzy_value_mm"]}
        row["operations"].append(item)
        try:
            result = _boolean(first, operands, operation, config["boolean_strategy"], config["boolean_fuzzy_value_mm"])
        except Exception as error:
            item.update(runtime_s=perf_counter()-operation_started, error=type(error).__name__+": "+str(error))
            fail(name+" native operation failed: "+str(error), first)
        item.update(runtime_s=perf_counter()-operation_started, warnings=getattr(result, "boolean_warnings", []))
        if result is None or result.wrapped is None or result.wrapped.IsNull() or item["warnings"]:
            fail(name+" has unresolved native warnings or null output", first)
        return result

    if not preserves:
        row.update(status="identity_empty", runtime_s=perf_counter()-started)
        _progress(progress, "Exact preserve preunion", report)
        return None
    inspection_started = perf_counter()
    row["inputs"] = [_solid_set_status(shape) for _, shape in preserves]
    row["input_inspection_s"] = perf_counter()-inspection_started
    if not all(item["valid"] and item["positive_outward_components"] for item in row["inputs"]):
        fail("original preserve requires valid positive outward components")
    original_shapes = [shape for _, shape in preserves]
    canonical = original_shapes[0] if len(preserves) == 1 else native(original_shapes[0], original_shapes[1:], "fuse", "Original preserve union")
    inspection_started = perf_counter()
    row["result"] = _solid_set_status(canonical)
    row["result_inspection_s"] = perf_counter()-inspection_started
    if not row["result"]["valid"] or not row["result"]["positive_outward_components"]:
        fail("union requires valid positive outward components", canonical)
    lower = max(item["volume_mm3"] for item in row["inputs"])-tolerance
    upper = sum(item["volume_mm3"] for item in row["inputs"])+tolerance
    row["volume_monotonicity"] = {"minimum_mm3": lower, "maximum_mm3": upper, "passed": lower <= canonical.volume <= upper}
    if not row["volume_monotonicity"]["passed"]:
        fail("union violated volume monotonicity", canonical)
    coverage_started = perf_counter()
    for index, (region, shape) in enumerate(preserves):
        missing = None if len(preserves) == 1 else native(shape, [canonical], "cut", "Preserve coverage "+region["name"])
        difference = None if missing is None else _solid_set_status(missing)
        coverage = {"name": region["name"], "required_volume_mm3": row["inputs"][index]["volume_mm3"],
                    "missing_volume_mm3": 0.0 if missing is None else float(missing.volume),
                    "valid": True if difference is None else difference["valid"], "difference_topology": difference,
                    "method": "identity" if missing is None else "original preserve minus canonical preserve union"}
        coverage["component_orientation_passed"] = difference is None or difference["solid_count"] == 0 or difference["positive_outward_components"]
        coverage["passed"] = coverage["valid"] and coverage["component_orientation_passed"] and 0 <= coverage["missing_volume_mm3"] <= tolerance
        row["coverage"].append(coverage)
        if not coverage["passed"]:
            fail("original preserve coverage failed for "+region["name"], canonical)
    row.update(status="identity_single" if len(preserves) == 1 else "passed", coverage_s=perf_counter()-coverage_started, runtime_s=perf_counter()-started)
    _progress(progress, "Exact preserve preunion", report, shape=canonical)
    return canonical


def _reconstruct_selected(domain, mesh, report, *, progress=None):
    started = perf_counter()
    sew_started = perf_counter()
    pieces = [mesh] if mesh.body_count == 1 else mesh.split(only_watertight=False)
    sewn = []
    for part, piece in enumerate(pieces):
        try:
            faces = []
            for index, vertices in enumerate(piece.faces):
                faces.append(Face(Wire.make_polygon(piece.vertices[vertices].tolist())))
            sewn.append(_single_solid(Solid(Shell(faces)), "Surface sewing", report))
        except SurfaceReconstructionError:
            raise
        except Exception as error:
            report["failed_operation"] = {"name": "Surface sewing", "part": part, "face": index,
                                           "triangle_mm": piece.vertices[vertices].tolist(), "error": type(error).__name__+": "+str(error)}
            raise SurfaceReconstructionError("Surface sewing failed: "+str(error), report, mesh=mesh) from error
    solid = sewn[0] if len(sewn) == 1 else Compound(children=sewn)
    report["sew_brep_s"] = perf_counter()-sew_started
    report["initial_brep"] = {"valid": bool(solid.is_valid), "solids": len(solid.solids()), "faces": len(solid.faces()), "volume_mm3": solid.volume,
                              "connection_policy": "Only prescribed preserve volumes may join separate density components; final exact CAD must be one solid."}
    _progress(progress, "before_exact_constraints", report, shape=solid, mesh=mesh)
    initial_volume = solid.volume
    preserves = [(region, region_shape(region)) for region in domain["regions"] if region["role"] == "preserve"]
    forbidden = [region_shape(region) for region in domain["regions"] if region["role"] == "forbidden"]
    report["preserve_connection_policy"] = "Prescribed preserve volumes may connect through other prescribed preserves. No preliminary diagnostic Booleans modify the density BRep; the protected exact union and final cut body must each be one valid solid, followed by independent attachment validation."
    operation_started = perf_counter()
    preserve_operands = [shape for _, shape in preserves]
    if report["settings"]["preserve_fusion_mode"] == "preunion":
        canonical = _preunion_preserves(preserves, report, progress)
        preserve_operands = [] if canonical is None else [canonical]
    if preserves:
        solid = _checked_operation(solid, preserve_operands, "fuse", "Exact preserve union", report, progress)
    report["preserve_union"] = {"mode": report["settings"]["preserve_fusion_mode"], "runtime_s": perf_counter()-operation_started, "added_volume_mm3": solid.volume-initial_volume}
    before_cut = solid.volume
    operation_started = perf_counter()
    if forbidden:
        solid = _checked_operation(solid, forbidden, "cut", "Exact forbidden subtraction", report, progress)
    report["forbidden_subtraction"] = {"runtime_s": perf_counter()-operation_started, "removed_volume_mm3": before_cut-solid.volume}
    before_clip = solid.volume
    grid = domain["grid"]
    lower = np.asarray(grid["origin_mm"], dtype=float)
    upper = lower+np.asarray(grid["spacing_mm"])*np.asarray(grid["shape"])
    envelope = region_shape({"kind": "box", "min_mm": lower.tolist(), "max_mm": upper.tolist()})
    solid = _checked_operation(solid, [envelope], "intersect", "Exact design envelope intersection", report, progress)
    report["envelope_intersection"] = {"removed_volume_mm3": before_clip-solid.volume}
    report["final_brep"] = {"valid": bool(solid.is_valid), "solids": len(solid.solids()), "faces": len(solid.faces()), "volume_mm3": solid.volume}
    report["runtime_s"] = perf_counter()-started
    solid.surface_report = report
    return solid


def reconstruct_surface(domain, density, settings, *, progress=None):
    
    started = perf_counter()
    config = _settings(settings)
    attempts = []
    budgets = config["decimation_face_budgets"]
    while budgets:
        try:
            mesh, report = extract_density_surface(domain, density, {**config, "decimation_face_budgets": budgets}, progress=progress)
        except SurfaceReconstructionError as error:
            error.report["reconstruction_attempts"] = deepcopy(attempts)
            raise
        report["requested_settings"] = config
        selected_budget = report["decimation_attempts"][-1]["face_budget"]
        try:
            solid = _reconstruct_selected(domain, mesh, report, progress=progress)
        except SurfaceReconstructionError as error:
            attempts.append(deepcopy(error.report))
            error.report["reconstruction_attempts"] = deepcopy(attempts)
            error.mesh = mesh
            _progress(progress, "reconstruction_failed_"+str(selected_budget), error.report, shape=error.shape, mesh=mesh)
            budgets = [value for value in budgets if value > selected_budget]
            if not budgets or "failed_operation" not in error.report:
                raise
        else:
            report["reconstruction_attempts"] = attempts
            report["runtime_s"] = perf_counter()-started
            return solid
