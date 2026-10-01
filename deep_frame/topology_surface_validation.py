from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
from time import perf_counter
import warnings

import numpy as np
import trimesh
from build123d import Align, Box, Pos, Vector
from build123d import Shape
from OCP.BOPAlgo import BOPAlgo_BOP, BOPAlgo_COMMON, BOPAlgo_CUT
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.Message import Message_Fail, Message_Warning
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_OUT, TopAbs_REVERSED
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from OCP.collections import List_TopoDS_Shape
from rtree import index as rtree_index

from deep_frame.topology_geometry import _compound, _primitive_wall_checks, _trapped_voids, _volume, region_shape

SURFACE_VALIDATION_SETTINGS = {
    "boolean_fuzzy_mm": 1e-7,
    "volume_tolerance_mm3": 1e-5,
    "tessellation_mm": 0.03,
    "tessellation_angle_rad": 0.08,
    "wall_sample_spacing_mm": 1.0,
    "maximum_wall_samples": 300000,
    "wall_tolerance_mm": 1e-5,
    "void_grid_spacing_mm": 1.0,
    "maximum_void_columns": 100000,
    "attachment_probe_depth_mm": 0.02,
    "attachment_offsets_mm": [0.0, 0.5, 1.0],
    "prescribed_surface_tolerance_mm": 0.04,
    "axis_normal_angle_deg": 2.0,
    "sharp_edge_angle_deg": 45.0,
    "source_plane_tolerance_mm": 0.04,
    "maximum_free_crease_ratio": 0.5,
    "maximum_free_axis_fraction_ratio": 0.5,
}

class _UnresolvedCSG(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__("Independent CAD operation is unresolved: " + report["operation"])

class _InvalidNativeMesh(ValueError):
    def __init__(self, reason, triangle_ids=()):
        indices = [int(index) for index in triangle_ids]
        self.report = {"reason": reason, "invalid_triangle_count": len(indices), "invalid_triangle_ids_first_100": indices[:100], "faces_removed": 0}
        super().__init__(reason + ": " + str(indices[:100]))

def _checked_csg(operation_name, first, *others, fuzzy_mm=1e-7):
    started = perf_counter()
    report = {"operation": operation_name, "passed": False, "complete": False, "non_destructive": True}
    try:
        operation = BOPAlgo_BOP()
        arguments, operands = List_TopoDS_Shape(), List_TopoDS_Shape()
        arguments.Append(first.wrapped)
        for other in others:
            operands.Append(other.wrapped)
        operation.SetArguments(arguments)
        operation.SetTools(operands)
        operation.SetOperation({"cut": BOPAlgo_CUT, "common": BOPAlgo_COMMON}[operation_name])
        operation.SetRunParallel(True)
        operation.SetNonDestructive(True)
        operation.SetFuzzyValue(fuzzy_mm)
        report["fuzzy_value_mm"] = float(operation.FuzzyValue())
        operation.Perform()
        report.update(has_errors=bool(operation.HasErrors()), has_warnings=bool(operation.HasWarnings()))
        report["native_alerts"] = {name: [alert.DynamicType().Name() for alert in operation.GetReport().GetAlerts(gravity)] for name, gravity in (("warnings", Message_Warning), ("errors", Message_Fail))}
        if report["has_errors"] or report["has_warnings"]:
            report["reason"] = "Native CAD errors or warnings prevent a reliable independent volume result"
            raise _UnresolvedCSG(report)
        native = operation.Shape()
        if native.IsNull():
            report["reason"] = "Native CAD result is null; it is not a verified empty difference"
            raise _UnresolvedCSG(report)
        result = Shape.cast(native)
        with warnings.catch_warnings(record=True) as cleaning_warnings:
            warnings.simplefilter("always")
            result.clean()
        if cleaning_warnings:
            report["cleanup_warnings"] = [str(item.message) for item in cleaning_warnings]
            report["reason"] = "CAD result cleanup did not complete reliably"
            raise _UnresolvedCSG(report)
        report["valid"] = bool(result.is_valid)
        if not report["valid"]:
            report["reason"] = "Native CAD result is invalid"
            raise _UnresolvedCSG(report)
        return result
    except _UnresolvedCSG:
        raise
    except Exception as error:
        report["reason"] = type(error).__name__ + ": " + str(error)
        raise _UnresolvedCSG(report) from error
    finally:
        report["elapsed_s"] = perf_counter() - started

def _settings(changes):
    settings = deepcopy(SURFACE_VALIDATION_SETTINGS)
    unknown = set(changes) - set(settings)
    if unknown:
        raise ValueError("Unknown surface validation settings: " + ", ".join(sorted(unknown)))
    settings.update(deepcopy(changes))
    for name, value in settings.items():
        values = value if isinstance(value, list) else [value]
        if not values or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not np.isfinite(item) or item < 0 for item in values):
            raise ValueError("Surface settings must be finite nonnegative numbers: " + name)
        if name != "attachment_offsets_mm" and any(item <= 0 for item in values):
            raise ValueError("Surface settings must be positive: " + name)
    for name in ("maximum_wall_samples", "maximum_void_columns"):
        if int(settings[name]) != settings[name]:
            raise ValueError("Sample limits must be integers")
    for name in ("axis_normal_angle_deg", "sharp_edge_angle_deg"):
        if settings[name] >= 90:
            raise ValueError("Surface normal angle must be below 90 degrees")
    for name in ("maximum_free_crease_ratio", "maximum_free_axis_fraction_ratio"):
        if settings[name] > 0.5:
            raise ValueError("Frozen surface reduction gates cannot be weakened")
    return settings

def _absolute_tessellation(shape, tolerance, angle):
    copied = Shape.cast(BRepBuilderAPI_Copy(shape.wrapped, True, False).Shape())
    BRepTools.Clean_s(copied.wrapped, True)
    native_faces = list(copied.faces())
    if any(BRep_Tool.Triangulation_s(face.wrapped, TopLoc_Location()) is not None for face in native_faces):
        raise ValueError("Copied CAD retains a triangulation cache before absolute meshing")
    mesher = BRepMesh_IncrementalMesh(copied.wrapped, tolerance, False, angle, True)
    if not mesher.IsDone():
        raise ValueError("Absolute CAD meshing did not finish: " + str(mesher.GetStatusFlags()))
    vertices, triangles, missing, deflections, face_counts = [], [], [], [], []
    for index, face in enumerate(native_faces):
        location = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face.wrapped, location)
        face_counts.append(0 if poly is None else poly.NbTriangles())
        if poly is None or poly.NbTriangles() == 0:
            missing.append({"face": index, "area_mm2": face.area, "type": str(face.geom_type)})
            continue
        offset = len(vertices)
        transform = location.Transformation()
        vertices.extend(poly.Node(i).Transformed(transform).Coord() for i in range(1, poly.NbNodes() + 1))
        reverse = face.wrapped.Orientation() == TopAbs_REVERSED
        for triangle in poly.Triangles():
            first, second, third = triangle.Get()
            triangles.append((first + offset - 1, third + offset - 1, second + offset - 1) if reverse else (first + offset - 1, second + offset - 1, third + offset - 1))
        deflections.append(float(poly.Deflection()))
    report = {"deflection_mode": "absolute_mm", "occt_is_relative": False,
              "requested_deflection_mm": tolerance, "angular_tolerance_rad": angle,
              "cad_face_count": len(native_faces), "triangulated_cad_face_count": len(native_faces) - len(missing),
              "untriangulated_faces": missing, "native_mesher_status_flags": int(mesher.GetStatusFlags()),
              "native_triangulation_deflection_range_mm": [min(deflections), max(deflections)] if deflections else None,
              "native_face_triangle_counts": face_counts,
              "input_cache_reused": False, "input_shape_modified": False,
              "cache_policy": "Fresh BRepBuilderAPI_Copy with copied geometry and no mesh; polygonal caches cleared only on this copy; native Poly_Triangulation extracted directly without build123d remeshing"}
    return vertices, triangles, report

def _face_tessellation(face, settings):
    failures = []
    for refinement in range(5):
        tolerance = settings["tessellation_mm"] / 10 ** refinement
        try:
            vertices, faces, report = _absolute_tessellation(face, tolerance, settings["tessellation_angle_rad"])
            if vertices and faces and not report["untriangulated_faces"]:
                return vertices, faces, tolerance, failures
            failures.append("empty triangulation")
        except (AttributeError, RuntimeError, ValueError) as error:
            failures.append(type(error).__name__ + ": " + str(error))
    raise ValueError("Final CAD face cannot be triangulated after five explicit local refinements: " + str(failures))

def _native_mesh(vertices, faces, native_face_triangle_counts=None):
    points, indices = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.all(np.isfinite(points)):
        raise _InvalidNativeMesh("Native mesh has empty, malformed or nonfinite coordinates")
    if indices.ndim != 2 or indices.shape[1] != 3 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
        raise _InvalidNativeMesh("Native mesh has empty or malformed triangle indices")
    invalid = np.flatnonzero(np.any((indices < 0) | (indices >= len(points)), axis=1))
    if len(invalid):
        raise _InvalidNativeMesh("Native triangle references an invalid vertex", invalid)
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    identified = inverse[indices]
    canonical = np.sort(identified, axis=1)
    zero_simplices = np.any(np.diff(canonical, axis=1) == 0, axis=1)
    kept = np.flatnonzero(~zero_simplices)
    if not len(kept):
        raise _InvalidNativeMesh("Native mesh contains no positive-area faces after exact repeated-coordinate simplices", np.flatnonzero(zero_simplices))
    if native_face_triangle_counts is not None:
        source_faces = np.repeat(np.arange(len(native_face_triangle_counts)), native_face_triangle_counts)
        if len(source_faces) != len(indices):
            raise _InvalidNativeMesh("Native CAD-face triangle coverage mapping is inconsistent")
        missing = np.setdiff1d(np.arange(len(native_face_triangle_counts)), np.unique(source_faces[kept]))
        if len(missing):
            raise _InvalidNativeMesh("Native CAD faces without positive-area triangle coverage: " + str(missing.tolist()))
    identified, canonical = identified[kept], canonical[kept]
    _, face_inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    invalid = np.flatnonzero(counts[face_inverse] > 1)
    if len(invalid):
        raise _InvalidNativeMesh("Native mesh has duplicate triangles", kept[invalid])
    referenced, compact_indices = np.unique(identified, return_inverse=True)
    unused_vertex_count = len(unique) - len(referenced)
    unique = unique[referenced]
    identified = compact_indices.reshape((-1, 3))
    triangles = unique[identified]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    crossed = np.cross(first, second)
    invalid = np.flatnonzero(~np.all(np.isfinite(crossed), axis=1))
    if len(invalid):
        raise _InvalidNativeMesh("Native triangle cross product is nonfinite", kept[invalid])
    uncertainty = 64 * np.finfo(np.float64).eps * np.max(np.abs(first), axis=1) * np.max(np.abs(second), axis=1)
    exact_indices = np.flatnonzero(np.max(np.abs(crossed), axis=1) <= uncertainty)
    for index in exact_indices:
        coordinates = [[Fraction.from_float(float(value)) for value in point] for point in triangles[index]]
        a, b = [[coordinates[side][axis] - coordinates[0][axis] for axis in range(3)] for side in (1, 2)]
        exact_cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        if not any(exact_cross):
            raise _InvalidNativeMesh("Native triangle is exactly collinear with distinct coordinates", [kept[index]])
        crossed[index] = [float(value) for value in exact_cross]
    scale = np.max(np.abs(crossed), axis=1)
    invalid = np.flatnonzero(~np.isfinite(scale) | (scale <= 0))
    if len(invalid):
        raise _InvalidNativeMesh("Native triangle cross product is unrepresentable", kept[invalid])
    scaled_cross = crossed / scale[:, None]
    lengths = np.linalg.norm(scaled_cross, axis=1)
    areas = lengths * (scale * 0.5)
    invalid = np.flatnonzero(~np.isfinite(areas) | (areas <= 0) | ~np.isfinite(lengths) | (lengths <= 0))
    if len(invalid):
        raise _InvalidNativeMesh("Native triangle has zero or unrepresentable area", kept[invalid])
    normals = np.ascontiguousarray(scaled_cross / lengths[:, None])
    if not np.all(np.isfinite(normals)):
        raise _InvalidNativeMesh("Native triangle has an unrepresentable unit normal")
    mesh = trimesh.Trimesh(vertices=unique, faces=identified, process=False)
    mesh._cache["face_normals"] = normals
    if not np.array_equal(mesh.face_normals, normals) or not np.allclose(np.linalg.norm(mesh.face_normals, axis=1), 1, rtol=0, atol=1e-12):
        raise _InvalidNativeMesh("Native unit normals were not retained by the mesh")
    if not np.array_equal(mesh.triangles, points[indices[kept]]):
        raise _InvalidNativeMesh("Native triangle coordinates changed during exact identification")
    mesh.metadata["native_triangle_preservation"] = {"native_triangle_count": len(indices), "retained_triangle_count": len(mesh.faces), "positive_area_faces_removed": 0, "exact_repeated_coordinate_simplices_removed": int(zero_simplices.sum()), "removed_zero_simplex_native_triangle_ids": np.flatnonzero(zero_simplices).tolist(), "unreferenced_vertices_compacted": unused_vertex_count, "coordinate_movement_mm": 0.0, "vertex_identification": "Exactly equal float64 coordinate triples only; no tolerance welding", "minimum_triangle_area_mm2": float(areas.min()), "exact_cross_product_fallback_triangle_ids": kept[exact_indices].tolist(), "normal_method": "Raw cross products; exact rational cross fallback near arithmetic cancellation; component-scaled normalization without an absolute area cutoff; explicit normal cache verified through mesh.face_normals", "finite_positive_area_faces": True, "unit_normals_verified": True}
    return mesh

def _mesh(solid, settings):
    failures = []
    for refinement in range(5):
        tolerance = settings["tessellation_mm"] / 10 ** refinement
        report = {}
        try:
            vertices, faces, report = _absolute_tessellation(solid, tolerance, settings["tessellation_angle_rad"])
            if report["untriangulated_faces"]:
                raise ValueError("CAD faces without triangle coverage: " + str(report["untriangulated_faces"]))
            mesh = _native_mesh(vertices, faces, report.pop("native_face_triangle_counts", None))
            mesh.metadata["adaptive_tessellation"] = {**report, "maximum_deflection_mm": settings["tessellation_mm"], "used_deflection_mm": tolerance, "failed_coarser_attempts": failures, "cad_geometry_modified": False, "method": "Absolute OCCT whole-shape meshing on a fresh copy preserves shared-edge topology; no independent face stitching or hole filling"}
            return mesh
        except _InvalidNativeMesh:
            raise
        except (AttributeError, RuntimeError, ValueError) as error:
            failures.append({**report, "deflection_mm": tolerance, "error": type(error).__name__ + ": " + str(error)})
    raise ValueError("Final CAD cannot be triangulated after five explicit whole-shape refinements: " + str(failures))

def _prescribed_triangles(triangles, normals, domain, settings):
    fixed = np.zeros(len(triangles), dtype=bool)
    tolerance = settings["prescribed_surface_tolerance_mm"]
    cosine = np.cos(np.deg2rad(settings["axis_normal_angle_deg"]))
    origin = np.asarray(domain["grid"]["origin_mm"], dtype=float)
    upper = origin + np.asarray(domain["grid"]["spacing_mm"]) * np.asarray(domain["grid"]["shape"])
    regions = [*domain["regions"], {"kind": "box", "min_mm": origin, "max_mm": upper}]
    for region in regions:
        if region["kind"] == "box":
            lower, high = np.asarray(region["min_mm"]), np.asarray(region["max_mm"])
            covered = np.all((triangles >= lower - tolerance) & (triangles <= high + tolerance), axis=(1, 2))
            for axis in range(3):
                parallel = np.abs(normals[:, axis]) >= cosine
                for coordinate in (lower[axis], high[axis]):
                    fixed |= covered & parallel & np.all(np.abs(triangles[:, :, axis] - coordinate) <= tolerance, axis=1)
        elif region["kind"] == "cylinder":
            axis = "xyz".index(region.get("axis", "z"))
            radial = [index for index in range(3) if index != axis]
            delta = triangles - np.asarray(region["center_mm"])
            radii = np.linalg.norm(delta[:, :, radial], axis=2)
            axial = np.all(np.abs(delta[:, :, axis]) <= region["height_mm"] / 2 + tolerance, axis=1)
            radial_covered = np.all(radii <= region["radius_mm"] + tolerance, axis=1)
            radial_normal = delta.mean(axis=1)
            radial_normal[:, axis] = 0
            lengths = np.linalg.norm(radial_normal, axis=1)
            radial_normal /= np.maximum(lengths, 1e-15)[:, None]
            fixed |= axial & np.all(np.abs(radii - region["radius_mm"]) <= tolerance, axis=1) & (np.abs(np.sum(normals * radial_normal, axis=1)) >= cosine)
            for side in (-1, 1):
                fixed |= radial_covered & (np.abs(normals[:, axis]) >= cosine) & np.all(np.abs(delta[:, :, axis] - side * region["height_mm"] / 2) <= tolerance, axis=1)
        else:
            raise ValueError("Unsupported prescribed surface primitive")
    return fixed

def _strict_selected_pair_certificates(triangles, selected):
    chosen = np.asarray(selected, dtype=np.int64)
    report = {"passed": False, "complete": False, "maximum_selected_faces": 16, "selected_face_count": len(chosen), "pair_count_required": len(chosen) * (len(chosen) - 1) // 2, "certificates": [], "unresolved_pairs": [], "method": "All selected-face pairs; exact Fraction.from_float binary64 coordinates; strict separating axes from triangle edges in XY/XZ/YZ projections; projected disjointness implies 3D disjointness without a coplanarity assumption", "tolerance": None}
    if len(chosen) < 2 or len(chosen) > report["maximum_selected_faces"]:
        report["reason"] = "Selected-face count outside bounded exact-refinement range"
        return report
    if len(np.unique(chosen)) != len(chosen) or np.any((chosen < 0) | (chosen >= len(triangles))):
        report["reason"] = "Invalid selected-face indices"
        return report
    coordinates = np.asarray(triangles)[chosen]
    if coordinates.shape != (len(chosen), 3, 3) or not np.all(np.isfinite(coordinates)):
        report["reason"] = "Nonfinite or malformed selected triangles"
        return report
    try:
        exact = [[[Fraction.from_float(float(value)) for value in point] for point in triangle] for triangle in coordinates]
        for first, second in combinations(range(len(chosen)), 2):
            certificate = None
            for axes in ((0, 1), (0, 2), (1, 2)):
                a = [[point[axis] for axis in axes] for point in exact[first]]
                b = [[point[axis] for axis in axes] for point in exact[second]]
                for origin in (a, b):
                    for edge in range(3):
                        start, end = origin[edge], origin[(edge + 1) % 3]
                        axis = [start[1] - end[1], end[0] - start[0]]
                        if not any(axis):
                            continue
                        supports = [[sum(coordinate * component for coordinate, component in zip(point, axis)) for point in triangle] for triangle in (a, b)]
                        intervals = [(min(values), max(values)) for values in supports]
                        if intervals[0][1] < intervals[1][0] or intervals[1][1] < intervals[0][0]:
                            certificate = {"faces": [int(chosen[first]), int(chosen[second])], "projection_axes": list(axes), "axis_exact": [str(value) for value in axis], "support_intervals_exact": [[str(value) for value in interval] for interval in intervals], "strictly_disjoint": True}
                            break
                    if certificate is not None:
                        break
                if certificate is not None:
                    break
            if certificate is None:
                report["unresolved_pairs"].append([int(chosen[first]), int(chosen[second])])
            else:
                report["certificates"].append(certificate)
        report["complete"] = True
        report["passed"] = not report["unresolved_pairs"] and len(report["certificates"]) == report["pair_count_required"]
        if not report["passed"]:
            report["reason"] = "At least one selected pair lacks a strict exact separation certificate; contact and intersection remain rejected"
    except Exception as error:
        report["reason"] = type(error).__name__ + ": " + str(error)
    return report

def _self_intersection_screen(mesh):
    started = perf_counter()
    vertices = np.ascontiguousarray(mesh.vertices, dtype="<f8")
    faces = np.ascontiguousarray(mesh.faces, dtype="<i8")
    vertex_hash = sha256(vertices.tobytes()).hexdigest()
    face_hash = sha256(faces.tobytes()).hexdigest()
    report = {"passed": False, "complete": False, "mesh_vertex_count": len(vertices), "mesh_face_count": len(faces),
              "mesh_vertices_float64_le_sha256": vertex_hash, "mesh_faces_int64_le_sha256": face_hash,
              "method": "MeshLab compute_selection_by_self_intersections_per_face on all retained positive native CAD triangles; only explicitly logged exact repeated-coordinate zero simplices may be absent; no positive triangle exclusion, repair, smoothing or remeshing",
              "limitations": "Self-intersection screen of the CAD tessellation at its separately logged resolution, not an analytic proof for all continuous CAD surfaces"}
    try:
        import pymeshlab
        report["pymeshlab_version"] = version("pymeshlab")
        native = pymeshlab.MeshSet()
        native.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces), "final_cad_tessellation")
        native.compute_selection_by_self_intersections_per_face()
        selected = np.flatnonzero(native.current_mesh().face_selection_array())
        native_vertex_hash = sha256(np.ascontiguousarray(native.current_mesh().vertex_matrix(), dtype="<f8").tobytes()).hexdigest()
        native_face_hash = sha256(np.ascontiguousarray(native.current_mesh().face_matrix(), dtype="<i8").tobytes()).hexdigest()
        report.update(complete=True, self_intersecting_face_count=int(len(selected)),
                      self_intersecting_faces_first_100=selected[:100].tolist(),
                      native_mesh_unchanged=vertex_hash == native_vertex_hash and face_hash == native_face_hash,
                      input_mesh_unchanged=vertex_hash == sha256(vertices.tobytes()).hexdigest() and face_hash == sha256(faces.tobytes()).hexdigest())
        if len(selected) and report["native_mesh_unchanged"]:
            triangles = mesh.triangles[selected]
            report["self_intersection_region_bounds_mm"] = [triangles.min(axis=(0, 1)).tolist(), triangles.max(axis=(0, 1)).tolist()]
        report["passed"] = not len(selected) and report["input_mesh_unchanged"] and report["native_mesh_unchanged"]
        if len(selected) and report["native_mesh_unchanged"] and report["input_mesh_unchanged"]:
            coverage = {"pymeshlab_version": "2025.7.post1", "pymeshlab_commit": "1dc199f9b6c43e58b6db346ba4600866b950b8ae", "meshlab_commit": "d876376e3cc4f92d257e248023d82cbac5b03c7d", "vcglib_commit": "c94ef4e12e9ea3ae986d9af91005be8328d13719", "selection_contract": "Pinned VCGLib SelfIntersections appends both faces of every reported pair; MeshLab selects every returned face. Every possible selected-face pair must be strictly certified disjoint."}
            report["exact_refinement_selection_coverage"] = coverage
            if report["pymeshlab_version"] == coverage["pymeshlab_version"]:
                report["exact_selected_pair_refinement"] = _strict_selected_pair_certificates(mesh.triangles, selected)
            else:
                report["exact_selected_pair_refinement"] = {"passed": False, "complete": False, "reason": "Installed MeshLab version lacks the pinned both-faces selection proof"}
            report["passed"] = report["exact_selected_pair_refinement"]["passed"]
    except Exception as error:
        report["reason"] = type(error).__name__ + ": " + str(error)
    report["elapsed_s"] = perf_counter() - started
    return report

def surface_metrics(mesh, domain, settings=None):
    settings = _settings(settings or {})
    fixed = _prescribed_triangles(mesh.triangles, mesh.face_normals, domain, settings)
    free = ~fixed
    area = float(mesh.area_faces[free].sum())
    axis = np.argmax(np.abs(mesh.face_normals), axis=1)
    aligned = np.max(np.abs(mesh.face_normals), axis=1) >= np.cos(np.deg2rad(settings["axis_normal_angle_deg"]))
    distances = np.abs((mesh.triangles_center - np.asarray(domain["grid"]["origin_mm"])) / np.asarray(domain["grid"]["spacing_mm"]))
    distances = np.abs(distances - np.round(distances)) * np.asarray(domain["grid"]["spacing_mm"])
    on_grid = distances[np.arange(len(axis)), axis] <= settings["source_plane_tolerance_mm"]
    adjacency_free = np.all(free[mesh.face_adjacency], axis=1)
    sharp = mesh.face_adjacency_angles >= np.deg2rad(settings["sharp_edge_angle_deg"])
    edges = mesh.vertices[mesh.face_adjacency_edges]
    crease = float(np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)[adjacency_free & sharp].sum())
    axis_area = float(mesh.area_faces[free & aligned].sum())
    grid_area = float(mesh.area_faces[free & aligned & on_grid].sum())
    return {
        "surface_area_mm2": float(mesh.area), "prescribed_surface_area_mm2": float(mesh.area_faces[fixed].sum()),
        "free_surface_area_mm2": area, "free_sharp_edge_length_mm": crease,
        "free_sharp_edge_length_per_area_per_mm": crease / area if area else None,
        "free_axis_normal_area_mm2": axis_area, "free_axis_normal_area_fraction": axis_area / area if area else None,
        "free_source_grid_plane_area_mm2": grid_area, "free_source_grid_plane_area_fraction": grid_area / area if area else None,
        "triangle_count": len(mesh.faces), "prescribed_triangle_count": int(fixed.sum()),
        "classification": "All three triangle vertices must lie on and within one finite prescribed primitive boundary; triangle normal must match it. Partial triangles remain free, conservatively.",
        "settings": {key: settings[key] for key in ("prescribed_surface_tolerance_mm", "axis_normal_angle_deg", "sharp_edge_angle_deg", "source_plane_tolerance_mm", "tessellation_mm", "tessellation_angle_rad")},
    }

def _triangle_samples(triangle, spacing):
    pending = [triangle]
    while pending:
        item = pending.pop()
        lengths = np.linalg.norm(item - np.roll(item, -1, axis=0), axis=1)
        longest = int(np.argmax(lengths))
        if lengths[longest] <= spacing:
            yield item.mean(axis=0)
        else:
            following, other = (longest + 1) % 3, (longest + 2) % 3
            middle = (item[longest] + item[following]) / 2
            pending.append(np.array([item[longest], middle, item[other]]))
            pending.append(np.array([middle, item[following], item[other]]))

class _CadRayIndex:
    def __init__(self, solid):
        started = perf_counter()
        self.faces = list(solid.faces())
        self.bounds = []
        self.tolerance = 1e-7
        for face in self.faces:
            bounds = Bnd_Box()
            BRepBndLib.AddOptimal_s(face.wrapped, bounds, False, True)
            if bounds.IsVoid() or bounds.IsOpen():
                raise ValueError("CAD ray index requires finite, nonempty face bounds")
            self.bounds.append((*bounds.CornerMin().Coord(), *bounds.CornerMax().Coord()))
        self.bounds = np.asarray(self.bounds)
        properties = rtree_index.Property()
        properties.dimension = 3
        self.tree = rtree_index.Index(((i, bounds, None) for i, bounds in enumerate(self.bounds)), properties=properties)
        self.intersectors = {}
        self.query_count = self.box_candidates = self.exact_face_queries = 0
        self.build_elapsed_s = perf_counter() - started
    def intersections(self, point, direction, lower, upper, *, nearest=True, with_faces=False):
        point, direction = np.asarray(tuple(point)), np.asarray(tuple(direction))
        direction = direction / np.linalg.norm(direction)
        endpoints = point + np.asarray([lower, upper])[:, None] * direction
        bounds = np.concatenate((endpoints.min(axis=0) - self.tolerance, endpoints.max(axis=0) + self.tolerance))
        candidates = np.asarray(list(self.tree.intersection(bounds)), dtype=int)
        self.query_count += 1
        self.box_candidates += len(candidates)
        if not len(candidates):
            return []
        boxes = self.bounds[candidates]
        enter, leave = np.full(len(boxes), lower), np.full(len(boxes), upper)
        possible = np.ones(len(boxes), dtype=bool)
        for axis in range(3):
            low, high = boxes[:, axis] - self.tolerance, boxes[:, axis + 3] + self.tolerance
            if direction[axis] == 0:
                possible &= (point[axis] >= low) & (point[axis] <= high)
            else:
                first, second = (low - point[axis]) / direction[axis], (high - point[axis]) / direction[axis]
                enter = np.maximum(enter, np.minimum(first, second))
                leave = np.minimum(leave, np.maximum(first, second))
        candidates = candidates[possible & (enter <= leave)]
        line = gp_Lin(gp_Pnt(*point), gp_Dir(*direction))
        hits = []
        for candidate in candidates:
            candidate = int(candidate)
            intersector = self.intersectors.get(candidate)
            if intersector is None:
                intersector = IntCurvesFace_ShapeIntersector()
                intersector.Load(self.faces[candidate].wrapped, self.tolerance)
                self.intersectors[candidate] = intersector
            self.exact_face_queries += 1
            if nearest:
                intersector.PerformNearest(line, lower, upper)
            else:
                intersector.Perform(line, lower, upper)
            if not intersector.IsDone():
                raise RuntimeError("Exact CAD intersection did not complete for face " + str(candidate))
            for i in range(1, intersector.NbPnt() + 1):
                value = intersector.WParameter(i)
                if lower < value <= upper:
                    hits.append((candidate, value) if with_faces else value)
        return hits
    def origin_is_resolved(self, point, direction, source_face):
        hits = self.intersections(point, direction, -self.tolerance, self.tolerance, nearest=False, with_faces=True)
        source = []
        for face, parameter in hits:
            if face != source_face:
                return False
            if not any(abs(parameter - other) <= 1e-12 for other in source):
                source.append(parameter)
        return len(source) == 1
    def report(self):
        return {"cad_face_count": len(self.faces), "index_build_elapsed_s": self.build_elapsed_s,
                "query_count": self.query_count, "segment_aabb_candidates": self.box_candidates,
                "exact_face_queries_after_ray_box_cull": self.exact_face_queries,
                "loaded_face_intersectors": len(self.intersectors), "intersection_tolerance_mm": self.tolerance,
                "method": "Rtree of original CAD face bounds computed without triangulation and expanded by CAD tolerances; exact ray-box cull; OCP intersections on every surviving original CAD face"}

def _wall_screen(solid, minimum, settings, progress=None):
    started = perf_counter()
    ray_index = _CadRayIndex(solid)
    maximum = minimum + settings["wall_tolerance_mm"]
    count, minimum_measured, thin_count, clear_count, unresolved_count = 0, None, 0, 0, 0
    completed_faces = 0
    tessellation_levels = {}
    tessellation_failures = 0
    unresolved, thin, thin_faces = [], [], {}
    minimum_sample = None
    def report(complete, reason=None):
        result = {"passed": complete and count > 0 and not unresolved_count and thin_count == 0,
                  "complete": complete, "minimum_required_mm": minimum, "minimum_measured_mm": minimum_measured,
                  "minimum_screened_thickness_lower_bound_mm": min(maximum, minimum_measured) if minimum_measured is not None else (maximum if count > unresolved_count else None),
                  "minimum_measured_scope": "Only exact intersections within the bounded query segment; absence of a hit establishes a lower bound, not a measured global minimum",
                  "query_upper_bound_mm": maximum, "ray_count": count, "rays_clear_through_upper_bound": clear_count,
                  "ray_origin_exclusion_mm": ray_index.tolerance,
                  "origin_guard": "For every otherwise passing ray, exact intersections in +/-CAD tolerance must contain exactly one source-face crossing and no other face; ambiguous micro-slivers or boundary origins fail unresolved",
                  "completed_cad_face_count": completed_faces, "total_cad_face_count": len(ray_index.faces),
                  "thin_sample_count": thin_count, "thin_samples": thin, "unresolved_sample_count": unresolved_count,
                  "minimum_sample": minimum_sample, "thin_faces": list(thin_faces.values()),
                  "unresolved_samples": unresolved, "maximum_sample_triangle_edge_mm": settings["wall_sample_spacing_mm"],
                  "tessellation": {"deflection_mode": "absolute_mm", "occt_is_relative": False, "maximum_requested_deflection_mm": settings["tessellation_mm"], "angular_tolerance_rad": settings["tessellation_angle_rad"], "used_deflection_face_counts": dict(tessellation_levels), "failed_coarser_attempt_count": tessellation_failures, "input_cache_reused": False, "input_shape_modified": False},
                  "elapsed_s": perf_counter() - started, "spatial_index": ray_index.report(),
                  "method": "Final CAD face tessellation subdivided until all sample triangle edges meet the spacing limit; centroids projected back onto their CAD face; exact inward normal intersections through required thickness plus tolerance",
                  "limitations": "Finite normal-chord screen, not a mathematical global minimum-thickness proof or print-process certification; sharp convex wedges can be conservatively rejected"}
        if reason:
            result["reason"] = reason
        return result
    for face_index, face in enumerate(ray_index.faces):
        vertices, triangles, tolerance, failures = _face_tessellation(face, settings)
        level = str(tolerance)
        tessellation_levels[level] = tessellation_levels.get(level, 0) + 1
        tessellation_failures += len(failures)
        array = np.asarray([tuple(point) for point in vertices])
        for triangle in triangles:
            points = array[list(triangle)]
            if np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) <= 1e-14:
                continue
            for position in _triangle_samples(points, settings["wall_sample_spacing_mm"]):
                if count >= settings["maximum_wall_samples"]:
                    return report(False, "wall sample budget exceeded")
                count += 1
                point = face.closest_points(Vector(*position))[0]
                direction = -face.normal_at(point)
                try:
                    hits = ray_index.intersections(point, direction, ray_index.tolerance, maximum)
                    if (not hits or min(hits) >= minimum - settings["wall_tolerance_mm"]) and not ray_index.origin_is_resolved(point, direction, face_index):
                        raise RuntimeError("Ray origin is ambiguous within CAD intersection tolerance")
                except RuntimeError as error:
                    unresolved_count += 1
                    if len(unresolved) < 30:
                        unresolved.append({"face": face_index, "position_mm": list(point), "reason": str(error)})
                    continue
                if not hits:
                    clear_count += 1
                else:
                    thickness = min(hits)
                    if minimum_measured is None or thickness < minimum_measured:
                        minimum_sample = {"face": face_index, "position_mm": list(point), "thickness_mm": thickness}
                    minimum_measured = thickness if minimum_measured is None else min(minimum_measured, thickness)
                    if thickness < minimum - settings["wall_tolerance_mm"]:
                        thin_count += 1
                        item = thin_faces.setdefault(face_index, {"face": face_index, "sample_count": 0, "minimum_mm": thickness, "position_mm": list(point)})
                        item["sample_count"] += 1
                        if thickness < item["minimum_mm"]:
                            item.update(minimum_mm=thickness, position_mm=list(point))
                        if len(thin) < 30:
                            thin.append({"face": face_index, "position_mm": list(point), "thickness_mm": thickness})
                if progress is not None and (count % 1000 == 0 or thin_count == 1 and len(thin) == 1):
                    progress({"stage": "wall", "ray_count": count, "face_index": face_index, "total_cad_face_count": len(ray_index.faces), "minimum_measured_mm": minimum_measured, "minimum_sample": minimum_sample, "thin_sample_count": thin_count, "thin_samples": thin, "elapsed_s": perf_counter() - started})
        completed_faces += 1
    return report(True)

def _accessibility(solid, domain, settings, progress=None):
    started = perf_counter()
    lower = np.asarray(domain["grid"]["origin_mm"], dtype=float)
    upper = lower + np.asarray(domain["grid"]["spacing_mm"]) * np.asarray(domain["grid"]["shape"])
    spacing = settings["void_grid_spacing_mm"]
    shape = np.ceil((upper - lower) / spacing).astype(int)
    if shape[0] * shape[1] > settings["maximum_void_columns"]:
        return {"passed": False, "reason": "void column budget exceeded"}
    occupied = np.zeros(shape, dtype=bool)
    ray_index = _CadRayIndex(solid)
    zvalues = lower[2] + (np.arange(shape[2]) + 0.5) * spacing
    unresolved = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            start = [lower[0] + (i + 0.5) * spacing, lower[1] + (j + 0.5) * spacing, lower[2] - spacing]
            try:
                hits = sorted(value + start[2] for value in ray_index.intersections(start, (0, 0, 1), 0, upper[2] - lower[2] + 2 * spacing, nearest=False))
            except RuntimeError:
                unresolved += 1
                continue
            unique = [value for index, value in enumerate(hits) if index == 0 or value - hits[index - 1] > 1e-6]
            if len(unique) % 2:
                unresolved += 1
                continue
            for bottom, top in zip(unique[::2], unique[1::2]):
                occupied[i, j] |= (zvalues > bottom) & (zvalues < top)
        if progress is not None and (i + 1) % 10 == 0:
            progress({"stage": "accessibility", "column_count": (i + 1) * int(shape[1]), "unresolved_columns": unresolved, "elapsed_s": perf_counter() - started})
    trapped = _trapped_voids(occupied)
    cavities = max(0, len(solid.shells()) - 1)
    return {"passed": not trapped and not cavities and not unresolved, "closed_cad_cavities": cavities, "trapped_void_cells": trapped, "unresolved_columns": unresolved, "grid_spacing_mm": spacing, "grid_shape": shape.tolist(), "occupied_cells": int(occupied.sum()), "spatial_index": ray_index.report(), "elapsed_s": perf_counter() - started, "method": "Fresh final-CAD exact vertical intersection pairs rasterized at cell centers, then six-connected exterior void flood fill plus closed CAD shell count", "limitations": "Finite geometric access screen; passages narrower than the grid may close numerically. Actual support generation, tool reach and support removal require slicer/physical review."}

def _intersection_volume(first, second, fuzzy_mm=1e-7):
    if first is None or second is None:
        return 0.0
    a, b = first.bounding_box(), second.bounding_box()
    if np.any(np.minimum(tuple(a.max), tuple(b.max)) <= np.maximum(tuple(a.min), tuple(b.min))):
        return 0.0
    return float(_volume(_checked_csg("common", first, second, fuzzy_mm=fuzzy_mm)))

def _preserve_neighborhood(solid, outer, settings):
    lower, upper = np.asarray(tuple(outer.bounding_box().min)), np.asarray(tuple(outer.bounding_box().max))
    padding = max(settings["attachment_offsets_mm"]) + settings["attachment_probe_depth_mm"]
    guard = 1e-6
    crop = Pos(*(lower - padding - guard)) * Box(*(upper - lower + 2 * (padding + guard)), align=(Align.MIN, Align.MIN, Align.MIN))
    local = _compound(_checked_csg("common", solid, crop, fuzzy_mm=settings["boolean_fuzzy_mm"]))
    return local, {"method": "Exact CAD intersection with preserve AABB expanded in every direction to contain every attachment slab", "padding_mm": padding, "inclusion_guard_mm": guard, "crop_min_mm": (lower - padding - guard).tolist(), "crop_max_mm": (upper + padding + guard).tolist(), "required_min_mm": (lower - padding).tolist(), "required_max_mm": (upper + padding).tolist(), "all_slabs_contained": True}

def _attachment_screen(solid, outer, region, manufacturing, settings):
    lower, upper = np.asarray(tuple(outer.bounding_box().min)), np.asarray(tuple(outer.bounding_box().max))
    levels = []
    for offset in settings["attachment_offsets_mm"]:
        sections = {}
        for axis in range(3):
            for side in (-1, 1):
                start = lower.copy()
                extent = upper - lower
                depth = settings["attachment_probe_depth_mm"]
                start[axis] = lower[axis] - offset - depth if side == -1 else upper[axis] + offset
                extent[axis] = depth
                slab = Pos(*start) * Box(*extent, align=(Align.MIN, Align.MIN, Align.MIN))
                sections[f"{'xyz'[axis]}{'-' if side < 0 else '+'}"] = _intersection_volume(solid, slab, fuzzy_mm=settings["boolean_fuzzy_mm"]) / depth
        levels.append({"offset_mm": offset, "sections_mm2": sections, "summed_area_mm2": sum(sections.values())})
    minimum = region.get("attachment_area_min_mm2", manufacturing["minimum_attachment_area_mm2"])
    return {"passed": bool(levels) and all(level["summed_area_mm2"] + 1e-6 >= minimum for level in levels), "minimum_required_area_mm2": minimum, "levels": levels, "method": "Summed mean areas in six slabs outside the exact preserve AABB at all prescribed offsets", "limitations": "Local attachment/near-junction screen; not a global minimum load-path cross-section proof. Independent FEA remains required."}

def _validate_surface(solid, domain: dict, settings: dict, *, reference_solid=None, progress=None) -> dict:
    settings = _settings(settings)
    tolerance = settings["volume_tolerance_mm3"]
    fuzzy_mm = settings["boolean_fuzzy_mm"]
    checks, violations = {}, []
    if progress is not None:
        progress({"stage": "topology", "status": "started"})
    try:
        mesh = _mesh(solid, settings)
    except ValueError as error:
        return {"passed": False, "violations": ["topology:surface_tessellation_failed"], "checks": {"topology": {"passed": False, "reason": str(error), "native_mesh_diagnostic": getattr(error, "report", None)}}, "settings": settings}
    infinite_states = []
    for item in solid.solids():
        classifier = BRepClass3d_SolidClassifier(item.wrapped)
        classifier.PerformInfinitePoint(1e-7)
        infinite_states.append(classifier.State())
    topology = {"solid_count": len(solid.solids()), "valid": bool(solid.is_valid), "closed": all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in solid.shells()), "mesh_watertight": bool(mesh.is_watertight), "mesh_winding_consistent": bool(mesh.is_winding_consistent), "mesh_body_count": int(mesh.body_count), "volume_mm3": float(solid.volume), "mesh_signed_volume_mm3": float(mesh.volume), "infinite_point_classifications": [str(state) for state in infinite_states], "outward_oriented": bool(infinite_states) and all(state == TopAbs_OUT for state in infinite_states)}
    topology["passed"] = topology["solid_count"] == topology["mesh_body_count"] == 1 and topology["valid"] and topology["closed"] and topology["mesh_watertight"] and topology["mesh_winding_consistent"] and topology["volume_mm3"] > tolerance and topology["mesh_signed_volume_mm3"] > tolerance and topology["outward_oriented"]
    topology["tessellation"] = mesh.metadata["adaptive_tessellation"]
    topology["native_triangle_preservation"] = mesh.metadata["native_triangle_preservation"]
    checks["topology"] = topology
    if progress is not None:
        progress({"stage": "topology", "status": "completed", "result": topology})
    if not topology["passed"]:
        return {"passed": False, "violations": ["topology:closed_manifold_single_solid"], "checks": checks, "settings": settings}
    if progress is not None:
        progress({"stage": "self_intersections", "status": "started"})
    checks["self_intersections"] = _self_intersection_screen(mesh)
    if progress is not None:
        progress({"stage": "self_intersections", "status": "completed", "result": checks["self_intersections"]})
    if not checks["self_intersections"]["passed"]:
        return {"passed": False, "violations": ["self_intersections"], "checks": checks, "settings": settings}
    lower = np.asarray(domain["grid"]["origin_mm"])
    upper = lower + np.asarray(domain["grid"]["spacing_mm"]) * np.asarray(domain["grid"]["shape"])
    box = region_shape({"kind": "box", "min_mm": lower, "max_mm": upper})
    if progress is not None:
        progress({"stage": "envelope", "status": "started"})
    outside = float(_volume(_checked_csg("cut", solid, box, fuzzy_mm=fuzzy_mm)))
    allowed = [region_shape(region) for region in domain["regions"] if region["role"] == "allowed"]
    remaining = solid
    for item in allowed:
        remaining = _compound(_checked_csg("cut", remaining, item, fuzzy_mm=fuzzy_mm)) if remaining is not None else None
    outside_allowed = float(_volume(remaining)) if allowed else outside
    checks["envelope"] = {"outside_grid_box_mm3": outside, "outside_allowed_regions_mm3": outside_allowed, "passed": max(outside, outside_allowed) <= tolerance}
    if progress is not None:
        progress({"stage": "envelope", "status": "completed", "result": checks["envelope"]})
    forbidden = [region for region in domain["regions"] if region["role"] == "forbidden"]
    cuts = [region_shape(region) for region in forbidden]
    checks["forbidden"] = {}
    for region, cut in zip(forbidden, cuts):
        if progress is not None:
            progress({"stage": "forbidden", "status": "started", "name": region["name"]})
        volume = _intersection_volume(solid, cut, fuzzy_mm=fuzzy_mm)
        checks["forbidden"][region["name"]] = {"intersection_mm3": volume, "passed": volume <= tolerance}
        if progress is not None:
            progress({"stage": "forbidden", "status": "completed", "name": region["name"], "result": checks["forbidden"][region["name"]]})
    manufacturing = domain["manufacturing"]
    minimum = max(manufacturing["nozzle_width_mm"] * manufacturing["minimum_wall_nozzles"], manufacturing["minimum_feature_mm"])
    checks["preserve"] = {}
    for region in domain["regions"]:
        if region["role"] != "preserve":
            continue
        if progress is not None:
            progress({"stage": "preserve", "status": "started", "name": region["name"]})
        outer = region_shape(region)
        expected = _checked_csg("cut", outer, *cuts, fuzzy_mm=fuzzy_mm) if cuts else outer
        local, neighborhood = _preserve_neighborhood(solid, outer, settings)
        missing = float(_volume(_checked_csg("cut", expected, local, fuzzy_mm=fuzzy_mm))) if local is not None else float(expected.volume)
        walls = _primitive_wall_checks(region, forbidden, max(minimum, region.get("minimum_wall_mm", minimum)))
        attachment = _attachment_screen(local, outer, region, manufacturing, settings)
        checks["preserve"][region["name"]] = {"required_volume_mm3": float(expected.volume), "missing_volume_mm3": missing, "walls": walls, "attachment": attachment, "exact_local_neighborhood": neighborhood, "passed": float(expected.volume) > tolerance and missing <= tolerance and walls["passed"] and attachment["passed"]}
        if progress is not None:
            progress({"stage": "preserve", "status": "completed", "name": region["name"], "result": checks["preserve"][region["name"]]})
    if progress is not None:
        progress({"stage": "wall", "status": "started"})
    checks["features"] = _wall_screen(solid, minimum, settings, progress)
    if progress is not None:
        progress({"stage": "wall", "status": "completed", "result": checks["features"]})
        progress({"stage": "accessibility", "status": "started"})
    accessibility = _accessibility(solid, domain, settings, progress)
    if progress is not None:
        progress({"stage": "accessibility", "status": "completed", "result": accessibility})
    needed = bool(np.any((mesh.face_normals[:, 2] < -0.1) & (mesh.triangles_center[:, 2] > mesh.bounds[0, 2] + 1e-6)))
    checks["supports"] = {"supports_required": needed, "build_direction": manufacturing["build_direction"], "supports_allowed": bool(manufacturing["supports_allowed"]), "accessibility": accessibility, "passed": manufacturing["build_direction"] == [0, 0, 1] and (manufacturing["supports_allowed"] or not needed) and accessibility["passed"]}
    metrics = surface_metrics(mesh, domain, settings)
    change = None
    if reference_solid is not None:
        if progress is not None:
            progress({"stage": "surface_maturity", "status": "started"})
        reference_settings = {**settings, "tessellation_mm": mesh.metadata["adaptive_tessellation"]["used_deflection_mm"]}
        reference_mesh = _mesh(reference_solid, reference_settings)
        reference = surface_metrics(reference_mesh, domain, settings)
        reference["tessellation"] = reference_mesh.metadata["adaptive_tessellation"]
        ratios = {}
        for key, limit_name in (("free_sharp_edge_length_per_area_per_mm", "maximum_free_crease_ratio"), ("free_axis_normal_area_fraction", "maximum_free_axis_fraction_ratio")):
            candidate_value, reference_value = metrics[key], reference[key]
            ratio = candidate_value / reference_value if candidate_value is not None and reference_value is not None and reference_value > 0 else None
            ratios[key] = {"candidate": candidate_value, "reference": reference_value, "ratio": ratio, "maximum_ratio": settings[limit_name], "passed": ratio is not None and ratio <= settings[limit_name]}
        checks["surface_maturity"] = {"passed": all(value["passed"] for value in ratios.values()), "ratios": ratios, "reference_metrics": reference}
        if progress is not None:
            progress({"stage": "surface_maturity", "status": "completed", "result": checks["surface_maturity"]})
            progress({"stage": "material_difference", "status": "started"})
        added = float(_volume(_checked_csg("cut", solid, reference_solid, fuzzy_mm=fuzzy_mm)))
        removed = float(_volume(_checked_csg("cut", reference_solid, solid, fuzzy_mm=fuzzy_mm)))
        change = {"added_volume_mm3": added, "removed_volume_mm3": removed, "symmetric_difference_volume_mm3": added + removed, "net_volume_change_mm3": float(solid.volume - reference_solid.volume), "volume_balance_residual_mm3": float(added - removed - (solid.volume - reference_solid.volume)), "method": "Exact CAD CSG difference in both directions; no equal-mass assumption"}
        if progress is not None:
            progress({"stage": "material_difference", "status": "completed", "result": change})
    else:
        checks["surface_maturity"] = {"passed": False, "reason": "No reference geometry supplied; geometry screen alone cannot establish maturity"}
    for key, value in checks.items():
        if key in ("forbidden", "preserve"):
            violations.extend(key + ":" + name for name, item in value.items() if not item["passed"])
        elif not value["passed"]:
            violations.append(key)
    return {"passed": not violations, "violations": violations, "checks": checks, "surface_metrics": metrics, "material_change": change, "settings": settings, "acceptance_scope": "Geometry screens only. Independent mechanical verification and final actual-geometry render/section review remain mandatory."}

def validate_surface(solid, domain: dict, settings: dict, *, reference_solid=None, progress=None) -> dict:
    settings = _settings(settings)
    try:
        result = _validate_surface(solid, domain, settings, reference_solid=reference_solid, progress=progress)
    except _UnresolvedCSG as error:
        if progress is not None:
            progress({"stage": "csg", "status": "unresolved", "result": error.report})
        result = {"passed": False, "violations": ["csg:unresolved"], "checks": {"csg": error.report}, "settings": settings, "acceptance_scope": "Unreliable CAD operation prevents geometry acceptance; no volume result is inferred from its output."}
    result["numerical_method"] = {"boolean_fuzzy_mm": settings["boolean_fuzzy_mm"], "default_boolean_fuzzy_mm": SURFACE_VALIDATION_SETTINGS["boolean_fuzzy_mm"], "method": "Explicit OCCT Boolean numerical tolerance; separate from physical volume, wall, attachment, surface and sampling criteria", "native_warning_policy": "Every native warning or error is unresolved and prevents geometry acceptance", "input_geometry_protection": "NonDestructive=True for every Boolean operation"}
    return result
