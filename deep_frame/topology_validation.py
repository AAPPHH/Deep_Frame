from itertools import combinations

import numpy as np
import trimesh
from build123d import Align, Box, Pos, Vector
from OCP.BRep import BRep_Tool
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from scipy.ndimage import generate_binary_structure, label

from deep_frame.topology_geometry import _compound, _volume, region_shape


def _attachment_sections(solid, outer, depth):
    bounds = outer.bounding_box()
    lower = np.array(tuple(bounds.min))
    upper = np.array(tuple(bounds.max))
    sections = {}
    for axis in range(3):
        for side in (-1, 1):
            start = lower.copy()
            extent = upper - lower
            start[axis] = lower[axis] - depth if side == -1 else upper[axis]
            extent[axis] = depth
            slab = Pos(*start) * Box(*extent, align=(Align.MIN, Align.MIN, Align.MIN))
            sections[f"{'xyz'[axis]}{'-' if side == -1 else '+'}"] = _volume(solid.intersect(slab)) / depth
    return sections


def _primitive_wall_checks(region, forbidden, minimum):
    checks = {}
    outer = region_shape(region)
    holes = [hole for hole in forbidden if hole["kind"] == "cylinder" and hole.get("rasterize", True) is False and _volume(outer.intersect(region_shape(hole))) > 1e-7]
    if region["kind"] == "box":
        low = np.array(region["min_mm"])
        high = np.array(region["max_mm"])
        checks["outer_minimum_dimension"] = float(np.min(high - low))
        for hole in holes:
            axis = "xyz".index(hole["axis"])
            center = np.array(hole["center_mm"])
            margin = [min(center[i] - low[i], high[i] - center[i]) - hole["radius_mm"] for i in range(3) if i != axis]
            if min(margin) >= -1e-7:
                checks["bore_to_outer:" + hole["name"]] = float(min(margin))
        for slot in forbidden:
            if slot["kind"] != "box" or slot.get("rasterize", True) or _volume(outer.intersect(region_shape(slot))) < 1e-7:
                continue
            for axis in range(3):
                margins = [slot["min_mm"][axis] - low[axis], high[axis] - slot["max_mm"][axis]]
                residual = [float(value) for value in margins if value > 1e-7]
                if residual:
                    checks[f"slot_rim:{slot['name']}:{'xyz'[axis]}"] = min(residual)
    else:
        checks["outer_minimum_dimension"] = float(min(region["height_mm"], 2 * region["radius_mm"]))
        for hole in holes:
            if hole["axis"] == region["axis"]:
                axis = "xyz".index(hole["axis"])
                offset = np.array(hole["center_mm"]) - np.array(region["center_mm"])
                offset[axis] = 0
                checks["bore_to_outer:" + hole["name"]] = float(region["radius_mm"] - np.linalg.norm(offset) - hole["radius_mm"])
    for first, second in combinations(holes, 2):
        if first["axis"] != second["axis"]:
            continue
        axis = "xyz".index(first["axis"])
        offset = np.array(first["center_mm"]) - np.array(second["center_mm"])
        offset[axis] = 0
        distance = float(np.linalg.norm(offset))
        if distance > 1e-7:
            checks["bore_ligament:" + first["name"] + ":" + second["name"]] = distance - first["radius_mm"] - second["radius_mm"]
    return {"margins_mm": checks, "minimum_mm": minimum, "passed": all(value >= minimum - 1e-6 for value in checks.values())}


def _wall_ray_screen(solid, minimum):
    measurements = []
    thin = []
    unresolved = []
    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(solid.wrapped, 1e-7)
    maximum_distance = solid.bounding_box().diagonal * 2
    for face_index, face in enumerate(solid.faces()):
        points = []
        for u in (0.2, 0.5, 0.8):
            for v in (0.2, 0.5, 0.8):
                point = face.position_at(u, v)
                if face.is_inside(point) and all(edge.distance_to(point) > 1e-5 for edge in face.edges()):
                    points.append(point)
        if not points:
            vertices, triangles = face.tessellate(0.1, 0.2)
            if triangles:
                array = np.array([tuple(point) for point in vertices])
                triangle = max(triangles, key=lambda indexes: np.linalg.norm(np.cross(array[indexes[1]] - array[indexes[0]], array[indexes[2]] - array[indexes[0]])))
                centroid = array[list(triangle)].mean(axis=0)
                points = [face.closest_points(Vector(*centroid))[0]]
        sampled = 0
        for point in points:
            direction = -face.normal_at(point)
            line = gp_Lin(gp_Pnt(*point), gp_Dir(*direction))
            intersector.PerformNearest(line, 1e-5, maximum_distance)
            positive = [intersector.WParameter(index) for index in range(1, intersector.NbPnt() + 1) if intersector.WParameter(index) > 1e-5]
            if not positive:
                continue
            thickness = min(positive)
            measurements.append(thickness)
            sampled += 1
            if thickness < minimum - 1e-5:
                thin.append({"face": face_index, "position_mm": list(point), "thickness_mm": thickness})
        if not sampled:
            unresolved.append(face_index)
    return {"method": "exact inward CAD normal rays, nine UV samples per face with projected largest-triangle fallback", "minimum_required_mm": minimum, "minimum_measured_mm": min(measurements) if measurements else None, "ray_count": len(measurements), "thin_samples": thin[:30], "thin_sample_count": len(thin), "unresolved_faces": unresolved, "passed": bool(measurements) and not thin and not unresolved, "limitations": "Finite surface sampling is a geometric screen, not a proof of global minimum thickness between samples"}


def _support_accessibility(solid, occupied):
    cavities = max(0, len(solid.shells()) - 1)
    if occupied is None:
        return {"passed": False, "reason": "No occupancy field available"}
    background = np.pad(~occupied, 1, constant_values=True)
    labels, _ = label(background, generate_binary_structure(3, 1))
    inaccessible = int(np.sum(background & (labels != labels[0, 0, 0])))
    return {"closed_cad_cavities": cavities, "trapped_void_voxels": inaccessible, "passed": cavities == 0 and inaccessible == 0, "method": "closed-shell cavity count plus face-connected flood fill from padded exterior", "limitations": "Coarse accessibility screen; support-tool reach and removal through narrow exact passages need slicer/physical review"}


def validate_topology(solid, domain: dict, settings: dict) -> dict:
    violations = []
    checks = {}
    tolerance = settings.get("volume_tolerance_mm3", 1e-5)
    topology = {"solid_count": len(solid.solids()), "valid": bool(solid.is_valid), "closed": all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in solid.shells())}
    vertices, faces = solid.tessellate(settings.get("tessellation_mm", 0.05), 0.1)
    mesh = trimesh.Trimesh(vertices=[tuple(point) for point in vertices], faces=faces, process=True)
    topology.update({"mesh_watertight": bool(mesh.is_watertight), "mesh_winding_consistent": bool(mesh.is_winding_consistent), "mesh_body_count": int(mesh.body_count), "volume_mm3": float(solid.volume)})
    topology["passed"] = topology["solid_count"] == topology["mesh_body_count"] == 1 and topology["valid"] and topology["closed"] and topology["mesh_watertight"] and topology["mesh_winding_consistent"]
    checks["topology"] = topology
    if not topology["passed"]:
        violations.append("topology:closed_manifold_single_solid")
    forbidden = [region for region in domain["regions"] if region["role"] == "forbidden"]
    cut_shapes = [region_shape(region) for region in forbidden]
    keepouts = {}
    for region, shape in zip(forbidden, cut_shapes):
        volume = _volume(solid.intersect(shape))
        keepouts[region["name"]] = {"intersection_mm3": volume, "purpose": region["purpose"], "passed": volume <= tolerance}
        if volume > tolerance:
            violations.append("forbidden:" + region["name"])
    checks["forbidden"] = keepouts
    manufacturing = domain["manufacturing"]
    nozzle_limit = manufacturing["nozzle_width_mm"] * manufacturing["minimum_wall_nozzles"]
    preserve_checks = {}
    for region in domain["regions"]:
        if region["role"] != "preserve":
            continue
        outer = region_shape(region)
        expected = outer.cut(*cut_shapes) if cut_shapes else outer
        volume = float(expected.volume)
        missing = _volume(expected.cut(solid))
        wall_limit = max(nozzle_limit, region.get("minimum_wall_mm", nozzle_limit))
        walls = _primitive_wall_checks(region, forbidden, wall_limit)
        sections = _attachment_sections(solid, outer, settings.get("attachment_probe_depth_mm", 0.02))
        area = sum(sections.values())
        minimum_area = region.get("attachment_area_min_mm2", manufacturing["minimum_attachment_area_mm2"])
        passed = volume > tolerance and missing <= tolerance and walls["passed"] and area + 1e-6 >= minimum_area
        preserve_checks[region["name"]] = {"required_volume_mm3": volume, "missing_volume_mm3": missing, "walls": walls, "attachment_sections_mm2": sections, "attachment_area_mm2": area, "minimum_attachment_area_mm2": minimum_area, "attachment_method": "summed outside AABB slab mean areas at six interfaces; local attachment check", "passed": passed}
        if not passed:
            violations.append("preserve:" + region["name"])
    checks["preserve"] = preserve_checks
    report = getattr(solid, "topology_report", None)
    occupied = getattr(solid, "topology_occupied", None)
    minimum_feature = max(nozzle_limit, manufacturing["minimum_feature_mm"])
    features = {"minimum_feature_mm": minimum_feature, "nozzle_limit_mm": nozzle_limit, "method": "voxel feature lower bound plus exact preserve ligament checks and post-Boolean normal-ray screen", "reconstruction_report_present": report is not None, "passed": False}
    if report is not None and occupied is not None:
        features["grid_feature_mm"] = report["minimum_voxel_feature_mm"]
        features["passed"] = features["grid_feature_mm"] + 1e-7 >= minimum_feature
        checks["reconstruction"] = report
    ray_screen = _wall_ray_screen(solid, minimum_feature)
    features["post_boolean_wall_screen"] = ray_screen
    features["passed"] = features["passed"] and ray_screen["passed"]
    if not features["passed"]:
        violations.append("manufacturing:minimum_feature_screen_failed")
    checks["features"] = features
    build_direction = manufacturing["build_direction"]
    support_needed = bool(np.any(mesh.face_normals[:, 2] < -0.1))
    support = {"build_direction": build_direction, "supports_allowed": bool(manufacturing["supports_allowed"]), "supports_required": support_needed, "method": "conservative downward-face flag; slicer planning required", "passed": build_direction == [0, 0, 1] and (manufacturing["supports_allowed"] or not support_needed)}
    support["accessibility"] = _support_accessibility(solid, occupied)
    support["passed"] = support["passed"] and support["accessibility"]["passed"]
    checks["supports"] = support
    if not support["passed"]:
        violations.append("manufacturing:unsupported_build_direction_or_overhangs")
    return {"passed": not violations, "violations": violations, "checks": checks}
