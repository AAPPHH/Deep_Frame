from copy import deepcopy
from itertools import combinations, permutations, product
from math import isfinite
from time import perf_counter

import numpy as np
import trimesh
from build123d import Align, Axis, Box, Compound, Cylinder, Pos, Rot, Solid, Vector
from OCP.BRep import BRep_Tool
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from scipy.ndimage import generate_binary_structure, label

from deep_frame.config import IMPLICIT_CONFIG, TOPOLOGY_CONFIG
from deep_frame.fea import prepare_frame_case
from deep_frame.frame import assembly_placements, build_components, camera_mount_z, motor_positions, mount_positions

def _box(name, role, minimum, maximum, purpose, **extra):
    return {"name": name, "role": role, "kind": "box", "min_mm": list(minimum), "max_mm": list(maximum), "purpose": purpose, **extra}

def _cylinder(name, role, center, radius, height, purpose, axis="z", **extra):
    return {"name": name, "role": role, "kind": "cylinder", "center_mm": list(center), "radius_mm": float(radius), "height_mm": float(height), "axis": axis, "purpose": purpose, **extra}

def region_bounds(region):
    if region["kind"] in ("box", "halfspace"):
        return np.asarray(region["min_mm"], dtype=float), np.asarray(region["max_mm"], dtype=float)
    half = np.full(3, region["radius_mm"], dtype=float)
    half[{"x": 0, "y": 1, "z": 2}[region.get("axis", "z")]] = region["height_mm"] / 2
    center = np.asarray(region["center_mm"], dtype=float)
    return center - half, center + half

def _preserve_subtractions(regions):
    result = []
    for preserve in (region for region in regions if region["role"] == "preserve"):
        pmin, pmax = region_bounds(preserve)
        for forbidden in (region for region in regions if region["role"] == "forbidden"):
            fmin, fmax = region_bounds(forbidden)
            overlap = np.minimum(pmax, fmax) - np.maximum(pmin, fmin)
            if np.all(overlap > 1e-8):
                if not forbidden.get("allow_preserve_subtraction", False):
                    raise ValueError(f"Undeclared preserve/forbidden overlap: {preserve['name']} / {forbidden['name']}")
                result.append({"preserve": preserve["name"], "forbidden": forbidden["name"], "reason": forbidden["purpose"], "detection": "conservative positive AABB overlap; exact CSG subtraction governs geometry"})
    return result

AXIS = {"x": 0, "y": 1, "z": 2}

def _bore(region):
    return region["role"] == "forbidden" and region["kind"] == "cylinder" and region.get("rasterize", True) is False

def _coaxial(a, b):
    axis = AXIS[a.get("axis", "z")]
    return a["kind"] == b["kind"] == "cylinder" and a.get("axis", "z") == b.get("axis", "z") and all(abs(a["center_mm"][i] - b["center_mm"][i]) <= 1e-6 for i in range(3) if i != axis)

def _footprint_gap(a, b, axis):
    disks = [region for region in (a, b) if region["kind"] == "cylinder" and AXIS[region.get("axis", "z")] == axis]
    plane = [i for i in range(3) if i != axis]
    if len(disks) == 2:
        return float(np.hypot(*(a["center_mm"][i] - b["center_mm"][i] for i in plane)) - a["radius_mm"] - b["radius_mm"])
    if len(disks) == 1:
        disk, other = disks[0], b if disks[0] is a else a
        low, high = region_bounds(other)
        offset = [max(low[i] - disk["center_mm"][i], disk["center_mm"][i] - high[i], 0.0) for i in plane]
        return float(np.hypot(*offset) - disk["radius_mm"])
    (alow, ahigh), (blow, bhigh) = region_bounds(a), region_bounds(b)
    return float(max(max(blow[i] - ahigh[i], alow[i] - bhigh[i]) for i in plane))

def _separation(a, b):
    axis = next((AXIS[region.get("axis", "z")] for region in (a, b) if region["kind"] == "cylinder"), 2)
    (alow, ahigh), (blow, bhigh) = region_bounds(a), region_bounds(b)
    return max(float(max(blow[axis] - ahigh[axis], alow[axis] - bhigh[axis])), _footprint_gap(a, b, axis))

def _flush_pairs(regions):
    bores = [region for region in regions if _bore(region)]
    for preserve in (region for region in regions if region["role"] == "preserve" and AXIS[region.get("axis", "z")] == 2):
        plow, phigh = region_bounds(preserve)
        for keepout in (region for region in regions if region["role"] == "forbidden" and not _bore(region)):
            if any(_coaxial(keepout, bore) and bore["radius_mm"] >= keepout["radius_mm"] for bore in bores) or _footprint_gap(preserve, keepout, 2) >= -1e-6:
                continue
            klow, khigh = region_bounds(keepout)
            for sign, face, plane in ((1, phigh[2], klow[2]), (-1, plow[2], khigh[2])):
                if abs(face - plane) <= 1e-6:
                    yield preserve, keepout, sign

def _near_wall(preserve, keepout, reserve):
    if _coaxial(preserve, keepout):
        insets = [keepout["radius_mm"] - preserve["radius_mm"]]
    else:
        (plow, phigh), (klow, khigh) = region_bounds(preserve), region_bounds(keepout)
        insets = [value for i in (0, 1) for value in (khigh[i] - phigh[i], plow[i] - klow[i])]
    return any(-1e-6 < value < reserve-1e-6 for value in insets)

def _extend_flush_contacts(regions, overlap, reserve):
    extended, moved = [], set()
    for preserve, keepout, sign in list(_flush_pairs(regions)):
        if _near_wall(preserve, keepout, reserve):
            continue
        keepout["allow_preserve_subtraction"] = True
        extended.append({"preserve": preserve["name"], "forbidden": keepout["name"], "direction": sign, "overlap_mm": overlap})
        if (preserve["name"], sign) in moved:
            continue
        moved.add((preserve["name"], sign))
        if preserve["kind"] == "box":
            preserve["max_mm" if sign > 0 else "min_mm"][2] += sign * overlap
        else:
            preserve["center_mm"][2] += sign * overlap / 2
            preserve["height_mm"] += overlap
    return extended

def prescribed_clearance(regions, minimum, margin, inflation):
    required, violations = minimum + margin, []
    preserves = [region for region in regions if region["role"] == "preserve"]
    for preserve, keepout, sign in _flush_pairs(regions):
        violations.append({"rule": "flush_contact", "preserve": preserve["name"], "forbidden": keepout["name"], "direction": sign, "near_wall": _near_wall(preserve, keepout, inflation + margin)})
    for preserve in preserves:
        plow, phigh = region_bounds(preserve)
        for keepout in (region for region in regions if region["role"] == "forbidden" and not _bore(region)):
            klow, khigh = region_bounds(keepout)
            if _coaxial(preserve, keepout):
                axis = AXIS[preserve.get("axis", "z")]
                chords = [preserve["radius_mm"] - keepout["radius_mm"]] if min(phigh[axis], khigh[axis]) - max(plow[axis], klow[axis]) >= -1e-6 else []
            elif np.all(np.minimum(phigh, khigh) - np.maximum(plow, klow) > 1e-6):
                chords = [value for axis in range(3) for value, face in ((phigh[axis] - khigh[axis], khigh[axis]), (klow[axis] - plow[axis], klow[axis])) if plow[axis] + 1e-6 < face < phigh[axis] - 1e-6]
            else:
                chords = []
            violations.extend({"rule": "rim_width", "preserve": preserve["name"], "forbidden": keepout["name"], "width_mm": float(value)} for value in chords if 1e-6 < value < required - 1e-6)
    for a, b in combinations(preserves, 2):
        gap = _separation(a, b)
        if 1e-6 < gap < required + 2 * inflation:
            violations.append({"rule": "preserve_gap", "preserve": a["name"], "other": b["name"], "gap_mm": gap})
    return {"passed": not violations, "violations": violations, "required_width_mm": required, "inflation_mm": inflation,
            "method": "Horizontal preserve faces coplanar with a keep-out face over a shared footprint are flush contacts (the inflated shell would be cut at its cap); every keep-out wall crossing a preserve, or coaxial keep-out cylinder, must leave a preserve rim of at least the minimum wall plus margin; distinct preserves either overlap or stay apart by more than that width plus both inflations. A flush contact whose preserve sits within one inflation plus margin of a keep-out side wall cannot be extended (the inflated shell would leave the keep-out as a sliver) and stays a violation; such a preserve has to be resized. Prescribed bores are excluded (their webs follow the C6 bore allowance)."}

def _merge(defaults, changes):
    result = deepcopy(defaults)
    for key, value in changes.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result

def grid_centers(grid):
    origin = np.asarray(grid["origin_mm"], dtype=float)
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    shape = np.asarray(grid["shape"])
    if origin.shape != (3,) or spacing.shape != (3,) or shape.shape != (3,):
        raise ValueError("Grid origin, spacing and shape must each have three entries")
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Grid coordinates must be finite with positive spacing")
    if not np.issubdtype(shape.dtype, np.integer) or np.any(shape < 1):
        raise ValueError("Grid shape must contain positive integers")
    if grid.get("axis_order", "xyz") != "xyz" or grid.get("order", "C") != "C":
        raise ValueError("Grid must use xyz axes and C array order")
    return np.stack(np.meshgrid(*(origin[axis] + spacing[axis] * (np.arange(shape[axis]) + 0.5) for axis in range(3)), indexing="ij"), axis=-1)

def region_contains(points, region, padding=0.0):
    points = np.asarray(points, dtype=float)
    padding = np.broadcast_to(np.asarray(padding, dtype=float), (3,))
    if points.shape[-1] != 3:
        raise ValueError("Points must have three coordinates")
    if region["kind"] == "halfspace":
        normal = np.asarray(region["normal"], dtype=float)
        if normal.shape != (3,) or not np.isclose(np.linalg.norm(normal), 1):
            raise ValueError("Halfspace requires a unit normal")
        return points @ normal - np.abs(normal) @ padding < float(region["offset_mm"]) - 1e-9
    if region["kind"] == "box":
        minimum = np.asarray(region["min_mm"], dtype=float)
        maximum = np.asarray(region["max_mm"], dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,) or not np.all(np.isfinite([minimum, maximum])) or np.any(maximum <= minimum):
            raise ValueError("Region boxes must have finite positive extents")
        return np.all((points >= minimum - padding - 1e-9) & (points <= maximum + padding + 1e-9), axis=-1)
    if region["kind"] == "cylinder":
        center = np.asarray(region["center_mm"], dtype=float)
        radius, height = float(region["radius_mm"]), float(region["height_mm"])
        axis = {"x": 0, "y": 1, "z": 2}[region.get("axis", "z")]
        radial = [index for index in range(3) if index != axis]
        if center.shape != (3,) or not np.all(np.isfinite(center)) or not np.isfinite(radius + height) or min(radius, height) <= 0:
            raise ValueError("Cylinder dimensions must be finite and positive")
        delta = points - center
        return (np.linalg.norm(delta[..., radial], axis=-1) <= radius + np.linalg.norm(padding[radial]) + 1e-9) & (np.abs(delta[..., axis]) <= height / 2 + padding[axis] + 1e-9)
    raise ValueError(f"Unsupported region kind: {region['kind']}")

def rasterize_regions(grid, regions):
    points = grid_centers(grid)
    shape = tuple(grid["shape"])
    masks = {name: np.zeros(shape, dtype=bool) for name in ("allowed", "preserve", "forbidden")}
    names = set()
    for region in regions:
        if region["name"] in names:
            raise ValueError("Region names must be unique")
        names.add(region["name"])
        if region["role"] not in masks:
            raise ValueError("Region role must be allowed, preserve or forbidden")
        inside = region_contains(points, region)
        if region["role"] == "forbidden" and region.get("rasterize", True):
            half = np.asarray(grid["spacing_mm"]) / 2
            if region["kind"] == "halfspace":
                normal = np.asarray(region["normal"])
                inside = points @ normal - np.abs(normal) @ half < region["offset_mm"] - 1e-9
            elif region["kind"] == "box":
                inside = np.all((points + half > np.asarray(region["min_mm"]) + 1e-9) & (points - half < np.asarray(region["max_mm"]) - 1e-9), axis=-1)
            else:
                axis = {"x": 0, "y": 1, "z": 2}[region.get("axis", "z")]
                radial = [index for index in range(3) if index != axis]
                delta = np.abs(points - np.asarray(region["center_mm"]))
                closest = np.maximum(delta[..., radial] - half[radial], 0)
                inside = (np.linalg.norm(closest, axis=-1) < region["radius_mm"] - 1e-9) & (delta[..., axis] - half[axis] < region["height_mm"] / 2 - 1e-9)
        if region.get("rasterize", True):
            masks[region["role"]] |= inside
    masks["allowed"] &= ~masks["forbidden"]
    masks["preserve"] &= masks["allowed"]
    masks["forbidden"] = ~masks["allowed"]
    if not masks["allowed"].any() or not (masks["allowed"] & ~masks["preserve"]).any():
        raise ValueError("The design domain needs allowed and freely optimizable cells")
    return masks

def _component_regions(parameters, settings, grid):
    f, c = parameters["frame"], parameters["components"]
    clearance = settings["component_clearance_mm"]
    placements = assembly_placements(parameters)
    components = build_components(parameters, placements)
    origin = np.asarray(grid["origin_mm"])
    upper = origin + np.asarray(grid["spacing_mm"]) * np.asarray(grid["shape"])
    regions = [_box("design_envelope", "allowed", origin, upper, "One connected cuboid, not a frame or arm mask")]
    for name, component in components.items():
        box = component["shape"].bounding_box()
        minimum, maximum = np.asarray(tuple(box.min)), np.asarray(tuple(box.max))
        if component["kind"] == "prop":
            center = component["center_of_mass_mm"]
            regions.append(_cylinder(name + "_swept_clearance", "forbidden", center, c["prop"]["diameter_mm"] / 2 + settings["prop_clearance_mm"], c["prop"]["thickness_mm"] + 2 * settings["prop_clearance_mm"], "Propeller swept volume including axial and radial clearance"))
        elif component["kind"] == "motor":
            position = component["position_mm"]
            regions.append(_cylinder(name + "_envelope", "forbidden", [position[0], position[1], (minimum[2] + maximum[2] + clearance) / 2], c["motor"]["diameter_mm"] / 2 + clearance, maximum[2] - minimum[2] + clearance, "Motor hardware envelope; lower mating face permits contact"))
        else:
            expanded_min, expanded_max = minimum - clearance, maximum + clearance
            if name in ("aio15", "battery"):
                expanded_min[2] = minimum[2]
            regions.append(_box(name + "_envelope", "forbidden", expanded_min, expanded_max, "Component envelope with clearance, excluding intentional lower support contact"))
    depth = settings["contact_depth_mm"]
    motor_radius = settings["motor_contact_radius_mm"]
    for name, (x, y) in motor_positions(parameters).items():
        regions.append(_cylinder(name + "_motor_contact", "preserve", [x, y, f["arm_height_mm"] / 2], motor_radius, f["arm_height_mm"], "Motor bolt attachment disk; no prescribed connecting arm", attachment_area_min_mm2=12.0, minimum_wall_mm=settings["manufacturing"]["minimum_feature_mm"]))
        bore_z, bore_height = (origin[2] + f["arm_height_mm"]) / 2, f["arm_height_mm"] - origin[2] + 2
        regions.append(_cylinder(name + "_shaft_clearance", "forbidden", [x, y, bore_z], f["motor_shaft_hole_mm"] / 2, bore_height, "Motor shaft clearance", rasterize=False))
        for index, (hx, hy) in enumerate(mount_positions(parameters)[name]):
            regions.append(_cylinder(f"{name}_motor_screw_{index}", "forbidden", [hx, hy, bore_z], (c["motor"]["screw_diameter_mm"] + f["hole_clearance_mm"]) / 2, bore_height, "Through screw bore and unobstructed underside screwdriver approach", rasterize=False))
        sign = -1 if x < 0 else 1
        corridor_min = [min(x - sign * 5, sign * 20), y - 2, f["arm_height_mm"] + 1]
        corridor_max = [max(x - sign * 5, sign * 20), y + 2, f["arm_height_mm"] + 5]
        regions.append(_box(name + "_motor_leads", "forbidden", corridor_min, corridor_max, "Provisional accessible straight motor lead corridor"))
    seat = f["base_thickness_mm"] + f["aio_standoff_mm"]
    post = settings["stack_post"]
    bore = post["bores"][post["fastening"]]
    foot = seat - max(f["aio_standoff_mm"], bore["depth_mm"] + post["bore_floor_mm"])
    layers = origin[2] + (np.arange(grid["shape"][2]) + 0.5) * grid["spacing_mm"][2]
    if not ((layers >= foot - 1e-9) & (layers + grid["spacing_mm"][2] / 2 <= seat + 1e-9)).any():
        foot = float(layers[layers < foot].max())
    for index, (x, y) in enumerate(mount_positions(parameters)["aio15"]):
        regions.append(_cylinder(f"aio_contact_{index}", "preserve", [x, y, (foot + seat) / 2], post["diameter_mm"] / 2, seat - foot, f"Stack post (whoop principle): AIO on its grommets on top, M2 from above, {post['fastening']} bore; connection to the frame left to the optimizer", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0, seat_mm=seat))
        regions.append(_cylinder(f"aio_screw_{index}", "forbidden", [x, y, seat + (1 - bore["depth_mm"]) / 2], bore["diameter_mm"] / 2, bore["depth_mm"] + 1, f"Blind M2 {post['fastening']} bore from the post top; closed below", rasterize=False, fastening=post["fastening"]))
    battery_y, battery_z = placements["battery"]["position"][1:]
    rail_width, rail_length = settings["battery_contact_width_mm"], settings["battery_contact_length_mm"]
    for sign in (-1, 1) if settings.get("battery_support", "rails") == "rails" else ():
        x, y = sign * (c["battery"]["width_mm"] / 2 - settings["battery_rail_edge_inset_mm"] - rail_width / 2), battery_y + settings["battery_contact_y_mm"]
        regions.append(_box(f"battery_rail_{sign}", "preserve", [x - rail_width / 2, y - rail_length / 2, battery_z - 3.0], [x + rail_width / 2, y + rail_length / 2, battery_z], "Longitudinal battery strap rail under the battery edge, ManaFly style; strap wraps battery and rail", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
    aio = placements["aio15"]["position"]
    aio_top = aio[2] + c["aio15"]["stack_height_mm"] + clearance
    regions.append(_box("elrs_antenna_clearance", "forbidden", [-c["aio15"]["width_mm"] / 2 - clearance, -c["aio15"]["length_mm"] / 2 - clearance, aio_top], [c["aio15"]["width_mm"] / 2 + clearance, c["aio15"]["length_mm"] / 2 + clearance, aio_top + c["aio15"]["elrs_antenna_clearance_mm"]], "ELRS wire antenna lifted at least 3 mm above the AIO board"))
    battery_half = c["battery"]["width_mm"] / 2 + clearance
    if settings.get("battery_support", "rails") == "free":
        regions.append(_box("battery_contact", "allowed", [-battery_half, battery_y - c["battery"]["length_mm"] / 2 - clearance, battery_z - 3.0], [battery_half, battery_y + c["battery"]["length_mm"] / 2 + clearance, battery_z], "Free battery support: contact plane under the battery (flatten target), no prescribed geometry"))
    regions.append(_box("battery_insertion", "forbidden", [-battery_half, battery_y - c["battery"]["length_mm"] / 2 - clearance, battery_z], [battery_half, battery_y + c["battery"]["length_mm"] / 2 + clearance, max(battery_z + c["battery"]["height_mm"] + clearance, upper[2] + 1)], "Battery insertion and removal vertically, no retaining roof"))
    guide = settings["battery_guide"]
    if guide["enabled"]:
        height, width = guide["height_mm"], guide["wall_zone_mm"]
        low_y, high_y = battery_y - c["battery"]["length_mm"] / 2 - clearance, battery_y + c["battery"]["length_mm"] / 2 + clearance
        regions.append(_box("battery_guide_side", "allowed", [battery_half, low_y, battery_z], [battery_half + width, high_y, battery_z + height], "Free low side guide, reflected by symmetry; no flight load interface", guide_axis=0))
        for name, lo, hi in (("front", high_y, high_y + width), ("rear", low_y - width, low_y)):
            regions.append(_box("battery_guide_" + name, "allowed", [-battery_half, lo, battery_z], [battery_half, hi, battery_z + height], "Free low end guide, handling only; no flight load interface", guide_axis=1))
        relief = guide["entry_relief_mm"]
        regions.append(_box("battery_guide_entry", "forbidden", [-battery_half-relief, low_y-relief, battery_z+height-relief], [battery_half+relief, high_y+relief, upper[2]+1], "Expanded upper insertion opening; no top overhang or latch"))
        regions.append(_box("battery_guide_height_cap", "forbidden", [-battery_half-12, low_y-12, battery_z+height], [battery_half+12, high_y+12, upper[2]+1], "Positioner limited to a low perimeter; higher cages and retention lips excluded"))
    camera_width = c["camera"]["width_mm"] + 2 * f["camera_side_clearance_mm"]
    for sign in (-1, 1) if settings.get("camera_support", "prescribed") == "prescribed" else ():
        x = sign * (camera_width / 2 + settings["camera_contact_width_mm"] / 2)
        width, length = settings["camera_contact_width_mm"], settings["camera_contact_length_mm"]
        regions.append(_box(f"camera_impact_contact_{sign}", "preserve", [x - width / 2, f["camera_y_mm"] - length / 2, f["cage_height_mm"] - depth], [x + width / 2, f["camera_y_mm"] + length / 2, f["cage_height_mm"]], "Two local protective impact contacts, without a predefined cage", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
        regions.append(_cylinder(f"camera_mount_{sign}", "preserve", [x, f["camera_y_mm"], camera_mount_z(parameters)], settings["camera_mount_radius_mm"], width, "Local camera screw lug; no prescribed connecting wall", axis="x", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
    lug_end = camera_width / 2 + settings["camera_contact_width_mm"]
    regions.append(_cylinder("camera_screw_axis", "forbidden", [0, f["camera_y_mm"], camera_mount_z(parameters)], f["camera_screw_diameter_mm"] / 2, 2 * lug_end, "Camera screw bore limited to exact mounting lugs", axis="x", rasterize=False))
    for sign in (-1, 1):
        end = upper[0] + 1 if sign > 0 else origin[0] - 1
        length = abs(end - sign * lug_end)
        regions.append(_cylinder(f"camera_tool_access_{sign}", "forbidden", [(end + sign * lug_end) / 2, f["camera_y_mm"], camera_mount_z(parameters)], settings["camera_tool_radius_mm"], length, "Conservative outboard screwdriver corridor; no subvoxel cut through free material", axis="x"))
    camera_box = components["camera"]["shape"].bounding_box()
    regions.append(_box("camera_front_access", "forbidden", [camera_box.min.X - clearance, camera_box.max.Y, camera_box.min.Z - clearance], [camera_box.max.X + clearance, upper[1] + 1, camera_box.max.Z + clearance], "Unobstructed camera front, insertion and lens field corridor"))
    regions.append(_box("aio_side_assembly_access", "forbidden", [0, -c["aio15"]["length_mm"] / 2 - clearance, placements["aio15"]["position"][2]], [upper[0] + 1, c["aio15"]["length_mm"] / 2 + clearance, placements["aio15"]["position"][2] + c["aio15"]["stack_height_mm"] + clearance], "AIO insertion/removal through right side with connectors unplugged"))
    return regions, placements, components

def _tool_access(regions, settings, top):
    result = []
    for region in (region for region in regions if region["name"].startswith("aio_contact_")):
        seat = region["seat_mm"]
        result.append(_cylinder(region["name"].replace("contact", "tool_access"), "forbidden", [*region["center_mm"][:2], (seat + top) / 2], settings["stack_post"]["tool_radius_mm"], top - seat, "Screwdriver corridor straight above the stack post through the AIO, grommet and battery zone; flush and rim rules do not apply", allow_preserve_subtraction=True))
    return result

def stack_pattern(regions):
    posts = [regions[f"aio_contact_{index}"] for index in range(4)]
    bores = [regions[f"aio_screw_{index}"] for index in range(4)]
    center = np.mean([bore["center_mm"] for bore in bores], axis=0)
    radius = float(np.hypot(*np.subtract(bores[0]["center_mm"], center)[:2]))
    if "seat_mm" not in posts[0]:
        return {"name": "stack_25.5", "center_mm": center[:2].tolist(), "z_mm": float(region_bounds(posts[0])[0][2]) + 1.5, "radius_mm": radius, "count": 4, "hole_diameter_mm": [1.6, 4.0], "screw_diameter_mm": 2.0, "tool_direction": [0, 0, -1]}
    diameter, top = 2 * bores[0]["radius_mm"], min(post["seat_mm"] for post in posts)
    return {"name": "stack_posts", "center_mm": center[:2].tolist(), "z_mm": top - 1.0, "radius_mm": radius, "count": 4,
            "hole_diameter_mm": [0.8 * diameter, 1.25 * diameter], "screw_diameter_mm": 0.9 * diameter, "tool_direction": [0, 0, 1], "post_top_mm": top, "post_diameter_mm": 2 * posts[0]["radius_mm"]}

def _connection_cases(regions, model, force):
    if force <= 0:
        return []
    fixtures = next(case["fixed_regions"] for case in model["load_cases"] if case["name"] == "battery_impact")
    cases = []
    for region in regions:
        name = region["name"]
        if region["role"] != "preserve" or not name.startswith(("aio_contact_", "battery_rail_", "camera_mount_")):
            continue
        if region["kind"] == "box":
            minimum, maximum = region["min_mm"], region["max_mm"]
        else:
            half = np.full(3, region["radius_mm"])
            half[{"x": 0, "y": 1, "z": 2}[region["axis"]]] = region["height_mm"] / 2
            minimum = (np.asarray(region["center_mm"]) - half).tolist()
            maximum = (np.asarray(region["center_mm"]) + half).tolist()
        selector = {"kind": "box", "min_mm": list(minimum), "max_mm": list(maximum)}
        cases.append({"name": "connection_" + name, "analysis": "static", "fixed_regions": deepcopy(fixtures), "loads": [{"region": selector, "force_n": [0.0, 0.0, -force]}], "purpose": "Small declared attachment proof load; enforces mechanically connected required interface"})
    return cases

def lowered_grid(grid, drop):
    grid = deepcopy(grid)
    cells = int(np.ceil(drop / grid["spacing_mm"][2] - 1e-9))
    grid["shape"][2] += cells
    grid["origin_mm"][2] -= cells * grid["spacing_mm"][2]
    return grid

def embed_field(field, grid):
    field = np.asarray(field)
    cells = grid["shape"][2] - field.shape[2]
    if cells < 0 or field.shape[:2] != tuple(grid["shape"][:2]):
        raise ValueError("Field does not fit the grid below its top layer")
    return np.pad(field, ((0, 0), (0, 0), (cells, 0)), mode="edge")

def functional_geometry(parameters, settings, components):
    result = {}
    if settings["battery_guide"]["enabled"]:
        result["battery_guide"] = deepcopy(settings["battery_guide"])
    flight = settings["low_flight"]
    if flight["enabled"]:
        pitch = np.radians(flight["pitch_deg"])
        vertical = np.array([0, -np.sin(pitch), np.cos(pitch)])
        camera, frame = parameters["components"]["camera"], parameters["frame"]
        tilt = np.radians(camera["tilt_deg"])
        lens = np.array([0, frame["camera_y_mm"]+camera["length_mm"]/2*np.cos(tilt), camera_mount_z(parameters)+camera["length_mm"]/2*np.sin(tilt)])
        bottoms = {name: float(item["shape"].rotate(Axis.X, -flight["pitch_deg"]).bounding_box().min.Z) for name, item in components.items()}
        lens_z = float(lens @ vertical)
        fixed_gap = lens_z - min(bottoms.values())
        camera_gap = lens_z - bottoms["camera"]
        maximum = max(fixed_gap, camera_gap+flight["guard_drop_mm"])
        result["low_flight"] = {**deepcopy(flight), "vertical_axis": vertical.tolist(), "lens_mm": lens.tolist(), "lens_world_z_mm": lens_z,
                                "component_bottoms_world_z_mm": bottoms, "fixed_hardware_gap_mm": fixed_gap, "camera_bottom_world_z_mm": bottoms["camera"],
                                "maximum_gap_mm": maximum, "minimum_frame_world_z_mm": lens_z-maximum,
                                "camera_center_mm": [0, frame["camera_y_mm"], camera_mount_z(parameters)], "camera_width_mm": camera["width_mm"], "camera_length_mm": camera["length_mm"],
                                "lens_source": "Front-face midpoint of the selected camera envelope; physical optical-centre offset remains unmeasured"}
    return result

def build_design_domain(parameters):
    settings = _merge(TOPOLOGY_CONFIG, parameters.get("topology", {}))
    grid = lowered_grid(settings["grid"], settings["floor_drop_mm"])
    centers = grid_centers(grid)
    manufacturing = settings["manufacturing"]
    required_feature = manufacturing["nozzle_width_mm"] * manufacturing["minimum_wall_nozzles"]
    if not np.isfinite(required_feature) or required_feature <= 0 or manufacturing["minimum_feature_mm"] < required_feature:
        raise ValueError("Minimum feature must satisfy nozzle width times track count")
    regions, placements, components = _component_regions(parameters, settings, grid)
    for region in regions:
        if region["role"] == "forbidden" and not region.get("rasterize", True):
            region["allow_preserve_subtraction"] = True
    regions.extend(deepcopy(settings["additional_regions"]))
    flush = _extend_flush_contacts(regions, settings["flush_overlap_mm"], IMPLICIT_CONFIG["preserve_inflation_mm"] + settings["prescribed_wall_margin_mm"])
    clearance = prescribed_clearance(regions, manufacturing["minimum_feature_mm"], settings["prescribed_wall_margin_mm"], IMPLICIT_CONFIG["preserve_inflation_mm"])
    regions.extend(_tool_access(regions, settings, grid["origin_mm"][2] + grid["spacing_mm"][2] * grid["shape"][2] + 1))
    subtractions = _preserve_subtractions(regions)
    functions = functional_geometry(parameters, settings, components)
    if functions.get("low_flight"):
        flight = functions["low_flight"]
        regions.append({"name": "flight_lens_clearance_floor", "role": "forbidden", "kind": "halfspace", "normal": flight["vertical_axis"], "offset_mm": flight["minimum_frame_world_z_mm"], "min_mm": grid["origin_mm"], "max_mm": (np.asarray(grid["origin_mm"])+np.asarray(grid["shape"])*grid["spacing_mm"]).tolist(), "purpose": "Material below the lens-relative whole-copter clearance in requested flight attitude is forbidden"})
    masks = rasterize_regions(grid, regions)
    _, allowed_components = label(masks["allowed"])
    if allowed_components != 1:
        raise ValueError("The allowed design domain must be face-connected")
    model = prepare_frame_case(parameters)
    tolerance = parameters["integration"]["selection_tolerance_mm"]
    fixture_radius = settings["aio_contact_radius_mm"]
    posts = [region_bounds(region) for region in regions if region["name"].startswith("aio_contact_")]
    aio_fixtures = [{"kind": "box", "min_mm": [*((low[:2] + high[:2]) / 2 - fixture_radius).tolist(), float(low[2]) - tolerance], "max_mm": [*((low[:2] + high[:2]) / 2 + fixture_radius).tolist(), float(high[2]) + tolerance]} for low, high in posts]
    for case in model["load_cases"]:
        if case["name"] in ("arm_tip", "thrust_all") or case["name"].startswith("crash_"):
            case["fixed_regions"] = deepcopy(aio_fixtures)
    model["fixture_model"] = "Arm-tip, thrust and crash cases: the four stack posts fixed over their height; other cases: four motor contact undersides fixed. Identical selectors must be used for v0 comparison."
    auxiliary_cases = _connection_cases(regions, model, settings["connection_proof_force_n"])
    weights = {case["name"]: 1.0 for case in model["load_cases"] if case["analysis"] == "static"}
    weights.update({case["name"]: len(weights) / (9 * len(auxiliary_cases)) for case in auxiliary_cases})
    weights.update(settings["optimizer"].get("case_weights", {}))
    settings["optimizer"]["case_weights"] = weights
    cell_volume = float(np.prod(grid["spacing_mm"]))
    counts = {name + "_cells": int(mask.sum()) for name, mask in masks.items()}
    counts["free_cells"] = counts["allowed_cells"] - counts["preserve_cells"]
    volumes = {key.replace("_cells", "_volume_mm3"): count * cell_volume for key, count in counts.items()}
    dimensions = {name: {"position_mm": list(component["position_mm"]), "center_of_mass_mm": list(component["center_of_mass_mm"]), "mass_g": component["mass_g"], "source": component["source"]} for name, component in components.items()}
    unresolved = []
    for region in regions:
        if region["role"] == "preserve" and not np.any(region_contains(centers, region) & masks["preserve"]):
            unresolved.append(region["name"])
    if unresolved:
        raise ValueError(f"Preserve regions unresolved by grid: {unresolved}; refine or shift the grid")
    return {
        "schema_version": "deep-frame-topology-domain-v1",
        "grid": grid,
        **masks,
        "regions": regions,
        "functional_requirements": functions,
        "material": deepcopy(model["material"]),
        "point_masses": deepcopy(model["point_masses"]),
        "load_cases": deepcopy(model["load_cases"]) + auxiliary_cases,
        "comparison_load_cases": deepcopy(model["load_cases"]),
        "fea_settings": deepcopy(model["settings"]),
        "manufacturing": deepcopy(manufacturing),
        "optimizer_settings": deepcopy(settings["optimizer"]),
        "reconstruction_settings": deepcopy(settings["reconstruction"]),
        "metadata": {
            "generator": "component envelopes and isolated interface contacts in one cuboid",
            "initial_density": "uniform in all freely optimizable cells; no v0 geometry or arm seed",
            "region_count": len(regions),
            **counts,
            **volumes,
            "preserve_fraction_of_allowed": counts["preserve_cells"] / counts["allowed_cells"],
            "free_fraction_of_allowed": counts["free_cells"] / counts["allowed_cells"],
            "allowed_face_connected_components": allowed_components,
            "declared_preserve_subtractions": subtractions,
            "flush_contact_extensions": flush,
            "prescribed_clearance": clearance,
            "component_placements": deepcopy(placements),
            "components": dimensions,
            "settings": deepcopy(settings),
            "reference": "Parametric v0 is used only for shared component placements and comparison load definitions, never its solid or connectivity.",
            "mass_scope": model["mass_scope"],
            "fixture_model": model["fixture_model"],
            "coupling_model": model["coupling_model"],
            "auxiliary_load_model": f"{settings['connection_proof_force_n']} N downward per required otherwise unforced interface; all evaluated independently",
            "default_objective_weights": "Normalized static compliances: primary mechanical cases 90 percent collectively; all attachment proof cases 10 percent collectively.",
            "rasterization": "Allowed and preserve primitives at cell centers; forbidden cells conservatively removed on any positive-volume overlap. Exact preserve unions and forbidden cuts remain mandatory after reconstruction.",
            "assumptions": [
                "Motor positions, component placements and local mounting faces are fixed interfaces, not design variables in this run.",
                "Camera screw position is provisional; final hardware fit requires measurement. XT30, balance plug and antennas have no prescribed seat; they are strapped where they fit.",
                "The initial design envelope is a full cuboid; no prescribed arms, central plate, battery walls, cage walls or tail links.",
                "Local contact primitives are human-specified functional interfaces; connecting branches are optimizer variables.",
                "Preserve volume fractions refer to raster cells; exact bores and interface boundaries change final physical volume.",
            ],
        },
    }

def symmetric_domains(domain, axis=0):
    grid = domain["grid"]
    shape, origin, spacing = list(grid["shape"]), np.asarray(grid["origin_mm"], dtype=float), np.asarray(grid["spacing_mm"], dtype=float)
    if shape[axis] % 2 or abs(origin[axis] + shape[axis] * spacing[axis] / 2) > 1e-9:
        raise ValueError("Symmetric half domains need an even cell count centered on the mirror plane")
    allowed = domain["allowed"] & np.flip(domain["allowed"], axis)
    preserve = (domain["preserve"] | np.flip(domain["preserve"], axis)) & allowed
    if label(allowed)[1] != 1 or not (allowed & ~preserve).any():
        raise ValueError("The symmetrized design domain must be face-connected with free cells")
    full = {**domain, "allowed": allowed, "preserve": preserve, "forbidden": ~allowed}
    full["metadata"] = {**domain.get("metadata", {}), "symmetrization": {"axis": axis, "removed_allowed_cells": int(np.count_nonzero(domain["allowed"] & ~allowed)), "added_preserve_cells": int(np.count_nonzero(preserve & ~domain["preserve"]))}}
    cut = [slice(None)] * 3
    cut[axis] = slice(shape[axis] // 2, None)
    half_grid = deepcopy(grid)
    half_grid["shape"][axis] = shape[axis] // 2
    half_grid["origin_mm"][axis] = 0.0
    half = {**full, "grid": half_grid, "symmetry": {"axis": axis, "plane_mm": 0.0}, **{name: full[name][tuple(cut)].copy() for name in ("allowed", "preserve", "forbidden")}}
    return full, half

def mirror_field(half, axis=0):
    return np.concatenate([np.flip(half, axis), half], axis=axis)

def region_shape(region):
    if region["kind"] == "halfspace":
        normal = np.asarray(region["normal"])
        if abs(normal[0]) > 1e-9:
            raise ValueError("Flight halfspace expects pitch about X")
        extent = 4*float(np.linalg.norm(np.asarray(region["max_mm"])-region["min_mm"]))
        angle = float(np.degrees(np.arctan2(-normal[1], normal[2])))
        return Pos(*(normal*(region["offset_mm"]-extent/2))) * Rot(angle, 0, 0) * Box(extent, extent, extent)
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

def _ambiguous_cells(mask):
    for axis in range(3):
        others = [index for index in range(3) if index != axis]
        for plane in range(mask.shape[axis]):
            square = np.take(mask, plane, axis=axis)
            a, b, c, d = square[:-1, :-1], square[1:, :-1], square[:-1, 1:], square[1:, 1:]
            ambiguous = (a & d & ~b & ~c) | (b & c & ~a & ~d)
            if np.any(ambiguous):
                i, j = np.argwhere(ambiguous)[0]
                cells = []
                for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    index = [0, 0, 0]
                    index[axis], index[others[0]], index[others[1]] = plane, i + di, j + dj
                    if not mask[tuple(index)]:
                        cells.append(tuple(index))
                return cells, "diagonal_edge_contact"
    for index in np.ndindex(tuple(value - 1 for value in mask.shape)):
        block = mask[tuple(slice(value, value + 2) for value in index)]
        if 1 < np.sum(block) < 8 and label(block, generate_binary_structure(3, 1))[1] > 1:
            cells = []
            for offset in np.argwhere(~block):
                if any(block[tuple(offset + direction)] for direction in np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) if np.all((offset + direction >= 0) & (offset + direction < 2))):
                    cells.append(tuple(np.array(index) + offset))
            return cells, "diagonal_vertex_contact"
    return [], None

def _repair_manifold_cells(occupied, domain, density, settings):
    allowed = np.asarray(domain["allowed"], dtype=bool)
    repaired = occupied.copy()
    maximum = settings.get("maximum_repair_voxels", max(1, int(np.sum(occupied) * 0.05)))
    if not isinstance(maximum, int) or maximum < 0:
        raise ValueError("maximum_repair_voxels must be a nonnegative integer")
    repairs = []
    volume = float(np.prod(domain["grid"]["spacing_mm"]))
    while True:
        candidates, cause = _ambiguous_cells(repaired)
        if not candidates:
            return repaired, repairs
        candidates = [index for index in candidates if allowed[index]]
        if not candidates:
            raise ValueError("Manifold raster repair would enter forbidden cells")
        if len(repairs) >= maximum:
            raise ValueError("Manifold raster repair exceeded maximum_repair_voxels")
        selected = min(candidates, key=lambda index: (-float(density[index]), index))
        repaired[selected] = True
        repairs.append({"method": "density_guided_local_manifold_fill", "cause": cause, "cell_xyz": [int(value) for value in selected], "material_before": 0, "material_after": 1, "density_before": float(density[selected]), "density_after": 1.0, "added_volume_mm3": volume})

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
    if settings.get("repair_manifold_voxels", False):
        occupied, changes = _repair_manifold_cells(occupied, domain, field, settings)
        repairs.extend(changes)
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
    return _validate_topology(solid, domain, settings)

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

def _trapped_voids(occupied):
    background = np.pad(~occupied, 1, constant_values=True)
    labels, _ = label(background, generate_binary_structure(3, 1))
    return int(np.sum(background & (labels != labels[0, 0, 0])))

def _support_accessibility(solid, occupied):
    cavities = max(0, len(solid.shells()) - 1)
    if occupied is None:
        return {"passed": False, "reason": "No occupancy field available"}
    inaccessible = _trapped_voids(occupied)
    return {"closed_cad_cavities": cavities, "trapped_void_voxels": inaccessible, "passed": cavities == 0 and inaccessible == 0, "method": "closed-shell cavity count plus face-connected flood fill from padded exterior", "limitations": "Coarse accessibility screen; support-tool reach and removal through narrow exact passages need slicer/physical review"}

def _validate_topology(solid, domain: dict, settings: dict) -> dict:
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
    above_build_plate = mesh.triangles_center[:, 2] > mesh.bounds[0, 2] + 1e-6
    support_needed = bool(np.any((mesh.face_normals[:, 2] < -0.1) & above_build_plate))
    support = {"build_direction": build_direction, "supports_allowed": bool(manufacturing["supports_allowed"]), "supports_required": support_needed, "method": "conservative downward-face flag; slicer planning required", "passed": build_direction == [0, 0, 1] and (manufacturing["supports_allowed"] or not support_needed)}
    support["accessibility"] = _support_accessibility(solid, occupied)
    support["passed"] = support["passed"] and support["accessibility"]["passed"]
    checks["supports"] = support
    if not support["passed"]:
        violations.append("manufacturing:unsupported_build_direction_or_overhangs")
    return {"passed": not violations, "violations": violations, "checks": checks}
