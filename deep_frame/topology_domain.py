from copy import deepcopy
from itertools import product
import numpy as np
from scipy.ndimage import label

from deep_frame.components import build_components
from deep_frame.frame import assembly_placements, camera_mount_z, motor_positions, mount_positions
from deep_frame.integration import prepare_frame_case
from deep_frame.topology_config import TOPOLOGY_CONFIG


def _box(name, role, minimum, maximum, purpose, **extra):
    return {"name": name, "role": role, "kind": "box", "min_mm": list(minimum), "max_mm": list(maximum), "purpose": purpose, **extra}


def _cylinder(name, role, center, radius, height, purpose, axis="z", **extra):
    return {"name": name, "role": role, "kind": "cylinder", "center_mm": list(center), "radius_mm": float(radius), "height_mm": float(height), "axis": axis, "purpose": purpose, **extra}


def region_bounds(region):
    if region["kind"] == "box":
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
            if region["kind"] == "box":
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
            if name in ("aio15", "battery", "xt30", "balancer"):
                expanded_min[2] = minimum[2]
            regions.append(_box(name + "_envelope", "forbidden", expanded_min, expanded_max, "Component envelope with clearance, excluding intentional lower support contact"))
    depth = settings["contact_depth_mm"]
    motor_radius = settings["motor_contact_radius_mm"]
    for name, (x, y) in motor_positions(parameters).items():
        regions.append(_cylinder(name + "_motor_contact", "preserve", [x, y, f["arm_height_mm"] / 2], motor_radius, f["arm_height_mm"], "Motor bolt attachment disk; no prescribed connecting arm", attachment_area_min_mm2=12.0, minimum_wall_mm=settings["manufacturing"]["minimum_feature_mm"]))
        regions.append(_cylinder(name + "_shaft_clearance", "forbidden", [x, y, f["arm_height_mm"] / 2], f["motor_shaft_hole_mm"] / 2, f["arm_height_mm"] + 2, "Motor shaft clearance", rasterize=False))
        for index, (hx, hy) in enumerate(mount_positions(parameters)[name]):
            regions.append(_cylinder(f"{name}_motor_screw_{index}", "forbidden", [hx, hy, f["arm_height_mm"] / 2], (c["motor"]["screw_diameter_mm"] + f["hole_clearance_mm"]) / 2, f["arm_height_mm"] + 2, "Through screw bore and unobstructed underside screwdriver approach", rasterize=False))
        sign = -1 if x < 0 else 1
        corridor_min = [min(x - sign * 5, sign * 20), y - 2, f["arm_height_mm"] + 1]
        corridor_max = [max(x - sign * 5, sign * 20), y + 2, f["arm_height_mm"] + 5]
        regions.append(_box(name + "_motor_leads", "forbidden", corridor_min, corridor_max, "Provisional accessible straight motor lead corridor"))
    for index, (x, y) in enumerate(mount_positions(parameters)["aio15"]):
        height = f["base_thickness_mm"] + f["aio_standoff_mm"]
        regions.append(_cylinder(f"aio_contact_{index}", "preserve", [x, y, height / 2], settings["aio_contact_radius_mm"], height, "AIO mounting boss; no prescribed central plate", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
        regions.append(_cylinder(f"aio_screw_{index}", "forbidden", [x, y, height / 2], (c["aio15"]["screw_diameter_mm"] + f["hole_clearance_mm"]) / 2, height + 2, "AIO through screw and underside assembly access", rasterize=False))
    battery_z = placements["battery"]["position"][2]
    contact_width, contact_length = settings["battery_contact_width_mm"], settings["battery_contact_length_mm"]
    for sx, sy in product((-1, 1), repeat=2):
        x = sx * (c["battery"]["width_mm"] / 2 - contact_width / 2)
        y = sy * settings["battery_contact_y_mm"]
        regions.append(_box(f"battery_contact_{sx}_{sy}", "preserve", [x - contact_width / 2, y - contact_length / 2, battery_z - depth], [x + contact_width / 2, y + contact_length / 2, battery_z], "Independent battery support pad; optimizer chooses supporting paths", attachment_area_min_mm2=12.0, minimum_wall_mm=2.0))
    band_half = parameters["integration"]["battery_attachment_band_width_mm"] / 2
    band_y = parameters["integration"]["battery_attachment_y_mm"]
    for sign in (-1, 1):
        x = sign * (c["battery"]["width_mm"] / 2 + 1)
        regions.append(_box(f"battery_coupling_{sign}", "preserve", [x - contact_width / 2, band_y - max(band_half, 4), battery_z - depth], [x + contact_width / 2, band_y + max(band_half, 4), battery_z], "Local transverse band for the same battery mass and impact coupling as v0", attachment_area_min_mm2=12.0, minimum_wall_mm=2.0))
    for sx, sy in product((-1, 1), repeat=2):
        x, y = sx * (c["battery"]["width_mm"] / 2 + 1.5), sy * 12.0
        regions.append(_box(f"strap_contact_{sx}_{sy}", "preserve", [x - 3, y - 8, battery_z - depth], [x + 3, y + 8, battery_z], "Local strap eyelet with two-millimeter rim; no prescribed deck or deck support", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
        regions.append(_box(f"strap_access_{sx}_{sy}", "forbidden", [x - 1.0, y - 6.0, battery_z - depth - 1], [x + 1.0, y + 6.0, upper[2] + 1], "Battery strap insertion slot beside wide battery face", rasterize=False))
    battery_half = c["battery"]["width_mm"] / 2 + clearance
    regions.append(_box("battery_insertion", "forbidden", [-battery_half, -c["battery"]["length_mm"] / 2 - clearance, battery_z], [battery_half, c["battery"]["length_mm"] / 2 + clearance, max(battery_z + c["battery"]["height_mm"] + clearance, upper[2] + 1)], "Battery removal vertically above its contact pads"))
    camera_width = c["camera"]["width_mm"] + 2 * f["camera_side_clearance_mm"]
    for sign in (-1, 1):
        x = sign * (camera_width / 2 + settings["camera_contact_width_mm"] / 2)
        width, length = settings["camera_contact_width_mm"], settings["camera_contact_length_mm"]
        regions.append(_box(f"camera_impact_contact_{sign}", "preserve", [x - width / 2, f["camera_y_mm"] - length / 2, f["cage_height_mm"] - depth], [x + width / 2, f["camera_y_mm"] + length / 2, f["cage_height_mm"]], "Two local protective impact contacts, without a predefined cage", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
        regions.append(_cylinder(f"camera_mount_{sign}", "preserve", [x, f["camera_y_mm"], camera_mount_z(parameters)], 4.0, width, "Local camera screw lug; no prescribed connecting wall", axis="x", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
    regions.append(_cylinder("camera_screw_axis", "forbidden", [0, f["camera_y_mm"], camera_mount_z(parameters)], f["camera_screw_diameter_mm"] / 2, camera_width + 2 * settings["camera_contact_width_mm"] + 20, "Camera side screw and screwdriver access", axis="x", rasterize=False))
    camera_box = components["camera"]["shape"].bounding_box()
    regions.append(_box("camera_front_access", "forbidden", [camera_box.min.X - clearance, camera_box.max.Y, camera_box.min.Z - clearance], [camera_box.max.X + clearance, upper[1] + 1, camera_box.max.Z + clearance], "Unobstructed camera front, insertion and lens field corridor"))
    for name in ("xt30", "balancer"):
        position = placements[name]["position"]
        width, length = c[name]["width_mm"], c[name]["length_mm"]
        regions.append(_box(name + "_contact", "preserve", [position[0] - width / 2, position[1] - length / 2 - depth, 0], [position[0] + width / 2, position[1] + length / 2 + depth, position[2]], "Local connector seat with exposed retention ends for tie or adhesive", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
        regions.append(_box(name + "_plug_access", "forbidden", [position[0] - width / 2 - clearance, position[1] - length / 2 - clearance, position[2]], [position[0] + width / 2 + clearance, position[1] + length / 2 + clearance, upper[2] + 1], "Top insertion and removal corridor for disconnected connector"))
    regions.append(_cylinder("antenna_contact", "preserve", [0, -f["antenna_y_mm"], f["antenna_holder_height_mm"] / 2], f["antenna_bore_mm"] / 2 + 2, f["antenna_holder_height_mm"], "Local antenna retention eyelet, not a prescribed tail", attachment_area_min_mm2=8.0, minimum_wall_mm=2.0))
    regions.append(_cylinder("antenna_bore", "forbidden", [0, -f["antenna_y_mm"], f["antenna_holder_height_mm"] / 2], f["antenna_bore_mm"] / 2, f["antenna_holder_height_mm"] + 2, "VTX antenna bore and axial assembly access", rasterize=False))
    regions.append(_box("aio_side_assembly_access", "forbidden", [0, -c["aio15"]["length_mm"] / 2 - clearance, placements["aio15"]["position"][2]], [upper[0] + 1, c["aio15"]["length_mm"] / 2 + clearance, placements["aio15"]["position"][2] + c["aio15"]["stack_height_mm"] + clearance], "AIO insertion/removal through right side with connectors unplugged"))
    regions.append(_box("balance_lead_routing", "forbidden", [f["connector_offset_x_mm"] - 2, -f["connector_y_mm"], 8], [f["connector_offset_x_mm"] + 2, -c["aio15"]["length_mm"] / 2, 12], "Accessible balance and power lead corridor; provisional connector routing"))
    return regions, placements, components


def _connection_cases(regions, model, force):
    if force <= 0:
        return []
    fixtures = next(case["fixed_regions"] for case in model["load_cases"] if case["name"] == "battery_impact")
    cases = []
    for region in regions:
        name = region["name"]
        if region["role"] != "preserve" or not name.startswith(("aio_contact_", "battery_contact_", "strap_contact_", "camera_mount_", "xt30_contact", "balancer_contact", "antenna_contact")):
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


def build_design_domain(parameters):
    settings = _merge(TOPOLOGY_CONFIG, parameters.get("topology", {}))
    grid = deepcopy(settings["grid"])
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
    subtractions = _preserve_subtractions(regions)
    masks = rasterize_regions(grid, regions)
    _, allowed_components = label(masks["allowed"])
    if allowed_components != 1:
        raise ValueError("The allowed design domain must be face-connected")
    model = prepare_frame_case(parameters)
    tolerance = parameters["integration"]["selection_tolerance_mm"]
    fixture_radius = settings["aio_contact_radius_mm"]
    aio_fixtures = [
        {"kind": "box", "min_mm": [x - fixture_radius, y - fixture_radius, -tolerance], "max_mm": [x + fixture_radius, y + fixture_radius, tolerance]}
        for x, y in mount_positions(parameters)["aio15"]
    ]
    next(case for case in model["load_cases"] if case["name"] == "arm_tip")["fixed_regions"] = aio_fixtures
    model["fixture_model"] = "Arm-tip case: undersides of the four mandatory AIO mounting contacts fixed; other cases: four motor contact undersides fixed. Identical selectors must be used for v0 comparison."
    auxiliary_cases = _connection_cases(regions, model, settings["connection_proof_force_n"])
    weights = {case["name"]: 1.0 for case in model["load_cases"] if case["analysis"] == "static"}
    weights.update({case["name"]: 1.0 / (3 * len(auxiliary_cases)) for case in auxiliary_cases})
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
                "Camera screw position, connector retention and cable envelopes are provisional; final hardware fit requires measurement.",
                "The initial design envelope is a full cuboid; no prescribed arms, central plate, battery walls, cage walls or tail links.",
                "Local contact primitives are human-specified functional interfaces; connecting branches are optimizer variables.",
                "Preserve volume fractions refer to raster cells; exact bores and interface boundaries change final physical volume.",
            ],
        },
    }
