from copy import deepcopy
from itertools import combinations
from math import atan2, cos, degrees, hypot, isfinite, radians, sin, sqrt
from pathlib import Path

import numpy as np
from build123d import Align, Axis, Box, CenterOf, Compound, Cylinder, GeomType, Location, Part, Pos, Rot, Solid, Vector, Vertex, export_stl
from OCP.BRep import BRep_Tool

from deep_frame.config import CONFIG

def build_smoke_body(config: dict) -> Part:
    dimensions = tuple(config[key] for key in ("length_mm", "width_mm", "thickness_mm"))
    if not all(isfinite(value) and value > 0 for value in dimensions):
        raise ValueError("Dimensions must be finite and greater than zero.")
    return Box(*dimensions, align=(Align.CENTER, Align.CENTER, Align.MIN))

def export_body(body: Part, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(body, path):
        raise RuntimeError(f"STL export failed: {path}")
    return path

def _positive(spec: dict, *keys: str) -> None:
    if not all(isfinite(spec[key]) and spec[key] > 0 for key in keys):
        raise ValueError(f"Component dimensions must be finite and positive: {keys}")

def _box(spec: dict, height_key: str = "height_mm"):
    _positive(spec, "width_mm", "length_mm", height_key)
    return Box(
        spec["width_mm"],
        spec["length_mm"],
        spec[height_key],
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

def _mount_holes(spec: dict) -> list[tuple[float, float, float]]:
    _positive(spec, "mount_pitch_mm", "screw_diameter_mm")
    half_pitch = spec["mount_pitch_mm"] / 2
    layout = spec.get("mount_layout", "square")
    if layout == "bolt_circle":
        return [(half_pitch * cos(radians(angle)), half_pitch * sin(radians(angle)), 0.0) for angle in (45, 135, 225, 315)]
    if layout != "square":
        raise ValueError("Mount layout must be square or bolt_circle.")
    return [(x, y, 0.0) for x in (-half_pitch, half_pitch) for y in (-half_pitch, half_pitch)]

def _drill(shape, holes: list, diameter: float, height: float):
    for hole in holes:
        tool = Cylinder(diameter / 2, height + 2, align=(Align.CENTER, Align.CENTER, Align.MIN))
        shape = shape - tool.moved(Location((hole[0], hole[1], -1.0)))
    return shape

def _record(shape, spec: dict, kind: str, holes: list | None = None) -> dict:
    mass = spec["mass_g"]
    if not isfinite(mass) or mass < 0:
        raise ValueError("Component mass must be finite and nonnegative.")
    return {
        "shape": shape,
        "mass_g": mass,
        "kind": kind,
        "mount_holes": holes or [],
        "center_of_mass_mm": tuple(shape.center(CenterOf.MASS)),
        "source": spec["source"],
        "parameter_sources": spec.get("parameter_sources", {}).copy(),
        "mass_scope": spec.get("mass_scope", "component"),
    }

def build_component_prototypes(config: dict) -> dict:
    specs = config["components"]
    aio = specs["aio15"]
    aio_shape = _box(aio, "stack_height_mm")
    aio_holes = _mount_holes(aio)
    if aio["mount_pitch_mm"] + aio["screw_diameter_mm"] >= min(aio["width_mm"], aio["length_mm"]):
        raise ValueError("AIO mounting holes must remain within the board envelope.")
    aio_shape = _drill(aio_shape, aio_holes, aio["screw_diameter_mm"], aio["stack_height_mm"])
    camera = specs["camera"]
    camera_shape = _box(camera)
    if not isfinite(camera["tilt_deg"]):
        raise ValueError("Camera tilt must be finite.")
    camera_shape = camera_shape.rotate(
        Axis((0, 0, camera["height_mm"] / 2), (1, 0, 0)), camera["tilt_deg"]
    )
    camera_shape = camera_shape.translate(Vector(0, 0, -camera_shape.bounding_box().min.Z))
    motor = specs["motor"]
    _positive(motor, "diameter_mm", "height_mm", "screw_clearance_mm")
    motor_holes = _mount_holes(motor)
    if 2 * max(hypot(x, y) for x, y, _ in motor_holes) + motor["screw_clearance_mm"] >= motor["diameter_mm"]:
        raise ValueError("Motor mounting holes must remain within the motor envelope.")
    if motor["screw_clearance_mm"] < motor["screw_diameter_mm"]:
        raise ValueError("Motor screw clearance must fit the screw diameter.")
    motor_shape = Cylinder(motor["diameter_mm"] / 2, motor["height_mm"], align=(Align.CENTER, Align.CENTER, Align.MIN))
    motor_shape = _drill(motor_shape, motor_holes, motor["screw_clearance_mm"], motor["height_mm"])
    prop = specs["prop"]
    _positive(prop, "diameter_mm", "thickness_mm")
    prop_shape = Cylinder(prop["diameter_mm"] / 2, prop["thickness_mm"], align=(Align.CENTER, Align.CENTER, Align.MIN))
    if specs["balancer"]["pins"] != 3:
        raise ValueError("The 2S balance connector must have three pins.")
    result = {
        "aio15": _record(aio_shape, aio, "aio15", aio_holes),
        "camera": _record(camera_shape, camera, "camera"),
        "battery": _record(_box(specs["battery"]), specs["battery"], "battery"),
        "motor": _record(motor_shape, motor, "motor", motor_holes),
        "prop": _record(prop_shape, prop, "prop"),
        "xt30": _record(_box(specs["xt30"]), specs["xt30"], "connector"),
        "balancer": _record(_box(specs["balancer"]), specs["balancer"], "connector"),
    }
    result["camera"]["tilt_deg"] = camera["tilt_deg"]
    return result

def build_components(config: dict, placements: dict | None = None) -> dict:
    prototypes = build_component_prototypes(config)
    if placements is None:
        placements = config.get("placements", {name: {"prototype": name} for name in prototypes})
    result = {}
    for name, placement in placements.items():
        prototype = prototypes[placement["prototype"]]
        position = placement.get("position", (0, 0, 0))
        rotation = placement.get("rotation", (0, 0, 0))
        if len(position) != 3 or len(rotation) != 3 or not all(isfinite(value) for value in (*position, *rotation)):
            raise ValueError("Placement position and rotation must contain three finite numbers.")
        location = Location(position, rotation)
        shape = prototype["shape"].moved(location)
        result[name] = prototype | {
            "shape": shape,
            "mount_holes": [tuple(Vertex(*hole).moved(location).center()) for hole in prototype["mount_holes"]],
            "center_of_mass_mm": tuple(shape.center(CenterOf.MASS)),
            "position_mm": tuple(position),
            "rotation_deg": tuple(rotation),
        }
    return result

def motor_positions(config: dict) -> dict:
    frame = config["frame"]
    ratio = frame["lateral_longitudinal_ratio"]
    longitudinal = frame["wheelbase_mm"] / sqrt(1 + ratio**2)
    lateral = longitudinal * ratio
    return {
        "front_left": (-lateral / 2, longitudinal / 2),
        "front_right": (lateral / 2, longitudinal / 2),
        "rear_left": (-lateral / 2, -longitudinal / 2),
        "rear_right": (lateral / 2, -longitudinal / 2),
    }

def mount_positions(config: dict) -> dict:
    components = config["components"]
    aio_pitch = components["aio15"]["mount_pitch_mm"]
    holes = {
        "aio15": [(sx * aio_pitch / 2, sy * aio_pitch / 2) for sx in (-1, 1) for sy in (-1, 1)]
    }
    for name, (x, y) in motor_positions(config).items():
        holes[name] = [(x + dx, y + dy) for dx, dy, _ in _mount_holes(components["motor"])]
    return holes

def box_at(width, length, height, x=0, y=0, z=0):
    return Pos(x, y, z) * Box(width, length, height, align=(Align.CENTER, Align.CENTER, Align.MIN))

def cylinder_at(radius, height, x=0, y=0, z=0):
    return Pos(x, y, z) * Cylinder(radius, height, align=(Align.CENTER, Align.CENTER, Align.MIN))

def camera_mount_z(config: dict) -> float:
    camera = config["components"]["camera"]
    tilt = radians(camera["tilt_deg"])
    extent = abs(camera["height_mm"] * cos(tilt)) + abs(camera["length_mm"] * sin(tilt))
    return config["frame"]["base_thickness_mm"] + config["frame"]["camera_bottom_clearance_mm"] + extent / 2

def structural_margins(config: dict) -> dict:
    f = config["frame"]
    c = config["components"]
    motor_hole_radius = (c["motor"]["screw_diameter_mm"] + f["hole_clearance_mm"]) / 2
    hole_radius = hypot(*_mount_holes(c["motor"])[0][:2])
    return {
        "base": f["base_thickness_mm"],
        "arm_height": f["arm_height_mm"],
        "arm_width_half": f["arm_width_mm"] / 2,
        "deck": f["deck_thickness_mm"],
        "walls": f["minimum_wall_mm"],
        "motor_shaft_web": hole_radius - motor_hole_radius - f["motor_shaft_hole_mm"] / 2,
        "motor_outer_web": f["motor_pad_radius_mm"] - hole_radius - motor_hole_radius,
        "aio_window_web": sqrt(2) * (c["aio15"]["mount_pitch_mm"] - f["base_window_mm"]) / 2 - (c["aio15"]["screw_diameter_mm"] + f["hole_clearance_mm"]) / 2,
        "wall_top_web": f["deck_top_mm"] - f["deck_thickness_mm"] - f["wall_window_bottom_mm"] - f["wall_window_height_mm"],
        "wall_bottom_web": f["wall_window_bottom_mm"] - f["base_thickness_mm"],
        "wall_end_web": (f["support_length_mm"] - f["wall_window_length_mm"]) / 2,
        "deck_window_side_web": (f["deck_width_mm"] - f["deck_window_width_mm"]) / 2,
        "deck_window_end_web": (f["deck_length_mm"] - f["deck_window_length_mm"]) / 2,
        "strap_outer_web": f["deck_width_mm"] / 2 - f["strap_slot_x_mm"] - f["strap_slot_width_mm"] / 2,
        "strap_inner_web": f["strap_slot_x_mm"] - f["strap_slot_width_mm"] / 2 - f["deck_window_width_mm"] / 2,
        "strap_end_web": f["deck_length_mm"] / 2 - f["strap_slot_y_mm"] - f["strap_slot_length_mm"] / 2,
    }

NONNEGATIVE_FRAME_KEYS = ("prop_motor_gap_mm",)

def build_frame(config: dict) -> Solid:
    f = config["frame"]
    c = config["components"]
    if any(not isfinite(value) or value < 0 or (value == 0 and key not in NONNEGATIVE_FRAME_KEYS) for key, value in f.items()):
        raise ValueError("Frame dimensions must be finite and positive.")
    wall = f["minimum_wall_mm"]
    base = f["base_thickness_mm"]
    deck_z = f["deck_top_mm"] - f["deck_thickness_mm"]
    if min(structural_margins(config).values()) < wall - 1e-7:
        raise ValueError("Structural dimensions violate the minimum wall thickness.")
    if f["deck_width_mm"] < c["battery"]["width_mm"] + 2 * f["battery_margin_mm"]:
        raise ValueError("The battery deck is too narrow.")
    if f["deck_width_mm"] - 2 * wall < c["aio15"]["width_mm"] + 2 * f["component_clearance_mm"]:
        raise ValueError("The AIO does not fit between the deck walls.")
    if deck_z <= base + f["aio_standoff_mm"] + c["aio15"]["stack_height_mm"] + f["component_clearance_mm"]:
        raise ValueError("The deck leaves insufficient AIO stack space.")
    frame = box_at(f["body_width_mm"], f["body_length_mm"], base)
    frame -= box_at(f["base_window_mm"], f["base_window_mm"], base + 2, z=-1)
    for x, y in motor_positions(config).values():
        root_x = (1 if x > 0 else -1) * f["arm_root_mm"]
        root_y = (1 if y > 0 else -1) * f["arm_root_mm"]
        length = hypot(x - root_x, y - root_y)
        angle = degrees(atan2(y - root_y, x - root_x))
        arm = Pos((x + root_x) / 2, (y + root_y) / 2, 0) * Rot(0, 0, angle) * Box(length, f["arm_width_mm"], f["arm_height_mm"], align=(Align.CENTER, Align.CENTER, Align.MIN))
        frame += arm + cylinder_at(f["motor_pad_radius_mm"], f["arm_height_mm"], x, y)
    for x, y in mount_positions(config)["aio15"]:
        radius = c["aio15"]["screw_diameter_mm"] / 2 + f["hole_clearance_mm"] / 2 + wall
        frame += cylinder_at(radius, base + f["aio_standoff_mm"], x, y)
    wall_x = f["deck_width_mm"] / 2 - wall / 2
    for sign in (-1, 1):
        support = box_at(wall, f["support_length_mm"], deck_z - base, sign * wall_x, z=base)
        support -= box_at(wall + 2, f["wall_window_length_mm"], f["wall_window_height_mm"], sign * wall_x, z=f["wall_window_bottom_mm"])
        frame += support
    deck = box_at(f["deck_width_mm"], f["deck_length_mm"], f["deck_thickness_mm"], z=deck_z)
    deck -= box_at(f["deck_window_width_mm"], f["deck_window_length_mm"], f["deck_thickness_mm"] + 2, z=deck_z - 1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            deck -= box_at(f["strap_slot_width_mm"], f["strap_slot_length_mm"], f["deck_thickness_mm"] + 2, sx * f["strap_slot_x_mm"], sy * f["strap_slot_y_mm"], deck_z - 1)
    frame += deck
    camera_width = c["camera"]["width_mm"] + 2 * f["camera_side_clearance_mm"]
    cage_width = camera_width + 2 * wall
    camera_y = f["camera_y_mm"]
    frame += box_at(cage_width, f["cage_length_mm"], base, y=camera_y)
    for sign in (-1, 1):
        frame += box_at(wall, f["cage_length_mm"], f["cage_height_mm"], sign * (camera_width + wall) / 2, camera_y)
    for sign in (-1, 1):
        frame += box_at(cage_width, wall, wall, y=camera_y + sign * (f["cage_length_mm"] - wall) / 2, z=f["cage_height_mm"] - wall)
    camera_mount = Pos(0, camera_y, camera_mount_z(config)) * Rot(0, 90, 0) * Cylinder(f["camera_screw_diameter_mm"] / 2, cage_width + 2)
    frame -= camera_mount
    tail = box_at(f["tail_width_mm"], f["tail_length_mm"], base, y=-f["tail_y_mm"])
    tail -= box_at(f["tail_window_width_mm"], f["tail_window_length_mm"], base + 2, y=-f["tail_window_y_mm"], z=-1)
    frame += tail
    for name, x in (("xt30", -f["connector_offset_x_mm"]), ("balancer", f["connector_offset_x_mm"])):
        connector = c[name]
        inner_width = connector["width_mm"] + 2 * f["connector_clearance_mm"]
        length = connector["length_mm"] + 2 * wall
        for sign in (-1, 1):
            frame += box_at(wall, length, f["connector_holder_height_mm"], x + sign * (inner_width + wall) / 2, -f["connector_y_mm"])
        frame += box_at(inner_width + 2 * wall, wall, f["connector_holder_height_mm"], x, -f["connector_y_mm"] + (length - wall) / 2)
    antenna = box_at(f["antenna_bore_mm"] + 2 * wall, f["antenna_bore_mm"] + 2 * wall, f["antenna_holder_height_mm"], y=-f["antenna_y_mm"])
    frame += antenna
    frame -= cylinder_at(f["antenna_bore_mm"] / 2, f["antenna_holder_height_mm"] + 2, y=-f["antenna_y_mm"], z=-1)
    frame -= box_at(f["cable_slot_width_mm"], wall + 2, f["cable_slot_height_mm"], f["connector_offset_x_mm"], -f["connector_y_mm"] + c["balancer"]["length_mm"] / 2 + wall / 2, base)
    for name, positions in mount_positions(config).items():
        screw = c["aio15"]["screw_diameter_mm"] if name == "aio15" else c["motor"]["screw_diameter_mm"]
        for x, y in positions:
            frame -= cylinder_at((screw + f["hole_clearance_mm"]) / 2, base + f["aio_standoff_mm"] + f["arm_height_mm"] + 2, x, y, -1)
    for x, y in motor_positions(config).values():
        frame -= cylinder_at(f["motor_shaft_hole_mm"] / 2, f["arm_height_mm"] + 2, x, y, -1)
    if not frame.is_valid or len(frame.solids()) != 1:
        raise ValueError("The frame must be one valid solid.")
    return frame.solids()[0]

def assembly_placements(config: dict) -> dict:
    f = config["frame"]
    c = config["components"]
    placements = {
        "aio15": {"prototype": "aio15", "position": (0, 0, f["base_thickness_mm"] + f["aio_standoff_mm"])},
        "camera": {"prototype": "camera", "position": (0, f["camera_y_mm"], f["base_thickness_mm"] + f["camera_bottom_clearance_mm"])},
        "battery": {"prototype": "battery", "position": (0, 0, f["deck_top_mm"])},
        "xt30": {"prototype": "xt30", "position": (-f["connector_offset_x_mm"], -f["connector_y_mm"], f["base_thickness_mm"])},
        "balancer": {"prototype": "balancer", "position": (f["connector_offset_x_mm"], -f["connector_y_mm"], f["base_thickness_mm"])},
    }
    for name, (x, y) in motor_positions(config).items():
        placements[f"motor_{name}"] = {"prototype": "motor", "position": (x, y, f["arm_height_mm"])}
        placements[f"prop_{name}"] = {"prototype": "prop", "position": (x, y, f["arm_height_mm"] + c["motor"]["height_mm"] + f["prop_motor_gap_mm"])}
    return placements

def intersection_shape(first, second):
    intersection = first.intersect(second)
    if isinstance(intersection, list):
        return Compound(children=intersection)
    return intersection

def assembly_mass_properties(frame, components: dict, density_g_cm3: float) -> dict:
    if not isfinite(density_g_cm3) or density_g_cm3 <= 0:
        raise ValueError("Material density must be finite and positive.")
    entries = {"frame": {"shape": frame, "mass_g": frame.volume * density_g_cm3 / 1000}} | components
    masses = {name: float(entry["mass_g"]) for name, entry in entries.items()}
    total = sum(masses.values())
    centers = {name: np.array(tuple(entry["shape"].center(CenterOf.MASS))) for name, entry in entries.items()}
    center = sum(masses[name] * centers[name] for name in entries) / total
    inertia = np.zeros((3, 3))
    for name, entry in entries.items():
        mass = masses[name]
        offset = centers[name] - center
        inertia += np.asarray(entry["shape"].matrix_of_inertia) * mass / entry["shape"].volume
        inertia += mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return {
        "mass_g": total,
        "frame_mass_g": masses["frame"],
        "component_masses_g": {name: mass for name, mass in masses.items() if name != "frame"},
        "center_of_mass_mm": center.tolist(),
        "inertia_tensor_g_mm2": inertia.tolist(),
        "inertia_about": "assembly center of mass, global x/y/z axes",
        "model": "exact CAD volume integrals, homogeneous component equivalent envelopes, parallel-axis theorem",
    }

def battery_prop_overlap(config: dict, components: dict) -> dict:
    battery = config["components"]["battery"]
    center = components["battery"]["center_of_mass_mm"]
    footprint = box_at(battery["width_mm"], battery["length_mm"], 1, center[0], center[1])
    radius = config["components"]["prop"]["diameter_mm"] / 2
    disks = [cylinder_at(radius, 1, *part["center_of_mass_mm"][:2]) for part in components.values() if part["kind"] == "prop"]
    union = disks[0]
    for disk in disks[1:]:
        union += disk
    intersection = intersection_shape(footprint, union)
    area = 0.0 if intersection is None else float(intersection.volume)
    return {
        "area_mm2": area,
        "percent_of_battery_area": 100 * area / footprint.volume,
        "battery_area_mm2": float(footprint.volume),
        "definition": "union of all prop swept disks projected along z, divided by battery top-view area",
    }

def collision_and_clearance(config: dict, frame, components: dict) -> dict:
    shapes = {"frame": frame} | {name: part["shape"] for name, part in components.items()}
    contacts = {"aio15", "battery", "xt30", "balancer"} | {name for name in components if name.startswith("motor_")}
    settings = config["checks"]
    distances = {}
    collisions = []
    violations = []
    for first, second in combinations(shapes, 2):
        name = first + ":" + second
        distance = float(shapes[first].distance_to(shapes[second]))
        volume = 0.0
        if distance <= settings["distance_tolerance_mm"]:
            overlap = intersection_shape(shapes[first], shapes[second])
            volume = 0.0 if overlap is None else float(overlap.volume)
        contact = first == "frame" and second in contacts or first.startswith("motor_") and second == "prop_" + first[len("motor_"):]
        required = 0.0 if contact else settings["prop_clearance_mm"] if first.startswith("prop_") or second.startswith("prop_") else settings["minimum_clearance_mm"]
        collided = volume > settings["intersection_tolerance_mm3"]
        passed = not collided and distance + settings["distance_tolerance_mm"] >= required
        distances[name] = {"distance_mm": distance, "required_mm": required, "intersection_mm3": volume, "planned_support_contact": contact, "passed": passed}
        if collided:
            collisions.append({"parts": [first, second], "intersection_mm3": volume})
            violations.append("collision:" + name)
        elif not passed:
            violations.append("clearance:" + name)
    return {"passed": not violations, "pairs": distances, "collisions": collisions, "violations": violations}

def mount_checks(config: dict, frame) -> dict:
    expected = mount_positions(config)
    surfaces = [face for face in frame.faces() if face.geom_type == GeomType.CYLINDER and face.radius is not None and abs(face.axis_of_rotation.direction.Z) > 0.999]
    groups = {}
    for name, positions in expected.items():
        spec = config["components"]["aio15" if name == "aio15" else "motor"]
        radius = (spec["screw_diameter_mm"] + config["frame"]["hole_clearance_mm"]) / 2
        checks = []
        for x, y in positions:
            matching = [face for face in surfaces if abs(face.radius - radius) < 1e-6 and np.linalg.norm(np.array([face.axis_of_rotation.position.X - x, face.axis_of_rotation.position.Y - y])) < 1e-6]
            checks.append({"position_mm": [x, y], "diameter_mm": 2 * radius, "passed": bool(matching)})
        groups[name] = checks
    return {"passed": all(check["passed"] for group in groups.values() for check in group), "groups": groups}

def check_assembly(config: dict) -> dict:
    try:
        frame = build_geometry(config)
        components = build_components(config, assembly_placements(config))
        topology = {"solid_count": len(frame.solids()), "valid": bool(frame.is_valid), "closed": all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in frame.shells())}
        topology["passed"] = topology["solid_count"] == 1 and topology["valid"] and topology["closed"]
        mounts = mount_checks(config, frame)
        clearances = collision_and_clearance(config, frame, components)
        overlap = battery_prop_overlap(config, components)
        overlap["maximum_percent"] = config["checks"]["maximum_battery_prop_overlap_percent"]
        overlap["passed"] = overlap["percent_of_battery_area"] <= overlap["maximum_percent"] + 1e-7
        walls = structural_margins(config)
        minimum_wall = config["frame"]["minimum_wall_mm"]
        wall_checks = {"minimum_wall_mm": minimum_wall, "margins_mm": walls, "passed": min(walls.values()) >= minimum_wall - 1e-7}
        checks = {"topology": topology, "mounts": mounts, "clearances": clearances, "battery_prop_overlap": overlap, "structural_walls": wall_checks, "mass_properties": assembly_mass_properties(frame, components, config["material"]["density_g_cm3"])}
        violations = clearances["violations"].copy()
        violations.extend(name for name, check in checks.items() if "passed" in check and not check["passed"] and name != "clearances")
        return {"passed": not violations, "violations": violations, "checks": checks}
    except (ValueError, KeyError, TypeError) as error:
        return {"passed": False, "violations": ["geometry:" + str(error)], "checks": {"error": str(error)}}

def reference_parameters() -> dict:
    return deepcopy({key: value for key, value in CONFIG.items() if key != "stl_path"})

def build_geometry(parameters: dict) -> Solid:
    return build_frame(parameters)

def validate_geometry(parameters: dict) -> dict:
    return check_assembly(parameters)

def assembly_scene(parameters: dict) -> dict:
    frame = build_geometry(parameters)
    components = build_components(parameters, assembly_placements(parameters))
    by_name = {"frame": frame} | {name: part["shape"] for name, part in components.items()}
    results = collision_and_clearance(parameters, frame, components)
    palette = {"frame": "#91a5b8", "aio15": "#237a44", "camera": "#42373b", "battery": "#4288bd", "xt30": "#efc528", "balancer": "#eeeeee"}
    names = list(by_name)
    shapes = list(by_name.values())
    colors = [palette.get(name, "#cccccc" if name.startswith("prop_") else "#383838") for name in names]
    alphas = [0.25 if name.startswith("prop_") else 0.75 if name == "frame" else 1.0 for name in names]
    for collision in results["collisions"]:
        first, second = collision["parts"]
        shapes.append(intersection_shape(by_name[first], by_name[second]))
        names.append("COLLISION " + first + " / " + second)
        colors.append("#ff2020")
        alphas.append(1.0)
    return {"shapes": shapes, "names": names, "colors": colors, "alphas": alphas, "collisions": results["collisions"]}

def show_assembly(parameters: dict) -> dict:
    from ocp_vscode import port_check, show
    port = parameters["viewer_port"]
    if not port_check(port):
        raise ConnectionError(f"Start OCP CAD Viewer in VS Code on port {port}.")
    scene = assembly_scene(parameters)
    show(*scene["shapes"], names=scene["names"], colors=scene["colors"], alphas=scene["alphas"], port=port, progress="", timeit=False)
    return {"shown": len(scene["shapes"]), "collisions": scene["collisions"]}
