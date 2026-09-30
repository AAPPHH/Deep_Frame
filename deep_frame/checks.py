from itertools import combinations
from math import isfinite

import numpy as np
from build123d import CenterOf, Compound, GeomType
from OCP.BRep import BRep_Tool

from deep_frame.components import build_components
from deep_frame.frame import assembly_placements, box_at, cylinder_at, mount_positions, structural_margins


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
        contact = first == "frame" and second in contacts
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
    from deep_frame.geometry import build_geometry

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
