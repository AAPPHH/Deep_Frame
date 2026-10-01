from math import atan2, cos, degrees, hypot, isfinite, radians, sin, sqrt

from build123d import Align, Box, Cylinder, Pos, Rot, Solid

from deep_frame.components import _mount_holes


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


def build_frame(config: dict) -> Solid:
    f = config["frame"]
    c = config["components"]
    if any(not isfinite(value) or value <= 0 for value in f.values()):
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
