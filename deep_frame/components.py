from math import cos, hypot, isfinite, radians, sin

from build123d import Align, Axis, Box, CenterOf, Cylinder, Location, Vector, Vertex


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
