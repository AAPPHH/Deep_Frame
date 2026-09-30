from deep_frame.checks import collision_and_clearance, intersection_shape
from deep_frame.components import build_components
from deep_frame.frame import assembly_placements
from deep_frame.geometry import build_geometry


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
