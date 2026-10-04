import json
import math
import re
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
from scipy import ndimage

from deep_frame.config import COMPONENT_DEFAULTS, EVALUATION_CONFIG, EVALUATION_KINDS, FEA_CONFIG, FRAME_DEFAULTS, IMPLICIT_CONFIG, PRINT_MATERIAL, TOPOLOGY_CONFIG, configure
from deep_frame.fea import MESH_KEYS, clean_slivers, evaluate, graded_surface, print_axes, robust_surface
from deep_frame.topology_geometry import region_contains

MOTORS = ("front_left", "front_right", "rear_left", "rear_right")
CRASH = ("crash_front", "crash_arm", "crash_back")

def box(low, high):
    return {"kind": "box", "min_mm": [float(v) for v in low], "max_mm": [float(v) for v in high]}

def shrink(region, margin):
    if region["kind"] == "box":
        return {**region, "min_mm": [v + margin for v in region["min_mm"]], "max_mm": [v - margin for v in region["max_mm"]]}
    return {**region, "radius_mm": region["radius_mm"] - margin, "height_mm": region["height_mm"] - 2 * margin}

def load_frame(spec):
    mesh = trimesh.load_mesh(spec["stl"], process=True)
    if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0):
        raise ValueError(f"{spec['stl']} is not a closed, consistently oriented mesh")
    return mesh

def frame_spec(overrides):
    spec = configure(EVALUATION_CONFIG, EVALUATION_KINDS, overrides, ("name", "stl", "output"))
    spec.update({key: {**EVALUATION_CONFIG[key], **spec[key]} for key in ("loads", "fea_settings", "fea_surface", "slicer")})
    if spec["ours"] or spec["domain"]:
        spec = {**spec, **ours(spec)}
    missing = [key for key in ("motors", "selectors") if not spec[key]]
    if missing:
        raise ValueError("frame.json needs " + ", ".join(missing) + " or ours=true")
    spec["motor_up"] = {name: spec["motor_up"].get(name, 1) for name in MOTORS}
    return spec

def ours(spec):
    from deep_frame.topology_geometry import build_design_domain
    from tools.topology_study import study_parameters
    stored = json.loads(Path(spec["domain"]).read_text(encoding="utf-8")) if spec["domain"] else None
    domain = stored.get("domain", stored) if stored else build_design_domain(study_parameters(spec["domain_grid"]))
    regions = {region["name"]: region for region in domain["regions"]}
    placements = domain["metadata"]["components"]
    motors = {name: [*placements["motor_" + name]["position_mm"]] for name in MOTORS}
    cases = {case["name"]: case for case in domain["comparison_load_cases"]}
    battery = domain["point_masses"][0]
    pitch = COMPONENT_DEFAULTS["motor"]["mount_pitch_mm"]
    patterns = [{"name": f"motor_{name}", "center_mm": xyz[:2], "z_mm": xyz[2] - 1.0, "radius_mm": pitch / 2, "count": 4, "hole_diameter_mm": [1.6, 4.0], "screw_diameter_mm": 2.0, "tool_direction": [0, 0, -1]} for name, xyz in motors.items()]
    aio = [regions[f"aio_screw_{index}"]["center_mm"] for index in range(4)]
    patterns.append({"name": "stack_25.5", "center_mm": np.mean(aio, axis=0)[:2].tolist(), "z_mm": 1.5, "radius_mm": float(np.hypot(*np.subtract(aio[0], np.mean(aio, axis=0))[:2])), "count": 4, "hole_diameter_mm": [1.6, 4.0], "screw_diameter_mm": 2.0, "tool_direction": [0, 0, -1]})
    components = [{"type": entry["prototype"] if "prototype" in entry else name.split("_")[0], "name": name, "center_mm": entry["center_of_mass_mm"]} for name, entry in placements.items() if entry["mass_g"] > 0]
    connectors = [{"name": name, "position_mm": placements[name]["center_of_mass_mm"], "direction": [0, 0, 1]} for name in ("xt30", "balancer") if name + "_contact" in regions]
    keep_outs = [region for region in domain["regions"] if region["role"] == "forbidden"]
    selectors = {"center_fixtures": cases["arm_tip"]["fixed_regions"], "motor_fixtures": cases["modes"]["fixed_regions"], "arm_tip": cases["arm_tip"]["loads"][0]["region"], "arm_motor": "front_left",
                 "camera": cases["camera_side"]["loads"][0]["region"], "deck": battery["attachment_region"], "battery_center_mm": battery["position_mm"], "battery_mass_g": battery["mass_g"]}
    return {"motors": motors, "prop_diameter_mm": COMPONENT_DEFAULTS["prop"]["diameter_mm"], "mount_patterns": patterns, "components": components, "connectors": connectors, "keep_outs": keep_outs,
            "selectors": selectors, "wall_zones": [region for region in domain["regions"] if region["role"] == "preserve" and "motor" in region["name"]]}

def component_list(spec):
    sizes = {"battery": ("width_mm", "length_mm", "height_mm"), "camera": ("width_mm", "length_mm", "height_mm"), "aio15": ("width_mm", "length_mm", "stack_height_mm"), "motor": ("diameter_mm", "diameter_mm", "height_mm")}
    items = [dict(item) for item in spec["components"]]
    if not any(item["type"] == "motor" for item in items):
        motor, gap = COMPONENT_DEFAULTS["motor"], FRAME_DEFAULTS["prop_motor_gap_mm"]
        for name, (x, y, z) in spec["motors"].items():
            up = spec["motor_up"][name]
            items += [{"type": "motor", "name": "motor_" + name, "center_mm": [x, y, z + up * motor["height_mm"] / 2]}, {"type": "prop", "name": "prop_" + name, "center_mm": [x, y, z + up * (motor["height_mm"] + gap)]}]
    result = []
    for item in items:
        base = COMPONENT_DEFAULTS[item["type"]]
        if item["type"] == "prop":
            size, shape = [spec["prop_diameter_mm"], spec["prop_diameter_mm"], base["thickness_mm"]], "disc"
        else:
            size, shape = [base[key] for key in sizes[item["type"]]], "box"
        result.append({"name": item.get("name", item["type"]), "type": item["type"], "mass_g": float(item.get("mass_g", base["mass_g"])), "center_mm": [float(v) for v in item["center_mm"]], "size_mm": item.get("size_mm", size), "shape": item.get("shape", shape)})
    return result

def component_inertia(component):
    mass, (a, b, c) = component["mass_g"], component["size_mm"]
    if component["shape"] == "disc":
        radius = a / 2
        return mass * np.diag([radius ** 2 / 4 + c ** 2 / 12] * 2 + [radius ** 2 / 2])
    return mass / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])

def mass_properties(mesh, components, density):
    bodies = [(mesh.volume * density / 1000, np.asarray(mesh.center_mass), np.asarray(mesh.moment_inertia) * density / 1000)]
    bodies += [(item["mass_g"], np.asarray(item["center_mm"]), component_inertia(item)) for item in components]
    total = sum(mass for mass, _, _ in bodies)
    center = sum(mass * position for mass, position, _ in bodies) / total
    inertia = sum(own + mass * (np.dot(position - center, position - center) * np.eye(3) - np.outer(position - center, position - center)) for mass, position, own in bodies)
    return {"frame_mass_g": bodies[0][0], "frame_volume_mm3": float(mesh.volume), "total_mass_g": float(total), "component_mass_g": float(total - bodies[0][0]), "center_of_mass_mm": center.tolist(),
            "frame_center_of_mass_mm": bodies[0][1].tolist(), "inertia_g_mm2": inertia.tolist(), "principal_inertia_g_mm2": np.linalg.eigvalsh(inertia).tolist(), "components": components,
            "model": "frame: exact mesh volume integrals x density; components: homogeneous boxes (props: discs) with parallel-axis theorem about the total centre of mass"}

def voxel_grid(mesh, h):
    from deep_frame.topology_implicit_validation import occupancy
    solid, lower, unbalanced = occupancy(mesh, h, 2 * h)
    return solid, lower, unbalanced

def top_view(solid, lower, h, spec, hub):
    silhouette = solid.any(axis=2)
    x = lower[0] + (np.arange(silhouette.shape[0]) + 0.5) * h
    y = lower[1] + (np.arange(silhouette.shape[1]) + 0.5) * h
    X, Y = np.meshgrid(x, y, indexing="ij")
    from scipy.spatial import ConvexHull
    points = np.column_stack((X[silhouette], Y[silhouette]))
    area, radius = silhouette.sum() * h * h, spec["prop_diameter_mm"] / 2
    distance = np.min([np.hypot(X - mx, Y - my) for mx, my, _ in spec["motors"].values()], axis=0)
    centres = np.asarray([xyz[:2] for xyz in spec["motors"].values()])
    spacing = min(np.linalg.norm(a - b) for index, a in enumerate(centres) for b in centres[index + 1:])
    disc, ring = distance <= radius, (distance <= radius) & (distance > hub)
    return {"projected_area_mm2": float(area), "bbox_share": float(area / np.prod(points.max(0) - points.min(0) + h)), "hull_share": float(area / ConvexHull(points).volume),
            "prop_disc_share": float((silhouette & disc).sum() * h * h / (len(centres) * math.pi * radius ** 2)), "prop_ring_share": float((silhouette & ring).sum() * h * h / (len(centres) * math.pi * (radius ** 2 - hub ** 2))),
            "prop_diameter_mm": spec["prop_diameter_mm"], "hub_radius_mm": hub, "discs_overlap": bool(spacing < 2 * radius),
            "method": "top-view silhouette of the winding-number voxel grid; prop share = material inside the four prop discs / disc area, ring share excludes the motor hub radius"}

def strut_form(solid, h):
    from skimage.morphology import skeletonize
    skeleton = skeletonize(solid) > 0
    widths = 2 * ndimage.distance_transform_edt(solid, sampling=h)[skeleton] - h
    planar = 2 * ndimage.distance_transform_edt(solid, sampling=(h, h, 1e6))[skeleton]
    levels = np.arange(solid.shape[2], dtype=np.int32)
    below = np.maximum.accumulate(np.where(solid, -1, levels[None, None, :]), axis=2)
    above = np.flip(np.minimum.accumulate(np.flip(np.where(solid, solid.shape[2], levels[None, None, :]), axis=2), axis=2), axis=2)
    ratio = ((above - below - 1) * h)[skeleton] / np.maximum(planar, h)
    percentiles = lambda values: {f"p{q}": float(np.percentile(values, q)) for q in (10, 50, 90)}
    return {"strut_width_mm": percentiles(widths), "share_below_2mm": float(np.mean(widths < 2.0)), "section_ratio": percentiles(ratio), "upright_share": float(np.mean(ratio > 1.5)), "flat_share": float(np.mean(ratio < 1 / 1.5)),
            "skeleton_voxels": int(skeleton.sum()), "method": "widths 2 x EDT on the 3D skeleton minus one voxel; section ratio = vertical material run / in-plane width at each skeleton voxel (> 1 upright, < 1 flat)"}

def loops(solid, h, radius):
    from skimage.measure import euler_number
    pad = int(math.ceil(radius / h)) + 2
    padded = np.pad(solid, pad)
    closed = ndimage.distance_transform_edt(~(ndimage.distance_transform_edt(~padded, sampling=h) <= radius), sampling=h) > radius
    bodies = ndimage.label(closed, structure=np.ones((3, 3, 3)))[1]
    cavities = ndimage.label(~closed)[1] - 1
    return {"closing_radius_mm": radius, "voxel_mm": h, "bodies": int(bodies), "cavities": int(cavities), "loops": int(bodies + cavities - euler_number(closed, connectivity=3)),
            "method": "first Betti number b1 = b0 + b2 - chi of the voxel solid after a morphological closing that fills bores and slots up to 2 x the closing radius"}

def symmetry(mesh, plane_x, samples, seed):
    points = trimesh.sample.sample_surface(mesh, samples, seed=seed)[0]
    mirrored = points * [-1, 1, 1] + [2 * plane_x, 0, 0]
    distance = trimesh.proximity.closest_point(mesh, mirrored)[1]
    return {"plane_x_mm": plane_x, "rms_mm": float(np.sqrt(np.mean(distance ** 2))), "p95_mm": float(np.percentile(distance, 95)), "max_mm": float(distance.max()), "samples": samples,
            "method": "surface samples mirrored at x = mean motor x, distance to the original surface"}

def printed(mesh, axis):
    result = mesh.copy()
    result.apply_transform(trimesh.geometry.align_vectors(print_axes(axis)[2], [0, 0, 1]))
    result.apply_translation(-result.bounds[0])
    return result

def overhangs(mesh, angle, bed):
    normal, area, centre = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    down = (normal[:, 2] < -math.cos(math.radians(angle))) & (centre[:, 2] > bed)
    contact = (normal[:, 2] < -0.99) & (centre[:, 2] <= bed)
    return {"overhang_area_mm2": float(area[down].sum()), "overhang_share": float(area[down].sum() / area.sum()), "bed_contact_mm2": float(area[contact].sum()), "print_height_mm": float(mesh.extents[2]),
            "print_footprint_mm": mesh.extents[:2].tolist(), "overhang_deg": angle, "method": f"faces whose normal lies within {90 - angle:g} deg of straight down above the bed, area share of the surface"}

def section(mesh, z, center, half, h):
    segments = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, z])
    count = int(round(2 * half / h))
    axis_x, axis_y = center[0] - half + (np.arange(count) + 0.5) * h, center[1] - half + (np.arange(count) + 0.5) * h
    image = np.zeros((count, count), dtype=bool)
    first, second = segments[:, 0, :2], segments[:, 1, :2]
    for column, y in enumerate(axis_y):
        crossing = (first[:, 1] <= y) != (second[:, 1] <= y)
        a, b = first[crossing], second[crossing]
        x = np.sort(a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
        for start, stop in zip(x[0::2], x[1::2]):
            image[(axis_x >= start) & (axis_x <= stop), column] = True
    return image, axis_x, axis_y

def holes(mesh, pattern, h, tolerance):
    center, radius = np.asarray(pattern["center_mm"], dtype=float), pattern["radius_mm"]
    image, axis_x, axis_y = section(mesh, pattern["z_mm"], center, radius + 4.0, h)
    labels, count = ndimage.label(~image)
    clearance = ndimage.distance_transform_edt(~image, sampling=h)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    found = []
    for index in range(1, count + 1):
        if index in border:
            continue
        cells = np.argwhere(labels == index)
        diameter = 2 * math.sqrt(len(cells) * h * h / math.pi)
        centroid = np.array([axis_x[cells[:, 0]].mean(), axis_y[cells[:, 1]].mean()])
        offset = centroid - center
        position = center + radius * offset / max(np.linalg.norm(offset), 1e-9)
        cells_at = [np.clip(np.round((point - [axis_x[0], axis_y[0]]) / h).astype(int), 0, len(axis_x) - 1) for point in (position, centroid)]
        fits = any(labels[i, j] == index and clearance[i, j] >= pattern["screw_diameter_mm"] / 2 - h for i, j in cells_at)
        if fits and abs(np.linalg.norm(offset) - radius) <= tolerance + diameter / 2 and pattern["hole_diameter_mm"][0] <= diameter <= pattern["hole_diameter_mm"][1]:
            found.append({"position_mm": position.tolist(), "centroid_mm": centroid.tolist(), "diameter_mm": diameter})
    return {"name": pattern["name"], "expected": pattern["count"], "found": len(found), "holes": found, "material_in_section": bool(image.any()), "passed": len(found) >= pattern["count"]}

def blocked(mesh, starts, direction, skip):
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    inside = mesh.contains(starts)
    locations, rays, _ = mesh.ray.intersects_location(starts, np.repeat([direction], len(starts), axis=0), multiple_hits=True)
    distance = np.round(np.einsum("ij,j->i", np.reshape(locations, (-1, 3)) - starts[rays], direction), 4)
    entries = [np.unique(distance[rays == index])[int(inside[index])::2] for index in range(len(starts))]
    return np.array([bool(np.any(values > skip)) for values in entries])

def reachable(mesh, start, direction, radius, skip):
    direction = np.asarray(direction, dtype=float)
    first, second, _ = print_axes(direction)
    starts = np.asarray([start] + [np.asarray(start) + radius * (math.cos(angle) * first + math.sin(angle) * second) for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False)])
    return not bool(blocked(mesh, starts, direction, skip).any())

def assembly(mesh, solid, lower, spec, config):
    h = config["voxel_mm"]
    patterns = [holes(mesh, pattern, config["section_voxel_mm"], config["hole_tolerance_mm"]) for pattern in spec["mount_patterns"]]
    for pattern, result in zip(spec["mount_patterns"], patterns):
        result["tool_reachable"] = [reachable(mesh, [*hole["centroid_mm"], pattern["z_mm"]], pattern["tool_direction"], config["screw_head_radius_mm"], config["tool_skip_mm"]) for hole in result["holes"]]
    points = lower + (np.argwhere(solid) + 0.5) * h
    fit = {region["name"]: float(region_contains(points, shrink(region, h)).sum() * h ** 3) for region in spec["keep_outs"]}
    connectors = {item["name"]: reachable(mesh, item["position_mm"], item["direction"], config["connector_radius_mm"], config["tool_skip_mm"]) for item in spec["connectors"]}
    return {"bolt_patterns": patterns, "keep_out_material_mm3": fit, "connector_reachable": connectors, "fit_tolerance_mm3": config["fit_tolerance_mm3"],
            "bolt_patterns_passed": all(item["passed"] for item in patterns), "tools_passed": all(all(item["tool_reachable"]) for item in patterns) and all(connectors.values()),
            "fit_passed": all(volume <= config["fit_tolerance_mm3"] for volume in fit.values()), "connectors_checked": bool(spec["connectors"]),
            "method": "holes from an exact plane section rasterised at the section voxel; reachability = 9 parallel rays (axis plus screw-head or connector radius) from the hole or connector along the tool axis must not re-enter material beyond the tool skip distance (the pad itself); fit = material voxels inside each keep-out shrunk by one voxel"}

def geometry(spec, config=None):
    config = config or spec
    started = perf_counter()
    mesh = load_frame(spec)
    h = config["voxel_mm"]
    solid, lower, unbalanced = voxel_grid(mesh, h)
    print_mesh = printed(mesh, spec["print_axis"])
    plane = float(np.mean([xyz[0] for xyz in spec["motors"].values()]))
    from deep_frame.topology_implicit_validation import ball_curvature
    curvature = ball_curvature(mesh, config["curvature_radius_mm"], max(config["surface_samples"], int(mesh.area / config["curvature_radius_mm"] ** 2 / 2)), config["seed"])
    result = {"mass": mass_properties(mesh, component_list(spec), PRINT_MATERIAL["density_g_cm3"]), "airflow": top_view(solid, lower, h, spec, config["hub_radius_mm"]), "assembly": assembly(mesh, solid, lower, spec, config),
              "printability": overhangs(print_mesh, config["overhang_deg"], config["bed_tolerance_mm"]),
              "form": {**strut_form(solid, h), "loops": loops(solid, h, config["loop_closing_mm"]), "mesh_bodies": int(mesh.body_count), "mesh_genus": int(round((2 * mesh.body_count - mesh.euler_number) / 2)),
                       "flight_height_mm": float(mesh.extents[2]), "symmetry": symmetry(mesh, plane, config["surface_samples"], config["seed"]),
                       "roughness": {"curvature_neighbour_rms_per_mm": curvature["neighbour_difference_rms_per_mm"], "mean_abs_curvature_per_mm": curvature["absolute_mean_curvature_per_mm"], "radius_mm": curvature["radius_mm"], "method": curvature["method"]}},
              "voxel_mm": h, "unbalanced_columns": unbalanced, "runtime_s": None}
    result["runtime_s"] = perf_counter() - started
    return result

def walls(spec):
    from deep_frame.topology_implicit_validation import wall_opening
    mesh = load_frame(spec)
    zones = spec.get("wall_zones") or [{"name": f"motor_{name}", "role": "preserve", "kind": "cylinder", "center_mm": [x, y, z - spec["motor_up"][name] * 2.0], "radius_mm": TOPOLOGY_CONFIG["motor_contact_radius_mm"], "height_mm": 6.0, "axis": "z"} for name, (x, y, z) in spec["motors"].items()]
    result = wall_opening(mesh, IMPLICIT_CONFIG, zones)
    result["calibrated_rule_passed"] = result.pop("passed")
    return result

def load_model(spec, config):
    selectors, loads = spec["selectors"], config["loads"]
    weight = loads["all_up_mass_g"] / 1000 * loads["standard_gravity_m_s2"]
    x, y = spec["motors"][selectors["arm_motor"]][:2]
    radial = np.array([x, y]) / math.hypot(x, y)
    oblique = np.append(-radial + [-radial[1], radial[0]], -0.5)
    oblique = oblique / np.linalg.norm(oblique) * weight * loads["crash_arm_g"]
    center, motors = selectors["center_fixtures"], selectors["motor_fixtures"]
    cases = [{"name": "arm_tip", "analysis": "static", "fixed_regions": center, "loads": [{"region": selectors["arm_tip"], "force_n": [0.0, 0.0, loads["arm_tip_force_n"]]}]},
             {"name": "modes", "analysis": "modal", "fixed_regions": motors},
             {"name": "crash_front", "analysis": "static", "fixed_regions": center, "loads": [{"region": selectors["camera"], "force_n": [0.0, -weight * loads["crash_front_g"], 0.0]}]},
             {"name": "crash_arm", "analysis": "static", "fixed_regions": center, "loads": [{"region": selectors["arm_tip"], "force_n": oblique.tolist()}]},
             {"name": "crash_back", "analysis": "static", "fixed_regions": motors, "loads": [{"region": selectors["deck"], "force_n": [0.0, 0.0, -weight * loads["crash_back_g"]]}]}]
    masses = [{"name": "battery", "mass_g": selectors.get("battery_mass_g", COMPONENT_DEFAULTS["battery"]["mass_g"]), "position_mm": selectors["battery_center_mm"], "attachment_region": selectors["deck"]}]
    material = {**PRINT_MATERIAL, "print_axis": list(spec["print_axis"])}
    return material, masses, cases

def mechanics(spec, config=None):
    config = config or spec
    mesh = load_frame(spec)
    material, masses, cases = load_model(spec, config)
    settings = {**FEA_CONFIG["settings"], **{key: IMPLICIT_CONFIG[key] for key in MESH_KEYS}, **config["fea_settings"], "stiffness_load_case": "arm_tip", "work_dir": str(Path(spec["output"]) / "fea")}
    sliver = None
    mesh, trials, chosen = robust_surface(mesh, settings, config["fea_surface"])
    settings.update(chosen)
    graded = None
    if not chosen and config["fea_surface"].get("graded"):
        boxes = [box for value in spec["selectors"].values() for box in (value if isinstance(value, list) else [value]) if isinstance(box, dict) and box.get("kind") == "box"]
        planes = sorted({round((box["min_mm"][2]+box["max_mm"][2])/2, 6) for box in boxes if box["max_mm"][2]-box["min_mm"][2] <= config["fea_surface"]["graded"]["snap_selector_mm"]})
        surface, graded = graded_surface(mesh, settings, config["fea_surface"]["graded"], planes)
        if graded["passed"]:
            mesh, chosen = surface, {"tet_attempts": ["direct_hxt"], "fea_direct_minimum_angle_deg": config["fea_surface"]["graded"]["minimum_angle_deg"]}
            settings.update(chosen)
    if not chosen and np.degrees(mesh.face_angles.min()) < settings["fea_direct_minimum_angle_deg"]:
        mesh, sliver = clean_slivers(mesh, settings)
    result = evaluate(mesh, material, masses, cases, settings)
    result.update(sliver_cleanup=sliver, fea_surface={"trials": trials, "graded": graded, "choice": chosen, "volume_mm3": float(mesh.volume)})
    result["model"] = {"material": material, "point_masses": masses, "load_cases": cases, "loads": config["loads"],
                       "support": {"arm_tip": "centre mount undersides fixed (all translations)", "modes": "four motor seat undersides fixed (all translations), battery as rigidly coupled point mass on the deck band",
                                   "crash_front": "centre mount undersides fixed", "crash_arm": "centre mount undersides fixed", "crash_back": "four motor seat undersides fixed"}}
    return result

def slice_frame(spec, config=None):
    config = config or spec
    slicer = config["slicer"]
    directory = Path(spec["output"]) / "slicer"
    directory.mkdir(parents=True, exist_ok=True)
    printed(load_frame(spec), spec["print_axis"]).export(directory / "printed.stl")
    result = {"version": slicer["version"], "profile": slicer["profile"], "options": slicer["options"]}
    for label, extra in (("supports", slicer["support"]), ("no_supports", [])):
        output = directory / f"{label}.gcode"
        started = perf_counter()
        completed = subprocess.run([slicer["executable"], "--export-gcode", *slicer["options"], *extra, "--output", str(output), str(directory / "printed.stl")], capture_output=True, text=True, timeout=slicer["timeout_s"])
        text = output.read_text(encoding="utf-8", errors="replace")[-20000:] if output.exists() else ""
        time_match = re.search(r"estimated printing time \(normal mode\) = (.+)", text)
        volume = re.search(r"filament used \[cm3\] = ([\d.]+)", text)
        grams = re.search(r"filament used \[g\] = ([\d.]+)", text)
        result[label] = {"returncode": completed.returncode, "runtime_s": perf_counter() - started, "log_tail": (completed.stdout + completed.stderr)[-1500:], "print_time": time_match.group(1).strip() if time_match else None,
                         "print_time_min": duration_minutes(time_match.group(1)) if time_match else None, "filament_mm3": 1000 * float(volume.group(1)) if volume else None, "filament_g": float(grams.group(1)) if grams else None}
    good = all(result[label]["filament_mm3"] is not None for label in ("supports", "no_supports"))
    result["support_mm3"] = result["supports"]["filament_mm3"] - result["no_supports"]["filament_mm3"] if good else None
    result["passed"] = bool(good and result["supports"]["returncode"] == 0)
    return result

def duration_minutes(text):
    units = {"d": 1440, "h": 60, "m": 1, "s": 1 / 60}
    return sum(float(value) * units[unit] for value, unit in re.findall(r"(\d+)([dhms])", text))

def lookup(data, path):
    for key in path.split("."):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data

def scaled(spec, result):
    geometry, fea, slicer = result.get("geometry") or {}, result.get("fea") or {}, result.get("slicer") or {}
    motors = np.array([spec["motors"][name][:2] for name in MOTORS], dtype=float)
    hub = motors.mean(axis=0)
    arm = float(np.linalg.norm(motors - hub, axis=1).mean())
    values = {"arm_mm": arm, "wheelbase_mm": 2 * arm}
    rules = {"arm_tip_slope": lambda: spec["loads"]["arm_tip_force_n"] / fea["stiffness_n_per_mm"] / arm,
             "support_per_volume": lambda: slicer["support_mm3"] / geometry["mass"]["frame_volume_mm3"],
             "print_min_per_g": lambda: slicer["supports"]["print_time_min"] / geometry["mass"]["frame_mass_g"],
             "symmetry_per_wheelbase": lambda: geometry["form"]["symmetry"]["rms_mm"] / (2 * arm),
             "cog_offset_per_wheelbase": lambda: math.hypot(*np.subtract(geometry["mass"]["center_of_mass_mm"][:2], hub)) / (2 * arm),
             "height_per_wheelbase": lambda: geometry["form"]["flight_height_mm"] / (2 * arm),
             "izz_per_mass_arm2": lambda: geometry["mass"]["inertia_g_mm2"][2][2] / geometry["mass"]["total_mass_g"] / arm ** 2}
    for name, rule in rules.items():
        try:
            values[name] = float(rule())
        except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
            values[name] = None
    return values

def within(result, path, low, high):
    value = lookup(result, path)
    return (low is None or value >= low) and (high is None or value <= high)

def assess(result, config):
    material, loads = PRINT_MATERIAL, config["loads"]
    geometry, fea, slicer = result.get("geometry") or {}, result.get("fea") or {}, result.get("slicer") or {}
    conditions = {}
    def condition(name, function):
        try:
            conditions[name] = bool(function())
        except (KeyError, TypeError, IndexError, ValueError):
            conditions[name] = False
    condition("one_body", lambda: geometry["form"]["mesh_bodies"] == 1)
    condition("bolt_patterns", lambda: geometry["assembly"]["bolt_patterns_passed"])
    condition("keep_outs_free", lambda: geometry["assembly"]["fit_passed"])
    condition("tools_reachable", lambda: geometry["assembly"]["tools_passed"])
    condition("fea_solved", lambda: fea["status"] == "ok")
    condition("slicer", lambda: slicer["passed"])
    opening, limits = result.get("walls") or {}, {"deep_max_fraction": IMPLICIT_CONFIG["wall_deep_max_fraction"], "deep_component_max_mm3": IMPLICIT_CONFIG["wall_deep_component_max_mm3"], "motor_zone_margin_mm": IMPLICIT_CONFIG["wall_motor_zone_margin_mm"]}
    condition("wall_deep_fraction", lambda: opening["part_volume_mm3"] > 0 and opening["unbalanced_columns"] == 0 and opening["deep_fraction"] <= limits["deep_max_fraction"])
    condition("wall_deep_component", lambda: opening["largest_deep_mm3"] <= limits["deep_component_max_mm3"])
    condition("wall_motor_zones", lambda: len(opening["motor_zones"]) > 0 and not opening["motor_zone_hits"])
    crash = {}
    for name in CRASH:
        try:
            case = fea["load_cases"][name]
            crash[name] = {"von_mises_p999_mpa": case["von_mises_p999_mpa"], "print_normal_p999_mpa": case["print_normal_abs_p999_mpa"],
                           "utilisation_xy": case["von_mises_p999_mpa"] * loads["safety_factor"] / material["strength_xy_mpa"], "utilisation_z": case["print_normal_abs_p999_mpa"] * loads["safety_factor"] / material["strength_z_mpa"]}
        except (KeyError, TypeError):
            crash[name] = None
        condition(f"{name}_strength", lambda: crash[name]["utilisation_xy"] <= 1 and crash[name]["utilisation_z"] <= 1)
    for name, (path, low, high) in config["targets"].items():
        condition("target:" + name, lambda: within(result, path, low, high))
    warnings = []
    for name, (path, low, high) in config["warnings"].items():
        try:
            passed = within(result, path, low, high)
        except (KeyError, TypeError, IndexError, ValueError):
            passed = False
        if not passed:
            warnings.append(name)
    return {"conditions": conditions, "missed": [name for name, passed in conditions.items() if not passed], "good": all(conditions.values()), "warnings": warnings, "crash": crash, "safety_factor": loads["safety_factor"],
            "targets": config["targets"], "warning_ranges": config["warnings"],
            "wall_rule": {**limits, "opening_radius_mm": IMPLICIT_CONFIG["wall_opening_radius_mm"], "deep_depth_mm": IMPLICIT_CONFIG["wall_deep_mm"], "status": "provisional, calibrated on ManaFly 3 BETA V4 (docs/validation/wall_calibration_manafly.md); final after own drop tests in M2"}, "rule": "a frame is good only if every condition holds; unevaluable conditions count as missed; warnings do not decide"}

def number(value, digits=1):
    return "–" if value is None else f"{value:.{digits}f}".replace(".", ",")

def report_line(result):
    g, f, s, a = result.get("geometry") or {}, result.get("fea") or {}, result.get("slicer") or {}, result["assessment"]
    w = result.get("walls") or {}
    cells = [result["name"]]
    try:
        m = g["mass"]
        inertia = np.diag(m["inertia_g_mm2"]) * 1e-3
        cells += [f"{number(m['frame_mass_g'])} g", f"SP z {number(m['center_of_mass_mm'][2])} mm; Ixx/Iyy/Izz {'/'.join(number(v, 0) for v in inertia)} kg·mm² (K)"]
        cells.append(f"{number(100 * g['airflow']['prop_ring_share'])} % ({number(g['airflow']['prop_diameter_mm'], 0)} mm)")
        asm = g["assembly"]
        cells.append(f"Bohrb. {sum(p['found'] for p in asm['bolt_patterns'])}/{sum(p['expected'] for p in asm['bolt_patterns'])}, Passung {'ok' if asm['fit_passed'] else 'nein'}, Werkzeug {'ok' if asm['tools_passed'] else 'nein'}{'' if asm['connectors_checked'] else ', Stecker n/a'}")
        p = g["printability"]
        cells.append(f"Überh. {number(100 * p['overhang_share'])} %, Öffn. r=1 {number(100 * w['deep_fraction'], 2) if w else '–'} %/{number(w.get('largest_deep_mm3'))} mm³/Motorzonen {len(w.get('motor_zone_hits') or {}) if w else '–'},Stütze {number((s.get('support_mm3') or 0) / 1000, 2) if s else '–'} cm³, {number((s.get('supports') or {}).get('print_time_min'), 0)} min")
        form = g["form"]
        cells.append(f"Strebe {'/'.join(number(form['strut_width_mm'][k]) for k in ('p10', 'p50', 'p90'))} mm, H/B {number(form['section_ratio']['p50'], 2)}, offen {number(100 - 100 * g['airflow']['bbox_share'], 0)} %, "
                     f"Höhe {number(form['flight_height_mm'])} mm, Körper {form['mesh_bodies']}, Schlaufen {form['loops']['loops']}, Sym. {number(form['symmetry']['rms_mm'], 2)} mm, Rauh. {number(form['roughness']['curvature_neighbour_rms_per_mm'], 3)}/mm")
    except (KeyError, TypeError):
        cells += ["–"] * (7 - len(cells))
    cases = f.get("load_cases") or {}
    cells.append(f"{number(f.get('stiffness_n_per_mm'))} N/mm (A)" if f.get("stiffness_n_per_mm") else "–")
    modes = (cases.get("modes") or {}).get("eigenfrequencies_hz") or []
    cells.append(f"f1 {number(modes[0], 0)} Hz; {'/'.join(number(v, 0) for v in modes[1:4])} (A)" if modes else "–")
    crash = [f"{name.split('_')[1]} {number(c['von_mises_p999_mpa'])}/{number(c['print_normal_p999_mpa'])} MPa" for name, c in a["crash"].items() if c]
    cells.append("; ".join(crash) + f" (SF {number(a['safety_factor'])}, A)" if crash else "–")
    cells.append((", ".join(a["missed"]) or "keine") + (f"; Warnung: {', '.join(a['warnings'])}" if a.get("warnings") else ""))
    return "| " + " | ".join(cells) + " |"

HEADER = ("| Frame | 1 Masse | 2 Schwerpunkt, Trägheit | 3 Luftstrom (Material im Propkreis) | 4 Montage | 5 Druckbarkeit | 6 Form | 7 Steifigkeit Armspitze | 8 Resonanz | 9 Crash p99,9 v. Mises/σ_Druckachse | Verfehlt |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")
LEGEND = ("(A) beruht auf Materialannahmen: ν = 0,30 und G13/G23 = E_z/(2(1+ν)) sind nicht im Bambu-PA6-CF-Datenblatt; E_xy, E_z, Festigkeiten und Dichte stammen aus dem Datenblatt. "
          "(K) Komponentenmassen und -maße aus unserer Hardware-Konfiguration, bei Referenzen auf deren Aufnahmen gesetzt. Crash: Spannung als 99,9-%-Wert der Integrationspunkte; Grenze σ_xy 102 MPa und σ_z 48 MPa geteilt durch SF.")

def summary(spec, parts, config=None):
    config = config or spec
    result = {"name": spec["name"], "stl": str(spec["stl"]), "print_axis": list(spec["print_axis"]), "material": PRINT_MATERIAL, **parts}
    result["scaled"] = scaled(spec, result)
    result["assessment"] = assess(result, config)
    result["line"] = report_line(result)
    return json.loads(json.dumps(result, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))

def datasheet_section(result, evaluation):
    return (f"\n## Bewertungszeile (neun Kriterien)\n\nAutomatisch von tools/evaluate_frame.py; Zielbereiche und Warnbereiche: docs/optimization_problem.md; Ergebnis: {evaluation}. "
            f"Gut (alle Bedingungen erfüllt): {'ja' if result['assessment']['good'] else 'nein'}.\n\n{HEADER}\n{result['line']}\n\n{LEGEND}\n")

def replace_line(text, result, occurrence=None):
    lines, prefix = text.split("\n"), f"| {result['name']} |"
    found = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    for i in found if occurrence is None else found[occurrence:occurrence + 1]:
        lines[i] = result["line"] + "\r" * lines[i].endswith("\r")
    return re.sub(r"Gut \(alle Bedingungen erfüllt\): (ja|nein)", f"Gut (alle Bedingungen erfüllt): {'ja' if result['assessment']['good'] else 'nein'}", "\n".join(lines)) if found else None

def append_datasheet(path, result, evaluation):
    path = Path(path)
    text = path.open(encoding="utf-8", newline="").read() if path.exists() else f"# {result['name']}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replace_line(text, result) or text.rstrip("\r\n") + "\n" + datasheet_section(result, evaluation), encoding="utf-8", newline="")
