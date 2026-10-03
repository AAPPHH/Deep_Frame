import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_frame.config import PRINT_MATERIAL, command_line, configure

DIRECTIONS = {"Fx": ("force", 0), "Fy": ("force", 1), "Fz": ("force", 2), "Mx": ("moment", 0), "My": ("moment", 1), "Mz": ("moment", 2)}
MOTORS = ("front_left", "front_right", "rear_left", "rear_right")
CONFIG = {"frame": None, "output": None, "mesh": None, "threads": None, "solver_timeout_s": 7200.0, "force_n": 1.0, "moment_nmm": 1.0,
          "candidates": [], "references": [], "threshold": 0.7, "size_mm": [80.0, 8.0, 8.0], "cell_mm": 1.0, "print_axis": [1.0, 0.0, 0.0]}
KINDS = {"frame": "path", "output": "path", "mesh": "path", "threads": "int", "solver_timeout_s": "float", "force_n": "float", "moment_nmm": "float",
         "candidates": ["path"], "references": ["path"], "threshold": "float", "size_mm": ["float", "float", "float"], "cell_mm": "float", "print_axis": ["float", "float", "float"]}
MODEL = {"load": "F: force_n split equally over the selector nodes; M: minimum-norm nodal force field over the selector nodes with zero resultant force and resultant moment moment_nmm about the global axis through the node centroid (distributed linear traction couple)",
         "stiffness": "work-conjugate: k = P^2 / sum_i f_i.u_i, i.e. load over the load-weighted mean displacement along F (= mean nodal displacement along F) or the load-weighted mean rotation about the moment axis; units N/mm and N*mm/rad",
         "support": "fixture as in the frame evaluation (no inertia relief there): stack mount undersides (center_fixtures) fixed in all translations for motor pads, battery rails and camera; the four motor seat undersides (motor_fixtures) fixed for the stack",
         "point_masses": "none (the battery rigid-body patch of the evaluation sits on the rail selector and would make the rail stiffness a rigid-patch artefact)"}

def settings_of(overrides):
    return configure(CONFIG, KINDS, overrides)

def nodal_loads(points, kind, axis, magnitude):
    points = np.asarray(points, dtype=float)
    if kind == "force":
        loads = np.zeros_like(points)
        loads[:, axis] = magnitude / len(points)
        return loads
    if np.linalg.matrix_rank(points - points[0], tol=1e-8) < 2:
        raise ValueError("A moment needs at least three non-collinear interface nodes")
    r = points - points.mean(axis=0)
    blocks = np.zeros((6, 3 * len(points)))
    for index, (x, y, z) in enumerate(r):
        blocks[:3, 3 * index:3 * index + 3] = np.eye(3)
        blocks[3:, 3 * index:3 * index + 3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    target = np.zeros(6)
    target[3 + axis] = magnitude
    loads = blocks.T @ np.linalg.solve(blocks @ blocks.T, target)
    if not np.allclose(blocks @ loads, target, atol=1e-9 * max(magnitude, 1.0)):
        raise ValueError("Couple field does not reproduce the requested moment")
    return loads.reshape(-1, 3)

def conjugate(loads, displacements):
    return float(np.sum(np.asarray(loads) * np.asarray(displacements)))

def select(nodes, regions):
    from deep_frame.fea import _select
    return sorted({node for region in regions for node in _select(nodes, region)})

def step_lines(name, chosen, loads, solver):
    lines = ["*STEP", "*STATIC" + solver, "*CLOAD,OP=NEW"]
    lines.extend(f"{node},{axis + 1},{value:.12g}" for node, load in zip(chosen, loads) for axis, value in enumerate(load) if value)
    return lines + [f"*NODE PRINT,NSET={name},GLOBAL=YES", "U", "*END STEP"]

def blocks(content, count):
    found = []
    for part in re.split(r"displacements \(vx,vy,vz\)", content, flags=re.IGNORECASE)[1:]:
        rows = []
        for line in part.splitlines()[1:]:
            fields = line.split()
            if len(fields) != 4:
                if rows:
                    break
                continue
            rows.append([float(value.replace("D", "E")) for value in fields])
        found.append({int(row[0]): np.array(row[1:]) for row in rows})
    if len(found) != count:
        raise RuntimeError(f"Expected {count} displacement blocks, found {len(found)}")
    return found

def solve(nodes, elements, material, groups, settings, directory):
    from deep_frame.fea import _model_lines, _run, _set_lines, resolve_solver
    directory.mkdir(parents=True, exist_ok=True)
    model, _ = _model_lines(nodes, elements, material, [])
    solver = resolve_solver({})
    threads = settings["threads"] or int(os.environ.get("OMP_NUM_THREADS", 2))
    results, runs = {}, []
    for index, group in enumerate(groups):
        fixed = select(nodes, group["support"])
        lines = [*_set_lines("NSET", "FIXED", fixed), "*BOUNDARY", "FIXED,1,3"]
        cases = []
        for slot, (interface, regions) in enumerate(group["interfaces"].items()):
            chosen = select(nodes, regions)
            if set(chosen) & set(fixed):
                raise ValueError(f"{interface} overlaps the support")
            lines.extend(_set_lines("NSET", f"IF{slot}", chosen))
            points = np.array([nodes[node] for node in chosen])
            for direction, (kind, axis) in DIRECTIONS.items():
                magnitude = settings["force_n"] if kind == "force" else settings["moment_nmm"]
                loads = nodal_loads(points, kind, axis, magnitude)
                cases.append((interface, direction, chosen, loads, magnitude))
                lines.extend(step_lines(f"IF{slot}", chosen, loads, ""))
        name = f"group_{index}"
        (directory / f"{name}.inp").write_text("\n".join(model + lines) + "\n", encoding="ascii")
        start = time.monotonic()
        log = _run([solver, "-i", name], directory, settings["solver_timeout_s"], threads, directory / f"{name}.log")
        if "Job finished" not in log:
            raise RuntimeError(f"CalculiX did not finish {name}")
        found = blocks((directory / f"{name}.dat").read_text(encoding="utf-8", errors="replace"), len(cases))
        runs.append({"group": name, "steps": len(cases), "fixed_nodes": len(fixed), "runtime_s": time.monotonic() - start, "reused_factorisation": "Reusing csc" in log})
        for (interface, direction, chosen, loads, magnitude), displacement in zip(cases, found):
            u = np.array([displacement[node] for node in chosen])
            work = conjugate(loads, u)
            row = results.setdefault(interface, {"nodes": len(chosen), "support": group["label"]})
            row[direction] = {"stiffness": magnitude ** 2 / work if work > 0 else None, "conjugate": work / magnitude, "mean_displacement_mm": u.mean(axis=0).tolist()}
    return results, runs

def interfaces(spec):
    selectors, motors = spec["selectors"], spec["motors"]
    pads = {}
    for region in selectors["motor_fixtures"]:
        center = (np.add(region["min_mm"], region["max_mm"]) / 2)[:2]
        pads["motor_" + min(MOTORS, key=lambda name: np.hypot(*(center - motors[name][:2])))] = [region]
    if len(pads) != 4:
        raise ValueError("motor_fixtures do not map one-to-one onto the four motors")
    return [{"label": "stack fixed", "support": selectors["center_fixtures"], "interfaces": {**{f"motor_{name}": pads[f"motor_{name}"] for name in MOTORS}, "battery_rails": [selectors["deck"]], "camera": [selectors["camera"]]}},
            {"label": "motor pads fixed", "support": selectors["motor_fixtures"], "interfaces": {"stack": selectors["center_fixtures"]}}]

def frame_mesh(spec, settings):
    from deep_frame.fea import _read_mesh
    path = settings["mesh"]
    record = Path(spec["output"]) / "fea.json"
    if not path and record.exists():
        evaluation = json.loads(record.read_text(encoding="utf-8"))
        candidate = Path(evaluation["artifacts"]["directory"]) / "mesh.inp"
        if evaluation.get("status") == "ok" and candidate.exists():
            nodes, elements = _read_mesh(candidate)
            if len(nodes) != evaluation["mesh"]["node_count"]:
                raise ValueError(f"{candidate} has {len(nodes)} nodes, evaluation recorded {evaluation['mesh']['node_count']}")
            return nodes, elements, {"source": "evaluation tet mesh", "path": str(candidate), "evaluation": str(record)}
    if path:
        nodes, elements = _read_mesh(path)
        return nodes, elements, {"source": "given tet mesh", "path": str(path)}
    return evaluation_mesh(spec, settings)

def evaluation_mesh(spec, settings):
    from deep_frame.config import FEA_CONFIG, IMPLICIT_CONFIG
    from deep_frame.fea import MESH_KEYS, _mesh_settings, _volume_mesh, clean_slivers, robust_surface
    from deep_frame.frame_evaluation import load_frame
    mesh = load_frame(spec)
    options = {**FEA_CONFIG["settings"], **{key: IMPLICIT_CONFIG[key] for key in MESH_KEYS}, **spec["fea_settings"]}
    mesh, _, chosen = robust_surface(mesh, options, spec["fea_surface"])
    options.update(chosen)
    if not chosen and np.degrees(mesh.face_angles.min()) < options["fea_direct_minimum_angle_deg"]:
        mesh, _ = clean_slivers(mesh, options)
    directory = Path(settings["output"]) / "mesh"
    directory.mkdir(parents=True, exist_ok=True)
    record = {}
    nodes, elements = _volume_mesh(mesh, directory, _mesh_settings(options), record)
    return nodes, elements, {"source": "evaluation mesh pipeline", "path": str(directory / "mesh.inp"), "surface_choice": chosen}

def table(result):
    lines = [f"| {result['name']} | " + " | ".join(DIRECTIONS) + " |", "|---|" + "---|" * len(DIRECTIONS)]
    for interface, row in result["interfaces"].items():
        lines.append(f"| {interface} | " + " | ".join(number(row[direction]["stiffness"]) for direction in DIRECTIONS) + " |")
    return "\n".join(lines) + "\n\nF in N/mm, M in N*mm/rad.\n"

def number(value):
    return "-" if value is None else f"{value:.4g}"

def write(settings, result, markdown):
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "interface_stiffness.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (output / "interface_stiffness.md").write_text(markdown, encoding="utf-8")
    print(markdown)

def arm(spec):
    motors = np.array([spec["motors"][name][:2] for name in MOTORS], dtype=float)
    return float(np.linalg.norm(motors - motors.mean(axis=0), axis=1).mean())

def run(overrides):
    from deep_frame.frame_evaluation import frame_spec, load_model
    settings = settings_of(overrides)
    spec = frame_spec(json.loads(Path(settings["frame"]).read_text(encoding="utf-8-sig")))
    settings["output"] = settings["output"] or str(ROOT / "exports" / "interface_stiffness" / Path(spec["output"]).name)
    material = load_model(spec, spec)[0]
    nodes, elements, mesh = frame_mesh(spec, settings)
    groups = interfaces(spec)
    results, runs = solve(nodes, elements, material, groups, settings, Path(settings["output"]) / "ccx")
    result = {"name": spec["name"], "frame": str(settings["frame"]), "stl": str(spec["stl"]), "arm_mm": arm(spec), "mesh": {**mesh, "nodes": len(nodes), "elements": len(elements)},
              "material": material, "model": MODEL, "selectors": {group["label"]: group for group in groups}, "runs": runs, "interfaces": results}
    write(settings, result, table(result))
    return 0

def block_mesh(size, cell):
    counts = [max(1, round(length / cell)) for length in size]
    grid = np.stack(np.meshgrid(*[np.linspace(0, length, count + 1) for length, count in zip(size, counts)], indexing="ij"), axis=-1)
    index = np.arange(grid[..., 0].size).reshape(grid.shape[:3])
    corners = {tuple(key): grid[tuple(key)] for key in np.ndindex(*grid.shape[:3])}
    nodes = {int(index[key]) + 1: tuple(float(v) for v in xyz) for key, xyz in corners.items()}
    middle, elements = {}, {}
    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in middle:
            middle[key] = len(nodes) + 1
            nodes[middle[key]] = tuple((np.add(nodes[a], nodes[b]) / 2).tolist())
        return middle[key]
    for cell_index in np.ndindex(*counts):
        for order in itertools.permutations(range(3)):
            path, vertex = [tuple(cell_index)], list(cell_index)
            for axis in order:
                vertex[axis] += 1
                path.append(tuple(vertex))
            a, b, c, d = (int(index[key]) + 1 for key in path)
            if np.dot(np.cross(np.subtract(nodes[b], nodes[a]), np.subtract(nodes[c], nodes[a])), np.subtract(nodes[d], nodes[a])) < 0:
                b, c = c, b
            elements[len(elements) + 1] = (a, b, c, d, mid(a, b), mid(b, c), mid(c, a), mid(a, d), mid(b, d), mid(c, d))
    return nodes, elements

def block(overrides):
    settings = settings_of(overrides)
    settings["output"] = settings["output"] or str(ROOT / "exports" / "interface_stiffness" / "block")
    length, width, height = settings["size_mm"]
    nodes, elements = block_mesh(settings["size_mm"], settings["cell_mm"])
    material = {**PRINT_MATERIAL, "print_axis": settings["print_axis"]}
    slab = lambda x: {"kind": "box", "min_mm": [x - 1e-6, -1.0, -1.0], "max_mm": [x + 1e-6, width + 1.0, height + 1.0]}
    groups = [{"label": "x=0 face fixed", "support": [slab(0.0)], "interfaces": {"tip": [slab(length)]}}]
    results, runs = solve(nodes, elements, material, groups, settings, Path(settings["output"]) / "ccx")
    tip = results["tip"]
    axis = np.abs(np.asarray(settings["print_axis"], dtype=float))
    along, across = (PRINT_MATERIAL["orthotropic"]["e_z_mpa"], PRINT_MATERIAL["orthotropic"]["g_z_mpa"]) if axis[0] > 0.99 else (PRINT_MATERIAL["orthotropic"]["e_xy_mpa"], None)
    inertia, area = width * height ** 3 / 12, width * height
    beam = {"Fx": along * area / length, "Fy": 3 * along * height * width ** 3 / 12 / length ** 3, "Fz": 3 * along * inertia / length ** 3, "My": along * inertia / length, "Mz": along * height * width ** 3 / 12 / length}
    if across and width == height:
        beam["Mx"] = across * 0.1406 * width ** 4 / length
    checks = {"Fy/Fz": tip["Fy"]["stiffness"] / tip["Fz"]["stiffness"], "My/Mz": tip["My"]["stiffness"] / tip["Mz"]["stiffness"], "My/Fz over L^2/3": tip["My"]["stiffness"] / tip["Fz"]["stiffness"] / (length ** 2 / 3),
              **{f"{direction} FE/beam": tip[direction]["stiffness"] / value for direction, value in beam.items()}}
    result = {"name": "block", "size_mm": settings["size_mm"], "cell_mm": settings["cell_mm"], "mesh": {"nodes": len(nodes), "elements": len(elements), "source": "structured Kuhn C3D10"},
              "material": material, "model": MODEL, "runs": runs, "interfaces": results, "beam_theory": beam, "checks": checks}
    write(settings, result, table(result) + "\n" + "\n".join(f"- {key}: {value:.4f}" for key, value in checks.items()) + "\n")
    return 0

def compare(overrides):
    settings = settings_of(overrides)
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    references, candidates = [load(path) for path in settings["references"]], [load(path) for path in settings["candidates"]]
    settings["output"] = settings["output"] or str(ROOT / "exports" / "interface_stiffness" / "comparison")
    proposals, sections = [], []
    for candidate in candidates:
        header = "| Schnittstelle | Richtung | " + " | ".join(item["name"] for item in references) + f" | {candidate['name']} | " + " | ".join(f"/{item['name']}" for item in references) + " | " + " | ".join(f"/{item['name']} skaliert" for item in references) + " | Lücke |"
        lines = [f"### {candidate['name']} (Arm {candidate['arm_mm']:.1f} mm)", "", header, "|" + "---|" * (3 + 3 * len(references))]
        for interface, row in candidate["interfaces"].items():
            for direction in DIRECTIONS:
                own = row[direction]["stiffness"]
                values = [item["interfaces"].get(interface, {}).get(direction, {}).get("stiffness") for item in references]
                factors = [(item["arm_mm"] / candidate["arm_mm"]) if DIRECTIONS[direction][0] == "force" else 1.0 for item in references]
                ratios = [own / value if own and value else None for value in values]
                scaled = [own / (value * factor) if own and value else None for value, factor in zip(values, factors)]
                gap = all(ratio is not None and ratio < settings["threshold"] for ratio in ratios)
                lines.append(f"| {interface} | {direction} | " + " | ".join(number(value) for value in [*values, own]) + " | " + " | ".join(number(value) for value in [*ratios, *scaled]) + f" | {'ja' if gap else ''} |")
                if gap:
                    weakest = min(range(len(values)), key=lambda index: values[index])
                    proposals.append({"candidate": candidate["name"], "interface": interface, "direction": direction, "candidate_stiffness": own, "references": dict(zip([item["name"] for item in references], values)),
                                      "limit_raw": values[weakest], "limit_arm_scaled": values[weakest] * factors[weakest], "limit_source": references[weakest]["name"], "below_scaled_threshold_too": all(value is not None and value < settings["threshold"] for value in scaled)})
        sections.append("\n".join(lines))
    rule = (f"Lücke: Kandidat < {settings['threshold']:.0%} beider Referenzen (roh). Grenzvorschlag = schwächere Referenz; skaliert = auf unsere Armlänge wie die Armspitze "
            "(gleiche Neigung F/(k*Arm), also k * Arm_ref/Arm_ours) für Kräfte, Momente unskaliert. Einheiten: F in N/mm, M in N*mm/rad.")
    proposal_lines = ["| Kandidat | Schnittstelle | Richtung | k Kandidat | Grenze roh | Grenze skaliert | Quelle | auch skaliert < Schwelle |", "|---|---|---|---|---|---|---|---|"]
    proposal_lines += [f"| {item['candidate']} | {item['interface']} | {item['direction']} | {number(item['candidate_stiffness'])} | {number(item['limit_raw'])} | {number(item['limit_arm_scaled'])} | {item['limit_source']} | {'ja' if item['below_scaled_threshold_too'] else 'nein'} |" for item in proposals]
    markdown = "\n\n".join([rule, *sections, "### Vorgeschlagene Bedingungen", "\n".join(proposal_lines)]) + "\n"
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "interface_gaps.json").write_text(json.dumps({"threshold": settings["threshold"], "rule": rule, "proposals": proposals, "references": settings["references"], "candidates": settings["candidates"]}, indent=1), encoding="utf-8")
    (output / "interface_gaps.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0

def main(argv=None):
    return command_line({"run": run, "block": block, "compare": compare}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
