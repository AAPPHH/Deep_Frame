import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

if __name__ != "__main__":
    from build123d import export_step
    from deep_frame.config import IMPLICIT_CONFIG, IMPLICIT_KINDS, _json_copy, _json_digest
    from deep_frame.frame import assembly_placements, build_components, build_geometry, motor_positions
    MESH_ATTEMPTS = IMPLICIT_KINDS["tet_attempts"][0]
    MESH_KEYS = ("tet_attempts", "mesh_minimum_sicn", "mesh_boundary_deviation_mm", "surface_deviation_mm", "relative_volume_change", *(key for key in IMPLICIT_CONFIG if key.startswith("fea_")))
    MESH_NUMBERS = tuple(key for key in MESH_KEYS[2:] if key != "fea_remesh_targets_mm")

def resolve_solver(settings):
    explicit = settings.get("solver_path") or os.environ.get("CALCULIX_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        resolved = str(path.resolve()) if path.is_file() else shutil.which(str(explicit))
        if not resolved:
            raise FileNotFoundError(f"CalculiX solver not found: {explicit}")
        return resolved
    for name in ("ccx", "ccx_static", "ccx_dynamic"):
        if shutil.which(name):
            return shutil.which(name)
    candidates = sorted((Path(sys.prefix) / "calculix").glob("**/ccx_static.exe"))
    if candidates:
        return str(candidates[0].resolve())
    raise FileNotFoundError("CalculiX missing: set solver_path or CALCULIX_PATH, or install in .venv/calculix")

def _run(command, directory, timeout, threads, log_path):
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": str(threads), "CCX_NPROC_RESULTS": str(threads),
                        "CCX_NPROC_STIFFNESS": str(threads), "CCX_NPROC_EQUATION_SOLVER": str(threads),
                        "NUMBER_OF_CPUS": str(threads)})
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH")]))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_path = Path(log_path)
    for attempt in range(2):
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=timeout, creationflags=flags)
        if 0 <= completed.returncode < 0xC0000000 or attempt:
            break
        log_path.replace(log_path.with_suffix(".crash.log"))
    content = log_path.read_text(encoding="utf-8", errors="replace")
    if completed.returncode or re.search(r"\*ERROR|\*FATAL", content, re.IGNORECASE):
        raise RuntimeError(f"External tool failed ({completed.returncode}); log: {log_path}; {content[-1200:]}")
    return content

def _read_mesh(path):
    nodes = {}
    elements = {}
    mode = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("**") or not line:
            continue
        if line.startswith("*"):
            upper = line.upper()
            mode = "nodes" if upper == "*NODE" else "elements" if upper.startswith("*ELEMENT") and "TYPE=C3D10" in upper else None
            continue
        values = [v.strip() for v in line.split(",") if v.strip()]
        if mode == "nodes":
            nodes[int(values[0])] = tuple(float(v) for v in values[1:4])
        elif mode == "elements":
            if len(values) != 11:
                raise ValueError("Invalid C3D10 connectivity")
            elements[int(values[0])] = tuple(int(v) for v in values[1:])
    if not nodes or not elements:
        raise ValueError("Empty volume mesh")
    used = {node for element in elements.values() for node in element}
    if not used.issubset(nodes):
        raise ValueError("Mesh references missing nodes")
    nodes = {node: nodes[node] for node in sorted(used)}
    parents = {node: node for node in nodes}
    def root(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node
    for element in elements.values():
        anchor = root(element[0])
        for node in element[1:]:
            parents[root(node)] = anchor
    if len({root(node) for node in nodes}) != 1:
        raise ValueError("Disconnected volume mesh")
    return nodes, elements

def _vector(value, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain three finite numbers")
    return result

def _select(nodes, selector):
    if selector.get("kind") != "box":
        raise ValueError("Only box node selectors are supported")
    minimum = _vector(selector["min_mm"], "min_mm")
    maximum = _vector(selector["max_mm"], "max_mm")
    if np.any(minimum > maximum):
        raise ValueError("Selector min_mm exceeds max_mm")
    selected = [node for node, xyz in nodes.items() if np.all(np.asarray(xyz) >= minimum - 1e-7) and np.all(np.asarray(xyz) <= maximum + 1e-7)]
    if not selected:
        raise ValueError(f"Empty node selector: {selector}")
    return selected

def _set_lines(kind, name, values):
    data = [f"*{kind},{kind}={name}"]
    values = list(values)
    data.extend(",".join(str(v) for v in values[start:start + 16]) for start in range(0, len(values), 16))
    return data

def _mass_lines(nodes, elements, point_masses):
    lines = []
    coupling_info = []
    used = set()
    for index, mass in enumerate(point_masses):
        position = _vector(mass["position_mm"], "position_mm")
        selected = _select(nodes, mass["attachment_region"])
        coordinates = np.asarray([nodes[n] for n in selected])
        if np.linalg.matrix_rank(coordinates - coordinates[0], tol=1e-8) < 2:
            raise ValueError("Point-mass patch must contain at least three non-collinear nodes")
        if used.intersection(selected):
            raise ValueError("Point-mass attachment regions must not share nodes")
        used.update(selected)
        node_id = max(nodes) + index + 1
        element_id = max(elements) + index + 1
        lines.extend(["*NODE", f"{node_id}," + ",".join(f"{v:.12g}" for v in position), f"*ELEMENT,TYPE=MASS,ELSET=PM{index}", f"{element_id},{node_id}", f"*MASS,ELSET=PM{index}", f"{mass['mass_g'] * 1e-6:.12g}"])
        lines.extend(_set_lines("NSET", f"ATTACH{index}", selected))
        lines.append(f"*RIGID BODY,NSET=ATTACH{index},REF NODE={node_id}")
        coupling_info.append({"name": mass["name"], "mass_g": mass["mass_g"], "position_mm": list(mass["position_mm"]), "attachment_nodes": len(selected), "coupling": "rigid attachment patch with reference at point-mass COM; patch deformation suppressed"})
    return lines, coupling_info

def print_axes(axis):
    normal = _vector(axis, "print_axis")
    normal = normal / np.linalg.norm(normal)
    first = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = first - first.dot(normal) * normal
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first), normal

def _elastic_lines(material):
    constants = material.get("orthotropic")
    if not constants:
        return ["*ELASTIC", f"{material['young_modulus_mpa']:.12g},{material['poisson_ratio']:.12g}"], [], ""
    c = constants
    first, second, _ = print_axes(material.get("print_axis", (0, 0, 1)))
    elastic = ["*ELASTIC,TYPE=ENGINEERING CONSTANTS", ",".join(f"{v:.12g}" for v in (c["e_xy_mpa"], c["e_xy_mpa"], c["e_z_mpa"], c["nu_xy"], c["nu_xz"], c["nu_xz"], c["g_xy_mpa"], c["g_z_mpa"])), f"{c['g_z_mpa']:.12g}"]
    return elastic, ["*ORIENTATION,NAME=PRINTAXES,SYSTEM=RECTANGULAR", ",".join(f"{v:.12g}" for v in (*first, *second))], ",ORIENTATION=PRINTAXES"

def _model_lines(nodes, elements, material, point_masses):
    lines = ["*HEADING", "Deep Frame linear elastic analysis", "*NODE"]
    lines.extend(f"{node}," + ",".join(f"{v:.12g}" for v in xyz) for node, xyz in nodes.items())
    lines.append("*ELEMENT,TYPE=C3D10,ELSET=FRAME")
    lines.extend(f"{element}," + ",".join(str(n) for n in connectivity) for element, connectivity in elements.items())
    lines.extend(_set_lines("NSET", "NALL", nodes))
    elastic, orientation, section = _elastic_lines(material)
    lines.extend(["*MATERIAL,NAME=PRINT", *elastic, "*DENSITY", f"{material['density_g_cm3'] * 1e-9:.12g}", *orientation, "*SOLID SECTION,ELSET=FRAME,MATERIAL=PRINT" + section])
    mass_lines, information = _mass_lines(nodes, elements, point_masses)
    return lines + mass_lines, information

def _case_lines(nodes, case, settings):
    fixed = sorted({node for region in case["fixed_regions"] for node in _select(nodes, region)})
    if not fixed or np.linalg.matrix_rank(np.asarray([nodes[n] for n in fixed]) - nodes[fixed[0]], tol=1e-8) < 2:
        raise ValueError("Fixture must restrain at least three non-collinear mesh nodes")
    lines = _set_lines("NSET", "FIXED", fixed)
    lines.extend(["*BOUNDARY", "FIXED,1,3", "*STEP"])
    solver_option = ",SOLVER=" + settings["linear_solver"] if settings.get("linear_solver") else ""
    load_nodes = []
    if case["analysis"] == "modal":
        lines.extend(["*FREQUENCY" + solver_option, str(settings["num_modes"])])
    elif case["analysis"] == "static":
        if not case.get("loads"):
            raise ValueError("Static load case needs loads")
        lines.extend(["*STATIC" + solver_option, "*CLOAD"])
        accumulated = {}
        for load in case["loads"]:
            selected = _select(nodes, load["region"])
            if set(selected).intersection(fixed):
                raise ValueError("A load region overlaps the fixed fixture")
            force = _vector(load["force_n"], "force_n")
            if np.linalg.norm(force) == 0:
                raise ValueError("A static load must have non-zero force")
            for node in selected:
                accumulated[node] = accumulated.get(node, np.zeros(3)) + force / len(selected)
            load_nodes.append((selected, force))
        for node, force in accumulated.items():
            lines.extend(f"{node},{axis + 1},{value:.12g}" for axis, value in enumerate(force) if value)
        lines.extend(["*NODE PRINT,NSET=NALL,GLOBAL=YES", "U", "*EL PRINT,ELSET=FRAME,GLOBAL=YES", "S"])
    else:
        raise ValueError(f"Unknown analysis type: {case['analysis']}")
    lines.append("*END STEP")
    return lines, load_nodes, len(fixed)

def _numeric_lines(content, header, columns):
    results = []
    active = False
    for line in content.splitlines():
        if header.lower() in line.lower():
            active = True
            continue
        if not active:
            continue
        fields = line.split()
        if len(fields) == columns:
            try:
                values = [float(v.replace("D", "E")) for v in fields]
            except ValueError:
                if results:
                    break
                continue
            results.append(values)
        elif results:
            break
    if not results or not np.all(np.isfinite(results)):
        raise RuntimeError(f"Missing or non-finite CalculiX output: {header}")
    return np.asarray(results)

def _static_result(content, load_nodes, axis=(0, 0, 1)):
    displacement_data = _numeric_lines(content, "displacements (", 4)
    stress_data = _numeric_lines(content, "stresses (", 8)
    displacements = {int(row[0]): row[1:] for row in displacement_data}
    sxx, syy, szz, sxy, sxz, syz = stress_data[:, 2:].T
    von_mises = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))
    n = print_axes(axis)[2]
    normal = sxx * n[0] ** 2 + syy * n[1] ** 2 + szz * n[2] ** 2 + 2 * (sxy * n[0] * n[1] + sxz * n[0] * n[2] + syz * n[1] * n[2])
    quantiles = {f"{name}_p{label}_mpa": float(np.percentile(values, q)) for label, q in (("99", 99.0), ("999", 99.9)) for name, values in (("von_mises", von_mises), ("print_normal_tension", np.maximum(normal, 0.0)), ("print_normal_abs", np.abs(normal)))}
    loading = []
    for selected, force in load_nodes:
        mean = np.mean([displacements[node] for node in selected], axis=0)
        directional = float(np.dot(mean, force / np.linalg.norm(force)))
        if directional <= 0:
            raise RuntimeError("Non-positive directional compliance")
        loading.append({"node_count": len(selected), "force_n": force.tolist(), "mean_displacement_mm": mean.tolist(), "directional_displacement_mm": directional, "stiffness_n_per_mm": float(np.linalg.norm(force) / directional)})
    return {"analysis": "static", "max_displacement_mm": float(np.max(np.linalg.norm(displacement_data[:, 1:], axis=1))), "max_von_mises_mpa": float(np.max(von_mises)), **quantiles, "stress_samples": len(von_mises), "print_axis": n.tolist(), "loads": loading, "stiffness_n_per_mm": loading[0]["stiffness_n_per_mm"] if len(loading) == 1 else None}

def _modal_result(content, settings):
    values = _numeric_lines(content, "E I G E N V A L U E", 5)
    frequencies = values[:, 3]
    cutoff = settings.get("minimum_elastic_frequency_hz", 0.1)
    elastic = [float(value) for value in frequencies if value > cutoff]
    if not elastic:
        raise RuntimeError("No elastic eigenfrequencies found")
    if np.any(values[:, 1] < -1e-6):
        raise RuntimeError("Negative modal eigenvalue")
    return {"analysis": "modal", "eigenfrequencies_hz": elastic, "discarded_rigid_modes": int(np.sum(frequencies <= cutoff)), "fixture": "all translations fixed on selected nodes"}

def _validate(solid, material, point_masses, load_cases, settings):
    if settings.get("linear_solver") not in (None, "SPOOLES", "PASTIX"):
        raise ValueError("linear_solver must be None, SPOOLES or PASTIX")
    threads = settings.get("threads", 2)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if _is_mesh(solid):
        if not (np.all(np.isfinite(solid.vertices)) and solid.is_watertight and solid.is_winding_consistent and solid.body_count == 1 and solid.volume > 0):
            raise ValueError("FEA requires one closed, consistently oriented triangle mesh with positive volume")
        if not settings["tet_attempts"] or not set(settings["tet_attempts"]) <= set(MESH_ATTEMPTS):
            raise ValueError(f"tet_attempts must be a non-empty subset of {MESH_ATTEMPTS}")
        targets = settings["fea_remesh_targets_mm"]
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < math.inf for value in [*(settings[key] for key in MESH_NUMBERS), *targets]) or not isinstance(settings["fea_remesh_iterations"], int):
            raise ValueError(f"{', '.join(MESH_KEYS[2:])} must be finite and positive, fea_remesh_iterations an integer")
        if not targets or any(finer >= coarser for coarser, finer in zip(targets, targets[1:])):
            raise ValueError("fea_remesh_targets_mm must be a non-empty, strictly decreasing list")
        if not 0 < settings["mesh_minimum_sicn"] < 1:
            raise ValueError("mesh_minimum_sicn must lie in (0, 1)")
    elif len(solid.solids()) != 1 or not solid.is_valid or solid.volume <= 0:
        raise ValueError("FEA requires exactly one valid solid with positive volume")
    for key in ("young_modulus_mpa", "density_g_cm3"):
        if not math.isfinite(material[key]) or material[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(material["poisson_ratio"]) or not -1 < material["poisson_ratio"] < 0.5:
        raise ValueError("poisson_ratio must be between -1 and 0.5")
    for key in ("mesh_size_mm", "mesh_timeout_s", "solver_timeout_s"):
        if not math.isfinite(settings.get(key, 180)) or settings.get(key, 180) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if settings.get("element_order", 2) != 2 or int(settings["num_modes"]) < 1:
        raise ValueError("C3D10 element order 2 and num_modes >= 1 required")
    if not load_cases or len({case["name"] for case in load_cases}) != len(load_cases):
        raise ValueError("Load cases must have unique names and must not be empty")
    for mass in point_masses:
        if not math.isfinite(mass["mass_g"]) or mass["mass_g"] <= 0:
            raise ValueError("Point masses must be finite and positive")
        _vector(mass["position_mm"], "position_mm")

def _is_mesh(solid):
    return hasattr(solid, "is_winding_consistent")

def _mesh_settings(settings):
    return {**{key: deepcopy(IMPLICIT_CONFIG[key]) for key in MESH_KEYS}, **settings}

def _element_budget(settings):
    import psutil
    own = psutil.Process().memory_info()
    process_mb, available_mb = getattr(own, "private", own.rss)/2**20, psutil.virtual_memory().available/2**20
    memory_mb = min(settings["fea_memory_budget_mb"]-process_mb, available_mb)
    return {"elements": max(0, int(memory_mb*1024/settings["fea_memory_per_element_kb"])), "solver_memory_mb": memory_mb, "process_mb": process_mb, "available_mb": available_mb,
            "memory_budget_mb": settings["fea_memory_budget_mb"], "memory_per_element_kb": settings["fea_memory_per_element_kb"]}

def _mesh_plan(settings):
    names, targets = settings["tet_attempts"], settings["fea_remesh_targets_mm"]
    return [(name, target) for target in targets for name in names if name.startswith("remesh")] + [(name, targets[0] if name == "refine_hxt" else None) for name in names if not name.startswith("remesh")]

def _skip_reason(name, target, attempts, angle, settings):
    if name == "direct_hxt" and angle < settings["fea_direct_minimum_angle_deg"]:
        return f"minimum input triangle angle {angle:.3g} deg below {settings['fea_direct_minimum_angle_deg']} deg"
    if not name.startswith("remesh"):
        return None
    for attempt in attempts:
        if attempt.get("over_budget") and attempt["name"].startswith("remesh") and attempt["target_mm"] >= target:
            return f"predicted element count exceeds the budget: {attempt['name']} at the coarser or equal surface target {attempt['target_mm']} mm already had {attempt['linear_element_count']} elements"
        if attempt.get("surface_failed") and attempt["target_mm"] == target and attempt["name"].startswith("remesh"):
            return f"prepared surface at {target} mm already failed in {attempt['name']}"
    return None

def _volume_mesh(solid, directory, settings, result):
    timeout, threads = settings.get("mesh_timeout_s", 180), settings.get("threads", 2)
    def mesher(name, **request):
        path = directory / name
        path.write_text(json.dumps({**request, "output_dir": str(directory), "settings": settings}), encoding="utf-8")
        return [sys.executable, "-m", "deep_frame.fea", str(path)]
    if not _is_mesh(solid):
        export_step(solid, directory / "solid.step")
        _run(mesher("mesh_request.json", step_path=str(directory / "solid.step")), directory, timeout, threads, directory / "gmsh.log")
        result["mesh"] = json.loads((directory / "mesh_metadata.json").read_text(encoding="utf-8"))
        return _read_mesh(directory / "mesh.inp")
    np.savez(directory / "surface.npz", vertices=np.asarray(solid.vertices, dtype=np.float64), faces=np.asarray(solid.faces, dtype=np.int64))
    angle = float(np.degrees(solid.face_angles.min()))
    attempts, budget = [], _element_budget(settings)
    result["mesh"] = {"attempts": attempts, "input_face_count": len(solid.faces), "input_minimum_angle_deg": angle, "memory_budget": budget}
    for index, (name, target) in enumerate(_mesh_plan(settings)):
        start = time.monotonic()
        attempt = {"name": name, "target_mm": target, "status": "failed"}
        attempts.append(attempt)
        try:
            for stale in ("mesh.inp", "mesh.msh", "mesh_metadata.json", "attempt_metadata.json"):
                (directory / stale).unlink(missing_ok=True)
            reason = _skip_reason(name, target, attempts[:-1], angle, settings)
            if reason:
                attempt.update(status="skipped", diagnostic=reason)
                continue
            label = f"{index:02d}_{name}" + (f"_{target:g}" if target else "")
            limit = timeout if name.startswith(("remesh", "refine")) else min(timeout, settings["fea_fallback_timeout_s"])
            attempt["timeout_s"] = limit
            _run(mesher(f"mesh_request_{label}.json", surface_path=str(directory / "surface.npz"), attempt=name, target_mm=target, element_budget=budget["elements"]), directory, limit, threads, directory / f"gmsh_{label}.log")
            nodes, elements = _read_mesh(directory / "mesh.inp")
            result["mesh"].update(json.loads((directory / "mesh_metadata.json").read_text(encoding="utf-8")))
            attempt.update(status="ok", linear_element_count=result["mesh"]["linear_element_count"])
            return nodes, elements
        except Exception as error:
            attempt["diagnostic"] = f"{type(error).__name__}: {str(error)[-800:]}"
            if (directory / "attempt_metadata.json").exists():
                attempt.update(json.loads((directory / "attempt_metadata.json").read_text(encoding="utf-8")))
            attempt["surface_failed"] = attempt.get("prepared_surface", {}).get("passed") is False or "Prepared FEA surface" in attempt["diagnostic"]
        finally:
            attempt["runtime_s"] = time.monotonic() - start
    raise RuntimeError("All tetrahedral meshing attempts failed: " + ", ".join(f"{attempt['name']} {attempt['status']}" for attempt in attempts))

def evaluate(solid, material, point_masses, load_cases, settings):
    start = time.monotonic()
    result = {"status": "failed", "mass_g": None, "eigenfrequencies_hz": [], "max_displacement_mm": None, "max_von_mises_mpa": None, "stiffness_n_per_mm": None, "load_cases": {}, "diagnostics": [], "artifacts": {}}
    directory = None
    try:
        if _is_mesh(solid):
            settings = _mesh_settings(settings)
        _validate(solid, material, point_masses, load_cases, settings)
        result.update(linear_solver=settings.get("linear_solver"), solver_threads=settings.get("threads", 2))
        solver = resolve_solver(settings)
        base = Path(settings["work_dir"]).resolve()
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(prefix="evaluation_", dir=base))
        result["artifacts"]["directory"] = str(directory)
        nodes, elements = _volume_mesh(solid, directory, settings, result)
        result["mesh"]["node_count"] = len(nodes)
        model, coupling = _model_lines(nodes, elements, material, point_masses)
        result["point_mass_coupling"] = coupling
        attachments = {node for mass in point_masses for node in _select(nodes, mass["attachment_region"])}
        for case in load_cases:
            fixed_nodes = {node for region in case["fixed_regions"] for node in _select(nodes, region)}
            if attachments.intersection(fixed_nodes):
                raise ValueError("A point-mass attachment region overlaps a fixed fixture")
        prepared = [(case, *_case_lines(nodes, case, settings)) for case in load_cases]
        result["frame_mass_g"] = float(solid.volume * material["density_g_cm3"] / 1000)
        if _is_mesh(solid):
            result["model_frame_mass_g"] = float(result["mesh"]["tet_volume_mm3"] * material["density_g_cm3"] / 1000)
        result["point_mass_g"] = float(sum(mass["mass_g"] for mass in point_masses))
        result["mass_g"] = result["frame_mass_g"] + result["point_mass_g"]
        for index, (case, lines, load_nodes, fixed_count) in enumerate(prepared):
            name = f"case_{index}"
            (directory / f"{name}.inp").write_text("\n".join(model + lines) + "\n", encoding="ascii")
            log = _run([solver, "-i", name], directory, settings.get("solver_timeout_s", 180), settings.get("threads", 2), directory / f"{name}.log")
            if "Job finished" not in log:
                raise RuntimeError(f"CalculiX did not finish case {case['name']}")
            content = (directory / f"{name}.dat").read_text(encoding="utf-8", errors="replace")
            current = _static_result(content, load_nodes, material.get("print_axis", (0, 0, 1))) if case["analysis"] == "static" else _modal_result(content, settings)
            current["fixed_node_count"] = fixed_count
            result["load_cases"][case["name"]] = current
            result["artifacts"][case["name"]] = {"input": str(directory / f"{name}.inp"), "data": str(directory / f"{name}.dat"), "log": str(directory / f"{name}.log")}
            version = re.search(r"Version\s+([\d.]+)", log)
            result["solver_version"] = version.group(1) if version else "unreported"
        static = [case for case in result["load_cases"].values() if case["analysis"] == "static"]
        modal = [case for case in result["load_cases"].values() if case["analysis"] == "modal"]
        result["max_displacement_mm"] = max((case["max_displacement_mm"] for case in static), default=None)
        result["max_von_mises_mpa"] = max((case["max_von_mises_mpa"] for case in static), default=None)
        result["eigenfrequencies_hz"] = sorted(value for case in modal for value in case["eigenfrequencies_hz"])
        stiffness_name = settings.get("stiffness_load_case")
        if static and stiffness_name:
            selected = result["load_cases"].get(stiffness_name, {})
            if selected.get("stiffness_n_per_mm") is None:
                raise ValueError("stiffness_load_case must identify a static case with one load")
            result["stiffness_n_per_mm"] = selected["stiffness_n_per_mm"]
        result["status"] = "ok"
    except (ValueError, KeyError, TypeError) as error:
        result["status"] = "invalid"
        result["diagnostics"].append(f"{type(error).__name__}: {error}")
    except Exception as error:
        result["status"] = "failed"
        result["diagnostics"].append(f"{type(error).__name__}: {error}")
    result["runtime_s"] = time.monotonic() - start
    if directory:
        result["artifacts"]["result"] = str(directory / "result.json")
        (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result

def _topology(mesh):
    return {"watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent), "body_count": int(mesh.body_count), "euler_number": int(mesh.euler_number)}

def _prepare_surface(source, settings, refine, target, record=None):
    import pymeshlab
    import trimesh
    from deep_frame.topology_implicit import mesh_checks
    start = time.monotonic()
    meshes = pymeshlab.MeshSet()
    def snapshot():
        current = meshes.current_mesh()
        return trimesh.Trimesh(current.vertex_matrix(), current.face_matrix(), process=False)
    def restore(mesh):
        meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64), face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    restore(source)
    if refine:
        meshes.meshing_surface_subdivision_midpoint(iterations=64, threshold=pymeshlab.PureValue(settings["fea_refine_edge_mm"]))
    meshes.meshing_isotropic_explicit_remeshing(iterations=settings["fea_remesh_iterations"], targetlen=pymeshlab.PureValue(target), featuredeg=settings["fea_remesh_feature_deg"],
                                                checksurfdist=True, maxsurfdist=pymeshlab.PureValue(settings["fea_refine_max_surface_distance_mm" if refine else "fea_remesh_max_surface_distance_mm"]))
    tolerance = settings["fea_merge_relative_tolerance"]*float(source.scale)
    steps = {"merge_close_vertices": lambda: meshes.meshing_merge_close_vertices(threshold=pymeshlab.PureValue(tolerance)),
             "remove_t_vertices": lambda: meshes.meshing_remove_t_vertices(method="Edge Collapse", threshold=settings["fea_t_vertex_ratio"], repeat=True)}
    surface, topology, rejected = snapshot(), {"input": _topology(source)}, {}
    topology["remeshed"] = _topology(surface)
    cleanup = topology["remeshed"] != topology["input"] or not mesh_checks(surface)["passed"] or bool(np.any(surface.face_adjacency_angles > math.radians(179)))
    for name, step in steps.items() if cleanup else ():
        step()
        meshes.meshing_remove_null_faces()
        meshes.meshing_remove_unreferenced_vertices()
        candidate = snapshot()
        if _topology(candidate) == _topology(surface):
            surface = candidate
        else:
            rejected[name] = _topology(candidate)
            restore(surface)
    topology["prepared"] = _topology(surface)
    checks = mesh_checks(surface)
    report = {"refined_input": refine, "target_mm": target, "merge_tolerance_mm": tolerance, "cleanup_applied": cleanup, "face_count": len(surface.faces), "minimum_angle_deg": float(np.degrees(surface.face_angles.min())),
              "topology": topology, "topology_changed": topology["prepared"] != topology["input"], "rejected_steps": rejected, "topology_passed": checks["topology"]["passed"],
              "self_intersections_passed": checks["self_intersections"]["passed"], "folded_edges": int(np.sum(surface.face_adjacency_angles > math.radians(179))), "runtime_s": time.monotonic() - start}
    report["passed"] = not report["topology_changed"] and checks["passed"] and not report["folded_edges"]
    if not report["passed"]:
        if record:
            Path(record).write_text(json.dumps({"prepared_surface": report}), encoding="utf-8")
        raise ValueError(f"Prepared FEA surface changed topology or is not one closed, oriented, fold- and self-intersection-free body: {report}")
    return surface, report

def uniform_surface(mesh, target, taubin):
    import pymeshlab
    import trimesh
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)))
    meshes.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(target), iterations=6, featuredeg=40)
    if taubin:
        meshes.apply_coord_taubin_smoothing(stepsmoothnum=taubin)
    return trimesh.Trimesh(meshes.current_mesh().vertex_matrix(), meshes.current_mesh().face_matrix())

def robust_surface(mesh, settings, plan):
    from deep_frame.topology_implicit import surface_fidelity
    settings, trials = _mesh_settings(settings), []
    for volume in settings["fea_remesh_targets_mm"]:
        for feature in plan["feature_degs"]:
            for target in plan["targets_mm"]:
                surface = uniform_surface(mesh, target, plan["taubin"])
                fidelity = surface_fidelity(mesh, surface)
                row = {"surface_mm": target, "taubin": plan["taubin"], "volume_mm": volume, "feature_deg": feature, "deviation_mm": fidelity["maximum_sampled_deviation_mm"], "relative_volume_change": fidelity["relative_volume_change"]}
                row["deviation_within_limit"] = row["deviation_mm"] <= settings["surface_deviation_mm"]
                if abs(row["relative_volume_change"]) > settings["relative_volume_change"]:
                    trials.append({**row, "passed": False, "diagnostic": "uniform surface changes the volume beyond the limit"})
                    continue
                try:
                    _prepare_surface(surface, {**settings, "fea_remesh_feature_deg": feature}, False, volume)
                    trials.append({**row, "passed": True})
                    return surface, trials, {"fea_remesh_targets_mm": [volume], "fea_remesh_feature_deg": feature}
                except ValueError as error:
                    trials.append({**row, "passed": False, "diagnostic": str(error)[-300:]})
    return mesh, trials, {}

def _collapse_short_edges(mesh, threshold):
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = mesh.edges_unique[mesh.edges_unique_length < threshold]
    count = len(mesh.vertices)
    _, labels = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count)), directed=False)
    vertices = np.zeros((labels.max() + 1, 3))
    np.add.at(vertices, labels, mesh.vertices)
    vertices /= np.bincount(labels)[:, None]
    faces = labels[mesh.faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    result = trimesh.Trimesh(vertices, faces, process=False)
    result.update_faces(result.unique_faces())
    result.remove_unreferenced_vertices()
    return result

def clean_slivers(mesh, settings):
    import pymeshlab
    import trimesh
    from deep_frame.topology_implicit import mesh_checks, surface_fidelity
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64), face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    meshes.meshing_isotropic_explicit_remeshing(iterations=settings["fea_sliver_iterations"], targetlen=pymeshlab.PureValue(settings["fea_sliver_target_mm"]), featuredeg=settings["fea_remesh_feature_deg"],
                                                checksurfdist=True, maxsurfdist=pymeshlab.PureValue(settings["fea_remesh_max_surface_distance_mm"]))
    current = meshes.current_mesh()
    surface = trimesh.Trimesh(current.vertex_matrix(), current.face_matrix(), process=False)
    collapsed = _collapse_short_edges(surface, settings["fea_sliver_collapse_mm"])
    if _topology(collapsed) == _topology(surface) and mesh_checks(collapsed)["passed"]:
        surface = collapsed
    fidelity = surface_fidelity(mesh, surface)
    report = {"target_mm": settings["fea_sliver_target_mm"], "iterations": settings["fea_sliver_iterations"], "collapse_mm": settings["fea_sliver_collapse_mm"], "collapsed": surface is collapsed, "input_minimum_angle_deg": float(np.degrees(mesh.face_angles.min())), "minimum_angle_deg": float(np.degrees(surface.face_angles.min())),
              "face_count": len(surface.faces), "topology_unchanged": _topology(surface) == _topology(mesh), "checks_passed": mesh_checks(surface)["passed"],
              "maximum_sampled_deviation_mm": fidelity["maximum_sampled_deviation_mm"], "relative_volume_change": fidelity["relative_volume_change"]}
    report["passed"] = report["topology_unchanged"] and report["checks_passed"] and report["maximum_sampled_deviation_mm"] <= settings["surface_deviation_mm"] and abs(report["relative_volume_change"]) <= settings["relative_volume_change"]
    return (surface if report["passed"] else mesh), report

def _cross(u, v, x, y):
    return (u[:, 0]-x)*(v[:, 1]-y)-(u[:, 1]-y)*(v[:, 0]-x)

def voxel_coverage(mesh, low, h, shape, block):
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    area = _cross(triangles[:, 1], triangles[:, 2], triangles[:, 0, 0], triangles[:, 0, 1])
    triangles, area = triangles[area != 0], area[area != 0]
    delta = np.zeros((shape[0], shape[1], shape[2]+1), dtype=np.float32)
    for start in range(0, len(triangles), block):
        t, a = triangles[start:start+block], area[start:start+block]
        first = np.ceil((t[:, :, :2].min(1)-low[:2])/h).astype(np.int64)
        span = np.maximum(np.floor((t[:, :, :2].max(1)-low[:2])/h).astype(np.int64)-first+1, 0)
        counts = span[:, 0]*span[:, 1]
        index = np.repeat(np.arange(len(t)), counts)
        local = np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts, counts)
        i, j = first[index, 0]+local//span[index, 1], first[index, 1]+local % span[index, 1]
        x, y, t, a = low[0]+i*h, low[1]+j*h, t[index], a[index]
        weights = np.stack([_cross(t[:, (k+1) % 3], t[:, (k+2) % 3], x, y)/a for k in range(3)])
        inside = np.all(weights >= 0, axis=0)
        q = (np.einsum("ki,ik->i", weights[:, inside], t[inside, :, 2])-low[2])/h
        cell, sign, i, j = np.floor(q+0.5).astype(np.int64), np.where(a[inside] > 0, -1.0, 1.0), i[inside], j[inside]
        fraction = cell+0.5-q
        np.add.at(delta, (i, j, cell), (sign*fraction).astype(np.float32))
        np.add.at(delta, (i, j, cell+1), (sign*(1-fraction)).astype(np.float32))
    np.cumsum(delta, axis=2, out=delta)
    tolerance = 1e-3
    bad = (np.abs(delta[:, :, -1]) > tolerance) | (delta.max(axis=2) > 1+tolerance) | (delta.min(axis=2) < -tolerance)
    return np.clip(delta[:, :, :-1], 0, 1), int(bad.sum())

def _local_depth(inside, h, tiers):
    from scipy.ndimage import distance_transform_edt, maximum_filter
    depth = distance_transform_edt(inside)*h
    return [maximum_filter(depth, size=2*int(np.ceil(radius/h))+1) for radius, _ in tiers]

def _levels(points, depths, low, h, tiers):
    index = tuple(np.clip(np.round((points-low)/h).astype(np.int64), 0, np.array(depths[0].shape)-1).T)
    level = np.full(len(points), len(tiers))
    for k in reversed(range(len(tiers))):
        level[depths[k][index] < tiers[k][0]] = k
    return level

def _isotropic(mesh, target, plan, level=None, minimum=0):
    import pymeshlab
    import trimesh
    meshes = pymeshlab.MeshSet()
    arrays = {} if level is None else {"v_scalar_array": level.astype(np.float64)}
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64), face_matrix=np.asarray(mesh.faces, dtype=np.int32), **arrays))
    if level is not None:
        meshes.compute_selection_by_condition_per_vertex(condselect=f"q >= {minimum}")
        meshes.compute_selection_transfer_vertex_to_face(inclusive=False)
    meshes.meshing_isotropic_explicit_remeshing(iterations=plan["iterations"], targetlen=pymeshlab.PureValue(target), featuredeg=plan["feature_deg"], checksurfdist=True,
                                                maxsurfdist=pymeshlab.PureValue(plan["max_surface_distance_mm"]), selectedonly=level is not None)
    current = meshes.current_mesh()
    return trimesh.Trimesh(current.vertex_matrix(), current.face_matrix(), process=False)

def _meshset(mesh):
    import pymeshlab
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64), face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    return meshes

def _current(meshes):
    import trimesh
    return trimesh.Trimesh(meshes.current_mesh().vertex_matrix(), meshes.current_mesh().face_matrix(), process=False)

def _repair_intersections(mesh, target, plan):
    import pymeshlab
    repaired = []
    for _ in range(plan["repair_rounds"]):
        meshes = _meshset(mesh)
        meshes.compute_selection_by_self_intersections_per_face()
        count = int(meshes.current_mesh().selected_face_number())
        if not count:
            break
        repaired.append(count)
        for _ in range(plan["repair_rings"]):
            meshes.apply_selection_dilatation()
        meshes.meshing_isotropic_explicit_remeshing(iterations=plan["iterations"], targetlen=pymeshlab.PureValue(target), featuredeg=plan["repair_feature_deg"], checksurfdist=True,
                                                    maxsurfdist=pymeshlab.PureValue(plan["max_surface_distance_mm"]), selectedonly=True)
        candidate = _current(meshes)
        if _topology(candidate) != _topology(mesh):
            break
        mesh = candidate
    return mesh, repaired

def graded_surface(mesh, settings, plan):
    import trimesh
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    from deep_frame.topology_implicit import mesh_checks, surface_fidelity
    start, h = time.monotonic(), plan["voxel_mm"]
    low = mesh.bounds[0]-(plan["pad_cells"]+0.5)*h+np.array([math.pi, math.e, math.sqrt(2)])*1e-5
    shape = np.ceil((mesh.bounds[1]-low)/h).astype(np.int64)+plan["pad_cells"]+1
    field, bad_columns = np.zeros(shape, dtype=np.float32), 0
    for axes in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        order = np.array(axes)
        coverage, bad = voxel_coverage(trimesh.Trimesh(np.asarray(mesh.vertices)[:, order], mesh.faces, process=False), low[order], h, shape[order], plan["block_faces"])
        field += np.transpose(coverage, np.argsort(order))/3
        bad_columns += bad
        del coverage
    if plan["sigma_cells"]:
        field = gaussian_filter(field, plan["sigma_cells"], output=np.float32)
    vertices, faces, _, _ = marching_cubes(field, 0.5, spacing=(h, h, h), allow_degenerate=False)
    voxel = trimesh.Trimesh(vertices+low, faces, process=True)
    if voxel.volume < 0:
        voxel.invert()
    stride = plan["sizing_stride"]
    depths = _local_depth(field[::stride, ::stride, ::stride] > 0.5, h*stride, plan["tiers"])
    del field
    sizes = [size for _, size in plan["tiers"]]+[plan["coarse_mm"]]
    surface = _isotropic(voxel, sizes[0], plan)
    for minimum, size in enumerate(sizes[1:], 1):
        surface = _isotropic(surface, size, plan, _levels(np.asarray(surface.vertices), depths, low, h*stride, plan["tiers"]), minimum)
    parts = trimesh.Trimesh(surface.vertices, surface.faces, process=True).split(only_watertight=False)
    surface = max(parts, key=lambda part: abs(part.volume))
    dropped = float(sum(abs(part.volume) for part in parts)-abs(surface.volume))
    collapsed = _collapse_short_edges(surface, plan["collapse_mm"])
    if _topology(collapsed) == _topology(surface) and collapsed.face_angles.min() > surface.face_angles.min():
        surface = collapsed
    surface, repaired = _repair_intersections(surface, sizes[0], plan)
    checks, fidelity = mesh_checks(surface), surface_fidelity(mesh, surface)
    level = _levels(np.asarray(surface.vertices), depths, low, h*stride, plan["tiers"])
    report = {"voxel_mm": h, "grid_shape": shape.tolist(), "inconsistent_columns": bad_columns, "voxel_face_count": len(voxel.faces), "voxel_volume_change": float(voxel.volume/mesh.volume-1), "sizes_mm": sizes, "tier_vertex_counts": np.bincount(level, minlength=len(sizes)).tolist(), "dropped_parts": len(parts)-1, "dropped_volume_mm3": dropped, "intersection_repairs": repaired,
              "face_count": len(surface.faces), "minimum_angle_deg": float(np.degrees(surface.face_angles.min())), "folded_edges": int(np.sum(surface.face_adjacency_angles > math.radians(179))),
              "topology": {"input": _topology(mesh), "graded": _topology(surface)}, "checks_passed": checks["passed"], "self_intersections_passed": checks["self_intersections"]["passed"],
              "print_deviation_mm": fidelity["maximum_sampled_deviation_mm"], "print_to_fea_mm": fidelity["reference_to_approximation"]["surface_distance_mm"], "fea_to_print_mm": fidelity["approximation_to_reference"]["surface_distance_mm"],
              "relative_volume_change": fidelity["relative_volume_change"], "runtime_s": time.monotonic()-start}
    report["deviation_within_limit"] = report["print_deviation_mm"] <= settings["surface_deviation_mm"]
    report["passed"] = checks["passed"] and not report["folded_edges"] and report["minimum_angle_deg"] >= plan["minimum_angle_deg"] and abs(report["relative_volume_change"]) <= settings["relative_volume_change"] and surface.body_count == 1
    return surface, report

def _surface_model(gmsh, request, settings):
    import trimesh
    data = np.load(request["surface_path"])
    source = trimesh.Trimesh(data["vertices"], data["faces"], process=False)
    attempt = request["attempt"]
    report = {"attempt": attempt, "target_mm": request["target_mm"]}
    surface = source
    if attempt.startswith(("remesh", "refine")):
        surface, report["prepared_surface"] = _prepare_surface(source, settings, attempt.startswith("refine"), request["target_mm"], Path(request["output_dir"]) / "attempt_metadata.json")
    tag = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, tag, np.arange(1, len(surface.vertices) + 1), np.asarray(surface.vertices, dtype=np.float64).ravel())
    gmsh.model.mesh.addElementsByType(tag, 2, [], np.asarray(surface.faces, dtype=np.int64).ravel() + 1)
    if attempt.startswith("classify"):
        gmsh.model.mesh.classifySurfaces(math.radians(settings["fea_classify_angle_deg"]), True, True, math.pi)
        gmsh.model.mesh.createGeometry()
    gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([entity for _, entity in gmsh.model.getEntities(2)])])
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm3D", 1 if attempt.endswith("delaunay") else 10)
    return source, report

def _boundary_report(gmsh, source, settings):
    import trimesh
    from deep_frame.topology_implicit import surface_fidelity
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    lookup = np.zeros(int(tags.max()) + 1, dtype=np.int64)
    lookup[tags] = np.arange(len(tags))
    coordinates = coordinates.reshape(-1, 3)
    types, _, connectivity = gmsh.model.mesh.getElements(2)
    if list(types) != [2]:
        raise ValueError("The tetrahedral boundary must consist of linear triangles")
    boundary = trimesh.Trimesh(coordinates, lookup[connectivity[0].astype(np.int64)].reshape(-1, 3), process=False)
    boundary.remove_unreferenced_vertices()
    _, distance, _ = trimesh.proximity.closest_point(source, boundary.vertices)
    fidelity = surface_fidelity(source, boundary)
    types, _, connectivity = gmsh.model.mesh.getElements(3)
    if list(types) != [4]:
        raise ValueError("The volume mesh must consist of linear tetrahedra before elevation")
    corners = coordinates[lookup[connectivity[0].astype(np.int64)].reshape(-1, 4)]
    volume = float(np.sum(np.einsum("ij,ij->i", corners[:, 1]-corners[:, 0], np.cross(corners[:, 2]-corners[:, 0], corners[:, 3]-corners[:, 0])))/6)
    report = {"boundary_node_count": len(boundary.vertices), "boundary_node_deviation_mm": float(distance.max()), "boundary_fidelity": fidelity, "input_volume_mm3": float(source.volume), "tet_volume_mm3": volume,
              "tet_volume_relative_change": volume/source.volume-1}
    if not distance.max() <= settings["mesh_boundary_deviation_mm"]:
        raise ValueError(f"Boundary nodes deviate {distance.max():.4g} mm from the input surface (limit {settings['mesh_boundary_deviation_mm']} mm)")
    if not fidelity["maximum_sampled_deviation_mm"] <= settings["surface_deviation_mm"] or not abs(fidelity["relative_volume_change"]) <= settings["relative_volume_change"] or not abs(report["tet_volume_relative_change"]) <= settings["relative_volume_change"]:
        raise ValueError(f"Tetrahedral body deviates {fidelity['maximum_sampled_deviation_mm']:.4g} mm and {report['tet_volume_relative_change']:.4%} in volume from the input surface (limits {settings['surface_deviation_mm']} mm, {settings['relative_volume_change']:.2%})")
    return report

def generate_mesh(request_path):
    import gmsh
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    settings = request["settings"]
    destination = Path(request["output_dir"])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", settings.get("mesh_threads", 1))
        gmsh.option.setNumber("Mesh.MeshSizeMax", settings["mesh_size_mm"])
        gmsh.option.setNumber("Mesh.MeshSizeMin", settings.get("mesh_min_size_mm", 0.0))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", settings.get("mesh_curvature_points", 12))
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        second_order_linear = settings.get("mesh_second_order_linear", False)
        high_order_optimize = settings.get("mesh_high_order_optimize", 0)
        if not isinstance(second_order_linear, bool) or high_order_optimize not in (0, 1, 2, 3, 4):
            raise ValueError("Explicit boolean mesh_second_order_linear and optimization mode 0..4 required")
        gmsh.option.setNumber("Mesh.SecondOrderLinear", int(second_order_linear))
        gmsh.model.add("frame")
        source, surface = None, {}
        if "surface_path" in request:
            source, surface = _surface_model(gmsh, request, settings)
        else:
            gmsh.model.occ.importShapes(request["step_path"])
            gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:
            raise ValueError("The STEP input must contain exactly one volume")
        gmsh.model.addPhysicalGroup(3, [volumes[0][1]], name="FRAME")
        gmsh.model.mesh.generate(3)
        if source is not None:
            count = len(gmsh.model.mesh.getElements(3)[1][0])
            measured = {"linear_element_count": count, "element_budget": request["element_budget"], "over_budget": count > request["element_budget"]}
            surface.update(measured)
            (destination / "attempt_metadata.json").write_text(json.dumps(surface), encoding="utf-8")
            if measured["over_budget"]:
                raise ValueError(f"{count} tetrahedra exceed the element budget of {request['element_budget']} derived from available memory")
            surface.update(_boundary_report(gmsh, source, settings))
        gmsh.model.mesh.setOrder(2)
        if high_order_optimize in (2, 3):
            gmsh.model.mesh.optimize("HighOrderElastic")
        if high_order_optimize in (1, 2):
            gmsh.model.mesh.optimize("HighOrder")
        if high_order_optimize == 4:
            gmsh.model.mesh.optimize("HighOrderFastCurving")
        types, tags, _ = gmsh.model.mesh.getElements(3)
        if list(types) != [11] or not len(tags[0]):
            raise ValueError("The volume mesh must consist of C3D10 tetrahedra")
        quality = gmsh.model.mesh.getElementQualities(tags[0], "minDetJac")
        if not np.all(np.isfinite(quality)) or np.min(quality) <= 0:
            raise ValueError("The mesh contains a non-positive Jacobian")
        if source is not None:
            sicn = gmsh.model.mesh.getElementQualities(tags[0], "minSICN")
            surface.update(minimum_sicn=float(np.min(sicn)), elements_below_sicn=int(np.sum(sicn < settings["mesh_minimum_sicn"])))
            (destination / "attempt_metadata.json").write_text(json.dumps(surface), encoding="utf-8")
            if not np.all(np.isfinite(sicn)) or np.min(sicn) < settings["mesh_minimum_sicn"]:
                raise ValueError(f"Minimum SICN {np.min(sicn):.4g} below {settings['mesh_minimum_sicn']}")
        gmsh.write(str(destination / "mesh.inp"))
        gmsh.write(str(destination / "mesh.msh"))
        metadata = {
            "gmsh_version": gmsh.__version__,
            "element_type": "C3D10",
            "element_count": len(tags[0]),
            "minimum_jacobian_mm3": float(np.min(quality)),
            "mesh_size_mm": settings["mesh_size_mm"],
            "second_order_linear": second_order_linear,
            "high_order_optimize": high_order_optimize,
            "boundary_geometry": "piecewise planar quadratic tetrahedra with straight midside nodes" if second_order_linear else "quadratic boundary nodes projected to CAD",
            **surface,
        }
        (destination / "mesh_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        gmsh.finalize()

def _box(minimum, maximum):
    return {"kind": "box", "min_mm": list(minimum), "max_mm": list(maximum)}

def solver_identity(settings):
    executable = resolve_solver(settings)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run([executable, "-v"], capture_output=True, text=True, timeout=10, creationflags=flags)
    match = re.search(r"Version\s+([\d.]+)", completed.stdout + completed.stderr)
    if not match:
        raise RuntimeError("CalculiX version could not be identified")
    return {
        "solver_path": executable,
        "calculix_version": match.group(1),
        "calculix_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
        "gmsh_version": version("gmsh"),
        "build123d_version": version("build123d"),
        "ocp_version": version("cadquery-ocp-novtk"),
    }

def prepare_frame_case(parameters, fea_config=None, integration_config=None):
    fea = deepcopy(fea_config if fea_config is not None else parameters["fea"])
    integration = deepcopy(integration_config if integration_config is not None else parameters["integration"])
    frame = parameters["frame"]
    if parameters["material"]["density_g_cm3"] != fea["material"]["density_g_cm3"]:
        raise ValueError("Geometry and FEA material densities must agree")
    for key, value in integration.items():
        if isinstance(value, (int, float)) and (not math.isfinite(value) or (value <= 0 and key != "battery_attachment_y_mm")):
            raise ValueError(f"Integration setting {key} must be finite and positive")
    for key in ("central_fixture_fraction", "camera_upper_height_fraction", "camera_length_fraction"):
        if integration[key] > 1:
            raise ValueError(f"Integration setting {key} must not exceed one")
    tolerance = integration["selection_tolerance_mm"]
    motors = motor_positions(parameters)
    pad_half = frame["motor_pad_radius_mm"] + integration["motor_pad_margin_mm"]
    motor_fixtures = [_box((x - pad_half, y - pad_half, -tolerance), (x + pad_half, y + pad_half, tolerance)) for x, y in motors.values()]
    fraction = integration["central_fixture_fraction"]
    central_fixture = _box(
        (-frame["body_width_mm"] * fraction / 2, -frame["body_length_mm"] * fraction / 2, -tolerance),
        (frame["body_width_mm"] * fraction / 2, frame["body_length_mm"] * fraction / 2, tolerance),
    )
    x, y = motors[integration["arm_tip_motor"]]
    arm_region = _box((x - pad_half, y - pad_half, frame["arm_height_mm"] - tolerance), (x + pad_half, y + pad_half, frame["arm_height_mm"] + tolerance))
    deck_half_width = frame["deck_width_mm"] / 2 + integration["battery_attachment_margin_mm"]
    band_half_width = integration["battery_attachment_band_width_mm"] / 2
    band_y = integration["battery_attachment_y_mm"]
    deck_region = _box(
        (-deck_half_width, band_y - band_half_width, frame["deck_top_mm"] - tolerance),
        (deck_half_width, band_y + band_half_width, frame["deck_top_mm"] + tolerance),
    )
    camera_half_width = (parameters["components"]["camera"]["width_mm"] + 2 * frame["camera_side_clearance_mm"] + 2 * frame["minimum_wall_mm"]) / 2
    camera_half_length = frame["cage_length_mm"] * integration["camera_length_fraction"] / 2
    camera_region = _box(
        (-camera_half_width - tolerance, frame["camera_y_mm"] - camera_half_length, frame["cage_height_mm"] * (1 - integration["camera_upper_height_fraction"])),
        (camera_half_width + tolerance, frame["camera_y_mm"] + camera_half_length, frame["cage_height_mm"] + tolerance),
    )
    battery = build_components(parameters, assembly_placements(parameters))["battery"]
    mass = {
        "name": "battery",
        "mass_g": battery["mass_g"],
        "position_mm": list(battery["center_of_mass_mm"]),
        "attachment_region": deck_region,
    }
    impact = mass["mass_g"] / 1000 * integration["standard_gravity_m_s2"] * integration["battery_impact_g_factor"]
    load_cases = [
        {"name": "arm_tip", "analysis": "static", "fixed_regions": [central_fixture], "loads": [{"region": arm_region, "force_n": [0.0, 0.0, -integration["arm_tip_force_n"]]}]},
        {"name": "battery_impact", "analysis": "static", "fixed_regions": motor_fixtures, "loads": [{"region": deck_region, "force_n": [0.0, 0.0, -impact]}]},
        {"name": "camera_side", "analysis": "static", "fixed_regions": motor_fixtures, "loads": [{"region": camera_region, "force_n": [integration["camera_side_force_n"], 0.0, 0.0]}]},
        {"name": "modes", "analysis": "modal", "fixed_regions": motor_fixtures},
    ]
    weight = integration["all_up_mass_g"] / 1000 * integration["standard_gravity_m_s2"]
    thrust = parameters["components"]["motor"]["thrust_n"] * integration["thrust_safety_factor"]
    pads = [_box((mx - pad_half, my - pad_half, frame["arm_height_mm"] - tolerance), (mx + pad_half, my + pad_half, frame["arm_height_mm"] + tolerance)) for mx, my in motors.values()]
    def oblique(mx, my):
        radial, tangential = np.array([mx, my]) / math.hypot(mx, my), np.array([-my, mx]) / math.hypot(mx, my)
        vector = np.append(-radial + tangential, -0.5)
        return vector / np.linalg.norm(vector)
    load_cases[3:3] = [
        {"name": "thrust_all", "analysis": "static", "fixed_regions": [central_fixture], "loads": [{"region": pad, "force_n": [0.0, 0.0, thrust]} for pad in pads]},
        {"name": "crash_front", "analysis": "static", "fixed_regions": [central_fixture], "loads": [{"region": camera_region, "force_n": [0.0, -weight * integration["crash_front_g_factor"], 0.0]}]},
        {"name": "crash_arm", "analysis": "static", "fixed_regions": [central_fixture], "loads": [{"region": arm_region, "force_n": (weight * integration["crash_arm_g_factor"] * oblique(x, y)).tolist()}]},
    ]
    if integration["crash_directions"]:
        sides = {name: [pad for pad, (mx, _) in zip(pads, motors.values()) if np.sign(mx) == sign] for name, sign in (("side_left", -1), ("side_right", 1))}
        patches = {"front": [(camera_region, (0.0, -1.0, 0.0))], "side_left": [(pad, (1.0, 0.0, 0.0)) for pad in sides["side_left"]], "side_right": [(pad, (-1.0, 0.0, 0.0)) for pad in sides["side_right"]],
                   **{"arm_" + name: [(pad, oblique(mx, my))] for pad, (name, (mx, my)) in zip(pads, motors.items())}, "below": [(camera_region, (0.0, 0.0, 1.0))], "back": [(deck_region, (0.0, 0.0, -1.0))]}
        unknown = sorted(set(integration["crash_directions"]) - set(patches))
        if unknown:
            raise ValueError("Unknown crash directions: " + ", ".join(unknown))
        crash = weight * integration["crash_front_g_factor"]
        load_cases[3:] = [case for case in load_cases[3:] if not case["name"].startswith("crash_")]
        load_cases[4:4] = [{"name": "crash_" + name, "analysis": "static", "fixed_regions": [central_fixture], "loads": [{"region": deepcopy(region), "force_n": (crash / len(patches[name]) * np.asarray(direction)).tolist()} for region, direction in patches[name]]} for name in integration["crash_directions"]]
    fea["settings"]["stiffness_load_case"] = "arm_tip"
    result = {
        "material": fea["material"],
        "point_masses": [mass],
        "load_cases": load_cases,
        "settings": fea["settings"],
        "integration": integration,
        "mass_scope": "frame plus battery point mass; motors, props, camera and AIO excluded from FEA",
        "fixture_model": "arm: central base bottom rim fixed; other cases: four motor pad bottom surfaces fixed",
        "coupling_model": "battery COM rigidly coupled to a narrow transverse band of the deck rails; local patch deformation suppressed; no battery rotational inertia",
        "impact_model": "equivalent static force, not a transient impact or crash strength prediction",
    }
    return _json_copy(result)

class FrameEvaluator:
    def __init__(self, reference_parameters, fea_config=None, integration_config=None):
        self.fea_config = deepcopy(fea_config if fea_config is not None else reference_parameters["fea"])
        self.integration_config = deepcopy(integration_config if integration_config is not None else reference_parameters["integration"])
        self.toolchain = solver_identity(self.fea_config["settings"])
        self.fea_config["settings"]["solver_path"] = self.toolchain["solver_path"]
        reference_case = prepare_frame_case(reference_parameters, self.fea_config, self.integration_config)
        physical_settings = {key: value for key, value in self.fea_config["settings"].items() if key not in ("work_dir", "solver_path")}
        directory = Path(__file__).resolve().parent
        sources = {name: hashlib.sha256((directory / name).read_text(encoding="utf-8").encode()).hexdigest() for name in ("fea.py", "frame.py")}
        self.evaluation_contract = {
            "model_version": self.integration_config["model_version"],
            "material": self.fea_config["material"],
            "integration": self.integration_config,
            "settings": physical_settings,
            "reference_point_masses": reference_case["point_masses"],
            "reference_load_cases": reference_case["load_cases"],
            "toolchain": {key: value for key, value in self.toolchain.items() if key != "solver_path"},
            "source_sha256": sources,
        }
        self.evaluation_id = self.integration_config["model_version"] + ":" + _json_digest(self.evaluation_contract)
    def __call__(self, parameters):
        model = prepare_frame_case(parameters, self.fea_config, self.integration_config)
        solid = build_geometry(parameters)
        result = evaluate(solid, model["material"], model["point_masses"], model["load_cases"], model["settings"])
        result["evaluation_id"] = self.evaluation_id
        result["model_inputs"] = model
        result["evaluation_contract"] = deepcopy(self.evaluation_contract)
        return _json_copy(result)

if __name__ == "__main__":
    generate_mesh(sys.argv[1])
