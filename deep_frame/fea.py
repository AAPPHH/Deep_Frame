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
    from deep_frame.frame import assembly_placements, build_components, build_geometry, motor_positions

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
    with Path(log_path).open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=timeout, creationflags=flags)
    content = Path(log_path).read_text(encoding="utf-8", errors="replace")
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

def _model_lines(nodes, elements, material, point_masses):
    lines = ["*HEADING", "Deep Frame linear elastic analysis", "*NODE"]
    lines.extend(f"{node}," + ",".join(f"{v:.12g}" for v in xyz) for node, xyz in nodes.items())
    lines.append("*ELEMENT,TYPE=C3D10,ELSET=FRAME")
    lines.extend(f"{element}," + ",".join(str(n) for n in connectivity) for element, connectivity in elements.items())
    lines.extend(_set_lines("NSET", "NALL", nodes))
    lines.extend(["*MATERIAL,NAME=PRINT", "*ELASTIC", f"{material['young_modulus_mpa']:.12g},{material['poisson_ratio']:.12g}", "*DENSITY", f"{material['density_g_cm3'] * 1e-9:.12g}", "*SOLID SECTION,ELSET=FRAME,MATERIAL=PRINT"])
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
        lines.extend(["*NODE PRINT,NSET=NALL", "U", "*EL PRINT,ELSET=FRAME", "S"])
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

def _static_result(content, load_nodes):
    displacement_data = _numeric_lines(content, "displacements (", 4)
    stress_data = _numeric_lines(content, "stresses (", 8)
    displacements = {int(row[0]): row[1:] for row in displacement_data}
    sxx, syy, szz, sxy, sxz, syz = stress_data[:, 2:].T
    von_mises = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))
    loading = []
    for selected, force in load_nodes:
        mean = np.mean([displacements[node] for node in selected], axis=0)
        directional = float(np.dot(mean, force / np.linalg.norm(force)))
        if directional <= 0:
            raise RuntimeError("Non-positive directional compliance")
        loading.append({"node_count": len(selected), "force_n": force.tolist(), "mean_displacement_mm": mean.tolist(), "directional_displacement_mm": directional, "stiffness_n_per_mm": float(np.linalg.norm(force) / directional)})
    return {"analysis": "static", "max_displacement_mm": float(np.max(np.linalg.norm(displacement_data[:, 1:], axis=1))), "max_von_mises_mpa": float(np.max(von_mises)), "loads": loading, "stiffness_n_per_mm": loading[0]["stiffness_n_per_mm"] if len(loading) == 1 else None}

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
    if len(solid.solids()) != 1 or not solid.is_valid or solid.volume <= 0:
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

def evaluate(solid, material, point_masses, load_cases, settings):
    start = time.monotonic()
    result = {"status": "failed", "mass_g": None, "eigenfrequencies_hz": [], "max_displacement_mm": None, "max_von_mises_mpa": None, "stiffness_n_per_mm": None, "load_cases": {}, "diagnostics": [], "artifacts": {}}
    directory = None
    try:
        _validate(solid, material, point_masses, load_cases, settings)
        result.update(linear_solver=settings.get("linear_solver"), solver_threads=settings.get("threads", 2))
        solver = resolve_solver(settings)
        base = Path(settings["work_dir"]).resolve()
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(prefix="evaluation_", dir=base))
        result["artifacts"]["directory"] = str(directory)
        step_path = directory / "solid.step"
        export_step(solid, step_path)
        request = {"step_path": str(step_path), "output_dir": str(directory), "settings": settings}
        request_path = directory / "mesh_request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        _run([sys.executable, "-m", "deep_frame.fea", str(request_path)], directory, settings.get("mesh_timeout_s", 180), settings.get("threads", 2), directory / "gmsh.log")
        nodes, elements = _read_mesh(directory / "mesh.inp")
        result["mesh"] = json.loads((directory / "mesh_metadata.json").read_text(encoding="utf-8"))
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
        result["point_mass_g"] = float(sum(mass["mass_g"] for mass in point_masses))
        result["mass_g"] = result["frame_mass_g"] + result["point_mass_g"]
        for index, (case, lines, load_nodes, fixed_count) in enumerate(prepared):
            name = f"case_{index}"
            (directory / f"{name}.inp").write_text("\n".join(model + lines) + "\n", encoding="ascii")
            log = _run([solver, "-i", name], directory, settings.get("solver_timeout_s", 180), settings.get("threads", 2), directory / f"{name}.log")
            if "Job finished" not in log:
                raise RuntimeError(f"CalculiX did not finish case {case['name']}")
            content = (directory / f"{name}.dat").read_text(encoding="utf-8", errors="replace")
            current = _static_result(content, load_nodes) if case["analysis"] == "static" else _modal_result(content, settings)
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
        gmsh.model.occ.importShapes(request["step_path"])
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:
            raise ValueError("The STEP input must contain exactly one volume")
        gmsh.model.addPhysicalGroup(3, [volumes[0][1]], name="FRAME")
        gmsh.model.mesh.generate(3)
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
        }
        (destination / "mesh_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        gmsh.finalize()

def _box(minimum, maximum):
    return {"kind": "box", "min_mm": list(minimum), "max_mm": list(maximum)}

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

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
    return json.loads(json.dumps(result, allow_nan=False))

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
        self.evaluation_id = self.integration_config["model_version"] + ":" + _digest(self.evaluation_contract)
    def __call__(self, parameters):
        model = prepare_frame_case(parameters, self.fea_config, self.integration_config)
        solid = build_geometry(parameters)
        result = evaluate(solid, model["material"], model["point_masses"], model["load_cases"], model["settings"])
        result["evaluation_id"] = self.evaluation_id
        result["model_inputs"] = model
        result["evaluation_contract"] = deepcopy(self.evaluation_contract)
        return json.loads(json.dumps(result, allow_nan=False))

if __name__ == "__main__":
    generate_mesh(sys.argv[1])
