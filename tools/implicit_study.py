from copy import deepcopy
import importlib.metadata
import json
from itertools import product
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from deep_frame.config import PATHS, IMPLICIT_CONFIG, IMPLICIT_KINDS, command_line, configure
from deep_frame.fea import MESH_KEYS, evaluate
from deep_frame.frame import build_geometry
from deep_frame.topology_geometry import build_design_domain
from deep_frame.topology_implicit import ImplicitError, build_implicit, export_mesh
from deep_frame.topology_implicit_validation import ball_curvature, validate_implicit
from deep_frame.topology_pipeline import _file_digest, _plot_modules, _provenance, _read, _save, _verify_cases, candidate_entry, compare_to_baseline, log_run, run_log_path
from deep_frame.topology_surface_validation import surface_metrics
from tools.mature_pipeline import acceptance, artifacts, baseline_fea, comparison_inputs, draw_view, save_provenance, snapshot_source
from tools.workstation_study import load_source

REFERENCE_STEP = Path(str(Path(PATHS["data"]) / 'topology/workstation_20260930/candidate_study/grid4_iter300/candidates/density_t01/geometry.step'))
PACKAGES = ("numpy", "scipy", "scikit-image", "trimesh", "rtree", "pymeshlab", "manifold3d", "gmsh", "matplotlib", "psutil")
PARAMETERS = {"transition_radius_mm": "k_mm", "density_sigma_mm": "sigma_d_mm", "preserve_inflation_mm": "delta_mm", "constraint_offset_mm": "c_mm", "opening_radius_mm": "r_open_mm", "ripple_sigma_mm": "sigma_r_mm", "subdivisions": "subdivisions"}
VIEWS = (("isometric", 28, -48, "Isometrie"), ("top", 90, -90, "Draufsicht (+z)"), ("front", 0, -90, "Vorderansicht (-y)"), ("side", 0, 0, "Seitenansicht (+x)"))
LABELS = {True: "Geometrie-Gates bestanden - mechanische Freigabe separat pruefen", False: "DIAGNOSE - Geometrie-Gates nicht bestanden"}
RUN_CONFIG = {"source": None, "output": None, "reference_step": REFERENCE_STEP, "geometry_only": False, "study_timeout_s": 14400.0, "run_log": None, "section_heights_mm": [2.0, 27.0], "diagnostic_fea_always": False, **IMPLICIT_CONFIG}
RUN_KINDS = {"source": "path", "output": "path", "reference_step": "path", "geometry_only": "flag", "study_timeout_s": "float", "run_log": "path", "section_heights_mm": ["float"], "diagnostic_fea_always": "flag", **IMPLICIT_KINDS}
RENDER_CONFIG = {"geometry": None, "output": None, "inputs": None, "label": "", "section_heights_mm": [2.0, 27.0], **{key: IMPLICIT_CONFIG[key] for key in ("render_faces", "curvature_radius_mm", "curvature_samples")}}
RENDER_KINDS = {"geometry": "path", "output": "path", "inputs": "path", "label": "text", "section_heights_mm": ["float"], **{key: IMPLICIT_KINDS[key] for key in ("render_faces", "curvature_radius_mm", "curvature_samples")}}
SUMMARIZE_CONFIG = {"run_log": None, "output": None, "runs": None}
SUMMARIZE_KINDS = {"run_log": "path", "output": "path", "runs": ["text"]}

def peak_rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    memory = psutil.Process().memory_info()
    return getattr(memory, "peak_wset", memory.rss)/2**20

def decimated(mesh, faces):
    if len(mesh.faces) <= faces:
        return mesh
    import pymeshlab
    import trimesh
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64), face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    meshes.meshing_decimation_quadric_edge_collapse(targetfacenum=faces, qualitythr=0.5, preservenormal=True, planarquadric=True)
    return trimesh.Trimesh(meshes.current_mesh().vertex_matrix(), meshes.current_mesh().face_matrix(), process=False)

def rectangle(low, high):
    return np.asarray([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]], [low[0], low[1]]])

def footprints(domain, height):
    shapes = []
    for region in domain["regions"]:
        if region["role"] not in ("preserve", "forbidden"):
            continue
        if region["kind"] == "box":
            if region["min_mm"][2] <= height <= region["max_mm"][2]:
                shapes.append((region["role"], rectangle(region["min_mm"], region["max_mm"])))
            continue
        center, radius, half = np.asarray(region["center_mm"], dtype=float), region["radius_mm"], region["height_mm"]/2
        if region.get("axis", "z") == "z":
            if abs(height-center[2]) <= half:
                angle = np.linspace(0, 2*np.pi, 97)
                shapes.append((region["role"], np.column_stack((center[0]+radius*np.cos(angle), center[1]+radius*np.sin(angle)))))
        elif abs(height-center[2]) < radius:
            width = np.sqrt(radius**2-(height-center[2])**2)
            extent = np.asarray([half, width] if region["axis"] == "x" else [width, half])
            shapes.append((region["role"], rectangle(center[:2]-extent, center[:2]+extent)))
    return shapes

def render_views(mesh, domain, directory, label, footer, faces, heights):
    plt, _ = _plot_modules()
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    shown = decimated(mesh, faces)
    grid = domain["grid"]
    lower = np.asarray(grid["origin_mm"], dtype=float)
    upper = lower+np.asarray(grid["shape"])*np.asarray(grid["spacing_mm"])
    lower, upper = np.minimum(lower, mesh.bounds[0]), np.maximum(upper, mesh.bounds[1])
    record = {"images": {}, "sections": [], "mesh_faces": len(mesh.faces), "render_faces": len(shown.faces),
              "method": "Views: flat-shaded orthographic matplotlib Agg of a quadric-decimated copy used for display only; sections: exact plane cuts of the delivered full-resolution mesh with the exact preserve (green dashed) and forbidden (red dotted) footprints"}
    def save(figure, name):
        path = directory/(name+".png")
        figure.text(0.5, 0.01, footer, ha="center", fontsize=8)
        figure.savefig(path, dpi=150, facecolor="white")
        plt.close(figure)
        record["images"][name] = {"path": path.name, "sha256": _file_digest(path), "size_bytes": path.stat().st_size}
    color = "#9b392b" if label.startswith("DIAGNOSE") else "#275e3c"
    for name, elevation, azimuth, title in VIEWS:
        figure = plt.figure(figsize=(9, 8.2), layout="constrained")
        draw_view(figure.add_subplot(projection="3d"), shown, lower, upper, elevation, azimuth, title)
        figure.suptitle(label, fontsize=13, color=color)
        save(figure, "view_"+name)
    for height in heights:
        figure, axis = plt.subplots(figsize=(10, 9), layout="constrained")
        cut = mesh.section(plane_origin=[0, 0, height], plane_normal=[0, 0, 1])
        lines = [] if cut is None else [line[:, :2] for line in cut.discrete]
        axis.add_collection(LineCollection(lines, colors="#194653", linewidths=0.8))
        for role, outline in footprints(domain, height):
            axis.plot(outline[:, 0], outline[:, 1], color="#46904b" if role == "preserve" else "#c6553c", linestyle="--" if role == "preserve" else ":", linewidth=0.9)
        axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), xlabel="x / mm", ylabel="y / mm", title=f"Schnitt z = {height:g} mm")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.legend(handles=[Line2D([0], [0], color="#194653", label="Netzschnitt"), Line2D([0], [0], color="#46904b", linestyle="--", label="Preserve"), Line2D([0], [0], color="#c6553c", linestyle=":", label="Forbidden")], loc="upper right", fontsize=8)
        figure.suptitle(label, fontsize=13, color=color)
        save(figure, f"section_z{height:g}")
        record["sections"].append({"height_mm": height, "loops": len(lines), "length_mm": float(sum(np.linalg.norm(np.diff(line, axis=0), axis=1).sum() for line in lines))})
    record["elapsed_s"] = perf_counter()-started
    return record

def domain_currency(parameters, domain):
    current = build_design_domain(parameters)
    same = current["grid"]["shape"] == domain["grid"]["shape"]
    return {"regions_current": json.loads(json.dumps(current["regions"])) == domain["regions"], "grid_current": same,
            "mask_cells_changed": {name: int(np.count_nonzero(current[name] != domain[name])) if same else None for name in ("allowed", "preserve", "forbidden")},
            "prescribed_clearance_passed": current["metadata"]["prescribed_clearance"]["passed"]}

def reference_geometry(path, domain, config):
    from build123d import import_step
    from deep_frame.topology_surface_validation import _mesh, _settings
    solid = import_step(path)
    if not solid.is_valid or len(solid.solids()) != 1 or solid.volume <= 0:
        raise ValueError("Reference STEP must be a valid single positive-volume solid")
    mesh = _mesh(solid, _settings({}))
    return mesh, {"source": str(Path(path).resolve()), "sha256": _file_digest(path), "volume_mm3": solid.volume, "metrics": surface_metrics(mesh, domain),
                  "curvature": ball_curvature(mesh, config["curvature_radius_mm"], config["curvature_samples"]), "triangle_count": len(mesh.faces)}

class ImplicitStudy:
    def __init__(self, config):
        self.config, self.records, self.baseline = config, [], None
        self.source, self.output = config["source"].resolve(), config["output"].resolve()
        self.started = perf_counter()
        self.candidates = list(product(config["thresholds"], config["extensions"]))
        self.manifest = {"schema_version": "deep-frame-implicit-geometry-study-v1", "status": "running", "stage": "preparing", "density_source": str(self.source),
                         "geometry_only": config["geometry_only"], "candidate_plan": [{"threshold": t, "extension": e} for t, e in self.candidates],
                         "accepted_count": 0, "selected_id": None, "overall_acceptance": False, "candidates": [], "study_timeout_s": config["study_timeout_s"],
                         "acceptance_scope": "Geometry checks C1-C10 and the mass screen on the final mesh, then independent FEA against v0; render review remains required",
                         "budget_policy": "Study and candidate budgets checked at stage boundaries; FEA meshing and each solver case have their own timeouts"}

    def status(self, **values):
        _save(self.output/"status.json", {"status": self.manifest["status"], "stage": self.manifest["stage"], **values})

    def prepare(self):
        config = self.config
        load_source(self.source)
        source_manifest, snapshot = snapshot_source(self.source, self.output)
        inputs, density_result, self.domain, self.density = load_source(snapshot)
        historical, self.settings = comparison_inputs(inputs, self.domain, config["mesh_timeout_s"], config["solver_timeout_s"])
        self.constraints = historical["settings"]["relative_constraints"]
        self.reference_mesh, self.reference = reference_geometry(config["reference_step"], self.domain, config)
        _save(self.output/"reference_metrics.json", self.reference)
        self.baseline_solid = build_geometry(deepcopy(inputs["parameters"]))
        self.baseline_mass = self.baseline_solid.volume*self.domain["material"]["density_g_cm3"]/1000
        self.manifest.update(source_manifest_sha256=_file_digest(snapshot/"manifest.json"), density_status=density_result["status"], domain_currency=domain_currency(inputs["parameters"], self.domain),
                             implicit_settings={key: config[key] for key in IMPLICIT_CONFIG}, fea_settings=self.settings, mesh_settings={key: config[key] for key in MESH_KEYS},
                             comparison_reference={key: self.reference[key] for key in ("source", "sha256", "volume_mm3", "metrics")},
                             material=self.domain["material"], point_masses=self.domain["point_masses"], load_cases=self.domain["comparison_load_cases"], relative_constraints=self.constraints,
                             packages={name: importlib.metadata.version(name) for name in PACKAGES},
                             source_artifacts=save_provenance(self.output, [Path(__file__), ROOT/"tools/mature_pipeline.py", ROOT/"tools/workstation_study.py"]),
                             runtime_provenance=_provenance({"builder": build_implicit, "validator": validate_implicit, "evaluator": evaluate, "baseline_builder": build_geometry}, self.settings, True),
                             v0_geometry={"volume_mm3": self.baseline_solid.volume, "frame_mass_g": self.baseline_mass})

    def progress(self, record, directory, state, started):
        def callback(event):
            event = dict(event)
            mesh, report = event.pop("mesh", None), event.pop("report", None)
            if mesh is not None:
                state["mesh"] = mesh
            if report is not None and event["stage"].startswith("validation_"):
                record.setdefault("validation_progress", {})[event["stage"][11:]] = report.get("passed")
            elif report is not None:
                record["geometry"] = report
            record["stage"] = event["stage"]
            event.update(candidate=record["id"], status=record["status"], elapsed_s=perf_counter()-started)
            with (directory/"progress.jsonl").open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(event, allow_nan=False)+"\n")
            self.status(candidate=record["id"], candidate_status=record["status"], candidate_stage=event["stage"])
            if event["elapsed_s"] >= self.config["candidate_timeout_s"]:
                raise TimeoutError("Candidate runtime budget exceeded at "+event["stage"])
            if perf_counter()-self.started >= self.config["study_timeout_s"]:
                raise TimeoutError("Study runtime budget exceeded at "+event["stage"])
        return callback

    def step(self, record, status, progress):
        record["status"] = status
        progress({"stage": status})

    def build(self, directory, record, progress):
        config, timings = self.config, record["timings_s"]
        settings = {**{key: config[key] for key in IMPLICIT_CONFIG}, "threshold": record["parameters"]["threshold"], "extension": record["parameters"]["extension"]}
        mesh, report, field = build_implicit(self.domain, self.density, settings, progress)
        record["geometry"] = report
        timings.update(report["timings_s"])
        record["parameters"]["h_mm"] = max(report["field"]["spacing_mm"])
        clock = perf_counter()
        record["final_mesh"] = {"files": export_mesh(mesh, directory), "faces": len(mesh.faces), "vertices": len(mesh.vertices), "volume_mm3": float(mesh.volume)}
        timings["export"] = perf_counter()-clock
        self.step(record, "validating", progress)
        validation = validate_implicit(mesh, self.domain, field, report, None, self.reference["metrics"], self.reference_mesh, progress)
        del field
        record["surface_metrics"], record["material_change"] = validation.pop("surface_metrics"), validation.pop("material_change")
        record["validation"], timings["validation"] = validation, validation["timings_s"]
        record["wall_warning"] = validation["checks"].get("features", {}).get("wall_warning")
        clock = perf_counter()
        record["curvature"] = ball_curvature(mesh, config["curvature_radius_mm"], config["curvature_samples"])
        timings["metrics"] = perf_counter()-clock
        record["mass_g"] = float(mesh.volume)*self.domain["material"]["density_g_cm3"]/1000
        ratio = record["mass_g"]/self.baseline_mass
        record["mass_screen"] = {"ratio_to_v0": ratio, "maximum_ratio": self.constraints["frame_mass_ratio_max"], "passed": ratio <= self.constraints["frame_mass_ratio_max"]}
        passed = bool(validation["passed"] and record["mass_screen"]["passed"])
        record["geometry_valid"] = passed
        if not passed:
            record["failure_stage"] = "validation"
            record["diagnostics"].extend(validation["violations"]+([] if record["mass_screen"]["passed"] else ["mass_screen"]))
        self.step(record, "geometry_valid" if passed else "geometry_invalid", progress)
        clock = perf_counter()
        footer = f"{record['id']} | t = {settings['threshold']:g}, {settings['extension']} | geometry.stl SHA256 {record['final_mesh']['files']['geometry.stl']['sha256'][:16]} | {len(mesh.faces)} Dreiecke"
        record["renders"] = render_views(mesh, self.domain, directory/"renders", LABELS[passed], footer, config["render_faces"], config["section_heights_mm"])
        timings["renders"] = perf_counter()-clock
        diagnostic = not passed and (config["diagnostic_fea_always"] or config["diagnostic_fea"] and validation["violations"] == ["features"] and record["mass_screen"]["passed"])
        if not (passed or diagnostic) or config["geometry_only"]:
            return
        if not diagnostic:
            return self.fea(directory, record, mesh, progress)
        record["fea_diagnostic_only"] = True
        try:
            self.fea(directory, record, mesh, progress)
        except Exception as error:
            record["diagnostics"].append("diagnostic FEA "+type(error).__name__+": "+str(error))
        if record.get("fea", {}).get("status") != "ok":
            record["diagnostics"].append("diagnostic_fea_failed")
        record.update(status="geometry_invalid", failure_stage="validation")

    def fea(self, directory, record, mesh, progress):
        timings = record["timings_s"]
        if self.baseline is None:
            self.step(record, "baseline_fea", progress)
            self.baseline = baseline_fea(self.output/"baseline", self.baseline_solid, self.domain, self.settings, evaluate)
        self.step(record, "candidate_fea", progress)
        clock = perf_counter()
        result = evaluate(mesh, self.domain["material"], self.domain["point_masses"], self.domain["comparison_load_cases"],
                          {**self.settings, **{key: self.config[key] for key in MESH_KEYS}, "work_dir": str(directory/"fea")})
        timings["fea"] = perf_counter()-clock
        timings["fea_mesh"] = sum(attempt.get("runtime_s", 0.0) for attempt in result.get("mesh", {}).get("attempts", []))
        timings["fea_solve"] = timings["fea"]-timings["fea_mesh"]
        _save(directory/"raw_fea_result.json", result)
        record["fea"] = result
        if result["status"] != "ok":
            record.update(status="failed", failure_stage="fea")
            record["diagnostics"].extend(result.get("diagnostics", []))
            return
        _verify_cases(result, self.domain["comparison_load_cases"])
        record["comparison"] = compare_to_baseline(result, self.baseline, self.constraints)
        record["status"] = "accepted" if record["comparison"]["passed"] else "mechanically_rejected"
        if not record["comparison"]["passed"]:
            record["failure_stage"] = "comparison"

    def candidate(self, index, threshold, extension):
        directory = self.output/"candidates"/f"c{index:02d}"
        directory.mkdir(parents=True)
        started, state = perf_counter(), {"mesh": None}
        record = {"id": directory.name, "parameters": {"threshold": threshold, "extension": extension, **{name: self.config[key] for key, name in PARAMETERS.items()}},
                  "status": "building_field", "stage": "building_field", "success": False, "geometry_valid": False, "failure_stage": None, "diagnostics": [], "timings_s": {}}
        try:
            self.build(directory, record, self.progress(record, directory, state, started))
        except Exception as error:
            record["status"] = error.status if isinstance(error, ImplicitError) else "failed"
            record["failure_stage"] = error.report.get("failure_stage", "field") if isinstance(error, ImplicitError) else record["failure_stage"] or record["stage"]
            record["diagnostics"].append(type(error).__name__+": "+str(error))
            if isinstance(error, ImplicitError):
                record["geometry"] = error.report
            mesh = state["mesh"] if getattr(error, "mesh", None) is None else error.mesh
            if mesh is not None:
                (directory/"failure").mkdir(exist_ok=True)
                mesh.export(directory/"failure"/(record["failure_stage"]+".ply"))
                record["failure_evidence"] = {"mesh": "failure/"+record["failure_stage"]+".ply", "faces": len(mesh.faces)}
        record.update(success=record["status"] == "accepted", runtime_s=perf_counter()-started, process_peak_rss_mb=peak_rss_mb())
        record["artifacts"] = artifacts(directory)
        _save(directory/"record.json", record)
        self.records.append(record)
        log_run(self.config["run_log"], candidate_entry("implicit", self.output, record, self.manifest.get("source_manifest_sha256")))
        self.manifest["candidates"].append({"id": record["id"], "directory": directory.relative_to(self.output).as_posix(), "record_sha256": _file_digest(directory/"record.json"),
                                           "threshold": threshold, "extension": extension, "status": record["status"], "success": record["success"], "runtime_s": record["runtime_s"]})
        print(json.dumps({"candidate": record["id"], "threshold": threshold, "extension": extension, "status": record["status"], "failure_stage": record["failure_stage"], "runtime_s": round(record["runtime_s"], 1)}), flush=True)

    def summary(self):
        count, runtimes = len(self.records), [record["runtime_s"] for record in self.records]
        return {"candidate_count": count, "success_rate": sum(record["success"] for record in self.records)/count if count else None,
                "geometry_success_rate": sum(record["geometry_valid"] for record in self.records)/count if count else None,
                "runtime_per_candidate_s": {"median": statistics.median(runtimes), "max": max(runtimes)} if runtimes else None}

    def run(self):
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError("A new empty output directory is required; old evidence is never overwritten")
        self.output.mkdir(parents=True, exist_ok=True)
        _save(self.output/"manifest.json", self.manifest)
        self.status()
        try:
            self.prepare()
            for index, (threshold, extension) in enumerate(self.candidates):
                if perf_counter()-self.started >= self.config["study_timeout_s"]:
                    raise TimeoutError("Study runtime budget exceeded before the next candidate")
                self.manifest["stage"] = f"candidate c{index:02d}"
                self.status()
                self.candidate(index, threshold, extension)
                self.manifest.update(acceptance(self.records, complete=False), **self.summary())
                _save(self.output/"manifest.json", self.manifest)
            self.manifest.update(acceptance(self.records, complete=True), status="complete", stage="finished")
        except Exception as error:
            if self.manifest["stage"] == "preparing":
                log_run(self.config["run_log"], {"run_dir": str(self.output), "kind": "implicit", "candidate": None, "parameters": None, "source_sha256": self.manifest.get("source_manifest_sha256"),
                                                 "success": False, "status": "failed", "failure_stage": "prepare", "runtime_s": perf_counter()-self.started, "timings_s": None, "error": type(error).__name__+": "+str(error)})
            self.manifest.update(status="failed", stage="failed", overall_acceptance=False, selected_id=None, error=type(error).__name__+": "+str(error))
        self.manifest.update(**self.summary(), runtime_s=perf_counter()-self.started, artifacts=artifacts(self.output))
        _save(self.output/"manifest.json", self.manifest)
        self.status(**{key: self.manifest[key] for key in ("accepted_count", "selected_id", "overall_acceptance", "success_rate", "runtime_s")})
        return self.manifest

def run_main(overrides):
    config = configure(RUN_CONFIG, RUN_KINDS, overrides, ("source", "output"))
    if not config["thresholds"] or not config["extensions"] or config["study_timeout_s"] <= 0:
        raise ValueError("At least one threshold and extension and a positive study budget are required")
    return 0 if ImplicitStudy(config).run()["status"] == "complete" else 1

def render_main(overrides):
    config = configure(RENDER_CONFIG, RENDER_KINDS, overrides, ("geometry", "output", "inputs"))
    import trimesh
    path, domain = config["geometry"], _read(config["inputs"])["domain"]
    if path.suffix.lower() in (".step", ".stp"):
        mesh = reference_geometry(path, domain, config)[0]
    else:
        mesh = trimesh.load_mesh(path, process=True)
    record = {"geometry": str(path.resolve()), "sha256": _file_digest(path), "inputs_sha256": _file_digest(config["inputs"]), "renderer_sha256": _file_digest(__file__),
              "watertight": bool(mesh.is_watertight), "body_count": int(mesh.body_count), "volume_mm3": float(mesh.volume),
              "surface_metrics": surface_metrics(mesh, domain), "curvature": ball_curvature(mesh, config["curvature_radius_mm"], config["curvature_samples"])}
    footer = f"{path.name} | SHA256 {record['sha256'][:16]} | {len(mesh.faces)} Dreiecke"
    record["renders"] = render_views(mesh, domain, config["output"], config["label"] or "DIAGNOSE - Darstellung ohne Abnahmeaussage", footer, config["render_faces"], config["section_heights_mm"])
    _save(config["output"]/"render_manifest.json", record)
    return 0

def summarize(entries):
    runs = {}
    for entry in entries:
        runs.setdefault((entry["kind"], entry["run_dir"]), []).append(entry)
    rows = []
    for (kind, run_dir), items in runs.items():
        runtimes = [item["runtime_s"] for item in items if item.get("runtime_s") is not None]
        rows.append({"kind": kind, "run_dir": run_dir, "candidates": len(items), "accepted": sum(bool(item["success"]) for item in items), "success_rate": sum(bool(item["success"]) for item in items)/len(items),
                     "statuses": {status: sum(item["status"] == status for item in items) for status in sorted({item["status"] for item in items})},
                     "runtime_per_candidate_s": {"median": statistics.median(runtimes), "max": max(runtimes)} if runtimes else None, "git_commits": sorted({item.get("git_commit") or "" for item in items})})
    return rows

def summarize_main(overrides):
    config = configure(SUMMARIZE_CONFIG, SUMMARIZE_KINDS, overrides, ("output",))
    path = run_log_path(config["run_log"])
    entries = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if config["runs"] is not None:
        entries = [entry for entry in entries if entry["run_dir"] in config["runs"] or Path(entry["run_dir"]).name in config["runs"]]
    _save(config["output"], {"run_log": str(path), "run_log_sha256": _file_digest(path), "runs": summarize(entries)})
    return 0

def main(argv=None):
    return command_line({"run": run_main, "render": render_main, "summarize": summarize_main}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
