import json
import sys
from copy import deepcopy
from pathlib import Path

RUN_CONFIG = {
    "n_trials": 4,
    "study_name": "frame-v0-probe-v1",
    "storage": "sqlite:///exports/optimization/probe/study.sqlite3",
    "output_dir": "exports/optimization/probe",
    "fea_work_dir": "exports/fea/optimization_probe",
    "show_viewer": True,
}

FRAME = {"style": "freestyle", "durability": "crash_resistant", "prop_size_in": 2.5, "layout": {"x_type": "compressed_x", "battery_mount": "top"},
         "components": {"motor": "GTS V3 1203", "aio": "HDZero AIO15", "camera": "HDZero Lux", "battery": "GNB5502S120A", "antennas": "HDZero VTX + ELRS"},
         "material": "PA6-CF", "print": {"nozzle_mm": 0.4, "layer_mm": 0.2}}

def frame():
    from build123d import export_step
    from deep_frame.frame import build_geometry, export_body, reference_parameters, show_assembly, validate_geometry
    parameters = reference_parameters()
    result = validate_geometry(parameters)
    stem = Path(parameters["frame_export_stem"])
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not result["passed"]:
        show_assembly(parameters)
        raise ValueError(result["violations"])
    frame = build_geometry(parameters)
    export_body(frame, stem.with_suffix(".stl"))
    export_step(frame, stem.with_suffix(".step"))
    show_assembly(parameters)

def optimization(config=RUN_CONFIG):
    from deep_frame.fea import FrameEvaluator
    from deep_frame.frame import build_geometry, reference_parameters, validate_geometry
    from deep_frame.optimization import optimize, show_candidates
    parameters = reference_parameters()
    fea_config = deepcopy(parameters["fea"])
    fea_config["settings"]["work_dir"] = config["fea_work_dir"]
    evaluator = FrameEvaluator(parameters, fea_config)
    settings = deepcopy(parameters["optimization"])
    settings.update({key: config[key] for key in ("n_trials", "study_name", "storage", "output_dir")})
    settings["evaluation_id"] = evaluator.evaluation_id
    result = optimize(parameters, parameters["optimization_search_space"], evaluator, validate_geometry, settings)
    if config["show_viewer"] and result["pareto_front"]:
        candidates = sorted(result["pareto_front"], key=lambda trial: (-trial["metrics"]["stiffness_n_per_mm"], trial["number"]))
        show_candidates(candidates[:1], build_geometry, viewer_settings={"port": parameters["viewer_port"], "progress": "", "timeit": False})
    return result

def smoke():
    from ocp_vscode import port_check, show
    from deep_frame.config import CONFIG
    from deep_frame.frame import build_smoke_body, export_body
    body = build_smoke_body(CONFIG)
    export_body(body, CONFIG["stl_path"])
    if not port_check(CONFIG["viewer_port"]):
        raise ConnectionError(f"Start OCP CAD Viewer in VS Code on port {CONFIG['viewer_port']}.")
    show(body, names=["Smoke"], port=CONFIG["viewer_port"], progress="", timeit=False)

def topology(settings=None):
    from deep_frame.frame import reference_parameters
    from deep_frame.topology_pipeline import PIPELINE_CONFIG, run_topology
    return run_topology(reference_parameters(), deepcopy(PIPELINE_CONFIG) if settings is None else settings)

def build(path=None, frame=FRAME):
    from deep_frame.frame_run import FrameRun
    request = {**deepcopy(frame), **(json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {})}
    manifest = FrameRun(request).run()
    print(json.dumps({"run": manifest["name"], "status": manifest["status"], "stages": {name: entry["status"] for name, entry in manifest["stages"].items()}}))
    return manifest

def datasheet(path):
    from deep_frame.frame_run import datasheet as write
    return write(path)

def layout(path=None):
    from deep_frame.frame_run import layout_study
    layout_study(path)
    return 0

def _update(target, values):
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _update(target[key], value)
        else:
            target[key] = deepcopy(value)

def _json(value):
    return value.tolist() if hasattr(value, "tolist") else str(value)

def stage(path):
    import importlib
    import importlib.util
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    root, here = Path(request["root"]).resolve(), Path(__file__).resolve().parent
    sys.path[:] = [str(root)] + [entry for entry in sys.path if Path(entry or ".").resolve() != here]
    config = importlib.import_module("deep_frame.config")
    for name, values in request["patch"].items():
        _update(getattr(config, name), values)
    tool = root / request["tool"]
    if request["action"] == "cli":
        import runpy
        overrides = Path(path).with_suffix(".overrides.json")
        overrides.write_text(json.dumps(request["overrides"], indent=1), encoding="utf-8")
        sys.argv = [str(tool), *request["argv"], str(overrides)]
        runpy.run_path(str(tool), run_name="__main__")
        return 0
    if request["action"] == "domain":
        from deep_frame.topology_geometry import build_design_domain
        from tools.topology_study import study_parameters
        domain = build_design_domain(study_parameters(request["shape"]))
        result = {key: domain[key] for key in ("grid", "regions", "load_cases", "comparison_load_cases", "point_masses", "material")}
        result["components"] = domain["metadata"]["components"]
    else:
        import trimesh
        spec = importlib.util.spec_from_file_location(tool.stem, tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kwargs = dict(request["kwargs"])
        kwargs["mesh"] = trimesh.load_mesh(kwargs["mesh"], process=True)
        if "out" in kwargs:
            kwargs["out"] = Path(kwargs["out"])
            kwargs["out"].mkdir(parents=True, exist_ok=True)
        if "views" in kwargs:
            kwargs["views"] = {name: (tuple(direction), tuple(up)) for name, (direction, up) in kwargs["views"].items()}
        result = getattr(module, request["function"])(**kwargs)
    if request.get("output") or request.get("result"):
        target = Path(request.get("output") or request["result"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=1, default=_json), encoding="utf-8")
    return 0

COMMANDS = {"frame": frame, "optimization": optimization, "smoke": smoke, "topology": topology, "build": build, "datasheet": datasheet, "stage": stage, "layout": layout}
ARGUMENTS = {"layout": (0, 1), "build": (0, 1), "datasheet": (1, 1), "stage": (1, 1)}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    low, high = ARGUMENTS.get(argv[0] if argv else None, (0, 0))
    if not argv or argv[0] not in COMMANDS or not low <= len(argv) - 1 <= high:
        raise SystemExit("usage: run.py {" + ",".join(COMMANDS) + "} [file.json]")
    command = argv[0]
    result = COMMANDS[command](*argv[1:])
    if command == "optimization":
        print(json.dumps({"status": result["status"], "trial_counts": result["trial_counts"], "has_improved_design": result["has_improved_design"], "artifacts": result["artifacts"]}))
    if command == "topology":
        print(json.dumps({"status": result["status"], "selected_id": result["selected_id"], "pareto_ids": result["pareto_ids"], "manifest": result["run_dir"] + "/manifest.json"}))
    if command in ("datasheet", "stage"):
        return result

if __name__ == "__main__":
    raise SystemExit(main())
