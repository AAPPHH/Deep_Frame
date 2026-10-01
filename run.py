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

COMMANDS = {"frame": frame, "optimization": optimization, "smoke": smoke, "topology": topology}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in COMMANDS:
        raise SystemExit("usage: run.py {" + ",".join(COMMANDS) + "}")
    command = argv[0]
    result = COMMANDS[command]()
    if command == "optimization":
        print(json.dumps({"status": result["status"], "trial_counts": result["trial_counts"], "has_improved_design": result["has_improved_design"], "artifacts": result["artifacts"]}))
    if command == "topology":
        print(json.dumps({"status": result["status"], "selected_id": result["selected_id"], "pareto_ids": result["pareto_ids"], "manifest": result["run_dir"] + "/manifest.json"}))

if __name__ == "__main__":
    main()
