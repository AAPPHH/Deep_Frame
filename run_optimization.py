import json
from copy import deepcopy

from deep_frame.geometry import build_geometry, reference_parameters, validate_geometry
from deep_frame.integration import FrameEvaluator
from deep_frame.optimization import optimize
from deep_frame.optimization_viewer import show_candidates


RUN_CONFIG = {
    "n_trials": 4,
    "study_name": "frame-v0-probe-v1",
    "storage": "sqlite:///exports/optimization/probe/study.sqlite3",
    "output_dir": "exports/optimization/probe",
    "fea_work_dir": "exports/fea/optimization_probe",
    "show_viewer": True,
}


def run(config=RUN_CONFIG):
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


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({"status": outcome["status"], "trial_counts": outcome["trial_counts"], "has_improved_design": outcome["has_improved_design"], "artifacts": outcome["artifacts"]}))
