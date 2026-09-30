import json
from copy import deepcopy

from deep_frame.geometry import reference_parameters
from deep_frame.topology_pipeline import PIPELINE_CONFIG, run_topology


SETTINGS = deepcopy(PIPELINE_CONFIG)


if __name__ == "__main__":
    result = run_topology(reference_parameters(), SETTINGS)
    print(json.dumps({"status": result["status"], "selected_id": result["selected_id"], "pareto_ids": result["pareto_ids"], "manifest": result["run_dir"] + "/manifest.json"}))
