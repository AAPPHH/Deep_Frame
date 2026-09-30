from copy import deepcopy


def show_candidates(candidates, build_geometry, show=None, viewer_settings=None):
    if show is None:
        from ocp_vscode import show
    if not candidates:
        raise ValueError("At least one Pareto candidate is required")
    shapes = []
    names = []
    for candidate in candidates:
        if candidate.get("outcome") != "valid":
            raise ValueError("Only valid candidates can be displayed")
        shapes.append(build_geometry(deepcopy(candidate["parameters"])))
        names.append(f"Trial {candidate['number']}")
    show(*shapes, names=names, **(viewer_settings or {}))
    return {"displayed_trials": [candidate["number"] for candidate in candidates]}
