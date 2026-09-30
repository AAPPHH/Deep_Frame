import json
from pathlib import Path

from build123d import export_step

from deep_frame.geometry import build_geometry, reference_parameters, validate_geometry
from deep_frame.model import export_body
from deep_frame.viewer import show_assembly


def main():
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


if __name__ == "__main__":
    main()
