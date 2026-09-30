from copy import deepcopy

from build123d import Solid

from deep_frame.config import CONFIG
from deep_frame.frame import build_frame


def reference_parameters() -> dict:
    return deepcopy({key: value for key, value in CONFIG.items() if key != "stl_path"})


def build_geometry(parameters: dict) -> Solid:
    return build_frame(parameters)


def validate_geometry(parameters: dict) -> dict:
    from deep_frame.checks import check_assembly

    return check_assembly(parameters)
