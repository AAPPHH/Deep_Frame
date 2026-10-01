from math import isfinite
from pathlib import Path

from build123d import Align, Box, Part, export_stl


def build_smoke_body(config: dict) -> Part:
    dimensions = tuple(config[key] for key in ("length_mm", "width_mm", "thickness_mm"))
    if not all(isfinite(value) and value > 0 for value in dimensions):
        raise ValueError("Dimensions must be finite and greater than zero.")
    return Box(*dimensions, align=(Align.CENTER, Align.CENTER, Align.MIN))


def export_body(body: Part, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(body, path):
        raise RuntimeError(f"STL export failed: {path}")
    return path
