from pathlib import Path


CONFIG = {
    "length_mm": 30.0,
    "width_mm": 20.0,
    "thickness_mm": 3.0,
    "stl_path": Path(__file__).resolve().parent.parent / "exports" / "smoke.stl",
    "viewer_port": 3939,
}
