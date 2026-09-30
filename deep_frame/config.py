from pathlib import Path

from deep_frame.component_defaults import COMPONENT_DEFAULTS
from deep_frame.fea_config import FEA_CONFIG
from deep_frame.frame_defaults import FRAME_DEFAULTS, FRAME_DEFAULT_SOURCES
from deep_frame.integration_config import INTEGRATION_CONFIG
from deep_frame.optimization_config import OPTIMIZATION_CONFIG, SEARCH_SPACE


CONFIG = {
    "length_mm": 30.0,
    "width_mm": 20.0,
    "thickness_mm": 3.0,
    "stl_path": Path(__file__).resolve().parent.parent / "exports" / "smoke.stl",
    "viewer_port": 3939,
    "frame": FRAME_DEFAULTS,
    "components": COMPONENT_DEFAULTS,
    "default_sources": FRAME_DEFAULT_SOURCES,
    "material": FEA_CONFIG["material"],
    "fea": FEA_CONFIG,
    "optimization": OPTIMIZATION_CONFIG,
    "optimization_search_space": SEARCH_SPACE,
    "integration": INTEGRATION_CONFIG,
    "checks": {
        "minimum_clearance_mm": 0.5,
        "prop_clearance_mm": 2.0,
        "intersection_tolerance_mm3": 1e-6,
        "distance_tolerance_mm": 1e-6,
        "maximum_battery_prop_overlap_percent": 0.0,
    },
    "frame_export_stem": "exports/frame_v0",
}
