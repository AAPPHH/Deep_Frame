from pathlib import Path

from deep_frame.component_defaults import COMPONENT_DEFAULTS
from deep_frame.frame_defaults import FRAME_DEFAULTS, FRAME_DEFAULT_SOURCES


CONFIG = {
    "length_mm": 30.0,
    "width_mm": 20.0,
    "thickness_mm": 3.0,
    "stl_path": Path(__file__).resolve().parent.parent / "exports" / "smoke.stl",
    "viewer_port": 3939,
    "frame": FRAME_DEFAULTS,
    "components": COMPONENT_DEFAULTS,
    "default_sources": FRAME_DEFAULT_SOURCES,
    "material": {
        "name": "Generic dry PA6-CF, provisional Bambu reference",
        "density_g_cm3": 1.09,
        "source": "https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf",
        "assumptions": "Homogeneous full-density envelope; filament, infill and anisotropy remain unselected",
    },
    "checks": {
        "minimum_clearance_mm": 0.5,
        "prop_clearance_mm": 2.0,
        "intersection_tolerance_mm3": 1e-6,
        "distance_tolerance_mm": 1e-6,
        "maximum_battery_prop_overlap_percent": 0.0,
    },
    "frame_export_stem": "exports/frame_v0",
}
