OPTIMIZATION_CONFIG = {
    "n_trials": 6,
    "seed": 42,
    "population_size": 8,
    "storage": "sqlite:///exports/optimization/study.sqlite3",
    "study_name": "frame-v0",
    "evaluation_id": "frame-v0-fea-v1",
    "output_dir": "exports/optimization",
    "relative_constraints": {
        "mass_ratio_max": 1.05,
        "stiffness_ratio_min": 0.95,
        "frequency_ratio_min": 0.95,
        "displacement_ratio_max": 1.05,
        "stress_ratio_max": 1.10,
    },
    "printability": {
        "nozzle_width_mm": 0.4,
        "minimum_wall_nozzles": 4,
        "wall_thickness_paths": [
            "frame.minimum_wall_mm",
            "frame.base_thickness_mm",
            "frame.deck_thickness_mm",
            "frame.arm_height_mm",
        ],
        "clamp_sections": [
            {
                "name": "arm_root",
                "width_path": "frame.arm_width_mm",
                "height_path": "frame.arm_height_mm",
                "minimum_area_mm2": 20.0,
            }
        ],
    },
    "initial_candidates": [
        {"frame.arm_height_mm": 4.2, "frame.arm_width_mm": 6.5},
        {"frame.arm_height_mm": 4.0, "frame.arm_width_mm": 6.7},
    ],
}


SEARCH_SPACE = {
    "frame.arm_height_mm": {"low": 3.8, "high": 4.4, "step": 0.2},
    "frame.arm_width_mm": {"low": 6.3, "high": 6.9, "step": 0.2},
}
