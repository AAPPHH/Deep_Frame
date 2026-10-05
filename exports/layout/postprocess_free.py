import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")

def export(cfg):
    from tools.formulation_study import configure, export_body
    out = ROOT / cfg["raw_directory"]
    out.mkdir(parents=True, exist_ok=True)
    result = export_body(configure(cfg["formulation"]), np.load(ROOT / cfg["field"])["density"], out)
    save(out / "export.json", result)

def stage(cfg):
    source, target = ROOT / cfg["reconstructed_stl"], ROOT / cfg["frame_stl"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def layout(cfg):
    import trimesh
    from deep_frame.config import LAYOUT_OPTIMIZATION, LAYOUT_DEFAULT, PRINT_MATERIAL
    from deep_frame.frame_run import layout_study
    from tools.formulation_study import digest
    stl = ROOT / cfg["frame_stl"]
    mesh = trimesh.load_mesh(stl, process=True)
    rho = PRINT_MATERIAL["density_g_cm3"] / 1000
    share = {"stl": cfg["frame_stl"], "sha256": digest(stl), "mass_g": float(mesh.volume*rho), "center_mm": np.asarray(mesh.center_mass).tolist(), "inertia_g_mm2": (np.asarray(mesh.moment_inertia)*rho).tolist(), "source": "v3c with preserved camera contact; PA6-CF 1.09 g/cm3", "layout": dict(LAYOUT_DEFAULT["layout"])}
    layout_study(str(ROOT / cfg["layout_output"]), {**deepcopy(LAYOUT_OPTIMIZATION), "cg_horizontal_mm": 1.0, "frame_share": share})

def contacts(cfg):
    import trimesh
    from matplotlib.path import Path as Polygon
    spec, out = cfg["camera"], {}
    center, radius, half, step = np.asarray(spec["center_mm"]), spec["radius_mm"], spec["half_width_mm"], spec["sample_mm"]
    yz = np.stack(np.meshgrid(np.arange(-radius+step/2, radius, step), np.arange(-radius+step/2, radius, step)), axis=-1).reshape(-1, 2)
    yz = yz[np.linalg.norm(yz, axis=1) < radius] + center[1:]
    for name, relative in cfg["bodies"].items():
        mesh, rows = trimesh.load_mesh(ROOT / relative), {}
        for sign in (-1, 1):
            mask = (np.linalg.norm(mesh.vertices[:, 1:]-center[1:], axis=1) <= radius) & (sign*(mesh.vertices[:, 0]-center[0]) > 0)
            sections = {}
            for distance in spec["distances_mm"]:
                section = mesh.section(plane_origin=[center[0]+sign*(half+distance), 0, 0], plane_normal=[1, 0, 0])
                inside = np.zeros(len(yz), dtype=bool)
                for loop in section.discrete if section is not None else []:
                    if len(loop) > 3:
                        inside ^= Polygon(loop[:, 1:]).contains_points(yz)
                sections[str(distance)] = float(inside.sum()*step**2)
            rows[str(sign)] = {"nearest_gap_mm": float(np.min(sign*(mesh.vertices[mask, 0]-center[0]))-half) if mask.any() else None, "patch_sections_mm2_by_distance_mm": sections}
        out[name] = {"stl": relative, "contacts": rows}
        print(name, rows, flush=True)
    save(cfg["contact_output"], out)

if __name__ == "__main__":
    os.chdir(ROOT)
    cfg = json.loads((ROOT / (sys.argv[2] if len(sys.argv) > 2 else "exports/layout/free_postprocess_dgx.json")).read_text(encoding="utf-8"))
    {"export": export, "stage": stage, "contacts": contacts, "layout": layout}[sys.argv[1]](cfg)
