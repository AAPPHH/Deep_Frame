import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "C:/clones/Deep_Frame-cov")
from tools.formulation_study import configure, export_body
out = Path("C:/clones/Deep_Frame-cov/exports/cov/symmetry/raw_fixed")
out.mkdir(parents=True, exist_ok=True)
cfg = configure(json.loads(Path("C:/clones/Deep_Frame-cov/exports/cov/symmetry/cov3_run.json").read_text()))
physical = np.load("C:/clones/Deep_Frame-cov/exports/runs/simp_mma_cov3_opt/simp_mma_cov3/density_half.npz")["density"]
body = export_body(cfg, physical, out)
(out / "body.json").write_text(json.dumps(body, indent=1, default=float))
print(json.dumps({k: body[k] for k in ("mass_g", "mesh_volume_mm3", "bodies", "watertight")}))
