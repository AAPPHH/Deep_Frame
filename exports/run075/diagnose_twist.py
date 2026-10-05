import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.formulation_study import configure, frame_setup, MemoryProbe
from deep_frame.topology_problem import TopologyProblem, prolongate

ROOT = Path(__file__).resolve().parent
SOURCE = Path("C:/clones/Deep_Frame-layout/exports/runs/free_layout4_opt/fine2")
VARIANTS = {"float64": {"precision": "float64"}, "pinned": {"projection": False}}
cfg = configure(json.loads((ROOT / "probe_restart2.json").read_text(encoding="utf-8-sig")))
out = ROOT / "twist_diagnostic"
out.mkdir(exist_ok=True)
results = {}
for name, settings in VARIANTS.items():
    started = perf_counter()
    probe = MemoryProbe(1.0, out / f"{name}_memory.jsonl")
    with probe.phase(name):
        half, problem = frame_setup(cfg)
        problem["multigrid"] = {**settings, "max_iterations": 400, "retries": 0, "trace": True}
        tp = TopologyProblem(half, problem, linear_solver="multigrid")
        while tp.advance():
            pass
        design = prolongate(np.load(SOURCE / "design.npz")["design"], json.loads((SOURCE / "result.json").read_text())["grid"], half)
        physical = tp.map.fields(design)[0]["eroded"][0]
        tp.body_update(physical)
        system = tp.system
        system.groups = {key: [(case, part) for case, part in members if case["name"] == "twist"] for key, members in system.groups.items() if any(case["name"] == "twist" for case, _ in members)}
        system.shared = {}
        norms = [{"sign": part["sign"], "norm": float(np.linalg.norm(part["force"])), "maximum": float(np.max(np.abs(part["force"]))) } for members in system.groups.values() for _, part in members]
        print(json.dumps({"variant": name, "forces": norms}), flush=True)
        result = {"settings": problem["multigrid"], "forces": norms}
        try:
            values = system.solve(physical, tp.penalization, tp.min_stiffness_ratio)
            result["values"] = {case: {key: value for key, value in row.items() if key in ("compliance_n_mm", "relative_residual")} for case, row in values.items()}
        except Exception as error:
            result["error"] = str(error)
        result["solves"] = system.multigrid.diagnostics()
        tp.close()
        system.multigrid.torch.cuda.empty_cache()
    probe.close()
    result.update(seconds=perf_counter() - started, memory=probe.summary())
    results[name] = result
    (out / "result.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({"variant": name, "seconds": result["seconds"], "values": result.get("values"), "error": result.get("error")}), flush=True)
    del tp, system, physical, design
