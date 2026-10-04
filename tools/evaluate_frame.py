import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_frame.config import command_line, configure
from deep_frame.frame_evaluation import HEADER, LEGEND, append_datasheet, frame_spec, geometry, limits_markdown, mechanics, sigma, sigma_limits, slice_frame, summary, walls

PARTS = {"geometry": (geometry, "cpu"), "walls": (walls, "wall_check"), "fea": (mechanics, "fea_modal"), "sigma": (sigma, "fea_modal"), "slicer": (slice_frame, "cpu")}

GAP = Path("C:/clones/Deep_Frame-gap/exports")
gap = lambda *names: str(next((path for path in (GAP / folder / name / "interface_stiffness.json" for folder in ("gap", "interface_stiffness") for name in names) if path.exists()), GAP / "gap" / names[0] / "interface_stiffness.json"))
LIMITS = {"references": {"manafly3": gap("manafly3"), "aether4": gap("aether4")}, "candidates": {"simp_mma_raw_1": gap("simp_mma_raw", "simp_mma_raw_1"), "simp_mma_recon_1": gap("simp_mma_recon", "simp_mma_recon_1")},
          "evaluations": {name: str(ROOT / "exports" / "cov" / "eval" / name / "sigma.json") for name in ("manafly3", "aether4", "simp_mma_raw_1", "simp_mma_recon_1")}
                          | {name: str(ROOT / "exports" / "runs" / name / "evaluation" / "sigma.json") for name in ("simp_mma_cov_raw_1", "simp_mma_cov_recon_1", "simp_mma_cov2_raw_1")}, "output": str(ROOT / "exports" / "cov")}
LIMITS_KINDS = {"references": "object", "candidates": "object", "evaluations": "object", "output": "path"}

def spec_of(overrides):
    spec = frame_spec(overrides)
    Path(spec["output"]).mkdir(parents=True, exist_ok=True)
    return spec

def part(name):
    def run(overrides):
        spec = spec_of(overrides)
        (Path(spec["output"]) / f"{name}.json").write_text(json.dumps(PARTS[name][0](spec), indent=1, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)), encoding="utf-8")
        return 0
    return run

def report(overrides):
    spec = spec_of(overrides)
    output = Path(spec["output"])
    parts = {name: json.loads((output / f"{name}.json").read_text(encoding="utf-8")) if (output / f"{name}.json").exists() else None for name in PARTS}
    result = summary(spec, parts)
    (output / "evaluation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (output / "bewertung.md").write_text(f"{HEADER}\n{result['line']}\n\n{LEGEND}\n", encoding="utf-8")
    if spec["datasheet"]:
        append_datasheet(spec["datasheet"], result, (output / "evaluation.json").resolve().as_posix())
    print(result["line"])
    return 0

def run(overrides):
    spec = spec_of(overrides)
    output = Path(spec["output"]).resolve()
    request = output / "frame.json"
    request.write_text(json.dumps(overrides, indent=1, default=str), encoding="utf-8")
    jobs = {name: subprocess.Popen([spec["python"], spec["compute"], PARTS[name][1], "--cwd", str(ROOT), "--", spec["python"], str(ROOT / "tools" / "evaluate_frame.py"), name, str(request)],
                                   stdout=(output / f"{name}.log").open("w", encoding="utf-8"), stderr=subprocess.STDOUT) for name in spec["parts"]}
    failed = [name for name, job in jobs.items() if job.wait()]
    report(overrides)
    return 1 if failed else 0

def limits(overrides):
    settings = configure(LIMITS, LIMITS_KINDS, overrides)
    result = sigma_limits(settings)
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    markdown = limits_markdown(result)
    (output / "limits.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    (output / "limits.md").write_text(markdown, encoding="utf-8")
    return 0

def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    return command_line({"run": run, "report": report, "limits": limits, **{name: part(name) for name in PARTS}}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
