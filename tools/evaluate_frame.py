import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_frame.config import command_line
from deep_frame.frame_evaluation import HEADER, LEGEND, frame_spec, geometry, mechanics, slice_frame, summary, walls

PARTS = {"geometry": (geometry, "cpu"), "walls": (walls, "wall_check"), "fea": (mechanics, "fea_modal"), "slicer": (slice_frame, "cpu")}

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

def main(argv=None):
    return command_line({"run": run, "report": report, **{name: part(name) for name in PARTS}}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
