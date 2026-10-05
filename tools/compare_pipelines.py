import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compute import request, JOB_TYPES
from ray.job_submission import JobStatus, JobSubmissionClient

class Comparison:
    def __init__(self, path):
        self.config = json.loads(Path(path).read_text())
        self.root = ROOT / self.config["state_directory"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.python = os.environ.get("DEEP_FRAME_PYTHON", sys.executable)
        self.client = JobSubmissionClient(os.environ["RAY_ADDRESS"])
        self.lock = threading.RLock()
        self.problem_ready = threading.Barrier(len(self.config["pipelines"]))
        self.data = json.loads((self.root/"status.json").read_text()) if self.config.get("resume") and (self.root/"status.json").exists() else {"created": datetime.now(timezone.utc).isoformat(), "config": self.config, "stages": {}, "branches": {}}
        self.data["config"]=self.config
        self.configs = {name: json.loads((ROOT / path).read_text()) for name, path in self.config["pipelines"].items()}
    def save(self):
        with self.lock:
            target = self.root / "status.tmp"
            target.write_text(json.dumps(self.data, indent=2))
            target.replace(self.root / "status.json")
    def update(self, stage, **fields):
        with self.lock:
            self.data["stages"].setdefault(stage, {}).update(fields)
            self.save()
        print(stage, fields, flush=True)
    def wait(self, job):
        while not self.client.get_job_status(job).is_terminal():
            time.sleep(5)
        info = self.client.get_job_info(job)
        if info.status != JobStatus.SUCCEEDED:
            raise RuntimeError(f"{job}: {info.status}: {info.message}")
    def ray(self, stage, action, path,kind="gpu_compare",tool="tools/formulation_study.py"):
        previous=self.data["stages"].get(stage,{})
        inputs={"action":action,"path":str(path),"tool":tool,"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        if previous.get("inputs") and previous["inputs"]!=inputs:
            raise RuntimeError(f"{stage}: recorded job inputs changed; use a new stage")
        if previous.get("job"):
            job=previous["job"]
            self.update(stage,resumed=True,inputs=inputs)
        else:
            job = self.client.submit_job(**request(kind, [self.python, tool, action, path], ROOT, python=self.python))
            self.update(stage, status="RUNNING", job=job,kind=kind,gpu_limit_gb=JOB_TYPES[kind]["gpu_gb"],host_limit_gb=JOB_TYPES[kind]["memory_gb"],started=datetime.now(timezone.utc).isoformat(),inputs=inputs)
        try:
            self.wait(job)
            self.update(stage, status="SUCCEEDED")
        except Exception as error:
            self.update(stage, status="FAILED", error=str(error))
            raise
        finally:
            (self.root / (stage + ".log")).write_text(self.client.get_job_logs(job))
    def gate(self, name):
        root = ROOT / (self.configs[name]["mma"]["root"] + "_probe")
        record = json.loads((root / "fine/result.json").read_text())
        memory = json.loads((root / "memory.json").read_text())["peaks"]["probe"]
        rows = (root / "fine/iterations.jsonl").read_text().splitlines()
        gpu = memory.get("process_gpu_gb")
        if record["iterations"] != 3 or len(rows) != 3 or gpu is None or not 0 < gpu <= 32 or not math.isfinite(gpu) or not 0 < memory["rss_gb"] <= 46 or not all(math.isfinite(record[key]) for key in ("mass_g", "max_violation")):
            raise RuntimeError(f"{name}: three-iteration solver/memory gate failed")
        self.update(name + "_probe", gate="PASSED", gpu_peak_gb=gpu, host_peak_gb=memory["rss_gb"], design_status=record["status"], note="Numerical/resource probe; no design acceptance")
    def shared(self):
        records={name:json.loads((ROOT/(cfg["mma"]["root"]+"_probe")/"fine/result.json").read_text()) for name,cfg in self.configs.items()}
        fingerprints={row["problem_definition_sha256"] for row in records.values()}
        if len(fingerprints)!=1:
            raise RuntimeError("Optimization routes use different problem definitions")
        for row in records.values():
            functions=row["functional_requirements"]
            if row["battery_retention"]!="ideal_press" or functions["low_flight"]["pitch_deg"]!=15 or functions["battery_guide"]["area_measure"]!="nearest_grid_layer_to_battery":
                raise RuntimeError("Required flight attitude, ideal retention or integrated guide formulation missing")
        self.update("shared_problem",status="VERIFIED",sha256=next(iter(fingerprints)),battery_retention="ideal_press",flight_pitch_deg=15)
    def local(self,stage,command):
        previous=self.data["stages"].get(stage,{})
        if previous.get("command") and previous["command"]!=command:
            raise RuntimeError(f"{stage}: recorded command changed")
        if previous.get("status")=="SUCCEEDED":
            return
        self.update(stage,status="RUNNING",command=command)
        with (self.root/(stage+".log")).open("w") as log:
            code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        self.update(stage,status="SUCCEEDED" if code==0 else "FAILED",exit_code=code)
        if code:
            raise RuntimeError(f"{stage}: exit {code}")
    def physical(self,name,path):
        cfg=json.loads((ROOT/path).read_text())
        root=ROOT/cfg["mma"]["root"]
        runs=json.loads((root/"runs.json").read_text())
        reviews=json.loads((root/"functional_reviews.json").read_text())
        accepted=[]
        for suffix,row in reviews.items():
            if row["exit_code"]:
                continue
            run=Path(runs[suffix])
            checked={**cfg,"review":{"frame_source":str(run/"frame.stl"),"output_directory":str(run/"functional_review")}}
            request=root/("physical_"+suffix+".json")
            request.write_text(json.dumps(checked,indent=2))
            try:
                self.ray(name+"_physical_"+suffix,"frame_physical",str(request),kind="gpu_075" if name=="075" else "gpu_compare")
            except Exception as error:
                self.update(name+"_physical_"+suffix,acceptance="FAILED",error=str(error))
                continue
            manifest=json.loads((run/"manifest.json").read_text())
            evaluation=json.loads((run/"evaluation.json").read_text())
            stand=((evaluation.get("geometry") or {}).get("mass") or {}).get("standing",{}).get("stability",{})
            standing=stand.get("reserve_passed") is True and stand.get("prop_clearance_passed") is True
            if not manifest.get("wall_rule_passed") or (manifest.get("evaluation_gates") or {}).get("missed") or not manifest.get("evaluation_gates") or not standing:
                self.update(name+"_physical_"+suffix,acceptance="FAILED",wall_rule_passed=manifest.get("wall_rule_passed"),independent_evaluation=manifest.get("evaluation_gates"),stand=stand)
                continue
            report=json.loads((run/"functional_review/physical_evaluation.json").read_text())
            accepted.append({"route":name,"suffix":suffix,"run":str(run),"mass_g":report["stl_mass_g"],"pipeline":path})
        self.update(name+"_acceptance",status="PASSED" if accepted else "FAILED",candidates=accepted)
        return accepted
    def branch(self, name, path):
        with self.lock:
            self.data["branches"][name] = "RUNNING"
            self.save()
        try:
            self.ray(name + "_probe", "frame_probe", path)
            self.gate(name)
            self.wait(self.config["validation_job"])
            self.problem_ready.wait()
            self.shared()
            action = "frame_neural" if name == "neural" else "frame_mma"
            self.ray(name + "_optimization", action, path)
            cfg = self.configs[name]
            record = json.loads((ROOT / cfg["mma"]["root"] / cfg["mma"]["fine_dir"] / "result.json").read_text())
            self.update(name + "_optimization", design_status=record["status"], mass_g=record["mass_g"], max_violation=record["max_violation"])
            if record["status"] != "converged":
                raise RuntimeError(f"{name}: {record['status']}; dependent production postprocessing withheld")
            self.local(name+"_postprocessing",[self.python,"tools/formulation_study.py","frame_runs",path])
            self.physical(name,path)
        except Exception as error:
            self.problem_ready.abort()
            with self.lock:
                self.data["branches"][name] = {"status": "FAILED", "error": str(error)}
                self.save()
        else:
            with self.lock:
                self.data["branches"][name] = "SUCCEEDED"
                self.save()
    def hardware(self):
        command = "LD_LIBRARY_PATH=" + shlex.quote(os.environ.get("LD_LIBRARY_PATH", "")) + " nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader"
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "node21", command], capture_output=True, text=True, timeout=15)
        cpu = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "node20", "ps -eo pid,pcpu,rss,comm --sort=-pcpu | head -n 8"], capture_output=True, text=True, timeout=15)
        with (self.root / "hardware.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "exit_code": result.returncode, "output": result.stdout.strip(), "error": result.stderr.strip(),"node20":{"exit_code":cpu.returncode,"output":cpu.stdout.strip(),"error":cpu.stderr.strip()}}) + "\n")
        for name, cfg in self.configs.items():
            phases=(("_probe", cfg["mma"]["root"] + "_probe"), ("_optimization", cfg["mma"]["root"])) if name!="075" else (("_probe" if cfg["mma"]["until"]=="fine" else "_optimization",cfg["mma"]["root"]),)
            for suffix, directory in phases:
                stage = name + suffix
                state = self.data["stages"].get(stage, {})
                path = ROOT / directory / "memory.jsonl"
                if state.get("status") != "RUNNING" or not path.exists():
                    continue
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    lines = stream.read().decode().splitlines()[1:]
                samples = []
                for line in lines:
                    try:
                        samples.append(json.loads(line))
                    except ValueError:
                        pass
                if any(row.get("process_gpu_gb", 0) > state.get("gpu_limit_gb",32) or row.get("rss_gb", 0) > state.get("host_limit_gb",46) for row in samples):
                    self.client.stop_job(state["job"])
                    self.update(stage, stop_reason="Measured memory exceeds declared reservation")
    def display(self,candidate,review=None):
        review=review or str(Path(candidate["run"])/"functional_review/request.json")
        tag="ocp_"+candidate["route"]+"_"+candidate["suffix"]
        try:
            self.ray(tag,"ocp",review,kind="reconstruction",tool="tools/functional_geometry_review.py")
            self.local(tag+"_send",[self.python,"tools/functional_geometry_review.py","send",review])
        except Exception as error:
            self.update(tag+"_display",status="PENDING_VIEWER",error=str(error),frame_source=str(Path(candidate["run"])/"frame.stl"))
    def channels(self,best):
        base=json.loads((ROOT/best["pipeline"]).read_text())
        run=Path(best["run"])
        tag="cables_"+best["route"]+"_"+best["suffix"]
        out=self.root/tag
        cables={"source":str(run/"reconstruction/density_fine.npz"),"domain":str(run/"reconstruction/domain.json"),"body":str(run/"frame.stl"),"output":str(out),"section_body":str(ROOT/base["mma"]["root"]/base["mma"]["variant"]/"geometry.stl")}
        path=self.root/(tag+".json")
        path.write_text(json.dumps(cables,indent=2))
        self.ray(tag,"cables",str(path),kind="reconstruction",tool="tools/reconstruction_study.py")
        report=json.loads((out/"cables.json").read_text())
        paths=report.get("paths",[])
        if not report.get("delivered") or len(paths)!=5 or not all(row.get("routed") and row.get("continuous") and row.get("closed_to_props") and not row.get("members_below_limit") for row in paths):
            raise RuntimeError("Cable reconstruction failed five-path continuity, prop closure, minimum width or closed-frame checks")
        spec={"pipeline":best["pipeline"],"frame_source":str(out/"frame.stl"),"output_directory":str(out/"functional_review"),"derived_cables":str(out/"cables.json")}
        review=out/"review.json"
        review.write_text(json.dumps(spec,indent=2))
        self.ray(tag+"_functional","run",str(review),kind="reconstruction",tool="tools/functional_geometry_review.py")
        checked={**base,"review":{"frame_source":str(out/"frame.stl"),"output_directory":str(out/"functional_review")}}
        physical=out/"physical.json"
        physical.write_text(json.dumps(checked,indent=2))
        self.ray(tag+"_physical","frame_physical",str(physical),kind="gpu_075" if best["route"]=="075" else "gpu_compare")
        evaluation=json.loads((run/"requests/evaluation_frame.json").read_text())
        evaluation.update(name=tag,stl=str(out/"frame.stl"),output=str(out/"evaluation"))
        request_path=out/"evaluation.json"
        request_path.write_text(json.dumps(evaluation,indent=2))
        self.local(tag+"_evaluation",[self.python,"tools/evaluate_frame.py","run",str(request_path)])
        after=json.loads((out/"evaluation/evaluation.json").read_text())
        (out/"evaluation.json").write_text(json.dumps(after,indent=2))
        before=json.loads((run/"evaluation.json").read_text())
        stand=((after.get("geometry") or {}).get("mass") or {}).get("standing",{}).get("stability",{})
        record={"parent":best,"frame_source":str(out/"frame.stl"),"frame_sha256":hashlib.sha256((out/"frame.stl").read_bytes()).hexdigest(),"cables":report,"before":before,"after":after,"stand":stand,"passed":bool(after["assessment"]["good"] and stand.get("reserve_passed") is True and stand.get("prop_clearance_passed") is True)}
        (out/"before_after.json").write_text(json.dumps(record,indent=2))
        self.update(tag+"_acceptance",status="PASSED" if record["passed"] else "FAILED",report=str(out/"before_after.json"))
        if not record["passed"]:
            raise RuntimeError("Cable frame failed independent wall, mechanics or assembly acceptance")
        candidate={**best,"run":str(out),"suffix":best["suffix"]+"_cables","mass_g":report["mass_after_g"],"parent":best}
        self.display(candidate,str(review))
        return candidate
    def comparisons(self,candidates):
        bodies={candidate["route"]+"_"+candidate["suffix"]:{"stl":str(Path(candidate["run"])/"frame.stl"),"request":json.loads((ROOT/candidate["pipeline"]).read_text())["layout"]} for candidate in candidates}
        bodies["ManaFly original"]={"stl":"exports/runs/reference_manafly_original/geometry.stl","request":self.configs["simp"]["layout"],"note":"Original geometry, not watertight; common assembly assumptions for visual/inertia comparison only. Frozen repaired mechanical references are reported separately."}
        cfg={"compare":{"bodies":bodies,"output":str(self.root/"comparison_4views.png"),"summary":str(self.root/"comparison_geometry.json")}}
        path=self.root/"comparisons.json"
        path.write_text(json.dumps(cfg,indent=2))
        self.ray("comparisons_"+str(len(candidates)),"battery_compare",str(path),kind="reconstruction")
        result={"candidates":candidates,"routes":{},"branch_status":self.data["branches"],"geometry_and_dynamics":json.loads((self.root/"comparison_geometry.json").read_text()),"shared_problem":self.data["stages"]["shared_problem"],"reference_mechanics":json.loads((ROOT/"docs/validation/formulation_reference.json").read_text())}
        lines=["# Fortgesetzter Pipeline-Vergleich","","| Route / Rekonstruktion | Masse [g] | Linsenabstand bei 15° [mm] | f1 [Hz] | Armsteifigkeit [N/mm] |","|---|---:|---:|---:|---:|"]
        for candidate in candidates:
            run=Path(candidate["run"])
            function=json.loads((run/"functional_review/report.json").read_text())
            physics=json.loads((run/"functional_review/physical_evaluation.json").read_text())
            evaluation=json.loads((run/("evaluation/evaluation.json" if candidate.get("parent") else "evaluation.json")).read_text())
            route=candidate["route"]+"_"+candidate["suffix"]
            result["routes"][route]={"candidate":candidate,"functions":function,"shared_physics":physics,"independent_evaluation":evaluation}
            fea=evaluation.get("fea") or {}
            lines.append(f"| {route} | {candidate['mass_g']:.3f} | {function['camera']['lens_to_whole_assembly_bottom_mm']:.3f} | {(fea.get('eigenfrequencies_hz') or [None])[0]} | {fea.get('stiffness_n_per_mm')} |")
        lines.extend(["","Die optische Linsenmitte ist im Bauteilmodell noch als Mittelpunkt der Kamerafront angenähert. Die Original-ManaFly-STL ist nicht wasserdicht; ihre Bild- und Trägheitswerte sind kein neuer mechanischer Referenznachweis.","","![Vier gemeinsame Ansichten](comparison_4views.png)"])
        (self.root/"comparison.json").write_text(json.dumps(result,indent=2))
        (self.root/"comparison.md").write_text("\n".join(lines)+"\n")
    def continuation(self):
        candidates=[candidate for name in self.configs for candidate in self.data["stages"].get(name+"_acceptance",{}).get("candidates",[])]
        if not candidates:
            self.update("continuation",status="AWAITING_DESIGN_CORRECTION",reason="No reconstructed frame passed geometry, shared physics, independent evaluation and wall rule; 0.75 input withheld")
            return
        best=min(candidates,key=lambda candidate:candidate["mass_g"])
        self.update("best_43",status="SELECTED",**best)
        self.display(best)
        cabled=self.channels(best)
        self.comparisons(candidates+[cabled])
        base=json.loads((ROOT/best["pipeline"]).read_text())
        for name in ("probe","full"):
            cfg=json.loads(json.dumps(base))
            cfg["shape"]=[182,172,44]
            cfg["mma"].update(root="exports/runs/ground15_075_"+name,variant="ground15_075",method="ground15_075",optimizer="mma",coarse=False,start=str(ROOT/base["mma"]["root"]/base["mma"]["fine_dir"]),fine_dir="fine",fine_start_level=6 if name=="probe" else 3,until="fine" if name=="probe" else "export",settings={"final_max_iterations":3,"final_min_iterations":10} if name=="probe" else {"final_max_iterations":300})
            cfg["v3"]={"compute":"reconstruction","copy":cfg["mma"]["root"]+"/v3_source"}
            path=self.root/("075_"+name+".json")
            path.write_text(json.dumps(cfg,indent=2))
            self.configs["075"]=cfg
            self.ray("075_"+("probe" if name=="probe" else "optimization"),"frame_mma",str(path),kind="gpu_075")
            root=ROOT/cfg["mma"]["root"]
            record=json.loads((root/"fine/result.json").read_text())
            if name=="probe":
                memory=json.loads((root/"memory.json").read_text())["peaks"]["fine"]
                lines=(root/"fine/iterations.jsonl").read_text().splitlines()
                if record["iterations"]!=3 or len(lines)!=3 or not all(math.isfinite(record[key]) for key in ("mass_g","max_violation","seconds_per_iteration")) or not 0<memory.get("process_gpu_gb",float("inf"))<=80 or not 0<memory["rss_gb"]<=160:
                    raise RuntimeError("0.75-mm numerical/resource probe failed; full run withheld")
                self.update("075_probe",gate="PASSED",seconds_per_iteration=record["seconds_per_iteration"],gpu_peak_gb=memory["process_gpu_gb"],host_peak_gb=memory["rss_gb"],note="Numerical probe, not design acceptance")
                self.update("075_schedule",status="MEASURED_PROBE_ESTIMATE",iteration_budget=540,iteration_budget_hours=540*record["seconds_per_iteration"]/3600,note="Projection levels 8/16/32: at most 80 iterations each; final level 64: at most 300. Additional initialization, export and acceptance time required; three probe iterations do not establish convergence time.")
            elif record["status"]!="converged":
                raise RuntimeError("0.75-mm optimizer did not converge; production postprocessing withheld")
            else:
                self.local("075_postprocessing",[self.python,"tools/formulation_study.py","frame_runs",str(path)])
                fine_candidates=self.physical("075",str(path))
                if not fine_candidates:
                    raise RuntimeError("No 0.75-mm reconstructed result passed acceptance")
                cabled_fine=self.channels(min(fine_candidates,key=lambda candidate:candidate["mass_g"]))
                self.comparisons(candidates+[cabled]+fine_candidates+[cabled_fine])
                self.update("continuation",status="RESULTS_VALIDATED",best_075=cabled_fine)
    def monitor(self,done):
        if done.wait(30):
            return
        while not done.is_set():
            try:
                self.hardware()
            except Exception as error:
                print("hardware",str(error),flush=True)
            done.wait(60)
    def run(self):
        with (self.root / "controller.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.root / "status.json").exists() and not self.config.get("resume"):
                raise RuntimeError("Comparison state exists; inspect recorded jobs before restarting")
            for cfg in self.configs.values():
                if not self.config.get("resume") and ((ROOT / cfg["mma"]["root"]).exists() or (ROOT / (cfg["mma"]["root"] + "_probe")).exists()):
                    raise RuntimeError("Fresh comparison output directories required")
            (self.root / "controller.pid").write_text(str(os.getpid()))
            self.save()
            done=threading.Event()
            monitor=threading.Thread(target=self.monitor,args=(done,),daemon=True)
            monitor.start()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.branch, name, path) for name, path in self.config["pipelines"].items()]
                for future in futures:
                    future.result()
            if self.config.get("continue_plan",True):
                try:
                    self.continuation()
                except Exception as error:
                    self.update("continuation",status="FAILED",error=str(error))
            done.set()
            monitor.join()
            self.data["finished"] = datetime.now(timezone.utc).isoformat()
            self.save()

if __name__ == "__main__":
    Comparison(sys.argv[1]).run()
