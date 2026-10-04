import asyncio
import base64
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {
    "address": os.environ.get("RAY_ADDRESS", "http://127.0.0.1:8265"),
    "ray_python": os.environ.get("RAY_PYTHON", "C:/clones/ray-venv/Scripts/python.exe"),
    "forward_env": ("CALCULIX_PATH", "PYTHONPATH", "CUDA_PATH"),
    "forward_prefix": "DEEP_FRAME_",
    "thread_env": ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
    "throttle": {"Deep_Frame-recon3": 2},
    "head": {"num_cpus": 24, "num_gpus": 1, "memory_gb": 40, "object_store_gb": 1, "gpu_gb": 14,
             "dashboard_port": 8265, "agent_ports": (53365, 53366, 53367, 53368), "start_timeout_s": 7 * 24 * 3600},
}
JOB_TYPES = {
    "density_neural": {"num_cpus": 4, "memory_gb": 24, "gpu_gb": 14},
    "density_simp": {"num_cpus": 4, "memory_gb": 8, "gpu_gb": 14},
    "density_simp_1mm": {"num_cpus": 8, "memory_gb": 28, "gpu_gb": 14},
    "geometry": {"num_cpus": 4, "memory_gb": 4, "gpu_gb": 0},
    "reconstruction": {"num_cpus": 4, "memory_gb": 26, "gpu_gb": 0},
    "fea_static": {"num_cpus": 2, "memory_gb": 8, "gpu_gb": 0},
    "fea_modal": {"num_cpus": 2, "memory_gb": 10, "gpu_gb": 0},
    "render": {"num_cpus": 2, "memory_gb": 3, "gpu_gb": 0},
    "wall_check": {"num_cpus": 2, "memory_gb": 3, "gpu_gb": 0},
    "suite": {"num_cpus": 4, "memory_gb": 6, "gpu_gb": 0},
    "cpu": {"num_cpus": 2, "memory_gb": 4, "gpu_gb": 0},
    "gpu": {"num_cpus": 4, "memory_gb": 6, "gpu_gb": 2},
}

def encode(data):
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def request(kind, command, cwd, environ=os.environ, config=CONFIG, python=getattr(sys, "_base_executable", sys.executable)):
    need = JOB_TYPES[kind]
    if not need["gpu_gb"] and any("gpu-venv" in str(part) or "Deep_Frame-gpu/" in str(part) for part in command):
        need = dict(need, gpu_gb=JOB_TYPES["gpu"]["gpu_gb"])
    env ={key: value for key, value in environ.items() if key in config["forward_env"] or key.startswith(config["forward_prefix"])}
    env.update({key: str(need["num_cpus"]) for key in config["thread_env"]})
    payload = {"command": list(command), "cwd": str(Path(cwd).resolve()), "env": env}
    return {"entrypoint": subprocess.list2cmdline([python, str(Path(__file__).resolve()), "exec", encode(payload)]),
            "entrypoint_num_cpus": need["num_cpus"], "entrypoint_memory": int(need["memory_gb"] * 2**30),
            "entrypoint_resources": {"gpu_gb": need["gpu_gb"]} if need["gpu_gb"] else None,
            "metadata": {"type": kind, "cwd": payload["cwd"], "command": subprocess.list2cmdline(command)[:500]}}

def contain():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]
    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000
    ok = kernel.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    ok = ok and kernel.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(-1))
    print("contain", ok, ctypes.get_last_error(), file=sys.stderr) if not ok else None
    return job

def execute(token):
    payload = json.loads(base64.urlsafe_b64decode(token))
    command = payload["command"]
    job = contain()
    watch(supervisor())
    process = subprocess.Popen(command[0] if len(command) == 1 else command, shell=len(command) == 1,
                               cwd=payload["cwd"], env={**os.environ, **payload["env"]})
    return process.wait()

def supervisor():
    if os.name != "nt":
        return None
    query = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={os.getppid()}').ParentProcessId"
    output = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True).stdout.strip()
    return int(output) if output.isdigit() else None

def watch(pid):
    if pid is None:
        return
    import ctypes
    import threading
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if handle:
        threading.Thread(target=lambda: (kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF), os._exit(1)), daemon=True).start()

async def follow(client, job):
    async for lines in client.tail_job_logs(job):
        print(lines, end="", flush=True)

def active(client, key):
    return sum(1 for job in client.list_jobs() if job.status in ("PENDING", "RUNNING") and key in (job.metadata or {}).get("cwd", ""))

def submit(kind, command, cwd, config=CONFIG):
    from ray.job_submission import JobStatus, JobSubmissionClient
    client = JobSubmissionClient(config["address"])
    for key, limit in config["throttle"].items():
        while key in str(Path(cwd).resolve()) and active(client, key) >= limit:
            time.sleep(15)
    job = client.submit_job(**request(kind, command, cwd, config=config))
    print(f"compute: {kind} job {job} submitted", file=sys.stderr, flush=True)
    try:
        asyncio.run(follow(client, job))
        while not client.get_job_status(job).is_terminal():
            time.sleep(1)
    except KeyboardInterrupt:
        client.stop_job(job)
        raise
    info = client.get_job_info(job)
    if info.driver_exit_code is not None:
        return info.driver_exit_code
    print(f"compute: job {job} {info.status}: {info.message}", file=sys.stderr)
    return 0 if info.status == JobStatus.SUCCEEDED else 1

def head(config=CONFIG):
    spec = config["head"]
    command = [str(Path(config["ray_python"]).with_name("ray.exe" if os.name == "nt" else "ray")), "start", "--head",
               "--num-cpus", str(spec["num_cpus"]), "--num-gpus", str(spec["num_gpus"]),
               "--memory", str(spec["memory_gb"] * 2**30), "--object-store-memory", str(spec["object_store_gb"] * 2**30),
               "--resources", json.dumps({"gpu_gb": spec["gpu_gb"]}), "--include-dashboard", "true",
               "--dashboard-host", "127.0.0.1", "--dashboard-port", str(spec["dashboard_port"]), "--disable-usage-stats",
               *[f"--{name}={port}" for name, port in zip(("dashboard-agent-listen-port", "dashboard-agent-grpc-port",
                                                          "metrics-export-port", "runtime-env-agent-port"), spec["agent_ports"])]]
    return subprocess.call(command, env={**os.environ, "RAY_JOB_START_TIMEOUT_SECONDS": str(spec["start_timeout_s"]),
                                               "RAY_num_workers_soft_limit": "2", "RAY_enable_worker_prestart": "0"})

def main(argv):
    if argv[:1] == ["exec"] and len(argv) == 2:
        return execute(argv[1])
    if argv == ["head"]:
        return head()
    cwd, rest = (argv[2], argv[:1] + argv[3:]) if argv[1:2] == ["--cwd"] else (os.getcwd(), argv)
    if len(rest) < 3 or rest[0] not in JOB_TYPES or rest[1] != "--":
        raise SystemExit("usage: compute.py {" + ",".join(JOB_TYPES) + "} [--cwd DIR] -- <command...> | head")
    if importlib.util.find_spec("ray") is None:
        return subprocess.call([CONFIG["ray_python"], __file__, rest[0], "--cwd", cwd, *rest[1:]])
    return submit(rest[0], rest[2:], cwd)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
