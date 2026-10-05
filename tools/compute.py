import asyncio
import base64
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {
    "address": os.environ.get("RAY_ADDRESS", "http://127.0.0.1:8265"),
    "machine": os.environ.get("DEEP_FRAME_MACHINE", "local"),
    "forward_env": ("CALCULIX_PATH", "PYTHONPATH", "CUDA_PATH", "LD_LIBRARY_PATH", "RAY_ADDRESS"),
    "forward_prefix": "DEEP_FRAME_",
    "thread_env": ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
    "heads": {"local": {"num_cpus": 24, "num_gpus": 1, "memory_gb": 40, "object_store_gb": 1, "gpu_gb": 14,
                        "ray_python": "C:/clones/ray-venv/Scripts/python.exe", "gpus_per_job": 0, "gpu_margin": 1.2, "card_gb": 14, "throttle": {}},
              "dgx": {"num_cpus": 128, "num_gpus": 8, "memory_gb": 768, "object_store_gb": 8, "gpu_gb": 640,
                      "ray_python": "/home/john/ray-venv/bin/python", "gpus_per_job": 1, "gpu_margin": 1.0, "card_gb": 80, "throttle": {}},
              "node20_21": {"num_cpus": 80, "num_gpus": 0, "memory_gb": 768, "object_store_gb": 8, "gpu_gb": 0,
                            "ray_python": str(Path(__file__).resolve().parents[1] / ".venv/bin/python"),
                            "gpus_per_job": 1, "gpu_margin": 1.0, "card_gb": 80,
                            "cpu_node": "192.168.2.20", "gpu_node": "192.168.2.21", "throttle": {},
                            "dashboard_host": "192.168.2.20", "dashboard_port": 8266,
                            "agent_ports": (53465, 53466, 53467, 53468),
                            "head_options": ("--node-ip-address=192.168.2.20", "--port=6380", "--ray-client-server-port=10003",
                                             "--node-manager-port=53469", "--object-manager-port=53470",
                                             "--min-worker-port=22000", "--max-worker-port=22999", "--temp-dir=/scratch/tmp/deep_frame_ray")}},
    "dashboard_port": 8265,
    "agent_ports": (53365, 53366, 53367, 53368),
    "start_timeout_s": 7 * 24 * 3600,
}
JOB_TYPES = {
    "density_neural": {"num_cpus": 4, "memory_gb": 24, "gpu_gb": 14.4},
    "density_simp": {"num_cpus": 4, "memory_gb": 8, "gpu_gb": 11.49},
    "density_simp_1mm": {"num_cpus": 8, "memory_gb": 28, "gpu_gb": 13.24},
    "geometry": {"num_cpus": 4, "memory_gb": 4, "gpu_gb": 0},
    "reconstruction": {"num_cpus": 4, "memory_gb": 26, "gpu_gb": 0},
    "fea_static": {"num_cpus": 2, "memory_gb": 8, "gpu_gb": 0},
    "fea_modal": {"num_cpus": 2, "memory_gb": 10, "gpu_gb": 0},
    "render": {"num_cpus": 2, "memory_gb": 3, "gpu_gb": 0},
    "wall_check": {"num_cpus": 2, "memory_gb": 3, "gpu_gb": 0},
    "suite": {"num_cpus": 4, "memory_gb": 6, "gpu_gb": 0},
    "cpu": {"num_cpus": 2, "memory_gb": 4, "gpu_gb": 0},
    "gpu": {"num_cpus": 4, "memory_gb": 6, "gpu_gb": 5.5},
    "gpu_a100": {"num_cpus": 16, "memory_gb": 46, "gpu_gb": 80},
    "gpu_compare": {"num_cpus": 16, "memory_gb": 46, "gpu_gb": 16.65, "gpu_margin": 1.3, "shared_gpu": True},
    "gpu_075": {"num_cpus": 16, "memory_gb": 160, "gpu_gb": 80},
}
join = subprocess.list2cmdline if os.name == "nt" else shlex.join

def encode(data):
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def profile(config=CONFIG, machine=None):
    spec = config["heads"][machine or config["machine"]]
    return dict(spec, ray_python=os.environ.get("RAY_PYTHON", spec["ray_python"]))

def request(kind, command, cwd, environ=os.environ, config=CONFIG, python=getattr(sys, "_base_executable", sys.executable), machine=None, gpu_gb=None):
    need, spec = dict(JOB_TYPES[kind], **({"gpu_gb": gpu_gb} if gpu_gb else {})), profile(config, machine)
    if spec["gpus_per_job"] and need["gpu_gb"] > spec["card_gb"]:
        raise ValueError(f"{kind} needs {need['gpu_gb']} GiB GPU memory; {machine or config['machine']} provides {spec['card_gb']} GiB per GPU")
    measured = need["gpu_gb"] or (JOB_TYPES["gpu"]["gpu_gb"] if any("gpu-venv" in str(part) or "Deep_Frame-gpu/" in str(part) for part in command) else 0)
    gpu_gb = min(round(measured * need.get("gpu_margin", spec["gpu_margin"]), 1), spec["card_gb"]) if measured else 0
    env = {key: value for key, value in environ.items() if key in config["forward_env"] or key.startswith(config["forward_prefix"])}
    env["DEEP_FRAME_GPU_RESERVED"] = "1" if gpu_gb else "0"
    env.update({key: str(need["num_cpus"]) for key in config["thread_env"]})
    payload = {"command": list(command), "cwd": str(Path(cwd).resolve()), "env": env}
    resources = {"gpu_gb": gpu_gb} if gpu_gb else {}
    if not gpu_gb and spec.get("cpu_node"):
        resources["node:" + spec["cpu_node"]] = 0.001
    if gpu_gb and spec.get("gpu_node"):
        resources["node:" + spec["gpu_node"]] = 0.001
    return {"entrypoint": join([python, str(Path(__file__).resolve()), "exec", encode(payload)]),
            "entrypoint_num_cpus": need["num_cpus"], "entrypoint_memory": int(need["memory_gb"] * 2**30),
            "entrypoint_resources": resources or None,
            "entrypoint_num_gpus": (round(gpu_gb / spec["card_gb"], 4) if need.get("shared_gpu") else spec["gpus_per_job"]) if gpu_gb and spec["gpus_per_job"] else None,
            "metadata": {"type": kind, "cwd": payload["cwd"], "command": join(command)[:500]}}

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
    pid = supervisor()
    if os.name == "nt" and pid is None:
        return 1
    watch(pid)
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
    if not handle:
        os._exit(1)
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
    for key, limit in profile(config)["throttle"].items():
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

def head_command(name="local", config=CONFIG):
    spec = profile(config, name)
    return [str(Path(spec["ray_python"]).with_name("ray.exe" if os.name == "nt" else "ray")), "start", "--head",
            "--num-cpus", str(spec["num_cpus"]), "--num-gpus", str(spec["num_gpus"]),
            "--memory", str(spec["memory_gb"] * 2**30), "--object-store-memory", str(spec["object_store_gb"] * 2**30),
            "--resources", json.dumps({"gpu_gb": spec["gpu_gb"]}), "--include-dashboard", "true",
            "--dashboard-host", spec.get("dashboard_host", "127.0.0.1"), "--dashboard-port", str(spec.get("dashboard_port", config["dashboard_port"])), "--disable-usage-stats",
            *[f"--{flag}={port}" for flag, port in zip(("dashboard-agent-listen-port", "dashboard-agent-grpc-port",
                                                       "metrics-export-port", "runtime-env-agent-port"), spec.get("agent_ports", config["agent_ports"]))],
            *spec.get("head_options", ())]

def head(name="local", config=CONFIG):
    return subprocess.call(head_command(name, config), env={**os.environ, "RAY_JOB_START_TIMEOUT_SECONDS": str(config["start_timeout_s"]),
                                                           "RAY_num_workers_soft_limit": "2", "RAY_enable_worker_prestart": "0"})

def main(argv):
    if argv[:1] == ["exec"] and len(argv) == 2:
        return execute(argv[1])
    if argv[:1] == ["head"] and len(argv) <= 2 and set(argv[1:]) <= set(CONFIG["heads"]):
        return head(*argv[1:])
    cwd, rest = (argv[2], argv[:1] + argv[3:]) if argv[1:2] == ["--cwd"] else (os.getcwd(), argv)
    if len(rest) < 3 or rest[0] not in JOB_TYPES or rest[1] != "--":
        raise SystemExit("usage: compute.py {" + ",".join(JOB_TYPES) + "} [--cwd DIR] -- <command...> | head [" + ",".join(CONFIG["heads"]) + "]")
    if importlib.util.find_spec("ray") is None:
        return subprocess.call([profile()["ray_python"], __file__, rest[0], "--cwd", cwd, *rest[1:]])
    return submit(rest[0], rest[2:], cwd)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
