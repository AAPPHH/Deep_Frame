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
    "ray_python": os.environ.get("RAY_PYTHON", "C:/clones/ray-venv/Scripts/python.exe" if os.name == "nt" else "/home/john/ray-venv/bin/python"),
    "forward_env": ("CALCULIX_PATH", "PYTHONPATH", "CUDA_PATH", "LD_LIBRARY_PATH"),
    "forward_prefix": "DEEP_FRAME_",
    "thread_env": ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
    "head": {"num_cpus": 24, "num_gpus": 1, "memory_gb": 40, "object_store_gb": 1, "gpu_gb": 14,
             "dashboard_port": 8265, "start_timeout_s": 7 * 24 * 3600},
}
JOB_TYPES = {
    "density_neural": {"num_cpus": 4, "memory_gb": 4, "gpu_gb": 12},
    "density_simp": {"num_cpus": 4, "memory_gb": 8, "gpu_gb": 10},
    "density_simp_1mm": {"num_cpus": 8, "memory_gb": 28, "gpu_gb": 14},
    "geometry": {"num_cpus": 4, "memory_gb": 4, "gpu_gb": 0},
    "reconstruction": {"num_cpus": 4, "memory_gb": 26, "gpu_gb": 0},
    "fea_static": {"num_cpus": 2, "memory_gb": 8, "gpu_gb": 0},
    "fea_modal": {"num_cpus": 2, "memory_gb": 10, "gpu_gb": 0},
    "render": {"num_cpus": 2, "memory_gb": 3, "gpu_gb": 0},
    "wall_check": {"num_cpus": 2, "memory_gb": 3, "gpu_gb": 0},
    "suite": {"num_cpus": 4, "memory_gb": 6, "gpu_gb": 0},
    "cpu": {"num_cpus": 2, "memory_gb": 4, "gpu_gb": 0},
    "gpu": {"num_cpus": 4, "memory_gb": 6, "gpu_gb": 14},
    "gpu_a100": {"num_cpus": 16, "memory_gb": 46, "gpu_gb": 80, "num_gpus": 1},
}
join = subprocess.list2cmdline if os.name == "nt" else shlex.join

def encode(data):
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def request(kind, command, cwd, environ=os.environ, config=CONFIG, python=sys.executable):
    need = JOB_TYPES[kind]
    env = {key: value for key, value in environ.items() if key in config["forward_env"] or key.startswith(config["forward_prefix"])}
    env.update({key: str(need["num_cpus"]) for key in config["thread_env"]})
    payload = {"command": list(command), "cwd": str(Path(cwd).resolve()), "env": env}
    return {"entrypoint": join([python, str(Path(__file__).resolve()), "exec", encode(payload)]),
            "entrypoint_num_cpus": need["num_cpus"], "entrypoint_memory": int(need["memory_gb"] * 2**30),
            "entrypoint_resources": {"gpu_gb": need["gpu_gb"]} if need["gpu_gb"] else None,
            "metadata": {"type": kind, "cwd": payload["cwd"], "command": join(command)[:500]}, **({"entrypoint_num_gpus": need["num_gpus"]} if "num_gpus" in need else {})}

def execute(token):
    payload = json.loads(base64.urlsafe_b64decode(token))
    command = payload["command"]
    return subprocess.call(command[0] if len(command) == 1 else command, shell=len(command) == 1,
                           cwd=payload["cwd"], env={**os.environ, **payload["env"]})

async def follow(client, job):
    async for lines in client.tail_job_logs(job):
        print(lines, end="", flush=True)

def submit(kind, command, cwd, config=CONFIG):
    from ray.job_submission import JobStatus, JobSubmissionClient
    client = JobSubmissionClient(config["address"])
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
               "--dashboard-host", "127.0.0.1", "--dashboard-port", str(spec["dashboard_port"]), "--disable-usage-stats"]
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
