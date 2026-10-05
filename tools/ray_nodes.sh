#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "$0")/.." && pwd)
python_path="$project_root/.venv/bin/python"
ray_path="$project_root/.venv/bin/ray"
state_dir="$project_root/.wf/ray_nodes"
mkdir -p "$state_dir"
if [[ ${1:-status} == status ]]; then
    "$ray_path" status --address=192.168.2.20:6380
    exit
fi
if [[ ${1:-} != start ]]; then
    echo "usage: tools/ray_nodes.sh [start|status]" >&2
    exit 2
fi
if curl -fsS --max-time 3 http://192.168.2.20:8266/api/version > /dev/null 2>&1; then
    echo "Deep_Frame Ray head already exists; inspect with tools/ray_nodes.sh status" >&2
    exit 1
fi
cat > "$state_dir/env.sh" <<EOF
export DEEP_FRAME_MACHINE=node20_21
export RAY_ADDRESS=http://192.168.2.20:8266
export RAY_PYTHON="$python_path"
export DEEP_FRAME_PYTHON="$python_path"
export DEEP_FRAME_GPU_PYTHON="$python_path"
export CALCULIX_PATH="$project_root/.wf/sysroot/usr/bin/ccx"
export DEEP_FRAME_SLICER="$project_root/.wf/sysroot/usr/bin/prusa-slicer"
export LD_LIBRARY_PATH="$project_root/.wf/nvidia_node21/usr/lib/x86_64-linux-gnu:$project_root/.wf/sysroot/lib/x86_64-linux-gnu:$project_root/.wf/sysroot/usr/lib/x86_64-linux-gnu:$project_root/.wf/sysroot/usr/lib/x86_64-linux-gnu/lapack:$project_root/.wf/sysroot/usr/lib/x86_64-linux-gnu/blas:$project_root/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$project_root/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$project_root/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
export CUPY_COMPILE_WITH_PTX=0
EOF
for node in node20 node21; do
    ssh -o BatchMode=yes "$node" bash -s -- "$project_root" "$node" <<'REMOTE'
set -euo pipefail
project_root=$1
node=$2
source "$project_root/.wf/ray_nodes/env.sh"
unset RAY_ADDRESS
export PATH="$project_root/.venv/bin:$PATH"
export RAY_USAGE_STATS_ENABLED=0
export RAY_enable_worker_prestart=0
export RAY_num_workers_soft_limit=2
export RAY_JOB_START_TIMEOUT_SECONDS=604800
options=(--disable-usage-stats --block --dashboard-agent-listen-port=53465
    --dashboard-agent-grpc-port=53466 --metrics-export-port=53467
    --runtime-env-agent-port=53468 --node-manager-port=53469 --object-manager-port=53470
    --min-worker-port=22000 --max-worker-port=22999)
if [[ $node == node20 ]]; then
    mkdir -p /scratch/tmp/deep_frame_ray
    options+=(--head --node-ip-address=192.168.2.20 --port=6380 --ray-client-server-port=10003
        --num-cpus=80 --num-gpus=0 --memory=824633720832 --object-store-memory=8589934592
        --resources='{"gpu_gb":0}' --temp-dir=/scratch/tmp/deep_frame_ray
        --include-dashboard=true --dashboard-host=192.168.2.20 --dashboard-port=8266)
else
    options+=(--address=192.168.2.20:6380 --node-ip-address=192.168.2.21
        --num-cpus=32 --num-gpus=1 --memory=274877906944 --object-store-memory=8589934592
        --resources='{"gpu_gb":80}')
fi
nohup "$project_root/.venv/bin/ray" start "${options[@]}" > "$project_root/.wf/ray_nodes/$node.log" 2>&1 < /dev/null &
echo $! > "$project_root/.wf/ray_nodes/$node.pid"
REMOTE
    if [[ $node == node20 ]]; then
        ready=false
        for attempt in {1..30}; do
            if curl -fsS --max-time 2 http://192.168.2.20:8266/api/version > /dev/null 2>&1; then
                ready=true
                break
            fi
            sleep 1
        done
        if [[ $ready != true ]]; then
            cat "$state_dir/node20.log" >&2
            exit 1
        fi
    fi
done
echo "Deep_Frame Ray: http://192.168.2.20:8266; source $state_dir/env.sh"
