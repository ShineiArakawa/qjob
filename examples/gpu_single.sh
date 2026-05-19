#!/usr/bin/env bash
#QJOB --name gpu-single-demo --cpus 4 --gpus 1
#QJOB --mem 8G --walltime 00:10:00 --priority high

set -euo pipefail

echo "job_id=${QJOB_JOB_ID:-}"
echo "job_name=${QJOB_JOB_NAME:-}"
echo "user=${QJOB_USER:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi

python3 - <<'PY'
import os
import time

print("Starting single-GPU workload")
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

try:
    import torch
except ImportError:
    print("torch is not installed; simulating GPU work.")
    for step in range(1, 16):
        print(f"step={step} simulated_gpu_batch={step * 32}")
        time.sleep(1)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch device =", device)
    x = torch.randn(2048, 2048, device=device)
    for step in range(1, 60):
        x = x.relu()
        print(f"step={step} mean={x.mean().item():.6f}")
        time.sleep(1)

print("Single-GPU workload finished")
PY
