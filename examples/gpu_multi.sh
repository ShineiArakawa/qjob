#!/usr/bin/env bash
#QJOB --name gpu-multi-demo --cpus 8 --gpus 2
#QJOB --mem 16G --walltime 00:15:00 --priority high

set -euo pipefail

echo "job_id=${QJOB_JOB_ID:-}"
echo "job_name=${QJOB_JOB_NAME:-}"
echo "user=${QJOB_USER:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

python - <<'PY'
import os
import time

visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
print("Starting multi-GPU workload")
print("visible GPU count =", len(visible))
print("CUDA_VISIBLE_DEVICES =", ",".join(visible))

try:
    import torch
except ImportError:
    print("torch is not installed; simulating work across visible GPUs.")
    for step in range(1, 6):
        print(f"step={step} simulated_devices={visible or ['none']}")
        time.sleep(1)
else:
    device_count = torch.cuda.device_count()
    print("torch visible GPU count =", device_count)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1024, 1024, device=device)
    for step in range(1, 6):
        x = x @ x.T
        print(f"step={step} norm={x.norm().item():.4f}")
        time.sleep(1)

print("Multi-GPU workload finished")
PY
