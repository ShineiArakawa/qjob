#!/usr/bin/env bash
#QJOB --name train-gpu --cpus 4 --gpus 3
#QJOB --mem 8G --walltime 00:30:00 --priority 100

set -euo pipefail

echo "job_id=${QJOB_JOB_ID:-}"
echo "job_name=${QJOB_JOB_NAME:-}"
echo "user=${QJOB_USER:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

python3 - <<'PY'
import os
import time

print("Starting example training job")
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

try:
    import torch
except ImportError:
    print("torch is not installed; simulating training on the assigned device.")
    for step in range(1, 6):
        print(f"step={step} loss={1.0 / step:.4f}")
        time.sleep(1)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch device =", device)
    x = torch.randn(4096, 4096, device=device)
    for step in range(1, 6):
        y = x @ x
        loss = y.mean()
        print(f"step={step} loss={loss.item():.4f}")
        time.sleep(1)

print("Training example finished")
PY
