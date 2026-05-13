#!/usr/bin/env bash
#QJOB --name mixed-cpu-gpu-demo --cpus 6 --gpus 1
#QJOB --mem 12G --walltime 00:10:00 --priority normal

set -euo pipefail

echo "job_id=${QJOB_JOB_ID:-}"
echo "job_name=${QJOB_JOB_NAME:-}"
echo "user=${QJOB_USER:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

python - <<'PY'
import math
import os
import time

print("Starting mixed CPU/GPU workload")
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

for step in range(1, 6):
    cpu_value = sum(math.sqrt(i) for i in range(1, 100_000))
    print(f"step={step} cpu_value={cpu_value:.2f}")
    time.sleep(1)

print("Mixed workload finished")
PY
