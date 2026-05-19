#!/usr/bin/env bash
#QJOB --name cpu-only-demo --cpus 2 --gpus 0
#QJOB --mem 1G --walltime 00:05:00 --priority normal

set -euo pipefail

echo "job_id=${QJOB_JOB_ID:-}"
echo "job_name=${QJOB_JOB_NAME:-}"
echo "user=${QJOB_USER:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

python3 - <<'PY'
import hashlib
import time

print("Starting CPU-only workload")

payload = b"qjob-cpu-demo" * 100_000
digest = ""

for step in range(1, 6):
    digest = hashlib.sha256(payload + str(step).encode()).hexdigest()
    print(f"step={step} digest={digest[:16]}")
    time.sleep(1)

print("CPU-only workload finished")
PY
