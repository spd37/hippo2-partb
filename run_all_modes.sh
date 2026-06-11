#!/bin/bash
# Drives the full 3-mode comparison unattended. Offline WandB.
set -u
cd /home/user/hippo2-partb
PY=/home/user/hippo2-partb/hippo2-partb/venv/bin/python
export WANDB_MODE=offline

# 1) Wait for the already-running 'none' job to finish
echo "[driver] waiting for current 'none' run (PID 1570405) to finish..."
while kill -0 1570405 2>/dev/null; do sleep 30; done
echo "[driver] 'none' done at $(date '+%H:%M:%S')"

# 2) poly, then 3) gplearn
for MODE in poly gplearn; do
  echo "[driver] === starting mode=$MODE at $(date '+%H:%M:%S') ==="
  FEATURE_ENG_MODE=$MODE $PY nm20275.py > run_${MODE}.log 2>&1
  echo "[driver] === mode=$MODE finished (exit $?) at $(date '+%H:%M:%S') ==="
done
echo "[driver] ALL THREE MODES COMPLETE at $(date '+%H:%M:%S')"
