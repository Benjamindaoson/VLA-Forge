#!/usr/bin/env bash
set -euo pipefail
cd /data/VLA-Forge/project/robot-vla
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export M04_DATASET_PATH=/data/VLA-Forge/Evidence/M04/dataset/lerobot_v3_verified_r7
unset LD_LIBRARY_PATH
PY=.venv/bin/python
case "${1:-audit}" in
 audit)
  "$PY" -m pytest tests benchmarks/robot-vla-engine/tests -q
  "$PY" benchmarks/robot-vla-engine/scripts/validate_m04_verified_v3.py
  "$PY" benchmarks/robot-vla-engine/scripts/audit_m04_artifact_integrity.py
  ;;
 benchmark)
  "$PY" benchmarks/robot-vla-engine/scripts/bench_m04_cuda_graph.py
  ;;
 rollout)
  : "${M04_RUN_ID:?Set a fresh unique M04_RUN_ID}"
  export M04_SUITE=goal M04_TASK_ID=6 M04_SOURCE_EPISODE=6 M04_POLICY_SEED=84182
  export M04_POLICY_CONDITIONS=brightness_first50 M04_BRIGHTNESS_CAMERA=wrist M04_BRIGHTNESS_GAIN=1.10 M04_INTERVENTION_STEPS=50
  export M04_INFERENCE_BACKEND="${M04_INFERENCE_BACKEND:-cuda_graph}"
  export M04_ATTEMPT_DIR=/data/VLA-Forge/Evidence/M04/performance/rollouts
  "$PY" benchmarks/robot-vla-engine/scripts/run_m04_policy_attempt.py
  ;;
 *) echo 'Usage: reproduce.sh audit|benchmark|rollout' >&2; exit 2 ;;
esac
