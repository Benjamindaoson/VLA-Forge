#!/usr/bin/env bash
set -euo pipefail
cd /data/VLA-Forge/project/robot-vla
unset LD_LIBRARY_PATH
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1
export PYTHONPATH=/data/VLA-Forge/project/robot-vla/benchmarks/robot-vla-engine/src
exec /data/VLA-Forge/project/robot-vla/.venv/bin/python benchmarks/robot-vla-engine/scripts/run_m06_independent.py "$@"
