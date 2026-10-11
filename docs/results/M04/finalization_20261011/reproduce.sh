#!/usr/bin/env bash
set -euo pipefail
cd /data/VLA-Forge/project/robot-vla
.venv/bin/ruff check benchmarks/robot-vla-engine
.venv/bin/pytest tests benchmarks/robot-vla-engine/tests -q
.venv/bin/python benchmarks/robot-vla-engine/scripts/audit_m04_final_admission.py
# The bounded controller is already terminal; this is a no-op on completed budget:
.venv/bin/python benchmarks/robot-vla-engine/scripts/run_m04_final_cross_task.py
# No policy training and no historical rollout rerun.
