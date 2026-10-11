#!/usr/bin/env bash
# Reproduce/verify the authorized M05 cloud run; GPU stages are explicit.
set -euo pipefail
unset LD_LIBRARY_PATH
VLA_ROOT="${VLA_ROOT:-/data/VLA-Forge/project/robot-vla}"
MODEL_OPS_ROOT="${MODEL_OPS_ROOT:-/data/ModelOps-Platform}"
M05_EVIDENCE="${M05_EVIDENCE:-/data/VLA-Forge/Evidence/M05}"
cd "$VLA_ROOT"
export PYTHONPATH="$VLA_ROOT/benchmarks/robot-vla-engine/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1
export LIBERO_CONFIG_PATH="$VLA_ROOT/work/libero"
case "${1:-verify}" in
  verify)
    .venv/bin/pytest -q benchmarks/robot-vla-engine/tests/test_m05_training.py benchmarks/robot-vla-engine/tests/test_m05_serving.py benchmarks/robot-vla-engine/tests/test_m05_development.py
    .venv/bin/python benchmarks/robot-vla-engine/scripts/validate_m05_development.py
    .venv/bin/python benchmarks/robot-vla-engine/scripts/summarize_m05_training.py
    .venv/bin/python benchmarks/robot-vla-engine/scripts/validate_m05_lineage.py
    .venv/bin/python benchmarks/robot-vla-engine/scripts/summarize_m05_system.py
    .venv/bin/python benchmarks/robot-vla-engine/scripts/summarize_m05_optimizations.py
    .venv/bin/python benchmarks/robot-vla-engine/scripts/compare_m05_graph_rollouts.py
    .venv/bin/python benchmarks/robot-vla-engine/scripts/validate_m05_development.py --root "$M05_EVIDENCE/development_graph" --conditions frozen corrective
    .venv/bin/python benchmarks/robot-vla-engine/scripts/validate_m05_artifacts.py --expect-service
    ;;
  development)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/run_m05_development.py --conditions frozen standard corrective replay
    ;;
  profile)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/profile_m05_inference.py --calls 100 --timeout 180
    ;;
  component-profile)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/run_m05_training.py --config "$M05_EVIDENCE/training_perf/configs/component_profile.json"
    ;;
  frozen-profile)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/profile_m05_frozen.py
    ;;
  graph-check)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/check_m05_graph_observations.py --model corrective
    .venv/bin/python benchmarks/robot-vla-engine/scripts/check_m05_graph_observations.py --model frozen
    ;;
  graph)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/run_m05_development.py --conditions frozen corrective --backend cuda_graph --output "$M05_EVIDENCE/development_graph"
    ;;
  memory-budget)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/check_m05_memory_budget.py
    ;;
  demo)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/build_m05_charts.py
    .venv/bin/python benchmarks/robot-vla-engine/scripts/build_m05_demo.py
    ;;
  managed-service)
    export PYTHONPATH="$MODEL_OPS_ROOT:$PYTHONPATH"
    .venv/bin/python -m pipelines.robotics.serve --registry "$M05_EVIDENCE/modelops/registry" --backend eager --log "$M05_EVIDENCE/modelops/serving_events.jsonl" --port 18769
    ;;
  serving-smoke)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/smoke_m05_managed_serving.py
    ;;
  serving-benchmark)
    .venv/bin/python benchmarks/robot-vla-engine/scripts/bench_m05_serving.py --version "${2:?model version required}"
    ;;
  *) echo "Use: $0 {verify|development|profile|frozen-profile|component-profile|graph-check|graph|memory-budget|demo|managed-service|serving-smoke|serving-benchmark VERSION}" >&2; exit 2;;
esac
