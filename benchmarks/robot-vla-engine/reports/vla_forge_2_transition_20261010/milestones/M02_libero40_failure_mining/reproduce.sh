#!/usr/bin/env bash
set -euo pipefail

# LeRobot's pinned PyTorch wheel must use its bundled cuDNN, not the base Conda copy.
unset LD_LIBRARY_PATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
REPORT="$ROOT/reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining"
ARTIFACTS="${VLA_M02_ARTIFACT_ROOT:-/data/vla-forge-artifacts/M02_libero40}"
PYTHON="$ROOT/.venv-policy/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing pinned project environment: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$ROOT/data/libero/meta/info.json" ]]; then
  echo "Missing pinned LIBERO data metadata: $ROOT/data/libero/meta/info.json" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/data/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET=1
export LIBERO_CONFIG_PATH="$ROOT/work/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$REPORT" "$ARTIFACTS"
exec > >(tee -a "$REPORT/run.log") 2>&1

"$PYTHON" "$ROOT/scripts/configure_libero.py"
"$PYTHON" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

snapshots = [
    (
        "lerobot/smolvla_libero",
        "31d453f7edd78c839a8bbc39744a292686daf0de",
        Path("/data/models/smolvla_libero/31d453f7edd78c839a8bbc39744a292686daf0de"),
        None,
    ),
    (
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        Path("/data/models/smolvlm_full/7b375e1b73b11138ff12fe22c8f2822d8fe03467"),
        [
            "config.json", "model.safetensors", "generation_config.json",
            "preprocessor_config.json", "processor_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
            "chat_template.json", "vocab.json", "merges.txt",
        ],
    ),
]
for repo, revision, destination, patterns in snapshots:
    if not (destination / "model.safetensors").is_file():
        snapshot_download(
            repo,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=patterns,
        )
PY

RUN_STATUS=0
"$PYTHON" "$ROOT/scripts/run_m02_libero40.py" \
  --root "$ROOT" \
  --output-dir "$REPORT" \
  --artifact-root "$ARTIFACTS" \
  --seed-base "${VLA_M02_SEED_BASE:-82000}" \
  --max-attempts "${VLA_M02_MAX_ATTEMPTS:-3}" || RUN_STATUS=$?

ANALYSIS_STATUS=0
"$PYTHON" "$ROOT/scripts/analyze_m02_libero40.py" --output-dir "$REPORT" || ANALYSIS_STATUS=$?

if [[ "$RUN_STATUS" -ne 0 ]]; then
  exit "$RUN_STATUS"
fi
exit "$ANALYSIS_STATUS"
