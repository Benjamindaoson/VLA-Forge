#!/usr/bin/env bash
set -euo pipefail

# The server's base Conda exports cuDNN 9.1, which shadows the pinned PyTorch
# wheel's cuDNN 9.7 runtime and aborts the process on its first CUDA inference.
unset LD_LIBRARY_PATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
REPORT="$ROOT/reports/vla_forge_2_transition_20261010/milestones/M01_real_libero"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project virtualenv: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$ROOT/data/libero/meta/info.json" ]]; then
  echo "Missing pinned LIBERO dataset at $ROOT/data/libero" >&2
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
mkdir -p "$HF_HOME" "$REPORT/videos"
exec > >(tee -a "$REPORT/run.log") 2>&1

"$PYTHON" "$ROOT/scripts/configure_libero.py"
"$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "lerobot/smolvla_libero",
    revision="31d453f7edd78c839a8bbc39744a292686daf0de",
    local_dir="/data/models/smolvla_libero/31d453f7edd78c839a8bbc39744a292686daf0de",
)
snapshot_download(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    local_dir="/data/models/smolvlm_full/7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    allow_patterns=[
        "config.json", "model.safetensors", "generation_config.json",
        "preprocessor_config.json", "processor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "chat_template.json", "vocab.json", "merges.txt",
    ],
)
PY

SOURCE_REVISION="${VLA_M01_SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}"
COMMON=(--root "$ROOT" --output-dir "$REPORT" --source-revision "$SOURCE_REVISION")
"$PYTHON" "$ROOT/scripts/run_m01_libero.py" --phase smoke "${COMMON[@]}"
"$PYTHON" - <<PY
import json
from pathlib import Path
m=json.loads(Path("$REPORT/metrics.json").read_text())
if m.get("phase_status") != "SMOKE_PASS":
    raise SystemExit("smoke gate did not pass; formal evaluation not started")
PY
"$PYTHON" "$ROOT/scripts/run_m01_libero.py" --phase formal "${COMMON[@]}"
