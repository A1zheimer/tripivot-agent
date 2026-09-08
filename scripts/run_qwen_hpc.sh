#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qwen-lora}"
DATA_DIR="${DATA_DIR:-data/processed}"
GLOSSARY="${GLOSSARY:-$DATA_DIR/glossary.jsonl}"
DIRECTIONS="${DIRECTIONS:-zh-en en-zh zh-my my-zh}"
DOMAINS="${DOMAINS:-tech intl finance}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
GRAD_ACC="${GRAD_ACC:-4}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
DRY_RUN="${DRY_RUN:-0}"
INSTALL_DEPENDENCIES="${INSTALL_DEPENDENCIES:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
CREATE_VENV="${CREATE_VENV:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-hpc}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p logs runs "$HF_HOME"

DATA_MARKER="$DATA_DIR/zh-en.tech.jsonl"
TRAINING_ARTIFACT="artifacts/qwen_training_data_v3.tar.gz"
EXPECTED_ARTIFACT_SHA256="${EXPECTED_ARTIFACT_SHA256:-899783fe45a11d9930cded97ac211c8a0eb9b7a27ca5fda04b21b022e7985160}"
if [[ ! -s "$DATA_MARKER" && -f "$TRAINING_ARTIFACT" ]]; then
  echo "[data] Extracting $TRAINING_ARTIFACT"
  if command -v sha256sum >/dev/null 2>&1; then
    echo "$EXPECTED_ARTIFACT_SHA256  $TRAINING_ARTIFACT" | sha256sum -c -
  elif command -v shasum >/dev/null 2>&1; then
    echo "$EXPECTED_ARTIFACT_SHA256  $TRAINING_ARTIFACT" | shasum -a 256 -c -
  else
    echo "[data] WARNING: sha256sum/shasum unavailable; skipping artifact checksum" >&2
  fi
  tar -xzf "$TRAINING_ARTIFACT"
elif [[ ! -s "$DATA_MARKER" ]]; then
  echo "ERROR: Missing $DATA_MARKER and $TRAINING_ARTIFACT." >&2
  echo "Either commit/copy the artifact, or place the processed JSONL files in $DATA_DIR." >&2
  exit 1
fi

LOG_FILE="logs/qwen-training-$(date +%Y%m%d-%H%M%S).log"
echo() { builtin echo "$@" | tee -a "$LOG_FILE"; }

echo "Repository: $ROOT"
echo "Log: $LOG_FILE"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-not-set}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

if [[ "$CREATE_VENV" == "1" ]]; then
  echo "[setup] Creating Python environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv --upgrade-deps "$VENV_DIR"
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

PYTHON="$PYTHON_BIN"
if [[ "$INSTALL_DEPENDENCIES" == "1" ]]; then
  if [[ "$INSTALL_TORCH" == "1" ]]; then
    if [[ -n "$TORCH_INDEX_URL" ]]; then
      "$PYTHON" -m pip install --index-url "$TORCH_INDEX_URL" ${TORCH_SPEC:-torch}
    else
      "$PYTHON" -m pip install ${TORCH_SPEC:-torch}
    fi
  fi
  echo "[setup] Installing project training dependencies"
  "$PYTHON" -m pip install -e '.[training]'
fi

echo "[check] Python and GPU"
"$PYTHON" - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import platform
import torch
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

read -r -a DIRECTION_ARGS <<< "$DIRECTIONS"
read -r -a DOMAIN_ARGS <<< "$DOMAINS"
EXTRA_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "[run] Qwen LoRA training"
"$PYTHON" scripts/train_qwen_lora.py \
  --model-name-or-path "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --data-dir "$DATA_DIR" \
  --directions "${DIRECTION_ARGS[@]}" \
  --domains "${DOMAIN_ARGS[@]}" \
  --glossary "$GLOSSARY" \
  --num-train-epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --per-device-train-batch-size "$TRAIN_BATCH" \
  --per-device-eval-batch-size "$EVAL_BATCH" \
  --gradient-accumulation-steps "$GRAD_ACC" \
  --max-length "$MAX_LENGTH" \
  "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"

echo "[done] Adapter: $OUTPUT_DIR/final_adapter"
echo "[done] Summary: $OUTPUT_DIR/training_summary.json"
