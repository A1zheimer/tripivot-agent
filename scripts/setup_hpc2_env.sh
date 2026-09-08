#!/usr/bin/env bash
# One-time HPC2 login-node setup: venv, training deps, and Qwen weights.
# Run on the login node (has internet); compute nodes then need no network.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

source /usr/local/Modules/init/bash 2>/dev/null || true
module load anaconda3

echo "[setup] Creating venv with $(python3 --version 2>&1)"
python3 -m venv .venv-hpc
PIP=".venv-hpc/bin/python -m pip"
$PIP install --upgrade pip

echo "[setup] Installing torch (PyPI linux wheel bundles CUDA 12)"
$PIP install torch

echo "[setup] Installing project training dependencies"
$PIP install -e '.[training]'

echo "[setup] Pre-downloading Qwen/Qwen2.5-7B-Instruct via hf-mirror"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
.venv-hpc/bin/python - <<'PY'
import torch
from huggingface_hub import snapshot_download
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
path = snapshot_download("Qwen/Qwen2.5-7B-Instruct")
print("MODEL_AT", path)
PY

echo SETUP_ALL_DONE
