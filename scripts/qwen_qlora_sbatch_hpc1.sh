#!/usr/bin/env bash
#SBATCH --job-name=tripivot-qlora
#SBATCH --output=logs/hpc1-qlora-%j.out
#SBATCH --error=logs/hpc1-qlora-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# HPC1 (Phase-1) A30-24GB: QLoRA 4-bit training for Qwen2.5-7B.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs

if ! type module >/dev/null 2>&1; then
  source /usr/local/Modules/init/bash 2>/dev/null \
    || source /etc/profile.d/modules.sh 2>/dev/null || true
fi
module load anaconda3 2>/dev/null || module load python 2>/dev/null || true

PY=python3
if [ -x .venv-hpc/bin/python ]; then PY=.venv-hpc/bin/python; fi
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)'; then
  echo "[setup] Installing Miniforge (system python too old)"
  MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-$(uname)-$(uname -m).sh"
  curl -fsSL "$MINIFORGE_URL" -o /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  "$HOME/miniforge3/bin/conda" create -y -p "$HOME/miniforge3/envs/training" python=3.11
  PY="$HOME/miniforge3/envs/training/bin/python"
fi

if [ ! -x .venv-hpc/bin/python ]; then
  echo "[setup] Creating venv"
  "$PY" -m venv .venv-hpc
  PY=.venv-hpc/bin/python
fi

"$PY" -m pip install --upgrade pip -q
echo "[setup] Installing training + qlora dependencies"
"$PY" -m pip install -e '.[training,qlora]'

export HF_HOME="$PWD/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "[check] GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[run] Qwen QLoRA training"
"$PY" scripts/train_qwen_lora.py \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir runs/qwen-lora-hpc1 \
  --data-dir data/processed \
  --directions zh-en en-zh zh-my my-zh \
  --domains tech intl finance \
  --glossary data/processed/glossary.jsonl \
  --num-train-epochs 2 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-length 1536 \
  --load-in-4bit 2>&1 | tee -a "logs/hpc1-qlora-train-$(date +%Y%m%d-%H%M%S).log"

echo "[done] Adapter: runs/qwen-lora-hpc1/final_adapter"
