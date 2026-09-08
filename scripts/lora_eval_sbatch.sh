#!/usr/bin/env bash
#SBATCH --job-name=lora-eval
#SBATCH --output=logs/lora-eval-%j.out
#SBATCH --error=logs/lora-eval-%j.err
#SBATCH -p emergency_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# LoRA effect evaluation: vanilla Qwen vs SFT LoRA across 4 directions.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
if ! type module >/dev/null 2>&1; then
  source /usr/local/Modules/init/bash 2>/dev/null || true
fi
module load anaconda3
export HF_HOME="$PWD/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

for t in raw sft; do
  for d in en-zh zh-en zh-my my-zh; do
    if [ -f "runs/eval/${t}_${d}.json" ]; then
      echo "[lora-eval] skip ${t} ${d} (cached)"
      continue
    fi
    echo "[lora-eval] running ${t} ${d}"
    .venv-hpc/bin/python scripts/eval_pilot.py --tag "$t" --direction "$d" --n 400 \
      || echo "[lora-eval] WARN failed ${t} ${d}"
  done
done
echo "LORA_EVAL_ALL_DONE"
