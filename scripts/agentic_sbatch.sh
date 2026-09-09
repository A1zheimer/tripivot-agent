#!/usr/bin/env bash
#SBATCH --job-name=agentic-v0
#SBATCH --output=logs/agentic-%j.out
#SBATCH --error=logs/agentic-%j.err
#SBATCH -p emergency_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
if ! type module >/dev/null 2>&1; then
  source /usr/local/Modules/init/bash 2>/dev/null || true
fi
module load anaconda3
export HF_HOME="$PWD/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_XET_HIGH_PERFORMANCE=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-hpc/bin/python scripts/agentic_grpo_v0.py 2>&1 | tee -a "logs/agentic-train-$(date +%Y%m%d-%H%M%S).log"
