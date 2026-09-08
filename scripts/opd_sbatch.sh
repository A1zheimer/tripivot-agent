#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot
#SBATCH --output=logs/opd-%j.out
#SBATCH --error=logs/opd-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=emergency_gpu

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs runs
if ! type module >/dev/null 2>&1; then
  source /usr/local/Modules/init/bash 2>/dev/null \
    || source /etc/profile.d/modules.sh 2>/dev/null || true
fi
module load anaconda3

export HF_HOME="$PWD/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_XET_HIGH_PERFORMANCE=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[env] GPUs:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-hpc/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
.venv-hpc/bin/python scripts/opd_pilot.py 2>&1 | tee -a "logs/opd-train-$(date +%Y%m%d-%H%M%S).log"
