#!/usr/bin/env bash
#SBATCH --job-name=tripivot-qwen-lora
#SBATCH --output=logs/qwen-sbatch-%j.out
#SBATCH --error=logs/qwen-sbatch-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# HKUST(GZ) HPC2: shared A800-80GB queue (up to 128 cores / 16 GPUs, 7-day limit).
#SBATCH --partition=i64m1tga800u
#SBATCH --mem=64G
if ! type module >/dev/null 2>&1; then
  source /usr/local/Modules/init/bash 2>/dev/null \
    || source /etc/profile.d/modules.sh 2>/dev/null || true
fi
module purge
module load anaconda3

set -Eeuo pipefail
# Slurm executes the script from its spool copy; $0 is not the repo path here.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs runs

# Keep dependencies installed inside the job scratch/repo environment.
export CREATE_VENV="${CREATE_VENV:-1}"
export VENV_DIR=.venv-hpc
export INSTALL_DEPENDENCIES="${INSTALL_DEPENDENCIES:-1}"

# The [training] extra does not include torch. Cluster drivers support CUDA <= 12.8,
# and the newest PyPI wheel (cu130) cannot initialize CUDA there — pin a cu126 build.
export INSTALL_TORCH="${INSTALL_TORCH:-1}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
export TORCH_SPEC="${TORCH_SPEC:-torch==2.7.1}"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/qwen-lora}"
export EPOCHS="${EPOCHS:-2}"
# HuggingFace direct is unreachable from the cluster; use the mirror.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# The xet CDN bypasses the mirror and fails with 401; force plain HTTP downloads.
export HF_HUB_DISABLE_XET=1
export HF_XET_HIGH_PERFORMANCE=0

bash scripts/run_qwen_hpc.sh
