#!/usr/bin/env bash
#SBATCH --job-name=tripivot-qwen-lora
#SBATCH --output=logs/qwen-sbatch-%j.out
#SBATCH --error=logs/qwen-sbatch-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Edit these three lines for your HPC partition/module setup.
#SBATCH --partition=CHANGE_ME
# module use /path/to/your/modules
# module load python cuda

set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs

# Keep dependencies installed inside the job scratch/repo environment.
export CREATE_VENV=1
export VENV_DIR=.venv-hpc
export INSTALL_DEPENDENCIES=1

# If the cluster does not provide torch through modules, uncomment:
# export INSTALL_TORCH=1
# export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export OUTPUT_DIR=runs/qwen-lora
export EPOCHS=2

bash scripts/run_qwen_hpc.sh
