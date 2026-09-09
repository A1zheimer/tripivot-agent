#!/usr/bin/env bash
#SBATCH --job-name=doc-agent-eval
#SBATCH --output=logs/docagent-%j.out
#SBATCH --error=logs/docagent-%j.err
#SBATCH -p emergency_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# P0-3: document-level A/B/C via offline LLM (no server).
#  A: vanilla Qwen + agent machinery   B: SFT + agent machinery
#  C: SFT + plain per-chunk calls (no glossary / no audit-retry)
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
if ! type module >/dev/null 2>&1; then
  source /usr/local/Modules/init/bash 2>/dev/null || true
fi
module load anaconda3
export HF_HOME="$PWD/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTD=runs/doc-agent
mkdir -p "$OUTD"

echo "[doc] group C: SFT + bypass"
.venv-hpc/bin/python scripts/agent_doc_run.py --input demo/doc_agent_eval.md \
  --output "$OUTD/C.md" --model runs/qwen-lora/pilot_base --agent 0

echo "[doc] group B: SFT + agent machinery"
.venv-hpc/bin/python scripts/agent_doc_run.py --input demo/doc_agent_eval.md \
  --output "$OUTD/B.md" --model runs/qwen-lora/pilot_base --agent 1

echo "[doc] group A: vanilla Qwen + agent machinery"
.venv-hpc/bin/python scripts/agent_doc_run.py --input demo/doc_agent_eval.md \
  --output "$OUTD/A.md" --model Qwen/Qwen2.5-7B-Instruct --agent 1

.venv-hpc/bin/python scripts/eval_document.py --dir "$OUTD"
echo "DOC_AGENT_EVAL_DONE"
