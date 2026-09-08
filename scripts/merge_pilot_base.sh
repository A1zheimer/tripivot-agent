#!/usr/bin/env bash
#SBATCH --job-name=pilot-merge
#SBATCH --output=logs/merge-%j.out
#SBATCH --error=logs/merge-%j.err
#SBATCH -p i64m512u
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=00:40:00

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
if ! type module >/dev/null 2>&1; then
  source /usr/local/Modules/init/bash 2>/dev/null || true
fi
module load anaconda3

export HF_HOME="$PWD/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

.venv-hpc/bin/python - <<'PYEOF'
import sys
import torch
sys.path.insert(0, "scripts")
from pilot_lib import latest_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ckpt = latest_checkpoint()
print("merging", ckpt, flush=True)
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float32, device_map="cpu")
m = PeftModel.from_pretrained(m, ckpt).merge_and_unload()
m.save_pretrained("runs/qwen-lora/pilot_base", safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained(
    "runs/qwen-lora/pilot_base")
print("MERGE_OK", flush=True)
PYEOF
