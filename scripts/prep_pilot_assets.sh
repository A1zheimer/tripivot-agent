#!/usr/bin/env bash
# Pilot asset prep v2 — idempotent, fault-tolerant, merge on the CPU partition.
set -uo pipefail
cd "$(dirname "$0")/.."
source /usr/local/Modules/init/bash 2>/dev/null || true
module load anaconda3 2>/dev/null || true
LOG=logs/prep_assets.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "prep v2 start"
PY=.venv-hpc/bin/python

# 1) verify venv deps (already installed in v1 run)
if $PY -c "import vllm, peft, sacrebleu" 2>>"$LOG"; then
  say "deps ok"
else
  say "deps missing -> installing"
  $PY -m pip install "vllm==0.9.2" >>"$LOG" 2>&1 || say "WARN vllm install issue"
  $PY -m pip install "transformers==4.46.3" "peft==0.14.0" sacrebleu >>"$LOG" 2>&1 || true
fi
$PY -m pip install unbabel-comet >>"$LOG" 2>&1 || say "comet pkg optional (skipped)"

# 2) teacher weights already cached (28G verified); confirm
$PY - <<'EOF' >>"$LOG" 2>&1 || say "WARN teacher snapshot incomplete"
import os
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
snapshot_download("Qwen/Qwen2.5-14B-Instruct", local_files_only=True)
print("teacher cache ok")
EOF

# 3) merge latest SFT checkpoint -> pilot_base, on a CPU partition job
if [ -f runs/qwen-lora/pilot_base/config.json ]; then
  say "pilot_base already merged"
else
  say "submitting merge job (CPU partition)"
  MJOB=$(sbatch --parsable -p i64m512u -n 4 --mem=48G --time=00:40:00 \
    --job-name=pilot-merge --output=logs/merge-%j.out --error=logs/merge-%j.err \
    --wrap 'cd '"$PWD"' && source /usr/local/Modules/init/bash 2>/dev/null || true; module load anaconda3; .venv-hpc/bin/python - <<PYEOF
import sys, torch
sys.path.insert(0, "scripts")
from pilot_lib import latest_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ckpt = latest_checkpoint()
print("merging", ckpt, flush=True)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                         dtype=torch.float32, device_map="cpu")
m = PeftModel.from_pretrained(m, ckpt).merge_and_unload()
m.save_pretrained("runs/qwen-lora/pilot_base", safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained(
    "runs/qwen-lora/pilot_base")
print("MERGE_OK", flush=True)
PYEOF')
  say "merge job $MJOB submitted"
  for i in $(seq 1 40); do
    [ -f runs/qwen-lora/pilot_base/config.json ] && break
    st=$(squeue -h -j "$MJOB" -o %T 2>/dev/null | head -1 | tr -d " ")
    if [ -z "$st" ]; then
      if [ ! -f runs/qwen-lora/pilot_base/config.json ]; then
        say "merge job left queue without output; last log:"
        tail -5 logs/merge-*.out 2>/dev/null >> "$LOG" || true
        break
      fi
    fi
    sleep 60
  done
fi

# 4) smoke tests (data schema + prompt head)
$PY - <<'EOF' >>"$LOG" 2>&1
import sys
sys.path.insert(0, "scripts")
from pilot_lib import load_examples, load_glossary, sft_lora_kwargs
exs = load_examples("en-zh", "train", 3000, seed=13)
val = load_examples("en-zh", "validation", 400, seed=7)
print("train rows", len(exs), "val rows", len(val))
kw = sft_lora_kwargs()
print("lora r=", kw["r"], "targets[:3]=", kw["target_modules"][:3])
from translation_agent.training_data import prompt_messages
print("prompt head:", prompt_messages(exs[0], load_glossary())[0]["content"][:60])
EOF

if [ -f runs/qwen-lora/pilot_base/config.json ]; then
  say "ALL ASSETS READY"
  touch logs/ASSETS_DONE
else
  say "FATAL: pilot_base still missing"
  exit 1
fi
