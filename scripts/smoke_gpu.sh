#!/usr/bin/env bash
#SBATCH --job-name=smoke-gpu
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
#SBATCH -p emergency_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

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

.venv-hpc/bin/python - <<'EOF'
import os
import sys
import torch
sys.path.insert(0, "scripts")
from pilot_lib import sft_lora_kwargs, load_examples, load_glossary, build_prompts, score_chrf

PILOT_BASE = "runs/qwen-lora/pilot_base"

print("[1] cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
assert torch.cuda.is_available()

# 2) transformers + peft runtime (the branch training path)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
tok = AutoTokenizer.from_pretrained(PILOT_BASE)
model = AutoModelForCausalLM.from_pretrained(
    PILOT_BASE, torch_dtype=torch.bfloat16, device_map="cuda:0")
model = get_peft_model(model, LoraConfig(**sft_lora_kwargs()))
model.print_trainable_parameters()
print("[2] peft wrap OK")

# 3) simulate the branch merge step (adapter -> merged dir WITH tokenizer)
from peft import PeftModel
adapter_tmp = "runs/opd-r1/_adapter_tmp"
merged_tmp = "runs/opd-r1/_merged_tmp_smoke"
if os.path.isdir(adapter_tmp):
    mc = AutoModelForCausalLM.from_pretrained(
        PILOT_BASE, torch_dtype=torch.bfloat16, device_map="cpu")
    pmc = PeftModel.from_pretrained(mc, adapter_tmp).merge_and_unload()
    pmc.save_pretrained(merged_tmp, safe_serialization=True)
    tok.save_pretrained(merged_tmp)
    del pmc, mc
    smoke_model = merged_tmp
    print("[3a] adapter merge + tokenizer save OK")
else:
    smoke_model = PILOT_BASE
    print("[3a] no adapter tmp; testing base dir")

# 4) vLLM engine init + real generation (the branch sampling path)
from vllm import LLM, SamplingParams
llm = LLM(model=smoke_model, dtype="bfloat16", max_model_len=1536,
          gpu_memory_utilization=0.4)
# tiny CPU merge simulation skipped; direct generate on base
exs = load_examples("en-zh", "train", 2, seed=13)
prompts = build_prompts(tok, exs, load_glossary())
outs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=64))
hyp = outs[0].outputs[0].text.strip()
print("[3] vLLM gen OK:", hyp[:60])
del llm
torch.cuda.empty_cache()

# 4) reward path (chrf primary)
refs = [exs[0]["target_text"]]
r = score_chrf([hyp], refs)
print("[4] chrf reward OK:", round(r[0], 3))

# 5) teacher load (OPD branch dependency, brief)
t = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda:0")
print("[5] teacher load OK:", sum(p.numel() for p in t.parameters()) // 10**9, "B params")
print("SMOKE_ALL_OK")
EOF
