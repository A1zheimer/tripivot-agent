#!/usr/bin/env bash
# Prepare demo assets after pilot adapters exist (run on the Mac, models
# pulled from the cluster first):
#   rsync -a hpc2:tripivot-agent/runs/{qwen-lora,opd-r1,grpo-r1} ../runs/
# Then this script merges + converts each variant to MLX 4bit and builds the
# offline cache used by the demo fallback backend.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PILOT_BASE="runs/qwen-lora/pilot_base"
mkdir -p demo/models

python3 - <<'EOF'
import json, os, sys
os.makedirs("demo/models", exist_ok=True)

VARIANTS = {
    "baseline": None,                      # pilot_base itself
    "opd": "runs/opd-r1/final_adapter",
    "grpo": "runs/grpo-r1/final_adapter",
}

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

for name, adapter in VARIANTS.items():
    out = f"demo/models/{name}"
    if os.path.exists(os.path.join(out, "config.json")):
        print(f"[demo] {name} already merged, skip")
        continue
    print(f"[demo] merging {name}")
    m = AutoModelForCausalLM.from_pretrained(
        "runs/qwen-lora/pilot_base", torch_dtype=torch.float32, device_map="cpu")
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.save_pretrained(out + "_bf16", safe_serialization=True)
    AutoTokenizer.from_pretrained("runs/qwen-lora/pilot_base").save_pretrained(out + "_bf16")
EOF

for v in baseline opd grpo; do
  [ -d "demo/models/${v}_bf16" ] || { echo "missing demo/models/${v}_bf16"; exit 1; }
  python3 -m mlx_lm.convert --hf-path "demo/models/${v}_bf16" \
    --mlx-path "demo/models/${v}" --quantize -q 2>/dev/null \
    || python3 -m mlx_lm.convert --hf-path "demo/models/${v}_bf16" --mlx-path "demo/models/${v}"
  rm -rf "demo/models/${v}_bf16"
done

python3 - <<'EOF'
# Build the offline cache from the same examples eval used (greedy outputs
# recorded during eval_pilot runs if present, otherwise sample now).
import json, os

cache = {"items": []}
src = "runs/eval/samples.json"
if os.path.exists(src):
    cache = json.load(open(src))
else:
    from pilot_examples import build  # noqa: F401  (placeholder, filled tomorrow)
json.dump(cache, open("demo/cache.json", "w"), ensure_ascii=False, indent=1)
print("[demo] cache.json ready:", len(cache["items"]), "items")
EOF
echo "[demo] done. Run: python demo/server.py"
