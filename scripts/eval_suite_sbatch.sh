#!/usr/bin/env bash
#SBATCH --job-name=full-eval
#SBATCH --output=logs/full-eval-%j.out
#SBATCH --error=logs/full-eval-%j.err
#SBATCH -p emergency_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Full evaluation: 3 variants x 4 directions, aggregated into one markdown table.
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

DIRECTIONS="${DIRECTIONS:-en-zh zh-en zh-my my-zh}"
N="${N:-400}"

for tag in baseline opd grpo; do
  for d in $DIRECTIONS; do
    if [ -f "runs/eval/${tag}_${d}.json" ]; then
      echo "[suite] skip ${tag} ${d} (cached)"
      continue
    fi
    if [ "$tag" != "baseline" ] && [ ! -f "runs/${tag}-r1/final_adapter/adapter_model.safetensors" ]; then
      echo "[suite] WARN adapter missing for $tag — skipped"
      continue
    fi
    echo "[suite] eval ${tag} ${d}"
    .venv-hpc/bin/python scripts/eval_pilot.py --tag "$tag" --direction "$d" --n "$N" \
      || echo "[suite] WARN eval failed for ${tag} ${d}"
  done
done

.venv-hpc/bin/python - <<'EOF'
import glob
import json
import os

rows = {}
for path in glob.glob("runs/eval/*_*.json"):
    d = json.load(open(path))
    rows[(d["tag"], d.get("direction", "en-zh"))] = d

directions = ["en-zh", "zh-en", "zh-my", "my-zh"]
tags = ["baseline", "opd", "grpo"]
metrics = ["comet_qe", "chrfpp", "len_ratio", "degenerate_rate"]
lines = ["# 全量评测：SFT vs OPD vs GRPO（四方向）", ""]
for m in metrics:
    lines += [f"## {m}", "",
              "| direction | baseline | opd | grpo |", "|---|---|---|---|"]
    for d in directions:
        cells = []
        for t in tags:
            r = rows.get((t, d))
            cells.append(str(r.get(m)) if r else "—")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("")
out = "runs/eval/full_suite_report.md"
open(out, "w").write("\n".join(lines))
print("SUITE_DONE", out)
EOF
