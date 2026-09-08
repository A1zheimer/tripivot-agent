#!/usr/bin/env bash
# Overnight conductor (nohup on login node): waits for assets, submits the two
# pilot branches, degrades to 1-GPU if stuck 75 min, runs unified eval, writes
# the comparison report. Everything logged to logs/orchestrator.log.
set -uo pipefail  # no -e: individual failures are handled, not fatal
cd "$(dirname "$0")/.."
source /usr/local/Modules/init/bash 2>/dev/null || true
module load anaconda3 2>/dev/null || true
export PATH="/opt/slurm/bin:$PATH"  # nohup scripts get a non-login shell without it
LOG=logs/orchestrator.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "orchestrator started"

# 1) wait for assets (max 3h)
for i in $(seq 1 90); do
  [ -f logs/ASSETS_DONE ] && break
  sleep 120
done
if [ ! -f logs/ASSETS_DONE ]; then
  say "FATAL: assets not ready after 3h; see logs/prep_assets.log"
  exit 1
fi
say "assets ready"

state_of() { squeue -h -j "$1" -o %T 2>/dev/null | head -1 | tr -d " "; }
submit() { sbatch --parsable "$@"; }

# 2) submit both branches (2-GPU, emergency)
OPD_JOB=$(submit scripts/opd_sbatch.sh)
GRPO_JOB=$(submit scripts/grpo_sbatch.sh)
say "submitted opd=$OPD_JOB grpo=$GRPO_JOB"

degrade() {  # job_id script single_gpu_env
  local jid="$1" script="$2" envvar="$3"
  local st; st=$(state_of "$jid")
  if [ "$st" = "PENDING" ]; then
    say "job $jid still pending after grace -> degrade to 1 GPU"
    scancel "$jid"
    local nj
    nj=$(sbatch --parsable -p emergency_gpu --cpus-per-task=8 --mem=48G \
         --gres=gpu:1 --time=10:00:00 --export=ALL,${envvar}=1 "$script")
    say "resubmitted $jid -> $nj (single GPU)"
    echo "$nj"
  else
    echo "$jid"
  fi
}

sleep 4500  # 75 min grace period
OPD_JOB=$(degrade "$OPD_JOB" scripts/opd_sbatch.sh OPD_SINGLE_GPU)
GRPO_JOB=$(degrade "$GRPO_JOB" scripts/grpo_sbatch.sh GRPO_SINGLE_GPU)

# 3) wait for both adapters (max 8h)
for i in $(seq 1 96); do
  opd_ok=0; grpo_ok=0
  [ -f runs/opd-r1/final_adapter/adapter_model.safetensors ] && opd_ok=1
  [ -f runs/grpo-r1/final_adapter/adapter_model.safetensors ] && grpo_ok=1
  ost=$(state_of "$OPD_JOB"); gst=$(state_of "$GRPO_JOB")
  if [ $opd_ok = 1 ] && [ $grpo_ok = 1 ]; then say "both adapters ready"; break; fi
  if [ $opd_ok = 0 ] && [ -z "$ost" ]; then say "WARN opd job left queue without adapter"; fi
  if [ $grpo_ok = 0 ] && [ -z "$gst" ]; then say "WARN grpo job left queue without adapter"; fi
  sleep 300
done

# 4) unified eval (single GPU job)
say "submitting eval job"
EVAL_JOB=$(sbatch --parsable -p emergency_gpu --cpus-per-task=8 --mem=32G \
  --gres=gpu:1 --time=03:00:00 --job-name=pilot-eval --output=logs/eval-%j.out \
  --wrap 'cd '"$PWD"' && source /usr/local/Modules/init/bash 2>/dev/null || true; module load anaconda3; export HF_HOME='"$PWD"'/.cache/huggingface HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false; .venv-hpc/bin/python scripts/eval_pilot.py --tag baseline && .venv-hpc/bin/python scripts/eval_pilot.py --tag opd && .venv-hpc/bin/python scripts/eval_pilot.py --tag grpo')
for i in $(seq 1 60); do
  st=$(state_of "$EVAL_JOB")
  [ -f runs/eval/grpo_en-zh.json ] && break
  [ -z "$st" ] && [ ! -f runs/eval/grpo_en-zh.json ] && sleep 60 && continue
  sleep 60
done

# 5) comparison report
say "writing comparison report"
{
  echo "# OPD vs GRPO 试点对比报告（en-zh）"
  echo
  echo "生成时间: $(date '+%F %T')"
  echo
  echo "| 指标 | baseline(SFT) | OPD-R1 | GRPO-R1 |"
  echo "|---|---|---|---|"
  for m in comet_qe chrfpp len_ratio degenerate_rate; do
    b=$(python3 -c "import json;print(json.load(open('runs/eval/baseline_en-zh.json')).get('$m','—'))" 2>/dev/null || echo —)
    o=$(python3 -c "import json;print(json.load(open('runs/eval/opd_en-zh.json')).get('$m','—'))" 2>/dev/null || echo —)
    g=$(python3 -c "import json;print(json.load(open('runs/eval/grpo_en-zh.json')).get('$m','—'))" 2>/dev/null || echo —)
    echo "| $m | $b | $o | $g |"
  done
  echo
  echo "训练日志: logs/opd-*.log, logs/grpo-*.log；评测明细: runs/eval/*.json"
} > runs/comparison_report.md
say "REPORT_READY runs/comparison_report.md"
