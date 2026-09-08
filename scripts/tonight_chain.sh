#!/usr/bin/env bash
# Tonight's autonomous chain: smoke pass -> relaunch orchestrator (branches
# -> degrade -> pilot eval -> comparison report) + LoRA eval. Run via nohup.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/slurm/bin:$PATH"
LOG=logs/tonight_chain.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "chain started"

# 1) wait for either smoke job to pass (max 4h)
smoke_ok=0
for i in $(seq 1 120); do
  for f in logs/smoke-12668431.out logs/smoke-long-12668866.out; do
    if grep -q SMOKE_ALL_OK "$f" 2>/dev/null; then smoke_ok=1; fi
  done
  [ "$smoke_ok" = 1 ] && break
  sleep 120
done
if [ "$smoke_ok" = 0 ]; then
  say "FATAL: smoke not passed within 4h"
  exit 1
fi
say "smoke PASSED — cancelling smoke jobs"
scancel 12668431 12668866 2>/dev/null || true

# 2) fresh orchestrator (branches -> degrade -> eval -> report)
pkill -f "overnight_orch[e]strator" 2>/dev/null || true
sleep 2
nohup bash scripts/overnight_orchestrator.sh >> logs/orchestrator.log 2>&1 &
say "orchestrator relaunched (pid $!)"

# 3) LoRA effect eval
EJ=$(sbatch --parsable scripts/lora_eval_sbatch.sh)
say "lora-eval submitted: $EJ"
say "CHAIN_HEAD_DONE — orchestrator owns the rest"
