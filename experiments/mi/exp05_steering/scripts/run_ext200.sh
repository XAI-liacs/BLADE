#!/bin/bash
# Budget 100 -> 200 for control and adaptive_cal on MA-BBOB (llamea-ext200,
# warm start from llamea-dial and llamea-stage1-cal). Waits until
# llamea-stage1-cal has finished, backs up its raw data, then runs 4 shards
# on GPU 2. Re-running resumes (finished runs are skipped).
#   setsid nohup bash scripts/run_ext200.sh > $MI_CACHE/logs/exp05_ext200.log 2>&1 &
set -u
cd "$(dirname "$0")/../../../.."  # BLADE root
export MI_CACHE=${MI_CACHE:-/local/steinnvan/cache}
source experiments/mi/env.sh >/dev/null 2>&1
export MLFLOW_DISABLE_AGENT_HINT=1
S=experiments/mi/exp05_steering/scripts
RAW=experiments/mi/exp05_steering/data/raw
LOGS=$MI_CACHE/logs
cal_done() {
  local n=0
  for f in $RAW/llamea-stage1-cal/shard-*/run-*/log.jsonl; do
    [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 100 ] && n=$((n + 1))
  done
  [ "$n" -ge 8 ]
}
until cal_done; do sleep 120; done
while ps aux | grep -q "[-]-tag llamea-stage1-cal "; do sleep 60; done
echo "$(date -u) llamea-stage1-cal finished"
(cd experiments/mi/exp05_steering/data && zip -qr -9 \
  $MI_CACHE/backups/exp05_raw_llamea-stage1-cal_$(date -u +%F).zip raw/llamea-stage1-cal)
echo "$(date -u) llamea-ext200 start"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python -X faulthandler $S/05_llamea.py \
    --tag llamea-ext200 --shard $k --n-shards 4 > $LOGS/exp05_llamea-ext200_$k.log 2>&1 &
done
wait
echo "$(date -u) llamea-ext200 done"
