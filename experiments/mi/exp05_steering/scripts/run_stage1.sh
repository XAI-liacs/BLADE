#!/bin/bash
# Stage 1 of paper_gecco2027/PLAN.md, on GPUs 2 and 3 only. In order of
# priority, so the chain can be stopped after any step:
#   GPU 2: llamea-stage1 (MA-BBOB: adaptive, prompt_explore, prompt_exploit
#          on the llamea-dial seeds), then shards 0-1 of llamea-stage1-seeds
#   GPU 3: llamea-stage1-{bragg,packing} (same arms, llamea-dial-<p> seeds),
#          the extra transfer-bragg samples (+ acts, report), then shards
#          2-3 of llamea-stage1-seeds
# Re-running resumes (finished runs are skipped).
#   setsid nohup bash scripts/run_stage1.sh > $MI_CACHE/logs/exp05_stage1.log 2>&1 &
set -u
cd "$(dirname "$0")/../../../.."  # BLADE root
export MI_CACHE=${MI_CACHE:-/local/steinnvan/cache}
source experiments/mi/env.sh >/dev/null 2>&1
export MLFLOW_DISABLE_AGENT_HINT=1
S=experiments/mi/exp05_steering/scripts
LOGS=$MI_CACHE/logs
PY=".venv/bin/python -X faulthandler"

llamea() {  # gpu tag shard n_shards
  CUDA_VISIBLE_DEVICES=$1 $PY $S/05_llamea.py --tag $2 --shard $3 --n-shards $4 \
    > $LOGS/exp05_$2_$3.log 2>&1
}

gpu2() {
  for k in 0 1 2 3; do llamea 2 llamea-stage1 $k 4 & done
  wait; echo "$(date -u) llamea-stage1 done"
  for k in 0 1; do llamea 2 llamea-stage1-seeds $k 4 & done
  wait; echo "$(date -u) llamea-stage1-seeds 0-1 done"
}

gpu3() {
  for k in 0 1; do
    llamea 3 llamea-stage1-bragg $k 2 &
    llamea 3 llamea-stage1-packing $k 2 &
  done
  wait; echo "$(date -u) llamea-stage1-{bragg,packing} done"
  export CUDA_VISIBLE_DEVICES=3
  $PY $S/10_transfer.py --tag transfer-bragg --phase init \
    > $LOGS/exp05_transfer-bragg-extra.log 2>&1
  $PY $S/11_transfer_analyze.py --tag transfer-bragg --stage acts \
    > $LOGS/exp05_transfer-bragg-acts2.log 2>&1
  $PY $S/11_transfer_analyze.py --tag transfer-bragg --stage report \
    > $LOGS/exp05_transfer-bragg-report2.log 2>&1
  echo "$(date -u) transfer-bragg extra samples done"
  for k in 2 3; do llamea 3 llamea-stage1-seeds $k 4 & done
  wait; echo "$(date -u) llamea-stage1-seeds 2-3 done"
}

echo "$(date -u) stage 1 start"
gpu2 &
gpu3 &
wait
echo "$(date -u) stage 1 done"
