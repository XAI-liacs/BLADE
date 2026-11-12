#!/bin/bash
# Step 5d chain: after the llamea-dial campaign (PIDs as arguments) has
# finished, per problem on its own GPU (bragg: 2, packing: 3): transfer
# screen, probe readout + report, LLaMEA smoke, full LLaMEA campaign.
#   setsid nohup bash scripts/run_5d.sh <pid>... > $MI_CACHE/logs/exp05_5d.log 2>&1 &
set -u
cd "$(dirname "$0")/../../../.."  # BLADE root
source experiments/mi/env.sh >/dev/null 2>&1
export MLFLOW_DISABLE_AGENT_HINT=1
S=experiments/mi/exp05_steering/scripts
LOGS=${MI_CACHE:-/local/steinnvan/cache}/logs
PY=".venv/bin/python -X faulthandler"

for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
echo "$(date -u) campaign done, starting 5d"

run() {  # gpu tag-suffix
  local gpu=$1 name=$2
  export CUDA_VISIBLE_DEVICES=$gpu
  $PY $S/10_transfer.py --tag transfer-$name > $LOGS/exp05_transfer-$name.log 2>&1
  echo "EXIT $?" >> $LOGS/exp05_transfer-$name.log
  $PY $S/11_transfer_analyze.py --tag transfer-$name --stage acts \
    > $LOGS/exp05_transfer-$name-acts.log 2>&1
  $PY $S/11_transfer_analyze.py --tag transfer-$name --stage report \
    > $LOGS/exp05_transfer-$name-report.log 2>&1
  echo "$(date -u) $name screen done"
  $PY $S/05_llamea.py --tag llamea-dial-$name-smoke \
    > $LOGS/exp05_llamea-dial-$name-smoke.log 2>&1
  if ! grep -q "shard 0: done" $LOGS/exp05_llamea-dial-$name-smoke.log; then
    echo "$(date -u) $name LLaMEA smoke failed"
    return 1
  fi
  for k in 0 1 2; do
    $PY $S/05_llamea.py --tag llamea-dial-$name --shard $k --n-shards 3 \
      > $LOGS/exp05_llamea-dial-${name}_$k.log 2>&1 &
  done
  wait
  echo "$(date -u) $name LLaMEA done"
}

run 2 bragg &
run 3 packing &
wait
echo "$(date -u) 5d chain done"
