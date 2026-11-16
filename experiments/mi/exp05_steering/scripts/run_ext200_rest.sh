#!/bin/bash
# Budget 100 -> 200 for the remaining MA-BBOB arms and seeds:
# llamea-ext200-rest (explore, exploit, prompt arms; seeds 300-307) and
# llamea-ext200-seeds (all six arms; seeds 308-315), warm-started from the
# finished 100-evaluation runs. Waits until the given pids (the f2
# campaign) have exited, backs up the source raw data, then starts 10
# shards (5 per campaign), each on an allowed GPU (default "2 3") as soon
# as one has >= 25 GB free. Re-running resumes (finished runs are skipped).
#   setsid nohup bash scripts/run_ext200_rest.sh "<pids>" \
#     > $MI_CACHE/logs/exp05_ext200_rest.log 2>&1 &
set -u
WAIT_PIDS=${1:-}
GPUS=${2:-2 3}
cd "$(dirname "$0")/../../../.."  # BLADE root
export MI_CACHE=${MI_CACHE:-/local/steinnvan/cache}
source experiments/mi/env.sh >/dev/null 2>&1
export MLFLOW_DISABLE_AGENT_HINT=1
S=experiments/mi/exp05_steering/scripts
N=5
free_gpu() {  # prints an allowed GPU with >= 25 GB free, or nothing
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v ok=" $GPUS " 'index(ok, " " $1 " ") && $2 >= 25000 {print $1; exit}'
}
for p in $WAIT_PIDS; do while kill -0 "$p" 2>/dev/null; do sleep 60; done; done
echo "$(date -u) waited pids exited"
(cd experiments/mi/exp05_steering/data && zip -qr -9 \
  $MI_CACHE/backups/exp05_raw_llamea-dial-stage1-seeds_$(date -u +%F).zip \
  raw/llamea-dial raw/llamea-stage1 raw/llamea-stage1-seeds)
echo "$(date -u) backup done"
for TAG in llamea-ext200-seeds llamea-ext200-rest; do
  for k in $(seq 0 $((N - 1))); do
    until g=$(free_gpu) && [ -n "$g" ]; do sleep 120; done
    echo "$(date -u) $TAG shard $k on GPU $g"
    CUDA_VISIBLE_DEVICES=$g .venv/bin/python -X faulthandler $S/05_llamea.py \
      --tag $TAG --shard $k --n-shards $N > $MI_CACHE/logs/exp05_${TAG}_$k.log 2>&1 &
    sleep 180  # let the model load before measuring free memory again
  done
done
wait
echo "$(date -u) llamea-ext200-rest and -seeds done"
