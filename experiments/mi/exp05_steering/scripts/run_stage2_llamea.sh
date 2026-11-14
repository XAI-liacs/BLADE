#!/bin/bash
# Stage 2 (E1) LLaMEA campaign on one BBOB function (default f20): 5
# shards of 2 seeds (12 runs each). Shards start on GPU 2 or 3 (only
# those) as soon as one has room (>= 25 GB free), one at a time.
# Re-running resumes (finished runs are skipped).
#   setsid nohup bash scripts/run_stage2_llamea.sh f20 \
#     > $MI_CACHE/logs/exp05_stage2_llamea_f20.log 2>&1 &
set -u
F=${1:-f20}
cd "$(dirname "$0")/../../../.."  # BLADE root
export MI_CACHE=${MI_CACHE:-/local/steinnvan/cache}
source experiments/mi/env.sh >/dev/null 2>&1
export MLFLOW_DISABLE_AGENT_HINT=1
S=experiments/mi/exp05_steering/scripts
TAG=llamea-bbob-$F
N=5
free_gpu() {  # prints a GPU (2 or 3) with >= 25 GB free, or nothing
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' '($1==2 || $1==3) && $2 >= 25000 {print $1; exit}'
}
for k in $(seq 0 $((N - 1))); do
  until g=$(free_gpu) && [ -n "$g" ]; do sleep 120; done
  echo "$(date -u) $TAG shard $k on GPU $g"
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python -X faulthandler $S/05_llamea.py \
    --tag $TAG --shard $k --n-shards $N > $MI_CACHE/logs/exp05_${TAG}_$k.log 2>&1 &
  sleep 180  # let the model load before measuring free memory again
done
wait
echo "$(date -u) $TAG done"
