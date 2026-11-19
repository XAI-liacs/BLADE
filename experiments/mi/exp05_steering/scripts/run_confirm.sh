#!/bin/bash
# Review follow-up (paper_gecco2027/PREREG_confirm.md): llamea-confirm,
# control / adaptive / adaptive_prompt / temp11 on seeds 316-339, budget
# 200, MA-BBOB. Starts the shards one by one, each on an allowed GPU
# (default "2 3") as soon as one has >= 25 GB free. Re-running resumes
# (finished runs are skipped).
#   setsid nohup bash scripts/run_confirm.sh \
#     > $MI_CACHE/logs/exp05_confirm.log 2>&1 &
set -u
GPUS=${1:-2 3}
cd "$(dirname "$0")/../../../.."  # BLADE root
export MI_CACHE=${MI_CACHE:-/local/steinnvan/cache}
source experiments/mi/env.sh >/dev/null 2>&1
export MLFLOW_DISABLE_AGENT_HINT=1
S=experiments/mi/exp05_steering/scripts
JOBS="llamea-confirm:6"
free_gpu() {  # prints an allowed GPU with >= 25 GB free, or nothing
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v ok=" $GPUS " 'index(ok, " " $1 " ") && $2 >= 25000 {print $1; exit}'
}
for job in $JOBS; do
  TAG=${job%%:*}
  N=${job##*:}
  for k in $(seq 0 $((N - 1))); do
    until g=$(free_gpu) && [ -n "$g" ]; do sleep 120; done
    echo "$(date -u) $TAG shard $k on GPU $g"
    CUDA_VISIBLE_DEVICES=$g .venv/bin/python -X faulthandler $S/05_llamea.py \
      --tag $TAG --shard $k --n-shards $N > $MI_CACHE/logs/exp05_${TAG}_$k.log 2>&1 &
    sleep 180  # let the model load before measuring free memory again
  done
done
wait
echo "$(date -u) confirmation campaign done"
