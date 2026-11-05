#!/usr/bin/env bash
# Launch a sharded script of experiment 2 as N detached processes over the GPUs.
# Usage: [SCRIPT=04_generate_counterfactuals.py] [PYTHON=<interpreter>] launch.sh <tag> [n_shards] [n_gpus] [extra args]
# Logs: $MI_CACHE/logs/exp02-<tag>-<shard>.log. Rerun to resume.
set -euo pipefail
TAG=${1:?tag, e.g. cf}
N=${2:-12}
GPUS=${3:-$(nvidia-smi -L | wc -l)}
shift $(( $# < 3 ? $# : 3 ))
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
source "$ROOT/experiments/mi/env.sh"
mkdir -p "$MI_CACHE/logs"
cd "$ROOT"
for k in $(seq 0 $((N - 1))); do
  # One visible GPU per shard: some libraries also allocate on the default
  # device, which would otherwise put every shard on GPU 0 as well.
  CUDA_VISIBLE_DEVICES=$((k % GPUS)) setsid nohup ${PYTHON:-uv run --no-sync python} \
    experiments/mi/exp02_feedback_attribution/scripts/${SCRIPT:-04_generate_counterfactuals.py} \
    --tag "$TAG" --device cuda:0 --shard "$k" --n-shards "$N" "$@" \
    > "$MI_CACHE/logs/exp02-$TAG-$k.log" 2>&1 < /dev/null &
done
echo "launched $N shards of '$TAG' on $GPUS GPUs; logs in $MI_CACHE/logs/exp02-$TAG-*.log"
