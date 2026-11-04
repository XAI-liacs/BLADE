#!/usr/bin/env bash
# Launch 01_generate.py (or $SCRIPT) as N detached shards spread over the GPUs.
# Usage: [SCRIPT=05_prescreen_generate.py] launch.sh <config> [n_shards] [n_gpus]
# Logs: $MI_CACHE/logs/<config>-<shard>.log. Rerun to resume unfinished runs.
set -euo pipefail
CONFIG=${1:?config preset, e.g. main}
N=${2:-12}
GPUS=${3:-$(nvidia-smi -L | wc -l)}
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
source "$ROOT/experiments/mi/env.sh"
mkdir -p "$MI_CACHE/logs"
cd "$ROOT"
for k in $(seq 0 $((N - 1))); do
  setsid nohup uv run --no-sync python \
    experiments/mi/exp01_probing/scripts/${SCRIPT:-01_generate.py} \
    --config "$CONFIG" --device "cuda:$((k % GPUS))" --shard "$k" --n-shards "$N" \
    > "$MI_CACHE/logs/$CONFIG-$k.log" 2>&1 < /dev/null &
done
echo "launched $N shards of '$CONFIG' on $GPUS GPUs; logs in $MI_CACHE/logs/"
