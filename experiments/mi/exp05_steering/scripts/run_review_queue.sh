#!/bin/bash
# Review follow-ups #5 and #6, queued after the confirmation campaign
# (run_confirm.sh). Each job waits for its own pids (the confirmation
# shards on its GPU, or the run_confirm.sh launcher), so the two jobs can
# start at different times:
#   GPU A: #5, the failure pre-screen inside LLaMEA on Bragg
#          (17_prescreen_loop.py, 12 seeds x shadow/screened, 3 shards);
#   GPU B: #6, the exploration dial on Qwen3-14B (EXP05_MODEL=qwen3-14b):
#          corpus activations, contrastive directions, the dial-qwen3-14b
#          screen (2 shards) and its analysis.
# Re-running resumes where the scripts support it.
#   setsid nohup bash scripts/run_review_queue.sh "<pids A>" "<pids B>" \
#     > $MI_CACHE/logs/exp05_review_queue.log 2>&1 &
set -u
WAIT_A=${1:-}
WAIT_B=${2:-}
GPU_A=${3:-2}
GPU_B=${4:-3}
cd "$(dirname "$0")/../../../.."  # BLADE root
export MI_CACHE=${MI_CACHE:-/local/steinnvan/cache}
source experiments/mi/env.sh >/dev/null 2>&1
export MLFLOW_DISABLE_AGENT_HINT=1
S=experiments/mi/exp05_steering/scripts
PY=".venv/bin/python -X faulthandler"
L=$MI_CACHE/logs
wait_for() {
  for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 60; done; done
}

prescreen_bragg() {
  wait_for $WAIT_A
  echo "$(date -u) #5 starts on GPU $GPU_A"
  for k in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPU_A $PY $S/17_prescreen_loop.py --shard $k --n-shards 3 \
      > $L/exp05_prescreen_bragg_$k.log 2>&1 &
    sleep 120
  done
  wait
  $PY $S/17_prescreen_loop.py --analyze > $L/exp05_prescreen_bragg_analyze.log 2>&1
  echo "$(date -u) #5 prescreen-bragg done"
}

dial_14b() {
  wait_for $WAIT_B
  echo "$(date -u) #6 starts on GPU $GPU_B"
  export EXP05_MODEL=qwen3-14b
  set -e
  D=experiments/mi/exp05_steering/data/processed
  if [ ! -f $D/corpus-qwen3-14b/directions.npz ]; then  # done in the check run?
    CUDA_VISIBLE_DEVICES=$GPU_B $PY $S/08_contrast.py --stage acts --device cuda:0 \
      > $L/exp05_contrast14b_acts.log 2>&1
    $PY $S/08_contrast.py --stage directions > $L/exp05_contrast14b_dirs.log 2>&1
  fi
  $PY $S/01_prepare.py --tag dial-qwen3-14b > $L/exp05_dial14b_prepare.log 2>&1
  rm -rf experiments/mi/exp05_steering/data/raw/dial-qwen3-14b  # check run
  set +e
  for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$GPU_B $PY $S/02_generate.py --tag dial-qwen3-14b \
      --shard $k --n-shards 2 > $L/exp05_dial14b_$k.log 2>&1 &
    sleep 180
  done
  wait
  $PY $S/03_analyze.py --tag dial-qwen3-14b > $L/exp05_dial14b_analyze.log 2>&1
  echo "$(date -u) #6 dial-qwen3-14b done"
}

prescreen_bragg &
dial_14b &
wait
echo "$(date -u) review queue done"
