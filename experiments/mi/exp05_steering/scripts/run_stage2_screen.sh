#!/bin/bash
# Stage 2 (E1) single-generation screens on BBOB 20-D, one function at a
# time (default f20), on one GPU (default 2): 10_transfer.py (init +
# refine), then 11_transfer_analyze.py acts + report. Re-running resumes.
#   setsid nohup bash scripts/run_stage2_screen.sh f20 2 \
#     > $MI_CACHE/logs/exp05_stage2_screen_f20.log 2>&1 &
set -u
F=${1:-f20}
GPU=${2:-2}
cd "$(dirname "$0")/../../../.."  # BLADE root
export MI_CACHE=${MI_CACHE:-/local/steinnvan/cache}
source experiments/mi/env.sh >/dev/null 2>&1
export MLFLOW_DISABLE_AGENT_HINT=1 CUDA_VISIBLE_DEVICES=$GPU
S=experiments/mi/exp05_steering/scripts
LOGS=$MI_CACHE/logs
PY=".venv/bin/python -X faulthandler"
TAG=transfer-bbob-$F
echo "$(date -u) $TAG start (GPU $GPU)"
$PY $S/10_transfer.py --tag $TAG > $LOGS/exp05_$TAG.log 2>&1 || exit 1
$PY $S/11_transfer_analyze.py --tag $TAG --stage acts > $LOGS/exp05_$TAG-acts.log 2>&1 || exit 1
$PY $S/11_transfer_analyze.py --tag $TAG --stage report > $LOGS/exp05_$TAG-report.log 2>&1 || exit 1
echo "$(date -u) $TAG done"
