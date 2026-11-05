#!/usr/bin/env bash
# Analyse a finished main campaign: dataset, probes, exported probes and the
# nested incremental-value analysis (steps 2, 3, 4 and 6).
# Usage: analyse_main.sh <tag>      e.g. analyse_main.sh main-qwen3-14b
set -euo pipefail
TAG=${1:?campaign tag}
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
source "$ROOT/experiments/mi/env.sh"
cd "$ROOT"
S=experiments/mi/exp01_probing/scripts
export OMP_NUM_THREADS=${OMP_NUM_THREADS_ANALYSIS:-16}
uv run --no-sync python $S/02_build_dataset.py --tag "$TAG"
uv run --no-sync python $S/03_train_probes.py --tag "$TAG"
uv run --no-sync python $S/04_export_probes.py --tag "$TAG"
uv run --no-sync python $S/06_incremental_value.py --tag "$TAG"
