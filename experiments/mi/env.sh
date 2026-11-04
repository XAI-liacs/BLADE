# Source before running MI experiments:  source experiments/mi/env.sh
# Puts every large cache (model weights, uv, pip, virtualenv app data) in one
# place. Set MI_CACHE first if $HOME is small, e.g.
#   export MI_CACHE=/local/$USER/cache
export MI_CACHE="${MI_CACHE:-$HOME/.cache/blade-mi}"
export HF_HOME="${HF_HOME:-$MI_CACHE/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$MI_CACHE/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MI_CACHE/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MI_CACHE/xdg}"
# BLADE builds a virtualenv per problem for sandboxed evaluation.
export VIRTUALENV_OVERRIDE_APP_DATA="${VIRTUALENV_OVERRIDE_APP_DATA:-$MI_CACHE/virtualenv}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$VIRTUALENV_OVERRIDE_APP_DATA"
# Optional: keep activations/BLADE logs outside the repo.
# export MI_DATA_ROOT=/local/$USER/mi-data
# One BLAS thread per evaluation subprocess. Otherwise every sandboxed
# evaluation starts one thread per core, parallel runs exhaust the thread
# limit, and the failures are wrongly logged as crashes of the generated code.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Keep idle thread pools small (Rust tokenizers, torch CPU ops). Some hosts
# cap tasks per user (e.g. systemd TasksMax=2048), and with many parallel
# generators the default one-thread-per-core pools exhaust it.
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
