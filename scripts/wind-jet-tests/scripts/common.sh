#!/usr/bin/env bash
set -euo pipefail
export REGIME_ROOT="${REGIME_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REGIME_ROOT/campaign.json" ]] || { echo 'Campaign root is missing.'; exit 1; }
IFS= read -r JULIA < "$REGIME_ROOT/runtime/julia-path.txt"
IFS= read -r PYTHON < "$REGIME_ROOT/runtime/python-path.txt"
IFS= read -r regime_depot < "$REGIME_ROOT/runtime/depot-path.txt"
export JULIA PYTHON JULIA_PROJECT="$REGIME_ROOT/environment" JULIA_DEPOT_PATH="$regime_depot:"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" JULIA_NUM_PRECOMPILE_TASKS=4 JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POISSON_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$REGIME_ROOT/runtime/matplotlib" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$REGIME_ROOT/logs" "$MPLCONFIGDIR"
[[ -x "$JULIA" && -x "$PYTHON" ]] || { echo 'Runtime unavailable.'; exit 1; }
