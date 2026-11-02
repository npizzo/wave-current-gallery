#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CAMPAIGN_ROOT="$PWD"
OLD=/work/pi_nicholas_pizzo_uri_edu/unity-packet-reproduction
export JULIA="$OLD/runtime/julia-1.11.9/bin/julia"
export PYTHON="$OLD/runtime/python-env/bin/python3"
export JULIA_PROJECT="$CAMPAIGN_ROOT/upstream"
export JULIA_DEPOT_PATH="$CAMPAIGN_ROOT/depot:$OLD/depot:"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" JULIA_NUM_PRECOMPILE_TASKS=4 JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POISSON_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$CAMPAIGN_ROOT/runtime/matplotlib" PYTHONDONTWRITEBYTECODE=1
mkdir -p runtime/matplotlib depot
