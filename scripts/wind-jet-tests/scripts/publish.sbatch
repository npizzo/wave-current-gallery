#!/usr/bin/env bash
#SBATCH --job-name=windjet-gallery
#SBATCH --partition=uri-cpu,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/common.sh
srun --ntasks=1 "$PYTHON" -u scripts/publish.py --root "$REGIME_ROOT"
