#!/usr/bin/env bash
#SBATCH --job-name=shearpilot-submit
#SBATCH --partition=uri-cpu,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/dispatch-%j.out
#SBATCH --error=logs/dispatch-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/common.sh
srun --ntasks=1 "$PYTHON" -u scripts/submit.py --root "$REGIME_ROOT" --concurrency 2
