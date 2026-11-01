#!/usr/bin/env bash
#SBATCH --job-name=shearbudget-run
#SBATCH --partition=uri-gpu,gpu-highvram
#SBATCH --constraint=a100|h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/common.sh
srun --ntasks=1 "$PYTHON" scripts/launch.py --root "$REGIME_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
