#!/bin/bash
#SBATCH --job-name=lj-dispatch
#SBATCH --partition=uri-cpu,cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/dispatch-%A_%a.out
#SBATCH --error=logs/dispatch-%A_%a.err
#SBATCH --no-requeue
source "${SLURM_SUBMIT_DIR}/scripts/common.sh"
srun --ntasks=1 "$PYTHON" scripts/ops.py submit
