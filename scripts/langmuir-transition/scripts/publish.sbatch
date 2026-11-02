#!/bin/bash
#SBATCH --job-name=lj-publish
#SBATCH --partition=uri-cpu,cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/publish-%A_%a.out
#SBATCH --error=logs/publish-%A_%a.err
#SBATCH --no-requeue
source "${SLURM_SUBMIT_DIR}/scripts/common.sh"
srun --ntasks=1 "$PYTHON" scripts/publish.py
