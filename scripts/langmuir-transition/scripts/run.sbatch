#!/bin/bash
#SBATCH --job-name=lj-transition
#SBATCH --partition=uri-gpu,gpu-highvram
#SBATCH --constraint=a100
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/run-%A_%a.out
#SBATCH --error=logs/run-%A_%a.err
#SBATCH --no-requeue
source "${SLURM_SUBMIT_DIR}/scripts/common.sh"
srun --ntasks=1 "$PYTHON" scripts/ops.py run "$SLURM_ARRAY_TASK_ID"
