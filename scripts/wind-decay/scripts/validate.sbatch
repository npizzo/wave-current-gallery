#!/usr/bin/env bash
#SBATCH --job-name=winddecay-valid
#SBATCH --partition=uri-gpu,gpu-highvram
#SBATCH --constraint=a100|h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/validate-%j.out
#SBATCH --error=logs/validate-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/common.sh
hostname
nvidia-smi
srun --ntasks=1 "$PYTHON" -u scripts/validate.py --root "$REGIME_ROOT"
