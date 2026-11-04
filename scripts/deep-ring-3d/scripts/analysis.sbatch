#!/bin/bash
#SBATCH --job-name=dr-analysis
#SBATCH --partition=uri-cpu,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analysis-%j.out
#SBATCH --no-requeue
# Usage: sbatch scripts/analysis.sbatch analysis/<script>.py   (all analysis runs here, not on the PI's computer)
source scripts/common.sh
srun --ntasks=1 "$PYTHON" "$@"
