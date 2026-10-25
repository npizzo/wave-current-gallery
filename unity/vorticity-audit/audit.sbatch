#!/usr/bin/env bash
#SBATCH --job-name=packet-vorticity
#SBATCH --partition=uri-cpu,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-6%2
#SBATCH --no-requeue
set -euo pipefail
audit_state=${1:?}; campaign=${2:?}; gallery=${3:?}
source "$campaign/scripts/common.sh"
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 "$PYTHON" "$audit_state/code/launch.py" "$audit_state" "$campaign" "$SLURM_ARRAY_TASK_ID"
