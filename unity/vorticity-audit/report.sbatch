#!/usr/bin/env bash
#SBATCH --job-name=packet-vort-report
#SBATCH --partition=uri-cpu,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
audit_state=${1:?}; campaign=${2:?}; gallery=${3:?}
source "$campaign/scripts/common.sh"
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 "$PYTHON" "$audit_state/code/publish.py" "$audit_state" "$campaign" "$gallery"
