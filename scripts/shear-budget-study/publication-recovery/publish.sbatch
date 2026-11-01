#!/usr/bin/env bash
#SBATCH --job-name=shearbudget-republish
#SBATCH --partition=uri-cpu,cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/recovery-%j.out
#SBATCH --error=logs/recovery-%j.err
set -euo pipefail
source scripts/common.sh
"$PYTHON" -m unittest discover -s ../publication-recovery -p test_check_moments.py -q
"$PYTHON" ../publication-recovery/publish_recovery.py
