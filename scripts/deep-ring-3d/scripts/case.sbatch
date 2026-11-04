#!/bin/bash
#SBATCH --job-name=dr-case
#SBATCH --partition=uri-gpu,gpu-highvram
#SBATCH --constraint=a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --no-requeue
# Usage: sbatch --array=... --job-name=<set> scripts/case.sbatch <cases-file>; line N+1 of the file = arguments for task N.
source scripts/common.sh
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1"); [ -n "$line" ] || { echo "no case $SLURM_ARRAY_TASK_ID"; exit 1; }
name=$(echo "$line" | tr ' ' '\n' | sed -n 's/^name=//p'); out="$ROOT/data/$name"
[ -e "$out/completed.txt" ] && { echo "already complete: $name"; exit 0; }
[ -e "$out" ] && mv -n "$out" "$out.failed-$(date +%Y%m%d%H%M%S)"
echo "CASE $line"; nvidia-smi -L || true
cd run && srun --ntasks=1 "$JULIA" --startup-file=no ring3d.jl $line out=$out
