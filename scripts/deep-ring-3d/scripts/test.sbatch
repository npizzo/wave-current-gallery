#!/bin/bash
#SBATCH --job-name=dr-test
#SBATCH --partition=uri-cpu,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/test-%j.out
#SBATCH --no-requeue
source scripts/common.sh
cd run && srun --ntasks=1 "$JULIA" --startup-file=no test_ring.jl
