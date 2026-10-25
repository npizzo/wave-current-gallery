#!/usr/bin/env bash
#SBATCH --job-name=wave-gallery
#SBATCH --partition=uri-cpu,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
gallery_root=${1:?Gallery checkout required}
campaign_root=${2:?Campaign root required}
repository=${3:?GitHub owner/name required}
cd "$gallery_root"
export JULIA_DEPOT_PATH="$campaign_root/depot"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 "$campaign_root/runtime/python-env/bin/python3" scripts/publish_update.py \
  --campaign "$campaign_root" --repository "$repository" \
  --key "$gallery_root/.unity-publisher/id_ed25519" \
  --known-hosts "$gallery_root/.unity-publisher/known_hosts"

