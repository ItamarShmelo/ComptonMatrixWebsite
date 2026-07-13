#!/bin/bash
#SBATCH --job-name=compton_mc128
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_mc_%j.out

TIDX="${1:?Usage: sbatch scripts/submit_mc_job.sh TEMPERATURE_INDEX SEED}"
SEED="${2:?Usage: sbatch scripts/submit_mc_job.sh TEMPERATURE_INDEX SEED}"

export OMP_NUM_THREADS=16

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
cd "$REPO_ROOT/logs"

"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" "$REPO_ROOT/scripts/compute_mc_matrix.py" \
    --temperature-index "$TIDX" \
    --seed "$SEED"
