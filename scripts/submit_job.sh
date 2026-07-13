#!/bin/bash
#SBATCH --job-name=compton_mg128
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_det_%j.out

TIDX="${1:?Usage: sbatch scripts/submit_job.sh TEMPERATURE_INDEX [QUADRATURE_SCALE]}"
QSCALE="${2:-1}"

export OMP_NUM_THREADS=16

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
cd "$REPO_ROOT/logs"

"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" "$REPO_ROOT/scripts/compute_matrix.py" \
    --temperature-index "$TIDX" \
    --quadrature-scale "$QSCALE"
