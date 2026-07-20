#!/bin/bash
#SBATCH --job-name=verify_collapse
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_verify_%j.out

export OMP_NUM_THREADS=16

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
cd "$REPO_ROOT/logs"

"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" "$REPO_ROOT/scripts/compute_verify_collapse.py"
