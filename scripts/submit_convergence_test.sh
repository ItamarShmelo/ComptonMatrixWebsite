#!/bin/bash
#SBATCH --job-name=quad_conv_test
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/quad_convergence_test_%j.out

export OMP_NUM_THREADS=16

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
cd "$REPO_ROOT/logs"

"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" "$REPO_ROOT/scripts/test_quadrature_convergence.py"
