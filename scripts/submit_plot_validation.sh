#!/bin/bash
#SBATCH --job-name=plot_valid
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/slurm_plot_validation_%j.out

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
cd "$REPO_ROOT/logs"

"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" "$REPO_ROOT/scripts/plot_validation.py"
