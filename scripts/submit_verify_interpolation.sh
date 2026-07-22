#!/bin/bash
#SBATCH --job-name=verify_interp
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_verify_interp_%j.out

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
cd "$REPO_ROOT"

mkdir -p logs output

python3 "$REPO_ROOT/scripts/verify_interpolation.py"
