#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/output/current_verify"

COUNT=0
for TIDX in 5 15 25 35 45 55; do
  for GRID in aligned random; do
    sbatch --job-name="cv_T${TIDX}_uniform_${GRID}" \
           --partition=bigrun \
           --ntasks=1 --cpus-per-task=4 --mem=4G \
           --time=00:30:00 \
           --output="$REPO_ROOT/logs/cv_T${TIDX}_uniform_${GRID}_%j.out" \
           --wrap="export OMP_NUM_THREADS=4; $PYTHON $REPO_ROOT/scripts/verify_current_one.py --tidx $TIDX --weighting uniform --grid $GRID"
    COUNT=$((COUNT+1))
  done
done
echo "Submitted $COUNT jobs."
