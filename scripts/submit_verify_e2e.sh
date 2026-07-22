#!/bin/bash
#SBATCH --job-name=verify_e2e
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_verify_e2e_%j.out

export OMP_NUM_THREADS=16

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
cd "$REPO_ROOT"

mkdir -p logs output

# Start HTTP server in background serving docs/
cd "$REPO_ROOT/docs"
python3 -m http.server 8791 --bind 127.0.0.1 &
HTTP_PID=$!
cd "$REPO_ROOT"

# Wait for server to be ready
sleep 2
echo "HTTP server started (PID $HTTP_PID) on port 8791"

# Run the verification (uses ComptonMatrixExact venv for the solver)
"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" \
    "$REPO_ROOT/scripts/verify_e2e_interpolation.py"
EXIT_CODE=$?

# Kill HTTP server
kill $HTTP_PID 2>/dev/null
wait $HTTP_PID 2>/dev/null

exit $EXIT_CODE
