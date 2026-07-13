#!/bin/bash
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Submitting 64 deterministic Compton matrix jobs ..."

for i in $(seq 0 20); do
    sbatch scripts/submit_job.sh "$i" 4
done
for i in $(seq 21 63); do
    sbatch scripts/submit_job.sh "$i"
done

echo "All 64 jobs submitted."
