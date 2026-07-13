#!/bin/bash
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Submitting 640 MC validation jobs (64 temperatures x 10 seeds) ..."

for tidx in $(seq 0 63); do
    for seed in $(seq 0 9); do
        sbatch scripts/submit_mc_job.sh "$tidx" "$seed"
    done
done

echo "All 640 MC jobs submitted."
