#!/usr/bin/env python3
"""
Validate deterministic Compton matrices against Monte Carlo realizations.

For each temperature, loads the deterministic result and 10 MC runs,
computes the MC mean and standard error of the mean (SEM) per element,
and flags any element where the deterministic value falls outside
2 SEM of the MC mean.

Usage:
  python scripts/validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

N_TEMPS = 64
N_SEEDS = 10

DET_DIR = ROOT / "output" / "tables"
MC_DIR = ROOT / "output" / "mc_tables"


def find_det_file(tidx: int) -> Path:
    pattern = f"T{tidx:03d}_*.npz"
    matches = sorted(DET_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No deterministic file for T-index {tidx}: {DET_DIR / pattern}")
    return matches[0]


def validate_temperature(tidx: int) -> dict:
    det_path = find_det_file(tidx)
    det_data = np.load(det_path)
    det_sigma = det_data["sigma_matrix"]
    T = float(det_data["temperature_K"])

    mc_sigmas = []
    for seed in range(N_SEEDS):
        mc_path = MC_DIR / f"mc_T{tidx:03d}_seed{seed}.npz"
        if not mc_path.exists():
            raise FileNotFoundError(f"Missing MC file: {mc_path}")
        mc_sigmas.append(np.load(mc_path)["sigma_matrix"])

    mc_stack = np.array(mc_sigmas)  # (N_SEEDS, G, G)
    mc_mean = mc_stack.mean(axis=0)
    mc_sem = mc_stack.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)

    diff = np.abs(det_sigma - mc_mean)

    mc_nonzero = mc_mean != 0
    testable = mc_nonzero & (mc_sem > 0)

    flagged = np.zeros_like(diff, dtype=bool)
    flagged[testable] = diff[testable] > 2.0 * mc_sem[testable]

    n_flagged = int(flagged.sum())
    n_tested = int(testable.sum())

    deviation_in_sem = np.zeros_like(diff)
    deviation_in_sem[testable] = diff[testable] / mc_sem[testable]
    worst_sem = float(deviation_in_sem.max()) if n_flagged > 0 else 0.0

    return {
        "tidx": tidx,
        "T_K": T,
        "n_flagged": n_flagged,
        "n_tested": n_tested,
        "worst_sem": worst_sem,
    }


def main() -> None:
    print(f"{'T-idx':>5s}  {'T [K]':>12s}  {'Flagged':>10s}  {'Tested':>10s}  {'Worst (SEM)':>12s}  Status")
    print("-" * 72)

    any_failed = False
    for tidx in range(N_TEMPS):
        try:
            result = validate_temperature(tidx)
        except FileNotFoundError as e:
            print(f"{tidx:5d}  {'---':>12s}  {'---':>10s}  {'---':>10s}  {'---':>12s}  MISSING: {e}")
            any_failed = True
            continue

        status = "PASS" if result["n_flagged"] == 0 else "FAIL"
        if result["n_flagged"] > 0:
            any_failed = True

        worst_str = f"{result['worst_sem']:.2f}" if np.isfinite(result["worst_sem"]) else "inf"
        print(f"{result['tidx']:5d}  {result['T_K']:12.6e}  {result['n_flagged']:10d}  "
              f"{result['n_tested']:10d}  {worst_str:>12s}  {status}")

    print("-" * 72)
    if any_failed:
        print("VALIDATION FAILED: some temperatures have flagged elements.")
        sys.exit(1)
    else:
        print("VALIDATION PASSED: all deterministic values within 2 SEM of MC mean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
