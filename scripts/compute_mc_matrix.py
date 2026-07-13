#!/usr/bin/env python3
"""
Compute one Monte Carlo realization of the multigroup Compton matrix (angle-integrated).

Headless, SLURM-friendly script. Computes the full 128x128 sigma and dsigma/dT
matrices on a 128-group geometric energy grid via MC sampling.

Usage:
  python scripts/compute_mc_matrix.py --temperature-index 0 --seed 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

ROOT = Path(__file__).resolve().parent.parent

N_GROUPS = 128
E_MIN_KEV = 1e-5  # 0.01 eV
E_MAX_KEV = 300.0

N_TEMPS = 64
T_MIN_K = 1000.0
T_MAX_K = 1e9

TEMPERATURES_K = np.geomspace(T_MIN_K, T_MAX_K, N_TEMPS)

CACHE_DIR = ROOT / "output" / "mc_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temperature-index", type=int, required=True,
        help=f"Temperature index (0..{N_TEMPS - 1})")
    parser.add_argument(
        "--seed", type=int, required=True,
        help="RNG seed for this realization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tidx = args.temperature_index
    seed = args.seed
    if not 0 <= tidx < N_TEMPS:
        raise ValueError(f"temperature-index must be in [0, {N_TEMPS - 1}], got {tidx}")

    T = TEMPERATURES_K[tidx]

    boundaries_kev = np.geomspace(E_MIN_KEV, E_MAX_KEV, N_GROUPS + 1)
    boundaries_erg = boundaries_kev * kev

    mc = cm.ComptonMonteCarloKernel(
        energy_group_boundaries=boundaries_erg.tolist(),
        weight_function=cm.PlanckWeightFunction(cap_x=300.0),
        config=cm.MCIntegrationConfig(
            num_samples=10_000_000,
            seed=seed,
            discard_out_of_grid=True,
        ),
    )

    print(f"[T-index {tidx:03d}, seed {seed}] Computing MC {N_GROUPS}x{N_GROUPS} matrix "
          f"at T = {T:.6e} K ...")

    t0 = time.time()
    sigma_matrix = np.asarray(mc.compute_sigma_matrix(T=T, Ne=1.0))
    elapsed_sigma = time.time() - t0
    print(f"  sigma_matrix done ({elapsed_sigma:.1f}s)")

    t0 = time.time()
    dsigma_dT_matrix = np.asarray(mc.compute_dsigma_dT_matrix(T=T, Ne=1.0))
    elapsed_deriv = time.time() - t0
    print(f"  dsigma_dT_matrix done ({elapsed_deriv:.1f}s)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"mc_T{tidx:03d}_seed{seed}.npz"
    np.savez_compressed(
        out_path,
        sigma_matrix=sigma_matrix,
        dsigma_dT_matrix=dsigma_dT_matrix,
        boundaries_keV=boundaries_kev,
        temperature_K=T,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
