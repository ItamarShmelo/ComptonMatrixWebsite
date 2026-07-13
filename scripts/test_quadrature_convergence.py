#!/usr/bin/env python3
"""
Test quadrature convergence at low temperatures.

Compares default, 2x, and 4x quadrature orders against MC mean
for the lowest temperatures to diagnose the systematic offset.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

ROOT = Path(__file__).resolve().parent.parent

N_GROUPS = 128
E_MIN_KEV = 1e-5
E_MAX_KEV = 300.0
N_SEEDS = 10

boundaries_kev = np.geomspace(E_MIN_KEV, E_MAX_KEV, N_GROUPS + 1)
boundaries_erg = boundaries_kev * kev

TEMPERATURES_K = np.geomspace(1000.0, 1e9, 64)
TEST_INDICES = [0, 2, 5, 10, 15, 20]

CONFIGS = {
    "default (48/16/24/24/12)": dict(cutoff_ratio=1e-10),
    "2x (96/32/48/48/24)": dict(
        cutoff_ratio=1e-10,
        xi_order=96, xi_tail_order=32,
        ep_edge_order=48, ep_interior_order=48,
        e_panel_order=24,
    ),
    "4x (192/64/96/96/48)": dict(
        cutoff_ratio=1e-10,
        xi_order=192, xi_tail_order=64,
        ep_edge_order=96, ep_interior_order=96,
        e_panel_order=48,
    ),
}

MC_DIR = ROOT / "output" / "mc_tables"


def load_mc_stats(tidx: int):
    mc_stack = np.array([
        np.load(MC_DIR / f"mc_T{tidx:03d}_seed{s}.npz")["sigma_matrix"]
        for s in range(N_SEEDS)
    ])
    return mc_stack.mean(axis=0), mc_stack.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)


def main() -> None:
    kernel = cds.ComptonKernelSolver()

    for tidx in TEST_INDICES:
        T = TEMPERATURES_K[tidx]
        mc_mean, mc_sem = load_mc_stats(tidx)

        print(f"=== T-index {tidx}, T = {T:.2e} K ===")

        for label, kwargs in CONFIGS.items():
            cfg = cm.MGIntegrationConfig(**kwargs)
            mg = cm.ComptonMultigroupKernel(
                energy_group_boundaries=boundaries_erg.tolist(),
                weight_function=cm.UniformWeightFunction(),
                config=cfg,
            )

            t0 = time.time()
            sigma = np.asarray(mg.compute_sigma_matrix(kernel, T=T, Ne=1.0))
            elapsed = time.time() - t0

            diff = np.abs(sigma - mc_mean)
            testable = (mc_mean != 0) & (mc_sem > 0)
            nsigma = np.zeros_like(diff)
            nsigma[testable] = diff[testable] / mc_sem[testable]
            fails = int((testable & (nsigma > 2.0)).sum())
            worst = float(nsigma[testable].max()) if testable.any() else 0

            diag_idx = 86
            diag_rel = (
                np.abs(sigma[diag_idx, diag_idx] - mc_mean[diag_idx, diag_idx])
                / np.abs(mc_mean[diag_idx, diag_idx])
                if mc_mean[diag_idx, diag_idx] != 0 else float("nan")
            )

            print(f"  {label}:")
            print(f"    Time: {elapsed:.1f}s  |  Fails: {fails}/{testable.sum()}  |  Worst: {worst:.2f} SEM  |  Diag[86,86] relErr: {diag_rel:.2e}")

        print()


if __name__ == "__main__":
    main()
