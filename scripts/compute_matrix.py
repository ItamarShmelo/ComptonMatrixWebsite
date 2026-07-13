#!/usr/bin/env python3
"""
Compute the deterministic multigroup Compton scattering matrix (angle-resolved).

Headless, SLURM-friendly script. Computes the full 128x128x100 sigma and dsigma/dT
matrices on a 128-group geometric energy grid with 100 angle bins.

Usage:
  python scripts/compute_matrix.py --temperature-index 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

ROOT = Path(__file__).resolve().parent.parent

N_GROUPS = 128
N_ANGLE_BINS = 100
E_MIN_KEV = 1e-5  # 0.01 eV
E_MAX_KEV = 300.0

N_TEMPS = 64
T_MIN_K = 1000.0
T_MAX_K = 1e9

TEMPERATURES_K = np.geomspace(T_MIN_K, T_MAX_K, N_TEMPS)

CACHE_DIR = ROOT / "docs" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temperature-index", type=int, required=True,
        help=f"Temperature index (0..{N_TEMPS - 1})")
    parser.add_argument(
        "--quadrature-scale", type=int, default=1,
        help="Multiply all quadrature orders by this factor (default: 1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tidx = args.temperature_index
    qscale = args.quadrature_scale
    if not 0 <= tidx < N_TEMPS:
        raise ValueError(f"temperature-index must be in [0, {N_TEMPS - 1}], got {tidx}")

    T = TEMPERATURES_K[tidx]

    boundaries_kev = np.geomspace(E_MIN_KEV, E_MAX_KEV, N_GROUPS + 1)
    boundaries_erg = boundaries_kev * kev

    config_kwargs = dict(cutoff_ratio=1e-10)
    if qscale > 1:
        config_kwargs.update(
            xi_order=48 * qscale,
            xi_tail_order=16 * qscale,
            ep_edge_order=24 * qscale,
            ep_interior_order=24 * qscale,
            e_panel_order=12 * qscale,
        )

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries_erg.tolist(),
        weight_function=cm.UniformWeightFunction(),
        config=cm.MGIntegrationConfig(**config_kwargs),
    )
    kernel = cds.ComptonKernelSolver()

    quad_label = f", quadrature x{qscale}" if qscale > 1 else ""
    print(f"[T-index {tidx:03d}] Computing deterministic {N_GROUPS}x{N_GROUPS}x{N_ANGLE_BINS} matrix "
          f"at T = {T:.6e} K{quad_label} ...")

    t0 = time.time()
    sigma_matrix = np.asarray(
        mg.compute_sigma_matrix(kernel, N_ANGLE_BINS, T=T, Ne=1.0))
    elapsed_sigma = time.time() - t0
    print(f"  sigma_matrix done ({elapsed_sigma:.1f}s)")

    t0 = time.time()
    dsigma_dT_matrix = np.asarray(
        mg.compute_dsigma_dT_matrix(kernel, N_ANGLE_BINS, T=T, Ne=1.0))
    elapsed_deriv = time.time() - t0
    print(f"  dsigma_dT_matrix done ({elapsed_deriv:.1f}s)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"T{tidx:03d}_{T:.6e}K.npz"
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
