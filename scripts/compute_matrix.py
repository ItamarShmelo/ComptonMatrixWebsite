#!/usr/bin/env python3
"""
Compute the deterministic multigroup Compton scattering matrix (angle-resolved).

Headless, SLURM-friendly script. Computes the full 128x128x100 sigma matrix
on a 128-group geometric energy grid with 100 angle bins using uniform weighting.

Usage:
  python scripts/compute_matrix.py --temperature-index 0
  python scripts/compute_matrix.py --temperature-index 5 --e-panel-order 96
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
        "--e-panel-order", type=int, default=None,
        help="Override e_panel_order (default: library default of 12)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tidx = args.temperature_index
    if not 0 <= tidx < N_TEMPS:
        raise ValueError(f"temperature-index must be in [0, {N_TEMPS - 1}], got {tidx}")

    T = TEMPERATURES_K[tidx]

    boundaries_kev = np.geomspace(E_MIN_KEV, E_MAX_KEV, N_GROUPS + 1)
    boundaries_erg = (boundaries_kev * kev).tolist()

    wf = cm.UniformWeightFunction()

    config_kwargs: dict = dict(cutoff_ratio=1e-10)
    if args.e_panel_order is not None:
        config_kwargs["e_panel_order"] = args.e_panel_order

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries_erg,
        weight_function=wf,
        config=cm.MGIntegrationConfig(**config_kwargs),
    )
    kernel = cds.ComptonKernelSolver()

    ep_label = f", e_panel_order={args.e_panel_order}" if args.e_panel_order else ""
    print(f"[T{tidx:03d}] Computing {N_GROUPS}x{N_GROUPS}x{N_ANGLE_BINS} matrix "
          f"at T = {T:.6e} K{ep_label} ...")

    t0 = time.time()
    sigma_matrix = np.asarray(
        mg.compute_sigma_matrix(kernel, N_ANGLE_BINS, T=T))
    elapsed = time.time() - t0
    print(f"  sigma_matrix done ({elapsed:.1f}s)")

    out_dir = CACHE_DIR / "uniform"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"T{tidx:03d}_{T:.6e}K.npz"
    np.savez_compressed(
        out_path,
        sigma_matrix=sigma_matrix.astype(np.float32),
        boundaries_keV=boundaries_kev,
        temperature_K=T,
    )
    print(f"Saved: {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
