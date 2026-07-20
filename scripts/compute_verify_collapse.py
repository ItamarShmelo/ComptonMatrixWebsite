#!/usr/bin/env python3
"""
Verify that collapsing the fine 128x128x100 grid to an arbitrary coarse grid
matches an exact computation on that coarse grid.

Generates a random 10x10x12 grid (10 energy groups, 12 angle bins) with
boundaries that do NOT align with the fine grid, computes the exact matrix
via ComptonMultigroupKernel, collapses the fine-grid data, and compares.

Usage:
  python scripts/compute_verify_collapse.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))

from collapse import collapse  # noqa: E402

DATA_DIR = ROOT / "docs" / "data"
OUT_DIR = ROOT / "output"

N_TEMPS = 64
T_MIN_K = 1000.0
T_MAX_K = 1e9
TEMPERATURES_K = np.geomspace(T_MIN_K, T_MAX_K, N_TEMPS)

TIDX = 30
T = TEMPERATURES_K[TIDX]


def generate_random_grid(seed=42):
    """Generate a 10-group, 12-angle-bin grid with random energy boundaries."""
    rng = np.random.default_rng(seed=seed)
    log_min, log_max = np.log10(1e-5), np.log10(300.0)
    min_spacing = 0.5

    raw = np.sort(rng.uniform(0, 1, size=9))
    available = (log_max - log_min) - min_spacing * 10
    inner = log_min + min_spacing + raw * available + np.arange(9) * min_spacing

    boundaries_keV = np.concatenate([[1e-5], 10**inner, [300.0]])
    n_angle_bins = 12

    log_spacings = np.diff(np.log10(boundaries_keV))
    assert np.all(log_spacings >= min_spacing - 1e-10), (
        f"Minimum log-spacing violated: {log_spacings.min():.3f} < {min_spacing}"
    )
    return boundaries_keV, n_angle_bins


def compute_exact(boundaries_keV, n_angle_bins, T):
    """Compute the exact matrix on the given grid using ComptonMultigroupKernel."""
    boundaries_erg = boundaries_keV * kev

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries_erg.tolist(),
        weight_function=cm.UniformWeightFunction(),
        config=cm.MGIntegrationConfig(cutoff_ratio=1e-10),
    )
    kernel = cds.ComptonKernelSolver()

    print(f"  Computing exact {len(boundaries_keV)-1}x{len(boundaries_keV)-1}x{n_angle_bins} "
          f"matrix at T = {T:.6e} K ...")

    t0 = time.time()
    sigma = np.asarray(mg.compute_sigma_matrix(kernel, n_angle_bins, T=T, Ne=1.0))
    elapsed = time.time() - t0
    print(f"  Exact computation done ({elapsed:.1f}s)")
    return sigma


def compute_collapsed(boundaries_keV, n_angle_bins, T, tidx):
    """Collapse the fine-grid data to the given coarse grid."""
    npz_path = None
    for path in sorted(DATA_DIR.glob("T*_*K.npz")):
        if path.name.startswith(f"T{tidx:03d}_"):
            npz_path = path
            break
    if npz_path is None:
        raise FileNotFoundError(f"No .npz file found for T-index {tidx}")

    print(f"  Collapsing {npz_path.name} to {len(boundaries_keV)-1}x"
          f"{len(boundaries_keV)-1}x{n_angle_bins} ...")

    t0 = time.time()
    npz_bytes = npz_path.read_bytes()
    result_bytes = collapse(npz_bytes, boundaries_keV.tolist(), n_angle_bins)
    result = np.load(io.BytesIO(result_bytes))
    elapsed = time.time() - t0
    print(f"  Collapse done ({elapsed:.1f}s)")
    return result


def compare(exact, collapsed, boundaries_keV):
    """Compare exact and collapsed matrices element-by-element.

    Uses two metrics:
    1. Peak-normalized absolute error: |exact - collapsed| / peak
    2. Relative error on significant elements (|exact| > 1% of peak)

    The linear splitting approximation introduces ~few-percent errors on
    significant elements, which is consistent with the fine-grid resolution
    (128 groups, ~13 fine groups per coarse group).
    """
    N, _, K = exact.shape
    print(f"\n  Shapes: exact={exact.shape}, collapsed={collapsed.shape}")

    peak = np.max(np.abs(exact))
    if peak == 0:
        print("  WARNING: exact matrix is all zeros")
        return True

    abs_diff = np.abs(exact - collapsed)

    peak_norm_err = abs_diff / peak
    print(f"  Peak-normalized max absolute error: {peak_norm_err.max():.6e}")
    print(f"  Peak-normalized mean absolute error: {peak_norm_err.mean():.6e}")

    for thresh_label, thresh_frac in [("1%", 0.01), ("0.1%", 0.001)]:
        mask = np.abs(exact) > thresh_frac * peak
        n = np.count_nonzero(mask)
        if n == 0:
            continue
        rel = abs_diff[mask] / np.abs(exact[mask])
        print(f"\n  Elements with |exact| > {thresh_label} of peak: {n} / {exact.size}")
        print(f"    Max relative error:    {rel.max():.6e}")
        print(f"    Mean relative error:   {rel.mean():.6e}")
        print(f"    Median relative error: {np.median(rel):.6e}")

    sig_mask = np.abs(exact) > 0.10 * peak
    if not np.any(sig_mask):
        print("  No significant elements to compare.")
        return True

    sig_rel = abs_diff[sig_mask] / np.abs(exact[sig_mask])
    max_sig_rel = sig_rel.max()
    print(f"\n  PASS criterion: max relative error on dominant elements "
          f"(>10% of peak) = {max_sig_rel:.4f} ({100*max_sig_rel:.2f}%)")
    passed = max_sig_rel < 0.10
    return passed


def main():
    print("=" * 60)
    print("Collapse verification: random 10x10x12 grid")
    print("=" * 60)

    boundaries_keV, n_angle_bins = generate_random_grid()
    print(f"\nGrid: {len(boundaries_keV)-1} energy groups, {n_angle_bins} angle bins")
    print(f"Temperature: T-index {TIDX}, T = {T:.6e} K")
    print(f"Energy boundaries (keV):")
    for i, b in enumerate(boundaries_keV):
        print(f"  [{i:2d}] {b:.6e}")

    print(f"\n--- Exact computation ---")
    exact = compute_exact(boundaries_keV, n_angle_bins, T)

    print(f"\n--- Collapse from fine grid ---")
    collapsed = compute_collapsed(boundaries_keV, n_angle_bins, T, TIDX)

    print(f"\n--- Comparison ---")
    passed = compare(exact, collapsed, boundaries_keV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "verify_collapse_results.npz"
    np.savez_compressed(
        out_path,
        exact=exact,
        collapsed=collapsed,
        boundaries_keV=boundaries_keV,
        n_angle_bins=n_angle_bins,
        temperature_K=T,
    )
    print(f"\nResults saved to {out_path}")

    if passed:
        print("\nVERIFICATION PASSED")
    else:
        print("\nVERIFICATION FAILED: relative errors exceed threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
