#!/usr/bin/env python3
"""
Verify that collapsing the fine 128x128x100 grid to an arbitrary coarse grid
with Planck/Wien/uniform weighting matches exact computation via
ComptonMatrixExact for all three weight functions.

Tests 6 temperatures x 3 weightings = 18 comparisons on a random 10x12 grid.

Requires ComptonMatrixExact installed in the venv.

Usage:
  python scripts/verify_weighted_collapse.py
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

from collapse import _collapse_to_array  # noqa: E402

DATA_DIR = ROOT / "docs" / "data"
OUT_DIR = ROOT / "output"

N_GROUPS = 128
E_MIN_KEV = 1e-5
E_MAX_KEV = 300.0

N_TEMPS = 64
T_MIN_K = 1000.0
T_MAX_K = 1e9
TEMPERATURES_K = np.geomspace(T_MIN_K, T_MAX_K, N_TEMPS)

TEST_INDICES = [5, 15, 25, 35, 45, 55]


def make_weighting_configs(boundaries_keV):
    """Build weight function factories using the given grid boundaries."""
    bounds_erg = (boundaries_keV * kev).tolist()
    return [
        ("uniform", lambda: cm.UniformWeightFunction()),
        ("planck", lambda b=bounds_erg: cm.PlanckWeightFunction(cap_x=25.0, group_boundaries=b)),
        ("wien", lambda b=bounds_erg: cm.WienWeightFunction(group_boundaries=b)),
    ]


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
    return boundaries_keV, n_angle_bins


def find_npz(tidx: int, weighting: str = "uniform") -> Path:
    """Find the .npz file for a given temperature index and weighting."""
    wdir = DATA_DIR / weighting
    for path in sorted(wdir.glob("T*_*K.npz")):
        if path.name.startswith(f"T{tidx:03d}_"):
            return path
    raise FileNotFoundError(f"No .npz file found for T-index {tidx} in {wdir}")


def compute_exact(boundaries_keV, n_angle_bins, T, weight_function):
    """Compute the exact matrix on the given grid using ComptonMultigroupKernel."""
    boundaries_erg = boundaries_keV * kev

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries_erg.tolist(),
        weight_function=weight_function,
        config=cm.MGIntegrationConfig(cutoff_ratio=1e-10),
    )
    kernel = cds.ComptonKernelSolver()

    t0 = time.time()
    sigma = np.asarray(mg.compute_sigma_matrix(kernel, n_angle_bins, T=T))
    elapsed = time.time() - t0
    return sigma, elapsed


def compute_collapsed(npz_path, boundaries_keV, n_angle_bins, weighting, T):
    """Collapse the fine-grid data to the given coarse grid."""
    npz_bytes = npz_path.read_bytes()
    t0 = time.time()
    result = _collapse_to_array(
        npz_bytes, boundaries_keV.tolist(), n_angle_bins,
        weighting=weighting,
        temperature_K=T if weighting != "uniform" else None,
    )
    elapsed = time.time() - t0
    return result, elapsed


def compute_error_metrics(exact, collapsed):
    """Compute error metrics comparing exact vs collapsed matrices."""
    peak = np.max(np.abs(exact))
    if peak == 0:
        return {"peak": 0, "max_peak_norm": 0, "mean_peak_norm": 0,
                "max_rel_10pct": 0, "mean_rel_10pct": 0,
                "max_rel_1pct": 0, "mean_rel_1pct": 0,
                "n_sig_10pct": 0, "n_sig_1pct": 0}

    abs_diff = np.abs(exact - collapsed)
    metrics = {
        "peak": peak,
        "max_peak_norm": (abs_diff / peak).max(),
        "mean_peak_norm": (abs_diff / peak).mean(),
    }

    for thresh_label, thresh_frac in [("1pct", 0.01), ("10pct", 0.10)]:
        mask = np.abs(exact) > thresh_frac * peak
        n = np.count_nonzero(mask)
        metrics[f"n_sig_{thresh_label}"] = n
        if n > 0:
            rel = abs_diff[mask] / np.abs(exact[mask])
            metrics[f"max_rel_{thresh_label}"] = rel.max()
            metrics[f"mean_rel_{thresh_label}"] = rel.mean()
        else:
            metrics[f"max_rel_{thresh_label}"] = 0.0
            metrics[f"mean_rel_{thresh_label}"] = 0.0

    return metrics


def main():
    print("=" * 75)
    print("Weighted Collapse Verification: 3 weightings x 6 temperatures")
    print("=" * 75)

    boundaries_keV, n_angle_bins = generate_random_grid()
    n_groups = len(boundaries_keV) - 1
    weighting_configs = make_weighting_configs(boundaries_keV)
    print(f"\nGrid: {n_groups} energy groups, {n_angle_bins} angle bins")
    print(f"Energy boundaries (keV):")
    for i, b in enumerate(boundaries_keV):
        print(f"  [{i:2d}] {b:.6e}")

    all_results = []
    any_failed = False

    print(f"\n{'Weighting':>10s}  {'T_idx':>5s}  {'T (K)':>12s}  "
          f"{'max_rel(>10%)':>14s}  {'mean_rel(>10%)':>15s}  "
          f"{'max_peak_norm':>14s}  {'exact_time':>10s}  {'coll_time':>10s}")
    print(f"  {'─' * 10}  {'─' * 5}  {'─' * 12}  {'─' * 14}  {'─' * 15}  "
          f"{'─' * 14}  {'─' * 10}  {'─' * 10}")

    for tidx in TEST_INDICES:
        T = TEMPERATURES_K[tidx]

        for wname, wf_factory in weighting_configs:
            npz_path = find_npz(tidx, wname)
            exact, exact_t = compute_exact(
                boundaries_keV, n_angle_bins, T, wf_factory()
            )
            collapsed, coll_t = compute_collapsed(
                npz_path, boundaries_keV, n_angle_bins, wname, T
            )

            metrics = compute_error_metrics(exact, collapsed)

            print(f"  {wname:>10s}  {tidx:5d}  {T:12.4e}  "
                  f"{metrics['max_rel_10pct']:14.6e}  "
                  f"{metrics['mean_rel_10pct']:15.6e}  "
                  f"{metrics['max_peak_norm']:14.6e}  "
                  f"{exact_t:9.1f}s  {coll_t:9.3f}s")

            passed = metrics["max_rel_10pct"] < 0.10
            if not passed:
                any_failed = True

            all_results.append({
                "weighting": wname,
                "tidx": tidx,
                "temperature_K": T,
                "passed": passed,
                **metrics,
            })

    # Summary
    print(f"\n{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")
    print(f"  Grid: {n_groups}g x {n_angle_bins}a (non-aligned)")
    print(f"  Temperatures: {len(TEST_INDICES)} indices, "
          f"3 weightings = {len(all_results)} comparisons")

    for wname in ["uniform", "planck", "wien"]:
        w_results = [r for r in all_results if r["weighting"] == wname]
        max_err = max(r["max_rel_10pct"] for r in w_results)
        mean_err = np.mean([r["mean_rel_10pct"] for r in w_results])
        worst = max(w_results, key=lambda r: r["max_rel_10pct"])
        print(f"\n  {wname:>8s}:  max_rel={100*max_err:.4f}%  "
              f"mean_rel={100*mean_err:.4f}%  "
              f"worst at T[{worst['tidx']}]={worst['temperature_K']:.4e} K")

    overall_max = max(r["max_rel_10pct"] for r in all_results)
    print(f"\n  Overall max relative error (>10% peak): "
          f"{overall_max:.6e} ({100*overall_max:.4f}%)")
    print(f"  Pass criterion: < 10%")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "verify_weighted_collapse_results.npz"
    np.savez_compressed(
        out_path,
        boundaries_keV=boundaries_keV,
        n_angle_bins=n_angle_bins,
        test_indices=np.array(TEST_INDICES),
        temperatures_K=TEMPERATURES_K[TEST_INDICES],
        weightings=np.array(["uniform", "planck", "wien"]),
        max_rel_10pct=np.array([r["max_rel_10pct"] for r in all_results]).reshape(
            len(TEST_INDICES), 3
        ),
        mean_rel_10pct=np.array([r["mean_rel_10pct"] for r in all_results]).reshape(
            len(TEST_INDICES), 3
        ),
    )
    print(f"\n  Results saved to {out_path}")

    if any_failed:
        print("\nVERIFICATION FAILED: some relative errors exceed 10%")
        sys.exit(1)
    else:
        print("\nVERIFICATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
