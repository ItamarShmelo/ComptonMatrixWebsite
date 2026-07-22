#!/usr/bin/env python3
"""
Verify temperature interpolation accuracy.

Tests log-space linear interpolation by comparing interpolated results
against stored ground truth at multiple temperatures. For stored T_i,
interpolates between neighbors T_{i-1} and T_{i+1} and compares against
the actual stored T_i data.

No external solver (ComptonMatrixExact) is required -- uses only stored .npz data.

Usage:
  python3 scripts/verify_interpolation.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))

from collapse import _collapse_to_array, collapse_interp  # noqa: E402

DATA_DIR = ROOT / "docs" / "data"
OUT_DIR = ROOT / "output"

N_TEMPS = 64
T_MIN_K = 1000.0
T_MAX_K = 1e9
TEMPERATURES_K = np.geomspace(T_MIN_K, T_MAX_K, N_TEMPS)


def find_npz(idx: int) -> Path:
    """Find the .npz file for a given temperature index."""
    for path in sorted(DATA_DIR.glob("T*_*K.npz")):
        if path.name.startswith(f"T{idx:03d}_"):
            return path
    raise FileNotFoundError(f"No .npz file found for T-index {idx}")


def load_npz_bytes(path: Path) -> bytes:
    return path.read_bytes()


def get_fine_boundaries(npz_bytes: bytes) -> np.ndarray:
    return np.load(io.BytesIO(npz_bytes))["boundaries_keV"]


def compute_error_metrics(exact: np.ndarray, interpolated: np.ndarray) -> dict:
    """Compute error metrics comparing exact vs interpolated matrices."""
    peak = np.max(np.abs(exact))
    if peak == 0:
        return {"peak": 0, "max_abs": 0, "mean_abs": 0,
                "max_rel_1pct": 0, "mean_rel_1pct": 0,
                "max_rel_10pct": 0, "mean_rel_10pct": 0,
                "n_sig_1pct": 0, "n_sig_10pct": 0}

    abs_diff = np.abs(exact - interpolated)

    metrics = {
        "peak": peak,
        "max_abs": abs_diff.max(),
        "mean_abs": abs_diff.mean(),
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
            metrics[f"median_rel_{thresh_label}"] = np.median(rel)
        else:
            metrics[f"max_rel_{thresh_label}"] = 0.0
            metrics[f"mean_rel_{thresh_label}"] = 0.0
            metrics[f"median_rel_{thresh_label}"] = 0.0

    return metrics


def verify_interpolation_at_index(
    idx: int,
    energy_boundaries: list[float],
    n_angle_bins: int,
    skip: int = 1,
) -> dict:
    """
    Verify interpolation at temperature index `idx` using neighbors at +-skip.

    Parameters
    ----------
    idx : int
        Target temperature index (ground truth).
    energy_boundaries : list[float]
        Coarse energy boundaries for collapse.
    n_angle_bins : int
        Number of angle bins.
    skip : int
        Number of indices to skip (1 = adjacent neighbors, 2 = skip-one).

    Returns
    -------
    dict
        Error metrics.
    """
    idx_lo = idx - skip
    idx_hi = idx + skip

    T_lo = TEMPERATURES_K[idx_lo]
    T_hi = TEMPERATURES_K[idx_hi]
    T_target = TEMPERATURES_K[idx]

    npz_lo = load_npz_bytes(find_npz(idx_lo))
    npz_hi = load_npz_bytes(find_npz(idx_hi))
    npz_exact = load_npz_bytes(find_npz(idx))

    # Ground truth: collapse the actual stored data at T_target
    exact = _collapse_to_array(npz_exact, energy_boundaries, n_angle_bins)

    # Interpolated: collapse neighbors and interpolate in log-T space
    interpolated = collapse_interp(
        npz_lo, npz_hi, T_lo, T_hi, T_target,
        energy_boundaries, n_angle_bins,
    )

    return compute_error_metrics(exact, interpolated)


def main():
    print("=" * 65)
    print("Temperature Interpolation Verification")
    print("=" * 65)

    # Load fine boundaries from the first file
    npz0 = load_npz_bytes(find_npz(0))
    fine_bounds = get_fine_boundaries(npz0)

    # Test configurations
    configs = [
        ("16 energy groups, 10 angle bins (coarse)", 16, 10),
        ("128 energy groups, 100 angle bins (identity/fine)", 128, 100),
    ]

    # Test indices: every 4th from 2 to 62 (avoids boundary issues)
    test_indices = list(range(2, 63, 4))

    for config_label, n_groups, n_angles in configs:
        print(f"\n{'─' * 65}")
        print(f"Grid: {config_label}")
        print(f"{'─' * 65}")

        # Generate log-spaced energy boundaries
        log_min = np.log(fine_bounds[0])
        log_max = np.log(fine_bounds[-1])
        energy_bounds = np.exp(np.linspace(log_min, log_max, n_groups + 1)).tolist()

        # ── Adjacent interpolation (skip=1) ──────────────────────
        print(f"\n  Adjacent interpolation (skip=1, ratio ~1.25x):")
        print(f"  {'Idx':>4s}  {'T (K)':>12s}  {'max_rel(>10%)':>14s}  {'mean_rel(>10%)':>15s}  {'max_peak_norm':>14s}")
        print(f"  {'─'*4}  {'─'*12}  {'─'*14}  {'─'*15}  {'─'*14}")

        all_metrics_adj = []
        for idx in test_indices:
            metrics = verify_interpolation_at_index(idx, energy_bounds, n_angles, skip=1)
            all_metrics_adj.append(metrics)
            print(f"  {idx:4d}  {TEMPERATURES_K[idx]:12.4e}  "
                  f"{metrics['max_rel_10pct']:14.6e}  "
                  f"{metrics['mean_rel_10pct']:15.6e}  "
                  f"{metrics['max_peak_norm']:14.6e}")

        max_rel_adj = max(m["max_rel_10pct"] for m in all_metrics_adj)
        mean_rel_adj = np.mean([m["mean_rel_10pct"] for m in all_metrics_adj])
        worst_idx_adj = test_indices[np.argmax([m["max_rel_10pct"] for m in all_metrics_adj])]

        print(f"\n  Summary (adjacent, skip=1):")
        print(f"    Max relative error (>10% peak):  {max_rel_adj:.6e} ({100*max_rel_adj:.4f}%)")
        print(f"    Mean relative error (>10% peak): {mean_rel_adj:.6e} ({100*mean_rel_adj:.4f}%)")
        print(f"    Worst-case temperature index:    {worst_idx_adj} (T = {TEMPERATURES_K[worst_idx_adj]:.4e} K)")

        # ── Skip-one interpolation (skip=2) ──────────────────────
        test_indices_skip2 = list(range(4, 61, 4))
        print(f"\n  Skip-one interpolation (skip=2, ratio ~1.56x):")
        print(f"  {'Idx':>4s}  {'T (K)':>12s}  {'max_rel(>10%)':>14s}  {'mean_rel(>10%)':>15s}  {'max_peak_norm':>14s}")
        print(f"  {'─'*4}  {'─'*12}  {'─'*14}  {'─'*15}  {'─'*14}")

        all_metrics_skip2 = []
        for idx in test_indices_skip2:
            metrics = verify_interpolation_at_index(idx, energy_bounds, n_angles, skip=2)
            all_metrics_skip2.append(metrics)
            print(f"  {idx:4d}  {TEMPERATURES_K[idx]:12.4e}  "
                  f"{metrics['max_rel_10pct']:14.6e}  "
                  f"{metrics['mean_rel_10pct']:15.6e}  "
                  f"{metrics['max_peak_norm']:14.6e}")

        max_rel_skip2 = max(m["max_rel_10pct"] for m in all_metrics_skip2)
        mean_rel_skip2 = np.mean([m["mean_rel_10pct"] for m in all_metrics_skip2])
        worst_idx_skip2 = test_indices_skip2[np.argmax([m["max_rel_10pct"] for m in all_metrics_skip2])]

        print(f"\n  Summary (skip-one, skip=2):")
        print(f"    Max relative error (>10% peak):  {max_rel_skip2:.6e} ({100*max_rel_skip2:.4f}%)")
        print(f"    Mean relative error (>10% peak): {mean_rel_skip2:.6e} ({100*mean_rel_skip2:.4f}%)")
        print(f"    Worst-case temperature index:    {worst_idx_skip2} (T = {TEMPERATURES_K[worst_idx_skip2]:.4e} K)")

    # ── Final verdict ────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("OVERALL VERDICT")
    print(f"{'=' * 65}")

    # Re-run with coarse grid for final pass/fail
    log_min = np.log(fine_bounds[0])
    log_max = np.log(fine_bounds[-1])
    energy_bounds_16 = np.exp(np.linspace(log_min, log_max, 17)).tolist()

    final_metrics = []
    for idx in test_indices:
        m = verify_interpolation_at_index(idx, energy_bounds_16, 10, skip=1)
        final_metrics.append(m)

    overall_max = max(m["max_rel_10pct"] for m in final_metrics)
    overall_mean = np.mean([m["mean_rel_10pct"] for m in final_metrics])

    print(f"  Adjacent interpolation (16g, 10a):")
    print(f"    Max relative error on dominant elements: {100*overall_max:.4f}%")
    print(f"    Mean relative error on dominant elements: {100*overall_mean:.4f}%")

    threshold = 0.10  # 10%
    passed = overall_max < threshold
    print(f"\n  Pass criterion: max relative error < {100*threshold:.0f}%")
    if passed:
        print("  VERIFICATION PASSED")
    else:
        print("  VERIFICATION FAILED: relative errors exceed threshold")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "verify_interpolation_results.npz"
    np.savez_compressed(
        out_path,
        test_indices=np.array(test_indices),
        temperatures_K=TEMPERATURES_K[test_indices],
        max_rel_errors=np.array([m["max_rel_10pct"] for m in final_metrics]),
        mean_rel_errors=np.array([m["mean_rel_10pct"] for m in final_metrics]),
    )
    print(f"\n  Results saved to {out_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
