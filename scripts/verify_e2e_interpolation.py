#!/usr/bin/env python3
"""
End-to-end verification of the website's temperature interpolation.

Downloads data from the HTTP server (simulating browser behavior), computes
interpolated matrices for custom temperatures using the same collapse/interp
logic as the website, then computes exact matrices via ComptonMatrixExact
and compares.

Tests several custom temperatures on a "weird" non-aligned grid.

Requires:
  - HTTP server running: python3 -m http.server 8791 --bind 127.0.0.1
    (from the docs/ directory)
  - ComptonMatrixExact installed in the venv

Usage:
  python3 scripts/verify_e2e_interpolation.py
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))

from collapse import _collapse_to_array, collapse_interp  # noqa: E402

BASE_URL = "http://127.0.0.1:8791"
OUT_DIR = ROOT / "output"


def fetch_manifest():
    """Fetch manifest.json from the HTTP server."""
    url = f"{BASE_URL}/data/manifest.json"
    print(f"  Fetching {url}")
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def fetch_npz(filename: str) -> bytes:
    """Fetch a .npz file from the HTTP server."""
    url = f"{BASE_URL}/data/{filename}"
    print(f"  Fetching {url}")
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def find_brackets(manifest, T_target):
    """Find bracketing stored temperatures for T_target (mirrors JS logic)."""
    temps = manifest["temperatures"]
    for i in range(len(temps) - 1):
        T_lo = temps[i]["temperature_K"]
        T_hi = temps[i + 1]["temperature_K"]
        if abs(T_target - T_lo) / T_lo < 1e-9:
            return i, i, True
        if abs(T_target - T_hi) / T_hi < 1e-9:
            return i + 1, i + 1, True
        if T_lo < T_target < T_hi:
            return i, i + 1, False
    return len(temps) - 1, len(temps) - 1, True


def compute_interpolated_from_server(manifest, T_target, energy_bounds, n_angle_bins):
    """
    Simulate what the website does: fetch .npz files via HTTP, collapse, interpolate.
    """
    idx_lo, idx_hi, exact = find_brackets(manifest, T_target)

    if exact:
        entry = manifest["temperatures"][idx_lo]
        npz_bytes = fetch_npz(entry["file"])
        return _collapse_to_array(npz_bytes, energy_bounds, n_angle_bins)
    else:
        entry_lo = manifest["temperatures"][idx_lo]
        entry_hi = manifest["temperatures"][idx_hi]
        npz_lo = fetch_npz(entry_lo["file"])
        npz_hi = fetch_npz(entry_hi["file"])
        T_lo = entry_lo["temperature_K"]
        T_hi = entry_hi["temperature_K"]
        return collapse_interp(
            npz_lo, npz_hi, T_lo, T_hi, T_target,
            energy_bounds, n_angle_bins,
        )


def compute_exact(boundaries_keV, n_angle_bins, T):
    """Compute the exact matrix using ComptonMatrixExact with uniform weighting."""
    boundaries_erg = np.asarray(boundaries_keV) * kev

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries_erg.tolist(),
        weight_function=cm.UniformWeightFunction(),
        config=cm.MGIntegrationConfig(cutoff_ratio=1e-10),
    )
    kernel = cds.ComptonKernelSolver()

    nG = len(boundaries_keV) - 1
    print(f"  Computing exact {nG}x{nG}x{n_angle_bins} matrix at T = {T:.6e} K ...")
    t0 = time.time()
    sigma = np.asarray(mg.compute_sigma_matrix(kernel, n_angle_bins, T=T, Ne=1.0))
    elapsed = time.time() - t0
    print(f"  Exact computation done ({elapsed:.1f}s)")
    return sigma


def compare(exact, interpolated, label):
    """Compare exact vs interpolated and report metrics."""
    peak = np.max(np.abs(exact))
    if peak == 0:
        print(f"  {label}: all zeros")
        return 0.0

    abs_diff = np.abs(exact - interpolated)
    peak_norm_max = (abs_diff / peak).max()
    peak_norm_mean = (abs_diff / peak).mean()

    print(f"  {label}:")
    print(f"    Peak-normalized max error:  {peak_norm_max:.6e}")
    print(f"    Peak-normalized mean error: {peak_norm_mean:.6e}")

    sig_mask = np.abs(exact) > 0.10 * peak
    if np.any(sig_mask):
        rel = abs_diff[sig_mask] / np.abs(exact[sig_mask])
        print(f"    Elements > 10% of peak: {np.count_nonzero(sig_mask)}/{exact.size}")
        print(f"    Max relative error:  {rel.max():.6e} ({100*rel.max():.4f}%)")
        print(f"    Mean relative error: {rel.mean():.6e} ({100*rel.mean():.4f}%)")
        return rel.max()
    else:
        print(f"    No elements > 10% of peak")
        return 0.0


def generate_weird_grid(seed=77):
    """
    Generate a non-aligned 7-group, 5-angle-bin grid with irregular spacing.
    Boundaries do NOT coincide with the stored fine grid.
    """
    rng = np.random.default_rng(seed=seed)

    log_min = np.log10(1e-5)
    log_max = np.log10(300.0)
    total_range = log_max - log_min

    # 7 groups with irregular spacing
    n_groups = 7
    fracs = np.sort(rng.uniform(0.08, 0.92, size=n_groups - 1))
    inner_log = log_min + fracs * total_range
    boundaries_keV = np.concatenate([[1e-5], 10**inner_log, [300.0]])

    n_angle_bins = 5
    return boundaries_keV, n_angle_bins


def main():
    print("=" * 70)
    print("End-to-End Interpolation Verification (HTTP server + exact solver)")
    print("=" * 70)

    # Fetch manifest from HTTP server
    print("\n--- Fetching manifest from server ---")
    manifest = fetch_manifest()
    print(f"  {len(manifest['temperatures'])} temperatures available")

    # Generate weird grid
    boundaries_keV, n_angle_bins = generate_weird_grid()
    n_groups = len(boundaries_keV) - 1
    print(f"\n--- Test grid ---")
    print(f"  {n_groups} energy groups, {n_angle_bins} angle bins")
    print(f"  Energy boundaries (keV):")
    for i, b in enumerate(boundaries_keV):
        print(f"    [{i:2d}] {b:.6e}")

    # Custom temperatures: midpoints between stored grid points (log-space)
    # Choose a variety across the temperature range
    test_temperatures = [
        1500.0,           # between T0 (1000) and T1 (1245)
        5.0e4,            # mid-range
        3.0e6,            # mid-high
        5.0e7,            # high
        5.0e8,            # near upper end
    ]

    print(f"\n--- Test temperatures ---")
    for T in test_temperatures:
        idx_lo, idx_hi, exact = find_brackets(manifest, T)
        if exact:
            print(f"  T = {T:.4e} K  (exact match at index {idx_lo})")
        else:
            T_lo = manifest["temperatures"][idx_lo]["temperature_K"]
            T_hi = manifest["temperatures"][idx_hi]["temperature_K"]
            print(f"  T = {T:.4e} K  (between T[{idx_lo}]={T_lo:.4e} and T[{idx_hi}]={T_hi:.4e})")

    # For each temperature: download + interpolate, then compute exact
    all_max_errors = []
    results = []

    for T in test_temperatures:
        print(f"\n{'─' * 70}")
        print(f"Temperature: T = {T:.6e} K")
        print(f"{'─' * 70}")

        print("\n  [1] Downloading and interpolating from HTTP server...")
        interpolated = compute_interpolated_from_server(
            manifest, T, boundaries_keV.tolist(), n_angle_bins
        )
        print(f"      Shape: {interpolated.shape}")

        print("\n  [2] Computing exact matrix with ComptonMatrixExact...")
        exact = compute_exact(boundaries_keV, n_angle_bins, T)
        print(f"      Shape: {exact.shape}")

        print("\n  [3] Comparison:")
        max_err = compare(exact, interpolated, f"T = {T:.4e} K")
        all_max_errors.append(max_err)
        results.append({
            "temperature_K": T,
            "max_rel_error": max_err,
            "exact": exact,
            "interpolated": interpolated,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Grid: {n_groups} groups x {n_angle_bins} angle bins (non-aligned)")
    print(f"  Temperatures tested: {len(test_temperatures)}")
    print(f"")
    print(f"  {'T (K)':>12s}  {'Max rel. error':>14s}")
    print(f"  {'─'*12}  {'─'*14}")
    for T, err in zip(test_temperatures, all_max_errors):
        print(f"  {T:12.4e}  {err:14.6e} ({100*err:.4f}%)")

    overall_max = max(all_max_errors)
    print(f"\n  Overall max relative error: {overall_max:.6e} ({100*overall_max:.4f}%)")

    threshold = 0.15  # 15% threshold (combines interpolation + collapse error)
    passed = overall_max < threshold
    print(f"  Pass criterion: max error < {100*threshold:.0f}%")
    if passed:
        print("  VERIFICATION PASSED")
    else:
        print("  VERIFICATION FAILED")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "verify_e2e_interpolation_results.npz"
    np.savez_compressed(
        out_path,
        temperatures_K=np.array(test_temperatures),
        max_rel_errors=np.array(all_max_errors),
        boundaries_keV=boundaries_keV,
        n_angle_bins=n_angle_bins,
    )
    print(f"\n  Results saved to {out_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
