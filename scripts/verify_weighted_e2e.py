#!/usr/bin/env python3
"""
End-to-end verification of weighted collapse + temperature interpolation.

Downloads data from the HTTP server (simulating browser behavior), computes
interpolated matrices for custom temperatures using collapse_interp with
uniform/Planck/Wien weighting, then computes exact matrices via
ComptonMatrixExact with the matching weight function and compares.

Tests 5 temperatures x 3 weightings = 15 comparisons on a non-aligned grid.

Requires:
  - HTTP server running: python3 -m http.server 8791 --bind 127.0.0.1
    (from the docs/ directory)
  - ComptonMatrixExact installed in the venv

Usage:
  python3 scripts/verify_weighted_e2e.py
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

def make_weighting_configs(boundaries_keV):
    """Build weight function factories using the given grid boundaries."""
    bounds_erg = (boundaries_keV * kev).tolist()
    return [
        ("uniform", lambda: cm.UniformWeightFunction()),
        ("planck", lambda b=bounds_erg: cm.PlanckWeightFunction(cap_x=25.0, group_boundaries=b)),
        ("wien", lambda b=bounds_erg: cm.WienWeightFunction(group_boundaries=b)),
    ]

TEST_TEMPERATURES = [
    1500.0,
    5.0e4,
    3.0e6,
    5.0e7,
    5.0e8,
]


def fetch_manifest():
    """Fetch manifest.json from the HTTP server."""
    url = f"{BASE_URL}/data/uniform/manifest.json"
    print(f"  Fetching {url}")
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def fetch_npz(filename: str, weighting: str = "uniform") -> bytes:
    """Fetch a .npz file from the HTTP server."""
    url = f"{BASE_URL}/data/{weighting}/{filename}"
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


def compute_from_server(manifest, T_target, energy_bounds, n_angle_bins,
                        weighting):
    """Simulate website: fetch .npz files via HTTP, collapse with weighting."""
    idx_lo, idx_hi, exact = find_brackets(manifest, T_target)

    temp_K = T_target if weighting != "uniform" else None

    if exact:
        entry = manifest["temperatures"][idx_lo]
        npz_bytes = fetch_npz(entry["file"], weighting)
        return _collapse_to_array(
            npz_bytes, energy_bounds, n_angle_bins,
            weighting=weighting, temperature_K=temp_K,
        )
    else:
        entry_lo = manifest["temperatures"][idx_lo]
        entry_hi = manifest["temperatures"][idx_hi]
        npz_lo = fetch_npz(entry_lo["file"], weighting)
        npz_hi = fetch_npz(entry_hi["file"], weighting)
        T_lo = entry_lo["temperature_K"]
        T_hi = entry_hi["temperature_K"]
        return collapse_interp(
            npz_lo, npz_hi, T_lo, T_hi, T_target,
            energy_bounds, n_angle_bins,
            weighting=weighting, temperature_K=temp_K,
        )


def compute_exact(boundaries_keV, n_angle_bins, T, weight_function):
    """Compute the exact matrix using ComptonMatrixExact."""
    boundaries_erg = np.asarray(boundaries_keV) * kev

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries_erg.tolist(),
        weight_function=weight_function,
        config=cm.MGIntegrationConfig(cutoff_ratio=1e-10),
    )
    kernel = cds.ComptonKernelSolver()

    nG = len(boundaries_keV) - 1
    t0 = time.time()
    sigma = np.asarray(mg.compute_sigma_matrix(kernel, n_angle_bins, T=T))
    elapsed = time.time() - t0
    return sigma, elapsed


def compute_error_metrics(exact, interpolated):
    """Compute error metrics comparing exact vs interpolated."""
    peak = np.max(np.abs(exact))
    if peak == 0:
        return {"max_rel_10pct": 0.0, "mean_rel_10pct": 0.0,
                "max_peak_norm": 0.0, "n_sig_10pct": 0}

    abs_diff = np.abs(exact - interpolated)
    metrics = {
        "max_peak_norm": (abs_diff / peak).max(),
        "mean_peak_norm": (abs_diff / peak).mean(),
    }

    sig_mask = np.abs(exact) > 0.10 * peak
    n = np.count_nonzero(sig_mask)
    metrics["n_sig_10pct"] = n
    if n > 0:
        rel = abs_diff[sig_mask] / np.abs(exact[sig_mask])
        metrics["max_rel_10pct"] = rel.max()
        metrics["mean_rel_10pct"] = rel.mean()
    else:
        metrics["max_rel_10pct"] = 0.0
        metrics["mean_rel_10pct"] = 0.0

    return metrics


def generate_weird_grid(seed=77):
    """Generate a non-aligned 7-group, 5-angle-bin grid."""
    rng = np.random.default_rng(seed=seed)

    log_min = np.log10(1e-5)
    log_max = np.log10(300.0)
    total_range = log_max - log_min

    n_groups = 7
    fracs = np.sort(rng.uniform(0.08, 0.92, size=n_groups - 1))
    inner_log = log_min + fracs * total_range
    boundaries_keV = np.concatenate([[1e-5], 10**inner_log, [300.0]])

    n_angle_bins = 5
    return boundaries_keV, n_angle_bins


def main():
    print("=" * 75)
    print("Weighted E2E Verification (HTTP + collapse + interp vs exact)")
    print("=" * 75)

    print("\n--- Fetching manifest from server ---")
    manifest = fetch_manifest()
    print(f"  {len(manifest['temperatures'])} temperatures available")

    boundaries_keV, n_angle_bins = generate_weird_grid()
    n_groups = len(boundaries_keV) - 1
    weighting_configs = make_weighting_configs(boundaries_keV)
    print(f"\n--- Test grid ---")
    print(f"  {n_groups} energy groups, {n_angle_bins} angle bins")
    print(f"  Energy boundaries (keV):")
    for i, b in enumerate(boundaries_keV):
        print(f"    [{i:2d}] {b:.6e}")

    print(f"\n--- Test temperatures ---")
    for T in TEST_TEMPERATURES:
        idx_lo, idx_hi, exact = find_brackets(manifest, T)
        if exact:
            print(f"  T = {T:.4e} K  (exact match at index {idx_lo})")
        else:
            T_lo = manifest["temperatures"][idx_lo]["temperature_K"]
            T_hi = manifest["temperatures"][idx_hi]["temperature_K"]
            print(f"  T = {T:.4e} K  (between T[{idx_lo}]={T_lo:.4e} "
                  f"and T[{idx_hi}]={T_hi:.4e})")

    all_results = []

    for T in TEST_TEMPERATURES:
        print(f"\n{'─' * 75}")
        print(f"Temperature: T = {T:.6e} K")
        print(f"{'─' * 75}")

        for wname, wf_factory in weighting_configs:
            print(f"\n  [{wname}]")

            print(f"    Downloading and processing from HTTP server...")
            interpolated = compute_from_server(
                manifest, T, boundaries_keV.tolist(), n_angle_bins, wname
            )
            print(f"    Shape: {interpolated.shape}")

            print(f"    Computing exact matrix with ComptonMatrixExact...")
            exact, elapsed = compute_exact(
                boundaries_keV, n_angle_bins, T, wf_factory()
            )
            print(f"    Exact done ({elapsed:.1f}s), shape: {exact.shape}")

            metrics = compute_error_metrics(exact, interpolated)
            max_err = metrics["max_rel_10pct"]
            print(f"    Peak-norm max:  {metrics['max_peak_norm']:.6e}")
            print(f"    Rel max (>10%): {max_err:.6e} ({100*max_err:.4f}%)")
            print(f"    Rel mean(>10%): {metrics['mean_rel_10pct']:.6e}")

            all_results.append({
                "temperature_K": T,
                "weighting": wname,
                "max_rel_10pct": max_err,
                **metrics,
            })

    # Summary
    print(f"\n{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")
    print(f"  Grid: {n_groups}g x {n_angle_bins}a (non-aligned)")
    print(f"  {len(TEST_TEMPERATURES)} temperatures x 3 weightings = "
          f"{len(all_results)} comparisons\n")

    print(f"  {'Weighting':>10s}  {'T (K)':>12s}  {'max_rel(>10%)':>14s}")
    print(f"  {'─' * 10}  {'─' * 12}  {'─' * 14}")
    for r in all_results:
        print(f"  {r['weighting']:>10s}  {r['temperature_K']:12.4e}  "
              f"{r['max_rel_10pct']:14.6e} ({100*r['max_rel_10pct']:.4f}%)")

    overall_max = max(r["max_rel_10pct"] for r in all_results)
    print(f"\n  Overall max relative error: {overall_max:.6e} "
          f"({100*overall_max:.4f}%)")

    threshold = 0.15
    passed = overall_max < threshold
    print(f"  Pass criterion: max error < {100*threshold:.0f}%")
    if passed:
        print("  VERIFICATION PASSED")
    else:
        print("  VERIFICATION FAILED")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "verify_weighted_e2e_results.npz"
    np.savez_compressed(
        out_path,
        temperatures_K=np.array(TEST_TEMPERATURES),
        weightings=np.array(["uniform", "planck", "wien"]),
        max_rel_errors=np.array(
            [r["max_rel_10pct"] for r in all_results]
        ).reshape(len(TEST_TEMPERATURES), 3),
        boundaries_keV=boundaries_keV,
        n_angle_bins=n_angle_bins,
    )
    print(f"\n  Results saved to {out_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
