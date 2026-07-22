#!/usr/bin/env python3
"""
End-to-end comparison: website collapse/interpolation vs exact vs Monte Carlo.

Tests on a non-aligned 32x32x1 grid at several temperatures.
Uses 1e8 MC samples for low statistical noise.

Requires:
  - HTTP server running on port 8791 serving docs/
  - ComptonMatrixExact installed in the venv

Usage (via SLURM):
  sbatch scripts/submit_verify_mc.sh
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
MC_SAMPLES = 100_000_000  # 1e8


def fetch_manifest():
    with urllib.request.urlopen(f"{BASE_URL}/data/manifest.json") as r:
        return json.loads(r.read())


def fetch_npz(filename: str) -> bytes:
    with urllib.request.urlopen(f"{BASE_URL}/data/{filename}") as r:
        return r.read()


def find_brackets(manifest, T_target):
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


def generate_grid(manifest):
    """Non-aligned 32-group grid using geometric midpoints of fine boundaries."""
    fine_bounds = np.array(manifest["boundaries_keV"])
    midpoints = np.sqrt(fine_bounds[:-1] * fine_bounds[1:])
    indices = np.linspace(0, 127, 33).astype(int)
    coarse_bounds = np.concatenate([[fine_bounds[0]], midpoints[indices[1:-1]], [fine_bounds[-1]]])
    coarse_bounds = np.sort(np.unique(coarse_bounds))
    return coarse_bounds


def compute_website_result(manifest, T, coarse_bounds, n_angle_bins):
    """Simulate what the website does: fetch .npz, collapse/interpolate."""
    idx_lo, idx_hi, is_exact = find_brackets(manifest, T)
    entry_lo = manifest["temperatures"][idx_lo]

    if is_exact:
        npz = fetch_npz(entry_lo["file"])
        return _collapse_to_array(npz, coarse_bounds.tolist(), n_angle_bins)
    else:
        entry_hi = manifest["temperatures"][idx_hi]
        npz_lo = fetch_npz(entry_lo["file"])
        npz_hi = fetch_npz(entry_hi["file"])
        T_lo = entry_lo["temperature_K"]
        T_hi = entry_hi["temperature_K"]
        return collapse_interp(
            npz_lo, npz_hi, T_lo, T_hi, T,
            coarse_bounds.tolist(), n_angle_bins,
        )


def compute_exact(coarse_bounds, n_angle_bins, T):
    """Compute exact deterministic matrix."""
    boundaries_erg = coarse_bounds * kev
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries_erg.tolist(),
        weight_function=cm.UniformWeightFunction(),
        config=cm.MGIntegrationConfig(cutoff_ratio=1e-10),
    )
    kernel = cds.ComptonKernelSolver()
    t0 = time.time()
    sigma = np.asarray(mg.compute_sigma_matrix(kernel, n_angle_bins, T=T, Ne=1.0))
    elapsed = time.time() - t0
    return sigma, elapsed


def compute_mc(coarse_bounds, n_angle_bins, T, seed=42):
    """Compute Monte Carlo matrix with MC_SAMPLES samples."""
    boundaries_erg = coarse_bounds * kev
    mc = cm.ComptonMonteCarloKernel(
        energy_group_boundaries=boundaries_erg.tolist(),
        weight_function=cm.UniformWeightFunction(),
        config=cm.MCIntegrationConfig(
            num_samples=MC_SAMPLES,
            seed=seed,
            discard_out_of_grid=True,
        ),
    )
    t0 = time.time()
    sigma = np.asarray(mc.compute_sigma_matrix(n_angle_bins, T=T, Ne=1.0))
    elapsed = time.time() - t0
    return sigma, elapsed


def compare(reference, test, label, peak=None):
    """Compare test against reference, report metrics."""
    if peak is None:
        peak = np.max(np.abs(reference))
    if peak == 0:
        print(f"    {label}: reference is all zeros")
        return {}

    abs_diff = np.abs(reference - test)
    metrics = {
        "peak_norm_max": (abs_diff / peak).max(),
        "peak_norm_mean": (abs_diff / peak).mean(),
    }

    mask = np.abs(reference) > 0.10 * peak
    n_sig = np.count_nonzero(mask)
    if n_sig > 0:
        rel = abs_diff[mask] / np.abs(reference[mask])
        metrics["max_rel_10pct"] = rel.max()
        metrics["mean_rel_10pct"] = rel.mean()
        metrics["n_sig"] = n_sig
    else:
        metrics["max_rel_10pct"] = 0.0
        metrics["mean_rel_10pct"] = 0.0
        metrics["n_sig"] = 0

    print(f"    {label}:")
    print(f"      Peak-norm max: {metrics['peak_norm_max']:.4e}  |  "
          f"Rel err (>10% peak): max={metrics['max_rel_10pct']:.4e} "
          f"({100*metrics['max_rel_10pct']:.2f}%), "
          f"mean={metrics['mean_rel_10pct']:.4e} "
          f"({100*metrics['mean_rel_10pct']:.2f}%)  "
          f"[{n_sig} elements]")
    return metrics


def main():
    print("=" * 75)
    print("Three-way comparison: Website (collapse+interp) vs Exact vs Monte Carlo")
    print(f"MC samples: {MC_SAMPLES:.0e}")
    print("=" * 75)

    manifest = fetch_manifest()
    coarse_bounds = generate_grid(manifest)
    n_groups = len(coarse_bounds) - 1
    n_angle_bins = 1

    print(f"\nGrid: {n_groups} energy groups, {n_angle_bins} angle bin (non-aligned)")
    print(f"  E range: [{coarse_bounds[0]:.4e}, {coarse_bounds[-1]:.4e}] keV")

    # Test temperatures: mix of stored and interpolated
    test_cases = [
        (2000.0, "interpolated"),
        (1e5, "stored (T[21])"),
        (1e7, "stored (T[42])"),
        (2e8, "interpolated"),
    ]

    all_results = []

    for T, desc in test_cases:
        print(f"\n{'━' * 75}")
        print(f"  T = {T:.4e} K  ({desc})")
        print(f"{'━' * 75}")

        # Website result
        print(f"  [1] Website collapse/interpolation...")
        website = compute_website_result(manifest, T, coarse_bounds, n_angle_bins)
        website_2d = website[:, :, 0]
        print(f"      Done. Shape: {website_2d.shape}")

        # Exact deterministic
        print(f"  [2] Exact deterministic computation...")
        exact_3d, t_exact = compute_exact(coarse_bounds, n_angle_bins, T)
        exact_2d = exact_3d[:, :, 0] if exact_3d.ndim == 3 else exact_3d
        print(f"      Done ({t_exact:.1f}s). Shape: {exact_2d.shape}")

        # Monte Carlo
        print(f"  [3] Monte Carlo ({MC_SAMPLES:.0e} samples)...")
        mc_3d, t_mc = compute_mc(coarse_bounds, n_angle_bins, T, seed=12345)
        mc_2d = mc_3d[:, :, 0] if mc_3d.ndim == 3 else mc_3d
        print(f"      Done ({t_mc:.1f}s). Shape: {mc_2d.shape}")

        # Comparisons (all relative to exact as ground truth)
        peak = np.max(np.abs(exact_2d))
        print(f"\n  Comparisons (reference = exact deterministic, peak = {peak:.4e}):")
        m_web = compare(exact_2d, website_2d, "Website vs Exact", peak)
        m_mc = compare(exact_2d, mc_2d, "MC vs Exact", peak)
        m_web_mc = compare(exact_2d, website_2d, "Website vs MC (using MC as ref)",
                           np.max(np.abs(mc_2d)))
        # Also direct website-MC comparison
        print(f"\n  Direct comparison (website vs MC):")
        compare(mc_2d, website_2d, "Website vs MC", np.max(np.abs(mc_2d)))

        all_results.append({
            "T": T,
            "desc": desc,
            "website_vs_exact": m_web,
            "mc_vs_exact": m_mc,
        })

    # Summary table
    print(f"\n{'━' * 75}")
    print("SUMMARY TABLE (max relative error on dominant elements, >10% of peak)")
    print(f"{'━' * 75}")
    print(f"  {'T (K)':>12s}  {'Type':>15s}  {'Website vs Exact':>16s}  {'MC vs Exact':>12s}")
    print(f"  {'─'*12}  {'─'*15}  {'─'*16}  {'─'*12}")
    for r in all_results:
        web_err = r["website_vs_exact"].get("max_rel_10pct", 0)
        mc_err = r["mc_vs_exact"].get("max_rel_10pct", 0)
        print(f"  {r['T']:12.4e}  {r['desc']:>15s}  "
              f"{100*web_err:15.2f}%  {100*mc_err:11.2f}%")

    print(f"\n  Note: MC statistical error decreases as 1/sqrt(N).")
    print(f"  With {MC_SAMPLES:.0e} samples, expected MC stat error ~ {1/np.sqrt(MC_SAMPLES):.2e}")
    print("\nDone.")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "verify_mc_comparison_results.npz"
    np.savez_compressed(
        out_path,
        temperatures=[r["T"] for r in all_results],
        website_max_rel=[r["website_vs_exact"].get("max_rel_10pct", 0) for r in all_results],
        mc_max_rel=[r["mc_vs_exact"].get("max_rel_10pct", 0) for r in all_results],
        coarse_bounds=coarse_bounds,
        n_angle_bins=n_angle_bins,
        mc_samples=MC_SAMPLES,
    )
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
