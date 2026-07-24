#!/usr/bin/env python3
"""
Diagnostic: investigate weighted collapse errors in detail.
Produces plots showing exact vs collapsed, error distribution, and cap_x effect.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))
from collapse import _collapse_to_array, _compute_spectral_weights, K_BOLTZ_KEV

DATA_DIR = ROOT / "docs" / "data"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)


def find_npz(tidx, weighting="uniform"):
    wdir = DATA_DIR / weighting
    for path in sorted(wdir.glob("T*_*K.npz")):
        if path.name.startswith(f"T{tidx:03d}_"):
            return path
    raise FileNotFoundError(f"No .npz for index {tidx} in {wdir}")


def generate_random_grid(seed=42):
    rng = np.random.default_rng(seed=seed)
    log_min, log_max = np.log10(1e-5), np.log10(300.0)
    min_spacing = 0.5
    raw = np.sort(rng.uniform(0, 1, size=9))
    available = (log_max - log_min) - min_spacing * 10
    inner = log_min + min_spacing + raw * available + np.arange(9) * min_spacing
    boundaries_keV = np.concatenate([[1e-5], 10**inner, [300.0]])
    return boundaries_keV, 12


def compute_exact(bounds_keV, n_angle, T, wf):
    bounds_erg = bounds_keV * kev
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds_erg.tolist(),
        weight_function=wf,
        config=cm.MGIntegrationConfig(cutoff_ratio=1e-10),
    )
    kernel = cds.ComptonKernelSolver()
    return np.asarray(mg.compute_sigma_matrix(kernel, n_angle, T=T))


def main():
    bounds_keV, n_angle = generate_random_grid()
    n_groups = len(bounds_keV) - 1

    test_cases = [
        (5, "2994 K"),
        (25, "2.4e5 K"),
        (55, "1.73e8 K"),
    ]

    N_TEMPS = 64
    TEMPERATURES_K = np.geomspace(1000.0, 1e9, N_TEMPS)

    fig_main, axes_main = plt.subplots(3, 3, figsize=(18, 14))
    fig_main.suptitle("Exact vs Collapsed: Angle-summed matrix (diagonal slice)", fontsize=14)

    fig_err, axes_err = plt.subplots(3, 3, figsize=(18, 14))
    fig_err.suptitle("Element-wise relative error distribution", fontsize=14)

    fig_scatter, axes_scatter = plt.subplots(3, 3, figsize=(18, 14))
    fig_scatter.suptitle("Exact vs Collapsed: all elements scatter plot", fontsize=14)

    for row, (tidx, label) in enumerate(test_cases):
        T = TEMPERATURES_K[tidx]
        npz_path = find_npz(tidx)
        npz_bytes = npz_path.read_bytes()
        kT_keV = K_BOLTZ_KEV * T

        print(f"\n{'='*60}")
        print(f"T[{tidx}] = {T:.4e} K,  kT = {kT_keV:.4e} keV")
        print(f"{'='*60}")

        # Show fine-group Planck weights
        fine_bounds = np.load(io.BytesIO(npz_bytes))["boundaries_keV"]
        fine_planck_w = _compute_spectral_weights(fine_bounds, "planck", kT_keV)
        nonzero_mask = fine_planck_w > 0
        print(f"  Fine groups with nonzero Planck weight: {nonzero_mask.sum()}/{len(fine_planck_w)}")
        if nonzero_mask.any():
            print(f"  Planck weight range: [{fine_planck_w[nonzero_mask].min():.4e}, {fine_planck_w.max():.4e}]")
            peak_idx = np.argmax(fine_planck_w)
            mid_e = 0.5 * (fine_bounds[peak_idx] + fine_bounds[peak_idx + 1])
            print(f"  Peak Planck weight at fine group {peak_idx}, E ~ {mid_e:.4e} keV (E/kT ~ {mid_e/kT_keV:.1f})")

        # Coarse group Planck weights
        coarse_planck_w = _compute_spectral_weights(bounds_keV, "planck", kT_keV)
        print(f"  Coarse Planck weights: {coarse_planck_w}")

        bounds_erg_coarse = (bounds_keV * kev).tolist()
        for col, (wname, wf) in enumerate([
            ("uniform", cm.UniformWeightFunction()),
            ("planck", cm.PlanckWeightFunction(cap_x=25.0, group_boundaries=bounds_erg_coarse)),
            ("wien", cm.WienWeightFunction(group_boundaries=bounds_erg_coarse)),
        ]):
            print(f"\n  [{wname}]")
            exact = compute_exact(bounds_keV, n_angle, T, wf)
            collapsed = _collapse_to_array(
                npz_bytes, bounds_keV.tolist(), n_angle,
                weighting=wname,
                temperature_K=T if wname != "uniform" else None,
            )

            peak = np.abs(exact).max()
            diff = np.abs(exact - collapsed)

            # Find elements > 10% of peak
            sig_mask = np.abs(exact) > 0.10 * peak
            n_sig = sig_mask.sum()
            if n_sig > 0:
                rel_err = diff[sig_mask] / np.abs(exact[sig_mask])
                worst_idx = np.unravel_index(np.argmax(diff * sig_mask), exact.shape)
                print(f"    Peak: {peak:.4e}")
                print(f"    Elements > 10% peak: {n_sig}")
                print(f"    Max rel error (>10%): {rel_err.max():.6f} ({100*rel_err.max():.2f}%)")
                print(f"    Mean rel error (>10%): {rel_err.mean():.6f}")
                print(f"    Worst element at {worst_idx}: exact={exact[worst_idx]:.4e}, collapsed={collapsed[worst_idx]:.4e}")

                # Check: are worst elements truly significant?
                print(f"    Worst element is {100*np.abs(exact[worst_idx])/peak:.1f}% of peak")

            # ── Plot 1: Diagonal slice (angle-summed) ──
            ax = axes_main[row, col]
            exact_diag = exact.sum(axis=2).diagonal()
            coll_diag = collapsed.sum(axis=2).diagonal()
            x = np.arange(n_groups)
            ax.semilogy(x, np.abs(exact_diag), 'o-', label='Exact', markersize=4)
            ax.semilogy(x, np.abs(coll_diag), 's--', label='Collapsed', markersize=4)
            ax.set_title(f"{wname}, T={label}")
            ax.set_xlabel("Group index (g→g)")
            ax.set_ylabel("|σ(g→g)| (angle-summed)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── Plot 2: Relative error histogram ──
            ax2 = axes_err[row, col]
            all_sig = np.abs(exact) > 0.01 * peak
            if all_sig.any():
                rel_all = diff[all_sig] / np.abs(exact[all_sig])
                rel_all_clipped = np.clip(rel_all, 0, 2.0)
                ax2.hist(rel_all_clipped, bins=50, edgecolor='black', linewidth=0.5)
                ax2.axvline(0.10, color='red', ls='--', label='10% threshold')
                ax2.set_title(f"{wname}, T={label}")
                ax2.set_xlabel("Relative error")
                ax2.set_ylabel("Count (elements > 1% peak)")
                ax2.legend(fontsize=8)

            # ── Plot 3: Scatter exact vs collapsed ──
            ax3 = axes_scatter[row, col]
            e_flat = exact.ravel()
            c_flat = collapsed.ravel()
            mask = np.abs(e_flat) > 0.001 * peak
            ax3.scatter(e_flat[mask], c_flat[mask], s=2, alpha=0.3)
            lims = [min(e_flat[mask].min(), c_flat[mask].min()),
                    max(e_flat[mask].max(), c_flat[mask].max())]
            ax3.plot(lims, lims, 'r-', lw=1, label='y=x')
            ax3.set_title(f"{wname}, T={label}")
            ax3.set_xlabel("Exact")
            ax3.set_ylabel("Collapsed")
            ax3.legend(fontsize=8)
            ax3.set_aspect('equal', adjustable='box')

    fig_main.tight_layout()
    fig_main.savefig(OUT_DIR / "diag_exact_vs_collapsed.png", dpi=150)
    print(f"\n  Saved {OUT_DIR / 'diag_exact_vs_collapsed.png'}")

    fig_err.tight_layout()
    fig_err.savefig(OUT_DIR / "diag_error_distribution.png", dpi=150)
    print(f"  Saved {OUT_DIR / 'diag_error_distribution.png'}")

    fig_scatter.tight_layout()
    fig_scatter.savefig(OUT_DIR / "diag_scatter.png", dpi=150)
    print(f"  Saved {OUT_DIR / 'diag_scatter.png'}")

    # ── Cap effect test ──
    print(f"\n{'='*60}")
    print("CAP EFFECT TEST: Planck with cap_x=25 vs no cap")
    print(f"{'='*60}")

    for tidx, label in test_cases:
        T = TEMPERATURES_K[tidx]
        npz_path = find_npz(tidx)
        npz_bytes = npz_path.read_bytes()

        bounds_erg_c = (bounds_keV * kev).tolist()
        exact_cap25 = compute_exact(bounds_keV, n_angle, T, cm.PlanckWeightFunction(cap_x=25.0, group_boundaries=bounds_erg_c))
        exact_nocap = compute_exact(bounds_keV, n_angle, T, cm.PlanckWeightFunction(cap_x=1e30, group_boundaries=bounds_erg_c))

        peak = np.abs(exact_cap25).max()
        diff = np.abs(exact_cap25 - exact_nocap)
        if peak > 0:
            max_diff = (diff / peak).max()
            print(f"  T[{tidx}]={T:.4e} K: max |cap25 - nocap| / peak = {max_diff:.6e} ({100*max_diff:.4f}%)")
        else:
            print(f"  T[{tidx}]={T:.4e} K: peak = 0")

    plt.close('all')


if __name__ == "__main__":
    main()
