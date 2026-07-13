#!/usr/bin/env python3
"""
Generate comparison plots of deterministic vs Monte Carlo Compton matrices.

Produces per-temperature pages with:
  - Heatmaps of det and MC mean sigma matrices (log scale)
  - Heatmap of deviation in SEM units
  - Diagonal profile comparison
  - Row profiles at selected groups through the peak
  - Relative error distribution on significant entries

Usage:
  python scripts/plot_validation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

N_GROUPS = 128
N_SEEDS = 10
N_TEMPS = 64

DET_DIR = ROOT / "output" / "tables"
MC_DIR = ROOT / "output" / "mc_tables"
PLOT_DIR = ROOT / "output" / "plots"

TEMPERATURES_K = np.geomspace(1000.0, 1e9, N_TEMPS)
E_MIN_KEV = 1e-5
E_MAX_KEV = 300.0
BOUNDARIES_KEV = np.geomspace(E_MIN_KEV, E_MAX_KEV, N_GROUPS + 1)
GROUP_CENTERS_KEV = np.sqrt(BOUNDARIES_KEV[:-1] * BOUNDARIES_KEV[1:])


def load_data(tidx: int):
    det_path = sorted(DET_DIR.glob(f"T{tidx:03d}_*.npz"))[0]
    det_data = np.load(det_path)
    det_sigma = det_data["sigma_matrix"]

    mc_stack = np.array([
        np.load(MC_DIR / f"mc_T{tidx:03d}_seed{s}.npz")["sigma_matrix"]
        for s in range(N_SEEDS)
    ])
    mc_mean = mc_stack.mean(axis=0)
    mc_std = mc_stack.std(axis=0, ddof=1)
    mc_sem = mc_std / np.sqrt(N_SEEDS)

    return det_sigma, mc_mean, mc_std, mc_sem


def plot_temperature(tidx: int, det_sigma, mc_mean, mc_std, mc_sem):
    T = TEMPERATURES_K[tidx]

    diff = det_sigma - mc_mean
    abs_diff = np.abs(diff)
    testable = (mc_mean != 0) & (mc_sem > 0)
    nsigma = np.zeros_like(diff)
    nsigma[testable] = diff[testable] / mc_sem[testable]

    sig_mask = np.abs(mc_mean) > np.abs(mc_mean).max() * 1e-6
    rel_err = np.full_like(diff, np.nan)
    rel_err[sig_mask] = abs_diff[sig_mask] / np.abs(mc_mean[sig_mask])

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(f"T-index {tidx:03d},  T = {T:.3e} K  ({T / 1.16045e7:.4f} keV)",
                 fontsize=14, fontweight="bold")

    extent = [np.log10(E_MIN_KEV), np.log10(E_MAX_KEV),
              np.log10(E_MAX_KEV), np.log10(E_MIN_KEV)]

    def _log_heatmap(ax, data, title):
        log_data = np.full_like(data, np.nan)
        pos = data > 0
        if pos.any():
            log_data[pos] = np.log10(data[pos])
        im = ax.imshow(log_data, extent=extent, aspect="equal", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("log10(E' / keV)")
        ax.set_ylabel("log10(E / keV)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="log10(|sigma|)")

    # --- (0,0) Deterministic sigma heatmap ---
    _log_heatmap(axes[0, 0], np.abs(det_sigma), "Deterministic |sigma|")

    # --- (0,1) MC mean sigma heatmap ---
    _log_heatmap(axes[0, 1], np.abs(mc_mean), "MC mean |sigma|")

    # --- (0,2) Deviation in SEM units ---
    ax = axes[0, 2]
    plot_nsigma = nsigma.copy()
    plot_nsigma[~testable] = np.nan
    vlim = np.nanmax(np.abs(plot_nsigma)) if np.any(np.isfinite(plot_nsigma)) else 3.0
    vlim = max(vlim, 3.0)
    im = ax.imshow(plot_nsigma, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   extent=extent, aspect="equal")
    ax.set_title("(Det - MC mean) / SEM")
    ax.set_xlabel("log10(E' / keV)")
    ax.set_ylabel("log10(E / keV)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="SEM units")

    # --- (1,0) Diagonal profile ---
    ax = axes[1, 0]
    diag_det = np.diag(det_sigma)
    diag_mc = np.diag(mc_mean)
    diag_std = np.diag(mc_std)
    nonzero = diag_mc != 0
    ax.plot(GROUP_CENTERS_KEV[nonzero], diag_det[nonzero], "b-", lw=1.5, label="Det")
    ax.plot(GROUP_CENTERS_KEV[nonzero], diag_mc[nonzero], "r--", lw=1.5, label="MC mean")
    ax.fill_between(GROUP_CENTERS_KEV[nonzero],
                    diag_mc[nonzero] - 2 * diag_std[nonzero],
                    diag_mc[nonzero] + 2 * diag_std[nonzero],
                    alpha=0.2, color="red", label="MC +/- 2 std")
    ax.set_xscale("log")
    ax.set_title("Diagonal sigma(g,g)")
    ax.set_xlabel("E (keV)")
    ax.set_ylabel("sigma")
    ax.legend(fontsize=8)

    # --- (1,1) Row profiles at selected groups ---
    ax = axes[1, 1]
    diag_peak = np.argmax(np.abs(np.diag(det_sigma)))
    rows_to_plot = sorted(set([
        max(0, diag_peak - 5), diag_peak, min(N_GROUPS - 1, diag_peak + 5),
        N_GROUPS // 4, 3 * N_GROUPS // 4
    ]))
    for row in rows_to_plot:
        row_det = det_sigma[row, :]
        row_mc = mc_mean[row, :]
        mask = row_mc != 0
        if mask.sum() == 0:
            continue
        e_center = GROUP_CENTERS_KEV[row]
        ax.plot(GROUP_CENTERS_KEV[mask], row_det[mask], "-",
                lw=1.2, label=f"Det g={row} ({e_center:.2e} keV)")
        ax.plot(GROUP_CENTERS_KEV[mask], row_mc[mask], "--", lw=1.0)
    ax.set_xscale("log")
    ax.set_title("Row profiles: sigma(g, g')")
    ax.set_xlabel("E' (keV)")
    ax.set_ylabel("sigma")
    ax.legend(fontsize=6, ncol=2)

    # --- (1,2) Relative error histogram on significant entries ---
    ax = axes[1, 2]
    valid_rel = rel_err[~np.isnan(rel_err)]
    if len(valid_rel) > 0:
        log_rel = np.log10(np.clip(valid_rel, 1e-10, None))
        ax.hist(log_rel, bins=50, color="steelblue", edgecolor="black", lw=0.5)
        ax.axvline(np.log10(0.01), color="green", ls="--", lw=1.5, label="1%")
        ax.axvline(np.log10(0.05), color="orange", ls="--", lw=1.5, label="5%")
        ax.axvline(np.log10(0.10), color="red", ls="--", lw=1.5, label="10%")
        ax.legend(fontsize=8)
    ax.set_title("Relative error distribution (significant entries)")
    ax.set_xlabel("log10(|det - mc_mean| / |mc_mean|)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    return fig


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    difficult_indices = [0, 1, 5, 10, 15, 20, 34, 43, 51, 52, 59, 63]

    for tidx in difficult_indices:
        print(f"Plotting T-index {tidx} (T = {TEMPERATURES_K[tidx]:.2e} K) ...")
        det_sigma, mc_mean, mc_std, mc_sem = load_data(tidx)
        fig = plot_temperature(tidx, det_sigma, mc_mean, mc_std, mc_sem)
        out_path = PLOT_DIR / f"validation_T{tidx:03d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
