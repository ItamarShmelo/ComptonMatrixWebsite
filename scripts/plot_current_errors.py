#!/usr/bin/env python3
"""Generate aligned vs random grid error plots with the current collapse code."""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))
from collapse import _collapse_to_array, K_BOLTZ_KEV

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = ROOT / "docs" / "data"
OUT_DIR = ROOT / "output"
N_TEMPS = 64
TEMPERATURES_K = np.geomspace(1000.0, 1e9, N_TEMPS)
TEST_INDICES = [5, 15, 25, 35, 45, 55]


def find_npz(tidx):
    wdir = DATA_DIR / "uniform"
    for p in sorted(wdir.glob("T*_*K.npz")):
        if p.name.startswith(f"T{tidx:03d}_"):
            return p
    raise FileNotFoundError(f"No .npz for T{tidx} in {wdir}")


def rel_error(sol, ref):
    peak = np.abs(ref).max()
    if peak == 0:
        return 0.0, 0.0
    sig = np.abs(ref) > 0.10 * peak
    if not sig.any():
        return 0.0, 0.0
    rel = np.abs(sol - ref)[sig] / np.abs(ref[sig])
    return float(rel.max()), float(rel.mean())


def generate_aligned_grid(fine_bounds, seed=42):
    rng = np.random.default_rng(seed=seed)
    inner_idx = np.sort(rng.choice(np.arange(1, len(fine_bounds) - 1), size=9, replace=False))
    return fine_bounds[np.concatenate([[0], inner_idx, [len(fine_bounds) - 1]])]


def generate_random_grid(seed=42):
    rng = np.random.default_rng(seed=seed)
    log_min, log_max = np.log10(1e-5), np.log10(300.0)
    min_spacing = 0.5
    raw = np.sort(rng.uniform(0, 1, size=9))
    available = (log_max - log_min) - min_spacing * 10
    inner = log_min + min_spacing + raw * available + np.arange(9) * min_spacing
    return np.concatenate([[1e-5], 10**inner, [300.0]])


def main():
    n_angle = 12
    kernel = cds.ComptonKernelSolver()
    results = {}

    for gt in ["aligned", "random"]:
        results[gt] = {"temps": [], "max_err": [], "mean_err": []}

    total = len(TEST_INDICES) * 2
    done = 0
    t_start = time.time()

    for tidx in TEST_INDICES:
        T = TEMPERATURES_K[tidx]
        npz_path = find_npz(tidx)
        fine_bounds = np.load(npz_path)["boundaries_keV"]
        npz_bytes = npz_path.read_bytes()

        for gt in ["aligned", "random"]:
            if gt == "aligned":
                coarse_bounds = generate_aligned_grid(fine_bounds)
            else:
                coarse_bounds = generate_random_grid()

            coarse_erg = (coarse_bounds * kev).tolist()

            collapsed = _collapse_to_array(
                npz_bytes, coarse_bounds.tolist(), n_angle,
                temperature_K=T,
            )

            wf = cm.UniformWeightFunction()
            mg = cm.ComptonMultigroupKernel(
                energy_group_boundaries=coarse_erg,
                weight_function=wf,
                config=cm.MGIntegrationConfig(cutoff_ratio=1e-10, e_panel_order=96),
            )
            exact = np.asarray(mg.compute_sigma_matrix(kernel, n_angle, T=T))

            mr, mn = rel_error(collapsed, exact)
            results[gt]["temps"].append(T)
            results[gt]["max_err"].append(100 * mr)
            results[gt]["mean_err"].append(100 * mn)

            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] T{tidx:03d} {gt:7s}: "
                  f"max={100*mr:.2f}% mean={100*mn:.2f}%  (ETA {eta/60:.0f}m)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, gt in enumerate(["aligned", "random"]):
        ax = axes[col]
        r = results[gt]
        ax.semilogy(r["temps"], r["max_err"], "o-",
                    color="#2196F3", label="max error", linewidth=2, markersize=8)
        ax.semilogy(r["temps"], r["mean_err"], "s--",
                    color="#F44336", label="mean error", linewidth=2, markersize=8)

        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xlabel("Temperature (K)", fontsize=11)
        ax.set_title(f"{gt.capitalize()} Grid — Relative Error (%)", fontsize=13)
        ax.axhline(1.0, color="green", ls=":", alpha=0.6, label="1%")
        ax.axhline(5.0, color="orange", ls=":", alpha=0.6, label="5%")

    axes[0].set_ylabel("Rel. Error on Significant Elements (%)", fontsize=11)

    fig.suptitle("Uniform Collapse Error: Aligned vs Random Coarse Grids\n"
                 "(10 coarse groups, 12 angles, e_panel=96 exact reference)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "uniform_aligned_vs_random.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
