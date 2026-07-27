#!/usr/bin/env python3
"""Verify one (tidx, grid_type) case with the current uniform collapse code."""
import argparse, sys, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))
from collapse import _collapse_to_array, K_BOLTZ_KEV

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

DATA_DIR = ROOT / "docs" / "data"
N_TEMPS = 64
TEMPERATURES_K = np.geomspace(1000.0, 1e9, N_TEMPS)

def find_npz(tidx):
    wdir = DATA_DIR / "uniform"
    for p in sorted(wdir.glob("T*_*K.npz")):
        if p.name.startswith(f"T{tidx:03d}_"):
            return p
    raise FileNotFoundError(f"No .npz for T{tidx} in {wdir}")

def rel_error(sol, ref):
    peak = np.abs(ref).max()
    if peak == 0: return 0.0, 0.0
    sig = np.abs(ref) > 0.10 * peak
    if not sig.any(): return 0.0, 0.0
    rel = np.abs(sol - ref)[sig] / np.abs(ref[sig])
    return float(rel.max()), float(rel.mean())

def make_wf(bounds_erg):
    return cm.UniformWeightFunction()

def generate_aligned_grid(fine_bounds, seed=42):
    rng = np.random.default_rng(seed=seed)
    inner_idx = np.sort(rng.choice(np.arange(1, len(fine_bounds)-1), size=9, replace=False))
    return fine_bounds[np.concatenate([[0], inner_idx, [len(fine_bounds)-1]])]

def generate_random_grid(seed=42):
    rng = np.random.default_rng(seed=seed)
    log_min, log_max = np.log10(1e-5), np.log10(300.0)
    min_spacing = 0.5
    raw = np.sort(rng.uniform(0, 1, size=9))
    available = (log_max - log_min) - min_spacing * 10
    inner = log_min + min_spacing + raw * available + np.arange(9) * min_spacing
    return np.concatenate([[1e-5], 10**inner, [300.0]])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tidx", type=int, required=True)
    p.add_argument("--weighting", default="uniform", help="ignored, kept for backward compat")
    p.add_argument("--grid", choices=["aligned","random"], required=True)
    args = p.parse_args()

    tidx, gt = args.tidx, args.grid
    T = TEMPERATURES_K[tidx]
    n_angle = 12

    npz_path = find_npz(tidx)
    fine_bounds = np.load(npz_path)["boundaries_keV"]

    if gt == "aligned":
        coarse_bounds = generate_aligned_grid(fine_bounds)
    else:
        coarse_bounds = generate_random_grid()

    coarse_erg = (coarse_bounds * kev).tolist()

    collapsed = _collapse_to_array(
        npz_path.read_bytes(), coarse_bounds.tolist(), n_angle,
        temperature_K=T)

    wf = make_wf(coarse_erg)
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=coarse_erg, weight_function=wf,
        config=cm.MGIntegrationConfig(cutoff_ratio=1e-10, e_panel_order=96))
    kernel = cds.ComptonKernelSolver()
    t0 = time.time()
    exact = np.asarray(mg.compute_sigma_matrix(kernel, n_angle, T=T))
    elapsed = time.time() - t0

    mr, mn = rel_error(collapsed, exact)
    out_dir = ROOT / "output" / "current_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"T{tidx:03d}_uniform_{gt}.json"
    out.write_text(json.dumps(dict(
        tidx=tidx, weighting="uniform", grid=gt, T_K=T,
        max_rel_pct=100*mr, mean_rel_pct=100*mn, exact_time=elapsed)))
    print(f"T{tidx:03d} uniform {gt:7s}: max={100*mr:.4f}% mean={100*mn:.4f}% ({elapsed:.1f}s)")

if __name__ == "__main__":
    main()
