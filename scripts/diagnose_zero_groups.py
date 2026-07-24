#!/usr/bin/env python3
"""
Diagnose why Planck-collapsed groups 4-8 are zero at T=2994K.
Check: is the stored data wrong, or is the collapse losing information?
"""
import io, sys
from pathlib import Path
import numpy as np

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import kev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))
from collapse import _collapse_to_array, _compute_spectral_weights, K_BOLTZ_KEV, _overlap_matrix

DATA_DIR = ROOT / "docs" / "data"

def find_npz(tidx, weighting="uniform"):
    wdir = DATA_DIR / weighting
    for p in sorted(wdir.glob("T*_*K.npz")):
        if p.name.startswith(f"T{tidx:03d}_"):
            return p

def generate_random_grid(seed=42):
    rng = np.random.default_rng(seed=seed)
    log_min, log_max = np.log10(1e-5), np.log10(300.0)
    min_spacing = 0.5
    raw = np.sort(rng.uniform(0, 1, size=9))
    available = (log_max - log_min) - min_spacing * 10
    inner = log_min + min_spacing + raw * available + np.arange(9) * min_spacing
    return np.concatenate([[1e-5], 10**inner, [300.0]]), 12

def compute_exact(bounds_keV, n_angle, T, wf):
    bounds_erg = bounds_keV * kev
    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=bounds_erg.tolist(),
        weight_function=wf,
        config=cm.MGIntegrationConfig(cutoff_ratio=1e-10),
    )
    kernel = cds.ComptonKernelSolver()
    return np.asarray(mg.compute_sigma_matrix(kernel, n_angle, T=T))

bounds_keV, n_angle = generate_random_grid()
n_groups = len(bounds_keV) - 1
T = np.geomspace(1000, 1e9, 64)[5]  # T=2994 K
kT_keV = K_BOLTZ_KEV * T
npz_path = find_npz(5)
npz_bytes = npz_path.read_bytes()
fine_data = np.load(io.BytesIO(npz_bytes))
fine_sigma = fine_data["sigma_matrix"]
fine_bounds = fine_data["boundaries_keV"]

print(f"T = {T:.4e} K,  kT = {kT_keV:.4e} keV")
print(f"Coarse grid: {bounds_keV}")
print(f"Fine grid: {fine_bounds.shape[0]-1} groups, {fine_sigma.shape[2]} angle bins")
print()

# ── 1. Exact solver with all three weightings ──
print("="*70)
print("EXACT SOLVER: diagonal elements σ(I→I), angle-summed")
print("="*70)
exact_uni = compute_exact(bounds_keV, n_angle, T, cm.UniformWeightFunction())
bounds_erg_fine = (np.geomspace(1e-5, 300.0, 129) * kev).tolist()
exact_pla = compute_exact(bounds_keV, n_angle, T, cm.PlanckWeightFunction(cap_x=25.0, group_boundaries=(bounds_keV * kev).tolist()))
exact_wie = compute_exact(bounds_keV, n_angle, T, cm.WienWeightFunction(group_boundaries=(bounds_keV * kev).tolist()))

print(f"{'Group':>5s}  {'E range (keV)':>24s}  {'E/kT range':>18s}  {'Uniform':>12s}  {'Planck':>12s}  {'Wien':>12s}")
for I in range(n_groups):
    e_lo, e_hi = bounds_keV[I], bounds_keV[I+1]
    x_lo, x_hi = e_lo/kT_keV, e_hi/kT_keV
    uni_val = exact_uni[I,I,:].sum()
    pla_val = exact_pla[I,I,:].sum()
    wie_val = exact_wie[I,I,:].sum()
    print(f"  {I:3d}  [{e_lo:10.3e}, {e_hi:10.3e}]  [{x_lo:7.1f}, {x_hi:7.1f}]  "
          f"{uni_val:12.4e}  {pla_val:12.4e}  {wie_val:12.4e}")

print()
print("Key observation: Planck exact ≈ Uniform exact for ALL groups,")
print("even where E/kT >> 25. The Planck function cancels in the ratio.")

# ── 2. Check Planck weights ──
print()
print("="*70)
print("PLANCK WEIGHTS: fine groups and coarse groups")
print("="*70)
fine_planck = _compute_spectral_weights(fine_bounds, "planck", kT_keV)
coarse_planck = _compute_spectral_weights(bounds_keV, "planck", kT_keV)
f_e = _overlap_matrix(fine_bounds, bounds_keV)
coarse_from_fine = np.einsum("gI,g->I", f_e, fine_planck)

print(f"{'Group':>5s}  {'Coarse Planck w (direct)':>24s}  {'Coarse w (from fine)':>20s}  {'Both zero?':>10s}")
for I in range(n_groups):
    direct = coarse_planck[I]
    from_fine = coarse_from_fine[I]
    both_zero = "YES" if direct == 0 and from_fine == 0 else "no"
    print(f"  {I:3d}  {direct:24.6e}  {from_fine:20.6e}  {both_zero:>10s}")

# ── 3. What the collapse produces ──
print()
print("="*70)
print("COLLAPSED RESULT: diagonal elements σ(I→I), angle-summed")
print("="*70)
coll_uni = _collapse_to_array(npz_bytes, bounds_keV.tolist(), n_angle, weighting="uniform")
coll_pla = _collapse_to_array(npz_bytes, bounds_keV.tolist(), n_angle, weighting="planck", temperature_K=T)
coll_wie = _collapse_to_array(npz_bytes, bounds_keV.tolist(), n_angle, weighting="wien", temperature_K=T)

print(f"{'Group':>5s}  {'Uni exact':>12s}  {'Uni coll':>12s}  {'Pla exact':>12s}  {'Pla coll':>12s}  {'Pla w':>12s}  {'Problem':>8s}")
for I in range(n_groups):
    ue = exact_uni[I,I,:].sum()
    uc = coll_uni[I,I,:].sum()
    pe = exact_pla[I,I,:].sum()
    pc = coll_pla[I,I,:].sum()
    w = coarse_from_fine[I]
    problem = "ZERO w" if w == 0 or pc == 0 else ""
    print(f"  {I:3d}  {ue:12.4e}  {uc:12.4e}  {pe:12.4e}  {pc:12.4e}  {w:12.4e}  {problem:>8s}")

# ── 4. Root cause ──
print()
print("="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)
print()
print("The exact solver computes:")
print("  σ_planck(I) = ∫ w(E) K(E,...) dE / ∫ w(E) dE")
print()
print("For groups where E >> kT, w(E) = E^3 exp(-E/kT).")
print("The exp(-E/kT) appears in BOTH numerator and denominator,")
print("so it CANCELS. The ratio reduces to an E^3-weighted average")
print("of K, which is close to the uniform average.")
print()
print("But in the collapse, we compute w_g as a number first.")
print("When E/kT ~ 1000, exp(-1000) underflows to 0.0 in float64.")
print("So w_g = 0 → 0/0 → guard sets result to 0.")
print()
print("The stored data is CORRECT. The collapse procedure loses")
print("information because double-precision can't represent the")
print("intermediate Planck weights at E >> kT.")
