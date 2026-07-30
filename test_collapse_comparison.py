"""
Compare collapse output against ComptonMatrixExact direct computation.
Measures timing and quantifies error.
"""
import sys
import time

sys.path.insert(0, "/home/itamarg/workspace_current/ComptonMatrixExact/.venv/lib/python3.12/site-packages")
sys.path.insert(0, "/home/itamarg/workspace_current/ComptonMatrixWebsite/docs/py")

import numpy as np
import compton_matrix as cm
from compton_matrix._compton_differential_cross_section import ComptonKernelSolver
from compton_matrix._units import kev, kev_kelvin, k_boltz

import collapse as collapse_mod

KERNEL = ComptonKernelSolver()

test_file = "/home/itamarg/workspace_current/ComptonMatrixWebsite/docs/data/uniform/T032_1.115884e+06K.npz"
T_K = 1.115884e+06
T_kev = k_boltz * T_K / kev

with open(test_file, "rb") as f:
    npz_bytes = f.read()

data = np.load(test_file)
fine_bounds_keV = data["boundaries_keV"]
sigma_fine = data["sigma_matrix"]
G, _, M = sigma_fine.shape

print(f"Test: T = {T_K:.3e} K  (kT = {T_kev:.3f} keV)")
print(f"Fine grid: {G} groups x {M} angle bins")
print(f"Fine energy range: [{fine_bounds_keV[0]:.4e}, {fine_bounds_keV[-1]:.4e}] keV")
print()

# --- Test case: collapse from 128 -> 20 groups, 100 -> 10 angle bins ---
N_COARSE = 20
K_COARSE = 10
coarse_bounds_keV = np.geomspace(fine_bounds_keV[0], fine_bounds_keV[-1], N_COARSE + 1)
coarse_xi = np.linspace(-1.0, 1.0, K_COARSE + 1)

print(f"Coarse grid: {N_COARSE} groups x {K_COARSE} angle bins")
print("=" * 70)

# 1. Time the new vectorized collapse
n_runs = 5
times_collapse = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    result_collapse = collapse_mod._collapse_to_array(npz_bytes, list(coarse_bounds_keV), K_COARSE)
    times_collapse.append(time.perf_counter() - t0)

t_collapse = np.median(times_collapse)
print(f"\n--- COLLAPSE TIMING (median of {n_runs} runs) ---")
print(f"New vectorized collapse: {t_collapse:.4f} s")

# 2. Compute exact result using ComptonMatrixExact
print("\n--- COMPUTING EXACT REFERENCE (ComptonMatrixExact) ---")
coarse_bounds_erg = [b * kev for b in coarse_bounds_keV]

cfg = cm.MGIntegrationConfig(
    xi_order=12,
    xi_tail_order=12,
    ep_edge_order=12,
    ep_interior_order=12,
    e_panel_order=12,
    cutoff_ratio=None,
)
mg = cm.ComptonMultigroupKernel(
    energy_group_boundaries=coarse_bounds_erg,
    weight_function=cm.UniformWeightFunction(),
    config=cfg,
)

t0 = time.perf_counter()
result_exact = mg.compute_sigma_matrix(KERNEL, num_angle_bins=K_COARSE, T=T_K)
t_exact = time.perf_counter() - t0
print(f"ComptonMatrixExact direct computation: {t_exact:.2f} s")

# 3. Compare
print(f"\n--- ACCURACY (collapse vs exact) ---")
print(f"Collapse result shape: {result_collapse.shape}")
print(f"Exact result shape:    {result_exact.shape}")

diff = np.abs(result_collapse - result_exact)

# Use significance threshold: only compare elements > 0.1% of the max
peak = np.abs(result_exact).max()
sig_threshold = 1e-3 * peak
sig_mask = np.abs(result_exact) > sig_threshold
n_sig = sig_mask.sum()
n_total = result_exact.size

rel_err_sig = diff[sig_mask] / np.abs(result_exact[sig_mask])

print(f"\nPeak matrix value:     {peak:.6e}")
print(f"Significance threshold: {sig_threshold:.6e} (0.1% of peak)")
print(f"Significant elements:   {n_sig} / {n_total} ({100*n_sig/n_total:.1f}%)")

print(f"\n[Significant elements only]")
print(f"  Max relative error:    {rel_err_sig.max():.6e}  ({rel_err_sig.max()*100:.4f}%)")
print(f"  Mean relative error:   {rel_err_sig.mean():.6e}  ({rel_err_sig.mean()*100:.4f}%)")
print(f"  Median relative error: {np.median(rel_err_sig):.6e}  ({np.median(rel_err_sig)*100:.4f}%)")
print(f"  RMS relative error:    {np.sqrt((rel_err_sig**2).mean()):.6e}")
print(f"  < 1% error:  {(rel_err_sig < 0.01).sum()}/{n_sig} ({100*(rel_err_sig < 0.01).sum()/n_sig:.1f}%)")
print(f"  < 5% error:  {(rel_err_sig < 0.05).sum()}/{n_sig} ({100*(rel_err_sig < 0.05).sum()/n_sig:.1f}%)")
print(f"  < 10% error: {(rel_err_sig < 0.10).sum()}/{n_sig} ({100*(rel_err_sig < 0.10).sum()/n_sig:.1f}%)")

# Row-sum comparison (total opacity per incoming group)
row_sum_collapse = result_collapse.sum(axis=(1, 2))
row_sum_exact = result_exact.sum(axis=(1, 2))
row_rel = np.abs(row_sum_collapse - row_sum_exact) / np.maximum(np.abs(row_sum_exact), 1e-30)
print(f"\n[Row-sum (total opacity)]")
print(f"  Max relative error:  {row_rel.max():.6e}  ({row_rel.max()*100:.4f}%)")
print(f"  Mean relative error: {row_rel.mean():.6e}  ({row_rel.mean()*100:.6f}%)")

# Diagonal comparison (elastic/forward scattering)
diag_collapse = np.array([result_collapse[i, i, :].sum() for i in range(N_COARSE)])
diag_exact = np.array([result_exact[i, i, :].sum() for i in range(N_COARSE)])
diag_mask = diag_exact > 1e-30
diag_rel = np.abs(diag_collapse[diag_mask] - diag_exact[diag_mask]) / diag_exact[diag_mask]
print(f"\n[Diagonal (elastic scattering)]")
print(f"  Max relative error:  {diag_rel.max():.6e}  ({diag_rel.max()*100:.4f}%)")
print(f"  Mean relative error: {diag_rel.mean():.6e}  ({diag_rel.mean()*100:.4f}%)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print(f"  Collapse time:         {t_collapse*1000:.1f} ms")
print(f"  Significant elements:  max {rel_err_sig.max()*100:.2f}%, median {np.median(rel_err_sig)*100:.2f}%")
print(f"  Row-sum (opacity):     max {row_rel.max()*100:.4f}%")
print(f"  Diagonal (elastic):    max {diag_rel.max()*100:.2f}%")

# --- Test 2: Higher temperature (hot plasma) ---
print("\n\n" + "=" * 70)
test_file_hot = "/home/itamarg/workspace_current/ComptonMatrixWebsite/docs/data/uniform/T058_3.340485e+08K.npz"
T_K_hot = 3.340485e+08
print(f"\nTest 2: T = {T_K_hot:.3e} K  (kT = {k_boltz * T_K_hot / kev:.1f} keV) — HOT PLASMA")

with open(test_file_hot, "rb") as f:
    npz_bytes_hot = f.read()

t0 = time.perf_counter()
result_collapse_hot = collapse_mod._collapse_to_array(npz_bytes_hot, list(coarse_bounds_keV), K_COARSE)
t_hot = time.perf_counter() - t0

mg_hot = cm.ComptonMultigroupKernel(
    energy_group_boundaries=coarse_bounds_erg,
    weight_function=cm.UniformWeightFunction(),
    config=cfg,
)
result_exact_hot = mg_hot.compute_sigma_matrix(KERNEL, num_angle_bins=K_COARSE, T=T_K_hot)

diff_hot = np.abs(result_collapse_hot - result_exact_hot)
peak_hot = np.abs(result_exact_hot).max()
sig_hot = np.abs(result_exact_hot) > 1e-3 * peak_hot
rel_hot = diff_hot[sig_hot] / np.abs(result_exact_hot[sig_hot])

row_collapse_hot = result_collapse_hot.sum(axis=(1, 2))
row_exact_hot = result_exact_hot.sum(axis=(1, 2))
row_rel_hot = np.abs(row_collapse_hot - row_exact_hot) / np.maximum(np.abs(row_exact_hot), 1e-30)

print(f"  Collapse time:         {t_hot*1000:.1f} ms")
print(f"  Significant elements:  {sig_hot.sum()}/{result_exact_hot.size}")
print(f"  Max relative error:    {rel_hot.max()*100:.2f}%")
print(f"  Median relative error: {np.median(rel_hot)*100:.2f}%")
print(f"  Row-sum max error:     {row_rel_hot.max()*100:.4f}%")
print(f"  < 5% error:  {(rel_hot < 0.05).sum()}/{sig_hot.sum()} ({100*(rel_hot < 0.05).sum()/sig_hot.sum():.1f}%)")
print(f"  < 10% error: {(rel_hot < 0.10).sum()}/{sig_hot.sum()} ({100*(rel_hot < 0.10).sum()/sig_hot.sum():.1f}%)")
