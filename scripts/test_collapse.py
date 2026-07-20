#!/usr/bin/env python3
"""
Offline correctness tests for the collapse function.

Imports collapse() from docs/py/collapse.py and runs it with known inputs
against expected outputs. No browser or Pyodide required.

Usage:
  python scripts/test_collapse.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))

from collapse import collapse  # noqa: E402

DATA_DIR = ROOT / "docs" / "data"


def find_3d_npz() -> Path:
    """Find the first .npz file with a 3D sigma_matrix."""
    for path in sorted(DATA_DIR.glob("T*_*K.npz")):
        with np.load(path) as d:
            if d["sigma_matrix"].ndim == 3:
                return path
    raise FileNotFoundError("No 3D .npz files found in docs/data/")


def load_npz_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _get_fine_boundaries(npz_bytes: bytes) -> np.ndarray:
    return np.load(io.BytesIO(npz_bytes))["boundaries_keV"]


# ── Identity & aligned-boundary tests ─────────────────────


def test_identity(npz_bytes: bytes):
    """Collapsing to the same grid must return the original matrix."""
    ref = np.load(io.BytesIO(npz_bytes))
    sigma = ref["sigma_matrix"]
    boundaries = ref["boundaries_keV"]
    G, _, M = sigma.shape

    result_bytes = collapse(npz_bytes, boundaries.tolist(), M)
    result = np.load(io.BytesIO(result_bytes))

    assert result.shape == sigma.shape, f"Shape mismatch: {result.shape} vs {sigma.shape}"
    np.testing.assert_allclose(result, sigma, rtol=1e-10, atol=1e-40,
                               err_msg="Identity collapse failed")
    print("  PASS: identity collapse")


def test_full_collapse(npz_bytes: bytes):
    """Collapsing to 1 group, 1 angle bin must equal the width-weighted sum."""
    ref = np.load(io.BytesIO(npz_bytes))
    sigma = ref["sigma_matrix"]
    boundaries = ref["boundaries_keV"]
    G, _, M = sigma.shape
    widths = boundaries[1:] - boundaries[:-1]

    result_bytes = collapse(npz_bytes, [boundaries[0], boundaries[-1]], 1)
    result = np.load(io.BytesIO(result_bytes))

    assert result.shape == (), f"Expected scalar (), got {result.shape}"

    sigma_angle_summed = sigma.sum(axis=2)
    sigma_out_summed = sigma_angle_summed.sum(axis=1)
    expected = (widths * sigma_out_summed).sum() / widths.sum()

    np.testing.assert_allclose(float(result), expected, rtol=1e-12,
                               err_msg="Full collapse value mismatch")
    print("  PASS: full collapse (scalar)")


def test_partial_collapse(npz_bytes: bytes):
    """Collapse to 2 groups at the midpoint with 2 angle bins (aligned)."""
    ref = np.load(io.BytesIO(npz_bytes))
    sigma = ref["sigma_matrix"]
    boundaries = ref["boundaries_keV"]
    G, _, M = sigma.shape
    widths = boundaries[1:] - boundaries[:-1]

    mid = G // 2
    coarse_bounds = [boundaries[0], boundaries[mid], boundaries[-1]]
    M_coarse = 2

    result_bytes = collapse(npz_bytes, coarse_bounds, M_coarse)
    result = np.load(io.BytesIO(result_bytes))
    assert result.shape == (2, 2, M_coarse), f"Expected (2,2,{M_coarse}), got {result.shape}"

    angle_factor = M // M_coarse
    sigma_a = sigma.reshape(G, G, M_coarse, angle_factor).sum(axis=3)
    idx = [0, mid, G]

    for G_in in range(2):
        g_lo, g_hi = idx[G_in], idx[G_in + 1]
        w = widths[g_lo:g_hi]
        for G_out in range(2):
            gp_lo, gp_hi = idx[G_out], idx[G_out + 1]
            sigma_ao = sigma_a[:, gp_lo:gp_hi, :].sum(axis=1)
            chunk = sigma_ao[g_lo:g_hi, :]
            expected = (w[:, None] * chunk).sum(axis=0) / w.sum()
            np.testing.assert_allclose(
                result[G_in, G_out, :], expected, rtol=1e-12,
                err_msg=f"Partial collapse mismatch at ({G_in},{G_out})"
            )
    print("  PASS: partial collapse (2x2x2)")


# ── Non-aligned boundary tests ────────────────────────────


def test_non_aligned_energy(npz_bytes: bytes):
    """Boundaries at geometric means of adjacent fine-grid boundaries."""
    ref = np.load(io.BytesIO(npz_bytes))
    sigma = ref["sigma_matrix"]
    boundaries = ref["boundaries_keV"]

    midpoints = np.sqrt(boundaries[:-1] * boundaries[1:])
    coarse_bounds = [boundaries[0]] + midpoints[::16].tolist() + [boundaries[-1]]

    result_bytes = collapse(npz_bytes, coarse_bounds, 100)
    result = np.load(io.BytesIO(result_bytes))
    nG = len(coarse_bounds) - 1
    assert result.shape == (nG, nG, 100), f"Shape mismatch: {result.shape}"
    assert np.all(np.isfinite(result)), "Non-finite values in result"

    total_collapsed = result.sum()
    identity_bytes = collapse(npz_bytes, boundaries.tolist(), 100)
    identity = np.load(io.BytesIO(identity_bytes))
    widths = boundaries[1:] - boundaries[:-1]
    total_identity = (identity * widths[:, None, None]).sum() / widths.sum()
    total_result_weighted = 0.0
    coarse_b = np.array(coarse_bounds)
    coarse_widths = coarse_b[1:] - coarse_b[:-1]
    total_result_weighted = (result * coarse_widths[:, None, None]).sum() / coarse_widths.sum()

    np.testing.assert_allclose(
        total_result_weighted, total_identity, rtol=1e-10,
        err_msg="Width-weighted total mismatch for non-aligned boundaries"
    )
    print("  PASS: non-aligned energy boundaries")


def test_non_divisor_angles(npz_bytes: bytes):
    """7 angle bins (doesn't divide 100). Angle-summed result should match."""
    ref = np.load(io.BytesIO(npz_bytes))
    boundaries = ref["boundaries_keV"]

    result_bytes_7 = collapse(npz_bytes, boundaries.tolist(), 7)
    result_7 = np.load(io.BytesIO(result_bytes_7))
    assert result_7.shape[2] == 7, f"Expected 7 angle bins, got {result_7.shape[2]}"

    result_bytes_1 = collapse(npz_bytes, boundaries.tolist(), 1)
    result_1 = np.load(io.BytesIO(result_bytes_1))

    np.testing.assert_allclose(
        result_7.sum(axis=2), result_1, rtol=1e-10,
        err_msg="Angle-summed 7-bin result should match 1-bin result"
    )
    print("  PASS: non-divisor angle bins (7)")


def test_arbitrary_grid(npz_bytes: bytes):
    """Random 10x10x12 grid: shape, non-negativity, and re-collapse transitivity."""
    ref = np.load(io.BytesIO(npz_bytes))
    boundaries = ref["boundaries_keV"]

    rng = np.random.default_rng(seed=12345)
    log_min, log_max = np.log10(boundaries[0]), np.log10(boundaries[-1])
    inner = np.sort(rng.uniform(log_min + 0.3, log_max - 0.3, size=9))
    for i in range(1, len(inner)):
        inner[i] = max(inner[i], inner[i - 1] + 0.3)
    coarse_bounds = np.concatenate([[boundaries[0]], 10**inner, [boundaries[-1]]])

    result_bytes = collapse(npz_bytes, coarse_bounds.tolist(), 12)
    result = np.load(io.BytesIO(result_bytes))
    assert result.shape == (10, 10, 12), f"Expected (10,10,12), got {result.shape}"
    assert np.all(np.isfinite(result)), "Non-finite values"

    coarser_bounds = [coarse_bounds[0], coarse_bounds[5], coarse_bounds[-1]]
    direct_bytes = collapse(npz_bytes, [float(x) for x in coarser_bounds], 4)
    direct = np.load(io.BytesIO(direct_bytes))

    synthetic_npz = io.BytesIO()
    np.savez_compressed(
        synthetic_npz,
        sigma_matrix=result,
        boundaries_keV=coarse_bounds,
        temperature_K=ref["temperature_K"],
    )
    synthetic_npz.seek(0)
    indirect_bytes = collapse(
        synthetic_npz.read(), [float(x) for x in coarser_bounds], 4
    )
    indirect = np.load(io.BytesIO(indirect_bytes))

    assert direct.shape == indirect.shape == (2, 2, 4)
    np.testing.assert_allclose(
        indirect, direct, rtol=1e-8,
        err_msg="Re-collapse transitivity failed"
    )
    print("  PASS: arbitrary grid (10x10x12) + re-collapse")


def test_sub_range(npz_bytes: bytes):
    """Boundaries [1, 100] keV -- only data in that sub-range."""
    ref = np.load(io.BytesIO(npz_bytes))
    boundaries = ref["boundaries_keV"]

    result_bytes = collapse(npz_bytes, [1.0, 10.0, 100.0], 10)
    result = np.load(io.BytesIO(result_bytes))
    assert result.shape == (2, 2, 10), f"Expected (2,2,10), got {result.shape}"
    assert np.all(np.isfinite(result)), "Non-finite values in sub-range result"
    print("  PASS: sub-range collapse ([1, 10, 100] keV)")


# ── Rejection tests ───────────────────────────────────────


def test_rejection_bad_boundaries(npz_bytes: bytes):
    """Invalid boundary arrays must raise ValueError."""
    ref = np.load(io.BytesIO(npz_bytes))
    boundaries = ref["boundaries_keV"]

    cases = [
        ([50.0, 30.0, 100.0], 10, "non-monotonic"),
        ([float("nan"), 100.0], 10, "NaN"),
        ([1e-8, 100.0], 10, "below fine grid"),
        ([1.0, 500.0], 10, "above fine grid"),
        ([50.0], 10, "fewer than 2 values"),
    ]
    for bounds, n_angles, label in cases:
        try:
            collapse(npz_bytes, bounds, n_angles)
            assert False, f"Should reject: {label}"
        except ValueError:
            pass
    print("  PASS: rejects invalid boundaries")


def test_rejection_bad_angle_bins(npz_bytes: bytes):
    """Invalid n_angle_bins must raise ValueError."""
    ref = np.load(io.BytesIO(npz_bytes))
    boundaries = ref["boundaries_keV"]

    for n in [0, -1]:
        try:
            collapse(npz_bytes, boundaries.tolist(), n)
            assert False, f"Should reject n_angle_bins={n}"
        except ValueError:
            pass
    print("  PASS: rejects invalid angle bins")


# ── Main ──────────────────────────────────────────────────


def main() -> None:
    npz_path = find_3d_npz()
    print(f"Using test file: {npz_path.name}")

    npz_bytes = load_npz_bytes(npz_path)

    test_identity(npz_bytes)
    test_full_collapse(npz_bytes)
    test_partial_collapse(npz_bytes)
    test_non_aligned_energy(npz_bytes)
    test_non_divisor_angles(npz_bytes)
    test_arbitrary_grid(npz_bytes)
    test_sub_range(npz_bytes)
    test_rejection_bad_boundaries(npz_bytes)
    test_rejection_bad_angle_bins(npz_bytes)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
