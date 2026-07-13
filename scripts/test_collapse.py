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


def test_identity(npz_bytes: bytes):
    """Collapsing to the same grid must return the original matrix."""
    ref = np.load(io.BytesIO(npz_bytes))
    sigma = ref["sigma_matrix"]
    G, _, M = sigma.shape

    all_indices = list(range(G + 1))
    result_bytes = collapse(npz_bytes, all_indices, 1)
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

    result_bytes = collapse(npz_bytes, [0, G], M)
    result = np.load(io.BytesIO(result_bytes))

    assert result.shape == (1, 1, 1), f"Expected (1,1,1), got {result.shape}"

    sigma_angle_summed = sigma.sum(axis=2)
    sigma_out_summed = sigma_angle_summed.sum(axis=1)
    expected = (widths * sigma_out_summed).sum() / widths.sum()

    np.testing.assert_allclose(result[0, 0, 0], expected, rtol=1e-12,
                               err_msg="Full collapse value mismatch")
    print("  PASS: full collapse (1x1x1)")


def test_partial_collapse(npz_bytes: bytes):
    """Collapse to 2 groups, verify consistency with manual sub-block computation."""
    ref = np.load(io.BytesIO(npz_bytes))
    sigma = ref["sigma_matrix"]
    boundaries = ref["boundaries_keV"]
    G, _, M = sigma.shape
    widths = boundaries[1:] - boundaries[:-1]

    mid = G // 2
    indices = [0, mid, G]
    angle_factor = M // 2
    M_coarse = 2

    result_bytes = collapse(npz_bytes, indices, angle_factor)
    result = np.load(io.BytesIO(result_bytes))
    assert result.shape == (2, 2, M_coarse), f"Expected (2,2,{M_coarse}), got {result.shape}"

    sigma_a = sigma.reshape(G, G, M_coarse, angle_factor).sum(axis=3)

    for G_in in range(2):
        g_lo = indices[G_in]
        g_hi = indices[G_in + 1]
        w = widths[g_lo:g_hi]
        for G_out in range(2):
            gp_lo = indices[G_out]
            gp_hi = indices[G_out + 1]
            sigma_ao = sigma_a[:, gp_lo:gp_hi, :].sum(axis=1)
            chunk = sigma_ao[g_lo:g_hi, :]
            expected = (w[:, None] * chunk).sum(axis=0) / w.sum()
            np.testing.assert_allclose(
                result[G_in, G_out, :], expected, rtol=1e-13,
                err_msg=f"Partial collapse mismatch at ({G_in},{G_out})"
            )
    print("  PASS: partial collapse (2x2x2)")


def test_rejection_bad_angle_factor(npz_bytes: bytes):
    """Non-divisible angle factor must raise ValueError."""
    ref = np.load(io.BytesIO(npz_bytes))
    G = ref["sigma_matrix"].shape[0]
    try:
        collapse(npz_bytes, list(range(G + 1)), 3)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASS: rejects non-divisible angle factor")


def test_rejection_bad_indices(npz_bytes: bytes):
    """Out-of-range and non-monotonic indices must raise ValueError."""
    ref = np.load(io.BytesIO(npz_bytes))
    G = ref["sigma_matrix"].shape[0]

    try:
        collapse(npz_bytes, [1, G], 1)
        assert False, "Should reject indices not starting at 0"
    except ValueError:
        pass

    try:
        collapse(npz_bytes, [0, G - 1], 1)
        assert False, "Should reject indices not ending at G"
    except ValueError:
        pass

    try:
        collapse(npz_bytes, [0, 50, 30, G], 1)
        assert False, "Should reject non-monotonic indices"
    except ValueError:
        pass

    print("  PASS: rejects invalid group indices")


def main() -> None:
    npz_path = find_3d_npz()
    print(f"Using test file: {npz_path.name}")

    npz_bytes = load_npz_bytes(npz_path)

    test_identity(npz_bytes)
    test_full_collapse(npz_bytes)
    test_partial_collapse(npz_bytes)
    test_rejection_bad_angle_factor(npz_bytes)
    test_rejection_bad_indices(npz_bytes)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
