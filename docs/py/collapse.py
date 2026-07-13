"""
Collapse a fine-grid Compton scattering matrix to a coarser energy/angle grid.

Runs in Pyodide (browser) or standalone CPython. Uses the same uniform
weighting that produced the original data:
  - Incoming groups: width-weighted average (w_g = E_{g+1} - E_g)
  - Outgoing groups: simple sum
  - Angle bins: simple sum
"""

from __future__ import annotations

import io

import numpy as np


def collapse(
    npz_bytes: bytes,
    coarse_group_indices: list[int],
    coarse_angle_factor: int,
) -> bytes:
    """
    Load a .npz from raw bytes, collapse to coarser grid, return .npy bytes.

    Parameters
    ----------
    npz_bytes : bytes
        Raw bytes of a .npz file (fetched via JS or read from disk).
    coarse_group_indices : list[int]
        Indices into the fine boundary array that define coarse group edges.
        Length = N_coarse + 1.  Must start at 0 and end at G (number of fine
        groups), and be strictly increasing.
    coarse_angle_factor : int
        How many fine angle bins per coarse bin.
        E.g. 10 means M_fine / 10 coarse bins.

    Returns
    -------
    bytes
        Raw bytes of a .npy file containing the collapsed (N, N, M_coarse)
        float64 array.
    """
    data = np.load(io.BytesIO(npz_bytes))
    sigma = data["sigma_matrix"]
    boundaries = data["boundaries_keV"]
    G, _, M = sigma.shape

    if sigma.ndim != 3:
        raise ValueError(f"Expected 3D sigma_matrix, got shape {sigma.shape}")
    if M % coarse_angle_factor != 0:
        raise ValueError(
            f"M={M} not divisible by coarse_angle_factor={coarse_angle_factor}"
        )
    idx = list(coarse_group_indices)
    if idx[0] != 0 or idx[-1] != G:
        raise ValueError(
            f"Group indices must span [0, {G}], got [{idx[0]}, {idx[-1]}]"
        )
    if not all(idx[i] < idx[i + 1] for i in range(len(idx) - 1)):
        raise ValueError("Group indices must be strictly increasing")

    M_coarse = M // coarse_angle_factor
    sigma_a = sigma.reshape(G, G, M_coarse, coarse_angle_factor).sum(axis=3)

    N_coarse = len(idx) - 1
    result = np.zeros((N_coarse, N_coarse, M_coarse))
    widths = boundaries[1:] - boundaries[:-1]

    for G_out in range(N_coarse):
        gp_lo, gp_hi = idx[G_out], idx[G_out + 1]
        sigma_ao = sigma_a[:, gp_lo:gp_hi, :].sum(axis=1)

        for G_in in range(N_coarse):
            g_lo, g_hi = idx[G_in], idx[G_in + 1]
            w = widths[g_lo:g_hi]
            chunk = sigma_ao[g_lo:g_hi, :]
            result[G_in, G_out, :] = (w[:, None] * chunk).sum(axis=0) / w.sum()

    buf = io.BytesIO()
    np.save(buf, result)
    return buf.getvalue()
