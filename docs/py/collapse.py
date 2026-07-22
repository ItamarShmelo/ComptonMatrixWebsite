"""
Collapse a fine-grid Compton scattering matrix to a coarser energy/angle grid.

Runs in Pyodide (browser) or standalone CPython. Supports arbitrary coarse
energy boundaries and any number of uniform angle bins -- not limited to
fine-grid-aligned boundaries or divisors of the fine angle count.

Uses the same uniform weighting that produced the original data:
  - Incoming groups: width-weighted average (w_g = E_{g+1} - E_g)
  - Outgoing groups: simple sum (with fractional overlap splitting)
  - Angle bins: simple sum (with fractional overlap splitting)
"""

from __future__ import annotations

import io

import numpy as np


def _overlap_matrix(fine_bounds: np.ndarray, coarse_bounds: np.ndarray) -> np.ndarray:
    """
    Build a dense overlap-fraction matrix F[g, I].

    F[g, I] = (overlap length of fine bin g with coarse bin I) / (fine bin g width).
    Rows sum to <= 1 (exactly 1 when coarse grid covers the fine bin).
    """
    G = len(fine_bounds) - 1
    N = len(coarse_bounds) - 1
    F = np.zeros((G, N))
    j_start = 0
    for g in range(G):
        f_lo, f_hi = fine_bounds[g], fine_bounds[g + 1]
        f_width = f_hi - f_lo
        if f_width <= 0:
            continue
        for j in range(j_start, N):
            c_lo, c_hi = coarse_bounds[j], coarse_bounds[j + 1]
            if c_lo >= f_hi:
                break
            if c_hi <= f_lo:
                j_start = j + 1
                continue
            overlap = min(f_hi, c_hi) - max(f_lo, c_lo)
            if overlap > 0:
                F[g, j] = overlap / f_width
    return F


def _validate_inputs(energy_boundaries_keV, n_angle_bins, angle_boundaries):
    """Validate and parse energy/angle boundary arguments. Returns (B, coarse_xi)."""
    if (n_angle_bins is None) == (angle_boundaries is None):
        raise ValueError("Provide exactly one of n_angle_bins or angle_boundaries")

    B = np.asarray(energy_boundaries_keV, dtype=np.float64)
    if B.ndim != 1 or len(B) < 2:
        raise ValueError("energy_boundaries_keV must have at least 2 values")
    if np.any(np.isnan(B)):
        raise ValueError("energy_boundaries_keV contains NaN")
    if not np.all(np.diff(B) > 0):
        raise ValueError("energy_boundaries_keV must be strictly increasing")

    if angle_boundaries is not None:
        coarse_xi = np.asarray(angle_boundaries, dtype=np.float64)
        if coarse_xi.ndim != 1 or len(coarse_xi) < 2:
            raise ValueError("angle_boundaries must have at least 2 values")
        if not np.all(np.diff(coarse_xi) > 0):
            raise ValueError("angle_boundaries must be strictly increasing")
        if coarse_xi[0] < -1 - 1e-12 or coarse_xi[-1] > 1 + 1e-12:
            raise ValueError(
                f"angle_boundaries [{coarse_xi[0]:.6e}, {coarse_xi[-1]:.6e}] "
                f"exceed [-1, 1]"
            )
    else:
        if not isinstance(n_angle_bins, (int, np.integer)) or n_angle_bins < 1:
            raise ValueError(f"n_angle_bins must be a positive integer, got {n_angle_bins}")
        coarse_xi = np.linspace(-1.0, 1.0, n_angle_bins + 1)

    return B, coarse_xi


def _collapse_to_array(
    npz_bytes: bytes,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
) -> np.ndarray:
    """
    Collapse a fine-grid matrix to a coarser grid, returning the full 3D array.

    Always returns shape (N, N, K) without any dimension reduction.

    Parameters
    ----------
    npz_bytes : bytes
        Raw bytes of a .npz file containing ``sigma_matrix`` (G, G, M)
        and ``boundaries_keV`` (G+1,).
    energy_boundaries_keV : list[float]
        Coarse energy group edges in keV.
    n_angle_bins : int, optional
        Number of uniform coarse angle bins.
    angle_boundaries : list[float], optional
        Explicit coarse angle bin edges.

    Returns
    -------
    np.ndarray
        Collapsed matrix with shape (N, N, K).
    """
    B, coarse_xi = _validate_inputs(energy_boundaries_keV, n_angle_bins, angle_boundaries)

    data = np.load(io.BytesIO(npz_bytes))
    sigma = data["sigma_matrix"]
    fine_bounds = data["boundaries_keV"]
    G, _, M = sigma.shape

    if sigma.ndim != 3:
        raise ValueError(f"Expected 3D sigma_matrix, got shape {sigma.shape}")
    if B[0] < fine_bounds[0] * (1 - 1e-12) or B[-1] > fine_bounds[-1] * (1 + 1e-12):
        raise ValueError(
            f"Coarse boundaries [{B[0]:.6e}, {B[-1]:.6e}] exceed fine grid "
            f"[{fine_bounds[0]:.6e}, {fine_bounds[-1]:.6e}]"
        )

    f_e = _overlap_matrix(fine_bounds, B)

    fine_xi = np.linspace(-1.0, 1.0, M + 1)
    K = len(coarse_xi) - 1
    f_xi = _overlap_matrix(fine_xi, coarse_xi)

    fine_widths = fine_bounds[1:] - fine_bounds[:-1]
    coarse_widths = B[1:] - B[:-1]

    sigma_a = np.einsum("gpa,aA->gpA", sigma, f_xi)
    sigma_ao = np.einsum("gpA,pJ->gJA", sigma_a, f_e)
    sigma_weighted = sigma_ao * fine_widths[:, None, None]
    result = np.einsum("gJA,gI->IJA", sigma_weighted, f_e) / coarse_widths[:, None, None]

    return result


def collapse(
    npz_bytes: bytes,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
) -> bytes:
    """
    Load a .npz from raw bytes, collapse to a coarser grid, return .npy bytes.

    Parameters
    ----------
    npz_bytes : bytes
        Raw bytes of a .npz file containing at least ``sigma_matrix`` (G, G, M)
        and ``boundaries_keV`` (G+1,).
    energy_boundaries_keV : list[float]
        Coarse energy group edges in keV.  Length N+1, strictly increasing.
        Values must lie within the fine-grid range.
    n_angle_bins : int, optional
        Number of uniform coarse angle bins in xi = cos(theta) over [-1, 1].
        Any positive integer is accepted.  Mutually exclusive with
        ``angle_boundaries``.
    angle_boundaries : list[float], optional
        Explicit coarse angle bin edges in xi = cos(theta).  Must be strictly
        increasing and lie within [-1, 1].  Mutually exclusive with
        ``n_angle_bins``.

    Exactly one of ``n_angle_bins`` or ``angle_boundaries`` must be provided.

    Returns
    -------
    bytes
        Raw bytes of a .npy file containing the collapsed float64 array.
        Shape is (N, N, K), reduced for edge cases:
        - (N, N) if K == 1
        - (K,) if N == 1
        - scalar if N == 1 and K == 1

    Notes
    -----
    When the requested energy boundaries coincide with the stored fine-grid
    boundaries, the collapse is exact.  When boundaries fall between
    fine-grid points, the affected fine groups are split proportionally by
    width (linear interpolation), which introduces a few-percent error on
    the dominant matrix elements.  Angle-bin splitting is accurate to
    machine precision because the fine angle bins are narrow.
    """
    result = _collapse_to_array(
        npz_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
    )
    N, _, K = result.shape

    if N == 1 and K == 1:
        result = result.ravel()[0]
    elif N == 1:
        result = result[0, 0, :]
    elif K == 1:
        result = result[:, :, 0]

    buf = io.BytesIO()
    np.save(buf, result)
    return buf.getvalue()


def collapse_interp(
    npz_lo_bytes: bytes,
    npz_hi_bytes: bytes,
    T_lo: float,
    T_hi: float,
    T_target: float,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
) -> np.ndarray:
    """
    Collapse two temperature grids and interpolate in log-T space.

    Parameters
    ----------
    npz_lo_bytes : bytes
        Raw bytes of the lower-temperature .npz file.
    npz_hi_bytes : bytes
        Raw bytes of the higher-temperature .npz file.
    T_lo : float
        Temperature (K) of the lower bracket.
    T_hi : float
        Temperature (K) of the upper bracket.
    T_target : float
        Desired interpolation temperature (K). Must satisfy T_lo <= T_target <= T_hi.
    energy_boundaries_keV : list[float]
        Coarse energy group edges in keV.
    n_angle_bins : int, optional
        Number of uniform coarse angle bins.
    angle_boundaries : list[float], optional
        Explicit coarse angle bin edges.

    Returns
    -------
    np.ndarray
        Interpolated collapsed matrix with shape (N, N, K).
    """
    alpha = (np.log(T_target) - np.log(T_lo)) / (np.log(T_hi) - np.log(T_lo))

    sigma_lo = _collapse_to_array(
        npz_lo_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
    )
    sigma_hi = _collapse_to_array(
        npz_hi_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
    )

    return (1.0 - alpha) * sigma_lo + alpha * sigma_hi
