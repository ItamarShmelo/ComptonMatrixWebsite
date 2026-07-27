"""
Collapse a fine-grid Compton scattering matrix to a coarser energy/angle grid.

Runs in Pyodide (browser) or standalone CPython. Supports arbitrary coarse
energy boundaries and any number of uniform angle bins -- not limited to
fine-grid-aligned boundaries or divisors of the fine angle count.

Uses uniform (width-weighted) energy averaging for the incoming-group average.
When a fine input group straddles a coarse energy boundary, a sub-group
minmod-limited linear reconstruction estimates the overlap-specific
cross-section.  At high temperatures (kT > 1 keV), straddling output groups
are split by overlap-fraction weighting with an output-direction slope
correction; at low temperatures, kinematic assignment via the Compton formula
determines the target coarse output bin.

Angle bins always use width-proportional fractional overlap.
"""

from __future__ import annotations

import io

import numpy as np

K_BOLTZ_KEV = 8.617333262e-8  # Boltzmann constant in keV/K
_MC2_KEV = 510.998950  # electron rest mass energy in keV

_STRADDLE_EPS = 1e-12


def _reconstruct_subgroup(sigma_a, g, ov_lo, ov_hi, E_c, G):
    """
    Reconstruct sub-group cross-section for fine group g over [ov_lo, ov_hi].

    Uses minmod-limited linear reconstruction: estimates the slope of sigma
    through group g using centered differences, limits it with the minmod
    function to prevent overshoots at kernel edges, and evaluates at the
    overlap midpoint.

    Operates element-wise on the (output_group, angle) dimensions so that
    each sigma[g, p, A] is limited independently.

    Returns the reconstructed array of shape (G_out, K) to replace sigma_a[g, :, :].
    """
    E_mid = 0.5 * (ov_lo + ov_hi)
    delta_E = E_mid - E_c[g]
    val_g = sigma_a[g, :, :]

    if G < 2:
        return val_g.copy()

    if g == 0:
        dE = E_c[1] - E_c[0]
        if dE == 0:
            return val_g.copy()
        slope = (sigma_a[1, :, :] - val_g) / dE
        result = val_g + slope * delta_E
        np.maximum(result, 0.0, out=result)
        return result

    if g == G - 1:
        dE = E_c[G - 1] - E_c[G - 2]
        if dE == 0:
            return val_g.copy()
        slope = (val_g - sigma_a[G - 2, :, :]) / dE
        result = val_g + slope * delta_E
        np.maximum(result, 0.0, out=result)
        return result

    dE_L = E_c[g] - E_c[g - 1]
    dE_R = E_c[g + 1] - E_c[g]

    slope_L = (val_g - sigma_a[g - 1, :, :]) / dE_L
    slope_R = (sigma_a[g + 1, :, :] - val_g) / dE_R

    same_sign = slope_L * slope_R > 0
    slope = np.where(
        same_sign,
        np.where(np.abs(slope_L) < np.abs(slope_R), slope_L, slope_R),
        0.0,
    )

    result = val_g + slope * delta_E
    np.maximum(result, 0.0, out=result)
    return result


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
    weighting: str = "uniform",
    temperature_K: float | None = None,
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
    weighting : str, optional
        Energy weighting mode.  Only ``'uniform'`` is supported.
    temperature_K : float, optional
        Electron temperature in Kelvin.  Used to select the output-splitting
        strategy: overlap-fraction at high kT, kinematic at low kT.

    Returns
    -------
    np.ndarray
        Collapsed matrix with shape (N, N, K).
    """
    if weighting != "uniform":
        raise ValueError(
            f"Only 'uniform' weighting is supported, got {weighting!r}"
        )
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

    kT_keV = K_BOLTZ_KEV * temperature_K if temperature_K is not None else None

    sigma_a = np.einsum("gpa,aA->gpA", sigma, f_xi)

    N = len(B) - 1
    E_c = np.sqrt(fine_bounds[:-1] * fine_bounds[1:])

    # Identify straddling output groups and precompute output-direction corrections
    straddle_out_info = []
    for p in range(G):
        overlapping_J = [J for J in range(N) if _STRADDLE_EPS < f_e[p, J] < 1.0 - _STRADDLE_EPS]
        if not overlapping_J:
            continue
        for J in overlapping_J:
            ov_lo_out = max(fine_bounds[p], B[J])
            ov_hi_out = min(fine_bounds[p + 1], B[J + 1])
            E_mid_out = 0.5 * (ov_lo_out + ov_hi_out)
            straddle_out_info.append((p, J, E_mid_out - E_c[p]))

    def _apply_output_recon(sigma_ao_I, sigma_a_used):
        """Apply output-direction slope correction for straddling output groups."""
        for p, J, delta_out in straddle_out_info:
            if p == 0:
                dE = E_c[1] - E_c[0]
                slope_out = (sigma_a_used[:, 1, :] - sigma_a_used[:, 0, :]) / dE
            elif p == G - 1:
                dE = E_c[G - 1] - E_c[G - 2]
                slope_out = (sigma_a_used[:, G - 1, :] - sigma_a_used[:, G - 2, :]) / dE
            else:
                dE_L = E_c[p] - E_c[p - 1]
                dE_R = E_c[p + 1] - E_c[p]
                sL = (sigma_a_used[:, p, :] - sigma_a_used[:, p - 1, :]) / dE_L
                sR = (sigma_a_used[:, p + 1, :] - sigma_a_used[:, p, :]) / dE_R
                same_sign = sL * sR > 0
                slope_out = np.where(
                    same_sign,
                    np.where(np.abs(sL) < np.abs(sR), sL, sR),
                    0.0,
                )
            sigma_ao_I[:, J, :] += slope_out * delta_out * f_e[p, J]
        np.maximum(sigma_ao_I, 0.0, out=sigma_ao_I)

    fine_weights = fine_bounds[1:] - fine_bounds[:-1]
    coarse_weights = np.einsum("gI,g->I", f_e, fine_weights)

    # At high kT, the Compton kernel is broad and overlap-fraction output
    # splitting outperforms kinematic (which does all-or-nothing assignment).
    # At low kT, the kernel is narrow and kinematic is more accurate.
    _KT_OVERLAP_THRESHOLD = 1.0  # keV
    use_overlap_output = (kT_keV is not None and kT_keV > _KT_OVERLAP_THRESHOLD)

    straddle_out = set()
    xi_centers = None
    if not use_overlap_output:
        for p in range(G):
            if np.count_nonzero(f_e[p, :]) >= 2:
                straddle_out.add(p)
        xi_centers = 0.5 * (coarse_xi[:-1] + coarse_xi[1:])

    result = np.zeros((N, N, K))

    for I in range(N):
        if coarse_weights[I] <= 0:
            continue

        straddle_in = [g for g in range(G)
                       if _STRADDLE_EPS < f_e[g, I] < 1.0 - _STRADDLE_EPS]
        if straddle_in:
            sigma_a_I = sigma_a.copy()
            for g in straddle_in:
                ov_lo = max(fine_bounds[g], B[I])
                ov_hi = min(fine_bounds[g + 1], B[I + 1])
                sigma_a_I[g, :, :] = _reconstruct_subgroup(
                    sigma_a, g, ov_lo, ov_hi, E_c, G,
                )
        else:
            sigma_a_I = sigma_a

        if use_overlap_output:
            sigma_ao_I = np.einsum("gpA,pJ->gJA", sigma_a_I, f_e)
            if straddle_out_info:
                _apply_output_recon(sigma_ao_I, sigma_a_I)
        elif not straddle_out:
            sigma_ao_I = np.einsum("gpA,pJ->gJA", sigma_a_I, f_e)
        else:
            non_straddle = np.array([p not in straddle_out for p in range(G)])
            sigma_ao_I = np.einsum(
                "gpA,pJ->gJA", sigma_a_I[:, non_straddle, :], f_e[non_straddle, :],
            )
            for p in straddle_out:
                for g in range(G):
                    if f_e[g, I] <= 0:
                        continue
                    if abs(f_e[g, I] - 1.0) < 1e-12:
                        E_eff = E_c[g]
                    else:
                        ov_lo = max(fine_bounds[g], B[I])
                        ov_hi = min(fine_bounds[g + 1], B[I + 1])
                        E_eff = 0.5 * (ov_lo + ov_hi)
                    for A in range(K):
                        val = sigma_a_I[g, p, A]
                        if val == 0.0:
                            continue
                        E_prime = E_eff / (1.0 + (E_eff / _MC2_KEV) * (1.0 - xi_centers[A]))
                        J = int(np.searchsorted(B, E_prime, side="right")) - 1
                        J = max(0, min(N - 1, J))
                        sigma_ao_I[g, J, A] += val

        w_gI = f_e[:, I] * fine_weights
        result[I, :, :] = np.einsum("g,gJA->JA", w_gI, sigma_ao_I) / coarse_weights[I]

    return result


def collapse(
    npz_bytes: bytes,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
    weighting: str = "uniform",
    temperature_K: float | None = None,
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
    weighting : str, optional
        Energy weighting mode.  Only ``'uniform'`` is supported.
    temperature_K : float, optional
        Electron temperature in Kelvin.  Used to select the output-splitting
        strategy: overlap-fraction at high kT, kinematic at low kT.

    Returns
    -------
    bytes
        Raw bytes of a .npy file containing the collapsed float64 array.
        Shape is (N, N, K), reduced for edge cases:
        - (N, N) if K == 1
        - (K,) if N == 1
        - scalar if N == 1 and K == 1
    """
    result = _collapse_to_array(
        npz_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
        weighting=weighting,
        temperature_K=temperature_K,
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
    weighting: str = "uniform",
    temperature_K: float | None = None,
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
    weighting : str, optional
        Energy weighting mode.  Only ``'uniform'`` is supported.
    temperature_K : float, optional
        Temperature for the kT-based output strategy (K).  Typically *T_target*.

    Returns
    -------
    np.ndarray
        Interpolated collapsed matrix with shape (N, N, K).
    """
    alpha = (np.log(T_target) - np.log(T_lo)) / (np.log(T_hi) - np.log(T_lo))

    sigma_lo = _collapse_to_array(
        npz_lo_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
        weighting=weighting,
        temperature_K=temperature_K,
    )
    sigma_hi = _collapse_to_array(
        npz_hi_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
        weighting=weighting,
        temperature_K=temperature_K,
    )

    return (1.0 - alpha) * sigma_lo + alpha * sigma_hi
