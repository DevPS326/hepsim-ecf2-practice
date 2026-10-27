"""Standalone jet-observable practice implementations (not official ML4SCI code)."""

from numbers import Real

import numpy as np


def ecf2(jet_particles, beta=1.0):
    r"""Calculate normalized two-point energy correlation ``e2^(beta)``.

    ``e2^(beta) = sum_(i<j) z_i * z_j * (Delta R_ij)**beta`` where
    ``z_i = pT_i / sum_k(pT_k)`` and
    ``Delta R_ij = hypot(y_i-y_j, wrapped(phi_i-phi_j))``.
    Each unordered pair is counted once. There is no jet-radius normalization.

    Parameters
    ----------
    jet_particles : array_like of shape (N, 4)
        Constituent rows ``[pT, rapidity, phi, PDG ID]``. All values must be
        finite; pT must be nonnegative. Zero-pT rows are treated as padding.
        PDG ID is accepted but does not affect this observable.
    beta : positive finite real number, optional
        Exponent of the angular separation (default 1.0).

    Returns
    -------
    float
        Dimensionless normalized ECF2, or 0.0 for fewer than two constituents
        with positive pT.

    Raises
    ------
    ValueError
        For invalid inputs or unrepresentable nonfinite angular distances.

    Notes
    -----
    Runtime and memory scale quadratically with constituent count. Scaling
    transverse momenta before summing avoids overflow of large finite pT.
    """
    if isinstance(beta, (bool, np.bool_)) or not isinstance(beta, Real):
        raise ValueError("beta must be a finite positive real scalar")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a finite positive real scalar")

    try:
        particles = np.asarray(jet_particles, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("jet_particles must be a numeric array") from exc

    if particles.ndim != 2 or particles.shape[1] != 4:
        raise ValueError("jet_particles must have shape (N, 4)")
    if not np.isfinite(particles).all():
        raise ValueError("jet_particles must contain only finite values")
    if np.any(particles[:, 0] < 0.0):
        raise ValueError("constituent pT must be nonnegative")

    active = particles[particles[:, 0] > 0.0]
    if len(active) < 2:
        return 0.0

    pt = active[:, 0]
    scaled_pt = pt / np.max(pt)
    z = scaled_pt / np.sum(scaled_pt, dtype=np.float64)

    dy = active[:, None, 1] - active[None, :, 1]
    dphi = active[:, None, 2] - active[None, :, 2]
    wrapped_dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    delta_r = np.hypot(dy, wrapped_dphi)
    if not np.isfinite(delta_r).all():
        raise ValueError("pairwise angular distances are not finite")

    i, j = np.triu_indices(len(active), k=1)
    pair_terms = z[i] * z[j] * np.power(delta_r[i, j], beta)
    result = float(np.sum(pair_terms, dtype=np.float64))
    if not np.isfinite(result):
        raise ValueError("ECF2 result is not finite for these inputs")
    return result
