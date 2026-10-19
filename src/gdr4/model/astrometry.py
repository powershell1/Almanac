"""Along-scan astrometric forward model and its design matrix.

For transit k (CCD samples already combined per transit):

    w_k = (da* + mua* t_k) sin psi_k + (dd + mud t_k) cos psi_k + plx f_k
          + (B X_k + G Y_k) sin psi_k + (A X_k + F Y_k) cos psi_k + eps_k

with t_k in years for the proper-motion terms. This matches gaiamock
(``predict_astrometry_binary_in_terms_of_a0``) plus the two position offsets.
See docs/derivations.md, section 2.

Linear parameter order (used everywhere in this package)::

    theta_lin = (dra_mas, ddec_mas, parallax_mas, pmra_masyr, pmdec_masyr, A, B, F, G)

Thiele-Innes constants A, B, F, G are in mas.
"""

from __future__ import annotations

import numpy as np

from gdr4.model.kepler import TWO_PI, orbit_xy

DAYS_PER_YEAR = 365.25
DR4_REF_EPOCH_JD = 2457936.875  # J2017.5 (TCB), same value gaiamock uses for DR4
LIN_NAMES = ("dra_mas", "ddec_mas", "parallax_mas", "pmra_masyr", "pmdec_masyr", "A", "B", "F", "G")
N_LIN = len(LIN_NAMES)
N_5P = 5


def t_peri_from_mean_anomaly(mean_anom0_rad, period_d):
    """Periastron time (days from the reference epoch) given the mean anomaly at t = 0.

    ``M(t) = M0 + 2 pi t / P`` and ``M(T_p) = 0``, so ``T_p = -M0 P / (2 pi)``.
    """
    return -np.asarray(mean_anom0_rad) * np.asarray(period_d) / TWO_PI


def mean_anomaly_from_t_peri(t_peri_d, period_d):
    """Inverse of :func:`t_peri_from_mean_anomaly`, wrapped to [0, 2 pi)."""
    return np.mod(-TWO_PI * np.asarray(t_peri_d) / np.asarray(period_d), TWO_PI)


def design_matrix_5p(t_d, psi_rad, plx_factor):
    """Single-star (5-parameter) design matrix.

    Parameters
    ----------
    t_d, psi_rad, plx_factor : array_like, shape (..., N)
        Time (days from reference epoch), scan angle (rad), along-scan parallax factor.

    Returns
    -------
    np.ndarray, shape (..., N, 5)
        Columns multiply (dra_mas, ddec_mas, parallax_mas, pmra_masyr, pmdec_masyr).
    """
    t_yr = np.asarray(t_d, dtype=np.float64) / DAYS_PER_YEAR
    s, c = np.sin(psi_rad), np.cos(psi_rad)
    f = np.asarray(plx_factor, dtype=np.float64)
    return np.stack([s, c, f, t_yr * s, t_yr * c], axis=-1)


def design_matrix_orbit(t_d, psi_rad, period_d, ecc, t_peri_d):
    """Orbital (Thiele-Innes) part of the design matrix.

    Parameters
    ----------
    t_d, psi_rad : array_like, shape (..., N)
    period_d, ecc, t_peri_d : array_like
        Must broadcast against ``t_d``; pass shape (S, 1) to evaluate S orbits at once.

    Returns
    -------
    np.ndarray, shape (..., N, 4)
        Columns multiply (A, B, F, G) in mas.
    """
    x, y = orbit_xy(t_d, period_d, ecc, t_peri_d)
    s, c = np.sin(psi_rad), np.cos(psi_rad)
    s, c, x, y = np.broadcast_arrays(s, c, x, y)
    return np.stack([x * c, x * s, y * c, y * s], axis=-1)


def design_matrix(t_d, psi_rad, plx_factor, period_d, ecc, t_peri_d):
    """Full 9-column design matrix ``M(theta_nl)``, shape (..., N, 9). See module docstring."""
    m_orb = design_matrix_orbit(t_d, psi_rad, period_d, ecc, t_peri_d)
    m_5p = design_matrix_5p(t_d, psi_rad, plx_factor)
    m_5p = np.broadcast_to(m_5p, m_orb.shape[:-1] + (N_5P,))
    return np.concatenate([m_5p, m_orb], axis=-1)


def predict_w(t_d, psi_rad, plx_factor, period_d, ecc, t_peri_d, theta_lin):
    """Noise-free along-scan abscissa ``w = M theta_lin`` in mas, shape (..., N)."""
    m = design_matrix(t_d, psi_rad, plx_factor, period_d, ecc, t_peri_d)
    return np.einsum("...nk,...k->...n", m, np.asarray(theta_lin, dtype=np.float64))


def thiele_innes_from_campbell(a0_mas, node_rad, argp_rad, inc_rad):
    """Thiele-Innes constants (mas) from Campbell elements.

    Same convention as gaiamock: ``node_rad`` is the longitude of the ascending node (Omega),
    ``argp_rad`` the argument of periastron (omega), ``inc_rad`` the inclination
    (0 or pi is face-on).
    """
    cw, sw = np.cos(argp_rad), np.sin(argp_rad)
    cn, sn = np.cos(node_rad), np.sin(node_rad)
    ci = np.cos(inc_rad)
    a = a0_mas * (cw * cn - sw * sn * ci)
    b = a0_mas * (cw * sn + sw * cn * ci)
    f = -a0_mas * (sw * cn + cw * sn * ci)
    g = -a0_mas * (sw * sn - cw * cn * ci)
    return a, b, f, g


def campbell_from_thiele_innes(a, b, f, g):
    """Campbell elements from Thiele-Innes constants (inverse of the function above).

    Follows the formulas in gaiamock ``get_Campbell_elements`` (Halbwachs et al. 2023,
    appendix). Omega is returned in [0, pi), omega in [0, 2 pi), i in [0, pi].

    Returns
    -------
    a0_mas, node_rad, argp_rad, inc_rad : np.ndarray
    """
    a, b, f, g = (np.asarray(v, dtype=np.float64) for v in (a, b, f, g))
    wp = np.arctan2(b - f, a + g)  # omega + Omega
    wm = np.arctan2(-b - f, a - g)  # omega - Omega
    argp = 0.5 * (wp + wm)
    node = 0.5 * (wp - wm)
    argp = np.where(node < 0, argp + np.pi, argp)
    node = np.where(node < 0, node + np.pi, node)
    t_ag = np.abs((a + g) * np.cos(wm))
    t_bf = np.abs((f - b) * np.sin(wm))
    inc = np.where(
        t_ag > t_bf,
        2.0 * np.arctan2(np.sqrt(np.abs((a - g) * np.cos(wp))), np.sqrt(t_ag)),
        2.0 * np.arctan2(np.sqrt(np.abs((b + f) * np.sin(wp))), np.sqrt(t_bf)),
    )
    a0 = photocentre_semimajor_axis(a, b, f, g)
    argp = np.mod(argp, TWO_PI)
    return a0, node, argp, inc


def photocentre_semimajor_axis(a, b, f, g):
    """Photocentre semi-major axis a0 (mas) from Thiele-Innes constants (mas)."""
    u = 0.5 * (a * a + b * b + f * f + g * g)
    v = a * g - b * f
    return np.sqrt(u + np.sqrt(np.clip((u + v) * (u - v), 0.0, None)))
