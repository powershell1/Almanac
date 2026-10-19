"""Vectorized solver for Kepler's equation, E - e sin E = M (float64).

See docs/derivations.md, section 1.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def solve_kepler(
    mean_anomaly_rad: np.ndarray,
    ecc: np.ndarray | float,
    tol: float = 1e-15,
    max_iter: int = 30,
) -> np.ndarray:
    """Return the eccentric anomaly E solving ``E - e sin E = M``.

    Uses Danby's (1988) starting guess and his quartic-convergence correction, which
    converges in 2-4 iterations for all 0 <= e < 1.

    Parameters
    ----------
    mean_anomaly_rad : array_like
        Mean anomaly M in radians. Any real value; it is reduced to [-pi, pi) internally
        and the multiple of 2 pi is added back, so E tracks M continuously.
    ecc : array_like or float
        Eccentricity e, 0 <= e < 1. Must broadcast against ``mean_anomaly_rad``.
    tol : float
        Stop when the largest correction is below ``tol`` (radians).
    max_iter : int
        Iteration cap.

    Returns
    -------
    np.ndarray
        Eccentric anomaly E in radians (float64), shape ``broadcast(M, e)``.
    """
    m_rad = np.asarray(mean_anomaly_rad, dtype=np.float64)
    e = np.asarray(ecc, dtype=np.float64)
    m_rad, e = np.broadcast_arrays(m_rad, e)
    if np.any((e < 0) | (e >= 1)):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")

    m_red = np.mod(m_rad + np.pi, TWO_PI) - np.pi  # in [-pi, pi)
    ecc_anom = m_red + 0.85 * e * np.sign(np.sin(m_red))
    for _ in range(max_iter):
        s, c = e * np.sin(ecc_anom), e * np.cos(ecc_anom)
        f0 = ecc_anom - s - m_red
        f1 = 1.0 - c
        f2 = s
        f3 = c
        d1 = -f0 / f1
        d2 = -f0 / (f1 + 0.5 * d1 * f2)
        d3 = -f0 / (f1 + 0.5 * d2 * f2 + d2 * d2 * f3 / 6.0)
        ecc_anom = ecc_anom + d3
        if np.max(np.abs(d3), initial=0.0) < tol:
            break
    return ecc_anom + (m_rad - m_red)


def orbit_xy(
    t_d: np.ndarray,
    period_d: np.ndarray | float,
    ecc: np.ndarray | float,
    t_peri_d: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Elliptical rectangular coordinates of the Thiele-Innes model.

    ``X = cos E - e`` and ``Y = sqrt(1 - e^2) sin E``, with ``E - e sin E = 2 pi (t - T_p) / P``.

    Parameters
    ----------
    t_d : array_like
        Observation times in days relative to the reference epoch (J2017.5 for DR4).
    period_d, ecc, t_peri_d : array_like
        Period (days), eccentricity, and periastron time (days, same origin as ``t_d``).
        All arguments must broadcast together, e.g. ``t_d`` of shape (N,) with parameters of
        shape (S, 1) gives outputs of shape (S, N).

    Returns
    -------
    X, Y : np.ndarray
        Dimensionless, float64.
    """
    t_d = np.asarray(t_d, dtype=np.float64)
    period_d = np.asarray(period_d, dtype=np.float64)
    ecc = np.asarray(ecc, dtype=np.float64)
    mean_anom = TWO_PI * (t_d - t_peri_d) / period_d
    ecc_anom = solve_kepler(mean_anom, ecc)
    x = np.cos(ecc_anom) - ecc
    y = np.sqrt(1.0 - ecc * ecc) * np.sin(ecc_anom)
    return x, y
