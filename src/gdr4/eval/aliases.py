"""Period recovery and alias-failure diagnostics (definitions pre-registered in the config)."""

from __future__ import annotations

import numpy as np


def period_metrics(
    period_samples: np.ndarray,
    period_true: np.ndarray,
    tol: float = 0.1,
    mass_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Per-star period diagnostics.

    Parameters
    ----------
    period_samples : (n, B) posterior samples of P in days.
    period_true : (B,)
    tol : relative tolerance, |P / P_true - 1| < tol.
    mass_threshold : alias failure if less than this posterior mass lies within ``tol``.

    Returns
    -------
    dict with ``median_d``, ``rel_err`` (of the median), ``recovered`` (median within tol),
    ``mass_within`` and ``alias_failure`` (bool).
    """
    rel = np.abs(period_samples / period_true[None, :] - 1.0)
    mass = np.mean(rel < tol, axis=0)
    med = np.median(period_samples, axis=0)
    rel_med = np.abs(med / period_true - 1.0)
    return {
        "median_d": med,
        "rel_err": rel_med,
        "recovered": rel_med < tol,
        "mass_within": mass,
        "alias_failure": mass < mass_threshold,
    }


def binned_fraction(x: np.ndarray, y: np.ndarray, flag: np.ndarray, x_edges, y_edges):
    """Fraction of ``flag`` in each (x, y) bin, and the counts (for alias maps)."""
    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    hits, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=flag.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(count > 0, hits / count, np.nan)
    return frac, count
