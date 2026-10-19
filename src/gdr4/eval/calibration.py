"""Calibration diagnostics: HPD expected coverage, TARP, marginal (SBC) ranks.

See docs/derivations.md, section 6. Sample arrays are (n_samples, n_stars[, dim]).
"""

from __future__ import annotations

import numpy as np


def hpd_levels(log_q_true: np.ndarray, log_q_samples: np.ndarray) -> np.ndarray:
    """Credibility level of the truth: fraction of samples with higher density than the truth.

    For a calibrated posterior this is Uniform(0, 1) across test stars (Hermans et al. 2022).
    """
    return np.mean(log_q_samples > log_q_true[None, :], axis=0)


def expected_coverage(levels: np.ndarray, alphas) -> np.ndarray:
    """Observed fraction of stars whose truth lies inside the alpha credible region."""
    alphas = np.asarray(alphas, float)
    return np.mean(levels[:, None] < alphas[None, :], axis=0)


def bootstrap_coverage(
    levels: np.ndarray, alphas, rng: np.random.Generator, n_boot=2000, ci=0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval of :func:`expected_coverage` over stars."""
    n = len(levels)
    boots = np.array(
        [expected_coverage(levels[rng.integers(0, n, n)], alphas) for _ in range(n_boot)]
    )
    lo, hi = np.quantile(boots, [(1 - ci) / 2, (1 + ci) / 2], axis=0)
    return lo, hi


def marginal_ranks(truth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """SBC-style normalised rank of the truth in each star's marginal samples, in [0, 1]."""
    return np.mean(samples < truth[None, :], axis=0)


def central_interval_coverage(ranks: np.ndarray, alphas) -> np.ndarray:
    """Fraction of stars whose truth is inside the central alpha interval of a marginal."""
    alphas = np.asarray(alphas, float)
    lo, hi = 0.5 - alphas / 2, 0.5 + alphas / 2
    return np.mean((ranks[:, None] > lo) & (ranks[:, None] < hi), axis=0)


def circular_embed(v: np.ndarray) -> np.ndarray:
    """(u_P, u_e, M0) -> (u_P, u_e, cos M0, sin M0) so Euclidean distance respects the circle."""
    return np.concatenate([v[..., :2], np.cos(v[..., 2:3]), np.sin(v[..., 2:3])], axis=-1)


def tarp_ecp(
    samples: np.ndarray, truth: np.ndarray, rng: np.random.Generator, alphas=None
) -> tuple[np.ndarray, np.ndarray]:
    """TARP expected coverage probability (Lemos et al. 2023).

    Parameters
    ----------
    samples : (n, B, D) embedded posterior samples; truth : (B, D).

    Returns
    -------
    alphas, ecp : arrays; a calibrated estimator gives ecp == alphas.
    """
    alphas = np.linspace(0, 1, 101) if alphas is None else np.asarray(alphas, float)
    mu, sd = truth.mean(0), truth.std(0) + 1e-12
    s = (samples - mu) / sd
    t = (truth - mu) / sd
    lo, hi = t.min(0), t.max(0)
    ref = rng.uniform(lo, hi, size=t.shape)  # one random reference point per star
    d_samp = np.linalg.norm(s - ref[None], axis=-1)
    d_true = np.linalg.norm(t - ref, axis=-1)
    f = np.mean(d_samp < d_true[None], axis=0)
    ecp = np.mean(f[:, None] < alphas[None, :], axis=0)
    return alphas, ecp
