"""Analytic treatment of the linear parameters (the "Joker trick").

Model: ``w = M theta + eps`` with ``eps ~ N(0, C)``, ``C = diag(sigma^2)``, and a Gaussian prior
``theta ~ N(mu0, L0)`` with diagonal ``L0 = diag(s0^2)``. Then (docs/derivations.md, section 3)

    H  = L0^-1 + M^T C^-1 M                  (posterior precision)
    b  = L0^-1 mu0 + M^T C^-1 w
    theta | w   ~ N(H^-1 b, H^-1)
    log p(w)    = -1/2 [w^T C^-1 w + mu0^T L0^-1 mu0 - b^T H^-1 b]
                  -1/2 [log|C| + log|L0| + log|H|] - n/2 log(2 pi)

Everything is batched over leading dimensions, so one call can evaluate many trial orbits
(brute-force reference) or many stars (simulation). Padded transits are handled with a
boolean ``mask``: masked entries get zero weight and do not count towards ``n``.
All computation is float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class LinearSolution:
    """Result of :func:`gaussian_linear_posterior` (shapes for batch shape ``S``)."""

    mean: np.ndarray  # (*S, K) posterior mean of theta_lin
    cov: np.ndarray  # (*S, K, K) posterior covariance
    log_marginal: np.ndarray  # (*S,) log p(w | theta_nl), marginalised over theta_lin


def _weights(sigma, mask):
    sigma = np.asarray(sigma, dtype=np.float64)
    inv_var = 1.0 / (sigma * sigma)
    if mask is not None:
        inv_var = np.where(mask, inv_var, 0.0)
    return inv_var


def gaussian_linear_posterior(
    m: np.ndarray,
    w: np.ndarray,
    sigma: np.ndarray,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    mask: np.ndarray | None = None,
    return_cov: bool = True,
) -> LinearSolution:
    """Posterior of theta_lin and marginal likelihood for a linear-Gaussian model.

    Parameters
    ----------
    m : np.ndarray, shape (*S, N, K)
        Design matrix.
    w : np.ndarray, shape (*S, N) or (N,)
        Observed along-scan abscissae (mas).
    sigma : np.ndarray, broadcastable to w
        Per-transit uncertainty (mas).
    prior_mean, prior_std : np.ndarray, broadcastable to (*S, K)
        Mean and standard deviation of the independent Gaussian prior on theta_lin.
    mask : np.ndarray of bool, broadcastable to w, optional
        True for real transits, False for padding.
    return_cov : bool
        If False, skip the K x K inverse (faster when only ``log_marginal`` is needed).

    Returns
    -------
    LinearSolution
    """
    m = np.asarray(m, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    inv_var = _weights(sigma, mask)
    inv_var = np.broadcast_to(inv_var, np.broadcast_shapes(inv_var.shape, w.shape))
    k = m.shape[-1]
    mu0 = np.asarray(prior_mean, dtype=np.float64)
    s0 = np.asarray(prior_std, dtype=np.float64)
    prior_prec = 1.0 / (s0 * s0)

    mt_ci = np.swapaxes(m * inv_var[..., None], -1, -2)  # (*S, K, N)
    h = mt_ci @ m  # (*S, K, K)
    h = h + np.eye(k) * prior_prec[..., None, :]
    b = prior_prec * mu0 + np.einsum("...kn,...n->...k", mt_ci, np.broadcast_to(w, inv_var.shape))

    chol = np.linalg.cholesky(h)
    # Solve H x = b via the Cholesky factor: L y = b, L^T x = y.
    y = np.linalg.solve(chol, b[..., None])
    mean = np.linalg.solve(np.swapaxes(chol, -1, -2), y)[..., 0]

    quad = (
        np.sum(w * w * inv_var, axis=-1)
        + np.sum(prior_prec * mu0 * mu0, axis=-1)
        - np.sum(y[..., 0] ** 2, axis=-1)  # b^T H^-1 b = |L^-1 b|^2
    )
    real = inv_var > 0
    n = np.sum(real, axis=-1)
    log_det_c = -np.sum(np.log(np.where(real, inv_var, 1.0)), axis=-1)
    log_det_l0 = np.sum(np.broadcast_to(2.0 * np.log(s0), mean.shape), axis=-1)
    log_det_h = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    log_marginal = -0.5 * (quad + log_det_c + log_det_l0 + log_det_h + n * LOG_2PI)

    cov = None
    if return_cov:
        chol_inv = np.linalg.inv(chol)
        cov = np.swapaxes(chol_inv, -1, -2) @ chol_inv
    return LinearSolution(mean=mean, cov=cov, log_marginal=log_marginal)


def wls_fit(m, w, sigma, mask=None):
    """Plain weighted least squares (no prior).

    Returns
    -------
    theta_hat : np.ndarray, shape (*S, K)
    residual : np.ndarray, shape (*S, N)
        ``w - M theta_hat`` (mas); zero where masked.
    chi2 : np.ndarray, shape (*S,)
    dof : np.ndarray, shape (*S,)
        Number of unmasked transits minus K.
    """
    m = np.asarray(m, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    inv_var = np.broadcast_to(_weights(sigma, mask), w.shape)
    mt_ci = np.swapaxes(m * inv_var[..., None], -1, -2)
    h = mt_ci @ m
    b = np.einsum("...kn,...n->...k", mt_ci, w)
    theta_hat = np.linalg.solve(h, b[..., None])[..., 0]
    resid = w - np.einsum("...nk,...k->...n", m, theta_hat)
    resid = np.where(inv_var > 0, resid, 0.0)
    chi2 = np.sum(resid * resid * inv_var, axis=-1)
    dof = np.sum(inv_var > 0, axis=-1) - m.shape[-1]
    return theta_hat, resid, chi2, dof
