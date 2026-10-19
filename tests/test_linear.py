import numpy as np
from scipy.stats import multivariate_normal

from gdr4.model.astrometry import design_matrix, t_peri_from_mean_anomaly
from gdr4.model.linear import gaussian_linear_posterior, wls_fit


def _problem(rng, n=60):
    t_d = np.sort(rng.uniform(-990, 990, n))
    psi = rng.uniform(-np.pi, np.pi, n)
    plx = rng.uniform(-1, 1, n)
    period, ecc, m0 = 412.0, 0.3, 1.1
    m = design_matrix(t_d, psi, plx, period, ecc, t_peri_from_mean_anomaly(m0, period))
    return m, rng.uniform(0.03, 0.1, n)


def test_log_marginal_matches_direct_gaussian(rng):
    m, sigma = _problem(rng)
    mu0 = rng.normal(0, 1, 9)
    s0 = rng.uniform(0.5, 3, 9)
    w = rng.normal(0, 2, len(sigma))
    sol = gaussian_linear_posterior(m, w, sigma, mu0, s0)
    cov_w = np.diag(sigma**2) + m @ np.diag(s0**2) @ m.T
    direct = multivariate_normal(mean=m @ mu0, cov=cov_w).logpdf(w)
    np.testing.assert_allclose(sol.log_marginal, direct, rtol=1e-10)


def test_posterior_matches_direct_formula(rng):
    m, sigma = _problem(rng)
    mu0, s0 = np.zeros(9), np.full(9, 2.0)
    w = rng.normal(0, 1, len(sigma))
    sol = gaussian_linear_posterior(m, w, sigma, mu0, s0)
    ci = np.diag(1 / sigma**2)
    h = np.diag(1 / s0**2) + m.T @ ci @ m
    np.testing.assert_allclose(sol.cov, np.linalg.inv(h), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(sol.mean, np.linalg.solve(h, m.T @ ci @ w), rtol=1e-9)


def test_recovers_injected_theta_lin_at_high_snr(rng):
    m, _ = _problem(rng, n=120)
    sigma = np.full(120, 1e-3)
    theta = rng.normal(0, 3, 9)
    w = m @ theta + sigma * rng.normal(size=120)
    sol = gaussian_linear_posterior(m, w, sigma, np.zeros(9), np.full(9, 100.0))
    z = (sol.mean - theta) / np.sqrt(np.diag(sol.cov))
    assert np.all(np.abs(z) < 5)
    np.testing.assert_allclose(sol.mean, theta, atol=1e-2)


def test_mask_equals_dropping_rows(rng):
    m, sigma = _problem(rng)
    w = rng.normal(0, 1, len(sigma))
    mask = rng.uniform(size=len(sigma)) > 0.3
    mu0, s0 = np.zeros(9), np.full(9, 2.0)
    full = gaussian_linear_posterior(m, w, sigma, mu0, s0, mask=mask)
    sub = gaussian_linear_posterior(m[mask], w[mask], sigma[mask], mu0, s0)
    np.testing.assert_allclose(full.log_marginal, sub.log_marginal, rtol=1e-12)
    np.testing.assert_allclose(full.mean, sub.mean, rtol=1e-10)


def test_batched_over_trial_orbits(rng):
    n, s = 50, 6
    t_d = np.sort(rng.uniform(-990, 990, n))
    psi, plx = rng.uniform(-np.pi, np.pi, n), rng.uniform(-1, 1, n)
    sigma = np.full(n, 0.05)
    w = rng.normal(0, 1, n)
    period = rng.uniform(30, 2000, (s, 1))
    ecc = rng.uniform(0, 0.9, (s, 1))
    tp = rng.uniform(0, 100, (s, 1))
    m = design_matrix(t_d, psi, plx, period, ecc, tp)
    s0 = rng.uniform(0.5, 2, (s, 9))
    batched = gaussian_linear_posterior(m, w, sigma, np.zeros(9), s0, return_cov=False)
    for i in range(s):
        single = gaussian_linear_posterior(m[i], w, sigma, np.zeros(9), s0[i])
        np.testing.assert_allclose(batched.log_marginal[i], single.log_marginal, rtol=1e-12)


def test_wls_residual_invariant_to_5p_signal(rng):
    m, sigma = _problem(rng)
    m5 = m[:, :5]
    w = rng.normal(0, 1, len(sigma))
    _, r1, chi1, dof = wls_fit(m5, w, sigma)
    _, r2, chi2, _ = wls_fit(m5, w + m5 @ rng.normal(0, 50, 5), sigma)
    np.testing.assert_allclose(r1, r2, atol=1e-9)
    assert dof == len(sigma) - 5
