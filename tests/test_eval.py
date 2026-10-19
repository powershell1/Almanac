import numpy as np

from gdr4.eval.aliases import binned_fraction, period_metrics
from gdr4.eval.c2st import c2st_accuracy
from gdr4.eval.calibration import (
    central_interval_coverage,
    circular_embed,
    expected_coverage,
    hpd_levels,
    marginal_ranks,
    tarp_ecp,
)

ALPHAS = [0.5, 0.68, 0.9]


def _gaussian_problem(rng, n_star=3000, n_samp=1000, width_scale=1.0):
    """theta ~ N(0, 1), x = theta + N(0, 1): exact posterior N(x / 2, 1 / 2)."""
    theta = rng.normal(size=(n_star, 2))
    x = theta + rng.normal(size=(n_star, 2))
    mean, sd = x / 2, np.sqrt(0.5) * width_scale
    samples = mean[None] + sd * rng.normal(size=(n_samp, n_star, 2))

    def logq(v):
        return -0.5 * np.sum(((v - mean) / sd) ** 2, axis=-1)

    return theta, samples, logq(samples), logq(theta)


def test_hpd_coverage_calibrated(rng):
    _, _, lq_s, lq_t = _gaussian_problem(rng)
    cov = expected_coverage(hpd_levels(lq_t, lq_s), ALPHAS)
    np.testing.assert_allclose(cov, ALPHAS, atol=0.03)


def test_hpd_coverage_detects_overconfidence(rng):
    _, _, lq_s, lq_t = _gaussian_problem(rng, width_scale=0.6)
    cov = expected_coverage(hpd_levels(lq_t, lq_s), ALPHAS)
    assert np.all(cov < np.array(ALPHAS) - 0.1)


def test_marginal_central_coverage(rng):
    theta, samples, _, _ = _gaussian_problem(rng)
    cov = central_interval_coverage(marginal_ranks(theta[:, 0], samples[..., 0]), ALPHAS)
    np.testing.assert_allclose(cov, ALPHAS, atol=0.03)


def test_tarp_calibrated_and_not(rng):
    theta, samples, _, _ = _gaussian_problem(rng)
    a, ecp = tarp_ecp(samples, theta, rng, ALPHAS)
    np.testing.assert_allclose(ecp, a, atol=0.04)
    theta, samples, _, _ = _gaussian_problem(rng, width_scale=0.5)
    _, ecp_bad = tarp_ecp(samples, theta, rng, ALPHAS)
    assert np.max(np.abs(ecp_bad - a)) > 0.08


def test_circular_embedding_wraps():
    v = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 2 * np.pi - 0.01]])
    e = circular_embed(v)
    assert np.linalg.norm(e[0] - e[1]) < 0.03


def test_c2st(rng):
    a, b = rng.normal(size=(1000, 3)), rng.normal(size=(1000, 3))
    assert c2st_accuracy(a, b) < 0.56
    assert c2st_accuracy(a, b + 3.0) > 0.95


def test_period_metrics():
    true = np.array([100.0, 100.0, 500.0])
    samples = np.stack(
        [np.full(100, 101.0), np.r_[np.full(40, 100.0), np.full(60, 50.0)], np.full(100, 700.0)],
        axis=1,
    )
    m = period_metrics(samples, true)
    np.testing.assert_array_equal(m["recovered"], [True, False, False])
    np.testing.assert_array_equal(m["alias_failure"], [False, True, True])
    frac, count = binned_fraction(
        np.log10(true), np.array([7, 7, 7.0]), m["alias_failure"], [1.5, 2.5, 3.0], [6, 8]
    )
    np.testing.assert_allclose(frac[:, 0], [0.5, 1.0])
