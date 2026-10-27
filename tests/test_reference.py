import numpy as np
import pytest

from gdr4.reference.bruteforce import _KernelMixture, reference_posterior
from gdr4.sim.noise import get_noise_model
from gdr4.sim.priors import Prior
from gdr4.sim.scan import build_cache_toy_nsl
from gdr4.sim.simulate import simulate_batch, star_view
from gdr4.utils import REPO_ROOT, load_config


@pytest.fixture(scope="module")
def setup():
    prior = Prior.from_config(load_config(REPO_ROOT / "configs" / "smoke.yaml")["prior"])
    batch = simulate_batch(
        np.random.default_rng(4), 200, prior, build_cache_toy_nsl(2), get_noise_model("nominal")
    )
    return prior, batch


def test_kernel_mixture_density_normalised_over_circular_dim():
    c = np.array([[0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 6.0, 0.0]])
    h = np.array([[1.0, 1.0, 1.5, 1.0], [1.0, 1.0, 0.3, 1.0]])
    k = _KernelMixture(c, h, np.log([0.3, 0.7]))
    m0 = np.linspace(0, 2 * np.pi, 20001)
    x = np.zeros((len(m0), 4))
    x[:, 2] = m0
    dens = np.exp(k.log_prob(x))
    expected = (2 * np.pi) ** -1.5  # three unit-width Gaussians evaluated at their centre
    np.testing.assert_allclose(np.trapezoid(dens, m0), expected, rtol=1e-6)


def test_kernel_samples_match_density(rng):
    k = _KernelMixture(np.array([[0.0, 1.0, 3.0, -1.0]]), np.array([[0.1, 0.2, 0.5, 0.3]]))
    x = k.sample(rng, 200000)
    assert np.all((x[:, 2] >= 0) & (x[:, 2] < 2 * np.pi))
    np.testing.assert_allclose(x.std(0)[[0, 1, 3]], [0.1, 0.2, 0.3], rtol=0.02)


@pytest.mark.slow
def test_reference_recovers_period_at_high_snr(setup):
    prior, batch = setup
    i = int(np.argsort(batch["snr_a0"])[-5])
    s = star_view(batch, i)
    r = reference_posterior(
        s,
        prior,
        np.random.default_rng(0),
        n_prior_samples=30000,
        n_refine_samples=30000,
        refine_rounds=2,
        chunk=10000,
    )
    assert r.ess > 500
    period = prior.from_unconstrained(r.resample(np.random.default_rng(1), 2000))["period_d"]
    assert np.mean(np.abs(period / s["period_d"] - 1) < 0.1) > 0.9


@pytest.mark.slow
def test_refined_agrees_with_brute_force_at_low_snr(setup):
    """Where plain prior sampling is efficient, refinement must not change the answer."""
    prior, batch = setup
    i = int(np.argsort(batch["snr_mf"])[3])
    s = star_view(batch, i)
    kw = dict(n_prior_samples=60000, chunk=10000)
    plain = reference_posterior(s, prior, np.random.default_rng(0), refine=False, **kw)
    refined = reference_posterior(
        s,
        prior,
        np.random.default_rng(1),
        refine=True,
        n_refine_samples=60000,
        refine_rounds=2,
        **kw,
    )
    assert plain.ess > 1000 and refined.ess > 1000
    for d in (0, 1, 3):  # u_P, u_e, log s_a posterior means agree within MC error
        m1 = np.sum(np.exp(plain.log_w) * plain.v[:, d])
        m2 = np.sum(np.exp(refined.log_w) * refined.v[:, d])
        sd = np.sqrt(np.sum(np.exp(plain.log_w) * (plain.v[:, d] - m1) ** 2))
        assert abs(m1 - m2) < 5 * sd / np.sqrt(min(plain.ess, refined.ess)) + 0.02
    np.testing.assert_allclose(plain.log_evidence, refined.log_evidence, atol=0.1)


KRANGE = (0.8, 1.5)


@pytest.fixture(scope="module")
def kappa_setup():
    prior = Prior.from_config(load_config(REPO_ROOT / "configs" / "smoke.yaml")["prior"])
    batch = simulate_batch(
        np.random.default_rng(6), 200, prior, build_cache_toy_nsl(2), get_noise_model("scale_prior")
    )
    return prior, batch


def test_kernel_mixture_normalised_in_five_dimensions():
    c = np.array([[0.0, 0.0, 0.2, 0.0, 0.1]])
    h = np.array([[1.0, 1.0, 0.8, 1.0, 1.0]])
    k = _KernelMixture(c, h)
    m0 = np.linspace(0, 2 * np.pi, 20001)
    x = np.tile(c, (len(m0), 1))
    x[:, 2] = m0
    dens = np.exp(k.log_prob(x))
    np.testing.assert_allclose(np.trapezoid(dens, m0), (2 * np.pi) ** -2, rtol=1e-6)


@pytest.mark.slow
def test_reference_with_noise_scale_recovers_kappa_and_period(kappa_setup):
    prior, batch = kappa_setup
    i = int(np.argsort(batch["snr_a0"])[-8])
    s = star_view(batch, i)
    r = reference_posterior(
        s,
        prior,
        np.random.default_rng(0),
        n_prior_samples=30000,
        n_refine_samples=30000,
        refine_rounds=2,
        chunk=10000,
        noise_scale_range=KRANGE,
    )
    assert r.v.shape[1] == 5 and r.ess > 300
    smp = r.resample(np.random.default_rng(1), 4000)
    kappa = np.exp(smp[:, 4])
    assert np.quantile(kappa, 0.005) < s["noise_scale"] < np.quantile(kappa, 0.995)
    assert abs(np.median(kappa) / s["noise_scale"] - 1) < 0.15  # ~100 transits: ~7% precision
    period = prior.from_unconstrained(smp[:, :4])["period_d"]
    assert np.mean(np.abs(period / s["period_d"] - 1) < 0.1) > 0.9


@pytest.mark.slow
def test_refined_agrees_with_brute_force_with_noise_scale(kappa_setup):
    prior, batch = kappa_setup
    i = int(np.argsort(batch["snr_mf"])[3])
    s = star_view(batch, i)
    kw = dict(n_prior_samples=200000, chunk=20000, noise_scale_range=KRANGE)
    plain = reference_posterior(s, prior, np.random.default_rng(0), refine=False, **kw)
    refined = reference_posterior(
        s,
        prior,
        np.random.default_rng(1),
        refine=True,
        n_refine_samples=60000,
        refine_rounds=2,
        **kw,
    )
    # kappa is pinned to ~7% by the data, so plain prior sampling is ~4x less efficient here
    assert plain.ess > 500 and refined.ess > 500
    for d in (0, 1, 3, 4):
        m1 = np.sum(np.exp(plain.log_w) * plain.v[:, d])
        m2 = np.sum(np.exp(refined.log_w) * refined.v[:, d])
        sd = np.sqrt(np.sum(np.exp(plain.log_w) * (plain.v[:, d] - m1) ** 2))
        assert abs(m1 - m2) < 5 * sd / np.sqrt(min(plain.ess, refined.ess)) + 0.02, d
    np.testing.assert_allclose(plain.log_evidence, refined.log_evidence, atol=0.1)
