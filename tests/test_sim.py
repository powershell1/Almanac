import numpy as np
import pytest

from gdr4.data.shards import (
    gather_padded,
    generate_shards,
    is_complete,
    list_complete,
    read_shard,
    shard_path,
    shard_rng,
    to_ragged,
)
from gdr4.model.astrometry import N_5P, design_matrix, t_peri_from_mean_anomaly
from gdr4.model.linear import gaussian_linear_posterior
from gdr4.sim.noise import get_noise_model, sigma_transit_mas
from gdr4.sim.priors import Prior
from gdr4.sim.scan import ScanCache, build_cache_toy_nsl, toy_nsl_transits
from gdr4.sim.simulate import simulate_batch, star_view
from gdr4.utils import REPO_ROOT, load_config


@pytest.fixture(scope="module")
def prior():
    return Prior.from_config(load_config(REPO_ROOT / "configs" / "smoke.yaml")["prior"])


@pytest.fixture(scope="module")
def scan():
    return build_cache_toy_nsl(nside=2)


def test_prior_unconstrained_roundtrip(prior, rng):
    nl = prior.sample_nonlinear(rng, 1000)
    v = prior.to_unconstrained(nl["period_d"], nl["ecc"], nl["mean_anom0_rad"], nl["amp_scale_mas"])
    back = prior.from_unconstrained(v)
    for k in back:
        np.testing.assert_allclose(back[k], nl[k], rtol=1e-10)
    assert np.all(np.isfinite(prior.log_prob_unconstrained(v)))


def test_prior_density_integrates_to_one(prior):
    u = np.linspace(-30, 30, 20001)
    v = np.zeros((len(u), 4))
    v[:, 0] = u
    v[:, 3] = np.log(0.1)
    lp = prior.log_prob_unconstrained(v)
    # Marginal over u_P of the normal density is 1; remaining factors are the M0 and s_a ones.
    rest = -np.log(2 * np.pi) - np.log(np.log(prior.amp_scale_max_mas / prior.amp_scale_min_mas))
    rest += -0.5 * np.log(2 * np.pi)  # standard-normal density of u_e at 0
    np.testing.assert_allclose(np.trapezoid(np.exp(lp - rest), u), 1.0, rtol=1e-6)


def test_prior_unconstrained_is_standard_normal(prior, rng):
    nl = prior.sample_nonlinear(rng, 200000)
    v = prior.to_unconstrained(nl["period_d"], nl["ecc"], nl["mean_anom0_rad"])
    np.testing.assert_allclose(v[:, :2].mean(0), 0, atol=0.01)
    np.testing.assert_allclose(v[:, :2].std(0), 1, atol=0.01)


def test_linear_prior_used_by_simulator(prior, rng):
    amp = np.full(200000, 0.3)
    draws = prior.sample_linear(rng, amp)
    mean, std = prior.linear_prior(amp)
    np.testing.assert_allclose(draws.mean(0), mean[0], atol=4 * std[0].max() / np.sqrt(len(amp)))
    np.testing.assert_allclose(draws.std(0), std[0], rtol=0.01)


def test_toy_scan_law_sane():
    t, psi, plx = toy_nsl_transits(np.deg2rad(45.0), np.deg2rad(30.0))
    assert 30 < len(t) < 400
    assert np.all(np.abs(plx) <= 1.02)
    assert np.all((psi >= -np.pi) & (psi <= np.pi))
    assert t.max() - t.min() > 1500  # spans most of the 66-month DR4 baseline


def test_scan_cache_save_load(tmp_path, scan):
    scan.save(tmp_path / "c.npz")
    back = ScanCache.load(tmp_path / "c.npz")
    assert back.source == "toy_nsl" and back.nside == scan.nside
    np.testing.assert_array_equal(back.n_transits, scan.n_transits)


def test_simulate_shapes_and_determinism(prior, scan):
    noise = get_noise_model("nominal")
    a = simulate_batch(np.random.default_rng(3), 50, prior, scan, noise)
    b = simulate_batch(np.random.default_rng(3), 50, prior, scan, noise)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert a["t_d"].dtype == np.float32 and a["mask"].dtype == bool
    assert np.all(a["n_obs"] == a["mask"].sum(1))
    # left-packed and time-ordered
    for i in range(50):
        n = a["n_obs"][i]
        assert a["mask"][i, :n].all() and not a["mask"][i, n:].any()
        assert np.all(np.diff(a["t_d"][i, :n]) >= 0)


def test_simulated_noise_level_matches_sigma(prior, scan):
    """With the orbit removed, normalised residuals of the full model are unit-variance."""
    rng = np.random.default_rng(5)
    batch = simulate_batch(rng, 300, prior, scan, get_noise_model("nominal"))
    z = []
    for i in range(300):
        s = star_view(batch, i)
        tp = t_peri_from_mean_anomaly(s["mean_anom0_rad"], s["period_d"])
        m = design_matrix(s["t_d"], s["psi_rad"], s["plx_factor"], s["period_d"], s["ecc"], tp)
        z.append((s["w_mas"] - m @ s["theta_lin"]) / s["sigma_mas"])
    z = np.concatenate(z)
    assert abs(z.std() - 1.0) < 0.02
    np.testing.assert_allclose(
        batch["sigma_mas"][0, 0], sigma_transit_mas(batch["gmag"][0]), rtol=1e-6
    )


def test_true_parameters_have_high_marginal_likelihood(prior, scan):
    """Sanity: at high S/N, the true theta_nl beats random prior draws in log p(w | theta_nl)."""
    rng = np.random.default_rng(9)
    batch = simulate_batch(rng, 400, prior, scan, get_noise_model("nominal"))
    i = int(np.argmax(batch["snr_a0"]))
    s = star_view(batch, i)
    mu0, sd0 = prior.linear_prior(np.array([s["amp_scale_mas"]]))

    def logml(period, ecc, m0):
        tp = t_peri_from_mean_anomaly(m0, period)
        m = design_matrix(s["t_d"], s["psi_rad"], s["plx_factor"], period, ecc, tp)
        return gaussian_linear_posterior(m, s["w_mas"], s["sigma_mas"], mu0[0], sd0[0]).log_marginal

    best = logml(s["period_d"], s["ecc"], s["mean_anom0_rad"])
    nl = prior.sample_nonlinear(rng, 200)
    others = [
        logml(p, e, m)
        for p, e, m in zip(nl["period_d"], nl["ecc"], nl["mean_anom0_rad"], strict=True)
    ]
    assert best > max(others)
    assert batch["theta_lin"].shape[1] - N_5P == 4


def test_shards_resumable(tmp_path, prior, scan):
    calls = []

    def make(rng):
        calls.append(1)
        return to_ragged(simulate_batch(rng, 20, prior, scan, get_noise_model("nominal")))

    generate_shards(tmp_path, range(3), make, entropy=42, meta={"test": True}, log=lambda s: None)
    assert list_complete(tmp_path) == [0, 1, 2] and len(calls) == 3
    generate_shards(tmp_path, range(4), make, entropy=42, meta={}, log=lambda s: None)
    assert len(calls) == 4  # only shard 3 was new
    a = read_shard(tmp_path, 1)
    padded = simulate_batch(shard_rng(42, 1), 20, prior, scan, get_noise_model("nominal"))
    back = gather_padded(a, np.arange(20), n_max=padded["mask"].shape[1])
    for k in ("w_mas", "t_d", "mask", "resid5_mas", "period_d", "theta_lin"):
        np.testing.assert_array_equal(back[k], padded[k])
    (shard_path(tmp_path, 2) / "_COMPLETE").unlink()
    assert not is_complete(tmp_path, 2)
