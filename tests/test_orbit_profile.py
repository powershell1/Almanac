import numpy as np
import torch

from gdr4.model.astrometry import design_matrix, design_matrix_5p, t_peri_from_mean_anomaly
from gdr4.model.kepler import solve_kepler
from gdr4.model.linear import wls_fit
from gdr4.nn.data import to_torch_batch
from gdr4.nn.orbit_profile import kepler_eccentric_anomaly, orbit_chi2_gain, profile_anchors
from gdr4.sim.noise import get_noise_model
from gdr4.sim.priors import Prior
from gdr4.sim.scan import build_cache_toy_nsl
from gdr4.sim.simulate import simulate_batch, star_view
from gdr4.utils import REPO_ROOT, load_config


def _sim(n=40, seed=3):
    prior = Prior.from_config(load_config(REPO_ROOT / "configs" / "smoke.yaml")["prior"])
    sim = simulate_batch(
        np.random.default_rng(seed), n, prior, build_cache_toy_nsl(2), get_noise_model("nominal")
    )
    return prior, sim


def test_torch_kepler_matches_numpy():
    m = torch.linspace(-30, 30, 5001, dtype=torch.float64)
    for e in (0.0, 0.4, 0.9, 0.95):
        ours = kepler_eccentric_anomaly(m, torch.tensor(e, dtype=torch.float64)).numpy()
        np.testing.assert_allclose(ours, solve_kepler(m.numpy(), e), atol=1e-12)


def test_gain_matches_direct_nine_parameter_fit():
    _, sim = _sim(n=4)
    rng = np.random.default_rng(0)
    for i in range(4):
        s = star_view(sim, i)
        m5 = design_matrix_5p(s["t_d"], s["psi_rad"], s["plx_factor"])
        _, resid, chi5, _ = wls_fit(m5, s["w_mas"], s["sigma_mas"])
        one = {k: torch.tensor(s[k])[None] for k in ("t_d", "psi_rad", "plx_factor", "sigma_mas")}
        one["resid5_mas"] = torch.tensor(resid)[None]
        one["mask"] = torch.ones_like(one["t_d"], dtype=torch.bool)
        p, e, m0 = rng.uniform(30, 2000, 5), rng.uniform(0, 0.9, 5), rng.uniform(0, 6.28, 5)
        gain = orbit_chi2_gain(one, *(torch.tensor(x)[None] for x in (p, e, m0)))[0].numpy()
        for j in range(5):
            m9 = design_matrix(
                s["t_d"],
                s["psi_rad"],
                s["plx_factor"],
                p[j],
                e[j],
                t_peri_from_mean_anomaly(m0[j], p[j]),
            )
            _, _, chi9, _ = wls_fit(m9, s["w_mas"], s["sigma_mas"])
            np.testing.assert_allclose(gain[j], chi5 - chi9, rtol=1e-7, atol=1e-6 * chi5)


def test_profile_anchor_recovers_e_and_m0_at_true_period():
    prior, sim = _sim(n=200, seed=11)
    sel = np.flatnonzero((sim["snr_a0"] > 30) & (sim["ecc"] > 0.3))[:6]
    assert len(sel) >= 3
    batch = to_torch_batch({k: v[sel] for k, v in sim.items()}, prior)
    prof = profile_anchors(batch, torch.tensor(sim["period_d"][sel])[:, None], prior.ecc_max)
    assert prof["map"].shape == (len(sel), 1, 8, 16)
    d_m0 = np.angle(np.exp(1j * (prof["m0"][:, 0].numpy() - sim["mean_anom0_rad"][sel])))
    d_e = prof["ecc"][:, 0].numpy() - sim["ecc"][sel]
    assert np.median(np.abs(d_m0)) < 0.05 and np.median(np.abs(d_e)) < 0.05
    assert np.all(np.isfinite(prof["sigma_m0"][:, 0].numpy()))


def test_keplerian_refinement_beats_circular_peak_for_eccentric_orbits():
    from gdr4.nn.periodogram import analyse_peaks, circular_periodogram, frequency_grid

    prior, sim = _sim(n=300, seed=21)
    sel = np.flatnonzero((sim["snr_a0"] > 20) & (sim["ecc"] > 0.5))[:12]
    assert len(sel) >= 6
    batch = to_torch_batch({k: v[sel] for k, v in sim.items()}, prior)
    grid = frequency_grid(20.0, 3000.0, 512)
    peaks = analyse_peaks(batch, grid, circular_periodogram(batch, grid), 1)
    f_c = peaks["freq"]
    sig_f = torch.sqrt(2.0 / (-peaks["d2"]).clamp_min(1e-300))
    prof = profile_anchors(batch, 1.0 / f_c, prior.ecc_max, sigma_f=sig_f)
    f_true = 1.0 / sim["period_d"][sel]
    err_c = np.abs(f_c[:, 0].numpy() - f_true) / sig_f[:, 0].numpy()
    err_k = np.abs(1.0 / prof["period"][:, 0].numpy() - f_true) / sig_f[:, 0].numpy()
    assert np.median(err_k) < np.median(err_c)
    assert np.median(err_k) < 3.0  # within a few Fisher widths of the truth


def _true_noise_scale(sim, i):
    s = star_view(sim, i)
    tp = t_peri_from_mean_anomaly(s["mean_anom0_rad"], s["period_d"])
    m = design_matrix(s["t_d"], s["psi_rad"], s["plx_factor"], s["period_d"], s["ecc"], tp)
    return np.std((s["w_mas"] - m @ s["theta_lin"]) / s["sigma_mas"])


def test_noise_scale_estimate_tracks_true_noise():
    from gdr4.nn.model import NPE

    prior = Prior.from_config(load_config(REPO_ROOT / "configs" / "smoke.yaml")["prior"])
    sim = simulate_batch(
        np.random.default_rng(5), 120, prior, build_cache_toy_nsl(2), get_noise_model("scale_prior")
    )
    batch = to_torch_batch(sim, prior)
    torch.manual_seed(0)
    model = NPE(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_freq=8,
        ctx_dim=32,
        flow_transforms=2,
        flow_hidden=(32, 32),
        flow_bins=4,
        n_vm=3,
        pgram_k=256,
        pgram_peaks=4,
        head="anchored",
        n_anchor=3,
        profile=True,
        refine_period=True,
        noise_rescale=True,
    )
    with torch.no_grad():
        _, peaks = model.encoder(batch)
    est = np.exp(peaks["log_kappa"].numpy())
    true = np.array([_true_noise_scale(sim, i) for i in range(120)])
    hi = sim["snr_a0"] > 6
    for sel in (hi, ~hi):  # valid at high and low S/N
        assert np.median(np.abs(np.log(est[sel] / true[sel]))) < 0.08
    assert np.corrcoef(est, true)[0, 1] > 0.8


def test_rescaled_features_invariant_to_reported_sigma():
    """Multiplying the reported sigma by c must not change what the network is given."""
    from gdr4.nn.model import NPE

    prior, sim = _sim(n=12, seed=7)
    batch = to_torch_batch(sim, prior)
    other = dict(batch)
    c = 1.3
    other["sigma_mas"] = batch["sigma_mas"] * c
    other["chi2_5p"] = batch["chi2_5p"] / c**2  # (not used by the estimate any more)
    torch.manual_seed(0)
    model = NPE(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_freq=8,
        ctx_dim=32,
        flow_transforms=2,
        flow_hidden=(32, 32),
        flow_bins=4,
        n_vm=3,
        pgram_k=256,
        pgram_peaks=4,
        head="anchored",
        n_anchor=3,
        profile=True,
        refine_period=True,
        noise_rescale=True,
    ).eval()
    with torch.no_grad():
        # log kappa_hat is deliberately an input (it says how far the error bars are off), so it
        # is the one input that must change; switch it off to test everything else.
        model.encoder.pgram.peaks[0].weight[:, -1] = 0.0
        emb_a, pk_a = model.encoder.pgram(batch)
        emb_b, pk_b = model.encoder.pgram(other)
        _, anc_a = model.context(batch)
        _, anc_b = model.context(other)
    # sigma -> c sigma scales chi^2 by 1/c^2, so kappa_hat -> kappa_hat / c
    np.testing.assert_allclose(
        pk_a["log_kappa"].numpy() - np.log(c), pk_b["log_kappa"].numpy(), atol=1e-5
    )
    torch.testing.assert_close(emb_a, emb_b, atol=2e-4, rtol=1e-4)
    for key in ("a", "log_tau", "m0", "log_tau_m0", "ecc_feat"):
        torch.testing.assert_close(anc_a[key], anc_b[key], atol=2e-4, rtol=1e-4)


def test_joint_refine_improves_fit_and_gives_positive_covariance():
    """Experimental (off by default): Newton must not lower Delta chi^2; covariance PD."""
    from gdr4.nn.orbit_profile import joint_refine

    prior, sim = _sim(n=200, seed=11)
    sel = np.flatnonzero((sim["snr_a0"] > 30) & (sim["ecc"] > 0.3))[:6]
    batch = to_torch_batch({k: v[sel] for k, v in sim.items()}, prior)
    f0 = torch.tensor(1.0 / sim["period_d"][sel])[:, None] * (1 + 1e-4)
    e0 = torch.tensor(sim["ecc"][sel])[:, None] - 0.02
    m0 = torch.tensor(sim["mean_anom0_rad"][sel])[:, None] + 0.02
    start_gain = orbit_chi2_gain(batch, 1.0 / f0, e0, m0)
    sig = torch.tensor([[1e-6, 0.01, 0.01]], dtype=torch.float64).expand(len(sel), 1, 3)
    jr = joint_refine(batch, f0, e0, m0, sig, prior.ecc_max)
    assert torch.all(jr["gain"] >= start_gain - 1e-6)
    assert jr["pd"].all()
    assert torch.all(torch.linalg.eigvalsh(jr["cov_x"]) > 0)
