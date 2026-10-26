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
