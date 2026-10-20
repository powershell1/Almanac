import numpy as np
import torch

from gdr4.model.astrometry import design_matrix_5p
from gdr4.model.linear import wls_fit
from gdr4.nn.data import to_torch_batch
from gdr4.nn.periodogram import circular_periodogram, frequency_grid
from gdr4.sim.noise import get_noise_model
from gdr4.sim.priors import Prior
from gdr4.sim.scan import build_cache_toy_nsl
from gdr4.sim.simulate import simulate_batch, star_view
from gdr4.utils import REPO_ROOT, load_config


def _batch():
    prior = Prior.from_config(load_config(REPO_ROOT / "configs" / "smoke.yaml")["prior"])
    sim = simulate_batch(
        np.random.default_rng(1), 6, prior, build_cache_toy_nsl(2), get_noise_model("nominal")
    )
    return sim, to_torch_batch(sim, prior)


def test_matches_direct_nine_parameter_fit():
    sim, _ = _batch()
    freqs = torch.tensor([1 / 37.0, 1 / 400.0, 1 / 1500.0], dtype=torch.float64)
    for i in range(6):
        s = star_view(sim, i)
        m5 = design_matrix_5p(s["t_d"], s["psi_rad"], s["plx_factor"])
        _, resid, chi5, _ = wls_fit(m5, s["w_mas"], s["sigma_mas"])
        one = {k: torch.tensor(s[k])[None] for k in ("t_d", "psi_rad", "plx_factor", "sigma_mas")}
        one["resid5_mas"] = torch.tensor(resid)[None]  # same float64 inputs for both paths
        one["mask"] = torch.ones_like(one["t_d"], dtype=torch.bool)
        pg = circular_periodogram(one, freqs).numpy()[0]
        for k, f in enumerate(freqs.numpy()):
            ph = 2 * np.pi * f * s["t_d"]
            c, sn, cp, sp = np.cos(ph), np.sin(ph), np.cos(s["psi_rad"]), np.sin(s["psi_rad"])
            m9 = np.column_stack([m5, c * cp, c * sp, sn * cp, sn * sp])
            _, _, chi9, _ = wls_fit(m9, s["w_mas"], s["sigma_mas"])
            np.testing.assert_allclose(pg[k], chi5 - chi9, rtol=1e-7, atol=1e-6 * chi5)


def test_peak_near_true_period_for_bright_circular_like_orbit():
    sim, batch = _batch()
    freqs = frequency_grid(20.0, 3000.0, 2048)
    pg = circular_periodogram(batch, freqs).numpy()
    i = int(np.argmax(sim["snr_a0"] * (sim["ecc"] < 0.5)))
    best_p = 1 / freqs.numpy()[np.argmax(pg[i])]
    assert abs(best_p / sim["period_d"][i] - 1) < 0.05
