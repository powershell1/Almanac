import numpy as np
import torch

from gdr4.model.astrometry import design_matrix_5p
from gdr4.model.linear import wls_fit
from gdr4.nn.data import to_torch_batch
from gdr4.nn.periodogram import (
    circular_periodogram,
    frequency_grid,
    peak_features,
    refine_peaks,
    top_peaks,
)
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


def test_per_star_grid_matches_shared_grid():
    _, batch = _batch()
    freqs = frequency_grid(20.0, 3000.0, 64)
    shared = circular_periodogram(batch, freqs)
    per_star = circular_periodogram(batch, freqs.expand(6, -1).clone())
    torch.testing.assert_close(shared, per_star)


def test_top_peaks_on_known_curve():
    x = torch.linspace(0, 1, 201, dtype=torch.float64)
    pg = 3 * torch.exp(-(((x - 0.3) / 0.01) ** 2)) + 5 * torch.exp(-(((x - 0.7) / 0.01) ** 2))
    idx, valid = top_peaks(pg[None], 3)
    assert idx[0, 0] == 140 and idx[0, 1] == 60  # highest first
    assert valid[0, :2].all()
    flat = torch.zeros(1, 50, dtype=torch.float64)
    _, valid_flat = top_peaks(flat, 3)
    assert valid_flat.sum() <= 1  # a flat curve has (at most) the trivial edge maximum


def test_refined_peak_matches_dense_grid():
    """Sub-grid refinement must land on the same peak a 32x denser grid finds."""
    sim, batch = _batch()
    grid = frequency_grid(20.0, 3000.0, 512)
    pg = circular_periodogram(batch, grid)
    idx, _ = top_peaks(pg, 1)
    f_ref, h_ref = refine_peaks(batch, grid, idx)
    dense = frequency_grid(20.0, 3000.0, 512 * 32)
    pg_dense = circular_periodogram(batch, dense)
    step = (grid[1] - grid[0]).item()
    for i in range(6):
        near = (dense - grid[idx[i, 0]]).abs() <= step
        j = torch.argmax(torch.where(near, pg_dense[i], torch.zeros_like(pg_dense[i])))
        assert abs(f_ref[i, 0].item() - dense[j].item()) < 0.05 * step
        assert h_ref[i, 0] >= pg[i, idx[i, 0]] - 1e-6  # refining never lowers the peak


def test_peak_features_shape_and_order():
    sim, batch = _batch()
    grid = frequency_grid(20.0, 3000.0, 512)
    feats = peak_features(batch, grid, circular_periodogram(batch, grid), 8, (20.0, 3000.0))
    assert feats.shape == (6, 8, 4) and feats.dtype == torch.float32
    assert torch.all(feats[..., 1][:, :-1] >= feats[..., 1][:, 1:])  # sorted by height
    # the highest peak sits near the true period for the brightest low-e star
    i = int(np.argmax(sim["snr_a0"] * (sim["ecc"] < 0.5)))
    lo, hi = np.log(20.0), np.log(3000.0)
    best_p = np.exp(lo + feats[i, 0, 0].item() * (hi - lo))
    assert abs(best_p / sim["period_d"][i] - 1) < 0.02
