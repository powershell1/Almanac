import math

import numpy as np
import pytest
import torch

from gdr4.nn.data import to_torch_batch
from gdr4.nn.model import NPE, VonMisesMixture
from gdr4.sim.noise import get_noise_model
from gdr4.sim.priors import Prior
from gdr4.sim.scan import build_cache_toy_nsl
from gdr4.sim.simulate import simulate_batch
from gdr4.utils import REPO_ROOT, load_config

SMALL = dict(
    d_model=32,
    n_layers=1,
    n_heads=2,
    n_freq=8,
    ctx_dim=32,
    flow_transforms=2,
    flow_hidden=(32, 32),
    flow_bins=4,
    n_vm=3,
)


@pytest.fixture(scope="module")
def batch_and_prior():
    prior = Prior.from_config(load_config(REPO_ROOT / "configs" / "smoke.yaml")["prior"])
    sim = simulate_batch(
        np.random.default_rng(0), 16, prior, build_cache_toy_nsl(2), get_noise_model("nominal")
    )
    return to_torch_batch(sim, prior), prior


def test_von_mises_mixture_normalised():
    torch.manual_seed(0)
    vm = VonMisesMixture(ctx_dim=4, hidden=16, k=3)
    m0 = torch.linspace(0, 2 * math.pi, 20001)
    u2, ctx = torch.randn(1, 2).expand(len(m0), 2), torch.randn(1, 4).expand(len(m0), 4)
    dens = torch.exp(vm.log_prob(m0, u2, ctx))
    assert abs(torch.trapezoid(dens, m0).item() - 1.0) < 1e-3
    # periodic: same density at 0 and 2 pi
    assert torch.allclose(dens[0], dens[-1], rtol=1e-4)


def test_von_mises_large_kappa_stable():
    vm = VonMisesMixture(ctx_dim=2, hidden=8, k=1)
    with torch.no_grad():
        vm.net[-1].weight.zero_()
        vm.net[-1].bias.copy_(torch.tensor([0.0, 1.0, 0.0, 14.0]))  # mu = 0, kappa = e^14
    lp = vm.log_prob(torch.tensor([0.0, 1e-3]), torch.zeros(2, 2), torch.zeros(2, 2))
    assert torch.isfinite(lp).all()
    expected_peak = 0.5 * math.log(math.exp(14.0) / (2 * math.pi))  # Gaussian limit
    assert abs(lp[0].item() - expected_peak) < 0.05


@pytest.mark.parametrize("features", ["raw", "normalized"])
def test_npe_shapes_and_gradients(batch_and_prior, features):
    batch, _ = batch_and_prior
    torch.manual_seed(0)
    model = NPE(features=features, **SMALL)
    loss = model(batch, batch["v"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    v, lq = model.sample(batch, 5, with_log_prob=True)
    assert v.shape == (5, 16, 3) and lq.shape == (5, 16)
    assert ((v[..., 2] >= 0) & (v[..., 2] < 2 * math.pi)).all()


def test_padding_does_not_change_output(batch_and_prior):
    batch, _ = batch_and_prior
    torch.manual_seed(0)
    model = NPE(**SMALL).eval()
    padded = {k: v for k, v in batch.items()}
    for k in ("t_d", "psi_rad", "plx_factor", "sigma_mas", "resid5_mas"):
        padded[k] = torch.cat([batch[k], torch.full((16, 7), 3.0)], dim=1)
    padded["mask"] = torch.cat([batch["mask"], torch.zeros(16, 7, dtype=bool)], dim=1)
    a = model.log_prob(batch, batch["v"])
    b = model.log_prob(padded, batch["v"])
    assert torch.allclose(a, b, atol=1e-4)


def test_npe_learns_toy_period(batch_and_prior):
    """A few hundred steps on a fixed batch must reduce the loss substantially (overfit check)."""
    batch, _ = batch_and_prior
    torch.manual_seed(1)
    model = NPE(**SMALL)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = model(batch, batch["v"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first - 1.0


def test_npe_with_periodogram_branch(batch_and_prior):
    batch, _ = batch_and_prior
    torch.manual_seed(0)
    model = NPE(**{**SMALL, "pgram_k": 64})
    loss = model(batch, batch["v"])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.pgram.out.weight.grad is not None
    v = model.sample(batch, 3)
    assert v.shape == (3, 16, 3)


def test_npe_with_periodogram_peaks(batch_and_prior):
    batch, _ = batch_and_prior
    torch.manual_seed(0)
    model = NPE(**{**SMALL, "pgram_k": 64, "pgram_peaks": 4})
    loss = model(batch, batch["v"])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.pgram.peaks[0].weight.grad is not None
    # padding must not change anything, including the peak features
    model.eval()
    padded = dict(batch)
    for k in ("t_d", "psi_rad", "plx_factor", "sigma_mas", "resid5_mas"):
        padded[k] = torch.cat([batch[k], torch.full((16, 7), 3.0)], dim=1)
    padded["mask"] = torch.cat([batch["mask"], torch.zeros(16, 7, dtype=bool)], dim=1)
    assert torch.allclose(
        model.log_prob(batch, batch["v"]), model.log_prob(padded, batch["v"]), atol=1e-4
    )


def test_peaks_require_periodogram():
    with pytest.raises(ValueError):
        NPE(**{**SMALL, "pgram_peaks": 4})


ANCHORED = {
    **SMALL,
    "pgram_k": 64,
    "pgram_peaks": 4,
    "head": "anchored",
    "n_anchor": 3,
    "target_period_d": (30.0, 2000.0),
}


def _anchored_model_and_context(batch, seed=0):
    torch.manual_seed(seed)
    model = NPE(**ANCHORED).eval()
    with torch.no_grad():  # random (non-zero) mixture weights, shifts and scales
        model.period.par[-1].weight.normal_(0, 0.3)
        model.period.par[-1].bias.normal_(0, 0.3)
    return model, model.context(batch)


def test_anchored_period_density_integrates_to_one(batch_and_prior):
    batch, _ = batch_and_prior
    model, (ctx, anc) = _anchored_model_and_context(batch)
    for i in range(4):
        c, a = ctx[i : i + 1], {k: v[i : i + 1] for k, v in anc.items()}
        # broad grid plus fine grids around every anchor (components can be ~1e-4 wide)
        grids = [torch.linspace(-12, 12, 40001)]
        for k in range(3):
            w = 60 * torch.exp(a["log_tau"][0, k] + 5.0)
            mid = a["a"][0, k] + torch.exp(a["log_tau"][0, k]) * 50 * torch.tensor([-1.0, 1.0])
            lo, hi = min(mid[0] - w, a["a"][0, k] - w), max(mid[1] + w, a["a"][0, k] + w)
            grids.append(torch.linspace(lo.item(), hi.item(), 40001))
        u = torch.sort(torch.cat(grids)).values
        with torch.no_grad():
            dens = torch.exp(model.period.log_prob(u[:, None].double().float(), c, a))[:, 0]
        assert abs(torch.trapezoid(dens, u).item() - 1.0) < 2e-3, i


def test_anchored_samples_match_density(batch_and_prior):
    batch, _ = batch_and_prior
    model, (ctx, anc) = _anchored_model_and_context(batch, seed=1)
    i = 0
    c, a = ctx[i : i + 1], {k: v[i : i + 1] for k, v in anc.items()}
    with torch.no_grad():
        s = model.period.sample(40000, c, a)[:, 0]
        lo, hi = s.quantile(0.2).item(), s.quantile(0.7).item()
        u = torch.linspace(lo, hi, 20001)
        mass = torch.trapezoid(torch.exp(model.period.log_prob(u[:, None], c, a))[:, 0], u)
    frac = ((s > lo) & (s < hi)).float().mean().item()
    assert abs(mass.item() - frac) < 0.015


def test_anchored_npe_trains_samples_and_ignores_padding(batch_and_prior):
    batch, _ = batch_and_prior
    torch.manual_seed(0)
    model = NPE(**ANCHORED)
    loss = model(batch, batch["v"])
    loss.backward()
    assert torch.isfinite(loss)
    for name in ("period.par.0.weight", "ecc_flow.transform.transforms.0.hyper.0.weight"):
        assert dict(model.named_parameters())[name].grad is not None, name
    model.eval()
    v, lq = model.sample(batch, 5, with_log_prob=True)
    assert v.shape == (5, 16, 3) and torch.isfinite(lq).all()
    assert ((v[..., 2] >= 0) & (v[..., 2] < 2 * math.pi)).all()
    padded = dict(batch)
    for k in ("t_d", "psi_rad", "plx_factor", "sigma_mas", "resid5_mas"):
        padded[k] = torch.cat([batch[k], torch.full((16, 7), 3.0)], dim=1)
    padded["mask"] = torch.cat([batch["mask"], torch.zeros(16, 7, dtype=bool)], dim=1)
    assert torch.allclose(
        model.log_prob(batch, batch["v"]), model.log_prob(padded, batch["v"]), atol=1e-3
    )


def test_anchored_requires_enough_peaks():
    with pytest.raises(ValueError):
        NPE(**{**ANCHORED, "pgram_peaks": 2})
