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
