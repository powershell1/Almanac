"""Benchmark (2026-10-27): can a conditional flow head represent posteriors 1e-4 to 1 wide?

Toy amortised problem: context c ~ N(0, I_8); target u | c ~ N(m(c), diag(s(c)^2)) with
log10 s(c) in [-4, 0]. Compares the plain zuko NSF head with a moment-standardised head
(an auxiliary Gaussian head predicts mean and width; the spline models (u - mu) / sigma).
Run from the repo root: python scripts/diagnostics/bench_flow_head.py [steps]
"""

import math
import sys
import time

import numpy as np
import torch
import zuko
from torch import nn

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def truth(c):
    m = torch.stack([2 * torch.tanh(c[:, 0]), c[:, 1]], -1)
    log10_s = -2 + 2 * torch.tanh(c[:, 2])  # spans 1e-4 .. 1
    s = torch.stack([10**log10_s, 10 ** (log10_s + 0.5).clamp(max=0)], -1)
    return m, s


class Plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(8, 128), nn.GELU(), nn.Linear(128, 64))
        self.flow = zuko.flows.NSF(
            features=2, context=64, transforms=4, bins=8, hidden_features=[128, 128]
        )

    def loss(self, u, c):
        return -self.log_prob(u, c).mean()

    def log_prob(self, u, c):
        return self.flow(self.enc(c)).log_prob(u)

    def sample(self, c, n):
        return self.flow(self.enc(c)).sample((n,))


class Standardised(Plain):
    def __init__(self):
        super().__init__()
        self.moments = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 4))

    def _ms(self, ctx):
        mu, log_s = self.moments(ctx).chunk(2, -1)
        return mu, log_s.clamp(-15, 3)

    def log_prob(self, u, c):
        ctx = self.enc(c)
        mu, log_s = (x.detach() for x in self._ms(ctx))
        return self.flow(ctx).log_prob((u - mu) * torch.exp(-log_s)) - log_s.sum(-1)

    def loss(self, u, c):
        ctx = self.enc(c)
        mu, log_s = self._ms(ctx)
        gauss = (0.5 * ((u - mu) * torch.exp(-log_s)) ** 2 + log_s).sum(-1)  # aux Gaussian NLL
        return -self.log_prob(u, c).mean() + gauss.mean()

    def sample(self, c, n):
        ctx = self.enc(c)
        mu, log_s = self._ms(ctx)
        return mu + torch.exp(log_s) * self.flow(ctx).sample((n,))


def run(cls):
    torch.manual_seed(0)
    model = cls()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(STEPS):
        c = torch.randn(512, 8)
        m, s = truth(c)
        u = m + s * torch.randn_like(m)
        loss = model.loss(u, c)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
    model.eval()
    gen = torch.Generator().manual_seed(1)
    c = torch.randn(2000, 8, generator=gen)
    m, s = truth(c)
    with torch.no_grad():
        u = m + s * torch.randn(m.shape, generator=gen)
        gap = -model.log_prob(u, c) - (torch.log(s) + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
        x = model.sample(c, 400)
    ratio = (x.std(0) / s)[:, 0].numpy()
    bins = np.log10(s[:, 0].numpy())
    print(
        f"{cls.__name__:12s} {STEPS} steps, {time.time() - t0:.0f}s | mean NLL gap {gap.mean():.2f} nats"
    )
    for lo, hi in ((-4, -3), (-3, -2), (-2, -1), (-1, 0.1)):
        sel = (bins >= lo) & (bins < hi)
        print(
            f"    true width 1e{lo:+d}..1e{hi:+.0f}: width ratio model/true median "
            f"{np.median(ratio[sel]):7.2f}  NLL gap {gap[sel].mean():6.2f}"
        )


run(Plain)
run(Standardised)
