"""Analytic circular-orbit periodogram used as an input feature (no learnable parameters).

For a circular orbit (e = 0) the along-scan model at a trial frequency f is *linear* in nine
parameters: the five single-star ones plus four orbital columns

    cos(2 pi f t) cos psi, cos(2 pi f t) sin psi, sin(2 pi f t) cos psi, sin(2 pi f t) sin psi

(any phase and orientation is a linear combination of these). The periodogram value is the
chi^2 improvement over the 5-parameter fit, Delta chi^2(f) = chi^2_5 - chi^2_9(f).

With r the weighted 5-parameter residual (already orthogonal to the 5-parameter columns in the
C^-1 metric), O the four orbital columns, M5 the single-star columns and W = C^-1:

    b = O^T W r
    G = O^T W O - O^T W M5 (M5^T W M5)^-1 M5^T W O
    Delta chi^2 = b^T G^-1 b

(the projection of r onto the part of the orbital columns not explained by M5).
Computed in float64; see tests/test_periodogram.py for the check against a direct 9-parameter
least-squares fit.
"""

from __future__ import annotations

import math

import torch


def frequency_grid(period_min_d: float, period_max_d: float, k: int) -> torch.Tensor:
    """K frequencies (cycles/day), uniform in frequency, covering the period range."""
    return torch.linspace(1.0 / period_max_d, 1.0 / period_min_d, k, dtype=torch.float64)


@torch.no_grad()
def circular_periodogram(batch: dict, freqs: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Delta chi^2 of the best circular orbit at each frequency, shape (B, K), float64."""
    dev = batch["t_d"].device
    f64 = torch.float64
    t = batch["t_d"].to(f64)
    psi = batch["psi_rad"].to(f64)
    mask = batch["mask"]
    sigma = batch["sigma_mas"].to(f64).clamp_min(1e-6)
    w = torch.where(mask, 1.0 / sigma**2, torch.zeros_like(sigma))  # (B, N)
    r = batch["resid5_mas"].to(f64) * mask
    cp, sp = torch.cos(psi), torch.sin(psi)
    t_yr = t / 365.25
    m5 = torch.stack([sp, cp, batch["plx_factor"].to(f64), t_yr * sp, t_yr * cp], -1)  # (B,N,5)
    h5 = torch.einsum("bn,bnp,bnq->bpq", w, m5, m5)
    h5 = h5 + 1e-9 * torch.eye(5, dtype=f64, device=dev)  # guard fully masked rows
    h5_inv = torch.linalg.inv(h5)
    freqs = freqs.to(dev)
    out = []
    for s in range(0, len(freqs), chunk):
        ph = 2 * math.pi * freqs[s : s + chunk, None, None] * t[None]  # (k, B, N)
        c, sn = torch.cos(ph), torch.sin(ph)
        o = torch.stack([c * cp, c * sp, sn * cp, sn * sp], -1)  # (k, B, N, 4)
        ow = o * w[None, ..., None]
        b = torch.einsum("kbno,bn->kbo", ow, r)
        oo = torch.einsum("kbno,kbnp->kbop", ow, o)
        om = torch.einsum("kbno,bnp->kbop", ow, m5)  # (k, B, 4, 5)
        g = oo - om @ h5_inv[None] @ om.transpose(-1, -2)
        g = g + 1e-9 * torch.eye(4, dtype=f64, device=dev)
        x = torch.linalg.solve(g, b[..., None])[..., 0]
        out.append((b * x).sum(-1).clamp_min(0.0).T)  # (B, k)
    return torch.cat(out, dim=1)
