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

``peak_features`` summarises the periodogram by its highest local maxima. Each peak is located
to a fraction of the coarse grid spacing by re-evaluating Delta chi^2 on a fine grid around it
and fitting a parabola through the three highest fine points.
"""

from __future__ import annotations

import math

import torch


def frequency_grid(period_min_d: float, period_max_d: float, k: int) -> torch.Tensor:
    """K frequencies (cycles/day), uniform in frequency, covering the period range."""
    return torch.linspace(1.0 / period_max_d, 1.0 / period_min_d, k, dtype=torch.float64)


@torch.no_grad()
def circular_periodogram(batch: dict, freqs: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Delta chi^2 of the best circular orbit at each frequency, shape (B, K), float64.

    ``freqs`` (cycles/day) is either one grid shared by all stars, shape (K,), or one grid per
    star, shape (B, K).
    """
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
    freqs = freqs.to(dev, f64)
    if freqs.dim() == 1:
        freqs = freqs.expand(t.shape[0], -1)
    out = []
    for s in range(0, freqs.shape[1], chunk):
        ph = 2 * math.pi * freqs[:, s : s + chunk].T[..., None] * t[None]  # (k, B, N)
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


def top_peaks(pg: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid indices of the ``n`` highest local maxima of each row of ``pg`` (B, K).

    Returns ``idx`` (B, n), sorted by height, and ``valid`` (B, n), False where a row has fewer
    than ``n`` local maxima.
    """
    neg = torch.full_like(pg[:, :1], -torch.inf)
    left = torch.cat([neg, pg[:, :-1]], dim=1)
    right = torch.cat([pg[:, 1:], neg], dim=1)
    score = torch.where((pg >= left) & (pg > right), pg, torch.full_like(pg, -torch.inf))
    height, idx = score.topk(n, dim=1)
    return idx, torch.isfinite(height)


@torch.no_grad()
def refine_peaks(batch: dict, grid: torch.Tensor, idx: torch.Tensor, n_fine: int = 9):
    """Sub-grid frequency (cycles/day) and height of each peak in ``idx`` (B, n).

    Evaluates Delta chi^2 at ``n_fine`` points spanning +-1 coarse grid step around each peak,
    then fits a parabola through the highest fine point and its two neighbours.
    """
    grid = grid.to(idx.device, torch.float64)
    step = grid[1] - grid[0]
    offsets = torch.linspace(-1.0, 1.0, n_fine, dtype=torch.float64, device=idx.device) * step
    fine = (grid[idx][..., None] + offsets).clamp_min(1e-6)  # (B, n, n_fine)
    b, n, _ = fine.shape
    y = circular_periodogram(batch, fine.reshape(b, n * n_fine)).reshape(b, n, n_fine)
    j = y.argmax(-1, keepdim=True).clamp(1, n_fine - 2)
    ym, y0, yp = (y.gather(-1, j + d)[..., 0] for d in (-1, 0, 1))
    curv = ym - 2 * y0 + yp
    delta = torch.where(
        curv < 0, 0.5 * (ym - yp) / curv.clamp(max=-1e-12), torch.zeros_like(y0)
    ).clamp(-1.0, 1.0)
    h_fine = offsets[1] - offsets[0]
    freq = fine.gather(-1, j)[..., 0] + delta * h_fine
    height = y0 - 0.25 * (ym - yp) * delta
    return freq, height.clamp_min(0.0)


N_PEAK_FEATURES = 4


@torch.no_grad()
def peak_features(
    batch: dict,
    grid: torch.Tensor,
    pg: torch.Tensor,
    n_peaks: int,
    period_range_d: tuple[float, float],
) -> torch.Tensor:
    """(B, n_peaks, 4) float32 features of the highest periodogram peaks, highest first.

    Per peak: log-period scaled to [0, 1] over ``period_range_d``, log(1 + Delta chi^2),
    that value minus the highest peak's, and a validity flag (missing peaks are all zeros).
    """
    idx, valid = top_peaks(pg, n_peaks)
    freq, height = refine_peaks(batch, grid, idx)
    height = torch.where(valid, height, torch.zeros_like(height))
    order = height.argsort(dim=1, descending=True)
    freq, height, valid = (x.gather(1, order) for x in (freq, height, valid))
    lo, hi = math.log(period_range_d[0]), math.log(period_range_d[1])
    x = (-torch.log(freq) - lo) / (hi - lo)
    lh = torch.log1p(height)
    rel = lh - lh[:, :1]
    feats = torch.stack([x, lh, rel, valid.to(lh.dtype)], dim=-1)
    return torch.where(valid[..., None], feats, torch.zeros_like(feats)).float()
