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


def single_star_terms(batch: dict):
    """float64 pieces shared by the periodogram and the orbit profile.

    Returns t (B,N), cos psi, sin psi, weights W = mask / sigma^2, residual r, the 5-parameter
    design matrix M5 (B,N,5) and (M5^T W M5)^-1 (B,5,5).
    """
    dev, f64 = batch["t_d"].device, torch.float64
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
    return t, cp, sp, w, r, m5, torch.linalg.inv(h5)


@torch.no_grad()
def circular_periodogram(batch: dict, freqs: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Delta chi^2 of the best circular orbit at each frequency, shape (B, K), float64.

    ``freqs`` (cycles/day) is either one grid shared by all stars, shape (K,), or one grid per
    star, shape (B, K).
    """
    f64 = torch.float64
    t, cp, sp, w, r, m5, h5_inv = single_star_terms(batch)
    dev = t.device
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
    """Sub-grid frequency (cycles/day), height and curvature of each peak in ``idx`` (B, n).

    Evaluates Delta chi^2 at ``n_fine`` points spanning +-1 coarse grid step around each peak,
    then fits a parabola through the highest fine point and its two neighbours.

    Returns
    -------
    freq, height, d2 : (B, n) float64
        Vertex frequency, vertex height and second derivative d^2(Delta chi^2)/df^2
        (negative at a proper maximum).
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
    return freq, height.clamp_min(0.0), curv / h_fine**2


@torch.no_grad()
def analyse_peaks(batch: dict, grid: torch.Tensor, pg: torch.Tensor, n: int) -> dict:
    """Top ``n`` refined peaks of ``pg``, highest first: freq, height, d2, valid (B, n)."""
    idx, valid = top_peaks(pg, n)
    freq, height, d2 = refine_peaks(batch, grid, idx)
    height = torch.where(valid, height, torch.zeros_like(height))
    order = height.argsort(dim=1, descending=True)
    return {
        k: x.gather(1, order)
        for k, x in (("freq", freq), ("height", height), ("d2", d2), ("valid", valid))
    }


N_PEAK_FEATURES = 4


def features_from_peaks(peaks: dict, period_range_d: tuple[float, float]) -> torch.Tensor:
    """(B, n, 4) float32: log-period scaled to [0, 1] over ``period_range_d``,
    log(1 + Delta chi^2), that value minus the highest peak's, and a validity flag."""
    valid = peaks["valid"]
    lo, hi = math.log(period_range_d[0]), math.log(period_range_d[1])
    x = (-torch.log(peaks["freq"]) - lo) / (hi - lo)
    lh = torch.log1p(peaks["height"])
    rel = lh - lh[:, :1]
    feats = torch.stack([x, lh, rel, valid.to(lh.dtype)], dim=-1)
    return torch.where(valid[..., None], feats, torch.zeros_like(feats)).float()


@torch.no_grad()
def peak_features(
    batch: dict,
    grid: torch.Tensor,
    pg: torch.Tensor,
    n_peaks: int,
    period_range_d: tuple[float, float],
) -> torch.Tensor:
    """(B, n_peaks, 4) float32 features of the highest periodogram peaks, highest first."""
    return features_from_peaks(analyse_peaks(batch, grid, pg, n_peaks), period_range_d)


@torch.no_grad()
def anchors_from_peaks(
    peaks: dict,
    n: int,
    target_period_d: tuple[float, float],
    freq: torch.Tensor | None = None,
    log_kappa: torch.Tensor | None = None,
) -> dict:
    """Anchors for the period posterior from the ``n`` highest peaks (see docs, section 5c).

    For each peak: its position in the flow's target coordinate
    ``u_P = probit((log P - log P_min) / (log P_max - log P_min))``, and the circular-orbit
    Fisher width there, ``tau = sigma_f / (f R phi(u_P))`` with
    ``sigma_f = sqrt(2 / -d2)`` (Delta chi^2 = -2 Delta log L, so d2 log L / df2 = d2 / 2).
    Peaks outside the prior period range or without negative curvature are invalid; invalid
    entries are set to a = 0, log_tau = 0 so that downstream arithmetic stays finite.
    ``freq`` (B, n) overrides the anchor positions (e.g. Keplerian re-refined frequencies); the
    width still comes from the circular-periodogram curvature. ``log_kappa`` (B,) widens every
    width by the estimated noise scale (Fisher widths scale linearly with the noise).
    """
    f = peaks["freq"][:, :n] if freq is None else freq
    d2 = peaks["d2"][:, :n]
    lo, hi = math.log(target_period_d[0]), math.log(target_period_d[1])
    frac = (-torch.log(f) - lo) / (hi - lo)
    valid = peaks["valid"][:, :n] & (d2 < 0) & (frac > 0) & (frac < 1)
    frac = frac.clamp(1e-6, 1 - 1e-6)
    a = torch.special.ndtri(frac)
    phi = torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    sigma_f = torch.sqrt(2.0 / (-d2).clamp_min(1e-300))
    log_tau = torch.log(sigma_f / (f * (hi - lo) * phi))
    if log_kappa is not None:
        log_tau = log_tau + log_kappa.to(log_tau)[:, None]
    log_tau = log_tau.clamp(math.log(1e-7), math.log(3.0))
    zero = torch.zeros_like(a)
    return {
        "a": torch.where(valid, a, zero).float(),
        "log_tau": torch.where(valid, log_tau, zero).float(),
        "valid": valid,
    }
