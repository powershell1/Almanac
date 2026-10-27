"""Eccentric-orbit profile at anchor periods: Delta chi^2(e, M0) maps and (e, M0) anchors.

At fixed (P, e, M0) the along-scan model is linear in the nine parameters (5 single-star +
Thiele-Innes A, B, F, G), so the best fit is exact. As in ``periodogram.py`` the feature is the
chi^2 improvement over the 5-parameter fit,

    Delta chi^2 = b^T G^-1 b,  b = O^T W r,  G = O^T W O - O^T W M5 (M5^T W M5)^-1 M5^T W O,

now with the Thiele-Innes columns O = [X cos psi, X sin psi, Y cos psi, Y sin psi] from Kepler's
equation. Evaluated on an (e, M0) grid at each anchor period, refined around the best grid point
by two rounds of local 3 x 3 grids and a parabola per axis, and turned into Fisher widths
sigma = sqrt(2 / -d2) as for the period (docs/derivations.md, section 5d). float64 throughout;
tests/test_orbit_profile.py checks it against a direct 9-parameter least-squares fit.
"""

from __future__ import annotations

import math

import torch

from gdr4.nn.periodogram import single_star_terms

TWO_PI = 2 * math.pi


def kepler_eccentric_anomaly(m: torch.Tensor, e: torch.Tensor, n_iter: int = 8) -> torch.Tensor:
    """Solve E - e sin E = M (torch, float64, broadcasting), Danby start + Halley-type steps."""
    m_red = torch.remainder(m + math.pi, TWO_PI) - math.pi
    ecc = m_red + 0.85 * e * torch.sign(torch.sin(m_red))
    for _ in range(n_iter):
        s, c = e * torch.sin(ecc), e * torch.cos(ecc)
        f0 = ecc - s - m_red
        d1 = -f0 / (1 - c)
        d2 = -f0 / (1 - c + 0.5 * d1 * s)
        ecc = ecc - f0 / (1 - c + 0.5 * d2 * s + d2 * d2 * c / 6)
    return ecc + (m - m_red)


@torch.no_grad()
def orbit_chi2_gain(
    batch: dict, period_d: torch.Tensor, ecc: torch.Tensor, m0_rad: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """Delta chi^2 of the best Keplerian fit at each (P, e, M0); inputs and output (B, S)."""
    f64 = torch.float64
    t, cp, sp, w, r, m5, h5_inv = single_star_terms(batch)
    period_d, ecc, m0_rad = (x.to(t.device, f64) for x in (period_d, ecc, m0_rad))
    out = []
    for s in range(0, period_d.shape[1], chunk):
        p, e, m0 = (x[:, s : s + chunk, None] for x in (period_d, ecc, m0_rad))  # (B, k, 1)
        big_e = kepler_eccentric_anomaly(m0 + TWO_PI * t[:, None, :] / p, e)  # (B, k, N)
        x = torch.cos(big_e) - e
        y = torch.sqrt(1 - e * e) * torch.sin(big_e)
        c, sn = cp[:, None], sp[:, None]
        o = torch.stack([x * c, x * sn, y * c, y * sn], -1)  # (B, k, N, 4)
        ow = o * w[:, None, :, None]
        b = torch.einsum("bkno,bn->bko", ow, r)
        oo = torch.einsum("bkno,bknp->bkop", ow, o)
        om = torch.einsum("bkno,bnp->bkop", ow, m5)
        g = oo - om @ h5_inv[:, None] @ om.transpose(-1, -2)
        g = g + 1e-9 * torch.eye(4, dtype=f64, device=t.device)
        sol = torch.linalg.solve(g, b[..., None])[..., 0]
        out.append((b * sol).sum(-1).clamp_min(0.0))
    return torch.cat(out, dim=1)


def _parabola(ym, y0, yp):
    """Vertex offset (in steps, clamped to +-1) and second difference of a 3-point parabola."""
    curv = ym - 2 * y0 + yp
    delta = torch.where(
        curv < 0, 0.5 * (ym - yp) / curv.clamp(max=-1e-12), torch.zeros_like(y0)
    ).clamp(-1.0, 1.0)
    return delta, curv


def _gain(batch, period, ecc, m0):
    """orbit_chi2_gain for parameter tensors of shape (B, K, J) -> (B, K, J)."""
    b, k, j = period.shape
    return orbit_chi2_gain(
        batch,
        period.reshape(b, -1),
        ecc.expand(b, k, j).reshape(b, -1),
        m0.expand(b, k, j).reshape(b, -1),
    ).reshape(b, k, j)


@torch.no_grad()
def _refine_e_m0(batch, period, e_c, m_c, de, dm, shrinks, ecc_max):
    """Local 3 x 3 refinement of (e, M0) at fixed period (B, K); returns e, M0, d2_e, d2_m."""
    dev, f64 = e_c.device, torch.float64
    b, k = period.shape
    lo_e, hi_e = 0.005, ecc_max - 0.005
    off = torch.tensor([-1.0, 0.0, 1.0], dtype=f64, device=dev)
    for shrink in shrinks:
        e_c = e_c.clamp(lo_e + shrink * de, hi_e - shrink * de)
        ge = (e_c[..., None, None] + shrink * de * off[:, None]).expand(b, k, 3, 3)
        gm = (m_c[..., None, None] + shrink * dm * off[None, :]).expand(b, k, 3, 3)
        y = _gain(
            batch, period[..., None].expand(b, k, 9), ge.reshape(b, k, 9), gm.reshape(b, k, 9)
        ).reshape(b, k, 3, 3)
        dlt_e, curv_e = _parabola(y[..., 0, 1], y[..., 1, 1], y[..., 2, 1])
        dlt_m, curv_m = _parabola(y[..., 1, 0], y[..., 1, 1], y[..., 1, 2])
        e_c = (e_c + dlt_e * shrink * de).clamp(lo_e, hi_e)
        m_c = torch.remainder(m_c + dlt_m * shrink * dm, TWO_PI)
        h_e, h_m = shrink * de, shrink * dm
    return e_c, m_c, curv_e / h_e**2, curv_m / h_m**2


@torch.no_grad()
def _refine_frequency(batch, freq, ecc, m0, step, n_scan: int = 0):
    """Keplerian Delta chi^2 at fixed (e, M0): optional scan of +-n_scan steps, then a parabola.

    ``freq``, ``ecc``, ``m0``, ``step`` (cycles/day) have shape (B, K).
    """
    dev, f64 = freq.device, torch.float64
    if n_scan:
        j = torch.arange(-n_scan, n_scan + 1, dtype=f64, device=dev)
        fs = (freq[..., None] + step[..., None] * j).clamp_min(1e-6)
        y = _gain(batch, 1.0 / fs, ecc[..., None], m0[..., None])
        freq = fs.gather(-1, y.argmax(-1, keepdim=True))[..., 0]
    off = torch.tensor([-1.0, 0.0, 1.0], dtype=f64, device=dev)
    fs = (freq[..., None] + step[..., None] * off).clamp_min(1e-6)
    y = _gain(batch, 1.0 / fs, ecc[..., None], m0[..., None])
    dlt, _ = _parabola(y[..., 0], y[..., 1], y[..., 2])
    return freq + dlt * step


@torch.no_grad()
def profile_anchors(
    batch: dict,
    period_d: torch.Tensor,
    ecc_max: float,
    n_e: int = 8,
    n_m: int = 16,
    sigma_f: torch.Tensor | None = None,
) -> dict:
    """(e, M0) maps and anchors at each anchor period ``period_d`` (B, K).

    If ``sigma_f`` (B, K), the circular-orbit Fisher width in frequency, is given, the period is
    also re-refined with the exact Keplerian Delta chi^2 by coordinate ascent: (e, M0) at the
    circular period, then the frequency at fixed (e, M0) (scan of +-24 sigma_f, then parabolas),
    then (e, M0) again at the new period, then the frequency once more. This removes the shift of
    the circular-periodogram peak for eccentric orbits (docs/derivations.md, section 5e).

    Returns
    -------
    dict with ``map`` (B, K, n_e, n_m) log(1 + Delta chi^2) on the grid at the input period;
    refined ``ecc``, ``m0`` and ``period`` (B, K); Fisher widths ``sigma_e``, ``sigma_m0``
    (B, K) (inf where the profile is not curved, e.g. M0 for a near-circular orbit).
    """
    dev, f64 = batch["t_d"].device, torch.float64
    b, k = period_d.shape
    period = period_d.to(f64)
    e_grid = (torch.arange(n_e, dtype=f64, device=dev) + 0.5) * ecc_max / n_e
    m_grid = torch.arange(n_m, dtype=f64, device=dev) * TWO_PI / n_m
    ee, mm = torch.meshgrid(e_grid, m_grid, indexing="ij")
    gain = _gain(
        batch, period[..., None].expand(b, k, n_e * n_m), ee.reshape(1, 1, -1), mm.reshape(1, 1, -1)
    ).reshape(b, k, n_e, n_m)
    best = gain.reshape(b, k, -1).argmax(-1)
    e_c, m_c = e_grid[best // n_m], m_grid[best % n_m]
    de, dm = ecc_max / n_e, TWO_PI / n_m
    e_c, m_c, d2e, d2m = _refine_e_m0(batch, period, e_c, m_c, de, dm, (1.0, 0.25), ecc_max)
    if sigma_f is not None:
        sig = sigma_f.to(f64)
        freq = _refine_frequency(batch, 1.0 / period, e_c, m_c, 2.0 * sig, n_scan=12)
        freq = _refine_frequency(batch, freq, e_c, m_c, 0.5 * sig)
        e_c, m_c, d2e, d2m = _refine_e_m0(
            batch, 1.0 / freq, e_c, m_c, de, dm, (0.25, 0.0625), ecc_max
        )
        freq = _refine_frequency(batch, freq, e_c, m_c, 0.125 * sig)
        period = 1.0 / freq
    inf = torch.full_like(d2e, torch.inf)
    sig_e = torch.where(d2e < 0, torch.sqrt(2.0 / (-d2e).clamp_min(1e-300)), inf)
    sig_m = torch.where(d2m < 0, torch.sqrt(2.0 / (-d2m).clamp_min(1e-300)), inf)
    return {
        "map": torch.log1p(gain),
        "ecc": e_c,
        "m0": m_c,
        "period": period,
        "sigma_e": sig_e,
        "sigma_m0": sig_m,
    }
