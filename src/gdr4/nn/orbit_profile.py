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


@torch.no_grad()
def profile_anchors(
    batch: dict, period_d: torch.Tensor, ecc_max: float, n_e: int = 8, n_m: int = 16
) -> dict:
    """(e, M0) maps and anchors at each anchor period ``period_d`` (B, K).

    Returns
    -------
    dict with ``map`` (B, K, n_e, n_m) log(1 + Delta chi^2) on the grid; refined ``ecc``,
    ``m0`` (B, K); Fisher widths ``sigma_e``, ``sigma_m0`` (B, K) (inf where the profile is not
    curved, e.g. M0 for a near-circular orbit).
    """
    dev, f64 = batch["t_d"].device, torch.float64
    b, k = period_d.shape
    e_grid = (torch.arange(n_e, dtype=f64, device=dev) + 0.5) * ecc_max / n_e
    m_grid = torch.arange(n_m, dtype=f64, device=dev) * TWO_PI / n_m
    ee, mm = torch.meshgrid(e_grid, m_grid, indexing="ij")
    p = period_d.to(f64)[..., None].expand(b, k, n_e * n_m).reshape(b, -1)
    e_all = ee.reshape(-1).repeat(k).expand(b, -1)  # (B, K * n_e * n_m), grid repeated per anchor
    m_all = mm.reshape(-1).repeat(k).expand(b, -1)
    gain = orbit_chi2_gain(batch, p, e_all, m_all).reshape(b, k, n_e, n_m)
    best = gain.reshape(b, k, -1).argmax(-1)
    e_c, m_c = e_grid[best // n_m], m_grid[best % n_m]
    de, dm = ecc_max / n_e, TWO_PI / n_m
    lo_e, hi_e = 0.005, ecc_max - 0.005
    for shrink in (1.0, 0.25):  # two rounds of local 3 x 3 refinement
        e_c = e_c.clamp(lo_e + shrink * de, hi_e - shrink * de)
        off = torch.tensor([-1.0, 0.0, 1.0], dtype=f64, device=dev)
        ge = (e_c[..., None, None] + shrink * de * off[:, None]).expand(b, k, 3, 3)
        gm = (m_c[..., None, None] + shrink * dm * off[None, :]).expand(b, k, 3, 3)
        y = orbit_chi2_gain(
            batch,
            period_d.to(f64)[..., None].expand(b, k, 9).reshape(b, -1),
            ge.reshape(b, -1),
            gm.reshape(b, -1),
        ).reshape(b, k, 3, 3)
        dlt_e, curv_e = _parabola(y[..., 0, 1], y[..., 1, 1], y[..., 2, 1])
        dlt_m, curv_m = _parabola(y[..., 1, 0], y[..., 1, 1], y[..., 1, 2])
        e_c = (e_c + dlt_e * shrink * de).clamp(lo_e, hi_e)
        m_c = torch.remainder(m_c + dlt_m * shrink * dm, TWO_PI)
        h_e, h_m = shrink * de, shrink * dm
    d2e, d2m = curv_e / h_e**2, curv_m / h_m**2
    inf = torch.full_like(d2e, torch.inf)
    sig_e = torch.where(d2e < 0, torch.sqrt(2.0 / (-d2e).clamp_min(1e-300)), inf)
    sig_m = torch.where(d2m < 0, torch.sqrt(2.0 / (-d2m).clamp_min(1e-300)), inf)
    return {"map": torch.log1p(gain), "ecc": e_c, "m0": m_c, "sigma_e": sig_e, "sigma_m0": sig_m}
