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
    joint: bool = False,
    noise_corr: torch.Tensor | None = None,
    regrid: bool = False,
    n_starts: int = 1,
) -> dict:
    """(e, M0) maps and anchors at each anchor period ``period_d`` (B, K).

    If ``sigma_f`` (B, K), the circular-orbit Fisher width in frequency, is given, the period is
    also re-refined with the exact Keplerian Delta chi^2 by coordinate ascent: (e, M0) at the
    circular period, then the frequency at fixed (e, M0) (scan of +-24 sigma_f, then parabolas),
    then (e, M0) again at the new period, then the frequency once more. This removes the shift of
    the circular-periodogram peak for eccentric orbits (docs/derivations.md, section 5e).

    Returns
    -------
    dict with ``map`` (B, K, n_e, n_m) log(1 + Delta chi^2) on the grid at the input period
    (``gain_grid``: the same without the log); ``gain_best`` (B, K), Delta chi^2 at the refined
    optimum;
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
    extra = {}
    if sigma_f is not None:
        sig = sigma_f.to(f64)
        freq = _refine_frequency(batch, 1.0 / period, e_c, m_c, 2.0 * sig, n_scan=12)
        freq = _refine_frequency(batch, freq, e_c, m_c, 0.5 * sig)
        if joint:  # joint Newton ascent in (f, e, M0) + Laplace covariance (section 5f)
            corr = (
                torch.ones_like(sig)
                if noise_corr is None
                else noise_corr.to(f64)[:, None].expand_as(sig)
            )

            def width(d2, default):
                w = torch.sqrt(2.0 / (-d2).clamp_min(1e-300)) * corr
                return torch.where(d2 < 0, w, torch.full_like(w, default))

            start = torch.stack([sig, width(d2e, 0.05), width(d2m, 0.3)], -1)
            if regrid:  # search (e, M0) again at the Keplerian-refined period
                g2 = _gain(
                    batch,
                    (1.0 / freq)[..., None].expand(b, k, n_e * n_m),
                    ee.reshape(1, 1, -1),
                    mm.reshape(1, 1, -1),
                ).reshape(b, k, n_e, n_m)
                e_s, m_s = _grid_maxima(g2, e_grid, m_grid, n_starts)  # (B, K, S)
            else:
                e_s, m_s = e_c[..., None], m_c[..., None]
            n_s = e_s.shape[-1]

            def rep(t):  # (B, K, ...) -> (B, K * S, ...)
                return (
                    t[:, :, None].expand(b, k, n_s, *t.shape[2:]).reshape(b, k * n_s, *t.shape[2:])
                )

            e_s, m_s, steps0 = e_s.reshape(b, -1), m_s.reshape(b, -1), rep(start)
            if regrid:  # coarse grid starts get the same local refinement as the first stage
                e_s, m_s, d2e_s, d2m_s = _refine_e_m0(
                    batch, rep(1.0 / freq), e_s, m_s, de, dm, (1.0, 0.25), ecc_max
                )
                c_s = rep(corr)

                def width_s(d2, default):
                    w = torch.sqrt(2.0 / (-d2).clamp_min(1e-300)) * c_s
                    return torch.where(d2 < 0, w, torch.full_like(w, default))

                steps0 = torch.stack([rep(sig), width_s(d2e_s, 0.05), width_s(d2m_s, 0.3)], -1)
            jr = joint_refine(batch, rep(freq), e_s, m_s, steps0, ecc_max)
            pick = jr["gain"].reshape(b, k, n_s).argmax(-1)  # best start per anchor

            def best(t):
                t = t.reshape(b, k, n_s, *t.shape[2:])
                idx = pick.reshape(b, k, 1, *([1] * (t.dim() - 3)))
                return t.gather(2, idx.expand(b, k, 1, *t.shape[3:]))[:, :, 0]

            jr = {key: best(val) for key, val in jr.items()}
            freq, e_c, m_c = jr["freq"], jr["ecc"], jr["m0"]
            var = torch.diagonal(jr["cov_x"], dim1=-2, dim2=-1)
            d2e = -2.0 / var[..., 1].clamp_min(1e-300)  # so that sigma_e = sqrt(var_e) below
            d2m = -2.0 / var[..., 2].clamp_min(1e-300)
            extra = {"cov_x": jr["cov_x"], "pd": jr["pd"], "newton_dist": jr["newton_dist"]}
        else:
            e_c, m_c, d2e, d2m = _refine_e_m0(
                batch, 1.0 / freq, e_c, m_c, de, dm, (0.25, 0.0625), ecc_max
            )
            freq = _refine_frequency(batch, freq, e_c, m_c, 0.125 * sig)
        period = 1.0 / freq
    inf = torch.full_like(d2e, torch.inf)
    sig_e = torch.where(d2e < 0, torch.sqrt(2.0 / (-d2e).clamp_min(1e-300)), inf)
    sig_m = torch.where(d2m < 0, torch.sqrt(2.0 / (-d2m).clamp_min(1e-300)), inf)
    best = _gain(batch, period[..., None], e_c[..., None], m_c[..., None])[..., 0]
    return {
        "map": torch.log1p(gain),
        "gain_grid": gain,
        "gain_best": best,
        "ecc": e_c,
        "m0": m_c,
        "period": period,
        "sigma_e": sig_e,
        "sigma_m0": sig_m,
        **extra,
    }


@torch.no_grad()
def circular_noise_scale(batch: dict, pg: torch.Tensor, lo: float = 1e-4, hi: float = 1e8):
    """Preliminary log kappa_hat (B,) from the best *circular* orbit: (chi^2_5 - max pg)/(N - 9).

    Used only to set noise-invariant step sizes before the Keplerian refinement, so the clamp
    is loose (a circular fit to a bright eccentric orbit leaves a large residual).
    """
    _, _, _, w, r, _, _ = single_star_terms(batch)
    chi2_5 = (w * r * r).sum(-1)
    dof = (batch["mask"].sum(-1).to(torch.float64) - 9.0).clamp_min(1.0)
    resid = (chi2_5 - pg.amax(-1).to(torch.float64)).clamp_min(0.0)
    return 0.5 * torch.log((resid / dof).clamp(lo, hi))


@torch.no_grad()
def noise_scale_estimate(batch: dict, prof: dict, lo: float = 0.25, hi: float = 9.0):
    """log kappa_hat (B,), the noise scale relative to the reported sigma.

    kappa_hat^2 = (chi^2_5 - max_k Delta chi^2_k) / (N - 12): the residual chi^2 per degree of
    freedom after the best Keplerian fit among the anchors (5 + 4 linear + 3 nonlinear
    parameters), clamped to [lo, hi]. The orbit is removed first, so this is valid at low and
    high S/N alike (docs/derivations.md, section 7).
    """
    # chi^2_5 from the same float64 residuals and weights as Delta chi^2, so the difference
    # does not suffer from float32 rounding of a stored chi^2 at high S/N.
    _, _, _, w, r, _, _ = single_star_terms(batch)
    chi2_5 = (w * r * r).sum(-1)
    dof = (batch["mask"].sum(-1).to(torch.float64) - 12.0).clamp_min(1.0)
    resid = (chi2_5 - prof["gain_best"].amax(-1)).clamp_min(0.0)
    return 0.5 * torch.log((resid / dof).clamp(lo, hi))


def _grid_maxima(g, e_grid, m_grid, n: int):
    """(e, M0) of the ``n`` highest local maxima of each (n_e, n_m) grid (M0 circular)."""
    neigh = []
    for de_ in (-1, 0, 1):
        for dm_ in (-1, 0, 1):
            if de_ or dm_:
                sh = torch.roll(g, shifts=dm_, dims=-1)  # circular in M0
                if de_:
                    sh = torch.roll(sh, shifts=de_, dims=-2)
                    edge = 0 if de_ == 1 else -1  # no wrap in e: rolled-in row never wins
                    sh[..., edge, :] = -torch.inf
                neigh.append(sh)
    is_max = (g[None] >= torch.stack(neigh)).all(0)
    score = torch.where(is_max, g, torch.full_like(g, -torch.inf)).flatten(-2)
    top = score.topk(n, dim=-1)
    idx = torch.where(torch.isfinite(top.values), top.indices, top.indices[..., :1])
    n_m = len(m_grid)
    return e_grid[idx // n_m], m_grid[idx % n_m]


# ---- joint Newton refinement with the Laplace covariance (docs/derivations.md, section 5f) ---

_PAIRS = ((0, 1), (0, 2), (1, 2))


def _fd_offsets(dtype, device):
    """19 finite-difference offsets in units of h: centre, +-each axis, and the 4 corners of
    each coordinate pair (enough for the gradient and the full 3 x 3 Hessian)."""
    pts = [[0.0, 0.0, 0.0]]
    for i in range(3):
        for s in (1.0, -1.0):
            p = [0.0, 0.0, 0.0]
            p[i] = s
            pts.append(p)
    for i, j in _PAIRS:
        for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            p = [0.0, 0.0, 0.0]
            p[i], p[j] = float(si), float(sj)
            pts.append(p)
    return torch.tensor(pts, dtype=dtype, device=device)  # (19, 3)


def _gain_x(batch, x):
    """Delta chi^2 at points x = (f, e, M0) of shape (B, K, J, 3)."""
    return _gain(batch, 1.0 / x[..., 0], x[..., 1], x[..., 2])


def _grad_hess(y, h):
    """Gradient (B,K,3) and Hessian (B,K,3,3) from the 19 values y (B,K,19), steps h (B,K,3)."""
    g = torch.stack(
        [(y[..., 1 + 2 * i] - y[..., 2 + 2 * i]) / (2 * h[..., i]) for i in range(3)], -1
    )
    hess = torch.zeros(*y.shape[:-1], 3, 3, dtype=y.dtype, device=y.device)
    for i in range(3):
        hess[..., i, i] = (y[..., 1 + 2 * i] + y[..., 2 + 2 * i] - 2 * y[..., 0]) / h[..., i] ** 2
    for n, (i, j) in enumerate(_PAIRS):
        c = 7 + 4 * n
        hij = (y[..., c] - y[..., c + 1] - y[..., c + 2] + y[..., c + 3]) / (
            4 * h[..., i] * h[..., j]
        )
        hess[..., i, j] = hess[..., j, i] = hij
    return g, hess


def _ascent_step(g, hess):
    """Newton step for a maximum, made safe: eigenvalues >= 0 are replaced by -|lambda|max/100."""
    lam, vec = torch.linalg.eigh(hess)
    floor = -1e-2 * lam.abs().amax(-1, keepdim=True).clamp_min(1e-300)
    lam = torch.minimum(lam, floor)
    return -(vec @ ((vec.transpose(-1, -2) @ g[..., None]) / lam[..., None]))[..., 0]


@torch.no_grad()
def joint_refine(batch, freq, ecc, m0, sigma, ecc_max, n_iter: int = 4, trust: float = 4.0):
    """Joint Newton ascent of the Keplerian Delta chi^2 in x = (f, e, M0), and its Laplace
    covariance.

    Parameters
    ----------
    freq, ecc, m0 : (B, K) starting point (cycles/day, -, rad).
    sigma : (B, K, 3) rough widths used for finite-difference steps (h = sigma / 2) and the
        trust region (each step moves at most ``trust`` sigma per coordinate).

    Returns
    -------
    dict with refined ``freq``, ``ecc``, ``m0``; ``gain`` (Delta chi^2 there); ``cov_x``
    (B, K, 3, 3), the inverse of the Fisher matrix -1/2 d2(Delta chi^2)/dx2 (a Laplace
    approximation); ``pd`` (B, K) whether that Hessian was negative definite (otherwise
    ``cov_x`` is diagonal sigma^2); ``newton_dist`` (B, K), the size of the next Newton step in
    units of the final widths (small = converged).
    """
    f64 = torch.float64
    dev = freq.device
    lo = torch.tensor([1e-6, 0.0, -1e9], dtype=f64, device=dev)
    x = torch.stack([freq, ecc, m0], -1).to(f64)
    sig = sigma.to(f64)
    off = _fd_offsets(f64, dev)
    cap_e = min(0.95, ecc_max + 0.04)

    def steps(s):
        h = 0.5 * s
        return torch.stack([h[..., 0], h[..., 1].clamp(1e-4, 0.05), h[..., 2].clamp(1e-4, 0.3)], -1)

    def centre_ok(x, h):  # keep the stencil inside 0 < e < 1
        e = x[..., 1].clamp(h[..., 1] + 1e-3, cap_e - h[..., 1])
        return torch.stack([x[..., 0].clamp_min(2 * h[..., 0]), e, x[..., 2]], -1)

    for _ in range(n_iter):
        h = steps(sig)
        x = centre_ok(x, h)
        y = _gain_x(batch, x[..., None, :] + off * h[..., None, :])
        g, hess = _grad_hess(y, h)
        d = _ascent_step(g, hess)
        scale = (d.abs() / (trust * sig)).amax(-1, keepdim=True).clamp_min(1.0)
        d = d / scale
        cand = torch.stack([x + d, x + 0.25 * d], -2)  # (B, K, 2, 3), backtracking option
        cand = centre_ok(cand, h[..., None, :])
        yc = _gain_x(batch, cand)
        best = torch.cat([y[..., :1], yc], -1).argmax(-1)  # 0 = stay, 1 = full, 2 = quarter
        options = torch.cat([x[..., None, :], cand], -2)
        x = options.gather(-2, best[..., None, None].expand(*best.shape, 1, 3))[..., 0, :]
        # update the widths from the local curvature where it is usable
        diag = torch.diagonal(hess, dim1=-2, dim2=-1)
        new_sig = torch.where(diag < 0, torch.sqrt(2.0 / (-diag).clamp_min(1e-300)), sig)
        sig = torch.minimum(torch.maximum(new_sig, 0.05 * sig), 20.0 * sig)
    h = steps(sig)
    x = centre_ok(x, h)
    y = _gain_x(batch, x[..., None, :] + off * h[..., None, :])
    g, hess = _grad_hess(y, h)
    lam = torch.linalg.eigvalsh(hess)
    pd = (lam < 0).all(-1)
    eye = torch.eye(3, dtype=f64, device=dev)
    safe = torch.where(pd[..., None, None], hess, -eye)
    cov = torch.where(pd[..., None, None], -2.0 * torch.linalg.inv(safe), torch.diag_embed(sig**2))
    sd = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-300))
    dist = (_ascent_step(g, hess).abs() / sd).amax(-1)
    x = torch.maximum(x, lo)
    return {
        "freq": x[..., 0],
        "ecc": x[..., 1],
        "m0": torch.remainder(x[..., 2], TWO_PI),
        "gain": y[..., 0],
        "cov_x": cov,
        "pd": pd,
        "newton_dist": dist,
    }


@torch.no_grad()
def laplace_to_target(prof: dict, period_range_d, ecc_max: float, log_kappa=None) -> dict:
    """Laplace covariance in the network's coordinates (u_P, u_e, M0), widened by kappa_hat.

    u_P = probit((log P - log P_min) / R) with P = 1/f, so du_P/df = -1 / (f R phi(u_P));
    u_e = probit(e / e_max), so du_e/de = 1 / (e_max phi(u_e)). With the Jacobian
    J = diag(du_P/df, du_e/de, 1): cov_u = J cov_x J, times kappa_hat^2 (widths scale with
    the noise). Returns the centre ``mu`` (B, K, 3) and ``cov`` (B, K, 3, 3).
    """
    f = 1.0 / prof["period"]
    lo, hi = math.log(period_range_d[0]), math.log(period_range_d[1])
    frac = ((-torch.log(f) - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
    u_p = torch.special.ndtri(frac)
    u_e = torch.special.ndtri((prof["ecc"] / ecc_max).clamp(1e-6, 1 - 1e-6))

    def phi(u):
        return torch.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)

    jac = torch.stack(
        [-1.0 / (f * (hi - lo) * phi(u_p)), 1.0 / (ecc_max * phi(u_e)), torch.ones_like(f)], -1
    )
    cov = prof["cov_x"] * jac[..., :, None] * jac[..., None, :]
    if log_kappa is not None:
        cov = cov * torch.exp(2.0 * log_kappa.to(cov))[:, None, None, None]
    return {"mu": torch.stack([u_p, u_e, prof["m0"]], -1), "cov": cov}
