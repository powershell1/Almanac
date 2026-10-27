"""Reference posteriors by importance sampling with the analytic marginal likelihood.

Parameters are handled in the unconstrained space of ``Prior`` with the amplitude
hyperparameter included: v = (u_P, u_e, M0, log s_a). If the training noise model has an
unknown per-star noise scale kappa (``scale_prior``, docs/derivations.md section 7), pass its
range as ``noise_scale_range`` and v gains a fifth coordinate, log kappa, with the same
LogUniform prior; the likelihood then uses kappa * sigma. The target is

    p(v | w) ∝ p(v) p(w | v),

with p(w | v) from ``model.linear`` (theta_lin integrated out exactly, for every kappa).

Stage 1 (exact brute force): draw v from the prior; weight = likelihood.
Refinement (optional): Nelder-Mead from the best distinct stage-1 samples to locate the
modes (period aliases), then R rounds of importance sampling from a *defensive* mixture
``q = alpha * prior + (1 - alpha) * kernel mixture`` built around the current samples
(Hesterberg 1995). Each round is an unbiased importance sampler on its own; the defensive
prior component bounds the weights by ``p L / (alpha p) = L / alpha``. Only the last round is
returned. Always check ``ess``: when it is small the reference is not trustworthy (flag it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from gdr4.model.astrometry import design_matrix, t_peri_from_mean_anomaly
from gdr4.model.kepler import TWO_PI
from gdr4.model.linear import gaussian_linear_posterior
from gdr4.sim.priors import Prior

# Per-coordinate constants for (u_P, u_e, M0, log s_a[, log kappa]).
_KERNEL_FLOOR = np.array([1e-6, 1e-4, 1e-5, 1e-4, 1e-4])
_KERNEL_CAP = np.array([1.0, 2.0, 1.5, 2.0, 1.0])
_FD_STEP = np.array([1e-5, 1e-4, 1e-4, 1e-3, 1e-3])
# Used when the curvature is not usable in a dimension, e.g. an optimum on the boundary of the
# amplitude or noise-scale prior, where the finite difference leaves the support.
_FALLBACK_WIDTH = np.array([1e-3, 0.3, 0.3, 0.5, 0.1])
_NM_STEP = np.array([2e-3, 0.05, 0.05, 0.2, 0.05])
_CIRCULAR = 2  # M0


class _Target:
    """Prior (and optional noise-scale prior) defining the reference parameter space."""

    def __init__(self, prior: Prior, noise_scale_range: tuple[float, float] | None = None):
        self.prior = prior
        self.krange = None
        if noise_scale_range is not None:
            self.krange = (float(np.log(noise_scale_range[0])), float(np.log(noise_scale_range[1])))
        self.dim = 5 if self.krange else 4

    def log_prior(self, v: np.ndarray) -> np.ndarray:
        lp = self.prior.log_prob_unconstrained(v[..., :4])
        if self.krange:
            lo, hi = self.krange
            inside = (v[..., 4] >= lo) & (v[..., 4] <= hi)
            lp = np.where(inside, lp - np.log(hi - lo), -np.inf)
        return lp

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        nl = self.prior.sample_nonlinear(rng, n)
        v = self.prior.to_unconstrained(
            nl["period_d"], nl["ecc"], nl["mean_anom0_rad"], nl["amp_scale_mas"]
        )
        if self.krange:
            v = np.concatenate([v, rng.uniform(*self.krange, (n, 1))], axis=1)
        return v


@dataclass
class ReferenceResult:
    v: np.ndarray  # (S, 4 or 5) samples (u_P, u_e, M0, log s_a[, log kappa])
    log_w: np.ndarray  # (S,) normalised log weights
    ess: float
    ess_stage1: float
    log_evidence: float  # log p(w), estimated from the returned stage
    n_likelihood_evals: int
    seconds: float

    def resample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        idx = rng.choice(len(self.log_w), size=n, replace=True, p=np.exp(self.log_w))
        return self.v[idx]


def log_likelihood(
    star: dict,
    prior: Prior,
    v: np.ndarray,
    chunk: int = 20000,
    noise_scale_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """log p(w | theta_nl, s_a[, kappa]) for each row of v; -inf outside the prior support."""
    return _log_likelihood(star, _Target(prior, noise_scale_range), v, chunk)


def _log_likelihood(star: dict, target: _Target, v: np.ndarray, chunk: int) -> np.ndarray:
    prior = target.prior
    out = np.full(len(v), -np.inf)
    idx = np.flatnonzero(np.isfinite(target.log_prior(v)))
    t, psi, plx = star["t_d"], star["psi_rad"], star["plx_factor"]
    w, sigma = star["w_mas"], star["sigma_mas"]
    for s in range(0, len(idx), chunk):
        sub = idx[s : s + chunk]
        par = prior.from_unconstrained(v[sub, :4])
        tp = t_peri_from_mean_anomaly(par["mean_anom0_rad"], par["period_d"])
        m = design_matrix(t, psi, plx, par["period_d"][:, None], par["ecc"][:, None], tp[:, None])
        mu0, sd0 = prior.linear_prior(par["amp_scale_mas"])
        sig = sigma * np.exp(v[sub, 4])[:, None] if target.dim == 5 else sigma
        out[sub] = gaussian_linear_posterior(m, w, sig, mu0, sd0, return_cov=False).log_marginal
    return out


def sample_prior_unconstrained(
    prior: Prior,
    rng: np.random.Generator,
    n: int,
    noise_scale_range: tuple[float, float] | None = None,
) -> np.ndarray:
    return _Target(prior, noise_scale_range).sample(rng, n)


def _ess(log_w_norm: np.ndarray) -> float:
    return float(np.exp(-logsumexp(2 * log_w_norm)))


def _wrap(d):
    return np.angle(np.exp(1j * d))


def _laplace_width(star, target, center, chunk):
    """Per-dimension 1/sqrt(-d2 logL / dv2) at ``center`` by central finite differences."""
    dim = target.dim
    h = _FD_STEP[:dim]
    pts = [center]
    for d in range(dim):
        for sgn in (+1, -1):
            p = center.copy()
            p[d] += sgn * h[d]
            pts.append(p)
    ll = _log_likelihood(star, target, np.array(pts), chunk)
    f0, rest = ll[0], ll[1:].reshape(dim, 2)
    curv = (rest[:, 0] + rest[:, 1] - 2 * f0) / h**2
    width = np.where(
        np.isfinite(curv) & (curv < 0), 1.0 / np.sqrt(-np.minimum(curv, -1e-300)), np.nan
    )
    return width


class _KernelMixture:
    """Weighted mixture of diagonal Gaussians (wrapped Gaussian in M0)."""

    def __init__(self, centers: np.ndarray, widths: np.ndarray, log_weights=None):
        self.c, self.h = centers, widths
        lw = np.zeros(len(centers)) if log_weights is None else np.asarray(log_weights, float)
        self.log_a = lw - logsumexp(lw)

    def sample(self, rng, n):
        j = rng.choice(len(self.c), size=n, p=np.exp(self.log_a))
        x = self.c[j] + self.h[j] * rng.standard_normal((n, self.c.shape[1]))
        x[:, _CIRCULAR] = np.mod(x[:, _CIRCULAR], TWO_PI)
        return x

    def log_prob(self, x, chunk=4096):
        out = np.empty(len(x))
        dim = self.c.shape[1]
        lin = [d for d in range(dim) if d != _CIRCULAR]
        log_norm = -0.5 * np.log(2 * np.pi) * dim - np.sum(np.log(self.h), axis=1) + self.log_a
        for s in range(0, len(x), chunk):
            d = x[s : s + chunk, None, :] - self.c[None]  # (n, J, dim)
            z2 = np.sum((d[..., lin] / self.h[None][..., lin]) ** 2, axis=-1)
            # M0 is sampled mod 2 pi, so its density sums over periodic images. With widths
            # capped at 1.5 rad, images beyond +-1 contribute < e^-20 and are dropped.
            dm = _wrap(d[..., 2])
            zm = np.stack([((dm + k * TWO_PI) / self.h[None, :, 2]) ** 2 for k in (-1, 0, 1)])
            log_k = logsumexp(log_norm[None, None] - 0.5 * (z2[None] + zm), axis=0)
            out[s : s + chunk] = logsumexp(log_k, axis=1)
        return out


def _neg_log_post(x, star, target, chunk):
    x = x.copy()
    x[_CIRCULAR] = np.mod(x[_CIRCULAR], TWO_PI)
    lp = target.log_prior(x[None])[0]
    if not np.isfinite(lp):
        return 1e300
    return -(lp + _log_likelihood(star, target, x[None], chunk)[0])


def find_modes(star, target, v, log_w, chunk, n_starts=16, min_sep_u=0.01):
    """Nelder-Mead from the best distinct stage-1 samples; returns optima, widths, log weights.

    Weights are Laplace evidence estimates, log p(v*) + log L(v*) + sum log(width) (+ const),
    so the proposal does not waste samples on poor local maxima.
    """
    dim = target.dim
    order = np.argsort(log_w)[::-1][:2000]
    starts = []
    for i in order:
        if all(abs(v[i, 0] - v[j, 0]) > min_sep_u for j in starts):
            starts.append(i)
        if len(starts) == n_starts:
            break
    steps = _NM_STEP[:dim]
    optima, widths, log_ev = [], [], []
    for i in starts:
        x0 = v[i]
        simplex = np.vstack([x0] + [x0 + np.eye(dim)[d] * steps[d] for d in range(dim)])
        res = minimize(
            _neg_log_post,
            x0,
            args=(star, target, chunk),
            method="Nelder-Mead",
            options={"initial_simplex": simplex, "xatol": 1e-8, "fatol": 1e-4, "maxfev": 3000},
        )
        x = res.x.copy()
        x[_CIRCULAR] = np.mod(x[_CIRCULAR], TWO_PI)
        lap = _laplace_width(star, target, x, chunk)
        if not np.isfinite(res.fun) or res.fun >= 1e299:
            continue
        lap = np.where(np.isfinite(lap), lap, _FALLBACK_WIDTH[:dim])
        optima.append(x)
        widths.append(lap)
        log_ev.append(-res.fun + np.sum(np.log(lap)))
    if not optima:
        return None
    return np.array(optima), np.array(widths), np.array(log_ev)


def _build_kernel(star, target, v, log_w, rng, n_centers, chunk, modes=None, window_u=0.05):
    dim = target.dim
    floor, cap = _KERNEL_FLOOR[:dim], _KERNEL_CAP[:dim]
    pool = v[rng.choice(len(v), size=20000, replace=True, p=np.exp(log_w))]
    uniq = np.unique(pool, axis=0)
    centers = uniq[rng.permutation(len(uniq))[:n_centers]]
    widths = np.empty_like(centers)
    for j, c in enumerate(centers):
        near = pool[np.abs(pool[:, 0] - c[0]) < window_u]
        d = near - c
        d[:, _CIRCULAR] = _wrap(d[:, _CIRCULAR])
        local = np.sqrt(np.mean(d**2, axis=0)) if len(near) > 1 else np.zeros(dim)
        lap = _laplace_width(star, target, c, chunk) if j < 32 else np.full(dim, np.nan)
        widths[j] = np.fmax(2.0 * local, 1.5 * np.nan_to_num(lap, nan=0.0))
    # centres without a Laplace estimate borrow the median of those that have one
    widths = np.clip(np.where(widths > 0, widths, np.median(widths[:32], axis=0)), floor, cap)
    log_a = np.zeros(len(centers))
    if modes is not None:  # half the kernel mass on the optimised modes
        opt, opt_w, log_ev = modes
        log_ev = np.maximum(log_ev - logsumexp(log_ev), np.log(1e-3))
        log_ev -= logsumexp(log_ev)
        centers = np.vstack([centers, opt])
        widths = np.vstack([widths, np.clip(2.0 * opt_w, floor, cap)])
        log_a = np.concatenate([log_a - logsumexp(log_a), log_ev]) + np.log(0.5)
    return _KernelMixture(centers, widths, log_a)


def reference_posterior(
    star: dict,
    prior: Prior,
    rng: np.random.Generator,
    n_prior_samples: int = 1_000_000,
    refine: bool = True,
    n_refine_samples: int = 1_000_000,
    refine_rounds: int = 3,
    chunk: int = 20000,
    alpha: float = 0.1,
    n_centers: int = 256,
    noise_scale_range: tuple[float, float] | None = None,
) -> ReferenceResult:
    """Reference posterior for one star (dict of unpadded float64 arrays, see star_view).

    ``noise_scale_range`` must be the training noise model's ``sigma_scale_range`` (or None for
    known error bars), so that the reference marginalises exactly the same noise model.
    """
    t0 = time.time()
    target = _Target(prior, noise_scale_range)
    v = target.sample(rng, n_prior_samples)
    ll = _log_likelihood(star, target, v, chunk)
    log_z = logsumexp(ll) - np.log(len(ll))
    log_w = ll - logsumexp(ll)
    ess1 = _ess(log_w)
    n_eval = len(v)
    if refine:
        modes = find_modes(star, target, v, log_w, chunk)
        for r in range(refine_rounds):
            kern = _build_kernel(
                star, target, v, log_w, rng, n_centers, chunk, modes=modes if r == 0 else None
            )
            n_def = rng.binomial(n_refine_samples, alpha)
            v = np.concatenate(
                [target.sample(rng, n_def), kern.sample(rng, n_refine_samples - n_def)]
            )
            lp = target.log_prior(v)
            log_q = np.logaddexp(np.log(alpha) + lp, np.log1p(-alpha) + kern.log_prob(v))
            ll = _log_likelihood(star, target, v, chunk)
            log_r = np.where(np.isfinite(lp), lp + ll - log_q, -np.inf)
            log_z = logsumexp(log_r) - np.log(len(log_r))
            log_w = log_r - logsumexp(log_r)
            n_eval += len(v) + (2 * target.dim + 1) * min(32, n_centers)
    return ReferenceResult(
        v=v,
        log_w=log_w,
        ess=_ess(log_w),
        ess_stage1=ess1,
        log_evidence=float(log_z),
        n_likelihood_evals=n_eval,
        seconds=time.time() - t0,
    )
