"""Encoder over transits + conditional posterior over (u_P, u_e, M0).

Batch convention (torch tensors, padded to the longest star in the batch):
``t_d, psi_rad, plx_factor, sigma_mas, resid5_mas`` (B, N) float32, ``mask`` (B, N) bool,
``n_obs`` (B,), ``chi2_5p`` (B,). Targets ``v`` (B, 3) = (u_P, u_e, M0) come from
``Prior.to_unconstrained`` (see docs/derivations.md, section 5).
"""

from __future__ import annotations

import math

import torch
import zuko
from torch import nn

from gdr4.nn.periodogram import (
    N_PEAK_FEATURES,
    analyse_peaks,
    anchors_from_peaks,
    circular_periodogram,
    features_from_peaks,
    frequency_grid,
)

FEATURE_VARIANTS = ("raw", "normalized")
_R0_MAS = 0.1  # fixed scale for raw residuals and sigmas (typical per-transit sigma ~0.05 mas)


def token_features(batch: dict, variant: str) -> torch.Tensor:
    """Per-transit features (B, N, F) excluding the Fourier time embedding."""
    t, psi = batch["t_d"], batch["psi_rad"]
    cols = [t / 1000.0, torch.sin(psi), torch.cos(psi), batch["plx_factor"]]
    sigma = batch["sigma_mas"].clamp_min(1e-6)
    if variant == "raw":  # (a) raw residual and sigma
        cols += [batch["resid5_mas"] / _R0_MAS, batch["sigma_mas"] / _R0_MAS]
    elif variant == "normalized":  # (b) sigma-normalised residual only
        cols += [batch["resid5_mas"] / sigma]
    else:
        raise ValueError(f"features must be one of {FEATURE_VARIANTS}")
    x = torch.stack(cols, dim=-1)
    return x * batch["mask"][..., None]


def global_features(batch: dict, variant: str) -> torch.Tensor:
    n = batch["n_obs"].float()
    dof = (n - 5).clamp_min(1)
    cols = [torch.log(n) - 4.5, torch.log((batch["chi2_5p"] / dof).clamp_min(1e-3))]
    if variant == "raw":
        m = batch["mask"]
        mean_sigma = (batch["sigma_mas"] * m).sum(1) / m.sum(1).clamp_min(1)
        cols.append(torch.log(mean_sigma.clamp_min(1e-6) / _R0_MAS))
    return torch.stack(cols, dim=-1)


def n_token_features(variant: str) -> int:
    return 6 if variant == "raw" else 5


def n_global_features(variant: str) -> int:
    return 3 if variant == "raw" else 2


class FourierTime(nn.Module):
    """sin/cos(2 pi f t) with learnable log-spaced frequencies f (cycles/day).

    Always evaluated in float32: at t ~ 1000 d the phase is ~600 rad, which fp16 cannot hold.
    """

    def __init__(self, n_freq: int, f_min: float = 1 / 4000, f_max: float = 1 / 10):
        super().__init__()
        self.log_f = nn.Parameter(torch.linspace(math.log(f_min), math.log(f_max), n_freq))

    def forward(self, t_d: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=t_d.device.type, enabled=False):
            phase = 2 * math.pi * t_d.float()[..., None] * torch.exp(self.log_f)
            return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class PeriodogramEncoder(nn.Module):
    """1-D CNN over log(1 + Delta chi^2) on a fixed frequency grid (see nn/periodogram.py).

    With ``n_peaks > 0`` it also embeds the sub-grid positions and heights of the highest
    peaks, so the network does not have to locate them from the coarse curve itself.
    """

    def __init__(
        self,
        k: int,
        period_range_d: tuple[float, float],
        out_dim: int,
        ch: int = 32,
        n_peaks: int = 0,
    ):
        super().__init__()
        self.period_range_d = tuple(period_range_d)
        self.n_peaks = n_peaks
        self.out_dim = out_dim * (2 if n_peaks else 1)
        self.register_buffer("freqs", frequency_grid(*period_range_d, k), persistent=False)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, ch, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(ch, ch, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(ch, ch, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(ch, ch, 7, stride=2, padding=3),
            nn.GELU(),
        )
        self.out = nn.Linear(ch * (k // 8), out_dim)
        if n_peaks:
            self.peaks = nn.Sequential(
                nn.Linear(n_peaks * N_PEAK_FEATURES, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict | None]:
        """Embedding (B, out_dim) and the refined peak analysis (None without peaks)."""
        with torch.autocast(device_type=batch["t_d"].device.type, enabled=False):
            pg = circular_periodogram(batch, self.freqs)
            emb = self.out(self.cnn(torch.log1p(pg).float()[:, None]).flatten(1))
            if not self.n_peaks:
                return emb, None
            peaks = analyse_peaks(batch, self.freqs, pg, self.n_peaks)
            pk = features_from_peaks(peaks, self.period_range_d)
            return torch.cat([emb, self.peaks(pk.flatten(1))], dim=-1), peaks


class TransitEncoder(nn.Module):
    def __init__(
        self,
        variant: str,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_freq: int,
        ctx_dim: int,
        pgram_k: int = 0,
        pgram_period_d: tuple[float, float] = (20.0, 3000.0),
        pgram_peaks: int = 0,
    ):
        super().__init__()
        self.variant = variant
        self.pgram = (
            PeriodogramEncoder(pgram_k, pgram_period_d, d_model, n_peaks=pgram_peaks)
            if pgram_k
            else None
        )
        n_pgram = self.pgram.out_dim if self.pgram is not None else 0
        self.time = FourierTime(n_freq)
        n_in = n_token_features(variant) + 2 * n_freq
        self.embed = nn.Sequential(nn.Linear(n_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        self.pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model + n_pgram + n_global_features(variant), ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, ctx_dim),
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict | None]:
        """Context vector (B, ctx_dim) and the periodogram peak analysis (or None)."""
        mask = batch["mask"]
        x = torch.cat([token_features(batch, self.variant), self.time(batch["t_d"])], dim=-1)
        h = self.embed(x)
        h = self.norm(self.transformer(h, src_key_padding_mask=~mask))
        q = self.query.expand(h.shape[0], -1, -1)
        attn, _ = self.pool(q, h, h, key_padding_mask=~mask, need_weights=False)
        mf = mask[..., None].to(h.dtype)
        mean = (h * mf).sum(1) / mf.sum(1).clamp_min(1)
        g = global_features(batch, self.variant).to(h.dtype)
        parts = [attn[:, 0], mean, g]
        peaks = None
        if self.pgram is not None:
            emb, peaks = self.pgram(batch)
            parts.append(emb.to(h.dtype))
        return self.head(torch.cat(parts, dim=-1)), peaks


class VonMisesMixture(nn.Module):
    """q(M0 | u_P, u_e, ctx) as a K-component von Mises mixture (exactly periodic)."""

    LOG_KAPPA_RANGE = (-4.0, 14.0)  # kappa from ~0.02 (flat) to ~1.2e6 (width ~1e-3 rad)

    def __init__(self, ctx_dim: int, hidden: int, k: int, cond_dim: int = 2):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4 * k),
        )

    def params(self, u2: torch.Tensor, ctx: torch.Tensor):
        out = self.net(torch.cat([ctx, u2], dim=-1))
        logits, cx, cy, log_kappa = out.split(self.k, dim=-1)
        mu = torch.atan2(cy, cx)
        log_kappa = log_kappa.clamp(*self.LOG_KAPPA_RANGE)
        return torch.log_softmax(logits, dim=-1), mu, log_kappa

    def log_prob(self, m0: torch.Tensor, u2: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        log_w, mu, log_kappa = self.params(u2, ctx)
        kappa = torch.exp(log_kappa)
        half = 0.5 * (m0[..., None] - mu)
        # kappa (cos d - 1) = -2 kappa sin^2(d/2): accurate in float32 even for huge kappa.
        lp = (
            -2.0 * kappa * torch.sin(half) ** 2
            - math.log(2 * math.pi)
            - torch.log(torch.special.i0e(kappa))
        )
        return torch.logsumexp(log_w + lp, dim=-1)

    @torch.no_grad()
    def sample(self, u2: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        log_w, mu, log_kappa = self.params(u2, ctx)
        comp = torch.distributions.Categorical(logits=log_w).sample()
        mu_c = mu.gather(-1, comp[..., None])[..., 0].double()
        kappa_c = torch.exp(log_kappa.gather(-1, comp[..., None])[..., 0]).double()
        m0 = torch.distributions.VonMises(mu_c, kappa_c).sample()
        return torch.remainder(m0, 2 * math.pi).to(ctx.dtype)


class AnchoredPeriodHead(nn.Module):
    """q(u_P | x): mixture of one global flow and K components anchored on periodogram peaks.

    Component k (one per anchor, see ``anchors_from_peaks``) is
        u_P = a_k + tau_k (d_k + exp(s_k) z),   z ~ g(z | ctx, k),
    where a_k and tau_k (peak position and circular-orbit Fisher width) come from the data, and
    d_k, s_k and the 1-D spline flow g are learned. The network therefore only has to output
    O(1) numbers even when the posterior is 1e-4 wide. The global component is a 1-D spline
    flow on u_P itself, for broad (low-S/N) posteriors. Mixture weights are learned; invalid
    anchors get zero weight. Every component is a normalised density on R, so the mixture is
    too. See docs/derivations.md, section 5c.
    """

    def __init__(
        self, ctx_dim: int, n_anchor: int, hidden: tuple[int, ...], transforms: int, bins: int
    ):
        super().__init__()
        self.k = n_anchor
        h = list(hidden)
        self.global_flow = zuko.flows.NSF(
            features=1, context=ctx_dim, transforms=transforms, bins=bins, hidden_features=h
        )
        self.local_flow = zuko.flows.NSF(
            features=1,
            context=ctx_dim + n_anchor,
            transforms=transforms,
            bins=bins,
            hidden_features=h,
        )
        self.par = nn.Sequential(
            nn.Linear(ctx_dim, h[0]), nn.GELU(), nn.Linear(h[0], 3 * n_anchor + 1)
        )
        nn.init.zeros_(self.par[-1].weight)
        nn.init.zeros_(self.par[-1].bias)

    def _params(self, ctx, anc):
        out = self.par(ctx)
        logits = out[:, : self.k + 1]
        shift = out[:, self.k + 1 : 2 * self.k + 1].clamp(-50.0, 50.0)
        log_s = out[:, 2 * self.k + 1 :].clamp(-5.0, 5.0)
        neg = torch.full_like(logits[:, 1:], -torch.inf)
        logits = torch.cat([logits[:, :1], torch.where(anc["valid"], logits[:, 1:], neg)], 1)
        return torch.log_softmax(logits, dim=-1), shift, log_s

    def _local_ctx(self, ctx):
        b = ctx.shape[0]
        onehot = torch.eye(self.k, device=ctx.device, dtype=ctx.dtype)[:, None, :].expand(
            self.k, b, self.k
        )
        return torch.cat([ctx[None].expand(self.k, b, ctx.shape[-1]), onehot], dim=-1)

    def log_prob(self, u: torch.Tensor, ctx: torch.Tensor, anc: dict) -> torch.Tensor:
        """log q(u_P) for u of shape (..., B)."""
        log_w, shift, log_s = self._params(ctx, anc)
        lg = self.global_flow(ctx).log_prob(u[..., None])  # (..., B)
        a, lt = anc["a"].T, anc["log_tau"].T  # (K, B)
        scale = torch.exp(lt + log_s.T)
        z = (u[..., None, :] - a - torch.exp(lt) * shift.T) / scale  # (..., K, B)
        ll = self.local_flow(self._local_ctx(ctx)).log_prob(z[..., None]) - lt - log_s.T
        comp = torch.cat([lg[..., None, :], ll], dim=-2)  # (..., K+1, B)
        return torch.logsumexp(log_w.T + comp, dim=-2)

    @torch.no_grad()
    def sample(self, n: int, ctx: torch.Tensor, anc: dict) -> torch.Tensor:
        """n samples of u_P per star, shape (n, B)."""
        log_w, shift, log_s = self._params(ctx, anc)
        comp = torch.distributions.Categorical(logits=log_w).sample((n,))  # (n, B)
        g = self.global_flow(ctx).sample((n,))[..., 0]  # (n, B)
        z = self.local_flow(self._local_ctx(ctx)).sample((n,))[..., 0]  # (n, K, B)
        tau = torch.exp(anc["log_tau"].T)
        loc = anc["a"].T + tau * (shift.T + torch.exp(log_s.T) * z)
        everything = torch.cat([g[:, None], loc], dim=1)  # (n, K+1, B)
        return everything.gather(1, comp[:, None]).squeeze(1)

    def features(self, u: torch.Tensor, anc: dict) -> torch.Tensor:
        """Conditioning features for u_e and M0: u_P and its offset from each anchor in units
        of that anchor's width (clipped), so downstream heads can resolve tiny changes of u_P."""
        z = (u[..., None] - anc["a"]) * torch.exp(-anc["log_tau"])
        z = torch.where(anc["valid"], z.clamp(-10.0, 10.0) / 10.0, torch.zeros_like(z))
        return torch.cat([u[..., None] / 3.0, z], dim=-1)  # (..., B, 1 + K)


class NPE(nn.Module):
    """Neural posterior estimator q(u_P, u_e, M0 | transits)."""

    def __init__(
        self,
        features: str = "raw",
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_freq: int = 32,
        ctx_dim: int = 256,
        flow_transforms: int = 6,
        flow_hidden: tuple[int, ...] = (256, 256),
        flow_bins: int = 16,
        n_vm: int = 8,
        pgram_k: int = 0,
        pgram_period_d: tuple[float, float] = (20.0, 3000.0),
        pgram_peaks: int = 0,
        head: str = "nsf",
        n_anchor: int = 3,
        target_period_d: tuple[float, float] = (30.0, 2000.0),
    ):
        super().__init__()
        if pgram_peaks and not pgram_k:
            raise ValueError("pgram_peaks needs the periodogram (pgram_k > 0)")
        if head not in ("nsf", "anchored"):
            raise ValueError("head must be 'nsf' or 'anchored'")
        if head == "anchored" and pgram_peaks < n_anchor:
            raise ValueError("head='anchored' needs pgram_peaks >= n_anchor")
        self.head = head
        self.n_anchor = n_anchor
        self.target_period_d = tuple(target_period_d)
        self.hparams = dict(
            features=features,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_freq=n_freq,
            ctx_dim=ctx_dim,
            flow_transforms=flow_transforms,
            flow_hidden=list(flow_hidden),
            flow_bins=flow_bins,
            n_vm=n_vm,
            pgram_k=pgram_k,
            pgram_period_d=list(pgram_period_d),
            pgram_peaks=pgram_peaks,
            head=head,
            n_anchor=n_anchor,
            target_period_d=list(target_period_d),
        )
        self.encoder = TransitEncoder(
            features,
            d_model,
            n_layers,
            n_heads,
            n_freq,
            ctx_dim,
            pgram_k,
            tuple(pgram_period_d),
            pgram_peaks,
        )
        if head == "nsf":  # joint 2-D spline flow over (u_P, u_e)
            self.flow = zuko.flows.NSF(
                features=2,
                context=ctx_dim,
                transforms=flow_transforms,
                bins=flow_bins,
                hidden_features=list(flow_hidden),
            )
            self.vm = VonMisesMixture(ctx_dim, flow_hidden[0], n_vm)
        else:  # u_P from the anchored mixture, then u_e | u_P, then M0 | u_P, u_e
            self.period = AnchoredPeriodHead(
                ctx_dim, n_anchor, tuple(flow_hidden), flow_transforms, flow_bins
            )
            n_feat = 1 + n_anchor
            self.ecc_flow = zuko.flows.NSF(
                features=1,
                context=ctx_dim + n_feat,
                transforms=flow_transforms,
                bins=flow_bins,
                hidden_features=list(flow_hidden),
            )
            self.vm = VonMisesMixture(ctx_dim, flow_hidden[0], n_vm, cond_dim=n_feat + 1)

    @classmethod
    def from_config(cls, cfg_model: dict) -> NPE:
        tuples = ("flow_hidden", "pgram_period_d", "target_period_d")
        return cls(**{k: (tuple(v) if k in tuples else v) for k, v in cfg_model.items()})

    def context(self, batch: dict, amp: bool = False) -> tuple[torch.Tensor, dict | None]:
        """Context vector (B, C) and, for the anchored head, the period anchors."""
        dev = batch["t_d"].device.type
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=amp and dev == "cuda"):
            ctx, peaks = self.encoder(batch)
        anchors = None
        if self.head == "anchored":
            anchors = anchors_from_peaks(peaks, self.n_anchor, self.target_period_d)
        return ctx.float(), anchors

    def log_prob_from_context(self, v: torch.Tensor, context) -> torch.Tensor:
        """log q(v | x) for v (..., B, 3); ``context`` from :meth:`context`."""
        ctx, anc = context
        ctx_b = ctx.expand(*v.shape[:-1], ctx.shape[-1])
        if self.head == "nsf":
            u2, m0 = v[..., :2], v[..., 2]
            return self.flow(ctx).log_prob(u2) + self.vm.log_prob(m0, u2, ctx_b)
        u_p, u_e, m0 = v[..., 0], v[..., 1], v[..., 2]
        feat = self.period.features(u_p, anc)
        lp = self.period.log_prob(u_p, ctx, anc)
        lp = lp + self.ecc_flow(torch.cat([ctx_b, feat], -1)).log_prob(u_e[..., None])
        return lp + self.vm.log_prob(m0, torch.cat([feat, u_e[..., None]], -1), ctx_b)

    def log_prob(self, batch: dict, v: torch.Tensor, amp: bool = False) -> torch.Tensor:
        return self.log_prob_from_context(v, self.context(batch, amp))

    def forward(self, batch: dict, v: torch.Tensor, amp: bool = False) -> torch.Tensor:
        """Training loss: mean negative log posterior density of the true parameters."""
        return -self.log_prob(batch, v, amp).mean()

    @torch.no_grad()
    def sample(self, batch: dict, n: int, with_log_prob: bool = False):
        """Draw n posterior samples per star: v of shape (n, B, 3) (and log q if requested)."""
        context = self.context(batch)
        ctx, anc = context
        ctx_b = ctx.expand(n, *ctx.shape)
        if self.head == "nsf":
            u2 = self.flow(ctx).sample((n,))
            m0 = self.vm.sample(u2, ctx_b)
            v = torch.cat([u2, m0[..., None]], dim=-1)
        else:
            u_p = self.period.sample(n, ctx, anc)
            feat = self.period.features(u_p, anc)
            u_e = self.ecc_flow(torch.cat([ctx_b, feat], -1)).sample()[..., 0]
            m0 = self.vm.sample(torch.cat([feat, u_e[..., None]], -1), ctx_b)
            v = torch.stack([u_p, u_e, m0], dim=-1)
        if with_log_prob:
            return v, self.log_prob_from_context(v, context)
        return v
