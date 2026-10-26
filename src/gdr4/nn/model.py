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
    circular_periodogram,
    frequency_grid,
    peak_features,
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

    def forward(self, batch: dict) -> torch.Tensor:
        with torch.autocast(device_type=batch["t_d"].device.type, enabled=False):
            pg = circular_periodogram(batch, self.freqs)
            emb = self.out(self.cnn(torch.log1p(pg).float()[:, None]).flatten(1))
            if not self.n_peaks:
                return emb
            pk = peak_features(batch, self.freqs, pg, self.n_peaks, self.period_range_d)
            return torch.cat([emb, self.peaks(pk.flatten(1))], dim=-1)


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

    def forward(self, batch: dict) -> torch.Tensor:
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
        if self.pgram is not None:
            parts.append(self.pgram(batch).to(h.dtype))
        return self.head(torch.cat(parts, dim=-1))


class VonMisesMixture(nn.Module):
    """q(M0 | u_P, u_e, ctx) as a K-component von Mises mixture (exactly periodic)."""

    LOG_KAPPA_RANGE = (-4.0, 14.0)  # kappa from ~0.02 (flat) to ~1.2e6 (width ~1e-3 rad)

    def __init__(self, ctx_dim: int, hidden: int, k: int):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + 2, hidden),
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
    ):
        super().__init__()
        if pgram_peaks and not pgram_k:
            raise ValueError("pgram_peaks needs the periodogram (pgram_k > 0)")
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
        self.flow = zuko.flows.NSF(
            features=2,
            context=ctx_dim,
            transforms=flow_transforms,
            bins=flow_bins,
            hidden_features=list(flow_hidden),
        )
        self.vm = VonMisesMixture(ctx_dim, flow_hidden[0], n_vm)

    @classmethod
    def from_config(cls, cfg_model: dict) -> NPE:
        tuples = ("flow_hidden", "pgram_period_d")
        return cls(**{k: (tuple(v) if k in tuples else v) for k, v in cfg_model.items()})

    def context(self, batch: dict, amp: bool = False) -> torch.Tensor:
        dev = batch["t_d"].device.type
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=amp and dev == "cuda"):
            ctx = self.encoder(batch)
        return ctx.float()

    def log_prob_from_context(self, v: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """log q(v | ctx) for v (..., B, 3); ctx (B, C) broadcasts over leading dims."""
        u2, m0 = v[..., :2], v[..., 2]
        ctx_b = ctx.expand(*u2.shape[:-1], ctx.shape[-1])
        return self.flow(ctx).log_prob(u2) + self.vm.log_prob(m0, u2, ctx_b)

    def log_prob(self, batch: dict, v: torch.Tensor, amp: bool = False) -> torch.Tensor:
        return self.log_prob_from_context(v, self.context(batch, amp))

    def forward(self, batch: dict, v: torch.Tensor, amp: bool = False) -> torch.Tensor:
        """Training loss: mean negative log posterior density of the true parameters."""
        return -self.log_prob(batch, v, amp).mean()

    @torch.no_grad()
    def sample(self, batch: dict, n: int, with_log_prob: bool = False):
        """Draw n posterior samples per star: v of shape (n, B, 3) (and log q if requested)."""
        ctx = self.context(batch)
        u2 = self.flow(ctx).sample((n,))
        ctx_b = ctx.expand(n, *ctx.shape)
        m0 = self.vm.sample(u2, ctx_b)
        v = torch.cat([u2, m0[..., None]], dim=-1)
        if with_log_prob:
            return v, self.log_prob_from_context(v, ctx)
        return v
