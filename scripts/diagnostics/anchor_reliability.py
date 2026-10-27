"""Anchor reliability (2026-10-28): old coordinate ascent vs joint Newton + Laplace covariance.

For every test star: where does the truth fall relative to the top-3 anchors, in units of each
anchor's width? Reliable anchors give |z| < 2 about 95% of the time and a robust spread ~1.
No network involved. Run: python scripts/diagnostics/anchor_reliability.py [test_set]
"""

import math
import sys
import time

import numpy as np
import torch

from gdr4.data.shards import gather_padded, read_shard
from gdr4.nn.data import to_torch_batch
from gdr4.nn.orbit_profile import (
    circular_noise_scale,
    laplace_to_target,
    noise_scale_estimate,
    profile_anchors,
)
from gdr4.nn.periodogram import analyse_peaks, circular_periodogram, frequency_grid
from gdr4.sim.priors import Prior
from gdr4.utils import load_config

TEST = sys.argv[1] if len(sys.argv) > 1 else "nominal"
prior = Prior.from_config(load_config("configs/smoke_local_check.yaml")["prior"])
sh = read_shard(f"runs_local/shards/smoke/test/{TEST}", 0)
n = 300
b = to_torch_batch(gather_padded(sh, np.arange(n)), prior)
snr = np.asarray(sh["snr_a0"])[:n]
vt = b["v"].numpy()
grid = frequency_grid(20.0, 3000.0, 512)
pg = circular_periodogram(b, grid)
pk = analyse_peaks(b, grid, pg, 8)
d2 = pk["d2"][:, :3]
step = (grid[1] - grid[0]).item()
kc = torch.exp(circular_noise_scale(b, pg))
sig_f = (torch.where(d2 < 0, torch.sqrt(2.0 / (-d2).clamp_min(1e-300)), step) * kc[:, None]).clamp(
    max=step
)
R = math.log(prior.period_max_d / prior.period_min_d)


def phi(u):
    return np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


def anchors(joint):
    t0 = time.time()
    prof = profile_anchors(
        b, 1.0 / pk["freq"][:, :3], prior.ecc_max, sigma_f=sig_f, joint=joint, noise_corr=kc
    )
    secs = time.time() - t0
    lk = noise_scale_estimate(b, prof)
    kap = torch.exp(lk).numpy()[:, None]
    if joint:
        lap = laplace_to_target(prof, (prior.period_min_d, prior.period_max_d), prior.ecc_max, lk)
        mu, sd = lap["mu"].numpy(), np.sqrt(np.diagonal(lap["cov"].numpy(), axis1=-2, axis2=-1))
        return mu, sd, secs
    f = 1.0 / prof["period"].numpy()
    u_p = (np.log(1 / f) - np.log(prior.period_min_d)) / R
    from scipy.special import ndtri

    u_p = ndtri(np.clip(u_p, 1e-6, 1 - 1e-6))
    tau_p = sig_f.numpy() / kc.numpy()[:, None] / (f * R * phi(u_p)) * kap  # circular width
    u_e = ndtri(np.clip(prof["ecc"].numpy() / prior.ecc_max, 1e-6, 1 - 1e-6))
    tau_e = prof["sigma_e"].numpy() / (prior.ecc_max * phi(u_e)) * kap
    tau_m = prof["sigma_m0"].numpy() * kap
    return np.stack([u_p, u_e, prof["m0"].numpy()], -1), np.stack([tau_p, tau_e, tau_m], -1), secs


def rs(z):
    return 1.4826 * np.median(np.abs(z - np.median(z)))


for joint in (False, True):
    mu, sd, secs = anchors(joint)
    z = vt[:, None, :] - mu
    z[..., 2] = np.angle(np.exp(1j * z[..., 2]))
    z = z / sd
    right = np.abs(z[..., 0]) < 5
    print(
        f"\n{'joint Newton + Laplace' if joint else 'coordinate ascent (current)'}"
        f"  [{secs:.1f} s for {n} stars x 3 anchors on CPU]"
    )
    print(
        " S/N bin   n | no anchor right | at the right anchor, |z|<2 for P / e / M0"
        " | robust sd P / e / M0"
    )
    for lo, hi in ((0, 3), (3, 10), (10, 30), (30, 1e9)):
        s = np.flatnonzero((snr >= lo) & (snr < hi))
        ok = [i for i in s if right[i].any()]
        k = [int(np.argmax(right[i])) for i in ok]
        zz = np.array([z[i, kk] for i, kk in zip(ok, k, strict=True)])
        fr = [100 * np.mean(np.abs(zz[:, d]) < 2) for d in range(3)]
        print(
            f"{lo:4g}-{hi:<5g} {len(s):3d} | {100 * (1 - len(ok) / len(s)):14.0f}% |"
            f"            {fr[0]:5.0f}% / {fr[1]:4.0f}% / {fr[2]:4.0f}%       |"
            f" {rs(zz[:, 0]):5.2f} / {rs(zz[:, 1]):4.2f} / {rs(zz[:, 2]):4.2f}"
        )
