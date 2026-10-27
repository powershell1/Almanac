"""Check (2026-10-27): distance of the period anchor from the exact posterior centre.

Circular-periodogram anchor vs the Keplerian re-refined anchor, in units of the exact
(reference) posterior standard deviation of u_P, for the 20 local reference stars.
Run from the repo root: python scripts/diagnostics/check_period_anchor.py
"""

import numpy as np
import torch
from scipy.special import ndtri

from gdr4.data.shards import gather_padded, read_shard
from gdr4.nn.data import to_torch_batch
from gdr4.nn.orbit_profile import profile_anchors
from gdr4.nn.periodogram import analyse_peaks, circular_periodogram, frequency_grid
from gdr4.sim.priors import Prior
from gdr4.utils import load_config

prior = Prior.from_config(load_config("configs/smoke_local_check.yaml")["prior"])
sh = read_shard("runs_local/shards/smoke/test/nominal", 0)
b = to_torch_batch(gather_padded(sh, np.arange(20)), prior)
grid = frequency_grid(20, 3000, 512)
peaks = analyse_peaks(b, grid, circular_periodogram(b, grid), 8)
f_c = peaks["freq"][:, :1]
sig_f = torch.sqrt(2.0 / (-peaks["d2"][:, :1]).clamp_min(1e-300))
prof = profile_anchors(b, 1.0 / f_c, prior.ecc_max, sigma_f=sig_f)
R = np.log(prior.period_max_d / prior.period_min_d)


def u_of_p(p):
    return ndtri(np.clip((np.log(p) - np.log(prior.period_min_d)) / R, 1e-9, 1 - 1e-9))


snr = np.asarray(sh["snr_a0"])[:20]
true_p = np.asarray(sh["period_d"])[:20]
print(" star   S/N | anchor offset from exact centre (exact sd): circular  Keplerian | e*  e_true")
for i in np.argsort(snr):
    z = np.load(f"runs_local/shards/smoke/reference/star_{i:05d}.npz")
    m, s = z["v"][:, 0].mean(), z["v"][:, 0].std()
    oc = (u_of_p(1 / f_c[i, 0].item()) - m) / s
    ok = (u_of_p(prof["period"][i, 0].item()) - m) / s
    print(
        f"{i:4d} {snr[i]:6.1f} | {oc:+10.1f} {ok:+10.1f}"
        f"            | {prof['ecc'][i, 0].item():.2f} {np.asarray(sh['ecc'])[i]:.2f}"
    )
