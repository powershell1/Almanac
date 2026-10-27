"""How often does the anchor refinement end in the wrong basin? (2026-10-28)

For S/N >= 10 smoke test stars, compares the best Keplerian Delta chi^2 found by each search
variant with the Delta chi^2 at the true parameters ("wrong basin" = truth fits better by > 2).
Run from the repo root: python scripts/diagnostics/wrong_basin.py
"""

import time

import numpy as np
import torch

from gdr4.data.shards import gather_padded, read_shard
from gdr4.nn.data import to_torch_batch
from gdr4.nn.orbit_profile import circular_noise_scale, orbit_chi2_gain, profile_anchors
from gdr4.nn.periodogram import analyse_peaks, circular_periodogram, frequency_grid
from gdr4.sim.priors import Prior
from gdr4.utils import load_config

prior = Prior.from_config(load_config("configs/smoke_local_check.yaml")["prior"])
sh = read_shard("runs_local/shards/smoke/test/nominal", 0)
snr = np.asarray(sh["snr_a0"])[:300]
idx = np.flatnonzero(snr >= 10)
b = to_torch_batch(gather_padded(sh, idx), prior)
grid = frequency_grid(20, 3000, 512)
pg = circular_periodogram(b, grid)
pk = analyse_peaks(b, grid, pg, 8)
d2 = pk["d2"][:, :3]
step = (grid[1] - grid[0]).item()
kc = torch.exp(circular_noise_scale(b, pg))
sf = (torch.where(d2 < 0, torch.sqrt(2 / (-d2).clamp_min(1e-300)), step) * kc[:, None]).clamp(
    max=step
)
P = torch.tensor(np.asarray(sh["period_d"])[idx])[:, None]
E = torch.tensor(np.asarray(sh["ecc"])[idx])[:, None]
M = torch.tensor(np.asarray(sh["mean_anom0_rad"])[idx])[:, None]
g_true = orbit_chi2_gain(b, P, E, M)[:, 0].numpy()
s = snr[idx]
for name, kw in [
    ("current: coord ascent, grid 8x16", dict(joint=False)),
    ("joint + regrid, grid 8x16", dict(joint=True, regrid=True, n_e=8, n_m=16)),
    ("joint + regrid, grid 16x32", dict(joint=True, regrid=True, n_e=16, n_m=32)),
    ("joint + regrid, grid 16x64", dict(joint=True, regrid=True, n_e=16, n_m=64)),
    ("joint + regrid, grid 24x96", dict(joint=True, regrid=True, n_e=24, n_m=96)),
    (
        "joint + regrid, grid 24x96, 3 starts",
        dict(joint=True, regrid=True, n_e=24, n_m=96, n_starts=3),
    ),
]:
    t0 = time.time()
    prof = profile_anchors(b, 1 / pk["freq"][:, :3], prior.ecc_max, sigma_f=sf, noise_corr=kc, **kw)
    dt = time.time() - t0
    d = g_true - prof["gain_best"].amax(1).numpy()
    hi = s >= 30
    print(
        "%-40s wrong basin: %2d/%d (%3.0f%%) [S/N>=30: %2d/%d] | %5.1f s"
        % (name, (d > 2).sum(), len(idx), 100 * (d > 2).mean(), (d[hi] > 2).sum(), hi.sum(), dt),
        flush=True,
    )
