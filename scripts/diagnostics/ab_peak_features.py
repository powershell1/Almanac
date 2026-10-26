"""A/B test (2026-10-27): periodogram only vs periodogram + peak features.

Same small model, seed and data; trained on S/N > 6 stars from the local sanity-check training
shards (a fresh batch every step, no repeats), scored on S/N > 6 stars of the smoke test set
(toy scanning law; pipeline check only).
Run from the repo root: python scripts/diagnostics/ab_peak_features.py <n_peaks>
"""

import sys
import time

import numpy as np
import torch

from gdr4.data.shards import gather_padded, read_shard
from gdr4.eval.aliases import period_metrics
from gdr4.nn.data import to_torch_batch
from gdr4.nn.model import NPE
from gdr4.sim.priors import Prior
from gdr4.utils import load_config

STEPS, BATCH = 800, 128
cfg = load_config("configs/smoke_local_check.yaml")
prior = Prior.from_config(cfg["prior"])
shards = [read_shard("runs_local/shards/local_check/train", i) for i in range(11)]
hi_snr = [np.flatnonzero(np.asarray(sh["snr_a0"]) > 6) for sh in shards]
data_rng = np.random.default_rng(0)  # same batch sequence for every variant


def fresh_batch():
    k = data_rng.integers(len(shards))
    pick = np.sort(data_rng.choice(hi_snr[k], BATCH, replace=False))
    return to_torch_batch(gather_padded(shards[k], pick), prior)


test = read_shard("runs_local/shards/smoke/test/nominal", 0)
t_idx = np.flatnonzero(np.asarray(test["snr_a0"]) > 6)
t_batch = to_torch_batch(gather_padded(test, t_idx), prior)
p_true = np.asarray(test["period_d"])[t_idx]

for peaks in [int(a) for a in sys.argv[1:]] or (0, 8):
    torch.manual_seed(0)
    model = NPE(
        features="raw",
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_freq=16,
        ctx_dim=64,
        flow_transforms=3,
        flow_hidden=(64, 64),
        flow_bins=8,
        n_vm=4,
        pgram_k=512,
        pgram_peaks=peaks,
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    t0, hist = time.time(), []
    for _ in range(STEPS):
        b = fresh_batch()
        loss = model(b, b["v"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        hist.append(loss.item())
    model.eval()
    with torch.no_grad():
        test_loss = model(t_batch, t_batch["v"]).item()
        v = model.sample(t_batch, 1000).numpy().astype(np.float64)
    pm = period_metrics(prior.from_unconstrained(v)["period_d"], p_true)
    print(
        f"peaks={peaks}: train loss last100 {np.mean(hist[-100:]):.3f} | test loss {test_loss:.3f}"
        f" | period recovered {pm['recovered'].mean():.3f} | wrong-period (alias) "
        f"{pm['alias_failure'].mean():.3f} | n={len(t_idx)} | {time.time() - t0:.0f}s"
    )
