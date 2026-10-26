"""Diagnostic from 2026-10-27 (see logs/research_log.md). Uses the local sanity-check run in
runs_local/ (toy scanning law, smoke_local_check config); paths are hard-coded on purpose.
Run from the repo root: python scripts/diagnostics/diag_ref.py
"""

import numpy as np
import torch

from gdr4.data.shards import gather_padded, read_shard
from gdr4.nn.data import to_torch_batch
from gdr4.nn.train import load_model
from gdr4.reference.bruteforce import log_likelihood
from gdr4.sim.priors import Prior
from gdr4.sim.simulate import star_view
from gdr4.utils import load_config

cfg = load_config("configs/smoke_local_check.yaml")
prior = Prior.from_config(cfg["prior"])
model, _ = load_model("runs_local/checkpoints/local_check/best.pt")
sh = read_shard("runs_local/shards/smoke/test/nominal", 0)
pad = gather_padded(sh, np.arange(20))
b = to_torch_batch(pad, prior)
torch.manual_seed(0)
v_npe = model.sample(b, 2000).numpy()
rng = np.random.default_rng(0)
snr = np.asarray(sh["snr_a0"])[:20]


def log_post(star, v3, amp):  # exact log p(theta_nl, s_a | w) up to a constant
    v4 = np.c_[v3, np.log(amp)]
    return prior.log_prob_unconstrained(v4) + log_likelihood(star, prior, v4)


print("star  S/N | median exact log-post: truth   ref   NPE | truth z in ref  in NPE (u_P)")
for i in np.argsort(snr)[-10:]:
    s = star_view(pad, i)
    z = np.load(f"runs_local/shards/smoke/reference/star_{i:05d}.npz")
    ref4 = z["v"][rng.choice(len(z["v"]), 400, replace=False)]
    npe3 = v_npe[rng.choice(2000, 400, replace=False), i]
    # give NPE samples the best amplitude scale (favours NPE: its true score is lower)
    amps = np.exp(np.linspace(np.log(0.01), np.log(2.0), 25))
    lp_npe = np.max([log_post(s, npe3, np.full(400, a)) for a in amps], axis=0)
    lp_ref = prior.log_prob_unconstrained(ref4) + log_likelihood(s, prior, ref4)
    vt3 = b["v"].numpy()[i]
    lp_true = log_post(s, vt3[None], np.array([s["amp_scale_mas"]]))[0]
    zr = (vt3[0] - ref4[:, 0].mean()) / ref4[:, 0].std()
    zn = (vt3[0] - npe3[:, 0].mean()) / npe3[:, 0].std()
    print(
        f"{i:4d} {snr[i]:5.1f} | {lp_true:10.1f} {np.median(lp_ref):10.1f} {np.median(lp_npe):10.1f} | {zr:+6.2f}  {zn:+6.2f}"
    )
