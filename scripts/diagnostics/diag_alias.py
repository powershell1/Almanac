"""Diagnostic from 2026-10-27 (see logs/research_log.md). Uses the local sanity-check run in
runs_local/ (toy scanning law, smoke_local_check config); paths are hard-coded on purpose.
Run from the repo root: python scripts/diagnostics/diag_alias.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from joblib import Parallel, delayed

from gdr4.data.shards import gather_padded, read_shard
from gdr4.reference.bruteforce import reference_posterior
from gdr4.sim.priors import Prior
from gdr4.sim.simulate import star_view
from gdr4.utils import load_config

cfg = load_config("configs/smoke_local_check.yaml")
prior = Prior.from_config(cfg["prior"])
z = np.load("runs_local/results/smoke_local_check/eval/per_star_nominal.npz")
fail = np.flatnonzero((z["snr_a0"] > 6) & z["alias_failure"])
sh = read_shard("runs_local/shards/smoke/test/nominal", 0)
pad = gather_padded(sh, fail)


def run(k):
    s = star_view(pad, k)
    r = reference_posterior(
        s,
        prior,
        np.random.default_rng(k),
        n_prior_samples=200000,
        n_refine_samples=200000,
        refine_rounds=3,
    )
    P = prior.from_unconstrained(r.resample(np.random.default_rng(1), 20000))["period_d"]
    rel = np.abs(P / s["period_d"] - 1)
    return (int(fail[k]), r.ess, np.mean(rel < 0.1), np.median(P), np.quantile(P, [0.05, 0.95]))


out = Parallel(n_jobs=14)(delayed(run)(k) for k in range(len(fail)))
print(
    " idx   S/N  P_true  e    | exact: mass<10%  median  90% interval      ESS | NPE: mass<10%  median"
)
for i, ess, mass, med, q in sorted(out, key=lambda r: z["snr_a0"][r[0]]):
    print(
        f"{i:4d} {z['snr_a0'][i]:6.1f} {z['period_true'][i]:7.0f} {z['ecc_true'][i]:4.2f} |"
        f"        {mass:5.2f}  {med:6.0f}  [{q[0]:5.0f}, {q[1]:5.0f}] {ess:6.0f} |"
        f"      {z['mass_within'][i]:5.2f}  {z['period_median'][i]:6.0f}"
    )
