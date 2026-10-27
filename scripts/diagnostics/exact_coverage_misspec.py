"""Is coverage under noise misspecification a network problem or a model problem? (2026-10-27)

For the first N smoke test stars, compute exact (reference) posteriors that use the *reported*
sigma, on the nominal test set and on sigma_inflated_1.2 (true noise 1.2x the reported sigma),
and compare their marginal central-interval coverage with the network's on the same stars.
Run from the repo root: python scripts/diagnostics/exact_coverage_misspec.py [N] [checkpoint]
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from gdr4.data.shards import gather_padded, read_shard  # noqa: E402
from gdr4.eval.calibration import central_interval_coverage, marginal_ranks  # noqa: E402
from gdr4.nn.data import to_torch_batch  # noqa: E402
from gdr4.nn.train import load_model  # noqa: E402
from gdr4.reference.bruteforce import reference_posterior  # noqa: E402
from gdr4.sim.priors import Prior  # noqa: E402
from gdr4.sim.simulate import star_view  # noqa: E402
from gdr4.utils import load_config  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
CKPT = sys.argv[2] if len(sys.argv) > 2 else "runs_local/checkpoints/local_check_refine/best.pt"
LEVELS = [0.5, 0.68, 0.9]
prior = Prior.from_config(load_config("configs/smoke_local_check.yaml")["prior"])


def exact_samples(star, seed):
    r = reference_posterior(
        star,
        prior,
        np.random.default_rng(seed),
        n_prior_samples=100000,
        n_refine_samples=100000,
        refine_rounds=2,
    )
    return r.resample(np.random.default_rng(seed + 1), 4000)[:, :2], r.ess


torch.set_num_threads(8)
model, _ = load_model(CKPT)
for noise in ("nominal", "sigma_inflated_1.2"):
    sh = read_shard(f"runs_local/shards/smoke/test/{noise}", 0)
    pad = gather_padded(sh, np.arange(N))
    out = Parallel(n_jobs=14)(delayed(exact_samples)(star_view(pad, i), i) for i in range(N))
    ex = np.stack([o[0] for o in out], axis=1)  # (4000, N, 2)
    ess = np.array([o[1] for o in out])
    b = to_torch_batch(pad, prior)
    with torch.no_grad():
        npe = model.sample(b, 2000).numpy()[..., :2]
    vt = b["v"].numpy()[:, :2]
    ok = ess > 200
    print(f"\n{noise}: {N} stars ({ok.sum()} with reference ESS > 200 used)")
    print("  dim   method  central coverage at 50/68/90%")
    for d, name in ((0, "u_P"), (1, "u_e")):
        for lab, smp in (("exact", ex), ("NPE", npe)):
            cov = central_interval_coverage(marginal_ranks(vt[ok, d], smp[:, ok, d]), LEVELS)
            print(f"  {name:4s}  {lab:6s}  " + "  ".join(f"{c:.2f}" for c in cov))
