"""Brute-force reference posteriors for the first n_reference test stars (resumable, CPU).

    python scripts/reference_posteriors.py --config configs/killtest.yaml --root /kaggle/working \
        --set paths.test_dir=/kaggle/input/gdr4-test-set/shards/test --n-jobs 4
    # split across sessions with --stars 0-99 and --stars 100-199
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process

import time  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from gdr4.cli import base_parser, parse  # noqa: E402
from gdr4.data.shards import gather_padded, read_shard  # noqa: E402
from gdr4.reference.bruteforce import reference_posterior  # noqa: E402
from gdr4.sim.noise import get_noise_model  # noqa: E402
from gdr4.sim.priors import Prior  # noqa: E402
from gdr4.sim.simulate import star_view  # noqa: E402
from gdr4.utils import load_yaml, resolve, write_run_info  # noqa: E402

N_KEEP = 20000  # resampled posterior draws stored per star


def run_one(i, star, prior_cfg, ref_cfg, out_dir, entropy, krange):
    out = out_dir / f"star_{i:05d}.npz"
    if out.exists():
        return i, "skip"
    prior = Prior.from_config(prior_cfg)
    rng = np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(i,)))
    r = reference_posterior(
        star,
        prior,
        rng,
        n_prior_samples=int(ref_cfg["n_prior_samples"]),
        refine=bool(ref_cfg["refine"]),
        n_refine_samples=int(ref_cfg["n_refine_samples"]),
        refine_rounds=int(ref_cfg["refine_rounds"]),
        chunk=int(ref_cfg["chunk"]),
        noise_scale_range=krange,
    )
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        noise_scale_marginalised=krange is not None,
        v=r.resample(rng, N_KEEP),
        ess=r.ess,
        ess_stage1=r.ess_stage1,
        log_evidence=r.log_evidence,
        n_likelihood_evals=r.n_likelihood_evals,
        seconds=r.seconds,
        star_index=i,
    )
    os.replace(tmp, out)
    return i, f"ess={r.ess:.0f} ess1={r.ess_stage1:.0f} {r.seconds:.0f}s"


def main():
    p = base_parser(__doc__)
    p.add_argument("--stars", default="", help="inclusive index range, e.g. 0-99")
    p.add_argument("--n-jobs", type=int, default=4)
    args, cfg = parse(p)
    seeds = load_yaml(cfg["test_set"]["seeds_file"])
    n_ref = int(seeds["n_reference"])
    lo, hi = (0, n_ref - 1)
    if args.stars:
        a, _, b = args.stars.partition("-")
        lo, hi = int(a), int(b or a)
    # References use the evaluation's reference test set and marginalise the same noise model
    # the network was trained on (consistency rule, CLAUDE.md).
    ref_noise = cfg["evaluation"].get("reference_noise_model", "nominal")
    krange = get_noise_model(cfg["simulation"]["noise_model"]).sigma_scale_range
    shard = read_shard(resolve(cfg, "test_dir", args.root) / ref_noise, 0)
    idx = np.arange(lo, min(hi, n_ref - 1) + 1)
    batch = gather_padded(shard, idx)
    out_dir = resolve(cfg, "reference_dir", args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(
        out_dir,
        cfg,
        {
            "stage": "reference",
            "stars": [int(lo), int(hi)],
            "test_set": ref_noise,
            "noise_scale_range": krange,
        },
    )
    entropy = int(seeds["test_entropy"]) + 1  # sampler randomness, distinct from the test draws
    t0 = time.time()
    jobs = (
        delayed(run_one)(
            int(i), star_view(batch, k), cfg["prior"], cfg["reference"], out_dir, entropy, krange
        )
        for k, i in enumerate(idx)
    )
    for i, msg in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(jobs):
        print(f"star {i}: {msg}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
