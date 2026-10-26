"""Kill-test evaluation: posterior sampling on the frozen test set, metrics, and the decision.

Every number in ``metrics.json`` / ``decision.md`` is computed here from model outputs, so the
write-up can quote them directly (CLAUDE.md: never hand-edit results).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from gdr4.data.shards import gather_padded, n_stars, read_shard
from gdr4.eval.aliases import period_metrics
from gdr4.eval.c2st import c2st_accuracy
from gdr4.eval.calibration import (
    bootstrap_coverage,
    central_interval_coverage,
    circular_embed,
    expected_coverage,
    hpd_levels,
    marginal_ranks,
    tarp_ecp,
)
from gdr4.nn.data import to_torch_batch
from gdr4.sim.priors import Prior


@torch.no_grad()
def sample_posteriors(model, shard, prior, n_samples, device, stars_per_batch=32):
    """Posterior samples v (n, B, 3), their log q (n, B), and log q of the truth (B,)."""
    n = n_stars(shard)
    vs, lqs, lqt = [], [], []
    for s in range(0, n, stars_per_batch):
        idx = np.arange(s, min(n, s + stars_per_batch))
        b = to_torch_batch(gather_padded(shard, idx), prior, device)
        v, lq = model.sample(b, n_samples, with_log_prob=True)
        vs.append(v.cpu().numpy())
        lqs.append(lq.cpu().numpy())
        lqt.append(model.log_prob(b, b["v"]).cpu().numpy())
    return np.concatenate(vs, 1), np.concatenate(lqs, 1), np.concatenate(lqt)


@torch.no_grad()
def time_inference(model, shard, prior, n_samples, device, n_stars_timed=50):
    """Median single-star latency (ms) for n_samples draws, plus batched throughput (ms/star)."""

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    b0 = to_torch_batch(gather_padded(shard, np.arange(1)), prior, device)
    for _ in range(3):  # warm-up
        model.sample(b0, n_samples)
    sync()
    lat = []
    for i in range(n_stars_timed):
        b = to_torch_batch(gather_padded(shard, np.array([i])), prior, device)
        t0 = time.perf_counter()
        model.sample(b, n_samples)
        sync()
        lat.append((time.perf_counter() - t0) * 1e3)
    nb = min(256, n_stars(shard))
    b = to_torch_batch(gather_padded(shard, np.arange(nb)), prior, device)
    sync()
    t0 = time.perf_counter()
    for s in range(0, nb, 32):
        model.sample({k: v[s : s + 32] for k, v in b.items()}, n_samples)
    sync()
    return float(np.median(lat)), float((time.perf_counter() - t0) * 1e3 / nb)


def _summary(levels, v_true, v_samples, period_true, period_samples, snr, ecfg, rng):
    alphas = ecfg["levels"]
    cov = expected_coverage(levels, alphas)
    lo, hi = bootstrap_coverage(levels, alphas, rng)
    _, tarp = tarp_ecp(circular_embed(v_samples), circular_embed(v_true), rng, alphas)
    pm = period_metrics(
        period_samples, period_true, ecfg["period_tol"], ecfg["alias_mass_threshold"]
    )
    hi_snr = snr > ecfg["snr_min"]
    out = {
        "n_stars": int(len(levels)),
        "levels": list(alphas),
        "hpd_coverage": cov.tolist(),
        "hpd_coverage_ci95": [lo.tolist(), hi.tolist()],
        "hpd_coverage_error_max": float(np.max(np.abs(cov - alphas))),
        "tarp_coverage": tarp.tolist(),
        "tarp_coverage_error_max": float(np.max(np.abs(tarp - alphas))),
        "n_high_snr": int(hi_snr.sum()),
        "period_recovery_high_snr": float(np.mean(pm["recovered"][hi_snr]))
        if hi_snr.any()
        else None,
        "alias_failure_high_snr": float(np.mean(pm["alias_failure"][hi_snr]))
        if hi_snr.any()
        else None,
    }
    for d, name in ((0, "u_P"), (1, "u_e")):
        ranks = marginal_ranks(v_true[:, d], v_samples[..., d])
        out[f"marginal_coverage_{name}"] = central_interval_coverage(ranks, alphas).tolist()
    return out, pm


def _c2st_vs_reference(v_npe, ref_dir: Path, ecfg, rng):
    rows = []
    for f in sorted(ref_dir.glob("star_*.npz")):
        z = np.load(f)
        i = int(z["star_index"])
        if i >= v_npe.shape[1]:
            continue
        row = {"star": i, "ess": float(z["ess"]), "c2st": None}
        if z["ess"] >= ecfg.get("c2st_min_ess", 2000):
            n = min(int(ecfg.get("c2st_n_samples", 2000)), v_npe.shape[0], len(z["v"]))
            ref = z["v"][rng.choice(len(z["v"]), size=n, replace=False), :3]
            npe = v_npe[rng.choice(v_npe.shape[0], size=n, replace=False), i]
            row["c2st"] = c2st_accuracy(circular_embed(npe), circular_embed(ref), seed=i)
        rows.append(row)
    return rows


def _reference_self_calibration(ref_dir: Path, v_true, alphas):
    """Marginal coverage of the *reference* posteriors themselves (sanity check of the sampler)."""
    files = sorted(ref_dir.glob("star_*.npz"))
    if not files:
        return None
    ranks = {0: [], 1: []}
    for f in files:
        z = np.load(f)
        i = int(z["star_index"])
        for d in ranks:
            ranks[d].append(np.mean(z["v"][:, d] < v_true[i, d]))
    return {
        name: central_interval_coverage(np.array(ranks[d]), alphas).tolist()
        for d, name in ((0, "u_P"), (1, "u_e"))
    } | {"n_stars": len(files)}


def decide(nominal: dict, ms_per_star: float, c2st_rows: list, thr: dict) -> dict:
    """Apply the pre-registered continue / abandon thresholds (CLAUDE.md)."""
    vals = [r["c2st"] for r in c2st_rows if r["c2st"] is not None]
    c2st_frac = float(np.mean(np.array(vals) <= thr["c2st_max"])) if vals else None
    checks = {
        "coverage_error": (nominal["hpd_coverage_error_max"], "<=", thr["max_coverage_error"]),
        "period_recovery_high_snr": (
            nominal["period_recovery_high_snr"],
            ">=",
            thr["min_period_recovery"],
        ),
        "ms_per_star": (ms_per_star, "<=", thr["max_ms_per_star"]),
        "c2st_pass_fraction": (c2st_frac, ">=", thr["c2st_min_fraction"]),
    }
    passed = {}
    for k, (v, op, t) in checks.items():
        passed[k] = None if v is None else bool(v <= t if op == "<=" else v >= t)
    abandon = {
        "coverage_error_gt_abandon": nominal["hpd_coverage_error_max"]
        > thr["abandon_coverage_error"],
        "alias_failure_gt_abandon": (nominal["alias_failure_high_snr"] or 0.0)
        > thr["abandon_alias_failure"],
    }
    if all(v is True for v in passed.values()):
        verdict = "CONTINUE: all continue criteria met"
    elif any(abandon.values()):
        verdict = (
            "ABANDON CRITERION MET (applies if it persists after one retrain): "
            "switch to the fallback contribution"
        )
    elif any(v is None for v in passed.values()):
        verdict = "INCOMPLETE: some criteria could not be evaluated (see null entries)"
    else:
        verdict = "NOT YET: continue criteria not all met; retrain once and re-evaluate"
    return {
        "checks": {
            k: {"value": v[0], "op": v[1], "threshold": v[2], "pass": passed[k]}
            for k, v in checks.items()
        },
        "abandon": abandon,
        "verdict": verdict,
        "n_c2st_eligible": len(vals),
    }


def _decision_markdown(run, decision, nominal, timing):
    lines = [
        f"# Kill-test decision: {run}",
        "",
        "Generated by scripts/evaluate.py. Do not edit.",
        "",
        f"**Verdict:** {decision['verdict']}",
        "",
        "| criterion | value | rule | pass |",
        "|---|---|---|---|",
    ]
    for k, c in decision["checks"].items():
        v = "n/a" if c["value"] is None else f"{c['value']:.4g}"
        lines.append(f"| {k} | {v} | {c['op']} {c['threshold']} | {c['pass']} |")
    lines += [
        "",
        f"Abandon checks: {decision['abandon']}",
        f"C2ST-eligible reference stars: {decision['n_c2st_eligible']}",
        f"Timing: single-star {timing['ms_per_star_single']:.1f} ms, batched "
        f"{timing['ms_per_star_batched']:.2f} ms/star on {timing['device']}",
        f"HPD coverage at {nominal['levels']}: {nominal['hpd_coverage']}",
        f"High-S/N stars: {nominal['n_high_snr']}",
    ]
    return "\n".join(lines) + "\n"


def evaluate(cfg: dict, model, test_dir: Path, ref_dir: Path, out_dir: Path, device) -> dict:
    ecfg = cfg["evaluation"]
    prior = Prior.from_config(cfg["prior"])
    rng = np.random.default_rng(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, c2st_rows, timing = {}, [], None
    for noise in cfg["test_set"]["noise_models"]:
        shard = read_shard(test_dir / noise, 0, mmap=False)
        v_s, lq_s, lq_t = sample_posteriors(
            model, shard, prior, int(ecfg["n_posterior_samples"]), device
        )
        truth = {k: np.asarray(shard[k]) for k in ("period_d", "ecc", "mean_anom0_rad")}
        v_true = prior.to_unconstrained(truth["period_d"], truth["ecc"], truth["mean_anom0_rad"])
        period_s = prior.from_unconstrained(v_s.astype(np.float64))["period_d"]
        snr = np.asarray(shard[ecfg["snr_key"]])
        summ, pm = _summary(
            hpd_levels(lq_t, lq_s), v_true, v_s, truth["period_d"], period_s, snr, ecfg, rng
        )
        metrics[noise] = summ
        np.savez(
            out_dir / f"per_star_{noise}.npz",
            hpd_level=hpd_levels(lq_t, lq_s),
            period_true=truth["period_d"],
            period_median=pm["median_d"],
            mass_within=pm["mass_within"],
            alias_failure=pm["alias_failure"],
            snr_a0=np.asarray(shard["snr_a0"]),
            snr_mf=np.asarray(shard["snr_mf"]),
            ecc_true=truth["ecc"],
            n_obs=np.asarray(shard["n_obs"]),
        )
        if noise == "nominal":
            np.savez(
                out_dir / "samples_nominal_first200.npz", v=v_s[:500, :200], v_true=v_true[:200]
            )
            if ref_dir.exists():
                c2st_rows = _c2st_vs_reference(v_s, ref_dir, ecfg, rng)
                metrics["reference_self_calibration"] = _reference_self_calibration(
                    ref_dir, v_true, ecfg["levels"]
                )
            lat, thr = time_inference(model, shard, prior, int(ecfg["n_posterior_samples"]), device)
            timing = {
                "ms_per_star_single": lat,
                "ms_per_star_batched": thr,
                "n_samples": int(ecfg["n_posterior_samples"]),
                "device": str(device),
            }
    metrics["timing"] = timing
    metrics["c2st"] = c2st_rows
    decision = decide(
        metrics["nominal"], timing["ms_per_star_single"], c2st_rows, ecfg["thresholds"]
    )
    metrics["decision"] = decision
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "decision.md").write_text(
        _decision_markdown(cfg["run_name"], decision, metrics["nominal"], timing)
    )
    return metrics
