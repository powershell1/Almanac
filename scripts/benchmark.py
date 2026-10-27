"""Time one training step and posterior sampling on this machine (run in notebook 01).

Uses simulated stars on the toy scanning law (no data needed) and the model in the config, so
the numbers tell you how many training steps fit in the time budget before any counted run.

    python scripts/benchmark.py --config configs/killtest.yaml
"""

import argparse
import time

import numpy as np
import torch

from gdr4.nn.data import to_torch_batch
from gdr4.nn.model import NPE
from gdr4.sim.noise import get_noise_model
from gdr4.sim.priors import Prior
from gdr4.sim.scan import build_cache_toy_nsl
from gdr4.sim.simulate import simulate_batch
from gdr4.utils import load_config


def timed(fn, device, reps):
    fn()  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = Prior.from_config(cfg["prior"])
    bs = int(cfg["training"]["batch_size"])
    sim = simulate_batch(
        np.random.default_rng(0),
        bs,
        prior,
        build_cache_toy_nsl(4),
        get_noise_model(cfg["simulation"]["noise_model"]),
    )
    batch = to_torch_batch(sim, prior, device)
    torch.manual_seed(0)
    model = NPE.from_config(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp = bool(cfg["training"].get("amp")) and device.type == "cuda"
    print(
        f"device {device} | {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M parameters"
        f" | batch {bs}"
    )

    def step():
        loss = model(batch, batch["v"], amp=amp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t_step = timed(step, device, args.reps)
    with torch.no_grad():
        t_enc = timed(lambda: model.context(batch, amp=amp), device, args.reps)
    n_samp = int(cfg["evaluation"]["n_posterior_samples"])
    b32 = {k: v[:32] for k, v in batch.items()}
    b1 = {k: v[:1] for k, v in batch.items()}
    model.eval()
    t32 = timed(lambda: model.sample(b32, n_samp), device, args.reps)
    t1 = timed(lambda: model.sample(b1, n_samp), device, args.reps)
    hours = float(cfg["training"]["max_hours"])
    n_dev = max(1, torch.cuda.device_count()) if device.type == "cuda" else 1
    steps = hours * 3600 / t_step
    print(
        f"training step: {t_step:.3f} s (of which physics features + encoder forward {t_enc:.3f} s)"
    )
    print(
        f"=> ~{steps:,.0f} steps per {hours:g} h session on one device "
        f"(~{steps * n_dev:,.0f} star-batches with {n_dev} device(s) in DDP); "
        f"config max_steps = {int(cfg['training']['max_steps']):,}"
    )
    print(
        f"sampling {n_samp} draws: {1e3 * t1:.1f} ms for one star, "
        f"{1e3 * t32 / 32:.1f} ms per star in batches of 32 (gate: 50 ms)"
    )


if __name__ == "__main__":
    main()
