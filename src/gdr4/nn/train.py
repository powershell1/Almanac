"""Training loop with checkpoint/resume, a wall-clock budget, AMP and optional DDP.

Single GPU:   python scripts/train.py --config configs/killtest.yaml --root /kaggle/working
Two T4s:      torchrun --nproc_per_node 2 scripts/train.py --config ... --root ...
Rerunning the same command resumes from ``<checkpoint_dir>/last.pt``.
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from gdr4.data.shards import list_complete, read_shard
from gdr4.nn.data import ShardStream, batches_from_shard
from gdr4.nn.model import NPE
from gdr4.sim.priors import Prior


def _ddp_setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, 0
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return rank, world, local


def _lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        frac = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac))

    return f


def check_model_matches_prior(cfg: dict) -> None:
    """The anchored head maps periods into the prior's u_P coordinate, so its range must match."""
    m = cfg["model"]
    if m.get("head", "nsf") == "anchored":
        want = [float(x) for x in cfg["prior"]["period_d"]]
        got = [float(x) for x in m.get("target_period_d", (30.0, 2000.0))]
        if got != want:
            raise SystemExit(f"model.target_period_d {got} must equal prior.period_d {want}")
        if m.get("profile") and float(m.get("target_ecc_max", 0.9)) != float(
            cfg["prior"]["ecc_max"]
        ):
            raise SystemExit("model.target_ecc_max must equal prior.ecc_max")


def _save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(cfg: dict, shards_dir: Path, ckpt_dir: Path, results_dir: Path, log=print) -> dict:
    tcfg = cfg["training"]
    rank, world, local = _ddp_setup()
    is_main = rank == 0
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    prior = Prior.from_config(cfg["prior"])
    check_model_matches_prior(cfg)

    shards = list_complete(shards_dir)
    n_val = int(tcfg["val_shards"])
    if len(shards) <= n_val:
        raise SystemExit(f"need more than {n_val} complete shards in {shards_dir}")
    train_idx, val_idx = shards[:-n_val], shards[-n_val:]

    torch.manual_seed(int(tcfg["seed"]) + rank)
    model = NPE.from_config(cfg["model"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
    )
    max_steps = int(tcfg["max_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(int(tcfg["warmup_steps"]), max_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    last = ckpt_dir / "last.pt"
    step, best_val, elapsed_prev = 0, float("inf"), 0.0
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step, best_val = state["step"], state["best_val"]
        elapsed_prev = state.get("train_seconds", 0.0)
        if is_main:
            log(f"resumed from {last} at step {step}")
    if step >= max_steps:
        if is_main:
            log("already finished")
        return {"step": step, "best_val": best_val}

    net = model
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None
        )

    stream = ShardStream(
        shards_dir,
        train_idx,
        int(tcfg["batch_size"]),
        prior,
        seed=int(tcfg["seed"]) * 1000 + step,
        rank=rank,
        world=world,
    )
    loader = torch.utils.data.DataLoader(
        stream,
        batch_size=None,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(tcfg["num_workers"]) > 0,
    )

    val_batches = None
    if is_main:  # fixed validation batches from the held-out training shard(s)
        val_batches = []
        for s in val_idx:
            val_batches += list(
                batches_from_shard(read_shard(shards_dir, s), int(tcfg["batch_size"]), prior)
            )
        val_batches = val_batches[:20]

    def validate():
        model.eval()
        with torch.no_grad():
            tot = []
            for b in val_batches:
                b = _to(b, device)
                tot.append(float(model(b, b["v"], amp=amp)))
        model.train()
        return float(np.mean(tot))

    def save(tag_best=False):
        state = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val": best_val,
            "hparams": model.hparams,
            "prior": cfg["prior"],
            "train_seconds": elapsed_prev + time.time() - t_start,
        }
        _save_atomic(state, last)
        if tag_best:
            _save_atomic(state, ckpt_dir / "best.pt")

    log_path = results_dir / "train_log.csv"
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="") if is_main else None
    writer = csv.writer(log_file) if is_main else None
    if is_main and new_log:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_h"])

    t_start = time.time()
    t_ckpt = t_start
    budget_s = float(tcfg["max_hours"]) * 3600
    ckpt_s = float(tcfg["checkpoint_minutes"]) * 60
    running = []
    model.train()
    for batch in loader:
        batch = _to(batch, device)
        loss = net(batch, batch["v"], amp=amp)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        running.append(loss.item())

        now = time.time()
        out_of_time = now - t_start > budget_s
        if world > 1:  # every rank must stop at the same step, so agree on the clock
            if step % 50 == 0:
                flag = torch.tensor(float(out_of_time), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                out_of_time = bool(flag.item())
            else:
                out_of_time = False
        done = step >= max_steps or out_of_time
        if is_main and (step % int(tcfg["val_every"]) == 0 or done):
            val = validate()
            improved = val < best_val
            best_val = min(best_val, val)
            writer.writerow(
                [
                    step,
                    f"{np.mean(running):.5f}",
                    f"{val:.5f}",
                    f"{sched.get_last_lr()[0]:.3e}",
                    f"{(elapsed_prev + now - t_start) / 3600:.3f}",
                ]
            )
            log_file.flush()
            log(
                f"step {step} train {np.mean(running):.4f} val {val:.4f}"
                + (" *" if improved else "")
            )
            running = []
            if improved:
                save(tag_best=True)
        if is_main and (now - t_ckpt > ckpt_s or done):
            save()
            t_ckpt = now
        if done:
            if is_main and out_of_time:
                log(f"time budget reached at step {step}; rerun the same command to resume")
            break
    if is_main:
        log_file.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"step": step, "best_val": best_val}


def load_model(path: str | Path, device="cpu") -> tuple[NPE, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = NPE.from_config(state["hparams"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state
