"""Streaming batches from ragged shards for training, and conversion to torch batches."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from gdr4.data.shards import gather_padded, n_stars, read_shard
from gdr4.sim.priors import Prior

MODEL_KEYS = ("t_d", "psi_rad", "plx_factor", "sigma_mas", "resid5_mas")
STAR_KEYS = ("n_obs", "chi2_5p", "period_d", "ecc", "mean_anom0_rad")


def to_torch_batch(padded: dict, prior: Prior | None, device="cpu") -> dict[str, torch.Tensor]:
    """Padded numpy batch -> dict of tensors; adds target ``v`` if the truth is present."""
    out = {k: torch.as_tensor(np.asarray(padded[k], np.float32)) for k in MODEL_KEYS}
    out["mask"] = torch.as_tensor(np.asarray(padded["mask"], bool))
    out["n_obs"] = torch.as_tensor(np.asarray(padded["n_obs"], np.float32))
    out["chi2_5p"] = torch.as_tensor(np.asarray(padded["chi2_5p"], np.float32))
    if prior is not None and "period_d" in padded:
        v = prior.to_unconstrained(padded["period_d"], padded["ecc"], padded["mean_anom0_rad"])
        out["v"] = torch.as_tensor(v.astype(np.float32))
    return {k: t.to(device, non_blocking=True) for k, t in out.items()}


def batches_from_shard(
    shard: dict, batch_size: int, prior: Prior | None, order: np.ndarray | None = None
) -> Iterator[dict]:
    n = n_stars(shard)
    order = np.arange(n) if order is None else order
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield to_torch_batch(gather_padded(shard, idx), prior)


class ShardStream(IterableDataset):
    """Infinite stream of training batches.

    Shards are split across (DDP rank, DataLoader worker) streams; each stream shuffles its
    shards and the stars inside each shard every epoch. Batches from one shard are i.i.d.
    because shards are i.i.d. simulations.
    """

    def __init__(
        self,
        root: str | Path,
        indices: list[int],
        batch_size: int,
        prior: Prior,
        seed: int,
        rank: int = 0,
        world: int = 1,
    ):
        self.root, self.indices = Path(root), list(indices)
        self.batch_size, self.prior = batch_size, prior
        self.seed, self.rank, self.world = seed, rank, world

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        k, total = self.rank * nw + wid, self.world * nw
        if len(self.indices) >= total:
            mine, stride, offset = self.indices[k::total], 1, 0
        else:  # fewer shards than streams: share shards, split their batches
            mine, stride, offset = self.indices, total, k
        rng = np.random.default_rng([self.seed, k])
        while True:
            for s in rng.permutation(mine):
                shard = read_shard(self.root, int(s))
                order = rng.permutation(n_stars(shard))
                n_batches = len(order) // self.batch_size
                for j in range(offset, n_batches, stride):
                    idx = np.sort(order[j * self.batch_size : (j + 1) * self.batch_size])
                    yield to_torch_batch(gather_padded(shard, idx), self.prior)
