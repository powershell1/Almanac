"""Shard writer/reader. One shard = one directory of ``.npy`` arrays + ``meta.json``.

Per-transit arrays are stored *ragged* (all stars concatenated, plus ``offsets`` of length
n + 1), so storage scales with the real number of transits instead of the longest star.

Writes are atomic (temp directory, then rename) and a shard counts as done only once its
``_COMPLETE`` marker exists, so generation can be killed and resumed at any point.
Seeds: shard ``i`` uses ``SeedSequence(entropy, spawn_key=(i,))``, so any shard can be
regenerated independently and bit-for-bit.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

COMPLETE = "_COMPLETE"
PER_TRANSIT_KEYS = ("t_d", "psi_rad", "plx_factor", "w_mas", "sigma_mas", "resid5_mas")


def to_ragged(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Padded batch (with ``mask``) -> ragged arrays + ``offsets``."""
    mask = batch["mask"]
    n_obs = mask.sum(axis=1)
    out = {k: v for k, v in batch.items() if k not in PER_TRANSIT_KEYS and k != "mask"}
    out["offsets"] = np.concatenate([[0], np.cumsum(n_obs)]).astype(np.int64)
    for k in PER_TRANSIT_KEYS:
        out[k] = batch[k][mask]
    return out


def gather_padded(
    shard: dict[str, np.ndarray],
    idx: np.ndarray,
    n_max: int | None = None,
    keys: Iterable[str] = PER_TRANSIT_KEYS,
) -> dict[str, np.ndarray]:
    """Rows ``idx`` of a ragged shard as padded arrays (plus ``mask``) and per-star values."""
    idx = np.asarray(idx)
    off = shard["offsets"]
    start, stop = off[idx], off[idx + 1]
    n_obs = stop - start
    n_max = int(n_obs.max()) if n_max is None else n_max
    pos = np.arange(n_max)
    mask = pos[None, :] < n_obs[:, None]
    flat = np.where(mask, start[:, None] + pos[None, :], 0)
    out = {"mask": mask}
    for k in keys:
        out[k] = np.where(mask, np.asarray(shard[k])[flat], 0)
    for k, v in shard.items():
        if k not in PER_TRANSIT_KEYS and k != "offsets":
            out[k] = np.asarray(v[idx])
    return out


def n_stars(shard: dict[str, np.ndarray]) -> int:
    return len(shard["offsets"]) - 1


def shard_path(root: str | Path, index: int) -> Path:
    return Path(root) / f"shard_{index:05d}"


def is_complete(root: str | Path, index: int) -> bool:
    return (shard_path(root, index) / COMPLETE).exists()


def shard_rng(entropy: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=entropy, spawn_key=(index,)))


def write_shard(root: str | Path, index: int, arrays: dict[str, np.ndarray], meta: dict) -> Path:
    final = shard_path(root, index)
    tmp = final.with_name(final.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for k, v in arrays.items():
        np.save(tmp / f"{k}.npy", np.ascontiguousarray(v))
    n = len(arrays["offsets"]) - 1 if "offsets" in arrays else len(next(iter(arrays.values())))
    meta = dict(meta, n=int(n), keys=sorted(arrays))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (tmp / COMPLETE).touch()
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final


def read_shard(
    root: str | Path, index: int, mmap: bool = True, keys: Iterable[str] | None = None
) -> dict[str, np.ndarray]:
    p = shard_path(root, index)
    if not (p / COMPLETE).exists():
        raise FileNotFoundError(f"incomplete or missing shard {p}")
    meta = json.loads((p / "meta.json").read_text())
    keys = meta["keys"] if keys is None else keys
    return {k: np.load(p / f"{k}.npy", mmap_mode="r" if mmap else None) for k in keys}


def read_meta(root: str | Path, index: int) -> dict:
    return json.loads((shard_path(root, index) / "meta.json").read_text())


def list_complete(root: str | Path) -> list[int]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.glob("shard_?????")):
        if (p / COMPLETE).exists():
            out.append(int(p.name.split("_")[1]))
    return out


def generate_shards(
    root: str | Path,
    indices: Iterable[int],
    make: Callable[[np.random.Generator], dict[str, np.ndarray]],
    entropy: int,
    meta: dict,
    log: Callable[[str], None] = print,
) -> list[int]:
    """Create every shard in ``indices`` that is not complete yet. Returns indices written."""
    written = []
    for i in indices:
        if is_complete(root, i):
            continue
        arrays = make(shard_rng(entropy, i))
        write_shard(root, i, arrays, dict(meta, shard_index=i, entropy=str(entropy)))
        written.append(i)
        log(f"wrote {shard_path(root, i)}")
    return written
