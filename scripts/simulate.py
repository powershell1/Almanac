"""Generate training shards (resumable). Run in CPU-only Kaggle sessions.

    python scripts/simulate.py --config configs/killtest.yaml --root /kaggle/working \
        --set paths.scan_cache=/kaggle/input/gdr4-scan-cache/scan_cache_dr4_gaiamock.npz
    # optionally a subset of shards, e.g. to split work across sessions:
    python scripts/simulate.py ... --shards 0-19
"""

import time

from gdr4.cli import base_parser, parse
from gdr4.data.shards import generate_shards, list_complete, to_ragged
from gdr4.sim.noise import get_noise_model
from gdr4.sim.priors import Prior
from gdr4.sim.scan import ScanCache
from gdr4.sim.simulate import simulate_batch
from gdr4.utils import git_hash, resolve


def parse_range(text, n_total):
    if not text:
        return range(n_total)
    lo, _, hi = text.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def check_scan_source(cfg, scan):
    want = cfg["scan"]["source"]
    if not scan.source.startswith(want):
        raise SystemExit(f"config wants scan.source={want} but cache is {scan.source}")


def main():
    p = base_parser(__doc__)
    p.add_argument("--shards", default="", help="inclusive range like 0-19 (default: all)")
    args, cfg = parse(p)
    sim = cfg["simulation"]
    scan = ScanCache.load(resolve(cfg, "scan_cache", args.root))
    check_scan_source(cfg, scan)
    prior = Prior.from_config(cfg["prior"])
    noise = get_noise_model(sim["noise_model"])
    n_shards = -(-int(sim["n_train"]) // int(sim["shard_size"]))
    out = resolve(cfg, "shards_dir", args.root)

    def make(rng):
        return to_ragged(
            simulate_batch(
                rng, int(sim["shard_size"]), prior, scan, noise, drop_frac=float(sim["drop_frac"])
            )
        )

    meta = {
        "kind": "train",
        "scan_source": scan.source,
        "noise_model": noise.name,
        "prior": cfg["prior"],
        "git_hash": git_hash(),
    }
    t0 = time.time()
    done = generate_shards(
        out, parse_range(args.shards, n_shards), make, entropy=int(sim["train_entropy"]), meta=meta
    )
    print(
        f"wrote {len(done)} shards in {time.time() - t0:.0f}s; "
        f"complete: {len(list_complete(out))}/{n_shards}"
    )


if __name__ == "__main__":
    main()
