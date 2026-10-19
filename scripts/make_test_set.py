"""Generate the frozen test set: the same stars under each noise model in the config.

The stars (sky position, G, theta) depend only on the seeds file; each noise model gets its
own noise realisation. Output: <test_dir>/<noise_model>/shard_00000.
"""

import numpy as np

from gdr4.cli import base_parser, parse
from gdr4.data.shards import is_complete, to_ragged, write_shard
from gdr4.sim.noise import get_noise_model
from gdr4.sim.priors import Prior
from gdr4.sim.scan import ScanCache
from gdr4.sim.simulate import simulate_batch
from gdr4.utils import git_hash, load_yaml, resolve


def main():
    args, cfg = parse(base_parser(__doc__))
    seeds = load_yaml(cfg["test_set"]["seeds_file"])
    scan = ScanCache.load(resolve(cfg, "scan_cache", args.root))
    if not scan.source.startswith(cfg["scan"]["source"]):
        raise SystemExit(f"cache source {scan.source} does not match config")
    prior = Prior.from_config(cfg["prior"])
    root = resolve(cfg, "test_dir", args.root)
    for name in cfg["test_set"]["noise_models"]:
        out = root / name
        if is_complete(out, 0):
            print(f"{out} exists, skipping")
            continue
        rng = np.random.default_rng(np.random.SeedSequence(int(seeds["test_entropy"])))
        batch = simulate_batch(
            rng,
            int(seeds["n_test"]),
            prior,
            scan,
            get_noise_model(name),
            drop_frac=float(cfg["simulation"]["drop_frac"]),
        )
        meta = {
            "kind": "test",
            "noise_model": name,
            "scan_source": scan.source,
            "seeds_file": cfg["test_set"]["seeds_file"],
            "git_hash": git_hash(),
        }
        write_shard(out, 0, to_ragged(batch), meta)
        print(f"wrote {out} ({seeds['n_test']} stars)")


if __name__ == "__main__":
    main()
