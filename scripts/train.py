"""Train the NPE (resumable). On Kaggle use a GPU session; rerun the same command to resume.

    python scripts/train.py --config configs/killtest.yaml --root /kaggle/working \
        --set paths.shards_dir=/kaggle/input/gdr4-train-shards/shards/train
    torchrun --nproc_per_node 2 scripts/train.py ...     # both T4s
"""

import os

from gdr4.cli import base_parser, parse
from gdr4.nn.train import train
from gdr4.utils import resolve, write_run_info


def main():
    args, cfg = parse(base_parser(__doc__))
    results = resolve(cfg, "results_dir", args.root)
    if int(os.environ.get("RANK", "0")) == 0:
        write_run_info(results, cfg, {"stage": "train"})
    out = train(
        cfg,
        resolve(cfg, "shards_dir", args.root),
        resolve(cfg, "checkpoint_dir", args.root),
        results,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(out)


if __name__ == "__main__":
    main()
