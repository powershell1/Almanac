"""Evaluate a trained NPE on the frozen test set and write the kill-test decision.

    python scripts/evaluate.py --config configs/killtest.yaml --root /kaggle/working \
        --set paths.test_dir=/kaggle/input/gdr4-test-set/shards/test \
              paths.reference_dir=/kaggle/input/gdr4-reference/shards/reference
Outputs <results_dir>/eval/{metrics.json, decision.md, per_star_*.npz}.
"""

import torch

from gdr4.cli import base_parser, parse
from gdr4.eval.pipeline import evaluate
from gdr4.nn.train import load_model
from gdr4.utils import resolve, write_run_info


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", default="best.pt", help="file name inside checkpoint_dir")
    args, cfg = parse(p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_model(resolve(cfg, "checkpoint_dir", args.root) / args.checkpoint, device)
    out = resolve(cfg, "results_dir", args.root) / "eval"
    write_run_info(
        out,
        cfg,
        {"stage": "evaluate", "checkpoint_step": state["step"], "checkpoint": args.checkpoint},
    )
    m = evaluate(
        cfg,
        model,
        resolve(cfg, "test_dir", args.root),
        resolve(cfg, "reference_dir", args.root),
        out,
        device,
    )
    print((out / "decision.md").read_text())
    print("nominal:", {k: m["nominal"][k] for k in ("hpd_coverage", "tarp_coverage")})


if __name__ == "__main__":
    main()
