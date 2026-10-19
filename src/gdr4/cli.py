"""Shared argument parsing for scripts/."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdr4.utils import load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/killtest.yaml")
    p.add_argument(
        "--root", default=".", help="base dir for relative paths (Kaggle: /kaggle/working)"
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides, e.g. paths.scan_cache=/kaggle/input/x/cache.npz",
    )
    return p


def parse(p: argparse.ArgumentParser):
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    args.root = Path(args.root)
    return args, cfg
