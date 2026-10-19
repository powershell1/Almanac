"""Config loading and run bookkeeping (config + git hash written next to every result)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply ``key.sub=value`` overrides (values parsed as YAML)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    cfg["_config_path"] = str(path)
    return cfg


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
        ).stdout
        return out + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resolve(cfg: dict, key: str, root: str | Path | None = None) -> Path:
    """Path from ``cfg['paths'][key]``; relative paths are taken relative to ``root``."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() or root is None else Path(root) / p


def write_run_info(out_dir: str | Path, cfg: dict, extra: dict | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = copy.deepcopy(cfg)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(clean, sort_keys=False))
    info = {
        "git_hash": git_hash(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "python": sys.version,
    }
    info.update(extra or {})
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
