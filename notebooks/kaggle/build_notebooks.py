"""Generate the thin Kaggle notebooks in this folder (they contain no logic, only commands).

    python notebooks/kaggle/build_notebooks.py

Edit this file, not the .ipynb files, and re-run it.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

SETUP = r"""# --- Setup: get the code (no project logic in notebooks; everything lives in src/ and scripts/) ---
import os
import subprocess
import sys

REPO_URL = "https://github.com/powershell1/Almanac.git"
BRANCH = "master"
SRC = "/tmp/Almanac"
if os.path.exists("/kaggle/input/almanac-src/pyproject.toml"):
    SRC = "/kaggle/input/almanac-src"          # offline option: repo uploaded as a Dataset
elif not os.path.exists(SRC):
    url = REPO_URL
    try:                                       # private repo: add a Kaggle Secret GITHUB_TOKEN
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
        url = REPO_URL.replace("https://", f"https://{tok}@")
    except Exception:
        pass
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, SRC], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zuko==1.6.0"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", SRC], check=True)
os.chdir(SRC)
print(subprocess.run(["git", "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout)
WORK = "/kaggle/working"
CFG = "configs/killtest.yaml"
"""

NOTEBOOKS = {
    "01_env_check.ipynb": [
        (
            "md",
            "# 01 · Environment check (GPU session)\n\nPhase 1 step 1. Record the Python "
            "version, GPU and package versions, then pin them in `pyproject.toml`.",
        ),
        ("code", SETUP),
        ("code", "!python scripts/env_check.py --out /kaggle/working/env_report.json"),
        ("code", "!python -m pytest -q -m 'not slow' tests"),
        (
            "md",
            "Time one training step and posterior sampling on this GPU. If the projected "
            "steps per session are far below `training.max_steps`, adjust the plan before "
            "the counted run (fewer steps, smaller batch, or precomputed physics features).",
        ),
        ("code", "!python scripts/benchmark.py --config {CFG}"),
    ],
    "02_scan_cache_cpu.ipynb": [
        (
            "md",
            "# 02 · Scan-geometry cache (CPU session)\n\nInput: a private Kaggle Dataset "
            "with gaiamock's `healpix_scans/` folder (49,152 FITS files, 984 MB unzipped, "
            "from the Caltech Box link in the gaiamock README; you upload it yourself).\n\n"
            "Output: `data_cache/scan_cache_dr4_gaiamock.npz`. After *Save Version*, attach "
            "this notebook's output to notebooks 03.",
        ),
        ("code", SETUP),
        (
            "code",
            'SCANS = "/kaggle/input/gaiamock-healpix-scans"   # must contain healpix_scans/\n'
            "!python scripts/build_scan_cache.py --config {CFG} --root {WORK} "
            "--gaiamock-dir {SCANS} --n-jobs 4",
        ),
    ],
    "03_simulate_cpu.ipynb": [
        (
            "md",
            "# 03 · Frozen test set + training simulations (CPU session)\n\nResumable: if "
            "the session dies, attach this notebook's previous output, copy it back "
            "into /kaggle/working and rerun (finished shards are skipped).\n\nOutput: "
            "`shards/test/<noise>/` and `shards/train/` (~4.5 GB for 2M stars).",
        ),
        ("code", SETUP),
        (
            "code",
            'CACHE = "/kaggle/input/gdr4-02-scan-cache/data_cache/scan_cache_dr4_gaiamock.npz"\n'
            "# Optional resume: copy shards from a previous version of this notebook\n"
            'PREV = "/kaggle/input/gdr4-03-simulate/shards"\n'
            "if os.path.exists(PREV):\n"
            '    subprocess.run(["cp", "-rn", PREV, WORK], check=True)',
        ),
        (
            "code",
            "!python scripts/make_test_set.py --config {CFG} --root {WORK} "
            "--set paths.scan_cache={CACHE}",
        ),
        (
            "code",
            "!python scripts/simulate.py --config {CFG} --root {WORK} "
            "--set paths.scan_cache={CACHE}",
        ),
        ("code", "!du -sh /kaggle/working/shards/*"),
    ],
    "04_reference_cpu.ipynb": [
        (
            "md",
            "# 04 · Brute-force reference posteriors (CPU session)\n\nRuns the first "
            "`n_reference` test stars (configs/test_seeds.yaml) on 4 cores. Split long "
            'runs across sessions with `STARS = "0-99"`, then `"100-199"`.',
        ),
        ("code", SETUP),
        (
            "code",
            'TEST = "/kaggle/input/gdr4-03-simulate/shards/test"\n'
            'PREV = "/kaggle/input/gdr4-04-reference/shards/reference"  # optional resume\n'
            "if os.path.exists(PREV):\n"
            '    os.makedirs(f"{WORK}/shards", exist_ok=True)\n'
            '    subprocess.run(["cp", "-rn", PREV, f"{WORK}/shards/"], check=True)\n'
            'STARS = "0-199"',
        ),
        (
            "code",
            "!python scripts/reference_posteriors.py --config {CFG} --root {WORK} "
            "--n-jobs 4 --stars {STARS} --set paths.test_dir={TEST}",
        ),
    ],
    "05_train_gpu.ipynb": [
        (
            "md",
            "# 05 · Train the NPE (GPU session, 2× T4)\n\nStops cleanly after "
            "`training.max_hours` (11 h). To continue, attach this notebook's previous "
            "output and rerun: the checkpoint is copied back and training resumes.\n\n"
            "Record the date of the first *counted* run in `logs/research_log.md`.",
        ),
        ("code", SETUP),
        (
            "code",
            'SHARDS = "/kaggle/input/gdr4-03-simulate/shards/train"\n'
            'PREV = "/kaggle/input/gdr4-05-train"          # optional resume\n'
            'for sub in ("checkpoints", "results"):\n'
            '    if os.path.exists(f"{PREV}/{sub}"):\n'
            '        subprocess.run(["cp", "-rn", f"{PREV}/{sub}", WORK], check=True)\n'
            "!nvidia-smi --query-gpu=name,memory.total --format=csv",
        ),
        (
            "code",
            "!torchrun --nproc_per_node 2 scripts/train.py --config {CFG} --root {WORK} "
            "--set paths.shards_dir={SHARDS}",
        ),
        ("code", "!tail -5 /kaggle/working/results/killtest_v1/train_log.csv"),
    ],
    "06_evaluate_gpu.ipynb": [
        (
            "md",
            "# 06 · Evaluate and decide (GPU session)\n\nWrites `results/killtest_v1/eval/"
            "metrics.json` and `decision.md`, then the figures. Copy the whole "
            "`results/killtest_v1/` folder back into the repo and commit it.",
        ),
        ("code", SETUP),
        (
            "code",
            'TEST = "/kaggle/input/gdr4-03-simulate/shards/test"\n'
            'REF = "/kaggle/input/gdr4-04-reference/shards/reference"\n'
            'TRAINED = "/kaggle/input/gdr4-05-train"\n'
            'subprocess.run(["cp", "-rn", f"{TRAINED}/checkpoints", f"{TRAINED}/results", '
            "WORK], check=True)",
        ),
        (
            "code",
            "!python scripts/evaluate.py --config {CFG} --root {WORK} "
            "--set paths.test_dir={TEST} paths.reference_dir={REF}",
        ),
        ("code", "!python scripts/figures/plot_killtest.py --config {CFG} --root {WORK}"),
    ],
}


def to_nb(cells):
    out = []
    for kind, src in cells:
        cell = {
            "cell_type": "markdown" if kind == "md" else "code",
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        (HERE / name).write_text(json.dumps(to_nb(cells), indent=1) + "\n")
        print("wrote", HERE / name)
