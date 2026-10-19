# Kaggle runbook (Phase 1 kill test)

Each step is one notebook in `notebooks/kaggle/`. Notebooks only call `scripts/`. Kaggle
mounts inputs read-only under `/kaggle/input/<slug>/`; outputs go to `/kaggle/working`
(20 GB cap) and become reusable inputs after **Save Version → Save & Run All**.
Adjust the `/kaggle/input/...` paths at the top of each notebook to your own slugs.

## 0. Before anything (you, once)

1. Push this repo to GitHub (the notebooks `git clone` it). If the repo is private, add a
   Kaggle Secret named `GITHUB_TOKEN` (a fine-grained, read-only token for this repo).
   Alternative: upload the repo folder as a Kaggle Dataset named `almanac-src`.
2. Download `healpix_scans.zip` from the Caltech Box link in the gaiamock README
   (984 MB unzipped; gaiamock commit `dd30fdbf787eb96878734605ac077ac69bf28c84`).
   Upload it as a **private** Kaggle Dataset (e.g. `gaiamock-healpix-scans`) so that
   `/kaggle/input/gaiamock-healpix-scans/healpix_scans/healpix_64_0.fits` exists.
3. Confirm the open decisions in `configs/killtest.yaml` (priors, S/N and alias definitions)
   and the target ISEF year. Write the Research Plan **before** step 4 below (ISEF rules).

## 1. `01_env_check` (GPU, ~5 min)

Prints Python/GPU/package versions to `env_report.json` and runs the fast tests.
Copy the versions into `pyproject.toml` and commit.

## 2. `02_scan_cache_cpu` (CPU, ~10–20 min)

Reads the 49,152 FITS files once and writes `data_cache/scan_cache_dr4_gaiamock.npz`.

## 3. `03_simulate_cpu` (CPU)

Writes the frozen test set (`shards/test/<noise>/`, 2,000 stars × 6 noise models) and
2,000,000 training stars (`shards/train/`, 40 shards, ~4.5 GB). Locally the simulator runs
at ~12k stars/s on one core, so expect well under an hour. Resumable.
**This is the first counted data collection: record the date in `logs/research_log.md`.**

## 4. `04_reference_cpu` (CPU, ~4 h for 200 stars)

Brute-force reference posteriors (prior sampling + defensive importance sampling).
Each star writes `shards/reference/star_XXXXX.npz` with its ESS; rerunning skips done stars.

## 5. `05_train_gpu` (2× T4, ≤ 11 h per session)

`torchrun` over both GPUs. Checkpoints every 30 min and at the time budget. To continue,
attach the previous output of this notebook and rerun (it copies `checkpoints/` back and
resumes from `last.pt`). Budget for the kill test: ≤ 12 GPU-hours.

## 6. `06_evaluate_gpu` (GPU, < 1 h)

Writes `results/killtest_v1/eval/metrics.json`, `decision.md` and the figures. Download
`results/killtest_v1/` (small) into the repo and commit it. Record the decision and the
evidence in `logs/research_log.md`.

## Local testing (this laptop)

```bash
source .env/miniforge3/bin/activate .env/gdr4
pytest -q                                       # all tests, ~30 s
python scripts/build_scan_cache.py --config configs/smoke.yaml --root runs_local
python scripts/simulate.py        --config configs/smoke.yaml --root runs_local
python scripts/make_test_set.py   --config configs/smoke.yaml --root runs_local
python scripts/reference_posteriors.py --config configs/smoke.yaml --root runs_local --n-jobs 8
python scripts/train.py           --config configs/smoke.yaml --root runs_local
python scripts/evaluate.py        --config configs/smoke.yaml --root runs_local
```

The smoke configs use the **toy** scanning law; their outputs are pipeline checks only.
