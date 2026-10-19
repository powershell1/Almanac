# CLAUDE.md — Gaia DR4 Amortized Orbit Posteriors

## Project in one paragraph

Amortized neural posterior estimation (NPE) of companion orbits from **Gaia DR4 epoch astrometry**. A network is trained once on simulated Gaia data. It then returns a calibrated posterior over a star's companion orbit in milliseconds, instead of minutes per star with nested sampling or MCMC. Target: ISEF 2027 (Physics & Astronomy) plus an arXiv preprint. This is a **student research project**: read "Integrity & AI use" before writing any code.

### The claimed contribution (keep the scope exactly this)

1. Full, calibrated posteriors over the nonlinear orbit parameters (P, e, T_peri) from 1-D along-scan epoch astrometry. The 9 linear parameters are handled analytically.
2. Correct representation of **period aliases** caused by Gaia's scanning law.
3. Calibration that **survives noise-model misspecification**. This is the documented weakness of the closest prior work.
4. First application to **real DR4 data**, validated against DPAC orbits and RV-confirmed systems.

**Not claimed:**
- Amortized *detection*. This is already done (Nuthakki & Nuthakki, NeurIPS 2026 Sim2Science workshop); we use it as a baseline.
- A new ML architecture. NPE with normalizing flows is a standard tool; the novelty is the Gaia-specific adaptation and the real-data results.

---

## Current status (as of 2026-10-20)

- Planning complete. **No code, no data, no results exist yet.**
- The direction was chosen after comparing 12 candidate projects.
- Novelty search on 2026-10-20 found no published amortized orbit-posterior work for Gaia epoch astrometry. Re-check monthly (see "Novelty watch").
- Nothing belongs in `results/` unless a script in this repo generated it.

---

## Key dates and external constraints

| Date | Event |
|---|---|
| 2026-11-25 | **Kill-test decision gate** (continue, or switch to the fallback contribution) |
| 2026-12-02 | Gaia DR4 public release (expect 60+ DPAC papers the same day; check them for overlap) |
| TBD, Feb–Apr 2027 | Affiliated fair (fill in name and date) |
| 2027-05-08 to 05-14 | Regeneron ISEF, Los Angeles |

**ISEF rules (2026–27):**
- At most 12 continuous months of research, none before January 2026.
- The project start date is the first data collection. Treat the first *counted* simulation or training run as the start, and record it in `logs/research_log.md`.
- The Research Plan and required forms must be completed before that date. Check the ISEF Rules Wizard.
- **OPEN DECISION:** confirm the target year. If it is ISEF 2028, do not start counted experiments before January 2027.

**Compute: Kaggle free tier**
- GPU is 2× T4. The P100 was retired in September 2026. The two T4s are **separate 16 GB devices**, so a model must fit in 16 GB.
- About 30 GPU-hours per week; 12-hour sessions.
- 4 CPU cores, about 29 GB RAM, 20 GB persistent `/kaggle/working`.
- Run simulations in **CPU-only sessions**. Confirm on the account page that these don't use GPU quota.
- T4 FP64 is weak. Do float64 physics on the CPU, or keep it light on the GPU.
- Checkpoint every 30 minutes. Every job must be resumable. Persist shards and checkpoints as Kaggle Datasets.

---

## Science model

Verify all sign and convention choices against gaiamock and the DR4 data model before relying on them.

For transit k (with CCD samples combined per transit):

```
w_k = (Δα* + μα*·t_k)·sin ψ_k + (Δδ + μδ·t_k)·cos ψ_k + ϖ·f_ϖ,k
      + (B·X_k + G·Y_k)·sin ψ_k + (A·X_k + F·Y_k)·cos ψ_k + ε_k

X = cos E − e,   Y = √(1−e²)·sin E,   E − e·sin E = 2π(t_k − T_p)/P
```

- `ψ` is the scan angle measured from local ICRS north.
- `f_ϖ` is the along-scan parallax factor.
- `t` is time relative to the DR4 reference epoch J2017.5.
- **Nonlinear** θ_nl = (P, e, T_p). These are learned by the flow.
- **Linear** θ_lin = (Δα*, Δδ, ϖ, μα*, μδ, A, B, F, G). Given θ_nl, the model is `w = M(θ_nl)·θ_lin + ε`. With a Gaussian prior on θ_lin:
  - p(θ_lin | θ_nl, w) is Gaussian, and p(w | θ_nl) is analytic.
  - This is the same trick The Joker uses for radial velocities.
- **Consistency rule (important):** the simulator must draw θ_lin from *the same Gaussian prior* used in the analytic step. Otherwise q(θ_nl | w) and the analytic conditional do not combine into the correct joint posterior.
  - Physical parameters (a₀, i, ω, Ω, companion mass) are derived *after* sampling.
  - Document the physical prior this implies, and check whether it needs reweighting.
- The photocentre is not the star when the companion is luminous. For planets and brown dwarfs the photocentre is effectively the star. **The kill test uses dark companions only.**

---

## Architecture

```
priors → simulator (gaiamock wrapper + noise models) → shards
       → encoder + flow training → posterior samples of θ_nl
       → analytic θ_lin conditional → evaluation / real-data inference
```

### Planned repo layout

```
gaia-dr4-npe/
  CLAUDE.md  README.md  pyproject.toml
  configs/                 # YAML run configs; test_seeds.yaml (FROZEN)
  src/gdr4/
    model/kepler.py        # vectorized Kepler solver (float64)
    model/astrometry.py    # design matrix M(θ_nl), forward model
    model/linear.py        # weighted least squares, analytic marginal likelihood
    sim/priors.py          # prior definitions (single source of truth)
    sim/simulate.py        # gaiamock wrapper → per-transit (t, ψ, f_ϖ, w, σ)
    sim/noise.py           # nominal + misspecified noise models
    data/shards.py         # shard writer/reader, resumable generation
    data/dr4.py            # pre-release XML loader now; archive/DataLink after Dec 2
    nn/encoder.py          # transformer over transits
    nn/flow.py             # conditional neural spline flow over θ_nl
    nn/train.py            # training loop, checkpointing, resume
    reference/bruteforce.py# prior sampling × analytic marginal likelihood
    reference/kima_run.py  # kima cross-check
    baselines/             # cascade/Laplace, MLP, GBDT, Gaussian head, detector
    eval/calibration.py    # coverage, SBC, TARP
    eval/c2st.py           # classifier two-sample test vs reference
    eval/aliases.py        # alias failure maps
  notebooks/kaggle/        # thin wrappers that import src/; no logic here
  scripts/                 # CLI entry points; scripts/figures/ makes every figure
  tests/
  logs/research_log.md     # ISEF logbook (dated entries)
  logs/ai_use_log.md       # AI-assistance record (see Integrity)
  results/<run_id>/        # config, git hash, metrics, figures (script-generated only)
```

### Model

**Encoder:** a transformer over transits (a set or sequence, with a time embedding).
- Per-transit features: `t`, `sin ψ`, `cos ψ`, `f_ϖ`, the residual after a 5-parameter single-star fit, and `σ`.
- Compare two input variants:
  - (a) raw residual and σ;
  - (b) σ-normalized residuals.
- The prior work's advantage over the classical cascade shrank under a corrected noise model because its network consumed noise estimates as input features. Test this directly.

**Flow:** a conditional neural spline flow over (log P, transformed e, orbital phase).
- Orbital phase is circular. Use a circular representation or flow, not a raw T_p.

**Size:** 5–20 M parameters.

### Reference posteriors (ground truth for calibration)

1. **Brute force.**
   - Draw 10⁶–10⁷ samples of (P, e, T_p) from the prior and weight each by the analytic marginal likelihood (vectorized, on GPU).
   - This is exact up to sampling noise.
   - It fails when the posterior is very sharp: monitor the effective sample size and flag low-ESS stars.
2. **kima** (diffusive nested sampling) on a subset, as an independent cross-check and as the speed baseline.
3. **Octofitter** (Julia), optional.

### Baselines

Give every baseline the same compute budget, and report both GPU-hours and CPU-hours.
- gaiamock cascade fit (simulated annealing plus a Hessian/Laplace approximation)
- MLP regressor (point estimates) and GBDT on periodogram features
- NPE with a Gaussian head (no flow)
- NPE over all 12 parameters with no analytic step (ablation)
- The amortized 3-class detector from Nuthakki & Nuthakki, re-implemented as a detection baseline

### Evaluation

- Coverage at 50/68/90% (SBC, TARP).
- C2ST against the reference posteriors.
- Period recovery.
- Alias failure maps over period and scan geometry.
- Wall-clock time per star.
- **Misspecification suite:**
  - Train on noise model A, test on B.
  - Inflated or deflated σ.
  - Outliers.
  - Scan-angle-dependent spurious signals (3rd, 5th and 7th harmonics).
  - Photocentre dilution from luminous companions.

---

## Data

### Simulations
- Use gaiamock (El-Badry et al. 2024, arXiv:2411.00088). Get the code from the paper's link and **record the commit hash**.
- Kill-test priors: G = 10–13, P = 30–2000 d (the DR4 baseline is about 1977 d), dark companions.

### Storage
- Write shards to `/kaggle/working` (20 GB cap) and persist them as Kaggle Datasets.
- **Never store times in float16.** At t ≈ 2000 d its resolution is about 1–2 days, which destroys phase information. Use float32 for times and abscissae, and float64 for all physics computations.
- Size estimate: 2 M stars × about 100 transits × 6 features × 4 B ≈ 4.8 GB.

### Frozen test set
- Seeds live in `configs/test_seeds.yaml`.
- Never use them for training, tuning or model selection.

### Real data
- **Now:** the DR4 pre-release (12 sources, XML, with the draft data model).
  - Use it **only** to build and test the loader.
  - Rows are per-CCD samples and must be combined per transit.
  - Helpful tools: `esa/gaia-supdate`, `esa/gaia-jupyter-notebooks` (data-release-4-tutorials), `exogaia`.
- **After 2026-12-02:** fetch `epoch_astrometry` via the archive/DataLink for a bounded volume (for example G < 13 within about 100–200 pc). **Never attempt a bulk download**: the full table is about 62 TB.

### Validation targets
- DPAC two-body orbits.
- The `spurious_signal` table.
- A **frozen** list of RV-confirmed holdout systems, never tuned on.
- RV-quiet stars, for false-alarm calibration.

### Licence
Gaia data is CC BY-NC 3.0 IGO. Credit "ESA/Gaia/DPAC".

---

## Next immediate goals: Phase 1 kill test (decide by 2026-11-25)

Work in this order:

1. **Environment.**
   - Set up a Kaggle notebook with 2× T4.
   - Check the Python version (`python --version`) and the GPU.
   - Install dependencies and pin exact versions in `pyproject.toml`.
2. **Forward model and tests.**
   - Vectorized float64 Kepler solver, tested to 1e-12 for e ∈ [0, 0.95].
   - Design matrix and weighted least squares.
   - Test: reproduce gaiamock abscissae for identical inputs.
   - Test: recover the injected θ_lin at high S/N.
3. **Simulator.**
   - Write the simulator wrapper, priors and noise models.
   - Freeze a 2,000-star test set.
   - Generate 2 M training simulations in CPU sessions.
4. **Reference posteriors.**
   - Brute force for 100–200 test stars.
   - kima for about 20 of them.
5. **Training.**
   - Train encoder plus flow in ≤ 12 GPU-hours, with checkpoints.
6. **Evaluate against the thresholds below.**

**Continue only if all of these hold:**
- Coverage error ≤ 5 percentage points at the 50%, 68% and 90% levels.
- P recovered within 10% for at least 90% of stars at S/N > 6.
- Inference under 50 ms per star.
- C2ST ≤ 0.65 on at least 70% of the reference stars.

**Abandon and switch to the fallback if either holds:**
- Coverage error is still above 10 points after one retrain.
- Alias failures exceed 20% at S/N > 6.

**Fallback contribution:** an open calibration benchmark plus alias failure maps for Gaia orbit inference.

Record the decision and the evidence in `logs/research_log.md`.

### Later phases (outline)

- **Phase 2:** scale up; add contamination and systematics; run all baselines; 3 seeds with 95% bootstrap confidence intervals.
- **Phase 3:** real DR4 validation (DPAC orbits, RV holdouts, false-alarm rate on RV-quiet stars), then rank candidates and cross-match them against RV archives.
- **Phase 4:** ablations (analytic step, scan-angle features, flow vs Gaussian head, data scaling), alias maps, and a misspecification failure analysis.
- **Phase 5:** laptop demo (ONNX on CPU, under 1 s per star), poster, and an arXiv preprint **before the affiliated fair**.

---

## Tech stack

- Use the Python version in the Kaggle image. Pin all versions once the environment works.
- **Core:** `numpy`, `scipy`, `astropy`, `astroquery` (archive access), `pytorch`.
- **Flows:** `zuko` (preferred) or `sbi`. **OPEN DECISION:** choose in Phase 1 step 1.
- **Evaluation:** `tarp` (or an in-house TARP), `scikit-learn` (C2ST, GBDT), `matplotlib`.
- **References and simulation:** gaiamock (pinned commit), `kima` (pip), Octofitter (optional, Julia).
- **Tooling:** `pytest`, `ruff` (format and lint), `pyyaml`.
- **Demo:** `onnxruntime`. Add a small UI later (Streamlit or plain HTML).

---

## Conventions

**PROPOSED, not yet agreed. Edit this section to make them official.**

- Python with type hints. Use numpy-style docstrings that **state units**.
- **Units go in variable names:** `t_d`, `period_d`, `w_mas`, `sigma_mas`, `psi_rad`, `parallax_mas`.
- Angles are in radians internally, astrometry in mas, time in days.
- Code in `model/` is pure functions with no hidden global state.
- Seeds are passed explicitly as `np.random.Generator` or `torch.Generator`. Never call `np.random.seed`.
- Every experiment runs from a YAML config. Each run writes its config, git hash, metrics and figures to `results/<run_id>/`.
- Notebooks are thin: they import from `src/` and contain no logic.
- Figures come only from `scripts/figures/`. Never hand-edit a figure or a results file.
- Run `ruff format` and `ruff check`, and make sure `pytest` passes before every commit.
- Keep commits small, with messages like `area: change` (for example `model: add vectorized Kepler solver`).
- Physics and linear solves use float64. Neural-network training uses float32 or AMP.

---

## Integrity & AI use (read before helping)

- **The student is the researcher.** Claude assists. The student must be able to explain and defend every equation and every line of code in a judge interview.
- The ISEF 2026–27 rules treat inappropriate AI use as an integrity violation. Society for Science publishes an **AI Use Table** with guidance on acceptable use and how to cite it. Some affiliated fairs require citing all AI use, including the prompts. Check both before the fair.
- **Keep `logs/ai_use_log.md`.** Log every session: date, what was asked, what Claude produced, and what the student changed or verified.
- For core science modules (`model/`, `eval/`), prefer that the student writes the code and Claude reviews it. If Claude drafts it, Claude explains the derivation step by step and the student re-derives it. Note which approach was used in the AI-use log.
- **Never invent results, numbers or citations.** Any number written in docs, the poster or the paper must come from a script output in `results/`.
- Never modify `configs/test_seeds.yaml` or the RV holdout list. Never tune on them.
- Ask before adding a dependency or downloading more than 1 GB.
- Report failures honestly. The fallback is a legitimate result.

---

## Novelty watch (re-run monthly; log the results in the research log)

Search arXiv astro-ph.EP and astro-ph.IM, alphaXiv, and the DR4 papers released on 2026-12-02.

Closest work to track:

| Work | Relation to this project |
|---|---|
| Nuthakki & Nuthakki, NeurIPS 2026 Sim2Science workshop | Amortized companion *detection* in Gaia-like epoch astrometry. **Most likely group to publish orbit posteriors next.** |
| Baycroft, Faria & Delisle 2026, kima, arXiv:2606.24132 | Nested sampling on Gaia epoch astrometry (+RV). Our reference and speed baseline. |
| G23H, arXiv:2602.00235 | Likelihood-based Hipparcos+Gaia companion modelling. |
| arXiv:2510.17459 (FM-MCMC) | Flow-matching orbit inference for **direct imaging** (different data). |
| Sahlmann & Gómez, arXiv:2404.09350; ExoDNN, arXiv:2602.02910 | ML classifiers on DR3 summaries. |

---

## Key references

- El-Badry et al. 2024, *A generative model for Gaia astrometric orbit catalogs* (gaiamock), arXiv:2411.00088
- Baycroft, Faria & Delisle 2026, kima for Gaia epoch astrometry, arXiv:2606.24132
- Price-Whelan et al. 2017, *The Joker* (ApJ): analytic marginalization of linear parameters
- Gaia Collaboration 2024, Gaia BH3 (arXiv:2404.10486), Appendix B: epoch-astrometry columns
- *On the Exoplanet Yield of Gaia Astrometry*, arXiv:2511.04673
- Talts et al. 2018 (SBC); Lemos et al. 2023 (TARP)
- ESA DR4 pages: https://www.cosmos.esa.int/web/gaia/dr4 and https://www.cosmos.esa.int/web/gaia/dr4-prerelease
- ISEF International Rules and AI Use Table: https://www.societyforscience.org/isef/international-rules
