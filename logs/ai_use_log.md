# AI-use log

One entry per session: date, what was asked, what the AI produced, and what I (the student)
changed or verified. Keep this honest and complete; some fairs require the prompts too.

Columns for the "verified" part: who wrote it, whether I re-derived it, which tests I ran.

---

## 2026-10-20: Claude Code (Opus 5.5), session 1

**Prompt (verbatim):** "I want to create Gaia DR research paper, can you write the code I have to
run on kaggle also commit occasionally. Also create .env folder and install conda inside there.
if you want to test anything you can run it on this device first."
(Context given to Claude: `CLAUDE.md` and the project-selection report `agent2.md`.)

**What Claude produced (all drafts, none yet reviewed by me):**
- Local conda environment in `.env/` and `scripts/setup_local_env.sh`.
- The whole Phase 1 code base under `src/gdr4/`, `scripts/`, `tests/`, `configs/`,
  `notebooks/kaggle/`, plus `docs/derivations.md`.
- **Core science modules drafted by Claude, not by me:** `model/kepler.py`,
  `model/astrometry.py`, `model/linear.py`, `eval/*`. CLAUDE.md asks that in this case I
  re-derive each step myself. Status: **not yet re-derived.**

**Design decisions Claude made that I must approve or change:**
- Miniforge (conda-forge) instead of Anaconda's Miniconda, to avoid accepting Anaconda's ToS.
- Flow library: zuko. Phase handled by a von Mises mixture (circular), (log P, e) by a spline flow.
- Amplitude prior: hierarchical scale `s_a ~ LogUniform(0.01, 2) mas` on A, B, F, G
  (`docs/derivations.md` §4). Its implied cos i prior is **not** isotropic
  (`results/prior_checks/ti_prior.png`).
- Kill-test definitions: S/N = a0 / sigma_transit; "recovered" = posterior median within 10%;
  "alias failure" = < 50% posterior mass within 10%; coverage gate on joint HPD coverage;
  C2ST only on reference stars with ESS ≥ 2000.
- Misspecification strengths in `sim/noise.py` are placeholders.
- Reference sampler: prior sampling + Nelder–Mead mode finding + defensive importance
  sampling (my own design, checked only on simulated stars; kima cross-check not written).
- A toy nominal scanning law (`sim/scan.py`) for local tests only.

**What Claude verified (tests in `tests/`, 49 passing on 2026-10-20):** Kepler residual
< 1e-12; forward model equals gaiamock's C model to 1e-8 mas; marginal likelihood equals a
direct Gaussian; simulator noise level; shards resume; flow density normalised; training
resume and 2-process DDP; reference sampler recovers periods up to S/N ≈ 100; coverage metrics
on toy Gaussians; loader reads the Gaia BH3 epoch file.

**What I still have to do before using any of it for a counted run:**
- [ ] Re-derive `docs/derivations.md` sections 1–4 on paper and check them against the code.
- [ ] Read every function in `model/` and `eval/` and be able to explain it line by line.
- [ ] Re-run `pytest` myself and read what each test checks.
- [ ] Decide the open choices listed in `logs/research_log.md` (2026-10-20).
- [ ] Record here what I changed.

---

## 2026-10-26 / 27: Claude Code (Opus 5.5), session 2

**Prompts:** "Can you investigate why Agreement with exact posteriors (C2ST) not passed the
target or Wrong-period rate ... Could it be code problem or about the equation?", "push it to
github", "add the periodogram peak features to the encoder".

**What Claude produced:** diagnostics in `scripts/diagnostics/` (findings in the research log,
2026-10-27); peak features in `src/gdr4/nn/periodogram.py` (`top_peaks`, `refine_peaks`,
`peak_features`) and `nn/model.py`; tests; derivation in `docs/derivations.md` §5b.

**Status:** drafted and tested by Claude; not yet reviewed or re-derived by me.
- [ ] Re-derive the parabolic peak refinement (`docs/derivations.md` §5b).
- [ ] Decide whether to keep the peak features (A/B evidence is weak: one seed, 144 stars).
