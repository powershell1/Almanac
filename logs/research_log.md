# Research log (ISEF logbook)

Dated entries, newest at the bottom. Record what was done, why, and what was decided.
The **project start date** is the first *counted* simulation or training run (see CLAUDE.md,
"ISEF rules"). Pipeline smoke tests on toy data are not counted runs; say so when you log them.

---

## 2026-10-20

- Local environment created in `.env/` (Miniforge conda, Python 3.11.17, CPU-only PyTorch).
- Phase 1 pipeline code drafted (by Claude; see `logs/ai_use_log.md`). Not yet reviewed by me.
- gaiamock inspected at commit `dd30fdbf787eb96878734605ac077ac69bf28c84` (2026-05-09).
  The scanning-law files (`healpix_scans.zip`, 984 MB unzipped) are on Caltech Box and have
  **not** been downloaded yet.
- Only local smoke tests were run, using the *toy* scanning law. **No counted runs.**
  Project start date: not yet (see OPEN DECISION on ISEF 2027 vs 2028 in CLAUDE.md).
- Open decisions still pending: target ISEF year; flow library (zuko drafted, not decided);
  S/N definition and alias-failure definition for the kill test (drafted in
  `configs/killtest.yaml`; must be fixed before the first counted run).

## 2026-10-20 (later): bug found in the posterior head during a local sanity run

- Local CPU sanity run (toy scanning law, 300k sims, 2,003 steps, not counted): validation
  loss went 6.25 → 6.15, but never reached the prior-only level (5.84 in those coordinates).
- Diagnostic: with the context zeroed, the von Mises (phase) part reached its target
  exactly; the spline flow over (u_P, u_e) did not (4.38 vs 4.00, spiky).
- Cause: u = logit(...) gives the prior exponential tails; zuko's spline is the identity
  outside a bounded window, where the Gaussian base cannot fit those tails.
- Fix: u = probit(...) (inverse normal CDF), so the prior is exactly N(0, 1) = the flow's
  base. Re-test: the no-information head reaches 4.68 vs the 4.676 target. All tests pass.
- After the probit fix the run sat at the prior level (val 4.70 vs 4.676 after 600 steps).
  Diagnostic on S/N > 20 stars (same model, seed, 400 steps): loss 4.68 → 4.50 without and
  → 2.89 **with** an analytic circular-orbit periodogram input (`nn/periodogram.py`). Added it
  to the encoder (`model.pgram_k`). This is an architecture decision for me to approve; turning
  it off is an ablation.
- Local sanity run with the periodogram (toy scanning law, 300k sims, small model, 3,555 CPU
  steps / 1.5 h; **pipeline check, not counted, not a research result**): val loss
  4.68 → 1.96. `scripts/evaluate.py` on the 300-star smoke test set: HPD coverage error
  2.0 pp (nominal), 6.7 pp when sigma is 20% too small; period recovery 0.91 at S/N > 6;
  alias failures 0.10; 21 ms/star on CPU; C2ST ≤ 0.65 on 9/20 reference stars. The C2ST
  failures are all at S/N ≥ ~2: the small model's posteriors are wider than the sharp
  references there. Expect the full model and 12 GPU-hours to be needed for this criterion.

## 2026-10-27: why C2ST failed and why 14 high-S/N stars got the wrong period (local run)

All on the local sanity-check run (toy scan law, small model, 3,555 CPU steps; not counted).
Scripts: `scripts/diagnostics/`.

- **The reference is correct.** Scored with the exact posterior (prior × analytic marginal
  likelihood, no network), reference samples score the same as the truth (e.g. 42.8 vs 42.8)
  and the truth lies within ±1.4 sd of the reference centre for all 10 highest-S/N stars.
- **The NPE is centred right but too wide at high S/N.** Width ratio NPE/reference in u_P:
  ≈1 for S/N < 1.5, 20–50 at S/N ≈ 5, 100–5000 at S/N 11–98. NPE samples score 10–17,500
  log-units below the truth. C2ST is correctly detecting this; coverage still passes because a
  wide but centred posterior can be calibrated on average (coverage by S/N bin is 0.44–0.62
  at the 50% level).
- **The 14 "alias failures" are not real aliases.** The exact posterior puts 94–100% of its
  mass within 10% of the true period for all 14. The network is wrong, not the data.
- The circular periodogram input peaks at the true period for 10/14 of them (also at
  e ≈ 0.85–0.89), so the information is in the input; the network does not read it out
  precisely. For the 4 long-period failures (P ≈ 1600–1880 d, close to the ~1977 d baseline),
  the periodogram peak itself is 5–18% low, and the NPE copies that biased peak.
- **Flow head test** (fit fixed 2-D Gaussians): widths 0.1 and 0.01 are fitted in 3,000 steps;
  1e-3 needs ~9,000 steps (reaches 1.6e-3); 1e-4 is still at 3e-2 after 9,000 steps. A naive
  context-dependent affine "zoom" layer made it worse. So: moderate sharpness is a
  training-length issue; very sharp posteriors (S/N ≳ 30) are hard for this head.
- No error found in the equations, the simulator, the reference sampler or the metrics.
  Decision needed before the counted run: see the options in the session notes.

## 2026-10-27 (later): periodogram peak features added to the encoder

- `model.pgram_peaks: 8`: the 8 highest Δχ² peaks, each refined below the grid spacing
  (fine re-evaluation + parabola; within 0.05 grid steps of a 32× denser grid in the tests),
  given to the network as (log-period, height, height relative to the top peak, valid flag).
- A/B on the local data (small model, same seed and batch sequence, 800 steps on S/N > 6
  training stars, scored on the 144 S/N > 6 smoke test stars; toy scan law; not counted):
  - First attempt reused 60 cached batches; peaks looked *worse* (test loss 3.07 vs 2.83).
    Train losses were equal, so this was over-fitting to the repeated batches. Discarded.
  - With a fresh batch every step: test loss 2.978 → 2.877; period recovered 46.5% → 50.7%;
    wrong-period rate 63.9% → 56.2%; ~5% slower. Single seed and 144 stars (binomial sd
    ≈ 4 pp), so this is suggestive, not conclusive. Both variants are far from converged.
- Kept on in `configs/killtest.yaml`; `pgram_peaks: 0` is the ablation. Needs my approval.

## 2026-10-27 (later): 1.5 h local check with peak features vs without

Same config, seed, data and time budget; only `pgram_peaks` differs (0 vs 8). Toy scan law,
small model, CPU; **pipeline check, not counted**. Evaluation on the 300-star smoke test set.

| S/N > 6 (144 stars)                         | no peaks | peaks |
|---|---|---|
| period recovered (median within 10%)        | 91.0%    | 98.6% |
| wrong-period rate                           | 9.7%     | 2.8%  |
| wrong-period rate, sigma 20% too small      | 8.3%     | 3.5%  |
| coverage error nominal / sigma_inflated_1.2 | 2.0 / 6.7 pp | 3.3 / 2.3 pp |
| C2ST ≤ 0.65 (of 20 reference stars)         | 9        | 9     |

- 12 of the 14 earlier wrong-period stars fixed, 2 new ones (paired exact McNemar test,
  two-sided p ≈ 0.013). Training randomness is not covered (one seed per variant).
- All 4 remaining failures have P = 1447–1653 d, close to the ~1977 d baseline, where the
  circular periodogram peak itself is biased low.
- C2ST unchanged: the high-S/N blurriness comes from the flow head, not period finding.
- Peaks run reached step 3389 vs 3555 (about 5% slower per step).

## 2026-10-27 (later): fixing the C2ST blurriness, anchored heads

Local 1.5 h runs, same config/seed/data (toy scan law, CPU; **not counted**), evaluated on the
300-star smoke test set and 20 reference stars. Code: `f262e26`, `41b80bf`; docs §5c, §5d.

| | peaks only | + anchored period head | + (e, M0) profile |
|---|---|---|---|
| training steps in 1.5 h      | 3389  | 3245  | 1597  |
| best val loss                | 1.93  | 0.48  | −1.55 |
| coverage error (nominal)     | 3.3 pp | 3.0 pp | 2.3 pp |
| period recovered, S/N > 6    | 98.6% | 99.3% | 98.6% |
| C2ST ≤ 0.65 (of 20)          | 9     | 9     | 13    |
| ms/star CPU, single / batched| 21.5 / 10.4 | 20.5 / 9.7 | 71.5 / 17.4 |

- Toy benchmark first (`scripts/diagnostics/bench_flow_head.py`): a plain or moment-standardised
  flow stays ~45× too wide at width 1e-4. Cause: a network cannot output a location to 1e-4
  from scratch. Fix: anchor on data-derived locations and Fisher widths (docs §5c).
- Anchored period head alone fixed the period dimension (per-dim C2ST 0.51–0.69 up to S/N 45)
  but e and M0 stayed 4–48× too wide, so joint C2ST did not move.
- Adding the (e, M0) profile: all 12 stars with S/N ≤ 6 pass; S/N 11–23 at 0.69–0.75;
  S/N 27–98 still fail (0.90–1.00). For stars 9 and 10 the circular-periodogram anchor is
  19 and 5.5 posterior widths off-centre (eccentric orbits shift the circular peak).
- Timing: the profile makes single-star CPU inference 71.5 ms (> 50 ms gate); needs measuring
  on the Kaggle GPU, where the gate applies.
- Next options: Keplerian re-refinement of the period anchor at (e*, M0*); longer training
  (this run had half the steps and its loss was still falling).

## 2026-10-27 (later): Keplerian re-refinement of the period anchor

Same local setup (toy scan law, CPU, 1.5 h, same seed/data; **not counted**). Code `b771d9a`,
docs §5e, check `scripts/diagnostics/check_period_anchor.py`.

| | profile (prev) | + Keplerian anchor refinement |
|---|---|---|
| steps in 1.5 h                          | 1597  | 1387  |
| best val loss                           | −1.55 | −2.11 |
| C2ST ≤ 0.65 (of 20)                     | 13    | 15 (gate: ≥ 14) |
| coverage error, nominal                 | 2.3 pp | 4.0 pp |
| period recovered / wrong-period, S/N > 6 | 98.6% / 2.1% | 100% / 0.0% |
| coverage error, sigma 20% too small     | 13.3 pp | 10.0 pp |
| ms/star CPU single / batched            | 71.5 / 17.4 | 94.8 / 24.5 |

- Anchor offsets from the exact centre: worst stars 19.0 / 6.0 / 5.5 → 1.8 / 0.8 / 0.5 widths.
- C2ST by S/N: 11–23 now 0.61–0.65; 27–98 now 0.74–0.84 (period dimension fine; the phase is
  still 9–22× too wide there). C2ST on a few hundred samples is noisy (±0.03 between runs),
  so 15/20 should be read as roughly 14–16.
- **Misspecification:** with true noise 20% above the reported sigma, coverage error is 10 pp
  for the sharp heads vs 2.3 pp for the old, blurry head. Expected: a correct sharp posterior
  becomes over-confident when sigma is wrong; the blurry head was robust by accident. Directly
  relevant to claimed contribution 3; needs its own design (e.g. a noise-scale nuisance
  parameter in training). The Phase 1 gate only uses nominal noise.
- **Timing:** single-star CPU inference is 95 ms (> 50 ms), 60 ms of it in the (e, M0) profile
  and period refinement (many small float64 ops). Batched: 24.5 ms/star. Must be measured on the
  Kaggle GPU. OPEN: decide whether the 50 ms gate means single-star latency or batched
  throughput, before the counted run.

## 2026-10-27 (later): can coverage error be reduced?

Local, toy scan law, 300 smoke test stars, **not counted**.

1. **Nominal coverage error (2.3–4.0 pp across runs) is consistent with zero.** Every run's 95%
   bootstrap interval contains 0.50 / 0.68 / 0.90; ±5 pp is the resolution of 300 stars.
   The 2,000-star Kaggle test set resolves ±2 pp.
2. **Misspecified noise is a model problem, not a network problem.** Exact posteriors computed
   with the reported sigma when the true noise is 1.2 sigma (150 stars,
   `scripts/diagnostics/exact_coverage_misspec.py`): u_P coverage 0.43 / 0.60 / 0.85; network
   0.46 / 0.64 / 0.93. With correct sigma both are calibrated.
3. **Training with an unknown noise scale (kappa ~ LogUniform(0.8, 1.5), 1376 steps; the
   baseline had 1387):** did not help yet.

| test set | baseline max cov. err. | kappa-trained |
|---|---|---|
| nominal (kappa = 1)        | 3.3 pp  | 3.3 pp  |
| kappa = 1.2                | 10.7 pp | 13.3 pp |
| kappa = 0.8                | 14.0 pp | 8.3 pp  |
| kappa from the prior       | 3.0 pp  | 9.7 pp  |

   Both models under-cover badly for stars with true kappa > 1.2 (50% level: 0.23–0.31), even
   at S/N < 6: extra noise produces periodogram peaks that look significant against the
   reported sigma, and the network anchors on them. The kappa-trained network has not learned
   to rescale the Delta chi^2 features by the noise level.
   C2ST (nominal) 16/20 for both. Single-star CPU timing differed 2x (95 vs 49 ms) between
   identical architectures, so local timing is too noisy to judge the 50 ms gate.
4. Proposed next step: estimate kappa_hat^2 = (chi2_5 − max Delta chi^2) / (N − 12) from the
   best Keplerian fit and normalise all physics features by it (Delta chi^2 / kappa_hat^2,
   anchor widths × kappa_hat).

## 2026-10-28: noise-level rescaling fixes coverage under misspecified noise

Code `5dee1ff` (`model.noise_rescale`), docs §7. Local, toy scan law, 300 stars per test set,
training on kappa ~ LogUniform(0.8, 1.5) noise; 1385 steps (kappa-only run: 1376; baseline:
1387). **Not counted.**

kappa_hat^2 = (chi2_5 − max Delta chi^2)/(N − 12) from the best Keplerian fit; Delta chi^2
inputs / kappa_hat^2, anchor widths × kappa_hat, log kappa_hat as an input. Tested: tracks the
true noise scale to 8% (median) at low and high S/N; all other inputs exactly invariant to
rescaling the reported sigma.

| test set (max HPD coverage error) | baseline | kappa only | kappa + rescale |
|---|---|---|---|
| nominal (kappa = 1)            | 3.3 pp  | 3.3 pp  | 2.0 pp |
| kappa = 1.2                    | 10.7 pp | 13.3 pp | 2.3 pp |
| kappa = 0.8                    | 14.0 pp | 8.3 pp  | 3.0 pp |
| kappa from the prior           | 3.0 pp  | 9.7 pp  | 3.7 pp |
| gaiamock jitter (out of prior) | 7.0 pp  | –       | 2.7 pp |
| 2% outliers (out of prior)     | 19.7 pp | –       | 3.7 pp |
| scan harmonics (out of prior)  | 7.0 pp  | –       | 5.0 pp |

- Stars with true noise 1.2–2× sigma: 50% coverage 0.23–0.31 → 0.55–0.56.
- Period recovery 100% and no wrong periods on every set; C2ST 15/20 (references assume
  kappa = 1 exactly, so slightly pessimistic; they should marginalise kappa).
- Outliers are handled only through an overall noise inflation, not an outlier model.
- OPEN: the kappa range (0.8–1.5) is a placeholder to justify; references need a kappa dimension.

## 2026-10-28 (later): noise scale added to the reference sampler

Code `ded0b3c`. With `noise_scale_range` (read from the training noise model, so simulator and
reference share one prior) the reference adds log kappa ~ U(log 0.8, log 1.5) and uses kappa
sigma in the analytic likelihood. Simulated data now record each star's true `noise_scale`
(random draws unchanged; the regenerated smoke test sets are bit-identical otherwise). Tests:
kappa recovered to < 15% with the truth inside the 99% interval; refined vs plain sampling
agree in all 5 coordinates; 5-D kernel density normalised.

Local check (noise-rescaled model from 2026-10-28, 1385 steps; 20 reference stars per set,
lowest ESS 2433; **not counted**):

| C2ST ≤ 0.65 | vs kappa = 1 references | vs kappa-marginalised references |
|---|---|---|
| nominal test set     | 15/20 | 15/20 (median 0.60) |
| scale_prior test set | –     | 11/20 (median 0.62) |

- Reference self-calibration incl. log kappa within the noise of 20 stars
  (log kappa coverage 0.55–0.60 / 0.65–0.70 / 0.85–0.95).
- On scale_prior, new failures at S/N 1.8–4.9 with kappa 1.12–1.33 (C2ST 0.68–0.89); S/N is
  quoted against the reported sigma, so these are effectively fainter. High-S/N stars fail on
  both sets, as before. Cannot separate under-training (1385 steps) from a real weakness here.
- If scale_prior becomes the gating set (OPEN), C2ST currently fails it locally.

## 2026-10-28 (later): why the high-S/N stars still fail C2ST

Noise-rescaled model (1385 steps) vs kappa-marginalised references, nominal smoke set
(`scripts/diagnostics/diag_c2st.py`, `diag_high_snr.py`; **not counted**).

- Failing parameters at S/N ≥ 10: u_e 1.6–3.9× too wide (per-dim C2ST 0.59–0.80) and M0
  1.4–12× (0.53–0.75); the period core is fine (0.51–0.56) except stars 10 and 9.
- On these stars the e and M0 anchors are right (offsets ≤ 2.2 / 1.7 exact sd; widths 0.56–0.96
  / 0.75–1.17 of the exact sd) and the phase head puts 98–100% weight on anchored components,
  yet its phase is 1.5–19× too wide even at fixed period. Not a correlation effect: exact
  correlations are mostly |rho| ≤ 0.3 and conditional widths are still 1.7–21× too wide.
- Across all 300 test stars the anchors are over-confident: at S/N > 10 the truth is within 2
  anchor widths for only 74–87% (M0) and 79% (e) instead of 95%; robust spread 1.1–1.8 widths.
  The widths are one-parameter-at-a-time Fisher widths, which ignore correlations.
- At S/N > 30, no anchor is within 5 widths of the truth for 10% of stars: near-misses on the
  right peak (193 vs 194 d at 10 widths, 93 d at 7, 363 vs 362 d at 6) where coordinate ascent
  has not converged for eccentric orbits; long periods near the baseline (1525 vs 1608 d,
  1639 vs 1733 d); one P = 1867 d star with no periodogram peak near the truth.
  Re-ranking anchors by their Keplerian Delta chi^2 does not help.
- Interpretation: the network cannot tell reliable from unreliable anchors, so the likelihood
  objective makes it widen everywhere.
- Proposed fix: joint Newton refinement of (P, e, M0) on the Keplerian Delta chi^2 with the full
  Laplace covariance (converged anchors, marginal widths including correlations, conditional
  anchors for e | P and M0 | P, e), plus anchor-reliability inputs.
