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
