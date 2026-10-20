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
