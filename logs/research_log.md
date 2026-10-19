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
