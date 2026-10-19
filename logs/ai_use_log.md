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

**What I still have to do before using any of it for a counted run:**
- [ ] Re-derive `docs/derivations.md` sections 1–4 on paper and check them against the code.
- [ ] Read every function in `model/` and `eval/` and be able to explain it line by line.
- [ ] Re-run `pytest` myself and read what each test checks.
- [ ] Decide the open choices listed in `logs/research_log.md` (2026-10-20).
- [ ] Record here what I changed.
