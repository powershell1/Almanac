"""kima cross-check (Baycroft, Faria & Delisle 2026, arXiv:2606.24132). NOT IMPLEMENTED YET.

Plan (Phase 1 step 4): run kima's Gaia epoch-astrometry model on the first ``n_kima`` test stars
and compare its (P, e, M0) posterior with the brute-force reference (C2ST) and its wall-clock
time with the NPE (speed baseline).

Before writing this, read kima's documentation for the Gaia model: its expected input columns,
time units, and whether its phase parameter matches our M0 (= mean anomaly at J2017.5).
Ask before adding kima as a dependency (CLAUDE.md). ``export_star`` writes the data in a neutral
format so the kima run can be scripted once the format is known.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gdr4.model.astrometry import DAYS_PER_YEAR, DR4_REF_EPOCH_JD


def export_star(star: dict, path: str | Path) -> None:
    """Write one star's transits as whitespace-separated columns with a header."""
    cols = np.column_stack(
        [
            star["t_d"] + DR4_REF_EPOCH_JD,
            star["t_d"] / DAYS_PER_YEAR,
            star["w_mas"],
            star["sigma_mas"],
            star["psi_rad"],
            star["plx_factor"],
        ]
    )
    np.savetxt(
        path,
        cols,
        fmt="%.10f",
        header="bjd_tcb t_yr_from_J2017.5 w_mas sigma_mas psi_rad plx_factor_al",
    )


def run_kima(*args, **kwargs):
    raise NotImplementedError("kima cross-check not written yet; see module docstring")
