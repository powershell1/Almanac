"""Load real epoch astrometry (per-CCD rows) and combine it per transit.

Current use (CLAUDE.md): build and test the loader on the DR4 pre-release files and on the
public Gaia BH3 epoch astrometry shipped with gaiamock (``data/epochast.dat``). After the
2026-12-02 release, fetch ``epoch_astrometry`` through the archive/DataLink for a bounded
volume only; never bulk-download.

VERIFY before any real-data run: the column names, the units of the scan angle, and the sign
conventions of ``scan_pos_angle`` and ``parallax_factor_al`` against the final DR4 data model.
The defaults below are the column names of the Gaia BH3 release, as used in gaiamock's demo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gdr4.model.astrometry import DR4_REF_EPOCH_JD, design_matrix_5p
from gdr4.model.linear import wls_fit

DEFAULT_COLUMNS = {
    "transit_id": "transit_id",
    "ccd": "ccd_id",
    "time_jd": "obs_time_tcb",
    "w_mas": "centroid_pos_al",
    "sigma_mas": "centroid_pos_error_al",
    "plx_factor": "parallax_factor_al",
    "scan_angle": "scan_pos_angle",
    "outlier": "outlier_flag",
}
_BH3_ORDER = [
    "transit_id",
    "ccd_id",
    "obs_time_tcb",
    "centroid_pos_al",
    "centroid_pos_error_al",
    "parallax_factor_al",
    "scan_pos_angle",
    "outlier_flag",
]


def read_epoch_table(path: str | Path, fmt: str | None = None):
    """Read a per-CCD epoch-astrometry table (VOTable/XML, FITS, ECSV, CSV or BH3 ``.dat``)."""
    from astropy.table import Table

    path = Path(path)
    if fmt is None and path.suffix == ".dat":
        return Table.read(path, format="ascii.no_header", names=_BH3_ORDER, comment="#")
    if fmt is None and path.suffix in (".xml", ".vot"):
        fmt = "votable"
    return Table.read(path, format=fmt)


def combine_ccd_to_transit(transit_id, t_jd, w_mas, sigma_mas, plx_factor, psi_rad):
    """Inverse-variance mean of the CCD abscissae within each transit.

    Geometry (time, scan angle, parallax factor) is averaged without weights, as in gaiamock's
    BH3 demo; it varies by < 1e-4 within a transit.

    Returns
    -------
    dict of per-transit arrays sorted by time: t_jd, w_mas, sigma_mas, plx_factor, psi_rad,
    n_ccd.
    """
    transit_id = np.asarray(transit_id)
    order = np.argsort(transit_id, kind="stable")
    ids, start, counts = np.unique(transit_id[order], return_index=True, return_counts=True)
    ivar = 1.0 / np.asarray(sigma_mas, float)[order] ** 2

    def seg_sum(x):
        return np.add.reduceat(np.asarray(x, float)[order], start)

    sum_ivar = np.add.reduceat(ivar, start)
    out = {
        "t_jd": seg_sum(t_jd) / counts,
        "w_mas": np.add.reduceat(ivar * np.asarray(w_mas, float)[order], start) / sum_ivar,
        "sigma_mas": 1.0 / np.sqrt(sum_ivar),
        "plx_factor": seg_sum(plx_factor) / counts,
        # circular mean of the scan angle (it may cross +-pi within a transit)
        "psi_rad": np.arctan2(seg_sum(np.sin(psi_rad)), seg_sum(np.cos(psi_rad))),
        "n_ccd": counts,
    }
    by_time = np.argsort(out["t_jd"])
    return {k: v[by_time] for k, v in out.items()}


def star_from_table(
    tab, columns: dict | None = None, scan_angle_unit: str = "deg", drop_outliers: bool = True
) -> dict:
    """Per-transit star dict in the simulator's format (float64, unpadded)."""
    c = dict(DEFAULT_COLUMNS, **(columns or {}))
    keep = np.ones(len(tab), bool)
    if drop_outliers and c.get("outlier") in tab.colnames:
        keep &= np.asarray(tab[c["outlier"]]) == 0
    psi = np.asarray(tab[c["scan_angle"]], float)[keep]
    psi = np.deg2rad(psi) if scan_angle_unit == "deg" else psi
    tr = combine_ccd_to_transit(
        np.asarray(tab[c["transit_id"]])[keep],
        np.asarray(tab[c["time_jd"]], float)[keep],
        np.asarray(tab[c["w_mas"]], float)[keep],
        np.asarray(tab[c["sigma_mas"]], float)[keep],
        np.asarray(tab[c["plx_factor"]], float)[keep],
        psi,
    )
    t_d = tr["t_jd"] - DR4_REF_EPOCH_JD
    m5 = design_matrix_5p(t_d, tr["psi_rad"], tr["plx_factor"])
    _, resid, chi2, _ = wls_fit(m5, tr["w_mas"], tr["sigma_mas"])
    return {
        "t_d": t_d,
        "psi_rad": tr["psi_rad"],
        "plx_factor": tr["plx_factor"],
        "w_mas": tr["w_mas"],
        "sigma_mas": tr["sigma_mas"],
        "resid5_mas": resid,
        "n_obs": len(t_d),
        "chi2_5p": chi2,
        "n_ccd": tr["n_ccd"],
    }


def pad_stars(stars: list[dict]) -> dict[str, np.ndarray]:
    """Stack unpadded star dicts into a padded batch (with ``mask``) for the network."""
    n_max = max(s["n_obs"] for s in stars)
    keys = ("t_d", "psi_rad", "plx_factor", "w_mas", "sigma_mas", "resid5_mas")
    out = {k: np.zeros((len(stars), n_max)) for k in keys}
    out["mask"] = np.zeros((len(stars), n_max), bool)
    for i, s in enumerate(stars):
        n = s["n_obs"]
        for k in keys:
            out[k][i, :n] = s[k]
        out["mask"][i, :n] = True
    out["n_obs"] = np.array([s["n_obs"] for s in stars])
    out["chi2_5p"] = np.array([s["chi2_5p"] for s in stars])
    return out
