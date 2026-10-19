import numpy as np
import pytest
from astropy.table import Table

from gdr4.data.dr4 import combine_ccd_to_transit, pad_stars, read_epoch_table, star_from_table


def test_combine_weighted_mean(rng):
    tid = np.repeat([7, 3, 5], [8, 9, 8])
    t = np.repeat([10.0, 1.0, 5.0], [8, 9, 8]) + rng.uniform(0, 1e-4, 25)
    sigma = rng.uniform(0.1, 0.4, 25)
    w = rng.normal(0, 1, 25)
    psi = np.repeat([3.13, 0.5, -1.0], [8, 9, 8]) + rng.normal(0, 1e-3, 25)
    psi = np.angle(np.exp(1j * psi))  # first transit straddles +-pi
    out = combine_ccd_to_transit(tid, t, w, sigma, np.zeros(25), psi)
    assert list(out["n_ccd"]) == [9, 8, 8]  # sorted by time: ids 3, 5, 7
    sel = tid == 3
    ivar = 1 / sigma[sel] ** 2
    np.testing.assert_allclose(out["w_mas"][0], np.sum(ivar * w[sel]) / ivar.sum())
    np.testing.assert_allclose(out["sigma_mas"][0], 1 / np.sqrt(ivar.sum()))
    assert abs(abs(out["psi_rad"][2]) - 3.13) < 0.01  # circular mean, not ~0


def test_star_from_table_roundtrip(rng):
    n_tr, n_ccd = 40, 8
    tab = Table(
        {
            "transit_id": np.repeat(np.arange(n_tr), n_ccd),
            "ccd_id": np.tile(np.arange(1, n_ccd + 1), n_tr),
            "obs_time_tcb": np.repeat(2457000.0 + np.sort(rng.uniform(0, 1800, n_tr)), n_ccd),
            "centroid_pos_al": rng.normal(0, 0.2, n_tr * n_ccd),
            "centroid_pos_error_al": np.full(n_tr * n_ccd, 0.13),
            "parallax_factor_al": np.repeat(rng.uniform(-0.7, 0.7, n_tr), n_ccd),
            "scan_pos_angle": np.repeat(rng.uniform(-180, 180, n_tr), n_ccd),
            "outlier_flag": np.zeros(n_tr * n_ccd, int),
        }
    )
    s = star_from_table(tab)
    assert s["n_obs"] == n_tr
    np.testing.assert_allclose(s["sigma_mas"], 0.13 / np.sqrt(8))
    batch = pad_stars([s, s])
    assert batch["mask"].sum() == 2 * n_tr


def test_reads_gaia_bh3_epoch_file(gaiamock):
    from pathlib import Path

    path = Path(gaiamock.__file__).parent / "data" / "epochast.dat"
    if not path.exists():
        pytest.skip("BH3 file not present")
    s = star_from_table(read_epoch_table(path))
    assert 30 < s["n_obs"] < 200
    assert np.all(np.isfinite(s["w_mas"])) and np.all(s["sigma_mas"] > 0)
    # BH3 has a large orbit (a0 ~ 27 mas), so a single-star fit must be terrible.
    assert s["chi2_5p"] / (s["n_obs"] - 5) > 100
