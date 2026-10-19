"""Scan geometry: transit times, scan angles and parallax factors per sky position.

Two sources produce the same :class:`ScanCache`:

- ``build_cache_from_gaiamock``: gaiamock's pre-computed scanning law (49152 HEALPix level-64
  FITS files in ``gaiamock/healpix_scans``). **Use this for every counted run.**
- ``build_cache_toy_nsl``: an approximate nominal scanning law written for this repo, only so
  the pipeline can be tested without the 1 GB gaiamock download. It ignores dead time, gaps,
  the real precession-rate law and the exact FOV geometry. **Never use it for results.**

Times are days from the DR4 reference epoch (J2017.5); angles in radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gdr4.model.astrometry import DR4_REF_EPOCH_JD

# DR4 data window used by gaiamock (get_gost_one_position): JD 2456891.5 to 2458868.5.
DR4_START_JD = 2456891.5
DR4_END_JD = 2458868.5

_GOST_TIME = "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"
_GOST_PSI = "scanAngle[rad]"
_GOST_PLX = "parallaxFactorAlongScan"


@dataclass
class ScanCache:
    """Padded per-pixel scan geometry. Arrays of shape (n_pix, n_max) are NaN-padded."""

    nside: int
    source: str  # "gaiamock@<commit>" or "toy_nsl"
    t_d: np.ndarray
    psi_rad: np.ndarray
    plx_factor: np.ndarray
    n_transits: np.ndarray  # (n_pix,)

    @property
    def n_pix(self) -> int:
        return self.t_d.shape[0]

    @property
    def n_max(self) -> int:
        return self.t_d.shape[1]

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            nside=self.nside,
            source=self.source,
            t_d=self.t_d,
            psi_rad=self.psi_rad,
            plx_factor=self.plx_factor,
            n_transits=self.n_transits,
        )

    @classmethod
    def load(cls, path: str | Path) -> ScanCache:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                nside=int(z["nside"]),
                source=str(z["source"]),
                t_d=z["t_d"],
                psi_rad=z["psi_rad"],
                plx_factor=z["plx_factor"],
                n_transits=z["n_transits"],
            )


def _pad(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    n = np.array([len(r[0]) for r in rows])
    n_max = int(n.max())
    out = [np.full((len(rows), n_max), np.nan) for _ in range(3)]
    for i, r in enumerate(rows):
        order = np.argsort(r[0])
        for j in range(3):
            out[j][i, : n[i]] = r[j][order]
    return out[0], out[1], out[2], n


# ---- gaiamock scanning law ----------------------------------------------------------------


def _read_gaiamock_pixel(path: Path):
    from astropy.table import Table

    tab = Table.read(path)
    jd = np.asarray(tab[_GOST_TIME], dtype=np.float64)
    keep = (jd > DR4_START_JD) & (jd < DR4_END_JD)
    return (
        jd[keep] - DR4_REF_EPOCH_JD,
        np.asarray(tab[_GOST_PSI], np.float64)[keep],
        np.asarray(tab[_GOST_PLX], np.float64)[keep],
    )


def build_cache_from_gaiamock(
    gaiamock_dir: str | Path, n_jobs: int = 4, commit: str = "unknown"
) -> ScanCache:
    """Read all 49152 ``healpix_64_<pix>.fits`` files (RING ordering) into one cache."""
    from joblib import Parallel, delayed

    scan_dir = Path(gaiamock_dir) / "healpix_scans"
    nside = 64
    n_pix = 12 * nside * nside
    paths = [scan_dir / f"healpix_64_{i}.fits" for i in range(n_pix)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} scan files missing, e.g. {missing[0]}")
    rows = Parallel(n_jobs=n_jobs, batch_size=256)(delayed(_read_gaiamock_pixel)(p) for p in paths)
    t, psi, plx, n = _pad(rows)
    return ScanCache(nside, f"gaiamock@{commit}", t, psi, plx, n)


# ---- toy nominal scanning law (tests only) ------------------------------------------------

_OBLIQUITY_RAD = np.deg2rad(23.4392911)
_XI_RAD = np.deg2rad(45.0)  # solar aspect angle
_PRECESSION_D = 63.12
_SPIN_D = 0.25  # 6 h
_BASIC_ANGLE_RAD = np.deg2rad(106.5)
_AC_HALF_WIDTH_RAD = np.deg2rad(0.35)
_L2_AU = 1.01


def _sun_lon(jd):
    d = jd - 2451545.0
    g = np.deg2rad(357.529 + 0.98560028 * d)
    return (
        np.deg2rad(280.459 + 0.98564736 * d)
        + np.deg2rad(1.915) * np.sin(g)
        + np.deg2rad(0.020) * np.sin(2 * g)
    )


def _spin_axis(jd):
    """Spin axis z (ecliptic frame) at angle xi from the Sun, precessing around it."""
    lam = _sun_lon(jd)
    s = np.stack([np.cos(lam), np.sin(lam), np.zeros_like(lam)], -1)
    k = np.array([0.0, 0.0, 1.0])
    perp = np.cross(s, k)
    nu = 2 * np.pi * (jd - DR4_START_JD) / _PRECESSION_D
    z = np.cos(_XI_RAD) * s + np.sin(_XI_RAD) * (
        np.cos(nu)[..., None] * k + np.sin(nu)[..., None] * perp
    )
    return s, z


def _scan_basis(z):
    k = np.array([0.0, 0.0, 1.0])
    a = np.cross(k, z)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b = np.cross(z, a)
    return a, b


def _eq_to_ecl(v):
    ce, se = np.cos(_OBLIQUITY_RAD), np.sin(_OBLIQUITY_RAD)
    return np.stack(
        [v[..., 0], ce * v[..., 1] + se * v[..., 2], -se * v[..., 1] + ce * v[..., 2]], -1
    )


def toy_nsl_transits(ra_rad: float, dec_rad: float):
    """Approximate transits of one source under a toy nominal scanning law.

    For each 6 h spin revolution and each of the two FOVs, find when the FOV crosses the
    source's along-scan longitude and keep it if the source is within the across-scan
    half-width at that moment. The scan direction is z x u; psi is its position angle
    (north through east); f_plx = 1.01 AU * (Sun direction . scan direction).
    """
    ca, sa, cd, sd = np.cos(ra_rad), np.sin(ra_rad), np.cos(dec_rad), np.sin(dec_rad)
    u = _eq_to_ecl(np.array([cd * ca, cd * sa, sd]))
    east = _eq_to_ecl(np.array([-sa, ca, 0.0]))
    north = _eq_to_ecl(np.array([-sd * ca, -sd * sa, cd]))

    jd_rev = np.arange(DR4_START_JD, DR4_END_JD, _SPIN_D)
    out_t, out_psi, out_plx = [], [], []
    for fov_offset in (+0.5 * _BASIC_ANGLE_RAD, -0.5 * _BASIC_ANGLE_RAD):
        _, z = _spin_axis(jd_rev)
        a, b = _scan_basis(z)
        phi_u = np.arctan2(b @ u, a @ u)
        spin_phase = 2 * np.pi * (jd_rev - DR4_START_JD) / _SPIN_D  # = 0 mod 2 pi at jd_rev
        dphi = np.mod(phi_u - (spin_phase + fov_offset), 2 * np.pi)
        jd = jd_rev + dphi / (2 * np.pi) * _SPIN_D
        s, z = _spin_axis(jd)
        ac = np.arcsin(np.clip(z @ u, -1, 1))
        keep = np.abs(ac) < _AC_HALF_WIDTH_RAD
        if not np.any(keep):
            continue
        d = np.cross(z[keep], u)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        out_t.append(jd[keep] - DR4_REF_EPOCH_JD)
        out_psi.append(np.arctan2(d @ east, d @ north))
        out_plx.append(_L2_AU * np.sum(s[keep] * d, axis=-1))
    if not out_t:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.concatenate(out_t), np.concatenate(out_psi), np.concatenate(out_plx)


def build_cache_toy_nsl(nside: int = 8) -> ScanCache:
    """Toy-NSL cache on a HEALPix RING grid (pixel centres). Tests only."""
    import healpy as hp

    n_pix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(n_pix))
    rows = [toy_nsl_transits(phi[i], 0.5 * np.pi - theta[i]) for i in range(n_pix)]
    t, psi, plx, n = _pad(rows)
    return ScanCache(nside, "toy_nsl", t, psi, plx, n)
