"""Simulate per-transit epoch astrometry for stars with one dark companion.

Physics in float64; returned arrays for the network are float32 except times, which are stored
as float32 *days* (resolution ~1e-4 d at 2000 d, never float16; see CLAUDE.md).
"""

from __future__ import annotations

import numpy as np

from gdr4.data.shards import PER_TRANSIT_KEYS
from gdr4.model.astrometry import (
    N_5P,
    design_matrix,
    design_matrix_5p,
    photocentre_semimajor_axis,
    t_peri_from_mean_anomaly,
)
from gdr4.model.linear import wls_fit
from gdr4.sim.noise import NoiseModel, sigma_transit_mas
from gdr4.sim.priors import Prior
from gdr4.sim.scan import ScanCache


def compact_transits(t_d, psi_rad, plx_factor, keep):
    """Left-pack kept transits (in time order) per row; returns arrays and a boolean mask."""
    order = np.argsort(np.where(keep, t_d, np.inf), axis=1, kind="stable")
    mask = np.take_along_axis(keep, order, axis=1)
    t_d, psi_rad, plx_factor = (
        np.where(mask, np.take_along_axis(a, order, axis=1), 0.0)
        for a in (t_d, psi_rad, plx_factor)
    )
    return t_d, psi_rad, plx_factor, mask


def simulate_batch(
    rng: np.random.Generator,
    n: int,
    prior: Prior,
    scan: ScanCache,
    noise: NoiseModel,
    drop_frac: float = 0.1,
) -> dict[str, np.ndarray]:
    """Simulate ``n`` stars.

    Steps: random sky pixel -> its transits; drop a random ``drop_frac`` of them (as gaiamock
    does); draw G, theta_nl, the amplitude scale and theta_lin from ``prior``; compute
    ``w = M theta_lin``; add noise; fit the single-star (5-parameter) model and store residuals.

    Returns
    -------
    dict
        Per-transit arrays of shape (n, scan.n_max): ``t_d``, ``psi_rad``, ``plx_factor``,
        ``w_mas``, ``sigma_mas``, ``resid5_mas``, ``mask``. Per-star arrays: ``period_d``,
        ``ecc``, ``mean_anom0_rad``, ``amp_scale_mas``, ``theta_lin`` (n, 9), ``gmag``,
        ``pix``, ``n_obs``, ``chi2_5p``, ``a0_mas``, ``snr_a0`` (a0 / sigma_transit),
        ``snr_mf`` (matched-filter S/N of the orbit signal).
    """
    pix = rng.integers(0, scan.n_pix, n)
    t_d, psi, plx = scan.t_d[pix], scan.psi_rad[pix], scan.plx_factor[pix]
    valid = np.isfinite(t_d)
    keep = valid & (rng.uniform(size=t_d.shape) >= drop_frac)
    t_d, psi, plx, mask = compact_transits(
        np.nan_to_num(t_d), np.nan_to_num(psi), np.nan_to_num(plx), keep
    )

    gmag = prior.sample_gmag(rng, n)
    nl = prior.sample_nonlinear(rng, n)
    theta_lin = prior.sample_linear(rng, nl["amp_scale_mas"])
    t_peri = t_peri_from_mean_anomaly(nl["mean_anom0_rad"], nl["period_d"])

    m = design_matrix(t_d, psi, plx, nl["period_d"][:, None], nl["ecc"][:, None], t_peri[:, None])
    w_clean = np.einsum("snk,sk->sn", m, theta_lin)
    w_orbit = np.einsum("snk,sk->sn", m[..., N_5P:], theta_lin[:, N_5P:])
    sigma = np.where(mask, sigma_transit_mas(gmag)[:, None], 1.0)
    w_obs = noise.apply(rng, w_clean, sigma, psi, mask)

    _, resid5, chi2_5p, _ = wls_fit(design_matrix_5p(t_d, psi, plx), w_obs, sigma, mask)
    a0 = photocentre_semimajor_axis(*theta_lin[:, N_5P:].T)
    snr_mf = np.sqrt(np.sum(np.where(mask, (w_orbit / sigma) ** 2, 0.0), axis=1))

    f32 = np.float32
    return {
        "t_d": t_d.astype(f32),
        "psi_rad": psi.astype(f32),
        "plx_factor": plx.astype(f32),
        "w_mas": w_obs.astype(f32),
        "sigma_mas": np.where(mask, sigma, 0.0).astype(f32),
        "resid5_mas": resid5.astype(f32),
        "mask": mask,
        "period_d": nl["period_d"],
        "ecc": nl["ecc"],
        "mean_anom0_rad": nl["mean_anom0_rad"],
        "amp_scale_mas": nl["amp_scale_mas"],
        "theta_lin": theta_lin,
        "gmag": gmag,
        "pix": pix,
        "n_obs": mask.sum(axis=1),
        "chi2_5p": chi2_5p,
        "a0_mas": a0,
        "snr_a0": a0 / sigma_transit_mas(gmag),
        "snr_mf": snr_mf,
    }


def star_view(batch: dict, i: int) -> dict:
    """Unpadded float64 arrays for star ``i`` of a padded batch (for references, plotting)."""
    m = np.asarray(batch["mask"][i])
    out = {}
    for k, v in batch.items():
        if k == "mask":
            continue
        v = np.asarray(v[i])
        out[k] = v[m].astype(np.float64) if k in PER_TRANSIT_KEYS else v
    return out
