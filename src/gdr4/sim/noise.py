"""Per-transit noise models: nominal (A) and misspecified variants (B) for robustness tests.

The *reported* uncertainty ``sigma_mas`` is always the nominal one; misspecified models change
the noise that is actually added, which is what happens when Gaia's error model is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Per-CCD along-scan uncertainty vs G, copied from gaiamock (al_uncertainty_per_ccd_interp):
# the "EDR3 adjusted" curve of Fig. 3 in Lindegren et al. 2022 (arXiv:2206.05439).
_G_GRID = np.array([4, 5, 6, 7, 8.2, 8.4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], float)
_SIGMA_CCD_MAS = np.array(
    [
        0.4,
        0.35,
        0.15,
        0.17,
        0.23,
        0.13,
        0.13,
        0.135,
        0.125,
        0.13,
        0.15,
        0.23,
        0.36,
        0.63,
        1.05,
        2.05,
        4.1,
    ]
)
N_CCD_PER_TRANSIT = 8  # gaiamock averages 8 CCDs per FOV transit


def sigma_ccd_mas(gmag):
    """Per-CCD along-scan uncertainty in mas."""
    return np.interp(gmag, _G_GRID, _SIGMA_CCD_MAS)


def sigma_transit_mas(gmag):
    """Per-transit uncertainty in mas (8 CCDs averaged)."""
    return sigma_ccd_mas(gmag) / np.sqrt(N_CCD_PER_TRANSIT)


@dataclass(frozen=True)
class NoiseModel:
    """Noise actually added to w, relative to the reported sigma.

    Attributes
    ----------
    sigma_scale : float
        True per-transit std = sigma_scale * reported sigma (1 = correct error bars).
    jitter_max_mas : float
        Extra white noise with per-star std ~ U(0, jitter_max_mas) (gaiamock uses 0.04 for G<13).
    outlier_frac, outlier_scale : float
        Fraction of transits replaced by noise with std = outlier_scale * sigma.
    harmonic_amp_mas : dict[int, float]
        Spurious scan-angle signal sum_k a_k cos(k psi + phi_k), phi_k ~ U(0, 2 pi) per star.
        Odd k only, so that the signal flips sign with the scan direction like a real
        along-scan displacement.
    """

    name: str = "nominal"
    sigma_scale: float = 1.0
    jitter_max_mas: float = 0.0
    outlier_frac: float = 0.0
    outlier_scale: float = 10.0
    harmonic_amp_mas: dict[int, float] = field(default_factory=dict)

    def apply(self, rng, w_clean_mas, sigma_mas, psi_rad, mask):
        """Return noisy abscissae, shape like ``w_clean_mas`` (n_star, n_transit)."""
        n_star, n_tr = w_clean_mas.shape
        noise = self.sigma_scale * sigma_mas * rng.standard_normal((n_star, n_tr))
        if self.jitter_max_mas > 0:
            jit = rng.uniform(0.0, self.jitter_max_mas, (n_star, 1))
            noise += jit * rng.standard_normal((n_star, n_tr))
        if self.outlier_frac > 0:
            is_out = rng.uniform(size=(n_star, n_tr)) < self.outlier_frac
            out = self.outlier_scale * sigma_mas * rng.standard_normal((n_star, n_tr))
            noise = np.where(is_out, out, noise)
        for k, amp in self.harmonic_amp_mas.items():
            if k % 2 == 0:
                raise ValueError("use odd harmonics only (see docstring)")
            phase = rng.uniform(0.0, 2 * np.pi, (n_star, 1))
            noise += amp * np.cos(k * psi_rad + phase)
        return np.where(mask, w_clean_mas + noise, 0.0)


# The misspecification strengths below are placeholders chosen to be comparable to the
# per-transit sigma at G = 10-13 (~0.046 mas). Justify or replace them (e.g. from the DR4
# spurious_signal table) before reporting any robustness result.
NOISE_MODELS: dict[str, NoiseModel] = {
    m.name: m
    for m in (
        NoiseModel("nominal"),
        NoiseModel("gaiamock_jitter", jitter_max_mas=0.04),
        NoiseModel("sigma_inflated_1.2", sigma_scale=1.2),
        NoiseModel("sigma_deflated_0.8", sigma_scale=0.8),
        NoiseModel("outliers_2pct", outlier_frac=0.02, outlier_scale=10.0),
        NoiseModel("scan_harmonics", harmonic_amp_mas={3: 0.02, 5: 0.01, 7: 0.005}),
    )
}
# TODO(Phase 2): photocentre dilution / partially resolved luminous companions. In the
# unresolved limit dilution only rescales A, B, F, G (absorbed by the amplitude prior), so a
# meaningful test needs gaiamock's predict_astrometry_luminous_binary (al_bias_binary).


def get_noise_model(name: str) -> NoiseModel:
    try:
        return NOISE_MODELS[name]
    except KeyError as err:
        raise KeyError(f"unknown noise model {name!r}; choose from {sorted(NOISE_MODELS)}") from err
