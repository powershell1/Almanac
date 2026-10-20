"""Prior definitions: the single source of truth for simulator, flow targets and references.

Nonlinear parameters (learned by the flow): period_d, ecc, mean_anom0_rad (mean anomaly at the
reference epoch). Hyperparameter (marginalised): amp_scale_mas, the prior standard deviation of
each Thiele-Innes constant. Linear parameters: see ``gdr4.model.astrometry.LIN_NAMES``.
See docs/derivations.md, sections 3-5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from gdr4.model.astrometry import LIN_NAMES, N_5P, N_LIN
from gdr4.model.kepler import TWO_PI

FIVE_PAR_NAMES = LIN_NAMES[:N_5P]


# Probit map: a quantity uniform on (0, 1) becomes standard normal. We use it (rather than a
# logit) so that the prior on (u_P, u_e) is exactly N(0, 1), the flow's base distribution.
# With a logit the prior has exponential tails, which zuko's spline (identity outside its
# bounded window) cannot represent; that made training unstable (found 2026-10-20).
_probit = ndtri
_inv_probit = ndtr


def _log_std_normal(u):
    return -0.5 * u * u - 0.5 * np.log(2 * np.pi)


@dataclass(frozen=True)
class Prior:
    period_min_d: float
    period_max_d: float
    ecc_max: float
    amp_scale_min_mas: float
    amp_scale_max_mas: float
    gmag_min: float
    gmag_max: float
    five_par_mean: tuple[float, ...]  # order FIVE_PAR_NAMES
    five_par_std: tuple[float, ...]

    @classmethod
    def from_config(cls, cfg: dict) -> Prior:
        return cls(
            period_min_d=float(cfg["period_d"][0]),
            period_max_d=float(cfg["period_d"][1]),
            ecc_max=float(cfg["ecc_max"]),
            amp_scale_min_mas=float(cfg["amp_scale_mas"][0]),
            amp_scale_max_mas=float(cfg["amp_scale_mas"][1]),
            gmag_min=float(cfg["gmag"][0]),
            gmag_max=float(cfg["gmag"][1]),
            five_par_mean=tuple(float(cfg["five_par_mean"][k]) for k in FIVE_PAR_NAMES),
            five_par_std=tuple(float(cfg["five_par_std"][k]) for k in FIVE_PAR_NAMES),
        )

    # ---- sampling -------------------------------------------------------------------------

    def sample_nonlinear(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        """Draw n samples of (period_d, ecc, mean_anom0_rad, amp_scale_mas)."""
        log_p = rng.uniform(np.log(self.period_min_d), np.log(self.period_max_d), n)
        log_s = rng.uniform(np.log(self.amp_scale_min_mas), np.log(self.amp_scale_max_mas), n)
        return {
            "period_d": np.exp(log_p),
            "ecc": rng.uniform(0.0, self.ecc_max, n),
            "mean_anom0_rad": rng.uniform(0.0, TWO_PI, n),
            "amp_scale_mas": np.exp(log_s),
        }

    def linear_prior(self, amp_scale_mas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of the Gaussian prior on theta_lin, shape (n, 9) each."""
        amp = np.asarray(amp_scale_mas, dtype=np.float64)[..., None]
        mean = np.concatenate(
            [
                np.broadcast_to(self.five_par_mean, amp.shape[:-1] + (N_5P,)),
                np.zeros(amp.shape[:-1] + (N_LIN - N_5P,)),
            ],
            axis=-1,
        )
        std = np.concatenate(
            [
                np.broadcast_to(self.five_par_std, amp.shape[:-1] + (N_5P,)),
                np.broadcast_to(amp, amp.shape[:-1] + (N_LIN - N_5P,)),
            ],
            axis=-1,
        )
        return mean, std

    def sample_linear(self, rng: np.random.Generator, amp_scale_mas: np.ndarray) -> np.ndarray:
        """Draw theta_lin from exactly the Gaussian used in the analytic step (consistency rule)."""
        mean, std = self.linear_prior(amp_scale_mas)
        return mean + std * rng.standard_normal(mean.shape)

    def sample_gmag(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.gmag_min, self.gmag_max, n)

    # ---- unconstrained coordinates (flow targets, reference proposals) --------------------

    def to_unconstrained(self, period_d, ecc, mean_anom0_rad, amp_scale_mas=None):
        """Map to v = (u_P, u_e, M0[, u_s]) with u_P, u_e (and u_s) in R and M0 in [0, 2 pi).

        ``u_P = probit(fraction of the log-period range)`` and ``u_e = probit(e / e_max)``,
        where probit is the inverse standard-normal CDF. Under the prior, u_P and u_e are
        independent N(0, 1).
        """
        frac_p = (np.log(period_d) - np.log(self.period_min_d)) / self._log_p_range
        cols = [
            _probit(frac_p),
            _probit(np.asarray(ecc) / self.ecc_max),
            np.mod(mean_anom0_rad, TWO_PI),
        ]
        if amp_scale_mas is not None:
            cols.append(np.log(amp_scale_mas))
        return np.stack(cols, axis=-1)

    def from_unconstrained(self, v: np.ndarray) -> dict[str, np.ndarray]:
        out = {
            "period_d": np.exp(
                np.log(self.period_min_d) + _inv_probit(v[..., 0]) * self._log_p_range
            ),
            "ecc": self.ecc_max * _inv_probit(v[..., 1]),
            "mean_anom0_rad": np.mod(v[..., 2], TWO_PI),
        }
        if v.shape[-1] > 3:
            out["amp_scale_mas"] = np.exp(v[..., 3])
        return out

    def log_prob_unconstrained(self, v: np.ndarray) -> np.ndarray:
        """Prior log density of v = (u_P, u_e, M0, log s_a); -inf outside the support."""
        lp = _log_std_normal(v[..., 0]) + _log_std_normal(v[..., 1]) - np.log(TWO_PI)
        if v.shape[-1] > 3:
            lo, hi = np.log(self.amp_scale_min_mas), np.log(self.amp_scale_max_mas)
            inside = (v[..., 3] >= lo) & (v[..., 3] <= hi)
            lp = np.where(inside, lp - np.log(hi - lo), -np.inf)
        return lp

    @property
    def _log_p_range(self) -> float:
        return float(np.log(self.period_max_d) - np.log(self.period_min_d))
