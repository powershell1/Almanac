"""What physical prior do i.i.d. Gaussian Thiele-Innes constants imply? (docs/derivations.md s.4)

Draws A, B, F, G ~ N(0, 1) and plots the implied a0 (in units of s_a), cos i, omega and Omega.
For reference, an isotropic orientation has cos i, omega, Omega uniform. Output:
results/prior_checks/ti_prior.png. This is a prior diagnostic, not a research result.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from style import INK_2, SERIES, plt  # noqa: E402

from gdr4.model.astrometry import campbell_from_thiele_innes  # noqa: E402
from gdr4.utils import REPO_ROOT  # noqa: E402


def main():
    style.apply()
    rng = np.random.default_rng(0)
    a0, node, argp, inc = campbell_from_thiele_innes(*rng.standard_normal((4, 400_000)))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    panels = [
        (a0, "a0 / s_a", None),
        (np.cos(inc), "cos i", (-1, 1)),
        (argp, "omega (rad)", (0, 2 * np.pi)),
        (node, "Omega (rad)", (0, np.pi)),
    ]
    for ax, (x, label, rng_) in zip(axes, panels, strict=True):
        ax.hist(x, bins=80, range=rng_, density=True, color=SERIES[0])
        if rng_ is not None:
            ax.axhline(1 / (rng_[1] - rng_[0]), color=INK_2, ls="--", lw=1, label="isotropic")
            ax.legend(loc="lower center")
        ax.set(xlabel=label, yticks=[])
    fig.suptitle("Orientation and size implied by i.i.d. Gaussian A, B, F, G", y=1.02)
    fig.tight_layout()
    out = REPO_ROOT / "results" / "prior_checks"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ti_prior.png", bbox_inches="tight")
    print("wrote", out / "ti_prior.png")


if __name__ == "__main__":
    main()
