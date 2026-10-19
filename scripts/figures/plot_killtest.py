"""Kill-test figures from <results_dir>/eval (written by scripts/evaluate.py).

Outputs coverage.png, period_recovery.png and alias_map.png into <results_dir>/figures/.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from style import INK_2, SERIES, plt  # noqa: E402

from gdr4.cli import base_parser, parse  # noqa: E402
from gdr4.eval.aliases import binned_fraction  # noqa: E402
from gdr4.eval.calibration import expected_coverage  # noqa: E402
from gdr4.utils import resolve  # noqa: E402


def coverage_figure(ev, metrics, noise_models, out):
    alphas = np.linspace(0, 1, 101)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    lv = np.load(ev / "per_star_nominal.npz")["hpd_level"]
    ax.fill_between(
        alphas,
        alphas - 0.05,
        alphas + 0.05,
        color=SERIES[0],
        alpha=0.12,
        lw=0,
        label="±5 pp (continue threshold)",
    )
    ax.plot(alphas, alphas, color=INK_2, lw=1, ls="--")
    ax.plot(alphas, expected_coverage(lv, alphas), color=SERIES[0], label="NPE, nominal noise")
    ax.set(
        xlabel="credibility level",
        ylabel="observed coverage (HPD)",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Expected coverage, frozen test set",
    )
    ax.legend(loc="upper left")
    errs = [metrics[n]["hpd_coverage_error_max"] * 100 for n in noise_models]
    y = np.arange(len(noise_models))
    ax2.barh(y, errs, color=SERIES[0], height=0.6)
    for yi, e in zip(y, errs, strict=True):
        ax2.text(e + 0.3, yi, f"{e:.1f}", va="center", color=INK_2, fontsize=9)
    ax2.axvline(5, color=INK_2, lw=1, ls="--")
    ax2.axvline(10, color=INK_2, lw=1, ls=":")
    ax2.set_yticks(y, noise_models)
    ax2.invert_yaxis()
    ax2.set(xlabel="max coverage error at 50/68/90% (pp)", title="Trained on nominal noise")
    fig.tight_layout()
    fig.savefig(out / "coverage.png")
    plt.close(fig)


def period_figure(ev, snr_key, snr_min, out):
    z = np.load(ev / "per_star_nominal.npz")
    hi = z[snr_key] > snr_min
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = [z["period_true"].min() * 0.8, z["period_true"].max() * 1.25]
    ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
    for sel, color, label in (
        (~hi, SERIES[1], f"S/N ≤ {snr_min:g}"),
        (hi, SERIES[0], f"S/N > {snr_min:g}"),
    ):
        ax.scatter(
            z["period_true"][sel],
            z["period_median"][sel],
            s=10,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=lim,
        ylim=lim,
        xlabel="true period (d)",
        ylabel="posterior median period (d)",
        title="Period recovery",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "period_recovery.png")
    plt.close(fig)


def alias_figure(ev, snr_key, out):
    z = np.load(ev / "per_star_nominal.npz")
    x, y = np.log10(z["period_true"]), np.log10(np.clip(z[snr_key], 1e-2, None))
    xe, ye = np.linspace(np.log10(30), np.log10(2000), 9), np.linspace(-1, 2.5, 8)
    frac, count = binned_fraction(x, y, z["alias_failure"], xe, ye)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.pcolormesh(
        xe, ye, frac.T, cmap="Blues", vmin=0, vmax=1, edgecolors="white", linewidth=2
    )
    for i in range(len(xe) - 1):
        for j in range(len(ye) - 1):
            if count[i, j] > 0:
                ax.text(
                    0.5 * (xe[i] + xe[i + 1]),
                    0.5 * (ye[j] + ye[j + 1]),
                    f"{int(count[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if frac[i, j] > 0.6 else INK_2,
                )
    ax.grid(False)
    ax.set(
        xlabel="log10 true period (d)",
        ylabel=f"log10 {snr_key}",
        title="Alias failure fraction (numbers = stars per cell)",
    )
    fig.colorbar(im, ax=ax, label="fraction with < 50% mass within 10% of P")
    fig.tight_layout()
    fig.savefig(out / "alias_map.png")
    plt.close(fig)


def main():
    args, cfg = parse(base_parser(__doc__))
    style.apply()
    res = resolve(cfg, "results_dir", args.root)
    ev, out = res / "eval", res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ev / "metrics.json").read_text())
    ecfg = cfg["evaluation"]
    coverage_figure(ev, metrics, cfg["test_set"]["noise_models"], out)
    period_figure(ev, ecfg["snr_key"], ecfg["snr_min"], out)
    alias_figure(ev, ecfg["snr_key"], out)
    print("figures in", out)


if __name__ == "__main__":
    main()
