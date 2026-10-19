"""Build the scan-geometry cache (one .npz) from gaiamock's scanning law or the toy law.

    python scripts/build_scan_cache.py --config configs/killtest.yaml \
        --gaiamock-dir /kaggle/input/gaiamock-scans/gaiamock
    python scripts/build_scan_cache.py --config configs/smoke.yaml   # toy law, tests only
"""

from gdr4.cli import base_parser, parse
from gdr4.sim.scan import build_cache_from_gaiamock, build_cache_toy_nsl
from gdr4.utils import resolve


def main():
    p = base_parser(__doc__)
    p.add_argument("--gaiamock-dir", help="gaiamock checkout containing healpix_scans/")
    p.add_argument("--n-jobs", type=int, default=4)
    args, cfg = parse(p)
    out = resolve(cfg, "scan_cache", args.root)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = cfg["scan"]["source"]
    if src == "gaiamock":
        if not args.gaiamock_dir:
            p.error("--gaiamock-dir is required for scan.source=gaiamock")
        cache = build_cache_from_gaiamock(
            args.gaiamock_dir,
            n_jobs=args.n_jobs,
            commit=cfg["scan"].get("gaiamock_commit", "unknown"),
        )
    elif src == "toy_nsl":
        cache = build_cache_toy_nsl(nside=int(cfg["scan"]["nside"]))
    else:
        p.error(f"unknown scan.source {src!r}")
    cache.save(out)
    n = cache.n_transits
    print(
        f"{out}: source={cache.source} n_pix={cache.n_pix} transits min/median/max = "
        f"{n.min()}/{int(sorted(n)[len(n) // 2])}/{n.max()}"
    )


if __name__ == "__main__":
    main()
