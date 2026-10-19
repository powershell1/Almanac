"""Record the environment (Phase 1 step 1): Python, GPU, and package versions -> JSON."""

import argparse
import importlib
import json
import platform
import subprocess
import sys

PACKAGES = [
    "numpy",
    "scipy",
    "astropy",
    "healpy",
    "torch",
    "zuko",
    "sklearn",
    "matplotlib",
    "joblib",
    "yaml",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="env_report.json")
    args = p.parse_args()
    rep = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in PACKAGES:
        try:
            rep["packages"][name] = importlib.import_module(name).__version__
        except Exception as err:  # noqa: BLE001
            rep["packages"][name] = f"MISSING ({err})"
    import torch

    rep["cuda"] = torch.cuda.is_available()
    rep["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        rep["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        rep["nvidia_smi"] = None
    rep["cpu_count"] = __import__("os").cpu_count()
    print(json.dumps(rep, indent=2))
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
