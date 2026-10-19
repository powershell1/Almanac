#!/usr/bin/env bash
# Reproduce the local test environment in ./.env (Miniforge = conda-forge only).
# Usage: bash scripts/setup_local_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD/.env"
mkdir -p "$ROOT"
if [ ! -x "$ROOT/miniforge3/bin/conda" ]; then
    curl -sL -o "$ROOT/Miniforge3.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$ROOT/Miniforge3.sh" -b -p "$ROOT/miniforge3"
    rm "$ROOT/Miniforge3.sh"
fi
if [ ! -x "$ROOT/gdr4/bin/python" ]; then
    # gsl + gcc are only needed to compile gaiamock's C Kepler solver (used in a cross-check test).
    "$ROOT/miniforge3/bin/conda" create -y -p "$ROOT/gdr4" python=3.11 \
        numpy scipy astropy healpy pyyaml pytest ruff matplotlib scikit-learn joblib gsl gcc
    # CPU-only torch keeps the download small; Kaggle already ships a CUDA build.
    "$ROOT/gdr4/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
    "$ROOT/gdr4/bin/pip" install zuko
fi
"$ROOT/gdr4/bin/pip" install -e . --no-deps
echo "Done. Activate with:  source .env/miniforge3/bin/activate .env/gdr4"
