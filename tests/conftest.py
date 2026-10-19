"""Shared fixtures. gaiamock-based cross-checks are skipped if gaiamock is not available.

Set ``GDR4_GAIAMOCK_DIR`` to a gaiamock checkout (default: ``external/gaiamock``).
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def gaiamock():
    path = Path(os.environ.get("GDR4_GAIAMOCK_DIR", REPO / "external" / "gaiamock"))
    if not (path / "gaiamock.py").exists():
        pytest.skip(f"gaiamock not found at {path}")
    sys.path.insert(0, str(path))
    try:
        return importlib.import_module("gaiamock")
    except ImportError as err:  # missing optional deps (healpy, ...)
        pytest.skip(f"cannot import gaiamock: {err}")


@pytest.fixture(scope="session")
def gaiamock_c(gaiamock):
    try:
        return gaiamock.read_in_C_functions()
    except (ValueError, OSError) as err:
        pytest.skip(f"gaiamock C functions not compiled: {err}")
