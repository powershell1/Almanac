"""Diagnostic from 2026-10-27 (see logs/research_log.md). Uses the local sanity-check run in
runs_local/ (toy scanning law, smoke_local_check config); paths are hard-coded on purpose.
Run from the repo root: python scripts/diagnostics/diag_c2st.py
"""

import sys

import numpy as np
import torch

from gdr4.data.shards import gather_padded, read_shard
from gdr4.eval.c2st import c2st_accuracy
from gdr4.eval.calibration import circular_embed
from gdr4.nn.data import to_torch_batch
from gdr4.nn.train import load_model
from gdr4.sim.priors import Prior
from gdr4.utils import load_config

cfg = load_config("configs/smoke_local_check.yaml")
prior = Prior.from_config(cfg["prior"])
CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs_local/checkpoints/local_check/best.pt"
REF = sys.argv[2] if len(sys.argv) > 2 else "runs_local/shards/smoke/reference"
TEST = sys.argv[3] if len(sys.argv) > 3 else "nominal"
model, _ = load_model(CKPT)
sh = read_shard(f"runs_local/shards/smoke/test/{TEST}", 0)
b = to_torch_batch(gather_padded(sh, np.arange(20)), prior)
torch.manual_seed(0)
v_npe = model.sample(b, 2000).numpy()  # (2000, 20, 3)
vt = b["v"].numpy()
snr = np.asarray(sh["snr_a0"])[:20]


def cstd(a):  # circular std
    R = np.abs(np.mean(np.exp(1j * a)))
    return np.sqrt(-2 * np.log(max(R, 1e-12)))


print(
    "star  S/N   ESS  | width ratio NPE/ref (uP, ue, M0) | offset uP (ref sd) | C2ST all  uP   ue   M0"
)
rng = np.random.default_rng(0)
rows = []
for i in np.argsort(snr):
    z = np.load(f"{REF}/star_{i:05d}.npz")
    ref = z["v"][rng.choice(len(z["v"]), 2000, replace=False), :3]
    npe = v_npe[:, i]
    w = [
        npe[:, 0].std() / ref[:, 0].std(),
        npe[:, 1].std() / ref[:, 1].std(),
        cstd(npe[:, 2]) / cstd(ref[:, 2]),
    ]
    off = (npe[:, 0].mean() - ref[:, 0].mean()) / ref[:, 0].std()
    full = c2st_accuracy(circular_embed(npe), circular_embed(ref), seed=1)
    per = [
        c2st_accuracy(npe[:, [0]], ref[:, [0]], seed=1),
        c2st_accuracy(npe[:, [1]], ref[:, [1]], seed=1),
        c2st_accuracy(
            np.c_[np.cos(npe[:, 2]), np.sin(npe[:, 2])],
            np.c_[np.cos(ref[:, 2]), np.sin(ref[:, 2])],
            seed=1,
        ),
    ]
    print(
        f"{i:4d} {snr[i]:5.1f} {z['ess']:6.0f} | {w[0]:5.2f} {w[1]:5.2f} {w[2]:5.2f}            | {off:+6.2f}             | {full:.2f}     {per[0]:.2f} {per[1]:.2f} {per[2]:.2f}"
    )
