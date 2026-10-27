"""High-S/N C2ST failures (2026-10-28): are the e / M0 anchors right, and are they used?

For the high-S/N reference stars, compares the anchors given to the network (u_e*, tau_e,
M0*, tau_M0 of the top period anchor, after noise rescaling) with the exact posterior, measures
the phase width at fixed period (NPE vs exact), and the mixture weight the phase head puts on
its anchored components.
Run: python scripts/diagnostics/diag_high_snr.py <checkpoint> <reference_dir> [test_set]
"""

import sys

import numpy as np
import torch

from gdr4.data.shards import gather_padded, read_shard
from gdr4.nn.data import to_torch_batch
from gdr4.nn.train import load_model
from gdr4.sim.priors import Prior
from gdr4.utils import load_config

CKPT, REF = sys.argv[1], sys.argv[2]
TEST = sys.argv[3] if len(sys.argv) > 3 else "nominal"
prior = Prior.from_config(load_config("configs/smoke_local_check.yaml")["prior"])
model, _ = load_model(CKPT)
sh = read_shard(f"runs_local/shards/smoke/test/{TEST}", 0)
snr = np.asarray(sh["snr_a0"])[:20]
stars = [i for i in np.argsort(snr) if snr[i] >= 10]
b = to_torch_batch(gather_padded(sh, np.array(stars)), prior)
torch.manual_seed(0)
with torch.no_grad():
    ctx, anc = model.context(b)
    v = model.sample(b, 4000).numpy()


def cmean(a):
    return np.angle(np.mean(np.exp(1j * a)))


def cstd(a):
    return np.sqrt(-2 * np.log(max(np.abs(np.mean(np.exp(1j * a))), 1e-12)))


def wrap(a):
    return np.angle(np.exp(1j * a))


print(
    " star   S/N | u_e anchor: offset  tau/sd | M0 anchor: offset  tau/sd valid | "
    "M0 width at fixed P: NPE/exact | anchored phase weight"
)
for j, i in enumerate(stars):
    z = np.load(f"{REF}/star_{i:05d}.npz")["v"]
    ue_m, ue_s = z[:, 1].mean(), z[:, 1].std()
    m0_m, m0_s = cmean(z[:, 2]), cstd(z[:, 2])
    ef = anc["ecc_feat"][j].numpy().reshape(-1, 3)[0]  # top anchor: u_e/3, log tau_e/5, valid
    ue_a, tau_e = 3 * ef[0], np.exp(5 * ef[1])
    m0_a, tau_m = anc["m0"][j, 0].item(), np.exp(anc["log_tau_m0"][j, 0].item())
    valid_m = bool(anc["valid_m0"][j, 0])
    # phase width at (nearly) fixed period: samples within +-0.5 exact sd of the exact centre
    up_m, up_s = z[:, 0].mean(), z[:, 0].std()
    sel_r = np.abs(z[:, 0] - up_m) < 0.5 * up_s
    sel_n = np.abs(v[:, j, 0] - up_m) < 0.5 * up_s
    w_r, w_n = cstd(z[sel_r, 2]), cstd(v[sel_n, j, 2]) if sel_n.sum() > 50 else np.nan
    # mixture weight on the anchored von Mises components, averaged over the NPE samples
    feat = model.period.features(
        torch.tensor(v[:, j, 0])[:, None].float(), {k: x[j : j + 1] for k, x in anc.items()}
    )
    cond = torch.cat([feat, torch.tensor(v[:, j, 1])[:, None, None].float()], -1)
    with torch.no_grad():
        log_w, _, _ = model.vm.params_anchored(
            cond, ctx[j : j + 1].expand(4000, 1, -1), {k: x[j : j + 1] for k, x in anc.items()}
        )
    w_anch = torch.exp(log_w[..., model.vm.k :]).sum(-1).mean().item()
    print(
        f"{i:4d} {snr[i]:6.1f} | {(ue_a - ue_m) / ue_s:+12.1f} {tau_e / ue_s:7.2f} |"
        f" {wrap(m0_a - m0_m) / m0_s:+11.1f} {tau_m / m0_s:7.2f} {valid_m!s:5s} |"
        f" {w_n / w_r:22.2f}         | {w_anch:6.2f}"
        f"   (n at fixed P: NPE {sel_n.sum()}, exact {sel_r.sum()})"
    )
