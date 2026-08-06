"""Phase 0 (medal R&D) — LEARNED neural GR↔typewell registration, training + CV.

Model: eval GR sequence (1D-conv encoder) cross-attends over the typewell GR
profile (downsampled to K keypoints) to retrieve TVT, then a GRU refines the
per-row trajectory for continuity/anchor consistency. Target = TVT − anchor.

This is the genuinely-untried primitive (R32/R33): it LEARNS the registration
that PF/beam hand-code, so it can (in principle) follow large excursions (the
extreme-drift tail = 46% of SSE where PF loses the track) that the oracle over
our PF/beam candidates caps at 10.84.

Gate (Phase 0): pooled val RMSE (well-GroupKFold fold 0 held out) meaningfully
below pf_blend (~11) and competitive with the stack (9.14). If it can't beat
~11 it isn't learning the registration → diagnose/kill.

Run:  uv run python scripts/train_registration.py [--smoke PATH] [--epochs N]
      [--fold F] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise SystemExit(f"torch required: {e}")

ROOT = Path(__file__).resolve().parents[1]
K_TW = 256          # typewell keypoints (downsampled)
STRIDE_TRAIN = 4    # subsample eval rows during training (speed); eval on all
D = 64


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    return d


def tw_to_grid(tw_gr, tw_tvt, k=K_TW):
    """Downsample a ragged typewell (GR vs TVT) onto k evenly-TVT-spaced keypoints."""
    order = np.argsort(tw_tvt)
    t = tw_tvt[order]
    g = tw_gr[order]
    lo, hi = t[0], t[-1]
    grid = np.linspace(lo, hi, k).astype(np.float32)
    gg = np.interp(grid, t, g).astype(np.float32)
    return gg, grid  # (k,), (k,)


class RegNet(nn.Module):
    def __init__(self, d=D, k=K_TW):
        super().__init__()
        self.k = k
        # eval encoder: channels [gr, md_rel_norm, z_rel_norm]
        self.eval_enc = nn.Sequential(
            nn.Conv1d(3, d, 7, padding=3), nn.GELU(),
            nn.Conv1d(d, d, 7, padding=3, dilation=1), nn.GELU(),
            nn.Conv1d(d, d, 7, padding=6, dilation=2), nn.GELU(),
        )
        # typewell encoder: channel [gr]
        self.tw_enc = nn.Sequential(
            nn.Conv1d(1, d, 7, padding=3), nn.GELU(),
            nn.Conv1d(d, d, 7, padding=3), nn.GELU(),
        )
        self.q = nn.Linear(d, d)
        self.kk = nn.Linear(d, d)
        self.scale = d ** -0.5
        # refinement GRU over [eval_feat, reg_tvt, md_rel, z_rel]
        self.gru = nn.GRU(d + 3, d, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * d, 1)

    def forward(self, e_gr, e_md, e_z, tw_gr, tw_tvt):
        # e_*: (B, L)  tw_gr/tw_tvt: (B, K)
        x = torch.stack([e_gr, e_md, e_z], dim=1)          # (B,3,L)
        ef = self.eval_enc(x).transpose(1, 2)              # (B,L,d)
        tf = self.tw_enc(tw_gr.unsqueeze(1)).transpose(1, 2)  # (B,K,d)
        Q = self.q(ef)                                     # (B,L,d)
        Kk = self.kk(tf)                                   # (B,K,d)
        att = torch.softmax((Q @ Kk.transpose(1, 2)) * self.scale, dim=-1)  # (B,L,K)
        reg_tvt = att @ tw_tvt.unsqueeze(-1)               # (B,L,1) soft-retrieved TVT-anchor
        seq = torch.cat([ef, reg_tvt, e_md.unsqueeze(-1), e_z.unsqueeze(-1)], dim=-1)
        h, _ = self.gru(seq)
        out = self.head(h).squeeze(-1) + reg_tvt.squeeze(-1)  # residual around registration
        return out, reg_tvt.squeeze(-1)


def iter_wells(d, wells, stride=1):
    e_well = d["e_well"]
    e_gr, e_md, e_z, e_tgt = d["e_gr"], d["e_md"], d["e_z"], d["e_target"]
    tw_gr_r, tw_tvt_r = d["tw_gr"], d["tw_tvt"]
    for w in wells:
        m = np.where(e_well == w)[0]
        if stride > 1:
            m = m[::stride]
        gg, grid = tw_to_grid(tw_gr_r[w], tw_tvt_r[w])
        # normalize md/z to ~O(1)
        md = e_md[m] / 1000.0
        z = e_z[m] / 100.0
        yield (e_gr[m].astype(np.float32), md.astype(np.float32), z.astype(np.float32),
               e_tgt[m].astype(np.float32), gg, grid / 100.0, len(m))


def run(path, epochs, fold, device):
    d = load(path)
    folds = d["fold"]
    nwell = len(d["used_wells"])
    tr_wells = [w for w in range(nwell) if folds[w] != fold]
    va_wells = [w for w in range(nwell) if folds[w] == fold]
    print(f"wells train={len(tr_wells)} val={len(va_wells)} | device={device} K={K_TW}")

    net = RegNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    huber = nn.SmoothL1Loss(beta=5.0)

    def eval_pooled(wells, stride=1):
        net.eval()
        se = n = 0.0
        se_tail = n_tail = 0.0  # extreme-drift tail (per-well target range > 40ft = medal zone)
        with torch.no_grad():
            for gr, md, z, tgt, gg, grid, L in iter_wells(d, wells, stride):
                gr_t = torch.from_numpy(gr)[None].to(device)
                md_t = torch.from_numpy(md)[None].to(device)
                z_t = torch.from_numpy(z)[None].to(device)
                tg = torch.from_numpy(gg)[None].to(device)
                tt = torch.from_numpy(grid)[None].to(device)
                out, _ = net(gr_t, md_t, z_t, tg, tt)
                pred = out.squeeze(0).cpu().numpy() * 100.0  # grid was /100
                sqerr = (pred - tgt) ** 2
                se += float(sqerr.sum()); n += L
                if (tgt.max() - tgt.min()) > 40.0:
                    se_tail += float(sqerr.sum()); n_tail += L
        tail = float(np.sqrt(se_tail / n_tail)) if n_tail > 0 else float("nan")
        return float(np.sqrt(se / n)), tail

    rng = np.random.RandomState(0)
    for ep in range(epochs):
        net.train()
        order = list(tr_wells); rng.shuffle(order)
        t0 = time.time(); tl = tn = 0.0
        for w in order:
            for gr, md, z, tgt, gg, grid, L in iter_wells(d, [w], STRIDE_TRAIN):
                gr_t = torch.from_numpy(gr)[None].to(device)
                md_t = torch.from_numpy(md)[None].to(device)
                z_t = torch.from_numpy(z)[None].to(device)
                tg = torch.from_numpy(gg)[None].to(device)
                tt = torch.from_numpy(grid)[None].to(device)
                y = torch.from_numpy(tgt / 100.0)[None].to(device)  # match /100 scale
                out, _ = net(gr_t, md_t, z_t, tg, tt)
                loss = huber(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item() * L; tn += L
        vr, vtail = eval_pooled(va_wells, stride=1)
        print(f"ep {ep:2d} train_huber={tl/tn:.4f} val_pooled_RMSE={vr:.4f} "
              f"tail_RMSE={vtail:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return vr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=str, default=None, help="path to smoke npz")
    ap.add_argument("--data", type=str, default=str(ROOT / "outputs" / "reg_dataset.npz"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()
    path = Path(a.smoke) if a.smoke else Path(a.data)
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    run(path, a.epochs, a.fold, dev)
