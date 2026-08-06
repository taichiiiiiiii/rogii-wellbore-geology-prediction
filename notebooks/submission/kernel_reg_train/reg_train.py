# Phase 0 medal R&D: learned GR<->typewell registration (Kaggle GPU T4)
# Self-contained (loads only reg_dataset.npz). Batched + TCN refine (no GRU) for GPU speed.
import glob
import time

import numpy as np
import torch
import torch.nn as nn

print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)

DATA = glob.glob("/kaggle/input/**/reg_dataset.npz", recursive=True)[0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K_TW = 256
MAX_L = 3072       # cap eval rows per well for training (subsample if longer)
D = 64
BATCH = 8
EPOCHS = 30
FOLD = 0

d = np.load(DATA, allow_pickle=True)
folds = d["fold"]; nwell = len(d["used_wells"])
e_well = d["e_well"]; e_gr = d["e_gr"]; e_md = d["e_md"]; e_z = d["e_z"]; e_tgt = d["e_target"]
tw_gr_r = d["tw_gr"]; tw_tvt_r = d["tw_tvt"]


def tw_grid(w, k=K_TW):
    t = tw_tvt_r[w]; g = tw_gr_r[w]
    o = np.argsort(t); t = t[o]; g = g[o]
    grid = np.linspace(t[0], t[-1], k).astype(np.float32)
    return np.interp(grid, t, g).astype(np.float32), (grid / 100.0).astype(np.float32)


def well_arrays(w, cap):
    m = np.where(e_well == w)[0]
    if cap and len(m) > cap:
        m = m[np.linspace(0, len(m) - 1, cap).astype(int)]
    gg, grid = tw_grid(w)
    return (e_gr[m].astype(np.float32), (e_md[m] / 1000.0).astype(np.float32),
            (e_z[m] / 100.0).astype(np.float32), (e_tgt[m] / 100.0).astype(np.float32), gg, grid)


tr_wells = [w for w in range(nwell) if folds[w] != FOLD]
va_wells = [w for w in range(nwell) if folds[w] == FOLD]
print(f"wells train={len(tr_wells)} val={len(va_wells)} MAX_L={MAX_L} K={K_TW} batch={BATCH}", flush=True)
tr_cache = {w: well_arrays(w, MAX_L) for w in tr_wells}
va_cache = {w: well_arrays(w, None) for w in va_wells}   # full length for honest val
va_range = {w: float(e_tgt[e_well == w].max() - e_tgt[e_well == w].min()) for w in va_wells}


class RegNet(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.eval_enc = nn.Sequential(
            nn.Conv1d(3, d, 7, padding=3), nn.GELU(),
            nn.Conv1d(d, d, 7, padding=3), nn.GELU())
        self.tw_enc = nn.Sequential(
            nn.Conv1d(1, d, 7, padding=3), nn.GELU(),
            nn.Conv1d(d, d, 7, padding=3), nn.GELU())
        self.q = nn.Linear(d, d); self.kk = nn.Linear(d, d); self.scale = d ** -0.5
        self.log_temp = nn.Parameter(torch.tensor(1.6))  # learnable attn sharpening (exp~5)
        chs = d + 5  # eval_feat(d) + reg_soft + reg_peak + conf + md + z
        layers = []
        for dil in (1, 2, 4, 8, 16, 32, 64, 128):
            layers += [nn.Conv1d(chs, d, 5, padding=2 * dil, dilation=dil), nn.GELU()]
            chs = d
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(d, 1, 1)

    def forward(self, e_gr, e_md, e_z, tw_gr, tw_tvt):
        x = torch.stack([e_gr, e_md, e_z], 1)              # (B,3,L)
        ef = self.eval_enc(x)                              # (B,d,L)
        tf = self.tw_enc(tw_gr.unsqueeze(1))               # (B,d,K)
        Q = self.q(ef.transpose(1, 2))                     # (B,L,d)
        Kk = self.kk(tf.transpose(1, 2))                   # (B,K,d)
        logits = (Q @ Kk.transpose(1, 2)) * self.scale * torch.exp(self.log_temp)  # sharpened
        att = torch.softmax(logits, -1)                    # (B,L,K)
        reg_soft = att @ tw_tvt.unsqueeze(-1)              # (B,L,1) soft-retrieved TVT
        peak = att.argmax(-1, keepdim=True)                # (B,L,1) hard peak index
        reg_peak = torch.gather(tw_tvt.unsqueeze(1).expand(-1, att.size(1), -1), 2, peak)  # (B,L,1)
        conf = att.max(-1, keepdim=True).values            # (B,L,1) peak confidence
        seq = torch.cat([ef, reg_soft.transpose(1, 2), reg_peak.transpose(1, 2),
                         conf.transpose(1, 2), e_md.unsqueeze(1), e_z.unsqueeze(1)], 1)  # (B,d+5,L)
        out = self.head(self.tcn(seq)).squeeze(1) + reg_soft.squeeze(-1)   # residual around soft reg
        return out


def make_batch(wells, cache):
    arrs = [cache[w] for w in wells]
    L = max(a[0].shape[0] for a in arrs); B = len(arrs)
    gr = np.zeros((B, L), np.float32); md = np.zeros((B, L), np.float32)
    z = np.zeros((B, L), np.float32); tg = np.zeros((B, L), np.float32)
    msk = np.zeros((B, L), np.float32)
    twg = np.zeros((B, K_TW), np.float32); twt = np.zeros((B, K_TW), np.float32)
    for i, a in enumerate(arrs):
        n = a[0].shape[0]
        gr[i, :n] = a[0]; md[i, :n] = a[1]; z[i, :n] = a[2]; tg[i, :n] = a[3]; msk[i, :n] = 1
        twg[i] = a[4]; twt[i] = a[5]
    t = lambda x: torch.from_numpy(x).to(DEV)
    return t(gr), t(md), t(z), t(tg), t(msk), t(twg), t(twt)


net = RegNet().to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
huber = nn.SmoothL1Loss(beta=0.1, reduction="none")  # beta in /100 units (~10ft)
rng = np.random.RandomState(0)
best = {"vr": 1e9, "vt": 1e9, "ep": -1}


def evaluate():
    net.eval(); se = n = se_t = n_t = 0.0
    with torch.no_grad():
        for w in va_wells:
            a = va_cache[w]
            gr, md, z, tg, msk, twg, twt = make_batch([w], va_cache)
            out = net(gr, md, z, twg, twt)
            pred = out[0].cpu().numpy() * 100.0
            tgt = a[3] * 100.0
            sq = (pred - tgt) ** 2
            se += sq.sum(); n += len(tgt)
            if va_range[w] > 40.0:
                se_t += sq.sum(); n_t += len(tgt)
    tail = float(np.sqrt(se_t / n_t)) if n_t else float("nan")
    return float(np.sqrt(se / n)), tail


t0 = time.time()
for ep in range(EPOCHS):
    net.train(); order = list(tr_wells); rng.shuffle(order)
    tl = tn = 0.0; te = time.time()
    for i in range(0, len(order), BATCH):
        wb = order[i:i + BATCH]
        gr, md, z, tg, msk, twg, twt = make_batch(wb, tr_cache)
        out = net(gr, md, z, twg, twt)
        loss = (huber(out, tg) * msk).sum() / msk.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * msk.sum().item(); tn += msk.sum().item()
    sched.step()
    vr, vt = evaluate()
    if vr < best["vr"]:
        best = {"vr": vr, "vt": vt, "ep": ep}
    print(f"ep {ep:2d} train={tl/tn:.4f} val_RMSE={vr:.4f} tail_RMSE={vt:.3f} "
          f"temp={float(torch.exp(net.log_temp)):.1f} ({time.time()-te:.0f}s)", flush=True)
print(f"=== PHASE0 DONE fold{FOLD} BEST val={best['vr']:.4f} tail={best['vt']:.3f} "
      f"@ep{best['ep']} in {time.time()-t0:.0f}s ===")
print("GATE: val<9.5 (beats pf_blend 11.06, ~stack 9.14) & tail<14.17 -> PROCEED; else diagnose")
