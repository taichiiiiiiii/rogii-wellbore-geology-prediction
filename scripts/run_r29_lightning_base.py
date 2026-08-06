"""R29: measure the *real* well-GroupKFold pooled RMSE of the public plateau
"lb-7-776-rogii-ridge-sp" notebook's pipeline, with its leak branch disabled.

Ported (not copy-pasted verbatim) from
``lightningv08/lb-7-776-rogii-ridge-sp (Kaggle)lb-7-776-rogii-ridge-sp.ipynb``.

Mission: the public notebook plateau (LB 7.09-7.16) reads ~2ft better than our
own well-GroupKFold CV base (9.1456 / LB 9.014). Cell 27 of that notebook has a
hidden-test dead branch::

    if wid in train_wells:
        ...
        tvt_phys = tvt_from_contacts(hw_tr, tw_tr)   # copies the *actual* train row

Locally the 3 bundled ``data/raw/test`` wells happen to duplicate train wells,
so this branch fires and returns the true answer verbatim -- a leak that never
fires against the real hidden test set (disjoint well ids). This script
**always** takes the ``else`` branch (the real, leak-free heuristic/ML
pipeline) and measures its pooled RMSE with our own well-GroupKFold(5, seed=42)
harness (``rogii.cv.well_folds``), never the notebook's local-CV numbers.

Two scored components, blended 0.3 / 0.7 exactly as the notebook does:

  * ``sub_1`` -- LightGBM(x3) + CatBoost(x2) -> Ridge meta-stack over engineered
    features (PF/beam/spatial-donor registration signals), + tuned postprocess
    blend with the raw PF-ANCC track. Trained with a *nested* well-grouped CV:
    a fit/early-stopping split is carved out of each fold's training wells so
    early stopping never sees the scored (validation) fold.
  * ``sub_2`` -- the "heuristic model": 128-seed likelihood-weighted particle
    filter + 14-config beam search + n_eval/z_span selector blend, applied
    directly to each well's own known zone (no cross-well fitting needed, so
    it is evaluated directly on every well, fold assignment is irrelevant to
    it -- this mirrors exactly what the notebook does at inference time for a
    genuine unseen test well).

GPU device settings from the original notebook (``device_type="gpu"`` /
``task_type="GPU"``) are switched to CPU. The ravaghi pretrained-artifact
short-circuit (``CFG.artifacts_path`` / ``koolbox.Trainer`` disk cache) is
removed entirely -- everything is trained from scratch against our own data.

Usage::

    uv run python scripts/run_r29_lightning_base.py --n-wells 30 --out /tmp/r29_smoke.json
    uv run python scripts/run_r29_lightning_base.py --all --out /tmp/r29_full.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import data as D  # noqa: E402
from rogii.cv import well_folds  # noqa: E402

warnings.filterwarnings("ignore")

NCPU = min(4, __import__("multiprocessing").cpu_count())

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10
DENSE_SPW = 60
DENSE_K = 20
N_SPLITS = 5
CV_SEED = 42

BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

PF_N = 600
ANCC_N = 600
PF_MOM = 0.993
PF_VN = 0.005
PF_PN = 0.01
PF_GR_SIG_MIN = 10.0
PF_GR_SIG_MAX = 60.0
PF_GR_SIG_DEF = 30.0
PF_INIT_V_STD = 0.02
PF_INIT_SPR = 0.5
PF_RESAMP = 0.5
PF_ROUGH_P = 0.2
PF_ROUGH_V = 0.003
PF_GR_WIN = 5
PF_GR_WT = 0.3
ANCC_ALPHA = 0.998
ANCC_RN = 0.002
ANCC_PN = 0.005
ANCC_IR = 0.01
ANCC_IS = 0.3
ANCC_RP = 0.1
ANCC_RR = 0.001

ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)

# ---- Cell 6: "heuristic model" (sub_2) constants -----------------------

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)

SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}

SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

BEAM_CONFIGS_NP = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]


# =========================================================================
# Cell 6 (verbatim port): PF likelihood ensemble + beam ensemble + selector
# =========================================================================


def run_particle_filter(hw: pd.DataFrame, tw: pd.DataFrame, n_particles: int = 500, seed: int = 42):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy(), 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_Z = float(last["Z"])
    last_MD = float(last["MD"])

    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + 4.5 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N

    MOM = 0.998
    VN = 0.002
    PN = 0.005
    RP = 0.1
    RR = 0.001
    RESAMP = 0.5

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw["TVT_input"].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.0))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):
    preds = []
    liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = (weights[:, None] * pred_arr).sum(0)
    out["pf_mean"] = pred_arr.mean(0)
    return out


def beam_search_np(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC = mc * np.array([2.0, 1.0, 0.0, 1.0, 2.0])

    bidx = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.0
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci]) ** 2 / es
        tot = bcost[:bn, None] + gr_e + MC[None, :]
        tot = np.where(valid, tot, np.inf)

        ni_f = ni.flatten()
        tot_f = tot.flatten()
        vf = valid.flatten()
        ni_f = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top = top[np.argsort(tot_u[top])]

        bidx[:kept] = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]["TVT_input"])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    hgr = gr_all[ev.index]

    beam_results = [
        beam_search_np(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r) in BEAM_CONFIGS_NP
    ]

    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw["TVT_input"].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split("_")
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if "beam" in parts:
        beam_weight = float(parts[parts.index("beam") + 1])
    if "hold" in parts:
        hold_weight = float(parts[parts.index("hold") + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}")
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split("_beam_")[0].split("_hold_")[0]]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


def heuristic_predict(hw: pd.DataFrame, tw: pd.DataFrame, n_seeds: int = 128) -> np.ndarray:
    """sub_2 equivalent, leak branch **always disabled** (always the notebook's
    ``else`` path -- the real, leave-self-out heuristic, never
    ``tvt_from_contacts`` on a copy of this well's own train row)."""
    try:
        pf_by_scale = run_pf_lik_ensemble_scales(hw, tw, n_particles=500, n_seeds=n_seeds)
        tvt_pf = pf_by_scale[f"pf_scale_{8:g}"]
    except Exception:
        last_known = hw["TVT_input"].dropna()
        last_val = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
        tvt_pf = hw["TVT_input"].fillna(last_val).values.astype(float)
        pf_by_scale = {f"pf_scale_{scale:g}": tvt_pf.copy() for scale in SELECTOR_SCALES}

    try:
        tvt_beam = run_beam_ensemble(hw, tw)
    except Exception:
        tvt_beam = tvt_pf.copy()

    selector_code, selector_variant, n_eval, z_span = selector_well_code(hw)
    last_known = hw["TVT_input"].dropna()
    last_known_tvt = float(last_known.iloc[-1]) if len(last_known) > 0 else float(np.nanmean(tvt_pf))
    tvt_selector = apply_selector_variant(selector_variant, pf_by_scale, tvt_beam, last_known_tvt)

    mask = D.eval_mask(hw)
    return np.asarray(tvt_selector, dtype=float)[mask]


# =========================================================================
# Cell 7 (verbatim port): numba PF/beam + registration features for the
# LGBM/CatBoost/Ridge stack (sub_1)
# =========================================================================


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr)
    nt = len(tw_gr)
    MAX = BS * 6
    bidx = np.zeros(BS, np.int64)
    bidx[0] = si
    bcost = np.full(BS, 1e30)
    bcost[0] = 0.0
    bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64)
    hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64)
    cC = np.full(MAX, 1e30)
    cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]
        nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                        cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni
                        cC[nc] = tot
                        cP[nc] = bi
                        nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]
        hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
        b = hP[s, b]
    return path


@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.0)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 50.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va**0.5
        pm = md_v[i]
    return pts, std_


@njit(cache=True)
def _pf_z(
    md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N, MOM, VN, PN, GR_WT, RP, RV, RESAMP
):
    pos = np.empty(N)
    vel = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0)
            pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.0)
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0, 1e-300)
                    lk = (1.0 - GR_WT) * lp + GR_WT * ls
                else:
                    lk = lp
                lk = max(lk, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0, 1e-300)
            w[j] *= lz
            ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1.0 / N
        wm = 0.0
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va**0.5
        pm = md_v[i]
        pz = z_v[i]
    return pts, std_


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return float(PF_GR_SIG_DEF)
    return float(
        np.clip(
            np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)),
            PF_GR_SIG_MIN,
            PF_GR_SIG_MAX,
        )
    )


def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)


def beam_search_jit(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc(
        ev["MD"].values.astype(np.float64),
        ev["Z"].values.astype(np.float64),
        ev["GR"].values.astype(np.float64),
        gg,
        gmin,
        gst,
        gs,
        ls,
        ir,
        N,
        ANCC_ALPHA,
        ANCC_RN,
        ANCC_PN,
        ANCC_IS,
        ANCC_RP,
        ANCC_RR,
        PF_RESAMP,
    )
    return pts.astype(np.float32), std.astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    dz_k = np.diff(kna["Z"].values)
    dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values)
    m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]
        vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1
    t2 = kna.tail(20)
    dvt2 = np.diff(t2["TVT_input"].values)
    dmd2 = np.diff(t2["MD"].values)
    m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(
        ev["MD"].values.astype(np.float64),
        ev["Z"].values.astype(np.float64),
        ev["GR"].values.astype(np.float64),
        gr_sm.loc[ev.index].values.astype(np.float64),
        gg,
        gs2,
        gmin,
        gst,
        gs,
        float(kna["TVT_input"].iloc[-1]),
        iv,
        beta,
        icpt,
        zsig,
        N,
        PF_MOM,
        PF_VN,
        PF_PN,
        PF_GR_WT,
        PF_ROUGH_P,
        PF_ROUGH_V,
        PF_RESAMP,
    )
    return pts.astype(np.float32), std.astype(np.float32)


def _warmup_numba():
    _md = np.linspace(1, 50, 20, np.float64)
    _z = np.zeros(20, np.float64)
    _gr = np.full(20, 50.0, np.float64)
    _gg = np.linspace(45, 55, 100, np.float64)
    _pf_ancc(_md, _z, _gr, _gg, 45.0, 0.1, 20.0, 50.0, 0.0, 8, 0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
    _pf_z(_md, _z, _gr, _gr, _gg, _gg, 45.0, 0.1, 20.0, 50.0, 0.0, -1.0, 0.0, 0.1, 8, 0.993, 0.005, 0.01, 0.3, 0.2, 0.003, 0.5)
    _beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15.0, 100.0)


def robust_slope(x, y, w=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50) :])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[: max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1 : max(t1 + 1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.0 * scores)
    sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1)
        ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]
        yn = self.ya[ik]
        fn = self.fa[ik]
        wx = w * xn
        wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1)
        A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9
        A[:, 1, 1] += 1e-9
        A[:, 2, 2] += 1e-9
        rhs = np.stack(
            [(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1
        )
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]
        Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s["X"].values)
            ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values)
            wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1)
        ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]
        ap = (an * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return (
            ap.astype(np.float32),
            np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
            np.where(vk, dk, np.inf).min(1).astype(np.float32),
        )


def build_well(hw_path, tw_path, is_train, FI, DI):
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception:
        return None
    if is_train and "TVT" not in hw.columns:
        return None
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    if is_train and hw["TVT"].isna().all():
        return None
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None

    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0:
        return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32)
    std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0] :].to_numpy(np.float32)
    kgr = gr_full.iloc[: len(kn)].to_numpy(np.float32)

    bpaths = {}
    for bs, mc, es, r, tag in BEAMS:
        bpaths[tag] = beam_search_jit(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0

    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]
    sc15, sc15s = sc_res[1]
    sc25, sc25s = sc_res[2]
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn["MD"].to_numpy(np.float32)
    kz = kn["Z"].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = robust_slope(kmd, ktvt)
    slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)

    swid = wid if is_train else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = FI.impute(xy_kn, self_wid=swid)
    z_kn = kn["Z"].to_numpy(np.float32)
    z_ev = ev["Z"].to_numpy(np.float32)

    tvt_fs = {}
    form_rmse = {}
    form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"] = tvt_f
        tvt_fs[f"tvtFw_{fn}"] = tvt_fw
        tvt_fs[f"tvtF50_{fn}"] = tvt_f50
        tvt_fs[f"bw_{fn}"] = np.float32(b_full)
        tvt_fs[f"bww_{fn}"] = np.float32(b_wls)
        tvt_fs[f"bw50_{fn}"] = np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"] = np.float32(b_early)
        tvt_fs[f"bw_mid_{fn}"] = np.float32(b_mid)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = DI.impute(xy_kn, self_wid=swid)
    b_vd = ktvt + z_kn - d_kn
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn)
    b_d = float(np.median(b_vd))
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + z_kn - d_kn
    d_rmse = float(np.sqrt(np.mean(res_kn**2)))
    d_bias = float(np.mean(res_kn))
    d_nb_std = float(np.mean(d_std_kn))

    all_sigs = [pf_use] + list(bpaths.values()) + [sc8, sc15, sc25, sc_ens, tvt_fs["tvtF_ANCC"], tvt_dense]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values)
    rolls = {}
    for w in [5, 21, 51, 101]:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f"grm{w}"] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f"grs{w}"] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f"glag{lag}"] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f"glead{lag}"] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum((gr_s**2).rolling(21, center=True, min_periods=1).mean(), 0.0)).iloc[
        ev.index
    ].values.astype(np.float32)

    hmd = ev["MD"].to_numpy(np.float32)
    md_since = hmd - float(lk["MD"])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)

    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    nh = len(ev)
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    def sc(v):
        return np.full(nh, np.float32(v), np.float32)

    feats = {
        "well": wid,
        "id": [f"{wid}_{i}" for i in ev.index],
        "last_known_tvt": sc(last_tvt),
        "pf_ancc": pf_use,
        "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
        "pf_z": (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        "pf_z_delta": ((pf_z - last_tvt).astype(np.float32) if has_z else sc(0.0)),
        "pf_vs_z": ((pf_use - pf_z.astype(np.float32)) if has_z else sc(0.0)),
        **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        "beam_mean_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).mean(1).astype(np.float32),
        "beam_std_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).std(1).astype(np.float32),
        "beam_med_d": np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(np.float32),
        "sc8_d": (sc8 - np.float32(last_tvt)).astype(np.float32),
        "sc8_sc": sc8s,
        "sc15_d": (sc15 - np.float32(last_tvt)).astype(np.float32),
        "sc15_sc": sc15s,
        "sc25_d": (sc25 - np.float32(last_tvt)).astype(np.float32),
        "sc25_sc": sc25s,
        "sc_cons_d": (sc_cons - np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d": (sc_ens - np.float32(last_tvt)).astype(np.float32),
        "sc_trust": sc(sc_trust),
        "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        "sig_std": sig_std,
        "sig_mean_d": sig_mean,
        **tvt_fs,
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d,
        "form_std_d": form_std_d,
        "form_rng_d": form_rng_d,
        "spatial_ancc_d": (form_ev[:, 0] - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))),
        "spatial_knn_dist": knn_d,
        "dense_ancc": d_ancc,
        "dense_std": d_std,
        "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
        "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse),
        "dense_bias": sc(d_bias),
        "dense_nb_std": sc(d_nb_std),
        "pf_vs_spatial": (pf_use - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "pf_vs_dense": (pf_use - tvt_dense).astype(np.float32),
        "spatial_vs_dense": (tvt_fs["tvtF_ANCC"] - tvt_dense).astype(np.float32),
        "beam_vs_spatial": (bpaths["cons"] - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam": (sc_ens - bpaths["cons"]).astype(np.float32),
        "cal_a": sc(a_cal),
        "cal_b": sc(b_cal),
        "pfx_rmse": sc(pfx_rmse),
        "known_len": sc(len(kn)),
        "eval_len": sc(nh),
        "slp_all": sc(slp_all),
        "slp_50": sc(slp_50),
        "slp_z": sc(slp_z),
        "slp_b_d_all": (slp_b_all - last_tvt).astype(np.float32),
        "slp_b_d_50": (slp_b_50 - last_tvt).astype(np.float32),
        "ktvt_range": sc(float(np.ptp(ktvt))),
        "ktvt_std": sc(float(ktvt.std())),
        "md_since": md_since,
        "frac": frac,
        "frac2": frac**2,
        "sqrt_frac": np.sqrt(frac),
        "z": z_ev,
        "dx": (ev["X"] - float(lk["X"])).to_numpy(np.float32),
        "dy": (ev["Y"] - float(lk["Y"])).to_numpy(np.float32),
        "dz": (z_ev - float(lk["Z"])).astype(np.float32),
        "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2).to_numpy(np.float32),
        "dzdmd": dzdmd,
        "dxdmd": dxdmd,
        "dydmd": dydmd,
        "gr": hgr,
        "gr_d1": gr_d1,
        "gr_d2": gr_d2,
        "gr_env": gr_env,
        "gr_nrg": gr_nrg,
        "gr_vs_tw_anc": hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        "gr_vs_slp_all": hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
        **{f"tda{int(o)}": hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}": hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}": hgr - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}": hgr - np.interp(pf_use + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range": sc(float(np.ptp(tw_tvt))),
        "tw_gr_mean": sc(float(tw_gr.mean())),
    }
    for k, v in rolls.items():
        feats[k] = v
    result = pd.DataFrame(feats)
    if is_train:
        if "TVT" not in ev.columns or ev["TVT"].isna().all():
            return None
        result["target"] = ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt)
    return result


def build_dataset(wells, split, FI, DI, label, n_jobs=None):
    """Build the LGBM/CatBoost feature matrix. ``n_jobs=1`` processes wells
    serially so the DenseANCCImputer's ~O(eval_rows x nfetch) query arrays
    (hundreds of MB per well) never stack up concurrently -- essential on the
    3.8GB local box; ``None`` uses NCPU threads (fast, higher peak RSS)."""
    if n_jobs is None:
        n_jobs = NCPU
    root = D.data_root()
    args = [
        (root / split / f"{w}__horizontal_well.csv", root / split / f"{w}__typewell.csv", split == "train")
        for w in wells
    ]
    t0 = time.time()
    if n_jobs == 1:
        res = [build_well(hp, tp, it, FI, DI) for hp, tp, it in args]
    else:
        res = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
            delayed(build_well)(hp, tp, it, FI, DI) for hp, tp, it in args
        )
    parts = [r for r in res if r is not None]
    n_failed = len(args) - len(parts)
    print(f"[R29] build_dataset({label}): {len(parts)}/{len(args)} wells built in {time.time() - t0:.1f}s"
          f" ({n_failed} failed/skipped, n_jobs={n_jobs})")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =========================================================================
# Nested well-grouped CV training for the LGBM/CatBoost/Ridge stack (sub_1)
# =========================================================================


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    ordered = sorted(train_wells)
    random.Random(seed).shuffle(ordered)
    n_es = max(1, round(len(ordered) * es_frac))
    es_wells = sorted(ordered[:n_es])
    fit_wells = sorted(ordered[n_es:])
    return fit_wells, es_wells


def lgb_params_cpu(n_estimators_cap: int | None) -> list[dict]:
    params = [
        dict(
            boosting_type="gbdt",
            num_leaves=255,
            min_child_samples=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            reg_alpha=0.05,
            objective="regression",
            verbose=-1,
            n_jobs=NCPU,
            learning_rate=0.030,
            n_estimators=5000,
            seed=123,
        ),
        dict(
            n_jobs=NCPU,
            verbose=-1,
            reg_alpha=10.788188919840913,
            subsample=0.47437582748953966,
            num_leaves=64,
            reg_lambda=95.75401894533888,
            n_estimators=10000,
            random_state=0,
            boosting_type="gbdt",
            learning_rate=0.00934485794382918,
            colsample_bytree=0.39283351290380497,
            min_child_weight=0.24081152127177283,
            min_child_samples=40,
        ),
        dict(
            n_jobs=NCPU,
            verbose=-1,
            reg_alpha=10.788188919840913,
            subsample=0.47437582748953966,
            num_leaves=64,
            reg_lambda=95.75401894533888,
            n_estimators=10000,
            random_state=29,
            boosting_type="gbdt",
            learning_rate=0.00934485794382918,
            colsample_bytree=0.39283351290380497,
            min_child_weight=0.24081152127177283,
            min_child_samples=40,
        ),
    ]
    if n_estimators_cap is not None:
        for p in params:
            p["n_estimators"] = min(p["n_estimators"], n_estimators_cap)
    return params


def cb_params_cpu(iterations_cap: int | None) -> list[dict]:
    params = [
        dict(
            iterations=8000,
            depth=7,
            l2_leaf_reg=2.0,
            min_data_in_leaf=15,
            border_count=254,
            loss_function="RMSE",
            task_type="CPU",
            od_type="Iter",
            od_wait=300,
            verbose=0,
            learning_rate=0.020,
            random_seed=7,
            thread_count=NCPU,
        ),
        dict(
            iterations=8000,
            depth=7,
            l2_leaf_reg=2.0,
            min_data_in_leaf=15,
            border_count=254,
            loss_function="RMSE",
            task_type="CPU",
            od_type="Iter",
            od_wait=300,
            verbose=0,
            learning_rate=0.030,
            random_seed=123,
            thread_count=NCPU,
        ),
    ]
    if iterations_cap is not None:
        for p in params:
            p["iterations"] = min(p["iterations"], iterations_cap)
    return params


RIDGE_PARAMS = {
    "random_state": 42,
    "alpha": 1.6602834637650032,
    "tol": 0.0005030247295617308,
    "positive": True,
    "fit_intercept": True,
}

PP_PARAMS = {"alpha": 1.0, "tau": 85, "w_pf": 0.09}


def fit_oof_lgbm(X, y, wells_arr, folds, params, es_frac=0.15, es_seed=42):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    oof = np.full(len(y), np.nan)
    for fold_i, val_wells in enumerate(folds):
        val_mask = wells_arr.isin(val_wells).to_numpy()
        train_wells_k = sorted(set(wells_arr.unique()) - set(val_wells))
        fit_wells, es_wells = _split_fit_es(train_wells_k, es_frac, es_seed + fold_i)
        fit_mask = wells_arr.isin(fit_wells).to_numpy()
        es_mask = wells_arr.isin(es_wells).to_numpy()

        model = LGBMRegressor(**params)
        model.fit(
            X[fit_mask],
            y[fit_mask],
            eval_set=[(X[es_mask], y[es_mask])],
            eval_metric="rmse",
            callbacks=[log_evaluation(period=0), early_stopping(stopping_rounds=100, verbose=False)],
        )
        oof[val_mask] = model.predict(X[val_mask])
    return oof


def fit_oof_catboost(X, y, wells_arr, folds, params, es_frac=0.15, es_seed=42):
    from catboost import CatBoostRegressor

    oof = np.full(len(y), np.nan)
    for fold_i, val_wells in enumerate(folds):
        val_mask = wells_arr.isin(val_wells).to_numpy()
        train_wells_k = sorted(set(wells_arr.unique()) - set(val_wells))
        fit_wells, es_wells = _split_fit_es(train_wells_k, es_frac, es_seed + fold_i)
        fit_mask = wells_arr.isin(fit_wells).to_numpy()
        es_mask = wells_arr.isin(es_wells).to_numpy()

        model = CatBoostRegressor(**params)
        model.fit(
            X[fit_mask],
            y[fit_mask],
            eval_set=(X[es_mask], y[es_mask]),
            use_best_model=True,
            verbose=False,
        )
        oof[val_mask] = model.predict(X[val_mask])
    return oof


def fit_oof_ridge(oof_base_df, y, wells_arr, folds):
    from sklearn.linear_model import Ridge

    oof = np.full(len(y), np.nan)
    Xb = oof_base_df.to_numpy()
    for val_wells in folds:
        val_mask = wells_arr.isin(val_wells).to_numpy()
        fit_mask = ~val_mask
        model = Ridge(**RIDGE_PARAMS)
        model.fit(Xb[fit_mask], y[fit_mask])
        oof[val_mask] = model.predict(Xb[val_mask])
    return oof


def apply_pp(df, md, pd_, alpha, tau, w_pf):
    d = md * (1 - w_pf) + pd_ * w_pf
    if tau:
        d *= 1.0 - np.exp(-np.maximum(df["md_since"].values, 0.0) / tau)
    return d * alpha


# =========================================================================
# Orchestration
# =========================================================================


def _score_one_heuristic(w: str, n_seeds: int):
    """Score sub_2 (leak-free heuristic) for a single well. Module-level so it
    is picklable by loky for process-level parallelism. Returns
    (well, pred|None, y_true|None, note|None)."""
    hw = D.load_horizontal(w, "train")
    tw = D.load_typewell(w, "train")
    mask = D.eval_mask(hw)
    if mask.sum() == 0 or hw["TVT_input"].notna().sum() == 0:
        return w, None, None, "skip(no eval zone / no anchor)"
    try:
        pred = heuristic_predict(hw, tw, n_seeds=n_seeds)
        y_true = hw["TVT"].to_numpy()[mask]
        if pred.shape != y_true.shape or not np.all(np.isfinite(pred)):
            raise ValueError("bad heuristic prediction shape/values")
    except Exception as e:  # noqa: BLE001
        return w, None, None, f"FAIL:{e}"
    return w, pred, y_true, None


def sample_wells(all_wells: list[str], n: int | None, seed: int = 42) -> list[str]:
    if n is None or n >= len(all_wells):
        return sorted(all_wells)
    return sorted(random.Random(seed).sample(sorted(all_wells), n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wells", type=int, default=30, help="smoke-test well count (deterministic sample)")
    ap.add_argument("--all", action="store_true", help="use all train wells (Phase 3, full run)")
    ap.add_argument("--n-seeds-pf", type=int, default=32, help="heuristic PF ensemble seeds (notebook default 128)")
    ap.add_argument("--lgb-n-estimators-cap", type=int, default=600)
    ap.add_argument("--cb-iterations-cap", type=int, default=600)
    ap.add_argument("--skip-ml-stack", action="store_true", help="only score the heuristic (sub_2) component")
    ap.add_argument("--heur-jobs", type=int, default=1,
                    help="processes for the sub_2 heuristic (each well independent; loky). >1 uses all cores")
    ap.add_argument("--serial-build", action="store_true",
                    help="build features serially (n_jobs=1) to cap peak RSS on small-RAM boxes")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print("[R29] leak branch DISABLED: cell27's `if wid in train_wells: tvt_phys = tvt_from_contacts(...)`"
          " is never taken. Every well is scored via the notebook's `else` path only"
          " (128->%d-seed PF ensemble + 14-beam ensemble + n_eval/z_span selector, ML stack trained via"
          " nested well-grouped CV)." % args.n_seeds_pf)

    all_wells = D.list_wells("train")
    wells = sorted(all_wells) if args.all else sample_wells(all_wells, args.n_wells)
    print(f"[R29] scoring {len(wells)} / {len(all_wells)} train wells"
          f" ({'FULL' if args.all else 'SMOKE'} run)")

    _warmup_numba()

    # ---- sub_2: heuristic model, direct per-well, no cross-well fitting ----
    # The sub_2 PF is a pure-Python per-row loop (GIL-bound), so we fan wells
    # out across processes with joblib/loky to use all cores. Each well is
    # independent (no cross-well state), so this is exact -- no leak, no change
    # to results, only wall-clock.
    t0 = time.time()
    heuristic_preds: dict[str, np.ndarray] = {}
    heuristic_ytrue: dict[str, np.ndarray] = {}
    n_heur_failed = 0

    if args.heur_jobs == 1:
        results_iter = (_score_one_heuristic(w, args.n_seeds_pf) for w in wells)
    else:
        results_iter = Parallel(n_jobs=args.heur_jobs, backend="loky", verbose=0)(
            delayed(_score_one_heuristic)(w, args.n_seeds_pf) for w in wells
        )
    for i, (w, pred, y_true, note) in enumerate(results_iter):
        if note is not None and note.startswith("FAIL"):
            n_heur_failed += 1
            print(f"[R29]   heuristic FAILED well={w}: {note[5:]}")
            continue
        if pred is None:
            continue
        heuristic_preds[w] = pred
        heuristic_ytrue[w] = y_true
        if (i + 1) % max(1, len(wells) // 10) == 0:
            print(f"[R29]   heuristic {i + 1}/{len(wells)} wells done ({time.time() - t0:.0f}s elapsed)")

    y_true_all = np.concatenate([heuristic_ytrue[w] for w in heuristic_preds])
    sub2_all = np.concatenate([heuristic_preds[w] for w in heuristic_preds])
    sub2_rmse = pooled_rmse(y_true_all, sub2_all)
    print(f"[R29] sub_2 (heuristic, n_seeds_pf={args.n_seeds_pf}) pooled RMSE = {sub2_rmse:.4f}"
          f"  over {len(sub2_all):,} rows / {len(heuristic_preds)} wells"
          f"  ({n_heur_failed} failed)  [{time.time() - t0:.0f}s]")

    result: dict[str, object] = {
        "n_wells_requested": len(wells),
        "n_wells_heuristic_scored": len(heuristic_preds),
        "n_heuristic_failed": n_heur_failed,
        "n_seeds_pf": args.n_seeds_pf,
        "sub2_heuristic_pooled_rmse": sub2_rmse,
        "leak_branch": "DISABLED (always else-path, no tvt_from_contacts)",
    }

    if args.skip_ml_stack:
        _dump(result, args.out)
        return

    # ---- sub_1: LGBM(x3) + CatBoost(x2) -> Ridge stack, trained w/ well-grouped nested CV ----
    print("[R29] building FormationPlaneKNN / DenseANCCImputer donor pool from ALL train wells "
          f"(n={len(all_wells)}), self-well always excluded per-row via self_wid...")
    root = D.data_root()
    FI = FormationPlaneKNN(all_wells, root / "train")
    DI = DenseANCCImputer(all_wells, root / "train")

    train_df = build_dataset(
        wells, "train", FI, DI, label="train(scored subset)",
        n_jobs=1 if args.serial_build else None,
    )
    if train_df.empty:
        print("[R29] ERROR: build_dataset produced 0 rows -- aborting ML stack.")
        _dump(result, args.out)
        return

    feature_cols = [c for c in train_df.columns if c not in {"well", "id", "target"}]
    X = train_df[feature_cols]
    y = train_df["target"].to_numpy()
    wells_arr = train_df["well"]
    folds = well_folds(sorted(train_df["well"].unique().tolist()), n_splits=N_SPLITS, seed=CV_SEED)
    print(f"[R29] well_folds(seed={CV_SEED}, n_splits={N_SPLITS}): "
          + ", ".join(f"fold{i}={len(f)}w" for i, f in enumerate(folds)))

    oof_base = {}
    t0 = time.time()
    for i, params in enumerate(lgb_params_cpu(args.lgb_n_estimators_cap)):
        oof_base[f"lightgbm-{i + 1}"] = fit_oof_lgbm(X, y, wells_arr, folds, params)
        print(f"[R29]   lightgbm-{i + 1} OOF done [{time.time() - t0:.0f}s]")

    for i, params in enumerate(cb_params_cpu(args.cb_iterations_cap)):
        oof_base[f"catboost-{i + 1}"] = fit_oof_catboost(X, y, wells_arr, folds, params)
        print(f"[R29]   catboost-{i + 1} OOF done [{time.time() - t0:.0f}s]")

    oof_base_df = pd.DataFrame(oof_base)
    valid_rows = oof_base_df.notna().all(axis=1).to_numpy()
    if not valid_rows.all():
        print(f"[R29] WARNING: {(~valid_rows).sum()} rows have NaN base OOF (dropped from stack score)")

    ridge_oof = fit_oof_ridge(oof_base_df.loc[valid_rows], y[valid_rows], wells_arr.loc[valid_rows], folds)

    base_full = train_df["last_known_tvt"].to_numpy()[valid_rows]
    ytrue_full = y[valid_rows] + base_full
    pf_oof = train_df["pf_ancc"].to_numpy()[valid_rows] - base_full
    train_df_v = train_df.loc[valid_rows].reset_index(drop=True)

    baseline_d = apply_pp(train_df_v, ridge_oof, pf_oof, **PP_PARAMS)
    baseline_pred = base_full + baseline_d
    baseline_score = pooled_rmse(ytrue_full, baseline_pred)

    pp_selected_params = PP_PARAMS.copy()
    pp_selected_score = baseline_score
    pp_grid = [
        {"alpha": alpha, "tau": tau, "w_pf": w_pf}
        for alpha in [0.98, 0.99, 1.0, 1.01, 1.02]
        for tau in [35, 50, 65, 85, 105, 130, 170, 220]
        for w_pf in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16]
    ]
    for p in pp_grid:
        d = apply_pp(train_df_v, ridge_oof, pf_oof, **p)
        score = pooled_rmse(ytrue_full, base_full + d)
        if score < pp_selected_score - 1e-9:
            pp_selected_score = score
            pp_selected_params = p

    print(f"[R29] sub_1 (LGBM x3 + CatBoost x2 -> Ridge, capped n_estimators="
          f"{args.lgb_n_estimators_cap}/{args.cb_iterations_cap}):")
    print(f"[R29]   ridge (raw, no pp) pooled RMSE  = {pooled_rmse(ytrue_full, base_full + ridge_oof):.4f}")
    print(f"[R29]   ridge (pp default) pooled RMSE  = {baseline_score:.4f}  params={PP_PARAMS}")
    print(f"[R29]   ridge (pp tuned)   pooled RMSE  = {pp_selected_score:.4f}  params={pp_selected_params}")

    # ---- Final blend: 0.3 * sub_1 + 0.7 * sub_2, same rows only ----
    id_to_sub2 = {}
    for w in heuristic_preds:
        ids = [f"{w}_{i}" for i in np.where(D.eval_mask(D.load_horizontal(w, "train")))[0]]
        for _id, val in zip(ids, heuristic_preds[w]):
            id_to_sub2[_id] = val

    common_ids = [i for i in train_df_v["id"] if i in id_to_sub2]
    if len(common_ids) == 0:
        print("[R29] WARNING: no overlapping ids between sub_1 and sub_2 -- skipping blend score.")
    else:
        idx = train_df_v.set_index("id")
        sub1_series = pd.Series(baseline_pred, index=train_df_v["id"])
        y_true_common = (idx["target"] + idx["last_known_tvt"]).loc[common_ids].to_numpy()
        sub1_common = sub1_series.loc[common_ids].to_numpy()
        sub2_common = np.array([id_to_sub2[i] for i in common_ids])
        blend_pred = 0.3 * sub1_common + 0.7 * sub2_common
        blend_rmse = pooled_rmse(y_true_common, blend_pred)
        print(f"[R29] FINAL blend (0.3*sub1 + 0.7*sub2) pooled RMSE = {blend_rmse:.4f}"
              f"  over {len(common_ids):,} rows")
        result["sub1_pp_tuned_pooled_rmse"] = pp_selected_score
        result["sub1_pp_default_pooled_rmse"] = baseline_score
        result["sub1_pp_params_tuned"] = pp_selected_params
        result["final_blend_pooled_rmse"] = blend_rmse
        result["n_blend_rows"] = len(common_ids)

    result["n_wells_ml_scored"] = int(train_df["well"].nunique())
    _dump(result, args.out)


def _dump(result: dict, out: str | None) -> None:
    print("[R29] RESULT SUMMARY:")
    print(json.dumps(result, indent=2, default=str))
    if out:
        Path(out).write_text(json.dumps(result, indent=2, default=str))
        print(f"[R29] wrote {out}")


if __name__ == "__main__":
    main()
