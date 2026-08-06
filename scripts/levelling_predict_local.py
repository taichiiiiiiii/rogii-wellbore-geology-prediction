"""Step 3b: LOCAL levelling -- one free datum offset per donor well, estimated
on the query well's own known zone (spec: local_levelling_spec.md, S1).

`levelling_predict.py` solves one global mis-tie constant `c_d` per donor well
(299 connected components over 773 wells, 170 never tied) and then applies one
query-well anchor on top.  `loc` never touches that network at all: for query
well q, every donor WELL d in its k-nearest-neighbour footprint gets its own
offset `o_d = weighted mean over q's known zone of (S_q - m_d)`, where `m_d` is
the per-well IDW mean and the weight is the well's own total point weight
`W_d`.  Every donor is then expressed in the query well's own datum, so a
change in *which* neighbours are visible as the bit advances along the lateral
can no longer move the estimate -- there is no shared gauge left to drift.

The tail anchor is NOT subsumed by `o_d`: `o_d` is a weight-weighted mean over
the *whole* known zone, the anchor is a plain mean over the *last* 300 known
rows, and it is kept on top of the local levelling exactly as it is kept on
top of the global one (spec S1.5).

Fallback chain, elementwise per row: loc -> glob (production pair450 c_d +
tail anchor, computed by the same per-well machinery with a different offset
array) -> pipe.  Every held well emits a prediction; none are ever dropped
(spec S1.6, pitfall #1).

Usage:
    uv run python scripts/levelling_predict_local.py \
        [variant] [evstride] [knstride] [anchor_tail] [lev_tag] [eval_shift_ft]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE, CHUNK, donor_mask, load  # noqa: E402
from levelling_predict import (  # noqa: E402
    HOLDBACK_FT,
    MIN_FIT,
    SET_TAG,
    load_pipeline,
    offsets_for,
)

K, RID, POWER, EPS = 256, 1500.0, 2.0, 1.0
MINW = 1e-4
MIN_KI, MIN_EI = 50, 10
# The cached network embeds the held-well exclusion set, so it must be keyed by
# the holdout set (ROGII_SET_TAG) -- reusing k40's network for k40b would leak.
NET_CACHE = CACHE / f"network_prod_local_a{SET_TAG}.npz"


def build_field(w, held: list[str]):
    """Raw, unlevelled donor field, variant 'a': all held wells removed.  No
    c_d is subtracted here -- the global network is a separate leg (glob).
    """
    well_of = np.repeat(np.arange(len(w.wells)), np.diff(w.starts))
    fmask = donor_mask(w, exclude=set(held))
    DX = np.stack([w.X[fmask], w.Y[fmask]], axis=1)
    DS = (w.TVT + w.Z)[fmask]
    DW = well_of[fmask]
    print(f"field: {int(fmask.sum())} donor points")
    return cKDTree(DX), DS, DW


def global_offsets(w, heldset: set[str]) -> np.ndarray:
    """Production mis-tie network c_d (knob #5 restored: known_only=held, the
    same recipe levelling_predict.py's main() uses), cached across CLI calls.
    """
    if NET_CACHE.exists():
        return np.load(NET_CACHE)["c"]
    net_mask = donor_mask(w, exclude=set(), known_only=heldset, stride=16)
    c = offsets_for(w, net_mask, "pair450")
    np.savez(NET_CACHE, c=c)
    return c


def knn_chunked(tree: cKDTree, xy: np.ndarray, k: int):
    """K-nearest donor points, chunked at production CHUNK to bound peak RAM
    (knob #9): (CHUNK, K) float64 per chunk instead of the whole well at once.
    """
    n = len(xy)
    kk = min(k, tree.n)
    dist = np.empty((n, kk))
    idx = np.empty((n, kk), dtype=np.int64)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        d, ii = tree.query(xy[a:b], k=kk, workers=-1)
        if d.ndim == 1:
            d, ii = d[:, None], ii[:, None]
        dist[a:b], idx[a:b] = d, ii
    return dist, idx


def point_level(dist: np.ndarray, idx: np.ndarray, DW: np.ndarray, DS: np.ndarray):
    """Per-point donor-well id, IDW weight, surface value, and per-row nearest-
    donor distance / distinct-well count (used only for feats diagnostics).
    """
    ok = np.isfinite(dist) & (dist <= RID)
    idx_safe = np.where(ok, idx, 0)
    wt = np.where(ok, 1.0 / np.maximum(dist, EPS) ** POWER, 0.0)
    wid = np.where(ok, DW[idx_safe], -1)
    sv = DS[idx_safe]
    dnn = np.where(ok, dist, np.inf).min(axis=1)
    ws = np.sort(wid, axis=1)
    chg = np.ones_like(ws, dtype=bool)
    chg[:, 1:] = ws[:, 1:] != ws[:, :-1]
    nw = ((ws >= 0) & chg).sum(axis=1)
    return wid, wt, sv, dnn, nw


def collapse_by_well(wid: np.ndarray, wt: np.ndarray, sv: np.ndarray, self_id: int):
    """-> (wells, W, M): per-donor-WELL total weight and IDW mean (spec S1.3).

    Weights stay per POINT throughout; grouping by well only changes what gets
    offset later, never the weighting.  `self_id` exclusion is belt-and-braces
    (variant 'a' already has no self points in the field).
    """
    wells = sorted(set(np.unique(wid).tolist()) - {-1, self_id})
    n, m = len(wid), len(wells)
    if m == 0:                                        # pitfall #4: guard first
        return wells, np.zeros((n, 0)), np.zeros((n, 0))
    pos = {v: i for i, v in enumerate(wells)}
    code = np.full(wid.shape, -1)
    for v, i in pos.items():
        code[wid == v] = i
    keep = code >= 0
    flat = (np.arange(n)[:, None] * m + np.where(keep, code, 0)).ravel()
    kw = (wt * keep).ravel()
    W = np.bincount(flat, weights=kw, minlength=n * m).reshape(n, m)
    Sm = np.bincount(flat, weights=(sv * wt * keep).ravel(), minlength=n * m).reshape(n, m)
    M = np.divide(Sm, W, out=np.full_like(Sm, np.nan), where=W > 0)
    return wells, W, M


def aggregate(W: np.ndarray, Me2: np.ndarray) -> np.ndarray:
    """Sigma_d W_d Me2_d / Sigma_d W_d, numerator and denominator sharing one
    isfinite mask (pitfall #7: nansum of independently-masked terms is wrong).
    """
    fin = np.isfinite(Me2)
    num = np.nansum(W * np.nan_to_num(Me2) * fin, axis=1)
    den = np.nansum(W * fin, axis=1)
    return np.divide(num, den, out=np.full(len(W), np.nan), where=den > 0)


def local_offsets(S_ki: np.ndarray, Wk: np.ndarray, Mk: np.ndarray):
    """One offset per donor well: weighted MEAN over the whole known zone of
    (S_q - m_d), weight = W_d(x) (spec S1.4).  Untied donors stay NaN -- that
    is how they are dropped, do not fill with 0 (pitfall #6).
    """
    diff = S_ki[:, None] - Mk
    wk = np.where(np.isfinite(diff), Wk, 0.0)
    tot = wk.sum(0)
    share = tot / max(float(wk.sum()), 1e-30)
    o_d = np.divide((np.nan_to_num(diff) * wk).sum(0), tot,
                    out=np.full(Mk.shape[1], np.nan), where=tot > 0)
    tied = np.isfinite(o_d) & (share >= MINW)
    return np.where(tied, o_d, np.nan), tied


def tail_anchor(resid: np.ndarray, anchor_tail: int) -> float:
    """nanmean over the last `anchor_tail` known rows -- guarded against an
    all-NaN slice, which nanmean would otherwise silently pass through as NaN
    with a RuntimeWarning (pitfall #3).
    """
    tail = resid[-anchor_tail:] if anchor_tail > 0 else resid
    if tail.size == 0 or not np.isfinite(tail).any():
        return float("nan")
    return float(np.nanmean(tail))


def spatial_leg(S_ki: np.ndarray, Wk: np.ndarray, Mk: np.ndarray,
                We: np.ndarray, Me: np.ndarray, offsets: np.ndarray, anchor_tail: int):
    """Two-stage local-datum estimate + tail anchor (spec S1.5): the SAME
    offsets are applied on the known zone (to derive the anchor) and on the
    eval zone (for the prediction) -- the anchor absorbs whatever the offsets
    left over on the last `anchor_tail` known rows.
    """
    loc_zone = aggregate(We, Me + offsets)
    lk = aggregate(Wk, Mk + offsets)
    a = tail_anchor(S_ki - lk, anchor_tail)
    return loc_zone, lk, a


def honest_gate_and_tilt(ki: np.ndarray, MD: np.ndarray, S_ki: np.ndarray,
                         lk: np.ndarray, Z: np.ndarray, TV: np.ndarray) -> dict:
    """Withhold the last HOLDBACK_FT of known MD, refit only the anchor (o_d
    itself is NOT refit -- spec S5), and score the held-back stretch.

    `hb_rmse` mirrors production's own honest-gate feature (median anchor over
    the whole fit window, spec S3 knob #4).  `lam` is the S5 tilt pre-test:
    lstsq intercept+slope on the SAME residual, evaluated at lambda in
    {0, .25, .5, 1} on the held-back stretch, pooled outside this function.
    """
    out = {"hb_rmse": float("nan"), "hb_n": 0, "lam": {}}
    if len(ki) < MIN_FIT:
        return out
    md_kn = MD[ki]
    cut = md_kn.max() - HOLDBACK_FT
    fit_m, hb_m = md_kn <= cut, md_kn > cut
    if fit_m.sum() < MIN_FIT or hb_m.sum() < 20:
        return out
    r_loc = S_ki - lk
    c_hb = float(np.nanmedian(r_loc[fit_m]))
    p_hb = lk[hb_m] + c_hb - Z[ki[hb_m]]
    out["hb_rmse"] = float(np.sqrt(np.mean((p_hb - TV[ki[hb_m]]) ** 2)))
    out["hb_n"] = int(hb_m.sum())
    valid = fit_m & np.isfinite(r_loc)
    if valid.sum() >= MIN_FIT:
        A = np.stack([np.ones(int(valid.sum())), md_kn[valid] - cut], axis=1)
        a, b = np.linalg.lstsq(A, r_loc[valid], rcond=None)[0]
        for lam in (0.0, 0.25, 0.5, 1.0):
            p = lk[hb_m] + a + lam * b * (md_kn[hb_m] - cut) - Z[ki[hb_m]]
            se = (p - TV[ki[hb_m]]) ** 2
            fin = np.isfinite(se)
            out["lam"][lam] = (float(np.nansum(se[fin])), int(fin.sum()))
    return out


def process_well(wid: str, w, widx: dict, tree: cKDTree, DS: np.ndarray, DW: np.ndarray,
                  c_glob: np.ndarray, sub: pd.DataFrame, evstride: int, knstride: int,
                  anchor_tail: int, shift: float) -> dict:
    """One held well end to end: donor query -> loc + glob legs -> fallback ->
    feats.  Never drops the well (spec pitfall #1): pipe is always emitted.
    """
    sl = w.slice(wid)
    kn = w.known[sl]
    X, Y, Z, MD, RI = w.X[sl], w.Y[sl], w.Z[sl], w.MD[sl], w.row_idx[sl]
    TV = w.TVT[sl].copy()
    if shift:
        TV[~kn] += shift          # mutation leak test: eval-zone truth only
    S = TV + Z
    ki = np.where(kn)[0][::knstride]
    ei = np.where(~kn)[0][::evstride]

    sp = sub[sub["well"] == wid].set_index("ri")["tvt"]
    pipe = np.array([sp.get(int(i), np.nan) for i in RI[ei]])
    truth = TV[ei]

    res = {"loc": np.full(len(ei), np.nan), "glob": np.full(len(ei), np.nan),
           "oracle_pred": np.full(len(ei), np.nan),
           "nwell": 0, "ntied": 0, "dnn_kn": np.nan, "dnn_ev": np.nan,
           "nwells_ev": np.nan, "sd_r_kn": np.nan, "oracle": np.nan,
           "a_loc": np.nan, "a_glob": np.nan,
           "hb_rmse": np.nan, "hb_n": 0, "lam": {}}

    if len(ki) >= MIN_KI and len(ei) >= MIN_EI:
        qi = np.concatenate([ki, ei])
        nk = len(ki)
        dist, idx = knn_chunked(tree, np.stack([X[qi], Y[qi]], axis=1), K)
        wid_pt, wt, sv, dnn, nw = point_level(dist, idx, DW, DS)
        wells, W, M = collapse_by_well(wid_pt, wt, sv, widx[wid])
        res["nwell"] = len(wells)
        if wells:
            Wk, Mk, We, Me = W[:nk], M[:nk], W[nk:], M[nk:]
            o_use, tied = local_offsets(S[ki], Wk, Mk)
            res["ntied"] = int(tied.sum())
            loc_zone, lk, a_loc = spatial_leg(S[ki], Wk, Mk, We, Me, o_use, anchor_tail)
            res["a_loc"] = a_loc
            if np.isfinite(a_loc):
                res["loc"] = loc_zone + a_loc - Z[ei]

            c_here = -c_glob[np.array(wells)]
            glob_zone, _glk, a_glob = spatial_leg(S[ki], Wk, Mk, We, Me, c_here, anchor_tail)
            res["a_glob"] = a_glob
            if np.isfinite(a_glob):
                res["glob"] = glob_zone + a_glob - Z[ei]

            covered_kn, covered_ev = np.isfinite(lk), np.isfinite(loc_zone)
            if covered_kn.any():
                res["dnn_kn"] = float(np.median(dnn[:nk][covered_kn]))
                res["sd_r_kn"] = float(np.nanstd((S[ki] - lk)[covered_kn]))
            if covered_ev.any():
                res["dnn_ev"] = float(np.median(dnn[nk:][covered_ev]))
                res["nwells_ev"] = float(np.median(nw[nk:][covered_ev]))
            ev_ok = covered_ev & np.isfinite(truth)
            if ev_ok.sum() >= MIN_FIT:
                res["oracle"] = float(np.nanmedian((S[ei] - loc_zone)[ev_ok]))
                res["oracle_pred"] = loc_zone + res["oracle"] - Z[ei]
            res.update(honest_gate_and_tilt(ki, MD, S[ki], lk, Z, TV))

    spat = np.where(np.isfinite(res["loc"]), res["loc"], res["glob"])
    spat = np.where(np.isfinite(spat), spat, pipe)
    res.update({"pipe": pipe, "truth": truth, "spat": spat})
    return res


def emit_rows(wid: str, r: dict) -> pd.DataFrame:
    """Schema-identical to rows_a_pair450.parquet.  `loc` has one production
    anchor design (no tail100/tail300/tail1000/full/net variants), so those
    columns alias the same fallback chain result; `sp_oracle` is the genuine
    diagnostic (median over the eval zone, never used downstream).
    """
    m = np.isfinite(r["pipe"]) & np.isfinite(r["truth"])
    spat = r["spat"]
    oracle_col = np.where(np.isfinite(r["oracle_pred"]), r["oracle_pred"], spat)
    return pd.DataFrame({
        "well": wid, "pipe": r["pipe"][m], "truth": r["truth"][m],
        "sp_tail": spat[m], "sp_tail100": spat[m], "sp_tail300": spat[m],
        "sp_tail1000": spat[m], "sp_full": spat[m], "sp_net": spat[m],
        "sp_oracle": oracle_col[m],
    })


def main(argv: list[str]) -> int:
    variant = argv[1] if len(argv) > 1 else "a"
    evstride = int(argv[2]) if len(argv) > 2 else 1
    knstride = int(argv[3]) if len(argv) > 3 else 1
    anchor_tail = int(argv[4]) if len(argv) > 4 else 300
    lev_tag = argv[5] if len(argv) > 5 else "local"
    shift = float(argv[6]) if len(argv) > 6 else 0.0
    if variant != "a":
        raise ValueError("only variant 'a' is ported (spec S3 knob #8)")

    sub, held = load_pipeline()
    heldset = set(held)
    w = load()
    widx = {name: i for i, name in enumerate(w.wells)}
    t0 = time.time()
    tree, DS, DW = build_field(w, held)
    c_glob = global_offsets(w, heldset)
    print(f"levelling=local variant={variant} evstride={evstride} knstride={knstride} "
          f"anchor_tail={anchor_tail} shift={shift}")

    rows, feats = [], []
    lam_se = {lam: [0.0, 0] for lam in (0.0, 0.25, 0.5, 1.0)}
    for wid in held:
        r = process_well(wid, w, widx, tree, DS, DW, c_glob, sub,
                         evstride, knstride, anchor_tail, shift)
        rows.append(emit_rows(wid, r))
        feats.append({
            "well": wid, "n": int(np.isfinite(r["pipe"]).sum()),
            "hb_rmse": r["hb_rmse"], "hb_n": r["hb_n"],
            "dnn_ev": r["dnn_ev"], "dnn_kn": r["dnn_kn"], "nwells_ev": r["nwells_ev"],
            "sd_r_kn": r["sd_r_kn"], "n_kn": int(w.known[w.slice(wid)].sum()),
            "c_tail": r["a_loc"], "c_full": np.nan, "c_net": r["a_glob"], "c_oracle": r["oracle"],
            "nwell": r["nwell"], "ntied": r["ntied"],
        })
        for lam, (se, n) in r["lam"].items():
            lam_se[lam][0] += se
            lam_se[lam][1] += n

    df = pd.concat(rows, ignore_index=True)
    ft = pd.DataFrame(feats)
    tag = f"{variant}_{lev_tag}{SET_TAG}"
    df.to_parquet(CACHE / f"rows_{tag}.parquet", index=False)
    ft.to_csv(CACHE / f"feats_{tag}.csv", index=False)

    truth = df["truth"].to_numpy()
    base = float(np.sqrt(np.mean((df["pipe"].to_numpy() - truth) ** 2)))
    spat_rmse = float(np.sqrt(np.mean((df["sp_tail"].to_numpy() - truth) ** 2)))
    print(f"\nrows {len(df)}  wells {df['well'].nunique()}  elapsed {time.time() - t0:.1f}s")
    print(f"  pipeline           {base:8.4f} ft")
    print(f"  loc->glob->pipe    {spat_rmse:8.4f} ft")
    print("\n  tilt pre-test (spec S6), pooled RMSE on the holdback stretch:")
    for lam in (0.0, 0.25, 0.5, 1.0):
        se, n = lam_se[lam]
        print(f"    lambda={lam:.2f}  n={n:6d}  rmse={np.sqrt(se / n) if n else float('nan'):8.4f}")
    print(f"\nwrote {CACHE / f'rows_{tag}.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
