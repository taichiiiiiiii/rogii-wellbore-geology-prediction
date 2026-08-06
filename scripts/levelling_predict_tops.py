"""Step 3 (tops-F variant): held-out prediction from a geological-top donor field.

Donor field F(row) = (TVT + Z)(row) - top_ref(row), evaluated ROW-WISE against
that row's own formation-top reading (levelling_tops.py, EGFDL reconstructed
from EGFDU for the one well missing it).  F is near-constant within a well
(~0.007 ft, see levelling_tops.py) even though the raw top column is not, so
subtracting it removes the per-well datum bias algebraically -- there is no
network to solve for kind "none": c_d = 0 everywhere.

Two levelling kinds for the donor datum constants c_d (both act on F, not S):
  none      c_d = 0 (the datum is already removed by the top subtraction)
  pair450   the SAME mis-tie network machinery as levelling_network.py, but
            solved on F instead of S -- mis-ties that survive the tops
            de-datum are real structure the tops didn't capture.

Everything else -- variant "a" donor mask (train wells only, all 40 held-out
wells stripped), K/radius/power IDW, tail-300 known-zone anchor, honest
holdback gate -- is unchanged from levelling_predict.py; the per-well loop
below is a near-literal copy of it with S_idw replaced by F_idw.

LEAK RULES (see the run report's grep verification):
  (i)  the query well's own rows never enter the donor field: fmask always
       excludes the 40 held-out wells (levelling_common.donor_mask), so a
       held-out well is never a donor for itself or anyone else.
  (ii) the query well's OWN top columns are never read anywhere in the
       per-well loop below -- it only reads w.X/Y/Z/TVT/MD/row_idx (native
       trajectory data, real at test time).  top_ref/F is read exactly once,
       to build the donor arrays (F[fmask] and, for kind "pair450", the
       network's point_mask), both of which are train-donor-only.  This is
       WHY the "pair450" network's own point_mask here is train-only and NOT
       levelling_predict.offsets_for's `known_only=heldset` mask: a real test
       well has no top reading at all, known zone included, so it cannot
       legally contribute to an F-based network the way it legally can to an
       S-based one (S_known is real test-time information; F_known is not).

Usage:
    uv run python scripts/levelling_predict_tops.py [none|pair450]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE, Field, Wells, donor_mask, load  # noqa: E402
from levelling_network import PAIR_RADIUS, PAIR_STRIDE, pair_observations, solve_network  # noqa: E402
from levelling_predict import (  # noqa: E402
    ANCHOR_TAIL,
    HOLDBACK_FT,
    MIN_FIT,
    load_pipeline,
    robust_c,
)

TOPS_NPZ = CACHE / "tops.npz"


def load_top_ref() -> np.ndarray:
    """Row-aligned EGFDL (reconstructed), same row order as wells.npz."""
    z = np.load(TOPS_NPZ, allow_pickle=False)
    return z["top_ref"]


def with_lookup(w: Wells) -> Wells:
    w._lookup = {name: i for i, name in enumerate(w.wells)}  # type: ignore[attr-defined]
    return w


def make_shifted_wells(w: Wells, wells_to_shift: set[str], shift_ft: float) -> Wells:
    """Copy of w with EVAL-zone TVT shifted for the given wells; known zone untouched.

    Used only by the mutation leak test: a real "rebuild from a copied cache"
    of the +777 ft eval-zone truth shift, so the test exercises the actual
    donor-field/network code path rather than reasoning about it in the abstract.
    """
    tvt = w.TVT.copy()
    for wid in wells_to_shift:
        sl = w.slice(wid)
        kn = w.known[sl]
        seg = tvt[sl].copy()
        seg[~kn] += shift_ft
        tvt[sl] = seg
    return with_lookup(Wells(wells=w.wells, starts=w.starts, row_idx=w.row_idx, MD=w.MD,
                              X=w.X, Y=w.Y, Z=w.Z, TVT=tvt, known=w.known))


def network_offsets_on_f(w: Wells, F: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mis-tie network solved on F instead of S, by rigging a throwaway Wells view.

    pair_observations/solve_network (levelling_network.py) only ever read
    w.TVT + w.Z; a Wells with TVT=F, Z=0 gets that unmodified code to operate
    on F without touching the committed script.
    """
    w_f = Wells(wells=w.wells, starts=w.starts, row_idx=w.row_idx, MD=w.MD,
                X=w.X, Y=w.Y, Z=np.zeros_like(w.Z), TVT=F, known=w.known)
    pairs = pair_observations(w_f, radius=PAIR_RADIUS, point_mask=mask)
    c, comp, res = solve_network(pairs, len(w.wells))
    info = {"n_pairs": len(pairs), "n_comp": int(len(np.unique(comp))),
            "res_rms": float(np.sqrt(np.mean(res ** 2))) if len(res) else float("nan")}
    return c, info


def solve_offsets(w: Wells, F: np.ndarray, heldset: set[str], lev: str) -> tuple[np.ndarray, dict]:
    if lev == "none":
        return np.zeros(len(w.wells)), {}
    if lev == "pair450":
        net_mask = donor_mask(w, exclude=heldset, stride=PAIR_STRIDE)
        return network_offsets_on_f(w, F, net_mask)
    raise ValueError(lev)


def predict_one_well(w, wid, fld, c_all, widx) -> tuple[dict, dict]:
    sl = w.slice(wid)
    kn = w.known[sl]
    X, Y, Z, TV, MD, RI = (w.X[sl], w.Y[sl], w.Z[sl], w.TVT[sl], w.MD[sl], w.row_idx[sl])
    Swell = TV + Z
    ev = ~kn

    q = fld.query(np.stack([X, Y], axis=1), offsets=c_all, exclude_well=widx[wid])
    sh, nwe, dnn = q["s"], q["nwells"], q["dnn"]
    ok = np.isfinite(sh)
    kz, ev_ok = kn & ok, ev & ok
    r = Swell - sh

    anchors: dict[str, float] = {}
    if kz.sum() >= MIN_FIT:
        ki = np.where(kz)[0]
        anchors["tail"] = float(np.mean(r[ki[-ANCHOR_TAIL:]]))
        anchors["tail100"] = robust_c(r[ki[-100:]])
        anchors["tail300"] = robust_c(r[ki[-300:]])
        anchors["tail1000"] = robust_c(r[ki[-1000:]])
        anchors["full"] = robust_c(r[kz])
    anchors["net"] = float(c_all[widx[wid]])
    anchors["oracle"] = robust_c(r[ev_ok]) if ev_ok.sum() >= MIN_FIT else np.nan

    hb_rmse, hb_n = np.nan, 0
    if kz.sum() >= MIN_FIT:
        cut = MD[kn].max() - HOLDBACK_FT
        fit_m, hb_m = kz & (MD <= cut), kz & (MD > cut)
        if fit_m.sum() >= MIN_FIT and hb_m.sum() >= 20:
            c_hb = robust_c(r[fit_m])
            p_hb = (sh[hb_m] + c_hb) - Z[hb_m]
            hb_rmse = float(np.sqrt(np.mean((p_hb - TV[hb_m]) ** 2)))
            hb_n = int(hb_m.sum())

    return {"X": X, "Y": Y, "Z": Z, "TV": TV, "RI": RI, "ev": ev, "sh": sh,
            "anchors": anchors}, {
        "hb_rmse": hb_rmse, "hb_n": hb_n, "cov": float(ok[ev].mean()) if ev.sum() else np.nan,
        "dnn_ev": float(np.median(dnn[ev_ok])) if ev_ok.sum() else np.nan,
        "dnn_kn": float(np.median(dnn[kz])) if kz.sum() else np.nan,
        "nwells_ev": float(np.median(nwe[ev_ok])) if ev_ok.sum() else np.nan,
        "sd_r_kn": float(np.std(r[kz])) if kz.sum() else np.nan,
        "n_kn": int(kn.sum()),
        "c_tail": anchors.get("tail", np.nan), "c_full": anchors.get("full", np.nan),
        "c_net": anchors["net"], "c_oracle": anchors["oracle"],
    }


def run_variant(w: Wells, top_ref: np.ndarray, lev: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the tops-F donor field, predict the 40 held-out wells."""
    sub, held = load_pipeline()
    heldset = set(held)
    widx = {name: i for i, name in enumerate(w.wells)}
    F = (w.TVT + w.Z) - top_ref

    c_all, info = solve_offsets(w, F, heldset, lev)

    fmask = donor_mask(w, exclude=heldset)
    well_of = np.repeat(np.arange(len(w.wells)), np.diff(w.starts))
    fld = Field(np.stack([w.X[fmask], w.Y[fmask]], axis=1), F[fmask], well_of[fmask])

    rows, feats = [], []
    for wid in held:
        res, ft = predict_one_well(w, wid, fld, c_all, widx)
        X, Z, TV, RI, ev, sh, anchors = (res[k] for k in ("X", "Z", "TV", "RI", "ev", "sh", "anchors"))
        sp = sub[sub["well"] == wid].set_index("ri")["tvt"]
        pipe = np.array([sp.get(int(i), np.nan) for i in RI[ev]])
        truth = TV[ev]

        pred = {k: np.array(pipe, dtype=float)
                for k in ("tail", "tail100", "tail300", "tail1000", "full", "net", "oracle")}
        for name, c in anchors.items():
            if not np.isfinite(c):
                continue
            p = (sh[ev] + c) - Z[ev]
            pred[name] = np.where(np.isfinite(p), p, pipe)

        m = np.isfinite(pipe) & np.isfinite(truth)
        rows.append(pd.DataFrame({
            "well": wid, "pipe": pipe[m], "truth": truth[m],
            **{f"sp_{k}": v[m] for k, v in pred.items()},
        }))
        feats.append({"well": wid, "n": int(m.sum()), **ft})

    df = pd.concat(rows, ignore_index=True)
    return df, pd.DataFrame(feats), info


def main(argv: list[str]) -> int:
    lev = argv[1] if len(argv) > 1 else "none"
    w = load()
    top_ref = load_top_ref()

    print(f"levelling=tops_{lev}")
    df, ft, info = run_variant(w, top_ref, lev)
    if info:
        print(f"  network on F: {info['n_pairs']} pairs, {info['n_comp']} components, "
              f"residual rms {info['res_rms']:.2f} ft")

    tag = f"tops_{lev}"
    df.to_parquet(CACHE / f"rows_{tag}.parquet", index=False)
    ft.to_csv(CACHE / f"feats_{tag}.csv", index=False)

    truth = df["truth"].to_numpy()
    print(f"\nrows {len(df)}  wells {df['well'].nunique()}")
    print(f"  pipeline                 {np.sqrt(np.mean((df['pipe'] - truth) ** 2)):8.4f} ft")
    for name in ("tail", "tail100", "tail300", "tail1000", "full", "net", "oracle"):
        col = f"sp_{name}"
        p = df[col].to_numpy()
        pooled = np.sqrt(np.nanmean((p - truth) ** 2))
        pw = df.assign(e=(p - truth) ** 2).groupby("well")["e"].mean().pow(0.5)
        print(f"  spatial anchor={name:6s}   {pooled:8.4f} ft   median per-well {pw.median():7.3f}")
    print(f"\nwrote {CACHE / f'rows_{tag}.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
