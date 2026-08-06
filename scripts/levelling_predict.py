"""Step 3: held-out prediction from a levelled donor field, with no eval-zone leak.

For each of the 40 held-out wells the query well is stripped from the IDW field
entirely, so the spatial leg carries only cross-well information; the pipeline
already owns the along-well extrapolation.

Donor variants
  a   donors are train wells only, all 40 held-out wells removed
  b   donors additionally include the OTHER held-out wells' KNOWN ZONES, which is
      what a mounted-but-unlabelled test well legally offers at inference

Levelling variants for the donor datum constants c_d
  none    c_d = 0
  pairNNN mis-tie network over near approaches within NNN ft
  trend   global degree-2 trend + per-well constants, alternating

Query anchors (all fitted on the known zone only)
  tail    mean of S - S_idw over the last 300 known rows  (the historical method)
  full    median of the same over the whole known zone
  net     the query well's own constant from the mis-tie network (known zone only)
  oracle  median over the EVAL zone -- diagnostic ceiling, never used downstream

Also emits the honest gate feature: refit the anchor with the last HOLDBACK_FT of
the known zone withheld, and score that withheld stretch.

Usage:
    uv run python scripts/levelling_predict.py <variant a|b> <levelling>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE, Field, donor_mask, load  # noqa: E402
from levelling_network import pair_observations, solve_network, trend_levelling  # noqa: E402

# ROGII_HARNESS_DIR / ROGII_SET_TAG point the same machinery at the untouched
# holdout sets (k40b/c/d) without overwriting the k40 development artifacts.
K40 = Path(os.environ.get(
    "ROGII_HARNESS_DIR",
    "/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/harness_audit/k40"))
SET_TAG = os.environ.get("ROGII_SET_TAG", "")
ANCHOR_TAIL = 300
HOLDBACK_FT = 500.0
MIN_FIT = 50


def load_pipeline() -> tuple[pd.DataFrame, list[str]]:
    sub = pd.read_csv(K40 / "submission.csv")
    sub["id"] = sub["id"].astype(str)
    parts = sub["id"].str.rsplit("_", n=1)
    sub["well"] = parts.str[0]
    sub["ri"] = parts.str[1].astype(int)
    return sub, sorted(sub["well"].unique())


def offsets_for(w, mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return np.zeros(len(w.wells))
    if kind == "trend":
        return trend_levelling(w, mask)
    if kind.startswith("pair"):
        radius = float(kind[4:])
        pairs = pair_observations(w, radius=radius, point_mask=mask)
        c, comp, res = solve_network(pairs, len(w.wells))
        print(f"  network: {len(pairs)} pairs, {len(np.unique(comp))} components, "
              f"residual rms {np.sqrt(np.mean(res**2)):.2f} ft")
        return c
    raise ValueError(kind)


def robust_c(resid: np.ndarray) -> float:
    return float(np.median(resid))


def main(argv: list[str]) -> int:
    variant = argv[1] if len(argv) > 1 else "a"
    lev = argv[2] if len(argv) > 2 else "none"
    sub, held = load_pipeline()
    heldset = set(held)
    w = load()
    widx = {name: i for i, name in enumerate(w.wells)}

    # Levelling is always solved on legally visible points: train wells in full,
    # held-out wells by their known zone only.  That is true for both variants.
    net_mask = donor_mask(w, exclude=set(), known_only=heldset, stride=16)
    print(f"levelling={lev} variant={variant}")
    c_all = offsets_for(w, net_mask, lev)

    if variant == "a":
        fmask = donor_mask(w, exclude=heldset)
    else:
        fmask = donor_mask(w, exclude=set(), known_only=heldset)
    well_of = np.repeat(np.arange(len(w.wells)), np.diff(w.starts))
    fld = Field(np.stack([w.X[fmask], w.Y[fmask]], axis=1),
                (w.TVT + w.Z)[fmask], well_of[fmask])
    print(f"field: {len(fld)} donor points")

    kq, keff = (256, None) if variant == "a" else (1024, 256)
    rows, feats = [], []
    for wid in held:
        sl = w.slice(wid)
        kn = w.known[sl]
        X, Y, Z, TV, MD, RI = (w.X[sl], w.Y[sl], w.Z[sl], w.TVT[sl], w.MD[sl], w.row_idx[sl])
        S = TV + Z
        q = fld.query(np.stack([X, Y], axis=1), k=kq, offsets=c_all,
                      exclude_well=widx[wid], k_eff=keff)
        sh, nwe, dnn = q["s"], q["nwells"], q["dnn"]
        ok = np.isfinite(sh)
        ev = ~kn
        kz = kn & ok
        r = S - sh

        sp = sub[sub["well"] == wid].set_index("ri")["tvt"]
        pipe = np.array([sp.get(int(i), np.nan) for i in RI[ev]])
        truth = TV[ev]

        anchors: dict[str, float] = {}
        if kz.sum() >= MIN_FIT:
            ki = np.where(kz)[0]
            anchors["tail"] = float(np.mean(r[ki[-ANCHOR_TAIL:]]))
            anchors["tail100"] = robust_c(r[ki[-100:]])
            anchors["tail300"] = robust_c(r[ki[-300:]])
            anchors["tail1000"] = robust_c(r[ki[-1000:]])
            anchors["full"] = robust_c(r[kz])
        anchors["net"] = float(c_all[widx[wid]])
        ev_ok = ev & ok
        anchors["oracle"] = robust_c(r[ev_ok]) if ev_ok.sum() >= MIN_FIT else np.nan

        # honest gate: withhold the last HOLDBACK_FT of known MD, refit, score it
        hb_rmse = np.nan
        hb_n = 0
        if kz.sum() >= MIN_FIT:
            md_kn = MD[kn]
            cut = md_kn.max() - HOLDBACK_FT
            fit_m = kz & (MD <= cut)
            hb_m = kz & (MD > cut)
            if fit_m.sum() >= MIN_FIT and hb_m.sum() >= 20:
                c_hb = robust_c(r[fit_m])
                p_hb = (sh[hb_m] + c_hb) - Z[hb_m]
                hb_rmse = float(np.sqrt(np.mean((p_hb - TV[hb_m]) ** 2)))
                hb_n = int(hb_m.sum())

        pred = {k: np.array(pipe, dtype=float)
                for k in ("tail", "tail100", "tail300", "tail1000", "full", "net", "oracle")}
        for name, c in anchors.items():
            if not np.isfinite(c):
                continue          # anchor unavailable -> the spatial leg is the pipeline
            p = (sh[ev] + c) - Z[ev]
            pred[name] = np.where(np.isfinite(p), p, pipe)   # no donor -> pipeline

        m = np.isfinite(pipe) & np.isfinite(truth)
        rows.append(pd.DataFrame({
            "well": wid, "pipe": pipe[m], "truth": truth[m],
            **{f"sp_{k}": v[m] for k, v in pred.items()},
        }))
        feats.append({
            "well": wid, "n": int(m.sum()),
            "hb_rmse": hb_rmse, "hb_n": hb_n,
            "cov": float(ok[ev].mean()),
            "dnn_ev": float(np.median(dnn[ev_ok])) if ev_ok.sum() else np.nan,
            "dnn_kn": float(np.median(dnn[kz])) if kz.sum() else np.nan,
            "nwells_ev": float(np.median(nwe[ev_ok])) if ev_ok.sum() else np.nan,
            "sd_r_kn": float(np.std(r[kz])) if kz.sum() else np.nan,
            "n_kn": int(kn.sum()),
            "c_tail": anchors.get("tail", np.nan), "c_full": anchors.get("full", np.nan),
            "c_net": anchors["net"], "c_oracle": anchors["oracle"],
        })

    df = pd.concat(rows, ignore_index=True)
    ft = pd.DataFrame(feats)
    tag = f"{variant}_{lev}{SET_TAG}"
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
