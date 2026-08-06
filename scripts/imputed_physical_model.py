"""Can a spatially-imputed physical model work on wells with no formation columns?

The pipeline computes tvt_from_contacts only when the well is also a train well
(`if wid in train_wells`), because the formation-depth columns exist in train and not
in test. On the ~200 hidden wells that component is therefore simply absent.

tvt_from_contacts reduces to TVT = F - Z + C, where F is a formation-top depth along
the trajectory and C is a constant that the known zone pins down. F is present for all
773 train wells, so it is a spatial field that can be interpolated to a well that lacks
it — which would give the hidden wells a component they currently do not have at all.

Each well is scored leave-one-out: its own formation columns are never used, only
other wells' via inverse-distance interpolation, and C comes from TVT_input on the
known zone. Compared against carry_last, the honest floor.

Usage:
    uv run python scripts/imputed_physical_model.py [n_wells] [formation]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
DONOR_STRIDE = 8
RADIUS = 1500.0
POWER = 2.0
EPS = 1.0
ANCHOR_TAIL = 300


def main(argv: list[str]) -> int:
    limit = int(argv[1]) if len(argv) > 1 else 250
    ref = argv[2] if len(argv) > 2 else "EGFDU"

    frames = []
    for p in sorted(DATA.glob("*__horizontal_well.csv"))[:limit]:
        wid = p.name.split("__")[0]
        df = pd.read_csv(p, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input", ref])
        df = df.dropna(subset=["X", "Y", "Z", "TVT", ref])
        if df["TVT_input"].isna().sum() < 50 or df["TVT_input"].notna().sum() < ANCHOR_TAIL:
            continue
        df["well"] = wid
        frames.append(df)
    allpts = pd.concat(frames, ignore_index=True)
    print(f"wells={len(frames)}  points={len(allpts)}  formation={ref}")

    donors = allpts.iloc[::DONOR_STRIDE].reset_index(drop=True)
    xy_d = donors[["X", "Y"]].to_numpy()
    f_d = donors[ref].to_numpy(float)
    w_d = donors["well"].to_numpy()
    tree = cKDTree(xy_d)
    print(f"donor points={len(donors)}")

    def impute(xy: np.ndarray, exclude: str) -> np.ndarray:
        out = np.full(len(xy), np.nan)
        for i, nbrs in enumerate(tree.query_ball_point(xy, RADIUS, workers=-1)):
            if not nbrs:
                continue
            j = np.asarray(nbrs)
            j = j[w_d[j] != exclude]
            if j.size == 0:
                continue
            dist = np.hypot(*(xy_d[j] - xy[i]).T)
            w = 1.0 / np.power(np.maximum(dist, EPS), POWER)
            out[i] = float(np.sum(w * f_d[j]) / np.sum(w))
        return out

    sse_p = sse_c = 0.0
    n = 0
    per = []
    for df in frames:
        wid = df["well"].iloc[0]
        kn = df[df["TVT_input"].notna()]
        ev = df[df["TVT_input"].isna()].iloc[::5]
        tail = kn.tail(ANCHOR_TAIL)

        f_tail = impute(tail[["X", "Y"]].to_numpy(), wid)
        ok = np.isfinite(f_tail)
        if ok.sum() < 20:
            continue
        # TVT = F - Z + C, with C pinned by the known zone.
        c = float(np.mean(tail["TVT_input"].to_numpy(float)[ok]
                          - (f_tail[ok] - tail["Z"].to_numpy(float)[ok])))

        f_ev = impute(ev[["X", "Y"]].to_numpy(), wid)
        if np.isfinite(f_ev).mean() < 0.5:
            continue
        anchor = float(kn["TVT_input"].iloc[-1])
        pred = np.where(np.isfinite(f_ev), (f_ev - ev["Z"].to_numpy(float)) + c, anchor)
        truth = ev["TVT"].to_numpy(float)
        carry = np.full(len(ev), anchor)

        sse_p += float(np.sum((pred - truth) ** 2))
        sse_c += float(np.sum((carry - truth) ** 2))
        n += len(ev)
        per.append({"well": wid, "phys": float(np.sqrt(np.mean((pred - truth) ** 2))),
                    "carry": float(np.sqrt(np.mean((carry - truth) ** 2)))})

    r = pd.DataFrame(per)
    print(f"\nwells scored={len(r)}  eval rows={n}")
    print(f"  imputed physical model : {np.sqrt(sse_p / n):8.3f} ft")
    print(f"  carry_last (floor)     : {np.sqrt(sse_c / n):8.3f} ft")
    print(f"  wells where physics wins: {(r['phys'] < r['carry']).mean()*100:5.1f}%")
    print(f"  median per-well  phys {r['phys'].median():.2f}   carry {r['carry'].median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
