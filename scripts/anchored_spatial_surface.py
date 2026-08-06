"""Interpolate the surface from neighbouring wells, then re-anchor it on our own.

R86 compared raw surface values between wells and found the nearest cross-well point
sits 204 ft away with a median disagreement of 8.7 ft — worse than the pipeline — and
closed the route. But that comparison carries both wells' datum error, and we know our
own surface exactly at the anchor. Calibrating the interpolation there cancels the
datum and leaves the shape, which is the curvature the gamma ray provably cannot give.

For each well:
  donors      every point of every *other* well, S = TVT + Z (comparable across wells)
  estimate    inverse-distance interpolation of S over the k nearest donor points
  offset      S_true(anchor) - S_hat(anchor), measured on the known zone only
  prediction  TVT = (S_hat + offset) - Z

Only the known zone and other wells' training labels are used, so this is legal at
inference: a test well may use all 773 train wells as donors.

Reported against carry_last (flat TVT from the anchor), which is the honest floor.

Usage:
    uv run python scripts/anchored_spatial_surface.py [n_wells] [k_neighbours]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
DONOR_STRIDE = 8
ANCHOR_TAIL = 300      # known-zone rows used to measure the offset
POWER = 2.0            # inverse-distance exponent
RADIUS = 1500.0        # ft, search radius for cross-well donors
EVAL_STRIDE = 5        # subsample eval rows; the surface is smooth
EPS = 1.0              # ft, keeps the weight finite at zero distance


def main(argv: list[str]) -> int:
    limit = int(argv[1]) if len(argv) > 1 else 773
    k = int(argv[2]) if len(argv) > 2 else 16

    wells, frames = [], []
    for p in sorted(DATA.glob("*__horizontal_well.csv"))[:limit]:
        wid = p.name.split("__")[0]
        df = pd.read_csv(p, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
        df = df.dropna(subset=["X", "Y", "Z", "TVT"])
        if df["TVT_input"].isna().sum() < 50 or df["TVT_input"].notna().sum() < ANCHOR_TAIL:
            continue
        df["well"] = wid
        df["S"] = df["TVT"] + df["Z"]
        wells.append(wid)
        frames.append(df)
    allpts = pd.concat(frames, ignore_index=True)
    print(f"wells={len(wells)}  points={len(allpts)}")

    donors = allpts.iloc[::DONOR_STRIDE].reset_index(drop=True)
    tree = cKDTree(donors[["X", "Y"]].to_numpy())
    d_well = donors["well"].to_numpy()
    d_S = donors["S"].to_numpy(float)
    donor_xy = donors[["X", "Y"]].to_numpy()
    print(f"donor points={len(donors)} (stride {DONOR_STRIDE}), k={k}")

    def estimate(xy: np.ndarray, exclude: str) -> np.ndarray:
        """IDW over donors within RADIUS, excluding the query well.

        A k-nearest query cannot work here: an eval point mid-well is surrounded by
        several hundred of its own well's donors, so any modest k returns nothing
        usable once the well is excluded.
        """
        out = np.full(len(xy), np.nan)
        for i, nbrs in enumerate(tree.query_ball_point(xy, RADIUS, workers=-1)):
            if not nbrs:
                continue
            j = np.asarray(nbrs)
            j = j[d_well[j] != exclude]
            if j.size == 0:
                continue
            dist = np.hypot(*(donor_xy[j] - xy[i]).T)
            w = 1.0 / np.power(np.maximum(dist, EPS), POWER)
            out[i] = float(np.sum(w * d_S[j]) / np.sum(w))
        return out

    sse_a = sse_c = 0.0
    n_tot = 0
    per = []
    for df in frames:
        wid = df["well"].iloc[0]
        kn = df[df["TVT_input"].notna()]
        ev = df[df["TVT_input"].isna()].iloc[::EVAL_STRIDE]
        tail = kn.tail(ANCHOR_TAIL)

        s_tail = estimate(tail[["X", "Y"]].to_numpy(), wid)
        ok = np.isfinite(s_tail)
        if ok.sum() < 20:
            continue
        offset = float(np.mean((tail["TVT_input"].to_numpy(float) + tail["Z"].to_numpy(float))[ok]
                               - s_tail[ok]))

        s_ev = estimate(ev[["X", "Y"]].to_numpy(), wid)
        cov = float(np.isfinite(s_ev).mean())
        if cov < 0.5:
            continue
        # Points with no cross-well donor fall back to the anchor level.
        anchor_lvl = float(kn["TVT_input"].iloc[-1])
        s_ev = np.where(np.isfinite(s_ev), s_ev,
                        anchor_lvl + ev["Z"].to_numpy(float) - offset)
        pred = (s_ev + offset) - ev["Z"].to_numpy(float)
        truth = ev["TVT"].to_numpy(float)
        carry = np.full(len(ev), float(kn["TVT_input"].iloc[-1]))

        sse_a += float(np.sum((pred - truth) ** 2))
        sse_c += float(np.sum((carry - truth) ** 2))
        n_tot += len(ev)
        per.append({"well": wid, "n": len(ev), "coverage": cov,
                    "anchored": float(np.sqrt(np.mean((pred - truth) ** 2))),
                    "carry": float(np.sqrt(np.mean((carry - truth) ** 2)))})

    r = pd.DataFrame(per)
    cov_med = r["coverage"].median() * 100
    print(f"\nwells scored: {len(r)}   eval rows: {n_tot}   "
          f"median donor coverage {cov_med:.0f}%")
    print(f"  anchored spatial surface : {np.sqrt(sse_a / n_tot):8.3f} ft")
    print(f"  carry_last (floor)       : {np.sqrt(sse_c / n_tot):8.3f} ft")
    print(f"  wells where anchored wins: {(r['anchored'] < r['carry']).mean()*100:5.1f}%")
    better = r[r["anchored"] < r["carry"]]
    print(f"  median per-well RMSE      anchored {r['anchored'].median():.2f}  "
          f"carry {r['carry'].median():.2f}")
    if len(better):
        gain = (better["carry"] - better["anchored"]).median()
        print(f"  among wins, median gain  {gain:.2f} ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
