"""How much surface information do neighbouring wells actually reach an eval row with?

surface_spatial_coherence.py established that the datum is shared: at 100 ft of
horizontal separation the typical cross-well surface disagreement is ~3 ft against
a 166 ft between-well spread. That only matters if eval rows are actually that
close to another well. This measures the reach directly, on the geometry a
held-out well really faces: for every eval-zone row of every train well, the
distance to the nearest point of any *other* well, and the surface disagreement at
that nearest point.

The comparison to beat is the pipeline itself (~6.4 ft), so the question is what
fraction of eval rows have a neighbour whose surface disagreement is below that.

Usage:
    uv run python scripts/surface_neighbour_reach.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
DONOR_STRIDE = 10   # donors: subsample along each well
EVAL_STRIDE = 40    # query rows: enough to characterise the distribution


def main() -> int:
    donors, queries = [], []
    for p in sorted(DATA.glob("*__horizontal_well.csv")):
        well = p.name.split("__")[0]
        df = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        ok = df.dropna(subset=["X", "Y", "Z", "TVT"])
        d = ok.iloc[::DONOR_STRIDE]
        donors.append(pd.DataFrame({
            "well": well, "X": d.X.to_numpy(float), "Y": d.Y.to_numpy(float),
            "surface": d.TVT.to_numpy(float) + d.Z.to_numpy(float)}))
        ev = ok[ok["TVT_input"].isna()].iloc[::EVAL_STRIDE]
        if not ev.empty:
            queries.append(pd.DataFrame({
                "well": well, "X": ev.X.to_numpy(float), "Y": ev.Y.to_numpy(float),
                "surface": ev.TVT.to_numpy(float) + ev.Z.to_numpy(float)}))

    dn = pd.concat(donors, ignore_index=True)
    qy = pd.concat(queries, ignore_index=True)
    print(f"donor points={len(dn)}  eval query rows={len(qy)}  wells={qy.well.nunique()}")

    dwell = dn["well"].to_numpy()
    dxy = dn[["X", "Y"]].to_numpy()
    dsurf = dn["surface"].to_numpy()

    best_d = np.full(len(qy), np.nan)
    best_err = np.full(len(qy), np.nan)

    # A k=64 shortlist off one shared tree is not enough: a query row mid-well sits
    # surrounded by its own well's donors, so the true nearest *other*-well point can
    # sit outside the shortlist even though it exists. Build one tree per query well
    # with that well's own donor points excluded, then take its true k=1 neighbour —
    # the same "exclude, don't post-filter" approach anchored_spatial_surface.py uses.
    for well, idx in qy.groupby("well").groups.items():
        mask = dwell != well
        if not mask.any():
            continue
        tree = cKDTree(dxy[mask])
        sub_surf = dsurf[mask]
        pos = qy.index.get_indexer(idx)
        dist, j = tree.query(qy.loc[idx, ["X", "Y"]].to_numpy(), k=1, workers=-1)
        best_d[pos] = dist
        best_err[pos] = sub_surf[j] - qy.loc[idx, "surface"].to_numpy()

    have = np.isfinite(best_d)
    print(f"eval rows with a cross-well neighbour: {have.mean()*100:.1f}%")

    d, e = best_d[have], np.abs(best_err[have])
    for q in (10, 25, 50, 75, 90):
        print(f"  nearest cross-well distance p{q:<2d} = {np.percentile(d, q):8.0f} ft")
    print("\nsurface disagreement at that nearest cross-well point:")
    print(f"  RMS      = {np.sqrt(np.mean(e**2)):8.2f} ft")
    for q in (25, 50, 75, 90):
        print(f"  |diff| p{q:<2d} = {np.percentile(e, q):8.2f} ft")
    for thr in (3.0, 6.4, 10.0):
        print(f"  share of eval rows within {thr:4.1f} ft of a neighbour's surface: "
              f"{(e < thr).mean()*100:5.1f}%")

    print("\nby distance band (the usable-reach question):")
    print(f"{'band(ft)':>14} {'rows%':>7} {'RMS':>8} {'median':>8}")
    bands = [(0, 200), (200, 500), (500, 1000), (1000, 2500), (2500, 1e9)]
    for lo, hi in bands:
        m = (d >= lo) & (d < hi)
        if not m.any():
            continue
        print(f"{f'{lo:.0f}-{hi:.0f}':>14} {m.mean()*100:6.1f}% "
              f"{np.sqrt(np.mean(e[m]**2)):8.2f} {np.median(e[m]):8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
