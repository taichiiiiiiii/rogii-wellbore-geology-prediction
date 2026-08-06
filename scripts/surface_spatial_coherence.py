"""Test whether the implied stratigraphic surface is shared between wells.

TVT = surface - Z, so surface = TVT + Z is measured along every train well's
path. If that surface is one geological object in a common datum, it is a smooth
function of (X, Y) across the field, and the eval-zone trend — the component the
gamma ray provably cannot recover — could be interpolated from neighbouring wells
instead. If instead each well's TVT is referenced to its own typewell datum, the
surface values are mutually incomparable and the whole route is dead.

The test is a crossing check: find pairs of points from *different* wells that are
close in (X, Y) and compare their surface values. Agreement at short range means a
shared datum; a spread comparable to the between-well spread means per-well datums.

Usage:
    uv run python scripts/surface_spatial_coherence.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
STRIDE = 25  # subsample along each well; the surface is smooth, so this loses nothing
RADII = (100.0, 250.0, 500.0, 1000.0, 2000.0)


def load_points() -> pd.DataFrame:
    rows = []
    for p in sorted(DATA.glob("*__horizontal_well.csv")):
        well = p.name.split("__")[0]
        df = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT"])
        df = df.dropna(subset=["X", "Y", "Z", "TVT"]).iloc[::STRIDE]
        if df.empty:
            continue
        rows.append(pd.DataFrame({
            "well": well,
            "X": df["X"].to_numpy(float),
            "Y": df["Y"].to_numpy(float),
            "surface": df["TVT"].to_numpy(float) + df["Z"].to_numpy(float),
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    pts = load_points()
    n_wells = pts["well"].nunique()
    print(f"wells={n_wells}  points={len(pts)}  (stride {STRIDE})")
    print(f"surface: mean={pts.surface.mean():.1f}  std={pts.surface.std():.1f}  "
          f"min={pts.surface.min():.1f}  max={pts.surface.max():.1f}")

    per_well = pts.groupby("well")["surface"].agg(["mean", "std"])
    print(f"between-well spread of surface means: std={per_well['mean'].std():.2f} ft")
    print(f"within-well spread of surface       : median std={per_well['std'].median():.2f} ft")

    xy = pts[["X", "Y"]].to_numpy()
    surf = pts["surface"].to_numpy()
    well = pts["well"].to_numpy()
    tree = cKDTree(xy)

    print("\ncross-well surface agreement vs horizontal separation")
    print(f"{'radius(ft)':>10} {'pairs':>9} {'RMS diff':>10} {'median |diff|':>14}")
    rng = np.random.default_rng(0)
    sample = rng.choice(len(pts), size=min(4000, len(pts)), replace=False)
    for r in RADII:
        diffs = []
        for i in sample:
            for j in tree.query_ball_point(xy[i], r):
                if well[j] != well[i]:
                    diffs.append(surf[i] - surf[j])
        if not diffs:
            print(f"{r:>10.0f} {0:>9} {'-':>10} {'-':>14}")
            continue
        d = np.asarray(diffs, float)
        rms, med = np.sqrt(np.mean(d**2)), np.median(np.abs(d))
        print(f"{r:>10.0f} {len(d):>9} {rms:>10.2f} {med:>14.2f}")

    spread = per_well["mean"].std()
    print("\nReference: a shared datum implies short-range RMS diff far below the")
    print(f"between-well spread ({spread:.2f} ft); comparable values mean per-well datums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
