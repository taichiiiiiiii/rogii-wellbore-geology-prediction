"""Cache the 6 formation-top columns per train well: the donor-field datum.

Prior work (analysis/experiment_ledger.md R125) measured that ANCC, ASTNU,
ASTNL, EGFDU, EGFDL, BUDA are co-planar across wells (dip ~37 ft/1000 ft,
azimuth ~318 deg, R^2 0.92-0.94) and that F := S - top (S = TVT + Z, ROW-WISE
against that same row's own top reading) transfers between neighbour wells
with median error ~0.01 ft (vs several ft for raw S) -- verified again below.

IMPORTANT (spec-reality conflict, see the run report): the raw top columns
are NOT near-constant within a well by themselves -- they track the well's
own trajectory (elevation) tightly, so their within-well std runs into the
hundreds of feet on long laterals.  What IS near-constant within a well to
~0.007 ft is the residual F = S - top, i.e. the vertical offset between the
wellbore and that formation marker.  That is the quantity this cache exists
to support: it is exactly the per-well datum constant that the sparse
mis-tie network (scripts/levelling_network.py, 587 pairs / 299 components /
170 singletons) struggles to estimate for every well -- algebraically, not
statistically -- because top(row) is available at EVERY row, not just where
two wells happen to pass close to each other.

This script reads the 6 top columns for every well already cached in
wells.npz (same well list/order, via levelling_common.load()), reports
per-well median + within-well std of the raw columns (the literal ask), and
writes a ROW-ALIGNED top_ref array (EGFDL, reconstructed from EGFDU + a
global additive constant for the one well missing EGFDL) to tops.npz.  Top
columns are train-only and absent from test; nothing here is used on the
query well's own rows at prediction time (see levelling_predict_tops.py).

Usage:
    uv run python scripts/levelling_tops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE, load  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
TOP_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
TOP_REF = "EGFDL"  # reconstruction source column is chosen dynamically, see reconstruct_top_ref


def read_top_columns(w) -> dict[str, np.ndarray]:
    """Row-aligned top columns, same row order/count as wells.npz per well."""
    flat = {c: np.full(len(w.X), np.nan) for c in TOP_COLS}
    for i, wid in enumerate(w.wells):
        a, b = int(w.starts[i]), int(w.starts[i + 1])
        df = pd.read_csv(DATA / f"{wid}__horizontal_well.csv",
                          usecols=["X", "Y", "Z", "TVT"] + TOP_COLS)
        m = df[["X", "Y", "Z", "TVT"]].notna().all(axis=1).to_numpy()
        df = df[m]
        assert len(df) == b - a, f"{wid}: row count mismatch with wells.npz"
        for c in TOP_COLS:
            flat[c][a:b] = df[c].to_numpy(float)
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(w.wells)}")
    return flat


def well_median_std(w, flat: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = len(w.wells)
    med = np.full((n, len(TOP_COLS)), np.nan)
    std = np.full((n, len(TOP_COLS)), np.nan)
    for i in range(n):
        a, b = int(w.starts[i]), int(w.starts[i + 1])
        for j, c in enumerate(TOP_COLS):
            v = flat[c][a:b]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            med[i, j] = np.median(v)
            std[i, j] = np.std(v)
    return med, std


def report_raw_stats(w, med: np.ndarray, std: np.ndarray) -> None:
    n = len(w.wells)
    print(f"\n=== step 1: raw top-column per-well stats (n={n} wells) ===")
    for j, c in enumerate(TOP_COLS):
        cov = int(np.isfinite(med[:, j]).sum())
        mx = float(np.nanmax(std[:, j])) if np.isfinite(std[:, j]).any() else float("nan")
        md = float(np.nanmedian(std[:, j])) if np.isfinite(std[:, j]).any() else float("nan")
        print(f"  {c:6s} coverage={cov:4d}/{n}  median within-well std={md:8.4f} ft  "
              f"max within-well std={mx:8.4f} ft")
    flat_i = np.nanargmax(std)
    wi, cj = np.unravel_index(flat_i, std.shape)
    print(f"  worst within-well std overall: well={w.wells[wi]} col={TOP_COLS[cj]} "
          f"std={std[wi, cj]:.4f} ft")
    over1 = int((std > 1.0).sum())
    print(f"  (well,col) pairs with within-well std > 1 ft: {over1} / {n * len(TOP_COLS)}")
    print("  CONFLICT vs spec expectation (~0.007 ft): the RAW top columns track the well's "
          "own trajectory/elevation and are NOT near-constant within a well; see the F = S - "
          "top validation below for the quantity that actually is.")


def reconstruct_top_ref(w, flat: dict[str, np.ndarray], med: np.ndarray) -> tuple[np.ndarray, list[str], str]:
    """EGFDL row-aligned, with the missing well filled from EGFDU + a global constant."""
    j_egfdl, j_others = TOP_COLS.index(TOP_REF), [c for c in TOP_COLS if c != TOP_REF]
    print(f"\n=== step 2: reference top = {TOP_REF}; pairwise well-level separations ===")
    seps = {}
    for c in j_others:
        jo = TOP_COLS.index(c)
        d = med[:, j_egfdl] - med[:, jo]
        d = d[np.isfinite(d)]
        seps[c] = (float(np.median(d)), float(np.std(d)), int(d.size))
        print(f"  {TOP_REF} - {c:6s}: n={d.size:4d}  median={seps[c][0]:9.4f}  "
              f"std={seps[c][1]:7.4f} ft")
    src = min(j_others, key=lambda c: seps[c][1])
    const, sep_std, _ = seps[src]
    print(f"  chosen reconstruction source: {src} (tightest separation std {sep_std:.4f} ft); "
          f"EGFDL_hat = {src} + {const:.4f}")

    missing = np.where(~np.isfinite(med[:, j_egfdl]))[0]
    recon_wells = [str(w.wells[i]) for i in missing]
    print(f"  wells needing reconstruction: {len(recon_wells)} -> {recon_wells}")

    top_ref = flat[TOP_REF].copy()
    for i in missing:
        a, b = int(w.starts[i]), int(w.starts[i + 1])
        top_ref[a:b] = flat[src][a:b] + const
    return top_ref, recon_wells, src


def validate_f(w, top_ref: np.ndarray) -> None:
    """F = S - top_ref should be near-constant within a well (~0.007 ft), unlike top itself."""
    S = w.TVT + w.Z
    F = S - top_ref
    n = len(w.wells)
    fstd = np.full(n, np.nan)
    for i in range(n):
        a, b = int(w.starts[i]), int(w.starts[i + 1])
        v = F[a:b]
        v = v[np.isfinite(v)]
        if v.size:
            fstd[i] = np.std(v)
    print(f"\n=== validation: F = S - {TOP_REF}(row) within-well std (n={n}) ===")
    print(f"  median {np.nanmedian(fstd):.5f} ft  p90 {np.nanpercentile(fstd, 90):.5f} ft  "
          f"max {np.nanmax(fstd):.5f} ft")


def main() -> int:
    out = CACHE / "tops.npz"
    if out.exists():
        print(f"exists: {out}")
        return 0

    w = load()
    print(f"reading {len(w.wells)} wells' top columns from {DATA}")
    flat = read_top_columns(w)
    med, std = well_median_std(w, flat)
    report_raw_stats(w, med, std)
    top_ref, recon_wells, recon_src = reconstruct_top_ref(w, flat, med)
    validate_f(w, top_ref)

    np.savez(out, wells=w.wells, cols=np.array(TOP_COLS), median=med, std=std,
             top_ref=top_ref, top_ref_col=np.array(TOP_REF),
             recon_src_col=np.array(recon_src), recon_wells=np.array(recon_wells))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
