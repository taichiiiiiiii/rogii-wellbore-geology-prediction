"""Does the anchored spatial surface add anything to the pipeline when blended?

R96 measured the anchored spatial surface as a standalone predictor: 30.7 ft pooled
against carry_last's 14.8, because the wells it loses on lose catastrophically. But it
wins on 47% of wells by a median of ~4 ft, and the award write-up blended a
13.4-scoring trellis into a much stronger ensemble purely because its errors were
decorrelated. A weak leg can still pay if it is differently wrong.

So this asks the only question that matters: on wells with real pipeline predictions
and known truth, does blending help, and does the blend weight survive cross-fitting?

Donor hygiene: every held-out well is removed from the donor set, not just the query
well. At real inference a test well's neighbours are train wells only, so leaving the
other held-out wells in would flatter the method.

A single hash-based 2-fold split is one draw among many: it is repeated 200 times
over random well partitions to see the actual spread of the pooled out-of-sample
delta, not just one point estimate. The IDW interpolation itself (the expensive part)
is computed once, up front, independent of the split; only the blend-weight fit and
evaluation are redone per split.

Usage:
    uv run python scripts/spatial_blend_crossfit.py <harness_output_dir>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
DONOR_STRIDE = 8
ANCHOR_TAIL = 300
RADIUS = 1500.0
POWER = 2.0
EPS = 1.0
N_SPLITS = 200


def build_donors(exclude: set[str], field: str) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    """Donor surface field. 'S' is TVT+Z; a formation name uses that picked top instead.

    The picked tops are geologist-interpreted and measurably smoother than TVT+Z, which
    carries the well's own tracking and steering noise.
    """
    cols = ["X", "Y", "Z", "TVT"] if field == "S" else ["X", "Y", field]
    xs, ss = [], []
    for p in sorted(DATA.glob("*__horizontal_well.csv")):
        wid = p.name.split("__")[0]
        if wid in exclude:
            continue
        df = pd.read_csv(p, usecols=cols).dropna()
        if df.empty:
            continue
        df = df.iloc[::DONOR_STRIDE]
        xs.append(df[["X", "Y"]].to_numpy())
        ss.append((df["TVT"] + df["Z"]).to_numpy(float) if field == "S"
                  else df[field].to_numpy(float))
    xy = np.vstack(xs)
    return cKDTree(xy), xy, np.concatenate(ss)


def idw(tree, xy_d, s_d, xy_q) -> np.ndarray:
    out = np.full(len(xy_q), np.nan)
    for i, nbrs in enumerate(tree.query_ball_point(xy_q, RADIUS, workers=-1)):
        if not nbrs:
            continue
        j = np.asarray(nbrs)
        dist = np.hypot(*(xy_d[j] - xy_q[i]).T)
        w = 1.0 / np.power(np.maximum(dist, EPS), POWER)
        out[i] = float(np.sum(w * s_d[j]) / np.sum(w))
    return out


def _rmse_arr(w: float, pipe: np.ndarray, spat: np.ndarray, truth: np.ndarray) -> float:
    p = (1 - w) * pipe + w * spat
    return float(np.sqrt(np.mean((p - truth) ** 2)))


def _best_w_arr(pipe: np.ndarray, spat: np.ndarray, truth: np.ndarray,
                 grid: np.ndarray) -> float:
    p = (1 - grid[:, None]) * pipe[None, :] + grid[:, None] * spat[None, :]
    sse = np.mean((p - truth[None, :]) ** 2, axis=1)
    return float(grid[int(np.argmin(sse))])


def _random_group(rng: np.random.Generator, n_wells: int) -> np.ndarray:
    """A random near-even partition of well indices into two groups (True=A, False=B)."""
    order = rng.permutation(n_wells)
    group_a = np.zeros(n_wells, dtype=bool)
    group_a[order[: n_wells // 2]] = True
    return group_a


def _split_delta(pipe: np.ndarray, spat: np.ndarray, truth: np.ndarray, row_well: np.ndarray,
                  group_a: np.ndarray, grid: np.ndarray) -> float:
    """Pooled out-of-sample RMSE delta (best_w vs w=0) for one random 2-fold split."""
    mask_a = group_a[row_well]
    mask_b = ~mask_a
    sse = base_sse = 0.0
    n = 0
    for src, dst in ((mask_a, mask_b), (mask_b, mask_a)):
        w = _best_w_arr(pipe[src], spat[src], truth[src], grid)
        got = _rmse_arr(w, pipe[dst], spat[dst], truth[dst])
        base = _rmse_arr(0.0, pipe[dst], spat[dst], truth[dst])
        sse += got**2 * dst.sum()
        base_sse += base**2 * dst.sum()
        n += int(dst.sum())
    return float(np.sqrt(sse / n) - np.sqrt(base_sse / n))


def repeated_splits(df: pd.DataFrame, wells: list[str], n_splits: int = N_SPLITS,
                     seed: int = 0) -> np.ndarray:
    """Pooled out-of-sample delta for n_splits independent random 2-fold splits by well.

    The IDW interpolation behind pipe/spat/truth in `df` is already fully cached (it
    was computed once, before any split existed); only the cheap blend-weight fit and
    evaluation are redone per split, so this does not re-run the KD-tree query.
    """
    well_idx = {w: i for i, w in enumerate(wells)}
    row_well = df["well"].map(well_idx).to_numpy()
    pipe = df["pipe"].to_numpy(float)
    spat = df["spat"].to_numpy(float)
    truth = df["truth"].to_numpy(float)
    grid = np.linspace(0.0, 0.5, 51)
    rng = np.random.default_rng(seed)
    return np.array([
        _split_delta(pipe, spat, truth, row_well, _random_group(rng, len(wells)), grid)
        for _ in range(n_splits)
    ])


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = Path(argv[1])
    sub = pd.read_csv(d / "submission.csv")
    sub["id"] = sub["id"].astype(str)
    held = sorted({i.rsplit("_", 1)[0] for i in sub["id"]})
    print(f"{d.name}: {len(held)} held-out wells")

    field = argv[2] if len(argv) > 2 else "S"
    tree, xy_d, s_d = build_donors(set(held), field)
    print(f"surface field: {field}")
    print(f"donors: {len(s_d)} points from {773 - len(held)} wells (held-out excluded)")

    rows = []
    for wid in held:
        hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
        hw["id"] = [f"{wid}_{i}" for i in hw.index]
        hw = hw.dropna(subset=["X", "Y", "Z", "TVT"])
        kn = hw[hw["TVT_input"].notna()]
        ev = hw[hw["TVT_input"].isna()]
        if len(kn) < ANCHOR_TAIL or len(ev) < 50:
            continue
        tail = kn.tail(ANCHOR_TAIL)
        s_tail = idw(tree, xy_d, s_d, tail[["X", "Y"]].to_numpy())
        ok = np.isfinite(s_tail)
        if ok.sum() < 20:
            continue
        offset = float(np.mean((tail["TVT_input"].to_numpy(float)
                                + tail["Z"].to_numpy(float))[ok] - s_tail[ok]))

        g = ev.merge(sub[["id", "tvt"]], on="id", how="left")
        if g["tvt"].isna().any():
            continue
        s_ev = idw(tree, xy_d, s_d, g[["X", "Y"]].to_numpy())
        good = np.isfinite(s_ev)
        if good.mean() < 0.5:
            continue
        spatial = (s_ev + offset) - g["Z"].to_numpy(float)
        pipe = g["tvt"].to_numpy(float)
        spatial = np.where(good, spatial, pipe)      # no donor -> keep the pipeline
        rows.append(pd.DataFrame({"well": wid, "pipe": pipe, "spat": spatial,
                                  "truth": g["TVT"].to_numpy(float)}))

    df = pd.concat(rows, ignore_index=True)
    wells = sorted(df["well"].unique())
    print(f"scored wells: {len(wells)}  rows: {len(df)}")

    def rmse(w: float, sub_df: pd.DataFrame) -> float:
        p = (1 - w) * sub_df["pipe"].to_numpy() + w * sub_df["spat"].to_numpy()
        return float(np.sqrt(np.mean((p - sub_df["truth"].to_numpy()) ** 2)))

    def best_w(sub_df: pd.DataFrame) -> float:
        grid = np.linspace(0.0, 0.5, 51)
        return float(grid[int(np.argmin([rmse(w, sub_df) for w in grid]))])

    print(f"\n  pipeline alone            {rmse(0.0, df):8.4f} ft")
    print(f"  spatial alone             {rmse(1.0, df):8.4f} ft")
    base0 = rmse(0.0, df)
    for w in (0.05, 0.10, 0.20, 0.30):
        got = rmse(w, df)
        print(f"  blend w={w:<4}              {got:8.4f} ft  ({got - base0:+.4f})")
    wb = best_w(df)
    got = rmse(wb, df)
    print(f"  best in-sample w={wb:<5}     {got:8.4f} ft  ({got - base0:+.4f})")

    print("\n  cross-fit (single sha256 split, for comparison with the historical number):")
    half = {w: int(hashlib.sha256(w.encode()).hexdigest(), 16) % 2 for w in wells}
    df["half"] = df["well"].map(half)
    a, b = df[df["half"] == 0], df[df["half"] == 1]
    sse = base = 0.0
    n = 0
    for src, dst, name in ((a, b, "A->B"), (b, a, "B->A")):
        w = best_w(src)
        got, bas = rmse(w, dst), rmse(0.0, dst)
        print(f"    {name}: w={w:.2f}  {bas:.4f} -> {got:.4f}  ({got - bas:+.4f})")
        sse += got**2 * len(dst)
        base += bas**2 * len(dst)
        n += len(dst)
    print(f"    pooled out-of-sample: {np.sqrt(base/n):.4f} -> {np.sqrt(sse/n):.4f} "
          f"({np.sqrt(sse/n) - np.sqrt(base/n):+.4f})")

    print(f"\n  repeated random 2-fold splits by well (N={N_SPLITS}, seed=0):")
    deltas = repeated_splits(df, wells)
    print(f"    median {np.median(deltas):+.4f}  sd {deltas.std(ddof=1):.4f}  "
          f"p5 {np.percentile(deltas, 5):+.4f}  p95 {np.percentile(deltas, 95):+.4f}  "
          f"helps(delta<0) {100 * float((deltas < 0).mean()):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
