"""Cross-fit the global steering coefficient: estimate on one set of wells, apply to another.

R94 found that a single global coefficient corrects the pipeline's residual by -0.27
to -0.29 on held-out wells, but that number was read off the same wells it was fitted
on, so it is an upper bound. The coefficient is only worth shipping if it survives
being fitted on wells it is not then judged on.

Wells are split in half by a hash of their id, beta is fitted on each half and applied
to the other, and both directions are reported. A coefficient that is real gives a
similar gain in both; one that is fitted noise gives a gain in-sample and little or
nothing out of sample.

That single hash split is only one draw, and it is an unstable one: repeating the
2-fold split 200 times over random well partitions shows the pooled out-of-sample
delta has real spread, so a single split can land anywhere from "clearly helps" to
"clearly hurts" by chance alone. Both the historical single split and the full
distribution over repeated splits are reported.

Also reports the per-well oracle as the ceiling and beta = 0 as the floor, so the
cross-fit number can be read against both.

Usage:
    uv run python scripts/steering_beta_crossfit.py <harness_output_dir>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
TAIL = 200
N_SPLITS = 200


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    d = float(np.sum(x * x))
    return float(np.sum(x * (y - y.mean())) / d) if d > 0 else 0.0


def collect(d: Path) -> pd.DataFrame:
    sub = pd.read_csv(d / "submission.csv")
    sub["id"] = sub["id"].astype(str)
    wells = sorted({i.rsplit("_", 1)[0] for i in sub["id"]})

    frames = []
    for wid in wells:
        hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
        hw["id"] = [f"{wid}_{i}" for i in hw.index]
        kn = hw[hw["TVT_input"].notna()]
        ev = hw[hw["TVT_input"].isna()]
        if len(kn) < 20 or len(ev) < 50:
            continue
        tail = kn.tail(TAIL)
        md_a, z_a = float(tail["MD"].iloc[-1]), float(tail["Z"].iloc[-1])
        b_z = slope(tail["MD"].to_numpy(float), tail["Z"].to_numpy(float))

        g = ev.merge(sub[["id", "tvt"]], on="id", how="left")
        if g["tvt"].isna().any():
            continue
        frames.append(pd.DataFrame({
            "well": wid,
            "err": g["tvt"].to_numpy(float) - g["TVT"].to_numpy(float),
            "steer": g["Z"].to_numpy(float) - (z_a + b_z * (g["MD"].to_numpy(float) - md_a)),
        }))
    return pd.concat(frames, ignore_index=True)


def fit_beta(df: pd.DataFrame) -> float:
    s, e = df["steer"].to_numpy(), df["err"].to_numpy()
    return float(np.sum(s * e) / max(np.sum(s * s), 1e-12))


def rmse_with(df: pd.DataFrame, beta: float) -> float:
    return float(np.sqrt(np.mean((df["err"].to_numpy() - beta * df["steer"].to_numpy()) ** 2)))


def _fit_beta_arr(steer: np.ndarray, err: np.ndarray) -> float:
    return float(np.sum(steer * err) / max(np.sum(steer * steer), 1e-12))


def _rmse_arr(err: np.ndarray, steer: np.ndarray, beta: float) -> float:
    return float(np.sqrt(np.mean((err - beta * steer) ** 2)))


def _random_group(rng: np.random.Generator, n_wells: int) -> np.ndarray:
    """A random near-even partition of well indices into two groups (True=A, False=B)."""
    order = rng.permutation(n_wells)
    group_a = np.zeros(n_wells, dtype=bool)
    group_a[order[: n_wells // 2]] = True
    return group_a


def _split_delta(err: np.ndarray, steer: np.ndarray, row_well: np.ndarray,
                  group_a: np.ndarray) -> float:
    """Pooled out-of-sample RMSE delta (beta vs beta=0) for one random 2-fold split."""
    mask_a = group_a[row_well]
    mask_b = ~mask_a
    sse = base_sse = 0.0
    n = 0
    for src, dst in ((mask_a, mask_b), (mask_b, mask_a)):
        beta = _fit_beta_arr(steer[src], err[src])
        got = _rmse_arr(err[dst], steer[dst], beta)
        base = _rmse_arr(err[dst], steer[dst], 0.0)
        sse += got**2 * dst.sum()
        base_sse += base**2 * dst.sum()
        n += int(dst.sum())
    return float(np.sqrt(sse / n) - np.sqrt(base_sse / n))


def repeated_splits(df: pd.DataFrame, wells: list[str], n_splits: int = N_SPLITS,
                     seed: int = 0) -> np.ndarray:
    """Pooled out-of-sample delta for n_splits independent random 2-fold splits by well."""
    well_idx = {w: i for i, w in enumerate(wells)}
    row_well = df["well"].map(well_idx).to_numpy()
    err_all, steer_all = df["err"].to_numpy(), df["steer"].to_numpy()
    rng = np.random.default_rng(seed)
    return np.array([
        _split_delta(err_all, steer_all, row_well, _random_group(rng, len(wells)))
        for _ in range(n_splits)
    ])


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = Path(argv[1])
    df = collect(d)
    wells = sorted(df["well"].unique())
    print(f"{d.name}: wells={len(wells)} rows={len(df)}  baseline RMSE={rmse_with(df, 0.0):.4f} ft")

    beta_all = fit_beta(df)
    got_all, base_all = rmse_with(df, beta_all), rmse_with(df, 0.0)
    print(f"\n  full-sample beta = {beta_all:+.5f}   in-sample RMSE "
          f"{got_all:.4f} ({got_all - base_all:+.4f})")

    print("\n  cross-fit (single sha256 split, for comparison with the historical number):")
    half = {w: int(hashlib.sha256(w.encode()).hexdigest(), 16) % 2 for w in wells}
    df["half"] = df["well"].map(half)
    a, b = df[df["half"] == 0], df[df["half"] == 1]
    print(f"    split: {a['well'].nunique()} wells / {b['well'].nunique()} wells")
    tot_sse, tot_n, base_sse = 0.0, 0, 0.0
    for src, dst, name in ((a, b, "A->B"), (b, a, "B->A")):
        beta = fit_beta(src)
        got, base = rmse_with(dst, beta), rmse_with(dst, 0.0)
        print(f"    {name}: beta={beta:+.5f}  {base:.4f} -> {got:.4f}  ({got - base:+.4f})")
        tot_sse += got**2 * len(dst)
        base_sse += base**2 * len(dst)
        tot_n += len(dst)
    oos, oos_base = np.sqrt(tot_sse / tot_n), np.sqrt(base_sse / tot_n)
    print(f"    pooled out-of-sample: {oos_base:.4f} -> {oos:.4f}  ({oos - oos_base:+.4f})")

    print(f"\n  repeated random 2-fold splits by well (N={N_SPLITS}, seed=0):")
    deltas = repeated_splits(df, wells)
    print(f"    median {np.median(deltas):+.4f}  sd {deltas.std(ddof=1):.4f}  "
          f"p5 {np.percentile(deltas, 5):+.4f}  p95 {np.percentile(deltas, 95):+.4f}  "
          f"helps(delta<0) {100 * float((deltas < 0).mean()):.0f}%")

    per = df.groupby("well").apply(
        lambda g: rmse_with(g, fit_beta(g)) ** 2 * len(g), include_groups=False)
    n = df.groupby("well").size()
    print(f"\n  per-well oracle beta (ceiling): {np.sqrt(per.sum() / n.sum()):.4f} ft")

    betas = df.groupby("well").apply(fit_beta, include_groups=False)
    print(f"  per-well beta: median {betas.median():+.4f}  IQR "
          f"[{betas.quantile(.25):+.4f}, {betas.quantile(.75):+.4f}]  "
          f"share>0 {100*(betas > 0).mean():.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
