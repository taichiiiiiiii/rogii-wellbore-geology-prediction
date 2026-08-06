"""Steps 4-5: gate the spatial leg, blend it into the pipeline, and cross-fit honestly.

Two-stage gate.  Stage one -- "how bad will the spatial leg be in the eval zone?" --
is fitted on ~380 train wells that are NOT among the 40 (scripts/levelling_calib.py);
it needs no pipeline prediction, so it costs nothing in evaluation-set information.
Stage two is a single scalar sigma_p, the pipeline error level the spatial leg is
being weighed against, giving the inverse-variance weight

    w_i = min(w_max, sigma_p^2 / (sigma_p^2 + s_hat_i^2))

sigma_p cannot be fitted outside the 40 because pipeline predictions exist only
there, so it is cross-fitted inside them: two halves, fit on one, score the other,
both directions, pooled -- then repeated over 200 random well partitions to show the
spread rather than one flattering draw.

Usage:
    uv run python scripts/levelling_gate_blend.py [variant] [levelling] [anchor]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE  # noqa: E402

N_SPLITS = 200
SIGMA_GRID = np.concatenate([[0.0], np.geomspace(0.5, 40.0, 60)])
WMAX_GRID = np.array([0.5, 0.75, 1.0])


def fit_calibration(cal: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """log-linear model for the spatial leg's eval-zone RMSE.

    Fitted on non-evaluation wells only.  Three coefficients on ~380 wells is a far
    safer ratio than anything that could be fitted inside the 40.
    """
    cols = ["hb_rmse", "dnn_ev", "sd_r_kn"]
    A = np.column_stack([np.ones(len(cal))] + [np.log1p(cal[c].to_numpy(float)) for c in cols])
    y = np.log(cal["sp_rmse"].to_numpy(float))
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ beta
    print(f"calibration on {len(cal)} non-evaluation wells; coefficients "
          + ", ".join(f"{c}={b:+.3f}" for c, b in zip(["const"] + cols, beta)))
    print(f"  in-calibration spearman(pred, actual) = "
          f"{pd.Series(pred).corr(pd.Series(y), method='spearman'):+.3f}")
    return beta, cols


def apply_calibration(ft: pd.DataFrame, beta: np.ndarray, cols: list[str]) -> np.ndarray:
    A = np.column_stack([np.ones(len(ft))] + [np.log1p(ft[c].to_numpy(float)) for c in cols])
    return np.exp(A @ beta)


def pooled(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def blend(pipe: np.ndarray, spat: np.ndarray, wrow: np.ndarray) -> np.ndarray:
    return (1.0 - wrow) * pipe + wrow * spat


def weights(shat: np.ndarray, sigma: float, wmax: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(shat)
    return np.minimum(wmax, sigma**2 / (sigma**2 + shat**2))


def fit_sigma(pipe, spat, truth, wpw, wells_idx) -> tuple[float, float]:
    best, arg = np.inf, (0.0, 1.0)
    for wmax in WMAX_GRID:
        for s in SIGMA_GRID:
            wrow = weights(wpw, s, wmax)[wells_idx]
            r = pooled(blend(pipe, spat, wrow), truth)
            if r < best:
                best, arg = r, (float(s), float(wmax))
    return arg


def crossfit(df: pd.DataFrame, shat: dict[str, float], n_splits: int = N_SPLITS,
             seed: int = 0) -> dict:
    wells = sorted(df["well"].unique())
    widx = {w: i for i, w in enumerate(wells)}
    row_well = df["well"].map(widx).to_numpy()
    pipe = df["pipe"].to_numpy(float)
    spat = df["spat"].to_numpy(float)
    truth = df["truth"].to_numpy(float)
    wpw = np.array([shat[w] for w in wells])

    def one(group_a: np.ndarray) -> tuple[float, float]:
        ma = group_a[row_well]
        sse = base = 0.0
        n = 0
        for src, dst in ((ma, ~ma), (~ma, ma)):
            s, wm = fit_sigma(pipe[src], spat[src], truth[src], wpw, row_well[src])
            wrow = weights(wpw, s, wm)[row_well[dst]]
            got = pooled(blend(pipe[dst], spat[dst], wrow), truth[dst])
            bas = pooled(pipe[dst], truth[dst])
            sse += got**2 * dst.sum()
            base += bas**2 * dst.sum()
            n += int(dst.sum())
        return float(np.sqrt(base / n)), float(np.sqrt(sse / n))

    rng = np.random.default_rng(seed)
    deltas, gots = [], []
    for _ in range(n_splits):
        order = rng.permutation(len(wells))
        ga = np.zeros(len(wells), dtype=bool)
        ga[order[: len(wells) // 2]] = True
        b, g = one(ga)
        deltas.append(g - b)
        gots.append(g)
    return {"deltas": np.array(deltas), "gots": np.array(gots)}


def worst_share(err2: np.ndarray, frac: float = 0.05) -> float:
    k = max(1, int(round(frac * len(err2))))
    return float(np.sort(err2)[-k:].sum() / err2.sum())


def main(argv: list[str]) -> int:
    variant = argv[1] if len(argv) > 1 else "a"
    lev = argv[2] if len(argv) > 2 else "none"
    anchor = argv[3] if len(argv) > 3 else "tail"
    tag = f"{variant}_{lev}"
    rows = pd.read_parquet(CACHE / f"rows_{tag}.parquet")
    ft = pd.read_csv(CACHE / f"feats_{tag}.csv")
    cal = pd.read_csv(CACHE / "calib.csv")

    df = rows[["well", "pipe", "truth"]].copy()
    df["spat"] = rows[f"sp_{anchor}"].to_numpy()
    df = df[np.isfinite(df["spat"])]

    beta, cols = fit_calibration(cal)
    ft["shat"] = apply_calibration(ft, beta, cols)
    shat = dict(zip(ft["well"], ft["shat"]))

    truth = df["truth"].to_numpy()
    pipe = df["pipe"].to_numpy()
    spat = df["spat"].to_numpy()
    print(f"\n=== variant {variant} / levelling {lev} / anchor {anchor} ===")
    print(f"rows {len(df)}  wells {df['well'].nunique()}")
    print(f"  pipeline alone        {pooled(pipe, truth):8.4f} ft")
    print(f"  levelled-IDW alone    {pooled(spat, truth):8.4f} ft")

    res = crossfit(df, shat)
    d, g = res["deltas"], res["gots"]
    base = pooled(pipe, truth)
    print(f"\n  gated blend, cross-fitted over {N_SPLITS} random 2-fold well splits:")
    print(f"    pooled RMSE  median {np.median(g):7.4f}  p5 {np.percentile(g, 5):7.4f}  "
          f"p95 {np.percentile(g, 95):7.4f}")
    print(f"    delta vs pipeline  median {np.median(d):+.4f}  sd {d.std(ddof=1):.4f}  "
          f"p5 {np.percentile(d, 5):+.4f}  p95 {np.percentile(d, 95):+.4f}  "
          f"helps {100 * float((d < 0).mean()):.0f}%")

    s_all, wm_all = fit_sigma(pipe, spat, truth, np.array([shat[w] for w in sorted(shat)]),
                              df["well"].map({w: i for i, w in enumerate(sorted(shat))}).to_numpy())
    wrow_all = weights(np.array([shat[w] for w in sorted(shat)]), s_all, wm_all)[
        df["well"].map({w: i for i, w in enumerate(sorted(shat))}).to_numpy()]
    print(f"\n  (in-sample sigma_p={s_all:.2f} w_max={wm_all:.2f} -> "
          f"{pooled(blend(pipe, spat, wrow_all), truth):.4f} ft; "
          "in-sample only, not a result)")

    e_before = (pipe - truth) ** 2
    e_after = (blend(pipe, spat, wrow_all) - truth) ** 2
    print(f"\n  worst 5% of rows share of squared error: "
          f"before {100 * worst_share(e_before):.1f}%  after {100 * worst_share(e_after):.1f}%")

    per = df.assign(ep=(pipe - truth) ** 2, eb=e_after).groupby("well").agg(
        n=("truth", "size"), rmse_pipe=("ep", lambda x: np.sqrt(x.mean())),
        rmse_blend=("eb", lambda x: np.sqrt(x.mean()))).reset_index()
    per = per.merge(ft[["well", "hb_rmse", "shat", "dnn_ev", "nwells_ev"]], on="well")
    per["w"] = weights(per["shat"].to_numpy(), s_all, wm_all)
    per["delta"] = per["rmse_blend"] - per["rmse_pipe"]
    per = per.sort_values("delta")
    per.to_csv(CACHE / f"perwell_{tag}_{anchor}.csv", index=False)
    print(f"\n{per.round(3).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
