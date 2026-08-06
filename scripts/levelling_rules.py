"""Steps 4-5: five gating rule families, each cross-fitted over the 40 wells.

The inverse-variance gate of the first attempt failed for a specific, instructive
reason: a well whose spatial leg is 120 ft wrong still receives w=0.05 from a
calibrator that predicted 10 ft, and 5% of a 120 ft error is a 6 ft hit.  Any rule
that spreads a little weight everywhere is therefore hostage to the tail.  So the
families below are all ways of refusing weight rather than shading it:

  const     one global w                                        (1 param)
  invvar    w = min(w_max, sigma^2/(sigma^2 + s_hat^2))          (2 params)
  hard      w = w_max where s_hat <= T, else 0                   (2 params)
  clip      global w on a spatial leg clipped to +/-L of the pipeline (2 params)
  disagree  w = w_max where median|spat - pipe| <= D, else 0     (2 params)
  cliprel   as clip, but L = kappa * MAD(spat - pipe) pooled over
            all eval rows, so the cap rescales with the prediction
            spread instead of being frozen in feet                 (2 params)

`s_hat` comes from the out-of-40 calibration; the remaining scalars are fitted on
one half of the 40 and scored on the other, both directions, over 200 random
partitions.  Picking the winner among five families is itself a selection made on
these 40 wells, so all five are reported.

Usage:
    uv run python scripts/levelling_rules.py [variant] [levelling] [anchor]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE  # noqa: E402
from levelling_gate_blend import (  # noqa: E402
    apply_calibration,
    fit_calibration,
    pooled,
    worst_share,
)

N_SPLITS = 200
W_GRID = np.linspace(0.0, 1.0, 21)
WMAX_GRID = np.array([0.25, 0.5, 0.75, 1.0])
SIGMA_GRID = np.concatenate([[0.0], np.geomspace(0.5, 40.0, 40)])
CLIP_GRID = np.concatenate([np.geomspace(0.5, 200.0, 30), [np.inf]])
KAPPA_GRID = np.concatenate([np.geomspace(0.1, 40.0, 30), [np.inf]])


def _thr_grid(v: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([[0.0], np.quantile(v, np.linspace(0, 1, 25)), [np.inf]]))


class Rule:
    """A gating family: enumerate parameter settings, each mapping to a row blend."""

    def __init__(self, name: str, params: list[tuple], apply_fn):
        self.name = name
        self.params = params
        self.apply_fn = apply_fn

    def predict(self, p, pipe, spat, wpw, rw) -> np.ndarray:
        return self.apply_fn(p, pipe, spat, wpw, rw)

    def fit(self, pipe, spat, truth, wpw, rw):
        best, arg = np.inf, self.params[0]
        for p in self.params:
            r = pooled(self.predict(p, pipe, spat, wpw, rw), truth)
            if r < best:
                best, arg = r, p
        return arg


def make_rules(shat: np.ndarray, dis: np.ndarray, mad: float = 1.0) -> list[Rule]:
    def mix(pipe, spat, w):
        return (1.0 - w) * pipe + w * spat

    return [
        Rule("const", [(w,) for w in W_GRID],
             lambda p, pipe, spat, wpw, rw: mix(pipe, spat, p[0])),
        Rule("invvar", [(s, wm) for s in SIGMA_GRID for wm in WMAX_GRID],
             lambda p, pipe, spat, wpw, rw: mix(
                 pipe, spat,
                 (np.minimum(p[1], p[0] ** 2 / (p[0] ** 2 + wpw["shat"] ** 2))
                  if p[0] > 0 else np.zeros(len(wpw["shat"])))[rw])),
        Rule("hard", [(t, wm) for t in _thr_grid(shat) for wm in WMAX_GRID],
             lambda p, pipe, spat, wpw, rw: mix(
                 pipe, spat, (p[1] * (wpw["shat"] <= p[0]))[rw])),
        Rule("clip", [(w, L) for w in W_GRID for L in CLIP_GRID],
             lambda p, pipe, spat, wpw, rw: mix(
                 pipe, pipe + np.clip(spat - pipe, -p[1], p[1]), p[0])),
        Rule("cliprel", [(w, kp) for w in W_GRID for kp in KAPPA_GRID],
             lambda p, pipe, spat, wpw, rw: mix(
                 pipe, pipe + np.clip(spat - pipe, -p[1] * mad, p[1] * mad), p[0])),
        Rule("disagree", [(t, wm) for t in _thr_grid(dis) for wm in WMAX_GRID],
             lambda p, pipe, spat, wpw, rw: mix(
                 pipe, spat, (p[1] * (wpw["dis"] <= p[0]))[rw])),
    ]


def crossfit(rule: Rule, pipe, spat, truth, wpw, rw, n_wells, n_splits=N_SPLITS, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        order = rng.permutation(n_wells)
        ga = np.zeros(n_wells, dtype=bool)
        ga[order[: n_wells // 2]] = True
        ma = ga[rw]
        sse = base = 0.0
        n = 0
        for src, dst in ((ma, ~ma), (~ma, ma)):
            p = rule.fit(pipe[src], spat[src], truth[src], wpw, rw[src])
            got = pooled(rule.predict(p, pipe[dst], spat[dst], wpw, rw[dst]), truth[dst])
            sse += got**2 * dst.sum()
            base += pooled(pipe[dst], truth[dst]) ** 2 * dst.sum()
            n += int(dst.sum())
        out.append((float(np.sqrt(base / n)), float(np.sqrt(sse / n))))
    a = np.array(out)
    return a[:, 1] - a[:, 0], a[:, 1]


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
    df = df[np.isfinite(df["spat"])].reset_index(drop=True)

    beta, cols = fit_calibration(cal)
    ft["shat"] = apply_calibration(ft, beta, cols)
    dis = df.assign(a=np.abs(df["spat"] - df["pipe"])).groupby("well")["a"].median()
    ft = ft.merge(dis.rename("dis"), left_on="well", right_index=True)

    wells = sorted(df["well"].unique())
    ft = ft[ft["well"].isin(wells)].set_index("well").loc[wells].reset_index()
    rw = df["well"].map({w: i for i, w in enumerate(wells)}).to_numpy()
    pipe, spat, truth = (df[c].to_numpy(float) for c in ("pipe", "spat", "truth"))
    wpw = {"shat": ft["shat"].to_numpy(float), "dis": ft["dis"].to_numpy(float)}
    mad = float(np.median(np.abs(spat - pipe)))

    print(f"\n=== variant {variant} / levelling {lev} / anchor {anchor} ===")
    print(f"rows {len(df)}  wells {len(wells)}")
    base = pooled(pipe, truth)
    print(f"  pipeline alone        {base:8.4f} ft")
    print(f"  levelled-IDW alone    {pooled(spat, truth):8.4f} ft")
    print(f"\n  cross-fitted over {N_SPLITS} random 2-fold well splits "
          f"(delta < 0 means the blend beat the pipeline):")
    print(f"    {'rule':10s} {'pooled med':>10s} {'delta med':>10s} {'p5':>8s} "
          f"{'p95':>8s} {'helps':>6s} {'in-sample':>10s}")
    results = {}
    for rule in make_rules(wpw["shat"], wpw["dis"], mad):
        d, g = crossfit(rule, pipe, spat, truth, wpw, rw, len(wells))
        p_in = rule.fit(pipe, spat, truth, wpw, rw)
        ins = pooled(rule.predict(p_in, pipe, spat, wpw, rw), truth)
        results[rule.name] = (d, g, p_in, ins)
        print(f"    {rule.name:10s} {np.median(g):10.4f} {np.median(d):+10.4f} "
              f"{np.percentile(d, 5):+8.4f} {np.percentile(d, 95):+8.4f} "
              f"{100 * float((d < 0).mean()):5.0f}% {ins:10.4f}")

    best = min(results, key=lambda k: np.median(results[k][0]))
    d, g, p_in, ins = results[best]
    print(f"\n  best cross-fitted family: {best}  (in-sample params {p_in})")
    rule = [r for r in make_rules(wpw["shat"], wpw["dis"], mad) if r.name == best][0]
    blended = rule.predict(p_in, pipe, spat, wpw, rw)
    e0, e1 = (pipe - truth) ** 2, (blended - truth) ** 2
    print(f"  worst 5% of rows share of squared error: "
          f"before {100 * worst_share(e0):.1f}%  after {100 * worst_share(e1):.1f}%")

    print(f"  pooled |spat-pipe| MAD = {mad:.2f} ft; cap in MAD units for the "
          f"in-sample clip pick: {p_in[1] / mad:.2f}")

    # placebo 1: flip the sign of the spatial correction, which preserves its
    # magnitude and its per-row structure but destroys its direction
    # placebo 2: give each well another well's correction curve
    rng = np.random.default_rng(7)
    for label, sp2 in (("sign-flip", pipe - (spat - pipe)),
                       ("well-permuted", None)):
        if sp2 is None:
            perm = rng.permutation(len(wells))
            off = {i: j for i, j in enumerate(perm)}
            corr = spat - pipe
            sp2 = pipe.copy()
            for i in range(len(wells)):
                src_m, dst_m = rw == off[i], rw == i
                c = corr[src_m]
                if c.size == 0:
                    continue
                reps = int(np.ceil(dst_m.sum() / c.size))
                sp2[dst_m] = pipe[dst_m] + np.tile(c, reps)[: dst_m.sum()]
        d2, g2 = crossfit(rule, pipe, sp2, truth, wpw, rw, len(wells), n_splits=40)
        print(f"  placebo {label:14s} cross-fitted delta median {np.median(d2):+.4f} "
              f"(real {np.median(d):+.4f})")

    # leave-one-well-out: is the gain carried by one well?
    lowo = []
    for i, _wl in enumerate(wells):
        m = rw != i
        rw2 = rw[m]
        remap = {v: k for k, v in enumerate(sorted(set(rw2)))}
        rw2 = np.array([remap[v] for v in rw2])
        keep = sorted(set(rw[m]))
        w2 = {k: v[keep] for k, v in wpw.items()}
        d3, _ = crossfit(rule, pipe[m], spat[m], truth[m], w2, rw2, len(keep), n_splits=25)
        lowo.append(float(np.median(d3)))
    print(f"  leave-one-well-out cross-fitted delta: range "
          f"[{min(lowo):+.4f}, {max(lowo):+.4f}]  median {np.median(lowo):+.4f}")

    per = df.assign(e0=e0, e1=e1).groupby("well").agg(
        n=("truth", "size"), rmse_pipe=("e0", lambda x: float(np.sqrt(x.mean()))),
        rmse_blend=("e1", lambda x: float(np.sqrt(x.mean())))).reset_index()
    per = per.merge(ft[["well", "hb_rmse", "shat", "dis", "dnn_ev"]], on="well")
    per["delta"] = per["rmse_blend"] - per["rmse_pipe"]
    per = per.sort_values("delta")
    per.to_csv(CACHE / f"rules_perwell_{tag}_{anchor}.csv", index=False)
    print(f"\n{per.round(3).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
