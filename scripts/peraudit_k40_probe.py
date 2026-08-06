"""Single-split probe: train on the 733 non-k40 wells, predict the 40 k40 wells.

Perfectly honest (the 40 scored wells are never in training) and ~5x cheaper
than 5-fold, which lets us run every feature-family ablation AND the decisive
question in one pass:

  does the per-well drift model add anything ON TOP of the production pipeline
  (pooled RMSE 8.172 on exactly these 40 wells / 199,746 rows)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
CACHE = Path(
    "/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/peraudit"
)
H = Path(
    "/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/"
    "harness_audit/k40"
)
STRIDE = 8
NEST = 400


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    import lightgbm as lgb
    from peraudit_ablate import GROUPS

    meta = json.loads((CACHE / "meta.json").read_text())
    feats = meta["features"]
    fi = {f: i for i, f in enumerate(feats)}
    wells = [s["well"] for s in meta["wells"]]
    n_traj = meta["wells"][0]["n_traj"]
    n_gr = meta["wells"][0]["n_gr"]
    y = np.load(CACHE / "y.npy")
    g = np.load(CACHE / "well.npy")
    row = np.load(CACHE / "row.npy")
    X = np.load(CACHE / "X.npy", mmap_mode="r")

    k40 = set(json.loads((H / "cv_summary.json").read_text())["well_ids"])
    kidx = np.array([i for i, w in enumerate(wells) if w in k40])
    va = np.isin(g, kidx)
    tr = np.where(~va)[0][::STRIDE]
    vi = np.where(va)[0]
    print(f"train rows {len(tr):,} ({len(wells) - len(kidx)} wells)  "
          f"valid rows {len(vi):,} ({len(kidx)} wells)")

    groups = dict(GROUPS)
    groups["A4 full trajectory"] = feats[:n_traj]
    groups["A5 traj+heelGR"] = feats[: n_traj + n_gr]
    groups["A6 all features"] = feats

    preds = {}
    print(f"\ncarry-last on k40: {rmse(y[vi], 0.0):.3f}")
    for name, names in groups.items():
        cols = [fi[f] for f in names]
        Xtr = np.ascontiguousarray(X[tr][:, cols])
        ds = lgb.Dataset(Xtr, label=y[tr])
        del Xtr
        m = lgb.train(
            dict(objective="l2", learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=300, feature_fraction=0.75,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                 num_threads=4, verbosity=-1, seed=0),
            ds, num_boost_round=NEST,
        )
        del ds
        p = np.zeros(len(vi))
        for s in range(0, len(vi), 100_000):
            sl = vi[s : s + 100_000]
            p[s : s + len(sl)] = m.predict(np.ascontiguousarray(X[sl][:, cols]))
        preds[name] = p
        imp = m.feature_importance("gain")
        top = [names[i] for i in np.argsort(-imp)[:6]]
        print(f"  {name:22s} ({len(cols):2d}f) k40 RMSE {rmse(y[vi], p):7.3f}   top={top}",
              flush=True)
        del m

    # ------- decisive: does it add on top of the production pipeline? -------
    from rogii.data import data_root

    sub = pd.read_csv(H / "latest_valid_submission.csv")
    sub["well"] = sub["id"].str.split("_").str[0]
    sub["rw"] = sub["id"].str.split("_").str[1].astype(int)
    key = pd.DataFrame({"well": [wells[i] for i in g[vi]], "rw": row[vi],
                        "ord": np.arange(len(vi))})
    s = sub.merge(key, on=["well", "rw"], how="inner").sort_values("ord")
    assert len(s) == len(vi) == len(sub), (len(s), len(vi), len(sub))
    truth = np.zeros(len(vi))
    carry = np.zeros(len(vi))
    for w in sorted(k40):
        h = pd.read_csv(data_root() / "train" / f"{w}__horizontal_well.csv",
                        usecols=["TVT", "TVT_input"])
        ti = h["TVT_input"].to_numpy()
        tv = h["TVT"].to_numpy()
        m = s["well"].to_numpy() == w
        truth[m] = tv[s["rw"].to_numpy()[m]]
        carry[m] = float(ti[~np.isnan(ti)][-1])
    assert np.allclose(y[vi], truth - carry, atol=1e-3)
    pipe = s["tvt"].to_numpy()
    resid = truth - pipe
    base = rmse(truth, pipe)
    print(f"\nproduction pipeline on k40: {base:.4f}   (carry-last {rmse(truth, carry):.4f})")
    wv = s["well"].to_numpy()
    for name, p in preds.items():
        r = float(np.corrcoef(p, resid)[0, 1])
        # leave-one-well-out blend weight: honest on these 40 wells
        pr = pipe.copy()
        for w in sorted(k40):
            m = wv == w
            o = ~m
            a = float(np.sum(p[o] * resid[o]) / max(np.sum(p[o] ** 2), 1e-9))
            pr[m] = pipe[m] + a * p[m]
        best = min((rmse(truth, pipe + a * p), a) for a in np.arange(-0.6, 1.01, 0.05))
        print(f"  {name:22s} corr(pred,resid)={r:+.3f}  "
              f"LOWO-blend {rmse(truth, pr):7.4f} ({base - rmse(truth, pr):+.4f})  "
              f"best-in-sample {best[0]:7.4f} @a={best[1]:+.2f} ({base - best[0]:+.4f})")

    # structure of the pipeline's own residual
    print("\noracle decomposition of the PIPELINE residual (k40, in-sample per well):")
    dmd = np.asarray(X[vi, fi["dmd"]], np.float64)
    dz = np.asarray(X[vi, fi["dz"]], np.float64)
    acc: dict[str, list[float]] = {}
    for w in sorted(k40):
        m = wv == w
        rr = resid[m]
        one = np.ones(m.sum())
        for nm, A in (("raw", None), ("per-well const", one[:, None]),
                      ("per-well lin(md)", np.c_[dmd[m], one]),
                      ("per-well lin(md,dz)", np.c_[dmd[m], dz[m], one])):
            e = rr if A is None else rr - A @ np.linalg.lstsq(A, rr, rcond=None)[0]
            acc.setdefault(nm, [0.0, 0.0])
            acc[nm][0] += float(np.sum(e**2))
            acc[nm][1] += len(e)
    for k, v in acc.items():
        print(f"  {k:22s} {np.sqrt(v[0] / v[1]):8.4f}")
    np.save(CACHE / "k40_probe_preds.npy", preds["A6 all features"])


if __name__ == "__main__":
    main()
