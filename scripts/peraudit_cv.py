"""Honest GroupKFold-by-well information audit of the eval-zone drift.

Ladder:
  L0  carry-last (predict d = 0)                          -- no-information floor
  L1  linear extrapolation of the known-zone tail slope   -- 1 param, no fit
  L1s cross-fitted global shrinkage on the tail slope     -- 1 fitted param
  L2  trajectory-only GBDT
  L3  L2 + heel-calibrated GR
  L4  L3 + known-zone summary stats  (all features)
  O*  oracle decompositions (upper bounds on what per-well info could give)

Every number is out-of-fold: no well appears in both train and valid.
No pretrained assets: the heel calibration, the typewell lookups and the model
are all computed inside the fold (heel calibration is per-well and uses only
that well's known zone, so it is fold-independent by construction).
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

CACHE = Path(
    os.environ.get("ROGII_WORK", "work") + "/peraudit"
)
K40 = Path(
    os.environ.get("ROGII_WORK", "work") + "/"
    "harness_audit/k40/cv_summary.json"
)
NFOLD = 5
SEED = 0
TRAIN_STRIDE = int(__import__("os").environ.get("PA_STRIDE", 4))  # subsample training rows (3.8 GB RAM box); scoring uses all rows


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load():
    X = np.load(CACHE / "X.npy", mmap_mode="r")
    y = np.load(CACHE / "y.npy")
    g = np.load(CACHE / "well.npy")
    meta = json.loads((CACHE / "meta.json").read_text())
    return X, y, g, meta


def fit_predict_lgb(X, y, g, cols, folds, seed=0, n_est=int(__import__("os").environ.get("PA_NEST", 700))):
    import lightgbm as lgb

    oof = np.zeros(len(y), np.float32)
    imps = np.zeros(len(cols))
    for f, (tr_w, va_w) in enumerate(folds):
        tr = np.isin(g, tr_w)
        va = np.isin(g, va_w)
        tri = np.where(tr)[0][::TRAIN_STRIDE]
        Xtr = np.asarray(X[tri][:, cols])
        ds = lgb.Dataset(Xtr, label=y[tri], free_raw_data=True)
        del Xtr
        params = dict(
            objective="l2",
            metric="l2",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=300,
            feature_fraction=0.75,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=5.0,
            num_threads=4,
            verbosity=-1,
            seed=seed,
        )
        m = lgb.train(params, ds, num_boost_round=n_est)
        imps += m.feature_importance("gain")
        vai = np.where(va)[0]
        for s in range(0, len(vai), 200_000):
            sl = vai[s : s + 200_000]
            oof[sl] = m.predict(np.asarray(X[sl][:, cols])).astype(np.float32)
        print(f"    fold {f}: valid rmse {rmse(y[va], oof[va]):.4f}", flush=True)
        del ds, m
    return oof, imps / NFOLD


def per_well_table(y, pred, g, nw):
    out = np.zeros(nw)
    for w in range(nw):
        m = g == w
        out[w] = rmse(y[m], pred[m]) if m.any() else np.nan
    return out


def main() -> None:
    X, y, g, meta = load()
    feats = meta["features"]
    wells = [s["well"] for s in meta["wells"]]
    nw = len(wells)
    n_traj = meta["wells"][0]["n_traj"]
    n_gr = meta["wells"][0]["n_gr"]
    print(f"rows={len(y):,} wells={nw} feats={len(feats)} "
          f"(traj={n_traj}, gr={n_gr}, known={len(feats) - n_traj - n_gr})")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(nw)
    fold_of = np.zeros(nw, int)
    for i, w in enumerate(perm):
        fold_of[w] = i % NFOLD
    folds = [
        (np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(NFOLD)
    ]
    # fold hygiene assertion
    for tr_w, va_w in folds:
        assert not set(tr_w) & set(va_w)
    assert sorted(np.concatenate([v for _, v in folds])) == list(range(nw))

    k40 = set(json.loads(K40.read_text())["well_ids"])
    k40_idx = np.array([i for i, w in enumerate(wells) if w in k40])
    m40 = np.isin(g, k40_idx)
    print(f"k40 overlap: {len(k40_idx)}/40 wells, {m40.sum():,} rows")

    fi = {f: i for i, f in enumerate(feats)}
    dmd = np.asarray(X[:, fi["dmd"]], np.float64)
    dz = np.asarray(X[:, fi["dz"]], np.float64)

    results: list[dict] = []

    def report(name: str, pred: np.ndarray, extra: dict | None = None) -> None:
        pw = per_well_table(y, pred, g, nw)
        r = dict(
            model=name,
            pooled=rmse(y, pred),
            k40=rmse(y[m40], pred[m40]),
            well_median=float(np.nanmedian(pw)),
            well_p90=float(np.nanpercentile(pw, 90)),
            well_mean=float(np.nanmean(pw)),
        )
        if extra:
            r.update(extra)
        results.append(r)
        print(
            f"{name:34s} pooled={r['pooled']:7.3f}  k40={r['k40']:7.3f}  "
            f"med={r['well_median']:6.3f} p90={r['well_p90']:7.3f}",
            flush=True,
        )
        return pw

    # ---------------- L0 carry-last ----------------
    report("L0 carry-last (d=0)", np.zeros_like(y))

    # ---------------- L1 linear tail ----------------
    for key in ("tsl200", "tsl500", "tsl1500"):
        sl = np.asarray(X[:, fi[key]], np.float64)
        report(f"L1 linear-tail {key}", (sl * dmd).astype(np.float32))
        for cap in (500.0, 1000.0, 2000.0):
            report(
                f"L1 linear-tail {key} cap{int(cap)}",
                (sl * np.minimum(dmd, cap)).astype(np.float32),
            )

    # cross-fitted global shrinkage beta on the best raw tail feature
    for key in ("tsl500", "tsl1500"):
        sl = np.asarray(X[:, fi[key]], np.float64)
        base = sl * dmd
        pred = np.zeros_like(y)
        betas = []
        for tr_w, va_w in folds:
            tr = np.isin(g, tr_w)
            va = np.isin(g, va_w)
            b = float(np.sum(base[tr] * y[tr]) / max(np.sum(base[tr] ** 2), 1e-9))
            betas.append(b)
            pred[va] = (b * base[va]).astype(np.float32)
        report(f"L1s xfit-shrunk {key}", pred, {"beta": float(np.mean(betas))})

    # steering-leak single-parameter probes (cross-fitted)
    for nm, base in (
        ("dz", dz),
        ("dz_dev_tail", np.asarray(X[:, fi["dz_dev_tail"]], np.float64)),
    ):
        pred = np.zeros_like(y)
        betas = []
        for tr_w, va_w in folds:
            tr = np.isin(g, tr_w)
            va = np.isin(g, va_w)
            b = float(np.sum(base[tr] * y[tr]) / max(np.sum(base[tr] ** 2), 1e-9))
            betas.append(b)
            pred[va] = (b * base[va]).astype(np.float32)
        report(f"L1s xfit-shrunk {nm}", pred, {"beta": float(np.mean(betas))})

    # ---------------- L2/L3/L4 GBDT ladder ----------------
    ladders = [
        ("L2 trajectory-only GBDT", list(range(n_traj))),
        ("L3 +heel-GR GBDT", list(range(n_traj + n_gr))),
        ("L4 +known-zone-stats GBDT", list(range(len(feats)))),
    ]
    imps_out = {}
    oofs = {}
    for name, cols in ladders:
        print(f"  training {name} ({len(cols)} feats)", flush=True)
        oof, imp = fit_predict_lgb(X, y, g, cols, folds)
        oofs[name] = oof
        imps_out[name] = sorted(
            ((feats[c], float(imp[i])) for i, c in enumerate(cols)),
            key=lambda t: -t[1],
        )[:25]
        report(name, oof)

    # GR-only sanity (does GR carry anything without the trajectory?)
    print("  training GRonly", flush=True)
    gr_cols = list(range(n_traj, n_traj + n_gr)) + [fi["dmd"], fi["log_dmd"]]
    oof_gr, _ = fit_predict_lgb(X, y, g, gr_cols, folds)
    report("Lx GR-only(+dmd) GBDT", oof_gr)

    # ---------------- Oracle bounds ----------------
    o_const = np.zeros_like(y)
    o_lin = np.zeros_like(y)
    o_dz = np.zeros_like(y)
    o_lin_dz = np.zeros_like(y)
    for w in range(nw):
        m = g == w
        yy = y[m].astype(np.float64)
        o_const[m] = yy.mean()
        A = np.c_[dmd[m], np.ones(m.sum())]
        o_lin[m] = A @ np.linalg.lstsq(A, yy, rcond=None)[0]
        B = np.c_[dz[m], np.ones(m.sum())]
        o_dz[m] = B @ np.linalg.lstsq(B, yy, rcond=None)[0]
        C = np.c_[dmd[m], dz[m], np.ones(m.sum())]
        o_lin_dz[m] = C @ np.linalg.lstsq(C, yy, rcond=None)[0]
    report("O1 oracle per-well constant", o_const)
    report("O2 oracle per-well linear(dmd)", o_lin)
    report("O3 oracle per-well linear(dz)", o_dz)
    report("O4 oracle per-well lin(dmd,dz)", o_lin_dz)

    # oracle: per-well affine rescale of the best learned model (how much of the
    # residual is a per-well bias/gain the model could not know?)
    best = oofs["L4 +known-zone-stats GBDT"]
    o_cal = np.zeros_like(y)
    for w in range(nw):
        m = g == w
        A = np.c_[best[m].astype(np.float64), np.ones(m.sum())]
        o_cal[m] = A @ np.linalg.lstsq(A, y[m].astype(np.float64), rcond=None)[0]
    report("O5 oracle affine-recal of L4", o_cal)

    out = dict(results=results, importances=imps_out, n_rows=int(len(y)), n_wells=nw)
    (CACHE / "cv_results.json").write_text(json.dumps(out, indent=1))
    np.save(CACHE / "oof_L4.npy", best)
    np.save(CACHE / "oof_L2.npy", oofs["L2 trajectory-only GBDT"])
    print("\nsaved ->", CACHE / "cv_results.json")


if __name__ == "__main__":
    main()
