"""Which feature family carries the trajectory-only gain?

Same folds / same target as peraudit_cv.py. Each group is a strict subset of the
trajectory-only feature block, so the comparison isolates what the GBDT is
actually using:

  A0  "how far along"     -- dmd, log_dmd, frac_along, eval_len  (no trajectory
                             SHAPE at all; can only learn E[d | distance], i.e.
                             a global drift curve + heteroscedastic shrinkage)
  A1  A0 + well geometry  -- tail slopes of Z, slope stability, displacement
  A2  A0 + dZ family      -- cumulative Z change, deviation from the tail line,
                             look-ahead/behind dZ  (the steering-leak carriers)
  A3  A0 + local shape    -- local slopes, curvature, dogleg, sign flips
  A4  full trajectory block
"""

from __future__ import annotations

import json
import os

import numpy as np
from peraudit_cv import CACHE, K40, NFOLD, SEED, fit_predict_lgb, per_well_table, rmse

GROUPS = {
    "A0 distance-only": ["dmd", "log_dmd", "frac_along", "eval_len"],
    "A1 +well geometry": [
        "dmd", "log_dmd", "frac_along", "eval_len",
        "zsl200", "zsl500", "zsl1500", "z_slope_stab", "hd", "dz_over_hd",
    ],
    "A2 +dZ family": [
        "dmd", "log_dmd", "frac_along", "eval_len",
        "dz", "dz_dev_tail", "dz_dev_tail200", "dz_dev_tail1500",
        "lookdz-800", "lookdz-300", "lookdz-100", "lookdz100", "lookdz300",
        "lookdz800", "z_minus_evalmean", "z_rank_eval", "z_minus_evalmin",
        "z_minus_evalmax",
    ],
    "A3 +local shape": [
        "dmd", "log_dmd", "frac_along", "eval_len",
        "slope11", "slope51", "slope151", "slope401", "slope1001",
        "slope11_rel", "slope51_rel", "slope151_rel", "slope401_rel",
        "slope1001_rel", "slope_ratio", "curv51", "abscurv_r", "cum_abscurv",
        "cum_abscurv_rate", "flips", "flips_rate", "dogleg", "slopestd201",
        "slopestd801",
    ],
}


def main() -> None:
    meta = json.loads((CACHE / "meta.json").read_text())
    feats = meta["features"]
    fi = {f: i for i, f in enumerate(feats)}
    wells = [s["well"] for s in meta["wells"]]
    nw = len(wells)
    n_traj = meta["wells"][0]["n_traj"]
    X = np.load(CACHE / "X.npy", mmap_mode="r")
    y = np.load(CACHE / "y.npy")
    g = np.load(CACHE / "well.npy")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(nw)
    fold_of = np.zeros(nw, int)
    for i, w in enumerate(perm):
        fold_of[w] = i % NFOLD
    folds = [
        (np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(NFOLD)
    ]
    k40 = set(json.loads(K40.read_text())["well_ids"])
    m40 = np.isin(g, np.array([i for i, w in enumerate(wells) if w in k40]))

    out = []
    groups = dict(GROUPS)
    groups["A4 full trajectory"] = feats[:n_traj]
    groups["A5 all features"] = feats
    for name, names in groups.items():
        cols = [fi[f] for f in names]
        print(f"  {name} ({len(cols)} feats)", flush=True)
        # Materialise just these columns in RAM once: fancy-indexing the 1 GB
        # memmap inside every fold was the real bottleneck (I/O, not LightGBM).
        Xg = np.ascontiguousarray(X[:, cols])
        oof, imp = fit_predict_lgb(Xg, y, g, list(range(len(cols))), folds)
        del Xg
        np.save(CACHE / f"oof_{name.split()[0]}.npy", oof)
        pw = per_well_table(y, oof, g, nw)
        r = dict(
            model=name,
            n_feats=len(cols),
            pooled=rmse(y, oof),
            k40=rmse(y[m40], oof[m40]),
            well_median=float(np.nanmedian(pw)),
            top=sorted(
                ((names[i], float(imp[i])) for i in range(len(cols))), key=lambda t: -t[1]
            )[:8],
        )
        out.append(r)
        print(f"{name:24s} pooled={r['pooled']:7.3f} k40={r['k40']:7.3f}", flush=True)
    (CACHE / "ablation.json").write_text(json.dumps(out, indent=1))
    print("\ncarry-last floor: pooled 15.910 / k40 17.142")
    for r in out:
        print(f"  {r['model']:24s} {r['pooled']:7.3f} {r['k40']:7.3f}  "
              f"top={[t[0] for t in r['top'][:4]]}")


if __name__ == "__main__":
    os.environ.setdefault("PA_NEST", "300")
    main()
