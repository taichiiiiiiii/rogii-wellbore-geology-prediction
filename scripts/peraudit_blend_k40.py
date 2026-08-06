"""Does the drift model carry information the PRODUCTION pipeline does not?

Loads the k40 harness submission (the production pipeline's held-out predictions
for 40 wells, pooled RMSE 8.172), computes its residual, and asks whether the
per-well drift model's out-of-fold prediction explains any of it.

If corr(pipeline residual, drift prediction) ~ 0 and no alpha improves the
pipeline RMSE, the drift model carries nothing new.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CACHE = Path(
    "/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/peraudit"
)
H = Path("/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/harness_audit/k40")


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    from rogii.data import data_root

    meta = json.loads((CACHE / "meta.json").read_text())
    wells = [s["well"] for s in meta["wells"]]
    widx = {w: i for i, w in enumerate(wells)}
    y = np.load(CACHE / "y.npy").astype(np.float64)
    g = np.load(CACHE / "well.npy")
    row = np.load(CACHE / "row.npy")
    oof = {p.stem[4:]: np.load(p).astype(np.float64) for p in sorted(CACHE.glob("oof_*.npy")) if "smoke" not in p.stem}

    sub = pd.read_csv(H / "latest_valid_submission.csv")
    sub["well"] = sub["id"].str.split("_").str[0]
    sub["row"] = sub["id"].str.split("_").str[1].astype(int)
    k40 = sorted(sub["well"].unique())
    print(f"submission: {len(sub):,} rows, {len(k40)} wells")

    # index my cache rows by (well, row)
    recs = []
    for w in k40:
        if w not in widx:
            print(f"  !! {w} missing from cache")
            continue
        wi = widx[w]
        m = np.where(g == wi)[0]
        key = pd.DataFrame({"row": row[m], "pos": m})
        s = sub[sub.well == w].merge(key, on="row", how="inner")
        assert len(s) == len(sub[sub.well == w]), f"{w}: row mismatch"
        h = pd.read_csv(data_root() / "train" / f"{w}__horizontal_well.csv",
                        usecols=["TVT", "TVT_input"])
        ti = h["TVT_input"].to_numpy()
        tvt_a = float(ti[~np.isnan(ti)][-1])
        s["tvt_true"] = h["TVT"].to_numpy()[s["row"].to_numpy()]
        s["tvt_a"] = tvt_a
        recs.append(s)
    d = pd.concat(recs, ignore_index=True)
    pos = d["pos"].to_numpy()
    truth = d["tvt_true"].to_numpy()
    pipe = d["tvt"].to_numpy()
    carry = d["tvt_a"].to_numpy()

    # consistency: my cached target must equal truth - anchor
    assert np.allclose(y[pos], truth - carry, atol=1e-3)
    resid = truth - pipe
    print(f"pipeline pooled RMSE      {rmse(truth, pipe):8.4f}")
    print(f"carry-last pooled RMSE    {rmse(truth, carry):8.4f}")
    print(f"pipeline drift capture    {1 - np.var(resid) / np.var(truth - carry):.3f}")

    for k, o in oof.items():
        p = o[pos]
        print(f"\n--- drift model {k} ---")
        print(f"  standalone RMSE       {rmse(truth, carry + p):8.4f}")
        print(f"  corr(pred, pipe-resid) {np.corrcoef(p, resid)[0, 1]:8.4f}")
        best = min(
            ((rmse(truth, pipe + a * p), a) for a in np.arange(-0.5, 1.01, 0.05)),
        )
        print(f"  best in-sample blend   {best[0]:8.4f} at alpha={best[1]:.2f}"
              f"  (gain {rmse(truth, pipe) - best[0]:+.4f})")
        # leave-one-well-out alpha (honest)
        pr = np.zeros(len(d))
        wv = d["well"].to_numpy()
        for w in k40:
            m = wv == w
            o_ = ~m
            a = float(np.sum(p[o_] * resid[o_]) / max(np.sum(p[o_] ** 2), 1e-9))
            pr[m] = pipe[m] + a * p[m]
        print(f"  LOWO-alpha blend       {rmse(truth, pr):8.4f}"
              f"  (gain {rmse(truth, pipe) - rmse(truth, pr):+.4f})")

    # structure of the pipeline's own residual: same oracle ladder
    print("\n--- oracle decomposition of the PIPELINE residual (k40) ---")
    from peraudit_oracles import ols_pred  # noqa: PLC0415

    X = np.load(CACHE / "X.npy", mmap_mode="r")
    fi = {f: i for i, f in enumerate(meta["features"])}
    dmd = np.asarray(X[pos, fi["dmd"]], np.float64)
    dz = np.asarray(X[pos, fi["dz"]], np.float64)
    wv = d["well"].to_numpy()
    acc = {}
    for w in k40:
        m = wv == w
        rr = resid[m]
        one = np.ones(m.sum())
        for nm, A in (
            ("raw", None),
            ("O1_const", one[:, None]),
            ("O2_lin_md", np.c_[dmd[m], one]),
            ("O4_lin_md_dz", np.c_[dmd[m], dz[m], one]),
        ):
            e = rr if A is None else rr - ols_pred(A, rr, A)
            acc.setdefault(nm, [0.0, 0.0])
            acc[nm][0] += float(np.sum(e**2))
            acc[nm][1] += len(e)
    for k, v in acc.items():
        print(f"  {k:16s} {np.sqrt(v[0] / v[1]):8.4f}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "scripts"))
    main()
