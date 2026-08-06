"""Step 1: reproduce the oracle-anchor vs realistic-anchor gap, then ask why it exists.

Prior agent: IDW spatial map with an ORACLE per-well anchor scores ~5.49 ft pooled;
with a tail-mean anchor it degrades to ~9.41 ft.  Everything downstream is an attempt
to close that gap, so it has to reproduce first.

The diagnostic that decides whether the gap is even closeable: the residual
r(x) = S_well(x) - S_idw(x) along the *whole* well.  If r is a constant plus noise,
the anchor is a measurable scalar and a better estimator wins.  If r drifts
systematically from the known zone into the eval zone, no anchor estimator can win
and the ceiling is the drift, not the measurement.

Usage:
    uv run python scripts/levelling_repro.py [k40_dir]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE, build_field, donor_mask, load  # noqa: E402

K40 = Path(os.environ.get("ROGII_WORK", "work") + "/harness_audit/k40")
ANCHOR_TAIL = 300


def main(argv: list[str]) -> int:
    d = Path(argv[1]) if len(argv) > 1 else K40
    sub = pd.read_csv(d / "submission.csv")
    held = sorted({str(i).rsplit("_", 1)[0] for i in sub["id"]})
    print(f"held-out wells: {len(held)}")

    w = load()
    mask = donor_mask(w, exclude=set(held))
    fld = build_field(w, mask)
    print(f"donors: {len(fld)} points from {len(w.wells) - len(held)} wells")

    rows = []
    drift = []
    for wid in held:
        sl = w.slice(wid)
        kn = w.known[sl]
        X, Y, Z, T = w.X[sl], w.Y[sl], w.Z[sl], w.TVT[sl]
        S = T + Z
        xy = np.stack([X, Y], axis=1)
        _q = fld.query(xy)
        s_hat, nw = _q["s"], _q["nwells"]
        ok = np.isfinite(s_hat)
        ev = (~kn) & ok
        kz = kn & ok
        if ev.sum() < 50 or kz.sum() < 50:
            print(f"  skip {wid}: ev={ev.sum()} kn={kz.sum()}")
            continue
        r = S - s_hat

        c_oracle = float(np.median(r[ev]))
        tail_i = np.where(kz)[0][-ANCHOR_TAIL:]
        c_tail = float(np.mean(r[tail_i]))
        c_full = float(np.median(r[kz]))

        truth = T[~kn]
        base_ok = np.isfinite(s_hat[~kn])
        for name, c in (("oracle", c_oracle), ("tail", c_tail), ("full", c_full)):
            pred = (s_hat[~kn] + c) - Z[~kn]
            pred = np.where(base_ok, pred, np.nan)
            rows.append({"well": wid, "anchor": name, "n": int(base_ok.sum()),
                         "sse": float(np.nansum((pred - truth) ** 2 * base_ok)),
                         "c": c})
        drift.append({
            "well": wid, "n_ev": int(ev.sum()), "n_kn": int(kz.sum()),
            "c_oracle": c_oracle, "c_tail": c_tail, "c_full": c_full,
            "err_tail": c_tail - c_oracle, "err_full": c_full - c_oracle,
            "sd_r_known": float(np.std(r[kz])), "sd_r_eval": float(np.std(r[ev])),
            "nwells_med": float(np.median(nw[ok])),
            "cov": float(ok.mean()),
        })

    df = pd.DataFrame(rows)
    dr = pd.DataFrame(drift)
    print(f"\nscored wells: {dr.shape[0]}")
    print("\npooled RMSE by anchor:")
    for name in ("oracle", "tail", "full"):
        g = df[df["anchor"] == name]
        print(f"  {name:7s}  {np.sqrt(g['sse'].sum() / g['n'].sum()):8.3f} ft")

    print("\nanchor error |c_est - c_oracle| (ft):")
    for col in ("err_tail", "err_full"):
        a = dr[col].abs()
        print(f"  {col:9s} median {a.median():6.2f}  mean {a.mean():6.2f}  "
              f"p90 {a.quantile(0.9):6.2f}  max {a.max():6.2f}")

    print("\ndrift diagnostic: sd of r(x)=S_well-S_idw within a well")
    print(f"  known zone  median {dr['sd_r_known'].median():.2f} ft")
    print(f"  eval zone   median {dr['sd_r_eval'].median():.2f} ft")
    print(f"  median donor wells per query point: {dr['nwells_med'].median():.0f}")
    print(f"  median coverage: {dr['cov'].median()*100:.0f}%")

    dr.to_csv(CACHE / "repro_drift.csv", index=False)
    print(f"\nwrote {CACHE / 'repro_drift.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
