"""Calibrate the gate on train wells that are NOT among the 40 evaluation wells.

The blend weight must not be chosen on the wells it is judged on.  The pipeline's
held-out predictions only exist for the 40, so the blend weight itself has to be
cross-fitted inside them -- but the expensive half of the gate, "given the holdback
error, how badly will the spatial leg do in the eval zone", needs no pipeline at all.
It can be fitted on hundreds of ordinary train wells, and that is what this does.

Donor hygiene: the field excludes all 40 evaluation wells, so none of their labels
touch the calibration; the calibration well itself is excluded at query time.

Usage:
    uv run python scripts/levelling_calib.py [n_wells]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE, Field, donor_mask, load  # noqa: E402
from levelling_predict import (  # noqa: E402
    ANCHOR_TAIL,
    HOLDBACK_FT,
    MIN_FIT,
    load_pipeline,
    robust_c,
)


def well_record(w, fld: Field, widx: dict[str, int], wid: str) -> dict | None:
    sl = w.slice(wid)
    kn = w.known[sl]
    X, Y, Z, TV, MD = w.X[sl], w.Y[sl], w.Z[sl], w.TVT[sl], w.MD[sl]
    S = TV + Z
    ev = ~kn
    if ev.sum() < 50 or kn.sum() < MIN_FIT:
        return None
    q = fld.query(np.stack([X, Y], axis=1), exclude_well=widx[wid])
    sh, nwe, dnn = q["s"], q["nwells"], q["dnn"]
    ok = np.isfinite(sh)
    kz, ev_ok = kn & ok, ev & ok
    if kz.sum() < MIN_FIT or ev_ok.sum() < 50:
        return None
    r = S - sh

    ki = np.where(kz)[0]
    c_tail = robust_c(r[ki[-ANCHOR_TAIL:]])
    pred = (sh[ev] + c_tail) - Z[ev]
    good = np.isfinite(pred)
    sp_rmse = float(np.sqrt(np.mean((pred[good] - TV[ev][good]) ** 2)))

    cut = MD[kn].max() - HOLDBACK_FT
    fit_m, hb_m = kz & (MD <= cut), kz & (MD > cut)
    if fit_m.sum() < MIN_FIT or hb_m.sum() < 20:
        return None
    c_hb = robust_c(r[fit_m])
    hb_rmse = float(np.sqrt(np.mean(((sh[hb_m] + c_hb) - Z[hb_m] - TV[hb_m]) ** 2)))

    return {
        "well": wid, "n": int(ev.sum()), "sp_rmse": sp_rmse, "hb_rmse": hb_rmse,
        "hb_n": int(hb_m.sum()), "cov": float(ok[ev].mean()),
        "dnn_ev": float(np.median(dnn[ev_ok])), "dnn_kn": float(np.median(dnn[kz])),
        "nwells_ev": float(np.median(nwe[ev_ok])), "sd_r_kn": float(np.std(r[kz])),
        "n_kn": int(kn.sum()),
    }


def main(argv: list[str]) -> int:
    limit = int(argv[1]) if len(argv) > 1 else 400
    _, held = load_pipeline()
    heldset = set(held)
    w = load()
    widx = {name: i for i, name in enumerate(w.wells)}
    mask = donor_mask(w, exclude=heldset)
    well_of = np.repeat(np.arange(len(w.wells)), np.diff(w.starts))
    fld = Field(np.stack([w.X[mask], w.Y[mask]], axis=1),
                (w.TVT + w.Z)[mask], well_of[mask])
    print(f"field {len(fld)} points; calibrating on up to {limit} non-evaluation wells")

    pool = [x for x in w.wells if x not in heldset]
    recs = []
    for i, wid in enumerate(pool[:limit]):
        rec = well_record(w, fld, widx, wid)
        if rec:
            recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{min(limit, len(pool))} -> {len(recs)} usable")
    df = pd.DataFrame(recs)
    df.to_csv(CACHE / "calib.csv", index=False)
    print(f"\ncalibration wells: {len(df)}")
    print(f"  spatial eval RMSE  median {df['sp_rmse'].median():.2f}  "
          f"p90 {df['sp_rmse'].quantile(0.9):.2f}  max {df['sp_rmse'].max():.1f}")
    print(f"  holdback RMSE      median {df['hb_rmse'].median():.2f}  "
          f"p90 {df['hb_rmse'].quantile(0.9):.2f}  max {df['hb_rmse'].max():.1f}")
    for c in ("hb_rmse", "dnn_ev", "nwells_ev", "sd_r_kn", "cov", "n_kn"):
        lo = np.log1p(np.abs(df[c])) if c != "cov" else df[c]
        print(f"  spearman(log sp_rmse, {c:9s}) = "
              f"{pd.Series(np.log(df['sp_rmse'])).corr(pd.Series(lo), method='spearman'):+.3f}")
    print(f"wrote {CACHE / 'calib.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
