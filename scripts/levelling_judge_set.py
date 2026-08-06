"""Three-way frozen-recipe judgment (pair450 / local / comb) on one holdout set.

Runs the R118 rule for pair450 and the R128 comparison table for local and
comb = 0.5*(spat_pair450 + spat_local) on the rows files produced for one
holdout set (ROGII_SET_TAG), never refitting anything: W and KAPPA are the
same frozen constants as the submission kernel.

Per-set R118 rule (pair450 only):   ADOPT  if D <= -0.60 and >=24/40 improved
                                    and worst well <= +6.0 ft
                                    REJECT if D > -0.30 or any well > +10 ft
R128 comb promotion (pooled over all judged sets, computed by the caller):
D_comb <= D_pair450 - 0.05 and worst <= +6.0 and >=60% wells improved.

Usage:
    ROGII_SET_TAG=_b uv run python scripts/levelling_judge_set.py
    (reads rows_a_pair450_b.parquet + rows_a_local_b.parquet, writes
     judge3_perwell_b.csv next to them)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_apply_frozen import KAPPA, W  # noqa: E402
from levelling_common import CACHE  # noqa: E402
from levelling_predict import SET_TAG  # noqa: E402


def frozen(pipe: np.ndarray, spat: np.ndarray) -> tuple[np.ndarray, float]:
    s = np.where(np.isfinite(spat), spat, pipe)          # fallback rows included
    mad = float(np.median(np.abs(s - pipe)))
    return pipe + W * np.clip(s - pipe, -KAPPA * mad, KAPPA * mad), mad


def per_well_delta(wells: np.ndarray, fin: np.ndarray, pipe: np.ndarray,
                   truth: np.ndarray) -> pd.Series:
    out = {}
    for wl in np.unique(wells):
        m = wells == wl
        out[wl] = (float(np.sqrt(np.mean((fin[m] - truth[m]) ** 2)))
                   - float(np.sqrt(np.mean((pipe[m] - truth[m]) ** 2))))
    return pd.Series(out)


def r118_verdict(delta: float, improved: int, n: int, worst: float) -> str:
    if delta > -0.30 or worst > 10.0:
        return "REJECT"
    if delta <= -0.60 and improved >= int(0.6 * n) and worst <= 6.0:
        return "ADOPT"
    return "MIDDLE"


def main() -> int:
    a = pd.read_parquet(CACHE / f"rows_a_pair450{SET_TAG}.parquet")
    b = pd.read_parquet(CACHE / f"rows_a_local{SET_TAG}.parquet")
    if not (a["well"].values == b["well"].values).all():
        raise SystemExit("row alignment mismatch between pair450 and local")
    pipe, truth = a["pipe"].to_numpy(), a["truth"].to_numpy()
    wells = a["well"].to_numpy()
    n = len(np.unique(wells))
    base = float(np.sqrt(np.mean((pipe - truth) ** 2)))
    print(f"set {SET_TAG or '(k40)'}: wells={n} rows={len(a)} base={base:.4f}")

    sa, sb = a["sp_tail"].to_numpy(), b["sp_tail"].to_numpy()
    comb = np.where(np.isfinite(sa) & np.isfinite(sb), 0.5 * (sa + sb),
                    np.where(np.isfinite(sa), sa, sb))
    table = {}
    for name, spat in (("pair450", sa), ("local", sb), ("comb", comb)):
        fin, mad = frozen(pipe, spat)
        dw = per_well_delta(wells, fin, pipe, truth)
        delta = float(np.sqrt(np.mean((fin - truth) ** 2))) - base
        table[name] = dw
        v = r118_verdict(delta, int((dw < 0).sum()), n, float(dw.max())) \
            if name == "pair450" else ""
        print(f"  {name:8s} MAD {mad:6.3f}  D {delta:+.4f}  impr {int((dw < 0).sum())}/{n}"
              f"  worst {dw.max():+.2f}  best {dw.min():+.2f}  {v}")

    out = CACHE / f"judge3_perwell{SET_TAG or '_k40'}.csv"
    pd.DataFrame(table).rename_axis("well").to_csv(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
