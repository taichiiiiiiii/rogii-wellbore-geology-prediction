"""Head-to-head: tops-F vs the pair450 (mis-tie network) champion (task step 4e).

Compares the FROZEN-recipe finals of the two donor-field constructions on
the exact same 40 held-out wells / base predictions:
  final_pair450 = frozen(rows_a_pair450.parquet, anchor=tail)
  final_tops    = frozen(rows_tops_none.parquet,  anchor=tail)

Reports the per-well paired delta, the pooled difference, the correlation of
(spat-truth) between the two constructions restricted to rows where BOTH
differ from base (only rows where either leg actually did something), and
the anchor diagnostic: pooled RMSE of the honest holdback gate (hb_rmse,
"S_known_tail - F/S_idw - offset" scored on a withheld stretch of the KNOWN
zone -- the closest quantity the existing scripts emit to the "7.04 -> 3.55"
number quoted in the task; that exact figure is from a different, uncached
run and could not be reproduced bit-for-bit here, see the run report).

Usage:
    uv run python scripts/levelling_headtohead.py [anchor]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_apply_frozen import apply_frozen  # noqa: E402
from levelling_common import CACHE  # noqa: E402


def load_final(tag: str, anchor: str) -> pd.DataFrame:
    df = pd.read_parquet(CACHE / f"rows_{tag}.parquet")
    pipe, truth = df["pipe"].to_numpy(float), df["truth"].to_numpy(float)
    spat = df[f"sp_{anchor}"].to_numpy(float)
    final, mad, cap = apply_frozen(pipe, spat, truth)
    return df.assign(spat=spat, final=final), mad, cap


def per_well_rmse(df: pd.DataFrame, col: str) -> pd.Series:
    e = (df[col] - df["truth"]) ** 2
    return df.assign(e=e).groupby("well")["e"].mean().pow(0.5)


def main(argv: list[str]) -> int:
    anchor = argv[1] if len(argv) > 1 else "tail"
    ref, mad_r, cap_r = load_final("a_pair450", anchor)
    tops, mad_t, cap_t = load_final("tops_none", anchor)

    per = pd.DataFrame({
        "rmse_pair450": per_well_rmse(ref, "final"),
        "rmse_tops": per_well_rmse(tops, "final"),
    })
    per["delta_tops_minus_pair450"] = per["rmse_tops"] - per["rmse_pair450"]
    per = per.sort_values("delta_tops_minus_pair450")

    pooled_pair450 = np.sqrt(np.mean((ref["final"] - ref["truth"]) ** 2))
    pooled_tops = np.sqrt(np.mean((tops["final"] - tops["truth"]) ** 2))
    print(f"=== head-to-head, anchor={anchor} ===")
    print(f"  pooled final RMSE  pair450={pooled_pair450:.4f}  tops={pooled_tops:.4f}  "
          f"diff={pooled_tops - pooled_pair450:+.4f}")
    print(f"  wells where tops beats pair450: {int((per['delta_tops_minus_pair450'] < 0).sum())}/40")
    print(f"\n{per.round(3).to_string()}")

    # error decorrelation: rows where BOTH legs' spat actually differ from base
    a = ref[["well", "pipe", "truth", "spat"]].reset_index(drop=True)
    b = tops[["spat"]].reset_index(drop=True)
    both_moved = (np.abs(a["spat"] - a["pipe"]) > 1e-9) & (np.abs(b["spat"] - a["pipe"]) > 1e-9)
    err_pair450 = (a["spat"] - a["truth"])[both_moved]
    err_tops = (b["spat"] - a["truth"])[both_moved]
    corr = np.corrcoef(err_pair450, err_tops)[0, 1] if both_moved.sum() > 2 else float("nan")
    print(f"\n  rows where BOTH legs' spat != base: {int(both_moved.sum())}")
    print(f"  corr[(spat_pair450-truth), (spat_tops-truth)] on those rows: {corr:+.4f}")

    # anchor diagnostic: honest holdback gate, pooled and median, both constructions
    print("\n  honest-holdback (hb_rmse) known-zone anchor diagnostic:")
    for tag in ("a_none", "a_pair450", "tops_none", "tops_pair450"):
        ft = pd.read_csv(CACHE / f"feats_{tag}.csv")
        sse = (ft["hb_rmse"] ** 2 * ft["hb_n"]).sum()
        n = ft["hb_n"].sum()
        print(f"    {tag:14s} pooled {np.sqrt(sse / n):8.3f}  median {ft['hb_rmse'].median():8.3f}")

    out = CACHE / f"headtohead_perwell_{anchor}.csv"
    per.to_csv(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
