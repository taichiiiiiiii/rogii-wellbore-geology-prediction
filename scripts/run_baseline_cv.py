"""Evaluate baseline predictors on all train wells' eval zones (LB proxy).

The competition metric is RMSE over all eval-zone rows pooled together, so we
accumulate squared errors across wells rather than averaging per-well RMSEs.

    uv run python scripts/run_baseline_cv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import baseline  # noqa: E402
from rogii import data as D

PREDICTORS = {
    "carry_last": baseline.predict_carry_last,
}


def pooled_rmse(sq_err_sum: float, n: int) -> float:
    return float(np.sqrt(sq_err_sum / n)) if n else float("nan")


def main() -> None:
    wells = D.list_wells("train")
    print(f"# train wells: {len(wells)}")

    acc = {name: [0.0, 0] for name in PREDICTORS}  # name -> [sq_err_sum, n]
    skipped = 0
    for w in wells:
        h = D.load_horizontal(w, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            skipped += 1
            continue
        y_true = h["TVT"].to_numpy()[ez]
        for name, fn in PREDICTORS.items():
            pred = fn(h)
            err = y_true - pred
            acc[name][0] += float(np.sum(err**2))
            acc[name][1] += err.size

    print(f"# skipped wells (no eval zone / no anchor): {skipped}\n")
    print(f"{'predictor':16s} {'pooled RMSE (ft)':>18s} {'rows':>12s}")
    print("-" * 48)
    for name, (sse, n) in acc.items():
        print(f"{name:16s} {pooled_rmse(sse, n):18.4f} {n:12,d}")


if __name__ == "__main__":
    main()
