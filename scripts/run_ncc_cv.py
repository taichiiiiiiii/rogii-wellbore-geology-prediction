"""Evaluate the multi-scale NCC registration tracker on all train wells.

Reports, for (a) ``carry_last``, (b) raw ``track_ncc`` output, and (c) blends
``anchor + w * (ncc - anchor)`` for ``w`` in ``{0.3, 0.5, 0.7, 1.0}``:

* pooled RMSE (the competition metric: squared error summed across every
  eval-zone row of every well, then a single sqrt at the end -- see
  ``scripts/run_baseline_cv.py``),
* the per-well RMSE distribution (median / p90 / max), and
* ``track_ncc()`` wall-clock time per well (mean / p95), relevant for the
  Kaggle notebook's <=9h CPU budget.

NCC is unsupervised (no fitting), so there is no train/valid split here --
every train well with an eval zone and a known anchor is scored directly.

    uv run python scripts/run_ncc_cv.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import baseline  # noqa: E402
from rogii import data as D  # noqa: E402
from rogii.registration.ncc import track_ncc  # noqa: E402

BLEND_WEIGHTS: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)


def pooled_rmse(sq_err_sum: float, n: int) -> float:
    return float(np.sqrt(sq_err_sum / n)) if n else float("nan")


def well_rmse(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def _predictor_names() -> list[str]:
    return ["carry_last", "ncc_raw", *[f"blend_w{w:.1f}" for w in BLEND_WEIGHTS]]


def _predictions_for_well(anchor_pred: np.ndarray, ncc_tvt: np.ndarray) -> dict[str, np.ndarray]:
    preds = {"carry_last": anchor_pred, "ncc_raw": ncc_tvt}
    for w in BLEND_WEIGHTS:
        preds[f"blend_w{w:.1f}"] = anchor_pred + w * (ncc_tvt - anchor_pred)
    return preds


def _print_pooled_table(names: list[str], sse: dict[str, float], n_rows: dict[str, int]) -> None:
    print("## Pooled RMSE (all eval-zone rows, all wells) -- the competition metric\n")
    print(f"{'predictor':16s} {'pooled RMSE (ft)':>18s} {'rows':>12s}")
    print("-" * 48)
    for name in names:
        print(f"{name:16s} {pooled_rmse(sse[name], n_rows[name]):18.4f} {n_rows[name]:12,d}")


def _print_per_well_table(names: list[str], per_well: dict[str, list[float]]) -> None:
    print("\n## Per-well RMSE distribution (ft)\n")
    print(f"{'predictor':16s} {'median':>10s} {'p90':>10s} {'max':>10s} {'n_wells':>10s}")
    print("-" * 60)
    for name in names:
        arr = np.array(per_well[name])
        print(
            f"{name:16s} {np.median(arr):10.3f} {np.percentile(arr, 90):10.3f} "
            f"{arr.max():10.3f} {arr.size:10,d}"
        )


def _print_timing(timings: np.ndarray) -> None:
    print("\n## track_ncc() wall-clock time per well (s)\n")
    print(
        f"mean={timings.mean():.4f}  p50={np.median(timings):.4f}  "
        f"p95={np.percentile(timings, 95):.4f}  max={timings.max():.4f}  n={timings.size}"
    )
    budget_s = 9 * 3600
    print(
        f"(9h Kaggle budget / mean per-well time => ~{budget_s / timings.mean():,.0f} wells "
        f"processable if track_ncc were the only cost)"
    )


def main() -> None:
    wells = D.list_wells("train")
    print(f"# train wells: {len(wells)}")

    names = _predictor_names()
    sse = {name: 0.0 for name in names}
    n_rows = {name: 0 for name in names}
    per_well: dict[str, list[float]] = {name: [] for name in names}
    timings: list[float] = []
    skipped = 0

    for well in wells:
        h = D.load_horizontal(well, "train")
        tw = D.load_typewell(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            skipped += 1
            continue

        y_true = h["TVT"].to_numpy()[ez]
        anchor_pred = baseline.predict_carry_last(h)

        t0 = time.perf_counter()
        result = track_ncc(h, tw)
        timings.append(time.perf_counter() - t0)

        for name, pred in _predictions_for_well(anchor_pred, result.tvt).items():
            err = y_true - pred
            sse[name] += float(np.sum(err**2))
            n_rows[name] += err.size
            per_well[name].append(well_rmse(y_true, pred))

    print(f"# skipped wells (no eval zone / no anchor): {skipped}\n")
    _print_pooled_table(names, sse, n_rows)
    _print_per_well_table(names, per_well)
    _print_timing(np.array(timings))


if __name__ == "__main__":
    main()
