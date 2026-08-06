"""Well-level cross-validation helpers.

Splitting must never mix rows of the same well across folds (see
``CLAUDE.md`` / ``docs/design.md``), so folds are built over well ids rather
than rows. Scoring mirrors the competition metric: squared errors are
accumulated across all wells and pooled into a single RMSE
(``scripts/run_baseline_cv.py::pooled_rmse``), not averaged per well.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import data as D

PredictFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def well_folds(wells: list[str], n_splits: int = 5, seed: int = 42) -> list[list[str]]:
    """Deterministically split ``wells`` into ``n_splits`` non-overlapping folds.

    ``wells`` is sorted (for a canonical order independent of input order),
    shuffled with a seeded RNG, and then dealt round-robin into folds. The
    same ``(wells, n_splits, seed)`` always produces the same folds, and every
    well appears in exactly one fold.
    """
    ordered = sorted(wells)
    random.Random(seed).shuffle(ordered)

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for i, well in enumerate(ordered):
        folds[i % n_splits].append(well)
    return folds


def score_predictor(
    predict_fn: PredictFn,
    wells: list[str],
    split: str = "train",
    root: str | None = None,
) -> dict[str, float | int]:
    """Score ``predict_fn`` over ``wells`` using the competition's pooled RMSE.

    ``predict_fn(h, tw)`` must return an array of predictions aligned to
    ``D.eval_mask(h)`` (same order and length). Wells with an empty eval zone
    or without a known ``TVT_input`` anchor are skipped. Wells whose
    prediction raises an exception or contains non-finite values are counted
    as failed rather than aborting the whole run.
    """
    sq_err_sum = 0.0
    n_rows = 0
    well_rmses: list[float] = []
    n_skipped = 0
    n_failed = 0

    for well in wells:
        h = D.load_horizontal(well, split, root)
        tw = D.load_typewell(well, split, root)
        mask = D.eval_mask(h)

        if mask.sum() == 0:
            n_skipped += 1
            continue

        try:
            D.last_known_tvt(h)
        except ValueError:
            n_skipped += 1
            continue

        try:
            y_true = h["TVT"].to_numpy()[mask]
            pred = np.asarray(predict_fn(h, tw), dtype=float)
            if pred.shape != y_true.shape or not np.all(np.isfinite(pred)):
                raise ValueError("predict_fn returned an invalid prediction")
        except Exception:  # any predictor failure counts as a "failed" well
            n_failed += 1
            continue

        err = y_true - pred
        sse = float(np.sum(err**2))
        sq_err_sum += sse
        n_rows += err.size
        well_rmses.append(float(np.sqrt(sse / err.size)))

    pooled_rmse = float(np.sqrt(sq_err_sum / n_rows)) if n_rows else float("nan")

    if well_rmses:
        rmse_arr = np.array(well_rmses)
        median_well_rmse = float(np.median(rmse_arr))
        p90_well_rmse = float(np.percentile(rmse_arr, 90))
        max_well_rmse = float(np.max(rmse_arr))
    else:
        median_well_rmse = float("nan")
        p90_well_rmse = float("nan")
        max_well_rmse = float("nan")

    return {
        "pooled_rmse": pooled_rmse,
        "median_well_rmse": median_well_rmse,
        "p90_well_rmse": p90_well_rmse,
        "max_well_rmse": max_well_rmse,
        "n_rows": n_rows,
        "n_wells_scored": len(well_rmses),
        "n_wells_skipped": n_skipped,
        "n_wells_failed": n_failed,
    }
