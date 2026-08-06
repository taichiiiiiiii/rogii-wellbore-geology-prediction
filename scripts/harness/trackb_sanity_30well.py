"""Track B plumbing sanity check: 30-well local subsample, tiny LightGBM.

Not a claim of Track B's real strength (30 wells / 50-tree GBDT is a
smoke test, not a measurement) -- it exists only to prove the plumbing works
end to end before any of this is trusted on Kaggle's full 773-well train set:

1. ``rogii.trackb_features.build_trackb_features`` builds a NaN/inf-clean
   feature matrix for a real subsample of train wells.
2. Well-level GroupKFold (``rogii.cv.well_folds``, 5 folds) trains/predicts
   without ever mixing a well's rows across train/valid.
3. The resulting OOF pooled RMSE of ``carry_last + predicted_drift`` beats
   plain ``carry_last`` on this subsample (sanity, not a leaderboard claim).

Runs streamingly (one well loaded/featurized/dropped at a time, float32
throughout) and reports peak RSS via ``resource.getrusage`` -- see module
docstring of ``notebooks/submission/kernel_nn_dataset/nn_dataset_builder.py``
for why this repo's local box (3.8GB RAM) needs this discipline even for a
"small" 30-well run.

Usage::

    uv run python scripts/harness/trackb_sanity_30well.py
    uv run python scripts/harness/trackb_sanity_30well.py --n-wells 50 --seed 7
"""

from __future__ import annotations

import argparse
import gc
import resource
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from rogii import cv as CV
from rogii import data as D
from rogii.trackb_features import FEATURE_COLUMNS, build_trackb_features, carry_last

SEED = 42
N_SPLITS = 5
N_ESTIMATORS = 50  # deliberately tiny -- plumbing check, not a strength claim

LGB_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def peak_rss_mb() -> float:
    """Peak resident set size in MB (Linux: ru_maxrss is already in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_matrix(
    wells: list[str], split: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stream-build features/target/carry_last/well_idx for ``wells``.

    Returns (X, y_drift, y_tvt_true, carry_last_per_row, well_id_per_row).
    One well's ``h``/``tw`` are read, featurized, and dropped before the
    next is loaded -- never holds more than one well's raw CSVs in memory
    at once.
    """
    x_parts: list[pd.DataFrame] = []
    y_drift_parts: list[np.ndarray] = []
    y_tvt_parts: list[np.ndarray] = []
    carry_parts: list[np.ndarray] = []
    well_id_parts: list[np.ndarray] = []
    n_skipped = 0

    for well in wells:
        h = D.load_horizontal(well, split)
        tw = D.load_typewell(well, split)

        mask = D.eval_mask(h)
        if mask.sum() == 0:
            n_skipped += 1
            del h, tw
            continue
        try:
            anchor = carry_last(h)
        except ValueError:
            n_skipped += 1
            del h, tw
            continue

        feats = build_trackb_features(h, tw)
        y_true = h["TVT"].to_numpy(dtype=np.float32)[mask]
        y_drift = (y_true - anchor).astype(np.float32)

        x_parts.append(feats)
        y_drift_parts.append(y_drift)
        y_tvt_parts.append(y_true)
        carry_parts.append(np.full(len(feats), anchor, dtype=np.float32))
        well_id_parts.append(np.full(len(feats), well))

        del h, tw, feats, y_true, y_drift
        gc.collect()

    x = pd.concat(x_parts, ignore_index=True) if x_parts else pd.DataFrame(columns=FEATURE_COLUMNS)
    y_drift_arr = np.concatenate(y_drift_parts) if y_drift_parts else np.array([], dtype=np.float32)
    y_tvt_arr = np.concatenate(y_tvt_parts) if y_tvt_parts else np.array([], dtype=np.float32)
    carry_arr = np.concatenate(carry_parts) if carry_parts else np.array([], dtype=np.float32)
    well_id_arr = np.concatenate(well_id_parts) if well_id_parts else np.array([], dtype=object)

    print(f"[TRACKB-SANITY] built matrix: {len(wells) - n_skipped}/{len(wells)} wells usable "
          f"({n_skipped} skipped: empty eval zone or no anchor), n_rows={len(x)}, "
          f"n_features={len(FEATURE_COLUMNS)}, peak_rss_mb={peak_rss_mb():.1f}")
    return x, y_drift_arr, y_tvt_arr, carry_arr, list(well_id_arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wells", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.perf_counter()
    all_wells = D.list_wells("train")
    ordered = sorted(all_wells)
    import random

    random.Random(args.seed).shuffle(ordered)
    wells = ordered[: args.n_wells]
    print(f"[TRACKB-SANITY] sampled {len(wells)}/{len(all_wells)} train wells (seed={args.seed})")

    x, y_drift, y_tvt, carry, well_id = build_matrix(wells, "train")
    usable_wells = sorted(set(well_id))
    print(f"[TRACKB-SANITY] usable_wells={len(usable_wells)} total_rows={len(x)} "
          f"peak_rss_mb={peak_rss_mb():.1f}")

    folds = CV.well_folds(usable_wells, n_splits=N_SPLITS, seed=args.seed)
    well_to_fold = {w: i for i, fold_wells in enumerate(folds) for w in fold_wells}
    fold_id = np.array([well_to_fold[w] for w in well_id])

    x_arr = x.to_numpy(dtype=np.float32)
    oof_drift_pred = np.full(len(x), np.nan, dtype=np.float64)

    for fold in range(N_SPLITS):
        valid_mask = fold_id == fold
        train_mask = ~valid_mask
        if valid_mask.sum() == 0 or train_mask.sum() == 0:
            print(f"[TRACKB-SANITY] fold={fold} skipped (empty train or valid)")
            continue

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(x_arr[train_mask], y_drift[train_mask])
        pred = model.predict(x_arr[valid_mask])
        oof_drift_pred[valid_mask] = pred

        fold_tvt_pred = carry[valid_mask] + pred
        fold_rmse_model = pooled_rmse(y_tvt[valid_mask], fold_tvt_pred)
        fold_rmse_carry = pooled_rmse(y_tvt[valid_mask], carry[valid_mask])
        n_train_wells = len({w for w, f in well_to_fold.items() if f != fold})
        n_valid_wells = len({w for w, f in well_to_fold.items() if f == fold})
        print(f"[TRACKB-SANITY] fold={fold} n_train_wells={n_train_wells} "
              f"n_valid_wells={n_valid_wells} n_valid_rows={int(valid_mask.sum())} "
              f"rmse_carry_last={fold_rmse_carry:.4f} rmse_trackb={fold_rmse_model:.4f} "
              f"delta={fold_rmse_model - fold_rmse_carry:+.4f}")
        del model
        gc.collect()

    scored = ~np.isnan(oof_drift_pred)
    tvt_pred_oof = carry[scored] + oof_drift_pred[scored]
    pooled_carry = pooled_rmse(y_tvt[scored], carry[scored])
    pooled_trackb = pooled_rmse(y_tvt[scored], tvt_pred_oof)

    elapsed = time.perf_counter() - t0
    print(f"\n[TRACKB-SANITY] === POOLED (n_wells={len(usable_wells)}, "
          f"n_rows={int(scored.sum())}, n_features={len(FEATURE_COLUMNS)}) ===")
    print(f"[TRACKB-SANITY] pooled_rmse_carry_last = {pooled_carry:.4f}")
    print(f"[TRACKB-SANITY] pooled_rmse_trackb_oof  = {pooled_trackb:.4f}")
    print(f"[TRACKB-SANITY] delta (trackb - carry)  = {pooled_trackb - pooled_carry:+.4f} "
          f"({'BEATS' if pooled_trackb < pooled_carry else 'DOES NOT BEAT'} carry_last)")
    print(f"[TRACKB-SANITY] elapsed_s={elapsed:.1f} peak_rss_mb={peak_rss_mb():.1f}")


if __name__ == "__main__":
    main()
