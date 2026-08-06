"""P1: residual GBDT over the anchor baseline (issue P1).

Trains a LightGBM regressor on the residual target ``U = TVT - anchor``
(``anchor = data.last_known_tvt(h)``) using well-level ``GroupKFold`` (5
folds, seed=42, ``rogii.cv.well_folds``) and reports the pooled OOF RMSE
against the ``carry_last`` baseline (15.9099 ft, all 773 train wells).

Leak boundary: features come from ``rogii.features.build_features``, which
never reads ``h["TVT"]``. The ``TVT`` column is read exactly once here, to
build the training label -- never inside the feature builder.

Usage::

    uv run python scripts/run_p1_gbdt_cv.py
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from rogii import cv as CV
from rogii import data as D
from rogii.features import build_features

CARRY_LAST_POOLED_RMSE = 15.9099
N_SPLITS = 5
SEED = 42

LGB_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def build_well_matrix(
    wells: list[str], split: str = "train"
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build the feature matrix + targets for all ``wells``.

    Returns ``(X, y_u, y_true_tvt, well_idx, used_wells)``:
      - ``X``: float32 feature DataFrame, one row per eval-zone row.
      - ``y_u``: residual target ``TVT - anchor`` (float32), aligned to ``X``.
      - ``y_true_tvt``: ground-truth TVT (float32), aligned to ``X``.
      - ``well_idx``: int32 index into ``used_wells`` per row (memory-cheap
        stand-in for a well-id string column).
      - ``used_wells``: wells that had a non-empty eval zone and a valid anchor
        (wells without either are skipped, matching ``cv.score_predictor``).
    """
    feat_parts: list[pd.DataFrame] = []
    y_u_parts: list[np.ndarray] = []
    y_true_parts: list[np.ndarray] = []
    well_idx_parts: list[np.ndarray] = []
    used_wells: list[str] = []

    for well in wells:
        h = D.load_horizontal(well, split)
        tw = D.load_typewell(well, split)
        mask = D.eval_mask(h)
        if mask.sum() == 0:
            continue
        try:
            anchor = D.last_known_tvt(h)
        except ValueError:
            continue

        feats = build_features(h, tw)
        y_true = h["TVT"].to_numpy(dtype=np.float32)[mask]  # label only, never a feature

        well_i = len(used_wells)
        used_wells.append(well)

        feat_parts.append(feats)
        y_u_parts.append((y_true - anchor).astype(np.float32))
        y_true_parts.append(y_true)
        well_idx_parts.append(np.full(y_true.shape[0], well_i, dtype=np.int32))

    X = pd.concat(feat_parts, ignore_index=True)
    y_u = np.concatenate(y_u_parts)
    y_true = np.concatenate(y_true_parts)
    well_idx = np.concatenate(well_idx_parts)
    return X, y_u, y_true, well_idx, used_wells


def main() -> None:
    t_start = time.time()

    wells = D.list_wells("train")
    print(f"train wells: {len(wells)}")

    t0 = time.time()
    X, y_u, y_true, well_idx, used_wells = build_well_matrix(wells, split="train")
    print(
        f"feature matrix: {X.shape}, rows={len(y_true):,}, "
        f"wells_used={len(used_wells)}/{len(wells)} "
        f"({time.time() - t0:.1f}s)"
    )
    feature_cols = list(X.columns)

    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    row_fold = np.array([well_to_fold[w] for w in used_wells], dtype=np.int32)[well_idx]

    oof_pred_tvt = np.full(len(y_true), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importances = np.zeros(len(feature_cols), dtype=np.float64)

    for fold_i in range(N_SPLITS):
        t_fold = time.time()
        train_mask = row_fold != fold_i
        valid_mask = row_fold == fold_i
        n_valid_wells = len(folds[fold_i])

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X.loc[train_mask],
            y_u[train_mask],
            eval_set=[(X.loc[valid_mask], y_u[valid_mask])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        pred_u = model.predict(X.loc[valid_mask])
        anchor_valid = y_true[valid_mask] - y_u[valid_mask]
        pred_tvt = anchor_valid + pred_u
        oof_pred_tvt[valid_mask] = pred_tvt

        fold_rmse = pooled_rmse(y_true[valid_mask], pred_tvt)
        importances += model.feature_importances_.astype(np.float64) / N_SPLITS

        fold_rows.append(
            {
                "fold": fold_i,
                "n_wells": n_valid_wells,
                "n_rows": int(valid_mask.sum()),
                "best_iteration": model.best_iteration_,
                "rmse": fold_rmse,
                "seconds": time.time() - t_fold,
            }
        )
        print(
            f"fold {fold_i}: n_wells={n_valid_wells} n_rows={int(valid_mask.sum()):,} "
            f"best_iter={model.best_iteration_} rmse={fold_rmse:.4f} "
            f"({time.time() - t_fold:.1f}s)"
        )

    assert not np.isnan(oof_pred_tvt).any(), "every row must be scored by exactly one fold"

    oof_pooled_rmse = pooled_rmse(y_true, oof_pred_tvt)

    well_rmses = []
    for w_i in range(len(used_wells)):
        rows = well_idx == w_i
        well_rmses.append(pooled_rmse(y_true[rows], oof_pred_tvt[rows]))
    well_rmses_arr = np.array(well_rmses)

    print()
    print("=" * 70)
    print(f"OOF pooled RMSE (GBDT, P1):     {oof_pooled_rmse:.4f}")
    print(f"carry_last pooled RMSE (P0):    {CARRY_LAST_POOLED_RMSE:.4f}")
    print(f"delta (negative = improved):    {oof_pooled_rmse - CARRY_LAST_POOLED_RMSE:+.4f}")
    print(f"total rows scored:              {len(y_true):,}")
    print("=" * 70)
    print()

    fold_df = pd.DataFrame(fold_rows)
    print("per-fold RMSE:")
    print(fold_df.to_string(index=False))
    print()

    print("per-well RMSE distribution (OOF):")
    print(f"  median: {np.median(well_rmses_arr):.4f}")
    print(f"  p90:    {np.percentile(well_rmses_arr, 90):.4f}")
    print(f"  max:    {np.max(well_rmses_arr):.4f}")
    print()

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_importance": importances})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    print("feature importance (top 20, mean split-count over folds):")
    print(importance_df.head(20).to_string(index=False))
    print()

    print(f"total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
