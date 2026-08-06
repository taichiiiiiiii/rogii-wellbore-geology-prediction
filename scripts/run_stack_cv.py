"""Stack-integration GBDT: tracker (PF/beam) + within-well (P1) features (issue: stack GBDT).

Two target formulations are measured side by side, both trained on the same
feature matrix (P1's 36 within-well features + 8 tracker features from
``rogii.stack_features``), well-level ``GroupKFold(5, seed=42)`` OOF, LightGBM:

  (a) U-direct:   target = TVT - anchor,               pred = anchor   + pred_target
  (b) PF-residual: target = TVT - pf_blend_w0.7,        pred = pf_blend + pred_target

(b) is a safe-by-construction design: if the model learns nothing useful,
``pred_target -> 0`` and the prediction collapses to the current-best PF
blend (11.0621) rather than regressing below it.

Uses the pre-computed tracker cache (``data/processed/tracker_cache/*.npz``,
built by ``scripts/build_tracker_cache.py`` -- NOT re-run here) so PF/beam
never have to be recomputed; this script only does feature assembly + GBDT
training/scoring.

Leak boundary: within-well features come from ``rogii.features.build_features``
(never reads ``h["TVT"]``); tracker features come from
``rogii.stack_features.build_tracker_features`` (pure arithmetic over
already-leak-safe cached arrays + the anchor). ``TVT`` is read exactly once
per well, here, to build the two regression targets -- never inside a feature
builder.

After the two GBDT runs, a post-hoc stacking section evaluates cheap
combination variants computed from the OOF predictions (shrinking (b)'s
residual correction, blending (a) with (b), pf_std-tercile-wise shrink).
Every post-hoc scalar is fit **nested**: the scalar applied to fold ``i``'s
rows is least-squares-fit on the *other* folds' OOF rows only, so the
reported pooled RMSE stays honest (no scalar is ever fit on rows it scores).

Usage::

    uv run python scripts/run_stack_cv.py                 # full 773-well experiment
    uv run python scripts/run_stack_cv.py --n-wells 80    # stratified health-check subset
    uv run python scripts/run_stack_cv.py --save-oof      # also dump OOF arrays to outputs/
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rogii import cv as CV
from rogii import data as D
from rogii.features import build_features
from rogii.stack_features import build_tracker_features

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "tracker_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.csv"

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
PF_BLEND_W = 0.7  # matches build_tracker_cache.py / experiment ledger's "PF blend w0.7"

N_SPLITS = 5
SEED = 42
N_STRATA = 10  # deciles of manifest pf_std_mean, for the --n-wells health-check sample

LGB_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _per_well_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, well_idx: np.ndarray, n_wells: int
) -> np.ndarray:
    out = np.full(n_wells, np.nan, dtype=np.float64)
    for w_i in range(n_wells):
        rows = well_idx == w_i
        if rows.any():
            out[w_i] = pooled_rmse(y_true[rows], y_pred[rows])
    return out


def _well_npz_path(well: str) -> Path:
    return CACHE_DIR / f"{well}.npz"


def _stratified_sample(all_wells: list[str], n: int, seed: int) -> list[str]:
    """Deterministic sample of ``n`` wells, stratified by manifest ``pf_std_mean`` decile."""
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["well"].isin(all_wells)].copy()
    manifest["decile"] = pd.qcut(manifest["pf_std_mean"], N_STRATA, labels=False, duplicates="drop")

    rng = random.Random(seed)
    per_stratum = max(1, n // manifest["decile"].nunique())
    picked: list[str] = []
    for _, group in manifest.groupby("decile"):
        wells_in_group = sorted(group["well"].tolist())
        rng.shuffle(wells_in_group)
        picked.extend(wells_in_group[:per_stratum])

    rng.shuffle(picked)
    return sorted(picked[:n]) if len(picked) >= n else sorted(picked)


_BuildMatrixResult = tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int, int
]


def build_well_matrix(wells: list[str], split: str = "train") -> _BuildMatrixResult:
    """Build the combined (P1 + tracker) feature matrix and both targets' ingredients.

    Returns ``(X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells,
    n_cache_missing, n_len_mismatch)``, all row-aligned across wells (eval-zone order
    within each well, matching ``data.eval_mask``).
    """
    feat_parts: list[pd.DataFrame] = []
    y_true_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    pf_blend_parts: list[np.ndarray] = []
    pf_std_parts: list[np.ndarray] = []
    well_idx_parts: list[np.ndarray] = []
    used_wells: list[str] = []
    n_cache_missing = 0
    n_len_mismatch = 0

    for well in wells:
        h = D.load_horizontal(well, split)
        mask = D.eval_mask(h)
        if mask.sum() == 0:
            continue
        try:
            D.last_known_tvt(h)
        except ValueError:
            continue

        npz_path = _well_npz_path(well)
        if not npz_path.exists():
            n_cache_missing += 1
            continue

        with np.load(npz_path) as npz:
            pf_tvt = npz["pf_tvt"].astype(np.float64)
            pf_std = npz["pf_std"].astype(np.float64)
            beam_tvt = npz["beam_tvt"].astype(np.float64)
            beam_margin = npz["beam_margin"].astype(np.float64)
            cached_anchor = float(npz["anchor"])

        if pf_tvt.size != int(mask.sum()):
            n_len_mismatch += 1
            continue

        tw = D.load_typewell(well, split)
        p1_feats = build_features(h, tw)
        trk_feats = build_tracker_features(cached_anchor, pf_tvt, pf_std, beam_tvt, beam_margin)
        feats = pd.concat(
            [p1_feats.reset_index(drop=True), trk_feats.reset_index(drop=True)], axis=1
        )

        y_true = h["TVT"].to_numpy(dtype=np.float32)[mask]  # label only, never a feature
        pf_blend = cached_anchor + PF_BLEND_W * (pf_tvt - cached_anchor)

        well_i = len(used_wells)
        used_wells.append(well)

        feat_parts.append(feats)
        y_true_parts.append(y_true.astype(np.float64))
        anchor_parts.append(np.full(y_true.shape[0], cached_anchor, dtype=np.float64))
        pf_blend_parts.append(pf_blend)
        pf_std_parts.append(pf_std)
        well_idx_parts.append(np.full(y_true.shape[0], well_i, dtype=np.int32))

    X = pd.concat(feat_parts, ignore_index=True)
    y_true = np.concatenate(y_true_parts)
    anchor_arr = np.concatenate(anchor_parts)
    pf_blend_arr = np.concatenate(pf_blend_parts)
    pf_std_arr = np.concatenate(pf_std_parts)
    well_idx = np.concatenate(well_idx_parts)
    return (
        X,
        y_true,
        anchor_arr,
        pf_blend_arr,
        pf_std_arr,
        well_idx,
        used_wells,
        n_cache_missing,
        n_len_mismatch,
    )


def run_target_cv(
    label: str,
    X: pd.DataFrame,
    target: np.ndarray,
    base_arr: np.ndarray,
    y_true: np.ndarray,
    row_fold: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    """Train+OOF-score one target formulation across ``n_splits`` well-folds.

    ``base_arr`` is the row-aligned baseline added back to the predicted
    target to get a TVT prediction (``anchor`` for (a), ``pf_blend`` for (b)).
    Returns ``(oof_pred_tvt, fold_rows, mean_importance)``.
    """
    feature_cols = list(X.columns)
    oof_pred_tvt = np.full(len(y_true), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importances = np.zeros(len(feature_cols), dtype=np.float64)

    for fold_i in range(n_splits):
        t_fold = time.time()
        train_mask = row_fold != fold_i
        valid_mask = row_fold == fold_i

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X.loc[train_mask],
            target[train_mask],
            eval_set=[(X.loc[valid_mask], target[valid_mask])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        pred_target = model.predict(X.loc[valid_mask])
        pred_tvt = base_arr[valid_mask] + pred_target
        oof_pred_tvt[valid_mask] = pred_tvt

        fold_rmse = pooled_rmse(y_true[valid_mask], pred_tvt)
        importances += model.feature_importances_.astype(np.float64) / n_splits

        fold_rows.append(
            {
                "target": label,
                "fold": fold_i,
                "n_rows": int(valid_mask.sum()),
                "best_iteration": model.best_iteration_,
                "rmse": fold_rmse,
                "seconds": time.time() - t_fold,
            }
        )
        print(
            f"  [{label}] fold {fold_i}: n_rows={int(valid_mask.sum()):,} "
            f"best_iter={model.best_iteration_} rmse={fold_rmse:.4f} "
            f"({time.time() - t_fold:.1f}s)",
            flush=True,
        )

    assert not np.isnan(oof_pred_tvt).any(), "every row must be scored by exactly one fold"
    return oof_pred_tvt, fold_rows, importances


def _nested_shrink(
    y_delta: np.ndarray, pred_delta: np.ndarray, row_fold: np.ndarray, n_splits: int
) -> tuple[np.ndarray, list[float]]:
    """Per-fold nested least-squares shrink ``s``: ``y_delta ~ s * pred_delta``.

    For each fold ``i``, ``s_i`` is fit on the OOF rows of every *other*
    fold, then applied only to fold ``i``'s rows -- an honest (no-refit-leak)
    stacking scalar. Returns ``(shrunk pred_delta, [s_0..s_{k-1}])``.
    """
    out = np.empty_like(pred_delta)
    s_values: list[float] = []
    for fold_i in range(n_splits):
        fit_rows = row_fold != fold_i
        den = float(np.sum(pred_delta[fit_rows] ** 2))
        s = float(np.sum(y_delta[fit_rows] * pred_delta[fit_rows]) / den) if den > 0 else 1.0
        out[row_fold == fold_i] = s * pred_delta[row_fold == fold_i]
        s_values.append(s)
    return out, s_values


def _nested_shrink_by_pf_std_tercile(
    y_delta: np.ndarray,
    pred_delta: np.ndarray,
    pf_std_arr: np.ndarray,
    row_fold: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, list[list[float]]]:
    """Like :func:`_nested_shrink` but with a separate ``s`` per pf_std tercile.

    Tercile edges are computed from the fitting folds' rows only (nested,
    like the scalars). Non-finite ``pf_std`` rows (PF fallback) land in the
    top bucket via ``np.digitize`` on ``inf``. Returns ``(shrunk pred_delta,
    per-fold [s_low, s_mid, s_high])``.
    """
    out = np.empty_like(pred_delta)
    s_table: list[list[float]] = []
    for fold_i in range(n_splits):
        fit_rows = row_fold != fold_i
        apply_rows = row_fold == fold_i
        finite_fit_std = pf_std_arr[fit_rows]
        finite_fit_std = finite_fit_std[np.isfinite(finite_fit_std)]
        edges = np.percentile(finite_fit_std, [100 / 3, 200 / 3])

        fold_s: list[float] = []
        fit_bucket = np.digitize(pf_std_arr[fit_rows], edges)
        apply_bucket = np.digitize(pf_std_arr[apply_rows], edges)
        apply_idx = np.where(apply_rows)[0]
        for bucket in range(3):
            fb = fit_bucket == bucket
            den = float(np.sum(pred_delta[fit_rows][fb] ** 2))
            s = (
                float(np.sum(y_delta[fit_rows][fb] * pred_delta[fit_rows][fb]) / den)
                if den > 0
                else 1.0
            )
            rows_b = apply_idx[apply_bucket == bucket]
            out[rows_b] = s * pred_delta[rows_b]
            fold_s.append(s)
        s_table.append(fold_s)
    return out, s_table


def _helped_hurt_report(
    label: str,
    y_true: np.ndarray,
    stack_pred: np.ndarray,
    pf_blend_arr: np.ndarray,
    pf_std_arr: np.ndarray,
    well_idx: np.ndarray,
    n_wells: int,
) -> None:
    stack_well_rmse = _per_well_rmse(y_true, stack_pred, well_idx, n_wells)
    pf_well_rmse = _per_well_rmse(y_true, pf_blend_arr, well_idx, n_wells)
    well_pf_std_mean = np.array(
        [float(np.nanmean(pf_std_arr[well_idx == w_i])) for w_i in range(n_wells)]
    )

    valid = np.isfinite(stack_well_rmse) & np.isfinite(pf_well_rmse)
    helped = valid & (stack_well_rmse < pf_well_rmse)
    hurt = valid & (stack_well_rmse > pf_well_rmse)
    tied = valid & (stack_well_rmse == pf_well_rmse)

    print(f"\nhelped/hurt vs PF blend w0.7 -- ({label}):")
    print(f"  helped: {int(helped.sum())}  hurt: {int(hurt.sum())}  tied: {int(tied.sum())}")
    if helped.any():
        print(
            f"  median RMSE improvement (helped wells): "
            f"{float(np.median(pf_well_rmse[helped] - stack_well_rmse[helped])):.4f} ft"
        )
    if hurt.any():
        print(
            f"  median RMSE regression (hurt wells):    "
            f"{float(np.median(stack_well_rmse[hurt] - pf_well_rmse[hurt])):.4f} ft"
        )
    helped_pf_std = float(np.nanmean(well_pf_std_mean[helped])) if helped.any() else float("nan")
    hurt_pf_std = float(np.nanmean(well_pf_std_mean[hurt])) if hurt.any() else float("nan")
    print(f"  mean per-well pf_std -- helped: {helped_pf_std:.3f}  hurt: {hurt_pf_std:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wells",
        type=int,
        default=None,
        help="Health-check subset size (stratified by manifest pf_std_mean decile). "
        "Omit for the full 773-well experiment.",
    )
    parser.add_argument(
        "--save-oof",
        action="store_true",
        help="Save OOF prediction arrays to outputs/stack_oof.npz for post-hoc analysis.",
    )
    args = parser.parse_args()

    t_start = time.time()

    all_wells = D.list_wells("train")
    if args.n_wells is not None:
        wells = _stratified_sample(all_wells, args.n_wells, SEED)
        print(
            f"health-check subset: {len(wells)}/{len(all_wells)} wells "
            "(stratified by pf_std_mean)"
        )
    else:
        wells = all_wells
        print(f"train wells: {len(wells)}")

    t0 = time.time()
    (
        X,
        y_true,
        anchor_arr,
        pf_blend_arr,
        pf_std_arr,
        well_idx,
        used_wells,
        n_cache_missing,
        n_len_mismatch,
    ) = build_well_matrix(wells, split="train")
    print(
        f"feature matrix: {X.shape}, rows={len(y_true):,}, "
        f"wells_used={len(used_wells)}/{len(wells)} "
        f"(cache_missing={n_cache_missing}, len_mismatch={n_len_mismatch}) "
        f"({time.time() - t0:.1f}s)"
    )

    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    row_fold = np.array([well_to_fold[w] for w in used_wells], dtype=np.int32)[well_idx]

    n_wells_used = len(used_wells)

    # --- baselines on this exact row set (for an apples-to-apples comparison) ---
    floor_pooled = pooled_rmse(y_true, anchor_arr)
    pf_blend_pooled = pooled_rmse(y_true, pf_blend_arr)
    floor_well_rmse = _per_well_rmse(y_true, anchor_arr, well_idx, n_wells_used)
    pf_well_rmse = _per_well_rmse(y_true, pf_blend_arr, well_idx, n_wells_used)

    # --- (a) U-direct: target = TVT - anchor ---
    y_u = y_true - anchor_arr
    print("\n=== (a) U-direct (target = TVT - anchor) ===")
    oof_a, fold_rows_a, importances_a = run_target_cv(
        "a:U-direct", X, y_u, anchor_arr, y_true, row_fold, N_SPLITS
    )

    # --- (b) PF-residual: target = TVT - pf_blend_w0.7 ---
    y_r = y_true - pf_blend_arr
    print("\n=== (b) PF-residual (target = TVT - pf_blend_w0.7) ===")
    oof_b, fold_rows_b, importances_b = run_target_cv(
        "b:PF-residual", X, y_r, pf_blend_arr, y_true, row_fold, N_SPLITS
    )

    a_pooled = pooled_rmse(y_true, oof_a)
    b_pooled = pooled_rmse(y_true, oof_b)
    a_well_rmse = _per_well_rmse(y_true, oof_a, well_idx, n_wells_used)
    b_well_rmse = _per_well_rmse(y_true, oof_b, well_idx, n_wells_used)

    print()
    print("=" * 78)
    print(f"{'predictor':<28}{'pooled RMSE':>14}{'median':>10}{'p90':>10}{'max':>10}")
    print("-" * 78)
    for name, pooled, well_arr in [
        ("carry_last (floor)", floor_pooled, floor_well_rmse),
        ("PF blend w0.7 (baseline)", pf_blend_pooled, pf_well_rmse),
        ("(a) U-direct stack", a_pooled, a_well_rmse),
        ("(b) PF-residual stack", b_pooled, b_well_rmse),
    ]:
        valid = well_arr[np.isfinite(well_arr)]
        print(
            f"{name:<28}{pooled:>14.4f}{np.median(valid):>10.4f}"
            f"{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )
    print("=" * 78)
    print(f"delta (a) vs PF blend (ledger 11.0621): {a_pooled - PF_BLEND_POOLED_RMSE:+.4f}")
    print(f"delta (b) vs PF blend (ledger 11.0621): {b_pooled - PF_BLEND_POOLED_RMSE:+.4f}")
    print(f"total rows scored: {len(y_true):,}")
    print()

    fold_df = pd.concat([pd.DataFrame(fold_rows_a), pd.DataFrame(fold_rows_b)], ignore_index=True)
    print("per-fold RMSE / best_iteration (stability check):")
    print(fold_df.to_string(index=False))
    print()

    feature_cols = list(X.columns)
    imp_a = (
        pd.DataFrame({"feature": feature_cols, "mean_importance": importances_a})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_b = (
        pd.DataFrame({"feature": feature_cols, "mean_importance": importances_b})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    print("feature importance top10 -- (a) U-direct:")
    print(imp_a.head(10).to_string(index=False))
    print()
    print("feature importance top10 -- (b) PF-residual:")
    print(imp_b.head(10).to_string(index=False))

    _helped_hurt_report("a", y_true, oof_a, pf_blend_arr, pf_std_arr, well_idx, n_wells_used)
    _helped_hurt_report("b", y_true, oof_b, pf_blend_arr, pf_std_arr, well_idx, n_wells_used)

    # --- post-hoc nested stacking variants (no retraining; scalars fit nested) ---
    print("\n=== post-hoc nested stacking variants (scalars fit on other folds only) ===")

    # (b)-shrink: pred = pf_blend + s * (oof_b - pf_blend)
    shrunk_b_delta, s_b = _nested_shrink(
        y_true - pf_blend_arr, oof_b - pf_blend_arr, row_fold, N_SPLITS
    )
    pred_b_shrink = pf_blend_arr + shrunk_b_delta

    # (a,b)-blend: pred = oof_b + w * (oof_a - oof_b)
    blend_delta, w_ab = _nested_shrink(y_true - oof_b, oof_a - oof_b, row_fold, N_SPLITS)
    pred_ab_blend = oof_b + blend_delta

    # (b)-shrink per pf_std tercile
    shrunk_b3_delta, s_b3 = _nested_shrink_by_pf_std_tercile(
        y_true - pf_blend_arr, oof_b - pf_blend_arr, pf_std_arr, row_fold, N_SPLITS
    )
    pred_b_shrink3 = pf_blend_arr + shrunk_b3_delta

    print(f"  (b)-shrink   per-fold s: {[f'{s:.3f}' for s in s_b]}")
    print(f"  (a,b)-blend  per-fold w: {[f'{w:.3f}' for w in w_ab]}")
    print(
        "  (b)-shrink-tercile per-fold [s_low, s_mid, s_high]: "
        + str([[f"{s:.3f}" for s in fold_s] for fold_s in s_b3])
    )
    print()
    print(f"{'post-hoc variant':<32}{'pooled RMSE':>14}{'median':>10}{'p90':>10}{'max':>10}")
    print("-" * 78)
    for name, pred in [
        ("(b)+nested shrink", pred_b_shrink),
        ("(a,b) nested blend", pred_ab_blend),
        ("(b)+shrink by pf_std tercile", pred_b_shrink3),
    ]:
        pooled = pooled_rmse(y_true, pred)
        well_arr = _per_well_rmse(y_true, pred, well_idx, n_wells_used)
        valid = well_arr[np.isfinite(well_arr)]
        print(
            f"{name:<32}{pooled:>14.4f}{np.median(valid):>10.4f}"
            f"{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )

    if args.save_oof:
        out_dir = Path(__file__).resolve().parents[1] / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        oof_path = out_dir / "stack_oof.npz"
        np.savez_compressed(
            oof_path,
            y_true=y_true.astype(np.float32),
            oof_a=oof_a.astype(np.float32),
            oof_b=oof_b.astype(np.float32),
            anchor=anchor_arr.astype(np.float32),
            pf_blend=pf_blend_arr.astype(np.float32),
            pf_std=pf_std_arr.astype(np.float32),
            well_idx=well_idx,
            row_fold=row_fold,
            used_wells=np.array(used_wells),
        )
        print(f"\nOOF arrays saved: {oof_path}")

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
