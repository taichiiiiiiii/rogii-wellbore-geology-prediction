"""Stack v2: honest-ES + well-aggregate + spatial features on top of v1's (b) formulation
(issue: stack v2).

v1 (``scripts/run_stack_cv.py``, NOT modified here, read-only reference) measured two
target formulations and found (b) PF-residual strictly better than (a) U-direct
(current best: pooled **10.6513**, see ``analysis/experiment_ledger.md``). v2 keeps
(b) exclusively (``target = TVT - pf_blend_w0.7``) and studies three independent levers
on top of it, each isolated by an ablation row that changes exactly one thing relative
to the v1-reproduction baseline:

  1. **ES holdout separation** -- v1 used the *score* fold itself as LightGBM's
     early-stopping validation set (an honest-but-optimistic informal admission in the
     ledger: "ES にスコア fold 使用=微楽観"). v2 instead carves an additional 15% of
     each fold's *training* wells out as a dedicated ES-holdout
     (:func:`_split_fit_es`), so the reported score fold is genuinely never touched by
     early stopping.
  2. **Well-aggregate tracker features** -- 8 new per-well scalars
     (``rogii.stack_features.build_well_aggregate_features``), broadcast to every row
     of that well: final/mean PF drift, well-wide PF-std and beam-margin stats, PF/beam
     disagreement, eval-zone length, and GR NaN fraction.
  3. **Spatial-prior features** -- 4 new per-row features
     (``rogii.stack_features.build_spatial_features``) built from the leave-self-out
     spatial-prior cache (``data/processed/spatial_cache/*.npz``, built by
     ``scripts/build_spatial_cache.py`` from ``rogii.spatial``; NOT built here, this
     script only reads it).

Five rows come out of a single run (one feature matrix built once, reused via column
subsetting across configs -- see ``ABLATION_CONFIGS``):

  (b) v1 repro        -- P1(36) + tracker(8) features, v1's score-fold-as-ES protocol
  +ES holdout only     -- same features, honest ES-holdout protocol
  +well-agg only       -- + well-aggregate(8) features, v1's ES protocol (isolates lever 2)
  +spatial only        -- + spatial(4) features, v1's ES protocol (isolates lever 3)
  all-in               -- all features + honest ES-holdout protocol (levers 1+2+3 combined)

Leak boundary: identical to v1 -- ``rogii.features.build_features`` and
``rogii.stack_features.build_*`` never read ``h["TVT"]`` or a train-only formation
column; ``TVT`` is read exactly once per well, here, to build the regression target.
``gr_nan_frac`` (fed into ``build_well_aggregate_features``) is computed from ``h["GR"]``,
a column present in both train and test schemas.

Usage::

    uv run python scripts/run_stack_v2_cv.py                 # full 773-well experiment
    uv run python scripts/run_stack_v2_cv.py --n-wells 80    # stratified health-check subset
    uv run python scripts/run_stack_v2_cv.py --save-oof      # dump all 5 configs' OOF to outputs/
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
from rogii.stack_features import (
    FEATURE_COLUMNS,
    SPATIAL_FEATURE_COLUMNS,
    WELL_FEATURE_COLUMNS,
    build_spatial_features,
    build_tracker_features,
    build_well_aggregate_features,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "tracker_cache"
TRACKER_MANIFEST_PATH = TRACKER_CACHE_DIR / "manifest.csv"
SPATIAL_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "spatial_cache"

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
V1_STACK_B_POOLED_RMSE = 10.6513  # scripts/run_stack_cv.py (b) PF-residual, ledger 2026-07-06
PF_BLEND_W = 0.7  # matches build_tracker_cache.py / build_spatial_cache.py / v1

N_SPLITS = 5
SEED = 42
N_STRATA = 10  # deciles of manifest pf_std_mean, for the --n-wells health-check sample
ES_FRAC = 0.15  # fraction of each fold's training wells carved out as ES holdout

# Identical to v1 (scripts/run_stack_cv.py) -- "LGBM設定はv1踏襲" (task spec).
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


def _tracker_npz_path(well: str) -> Path:
    return TRACKER_CACHE_DIR / f"{well}.npz"


def _spatial_npz_path(well: str) -> Path:
    return SPATIAL_CACHE_DIR / f"{well}.npz"


def _stratified_sample(all_wells: list[str], n: int, seed: int) -> list[str]:
    """Deterministic sample of ``n`` wells, stratified by tracker-manifest pf_std_mean decile.

    Same recipe as ``scripts/run_stack_cv.py::_stratified_sample`` (not imported --
    that script is v1, off-limits to modify/import-couple with).
    """
    manifest = pd.read_csv(TRACKER_MANIFEST_PATH)
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


class _WellMatrix:
    """Container for the single feature matrix all 5 ablation configs subset from."""

    def __init__(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        anchor_arr: np.ndarray,
        pf_blend_arr: np.ndarray,
        pf_std_arr: np.ndarray,
        well_idx: np.ndarray,
        used_wells: list[str],
        p1_cols: list[str],
        counts: dict[str, int],
    ) -> None:
        self.X = X
        self.y_true = y_true
        self.anchor_arr = anchor_arr
        self.pf_blend_arr = pf_blend_arr
        self.pf_std_arr = pf_std_arr
        self.well_idx = well_idx
        self.used_wells = used_wells
        self.p1_cols = p1_cols
        self.counts = counts


def build_well_matrix_v2(wells: list[str], split: str = "train") -> _WellMatrix:
    """Build the combined (P1 + tracker + well-agg + spatial) feature matrix.

    A well is included only if it has a non-empty eval zone, a known anchor, a
    cached tracker result of matching length, *and* a cached spatial-prior result of
    matching length -- so every ablation config trains/scores on exactly the same
    row set (feature-column subsetting is the only thing that varies between
    configs, never the underlying wells/rows).
    """
    feat_parts: list[pd.DataFrame] = []
    y_true_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    pf_blend_parts: list[np.ndarray] = []
    pf_std_parts: list[np.ndarray] = []
    well_idx_parts: list[np.ndarray] = []
    used_wells: list[str] = []
    p1_cols: list[str] = []
    counts = {
        "n_no_eval_zone_or_anchor": 0,
        "n_tracker_cache_missing": 0,
        "n_tracker_len_mismatch": 0,
        "n_spatial_cache_missing": 0,
        "n_spatial_len_mismatch": 0,
    }

    for well in wells:
        h = D.load_horizontal(well, split)
        mask = D.eval_mask(h)
        if mask.sum() == 0:
            counts["n_no_eval_zone_or_anchor"] += 1
            continue
        try:
            D.last_known_tvt(h)
        except ValueError:
            counts["n_no_eval_zone_or_anchor"] += 1
            continue

        trk_path = _tracker_npz_path(well)
        if not trk_path.exists():
            counts["n_tracker_cache_missing"] += 1
            continue
        with np.load(trk_path) as npz:
            pf_tvt = npz["pf_tvt"].astype(np.float64)
            pf_std = npz["pf_std"].astype(np.float64)
            beam_tvt = npz["beam_tvt"].astype(np.float64)
            beam_margin = npz["beam_margin"].astype(np.float64)
            cached_anchor = float(npz["anchor"])
        if pf_tvt.size != int(mask.sum()):
            counts["n_tracker_len_mismatch"] += 1
            continue

        sp_path = _spatial_npz_path(well)
        if not sp_path.exists():
            counts["n_spatial_cache_missing"] += 1
            continue
        with np.load(sp_path) as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
            prefix_rmse = float(npz["prefix_rmse"])
            nn_dist_median = float(npz["nn_dist_median"])
        if spatial_tvt.size != int(mask.sum()):
            counts["n_spatial_len_mismatch"] += 1
            continue

        tw = D.load_typewell(well, split)
        p1_feats = build_features(h, tw)
        if not p1_cols:
            p1_cols = list(p1_feats.columns)

        trk_feats = build_tracker_features(cached_anchor, pf_tvt, pf_std, beam_tvt, beam_margin)
        gr_nan_frac = float(h["GR"].isna().mean()) if "GR" in h.columns else 1.0
        well_agg_feats = build_well_aggregate_features(
            cached_anchor, pf_tvt, pf_std, beam_tvt, beam_margin, gr_nan_frac
        )
        spatial_feats = build_spatial_features(
            cached_anchor, spatial_tvt, prefix_rmse, nn_dist_median
        )

        feats = pd.concat(
            [
                p1_feats.reset_index(drop=True),
                trk_feats.reset_index(drop=True),
                well_agg_feats.reset_index(drop=True),
                spatial_feats.reset_index(drop=True),
            ],
            axis=1,
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
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, p1_cols, counts
    )


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Deterministic: ``train_wells`` is sorted first (canonical order), then shuffled
    with a seeded RNG and dealt into a leading ES-holdout slice and a trailing fit
    slice. The ES-holdout wells are used *only* for LightGBM's early-stopping metric
    -- they are excluded from both the training rows and (by construction, since they
    come from the fold's training wells) the score fold. This is the honest
    counterpart to v1's protocol, which reused the score fold itself as the
    early-stopping validation set (ledger: "ES にスコア fold 使用=微楽観").
    """
    ordered = sorted(train_wells)
    random.Random(seed).shuffle(ordered)
    n_es = max(1, round(len(ordered) * es_frac))
    es_wells = sorted(ordered[:n_es])
    fit_wells = sorted(ordered[n_es:])
    return fit_wells, es_wells


def run_stack_fold_cv(
    label: str,
    X: pd.DataFrame,
    target: np.ndarray,
    base_arr: np.ndarray,
    y_true: np.ndarray,
    row_fold: np.ndarray,
    well_idx: np.ndarray,
    well_fold_arr: np.ndarray,
    used_wells: list[str],
    n_splits: int,
    use_es_holdout: bool,
    es_frac: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    """Train+OOF-score the (b) PF-residual target for one ablation config.

    ``use_es_holdout=False`` reproduces v1's protocol exactly: train on every row
    outside the score fold, early-stop on the score fold itself. ``use_es_holdout=True``
    additionally carves a 15% well-level ES-holdout out of the training wells
    (:func:`_split_fit_es`) and early-stops on that instead, never touching the score
    fold until final prediction.
    """
    feature_cols = list(X.columns)
    oof_pred_tvt = np.full(len(y_true), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importances = np.zeros(len(feature_cols), dtype=np.float64)
    used_well_to_idx = {w: i for i, w in enumerate(used_wells)}

    for fold_i in range(n_splits):
        t_fold = time.time()
        score_mask = row_fold == fold_i

        if use_es_holdout:
            train_well_positions = np.where(well_fold_arr != fold_i)[0]
            train_wells_all = [used_wells[wi] for wi in train_well_positions]
            fit_wells, es_wells = _split_fit_es(train_wells_all, es_frac, seed)
            fit_idx = np.array([used_well_to_idx[w] for w in fit_wells], dtype=np.int32)
            es_idx = np.array([used_well_to_idx[w] for w in es_wells], dtype=np.int32)
            train_mask = np.isin(well_idx, fit_idx)
            es_mask = np.isin(well_idx, es_idx)
            assert not (train_mask & es_mask).any(), "fit/ES-holdout rows must not overlap"
            assert not (train_mask & score_mask).any(), "fit rows must not overlap score fold"
            assert not (es_mask & score_mask).any(), "ES-holdout rows must not overlap score fold"
            assert bool(
                (train_mask | es_mask | score_mask).all()
            ), "every row must land in exactly one bucket"
            eval_set = [(X.loc[es_mask], target[es_mask])]
        else:
            train_mask = row_fold != fold_i
            eval_set = [(X.loc[score_mask], target[score_mask])]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X.loc[train_mask],
            target[train_mask],
            eval_set=eval_set,
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        pred_target = model.predict(X.loc[score_mask])
        pred_tvt = base_arr[score_mask] + pred_target
        oof_pred_tvt[score_mask] = pred_tvt

        fold_rmse = pooled_rmse(y_true[score_mask], pred_tvt)
        importances += model.feature_importances_.astype(np.float64) / n_splits

        fold_rows.append(
            {
                "config": label,
                "fold": fold_i,
                "n_train_rows": int(train_mask.sum()),
                "n_score_rows": int(score_mask.sum()),
                "best_iteration": model.best_iteration_,
                "rmse": fold_rmse,
                "seconds": time.time() - t_fold,
            }
        )
        print(
            f"  [{label}] fold {fold_i}: n_train={int(train_mask.sum()):,} "
            f"n_score={int(score_mask.sum()):,} best_iter={model.best_iteration_} "
            f"rmse={fold_rmse:.4f} ({time.time() - t_fold:.1f}s)",
            flush=True,
        )

    assert not np.isnan(oof_pred_tvt).any(), "every row must be scored by exactly one fold"
    return oof_pred_tvt, fold_rows, importances


def _helped_hurt_report(
    label: str,
    y_true: np.ndarray,
    stack_pred: np.ndarray,
    pf_blend_arr: np.ndarray,
    well_idx: np.ndarray,
    n_wells: int,
) -> None:
    stack_well_rmse = _per_well_rmse(y_true, stack_pred, well_idx, n_wells)
    pf_well_rmse = _per_well_rmse(y_true, pf_blend_arr, well_idx, n_wells)

    valid = np.isfinite(stack_well_rmse) & np.isfinite(pf_well_rmse)
    helped = valid & (stack_well_rmse < pf_well_rmse)
    hurt = valid & (stack_well_rmse > pf_well_rmse)
    tied = valid & (stack_well_rmse == pf_well_rmse)

    print(f"\nhelped/hurt vs PF blend w0.7 -- ({label}):")
    print(f"  helped: {int(helped.sum())}  hurt: {int(hurt.sum())}  tied: {int(tied.sum())}")
    if helped.any():
        print(
            "  median RMSE improvement (helped wells): "
            f"{float(np.median(pf_well_rmse[helped] - stack_well_rmse[helped])):.4f} ft"
        )
    if hurt.any():
        print(
            "  median RMSE regression (hurt wells):    "
            f"{float(np.median(stack_well_rmse[hurt] - pf_well_rmse[hurt])):.4f} ft"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wells",
        type=int,
        default=None,
        help="Health-check subset size (stratified by tracker-manifest pf_std_mean "
        "decile). Omit for the full 773-well experiment.",
    )
    parser.add_argument(
        "--save-oof",
        action="store_true",
        help="Save every config's OOF prediction arrays to outputs/stack_v2_oof.npz.",
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
    wm = build_well_matrix_v2(wells, split="train")
    print(
        f"feature matrix: {wm.X.shape}, rows={len(wm.y_true):,}, "
        f"wells_used={len(wm.used_wells)}/{len(wells)} counts={wm.counts} "
        f"({time.time() - t0:.1f}s)"
    )

    # GroupKFold(5, seed=42) computed from the SAME full wells list as v1
    # (scripts/run_stack_cv.py) -> identical fold assignment (task requirement).
    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr = np.array([well_to_fold[w] for w in wm.used_wells], dtype=np.int32)
    row_fold = well_fold_arr[wm.well_idx]

    n_wells_used = len(wm.used_wells)

    floor_pooled = pooled_rmse(wm.y_true, wm.anchor_arr)
    pf_blend_pooled = pooled_rmse(wm.y_true, wm.pf_blend_arr)
    floor_well_rmse = _per_well_rmse(wm.y_true, wm.anchor_arr, wm.well_idx, n_wells_used)
    pf_well_rmse = _per_well_rmse(wm.y_true, wm.pf_blend_arr, wm.well_idx, n_wells_used)

    tracker_cols = list(FEATURE_COLUMNS)
    wellagg_cols = list(WELL_FEATURE_COLUMNS)
    spatial_cols = list(SPATIAL_FEATURE_COLUMNS)

    # Each row changes exactly ONE thing relative to the v1-repro baseline (row 1),
    # except the final "all-in" row which combines all three levers.
    ablation_configs: list[tuple[str, list[str], bool]] = [
        ("(b) v1 repro", wm.p1_cols + tracker_cols, False),
        ("+ES holdout only", wm.p1_cols + tracker_cols, True),
        ("+well-agg only", wm.p1_cols + tracker_cols + wellagg_cols, False),
        ("+spatial only", wm.p1_cols + tracker_cols + spatial_cols, False),
        ("all-in (levers 1+2+3)", wm.p1_cols + tracker_cols + wellagg_cols + spatial_cols, True),
    ]

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, the only formulation in v2

    results: dict[str, np.ndarray] = {}
    all_fold_rows: list[dict[str, object]] = []
    importances_by_config: dict[str, pd.DataFrame] = {}

    for label, cols, use_es_holdout in ablation_configs:
        print(f"\n=== {label} (use_es_holdout={use_es_holdout}, n_features={len(cols)}) ===")
        X_cfg = wm.X[cols]
        oof, fold_rows, importances = run_stack_fold_cv(
            label,
            X_cfg,
            y_r,
            wm.pf_blend_arr,
            wm.y_true,
            row_fold,
            wm.well_idx,
            well_fold_arr,
            wm.used_wells,
            N_SPLITS,
            use_es_holdout,
            ES_FRAC,
            SEED,
        )
        results[label] = oof
        all_fold_rows.extend(fold_rows)
        importances_by_config[label] = (
            pd.DataFrame({"feature": cols, "mean_importance": importances})
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )

    print()
    print("=" * 96)
    print(
        f"{'predictor':<26}{'pooled RMSE':>13}{'vs PF blend':>13}{'vs v1(b) 10.6513':>18}"
        f"{'median':>10}{'p90':>10}{'max':>10}"
    )
    print("-" * 96)
    for name, pooled, well_arr in [
        ("carry_last (floor)", floor_pooled, floor_well_rmse),
        ("PF blend w0.7", pf_blend_pooled, pf_well_rmse),
    ]:
        valid = well_arr[np.isfinite(well_arr)]
        print(
            f"{name:<26}{pooled:>13.4f}{'--':>13}{'--':>18}"
            f"{np.median(valid):>10.4f}{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )
    for label, _cols, _use_es in ablation_configs:
        oof = results[label]
        pooled = pooled_rmse(wm.y_true, oof)
        well_arr = _per_well_rmse(wm.y_true, oof, wm.well_idx, n_wells_used)
        valid = well_arr[np.isfinite(well_arr)]
        print(
            f"{label:<26}{pooled:>13.4f}{pooled - pf_blend_pooled:>+13.4f}"
            f"{pooled - V1_STACK_B_POOLED_RMSE:>+18.4f}"
            f"{np.median(valid):>10.4f}{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )
    print("=" * 96)
    print(f"total rows scored: {len(wm.y_true):,}")

    fold_df = pd.DataFrame(all_fold_rows)
    print("\nper-fold RMSE / best_iteration (stability check):")
    print(fold_df.to_string(index=False))

    final_label = ablation_configs[-1][0]
    print(f"\nfeature importance top10 -- {final_label}:")
    print(importances_by_config[final_label].head(10).to_string(index=False))

    _helped_hurt_report(
        final_label, wm.y_true, results[final_label], wm.pf_blend_arr, wm.well_idx, n_wells_used
    )

    if args.save_oof:
        out_dir = _REPO_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        oof_path = out_dir / "stack_v2_oof.npz"
        save_kwargs = {
            f"oof_{i}": results[label].astype(np.float32)
            for i, (label, _cols, _use_es) in enumerate(ablation_configs)
        }
        np.savez_compressed(
            oof_path,
            y_true=wm.y_true.astype(np.float32),
            anchor=wm.anchor_arr.astype(np.float32),
            pf_blend=wm.pf_blend_arr.astype(np.float32),
            well_idx=wm.well_idx,
            row_fold=row_fold,
            used_wells=np.array(wm.used_wells),
            config_labels=np.array([label for label, _c, _e in ablation_configs]),
            **save_kwargs,
        )
        print(f"\nOOF arrays saved: {oof_path}")

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
