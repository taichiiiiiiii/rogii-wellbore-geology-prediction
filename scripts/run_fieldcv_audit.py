"""Field-grouped CV audit for stack(b): is well-level GroupKFold optimistic? (epistemics).

Host claim: nearby wells are highly informative of each other's surface depth
("R^2>0.99 from ~10 neighboring wells"). The mandated CV split is **well-unit**
``GroupKFold`` (``rogii.cv.well_folds``) -- it guarantees a well's own rows never
straddle train/valid, but it does *not* guarantee a validation well's spatial
*neighbors* are excluded from training. If hidden test wells sit outside the
spatial footprint of train, well-unit OOF could be optimistic relative to LB.

This script re-measures the current-best stack(b) (PF-residual GBDT, same
config as ``scripts/run_stack_cv.py``) under a **field-grouped** GroupKFold:

  1. cluster train-well centroids (mean X, mean Y over the whole horizontal
     log -- always known, never a leak) into ``k`` spatial "fields" (KMeans);
  2. assign whole fields to folds with a greedy well-count-balanced bin-pack
     (LPT heuristic), so a fold's validation wells' nearest train well is a
     *different field*, not just a different well;
  3. retrain stack(b) (identical features/target/LightGBM config to
     ``run_stack_cv.py``) under these field folds and report OOF pooled RMSE;
  4. diagnose how much spatial isolation each split scheme actually buys, via
     the fold-wise valid-well -> nearest-train-well centroid distance.

Self-contained by design (v1 files must not be modified/imported): the
feature-matrix-building and GBDT-training code below is a deliberate
duplicate of ``run_stack_cv.py``'s ``build_well_matrix`` / ``run_target_cv``,
not an import. Reads (read-only, never writes):
  - ``data/processed/tracker_cache/*.npz`` (PF/beam cache, built by
    ``scripts/build_tracker_cache.py``);
  - ``outputs/stack_oof.npz`` (well-GroupKFold OOF, saved by
    ``run_stack_cv.py --save-oof``) for an apples-to-apples per-well-stats
    comparison and the well-GroupKFold half of the distance diagnostic; if
    absent, falls back to ``rogii.cv.well_folds`` recomputation for the
    diagnostic and the ledger scalar (10.6513) for the pooled-RMSE row.

Leak boundary: identical to ``run_stack_cv.py`` -- ``TVT`` is read exactly
once per well to build ``y_true`` / the residual target, never inside a
feature builder; well centroids use ``X``/``Y`` only (present in test too).

Usage::

    uv run python scripts/run_fieldcv_audit.py --n-wells 80   # health check
    uv run python scripts/run_fieldcv_audit.py                # full audit
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rogii import data as D
from rogii.features import build_features
from rogii.stack_features import build_tracker_features

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "processed" / "tracker_cache"
STACK_OOF_PATH = REPO_ROOT / "outputs" / "stack_oof.npz"  # read-only baseline artifact

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621  # ledger: PF blend w0.7 (split-invariant, no training)
STACK_B_WELL_GROUPKFOLD_LEDGER = 10.6513  # ledger: stack(b), well-level GroupKFold OOF
PF_BLEND_W = 0.7  # matches build_tracker_cache.py / run_stack_cv.py

N_SPLITS = 5
SEED = 42
DEFAULT_K = 15
IMBALANCE_CV_THRESHOLD = 0.5  # coefficient of variation (std/mean) of field well-counts
IMBALANCE_MIN_FIELD = 5  # smallest acceptable field before treating k as "too fine"

# Identical to run_stack_cv.py -- same feature set / target / model config.
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


# --------------------------------------------------------------------------
# Feature matrix / GBDT training -- self-contained duplicate of
# run_stack_cv.py's build_well_matrix / run_target_cv (target (b) only).
# --------------------------------------------------------------------------

_BuildMatrixResult = tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int, int
]


def build_well_matrix(wells: list[str], split: str = "train") -> _BuildMatrixResult:
    """Build the combined (P1 + tracker) feature matrix and both targets' ingredients.

    Also returns each used well's centroid (mean X, mean Y over the *entire*
    horizontal log, not just the eval zone) as a side channel via the module-
    level ``_last_centroids`` dict populated during the same pass (avoids a
    second CSV read per well for the field-clustering step).
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
    centroids: dict[str, tuple[float, float]] = {}

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
        centroids[well] = (float(h["X"].mean()), float(h["Y"].mean()))  # X/Y: known in test too

        feat_parts.append(feats)
        y_true_parts.append(y_true.astype(np.float64))
        anchor_parts.append(np.full(y_true.shape[0], cached_anchor, dtype=np.float64))
        pf_blend_parts.append(pf_blend)
        pf_std_parts.append(pf_std)
        well_idx_parts.append(np.full(y_true.shape[0], well_i, dtype=np.int32))

    X = pd.concat(feat_parts, ignore_index=True)
    y_true_arr = np.concatenate(y_true_parts)
    anchor_arr = np.concatenate(anchor_parts)
    pf_blend_arr = np.concatenate(pf_blend_parts)
    pf_std_arr = np.concatenate(pf_std_parts)
    well_idx = np.concatenate(well_idx_parts)
    build_well_matrix.last_centroids = centroids  # type: ignore[attr-defined]
    return (
        X,
        y_true_arr,
        anchor_arr,
        pf_blend_arr,
        pf_std_arr,
        well_idx,
        used_wells,
        n_cache_missing,
        n_len_mismatch,
    )


def run_target_cv(
    X: pd.DataFrame,
    target: np.ndarray,
    base_arr: np.ndarray,
    y_true: np.ndarray,
    row_fold: np.ndarray,
    n_splits: int,
    label: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Train+OOF-score one target formulation across ``n_splits`` folds (``row_fold``)."""
    oof_pred_tvt = np.full(len(y_true), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []

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
    return oof_pred_tvt, fold_rows


# --------------------------------------------------------------------------
# Field clustering / fold assignment
# --------------------------------------------------------------------------


def cluster_fields(
    centroid_map: dict[str, tuple[float, float]], wells: list[str], k: int, seed: int
) -> dict[str, int]:
    coords = np.array([centroid_map[w] for w in wells])
    scaled = StandardScaler().fit_transform(coords)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(scaled)
    return {w: int(lbl) for w, lbl in zip(wells, labels, strict=True)}


def field_balance(well_to_field: dict[str, int], k: int) -> tuple[pd.Series, float]:
    counts = pd.Series(well_to_field).value_counts().reindex(range(k), fill_value=0).sort_index()
    cv_score = float(counts.std() / counts.mean())
    return counts, cv_score


def assign_fields_to_folds(
    field_sizes: pd.Series, n_splits: int
) -> tuple[dict[int, int], list[int]]:
    """Greedy well-count-balanced bin-pack (LPT heuristic): largest field first,
    always placed into the currently-smallest fold."""
    field_ids = sorted(field_sizes.index, key=lambda f: (-field_sizes[f], f))
    fold_totals = [0] * n_splits
    field_to_fold: dict[int, int] = {}
    for fid in field_ids:
        t = int(np.argmin(fold_totals))
        field_to_fold[fid] = t
        fold_totals[t] += int(field_sizes[fid])
    return field_to_fold, fold_totals


# --------------------------------------------------------------------------
# Diagnostic: valid-well -> nearest train-well centroid distance
# --------------------------------------------------------------------------


def nearest_train_distances(
    well_to_fold: dict[str, int], centroid_map: dict[str, tuple[float, float]], n_splits: int
) -> dict[int, np.ndarray]:
    wells = list(well_to_fold.keys())
    coords = np.array([centroid_map[w] for w in wells])
    fold_arr = np.array([well_to_fold[w] for w in wells])
    result: dict[int, np.ndarray] = {}
    for fold_i in range(n_splits):
        valid_mask = fold_arr == fold_i
        train_mask = ~valid_mask
        if valid_mask.sum() == 0 or train_mask.sum() == 0:
            result[fold_i] = np.array([])
            continue
        d = cdist(coords[valid_mask], coords[train_mask])
        result[fold_i] = d.min(axis=1)
    return result


def summarize_distances(dist_by_fold: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    all_d = []
    for fold_i, d in dist_by_fold.items():
        if d.size == 0:
            continue
        rows.append(
            {
                "fold": fold_i,
                "n_valid_wells": int(d.size),
                "median_ft": float(np.median(d)),
                "p90_ft": float(np.percentile(d, 90)),
            }
        )
        all_d.append(d)
    df = pd.DataFrame(rows)
    pooled = np.concatenate(all_d)
    overall = pd.DataFrame(
        [
            {
                "fold": "ALL",
                "n_valid_wells": int(pooled.size),
                "median_ft": float(np.median(pooled)),
                "p90_ft": float(np.percentile(pooled, 90)),
            }
        ]
    )
    return pd.concat([df, overall], ignore_index=True)


def _print_table(rows: list[tuple[str, float, np.ndarray]]) -> None:
    print(f"{'predictor':<40}{'pooled RMSE':>14}{'median':>10}{'p90':>10}{'max':>10}")
    print("-" * 84)
    for name, pooled, well_arr in rows:
        valid = well_arr[np.isfinite(well_arr)]
        if valid.size == 0:
            print(f"{name:<40}{pooled:>14.4f}{'n/a':>10}{'n/a':>10}{'n/a':>10}")
            continue
        print(
            f"{name:<40}{pooled:>14.4f}{np.median(valid):>10.4f}"
            f"{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wells",
        type=int,
        default=None,
        help="Health-check subset size (uniform random sample). Omit for the full 773-well audit.",
    )
    args = parser.parse_args()
    t_start = time.time()

    all_wells = D.list_wells("train")
    is_health_check = args.n_wells is not None
    if is_health_check:
        wells = sorted(random.Random(SEED).sample(all_wells, args.n_wells))
        print(f"health-check subset: {len(wells)}/{len(all_wells)} wells")
    else:
        wells = all_wells
        print(f"train wells: {len(wells)}")

    # --- 1. feature matrix (self-contained duplicate of run_stack_cv.py) ---
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
    centroid_map: dict[str, tuple[float, float]] = build_well_matrix.last_centroids  # type: ignore[attr-defined]
    n_wells_used = len(used_wells)
    print(
        f"feature matrix: {X.shape}, rows={len(y_true):,}, "
        f"wells_used={n_wells_used}/{len(wells)} "
        f"(cache_missing={n_cache_missing}, len_mismatch={n_len_mismatch}) "
        f"({time.time() - t0:.1f}s)"
    )

    # --- 2. spatial "field" clustering (KMeans on standardized well centroids) ---
    t0 = time.time()
    k = DEFAULT_K
    well_to_field = cluster_fields(centroid_map, used_wells, k, SEED)
    counts, cv_score = field_balance(well_to_field, k)
    print(f"\nfield clustering: k={k} seed={SEED}, well-count CV(std/mean)={cv_score:.3f}")
    print(counts.rename("n_wells").to_string())

    retried = False
    if cv_score > IMBALANCE_CV_THRESHOLD or counts.min() < IMBALANCE_MIN_FIELD:
        retried = True
        dominant_frac = counts.max() / n_wells_used
        alt_k = 20 if dominant_frac > 0.25 else 10
        well_to_field_alt = cluster_fields(centroid_map, used_wells, alt_k, SEED)
        counts_alt, cv_alt = field_balance(well_to_field_alt, alt_k)
        print(
            f"\nimbalance retry (CV={cv_score:.3f} > {IMBALANCE_CV_THRESHOLD} or "
            f"min_field={int(counts.min())} < {IMBALANCE_MIN_FIELD}): "
            f"k={alt_k}, well-count CV={cv_alt:.3f}"
        )
        print(counts_alt.rename("n_wells").to_string())
        if cv_alt < cv_score:
            k, well_to_field, counts, cv_score = alt_k, well_to_field_alt, counts_alt, cv_alt
            print(f"-> adopting k={k} (better balance)")
        else:
            print(f"-> keeping k={DEFAULT_K} (retry did not improve balance)")
    print(f"field clustering done ({time.time() - t0:.1f}s, retried={retried})")

    # --- 3. field -> fold assignment (greedy well-count-balanced bin-pack) ---
    field_sizes = pd.Series(well_to_field).value_counts()
    field_to_fold, fold_totals = assign_fields_to_folds(field_sizes, N_SPLITS)
    target_per_fold = n_wells_used / N_SPLITS
    print(f"\nfield->fold well-count totals (target ~{target_per_fold:.0f}/fold): {fold_totals}")
    well_to_fold_field = {w: field_to_fold[well_to_field[w]] for w in used_wells}
    row_fold_field = np.array([well_to_fold_field[w] for w in used_wells], dtype=np.int32)[well_idx]

    # --- 4. baselines on this exact row set ---
    floor_pooled = pooled_rmse(y_true, anchor_arr)
    pf_blend_pooled = pooled_rmse(y_true, pf_blend_arr)
    floor_well_rmse = _per_well_rmse(y_true, anchor_arr, well_idx, n_wells_used)
    pf_well_rmse = _per_well_rmse(y_true, pf_blend_arr, well_idx, n_wells_used)

    # --- 5. train stack(b) [PF-residual] under field-grouped folds ---
    print("\n=== (b) PF-residual stack, field-GroupKFold (same features/target/LGBM as v1) ===")
    y_r = y_true - pf_blend_arr
    oof_b_field, fold_rows_b = run_target_cv(
        X, y_r, pf_blend_arr, y_true, row_fold_field, N_SPLITS, label="b:field-GroupKFold"
    )
    b_field_pooled = pooled_rmse(y_true, oof_b_field)
    b_field_well_rmse = _per_well_rmse(y_true, oof_b_field, well_idx, n_wells_used)

    # --- 6. well-GroupKFold baseline (read-only from outputs/stack_oof.npz, full run only) ---
    well_stats: dict[str, object] | None = None
    if not is_health_check and STACK_OOF_PATH.exists():
        with np.load(STACK_OOF_PATH, allow_pickle=True) as npz:
            y_true_wg = npz["y_true"].astype(np.float64)
            oof_b_wg = npz["oof_b"].astype(np.float64)
            well_idx_wg = npz["well_idx"]
            row_fold_wg = npz["row_fold"]
            used_wells_wg = [str(w) for w in npz["used_wells"]]
        b_well_pooled = pooled_rmse(y_true_wg, oof_b_wg)
        b_well_well_rmse = _per_well_rmse(y_true_wg, oof_b_wg, well_idx_wg, len(used_wells_wg))
        well_to_fold_wg = {
            w: int(row_fold_wg[well_idx_wg == i][0]) for i, w in enumerate(used_wells_wg)
        }
        well_stats = {
            "pooled": b_well_pooled,
            "well_rmse": b_well_well_rmse,
            "well_to_fold": well_to_fold_wg,
            "used_wells": used_wells_wg,
        }
        print(
            f"\nwell-GroupKFold baseline (recomputed from outputs/stack_oof.npz): "
            f"pooled={b_well_pooled:.4f} (ledger {STACK_B_WELL_GROUPKFOLD_LEDGER:.4f}, "
            f"delta={b_well_pooled - STACK_B_WELL_GROUPKFOLD_LEDGER:+.4f})"
        )
        if set(used_wells_wg) != set(used_wells):
            print(
                f"[warn] outputs/stack_oof.npz used_wells ({len(used_wells_wg)}) != "
                f"this run's used_wells ({len(used_wells)}) -- comparison may be inexact"
            )
    elif not is_health_check:
        print(f"\n[warn] {STACK_OOF_PATH} not found -- falling back to ledger scalar only")

    # --- 7. comparison table ---
    print()
    print("=" * 84)
    rows: list[tuple[str, float, np.ndarray]] = [
        ("carry_last (floor)", floor_pooled, floor_well_rmse),
        ("PF blend w0.7 (split-invariant ref)", pf_blend_pooled, pf_well_rmse),
    ]
    if well_stats is not None:
        wg_pooled = well_stats["pooled"]  # type: ignore[assignment]
        wg_well_rmse = well_stats["well_rmse"]  # type: ignore[assignment]
        rows.append(("(b) stack, well-GroupKFold [recomputed]", wg_pooled, wg_well_rmse))  # type: ignore[arg-type]
    else:
        rows.append(
            (
                "(b) stack, well-GroupKFold [ledger scalar]",
                STACK_B_WELL_GROUPKFOLD_LEDGER,
                np.array([]),
            )
        )
    rows.append(("(b) stack, field-GroupKFold [NEW, this run]", b_field_pooled, b_field_well_rmse))
    _print_table(rows)
    print("=" * 84)
    ledger_delta = b_field_pooled - STACK_B_WELL_GROUPKFOLD_LEDGER
    print(f"delta: field-GroupKFold - well-GroupKFold(ledger 10.6513) = {ledger_delta:+.4f}")
    print(f"total rows scored: {len(y_true):,}")

    fold_df = pd.DataFrame(fold_rows_b)
    print("\nper-fold RMSE / best_iteration (field-GroupKFold, stability check):")
    print(fold_df.to_string(index=False))

    # --- 8. diagnostic: valid-well -> nearest train-well centroid distance ---
    print("\n=== diagnostic: valid-well -> nearest train-well centroid distance ===")
    print("(ft, X/Y projected coords -- consistent with MD/TVT units)")
    dist_field = nearest_train_distances(well_to_fold_field, centroid_map, N_SPLITS)
    print("\nfield-GroupKFold:")
    print(summarize_distances(dist_field).to_string(index=False))

    if well_stats is not None:
        dist_well = nearest_train_distances(well_stats["well_to_fold"], centroid_map, N_SPLITS)  # type: ignore[arg-type]
        print("\nwell-GroupKFold (folds as actually used for the ledger 10.6513 run):")
    else:
        from rogii.cv import well_folds as _well_folds  # local import: diagnostic-only fallback

        folds = _well_folds(used_wells, n_splits=N_SPLITS, seed=SEED)
        well_to_fold_wg = {w: i for i, fold in enumerate(folds) for w in fold}
        dist_well = nearest_train_distances(well_to_fold_wg, centroid_map, N_SPLITS)
        print("\nwell-GroupKFold (recomputed via rogii.cv.well_folds, diagnostic only):")
    print(summarize_distances(dist_well).to_string(index=False))

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
