"""Stack v2.7: LGBM hyperparameter coarse grid on the 56 original features (issue: R14).

**Why R14.** Every attempt to add information (bank candidates: R7-R9 all
LB-toxic; raw features: R12/R13 sub-gate) has stalled, but the LGBM
hyperparameters themselves have been frozen at stack v2's original values
(``num_leaves=63, min_child_samples=50, learning_rate=0.05/800est,
feature_fraction=0.9``) since the first stack run -- they were never swept.
R14 is a **transfer-safe** lever: the feature set stays the LB-verified 56
columns (CV 9.2309 -> LB 9.022, the only confirmed transfer), so any CV gain
here cannot come from bank-candidate leak-through -- only from a better
bias/variance point of the same information.

**Design.** One-dimensional sweeps around the current base params, seed 42
only (time-saving; matches how stack v2's 9.2309 reference was measured).
The coordinator's requested grid deduplicates against the current values
(num_leaves current=63, min_child_samples current=50, feature_fraction
current=0.9 -- the "現行" arms ARE the baseline) to **8 configs**:

  ==========  ==================================================
  key         params delta vs LGB_PARAMS_BASE
  ==========  ==================================================
  base        (none -- must reproduce stack v2 all-in 9.2309)
  nl31        num_leaves=31
  nl127       num_leaves=127
  nl255       num_leaves=255
  mcs200      min_child_samples=200
  mcs500      min_child_samples=500
  lr02        learning_rate=0.02, n_estimators=2000 (x2.5)
  ff06        feature_fraction=0.6
  ==========  ==================================================

Everything else is frozen, identical to v2/R7-R13: fold assignment
(``cv.well_folds(wells, 5, seed=42)``), ES-holdout split
(``_split_fit_es(..., seed=42)``), early stopping 50 rounds, PF-residual
target. Each sweep config runs in its own OS process (checkpoint npz flushed
per config), strictly serial.

**Gate / confirmation.** ``--compare`` prints the sweep table and the
machine-readable markers ``CONFIRM_TRIGGER: YES|NO`` and ``CONFIRM_KEY:
<key>``. Trigger = best config pooled <= (measured base pooled - 0.05). The
night-batch driver then runs the winning config for the 4 remaining seeds
(101/202/303/404; the sweep's own seed-42 checkpoint IS member 0) and
``--aggregate-confirm`` prints the 5-seed-mean final value.

**Leak boundary.** Identical to R11 (``run_stack_v24_cv.py``): the feature
matrix is a byte-copy of ``build_well_matrix_v2`` -- no bank npz, no
``h["TVT"]`` feature reads, no train-only formation columns.

Usage::

    # smoke test (one sweep config end-to-end, ~30 wells)
    uv run python scripts/run_stack_v27_cv.py --sweep-idx 0 --n-wells 30

    # full 773-well sweep, ONE PROCESS PER CONFIG, strictly serial
    uv run python scripts/run_stack_v27_cv.py --sweep-idx 0
    ...
    uv run python scripts/run_stack_v27_cv.py --sweep-idx 7

    # table + confirmation-trigger decision
    uv run python scripts/run_stack_v27_cv.py --compare

    # 5-seed confirmation of the winning config (driver runs these on trigger)
    uv run python scripts/run_stack_v27_cv.py --confirm --sweep-key <key> --seed-idx 1
    ...
    uv run python scripts/run_stack_v27_cv.py --aggregate-confirm --sweep-key <key>
"""

from __future__ import annotations

import argparse
import gc
import random
import resource
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rogii import cv as CV
from rogii import data as D
from rogii.features import build_features
from rogii.stack_features import (
    build_spatial_features,
    build_tracker_features,
    build_well_aggregate_features,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "tracker_cache"
TRACKER_MANIFEST_PATH = TRACKER_CACHE_DIR / "manifest.csv"
SPATIAL_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "spatial_cache"
OUTPUTS_DIR = _REPO_ROOT / "outputs"

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
STACK_V2_ALLIN_POOLED_RMSE = 9.2309  # ledger 2026-07-07 (LB 9.022) -- base config must repro this
PF_BLEND_W = 0.7

CONFIRM_TRIGGER_GATE_FT = 0.05  # best <= measured base - 0.05 -> run 5-seed confirmation
Y_TRUE_ATOL = 1e-2
BASE_REPRO_ATOL = 0.01  # base config vs stack v2 "all-in" 9.2309, full scale only

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN (matches v2/R7-R13)
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN (matches v2/R7-R13)
N_STRATA = 10
ES_FRAC = 0.15

SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404)
SWEEP_SEED = 42  # every sweep config runs this single seed

LGB_PARAMS_BASE: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50

N_FEATURES_EXPECTED = 56  # P1(36) + tracker(8) + well-agg(8) + spatial(4), v2's "all-in" columns

# One-dimensional sweep around LGB_PARAMS_BASE (see module docstring for the
# dedup rationale: 現行 num_leaves=63 / min_child_samples=50 /
# feature_fraction=0.9 arms are all the "base" row).
SWEEP: tuple[tuple[str, dict[str, object]], ...] = (
    ("base", {}),
    ("nl31", {"num_leaves": 31}),
    ("nl127", {"num_leaves": 127}),
    ("nl255", {"num_leaves": 255}),
    ("mcs200", {"min_child_samples": 200}),
    ("mcs500", {"min_child_samples": 500}),
    ("lr02", {"learning_rate": 0.02, "n_estimators": 2000}),
    ("ff06", {"feature_fraction": 0.6}),
)
SWEEP_KEYS: tuple[str, ...] = tuple(k for k, _ in SWEEP)


def _sweep_overrides(key: str) -> dict[str, object]:
    for k, overrides in SWEEP:
        if k == key:
            return overrides
    raise ValueError(f"unknown sweep key {key!r}, expected one of {SWEEP_KEYS}")


def _lgb_params_for(key: str, seed: int) -> dict[str, object]:
    return {**LGB_PARAMS_BASE, **_sweep_overrides(key), "random_state": seed}


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _peak_rss_mb() -> float:
    """Current process's peak (high-water-mark) RSS in MB, Linux ``ru_maxrss`` is KB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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

    Same recipe as ``run_stack_v2/.../v26_cv.py``.
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


# --------------------------------------------------------------------------- #
# 56-col feature builder, byte-copied from run_stack_v24_cv.py (R11)
# --------------------------------------------------------------------------- #


class _WellMatrix:
    """Container for the single 56-column feature matrix every config trains on."""

    def __init__(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        anchor_arr: np.ndarray,
        pf_blend_arr: np.ndarray,
        pf_std_arr: np.ndarray,
        well_idx: np.ndarray,
        used_wells: list[str],
        counts: dict[str, int],
    ) -> None:
        self.X = X
        self.y_true = y_true
        self.anchor_arr = anchor_arr
        self.pf_blend_arr = pf_blend_arr
        self.pf_std_arr = pf_std_arr
        self.well_idx = well_idx
        self.used_wells = used_wells
        self.counts = counts


def build_well_matrix_v2(wells: list[str], split: str = "train") -> _WellMatrix:
    """Two-pass, preallocating build of the v2 56-column feature matrix.

    Byte-copied from ``run_stack_v24_cv.py`` (R11). This is the ENTIRE
    feature set R14 trains on -- hyperparameters are the only thing swept.
    """
    counts = {
        "n_no_eval_zone_or_anchor": 0,
        "n_tracker_cache_missing": 0,
        "n_tracker_len_mismatch": 0,
        "n_spatial_cache_missing": 0,
        "n_spatial_len_mismatch": 0,
    }

    used_wells: list[str] = []
    row_counts: list[int] = []
    for well in wells:
        h = D.load_horizontal(well, split)
        mask = D.eval_mask(h)
        n = int(mask.sum())
        if n == 0:
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
            pf_tvt_size = int(npz["pf_tvt"].shape[0])
        if pf_tvt_size != n:
            counts["n_tracker_len_mismatch"] += 1
            continue

        sp_path = _spatial_npz_path(well)
        if not sp_path.exists():
            counts["n_spatial_cache_missing"] += 1
            continue
        with np.load(sp_path) as npz:
            spatial_size = int(npz["spatial_tvt"].shape[0])
        if spatial_size != n:
            counts["n_spatial_len_mismatch"] += 1
            continue

        used_wells.append(well)
        row_counts.append(n)

    if not used_wells:
        raise ValueError(
            f"build_well_matrix_v2: 0 wells passed validation -- no rows to build "
            f"(counts={counts})"
        )

    total_rows = int(sum(row_counts))
    boundaries = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int64)

    y_true = np.empty(total_rows, dtype=np.float64)
    anchor_arr = np.empty(total_rows, dtype=np.float64)
    pf_blend_arr = np.empty(total_rows, dtype=np.float64)
    pf_std_arr = np.empty(total_rows, dtype=np.float64)
    well_idx = np.empty(total_rows, dtype=np.int32)

    all_cols: list[str] = []
    col_arrays: dict[str, np.ndarray] = {}

    for well_i, well in enumerate(used_wells):
        s, e = int(boundaries[well_i]), int(boundaries[well_i + 1])

        h = D.load_horizontal(well, split)
        mask = D.eval_mask(h)
        with np.load(_tracker_npz_path(well)) as npz:
            pf_tvt = npz["pf_tvt"].astype(np.float64)
            pf_std = npz["pf_std"].astype(np.float64)
            beam_tvt = npz["beam_tvt"].astype(np.float64)
            beam_margin = npz["beam_margin"].astype(np.float64)
            cached_anchor = float(npz["anchor"])
        with np.load(_spatial_npz_path(well)) as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
            prefix_rmse = float(npz["prefix_rmse"])
            nn_dist_median = float(npz["nn_dist_median"])

        tw = D.load_typewell(well, split)
        p1_feats = build_features(h, tw)

        trk_feats = build_tracker_features(cached_anchor, pf_tvt, pf_std, beam_tvt, beam_margin)
        gr_nan_frac = float(h["GR"].isna().mean()) if "GR" in h.columns else 1.0
        well_agg_feats = build_well_aggregate_features(
            cached_anchor, pf_tvt, pf_std, beam_tvt, beam_margin, gr_nan_frac
        )
        spatial_feats = build_spatial_features(
            cached_anchor, spatial_tvt, prefix_rmse, nn_dist_median
        )

        if not col_arrays:
            all_cols = (
                list(p1_feats.columns)
                + list(trk_feats.columns)
                + list(well_agg_feats.columns)
                + list(spatial_feats.columns)
            )
            col_arrays = {c: np.empty(total_rows, dtype=np.float32) for c in all_cols}

        for block in (p1_feats, trk_feats, well_agg_feats, spatial_feats):
            for c in block.columns:
                col_arrays[c][s:e] = block[c].to_numpy(dtype=np.float32, copy=False)

        y_true_well = h["TVT"].to_numpy(dtype=np.float32)[mask]  # label only, never a feature
        y_true[s:e] = y_true_well.astype(np.float64)
        anchor_arr[s:e] = cached_anchor
        pf_blend_arr[s:e] = cached_anchor + PF_BLEND_W * (pf_tvt - cached_anchor)
        pf_std_arr[s:e] = pf_std
        well_idx[s:e] = well_i

    X = pd.DataFrame(col_arrays)
    assert list(X.columns) == all_cols, "dict-insertion column order must match all_cols"
    if len(all_cols) != N_FEATURES_EXPECTED:
        raise AssertionError(
            f"R14 must train on exactly {N_FEATURES_EXPECTED} (stack v2 'all-in') "
            f"features, got {len(all_cols)}"
        )
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, counts
    )


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Byte-for-byte identical to ``run_stack_v2/.../v26_cv.py``'s
    ``_split_fit_es``. Always called with ``ES_SPLIT_SEED`` (42, frozen).
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
    es_frac: float,
    lgb_params: dict[str, object],
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    """Train+OOF-score the (b) PF-residual target for one (config, seed) member.

    Identical protocol to ``run_stack_v22/.../v26_cv.py``'s
    ``run_stack_fold_cv`` with ES-holdout always on.
    """
    feature_cols = list(X.columns)
    oof_pred_tvt = np.full(len(y_true), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importances = np.zeros(len(feature_cols), dtype=np.float64)
    used_well_to_idx = {w: i for i, w in enumerate(used_wells)}

    for fold_i in range(n_splits):
        t_fold = time.time()
        score_mask = row_fold == fold_i

        train_well_positions = np.where(well_fold_arr != fold_i)[0]
        train_wells_all = [used_wells[wi] for wi in train_well_positions]
        fit_wells, es_wells = _split_fit_es(train_wells_all, es_frac, ES_SPLIT_SEED)
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

        model = lgb.LGBMRegressor(**lgb_params)
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
        del model, eval_set, train_mask, score_mask
        gc.collect()

    assert not np.isnan(oof_pred_tvt).any(), "every row must be scored by exactly one fold"
    return oof_pred_tvt, fold_rows, importances


def run_one_member(
    wm: _WellMatrix,
    key: str,
    seed: int,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
) -> dict[str, object]:
    """Run one (sweep-config, seed) member to completion."""
    params = _lgb_params_for(key, seed)
    overrides = _sweep_overrides(key)
    label = f"{key} seed {seed}"
    print(f"\n=== {label} (56 cols, overrides={overrides or 'none (base)'}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1..v2.6
    oof, fold_rows, importances = run_stack_fold_cv(
        label,
        wm.X,
        y_r,
        wm.pf_blend_arr,
        wm.y_true,
        row_fold,
        wm.well_idx,
        well_fold_arr,
        wm.used_wells,
        N_SPLITS,
        ES_FRAC,
        params,
    )
    gc.collect()

    pooled = pooled_rmse(wm.y_true, oof)
    importance_df = (
        pd.DataFrame({"feature": list(wm.X.columns), "mean_importance": importances})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "key": key,
        "seed": seed,
        "label": label,
        "cols": list(wm.X.columns),
        "oof": oof,
        "fold_rows": fold_rows,
        "importance_df": importance_df,
        "pooled_rmse": pooled,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _result_path(key: str, seed: int, n_wells: int | None) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    return OUTPUTS_DIR / f"stack_v27_result_{key}_s{seed}{suffix}.npz"


def _save_result(
    result: dict[str, object], wm: _WellMatrix, row_fold: np.ndarray, n_wells: int | None
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(result["fold_rows"])
    imp_df = result["importance_df"]
    path = _result_path(str(result["key"]), int(result["seed"]), n_wells)
    np.savez_compressed(
        path,
        key=str(result["key"]),
        seed=np.int64(result["seed"]),
        label=result["label"],
        cols=np.array(result["cols"]),
        oof=np.asarray(result["oof"], dtype=np.float32),
        y_true=wm.y_true.astype(np.float32),
        anchor=wm.anchor_arr.astype(np.float32),
        pf_blend=wm.pf_blend_arr.astype(np.float32),
        well_idx=wm.well_idx,
        row_fold=row_fold,
        used_wells=np.array(wm.used_wells),
        pooled_rmse=np.float64(result["pooled_rmse"]),
        peak_rss_mb=np.float64(result["peak_rss_mb"]),
        fold_id=fold_df["fold"].to_numpy(),
        fold_rmse=fold_df["rmse"].to_numpy(),
        fold_best_iteration=fold_df["best_iteration"].to_numpy(),
        fold_n_train=fold_df["n_train_rows"].to_numpy(),
        fold_n_score=fold_df["n_score_rows"].to_numpy(),
        fold_seconds=fold_df["seconds"].to_numpy(),
        importance_feature=imp_df["feature"].to_numpy(),
        importance_value=imp_df["mean_importance"].to_numpy(),
    )
    print(f"\nresult saved: {path} (peak RSS this process: {result['peak_rss_mb']:.0f} MB)")


def _load_result(key: str, seed: int, n_wells: int | None) -> dict[str, object]:
    path = _result_path(key, seed, n_wells)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with np.load(path, allow_pickle=True) as npz:
        return {
            "key": str(npz["key"]),
            "seed": int(npz["seed"]),
            "label": str(npz["label"]),
            "oof": npz["oof"].astype(np.float64),
            "y_true": npz["y_true"].astype(np.float64),
            "anchor": npz["anchor"].astype(np.float64),
            "pf_blend": npz["pf_blend"].astype(np.float64),
            "well_idx": npz["well_idx"],
            "row_fold": npz["row_fold"],
            "used_wells": [str(w) for w in npz["used_wells"]],
            "pooled_rmse": float(npz["pooled_rmse"]),
            "fold_rmse": npz["fold_rmse"],
            "fold_best_iteration": npz["fold_best_iteration"],
        }


def _compare(n_wells: int | None) -> None:
    """Sweep table + machine-readable confirmation-trigger markers for the driver."""
    rows: list[tuple[str, float, float]] = []
    base_pooled: float | None = None
    for key in SWEEP_KEYS:
        try:
            res = _load_result(key, SWEEP_SEED, n_wells)
        except FileNotFoundError:
            print(f"WARNING: sweep config {key!r} has no result npz -- skipped")
            continue
        mean_best_iter = float(np.mean(res["fold_best_iteration"]))
        rows.append((key, res["pooled_rmse"], mean_best_iter))
        if key == "base":
            base_pooled = res["pooled_rmse"]

    if not rows:
        print("no sweep results found -- nothing to compare")
        return
    if base_pooled is None:
        print("ERROR: base config result missing -- cannot compute the trigger gate")
        print("CONFIRM_TRIGGER: NO")
        return

    print("\n" + "=" * 96)
    print("R14 LGBM hyperparameter sweep (56 features, seed 42, 5-fold pooled OOF)")
    print("=" * 96)
    print(
        f"{'key':>8}{'overrides':>36}{'pooled OOF':>14}{'vs base':>12}{'mean best_iter':>16}"
    )
    print("-" * 96)
    for key, pooled, mean_best_iter in rows:
        overrides = _sweep_overrides(key)
        ov_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(base)"
        print(
            f"{key:>8}{ov_str:>36}{pooled:>14.4f}{pooled - base_pooled:>+12.4f}"
            f"{mean_best_iter:>16.1f}"
        )
    print("=" * 96)

    if n_wells is None:
        diff = base_pooled - STACK_V2_ALLIN_POOLED_RMSE
        ok = abs(diff) <= BASE_REPRO_ATOL
        print(
            f"\nbase repro check: {base_pooled:.4f} vs stack v2 all-in ledger 9.2309 "
            f"(diff {diff:+.4f}ft, tol {BASE_REPRO_ATOL}ft) -> {'PASS' if ok else 'FAIL'}"
        )

    best_key, best_pooled, _ = min(rows, key=lambda r: r[1])
    gate = base_pooled - CONFIRM_TRIGGER_GATE_FT
    triggered = best_pooled <= gate
    print(
        f"\nbest: {best_key} = {best_pooled:.4f} (base {base_pooled:.4f}, "
        f"delta {best_pooled - base_pooled:+.4f}ft, trigger gate <= {gate:.4f})"
    )
    # Machine-readable markers -- the night-batch driver greps for these.
    print(f"CONFIRM_TRIGGER: {'YES' if triggered else 'NO'}")
    print(f"CONFIRM_KEY: {best_key}")


def _aggregate_confirm(key: str, n_wells: int | None) -> None:
    """5-seed mean for the confirmed config (sweep's s42 npz is member 0)."""
    results: list[dict[str, object]] = []
    for seed in SEEDS:
        try:
            results.append(_load_result(key, seed, n_wells))
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
    if not results:
        print(f"no result files for config {key!r} -- nothing to aggregate")
        return

    ref = results[0]
    y_true = ref["y_true"]
    well_idx = ref["well_idx"]
    used_wells = ref["used_wells"]
    for res in results[1:]:
        if not np.allclose(res["y_true"], y_true, atol=Y_TRUE_ATOL):
            raise AssertionError(
                f"config {key} seed {res['seed']}'s y_true disagrees with seed "
                f"{ref['seed']}'s -- not the same wells list"
            )

    cum_sum = np.zeros_like(y_true, dtype=np.float64)
    print()
    print("=" * 96)
    print(f"R14 confirmation: config {key} x {len(results)} seeds")
    print(f"{'k':>3}{'seed':>7}{'solo pooled':>14}{'cum-mean pooled (1..k)':>24}")
    print("-" * 96)
    cum_pooled_list: list[float] = []
    for k, res in enumerate(results, start=1):
        cum_sum += res["oof"]
        cum_pooled = pooled_rmse(y_true, cum_sum / k)
        cum_pooled_list.append(cum_pooled)
        print(f"{k:>3}{res['seed']:>7}{res['pooled_rmse']:>14.4f}{cum_pooled:>24.4f}")
    print("=" * 96)
    oof_mean = cum_sum / len(results)
    mean_pooled = cum_pooled_list[-1]

    mean_well_rmse = _per_well_rmse(y_true, oof_mean, well_idx, len(used_wells))
    valid = mean_well_rmse[np.isfinite(mean_well_rmse)]
    print(
        f"\n{len(results)}-seed mean OOF (config {key}): pooled={mean_pooled:.4f}  "
        f"median={np.median(valid):.4f}  p90={np.percentile(valid, 90):.4f}  "
        f"max={np.max(valid):.4f}"
    )

    out_path = OUTPUTS_DIR / "stack_v27_oof.npz"
    save_kwargs = {
        f"oof_{i}": np.asarray(res["oof"], dtype=np.float32) for i, res in enumerate(results)
    }
    np.savez_compressed(
        out_path,
        key=key,
        y_true=y_true.astype(np.float32),
        anchor=ref["anchor"].astype(np.float32),
        pf_blend=ref["pf_blend"].astype(np.float32),
        oof_mean=oof_mean.astype(np.float32),
        well_idx=well_idx,
        row_fold=ref["row_fold"],
        used_wells=np.array(used_wells),
        seeds=np.array([res["seed"] for res in results], dtype=np.int64),
        pooled_per_seed=np.array([res["pooled_rmse"] for res in results], dtype=np.float64),
        pooled_cumulative=np.array(cum_pooled_list, dtype=np.float64),
        **save_kwargs,
    )
    print(f"\nOOF saved: {out_path}")
    print(
        f"\nR14 CONFIRMED VALUE: config {key} {len(results)}-seed mean = {mean_pooled:.4f} "
        f"(R11 56-feat 5-seed mean was 9.1843)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wells", type=int, default=None, help="Health-check subset size.")
    parser.add_argument(
        "--sweep-idx",
        type=int,
        choices=range(len(SWEEP)),
        default=None,
        help=f"Run sweep config i (SWEEP keys={SWEEP_KEYS}) with seed {SWEEP_SEED}, then exit.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print the sweep table + CONFIRM_TRIGGER/CONFIRM_KEY markers.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Run one confirmation member: requires --sweep-key and --seed-idx.",
    )
    parser.add_argument("--sweep-key", choices=SWEEP_KEYS, default=None)
    parser.add_argument("--seed-idx", type=int, choices=range(len(SEEDS)), default=None)
    parser.add_argument(
        "--aggregate-confirm",
        action="store_true",
        help="Aggregate the confirmed config's 5 seeds (requires --sweep-key).",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.compare:
        _compare(args.n_wells)
        print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
        return

    if args.aggregate_confirm:
        if args.sweep_key is None:
            parser.error("--aggregate-confirm requires --sweep-key")
        _aggregate_confirm(args.sweep_key, args.n_wells)
        print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
        return

    if args.confirm:
        if args.sweep_key is None or args.seed_idx is None:
            parser.error("--confirm requires --sweep-key and --seed-idx")
        key, seed = args.sweep_key, SEEDS[args.seed_idx]
    elif args.sweep_idx is not None:
        key, seed = SWEEP_KEYS[args.sweep_idx], SWEEP_SEED
    else:
        parser.error("one of --sweep-idx / --confirm / --compare / --aggregate-confirm required")
        return  # unreachable, satisfies type-checkers

    all_wells = D.list_wells("train")
    if args.n_wells is not None:
        wells = _stratified_sample(all_wells, args.n_wells, FOLD_SEED)
        print(
            f"health-check subset: {len(wells)}/{len(all_wells)} wells "
            "(stratified by pf_std_mean)"
        )
    else:
        wells = all_wells
        print(f"train wells: {len(wells)}")

    t0 = time.time()
    wm = build_well_matrix_v2(wells, split="train")
    gc.collect()
    print(
        f"feature matrix (v2, 56 cols, R14): {wm.X.shape}, rows={len(wm.y_true):,}, "
        f"wells_used={len(wm.used_wells)}/{len(wells)} counts={wm.counts} "
        f"({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )

    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=FOLD_SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr = np.array([well_to_fold[w] for w in wm.used_wells], dtype=np.int32)
    row_fold = well_fold_arr[wm.well_idx]

    result = run_one_member(wm, key, seed, row_fold, well_fold_arr)
    print(
        f"  -> {key} seed {seed}: pooled RMSE={result['pooled_rmse']:.4f} "
        f"peak_rss={result['peak_rss_mb']:.0f}MB"
    )
    _save_result(result, wm, row_fold, args.n_wells)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
    print(f"this process's peak RSS: {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
