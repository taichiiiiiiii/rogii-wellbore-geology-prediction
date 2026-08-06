"""Stack v2.4: LGBM 5-seed ensemble of stack v2's ORIGINAL 56 features only (issue: R11).

**Why R11.** sub-v6 (stack v2.1 solo, R7's 60-feature config) scored **LB 10.009**
-- worse than stack v2's own LB 9.022 -- despite a CV improvement (9.2309 -> 8.9805).
Combined with the sub-v4 blend regression (CV 9.1104 -> LB 9.456) and the
pf_bma_scale3_v2 probe (train 11.80 -> LB 14.406), the ledger's 2026-07-10 entries
establish **calibration point #5**: any feature/candidate derived from the new
pf_bma_v2/beam_grid bank -- whether blended at the output level or injected as LGBM
features -- has failed to transfer to the hidden test set so far. **The only
CV-to-LB transfer that has ever been confirmed is stack v2's original 56 features**
(CV 9.2309 -> LB 9.022, calibration point #3). R11 asks a narrower, transfer-safe
question: does seed-averaging that *exact* 56-feature model (the "P3 lever" already
validated as low-risk in R9, where seed-ensembling stack v2.3's 66-feature config
moved 8.9152 -> 8.8573) also help on the 56-feature input, with **zero exposure**
to the new bank?

**Design.**

  - Feature set: **only** :func:`build_well_matrix_v2`'s 56 columns (P1(36) +
    tracker(8) + well-agg(8) + spatial(4)) -- byte-copied unmodified from
    ``run_stack_v23_cv.py`` (itself byte-copied from ``run_stack_v22_cv.py``). R7's
    4 (``pfbma_v2_d`` etc.), G1's 4, and G2's 2 are **not added** -- there is no
    call to anything resembling v23's ``add_config_c_columns``, and the
    ``pf_bma_v2``/``beam`` bank npz files are never opened by this script.
  - :data:`SEEDS` = (42, 101, 202, 303, 404), identical set and order to R9. Each
    seed runs the full 5-fold well-GroupKFold CV with ``random_state=seed`` in the
    LightGBM params -- same mechanism as R9 (LightGBM derives
    ``bagging_seed``/``feature_fraction_seed``/``data_random_seed`` from ``seed``
    when unset).
  - **Everything else is frozen**, identical to v2/R7/R8/R9: fold assignment
    (``cv.well_folds(wells, 5, seed=42)``) and ES-holdout well split
    (``_split_fit_es(..., seed=42)``) use the fixed seed 42 for every member --
    only LGBM training randomness moves.
  - Seed **42 is deliberately first**: with ``random_state=42`` and the frozen
    splits + 56-column feature set, member 0 is parameter-identical to
    ``run_stack_v2_cv.py``'s ``"all-in (levers 1+2+3)"`` row -- a built-in
    regression check that must reproduce **9.2309** (+/- 0.01).
  - Report: each seed's solo pooled OOF, the cumulative-mean pooled OOF after
    averaging members 1..k (k = 1..5), and the final 5-seed mean vs stack v2 solo
    (9.2309). Gate: **<= 9.1309** (9.2309 - 0.10) -- reported regardless of
    outcome, per task spec (seed-averaging alone is expected to buy roughly
    -0.05..-0.10ft, i.e. plausibly short of the gate; that is not a failure of the
    experiment).

**Memory.** 56 columns is lighter than v2.3's 66 (which peaked at 3125MB RSS) --
no bank-array loads, no G1/G2/R7 augmentation pass. Seeds still run strictly
serially, one OS process each, for parity with R9's harness and to keep peak RSS
low on this 3.8GB box. Per-seed results are checkpointed to
``outputs/stack_v24_result_s{seed}.npz`` immediately on completion.

**Leak boundary.** Identical to v2/R7-R9: all features come from
``rogii.features.build_features`` + ``rogii.stack_features.build_*`` (tracker/
spatial caches only); no ``h["TVT"]``/eval-zone ``TVT_input`` reads. No bank npz
(pf_bma_v2/beam) is read at all in this script -- the R11 input surface is a strict
subset of v2's, which is itself LB-verified leak-free.

Usage::

    # smoke test (all 5 seeds + aggregate in one process, ~40 wells)
    uv run python scripts/run_stack_v24_cv.py --n-wells 40

    # full 773-well run, ONE PROCESS PER SEED, strictly serial
    uv run python scripts/run_stack_v24_cv.py --seed-idx 0
    ...
    uv run python scripts/run_stack_v24_cv.py --seed-idx 4

    # after all 5 members have saved outputs/stack_v24_result_s*.npz:
    uv run python scripts/run_stack_v24_cv.py --aggregate
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
STACK_V2_ALLIN_POOLED_RMSE = 9.2309  # ledger 2026-07-07 (run_stack_v2_cv.py "all-in", LB 9.022)
PF_BLEND_W = 0.7

IMPROVEMENT_GATE_FT = 0.10
ADOPTION_GATE_RMSE = STACK_V2_ALLIN_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.1309
Y_TRUE_ATOL = 1e-2
REPRO_ATOL = 0.01  # seed-42 member vs stack v2 "all-in" 9.2309, full scale only

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN for every member (matches v2/R7-R9)
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN for every member (matches v2/R7-R9)
N_STRATA = 10
ES_FRAC = 0.15

# The ONLY thing that varies across ensemble members (see module docstring).
# 42 first = parameter-identical to stack v2's "all-in" row -> regression check.
SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404)

# Identical to v2/R7/R8/R9 except random_state, injected per member by
# _lgb_params_for_seed(). LightGBM derives bagging_seed / feature_fraction_seed /
# data_random_seed from `seed` when unset, so this one knob varies all training
# randomness.
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


def _lgb_params_for_seed(seed: int) -> dict[str, object]:
    return {**LGB_PARAMS_BASE, "random_state": seed}


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

    Same recipe as ``run_stack_v2/v21/v22/v23_cv.py`` (not imported -- ``scripts/``
    files are not a shared package here).
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
# alignment primitives + feature builder, byte-copied from run_stack_v23_cv.py
# (which carries them from v22/v21/run_router_v2_cv.py). Only the bank-loading
# helpers (_load_bank_array, add_config_c_columns) are dropped -- R11 never
# reads the pf_bma_v2/beam bank npz files.
# --------------------------------------------------------------------------- #


def block_slices(well_idx: np.ndarray, used_wells: np.ndarray) -> dict[str, tuple[int, int]]:
    """well name -> (start, end) row-slice from a well-block-contiguous ``well_idx``."""
    if well_idx.size and np.any(np.diff(well_idx) < 0):
        raise ValueError("well_idx blocks are not contiguous/sorted")
    n_wells = len(used_wells)
    boundaries = np.searchsorted(well_idx, np.arange(n_wells + 1))
    if boundaries[-1] != well_idx.shape[0]:
        raise ValueError("well_idx blocks are not contiguous/sorted")
    return {str(w): (int(boundaries[i]), int(boundaries[i + 1])) for i, w in enumerate(used_wells)}


def reindex_to_master(
    source_array: np.ndarray,
    source_slices: dict[str, tuple[int, int]],
    master_wells: list[str],
    master_boundaries: np.ndarray,
) -> np.ndarray:
    """Gather ``source_array`` (in its own well-block order) into master row order.

    A well missing from ``source_slices``, or whose source segment length
    disagrees with the master length for that well, is left as ``NaN`` for
    its whole block (never a partial/garbled copy).
    """
    total = int(master_boundaries[-1])
    out = np.full(total, np.nan, dtype=np.float32)
    for i, well in enumerate(master_wells):
        ms, me = int(master_boundaries[i]), int(master_boundaries[i + 1])
        src = source_slices.get(well)
        if src is None:
            continue
        ss, se = src
        seg = source_array[ss:se]
        if seg.shape[0] != (me - ms):
            continue
        out[ms:me] = seg
    return out


class _WellMatrix:
    """Container for the single feature matrix every ensemble member trains on."""

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
    """Two-pass, preallocating build of the v2 56-column feature matrix.

    Byte-copied from ``run_stack_v22_cv.py``/``run_stack_v23_cv.py``'s
    memory-safe re-engineering: pass 1 validates wells + collects row counts,
    pass 2 writes every feature block straight into pre-sized arrays. This is
    the ENTIRE feature set R11 trains on -- no bank augmentation follows.
    """
    counts = {
        "n_no_eval_zone_or_anchor": 0,
        "n_tracker_cache_missing": 0,
        "n_tracker_len_mismatch": 0,
        "n_spatial_cache_missing": 0,
        "n_spatial_len_mismatch": 0,
    }

    # --- pass 1: validation + row counts only (no feature computation) ---
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
            "build_well_matrix_v2: 0 wells passed validation -- no rows to build "
            f"(counts={counts})"
        )

    total_rows = int(sum(row_counts))
    boundaries = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int64)

    y_true = np.empty(total_rows, dtype=np.float64)
    anchor_arr = np.empty(total_rows, dtype=np.float64)
    pf_blend_arr = np.empty(total_rows, dtype=np.float64)
    pf_std_arr = np.empty(total_rows, dtype=np.float64)
    well_idx = np.empty(total_rows, dtype=np.int32)

    p1_cols: list[str] = []
    all_cols: list[str] = []
    col_arrays: dict[str, np.ndarray] = {}

    # --- pass 2: recompute every feature block and write into place ---
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
            f"R11 must train on exactly {N_FEATURES_EXPECTED} (stack v2 'all-in') "
            f"features, got {len(all_cols)}"
        )
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, p1_cols, counts
    )


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Byte-for-byte identical to ``run_stack_v2/v21/v22/v23_cv.py``'s
    ``_split_fit_es``. R11 always calls it with ``ES_SPLIT_SEED`` (42, frozen)
    -- NEVER the member's LGBM seed (matches R9's task spec item 5).
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
    """Train+OOF-score the (b) PF-residual target for one ensemble member.

    Identical protocol to ``run_stack_v22/v23_cv.py``'s ``run_stack_fold_cv``
    with ES-holdout always on (matches v2's "all-in" row), except ``lgb_params``
    is a parameter so each member injects its own ``random_state`` -- the ES
    split itself stays on ``ES_SPLIT_SEED``.
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
        gc.collect()  # reclaim the fold's LightGBM Dataset + X.loc(...) row-slice copies promptly

    assert not np.isnan(oof_pred_tvt).any(), "every row must be scored by exactly one fold"
    return oof_pred_tvt, fold_rows, importances


def _helped_hurt_report(
    label: str,
    y_true: np.ndarray,
    stack_pred: np.ndarray,
    baseline_pred: np.ndarray,
    baseline_label: str,
    well_idx: np.ndarray,
    n_wells: int,
) -> None:
    stack_well_rmse = _per_well_rmse(y_true, stack_pred, well_idx, n_wells)
    base_well_rmse = _per_well_rmse(y_true, baseline_pred, well_idx, n_wells)

    valid = np.isfinite(stack_well_rmse) & np.isfinite(base_well_rmse)
    helped = valid & (stack_well_rmse < base_well_rmse)
    hurt = valid & (stack_well_rmse > base_well_rmse)
    tied = valid & (stack_well_rmse == base_well_rmse)

    print(f"\nhelped/hurt vs {baseline_label} -- ({label}):")
    print(f"  helped: {int(helped.sum())}  hurt: {int(hurt.sum())}  tied: {int(tied.sum())}")
    if helped.any():
        print(
            "  median RMSE improvement (helped wells): "
            f"{float(np.median(base_well_rmse[helped] - stack_well_rmse[helped])):.4f} ft"
        )
    if hurt.any():
        print(
            "  median RMSE regression (hurt wells):    "
            f"{float(np.median(stack_well_rmse[hurt] - base_well_rmse[hurt])):.4f} ft"
        )


def run_one_seed(
    wm: _WellMatrix,
    cols: list[str],
    seed: int,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
) -> dict[str, object]:
    """Run one ensemble member (56-col v2 config, given LGBM seed) to completion."""
    label = f"seed {seed}"
    print(f"\n=== {label} (stack v2 56-col config, n_features={len(cols)}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1..v2.3
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
        _lgb_params_for_seed(seed),
    )
    gc.collect()

    pooled = pooled_rmse(wm.y_true, oof)
    importance_df = (
        pd.DataFrame({"feature": cols, "mean_importance": importances})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "seed": seed,
        "label": label,
        "cols": cols,
        "oof": oof,
        "fold_rows": fold_rows,
        "importance_df": importance_df,
        "pooled_rmse": pooled,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _result_path(seed: int, n_wells: int | None) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    return OUTPUTS_DIR / f"stack_v24_result_s{seed}{suffix}.npz"


def _save_result(
    result: dict[str, object], wm: _WellMatrix, row_fold: np.ndarray, n_wells: int | None
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(result["fold_rows"])
    imp_df = result["importance_df"]
    path = _result_path(int(result["seed"]), n_wells)
    np.savez_compressed(
        path,
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


def _load_result(seed: int, n_wells: int | None) -> dict[str, object]:
    path = _result_path(seed, n_wells)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python scripts/run_stack_v24_cv.py "
            f"--seed-idx {SEEDS.index(seed)}"
            + (f" --n-wells {n_wells}" if n_wells is not None else "")
            + "` first"
        )
    with np.load(path, allow_pickle=True) as npz:
        return {
            "seed": int(npz["seed"]),
            "label": str(npz["label"]),
            "cols": [str(c) for c in npz["cols"]],
            "oof": npz["oof"].astype(np.float64),
            "y_true": npz["y_true"].astype(np.float64),
            "anchor": npz["anchor"].astype(np.float64),
            "pf_blend": npz["pf_blend"].astype(np.float64),
            "well_idx": npz["well_idx"],
            "row_fold": npz["row_fold"],
            "used_wells": [str(w) for w in npz["used_wells"]],
            "pooled_rmse": float(npz["pooled_rmse"]),
            "peak_rss_mb": float(npz["peak_rss_mb"]),
            "fold_id": npz["fold_id"],
            "fold_rmse": npz["fold_rmse"],
            "fold_best_iteration": npz["fold_best_iteration"],
            "importance_feature": [str(f) for f in npz["importance_feature"]],
            "importance_value": npz["importance_value"],
        }


def _aggregate(n_wells: int | None) -> None:
    results: list[dict[str, object]] = []
    for seed in SEEDS:
        try:
            results.append(_load_result(seed, n_wells))
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
    if not results:
        print("no result files found -- nothing to aggregate")
        return

    ref = results[0]
    y_true = ref["y_true"]
    well_idx = ref["well_idx"]
    used_wells = ref["used_wells"]
    anchor = ref["anchor"]
    pf_blend = ref["pf_blend"]
    n_wells_used = len(used_wells)
    for res in results[1:]:
        if not np.allclose(res["y_true"], y_true, atol=Y_TRUE_ATOL):
            raise AssertionError(
                f"seed {res['seed']}'s y_true disagrees with seed {ref['seed']}'s -- these "
                "were not run on the same wells list, aggregation would be meaningless"
            )

    floor_pooled = pooled_rmse(y_true, anchor)
    pf_pooled = pooled_rmse(y_true, pf_blend)

    # per-seed solo + cumulative-mean pooled RMSE (diminishing-returns curve)
    cum_sum = np.zeros_like(y_true, dtype=np.float64)
    print()
    print("=" * 90)
    print(
        f"{'k':>3}{'seed':>7}{'solo pooled':>14}{'cum-mean pooled (1..k)':>24}"
        f"{'vs stack v2 9.2309':>21}"
    )
    print("-" * 90)
    cum_pooled_list: list[float] = []
    for k, res in enumerate(results, start=1):
        cum_sum += res["oof"]
        cum_mean = cum_sum / k
        solo = res["pooled_rmse"]
        cum_pooled = pooled_rmse(y_true, cum_mean)
        cum_pooled_list.append(cum_pooled)
        print(
            f"{k:>3}{res['seed']:>7}{solo:>14.4f}{cum_pooled:>24.4f}"
            f"{cum_pooled - STACK_V2_ALLIN_POOLED_RMSE:>+21.4f}"
        )
    print("=" * 90)
    oof_mean = cum_sum / len(results)
    mean_pooled = cum_pooled_list[-1]

    print(f"\nreference: carry_last={floor_pooled:.4f}  PF blend w0.7={pf_pooled:.4f}")
    print(f"total rows scored: {len(y_true):,}  members aggregated: {len(results)}/{len(SEEDS)}")

    # regression check: seed-42 member == stack v2 "all-in" rerun (full scale only)
    seed42 = next((r for r in results if r["seed"] == 42), None)
    if seed42 is not None:
        diff = seed42["pooled_rmse"] - STACK_V2_ALLIN_POOLED_RMSE
        if n_wells is not None:
            print(
                f"\nseed-42 repro check (n_wells={n_wells} subsample, informational only): "
                f"{seed42['pooled_rmse']:.4f} vs stack v2 all-in ledger 9.2309 (diff {diff:+.4f}ft)"
            )
        else:
            ok = abs(diff) <= REPRO_ATOL
            print(
                f"\nseed-42 repro check: {seed42['pooled_rmse']:.4f} vs stack v2 all-in ledger "
                f"9.2309 (diff {diff:+.4f}ft, tol {REPRO_ATOL}ft) -> "
                f"{'PASS' if ok else 'FAIL -- investigate before trusting the ensemble'}"
            )

    # per-well stats of the final mean
    mean_well_rmse = _per_well_rmse(y_true, oof_mean, well_idx, n_wells_used)
    valid = mean_well_rmse[np.isfinite(mean_well_rmse)]
    print(
        f"\n{len(results)}-seed mean OOF: pooled={mean_pooled:.4f}  "
        f"median={np.median(valid):.4f}  p90={np.percentile(valid, 90):.4f}  "
        f"max={np.max(valid):.4f}"
    )

    if seed42 is not None:
        _helped_hurt_report(
            f"{len(results)}-seed mean",
            y_true,
            oof_mean,
            seed42["oof"],
            f"seed-42 solo ({seed42['pooled_rmse']:.4f})",
            well_idx,
            n_wells_used,
        )

    out_path = OUTPUTS_DIR / "stack_v24_oof.npz"
    save_kwargs = {
        f"oof_{i}": np.asarray(res["oof"], dtype=np.float32) for i, res in enumerate(results)
    }
    np.savez_compressed(
        out_path,
        y_true=y_true.astype(np.float32),
        anchor=anchor.astype(np.float32),
        pf_blend=pf_blend.astype(np.float32),
        oof_mean=oof_mean.astype(np.float32),
        well_idx=well_idx,
        row_fold=ref["row_fold"],
        used_wells=np.array(used_wells),
        seeds=np.array([res["seed"] for res in results], dtype=np.int64),
        pooled_per_seed=np.array([res["pooled_rmse"] for res in results], dtype=np.float64),
        pooled_cumulative=np.array(cum_pooled_list, dtype=np.float64),
        cols=np.array(ref["cols"]),
        **save_kwargs,
    )
    print(f"\nOOF saved: {out_path} (mean of {len(results)} seeds + each member)")

    improvement = STACK_V2_ALLIN_POOLED_RMSE - mean_pooled
    verdict = "採用" if mean_pooled <= ADOPTION_GATE_RMSE else "却下（非改善、実測値は報告のみ）"
    print(f"\n{'=' * 90}")
    print(
        f"判定: {verdict} -- {len(results)}-seed mean={mean_pooled:.4f}, "
        f"stack v2 solo={STACK_V2_ALLIN_POOLED_RMSE:.4f}, "
        f"改善={improvement:+.4f}ft (gate: <= {ADOPTION_GATE_RMSE:.4f}ft)"
    )
    print(f"{'=' * 90}")


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
        "--seed-idx",
        type=int,
        choices=range(len(SEEDS)),
        default=None,
        help=f"Run only SEEDS[i] (SEEDS={SEEDS}) in isolation, then exit -- the "
        "memory-safe path for the full 773-well run (one OS process per member, "
        "strictly serial). Omit to run all members in one process (smoke tests only).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Skip CV entirely; load all outputs/stack_v24_result_s*.npz, print the "
        "per-seed + cumulative-mean table and gate verdict, and write "
        "outputs/stack_v24_oof.npz.",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.aggregate:
        _aggregate(args.n_wells)
        print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
        return

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
        f"feature matrix (v2, 56 cols, R11 -- no bank augmentation): {wm.X.shape}, "
        f"rows={len(wm.y_true):,}, wells_used={len(wm.used_wells)}/{len(wells)} "
        f"counts={wm.counts} ({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )
    cols = list(wm.X.columns)

    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=FOLD_SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr = np.array([well_to_fold[w] for w in wm.used_wells], dtype=np.int32)
    row_fold = well_fold_arr[wm.well_idx]

    seeds_to_run = [SEEDS[args.seed_idx]] if args.seed_idx is not None else list(SEEDS)
    for seed in seeds_to_run:
        result = run_one_seed(wm, cols, seed, row_fold, well_fold_arr)
        print(
            f"  -> seed {seed}: pooled RMSE={result['pooled_rmse']:.4f} "
            f"peak_rss={result['peak_rss_mb']:.0f}MB"
        )
        _save_result(result, wm, row_fold, args.n_wells)
        del result
        gc.collect()

    if args.seed_idx is None:
        # all-seeds-in-one-process path (smoke tests only) -- reuse the same
        # aggregation printer the multi-process path uses, by reloading what
        # was just saved (keeps exactly one aggregation code path).
        _aggregate(args.n_wells)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
    print(f"this process's peak RSS: {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
