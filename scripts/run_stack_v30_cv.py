"""Stack v3.0: hard-well-specialized R13 variants (issue: R18).

**Why R18.** Analysis A (2026-07-11) found the top-20 wells by SSE (2.6% of
773) carry 26% of total SSE, and that hard wells are dominated by "mid-zone
mis-track -> constant offset" failures. Well-level diagnostics alone give
weak hard-well *identification* (past AUC ~0.69), but the router path
(picking among whole candidate predictors) has failed 3x (ledger 2026-07-09
R3-v1/v2/v3). This script tests a *different* lever: keep the single LGBM
family from R13 (``run_stack_v26_cv.py``, 56 + H3 = 59 features, 5-seed),
but make the loss itself hard-well-aware, in two variants:

  - **w1 / w3** (variant 1, cheapest): reweight training rows by their own
    well's carry_last RMSE. ``sample_weight = 1 + alpha * normalize(well_rmse)``,
    alpha in {1.0, 3.0}. Only the LGBM ``sample_weight`` argument changes --
    same features, same targets, same params as R13.
  - **2stage** (variant 2): a well-level difficulty regressor (diagnostic
    features -> predicted per-well carry_last RMSE) flags the riskiest 20%
    of a fold's score-side wells; those rows are predicted by a *second*
    LGBM trained with 3x sample weight on the fold's own top-20%-hardest
    train wells, while every other row keeps the normal (R13-identical,
    unweighted) model's prediction.

**Leak boundary (both variants).** The per-well "difficulty" signal used for
weighting/labeling a training well is that well's OWN true-label carry_last
RMSE -- always computed only over wells that are *entirely* on the training
side of the current well-level CV fold (``train_wells_all``, i.e. the wells
NOT in ``score_mask``'s fold). This is not a leak: LGBM already trains on
those wells' true ``y_true`` every run (R11..R13); this script only reweights
*which rows of already-visible training labels* the loss emphasizes. No
score-fold well's true difficulty is ever read -- the two-stage variant's
risk score for score-side wells comes from a regressor fit on train-side
wells' diagnostics -> true RMSE, then *applied* (not fit) to score-side
wells' diagnostics only (pre-inference-safe features, byte-copied from
``scripts/run_router_cv.py``'s ``ROUTER_FEATURE_COLUMNS`` machinery).

**Gate.** seed-42 solo pooled OOF <= R13 seed-42 solo (9.2515) - 0.10 =
9.1515. If a variant misses this, it is rejected WITHOUT running the other 4
seeds (playbook: report negative results honestly, no sunk-cost 5-seeding).
Separately (regardless of gate outcome), the RMSE of the top-20 hard wells
(by SSE, ranked from ``outputs/stack_v26_result_s42.npz``'s OOF -- the R13
seed-42 baseline) is reported, since a variant could fail the pooled gate
while still meaningfully fixing the hard-well subset.

Usage::

    # smoke test (~30 wells, one config, one seed, single process)
    uv run python scripts/run_stack_v30_cv.py --config w1 --n-wells 30 --seed-idx 0

    # seed-42-only gate check (full 773 wells)
    uv run python scripts/run_stack_v30_cv.py --config w1 --seed-idx 0
    uv run python scripts/run_stack_v30_cv.py --config w3 --seed-idx 0
    uv run python scripts/run_stack_v30_cv.py --config 2stage --seed-idx 0

    # only if a config clears the gate: remaining seeds + aggregate
    uv run python scripts/run_stack_v30_cv.py --config w1 --seed-idx 1
    ...
    uv run python scripts/run_stack_v30_cv.py --config w1 --seed-idx 4
    uv run python scripts/run_stack_v30_cv.py --config w1 --aggregate
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
STACK_V2_ALLIN_POOLED_RMSE = 9.2309  # ledger 2026-07-07 (LB 9.022)
R13_5SEED_MEAN_POOLED_RMSE = 9.1456  # ledger 2026-07-10 (run_stack_v26_cv.py, R13 5-seed mean)
R13_SEED42_POOLED_RMSE = 9.251504247013594  # outputs/stack_v26_result_s42.npz, exact readback
PF_BLEND_W = 0.7

IMPROVEMENT_GATE_FT = 0.10
SEED42_GATE_RMSE = R13_SEED42_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.1515
ADOPTION_GATE_5SEED_RMSE = R13_5SEED_MEAN_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.0456
Y_TRUE_ATOL = 1e-2

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN, matches v2/R7-R13
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN, matches v2/R7-R13
N_STRATA = 10
ES_FRAC = 0.15

SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404)

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

N_FEATURES_EXPECTED = 59  # P1(36) + tracker(8) + well-agg(8) + spatial(4) + H3(3), same as R13

CONFIGS: tuple[str, ...] = ("w1", "w3", "2stage")
REWEIGHT_ALPHA: dict[str, float] = {"w1": 1.0, "w3": 3.0}
TWOSTAGE_RISK_FRAC = 0.20  # top 20% of a fold's wells (by risk / true RMSE) are "difficult"
TWOSTAGE_SPECIALIZED_WEIGHT = 3.0  # sample_weight for difficult-well rows in the specialized model
HARD20_N = 20  # matches analysis A's "SSE top-20 wells (2.6%) -> 26% of total SSE"


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

    Same recipe as ``run_stack_v24/.../v26_cv.py``.
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
# H3 feature block, byte-copied from run_stack_v25/v26_cv.py (R12/R13). Reads
# only MD, the known-zone TVT_input, and Z. Never reads the train-only ANCC
# column or h["TVT"].
# --------------------------------------------------------------------------- #


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y ~ x``; ``0.0`` when degenerate. Byte-copied from R13."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    xs, ys = x[mask], y[mask]
    if np.std(xs) < 1e-9:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


H3_FEATURE_COLUMNS: tuple[str, ...] = (
    "known_linear_trend_extrap_minus_anchor",
    "known_tvt_z_identity_resid_std",
    "zone_len_known_len_ratio",
)

_H3_TREND_TAIL = 200  # matches rogii.features._KNOWN_TAIL_LONG


def build_h3_features(h: pd.DataFrame) -> pd.DataFrame:
    """H3: 3 columns from ``MD``, the known-zone ``TVT_input``, and ``Z`` only.

    Byte-copied from ``run_stack_v26_cv.py`` (R13).
    """
    mask = D.eval_mask(h)
    n_eval = int(mask.sum())
    known_mask = ~mask
    n_known = int(known_mask.sum())
    if n_eval == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in H3_FEATURE_COLUMNS})

    md = h["MD"].to_numpy(dtype=float)
    idx = np.where(mask)[0]

    trend_delta = np.zeros(n_eval, dtype=float)
    identity_resid_std = 0.0

    if n_known > 0:
        known_md = h.loc[known_mask, "MD"].to_numpy(dtype=float)
        known_tvt = h.loc[known_mask, "TVT_input"].to_numpy(dtype=float)
        anchor_md = float(known_md[-1])

        tail_n = min(_H3_TREND_TAIL, n_known)
        tail_md = known_md[-tail_n:]
        tail_tvt = known_tvt[-tail_n:]
        slope = _robust_slope(tail_md, tail_tvt)
        trend_delta = slope * (md[idx] - anchor_md)

        z = h["Z"].to_numpy(dtype=float)
        known_z = z[known_mask]
        identity_val = known_tvt + known_z
        finite = np.isfinite(known_md) & np.isfinite(identity_val)
        if int(finite.sum()) >= 2 and np.std(known_md[finite]) > 1e-9:
            coeffs = np.polyfit(known_md[finite], identity_val[finite], 1)
            fitted = np.polyval(coeffs, known_md[finite])
            identity_resid_std = float(np.std(identity_val[finite] - fitted))

    ratio = float(n_eval) / float(n_known) if n_known > 0 else 0.0

    trend_delta = np.where(np.isfinite(trend_delta), trend_delta, 0.0)
    identity_resid_std = identity_resid_std if np.isfinite(identity_resid_std) else 0.0
    ratio = ratio if np.isfinite(ratio) else 0.0

    out = pd.DataFrame(
        {
            "known_linear_trend_extrap_minus_anchor": trend_delta,
            "known_tvt_z_identity_resid_std": np.full(n_eval, identity_resid_std, dtype=np.float64),
            "zone_len_known_len_ratio": np.full(n_eval, ratio, dtype=np.float64),
        }
    )
    out = out[list(H3_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# well-level diagnostic features (variant 2 stage-1 detector input). Byte-
# copied semantics from run_router_cv.py's ROUTER_FEATURE_COLUMNS / helpers,
# but assembled inline (no extra I/O -- reuses arrays already loaded by the
# main matrix-build loop) and NEVER broadcast to rows: one row per well.
# --------------------------------------------------------------------------- #

WELL_DIAG_COLUMNS: tuple[str, ...] = (
    "pf_d_final",
    "pf_d_mean",
    "pf_std_well_mean",
    "pf_std_well_max",
    "beam_margin_well_mean",
    "pf_beam_abs_diff_well_mean",
    "beam_d_final",
    "eval_len",
    "gr_nan_frac",
    "spatial_d_final",
    "spatial_prefix_rmse",
    "spatial_nn_dist_median",
    "spatial_gated_d_final",
    "z_span",
    "prefix_slope",
    "disagree_std",
)

_PREFIX_SLOPE_K = 50  # matches run_router_cv.py's PREFIX_SLOPE_K


def _prefix_slope(h: pd.DataFrame, k: int = _PREFIX_SLOPE_K) -> float:
    """Local linear slope of ``TVT_input`` vs ``MD`` over the last ``k`` known rows.

    Byte-copied from ``run_router_cv.py``. Known-zone only -- no leakage.
    """
    known = h["TVT_input"].to_numpy(dtype=np.float64)
    md = h["MD"].to_numpy(dtype=np.float64)
    valid_idx = np.where(np.isfinite(known))[0]
    if valid_idx.size < 2:
        return 0.0
    tail = valid_idx[-k:] if valid_idx.size > k else valid_idx
    x, y = md[tail], known[tail]
    if np.ptp(x) == 0:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(slope) if np.isfinite(slope) else 0.0


def _z_span(h: pd.DataFrame, mask: np.ndarray) -> float:
    """Z (TVD) range across the eval zone -- a geometric survey column, always known.

    Byte-copied from ``run_router_cv.py``.
    """
    if "Z" not in h.columns:
        return 0.0
    z = h["Z"].to_numpy(dtype=np.float64)[mask]
    z = z[np.isfinite(z)]
    return float(np.max(z) - np.min(z)) if z.size else 0.0


def _well_diag_row(
    h: pd.DataFrame,
    mask: np.ndarray,
    anchor: float,
    pf_tvt: np.ndarray,
    pf_std: np.ndarray,
    beam_tvt: np.ndarray,
    beam_margin: np.ndarray,
    gr_nan_frac: float,
    spatial_tvt: np.ndarray,
    prefix_rmse: float,
    nn_dist_median: float,
) -> dict[str, float]:
    """One well's diagnostic scalar row (``WELL_DIAG_COLUMNS``), pre-inference safe."""
    well_agg = build_well_aggregate_features(
        anchor, pf_tvt, pf_std, beam_tvt, beam_margin, gr_nan_frac
    )
    spatial_feats = build_spatial_features(anchor, spatial_tvt, prefix_rmse, nn_dist_median)

    anchor_f = float(anchor) if np.isfinite(anchor) else 0.0
    beam_d = np.asarray(beam_tvt, dtype=np.float64) - anchor_f
    beam_d_final = float(beam_d[-1]) if beam_d.size and np.isfinite(beam_d[-1]) else 0.0
    pf_d_final = float(well_agg["pf_d_final"].iloc[-1])
    spatial_d_final = float(spatial_feats["spatial_d"].iloc[-1])
    disagree_std = float(np.std([pf_d_final, beam_d_final, spatial_d_final]))

    return {
        "pf_d_final": pf_d_final,
        "pf_d_mean": float(well_agg["pf_d_mean"].iloc[-1]),
        "pf_std_well_mean": float(well_agg["pf_std_well_mean"].iloc[-1]),
        "pf_std_well_max": float(well_agg["pf_std_well_max"].iloc[-1]),
        "beam_margin_well_mean": float(well_agg["beam_margin_well_mean"].iloc[-1]),
        "pf_beam_abs_diff_well_mean": float(well_agg["pf_beam_abs_diff_well_mean"].iloc[-1]),
        "beam_d_final": beam_d_final,
        "eval_len": float(well_agg["eval_len"].iloc[-1]),
        "gr_nan_frac": float(gr_nan_frac) if np.isfinite(gr_nan_frac) else 0.0,
        "spatial_d_final": spatial_d_final,
        "spatial_prefix_rmse": float(spatial_feats["spatial_prefix_rmse"].iloc[-1]),
        "spatial_nn_dist_median": float(spatial_feats["spatial_nn_dist_median"].iloc[-1]),
        "spatial_gated_d_final": float(spatial_feats["spatial_gated_d"].iloc[-1]),
        "z_span": _z_span(h, mask),
        "prefix_slope": _prefix_slope(h),
        "disagree_std": disagree_std,
    }


# --------------------------------------------------------------------------- #
# well-matrix builder (56 v2 cols + H3 = 59), structure byte-copied from
# run_stack_v26_cv.py, extended with a per-well diagnostic table (no extra
# I/O -- reuses the same tracker/spatial arrays already loaded per well).
# --------------------------------------------------------------------------- #


class _WellMatrix:
    """Container for the 59-column feature matrix + per-well diagnostics."""

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
        well_diag: pd.DataFrame,
    ) -> None:
        self.X = X
        self.y_true = y_true
        self.anchor_arr = anchor_arr
        self.pf_blend_arr = pf_blend_arr
        self.pf_std_arr = pf_std_arr
        self.well_idx = well_idx
        self.used_wells = used_wells
        self.counts = counts
        self.well_diag = well_diag  # index 0..n_wells-1 aligned to used_wells


def build_well_matrix_v30(wells: list[str], split: str = "train") -> _WellMatrix:
    """Two-pass, preallocating build of the 59-column matrix (v2's 56 + H3) + well_diag.

    Byte-copied validation/pass structure from ``run_stack_v26_cv.py``.
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
            f"build_well_matrix_v30: 0 wells passed validation -- no rows to build "
            f"(counts={counts})"
        )

    total_rows = int(sum(row_counts))
    boundaries = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int64)
    n_wells = len(used_wells)

    y_true = np.empty(total_rows, dtype=np.float64)
    anchor_arr = np.empty(total_rows, dtype=np.float64)
    pf_blend_arr = np.empty(total_rows, dtype=np.float64)
    pf_std_arr = np.empty(total_rows, dtype=np.float64)
    well_idx = np.empty(total_rows, dtype=np.int32)

    all_cols: list[str] = []
    col_arrays: dict[str, np.ndarray] = {}
    diag_rows: list[dict[str, float]] = [{} for _ in range(n_wells)]

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
        h3_feats = build_h3_features(h)

        if not col_arrays:
            all_cols = (
                list(p1_feats.columns)
                + list(trk_feats.columns)
                + list(well_agg_feats.columns)
                + list(spatial_feats.columns)
                + list(h3_feats.columns)
            )
            col_arrays = {c: np.empty(total_rows, dtype=np.float32) for c in all_cols}

        for block in (p1_feats, trk_feats, well_agg_feats, spatial_feats, h3_feats):
            for c in block.columns:
                col_arrays[c][s:e] = block[c].to_numpy(dtype=np.float32, copy=False)

        y_true_well = h["TVT"].to_numpy(dtype=np.float32)[mask]  # label only, never a feature
        y_true[s:e] = y_true_well.astype(np.float64)
        anchor_arr[s:e] = cached_anchor
        pf_blend_arr[s:e] = cached_anchor + PF_BLEND_W * (pf_tvt - cached_anchor)
        pf_std_arr[s:e] = pf_std
        well_idx[s:e] = well_i

        diag_rows[well_i] = _well_diag_row(
            h,
            mask,
            cached_anchor,
            pf_tvt,
            pf_std,
            beam_tvt,
            beam_margin,
            gr_nan_frac,
            spatial_tvt,
            prefix_rmse,
            nn_dist_median,
        )

    X = pd.DataFrame(col_arrays)
    assert list(X.columns) == all_cols, "dict-insertion column order must match all_cols"
    if len(all_cols) != N_FEATURES_EXPECTED:
        raise AssertionError(
            f"R18 must train on exactly {N_FEATURES_EXPECTED} (56 + H3) features, "
            f"got {len(all_cols)}"
        )
    well_diag = pd.DataFrame(diag_rows, index=used_wells)[list(WELL_DIAG_COLUMNS)]
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, counts, well_diag
    )


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Byte-for-byte identical to ``run_stack_v26_cv.py``'s ``_split_fit_es``.
    """
    ordered = sorted(train_wells)
    random.Random(seed).shuffle(ordered)
    n_es = max(1, round(len(ordered) * es_frac))
    es_wells = sorted(ordered[:n_es])
    fit_wells = sorted(ordered[n_es:])
    return fit_wells, es_wells


# --------------------------------------------------------------------------- #
# variant 1: sample-weight reweighting by own well's carry_last RMSE
# --------------------------------------------------------------------------- #


def _well_weight_array(
    train_wells_all: list[str],
    well_carry_rmse: np.ndarray,
    well_to_idx: dict[str, int],
    alpha: float,
) -> dict[str, float]:
    """``weight = 1 + alpha * minmax_normalize(well's own carry_last RMSE)``.

    Normalization is over ``train_wells_all`` (this fold's full train side,
    fit + ES-holdout wells) ONLY -- never over score-fold wells. Degenerate
    (constant or all-non-finite) RMSE distributions fall back to weight 1.0
    everywhere (no reweighting rather than a division by ~0).
    """
    vals = np.array([well_carry_rmse[well_to_idx[w]] for w in train_wells_all], dtype=np.float64)
    finite = np.isfinite(vals)
    norm = np.zeros_like(vals)
    if int(finite.sum()) >= 2 and np.ptp(vals[finite]) > 1e-9:
        lo, hi = float(vals[finite].min()), float(vals[finite].max())
        norm[finite] = (vals[finite] - lo) / (hi - lo)
    weight = 1.0 + alpha * norm
    return {w: float(wt) for w, wt in zip(train_wells_all, weight, strict=True)}


def run_stack_fold_cv_reweighted(
    label: str,
    X: pd.DataFrame,
    target: np.ndarray,
    base_arr: np.ndarray,
    y_true: np.ndarray,
    row_fold: np.ndarray,
    well_idx: np.ndarray,
    well_fold_arr: np.ndarray,
    used_wells: list[str],
    well_carry_rmse: np.ndarray,
    alpha: float,
    n_splits: int,
    es_frac: float,
    lgb_params: dict[str, object],
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    """Variant 1: identical to R13's ``run_stack_fold_cv`` except ``sample_weight``
    on the fit rows only (ES holdout + score fold stay unweighted, matching R13's
    eval_set/metric exactly so early stopping is not itself biased toward hard wells).
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

        weight_map = _well_weight_array(train_wells_all, well_carry_rmse, used_well_to_idx, alpha)
        sample_weight = np.array(
            [weight_map[used_wells[wi]] for wi in well_idx[train_mask]], dtype=np.float64
        )

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X.loc[train_mask],
            target[train_mask],
            sample_weight=sample_weight,
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
                "weight_min": float(sample_weight.min()),
                "weight_max": float(sample_weight.max()),
            }
        )
        print(
            f"  [{label}] fold {fold_i}: n_train={int(train_mask.sum()):,} "
            f"n_score={int(score_mask.sum()):,} best_iter={model.best_iteration_} "
            f"rmse={fold_rmse:.4f} w=[{sample_weight.min():.2f},{sample_weight.max():.2f}] "
            f"({time.time() - t_fold:.1f}s)",
            flush=True,
        )
        del model, eval_set, train_mask, score_mask
        gc.collect()

    assert not np.isnan(oof_pred_tvt).any(), "every row must be scored by exactly one fold"
    return oof_pred_tvt, fold_rows, importances


# --------------------------------------------------------------------------- #
# variant 2: well-level difficulty detector + specialized model for the
# riskiest 20% of a fold's score-side wells
# --------------------------------------------------------------------------- #


def _fit_difficulty_regressor(X_diag: pd.DataFrame, y_diag: np.ndarray, seed: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=200, max_depth=5, random_state=seed, n_jobs=-1
                ),
            ),
        ]
    )
    pipe.fit(X_diag, y_diag)
    return pipe


def run_stack_fold_cv_twostage(
    label: str,
    X: pd.DataFrame,
    target: np.ndarray,
    base_arr: np.ndarray,
    y_true: np.ndarray,
    row_fold: np.ndarray,
    well_idx: np.ndarray,
    well_fold_arr: np.ndarray,
    used_wells: list[str],
    well_carry_rmse: np.ndarray,
    well_diag: pd.DataFrame,
    n_splits: int,
    es_frac: float,
    lgb_params: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray, dict[str, object]]:
    """Variant 2. Per fold:

      1. Rank ``train_wells_all`` by their OWN true carry_last RMSE; top 20%
         are "difficult train wells" (train-side true label, leak-safe).
      2. Fit a well-level difficulty regressor (``well_diag`` -> true
         carry_last RMSE) on ``train_wells_all`` only.
      3. Apply the regressor to the fold's SCORE-side wells' ``well_diag``
         (no true label read) -> predicted risk; top 20% of THIS fold's
         score wells are routed to the specialized model.
      4. Train a "normal" model (unweighted, byte-identical procedure to
         R13) and a "specialized" model (same features/target, but
         sample_weight=3.0 on difficult-train-well rows, 1.0 elsewhere).
      5. Score-fold rows: difficult-flagged wells get the specialized
         model's prediction, everyone else gets the normal model's.
    """
    feature_cols = list(X.columns)
    oof_pred_tvt = np.full(len(y_true), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importances = np.zeros(len(feature_cols), dtype=np.float64)
    used_well_to_idx = {w: i for i, w in enumerate(used_wells)}
    diag_diagnostics: dict[str, object] = {"fold_n_flagged": [], "fold_r2": []}

    for fold_i in range(n_splits):
        t_fold = time.time()
        score_mask = row_fold == fold_i

        train_well_positions = np.where(well_fold_arr != fold_i)[0]
        score_well_positions = np.where(well_fold_arr == fold_i)[0]
        train_wells_all = [used_wells[wi] for wi in train_well_positions]
        score_wells_all = [used_wells[wi] for wi in score_well_positions]
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

        # ---- 1. difficult TRAIN wells (true label, train side -- leak-safe) ----
        train_rmse_vals = np.array(
            [well_carry_rmse[used_well_to_idx[w]] for w in train_wells_all], dtype=np.float64
        )
        train_finite = np.isfinite(train_rmse_vals)
        if int(train_finite.sum()) >= 5:
            thresh_train = float(np.percentile(train_rmse_vals[train_finite], 80))
        else:
            thresh_train = float("inf")
        difficult_train_wells = {
            w
            for w, v in zip(train_wells_all, train_rmse_vals, strict=True)
            if np.isfinite(v) and v >= thresh_train
        }

        # ---- 2. fit difficulty regressor on train_wells_all diagnostics ----
        X_diag_train = well_diag.loc[train_wells_all]
        reg = _fit_difficulty_regressor(X_diag_train, train_rmse_vals, seed)
        train_pred = reg.predict(X_diag_train)
        finite_pair = train_finite
        if int(finite_pair.sum()) >= 2 and np.var(train_rmse_vals[finite_pair]) > 1e-9:
            ss_res = float(np.sum((train_rmse_vals[finite_pair] - train_pred[finite_pair]) ** 2))
            ss_tot = float(
                np.sum(
                    (train_rmse_vals[finite_pair] - np.mean(train_rmse_vals[finite_pair])) ** 2
                )
            )
            fold_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")
        else:
            fold_r2 = float("nan")

        # ---- 3. predicted risk for SCORE-side wells (no true label read) ----
        X_diag_score = well_diag.loc[score_wells_all]
        risk_score = reg.predict(X_diag_score)
        risk_finite = np.isfinite(risk_score)
        if int(risk_finite.sum()) >= 5:
            thresh_score = float(np.percentile(risk_score[risk_finite], 80))
        else:
            thresh_score = float("inf")
        difficult_score_wells = {
            w
            for w, v in zip(score_wells_all, risk_score, strict=True)
            if np.isfinite(v) and v >= thresh_score
        }

        # ---- 4a. normal model (byte-identical procedure to R13) ----
        normal_model = lgb.LGBMRegressor(**lgb_params)
        normal_model.fit(
            X.loc[train_mask],
            target[train_mask],
            eval_set=eval_set,
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        pred_normal_all = normal_model.predict(X.loc[score_mask])
        importances += normal_model.feature_importances_.astype(np.float64) / n_splits
        del normal_model
        gc.collect()

        # ---- 4b. specialized model (3x weight on difficult-train-well rows) ----
        spec_weight = np.where(
            np.isin(well_idx[train_mask], [used_well_to_idx[w] for w in difficult_train_wells]),
            TWOSTAGE_SPECIALIZED_WEIGHT,
            1.0,
        ).astype(np.float64)
        spec_model = lgb.LGBMRegressor(**lgb_params)
        spec_model.fit(
            X.loc[train_mask],
            target[train_mask],
            sample_weight=spec_weight,
            eval_set=eval_set,
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        pred_spec_all = spec_model.predict(X.loc[score_mask])
        del spec_model
        gc.collect()

        # ---- 5. route: difficult-flagged score wells -> specialized, else normal ----
        score_well_idx_arr = well_idx[score_mask]
        difficult_score_idx = {used_well_to_idx[w] for w in difficult_score_wells}
        use_specialized = np.isin(score_well_idx_arr, list(difficult_score_idx))
        pred_target = np.where(use_specialized, pred_spec_all, pred_normal_all)
        pred_tvt = base_arr[score_mask] + pred_target
        oof_pred_tvt[score_mask] = pred_tvt

        fold_rmse = pooled_rmse(y_true[score_mask], pred_tvt)
        diag_diagnostics["fold_n_flagged"].append(len(difficult_score_wells))
        diag_diagnostics["fold_r2"].append(fold_r2)

        fold_rows.append(
            {
                "config": label,
                "fold": fold_i,
                "n_train_rows": int(train_mask.sum()),
                "n_score_rows": int(score_mask.sum()),
                "best_iteration": -1,  # two models this fold; not a single scalar
                "rmse": fold_rmse,
                "seconds": time.time() - t_fold,
                "n_difficult_train_wells": len(difficult_train_wells),
                "n_difficult_score_wells": len(difficult_score_wells),
                "diag_r2": fold_r2,
            }
        )
        print(
            f"  [{label}] fold {fold_i}: n_train={int(train_mask.sum()):,} "
            f"n_score={int(score_mask.sum()):,} rmse={fold_rmse:.4f} "
            f"difficult_train={len(difficult_train_wells)}/{len(train_wells_all)} "
            f"difficult_score={len(difficult_score_wells)}/{len(score_wells_all)} "
            f"diag_R2={fold_r2:.3f} ({time.time() - t_fold:.1f}s)",
            flush=True,
        )
        del eval_set, train_mask, score_mask
        gc.collect()

    assert not np.isnan(oof_pred_tvt).any(), "every row must be scored by exactly one fold"
    return oof_pred_tvt, fold_rows, importances, diag_diagnostics


# --------------------------------------------------------------------------- #
# hard-20-well reference (from the R13 seed-42 baseline OOF)
# --------------------------------------------------------------------------- #


def _load_r13_seed42_reference() -> dict[str, object]:
    path = OUTPUTS_DIR / "stack_v26_result_s42.npz"
    with np.load(path, allow_pickle=True) as npz:
        return {
            "used_wells": [str(w) for w in npz["used_wells"]],
            "well_idx": npz["well_idx"],
            "y_true": npz["y_true"].astype(np.float64),
            "oof": npz["oof"].astype(np.float64),
            "pooled_rmse": float(npz["pooled_rmse"]),
        }


def _hard20_well_names() -> list[str]:
    """Top-20 wells by SSE, ranked from the R13 seed-42 baseline OOF (analysis A's
    "top-20 wells (2.6%) carry 26% of SSE" ranking source)."""
    ref = _load_r13_seed42_reference()
    n_wells = len(ref["used_wells"])
    well_idx = ref["well_idx"]
    y_true = ref["y_true"]
    oof = ref["oof"]
    sse = np.zeros(n_wells, dtype=np.float64)
    for w_i in range(n_wells):
        rows = well_idx == w_i
        if rows.any():
            sse[w_i] = float(np.sum((y_true[rows] - oof[rows]) ** 2))
    order = np.argsort(-sse)[:HARD20_N]
    return [ref["used_wells"][i] for i in order]


def _report_hard20(
    label: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    well_idx: np.ndarray,
    used_wells: list[str],
) -> None:
    """Print RMSE of the R13-baseline hard-20 wells under this run's prediction."""
    try:
        ref = _load_r13_seed42_reference()
    except FileNotFoundError:
        print(
            f"\n({label}) hard-20-well comparison skipped: "
            f"{OUTPUTS_DIR}/stack_v26_result_s42.npz not found"
        )
        return
    hard_names = _hard20_well_names()
    well_to_i = {w: i for i, w in enumerate(used_wells)}
    ref_well_to_i = {w: i for i, w in enumerate(ref["used_wells"])}

    rows = []
    for w in hard_names:
        if w not in well_to_i or w not in ref_well_to_i:
            continue
        i_new = well_to_i[w]
        i_ref = ref_well_to_i[w]
        rows_new = well_idx == i_new
        rows_ref = ref["well_idx"] == i_ref
        if not rows_new.any() or not rows_ref.any():
            continue
        rmse_new = pooled_rmse(y_true[rows_new], pred[rows_new])
        rmse_ref = pooled_rmse(ref["y_true"][rows_ref], ref["oof"][rows_ref])
        rows.append(
            {
                "well": w,
                "r13_seed42_rmse": rmse_ref,
                f"{label}_rmse": rmse_new,
                "delta": rmse_new - rmse_ref,
            }
        )

    if not rows:
        print(f"\n({label}) hard-20-well comparison: no overlapping wells found")
        return
    df = pd.DataFrame(rows)
    pooled_hard_ref = float(np.sqrt(np.mean(df["r13_seed42_rmse"] ** 2)))
    pooled_hard_new = float(np.sqrt(np.mean(df[f"{label}_rmse"] ** 2)))
    print(
        f"\n--- hard-20-well RMSE ({label} vs R13 seed-42 baseline, "
        f"{len(df)}/{HARD20_N} matched) ---"
    )
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"  well-mean-RMSE-of-hard20: R13={pooled_hard_ref:.4f}  {label}={pooled_hard_new:.4f}  "
        f"delta={pooled_hard_new - pooled_hard_ref:+.4f}  "
        f"(median delta={float(df['delta'].median()):+.4f}, helped={int((df['delta'] < 0).sum())}/"
        f"{len(df)})"
    )


# --------------------------------------------------------------------------- #
# per-seed run + save/load/aggregate
# --------------------------------------------------------------------------- #


def run_one_seed(
    config: str,
    wm: _WellMatrix,
    cols: list[str],
    seed: int,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
    well_carry_rmse: np.ndarray,
) -> dict[str, object]:
    """Run one ensemble member (given config, given LGBM seed) to completion."""
    label = f"{config} seed {seed}"
    print(f"\n=== {label} (R18 {config} config, n_features={len(cols)}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from R7..R13
    diag_diagnostics: dict[str, object] | None = None

    if config in ("w1", "w3"):
        alpha = REWEIGHT_ALPHA[config]
        oof, fold_rows, importances = run_stack_fold_cv_reweighted(
            label,
            wm.X,
            y_r,
            wm.pf_blend_arr,
            wm.y_true,
            row_fold,
            wm.well_idx,
            well_fold_arr,
            wm.used_wells,
            well_carry_rmse,
            alpha,
            N_SPLITS,
            ES_FRAC,
            _lgb_params_for_seed(seed),
        )
    elif config == "2stage":
        oof, fold_rows, importances, diag_diagnostics = run_stack_fold_cv_twostage(
            label,
            wm.X,
            y_r,
            wm.pf_blend_arr,
            wm.y_true,
            row_fold,
            wm.well_idx,
            well_fold_arr,
            wm.used_wells,
            well_carry_rmse,
            wm.well_diag,
            N_SPLITS,
            ES_FRAC,
            _lgb_params_for_seed(seed),
            seed,
        )
    else:
        raise ValueError(f"unknown config {config!r}")
    gc.collect()

    pooled = pooled_rmse(wm.y_true, oof)
    importance_df = (
        pd.DataFrame({"feature": cols, "mean_importance": importances})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    _report_hard20(label, wm.y_true, oof, wm.well_idx, wm.used_wells)
    return {
        "config": config,
        "seed": seed,
        "label": label,
        "cols": cols,
        "oof": oof,
        "fold_rows": fold_rows,
        "importance_df": importance_df,
        "pooled_rmse": pooled,
        "peak_rss_mb": _peak_rss_mb(),
        "diag_diagnostics": diag_diagnostics,
    }


def _result_path(config: str, seed: int, n_wells: int | None) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    return OUTPUTS_DIR / f"stack_v30_result_{config}_s{seed}{suffix}.npz"


def _save_result(
    result: dict[str, object], wm: _WellMatrix, row_fold: np.ndarray, n_wells: int | None
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(result["fold_rows"])
    imp_df = result["importance_df"]
    path = _result_path(str(result["config"]), int(result["seed"]), n_wells)
    np.savez_compressed(
        path,
        config=result["config"],
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
        fold_n_train=fold_df["n_train_rows"].to_numpy(),
        fold_n_score=fold_df["n_score_rows"].to_numpy(),
        fold_seconds=fold_df["seconds"].to_numpy(),
        importance_feature=imp_df["feature"].to_numpy(),
        importance_value=imp_df["mean_importance"].to_numpy(),
    )
    print(f"\nresult saved: {path} (peak RSS this process: {result['peak_rss_mb']:.0f} MB)")


def _load_result(config: str, seed: int, n_wells: int | None) -> dict[str, object]:
    path = _result_path(config, seed, n_wells)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python scripts/run_stack_v30_cv.py "
            f"--config {config} --seed-idx {SEEDS.index(seed)}"
            + (f" --n-wells {n_wells}" if n_wells is not None else "")
            + "` first"
        )
    with np.load(path, allow_pickle=True) as npz:
        return {
            "config": str(npz["config"]),
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
        }


def _aggregate(config: str, n_wells: int | None) -> None:
    results: list[dict[str, object]] = []
    for seed in SEEDS:
        try:
            results.append(_load_result(config, seed, n_wells))
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

    cum_sum = np.zeros_like(y_true, dtype=np.float64)
    print()
    print("=" * 96)
    print(f"R18 config={config} ({len(ref['cols'])} features)")
    print(
        f"{'k':>3}{'seed':>7}{'solo pooled':>14}{'cum-mean pooled (1..k)':>24}"
        f"{'vs R13 5s 9.1456':>18}"
    )
    print("-" * 96)
    cum_pooled_list: list[float] = []
    for k, res in enumerate(results, start=1):
        cum_sum += res["oof"]
        cum_mean = cum_sum / k
        solo = res["pooled_rmse"]
        cum_pooled = pooled_rmse(y_true, cum_mean)
        cum_pooled_list.append(cum_pooled)
        print(
            f"{k:>3}{res['seed']:>7}{solo:>14.4f}{cum_pooled:>24.4f}"
            f"{cum_pooled - R13_5SEED_MEAN_POOLED_RMSE:>+18.4f}"
        )
    print("=" * 96)
    oof_mean = cum_sum / len(results)
    mean_pooled = cum_pooled_list[-1]

    print(f"\nreference: carry_last={floor_pooled:.4f}  PF blend w0.7={pf_pooled:.4f}")
    print(f"total rows scored: {len(y_true):,}  members aggregated: {len(results)}/{len(SEEDS)}")

    mean_well_rmse = _per_well_rmse(y_true, oof_mean, well_idx, n_wells_used)
    valid = mean_well_rmse[np.isfinite(mean_well_rmse)]
    print(
        f"\n{len(results)}-seed mean OOF (R18 {config}): pooled={mean_pooled:.4f}  "
        f"median={np.median(valid):.4f}  p90={np.percentile(valid, 90):.4f}  "
        f"max={np.max(valid):.4f}"
    )

    _report_hard20(f"R18-{config}-5seed-mean", y_true, oof_mean, well_idx, used_wells)

    out_path = OUTPUTS_DIR / f"stack_v30_oof_{config}.npz"
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
        used_wells=np.array(used_wells),
        seeds=np.array([res["seed"] for res in results], dtype=np.int64),
        pooled_per_seed=np.array([res["pooled_rmse"] for res in results], dtype=np.float64),
        pooled_cumulative=np.array(cum_pooled_list, dtype=np.float64),
        cols=np.array(ref["cols"]),
        **save_kwargs,
    )
    print(f"\nOOF saved: {out_path} (mean of {len(results)} seeds + each member)")

    improvement_5seed = R13_5SEED_MEAN_POOLED_RMSE - mean_pooled
    if mean_pooled <= ADOPTION_GATE_5SEED_RMSE:
        verdict = "採用候補"
    else:
        verdict = "却下（非改善、実測値は報告のみ）"
    print(f"\n{'=' * 96}")
    print(
        f"判定 (R18 {config}, 5-seed): {verdict} -- 5-seed mean={mean_pooled:.4f}, "
        f"R13 5-seed baseline={R13_5SEED_MEAN_POOLED_RMSE:.4f} (改善 {improvement_5seed:+.4f}ft), "
        f"gate <= {ADOPTION_GATE_5SEED_RMSE:.4f}ft"
    )
    print(f"{'=' * 96}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, required=True, help="Which R18 variant.")
    parser.add_argument("--n-wells", type=int, default=None, help="Health-check subset size.")
    parser.add_argument(
        "--seed-idx",
        type=int,
        choices=range(len(SEEDS)),
        default=None,
        help=f"Run only SEEDS[i] (SEEDS={SEEDS}) in isolation, then exit.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Skip CV; load outputs/stack_v30_result_{config}_s*.npz, print the table, "
        "write outputs/stack_v30_oof_{config}.npz.",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.aggregate:
        _aggregate(args.config, args.n_wells)
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
    wm = build_well_matrix_v30(wells, split="train")
    gc.collect()
    print(
        f"feature matrix (R18, 56+H3 = {wm.X.shape[1]} cols): {wm.X.shape}, "
        f"rows={len(wm.y_true):,}, wells_used={len(wm.used_wells)}/{len(wells)} "
        f"counts={wm.counts} ({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )
    cols = list(wm.X.columns)

    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=FOLD_SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr = np.array([well_to_fold[w] for w in wm.used_wells], dtype=np.int32)
    row_fold = well_fold_arr[wm.well_idx]

    n_wells_used = len(wm.used_wells)
    well_carry_rmse = _per_well_rmse(wm.y_true, wm.anchor_arr, wm.well_idx, n_wells_used)

    seeds_to_run = [SEEDS[args.seed_idx]] if args.seed_idx is not None else list(SEEDS)
    for seed in seeds_to_run:
        result = run_one_seed(
            args.config, wm, cols, seed, row_fold, well_fold_arr, well_carry_rmse
        )
        print(
            f"  -> {args.config} seed {seed}: pooled RMSE={result['pooled_rmse']:.4f} "
            f"peak_rss={result['peak_rss_mb']:.0f}MB"
        )
        if args.seed_idx is not None and seed == 42 and args.n_wells is None:
            gate_verdict = "PASS" if result["pooled_rmse"] <= SEED42_GATE_RMSE else "FAIL"
            gain = R13_SEED42_POOLED_RMSE - result["pooled_rmse"]
            print(
                f"\n{'=' * 96}\n"
                f"seed-42 gate ({args.config}): {gate_verdict} -- "
                f"pooled={result['pooled_rmse']:.4f}, R13 seed-42={R13_SEED42_POOLED_RMSE:.4f}, "
                f"gate <= {SEED42_GATE_RMSE:.4f} (改善 {gain:+.4f}ft)\n"
                f"{'=' * 96}"
            )
        _save_result(result, wm, row_fold, args.n_wells)
        del result
        gc.collect()

    if args.seed_idx is None:
        _aggregate(args.config, args.n_wells)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
    print(f"this process's peak RSS: {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
