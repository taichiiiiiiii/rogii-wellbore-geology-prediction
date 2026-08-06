"""Stack v2.9: R13's 59 features + H4 trend-family extension (issue: R16).

**Why R16.** The 2026-07-11 ledger confirms sub-v8 (= R13's 56+H3=59-feature,
5-seed config, ``run_stack_v26_cv.py``) is the new LB best (9.014, -0.008 vs
stack v2's 9.022) -- the first raw-feature-engineering increment ever
confirmed to transfer to the hidden test set. H3's
``known_linear_trend_extrap_minus_anchor`` (a known-zone tail-200 linear
``TVT_input ~ MD`` trend, extrapolated to each eval row and expressed
relative to the anchor) ranked #2 in R13's importances. R16 digs the same
vein: does the trend family generalize across window length, polynomial
order, and robust-vs-OLS fitting?

**Design.** One superset feature matrix (59 + 5 = 64 cols) is built per
process (R13's 59 columns, byte-copied from ``run_stack_v26_cv.py``, plus
:func:`build_h4_features`'s 5 new columns), then column-sliced per
``--config``:

  - (a) = R13's 59 columns only (**regression check** -- must reproduce
    R13's 5-seed mean 9.1456 +/- 0.02).
  - (b) = (a) + H4 (64 cols).
  - (c) = (b) minus the **bottom-2 mean-importance columns from (b)'s
    5-seed-mean feature importance** (62 cols) -- selected automatically
    from config (b)'s aggregated results and persisted to
    ``outputs/stack_v29_c_cols{suffix}.npz`` (run config b to completion
    before config c).

``SEEDS`` = ``(42, 101, 202, 303, 404)``, identical set/order/fold-seed/
ES-split-seed/LGBM base params to R11-R13/R15 (no hyperparameter changes --
R15 showed the R14 ``min_child_samples=200`` "gain" was seed noise).

**H4 feature block (5 cols), all reading ONLY the known-zone (``TVT_input``
non-NaN prefix) ``TVT_input`` and ``MD`` -- a strict subset of H3's leak
boundary (H4 never touches ``Z``):

  - ``trend_extrap_w50`` / ``trend_extrap_w400``: OLS linear trend of
    ``TVT_input ~ MD`` over the known-zone tail-50 / tail-400 rows (vs. H3's
    tail-200), extrapolated to each eval row's MD relative to the anchor --
    different window lengths capture different drift time constants.
  - ``trend_extrap_quad``: 2nd-order polynomial fit of the known-zone
    tail-400, extrapolated to each eval row's MD relative to the fit's own
    anchor-MD value, clipped to +/-50ft to bound over-extrapolation risk
    from curvature.
  - ``trend_w50_minus_w400``: ``trend_extrap_w50 - trend_extrap_w400``, a
    window-difference (trend acceleration) proxy.
  - ``trend_slope_robust``: Theil-Sen (median-of-pairwise-slopes) robust
    slope of the known-zone tail-200, extrapolated to each eval row's MD
    relative to the anchor -- resistant to a handful of noisy/spiky
    known-TVT rows that would bias H3's OLS slope.

**Gate.** 5-seed mean pooled OOF <= **9.0456** (R13's 9.1456 - 0.10) for the
best of (b)/(c), reported regardless of outcome. If the best result misses
the full gate but still beats R13 by > 0.05ft, it is flagged as a
"sub-v9 candidate (pending LB calibration)" per the task brief.

**Memory.** ~2.6GB target (64 float32 columns over 3,783,989 rows is only
~76MB more than R13's 59-column matrix). Two-pass preallocating build
(byte-copied structure from v25/v26), one seed per OS process, strictly
serial. Per-(config, seed) results checkpoint to
``outputs/stack_v29_result_{config}_s{seed}.npz`` immediately.

**Leak boundary.** H4 reads only ``MD`` and the known-zone (``TVT_input``
non-NaN prefix) ``TVT_input`` -- no ``Z``, no ``h["TVT"]``, no eval-zone
``TVT_input``, no train-only formation columns, no pf_bma_v2/beam_grid bank
npz. H3 (byte-copied from R13) keeps its existing leak boundary (``MD``,
known-zone ``TVT_input``, ``Z``).

Usage::

    # smoke test (all 5 seeds + aggregate in one process, ~30 wells)
    uv run python scripts/run_stack_v29_cv.py --config a --n-wells 30
    uv run python scripts/run_stack_v29_cv.py --config b --n-wells 30
    uv run python scripts/run_stack_v29_cv.py --config c --n-wells 30
    uv run python scripts/run_stack_v29_cv.py --compare --n-wells 30

    # full 773-well run, ONE PROCESS PER (config, SEED), strictly serial
    uv run python scripts/run_stack_v29_cv.py --config a --seed-idx 0
    ...
    uv run python scripts/run_stack_v29_cv.py --config c --seed-idx 4

    # after all 5 members of a config have saved their npz:
    uv run python scripts/run_stack_v29_cv.py --config a --aggregate

    # after all 3 configs are aggregated:
    uv run python scripts/run_stack_v29_cv.py --compare
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
from scipy import stats

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
R13_5SEED_MEAN_POOLED_RMSE = 9.1456  # ledger 2026-07-10 (v26 5-seed mean, sub-v8 LB 9.014)
PF_BLEND_W = 0.7

IMPROVEMENT_GATE_FT = 0.10
ADOPTION_GATE_RMSE = R13_5SEED_MEAN_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.0456
SUBV9_CANDIDATE_GATE_FT = 0.05
SUBV9_CANDIDATE_RMSE = R13_5SEED_MEAN_POOLED_RMSE - SUBV9_CANDIDATE_GATE_FT  # 9.0956
Y_TRUE_ATOL = 1e-2
R13_REPRO_ATOL = 0.02  # config (a) 5-seed mean vs R13's 9.1456

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN for every member (matches v2/R7-R13/R15)
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN for every member (matches v2/R7-R13/R15)
N_STRATA = 10
ES_FRAC = 0.15

# The ONLY thing that varies across ensemble members within one config.
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

N_FEATURES_SUPERSET = 64  # R13's 59 (P1 36 + tracker 8 + well-agg 8 + spatial 4 + H3 3) + H4(5)
CONFIG_KEYS: tuple[str, ...] = ("a", "b", "c")


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
# H3 feature block, byte-copied from run_stack_v26_cv.py (R13). Reads only
# MD, the known-zone TVT_input, and Z. Never reads the train-only ANCC
# column or h["TVT"].
# --------------------------------------------------------------------------- #


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y ~ x``; ``0.0`` when degenerate.

    Byte-copied from ``run_stack_v25/v26_cv.py`` (itself a byte-copy of
    ``rogii.features._robust_slope``).
    """
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

    Byte-copied from ``run_stack_v26_cv.py`` (R13) -- see that module's
    docstring for the full column definitions:

      - known_linear_trend_extrap_minus_anchor: known-zone tail-200
        ``TVT_input ~ MD`` linear trend extrapolated to each eval row's MD,
        relative to the anchor (``slope * (MD[i] - anchor_MD)``).
      - known_tvt_z_identity_resid_std: residual std of a linear fit of
        ``(TVT_input + Z) ~ MD`` over the FULL known zone (leak-safe
        ANCC=TVT+Z identity proxy; never reads the ``ANCC`` column).
      - zone_len_known_len_ratio: ``n_eval_rows / n_known_rows``.

    Never raises; one row per eval-zone row (``data.eval_mask(h)`` order).
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
# H4 feature block (NEW, R16): trend-family extension. Reads only MD and the
# known-zone TVT_input -- a strict subset of H3's leak boundary (no Z).
# --------------------------------------------------------------------------- #

H4_FEATURE_COLUMNS: tuple[str, ...] = (
    "trend_extrap_w50",
    "trend_extrap_w400",
    "trend_extrap_quad",
    "trend_w50_minus_w400",
    "trend_slope_robust",
)

_H4_TAIL_W50 = 50
_H4_TAIL_W400 = 400
_H4_QUAD_TAIL = 400
_H4_QUAD_CLIP_FT = 50.0
_H4_ROBUST_TAIL = 200  # matches H3's known_linear_trend_extrap window


def build_h4_features(h: pd.DataFrame) -> pd.DataFrame:
    """H4: trend-family extension of H3's known-zone linear extrapolation (R16).

    All five columns read only ``MD`` and the known-zone (``TVT_input``
    non-NaN prefix) ``TVT_input`` -- never ``Z``, never ``h["TVT"]``, never
    eval-zone ``TVT_input`` (NaN there by construction), never a train-only
    formation column, never the pf_bma_v2/beam_grid bank npz.

    Columns:
      - trend_extrap_w50 / trend_extrap_w400: OLS linear trend of
        ``TVT_input ~ MD`` fit over the known-zone tail-50 / tail-400 rows
        (vs. H3's tail-200), extrapolated to each eval row's MD and
        expressed relative to the anchor (``slope * (MD[i] - anchor_MD)``) --
        different window lengths capture different drift time constants.
      - trend_extrap_quad: 2nd-order polynomial fit of the known-zone
        tail-400, extrapolated to each eval row's MD, expressed relative to
        the fit's own value at the anchor MD, clipped to
        ``[-_H4_QUAD_CLIP_FT, +_H4_QUAD_CLIP_FT]`` to bound over-extrapolation
        risk from curvature. Falls back to ``trend_extrap_w400`` when the
        tail is too short/degenerate for a stable quadratic fit.
      - trend_w50_minus_w400: ``trend_extrap_w50 - trend_extrap_w400``, a
        window-difference (trend acceleration) proxy that grows with
        distance from the anchor when the short- and long-window trends
        disagree.
      - trend_slope_robust: Theil-Sen (median-of-pairwise-slopes) robust
        slope of the known-zone tail-200 ``TVT_input ~ MD``, extrapolated to
        each eval row's MD relative to the anchor -- resistant to a handful
        of noisy/outlier known-TVT rows that would bias H3's OLS slope.

    Never raises; one row per eval-zone row (``data.eval_mask(h)`` order).
    """
    mask = D.eval_mask(h)
    n_eval = int(mask.sum())
    known_mask = ~mask
    n_known = int(known_mask.sum())
    if n_eval == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in H4_FEATURE_COLUMNS})

    md = h["MD"].to_numpy(dtype=float)
    idx = np.where(mask)[0]

    trend_w50 = np.zeros(n_eval, dtype=float)
    trend_w400 = np.zeros(n_eval, dtype=float)
    trend_quad = np.zeros(n_eval, dtype=float)
    trend_robust = np.zeros(n_eval, dtype=float)

    if n_known > 0:
        known_md = h.loc[known_mask, "MD"].to_numpy(dtype=float)
        known_tvt = h.loc[known_mask, "TVT_input"].to_numpy(dtype=float)
        anchor_md = float(known_md[-1])
        dmd = md[idx] - anchor_md

        n50 = min(_H4_TAIL_W50, n_known)
        slope50 = _robust_slope(known_md[-n50:], known_tvt[-n50:])
        trend_w50 = slope50 * dmd

        n400 = min(_H4_TAIL_W400, n_known)
        slope400 = _robust_slope(known_md[-n400:], known_tvt[-n400:])
        trend_w400 = slope400 * dmd

        nq = min(_H4_QUAD_TAIL, n_known)
        quad_md = known_md[-nq:]
        quad_tvt = known_tvt[-nq:]
        finite_q = np.isfinite(quad_md) & np.isfinite(quad_tvt)
        if int(finite_q.sum()) >= 3 and np.std(quad_md[finite_q]) > 1e-9:
            coeffs = np.polyfit(quad_md[finite_q], quad_tvt[finite_q], 2)
            fitted_anchor = np.polyval(coeffs, anchor_md)
            fitted_eval = np.polyval(coeffs, md[idx])
            trend_quad = fitted_eval - fitted_anchor
            trend_quad = np.clip(trend_quad, -_H4_QUAD_CLIP_FT, _H4_QUAD_CLIP_FT)
        else:
            trend_quad = trend_w400.copy()

        nr = min(_H4_ROBUST_TAIL, n_known)
        rob_md = known_md[-nr:]
        rob_tvt = known_tvt[-nr:]
        finite_r = np.isfinite(rob_md) & np.isfinite(rob_tvt)
        slope_robust = 0.0
        if int(finite_r.sum()) >= 2 and np.std(rob_md[finite_r]) > 1e-9:
            try:
                slope_robust = float(stats.theilslopes(rob_tvt[finite_r], rob_md[finite_r])[0])
            except (ValueError, FloatingPointError):
                slope_robust = 0.0
        trend_robust = slope_robust * dmd

    trend_w50 = np.where(np.isfinite(trend_w50), trend_w50, 0.0)
    trend_w400 = np.where(np.isfinite(trend_w400), trend_w400, 0.0)
    trend_quad = np.where(np.isfinite(trend_quad), trend_quad, 0.0)
    trend_robust = np.where(np.isfinite(trend_robust), trend_robust, 0.0)
    trend_diff = trend_w50 - trend_w400

    out = pd.DataFrame(
        {
            "trend_extrap_w50": trend_w50,
            "trend_extrap_w400": trend_w400,
            "trend_extrap_quad": trend_quad,
            "trend_w50_minus_w400": trend_diff,
            "trend_slope_robust": trend_robust,
        }
    )
    out = out[list(H4_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# superset well-matrix builder (R13's 59 cols + H4 = 64)
# --------------------------------------------------------------------------- #


class _WellMatrix:
    """Container for the 64-column superset feature matrix every config slices."""

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


def build_well_matrix_v29(wells: list[str], split: str = "train") -> _WellMatrix:
    """Two-pass, preallocating build of the 64-column superset (R13's 59 + H4).

    Byte-copied validation/pass structure from ``run_stack_v25/v26_cv.py``.
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
            f"build_well_matrix_v29: 0 wells passed validation -- no rows to build "
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
        h3_feats = build_h3_features(h)
        h4_feats = build_h4_features(h)

        if not col_arrays:
            all_cols = (
                list(p1_feats.columns)
                + list(trk_feats.columns)
                + list(well_agg_feats.columns)
                + list(spatial_feats.columns)
                + list(h3_feats.columns)
                + list(h4_feats.columns)
            )
            col_arrays = {c: np.empty(total_rows, dtype=np.float32) for c in all_cols}

        for block in (p1_feats, trk_feats, well_agg_feats, spatial_feats, h3_feats, h4_feats):
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
    if len(all_cols) != N_FEATURES_SUPERSET:
        raise AssertionError(
            f"R16 superset must have {N_FEATURES_SUPERSET} cols, got {len(all_cols)}"
        )
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, counts
    )


def _c_cols_path(n_wells: int | None) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    return OUTPUTS_DIR / f"stack_v29_c_cols{suffix}.npz"


def _load_config_c_cols(n_wells: int | None) -> list[str]:
    path = _c_cols_path(n_wells)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run config b to completion first "
            "(`--config b --seed-idx 0..4` then `--config b --aggregate`, or "
            "`--config b --n-wells N` for a smoke test): config c's 62-column "
            "selection is derived from config b's 5-seed-mean feature importance"
        )
    with np.load(path, allow_pickle=True) as npz:
        return [str(c) for c in npz["cols"]]


def _config_columns(wm: _WellMatrix, config: str, n_wells: int | None) -> list[str]:
    """Column subset for ``config`` in {a, b, c} -- see module docstring."""
    p1_end = 59  # R13's 59: P1(36) + tracker(8) + well-agg(8) + spatial(4) + H3(3)
    h4_cols = set(H4_FEATURE_COLUMNS)
    base59 = [c for c in wm.X.columns if c not in h4_cols]
    if len(base59) != p1_end:
        raise AssertionError(f"expected {p1_end} base (R13) columns, got {len(base59)}")

    if config == "a":
        return base59
    if config == "b":
        return base59 + list(H4_FEATURE_COLUMNS)
    if config == "c":
        return _load_config_c_cols(n_wells)
    raise ValueError(f"unknown config {config!r}, expected one of {CONFIG_KEYS}")


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
    """Train+OOF-score the (b) PF-residual target for one ensemble member.

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
    config: str,
    cols: list[str],
    seed: int,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
) -> dict[str, object]:
    """Run one ensemble member (config's column subset, given LGBM seed) to completion."""
    label = f"config {config} seed {seed}"
    print(f"\n=== {label} (n_features={len(cols)}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1..v2.8
    oof, fold_rows, importances = run_stack_fold_cv(
        label,
        wm.X[cols],
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
        "config": config,
        "seed": seed,
        "label": label,
        "cols": cols,
        "oof": oof,
        "fold_rows": fold_rows,
        "importance_df": importance_df,
        "pooled_rmse": pooled,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _result_path(config: str, seed: int, n_wells: int | None) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    return OUTPUTS_DIR / f"stack_v29_result_{config}_s{seed}{suffix}.npz"


def _save_result(
    result: dict[str, object], wm: _WellMatrix, row_fold: np.ndarray, n_wells: int | None
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(result["fold_rows"])
    imp_df = result["importance_df"]
    path = _result_path(str(result["config"]), int(result["seed"]), n_wells)
    np.savez_compressed(
        path,
        config=str(result["config"]),
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


def _load_result(config: str, seed: int, n_wells: int | None) -> dict[str, object]:
    path = _result_path(config, seed, n_wells)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python scripts/run_stack_v29_cv.py "
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
            "fold_id": npz["fold_id"],
            "fold_rmse": npz["fold_rmse"],
            "fold_best_iteration": npz["fold_best_iteration"],
            "importance_feature": [str(f) for f in npz["importance_feature"]],
            "importance_value": npz["importance_value"],
        }


def _mean_importance_across_seeds(results: list[dict[str, object]]) -> pd.DataFrame:
    """Mean (across seeds) of each seed's already-5-fold-mean feature importance.

    Sorted ascending by ``mean_importance`` so ``.head(2)`` gives config c's
    drop set.
    """
    frames = [
        pd.DataFrame({"feature": res["importance_feature"], "importance": res["importance_value"]})
        for res in results
    ]
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("feature", as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": "mean_importance"})
        .sort_values("mean_importance", ascending=True)
        .reset_index(drop=True)
    )


def _write_config_c_selection(
    results: list[dict[str, object]], ref_cols: list[str], n_wells: int | None
) -> None:
    """From config (b)'s 5-seed-mean importance, drop the bottom-2 columns; persist for (c)."""
    imp_mean = _mean_importance_across_seeds(results)
    bottom2 = imp_mean.head(2)
    drop_set = set(bottom2["feature"].tolist())
    c_cols = [c for c in ref_cols if c not in drop_set]
    if len(c_cols) != len(ref_cols) - 2:
        raise AssertionError(
            f"expected exactly 2 columns dropped for config c, got "
            f"{len(ref_cols) - len(c_cols)} (bottom2={bottom2['feature'].tolist()})"
        )
    c_path = _c_cols_path(n_wells)
    np.savez_compressed(c_path, cols=np.array(c_cols))
    print("\nconfig c column selection (config b's 5-seed-mean importance, bottom-2 dropped):")
    print(imp_mean.head(6).to_string(index=False))
    print(f"config c cols saved: {c_path} ({len(c_cols)} columns)")


def _aggregate(config: str, n_wells: int | None) -> float | None:
    """Aggregate one config's 5 seed results; returns the 5-seed mean pooled RMSE (or None)."""
    results: list[dict[str, object]] = []
    for seed in SEEDS:
        try:
            results.append(_load_result(config, seed, n_wells))
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
    if not results:
        print(f"no result files found for config {config} -- nothing to aggregate")
        return None

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
                f"config {config} seed {res['seed']}'s y_true disagrees with seed "
                f"{ref['seed']}'s -- these were not run on the same wells list"
            )

    floor_pooled = pooled_rmse(y_true, anchor)
    pf_pooled = pooled_rmse(y_true, pf_blend)

    cum_sum = np.zeros_like(y_true, dtype=np.float64)
    print()
    print("=" * 96)
    print(f"config ({config}), n_features={len(ref['cols'])}")
    print(
        f"{'k':>3}{'seed':>7}{'solo pooled':>14}{'cum-mean pooled (1..k)':>24}"
        f"{'vs R13 9.1456':>16}"
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
            f"{cum_pooled - R13_5SEED_MEAN_POOLED_RMSE:>+16.4f}"
        )
    print("=" * 96)
    oof_mean = cum_sum / len(results)
    mean_pooled = cum_pooled_list[-1]

    print(f"\nreference: carry_last={floor_pooled:.4f}  PF blend w0.7={pf_pooled:.4f}")
    print(f"total rows scored: {len(y_true):,}  members aggregated: {len(results)}/{len(SEEDS)}")

    mean_well_rmse = _per_well_rmse(y_true, oof_mean, well_idx, n_wells_used)
    valid = mean_well_rmse[np.isfinite(mean_well_rmse)]
    print(
        f"\n{len(results)}-seed mean OOF (config {config}): pooled={mean_pooled:.4f}  "
        f"median={np.median(valid):.4f}  p90={np.percentile(valid, 90):.4f}  "
        f"max={np.max(valid):.4f}"
    )

    seed42 = next((r for r in results if r["seed"] == 42), None)
    if seed42 is not None:
        _helped_hurt_report(
            f"config {config} {len(results)}-seed mean",
            y_true,
            oof_mean,
            seed42["oof"],
            f"config {config} seed-42 solo ({seed42['pooled_rmse']:.4f})",
            well_idx,
            n_wells_used,
        )

    out_path = OUTPUTS_DIR / f"stack_v29_oof_{config}.npz"
    save_kwargs = {
        f"oof_{i}": np.asarray(res["oof"], dtype=np.float32) for i, res in enumerate(results)
    }
    np.savez_compressed(
        out_path,
        config=config,
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

    if config == "a" and n_wells is None and len(results) == len(SEEDS):
        diff_r13 = mean_pooled - R13_5SEED_MEAN_POOLED_RMSE
        ok = abs(diff_r13) <= R13_REPRO_ATOL
        print(
            f"\nR13 REGRESSION CHECK (config a, 5-seed mean): {mean_pooled:.4f} vs R13 ledger "
            f"9.1456 (diff {diff_r13:+.4f}ft, tol {R13_REPRO_ATOL}ft) -> "
            f"{'PASS' if ok else 'FAIL -- investigate before trusting configs b/c'}"
        )
    elif config == "a" and n_wells is not None:
        print(
            f"\n(n_wells={n_wells} subsample -- R13 regression check is informational-only "
            "and skipped; the full 773-well run gates on it)"
        )

    if config == "b" and len(results) == len(SEEDS):
        _write_config_c_selection(results, list(ref["cols"]), n_wells)

    improvement = R13_5SEED_MEAN_POOLED_RMSE - mean_pooled
    if mean_pooled <= ADOPTION_GATE_RMSE:
        verdict = "採用候補"
    elif mean_pooled <= SUBV9_CANDIDATE_RMSE:
        verdict = "sub-v9候補（LB較正込みで判断）"
    else:
        verdict = "却下（非改善、実測値は報告のみ）"
    print(f"\n{'=' * 96}")
    print(
        f"判定 (config {config}): {verdict} -- 5-seed mean={mean_pooled:.4f}, "
        f"R13 baseline={R13_5SEED_MEAN_POOLED_RMSE:.4f}, "
        f"改善={improvement:+.4f}ft (gate: <= {ADOPTION_GATE_RMSE:.4f}ft, "
        f"sub-v9候補: <= {SUBV9_CANDIDATE_RMSE:.4f}ft)"
    )
    print(f"{'=' * 96}")
    return mean_pooled


def _compare_all(n_wells: int | None) -> None:
    """Load all 3 configs' saved OOF npz and print the final R16 ablation table."""
    print("\n" + "=" * 96)
    print("R16 ablation: a=R13 59-feat repro, b=+H4(64), c=b minus bottom-2 importance(62)")
    print("=" * 96)
    print(
        f"{'config':>8}{'n_features':>12}{'5-seed mean pooled':>22}"
        f"{'vs R13 9.1456':>16}{'gate<=9.0456':>14}{'sub-v9<=9.0956':>16}"
    )
    print("-" * 96)
    results: dict[str, float] = {}
    for config in CONFIG_KEYS:
        path = OUTPUTS_DIR / f"stack_v29_oof_{config}.npz"
        if not path.exists():
            print(f"{config:>8}  (missing: {path})")
            continue
        with np.load(path, allow_pickle=True) as npz:
            y_true = npz["y_true"].astype(np.float64)
            oof_mean = npz["oof_mean"].astype(np.float64)
            n_feat = len(npz["cols"])
        pooled = pooled_rmse(y_true, oof_mean)
        results[config] = pooled
        gate_pass = "PASS" if pooled <= ADOPTION_GATE_RMSE else "fail"
        subv9_pass = "yes" if pooled <= SUBV9_CANDIDATE_RMSE else "no"
        print(
            f"{config:>8}{n_feat:>12}{pooled:>22.4f}"
            f"{pooled - R13_5SEED_MEAN_POOLED_RMSE:>+16.4f}{gate_pass:>14}{subv9_pass:>16}"
        )
    print("=" * 96)
    bc_results = {k: v for k, v in results.items() if k in ("b", "c")}
    if bc_results:
        best_config = min(bc_results, key=lambda k: bc_results[k])
        best_pooled = bc_results[best_config]
        if best_pooled <= ADOPTION_GATE_RMSE:
            tag = "PASS gate -- 採用候補"
        elif best_pooled <= SUBV9_CANDIDATE_RMSE:
            tag = "sub-v9候補（LB較正込みで判断）"
        else:
            tag = "却下（非改善）"
        print(f"\nbest of (b, c): config {best_config} = {best_pooled:.4f} -- {tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=CONFIG_KEYS,
        default=None,
        help="Feature-set config: a=59 (R13 repro), b=+H4(64), c=b minus bottom-2 importance(62).",
    )
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
        help="Skip CV; load outputs/stack_v29_result_{config}_s*.npz, print the table, "
        "write outputs/stack_v29_oof_{config}.npz.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Skip CV; load all 3 configs' outputs/stack_v29_oof_{a,b,c}.npz and print "
        "the final ablation table.",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.compare:
        _compare_all(args.n_wells)
        print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
        return

    if args.config is None:
        parser.error("--config is required unless --compare is given")

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
    wm = build_well_matrix_v29(wells, split="train")
    gc.collect()
    print(
        f"feature matrix (superset, {wm.X.shape[1]} cols): {wm.X.shape}, "
        f"rows={len(wm.y_true):,}, wells_used={len(wm.used_wells)}/{len(wells)} "
        f"counts={wm.counts} ({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )
    cols = _config_columns(wm, args.config, args.n_wells)
    print(f"config {args.config}: {len(cols)} columns")

    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=FOLD_SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr = np.array([well_to_fold[w] for w in wm.used_wells], dtype=np.int32)
    row_fold = well_fold_arr[wm.well_idx]

    seeds_to_run = [SEEDS[args.seed_idx]] if args.seed_idx is not None else list(SEEDS)
    for seed in seeds_to_run:
        result = run_one_seed(wm, args.config, cols, seed, row_fold, well_fold_arr)
        print(
            f"  -> config {args.config} seed {seed}: pooled RMSE={result['pooled_rmse']:.4f} "
            f"peak_rss={result['peak_rss_mb']:.0f}MB"
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
