"""Stack v3.2: LGBM objective (loss function) sweep on R13's 59 features (issue: R20).

**Why R20.** R14 (``run_stack_v27_cv.py``) swept ``num_leaves`` /
``min_child_samples`` / ``learning_rate`` / ``feature_fraction`` around
LGBM's regression defaults and found the current point is ~locally optimal
(only ``min_child_samples=200`` looked promising at -0.028, and R15 proved
that was single-seed noise). One axis R14 never touched: the **objective**
itself. Every stack run since v2 has trained with LightGBM's default
``objective="regression"`` (plain L2 / squared error). Per-well error is
extremely heavy-tailed (top-20 wells = 26% of total SSE, ledger analysis A),
so the L2 gradient on those rows dominates training and may be pulling the
model toward fitting noisy outlier rows at the expense of the bulk. A
robust loss (Huber / Fair) down-weights large-residual gradients and could
either help (less outlier-chasing -> better bulk fit -> lower pooled RMSE,
which is itself an L2-flavoured metric so the trade-off is not obvious) or
hurt (explicitly told not to fit the tail as hard, when pooled RMSE
rewards fitting the tail). This must be settled by measurement, not
argued from priors.

**Design.** Byte-copies ``run_stack_v26_cv.py``'s (R13, current best) full
harness verbatim: the 59-column feature matrix (P1 56 cols + H3 3 cols),
fold assignment (``cv.well_folds(wells, 5, seed=42)``), ES-holdout split
(``_split_fit_es(..., seed=42)``), early stopping 50 rounds on ``rmse``
(the *evaluation* metric, independent of the training *objective* -- so
even huber/fair runs are early-stopped on the actual pooled-RMSE-relevant
signal, not on their own loss), and PF-residual target
(``y_r = TVT - pf_blend``). The **only** thing that varies across configs
is ``objective`` (+ its shape parameter):

  ======  ==========================================  =================
  key     lgbm params                                  rationale
  ======  ==========================================  =================
  base    objective="regression" (l2, unchanged)        regression guard: must reproduce
                                                          R13 seed42 solo = 9.2515
  h1      objective="huber", alpha=10.0                  ~p75 |residual| (9.76) -- mild
                                                          robustness, most rows stay quadratic
  h2      objective="huber", alpha=17.0                  ~p90 |residual| (17.25) -- only the
                                                          worst decile gets linear treatment
  fair    objective="fair",  fair_c=10.0                 same scale as h1's alpha; fair's
                                                          transition is smooth (no huber kink)
  ======  ==========================================  =================

**Target-residual distribution (measured, not assumed).** Computed from
``outputs/stack_v26_result_s42.npz`` (R13's own seed-42 run: 773 wells,
3,783,989 eval rows, ``y_r = TVT - pf_blend``, i.e. exactly the training
target every config in this sweep regresses on -- huber/fair alpha only
change how *this same target's* residuals are weighted during training,
so this file's own distribution is the correct scale reference):

  std=11.02  mean|.|=7.37  p50|.|=4.70  p75|.|=9.76  p80|.|=11.41
  p85|.|=13.75  p90|.|=17.25  p95|.|=23.86  p99|.|=41.01  max|.|=81.94

Right-skew is severe (mean|.| 7.37 vs p50|.| 4.70 -- driven by the same
heavy per-well tail as the top-level pooled-RMSE story). h1's alpha=10 sits
right at p75 (residuals below this stay quadratic -- roughly 3/4 of rows);
h2's alpha=17 sits at p90 (only the worst decile switches to linear
gradient). fair_c=10 matches h1's scale with a softer transition. This
sweep only tests two huber points + one fair point (not a wide grid) per
the ≤3-iteration tuning budget in the playbook -- if a direction shows
signal, a follow-up can refine around it.

**Gate.** Screening is single-seed (42) only, matching R14's convention.
Trigger for 5-seed confirmation (a *separate* task, not run here): best
config's seed-42 pooled <= (base seed-42 pooled - 0.05). R15's lesson
applies without exception -- a seed-42-only "improvement" is NOT an
adoption decision; only a 5-seed mean vs the R13 5-seed-mean gate
(<= 9.1456 - 0.10 = 9.0456) can adopt.

**Leak boundary.** Byte-identical to R13 (``run_stack_v26_cv.py``): the
feature matrix reads no ``h["TVT"]``, no eval-zone ``TVT_input``, no
train-only formation columns. Only the loss function changes; the feature
set, fold structure, and target definition are untouched.

Usage::

    # smoke test (one config end-to-end, ~30 wells)
    uv run python scripts/run_stack_v32_cv.py --config base --seed-idx 0 --n-wells 30
    uv run python scripts/run_stack_v32_cv.py --config h1   --seed-idx 0 --n-wells 30
    uv run python scripts/run_stack_v32_cv.py --config h2   --seed-idx 0 --n-wells 30
    uv run python scripts/run_stack_v32_cv.py --config fair --seed-idx 0 --n-wells 30

    # full 773-well seed-42 screen, ONE PROCESS PER CONFIG, strictly serial
    uv run python scripts/run_stack_v32_cv.py --config base --seed-idx 0
    uv run python scripts/run_stack_v32_cv.py --config h1   --seed-idx 0
    uv run python scripts/run_stack_v32_cv.py --config h2   --seed-idx 0
    uv run python scripts/run_stack_v32_cv.py --config fair --seed-idx 0

    # screen table + regression guard + CONFIRM_TRIGGER/CONFIRM_KEY markers
    uv run python scripts/run_stack_v32_cv.py --aggregate

    # (follow-up task, not run here) 5-seed confirmation of the winning config
    uv run python scripts/run_stack_v32_cv.py --confirm --config <key> --seed-idx 1
    ...
    uv run python scripts/run_stack_v32_cv.py --confirm --config <key> --seed-idx 4
    uv run python scripts/run_stack_v32_cv.py --aggregate-confirm --config <key>
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
R13_SEED42_POOLED_RMSE = 9.2515  # ledger 2026-07-10/11 (run_stack_v26_cv.py seed-42 solo)
R13_5SEED_MEAN_POOLED_RMSE = 9.1456  # ledger 2026-07-10/11 (v26 5-seed mean, current best)
PF_BLEND_W = 0.7

BASE_REPRO_ATOL = 0.001  # base config (l2) must reproduce R13 seed42 9.2515 within this
SCREEN_IMPROVEMENT_GATE_FT = 0.05  # seed42 screen trigger: best <= base - 0.05 -> 5-seed confirm
ADOPTION_IMPROVEMENT_GATE_FT = 0.10  # 5-seed adoption gate (separate task): <= R13 5-seed - 0.10
Y_TRUE_ATOL = 1e-2

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN (matches v2/R7-R13)
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN (matches v2/R7-R13)
N_STRATA = 10
ES_FRAC = 0.15

SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404)
SCREEN_SEED = 42  # every sweep config's screening member uses this seed

# Measured target-residual distribution (see module docstring). abs(y_r) over
# all 773 wells / 3,783,989 eval rows, y_r = TVT - pf_blend (the training
# target every config here regresses on), sourced from R13's own seed-42 run.
RESID_ABS_STD = 11.0177
RESID_ABS_P75 = 9.7607
RESID_ABS_P90 = 17.25
ALPHA_H1 = 10.0  # ~p75 |residual|
ALPHA_H2 = 17.0  # ~p90 |residual|
FAIR_C = 10.0  # same scale as h1's alpha

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

N_FEATURES_EXPECTED = 59  # P1(36) + tracker(8) + well-agg(8) + spatial(4) + H3(3), R13's set

# Objective sweep. "base" overrides nothing (stays l2 / objective="regression").
# huber/fair override "objective" plus their own shape parameter.
SWEEP: tuple[tuple[str, dict[str, object]], ...] = (
    ("base", {}),
    ("h1", {"objective": "huber", "alpha": ALPHA_H1}),
    ("h2", {"objective": "huber", "alpha": ALPHA_H2}),
    ("fair", {"objective": "fair", "fair_c": FAIR_C}),
)
SWEEP_KEYS: tuple[str, ...] = tuple(k for k, _ in SWEEP)


def _sweep_overrides(key: str) -> dict[str, object]:
    for k, overrides in SWEEP:
        if k == key:
            return overrides
    raise ValueError(f"unknown config key {key!r}, expected one of {SWEEP_KEYS}")


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

    Same recipe as ``run_stack_v2/.../v27_cv.py``.
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
# H3 feature block, byte-copied from run_stack_v26_cv.py (R13). Reads only MD,
# the known-zone TVT_input, and Z. Never reads the train-only ANCC column or
# h["TVT"].
# --------------------------------------------------------------------------- #


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y ~ x``; ``0.0`` when degenerate.

    Byte-copied from ``run_stack_v26_cv.py`` (itself a byte-copy of
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
# well-matrix builder (56 v2 cols + H3 = 59), byte-copied from
# run_stack_v26_cv.py's two-pass preallocating build. Identical feature set
# for every config in this sweep -- only the LGBM objective varies.
# --------------------------------------------------------------------------- #


class _WellMatrix:
    """Container for the 59-column feature matrix every config trains on."""

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


def build_well_matrix_v32(wells: list[str], split: str = "train") -> _WellMatrix:
    """Two-pass, preallocating build of the 59-column matrix (R13's 56 + H3).

    Byte-copied from ``run_stack_v26_cv.py``. R20 sweeps only the LGBM
    objective, so this feature matrix is identical for every config.
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
            f"build_well_matrix_v32: 0 wells passed validation -- no rows to build "
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

    X = pd.DataFrame(col_arrays)
    assert list(X.columns) == all_cols, "dict-insertion column order must match all_cols"
    if len(all_cols) != N_FEATURES_EXPECTED:
        raise AssertionError(
            f"R20 must train on exactly {N_FEATURES_EXPECTED} (R13's 56 + H3) features, "
            f"got {len(all_cols)}"
        )
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, counts
    )


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Byte-for-byte identical to ``run_stack_v26_cv.py``'s ``_split_fit_es``.
    Always called with ``ES_SPLIT_SEED`` (42, frozen).
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

    Identical protocol to ``run_stack_v26_cv.py``'s ``run_stack_fold_cv``
    with ES-holdout always on. ``eval_metric="rmse"`` is fixed regardless of
    ``lgb_params["objective"]`` -- early stopping always tracks the actual
    pooled-RMSE-relevant signal, not the training loss shape.
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


def _print_target_residual_stats(wm: _WellMatrix) -> None:
    """Print the actual (this-run) target-residual distribution, for the report.

    ``y_r = TVT - pf_blend`` is the training target every config regresses
    on; huber/fair alpha choices are calibrated against this distribution
    (see module docstring). Printed every full run so the report always
    cites a real, run-specific measurement rather than a stale constant.
    """
    resid = wm.y_true - wm.pf_blend_arr
    abs_resid = np.abs(resid)
    percentiles = [50, 75, 80, 85, 90, 95, 99]
    pvals = np.percentile(abs_resid, percentiles)
    pstr = "  ".join(f"p{p}={v:.2f}" for p, v in zip(percentiles, pvals, strict=True))
    print(
        f"\ntarget residual |y_r|=|TVT-pf_blend| over {len(resid):,} rows: "
        f"std={np.std(resid):.4f}  mean|.|={np.mean(abs_resid):.4f}  {pstr}  "
        f"max|.|={np.max(abs_resid):.2f}",
        flush=True,
    )
    print(
        f"chosen: h1 alpha={ALPHA_H1} (~p75)  h2 alpha={ALPHA_H2} (~p90)  "
        f"fair fair_c={FAIR_C} (~p75 scale)",
        flush=True,
    )


def run_one_member(
    wm: _WellMatrix,
    key: str,
    seed: int,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
) -> dict[str, object]:
    """Run one (config, seed) member to completion."""
    params = _lgb_params_for(key, seed)
    overrides = _sweep_overrides(key)
    label = f"{key} seed {seed}"
    print(f"\n=== {label} (59 cols, overrides={overrides or 'none (base, l2)'}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1..v2.6/R13
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
    return OUTPUTS_DIR / f"stack_v32_result_{key}_s{seed}{suffix}.npz"


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


def _aggregate(n_wells: int | None) -> None:
    """Seed-42 screen table + regression guard + CONFIRM_TRIGGER/CONFIRM_KEY markers."""
    rows: list[tuple[str, float, float]] = []
    base_pooled: float | None = None
    for key in SWEEP_KEYS:
        try:
            res = _load_result(key, SCREEN_SEED, n_wells)
        except FileNotFoundError:
            print(f"WARNING: config {key!r} has no seed-{SCREEN_SEED} result npz -- skipped")
            continue
        mean_best_iter = float(np.mean(res["fold_best_iteration"]))
        rows.append((key, res["pooled_rmse"], mean_best_iter))
        if key == "base":
            base_pooled = res["pooled_rmse"]

    if not rows:
        print("no results found -- nothing to compare")
        return
    if base_pooled is None:
        print("ERROR: base config result missing -- cannot compute the trigger gate")
        print("CONFIRM_TRIGGER: NO")
        return

    print("\n" + "=" * 96)
    print(f"R20 LGBM objective sweep (59 features, seed {SCREEN_SEED}, 5-fold pooled OOF)")
    print("=" * 96)
    print(f"{'key':>8}{'overrides':>40}{'pooled OOF':>14}{'vs base':>12}{'mean best_iter':>16}")
    print("-" * 96)
    for key, pooled, mean_best_iter in rows:
        overrides = _sweep_overrides(key)
        ov_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(base, l2)"
        print(
            f"{key:>8}{ov_str:>40}{pooled:>14.4f}{pooled - base_pooled:>+12.4f}"
            f"{mean_best_iter:>16.1f}"
        )
    print("=" * 96)

    if n_wells is None:
        diff = base_pooled - R13_SEED42_POOLED_RMSE
        ok = abs(diff) <= BASE_REPRO_ATOL
        print(
            f"\nbase repro check (regression guard): {base_pooled:.4f} vs R13 seed42 ledger "
            f"9.2515 (diff {diff:+.4f}ft, tol {BASE_REPRO_ATOL}ft) -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            print(
                "WARNING: base config did not reproduce R13 seed42 within tolerance -- "
                "investigate before trusting the huber/fair deltas below."
            )

    best_key, best_pooled, _ = min(rows, key=lambda r: r[1])
    gate = base_pooled - SCREEN_IMPROVEMENT_GATE_FT
    triggered = best_key != "base" and best_pooled <= gate
    print(
        f"\nbest: {best_key} = {best_pooled:.4f} (base {base_pooled:.4f}, "
        f"delta {best_pooled - base_pooled:+.4f}ft, screen trigger gate <= {gate:.4f})"
    )
    print(
        "NOTE (R15 lesson): this is a SEED-42-ONLY screen. A seed-42 'improvement' is not an "
        "adoption decision -- only a 5-seed mean vs the R13 5-seed-mean gate "
        f"(<= {R13_5SEED_MEAN_POOLED_RMSE - ADOPTION_IMPROVEMENT_GATE_FT:.4f}) can adopt."
    )
    # Machine-readable markers for a follow-up driver.
    print(f"CONFIRM_TRIGGER: {'YES' if triggered else 'NO'}")
    print(f"CONFIRM_KEY: {best_key}")


def _aggregate_confirm(key: str, n_wells: int | None) -> None:
    """5-seed mean for a confirmed config (the seed-42 screen npz is member 0)."""
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
    print(f"R20 confirmation: config {key} x {len(results)} seeds")
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

    out_path = OUTPUTS_DIR / "stack_v32_oof.npz"
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

    adoption_gate = R13_5SEED_MEAN_POOLED_RMSE - ADOPTION_IMPROVEMENT_GATE_FT
    verdict = "採用候補" if mean_pooled <= adoption_gate else "却下（非改善、実測値は報告のみ）"
    print(
        f"\n判定 (R20 config {key}): {verdict} -- 5-seed mean={mean_pooled:.4f}, "
        f"R13 5-seed baseline={R13_5SEED_MEAN_POOLED_RMSE:.4f} "
        f"(改善 {R13_5SEED_MEAN_POOLED_RMSE - mean_pooled:+.4f}ft), gate <= {adoption_gate:.4f}ft"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wells", type=int, default=None, help="Health-check subset size.")
    parser.add_argument("--config", choices=SWEEP_KEYS, default=None, help="Config key to run.")
    parser.add_argument(
        "--seed-idx",
        type=int,
        choices=range(len(SEEDS)),
        default=None,
        help=f"Run SEEDS[i] (SEEDS={SEEDS}) for --config, then exit.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Marks this run as part of 5-seed confirmation (same mechanics as a normal run; "
        "kept for symmetry with run_stack_v27_cv.py's CLI).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Print the seed-42 screen table + regression guard + CONFIRM_TRIGGER/CONFIRM_KEY.",
    )
    parser.add_argument(
        "--aggregate-confirm",
        action="store_true",
        help="Aggregate --config's 5 seeds into a mean pooled RMSE + adoption verdict.",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.aggregate:
        _aggregate(args.n_wells)
        print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
        return

    if args.aggregate_confirm:
        if args.config is None:
            parser.error("--aggregate-confirm requires --config")
        _aggregate_confirm(args.config, args.n_wells)
        print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
        return

    if args.config is None:
        parser.error("one of --config / --aggregate / --aggregate-confirm required")
        return  # unreachable, satisfies type-checkers

    seed = SEEDS[args.seed_idx] if args.seed_idx is not None else SCREEN_SEED
    key = args.config

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
    wm = build_well_matrix_v32(wells, split="train")
    gc.collect()
    print(
        f"feature matrix (R20, 59 cols): {wm.X.shape}, rows={len(wm.y_true):,}, "
        f"wells_used={len(wm.used_wells)}/{len(wells)} counts={wm.counts} "
        f"({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )
    _print_target_residual_stats(wm)

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
