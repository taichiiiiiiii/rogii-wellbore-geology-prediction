"""Stack v3.3: R22-lite prefix-cut pseudo-well augmentation on top of v26's 59-feature CV.

**Why R22-lite.** ``analysis/experiment_ledger.md``'s cross-well ablations
(R14-R21) all failed to beat R13 (``run_stack_v26_cv.py`` = the current best,
56 v2 cols + H3 = 59 features, 5-seed mean pooled OOF **9.1456**, seed-42
solo **9.2515**) -- every new *feature* stalled. The next lever this script
tests is not a feature but a **sample-count** one: the hypothesis is that
cross-well generalization is bottlenecked by the effective training-example
count = 773 train wells, not by the row count within any one well. If a
train well's known-zone prefix is cut short partway through, the newly-
hidden tail (still real, still has ground-truth ``TVT``) becomes a *pseudo*
evaluation zone for a *pseudo*, earlier-anchored version of the same well --
additional (well, anchor-position) training signal that costs nothing at
inference time (a submission notebook is completely unaffected; only the
CV/training harness below changes).

**Design: identical to v26, plus ONE change.** Same 59-column feature set
(v2's 56 + H3's 3, byte-copied below), same fold seed 42, same ES-holdout
protocol/seed, same LGBM base params, same 5 seeds
``(42, 101, 202, 303, 404)``. The only difference: each fold's TRAINING data
gets pseudo-well rows appended, built from
``scripts/build_aug_caches.py``'s ``data/processed/aug_cache_lite/*.npz``
cache (150-well deterministic sample, see that script's docstring for the
cut-point selection and its measured feasibility finding). A pseudo well's
rows are tagged with its SOURCE well's identity, so they are added to a fold
only when the source well itself is a FIT well for that fold (never an
ES-holdout well, never a rows-being-scored well) -- this keeps the ES
protocol and the scored OOF rows byte-identical to v26's. **Validation/
scoring is always the real evaluation zone only** (v26's exact row set), so
this script's OOF is directly pooled-RMSE-comparable to R13's ledger numbers
(5-seed mean 9.1456, seed-42 solo 9.2515).

``--no-aug`` disables the augmentation entirely (no aug matrix is even
built) -- the remaining code path is then byte-identical to v26's, and a
``--no-aug`` run on the same wells/seed must reproduce v26's pooled RMSE and
OOF array exactly (regression guard for this script's own harness).

**Leak boundary.** Strict superset of v26's, extended by exactly the aug
cache's own leak boundary (``scripts/build_aug_caches.py``'s docstring): a
pseudo well's features are built from
``build_pseudo_well(h_full, known_len, cut_idx)`` -- ``h_full`` sliced to
``[0, known_len)`` with ``TVT_input[cut_idx:known_len)`` NaN-ed out and the
``TVT`` column dropped outright, so no feature builder (``rogii.features``,
this file's ``build_h3_features``, the tracker/spatial feature blocks) can
read the pseudo evaluation zone's true values. Only the *label* for those
rows (``y_true_aug = h_full["TVT"][cut_idx:known_len]``, the real ground
truth) is read directly by this script -- never fed to a feature builder,
exactly the same "label assembled outside the leak-safe layer" pattern v26
already uses for its real rows.

**IMPORTANT (see ``scripts/build_aug_caches.py``'s "IMPORTANT FINDING").**
At the aug cache's task-spec default thresholds (3000-row pseudo-eval-zone
cap, 800-row/40% residual-known floor), essentially no well in this dataset
passes the cut-point feasibility check (measured: 0/150 of the deterministic
sample, 1/773 overall) -- this dataset's known-zone rows top out at 2392
across all 773 train wells, well under the 3000-row cap. A full aug-cache
build at the literal spec defaults would therefore produce an aug matrix
with ~0 rows, and this script's ``--no-aug``-equivalent behavior would
silently be all this configuration can ever do. This script does not work
around that (it consumes whatever ``aug_cache_lite/`` contains, however
many wells that is); the finding is flagged here for the same reason it is
flagged in ``build_aug_caches.py`` -- a design decision (recalibrating the
cap) is needed before a full run is worth launching.

Usage::

    # smoke: reproduce v26 exactly (no aug matrix even built)
    uv run python scripts/run_stack_v33_cv.py --n-wells 30 --seed-idx 0 --no-aug

    # smoke: aug-on, using whatever aug_cache_lite/ currently holds (e.g. an
    # 8-well demo cache) -- must not crash even if 0 wells' aug data intersects
    # the --n-wells 30 subset
    uv run python scripts/run_stack_v33_cv.py --n-wells 30 --seed-idx 0

    # full 773-well run, ONE PROCESS PER SEED, strictly serial (NOT launched
    # by this task -- see the module docstring's IMPORTANT note above first)
    uv run python scripts/run_stack_v33_cv.py --seed-idx 0
    ...
    uv run python scripts/run_stack_v33_cv.py --seed-idx 4
    uv run python scripts/run_stack_v33_cv.py --aggregate
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
AUG_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "aug_cache_lite"
OUTPUTS_DIR = _REPO_ROOT / "outputs"

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
# R13 (run_stack_v26_cv.py) ledger numbers, 2026-07-10/11 -- this script's
# direct comparison points (analysis/experiment_ledger.md).
R13_SEED42_POOLED_RMSE = 9.2515  # v26 seed-42 solo, full 773 wells
R13_5SEED_MEAN_POOLED_RMSE = 9.1456  # v26 5-seed mean, full 773 wells
PF_BLEND_W = 0.7

IMPROVEMENT_GATE_FT = 0.05
ADOPTION_GATE_RMSE = R13_5SEED_MEAN_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.0956
Y_TRUE_ATOL = 1e-2

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN, matches v26/R7-R21
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN, matches v26/R7-R21
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

N_FEATURES_EXPECTED = 59  # P1(36) + tracker(8) + well-agg(8) + spatial(4) + H3(3), same as v26


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


def _aug_npz_path(well: str) -> Path:
    return AUG_CACHE_DIR / f"{well}.npz"


def _stratified_sample(all_wells: list[str], n: int, seed: int) -> list[str]:
    """Deterministic sample of ``n`` wells, stratified by tracker-manifest pf_std_mean decile.

    Byte-copied from ``run_stack_v26_cv.py`` (unchanged).
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
# H3 feature block, byte-copied from run_stack_v26_cv.py (itself byte-copied
# from run_stack_v25_cv.py). Reads only MD, the known-zone TVT_input, and Z.
# Never reads the train-only ANCC column or h["TVT"].
# --------------------------------------------------------------------------- #


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y ~ x``; ``0.0`` when degenerate. Byte-copied from v26."""
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

    Byte-copied from ``run_stack_v26_cv.py`` -- see that module's docstring
    for the full column definitions. Never raises; one row per eval-zone row
    (``data.eval_mask(h)`` order) -- for an aug-cache pseudo well, ``h`` is
    the truncated ``h_pseudo`` and "eval-zone row" means its pseudo eval
    zone ``[cut_idx, known_len)``.
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
# Prefix-cut pseudo-well reconstruction, byte-copied from
# scripts/build_aug_caches.py::build_pseudo_well (unchanged). Only this one
# helper is needed here -- cut-point SELECTION already happened when the aug
# cache was built; this script only rebuilds the same h_pseudo from the
# cached (known_len, cut_idx) metadata so it can recompute the cheap P1/H3
# features (which the aug cache does NOT store -- only the expensive
# tracker/spatial arrays are cached).
# --------------------------------------------------------------------------- #


def build_pseudo_well(
    h: pd.DataFrame, known_len: int, cut_idx: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Rebuild the pseudo well: rows ``[0, known_len)`` with ``TVT_input``
    NaN-ed out on ``[cut_idx, known_len)`` and ``TVT`` dropped entirely.

    Byte-copied from ``scripts/build_aug_caches.py`` (see that module's
    docstring for the full leak-safety rationale). Returns ``(h_pseudo,
    y_true)`` -- ``y_true`` is this well's real ground truth for the pseudo
    eval zone, captured before the NaN-out, used ONLY as this script's
    training label (never as a feature-builder input).
    """
    h_trunc = h.iloc[:known_len].reset_index(drop=True).copy()
    y_true = h_trunc["TVT"].to_numpy(dtype=float)[cut_idx:known_len].copy()

    tvt_input = h_trunc["TVT_input"].to_numpy(dtype=float, copy=True)
    tvt_input[cut_idx:known_len] = np.nan
    h_trunc["TVT_input"] = tvt_input

    h_pseudo = h_trunc.drop(columns=["TVT"]) if "TVT" in h_trunc.columns else h_trunc
    return h_pseudo, y_true


# --------------------------------------------------------------------------- #
# well-matrix builder (56 v2 cols + H3 = 59), byte-copied from
# run_stack_v26_cv.py::build_well_matrix_v26 (unchanged -- real rows only).
# --------------------------------------------------------------------------- #


class _WellMatrix:
    """Container for the 59-column feature matrix every ensemble member trains/scores on."""

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


def build_well_matrix_v33(wells: list[str], split: str = "train") -> _WellMatrix:
    """Two-pass, preallocating build of the 59-column REAL-row matrix.

    Byte-copied from ``run_stack_v26_cv.py::build_well_matrix_v26``
    (unchanged) -- v33's only change vs. v26 is the additive pseudo-well
    augmentation matrix built separately by :func:`build_aug_matrix` below,
    never mixed into this function's real-row output.
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
            f"build_well_matrix_v33: 0 wells passed validation -- no rows to build "
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
            f"v33 must train on exactly {N_FEATURES_EXPECTED} (56 + H3) real-row features, "
            f"got {len(all_cols)}"
        )
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, counts
    )


# --------------------------------------------------------------------------- #
# R22-lite: pseudo-well augmentation matrix (NEW vs. v26). Features are
# recomputed live from data/processed/aug_cache_lite (tracker/spatial arrays
# + cut-plan metadata); targets come from the REAL, untruncated well's own
# TVT column -- never from the aug cache (which never stores TVT).
# --------------------------------------------------------------------------- #


class _AugMatrix:
    """Container for the pseudo-well augmentation rows (same 59 columns as ``_WellMatrix.X``).

    ``well_idx`` indexes into the SAME ``used_wells`` list as the real
    ``_WellMatrix`` -- i.e. a pseudo well's rows are tagged with its SOURCE
    well's real-matrix index, so fold/ES membership can be looked up via the
    same ``well_fold_arr`` the real matrix uses. No separate fold logic.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y_r: np.ndarray,
        well_idx: np.ndarray,
        used_aug_wells: list[str],
        counts: dict[str, int],
    ) -> None:
        self.X = X
        self.y_r = y_r  # PF-residual target: y_true_aug - pf_blend_aug
        self.well_idx = well_idx
        self.used_aug_wells = used_aug_wells
        self.counts = counts


def build_aug_matrix(
    wells: list[str],
    well_to_idx: dict[str, int],
    all_cols: list[str],
    split: str = "train",
) -> _AugMatrix:
    """Two-pass, preallocating build of the pseudo-well augmentation matrix.

    Only wells that (a) have an ``aug_cache_lite`` npz AND (b) are a member
    of ``well_to_idx`` (i.e. passed :func:`build_well_matrix_v33`'s own
    validation, so their fold assignment is defined) contribute rows. This
    is a strict subset relationship -- a well absent from the real matrix
    (e.g. missing tracker/spatial cache) can never contribute pseudo rows
    either, since its fold membership would be undefined.
    """
    counts = {"n_no_aug_cache": 0, "n_not_a_used_well": 0, "n_size_mismatch": 0}

    candidates: list[str] = []
    row_counts: list[int] = []
    for well in wells:
        if well not in well_to_idx:
            counts["n_not_a_used_well"] += 1
            continue
        npz_path = _aug_npz_path(well)
        if not npz_path.exists():
            counts["n_no_aug_cache"] += 1
            continue
        with np.load(npz_path) as npz:
            n = int(npz["pseudo_eval_len"])
            pf_size = int(npz["pf_tvt"].shape[0])
        if n <= 0 or pf_size != n:
            counts["n_size_mismatch"] += 1
            continue
        candidates.append(well)
        row_counts.append(n)

    if not candidates:
        empty_X = pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in all_cols})
        return _AugMatrix(
            empty_X, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32), [], counts
        )

    total_rows = int(sum(row_counts))
    boundaries = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int64)

    y_r = np.empty(total_rows, dtype=np.float64)
    well_idx_arr = np.empty(total_rows, dtype=np.int32)
    col_arrays: dict[str, np.ndarray] = {
        c: np.empty(total_rows, dtype=np.float32) for c in all_cols
    }

    for i, well in enumerate(candidates):
        s, e = int(boundaries[i]), int(boundaries[i + 1])

        with np.load(_aug_npz_path(well)) as npz:
            pf_tvt = npz["pf_tvt"].astype(np.float64)
            pf_std = npz["pf_std"].astype(np.float64)
            beam_tvt = npz["beam_tvt"].astype(np.float64)
            beam_margin = npz["beam_margin"].astype(np.float64)
            cached_anchor = float(npz["anchor"])
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
            prefix_rmse = float(npz["prefix_rmse"])
            nn_dist_median = float(npz["nn_dist_median"])
            known_len = int(npz["known_len"])
            cut_idx = int(npz["cut_idx"])

        h_full = D.load_horizontal(well, split)
        tw = D.load_typewell(well, split)
        h_pseudo, y_true = build_pseudo_well(h_full, known_len, cut_idx)
        del h_full

        if y_true.size != e - s:
            raise AssertionError(
                f"aug well {well}: y_true size {y_true.size} != cached pseudo_eval_len {e - s}"
            )

        p1_feats = build_features(h_pseudo, tw)
        trk_feats = build_tracker_features(cached_anchor, pf_tvt, pf_std, beam_tvt, beam_margin)
        gr_nan_frac = float(h_pseudo["GR"].isna().mean()) if "GR" in h_pseudo.columns else 1.0
        well_agg_feats = build_well_aggregate_features(
            cached_anchor, pf_tvt, pf_std, beam_tvt, beam_margin, gr_nan_frac
        )
        spatial_feats = build_spatial_features(
            cached_anchor, spatial_tvt, prefix_rmse, nn_dist_median
        )
        h3_feats = build_h3_features(h_pseudo)

        for block in (p1_feats, trk_feats, well_agg_feats, spatial_feats, h3_feats):
            for c in block.columns:
                col_arrays[c][s:e] = block[c].to_numpy(dtype=np.float32, copy=False)

        pf_blend = cached_anchor + PF_BLEND_W * (pf_tvt - cached_anchor)
        y_r[s:e] = y_true - pf_blend
        well_idx_arr[s:e] = well_to_idx[well]

        del h_pseudo, tw

    X = pd.DataFrame(col_arrays)
    if list(X.columns) != list(all_cols):
        raise AssertionError("aug matrix column order must match the real matrix's all_cols")
    return _AugMatrix(X, y_r, well_idx_arr, candidates, counts)


# --------------------------------------------------------------------------- #
# fold/ES split + per-fold training, byte-copied from v26 with the aug
# concatenation as the ONLY addition (guarded so --no-aug / no-aug-rows-this-
# fold takes the exact same code path as v26 -- see module docstring).
# --------------------------------------------------------------------------- #


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


def run_stack_fold_cv_aug(
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
    am: _AugMatrix | None,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    """Train+OOF-score the (b) PF-residual target for one ensemble member.

    Identical protocol to ``run_stack_v26_cv.py``'s ``run_stack_fold_cv``
    with ES-holdout always on, PLUS: when ``am`` is not ``None`` and has
    rows, each fold's fit-well set additionally pulls in pseudo-well rows
    whose SOURCE well is a fit well for that fold (never an ES-holdout well,
    never a scored-fold well -- guaranteed by construction, since
    ``fit_idx`` is derived from ``train_wells_all`` minus the ES split,
    which itself already excludes the current score fold). When ``am`` is
    ``None`` or contributes 0 rows to a given fold, that fold's ``model.fit``
    call uses EXACTLY the same ``X.loc[train_mask]``/``target[train_mask]``
    slices v26 uses -- no concatenation, no extra object -- so scoring is
    bit-for-bit reproducible against v26 in that case.
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

        n_aug_fit = 0
        if am is not None and len(am.y_r):
            aug_fit_mask = np.isin(am.well_idx, fit_idx)
            n_aug_fit = int(aug_fit_mask.sum())
        else:
            aug_fit_mask = None

        if n_aug_fit > 0:
            X_train = pd.concat([X.loc[train_mask], am.X.loc[aug_fit_mask]], ignore_index=True)
            target_train = np.concatenate([target[train_mask], am.y_r[aug_fit_mask]])
        else:
            X_train = X.loc[train_mask]
            target_train = target[train_mask]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train,
            target_train,
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
                "n_aug_train_rows": n_aug_fit,
                "n_score_rows": int(score_mask.sum()),
                "best_iteration": model.best_iteration_,
                "rmse": fold_rmse,
                "seconds": time.time() - t_fold,
            }
        )
        print(
            f"  [{label}] fold {fold_i}: n_train={int(train_mask.sum()):,} "
            f"n_aug_train={n_aug_fit:,} n_score={int(score_mask.sum()):,} "
            f"best_iter={model.best_iteration_} rmse={fold_rmse:.4f} "
            f"({time.time() - t_fold:.1f}s)",
            flush=True,
        )
        del model, eval_set, train_mask, score_mask, X_train, target_train
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
    am: _AugMatrix | None,
    cols: list[str],
    seed: int,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
) -> dict[str, object]:
    """Run one ensemble member (59-col config, given LGBM seed) to completion."""
    label = f"seed {seed}"
    aug_note = "aug=OFF" if am is None else f"aug=ON({len(am.y_r):,} rows)"
    print(f"\n=== {label} (v33 R22-lite, n_features={len(cols)}, {aug_note}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1..v26
    oof, fold_rows, importances = run_stack_fold_cv_aug(
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
        am,
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


def _result_path(seed: int, n_wells: int | None, aug: bool) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    suffix += "" if aug else "_noaug"
    return OUTPUTS_DIR / f"stack_v33_result_s{seed}{suffix}.npz"


def _save_result(
    result: dict[str, object],
    wm: _WellMatrix,
    am: _AugMatrix | None,
    row_fold: np.ndarray,
    n_wells: int | None,
    aug: bool,
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(result["fold_rows"])
    imp_df = result["importance_df"]
    path = _result_path(int(result["seed"]), n_wells, aug)
    n_aug_rows = int(len(am.y_r)) if am is not None else 0
    n_aug_wells = len(am.used_aug_wells) if am is not None else 0
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
        aug_enabled=np.bool_(aug),
        n_aug_rows=np.int64(n_aug_rows),
        n_aug_wells=np.int64(n_aug_wells),
        fold_id=fold_df["fold"].to_numpy(),
        fold_rmse=fold_df["rmse"].to_numpy(),
        fold_best_iteration=fold_df["best_iteration"].to_numpy(),
        fold_n_train=fold_df["n_train_rows"].to_numpy(),
        fold_n_aug_train=fold_df["n_aug_train_rows"].to_numpy(),
        fold_n_score=fold_df["n_score_rows"].to_numpy(),
        fold_seconds=fold_df["seconds"].to_numpy(),
        importance_feature=imp_df["feature"].to_numpy(),
        importance_value=imp_df["mean_importance"].to_numpy(),
    )
    print(
        f"\nresult saved: {path} (n_aug_rows={n_aug_rows} n_aug_wells={n_aug_wells} "
        f"peak RSS this process: {result['peak_rss_mb']:.0f} MB)"
    )


def _load_result(seed: int, n_wells: int | None, aug: bool) -> dict[str, object]:
    path = _result_path(seed, n_wells, aug)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python scripts/run_stack_v33_cv.py "
            f"--seed-idx {SEEDS.index(seed)}"
            + (f" --n-wells {n_wells}" if n_wells is not None else "")
            + ("" if aug else " --no-aug")
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
            "n_aug_rows": int(npz["n_aug_rows"]),
            "n_aug_wells": int(npz["n_aug_wells"]),
            "fold_id": npz["fold_id"],
            "fold_rmse": npz["fold_rmse"],
            "fold_best_iteration": npz["fold_best_iteration"],
            "importance_feature": [str(f) for f in npz["importance_feature"]],
            "importance_value": npz["importance_value"],
        }


def _aggregate(n_wells: int | None, aug: bool) -> None:
    results: list[dict[str, object]] = []
    for seed in SEEDS:
        try:
            results.append(_load_result(seed, n_wells, aug))
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
    print("=" * 100)
    print(f"v33 R22-lite ({len(ref['cols'])} features, aug={'ON' if aug else 'OFF'})")
    print(
        f"{'k':>3}{'seed':>7}{'solo pooled':>14}{'cum-mean pooled (1..k)':>24}"
        f"{'vs R13 5seed 9.1456':>22}{'vs R13 seed42 9.2515':>22}"
    )
    print("-" * 100)
    cum_pooled_list: list[float] = []
    for k, res in enumerate(results, start=1):
        cum_sum += res["oof"]
        cum_mean = cum_sum / k
        solo = res["pooled_rmse"]
        cum_pooled = pooled_rmse(y_true, cum_mean)
        cum_pooled_list.append(cum_pooled)
        print(
            f"{k:>3}{res['seed']:>7}{solo:>14.4f}{cum_pooled:>24.4f}"
            f"{cum_pooled - R13_5SEED_MEAN_POOLED_RMSE:>+22.4f}"
            f"{cum_pooled - R13_SEED42_POOLED_RMSE:>+22.4f}"
        )
    print("=" * 100)
    oof_mean = cum_sum / len(results)
    mean_pooled = cum_pooled_list[-1]

    print(f"\nreference: carry_last={floor_pooled:.4f}  PF blend w0.7={pf_pooled:.4f}")
    print(f"total rows scored: {len(y_true):,}  members aggregated: {len(results)}/{len(SEEDS)}")
    print(f"aug rows used (seed 42): {ref['n_aug_rows']:,} from {ref['n_aug_wells']} source wells")

    mean_well_rmse = _per_well_rmse(y_true, oof_mean, well_idx, n_wells_used)
    valid = mean_well_rmse[np.isfinite(mean_well_rmse)]
    print(
        f"\n{len(results)}-seed mean OOF (v33): pooled={mean_pooled:.4f}  "
        f"median={np.median(valid):.4f}  p90={np.percentile(valid, 90):.4f}  "
        f"max={np.max(valid):.4f}"
    )

    seed42 = next((r for r in results if r["seed"] == 42), None)
    if seed42 is not None:
        _helped_hurt_report(
            f"v33 {len(results)}-seed mean",
            y_true,
            oof_mean,
            seed42["oof"],
            f"v33 seed-42 solo ({seed42['pooled_rmse']:.4f})",
            well_idx,
            n_wells_used,
        )

    suffix = "" if aug else "_noaug"
    out_path = OUTPUTS_DIR / f"stack_v33_oof{suffix}.npz"
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
        aug_enabled=np.bool_(aug),
        **save_kwargs,
    )
    print(f"\nOOF saved: {out_path} (mean of {len(results)} seeds + each member)")

    improvement_r13 = R13_5SEED_MEAN_POOLED_RMSE - mean_pooled
    verdict = (
        "採用候補" if mean_pooled <= ADOPTION_GATE_RMSE else "却下（非改善、実測値は報告のみ）"
    )
    print(f"\n{'=' * 100}")
    print(
        f"判定 (v33, aug={'ON' if aug else 'OFF'}): {verdict} -- "
        f"5-seed mean={mean_pooled:.4f}, R13 5-seed baseline={R13_5SEED_MEAN_POOLED_RMSE:.4f} "
        f"(改善 {improvement_r13:+.4f}ft), gate <= {ADOPTION_GATE_RMSE:.4f}ft"
    )
    print(f"{'=' * 100}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Skip CV; load outputs/stack_v33_result_s*.npz, print the table, "
        "write outputs/stack_v33_oof[_noaug].npz.",
    )
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable the R22-lite pseudo-well augmentation entirely -- the remaining code "
        "path is byte-identical to run_stack_v26_cv.py (regression guard).",
    )
    args = parser.parse_args()
    aug_enabled = not args.no_aug

    t_start = time.time()

    if args.aggregate:
        _aggregate(args.n_wells, aug_enabled)
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
    wm = build_well_matrix_v33(wells, split="train")
    gc.collect()
    print(
        f"real-row feature matrix (v33, 56+H3 = {wm.X.shape[1]} cols): {wm.X.shape}, "
        f"rows={len(wm.y_true):,}, wells_used={len(wm.used_wells)}/{len(wells)} "
        f"counts={wm.counts} ({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )
    cols = list(wm.X.columns)

    well_to_idx = {w: i for i, w in enumerate(wm.used_wells)}
    am: _AugMatrix | None = None
    if aug_enabled:
        t0 = time.time()
        am = build_aug_matrix(wells, well_to_idx, cols, split="train")
        gc.collect()
        print(
            f"aug matrix (R22-lite pseudo-well rows): rows={len(am.y_r):,} "
            f"wells_used={len(am.used_aug_wells)}/{len(wells)} counts={am.counts} "
            f"({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
        )
    else:
        print("aug matrix: DISABLED (--no-aug) -- not built, regression-guard mode")

    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=FOLD_SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr = np.array([well_to_fold[w] for w in wm.used_wells], dtype=np.int32)
    row_fold = well_fold_arr[wm.well_idx]

    seeds_to_run = [SEEDS[args.seed_idx]] if args.seed_idx is not None else list(SEEDS)
    for seed in seeds_to_run:
        result = run_one_seed(wm, am, cols, seed, row_fold, well_fold_arr)
        print(
            f"  -> seed {seed}: pooled RMSE={result['pooled_rmse']:.4f} "
            f"peak_rss={result['peak_rss_mb']:.0f}MB"
        )
        _save_result(result, wm, am, row_fold, args.n_wells, aug_enabled)
        del result
        gc.collect()

    if args.seed_idx is None:
        _aggregate(args.n_wells, aug_enabled)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
    print(f"this process's peak RSS: {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
