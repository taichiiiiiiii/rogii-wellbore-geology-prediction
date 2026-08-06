"""Stack v2.5: transfer-safe RAW feature ablation on top of R11 (issue: R12).

**Why R12.** The 2026-07-10 ledger establishes that every feature/candidate
derived from the new ``pf_bma_v2``/``beam_grid`` bank has failed to transfer
to the hidden test set -- whether blended (sub-v4 CV 9.1104 -> LB 9.456),
injected as GBDT features (sub-v6 CV 8.9805 -> LB 10.009, sub-v7 CV 8.8573 ->
LB 9.662), or probed solo (pf_bma_scale3_v2: train 11.80 -> LB 14.406). The
**only** confirmed CV-to-LB transfer is stack v2's original 56 features (CV
9.2309 -> LB 9.022). R11 (``run_stack_v24_cv.py``) seed-averaged that exact
56-feature model and got only -0.047ft (9.1843, gate 9.1309 missed).

R12 asks a narrower question: are there **raw-data-derived** features (never
touching the pf_bma_v2/beam_grid bank, never touching eval-zone
``TVT``/``TVT_input``, never touching train-only formation columns) that
improve on R11's 9.1843? Three groups, each computable identically at test
time:

  - **H1** (typewell-match reinforcement, 4 cols): known-zone-tail GR vs.
    affine-calibrated typewell GR correlation / best local lag / residual
    std, plus the typewell GR curve's local gradient at the anchor. Uses
    only the known-zone (``TVT_input`` non-NaN prefix) ``GR``/``TVT_input``
    and the typewell -- see :func:`build_h1_features`.
  - **H2** (pure trajectory geometry, 3 cols): cumulative ``-dz`` from the
    anchor, a curvature (second-difference) rolling std, and a horizontal
    path-length-so-far ratio. Uses only ``MD, X, Y, Z`` -- see
    :func:`build_h2_features`.
  - **H3** (well context, 3 cols): the known-zone linear ``TVT_input ~ MD``
    trend extrapolated to each eval row (minus anchor), a leak-safe
    ``TVT_input + Z`` linear-fit residual std (a proxy for the ANCC=TVT+Z
    identity from the 2026-07-10 R6-a ledger entry, computed WITHOUT reading
    the train-only ``ANCC`` column), and the eval-zone-length /
    known-zone-length ratio. Uses only ``MD``, the known-zone
    ``TVT_input``, and ``Z`` -- see :func:`build_h3_features`.

**Design.** One superset feature matrix (56 + 4 + 3 + 3 = 66 cols) is built
per process (byte-identical column-block order to
``run_stack_v22/v23_cv.py``'s "config c/d" naming), then column-sliced per
``--config``:

  - (a) = the 56 original columns only (**R11 regression check** -- must
    reproduce R11's 5-seed mean 9.1843, not stack v2's 9.2309, since the code
    path/seeds/params are otherwise identical to ``run_stack_v24_cv.py``).
  - (b) = (a) + H1 (60 cols)
  - (c) = (b) + H2 (63 cols)
  - (d) = (c) + H3 (66 cols)

``SEEDS`` = ``(42, 101, 202, 303, 404)``, identical set/order to R9/R11 --
seed 42 first for the regression check. Fold assignment
(``cv.well_folds(wells, 5, seed=42)``) and ES-holdout split
(``_split_fit_es(..., seed=42)``) are frozen for every member, exactly as in
R9/R11.

**Gate.** best 5-seed-mean pooled RMSE across (b)/(c)/(d) <= 9.0843 (R11's
9.1843 minus 0.10ft), reported regardless of outcome.

**Memory.** ~2.6GB target; the 66-column superset is ~10 more float32
columns than R11's 56 (~+150MB over R11's ~2.4GB peak). Two-pass
preallocating build (byte-copied structure from v22/v23/v24), one seed per OS
process, strictly serial. Per-(config, seed) results checkpoint to
``outputs/stack_v25_result_{config}_s{seed}.npz`` immediately.

**Leak boundary.** H1/H2/H3 read only: the known-zone (``TVT_input``
non-NaN prefix) columns of ``h`` (``MD, X, Y, Z, GR, TVT_input``), the
typewell (``TVT, GR``), and ``data.eval_mask(h)``/``data.last_known_tvt(h)``
for row selection. None of the three groups reads ``h["TVT"]``, any
eval-zone ``TVT_input`` (NaN there by construction), any train-only
formation column (``ANCC``/``ASTNU``/``ASTNL``/``EGFDU``/``EGFDL``/``BUDA``),
or the ``pf_bma_v2``/``beam_grid`` bank npz files (never opened by this
script, same as R11).

Usage::

    # smoke test (all 5 seeds + aggregate in one process, ~40 wells)
    uv run python scripts/run_stack_v25_cv.py --config d --n-wells 40

    # full 773-well run, ONE PROCESS PER (config, SEED), strictly serial
    uv run python scripts/run_stack_v25_cv.py --config a --seed-idx 0
    ...
    uv run python scripts/run_stack_v25_cv.py --config d --seed-idx 4

    # after all 5 members of a config have saved their npz:
    uv run python scripts/run_stack_v25_cv.py --config a --aggregate

    # after all 4 configs are aggregated:
    uv run python scripts/run_stack_v25_cv.py --compare
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
from rogii import typewell as TW
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
R11_5SEED_MEAN_POOLED_RMSE = 9.1843  # ledger 2026-07-10 (run_stack_v24_cv.py 5-seed mean)
PF_BLEND_W = 0.7

IMPROVEMENT_GATE_FT = 0.10
ADOPTION_GATE_RMSE = R11_5SEED_MEAN_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.0843
Y_TRUE_ATOL = 1e-2
R11_REPRO_ATOL = 0.02  # config (a) 5-seed mean vs R11's 9.1843

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN for every member (matches v2/R7-R11)
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN for every member (matches v2/R7-R11)
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

N_FEATURES_SUPERSET = 66  # P1(36) + tracker(8) + well-agg(8) + spatial(4) + H1(4) + H2(3) + H3(3)
CONFIG_KEYS: tuple[str, ...] = ("a", "b", "c", "d")


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

    Same recipe as ``run_stack_v2/v21/v22/v23/v24_cv.py``.
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
# R11 alignment primitives + P1 feature builder, byte-copied from
# run_stack_v24_cv.py (unused v21-v24 helpers block_slices/reindex_to_master
# dropped -- R12 does not touch the bank).
# --------------------------------------------------------------------------- #


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y ~ x``; ``0.0`` when degenerate.

    Byte-copied from ``rogii.features._robust_slope`` (kept local rather than
    importing a private helper across modules, matching this script family's
    existing byte-copy convention).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    xs, ys = x[mask], y[mask]
    if np.std(xs) < 1e-9:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


# --------------------------------------------------------------------------- #
# H1: typewell-match reinforcement (4 cols) -- see module docstring for the
# leak boundary. Reads only the known-zone (TVT_input non-NaN prefix) GR/
# TVT_input of `h` and the typewell (TVT, GR).
# --------------------------------------------------------------------------- #

H1_FEATURE_COLUMNS: tuple[str, ...] = (
    "known_tail_gr_twgr_corr",
    "known_tail_gr_twgr_best_lag",
    "known_tail_gr_twgr_resid_std",
    "twgr_gradient_at_anchor",
)

_H1_TAIL_WINDOW = 200  # matches rogii.features._KNOWN_TAIL_LONG
_H1_LAG_RANGE_FT = 15.0  # +/- TVT-space search range, ft
_H1_LAG_STEP_FT = 1.0
_H1_GRADIENT_EPS_FT = 1.0
_H1_MIN_PAIRS = 10


def build_h1_features(h: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    """H1: 4 well-level scalars, broadcast to every eval-zone row.

    Inputs (all present in train AND test, computable identically at test
    time): the known-zone (``data.eval_mask(h)`` False rows, i.e. the
    ``TVT_input`` non-NaN prefix) ``GR`` and ``TVT_input`` columns of ``h``,
    and the typewell ``tw`` (``TVT``, ``GR``). Never reads ``h["TVT"]`` or
    any eval-zone ``TVT_input`` (NaN there by construction).

    Columns:
      - known_tail_gr_twgr_corr: Pearson correlation between the known
        zone's tail-window ``GR`` and the affine-calibrated typewell GR
        (``TW.fit_affine_gr`` gain/offset fit on the FULL known zone,
        leak-safe) sampled at the same known ``TVT_input`` values, over the
        last ``min(_H1_TAIL_WINDOW, n_known)`` rows. ``0.0`` when <10
        finite pairs or zero variance on either side.
      - known_tail_gr_twgr_best_lag: the TVT-space shift (ft, grid
        ``[-15, +15]`` step 1) of the typewell-GR resampling that maximizes
        that same correlation -- does the known-zone registration already
        sit at the locally best GR match, or would a small depth shift fit
        better. ``0.0`` when degenerate.
      - known_tail_gr_twgr_resid_std: std of ``GR - calibrated twGR`` over
        the tail window at zero lag (known-zone match residual/noise level).
      - twgr_gradient_at_anchor: finite-difference slope of the typewell's
        ``TVT -> GR`` curve at the anchor TVT (``+/- 1ft``, gain/offset
        calibrated) -- local discriminability of the typewell signature at
        the current registration point.

    Never raises; ``0.0`` fallback on any degenerate/missing input.
    """
    mask = D.eval_mask(h)
    n_eval = int(mask.sum())
    known_mask = ~mask
    n_known = int(known_mask.sum())

    if n_eval == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in H1_FEATURE_COLUMNS})

    corr = 0.0
    best_lag = 0.0
    resid_std = 0.0
    grad = 0.0

    has_gr = "GR" in h.columns
    tw_ok = {"TVT", "GR"}.issubset(tw.columns) and not tw.empty
    if n_known > 0 and has_gr and tw_ok:
        known_tvt = h.loc[known_mask, "TVT_input"].to_numpy(dtype=float)
        known_gr = h.loc[known_mask, "GR"].to_numpy(dtype=float)
        tail_n = min(_H1_TAIL_WINDOW, n_known)
        tail_tvt = known_tvt[-tail_n:]
        tail_gr = known_gr[-tail_n:]

        gain, offset = TW.fit_affine_gr(h, tw)
        lookup = TW.tw_gr_lookup(tw)

        def _corr_at_lag(lag: float) -> float:
            twgr = TW.apply_affine(lookup(tail_tvt + lag), gain, offset)
            m = np.isfinite(tail_gr) & np.isfinite(twgr)
            if int(m.sum()) < _H1_MIN_PAIRS:
                return 0.0
            a, b = tail_gr[m], twgr[m]
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return 0.0
            c = float(np.corrcoef(a, b)[0, 1])
            return c if np.isfinite(c) else 0.0

        corr = _corr_at_lag(0.0)

        lags = np.arange(-_H1_LAG_RANGE_FT, _H1_LAG_RANGE_FT + 1e-9, _H1_LAG_STEP_FT)
        lag_corrs = np.array([_corr_at_lag(float(lag)) for lag in lags])
        if lag_corrs.size and np.any(lag_corrs != 0.0):
            best_lag = float(lags[int(np.argmax(lag_corrs))])

        twgr0 = TW.apply_affine(lookup(tail_tvt), gain, offset)
        m0 = np.isfinite(tail_gr) & np.isfinite(twgr0)
        if int(m0.sum()) >= 2:
            resid_std = float(np.std(tail_gr[m0] - twgr0[m0]))

        anchor_tvt = float(tail_tvt[-1]) if tail_tvt.size else float("nan")
        if np.isfinite(anchor_tvt):
            q = np.array([anchor_tvt - _H1_GRADIENT_EPS_FT, anchor_tvt + _H1_GRADIENT_EPS_FT])
            g_lo, g_hi = TW.apply_affine(lookup(q), gain, offset)
            if np.isfinite(g_hi) and np.isfinite(g_lo):
                grad = float((g_hi - g_lo) / (2.0 * _H1_GRADIENT_EPS_FT))

    corr = corr if np.isfinite(corr) else 0.0
    best_lag = best_lag if np.isfinite(best_lag) else 0.0
    resid_std = resid_std if np.isfinite(resid_std) else 0.0
    grad = grad if np.isfinite(grad) else 0.0

    out = pd.DataFrame(
        {
            "known_tail_gr_twgr_corr": np.full(n_eval, corr, dtype=np.float64),
            "known_tail_gr_twgr_best_lag": np.full(n_eval, best_lag, dtype=np.float64),
            "known_tail_gr_twgr_resid_std": np.full(n_eval, resid_std, dtype=np.float64),
            "twgr_gradient_at_anchor": np.full(n_eval, grad, dtype=np.float64),
        }
    )
    out = out[list(H1_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# H2: pure well-trajectory geometry (3 cols). Reads only MD, X, Y, Z (always
# present) plus data.eval_mask(h) for row selection. Never reads GR, TVT, or
# TVT_input.
# --------------------------------------------------------------------------- #

H2_FEATURE_COLUMNS: tuple[str, ...] = (
    "eval_cum_neg_dz_from_anchor",
    "dz_curvature_roll_std",
    "horiz_dist_ratio",
)

_H2_CURV_ROLL_WINDOW = 51  # matches rogii.features._TRAJ_ROLL_WINDOW


def build_h2_features(h: pd.DataFrame) -> pd.DataFrame:
    """H2: 3 columns from ``MD, X, Y, Z`` only (never ``GR``/``TVT``/``TVT_input``).

    Columns:
      - eval_cum_neg_dz_from_anchor: for eval row i, ``-(Z[i] -
        Z[anchor_row])`` -- the cumulative sum of ``-dz`` (MD is always
        exactly 1ft-spaced, so this is one ``-dz`` term per row) from the
        anchor row (the last known-zone row, ``data.eval_mask(h)`` False) up
        to row i. A pure-trajectory, GR-free naive TVT-delta.
      - dz_curvature_roll_std: rolling std (window 51, centered) of the
        second difference of ``Z`` (curvature / build-rate-change proxy),
        evaluated at each eval-zone row.
      - horiz_dist_ratio: cumulative horizontal (X-Y plane) path length from
        the well's first row to row i, divided by the well's total
        horizontal path length (first row to last row). ``0.0`` when the
        well's total horizontal path length is ~0.

    Never raises; one row per eval-zone row (``data.eval_mask(h)`` order).
    """
    mask = D.eval_mask(h)
    idx = np.where(mask)[0]
    n_eval = int(idx.size)
    if n_eval == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in H2_FEATURE_COLUMNS})

    known_mask = ~mask
    known_idx = np.where(known_mask)[0]
    anchor_row = int(known_idx[-1]) if known_idx.size else 0

    z = h["Z"].to_numpy(dtype=float)
    x = h["X"].to_numpy(dtype=float)
    y = h["Y"].to_numpy(dtype=float)

    anchor_z = z[anchor_row] if np.isfinite(z[anchor_row]) else 0.0
    eval_cum_neg_dz = -(z[idx] - anchor_z)
    eval_cum_neg_dz = np.where(np.isfinite(eval_cum_neg_dz), eval_cum_neg_dz, 0.0)

    dz1 = np.diff(z, prepend=z[0] if len(z) else 0.0)
    d2z = np.diff(dz1, prepend=dz1[0] if len(dz1) else 0.0)
    curv_roll = (
        pd.Series(d2z).rolling(_H2_CURV_ROLL_WINDOW, min_periods=1, center=True).std().to_numpy()
    )
    curv_roll_eval = curv_roll[idx]
    curv_roll_eval = np.where(np.isfinite(curv_roll_eval), curv_roll_eval, 0.0)

    dx = np.diff(x, prepend=x[0] if len(x) else 0.0)
    dy = np.diff(y, prepend=y[0] if len(y) else 0.0)
    seg_len = np.sqrt(dx**2 + dy**2)
    seg_len = np.where(np.isfinite(seg_len), seg_len, 0.0)
    cum_path = np.cumsum(seg_len)
    total_path = float(cum_path[-1]) if cum_path.size else 0.0
    if total_path > 1e-9:
        horiz_dist_ratio = cum_path[idx] / total_path
    else:
        horiz_dist_ratio = np.zeros(n_eval, dtype=float)

    out = pd.DataFrame(
        {
            "eval_cum_neg_dz_from_anchor": eval_cum_neg_dz,
            "dz_curvature_roll_std": curv_roll_eval,
            "horiz_dist_ratio": horiz_dist_ratio,
        }
    )
    out = out[list(H2_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# H3: well context from the known zone only (3 cols). Reads only MD, the
# known-zone TVT_input, and Z. Never reads the train-only ANCC column or
# h["TVT"].
# --------------------------------------------------------------------------- #

H3_FEATURE_COLUMNS: tuple[str, ...] = (
    "known_linear_trend_extrap_minus_anchor",
    "known_tvt_z_identity_resid_std",
    "zone_len_known_len_ratio",
)

_H3_TREND_TAIL = 200  # matches rogii.features._KNOWN_TAIL_LONG


def build_h3_features(h: pd.DataFrame) -> pd.DataFrame:
    """H3: 3 columns from ``MD``, the known-zone ``TVT_input``, and ``Z`` only.

    Columns:
      - known_linear_trend_extrap_minus_anchor: per eval row i, the linear
        trend of ``TVT_input ~ MD`` fit over the last
        ``min(_H3_TREND_TAIL, n_known)`` known-zone rows, extrapolated to
        ``MD[i]`` and expressed relative to the anchor
        (``slope * (MD[i] - anchor_MD)``) -- the design.md "per-well 直線"
        oracle's delta prediction, exposed as a feature, computed only from
        the known prefix.
      - known_tvt_z_identity_resid_std: residual std of a linear fit of
        ``(TVT_input + Z) ~ MD`` over the FULL known zone -- a leak-safe
        proxy for the ANCC=TVT+Z+const identity (ledger 2026-07-10 R6-a),
        using only ``TVT_input`` (known prefix, never eval-zone) and ``Z``
        (never the train-only ``ANCC`` column).
      - zone_len_known_len_ratio: ``n_eval_rows / n_known_rows`` for the
        well, broadcast to every eval row.

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
# superset well-matrix builder (56 v2 cols + H1 + H2 + H3 = 66)
# --------------------------------------------------------------------------- #


class _WellMatrix:
    """Container for the 66-column superset feature matrix every config slices."""

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


def build_well_matrix_v25(wells: list[str], split: str = "train") -> _WellMatrix:
    """Two-pass, preallocating build of the 66-column superset (v2's 56 + H1 + H2 + H3).

    Byte-copied validation/pass structure from ``run_stack_v22/v23/v24_cv.py``
    (``build_well_matrix_v2``): pass 1 validates wells + collects row counts
    (unchanged -- H1/H2/H3 need no extra cache files, only ``h``/``tw`` which
    are already loaded for the P1 block), pass 2 writes every feature block
    (P1, tracker, well-agg, spatial, H1, H2, H3) into pre-sized arrays.
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
            f"build_well_matrix_v25: 0 wells passed validation -- no rows to build "
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
        h1_feats = build_h1_features(h, tw)
        h2_feats = build_h2_features(h)
        h3_feats = build_h3_features(h)

        if not col_arrays:
            all_cols = (
                list(p1_feats.columns)
                + list(trk_feats.columns)
                + list(well_agg_feats.columns)
                + list(spatial_feats.columns)
                + list(h1_feats.columns)
                + list(h2_feats.columns)
                + list(h3_feats.columns)
            )
            col_arrays = {c: np.empty(total_rows, dtype=np.float32) for c in all_cols}

        all_blocks = (
            p1_feats,
            trk_feats,
            well_agg_feats,
            spatial_feats,
            h1_feats,
            h2_feats,
            h3_feats,
        )
        for block in all_blocks:
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
            f"R12 superset must have {N_FEATURES_SUPERSET} cols, got {len(all_cols)}"
        )
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, counts
    )


def _config_columns(wm: _WellMatrix, config: str) -> list[str]:
    """Column subset for ``config`` in {a, b, c, d} -- see module docstring.

    Selects by name-set membership against the H1/H2/H3 column-name
    constants (not a fragile positional slice), so a future change to the
    superset's column order fails loudly via the length assertion below
    rather than silently mis-slicing.
    """
    p1_end = 56  # P1(36) + tracker(8) + well-agg(8) + spatial(4), stack v2's "all-in" columns
    new_cols = set(H1_FEATURE_COLUMNS) | set(H2_FEATURE_COLUMNS) | set(H3_FEATURE_COLUMNS)
    base56 = [c for c in wm.X.columns if c not in new_cols]
    if len(base56) != p1_end:
        raise AssertionError(f"expected {p1_end} base columns, got {len(base56)}")

    if config == "a":
        return base56
    if config == "b":
        return base56 + list(H1_FEATURE_COLUMNS)
    if config == "c":
        return base56 + list(H1_FEATURE_COLUMNS) + list(H2_FEATURE_COLUMNS)
    if config == "d":
        return (
            base56
            + list(H1_FEATURE_COLUMNS)
            + list(H2_FEATURE_COLUMNS)
            + list(H3_FEATURE_COLUMNS)
        )
    raise ValueError(f"unknown config {config!r}, expected one of {CONFIG_KEYS}")


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Byte-for-byte identical to ``run_stack_v2/v21/v22/v23/v24_cv.py``'s
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

    Identical protocol to ``run_stack_v22/v23/v24_cv.py``'s
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

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1..v2.4
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
    return OUTPUTS_DIR / f"stack_v25_result_{config}_s{seed}{suffix}.npz"


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
            f"{path} not found -- run `uv run python scripts/run_stack_v25_cv.py "
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
        f"{'vs R11 9.1843':>16}"
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
            f"{cum_pooled - R11_5SEED_MEAN_POOLED_RMSE:>+16.4f}"
        )
    print("=" * 96)
    oof_mean = cum_sum / len(results)
    mean_pooled = cum_pooled_list[-1]

    print(f"\nreference: carry_last={floor_pooled:.4f}  PF blend w0.7={pf_pooled:.4f}")
    print(f"total rows scored: {len(y_true):,}  members aggregated: {len(results)}/{len(SEEDS)}")

    seed42 = next((r for r in results if r["seed"] == 42), None)
    if seed42 is not None and config == "a":
        diff = seed42["pooled_rmse"] - STACK_V2_ALLIN_POOLED_RMSE
        if n_wells is not None:
            print(
                f"\nseed-42 solo repro check (n_wells={n_wells} subsample, informational only): "
                f"{seed42['pooled_rmse']:.4f} vs stack v2 all-in ledger 9.2309 (diff {diff:+.4f}ft)"
            )
        else:
            print(
                f"\nseed-42 solo repro check: {seed42['pooled_rmse']:.4f} vs stack v2 all-in "
                f"ledger 9.2309 (diff {diff:+.4f}ft, informational -- config (a)'s regression "
                "check is the 5-seed MEAN vs R11's 9.1843, printed below)"
            )

    mean_well_rmse = _per_well_rmse(y_true, oof_mean, well_idx, n_wells_used)
    valid = mean_well_rmse[np.isfinite(mean_well_rmse)]
    print(
        f"\n{len(results)}-seed mean OOF (config {config}): pooled={mean_pooled:.4f}  "
        f"median={np.median(valid):.4f}  p90={np.percentile(valid, 90):.4f}  "
        f"max={np.max(valid):.4f}"
    )

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

    out_path = OUTPUTS_DIR / f"stack_v25_oof_{config}.npz"
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
        diff_r11 = mean_pooled - R11_5SEED_MEAN_POOLED_RMSE
        ok = abs(diff_r11) <= R11_REPRO_ATOL
        print(
            f"\nR11 REGRESSION CHECK (config a, 5-seed mean): {mean_pooled:.4f} vs R11 ledger "
            f"9.1843 (diff {diff_r11:+.4f}ft, tol {R11_REPRO_ATOL}ft) -> "
            f"{'PASS' if ok else 'FAIL -- investigate before trusting configs b/c/d'}"
        )

    improvement = R11_5SEED_MEAN_POOLED_RMSE - mean_pooled
    verdict = (
        "採用候補" if mean_pooled <= ADOPTION_GATE_RMSE else "却下（非改善、実測値は報告のみ）"
    )
    print(f"\n{'=' * 96}")
    print(
        f"判定 (config {config}): {verdict} -- 5-seed mean={mean_pooled:.4f}, "
        f"R11 baseline={R11_5SEED_MEAN_POOLED_RMSE:.4f}, "
        f"改善={improvement:+.4f}ft (gate: <= {ADOPTION_GATE_RMSE:.4f}ft)"
    )
    print(f"{'=' * 96}")
    return mean_pooled


def _compare_all(n_wells: int | None) -> None:
    """Load all 4 configs' saved OOF npz and print the final R12 ablation table."""
    print("\n" + "=" * 96)
    print("R12 ablation summary (a=R11 56-feat repro, b=+H1, c=+H1+H2, d=+H1+H2+H3)")
    print("=" * 96)
    print(
        f"{'config':>8}{'n_features':>12}{'5-seed mean pooled':>22}"
        f"{'vs R11 9.1843':>16}{'gate<=9.0843':>14}"
    )
    print("-" * 96)
    results: dict[str, float] = {}
    for config in CONFIG_KEYS:
        path = OUTPUTS_DIR / f"stack_v25_oof_{config}.npz"
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
        print(
            f"{config:>8}{n_feat:>12}{pooled:>22.4f}"
            f"{pooled - R11_5SEED_MEAN_POOLED_RMSE:>+16.4f}{gate_pass:>14}"
        )
    print("=" * 96)
    if results:
        best_config = min(results, key=lambda k: results[k])
        print(
            f"\nbest: config {best_config} = {results[best_config]:.4f} "
            f"({'PASS' if results[best_config] <= ADOPTION_GATE_RMSE else 'FAIL'} gate "
            f"<= {ADOPTION_GATE_RMSE:.4f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=CONFIG_KEYS,
        default=None,
        help="Feature-set config: a=56 (R11 repro), b=+H1(60), c=+H1+H2(63), d=+H1+H2+H3(66).",
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
        help="Skip CV; load outputs/stack_v25_result_{config}_s*.npz, print the table, "
        "write outputs/stack_v25_oof_{config}.npz.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Skip CV; load all 4 configs' outputs/stack_v25_oof_{a,b,c,d}.npz and print "
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
    wm = build_well_matrix_v25(wells, split="train")
    gc.collect()
    print(
        f"feature matrix (superset, {wm.X.shape[1]} cols): {wm.X.shape}, "
        f"rows={len(wm.y_true):,}, wells_used={len(wm.used_wells)}/{len(wells)} "
        f"counts={wm.counts} ({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )
    cols = _config_columns(wm, args.config)
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
