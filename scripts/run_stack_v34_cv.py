"""Stack v3.4: R13's 59 features + pf_std "overconfidence lock-on" time-series
shape (4 new cols, config b = 63 cols) (issue: R24).

**Why R24.** Analysis C (scratchpad qualitative case study of the worst-SSE
wells, see well ``4c2208f5``) found a time-series signature on the well's
catastrophic mid-zone step: the per-seed PF variance ``pf_std`` spikes
sharply (0 -> ~8 ft) just before the model's error takes a step (the PF
"locks on" to the wrong registration path), then collapses and stays low for
the rest of the zone even though the prediction is now durably wrong. Every
existing pf_std feature -- ``build_tracker_features``'s raw per-row
``pf_std``, ``build_well_aggregate_features``'s well-wide ``pf_std_well_
mean``/``pf_std_well_max`` -- is a *static* statistic (current value, or one
scalar for the whole well) and cannot express "pf_std used to be much higher
than it is now" or "how long ago was the last spike". R24 adds exactly that,
computed purely from the already-cached per-row ``pf_std`` array (same source
``build_tracker_features`` reads) -- no new I/O, no new leak surface.

**Design.** Byte-copy of ``run_stack_v26_cv.py`` (R13, 56 v2 cols + H3 = 59
cols) with one additive feature block appended, gated by ``--config``:

  - ``--config a``: reproduces R13 exactly (59 cols) -- a regression guard,
    not a new hypothesis.
  - ``--config b``: R13's 59 cols + 4 new pf_std time-series columns (63
    cols) -- see :func:`build_pf_std_ts_features` for the column definitions.

Same fold seed 42, same ES split seed 42, same LGBM base params, same 5 seeds
``(42, 101, 202, 303, 404)``, one OS process per seed, strictly serial.

**Leak boundary.** Strict subset of R13's. The new block reads only the
eval-zone-aligned ``pf_std`` array already loaded from the tracker cache (no
new files) and is **backward-looking only**: every column is an expanding
max or a trailing window max/argmax ending at (and including) row ``i`` --
never a centered or future window. No ``h["TVT"]`` reads, no eval-zone
``TVT_input``, no train-only formation columns.

**Gate.** Screen: seed-42 solo pooled OOF <= R13 seed-42 solo (**9.2515**) -
0.05 = **9.2015** -> proceed to a 5-seed full judgment (separate task; single-
seed judgments are explicitly banned by the R15 lesson -- a single seed's
apparent improvement has repeatedly turned out to be seed noise, e.g. R17's
seed-42 -0.09 that reversed to +0.013 over 5 seeds). Full-judgment adoption
gate (once run): 5-seed mean pooled OOF <= R13 5-seed mean (**9.1456**) -
0.10 = **9.0456**.

Usage::

    # smoke test (all 5 seeds + aggregate in one process, ~30 wells)
    uv run python scripts/run_stack_v34_cv.py --config a --n-wells 30
    uv run python scripts/run_stack_v34_cv.py --config b --n-wells 30

    # full 773-well run, ONE PROCESS PER SEED, strictly serial
    uv run python scripts/run_stack_v34_cv.py --config b --seed-idx 0
    ...
    uv run python scripts/run_stack_v34_cv.py --config b --seed-idx 4

    # after all 5 members have saved outputs/stack_v34_result_s*_cfgb.npz:
    uv run python scripts/run_stack_v34_cv.py --config b --aggregate
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
STACK_V2_ALLIN_POOLED_RMSE = 9.2309  # ledger 2026-07-07 (LB 9.022)
R13_5SEED_MEAN_POOLED_RMSE = 9.1456  # ledger 2026-07-10 (v26 5-seed mean, sub-v8)
R13_SEED42_POOLED_RMSE = 9.2515  # ledger 2026-07-11 (v26 seed-42 solo, R20/R22 base)
PF_BLEND_W = 0.7

FULL_ADOPTION_GATE_IMPROVEMENT_FT = 0.10
FULL_ADOPTION_GATE_RMSE = R13_5SEED_MEAN_POOLED_RMSE - FULL_ADOPTION_GATE_IMPROVEMENT_FT  # 9.0456
SEED42_SCREEN_IMPROVEMENT_FT = 0.05
SEED42_SCREEN_GATE_RMSE = R13_SEED42_POOLED_RMSE - SEED42_SCREEN_IMPROVEMENT_FT  # 9.2015
Y_TRUE_ATOL = 1e-2

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN for every member (matches v2/R7-R13)
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN for every member (matches v2/R7-R13)
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

CONFIGS: tuple[str, ...] = ("a", "b")
N_FEATURES_EXPECTED: dict[str, int] = {
    "a": 59,  # R13 repro: P1(36) + tracker(8) + well-agg(8) + spatial(4) + H3(3)
    "b": 63,  # (a) + pf_std time-series (4)
}


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
# H3 feature block, byte-copied from run_stack_v25/v26_cv.py (R12/R13). Reads
# only MD, the known-zone TVT_input, and Z. Never reads the train-only ANCC
# column or h["TVT"].
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
# pf_std timeseries "overconfidence lock-on" features (R24, config b only).
# Reads only the per-row pf_std array already cached by build_tracker_cache.py
# (same source build_tracker_features's raw pf_std column reads) -- no new
# I/O, no new leak surface. Every column is backward-looking only (expanding
# max, or a trailing window max/argmax ending at and including row i); never
# a centered or future window, so there is no eval-zone lookahead.
# --------------------------------------------------------------------------- #

PF_STD_TS_FEATURE_COLUMNS: tuple[str, ...] = (
    "pf_std_trailing_max_ratio",
    "pf_std_spike_decay",
    "pf_std_cummax",
    "pf_std_since_spike",
)

PF_STD_TS_WINDOW = 300  # trailing rows considered "recent" for spike_decay/since_spike
PF_STD_TS_EPS = 1e-3  # ft; floor for the current pf_std value in the ratio's denominator


def build_pf_std_ts_features(pf_std: np.ndarray, window: int = PF_STD_TS_WINDOW) -> pd.DataFrame:
    """R24: pf_std "spike-then-lock-on" time-series shape, 4 columns.

    Analysis C (scratchpad qualitative case study, well ``4c2208f5``) found
    that on the worst-SSE wells the per-seed PF variance ``pf_std`` spikes
    sharply (there: 0 -> ~8 ft) just before the model's error takes a step
    (the PF locking on to the wrong registration path), then collapses and
    stays low for the rest of the zone even though the prediction is now
    durably wrong. I.e. *a currently-low pf_std is not reliable evidence of a
    good fix* if pf_std was recently much higher. The existing pf_std
    features (``build_tracker_features``'s raw per-row ``pf_std``,
    ``build_well_aggregate_features``'s well-wide ``pf_std_well_mean``/
    ``pf_std_well_max``) are static snapshots and cannot express this shape.

    All four columns look only backward from row ``i`` (expanding max, or a
    trailing window of size ``window`` ending at and including row ``i``):

      - ``pf_std_cummax``: ``expanding_max(pf_std)[i]`` -- the highest pf_std
        seen up to and including row ``i`` (raw level information).
      - ``pf_std_trailing_max_ratio``: ``pf_std_cummax[i] / max(pf_std[i],
        eps)`` -- large when the well "spiked and is now low" (the lock-on
        signature); ~1.0 when pf_std is currently at (or near) its own
        running peak.
      - ``pf_std_spike_decay``: ``trailing_window_max(pf_std, window)[i] -
        pf_std[i]`` -- how much pf_std has collapsed from its local
        (last-``window``-row) peak, in raw ft units rather than a ratio.
      - ``pf_std_since_spike``: rows since the trailing-``window`` peak,
        normalized by ``eval_len`` (``n``) -- ``0.0`` if the peak is at the
        current row, up to ``(window-1)/n`` if the peak sits at the far edge
        of the window (saturates there by design -- this column asks "was
        there a *recent* spike", not "when was the well's all-time peak",
        which is what ``pf_std_cummax``/``pf_std_trailing_max_ratio`` answer).

    ``pf_std`` must be ``data.eval_mask(h)``-aligned (the same array
    ``build_tracker_features`` takes). Never raises: non-finite entries
    (``NaN``/``inf``, e.g. the PF's exception-fallback path) are treated as
    ``0.0`` before any rolling computation (matches this module's H3 block /
    ``build_well_aggregate_features``'s finite-or-default convention), and an
    empty ``pf_std`` yields a zero-row DataFrame with the same columns.
    """
    pf_std = np.asarray(pf_std, dtype=np.float64)
    n = pf_std.size
    if n == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in PF_STD_TS_FEATURE_COLUMNS})

    filled = np.where(np.isfinite(pf_std), pf_std, 0.0)

    cummax = np.maximum.accumulate(filled)
    trailing_max_ratio = cummax / np.maximum(filled, PF_STD_TS_EPS)

    w = min(window, n)
    pad = np.full(w - 1, -np.inf, dtype=np.float64)
    padded = np.concatenate([pad, filled])
    # right-aligned trailing window: windows[i] == padded[i : i + w] == filled
    # rows [i - w + 1 .. i] (clipped to the start of the array via -inf pad).
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)
    trailing_window_max = windows.max(axis=1)
    trailing_window_argmax = windows.argmax(axis=1)  # 0..w-1, w-1 == current row

    spike_decay = trailing_window_max - filled
    rows_since_peak = (w - 1) - trailing_window_argmax
    since_spike = rows_since_peak.astype(np.float64) / float(max(n, 1))

    out = pd.DataFrame(
        {
            "pf_std_trailing_max_ratio": trailing_max_ratio,
            "pf_std_spike_decay": spike_decay,
            "pf_std_cummax": cummax,
            "pf_std_since_spike": since_spike,
        }
    )
    out = out[list(PF_STD_TS_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# well-matrix builder (config a: 56 v2 cols + H3 = 59; config b: + pf_std-ts
# = 63), structure byte-copied from run_stack_v24/v25/v26_cv.py's two-pass
# preallocating build
# --------------------------------------------------------------------------- #


class _WellMatrix:
    """Container for the feature matrix every ensemble member trains on."""

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


def build_well_matrix_v34(wells: list[str], config: str, split: str = "train") -> _WellMatrix:
    """Two-pass, preallocating build of the config-``a``/``b`` feature matrix.

    ``config="a"`` reproduces R13 (v2's 56 + H3 = 59 cols) exactly.
    ``config="b"`` additionally appends :func:`build_pf_std_ts_features`'s 4
    columns (63 total). H1/H2 are never computed (R13 already dropped them).
    """
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}, got {config!r}")

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
            f"build_well_matrix_v34: 0 wells passed validation -- no rows to build "
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

        blocks = [p1_feats, trk_feats, well_agg_feats, spatial_feats, h3_feats]
        if config == "b":
            blocks.append(build_pf_std_ts_features(pf_std))

        if not col_arrays:
            all_cols = [c for block in blocks for c in block.columns]
            col_arrays = {c: np.empty(total_rows, dtype=np.float32) for c in all_cols}

        for block in blocks:
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
    n_expected = N_FEATURES_EXPECTED[config]
    if len(all_cols) != n_expected:
        raise AssertionError(
            f"R24 config {config!r} must train on exactly {n_expected} features, "
            f"got {len(all_cols)}"
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
    cols: list[str],
    config: str,
    seed: int,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
) -> dict[str, object]:
    """Run one ensemble member (config-``a``/``b``, given LGBM seed) to completion."""
    label = f"cfg{config} seed {seed}"
    print(f"\n=== {label} (R24 config {config}, n_features={len(cols)}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged since v1..v2.6
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
        "config": config,
        "label": label,
        "cols": cols,
        "oof": oof,
        "fold_rows": fold_rows,
        "importance_df": importance_df,
        "pooled_rmse": pooled,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _result_path(seed: int, config: str, n_wells: int | None) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    return OUTPUTS_DIR / f"stack_v34_result_s{seed}_cfg{config}{suffix}.npz"


def _save_result(
    result: dict[str, object], wm: _WellMatrix, row_fold: np.ndarray, n_wells: int | None
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(result["fold_rows"])
    imp_df = result["importance_df"]
    path = _result_path(int(result["seed"]), str(result["config"]), n_wells)
    np.savez_compressed(
        path,
        seed=np.int64(result["seed"]),
        config=str(result["config"]),
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


def _load_result(seed: int, config: str, n_wells: int | None) -> dict[str, object]:
    path = _result_path(seed, config, n_wells)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python scripts/run_stack_v34_cv.py "
            f"--config {config} --seed-idx {SEEDS.index(seed)}"
            + (f" --n-wells {n_wells}" if n_wells is not None else "")
            + "` first"
        )
    with np.load(path, allow_pickle=True) as npz:
        return {
            "seed": int(npz["seed"]),
            "config": str(npz["config"]),
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


def _aggregate(config: str, n_wells: int | None) -> None:
    results: list[dict[str, object]] = []
    for seed in SEEDS:
        try:
            results.append(_load_result(seed, config, n_wells))
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
    print(f"R24 config {config} ({len(ref['cols'])} features)")
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
        f"\n{len(results)}-seed mean OOF (R24 cfg {config}): pooled={mean_pooled:.4f}  "
        f"median={np.median(valid):.4f}  p90={np.percentile(valid, 90):.4f}  "
        f"max={np.max(valid):.4f}"
    )

    seed42 = next((r for r in results if r["seed"] == 42), None)
    if seed42 is not None:
        _helped_hurt_report(
            f"R24 cfg{config} {len(results)}-seed mean",
            y_true,
            oof_mean,
            seed42["oof"],
            f"R24 cfg{config} seed-42 solo ({seed42['pooled_rmse']:.4f})",
            well_idx,
            n_wells_used,
        )
        screen_pass = seed42["pooled_rmse"] <= SEED42_SCREEN_GATE_RMSE
        screen_msg = (
            "PASS (proceed to 5-seed full judgment)"
            if screen_pass
            else "FAIL (do not proceed on single seed alone -- report only, R15 lesson)"
        )
        print(
            f"\nseed-42 screen (cfg {config}): solo={seed42['pooled_rmse']:.4f}  "
            f"vs R13 seed42 {R13_SEED42_POOLED_RMSE:.4f} "
            f"(delta {seed42['pooled_rmse'] - R13_SEED42_POOLED_RMSE:+.4f}ft)  "
            f"screen gate <= {SEED42_SCREEN_GATE_RMSE:.4f}ft -> {screen_msg}"
        )

    out_path = OUTPUTS_DIR / f"stack_v34_oof_cfg{config}.npz"
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

    if len(results) == len(SEEDS):
        improvement_r13 = R13_5SEED_MEAN_POOLED_RMSE - mean_pooled
        verdict = (
            "採用候補"
            if mean_pooled <= FULL_ADOPTION_GATE_RMSE
            else "却下（非改善、実測値は報告のみ）"
        )
        print(f"\n{'=' * 96}")
        print(
            f"判定 (R24 cfg {config}, 5-seed): {verdict} -- 5-seed mean={mean_pooled:.4f}, "
            f"R13 baseline={R13_5SEED_MEAN_POOLED_RMSE:.4f} (改善 {improvement_r13:+.4f}ft), "
            f"gate <= {FULL_ADOPTION_GATE_RMSE:.4f}ft"
        )
        print(f"{'=' * 96}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        choices=list(CONFIGS),
        required=True,
        help="a = R13 repro (59 cols, regression guard). b = R13 + pf_std-ts (63 cols, R24).",
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
        help="Skip CV; load outputs/stack_v34_result_s*_cfg<config>.npz, print the table, "
        "write outputs/stack_v34_oof_cfg<config>.npz.",
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
    wm = build_well_matrix_v34(wells, args.config, split="train")
    gc.collect()
    print(
        f"feature matrix (R24 cfg {args.config}, {wm.X.shape[1]} cols): {wm.X.shape}, "
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
        result = run_one_seed(wm, cols, args.config, seed, row_fold, well_fold_arr)
        print(
            f"  -> cfg {args.config} seed {seed}: pooled RMSE={result['pooled_rmse']:.4f} "
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
