"""Stack v3.1: R13's 59 features + R17 SurfaceBank V4 diagnostics (issue: R17).

**Why R17.** The 2026-07-11 ledger's error-anatomy analysis (分析A/B) found
SSE decomposes as **55% per-well constant offset / 19.6% slope / 25.3%
shape**, and that the only *real* signal for the eval-zone slope is
``spatial_slope`` (r=0.49, a well-level plane-fit slope from the SurfaceBank)
-- donor raw slope is dead, known-zone trend extrapolation is dead (R16
confirmed this empirically), but the structural surface itself carries
signal because geology is spatially continuous. R17 asks: does sharpening
the SurfaceBank's local-surface estimate -- a genuine quadratic fit instead
of a plane, plus exposing the fit's own confidence/gradient as features --
give the R13 stack anything on top of the existing (IDW-based) ``spatial_d``
block?

**Design.** Three configs, identical LGBM/CV harness to R11-R16 (fold seed
42, ES-split seed 42, same base params, ``SEEDS=(42,101,202,303,404)``):

  - (a) = R13's 59 columns, byte-copied from ``run_stack_v26_cv.py``
    (**regression check** -- must reproduce 9.2515 solo seed42 / 9.1456
    5-seed mean).
  - (b) = (a) + 4 new columns from a **local-plane** SurfaceBank fit
    (``k=10``, unchanged from the existing IDW cache's ``k``,
    ``weight_power=1.0``): ``spatial_v2_resid_std`` (local fit confidence),
    ``spatial_v2_n_donors``, ``spatial_v2_grad_x``/``grad_y`` (per-row local
    dip vector, mirrors ``spatial_slope`` at row granularity). 63 columns.
  - (c) = (a) + the same 4 columns from a **quadratic-surface** fit
    (``k=12`` for more donors, ``weight_power=2.0`` for tighter local
    weighting -- V1+V2 combined). 63 columns.

(b)/(c) *add* the 4 diagnostic columns rather than *replacing* the existing
``spatial_d``/``spatial_prefix_rmse``/``spatial_nn_dist_median``/
``spatial_gated_d`` block (still sourced from the original, LB-transfer-
confirmed IDW cache, calibration point #3) -- ``run_spatial_cv.py``'s own
history found a raw local-plane fit has a *worse* heavy tail than IDW
(pooled 26.8 vs 23.0 on a 20-well subset), so replacing the proven IDW block
outright would be a much bigger, unvalidated bet than adding diagnostics
alongside it.

**Gate (two-phase, per the task brief).** Phase 1: each of (b)/(c) is run
**seed42 solo only** first and compared against R13's seed42 solo
(``R13_SEED42_SOLO = 9.2515``) with a ``-0.10`` gate
(``SEED42_GATE = 9.1515``). Only a config that clears this gate escalates to
the full 5-seed run, compared against R13's 5-seed mean
(``R13_5SEED_MEAN = 9.1456``, same ``-0.10`` adoption gate as R11-R16 ->
``ADOPTION_GATE = 9.0456``). Both are reported regardless of outcome
(negative results reported honestly, per project policy).

**Donor fold boundary.** V4's diagnostics are computed by
``scripts/build_spatial_cache_v2.py`` against a SurfaceBank built once from
*every* train well (not fold-train-only) -- the same (small, already-
diagnosed, LB-absolved) donor policy as the existing ``spatial_cache``. See
that script's docstring for the full argument; R17 does not change this.

**Leak boundary.** Identical to R13's for the 59 original columns. The new
V4 columns read only ``X``/``Y``/``Z`` (via the SurfaceBank, which itself
only ever reads *other* train wells' train-only formation-depth columns)
and the target well's own known-zone ``TVT_input`` (for offset calibration,
leak-safe per ``rogii.spatial``'s existing contract) -- no ``h["TVT"]``, no
eval-zone ``TVT_input``.

Usage::

    # smoke test (one config, all 5 seeds + aggregate in one process, ~30 wells)
    uv run python scripts/run_stack_v31_cv.py --config a --n-wells 30
    uv run python scripts/run_stack_v31_cv.py --config b --n-wells 30
    uv run python scripts/run_stack_v31_cv.py --config c --n-wells 30

    # phase 1 (seed42 solo, full 773 wells)
    uv run python scripts/run_stack_v31_cv.py --config a --seed-idx 0
    uv run python scripts/run_stack_v31_cv.py --config b --seed-idx 0
    uv run python scripts/run_stack_v31_cv.py --config c --seed-idx 0
    uv run python scripts/run_stack_v31_cv.py --compare --n-wells 30   # or full

    # phase 2 (5-seed, only for a config that cleared the phase-1 gate)
    uv run python scripts/run_stack_v31_cv.py --config b --seed-idx 1
    ...
    uv run python scripts/run_stack_v31_cv.py --config b --seed-idx 4
    uv run python scripts/run_stack_v31_cv.py --config b --aggregate
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
R13_SEED42_SOLO = 9.2515  # outputs/stack_v26_result_s42.npz (measured, this repo)
R13_5SEED_MEAN_POOLED_RMSE = 9.1456  # ledger 2026-07-10/11 (run_stack_v26_cv.py 5-seed mean)
PF_BLEND_W = 0.7

IMPROVEMENT_GATE_FT = 0.10
SEED42_GATE = R13_SEED42_SOLO - IMPROVEMENT_GATE_FT  # 9.1515 -- phase 1 (screening)
ADOPTION_GATE_RMSE = R13_5SEED_MEAN_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.0456 -- phase 2
Y_TRUE_ATOL = 1e-2

N_SPLITS = 5
FOLD_SEED = 42
ES_SPLIT_SEED = 42
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

# name -> (SurfaceBank v2 cache dir, method, k, weight_power) -- must match
# scripts/build_spatial_cache_v2.py's CONFIGS exactly.
V2_CONFIGS: dict[str, tuple[str, int, float]] = {
    "b": ("plane", 10, 1.0),
    "c": ("quadratic", 12, 2.0),
}
CONFIG_KEYS: tuple[str, ...] = ("a", "b", "c")
N_HARD_WELLS = 20


def _spatial_v2_cache_dir(config: str) -> Path:
    return _REPO_ROOT / "data" / "processed" / f"spatial_cache_v2_{config}"


def n_features_expected(config: str) -> int:
    return 59 if config == "a" else 63  # P1(36)+tracker(8)+well-agg(8)+spatial(4)+H3(3)[+V4(4)]


def _lgb_params_for_seed(seed: int) -> dict[str, object]:
    return {**LGB_PARAMS_BASE, "random_state": seed}


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _peak_rss_mb() -> float:
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

    Byte-copied recipe from ``run_stack_v26/v29_cv.py``; the SAME seed (42)
    used by ``scripts/build_spatial_cache_v2.py``'s own ``--n-wells`` sampler,
    so a smoke run here sees exactly the wells cached by that script's smoke
    build.
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
# only MD, the known-zone TVT_input, and Z. Never reads ANCC or h["TVT"].
# --------------------------------------------------------------------------- #


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
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

_H3_TREND_TAIL = 200


def build_h3_features(h: pd.DataFrame) -> pd.DataFrame:
    """H3: 3 columns from ``MD``, the known-zone ``TVT_input``, and ``Z`` only.

    Byte-copied from ``run_stack_v26_cv.py`` (R13) -- see that module for the
    full column definitions. Never raises; one row per eval-zone row.
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
# R17 V4 feature block: 4 new columns from the SurfaceBank local-fit
# diagnostics cached by build_spatial_cache_v2.py. Pure arithmetic over
# already-computed arrays -- no I/O, no ground-truth read.
# --------------------------------------------------------------------------- #

SPATIAL_V4_FEATURE_COLUMNS: tuple[str, ...] = (
    "spatial_v2_resid_std",
    "spatial_v2_n_donors",
    "spatial_v2_grad_x",
    "spatial_v2_grad_y",
)


def build_spatial_v4_features(
    resid_std: np.ndarray, n_donors: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray
) -> pd.DataFrame:
    """Turn one well's cached R17 surface-fit diagnostics into a feature block.

    ``resid_std``/``n_donors``/``grad_x``/``grad_y`` are the eval_mask-
    aligned arrays cached by ``scripts/build_spatial_cache_v2.py``
    (``rogii.spatial.predict_spatial_v2``). Never raises: non-finite inputs
    are replaced with ``0.0`` (LightGBM would otherwise see stray NaN/inf
    exactly where the original SurfaceBank fit had no valid donors, which is
    already communicated by ``spatial_gated_d``/``spatial_prefix_rmse`` in
    the existing 4-column block).
    """
    n = int(np.asarray(resid_std).size)
    if n == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in SPATIAL_V4_FEATURE_COLUMNS})

    def _clean(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        return np.where(np.isfinite(arr), arr, 0.0)

    out = pd.DataFrame(
        {
            "spatial_v2_resid_std": _clean(resid_std),
            "spatial_v2_n_donors": _clean(n_donors),
            "spatial_v2_grad_x": _clean(grad_x),
            "spatial_v2_grad_y": _clean(grad_y),
        }
    )
    return out[list(SPATIAL_V4_FEATURE_COLUMNS)].astype(np.float32)


# --------------------------------------------------------------------------- #
# well-matrix builder: R13's 59 cols, +4 V4 cols for config b/c
# --------------------------------------------------------------------------- #


class _WellMatrix:
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


def build_well_matrix_v31(wells: list[str], config: str, split: str = "train") -> _WellMatrix:
    """Two-pass, preallocating build of R13's 59 cols (+4 V4 cols for config b/c)."""
    use_v2 = config != "a"
    v2_dir = _spatial_v2_cache_dir(config) if use_v2 else None

    counts = {
        "n_no_eval_zone_or_anchor": 0,
        "n_tracker_cache_missing": 0,
        "n_tracker_len_mismatch": 0,
        "n_spatial_cache_missing": 0,
        "n_spatial_len_mismatch": 0,
        "n_spatial_v2_cache_missing": 0,
        "n_spatial_v2_len_mismatch": 0,
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

        if use_v2:
            v2_path = v2_dir / f"{well}.npz"  # type: ignore[union-attr]
            if not v2_path.exists():
                counts["n_spatial_v2_cache_missing"] += 1
                continue
            with np.load(v2_path) as npz:
                v2_size = int(npz["resid_std"].shape[0])
            if v2_size != n:
                counts["n_spatial_v2_len_mismatch"] += 1
                continue

        used_wells.append(well)
        row_counts.append(n)

    if not used_wells:
        raise ValueError(
            f"build_well_matrix_v31[{config}]: 0 wells passed validation (counts={counts})"
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
        if use_v2:
            with np.load(v2_dir / f"{well}.npz") as npz:  # type: ignore[union-attr]
                v4_feats = build_spatial_v4_features(
                    npz["resid_std"], npz["n_donors"], npz["grad_x"], npz["grad_y"]
                )
            blocks.append(v4_feats)

        if not col_arrays:
            for block in blocks:
                all_cols.extend(block.columns)
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
    expected = n_features_expected(config)
    if len(all_cols) != expected:
        raise AssertionError(
            f"R17 config {config} must train on exactly {expected} features, got {len(all_cols)}"
        )
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, counts
    )


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
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


def run_one_seed(
    wm: _WellMatrix,
    cols: list[str],
    config: str,
    seed: int,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
) -> dict[str, object]:
    label = f"config {config} seed {seed}"
    print(f"\n=== {label} (n_features={len(cols)}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged since v1
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


def _result_path(config: str, seed: int, n_wells: int | None) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    return OUTPUTS_DIR / f"stack_v31_result_{config}_s{seed}{suffix}.npz"


def _save_result(
    result: dict[str, object], wm: _WellMatrix, row_fold: np.ndarray, n_wells: int | None
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(result["fold_rows"])
    imp_df = result["importance_df"]
    path = _result_path(str(result["config"]), int(result["seed"]), n_wells)
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


def _load_result(config: str, seed: int, n_wells: int | None) -> dict[str, object]:
    path = _result_path(config, seed, n_wells)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python scripts/run_stack_v31_cv.py "
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
    n_wells_used = len(used_wells)
    for res in results[1:]:
        if not np.allclose(res["y_true"], y_true, atol=Y_TRUE_ATOL):
            raise AssertionError(
                f"seed {res['seed']}'s y_true disagrees with seed {ref['seed']}'s -- not the "
                "same wells list, aggregation would be meaningless"
            )

    cum_sum = np.zeros_like(y_true, dtype=np.float64)
    print()
    print("=" * 96)
    print(f"R17 config {config} ({len(ref['cols'])} features)")
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

    mean_well_rmse = _per_well_rmse(y_true, oof_mean, well_idx, n_wells_used)
    valid = mean_well_rmse[np.isfinite(mean_well_rmse)]
    print(
        f"\n{len(results)}-seed mean OOF (config {config}): pooled={mean_pooled:.4f}  "
        f"median={np.median(valid):.4f}  p90={np.percentile(valid, 90):.4f}  "
        f"max={np.max(valid):.4f}"
    )

    out_path = OUTPUTS_DIR / f"stack_v31_oof_{config}.npz"
    save_kwargs = {
        f"oof_{i}": np.asarray(res["oof"], dtype=np.float32) for i, res in enumerate(results)
    }
    np.savez_compressed(
        out_path,
        y_true=y_true.astype(np.float32),
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
    print(f"\nOOF saved: {out_path}")

    if len(results) == len(SEEDS):
        improvement = R13_5SEED_MEAN_POOLED_RMSE - mean_pooled
        verdict = (
            "採用候補" if mean_pooled <= ADOPTION_GATE_RMSE else "却下（非改善、実測値のみ報告）"
        )
        print(f"\n{'=' * 96}")
        print(
            f"判定 (config {config}, 5-seed): {verdict} -- mean={mean_pooled:.4f}, "
            f"R13 5-seed baseline={R13_5SEED_MEAN_POOLED_RMSE:.4f} (改善 {improvement:+.4f}ft), "
            f"gate <= {ADOPTION_GATE_RMSE:.4f}ft"
        )
        print(f"{'=' * 96}")


def _compare(n_wells: int | None) -> None:
    """Phase-1 decision table: each config's seed42 solo vs R13 seed42 solo,
    plus the top-N_HARD_WELLS hardest wells (by config a's per-well RMSE) and
    how b/c's per-well RMSE moves on those SAME wells."""
    loaded: dict[str, dict[str, object]] = {}
    for config in CONFIG_KEYS:
        try:
            loaded[config] = _load_result(config, 42, n_wells)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")

    if "a" not in loaded:
        print("config a (control) missing -- cannot compare (need it for the hard-well baseline)")
        return

    print()
    print("=" * 96)
    print("R17 phase-1 decision table (seed42 solo)")
    print(f"{'config':>8}{'n_features':>12}{'pooled':>12}{'vs R13 9.2515':>16}{'gate<=9.1515':>14}")
    print("-" * 96)
    for config in CONFIG_KEYS:
        if config not in loaded:
            continue
        res = loaded[config]
        pooled = float(res["pooled_rmse"])
        passed = "PASS" if pooled <= SEED42_GATE else "fail"
        print(
            f"{config:>8}{len(res['cols']):>12}{pooled:>12.4f}"
            f"{pooled - R13_SEED42_SOLO:>+16.4f}{passed:>14}"
        )
    print("=" * 96)

    ref = loaded["a"]
    y_true = ref["y_true"]
    well_idx = ref["well_idx"]
    used_wells = ref["used_wells"]
    n_wells_used = len(used_wells)

    base_well_rmse = _per_well_rmse(y_true, ref["oof"], well_idx, n_wells_used)
    finite = np.isfinite(base_well_rmse)
    order = np.argsort(-np.where(finite, base_well_rmse, -np.inf))
    hard_idx = [int(i) for i in order[: min(N_HARD_WELLS, n_wells_used)] if finite[i]]

    print(f"\nhard wells (top {len(hard_idx)} by config a per-well RMSE):")
    header = f"{'well':>10}{'a (base)':>12}"
    other_configs = [c for c in ("b", "c") if c in loaded and c != "a"]
    for c in other_configs:
        header += f"{c:>12}{'delta':>10}"
    print(header)

    other_well_rmse: dict[str, np.ndarray] = {}
    for c in other_configs:
        res_c = loaded[c]
        if res_c["used_wells"] != used_wells:
            print(f"  NOTE: config {c}'s used_wells differs from config a's -- skipping delta")
            continue
        other_well_rmse[c] = _per_well_rmse(y_true, res_c["oof"], res_c["well_idx"], n_wells_used)

    for i in hard_idx:
        row = f"{used_wells[i]:>10}{base_well_rmse[i]:>12.4f}"
        for c in other_configs:
            if c in other_well_rmse:
                v = other_well_rmse[c][i]
                row += f"{v:>12.4f}{v - base_well_rmse[i]:>+10.4f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIG_KEYS, default=None)
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
        help="Skip CV; load outputs/stack_v31_result_{config}_s*.npz, print the table.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Phase-1 decision table across configs a/b/c's seed42-solo results.",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.compare:
        _compare(args.n_wells)
        print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
        return

    if args.config is None:
        parser.error("--config is required unless --compare is given")
    config = args.config

    if args.aggregate:
        _aggregate(config, args.n_wells)
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
    wm = build_well_matrix_v31(wells, config, split="train")
    gc.collect()
    print(
        f"feature matrix (config {config}, {wm.X.shape[1]} cols): {wm.X.shape}, "
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
        result = run_one_seed(wm, cols, config, seed, row_fold, well_fold_arr)
        print(
            f"  -> config {config} seed {seed}: pooled RMSE={result['pooled_rmse']:.4f} "
            f"peak_rss={result['peak_rss_mb']:.0f}MB"
        )
        _save_result(result, wm, row_fold, args.n_wells)
        del result
        gc.collect()

    if args.seed_idx is None:
        _aggregate(config, args.n_wells)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
    print(f"this process's peak RSS: {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
