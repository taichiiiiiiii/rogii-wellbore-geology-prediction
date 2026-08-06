"""Stack v2.3: LGBM seed-ensemble of stack v2.2's config (c) (issue: R9).

**Why R9.** R8 (``scripts/run_stack_v22_cv.py``) established that config (c)
= R7 + G1 + G2 (66 features) is the best feature set to date -- pooled OOF
**8.9152** vs R7's 8.9805 -- but the -0.0653ft gain missed the adoption gate
(<= 8.8805, R7 - 0.10), so R8 alone was recorded-but-rejected. The strategy
doc's P3 lever (2026-07-10, Sec 3) is the natural complement: a **seed
ensemble of the same model family** ("Tucker's equal-weight ensemble
5.4 -> 5.0" is the same shape) -- unlike output blending of *different*
candidate families (sub-v4, frozen), averaging the same LightGBM stack over
its own training randomness carries minimal transfer risk, because every
member is the identical architecture/features/folds, differing only in
bagging/feature-subsample draws. R9 asks whether variance reduction over
LGBM's training randomness moves 8.9152 past the 8.8805 gate.

**Design.**

  - Feature set: exactly v2.2 config (c) -- v2's 56 columns + R7's 4
    (:data:`NEW_FEATURE_COLUMNS`) + G1's 4 (:data:`G1_FEATURE_COLUMNS`) +
    G2's 2 (:data:`G2_FEATURE_COLUMNS`) = 66. G3 (rejected in R8) is not
    carried. Column construction is byte-copied from ``run_stack_v22_cv.py``.
  - :data:`SEEDS` = (42, 101, 202, 303, 404). Each seed runs the **full**
    5-fold well-GroupKFold CV with ``random_state=seed`` in the LightGBM
    params -- LightGBM's ``seed`` deterministically derives its
    ``bagging_seed``/``feature_fraction_seed``/``data_random_seed`` etc.
    when those are not set explicitly (they are not, matching v1/v2/R7/R8),
    so varying ``random_state`` alone varies all training randomness.
  - **Everything else is frozen** (task spec item 5): the fold assignment
    (``cv.well_folds(wells, 5, seed=FOLD_SEED)``) and the ES-holdout well
    split (``_split_fit_es(..., ES_SPLIT_SEED)``) use the same fixed seed 42
    for every member -- only the LGBM training randomness moves, otherwise
    the seed comparison would be confounded with split luck.
  - Seed **42 is deliberately first**: with ``random_state=42`` and the
    frozen splits, member 0 is parameter-identical to R8's config (c) run,
    so its pooled OOF must reproduce **8.9152** (+/- 0.01, multi-threaded
    LightGBM float nondeterminism) -- a built-in regression check that the
    R9 harness didn't silently change the model (same idiom as R8's (a)).
  - Report: each seed's solo pooled OOF, and the **cumulative-mean** pooled
    OOF after averaging members 1..k (k = 1..5) -- the diminishing-returns
    curve the task asks for. Gate on the final 5-seed mean: **<= 8.8805**.

**Memory.** Config (c) peaked at **3125MB RSS** in R8 on this 3.8GB box
(measured, ``ru_maxrss``), which is why seeds run **strictly serially, one
OS process each** (``--seed-idx k`` runs one member and exits; the driver
shell launches the next only after the previous exits -- never in parallel,
per the task spec). All of v2.2's memory-safety engineering is carried
verbatim: two-pass preallocating :func:`build_well_matrix_v2` (no
``pd.concat`` assembly), in-place per-column feature augmentation, explicit
``del`` + ``gc.collect()`` at each large-intermediate death point, measured
peak-RSS printouts. Per-seed results are flushed to
``outputs/stack_v23_result_s{seed}.npz`` immediately on completion
(checkpointing -- a crash loses at most the in-flight member).

**Leak boundary.** Identical to R8: all features come from candidate
predictions/statistics (bank npz files) + the same ``pf_blend`` basis; no
``h["TVT"]``/eval-zone ``TVT_input`` reads; alignment via
:func:`block_slices`/:func:`reindex_to_master` with y_true
cross-check-or-raise. Averaging OOF predictions across seeds adds no new
information source at all -- every member sees exactly the same data.

Usage::

    # smoke test (all 5 seeds + aggregate in one process, ~40 wells)
    uv run python scripts/run_stack_v23_cv.py --n-wells 40

    # full 773-well run, ONE PROCESS PER SEED, strictly serial
    uv run python scripts/run_stack_v23_cv.py --seed-idx 0
    ...
    uv run python scripts/run_stack_v23_cv.py --seed-idx 4

    # after all 5 members have saved outputs/stack_v23_result_s*.npz:
    uv run python scripts/run_stack_v23_cv.py --aggregate
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
PFBMA_V2_BANK_PATH = OUTPUTS_DIR / "path_bank_pfbma_v2.npz"
BEAM_BANK_PATH = OUTPUTS_DIR / "path_bank_beam.npz"
STACK_V21_OOF_PATH = OUTPUTS_DIR / "stack_v21_oof.npz"

PFBMA_V2_VALUE_KEY = "pf_bma_scale3_tvt"
BEAM_GRID_CONFIG_LABEL = "mid_cal_sm2_mm3"  # config index 4, R7's beam_grid_d
STACK_V21_R7_LABEL = "all-in + pfbma_v2/beam_grid (R7)"

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
STACK_V21_R7_POOLED_RMSE = 8.9805  # ledger 2026-07-10 (R7, adopted)
STACK_V22_C_POOLED_RMSE = 8.9152  # ledger 2026-07-10 (R8 config (c), best-but-rejected)
PF_BLEND_W = 0.7

IMPROVEMENT_GATE_FT = 0.10
ADOPTION_GATE_RMSE = STACK_V21_R7_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 8.8805, same gate as R8
Y_TRUE_ATOL = 1e-2
REPRO_ATOL = 0.01  # seed-42 member vs R8 config (c) 8.9152, full scale only

N_SPLITS = 5
FOLD_SEED = 42  # cv.well_folds seed -- FROZEN for every member (task spec item 5)
ES_SPLIT_SEED = 42  # _split_fit_es seed -- FROZEN for every member (task spec item 5)
N_STRATA = 10
ES_FRAC = 0.15

# The ONLY thing that varies across ensemble members (see module docstring).
# 42 first = parameter-identical to R8 config (c) -> built-in regression check.
SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404)

# Identical to v1/v2/R7/R8 except random_state, which is injected per member
# by _lgb_params_for_seed(). LightGBM derives bagging_seed /
# feature_fraction_seed / data_random_seed from `seed` when (as here) they
# are not set explicitly, so this single knob varies all training randomness.
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

# R7's 4 features (byte-identical definitions to run_stack_v21/v22_cv.py).
NEW_FEATURE_COLUMNS: tuple[str, ...] = (
    "pfbma_v2_d",
    "beam_grid_d",
    "pfbma_v2_minus_pf",
    "pfbma_v2_d_well_mean",
)

# R8's G1/G2 (byte-identical definitions to run_stack_v22_cv.py). G3 was
# rejected in R8 and is not carried into v2.3.
PFBMA_V2_SCALE5_KEY = "pf_bma_scale5_tvt"
PFBMA_V2_SCALE8_KEY = "pf_bma_scale8_tvt"
PFBMA_V2_SCALE12_KEY = "pf_bma_scale12_tvt"
G1_FEATURE_COLUMNS: tuple[str, ...] = (
    "pfbma_v2_s5_d",
    "pfbma_v2_s8_d",
    "pfbma_v2_s12_d",
    "pfbma_v2_scale_range",
)
PFBMA_V2_SCALE3_STD_KEY = "pf_bma_scale3_std"
G2_FEATURE_COLUMNS: tuple[str, ...] = (
    "pfbma_v2_s3_std",
    "pfbma_v2_s3_std_well_mean",
)

N_FEATURES_EXPECTED = 66  # 44 (P1+tracker) + 8 well-agg + 4 spatial + 4 R7 + 4 G1 + 2 G2


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

    Same recipe as ``run_stack_v2/v21/v22_cv.py`` (not imported -- ``scripts/``
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
# alignment primitives + feature builders, byte-copied from run_stack_v22_cv.py
# (which itself carries them from run_stack_v21_cv.py / run_router_v2_cv.py).
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

    Byte-copied from ``run_stack_v22_cv.py``'s memory-safe re-engineering
    (see that script's docstring for the measured 3.0GB-with-swap failure the
    original single-pass accumulate-then-``pd.concat`` version caused on
    this 3.8GB box): pass 1 validates wells + collects row counts, pass 2
    writes every feature block straight into pre-sized arrays.
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
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, p1_cols, counts
    )


def _master_bounds(wm: _WellMatrix) -> tuple[list[str], np.ndarray]:
    """Shared (master_wells, master_boundaries) precondition for bank reindexing."""
    n_wells = len(wm.used_wells)
    master_wells = list(wm.used_wells)
    master_boundaries = np.searchsorted(wm.well_idx, np.arange(n_wells + 1)).astype(np.int64)
    if master_boundaries[-1] != wm.well_idx.shape[0]:
        raise ValueError(
            "wm.well_idx blocks are not contiguous/sorted -- alignment precondition violated"
        )
    return master_wells, master_boundaries


def _load_bank_array(
    path: Path,
    value_key: str,
    master_wells: list[str],
    master_boundaries: np.ndarray,
    wm_y_true: np.ndarray,
) -> np.ndarray:
    """Reindex one ``value_key`` array from a bank npz onto the master row order.

    Raises (never silently misaligns) if any well is missing after reindex or
    the bank's own ``y_true`` disagrees with ours.
    """
    with np.load(path) as npz:
        well_idx = npz["well_idx"]
        used_wells = npz["used_wells"]
        y_true = npz["y_true"].astype(np.float32)
        val = npz[value_key].astype(np.float32)
    slices = block_slices(well_idx, used_wells)
    row = reindex_to_master(val, slices, master_wells, master_boundaries)
    y_row = reindex_to_master(y_true, slices, master_wells, master_boundaries)

    if np.isnan(row).any():
        raise AssertionError(
            f"{int(np.isnan(row).sum())} row(s) missing {value_key} after reindex from "
            f"{path.name} -- a well in wm.used_wells is absent from that bank"
        )
    if not np.allclose(y_row, wm_y_true, atol=Y_TRUE_ATOL):
        raise AssertionError(
            f"{path.name} y_true disagrees with stack v2.3's own y_true after reindex "
            f"(value_key={value_key}) -- alignment bug, not a leak, but must not be "
            "silently ignored"
        )
    return row


def add_config_c_columns(wm: _WellMatrix) -> list[str]:
    """Augment ``wm.X`` in place with R7's 4 + G1's 4 + G2's 2 columns; return the 66-col list.

    Column definitions are byte-identical to ``run_stack_v22_cv.py``'s
    :func:`augment_with_bank_deltas` / :func:`build_g1_features` /
    :func:`build_g2_features`, fused here into one in-place pass (config (c)
    is the only feature set v2.3 ever trains, so the group-at-a-time
    ablation machinery is not carried). In-place per-column assignment, not
    ``pd.concat`` -- see ``run_stack_v22_cv.py``'s memory notes.
    """
    master_wells, master_boundaries = _master_bounds(wm)
    pf_blend_f64 = wm.pf_blend_arr.astype(np.float64)

    # --- R7's 4 columns ---
    pfbma_row = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_VALUE_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)
    with np.load(BEAM_BANK_PATH) as npz:
        beam_labels = [str(x) for x in npz["config_labels"]]
    beam_cfg_i = beam_labels.index(BEAM_GRID_CONFIG_LABEL)
    beam_row = _load_bank_array(
        BEAM_BANK_PATH, f"beam_tvt_{beam_cfg_i}", master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)

    pf_tvt_arr = wm.X["pf_d"].to_numpy(dtype=np.float64) + wm.anchor_arr.astype(np.float64)
    pfbma_v2_d = pfbma_row - pf_blend_f64
    r7_block = pd.DataFrame(
        {
            "pfbma_v2_d": pfbma_v2_d,
            "beam_grid_d": beam_row - pf_blend_f64,
            "pfbma_v2_minus_pf": pfbma_row - pf_tvt_arr,
            "pfbma_v2_d_well_mean": pd.Series(pfbma_v2_d)
            .groupby(wm.well_idx)
            .transform("mean")
            .to_numpy(),
        }
    )
    r7_block = r7_block[list(NEW_FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    for col in NEW_FEATURE_COLUMNS:
        wm.X[col] = r7_block[col].to_numpy(dtype=np.float32)
    del r7_block, beam_row, pf_tvt_arr, pfbma_v2_d
    gc.collect()

    # --- G1: pf_bma_v2 scale-5/8/12 deltas + per-row scale range ---
    scale5_tvt = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_SCALE5_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)
    scale8_tvt = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_SCALE8_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)
    scale12_tvt = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_SCALE12_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)

    # scale3 is deliberately RECOVERED from the float32-rounded pfbma_v2_d
    # column (+ pf_blend), not taken from the raw bank array -- that is
    # exactly what R8's build_g1_features did (it read wm.X["pfbma_v2_d"]),
    # and the seed-42 member must be bit-input-identical to R8's config (c)
    # for the built-in 8.9152 regression check to be meaningful.
    scale3_tvt = wm.X["pfbma_v2_d"].to_numpy(dtype=np.float64) + pf_blend_f64
    all_scales = np.stack([scale3_tvt, scale5_tvt, scale8_tvt, scale12_tvt], axis=1)
    g1_block = pd.DataFrame(
        {
            "pfbma_v2_s5_d": scale5_tvt - pf_blend_f64,
            "pfbma_v2_s8_d": scale8_tvt - pf_blend_f64,
            "pfbma_v2_s12_d": scale12_tvt - pf_blend_f64,
            "pfbma_v2_scale_range": all_scales.max(axis=1) - all_scales.min(axis=1),
        }
    )
    g1_block = g1_block[list(G1_FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    for col in G1_FEATURE_COLUMNS:
        wm.X[col] = g1_block[col].to_numpy(dtype=np.float32)
    del g1_block, all_scales, scale3_tvt, scale5_tvt, scale8_tvt, scale12_tvt, pfbma_row
    gc.collect()

    # --- G2: pf_bma_v2 scale-3 uncertainty (raw + finite-only well-mean) ---
    s3_std = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_SCALE3_STD_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)
    finite_or_nan = np.where(np.isfinite(s3_std), s3_std, np.nan)
    well_mean = pd.Series(finite_or_nan).groupby(wm.well_idx).transform("mean").to_numpy()
    g2_block = pd.DataFrame(
        {"pfbma_v2_s3_std": s3_std, "pfbma_v2_s3_std_well_mean": well_mean}
    )
    g2_block = g2_block[list(G2_FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    for col in G2_FEATURE_COLUMNS:
        wm.X[col] = g2_block[col].to_numpy(dtype=np.float32)
    del g2_block, s3_std, finite_or_nan, well_mean
    gc.collect()

    cols = list(wm.X.columns)
    if len(cols) != N_FEATURES_EXPECTED:
        raise AssertionError(
            f"config (c) must have exactly {N_FEATURES_EXPECTED} features, got {len(cols)}"
        )
    return cols


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Byte-for-byte identical to ``run_stack_v2/v21/v22_cv.py``'s
    ``_split_fit_es``. v2.3 always calls it with ``ES_SPLIT_SEED`` (42,
    frozen) -- NEVER the member's LGBM seed (task spec item 5).
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

    Identical protocol to ``run_stack_v22_cv.py``'s ``run_stack_fold_cv``
    with ES-holdout always on (config (c) always used it), except
    ``lgb_params`` is a parameter so each member injects its own
    ``random_state`` -- the ES split itself stays on ``ES_SPLIT_SEED``.
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
    """Run one ensemble member (config (c), given LGBM seed) to completion."""
    label = f"seed {seed}"
    print(f"\n=== {label} (config (c), n_features={len(cols)}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1..v2.2
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
    return OUTPUTS_DIR / f"stack_v23_result_s{seed}{suffix}.npz"


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
            f"{path} not found -- run `uv run python scripts/run_stack_v23_cv.py "
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


def _report_r7_helped_hurt(
    label: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    well_idx: np.ndarray,
    used_wells: list[str],
) -> None:
    """Diagnostic helped/hurt vs stack v2.1's R7 OOF (reindexed). Never crashes the run."""
    try:
        if not STACK_V21_OOF_PATH.exists():
            return
        n_wells_used = len(used_wells)
        with np.load(STACK_V21_OOF_PATH, allow_pickle=True) as npz:
            r7_labels = [str(x) for x in npz["config_labels"]]
            r7_well_idx = npz["well_idx"]
            r7_used_wells = npz["used_wells"]
            r7_y_true = npz["y_true"].astype(np.float32)
            r7_i = r7_labels.index(STACK_V21_R7_LABEL)
            r7_oof_raw = npz[f"oof_{r7_i}"].astype(np.float32)
        master_boundaries = np.searchsorted(well_idx, np.arange(n_wells_used + 1)).astype(np.int64)
        r7_slices = block_slices(r7_well_idx, r7_used_wells)
        r7_oof_row = reindex_to_master(r7_oof_raw, r7_slices, used_wells, master_boundaries)
        r7_y_row = reindex_to_master(r7_y_true, r7_slices, used_wells, master_boundaries)
        if np.isnan(r7_oof_row).any() or not np.allclose(r7_y_row, y_true, atol=Y_TRUE_ATOL):
            return
        _helped_hurt_report(
            label, y_true, pred, r7_oof_row, "stack v2.1 R7 OOF (8.9805)", well_idx, n_wells_used
        )
    except (KeyError, ValueError, IndexError) as exc:
        print(f"\n(skipped R7-OOF helped/hurt diagnostic: {exc})")


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
    print("=" * 96)
    print(
        f"{'k':>3}{'seed':>7}{'solo pooled':>14}{'cum-mean pooled (1..k)':>24}"
        f"{'vs R8(c) 8.9152':>17}{'vs R7 8.9805':>14}"
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
            f"{cum_pooled - STACK_V22_C_POOLED_RMSE:>+17.4f}"
            f"{cum_pooled - STACK_V21_R7_POOLED_RMSE:>+14.4f}"
        )
    print("=" * 96)
    oof_mean = cum_sum / len(results)
    mean_pooled = cum_pooled_list[-1]

    print(f"\nreference: carry_last={floor_pooled:.4f}  PF blend w0.7={pf_pooled:.4f}")
    print(f"total rows scored: {len(y_true):,}  members aggregated: {len(results)}/{len(SEEDS)}")

    # regression check: seed-42 member == R8 config (c) rerun (full scale only)
    seed42 = next((r for r in results if r["seed"] == 42), None)
    if seed42 is not None:
        diff = seed42["pooled_rmse"] - STACK_V22_C_POOLED_RMSE
        if n_wells is not None:
            print(
                f"\nseed-42 repro check (n_wells={n_wells} subsample, informational only): "
                f"{seed42['pooled_rmse']:.4f} vs R8(c) ledger 8.9152 (diff {diff:+.4f}ft)"
            )
        else:
            ok = abs(diff) <= REPRO_ATOL
            print(
                f"\nseed-42 repro check: {seed42['pooled_rmse']:.4f} vs R8(c) ledger 8.9152 "
                f"(diff {diff:+.4f}ft, tol {REPRO_ATOL}ft) -> "
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
    _report_r7_helped_hurt(
        f"{len(results)}-seed mean", y_true, oof_mean, well_idx, used_wells
    )

    out_path = OUTPUTS_DIR / "stack_v23_oof.npz"
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

    improvement = STACK_V21_R7_POOLED_RMSE - mean_pooled
    verdict = "採用" if mean_pooled <= ADOPTION_GATE_RMSE else "却下（非改善）"
    print(f"\n{'=' * 96}")
    print(
        f"判定: {verdict} -- {len(results)}-seed mean={mean_pooled:.4f}, "
        f"R8(c) solo={STACK_V22_C_POOLED_RMSE:.4f}, R7={STACK_V21_R7_POOLED_RMSE:.4f}, "
        f"R7比改善={improvement:+.4f}ft (gate: <= {ADOPTION_GATE_RMSE:.4f}ft)"
    )
    print(f"{'=' * 96}")


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
        help="Skip CV entirely; load all outputs/stack_v23_result_s*.npz, print the "
        "per-seed + cumulative-mean table and gate verdict, and write "
        "outputs/stack_v23_oof.npz.",
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
        f"feature matrix (v2, 56 cols): {wm.X.shape}, rows={len(wm.y_true):,}, "
        f"wells_used={len(wm.used_wells)}/{len(wells)} counts={wm.counts} "
        f"({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )

    t0 = time.time()
    cols = add_config_c_columns(wm)
    print(
        f"config (c) augmentation: -> {wm.X.shape} ({time.time() - t0:.1f}s, "
        f"y_true cross-checks PASSED) peak_rss={_peak_rss_mb():.0f}MB"
    )

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
