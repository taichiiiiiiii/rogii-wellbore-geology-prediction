"""Stack v2.2: greedy addition of the bank's remaining candidates as features (issue: R8).

**Why R8.** The 2026-07-10 strategy doc (Sec 2.1, "特徴注入") locks in the
architecture R7 validated: every new candidate/information source is fed to
the same LightGBM stack as a ``*_d`` (delta-from-``pf_blend``) feature and
judged by well-GroupKFold pooled OOF, never blended into the output. R7 added
4 such features (``pfbma_v2_d``/``beam_grid_d``/``pfbma_v2_minus_pf``/
``pfbma_v2_d_well_mean``) and moved stack v2's 9.2309 to **8.9805**. R8 asks
whether the *rest* of the bank -- scales the R7 delta didn't touch, PF-BMA's
own uncertainty, and a handful of maximally-diverse beam configs -- is worth
adding on top, via a strict greedy group-at-a-time ablation.

**Three new feature groups**, all built from already-computed candidate
*predictions*/*statistics* already sitting in ``outputs/path_bank_pfbma_v2.npz``
(``init_pos_std=2.0`` bank) and ``outputs/path_bank_beam.npz`` (10-config beam
grid) -- see :data:`G1_FEATURE_COLUMNS`/:data:`G2_FEATURE_COLUMNS`/
:data:`G3_FEATURE_COLUMNS` for the exact column lists and
:func:`build_g1_features`/:func:`build_g2_features`/:func:`build_g3_features`
for how each is computed:

  - **G1** (4 cols): the pf_bma_v2 scales R7 didn't use (5/8/12, R7 only used
    scale 3) as deltas from ``pf_blend``, plus the per-row range across all 4
    scales (max-min) -- "how much does PF-BMA's answer change with its own
    smoothing scale" is new information R7 never exposed to the GBDT.
  - **G2** (2 cols): PF-BMA's own uncertainty (``pf_bma_scale3_std``, the
    scale R7's ``pfbma_v2_d`` already uses) as a raw per-row column plus its
    well-mean -- R7 fed pf_bma_v2's point *estimate* but never its declared
    *confidence*, unlike the tracker-PF's ``pf_std``/``pf_std_is_inf``
    (``rogii.stack_features.build_tracker_features``) which the stack already
    trusts.
  - **G3** (5 cols): 4 beam-grid deltas from configs chosen to be maximally
    far apart in (product, max_move, gr_calibration, smooth_radius) parameter
    space -- an exhaustive max-min-pairwise-distance search over all
    C(9,4)=126 combinations of the 9 configs *not* already used by R7's
    ``beam_grid_d`` (config 4, ``mid_cal_sm2_mm3``) picked configs
    **1/2/8/9** (``loose_cal_sm0_mm3``/``stiff_cal_sm0_mm3``/
    ``loose_nocal_sm5_mm2``/``stiff_cal_sm5_mm1``, min pairwise normalized
    distance 3.125 -- see the module's development notes) -- plus the
    per-row std across those 4 raw beam_tvt values, a cheap disagreement
    signal the single-config ``beam_grid_d`` cannot express.

**Ablation, strictly additive, gate ``<= 8.8805`` (R7 - 0.10):**

  - (a) R7 repro -- regression check, must reproduce 8.9805 +/- 0.01 (same
    columns as ``run_stack_v21_cv.py``'s final row, computed via the
    byte-copied :func:`augment_with_bank_deltas`).
  - (b) R7 + G1
  - (c) R7 + G1 + G2
  - (d) R7 + G1 + G2 + G3

**Memory.** The task's ``~2.8GB`` figure for R7 turned out to be optimistic
for *this* run: a live measurement (below) showed the *unmodified,
byte-copied* ``build_well_matrix_v2`` alone -- before any R8 column, before
even R7's own 4 columns -- peaking at **2.1GB RSS** for the 773-well matrix,
and a process running just the (a) R7-repro config (zero new R8 columns)
reaching **3.0GB RSS with 523MB already swapped** against only ~2.7GB
available on this box, i.e. a real near-OOM caught mid-run, not a
hypothetical one. Mitigations, in the order they were needed:

  1. **Process isolation** -- ``--config {a,b,c,d}`` runs *one* ablation row
     and exits, so the OS reclaims all of that process's RSS before the next
     config starts. The full 773-well run is driven as 4 separate ``uv run``
     invocations (see Usage below), never 4 configs in one process -- that
     all-in-one-process path only exists for ``--n-wells`` smoke tests.
  2. **No ``pd.concat``-based assembly, anywhere new columns are added.**
     ``pd.concat`` allocates an entirely new block while its inputs stay
     alive (~2x peak). Three call sites were rewritten to avoid it: (a)
     :func:`build_well_matrix_v2` was re-engineered into a two-pass
     validate-then-preallocate design (see its docstring) instead of
     "append 773 small frames to a list, concat once" -- this alone was the
     single largest contributor to the observed 3.0GB peak; (b) R7's 4-column
     augmentation in ``main()`` now assigns each column onto ``wm.X`` in
     place (``wm.X[col] = array``) instead of concatenating; (c)
     :func:`_cfg_columns` does the same for G1/G2/G3, exploiting that
     :data:`ABLATION_GROUPS` is strictly cumulative (a subset-of b
     subset-of c subset-of d) so in-place column assignment is always safe,
     never overwrites/reorders an already-added group.
  3. **Explicit ``del`` + ``gc.collect()``** at every point a large
     intermediate becomes dead (post-build, post-augmentation, post-fold,
     post-group-assignment) so Python's refcounting reclaim isn't left to
     chance/timing. Peak RSS is printed at each of these checkpoints.

  ``LGB_PARAMS``/fold assignment/ES-holdout method are copied byte-for-byte
  from ``run_stack_v21_cv.py`` per the task spec ("CV設定はR7と完全同一...
  触らない") -- ``max_bin`` is deliberately left at LightGBM's default
  (unset in v1/v2/R7's ``LGB_PARAMS`` too, so "keep max_bin=127" from the
  task's memory-tactics list does not apply here; noted, not silently
  invented). Peak RSS is measured (``resource.getrusage(...).ru_maxrss``,
  never estimated) and printed at the end of every run.

**Alignment / leak boundary.** Identical story to R7: G1/G2/G3 are built
entirely from already-computed candidate predictions/statistics and the same
``pf_blend`` basis the stack's own PF-residual target uses -- none of them
reads ``h["TVT"]`` or the eval-zone ``TVT_input``. Every bank array is
reindexed onto this script's own master row order via the byte-copied
:func:`block_slices`/:func:`reindex_to_master`, followed by a ``y_true``
cross-check-or-raise (:func:`_load_bank_array`, generalizing R7's inline
per-array checks) -- never a silent misalignment. The PF-BMA/beam trackers
are themselves fold-independent physical computations, so there is no OOF
leak from adding them as features (same reasoning as R7).

Usage::

    # smoke test (health check, all 4 configs in one process, ~40 wells)
    uv run python scripts/run_stack_v22_cv.py --n-wells 40

    # full 773-well run, ONE PROCESS PER CONFIG (memory isolation)
    uv run python scripts/run_stack_v22_cv.py --config a --save-oof
    uv run python scripts/run_stack_v22_cv.py --config b --save-oof
    uv run python scripts/run_stack_v22_cv.py --config c --save-oof
    uv run python scripts/run_stack_v22_cv.py --config d --save-oof

    # after all 4 configs above have run and saved outputs/stack_v22_result_*.npz:
    uv run python scripts/run_stack_v22_cv.py --aggregate
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
    FEATURE_COLUMNS,
    SPATIAL_FEATURE_COLUMNS,
    WELL_FEATURE_COLUMNS,
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
STACK_V2_OOF_PATH = OUTPUTS_DIR / "stack_v2_oof.npz"
STACK_V21_OOF_PATH = OUTPUTS_DIR / "stack_v21_oof.npz"

PFBMA_V2_VALUE_KEY = "pf_bma_scale3_tvt"  # pf_bma_scale3_v2, from the init_pos_std=2.0 bank
BEAM_GRID_CONFIG_LABEL = "mid_cal_sm2_mm3"  # config index 4 -> "beam:mid_cal_sm2_mm3" in the ledger
STACK_V2_ALL_IN_LABEL = "all-in (levers 1+2+3)"
STACK_V21_R7_LABEL = "all-in + pfbma_v2/beam_grid (R7)"

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
V1_STACK_B_POOLED_RMSE = 10.6513  # scripts/run_stack_cv.py (b) PF-residual, ledger 2026-07-06
STACK_V2_ALL_IN_POOLED_RMSE = 9.2309  # scripts/run_stack_v2_cv.py all-in, ledger 2026-07-07
STACK_V21_R7_POOLED_RMSE = 8.9805  # scripts/run_stack_v21_cv.py R7 row, ledger 2026-07-10
PF_BLEND_W = 0.7  # matches build_tracker_cache.py / build_spatial_cache.py / v1 / v2 / R7

IMPROVEMENT_GATE_FT = 0.10
ADOPTION_GATE_RMSE = STACK_V21_R7_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 8.8805
Y_TRUE_ATOL = 1e-2  # cross-bank alignment sanity check tolerance (matches run_router_v2_cv.py)
REPRO_ATOL = 0.01  # (a) R7-repro regression-check tolerance vs STACK_V21_R7_POOLED_RMSE

N_SPLITS = 5
SEED = 42
N_STRATA = 10  # deciles of manifest pf_std_mean, for the --n-wells health-check sample
ES_FRAC = 0.15  # fraction of each fold's training wells carved out as ES holdout

# Identical to v1/v2/R7 (scripts/run_stack_cv.py, run_stack_v2_cv.py,
# run_stack_v21_cv.py) -- "CV設定はR7と完全同一（...触らない）" (task spec).
LGB_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50

# R7's 4 features, byte-identical to run_stack_v21_cv.py -- reproduced here so
# (a) "R7 repro" needs no import coupling to that script (scripts/ files are
# not a shared package in this repo, same convention R7 itself followed vs v2).
NEW_FEATURE_COLUMNS: tuple[str, ...] = (
    "pfbma_v2_d",
    "beam_grid_d",
    "pfbma_v2_minus_pf",
    "pfbma_v2_d_well_mean",
)

# --------------------------------------------------------------------------- #
# R8's 3 new feature groups (task spec Sec "実装", items G1/G2/G3).
# --------------------------------------------------------------------------- #

# G1: pf_bma_v2 scales R7 didn't use (R7 only used pf_bma_scale3_tvt), as
# deltas from pf_blend, plus the per-row range across all 4 scales.
PFBMA_V2_SCALE5_KEY = "pf_bma_scale5_tvt"
PFBMA_V2_SCALE8_KEY = "pf_bma_scale8_tvt"
PFBMA_V2_SCALE12_KEY = "pf_bma_scale12_tvt"
G1_FEATURE_COLUMNS: tuple[str, ...] = (
    "pfbma_v2_s5_d",
    "pfbma_v2_s8_d",
    "pfbma_v2_s12_d",
    "pfbma_v2_scale_range",
)

# G2: pf_bma_v2's own declared uncertainty for the scale R7's pfbma_v2_d
# already uses (pf_bma_scale3_std), raw per-row + well-mean.
PFBMA_V2_SCALE3_STD_KEY = "pf_bma_scale3_std"
G2_FEATURE_COLUMNS: tuple[str, ...] = (
    "pfbma_v2_s3_std",
    "pfbma_v2_s3_std_well_mean",
)

# G3: 4 beam-grid configs chosen to be maximally mutually distant in
# (product, max_move, gr_calibration, smooth_radius) parameter space, among
# the 9 configs NOT already used by R7's beam_grid_d (config 4,
# mid_cal_sm2_mm3). Found by an exhaustive max-min-pairwise-distance search
# over all C(9,4)=126 combinations of z-normalized config parameters (see
# module docstring) -- NOT the task prompt's illustrative "{0,2,7,9}" example,
# which covers only mid/stiff and misses "loose"/"nocal" entirely; {1,2,8,9}
# scores a strictly larger min pairwise distance (3.125 vs {0,2,7,9}'s 2.5)
# AND covers all three families (loose/stiff/nocal) the task asks for.
G3_BEAM_CONFIG_INDICES: tuple[int, ...] = (1, 2, 8, 9)
G3_BEAM_CONFIG_LABELS: tuple[str, ...] = (
    "loose_cal_sm0_mm3",
    "stiff_cal_sm0_mm3",
    "loose_nocal_sm5_mm2",
    "stiff_cal_sm5_mm1",
)
G3_FEATURE_COLUMNS: tuple[str, ...] = (
    "beam_cfg1_d",
    "beam_cfg2_d",
    "beam_cfg8_d",
    "beam_cfg9_d",
    "beam_g3_std",
)

ABLATION_ORDER: tuple[str, ...] = ("a", "b", "c", "d")
ABLATION_LABELS: dict[str, str] = {
    "a": "(a) R7 repro",
    "b": "(b) R7 + G1",
    "c": "(c) R7 + G1 + G2",
    "d": "(d) R7 + G1 + G2 + G3",
}
ABLATION_GROUPS: dict[str, tuple[str, ...]] = {
    "a": (),
    "b": ("G1",),
    "c": ("G1", "G2"),
    "d": ("G1", "G2", "G3"),
}


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

    Same recipe as ``run_stack_v2_cv.py``/``run_stack_v21_cv.py``'s
    ``_stratified_sample`` (not imported -- ``scripts/`` files are not a
    shared package here).
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
# well-name alignment primitives, byte-copied from run_stack_v21_cv.py (itself
# ported verbatim from scripts/run_router_v2_cv.py -- not imported).
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
    """Container for the single feature matrix all ablation configs subset from."""

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
    """Build the combined (P1 + tracker + well-agg + spatial) feature matrix.

    Same validation logic, same 56-column feature set, same row/well set as
    ``run_stack_v2_cv.py``/``run_stack_v21_cv.py``'s ``build_well_matrix_v2``
    (values verified numerically identical via the (a) R7-repro regression
    check, since this feeds every ablation config here) -- but re-engineered
    into **two passes over ``wells``** rather than one, per the task's
    memory-safety mandate:

      1. **Validation pass** (no feature computation, no large arrays kept):
         replays the exact same 5 gates the original single-pass loop used
         (eval-zone/anchor present, tracker cache present + length-matched,
         spatial cache present + length-matched), just to learn ``used_wells``
         and each one's eval-zone row count.
      2. **Fill pass**: with the total row count known upfront, every output
         array (``y_true``/``anchor_arr``/.../each ``X`` column) is
         **pre-allocated once** and each well's rows are written directly into
         its ``[start:end)`` slice.

    The original implementation instead ``.append()``-ed one small DataFrame
    per well to a 773-element list, then called one ``pd.concat`` at the end
    -- which needs the ~900MB input list *and* the ~900MB output array alive
    simultaneously (a ~2x transient peak). Measured on this box: that
    single-pass version alone peaked at **2.1GB RSS** for the 773-well matrix,
    before any LightGBM training even started, and pushed a
    process-isolated single-ablation-config run to **3.0GB+ RSS with active
    swapping** (VmSwap 523MB) against ~2.7GB available -- a real, observed
    near-OOM, not a hypothetical one. Pre-allocation removes that multiplier
    (peak here is bounded by ~1x the final matrix size plus one well's worth
    of transient per-well feature DataFrames, i.e. materially less than the
    ~900MB final ``X`` itself). The tradeoff is one extra full read of each
    well's horizontal-log CSV + tracker/spatial ``.npz`` headers in the
    validation pass (I/O-bound, not memory-bound; the CSV/npz reader is never
    memory-bottlenecked the way the accumulate-then-concat pattern was).
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

    # --- pass 2: recompute every feature block (same calls, same order as
    # the original single pass) and write straight into the pre-sized arrays.
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


def augment_with_bank_deltas(wm: _WellMatrix) -> pd.DataFrame:
    """Build R7's 4 features in ``wm``'s own master row order.

    Byte-for-byte identical to ``run_stack_v21_cv.py::augment_with_bank_deltas``
    (copied, not imported) -- reindexes ``path_bank_pfbma_v2.npz``'s
    ``pf_bma_scale3_tvt`` and ``path_bank_beam.npz``'s ``beam_tvt_4``
    (``mid_cal_sm2_mm3``) onto ``wm.well_idx``/``wm.used_wells`` via
    :func:`block_slices`/:func:`reindex_to_master`, cross-checks each bank's
    own ``y_true`` against ``wm.y_true`` (raises, never silently misaligns),
    then computes :data:`NEW_FEATURE_COLUMNS`.
    """
    n_wells = len(wm.used_wells)
    master_wells = list(wm.used_wells)
    master_boundaries = np.searchsorted(wm.well_idx, np.arange(n_wells + 1)).astype(np.int64)
    if master_boundaries[-1] != wm.well_idx.shape[0]:
        raise ValueError(
            "wm.well_idx blocks are not contiguous/sorted -- alignment precondition violated"
        )

    with np.load(PFBMA_V2_BANK_PATH) as npz:
        pfbma_well_idx = npz["well_idx"]
        pfbma_used_wells = npz["used_wells"]
        pfbma_y_true = npz["y_true"].astype(np.float32)
        pfbma_scale3_tvt = npz[PFBMA_V2_VALUE_KEY].astype(np.float32)
    pfbma_slices = block_slices(pfbma_well_idx, pfbma_used_wells)
    pfbma_row = reindex_to_master(pfbma_scale3_tvt, pfbma_slices, master_wells, master_boundaries)
    pfbma_y_row = reindex_to_master(pfbma_y_true, pfbma_slices, master_wells, master_boundaries)

    with np.load(BEAM_BANK_PATH) as npz:
        beam_well_idx = npz["well_idx"]
        beam_used_wells = npz["used_wells"]
        beam_y_true = npz["y_true"].astype(np.float32)
        beam_labels = [str(x) for x in npz["config_labels"]]
        beam_cfg_i = beam_labels.index(BEAM_GRID_CONFIG_LABEL)
        beam_grid_tvt = npz[f"beam_tvt_{beam_cfg_i}"].astype(np.float32)
    beam_slices = block_slices(beam_well_idx, beam_used_wells)
    beam_row = reindex_to_master(beam_grid_tvt, beam_slices, master_wells, master_boundaries)
    beam_y_row = reindex_to_master(beam_y_true, beam_slices, master_wells, master_boundaries)

    if np.isnan(pfbma_row).any():
        raise AssertionError(
            f"{int(np.isnan(pfbma_row).sum())} row(s) missing {PFBMA_V2_VALUE_KEY} after "
            "reindex -- a well in wm.used_wells is absent from path_bank_pfbma_v2.npz"
        )
    if np.isnan(beam_row).any():
        raise AssertionError(
            f"{int(np.isnan(beam_row).sum())} row(s) missing beam-grid "
            f"{BEAM_GRID_CONFIG_LABEL} after reindex -- a well in wm.used_wells is absent "
            "from path_bank_beam.npz"
        )
    if not np.allclose(pfbma_y_row, wm.y_true, atol=Y_TRUE_ATOL):
        raise AssertionError(
            "path_bank_pfbma_v2.npz y_true disagrees with stack v2.2's own y_true after "
            "reindex -- alignment bug, not a leak, but must not be silently ignored"
        )
    if not np.allclose(beam_y_row, wm.y_true, atol=Y_TRUE_ATOL):
        raise AssertionError(
            "path_bank_beam.npz y_true disagrees with stack v2.2's own y_true after "
            "reindex -- alignment bug, not a leak, but must not be silently ignored"
        )

    pf_tvt_arr = wm.X["pf_d"].to_numpy(dtype=np.float64) + wm.anchor_arr.astype(np.float64)
    pf_blend_f64 = wm.pf_blend_arr.astype(np.float64)
    pfbma_v2_d = pfbma_row.astype(np.float64) - pf_blend_f64
    beam_grid_d = beam_row.astype(np.float64) - pf_blend_f64
    pfbma_v2_minus_pf = pfbma_row.astype(np.float64) - pf_tvt_arr
    pfbma_v2_d_well_mean = (
        pd.Series(pfbma_v2_d).groupby(wm.well_idx).transform("mean").to_numpy()
    )

    out = pd.DataFrame(
        {
            "pfbma_v2_d": pfbma_v2_d,
            "beam_grid_d": beam_grid_d,
            "pfbma_v2_minus_pf": pfbma_v2_minus_pf,
            "pfbma_v2_d_well_mean": pfbma_v2_d_well_mean,
        }
    )
    out = out[list(NEW_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


def _master_bounds(wm: _WellMatrix) -> tuple[list[str], np.ndarray]:
    """Shared (master_wells, master_boundaries) precondition for bank reindexing.

    Factored out of R7's inline logic (repeated verbatim in
    :func:`augment_with_bank_deltas` above for that function's untouched
    byte-copy status) so :func:`build_g1_features`/:func:`build_g2_features`/
    :func:`build_g3_features` don't each duplicate it.
    """
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

    Generalizes the per-array reindex-plus-y_true-cross-check pattern R7's
    :func:`augment_with_bank_deltas` inlines twice (once per bank); used by
    G1/G2/G3 so every new bank column gets the same leak-boundary guarantee
    (raise, never silently misalign) without repeating the block.
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
            f"{path.name} y_true disagrees with stack v2.2's own y_true after reindex "
            f"(value_key={value_key}) -- alignment bug, not a leak, but must not be "
            "silently ignored"
        )
    return row


def build_g1_features(wm: _WellMatrix) -> pd.DataFrame:
    """G1: pf_bma_v2 scale-5/8/12 deltas from ``pf_blend`` + per-row scale range.

    Requires ``wm.X`` to already carry R7's ``pfbma_v2_d`` column (scale 3
    delta) -- scale 3's raw value is *recovered* from it
    (``pfbma_v2_d + pf_blend``) rather than re-reading the bank a second time,
    the same idiom :func:`augment_with_bank_deltas` uses to recover the
    tracker-PF's raw value from ``pf_d + anchor``.
    """
    master_wells, master_boundaries = _master_bounds(wm)
    pf_blend_f64 = wm.pf_blend_arr.astype(np.float64)

    scale3_tvt = wm.X["pfbma_v2_d"].to_numpy(dtype=np.float64) + pf_blend_f64
    scale5_tvt = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_SCALE5_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)
    scale8_tvt = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_SCALE8_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)
    scale12_tvt = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_SCALE12_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)

    pfbma_v2_s5_d = scale5_tvt - pf_blend_f64
    pfbma_v2_s8_d = scale8_tvt - pf_blend_f64
    pfbma_v2_s12_d = scale12_tvt - pf_blend_f64

    all_scales = np.stack([scale3_tvt, scale5_tvt, scale8_tvt, scale12_tvt], axis=1)
    pfbma_v2_scale_range = all_scales.max(axis=1) - all_scales.min(axis=1)
    del all_scales, scale3_tvt, scale5_tvt, scale8_tvt, scale12_tvt

    out = pd.DataFrame(
        {
            "pfbma_v2_s5_d": pfbma_v2_s5_d,
            "pfbma_v2_s8_d": pfbma_v2_s8_d,
            "pfbma_v2_s12_d": pfbma_v2_s12_d,
            "pfbma_v2_scale_range": pfbma_v2_scale_range,
        }
    )
    out = out[list(G1_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


def build_g2_features(wm: _WellMatrix) -> pd.DataFrame:
    """G2: pf_bma_v2's own scale-3 uncertainty, raw per-row + well-mean.

    ``pf_bma_scale3_std`` can be ``inf`` on the PF's own exception-fallback
    path (same semantics as the tracker-PF's ``pf_std``, see
    ``rogii.stack_features`` module docstring); the well-mean is computed
    over the *finite* subset only (an all-inf well falls back to ``NaN``,
    left for LightGBM's native NaN handling rather than a fabricated 0.0 --
    unlike ``build_well_aggregate_features``'s finite-or-default pattern,
    there is no natural "0 uncertainty" default to fall back to here).
    """
    master_wells, master_boundaries = _master_bounds(wm)
    s3_std = _load_bank_array(
        PFBMA_V2_BANK_PATH, PFBMA_V2_SCALE3_STD_KEY, master_wells, master_boundaries, wm.y_true
    ).astype(np.float64)

    finite_or_nan = np.where(np.isfinite(s3_std), s3_std, np.nan)
    well_mean = pd.Series(finite_or_nan).groupby(wm.well_idx).transform("mean").to_numpy()
    del finite_or_nan

    out = pd.DataFrame(
        {
            "pfbma_v2_s3_std": s3_std,
            "pfbma_v2_s3_std_well_mean": well_mean,
        }
    )
    out = out[list(G2_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


def build_g3_features(wm: _WellMatrix) -> pd.DataFrame:
    """G3: 4 maximally-diverse beam-grid deltas from ``pf_blend`` + per-row std.

    Configs are :data:`G3_BEAM_CONFIG_INDICES` (see module docstring for the
    farthest-point-sampling selection). ``beam_g3_std`` is the per-row std
    across the 4 *raw* ``beam_tvt`` values -- identical to the std across the
    4 deltas, since subtracting the same per-row ``pf_blend`` from all 4
    values is a per-row constant shift that does not change their spread.
    """
    master_wells, master_boundaries = _master_bounds(wm)
    pf_blend_f64 = wm.pf_blend_arr.astype(np.float64)

    raw_vals: list[np.ndarray] = []
    deltas: dict[str, np.ndarray] = {}
    for idx, col in zip(G3_BEAM_CONFIG_INDICES, G3_FEATURE_COLUMNS[:-1], strict=True):
        val = _load_bank_array(
            BEAM_BANK_PATH, f"beam_tvt_{idx}", master_wells, master_boundaries, wm.y_true
        ).astype(np.float64)
        raw_vals.append(val)
        deltas[col] = val - pf_blend_f64

    stacked = np.stack(raw_vals, axis=1)
    beam_g3_std = stacked.std(axis=1)
    del raw_vals, stacked

    out = pd.DataFrame({**deltas, "beam_g3_std": beam_g3_std})
    out = out[list(G3_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


def load_reindexed_oof(
    path: Path, label: str, wm: _WellMatrix
) -> np.ndarray:
    """Reindex a saved ``oof_{i}`` array (``config_labels`` lookup) onto ``wm``'s row order.

    Generalizes ``run_stack_v21_cv.py::load_stack_v2_all_in_oof_reindexed``
    (used there only for the stack-v2 comparison) to any prior stack's saved
    OOF npz, so R8 can additionally report helped/hurt vs stack v2.1's own
    R7 OOF. Used only for diagnostics -- never as a feature.
    """
    n_wells = len(wm.used_wells)
    master_wells = list(wm.used_wells)
    master_boundaries = np.searchsorted(wm.well_idx, np.arange(n_wells + 1)).astype(np.int64)

    with np.load(path, allow_pickle=True) as npz:
        well_idx = npz["well_idx"]
        used_wells = npz["used_wells"]
        y_true = npz["y_true"].astype(np.float32)
        config_labels = [str(x) for x in npz["config_labels"]]
        cfg_i = config_labels.index(label)
        oof = npz[f"oof_{cfg_i}"].astype(np.float32)
    slices = block_slices(well_idx, used_wells)
    oof_row = reindex_to_master(oof, slices, master_wells, master_boundaries)
    y_row = reindex_to_master(y_true, slices, master_wells, master_boundaries)

    if np.isnan(oof_row).any():
        raise AssertionError(
            f"{int(np.isnan(oof_row).sum())} row(s) missing {path.name} OOF ({label}) "
            "after reindex -- a well in wm.used_wells is absent from that file"
        )
    if not np.allclose(y_row, wm.y_true, atol=Y_TRUE_ATOL):
        raise AssertionError(f"{path.name} y_true disagrees with stack v2.2's own y_true")
    return oof_row


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Byte-for-byte identical to ``run_stack_v2_cv.py``/``run_stack_v21_cv.py``'s
    ``_split_fit_es`` (copied, not imported): deterministic, sorted-then-
    seeded-shuffled, leading slice -> ES-holdout, trailing slice -> fit.
    ES-holdout wells are excluded from both the training rows and (by
    construction) the score fold.
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
    use_es_holdout: bool,
    es_frac: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    """Train+OOF-score the (b) PF-residual target for one ablation config.

    Identical protocol to ``run_stack_v2_cv.py``/``run_stack_v21_cv.py``'s
    ``run_stack_fold_cv`` (copied, not imported) -- ``LGB_PARAMS``,
    ``EARLY_STOPPING_ROUNDS``, and the ES-holdout mechanism are untouched
    per the task spec.
    """
    feature_cols = list(X.columns)
    oof_pred_tvt = np.full(len(y_true), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importances = np.zeros(len(feature_cols), dtype=np.float64)
    used_well_to_idx = {w: i for i, w in enumerate(used_wells)}

    for fold_i in range(n_splits):
        t_fold = time.time()
        score_mask = row_fold == fold_i

        if use_es_holdout:
            train_well_positions = np.where(well_fold_arr != fold_i)[0]
            train_wells_all = [used_wells[wi] for wi in train_well_positions]
            fit_wells, es_wells = _split_fit_es(train_wells_all, es_frac, seed)
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
        else:
            train_mask = row_fold != fold_i
            eval_set = [(X.loc[score_mask], target[score_mask])]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
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


def _cfg_columns(wm: _WellMatrix, key: str) -> list[str]:
    """Ensure ``wm.X`` carries exactly the columns ``key`` needs; return that column list.

    **Mutates ``wm.X`` in place via per-column assignment** (``wm.X[col] =
    array``), not ``pd.concat``. ``pd.concat`` allocates an entirely new
    ~1GB block while ``wm.X`` is still alive (~2x peak -- the same failure
    mode fixed in :func:`build_well_matrix_v2` and the R7-augmentation step
    in ``main()``); assigning one small (4/2/5-column) group's arrays onto an
    existing DataFrame only allocates those small arrays, not a full-matrix
    copy. This is safe/idempotent because :data:`ABLATION_GROUPS` is strictly
    cumulative (a subset-of b subset-of c subset-of d): calling this with
    'b' then 'c' then 'd' on the *same* ``wm`` (the all-4-in-one-process
    smoke-test path) only ever *adds* columns, never removes or overwrites
    an already-added group; calling it with 'd' directly on a fresh ``wm``
    (the process-isolated full-run path) adds G1+G2+G3 all at once. Either
    way, after this returns, ``wm.X``'s columns equal the returned list
    exactly (verified by assertion) -- callers use ``wm.X`` itself as the
    training matrix, no further subsetting/copying needed.
    """
    groups = ABLATION_GROUPS[key]
    tracker_cols = list(FEATURE_COLUMNS)
    wellagg_cols = list(WELL_FEATURE_COLUMNS)
    spatial_cols = list(SPATIAL_FEATURE_COLUMNS)
    cols = wm.p1_cols + tracker_cols + wellagg_cols + spatial_cols + list(NEW_FEATURE_COLUMNS)

    if "G1" in groups:
        cols += list(G1_FEATURE_COLUMNS)
        if G1_FEATURE_COLUMNS[0] not in wm.X.columns:
            g1 = build_g1_features(wm)
            for col in G1_FEATURE_COLUMNS:
                wm.X[col] = g1[col].to_numpy()
            del g1
            gc.collect()
    if "G2" in groups:
        cols += list(G2_FEATURE_COLUMNS)
        if G2_FEATURE_COLUMNS[0] not in wm.X.columns:
            g2 = build_g2_features(wm)
            for col in G2_FEATURE_COLUMNS:
                wm.X[col] = g2[col].to_numpy()
            del g2
            gc.collect()
    if "G3" in groups:
        cols += list(G3_FEATURE_COLUMNS)
        if G3_FEATURE_COLUMNS[0] not in wm.X.columns:
            g3 = build_g3_features(wm)
            for col in G3_FEATURE_COLUMNS:
                wm.X[col] = g3[col].to_numpy()
            del g3
            gc.collect()

    assert cols == list(wm.X.columns), "wm.X's columns must match the requested config exactly"
    return cols


def run_one_config(
    wm: _WellMatrix,
    key: str,
    row_fold: np.ndarray,
    well_fold_arr: np.ndarray,
) -> dict[str, object]:
    """Run one ablation config (a/b/c/d) to completion and return its results dict."""
    label = ABLATION_LABELS[key]
    cols = _cfg_columns(wm, key)  # mutates wm.X in place (see docstring); wm.X IS the config matrix
    print(f"\n=== {label} (use_es_holdout=True, n_features={len(cols)}) ===", flush=True)

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1/v2/R7
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
        True,
        ES_FRAC,
        SEED,
    )
    gc.collect()

    pooled = pooled_rmse(wm.y_true, oof)
    importance_df = (
        pd.DataFrame({"feature": cols, "mean_importance": importances})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "key": key,
        "label": label,
        "cols": cols,
        "oof": oof,
        "fold_rows": fold_rows,
        "importance_df": importance_df,
        "pooled_rmse": pooled,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _result_path(key: str, n_wells: int | None) -> Path:
    suffix = f"_n{n_wells}" if n_wells is not None else ""
    return OUTPUTS_DIR / f"stack_v22_result_{key}{suffix}.npz"


def _save_result(
    result: dict[str, object], wm: _WellMatrix, row_fold: np.ndarray, n_wells: int | None
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(result["fold_rows"])
    imp_df = result["importance_df"]
    path = _result_path(result["key"], n_wells)
    np.savez_compressed(
        path,
        key=result["key"],
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


def _load_result(key: str, n_wells: int | None) -> dict[str, object]:
    path = _result_path(key, n_wells)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python scripts/run_stack_v22_cv.py "
            f"--config {key}"
            + (f" --n-wells {n_wells}" if n_wells is not None else "")
            + "` first"
        )
    with np.load(path, allow_pickle=True) as npz:
        return {
            "key": str(npz["key"]),
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
            "fold_n_train": npz["fold_n_train"],
            "fold_n_score": npz["fold_n_score"],
            "fold_seconds": npz["fold_seconds"],
            "importance_feature": [str(f) for f in npz["importance_feature"]],
            "importance_value": npz["importance_value"],
        }


def _print_comparison_table(
    results: dict[str, dict[str, object]], n_wells: int | None = None
) -> None:
    any_res = next(iter(results.values()))
    y_true = any_res["y_true"]
    well_idx = any_res["well_idx"]
    used_wells = any_res["used_wells"]
    anchor = any_res["anchor"]
    pf_blend = any_res["pf_blend"]
    n_wells_used = len(used_wells)

    floor_pooled = pooled_rmse(y_true, anchor)
    pf_pooled = pooled_rmse(y_true, pf_blend)
    floor_well_rmse = _per_well_rmse(y_true, anchor, well_idx, n_wells_used)
    pf_well_rmse = _per_well_rmse(y_true, pf_blend, well_idx, n_wells_used)

    print()
    print("=" * 112)
    print(
        f"{'predictor':<26}{'pooled RMSE':>13}{'vs PF blend':>13}{'vs R7 8.9805':>16}"
        f"{'median':>10}{'p90':>10}{'max':>10}"
    )
    print("-" * 112)
    for name, pooled, well_arr in [
        ("carry_last (floor)", floor_pooled, floor_well_rmse),
        ("PF blend w0.7", pf_pooled, pf_well_rmse),
    ]:
        valid = well_arr[np.isfinite(well_arr)]
        print(
            f"{name:<26}{pooled:>13.4f}{'--':>13}{'--':>16}"
            f"{np.median(valid):>10.4f}{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )
    best_key = None
    best_pooled = float("inf")
    for key in ABLATION_ORDER:
        if key not in results:
            continue
        res = results[key]
        oof = res["oof"]
        pooled = res["pooled_rmse"]
        well_arr = _per_well_rmse(y_true, oof, well_idx, n_wells_used)
        valid = well_arr[np.isfinite(well_arr)]
        print(
            f"{res['label']:<26}{pooled:>13.4f}{pooled - pf_pooled:>+13.4f}"
            f"{pooled - STACK_V21_R7_POOLED_RMSE:>+16.4f}"
            f"{np.median(valid):>10.4f}{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
            f"   peak_rss={res['peak_rss_mb']:.0f}MB"
        )
        if pooled < best_pooled:
            best_pooled = pooled
            best_key = key
    print("=" * 112)
    print(f"total rows scored: {len(y_true):,}")

    if "a" in results:
        repro_diff = results["a"]["pooled_rmse"] - STACK_V21_R7_POOLED_RMSE
        if n_wells is not None:
            # The 8.9805 ledger figure is a full-773-well pooled RMSE; a
            # stratified n_wells<773 subsample is expected to disagree by
            # more than REPRO_ATOL just from sampling noise, so PASS/FAIL
            # wording here would be misleading -- report the diff only.
            print(
                f"\n(a) R7-repro check (n_wells={n_wells} subsample, informational "
                f"only -- REPRO_ATOL={REPRO_ATOL}ft only applies at full 773-well scale): "
                f"{results['a']['pooled_rmse']:.4f} vs R7 ledger 8.9805 (diff {repro_diff:+.4f}ft)"
            )
        else:
            repro_ok = abs(repro_diff) <= REPRO_ATOL
            print(
                f"\n(a) R7-repro regression check: {results['a']['pooled_rmse']:.4f} vs "
                f"R7 ledger 8.9805 (diff {repro_diff:+.4f}ft, tol {REPRO_ATOL}ft) "
                f"-> {'PASS' if repro_ok else 'FAIL -- investigate before trusting b/c/d'}"
            )

    if best_key is not None:
        best = results[best_key]
        print(f"\nfeature importance top20 -- {best['label']}:")
        imp_df = pd.DataFrame(
            {"feature": best["importance_feature"], "mean_importance": best["importance_value"]}
        ).sort_values("mean_importance", ascending=False)
        print(imp_df.head(20).to_string(index=False))

        new_cols_all = set(G1_FEATURE_COLUMNS) | set(G2_FEATURE_COLUMNS) | set(G3_FEATURE_COLUMNS)
        new_ranks = imp_df.reset_index(drop=True).reset_index()
        new_ranks = new_ranks[new_ranks["feature"].isin(new_cols_all)]
        if len(new_ranks):
            print(f"\nR8 new-feature ranks (of {len(imp_df)}):")
            print(new_ranks.rename(columns={"index": "rank"}).to_string(index=False))

        _helped_hurt_report(
            best["label"], y_true, best["oof"], pf_blend, "PF blend w0.7", well_idx, n_wells_used
        )

        try:
            r7_oof_path = STACK_V21_OOF_PATH
            if r7_oof_path.exists():
                with np.load(r7_oof_path, allow_pickle=True) as npz:
                    r7_labels = [str(x) for x in npz["config_labels"]]
                    r7_well_idx = npz["well_idx"]
                    r7_used_wells = npz["used_wells"]
                    r7_y_true = npz["y_true"].astype(np.float32)
                    r7_i = r7_labels.index(STACK_V21_R7_LABEL)
                    r7_oof_raw = npz[f"oof_{r7_i}"].astype(np.float32)
                master_boundaries = np.searchsorted(well_idx, np.arange(n_wells_used + 1)).astype(
                    np.int64
                )
                r7_slices = block_slices(r7_well_idx, r7_used_wells)
                r7_oof_row = reindex_to_master(r7_oof_raw, r7_slices, used_wells, master_boundaries)
                r7_y_row = reindex_to_master(r7_y_true, r7_slices, used_wells, master_boundaries)
                if not np.isnan(r7_oof_row).any() and np.allclose(
                    r7_y_row, y_true, atol=Y_TRUE_ATOL
                ):
                    _helped_hurt_report(
                        best["label"],
                        y_true,
                        best["oof"],
                        r7_oof_row,
                        "stack v2.1 R7 OOF (8.9805)",
                        well_idx,
                        n_wells_used,
                    )
        except (KeyError, ValueError, IndexError) as exc:  # never let a diagnostic crash the run
            print(f"\n(skipped R7-OOF helped/hurt diagnostic: {exc})")

    print(f"\n{'=' * 112}")
    if best_key is not None:
        best_pooled_v = results[best_key]["pooled_rmse"]
        improvement = STACK_V21_R7_POOLED_RMSE - best_pooled_v
        verdict = "採用" if best_pooled_v <= ADOPTION_GATE_RMSE else "却下（非改善）"
        print(
            f"判定: {verdict} -- best={results[best_key]['label']} ({best_pooled_v:.4f}), "
            f"R7={STACK_V21_R7_POOLED_RMSE:.4f}, 改善={improvement:+.4f}ft "
            f"(gate: <= {ADOPTION_GATE_RMSE:.4f}ft)"
        )
    print(f"{'=' * 112}")


def _aggregate(n_wells: int | None, save_oof: bool) -> None:
    results: dict[str, dict[str, object]] = {}
    for key in ABLATION_ORDER:
        try:
            results[key] = _load_result(key, n_wells)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
    if not results:
        print("no result files found -- nothing to aggregate")
        return

    y_true_ref = results[next(iter(results))]["y_true"]
    for key, res in results.items():
        if not np.allclose(res["y_true"], y_true_ref, atol=Y_TRUE_ATOL):
            raise AssertionError(
                f"config {key}'s y_true disagrees with the other configs' -- these were not "
                "run on the same wells list, aggregation would be meaningless"
            )

    _print_comparison_table(results, n_wells=n_wells)

    if save_oof:
        best_key = min(results, key=lambda k: results[k]["pooled_rmse"])
        best = results[best_key]
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / "stack_v22_oof.npz"
        np.savez_compressed(
            out_path,
            y_true=best["y_true"].astype(np.float32),
            anchor=best["anchor"].astype(np.float32),
            pf_blend=best["pf_blend"].astype(np.float32),
            oof=best["oof"].astype(np.float32),
            well_idx=best["well_idx"],
            row_fold=best["row_fold"],
            used_wells=np.array(best["used_wells"]),
            best_config=best["key"],
            best_label=best["label"],
            best_cols=np.array(best["cols"]),
            pooled_rmse=np.float64(best["pooled_rmse"]),
        )
        print(f"\nbest OOF saved: {out_path} (config={best['label']}, {best['pooled_rmse']:.4f})")


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
        "--config",
        choices=list(ABLATION_ORDER),
        default=None,
        help="Run only this ablation config (a/b/c/d) in isolation, then exit -- the "
        "memory-safe path for the full 773-well run (one OS process per config). "
        "Omit to run all 4 configs in one process (fine for --n-wells smoke tests).",
    )
    parser.add_argument(
        "--save-oof",
        action="store_true",
        help="With --config: also dump this config's per-config result npz (always saved "
        "regardless of this flag; kept for CLI parity with run_stack_v21_cv.py). "
        "With --aggregate: write the best config's OOF to outputs/stack_v22_oof.npz.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Skip CV entirely; load all 4 outputs/stack_v22_result_{a,b,c,d}*.npz, print "
        "the comparison table + gate verdict, and (with --save-oof) write "
        "outputs/stack_v22_oof.npz for the best config.",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.aggregate:
        _aggregate(args.n_wells, args.save_oof)
        print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
        return

    all_wells = D.list_wells("train")
    if args.n_wells is not None:
        wells = _stratified_sample(all_wells, args.n_wells, SEED)
        print(
            f"health-check subset: {len(wells)}/{len(all_wells)} wells "
            "(stratified by pf_std_mean)"
        )
    else:
        wells = all_wells
        print(f"train wells: {len(wells)}")

    t0 = time.time()
    wm = build_well_matrix_v2(wells, split="train")
    gc.collect()  # reclaim build_well_matrix_v2's per-well feat_parts list overhead
    print(
        f"feature matrix (v2, 56 cols): {wm.X.shape}, rows={len(wm.y_true):,}, "
        f"wells_used={len(wm.used_wells)}/{len(wells)} counts={wm.counts} "
        f"({time.time() - t0:.1f}s) peak_rss={_peak_rss_mb():.0f}MB"
    )

    t0 = time.time()
    r7_feats = augment_with_bank_deltas(wm)
    # In-place per-column assignment, not pd.concat: concat would allocate an
    # entirely new ~900MB block while the old one is still alive (~2x peak,
    # the same failure mode fixed in build_well_matrix_v2 above); assigning
    # one small column at a time only allocates that column's own array.
    for col in NEW_FEATURE_COLUMNS:
        wm.X[col] = r7_feats[col].to_numpy()
    del r7_feats
    gc.collect()
    print(
        f"R7 augmentation: +{len(NEW_FEATURE_COLUMNS)} cols -> {wm.X.shape} "
        f"({time.time() - t0:.1f}s, y_true cross-checks PASSED) peak_rss={_peak_rss_mb():.0f}MB"
    )

    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr = np.array([well_to_fold[w] for w in wm.used_wells], dtype=np.int32)
    row_fold = well_fold_arr[wm.well_idx]

    configs_to_run = [args.config] if args.config else list(ABLATION_ORDER)
    results: dict[str, dict[str, object]] = {}
    for key in configs_to_run:
        result = run_one_config(wm, key, row_fold, well_fold_arr)
        results[key] = result
        print(
            f"  -> {result['label']}: pooled RMSE={result['pooled_rmse']:.4f} "
            f"peak_rss={result['peak_rss_mb']:.0f}MB"
        )
        _save_result(result, wm, row_fold, args.n_wells)

    if args.config is None:
        # all-4-in-one-process path (smoke tests only) -- reuse the same
        # comparison/aggregation printer the multi-process --aggregate path uses,
        # by reloading what was just saved (keeps exactly one code path).
        _aggregate(args.n_wells, args.save_oof)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
    print(f"this process's peak RSS: {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
