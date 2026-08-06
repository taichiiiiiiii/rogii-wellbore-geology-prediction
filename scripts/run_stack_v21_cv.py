"""Stack v2.1: pf_bma_v2 / beam-grid per-row deltas as stack features (issue: R7).

**Why R7.** The 2026-07-10 ledger established two facts that point in opposite
directions for how to use the bank_builder Kaggle kernel's candidate paths
(``outputs/path_bank_pfbma_v2.npz`` -- PF-BMA with ``init_pos_std=2.0`` --
and ``outputs/path_bank_beam.npz`` -- the 10-config beam-DP grid):

  1. **sub-v4** (a *fixed-weight blend* of stack v2 + pf_bma_scale3 + one beam
     config + spatial, honest CV **9.1104**) scored **LB 9.456**, *worse*
     than stack v2 alone (LB 9.022). A blend-type CV improvement did not
     transfer to the hidden test set (calibration point #4).
  2. Stack v2 itself uses tracker info (``pf_d``/``beam_d``/``spatial_d``,
     ``rogii.stack_features``) purely as **GBDT features**, not as blend
     components -- and *that* mechanism demonstrably transferred (well-CV
     9.2309 -> LB 9.022, calibration point #3).

R7's hypothesis: stack v2's own feature-based mechanism, not blending, is
what transfers. So instead of blending in pf_bma_v2/beam-grid predictions,
this script feeds their **per-row disagreement with the stack's own
PF-residual baseline** to the same LightGBM stack v2 already trains, letting
it learn row-by-row when the alternate candidate should move the residual
prediction and by how much -- the same idiom as v2's existing ``pf_d``/
``beam_d`` features, just against two new inputs.

**56 -> 60 features.** All of stack v2's existing feature blocks (P1
tabular, 8 tracker, 8 well-aggregate, 4 spatial -- see
``run_stack_v2_cv.py``, NOT modified here, read-only reference) are
unchanged. Four new columns (:data:`NEW_FEATURE_COLUMNS`, built by
:func:`augment_with_bank_deltas`) are appended, all deltas from the same
``pf_blend`` basis the stack's own ``(b) PF-residual`` target already uses
(``target = TVT - pf_blend``, PF_BLEND_W=0.7):

  - ``pfbma_v2_d``          = pf_bma_scale3_v2 (init_pos_std=2.0 bank) - pf_blend
  - ``beam_grid_d``         = beam-grid config 4 (``mid_cal_sm2_mm3``)  - pf_blend
  - ``pfbma_v2_minus_pf``   = pf_bma_scale3_v2 - the existing tracker PF (recovered
                              from the already-present ``pf_d`` column + anchor;
                              candidate-vs-candidate disagreement, never TVT)
  - ``pfbma_v2_d_well_mean``= well-level mean of ``pfbma_v2_d``, broadcast per row

**Alignment.** ``path_bank_pfbma_v2.npz`` / ``path_bank_beam.npz`` /
``outputs/stack_v2_oof.npz`` each carry their own ``well_idx``/``used_wells``
row-block bookkeeping (independently built by different scripts on
different runs). :func:`block_slices`/:func:`reindex_to_master` (ported
verbatim from ``scripts/run_router_v2_cv.py`` -- not imported, ``scripts/``
files are not a shared package here, that script's own convention) gather
each bank's row-level array into *this* script's own well-block row order
(the master), and every reindex is followed by a ``y_true``
cross-check-or-raise against ``wm.y_true`` (never a silent misalignment).

**Leak boundary.** ``pfbma_v2_d``/``beam_grid_d``/``pfbma_v2_minus_pf`` are
built entirely from already-computed candidate *predictions*
(``pf_bma_scale3_v2``/beam-grid/tracker-PF) and the same ``pf_blend``/
``anchor`` scalars stack v2's own features already use -- none of them reads
``h["TVT"]`` or the eval-zone ``TVT_input``. The PF-BMA and beam trackers are
themselves fold-independent physical computations (leave-self-out is not
applicable -- they never read other wells' ground truth either), so there is
no OOF leak from adding them as features, unlike a supervised meta-model.

**Fair comparison.** CV protocol -- well-GroupKFold(5, seed=42), the honest
ES-holdout split (:func:`_split_fit_es`, ``ES_FRAC=0.15``), and
``LGB_PARAMS``/``EARLY_STOPPING_ROUNDS`` -- is copied byte-for-byte from
``run_stack_v2_cv.py``. All 5 of v2's original ablation rows are reproduced
unchanged (same feature columns => should numerically match the v2 ledger
exactly, a built-in regression check); a 6th row adds exactly
:data:`NEW_FEATURE_COLUMNS` on top of v2's ``all-in`` config, isolating the
R7 lever the same way v2 isolated ES-holdout/well-agg/spatial.

Usage::

    uv run python scripts/run_stack_v21_cv.py                 # full 773-well experiment
    uv run python scripts/run_stack_v21_cv.py --n-wells 80    # stratified health-check subset
    uv run python scripts/run_stack_v21_cv.py --save-oof      # dump all 6 configs' OOF to outputs/
"""

from __future__ import annotations

import argparse
import random
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

PFBMA_V2_VALUE_KEY = "pf_bma_scale3_tvt"  # pf_bma_scale3_v2, from the init_pos_std=2.0 bank
BEAM_GRID_CONFIG_LABEL = "mid_cal_sm2_mm3"  # config index 4 -> "beam:mid_cal_sm2_mm3" in the ledger
STACK_V2_ALL_IN_LABEL = "all-in (levers 1+2+3)"

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
V1_STACK_B_POOLED_RMSE = 10.6513  # scripts/run_stack_cv.py (b) PF-residual, ledger 2026-07-06
STACK_V2_ALL_IN_POOLED_RMSE = 9.2309  # scripts/run_stack_v2_cv.py all-in, ledger 2026-07-07
PF_BLEND_W = 0.7  # matches build_tracker_cache.py / build_spatial_cache.py / v1 / v2

IMPROVEMENT_GATE_FT = 0.10
ADOPTION_GATE_RMSE = STACK_V2_ALL_IN_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.1309
Y_TRUE_ATOL = 1e-2  # cross-bank alignment sanity check tolerance (matches run_router_v2_cv.py)

N_SPLITS = 5
SEED = 42
N_STRATA = 10  # deciles of manifest pf_std_mean, for the --n-wells health-check sample
ES_FRAC = 0.15  # fraction of each fold's training wells carved out as ES holdout

# Identical to v1/v2 (scripts/run_stack_cv.py, run_stack_v2_cv.py) -- "LGBM設定は
# v1踏襲" (task spec), and R7 explicitly requires this untouched for a fair comparison.
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

# R7's 4 new per-row features, all deltas from the same pf_blend basis the
# stack's own (b) PF-residual target already uses (see module docstring).
NEW_FEATURE_COLUMNS: tuple[str, ...] = (
    "pfbma_v2_d",
    "beam_grid_d",
    "pfbma_v2_minus_pf",
    "pfbma_v2_d_well_mean",
)


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


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

    Same recipe as ``scripts/run_stack_v2_cv.py::_stratified_sample`` (not
    imported -- that script is v2, off-limits to modify/import-couple with;
    ``scripts/`` files are not a shared package in this repo).
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
# well-name alignment primitives, ported verbatim from
# scripts/run_router_v2_cv.py (not imported, see module docstring).
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

    Byte-for-byte identical to ``scripts/run_stack_v2_cv.py::build_well_matrix_v2``
    (copied, not imported -- ``scripts/`` files are not a shared package here).
    R7's new features are added afterwards by :func:`augment_with_bank_deltas`,
    so this function -- and therefore v2's original 56-column feature set and
    row/well set -- is untouched.
    """
    feat_parts: list[pd.DataFrame] = []
    y_true_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    pf_blend_parts: list[np.ndarray] = []
    pf_std_parts: list[np.ndarray] = []
    well_idx_parts: list[np.ndarray] = []
    used_wells: list[str] = []
    p1_cols: list[str] = []
    counts = {
        "n_no_eval_zone_or_anchor": 0,
        "n_tracker_cache_missing": 0,
        "n_tracker_len_mismatch": 0,
        "n_spatial_cache_missing": 0,
        "n_spatial_len_mismatch": 0,
    }

    for well in wells:
        h = D.load_horizontal(well, split)
        mask = D.eval_mask(h)
        if mask.sum() == 0:
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
            pf_tvt = npz["pf_tvt"].astype(np.float64)
            pf_std = npz["pf_std"].astype(np.float64)
            beam_tvt = npz["beam_tvt"].astype(np.float64)
            beam_margin = npz["beam_margin"].astype(np.float64)
            cached_anchor = float(npz["anchor"])
        if pf_tvt.size != int(mask.sum()):
            counts["n_tracker_len_mismatch"] += 1
            continue

        sp_path = _spatial_npz_path(well)
        if not sp_path.exists():
            counts["n_spatial_cache_missing"] += 1
            continue
        with np.load(sp_path) as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
            prefix_rmse = float(npz["prefix_rmse"])
            nn_dist_median = float(npz["nn_dist_median"])
        if spatial_tvt.size != int(mask.sum()):
            counts["n_spatial_len_mismatch"] += 1
            continue

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

        feats = pd.concat(
            [
                p1_feats.reset_index(drop=True),
                trk_feats.reset_index(drop=True),
                well_agg_feats.reset_index(drop=True),
                spatial_feats.reset_index(drop=True),
            ],
            axis=1,
        )

        y_true = h["TVT"].to_numpy(dtype=np.float32)[mask]  # label only, never a feature
        pf_blend = cached_anchor + PF_BLEND_W * (pf_tvt - cached_anchor)

        well_i = len(used_wells)
        used_wells.append(well)

        feat_parts.append(feats)
        y_true_parts.append(y_true.astype(np.float64))
        anchor_parts.append(np.full(y_true.shape[0], cached_anchor, dtype=np.float64))
        pf_blend_parts.append(pf_blend)
        pf_std_parts.append(pf_std)
        well_idx_parts.append(np.full(y_true.shape[0], well_i, dtype=np.int32))

    X = pd.concat(feat_parts, ignore_index=True)
    y_true = np.concatenate(y_true_parts)
    anchor_arr = np.concatenate(anchor_parts)
    pf_blend_arr = np.concatenate(pf_blend_parts)
    pf_std_arr = np.concatenate(pf_std_parts)
    well_idx = np.concatenate(well_idx_parts)
    return _WellMatrix(
        X, y_true, anchor_arr, pf_blend_arr, pf_std_arr, well_idx, used_wells, p1_cols, counts
    )


def augment_with_bank_deltas(wm: _WellMatrix) -> pd.DataFrame:
    """Build the 4 R7 features in ``wm``'s own master row order.

    Reindexes ``path_bank_pfbma_v2.npz``'s ``pf_bma_scale3_tvt`` and
    ``path_bank_beam.npz``'s ``beam_tvt_4`` (``mid_cal_sm2_mm3``) onto
    ``wm.well_idx``/``wm.used_wells`` via :func:`block_slices`/
    :func:`reindex_to_master`, cross-checks each bank's own ``y_true``
    against ``wm.y_true`` (raises, never silently misaligns), then computes
    :data:`NEW_FEATURE_COLUMNS` -- all deltas from ``wm.pf_blend_arr``
    (the same basis the stack's PF-residual target uses) or from the
    existing tracker PF (recovered from ``wm.X["pf_d"] + wm.anchor_arr"``,
    so no extra row-level array needs to be carried).
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
            "path_bank_pfbma_v2.npz y_true disagrees with stack v2.1's own y_true after "
            "reindex -- alignment bug, not a leak, but must not be silently ignored"
        )
    if not np.allclose(beam_y_row, wm.y_true, atol=Y_TRUE_ATOL):
        raise AssertionError(
            "path_bank_beam.npz y_true disagrees with stack v2.1's own y_true after "
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


def load_stack_v2_all_in_oof_reindexed(wm: _WellMatrix) -> np.ndarray:
    """Reindex stack v2's ``all-in`` OOF (``outputs/stack_v2_oof.npz``) onto ``wm``'s row order.

    Used only for the task-required "helped/hurt vs stack v2 OOF" report --
    never as a feature.
    """
    n_wells = len(wm.used_wells)
    master_wells = list(wm.used_wells)
    master_boundaries = np.searchsorted(wm.well_idx, np.arange(n_wells + 1)).astype(np.int64)

    with np.load(STACK_V2_OOF_PATH, allow_pickle=True) as npz:
        well_idx = npz["well_idx"]
        used_wells = npz["used_wells"]
        y_true = npz["y_true"].astype(np.float32)
        config_labels = [str(x) for x in npz["config_labels"]]
        all_in_i = config_labels.index(STACK_V2_ALL_IN_LABEL)
        oof_all_in = npz[f"oof_{all_in_i}"].astype(np.float32)
    slices = block_slices(well_idx, used_wells)
    oof_row = reindex_to_master(oof_all_in, slices, master_wells, master_boundaries)
    y_row = reindex_to_master(y_true, slices, master_wells, master_boundaries)

    if np.isnan(oof_row).any():
        raise AssertionError(
            f"{int(np.isnan(oof_row).sum())} row(s) missing stack_v2_oof.npz all-in OOF "
            "after reindex -- a well in wm.used_wells is absent from stack_v2_oof.npz"
        )
    if not np.allclose(y_row, wm.y_true, atol=Y_TRUE_ATOL):
        raise AssertionError(
            "stack_v2_oof.npz y_true disagrees with stack v2.1's own y_true after reindex"
        )
    return oof_row


def _split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split of a fold's training wells into fit vs. ES-holdout.

    Byte-for-byte identical to ``run_stack_v2_cv.py::_split_fit_es`` (copied,
    not imported): deterministic, sorted-then-seeded-shuffled, leading slice
    -> ES-holdout, trailing slice -> fit. ES-holdout wells are excluded from
    both the training rows and (by construction) the score fold.
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

    Identical protocol to ``run_stack_v2_cv.py::run_stack_fold_cv`` (copied,
    not imported).
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
        "--save-oof",
        action="store_true",
        help="Save every config's OOF prediction arrays to outputs/stack_v21_oof.npz.",
    )
    args = parser.parse_args()

    t_start = time.time()

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
    print(
        f"feature matrix (v2, 56 cols): {wm.X.shape}, rows={len(wm.y_true):,}, "
        f"wells_used={len(wm.used_wells)}/{len(wells)} counts={wm.counts} "
        f"({time.time() - t0:.1f}s)"
    )

    t0 = time.time()
    new_feats = augment_with_bank_deltas(wm)
    wm.X = pd.concat([wm.X, new_feats.reset_index(drop=True)], axis=1)
    print(
        f"R7 augmentation: +{new_feats.shape[1]} cols -> {wm.X.shape} "
        f"({time.time() - t0:.1f}s, y_true cross-checks PASSED)"
    )
    for col in NEW_FEATURE_COLUMNS:
        vals = new_feats[col].to_numpy()
        print(
            f"  {col:<22} mean={np.nanmean(vals):+.4f} std={np.nanstd(vals):.4f} "
            f"nan_frac={np.isnan(vals).mean():.4f}"
        )

    stack_v2_all_in_oof = load_stack_v2_all_in_oof_reindexed(wm)

    # GroupKFold(5, seed=42) computed from the SAME full wells list as v1/v2
    # (scripts/run_stack_cv.py / run_stack_v2_cv.py) -> identical fold assignment.
    folds = CV.well_folds(wells, n_splits=N_SPLITS, seed=SEED)
    well_to_fold = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr = np.array([well_to_fold[w] for w in wm.used_wells], dtype=np.int32)
    row_fold = well_fold_arr[wm.well_idx]

    n_wells_used = len(wm.used_wells)

    floor_pooled = pooled_rmse(wm.y_true, wm.anchor_arr)
    pf_blend_pooled = pooled_rmse(wm.y_true, wm.pf_blend_arr)
    stack_v2_pooled = pooled_rmse(wm.y_true, stack_v2_all_in_oof)
    floor_well_rmse = _per_well_rmse(wm.y_true, wm.anchor_arr, wm.well_idx, n_wells_used)
    pf_well_rmse = _per_well_rmse(wm.y_true, wm.pf_blend_arr, wm.well_idx, n_wells_used)

    tracker_cols = list(FEATURE_COLUMNS)
    wellagg_cols = list(WELL_FEATURE_COLUMNS)
    spatial_cols = list(SPATIAL_FEATURE_COLUMNS)
    new_cols = list(NEW_FEATURE_COLUMNS)

    # Rows 1-5 reproduce v2's original ablation exactly (same columns -> same
    # OOF, a built-in regression check). Row 6 is the only R7 change: v2's
    # all-in config + exactly the 4 new columns, same ES-holdout protocol.
    ablation_configs: list[tuple[str, list[str], bool]] = [
        ("(b) v1 repro", wm.p1_cols + tracker_cols, False),
        ("+ES holdout only", wm.p1_cols + tracker_cols, True),
        ("+well-agg only", wm.p1_cols + tracker_cols + wellagg_cols, False),
        ("+spatial only", wm.p1_cols + tracker_cols + spatial_cols, False),
        ("all-in (levers 1+2+3)", wm.p1_cols + tracker_cols + wellagg_cols + spatial_cols, True),
        (
            "all-in + pfbma_v2/beam_grid (R7)",
            wm.p1_cols + tracker_cols + wellagg_cols + spatial_cols + new_cols,
            True,
        ),
    ]

    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, unchanged from v1/v2

    results: dict[str, np.ndarray] = {}
    all_fold_rows: list[dict[str, object]] = []
    importances_by_config: dict[str, pd.DataFrame] = {}

    for label, cols, use_es_holdout in ablation_configs:
        print(f"\n=== {label} (use_es_holdout={use_es_holdout}, n_features={len(cols)}) ===")
        X_cfg = wm.X[cols]
        oof, fold_rows, importances = run_stack_fold_cv(
            label,
            X_cfg,
            y_r,
            wm.pf_blend_arr,
            wm.y_true,
            row_fold,
            wm.well_idx,
            well_fold_arr,
            wm.used_wells,
            N_SPLITS,
            use_es_holdout,
            ES_FRAC,
            SEED,
        )
        results[label] = oof
        all_fold_rows.extend(fold_rows)
        importances_by_config[label] = (
            pd.DataFrame({"feature": cols, "mean_importance": importances})
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )

    print()
    print("=" * 112)
    print(
        f"{'predictor':<26}{'pooled RMSE':>13}{'vs PF blend':>13}{'vs stack v2 9.2309':>20}"
        f"{'median':>10}{'p90':>10}{'max':>10}"
    )
    print("-" * 112)
    for name, pooled, well_arr in [
        ("carry_last (floor)", floor_pooled, floor_well_rmse),
        ("PF blend w0.7", pf_blend_pooled, pf_well_rmse),
    ]:
        valid = well_arr[np.isfinite(well_arr)]
        print(
            f"{name:<26}{pooled:>13.4f}{'--':>13}{'--':>20}"
            f"{np.median(valid):>10.4f}{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )
    for label, _cols, _use_es in ablation_configs:
        oof = results[label]
        pooled = pooled_rmse(wm.y_true, oof)
        well_arr = _per_well_rmse(wm.y_true, oof, wm.well_idx, n_wells_used)
        valid = well_arr[np.isfinite(well_arr)]
        print(
            f"{label:<26}{pooled:>13.4f}{pooled - pf_blend_pooled:>+13.4f}"
            f"{pooled - STACK_V2_ALL_IN_POOLED_RMSE:>+20.4f}"
            f"{np.median(valid):>10.4f}{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )
    print("=" * 112)
    print(f"total rows scored: {len(wm.y_true):,}")
    print(f"stack v2 all-in OOF (reindexed reference) pooled RMSE = {stack_v2_pooled:.4f}")

    fold_df = pd.DataFrame(all_fold_rows)
    print("\nper-fold RMSE / best_iteration (stability check):")
    print(fold_df.to_string(index=False))

    final_label = ablation_configs[-1][0]
    print(f"\nfeature importance top20 -- {final_label}:")
    print(importances_by_config[final_label].head(20).to_string(index=False))
    new_ranks = importances_by_config[final_label].reset_index()
    new_ranks = new_ranks[new_ranks["feature"].isin(new_cols)]
    print(f"\nR7 new-feature ranks (of {len(importances_by_config[final_label])}):")
    print(new_ranks.rename(columns={"index": "rank"}).to_string(index=False))

    _helped_hurt_report(
        final_label, wm.y_true, results[final_label], wm.pf_blend_arr, "PF blend w0.7",
        wm.well_idx, n_wells_used,
    )
    _helped_hurt_report(
        final_label, wm.y_true, results[final_label], stack_v2_all_in_oof, "stack v2 OOF (9.2309)",
        wm.well_idx, n_wells_used,
    )

    if args.save_oof:
        out_dir = OUTPUTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        oof_path = out_dir / "stack_v21_oof.npz"
        save_kwargs = {
            f"oof_{i}": results[label].astype(np.float32)
            for i, (label, _cols, _use_es) in enumerate(ablation_configs)
        }
        np.savez_compressed(
            oof_path,
            y_true=wm.y_true.astype(np.float32),
            anchor=wm.anchor_arr.astype(np.float32),
            pf_blend=wm.pf_blend_arr.astype(np.float32),
            stack_v2_all_in_oof=stack_v2_all_in_oof.astype(np.float32),
            well_idx=wm.well_idx,
            row_fold=row_fold,
            used_wells=np.array(wm.used_wells),
            config_labels=np.array([label for label, _c, _e in ablation_configs]),
            **save_kwargs,
        )
        print(f"\nOOF arrays saved: {oof_path}")

    final_pooled = pooled_rmse(wm.y_true, results[final_label])
    improvement = STACK_V2_ALL_IN_POOLED_RMSE - final_pooled
    verdict = "採用" if final_pooled <= ADOPTION_GATE_RMSE else "却下（非改善）"
    print(f"\n{'=' * 112}")
    print(
        f"判定: {verdict} -- stack v2.1={final_pooled:.4f}, "
        f"stack v2={STACK_V2_ALL_IN_POOLED_RMSE:.4f}, "
        f"改善={improvement:+.4f}ft (gate: <= {ADOPTION_GATE_RMSE:.4f}ft)"
    )
    print(f"{'=' * 112}")

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
