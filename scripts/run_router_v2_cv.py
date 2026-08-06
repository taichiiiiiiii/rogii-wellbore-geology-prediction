"""R3-v2 candidate-path router: backtest + cross-candidate-agreement features
(design.md §11 R3-v2, ledger 2026-07-09 R3-v1 follow-up).

**Why R3-v2.** R3-v1 (``scripts/run_router_cv.py``, read-only reference,
never modified) trained a 4-class classifier over {stack, carry_last,
pf_blend, spatial} and was rejected: nested pooled RMSE 9.2161 (-0.0148 vs
the 9.2309 baseline, gate is >=0.10ft) because none of its features directly
measured *relative candidate quality* (4-class argmax 33.6% < the 46.7%
majority floor). R3-v2 adds two leak-safe blocks that do:

1. **Backtest features** (``scripts/build_backtest_features.py``'s 48-column
   schema, ``outputs/backtest_features.npz``): each candidate actually run on
   a prefix-truncated pseudo eval zone and scored against real ground truth.
2. **Cross-candidate agreement features** (:func:`compute_agreement_features`):
   RMS distance / rank correlation / spread *among candidate predictions*
   (never against ``TVT``) -- legal at hidden-test inference time.

Both combine with R3-v1's own well-diagnostic block (tracker/spatial cache
scalars, GR NaN rate, prefix slope, Z span -- reused verbatim, not imported;
``scripts/`` files are not a shared package here, see ``run_beam_grid_cv.py``
for the one precedent of cross-script reuse, not followed to keep this
module self-contained like ``run_pf_bma_cv.py``).

**Candidate universe (<=18).** The 4 R3-v1 base candidates
(``stack``/``carry_last``/``pf_blend``/``spatial``) plus the bank_builder
Kaggle kernel's two new families: PF-BMA at 4 softmax temperatures
(``outputs/path_bank_pfbma.npz``) and the 10-config beam-DP grid
(``outputs/path_bank_beam.npz``, prefixed ``beam:<label>``).

**Regression, not classification** (R3-v1's failure mode). Instead of one
classifier over K classes, :func:`fit_predict_rmse_matrix` fits one small
LightGBM regressor per candidate predicting its per-well RMSE;
:func:`decide` argmins the predicted-RMSE row and trust-gates with ``tau``
(a predicted-*improvement*-in-feet threshold, not a class probability),
mirroring R3-v1's nested tau selection but also nested-selecting a tiny
2-entry GBDT hyperparameter grid (``MODEL_GRID``).

**Leak boundary.** Every feature is computable from a well's own known-zone
prefix + already-computed candidate *predictions* (never ``TVT``/eval-zone
``TVT_input``). Agreement features never take ``y_true`` as an argument
(asserted by ``tests/test_router_v2.py``). Per-well *training labels* (true
RMSE) legitimately use ground truth like any supervised target, but are
never fed back in as a feature, and every outer test fold is scored by a
regressor (and tau) that never saw that fold's wells (nested well-GroupKFold).

**Memory discipline** (3.8GB dev box). Row-level sources are loaded once as
float32 and reindexed into per-candidate master-aligned arrays (a few
hundred MB at 773 wells) -- never an all-candidate x all-row pairwise
matrix; :func:`build_well_matrices` streams well-by-well, holding only
O(n_wells x K) aggregates after the loop. Peak RSS printed via
``resource.getrusage``.

Usage::

    # full run (once the rogii-bank-builder kernel's 3 npz land in outputs/)
    uv run python scripts/run_router_v2_cv.py

    # smoke run against a bank_builder.py --smoke output directory
    ROGII_BANK_LIMIT=8 uv run python notebooks/submission/kernel_bank_builder/bank_builder.py
    uv run python scripts/run_router_v2_cv.py --smoke outputs_smoke --skip-oracle-guard
"""

from __future__ import annotations

import argparse
import gc
import resource
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from rogii import data as D
from rogii.cv import well_folds
from rogii.stack_features import build_spatial_features, build_well_aggregate_features

_REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = _REPO_ROOT / "outputs"
STACK_OOF_PATH = OUTPUTS_DIR / "stack_v2_oof.npz"
TRACKER_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "tracker_cache"
SPATIAL_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "spatial_cache"

BACKTEST_FEATURES_FILENAME = "backtest_features.npz"
PFBMA_BANK_FILENAME = "path_bank_pfbma.npz"
BEAM_BANK_FILENAME = "path_bank_beam.npz"

ALL_IN_CONFIG_LABEL = "all-in (levers 1+2+3)"
STACK_V2_ALL_IN_POOLED_RMSE = 9.2309  # ledger 2026-07-07, sanity anchor
# ledger 2026-07-08, analyze_hard_wells.py oracle_substitution()
HARD_WELL_ORACLE_POOLED_RMSE = 7.7732
GUARD_TOL = 0.005
TOP_FRAC = 0.10
IMPROVEMENT_GATE_FT = 0.10
ADOPTION_GATE_RMSE = STACK_V2_ALL_IN_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.1309

BASE_CANDIDATES: tuple[str, ...] = ("stack", "carry_last", "pf_blend", "spatial")
PFBMA_REP_TARGET_SCALE = 8.0  # nearest-to-8 scale used as the pf_bma "family rep"

N_SPLITS_DEFAULT = 5
TAU_GRID_FT: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
SENTINEL_RMSE_FT = 200.0  # replaces inf/NaN candidate-RMSE labels for GBDT training
RANDOM_STATE = 42
PREFIX_SLOPE_K = 50  # last-K known-zone points used for the local TVT_input slope feature

# Two tiny, fixed LightGBM configs nested-selected alongside tau (task: "内側
# fold=trust-gate τ とハイパラ選択"). Not grid-searched beyond this -- the
# ledger's 3-iteration tuning budget is reserved for the real (773-well) run.
MODEL_GRID: tuple[dict[str, int], ...] = (
    {"n_estimators": 60, "num_leaves": 7, "max_depth": 3, "min_child_samples": 5},
    {"n_estimators": 30, "num_leaves": 3, "max_depth": 2, "min_child_samples": 3},
)
LGB_FIXED: dict[str, object] = {
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "n_jobs": 1,
}

# R3-v1's well-diagnostic block, reused verbatim (run_router_cv.py, read-only reference).
ROUTER_FEATURE_COLUMNS: tuple[str, ...] = (
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

# New for R3-v2: candidate-vs-candidate agreement, never candidate-vs-truth.
AGREE_FEATURE_COLUMNS: tuple[str, ...] = (
    "agree_pfbma_std_final",
    "agree_pfbma_std_mean",
    "agree_beam_std_final",
    "agree_beam_std_mean",
    "agree_family_std_final",
    "agree_family_std_mean",
    "agree_family_rms_max",
    "agree_family_rank_corr_min",
    "agree_pfbma_beam_rms",
    "agree_stack_bank_best_rms",
)


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pooled_rmse_from_sse(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


# --------------------------------------------------------------------------- #
# 1. well-name alignment primitives (pure, unit-tested)
# --------------------------------------------------------------------------- #


def join_well_names(*well_name_lists: Sequence[str]) -> list[str]:
    """Inner-join well names, preserving ``well_name_lists[0]``'s order.

    Robust to each list's own (possibly shuffled/unrelated) ordering -- only
    set membership matters for which wells survive; the *output* order is
    always the first list's order (by convention here, ``stack_v2_oof``'s
    ``used_wells``, per the task spec: "順序は stack_v2_oof の used_wells を
    正とし").
    """
    if not well_name_lists:
        return []
    common = set(well_name_lists[0])
    for lst in well_name_lists[1:]:
        common &= set(lst)
    return [w for w in well_name_lists[0] if w in common]


def block_slices(well_idx: np.ndarray, used_wells: np.ndarray) -> dict[str, tuple[int, int]]:
    """well name -> (start, end) row-slice from a well-block-contiguous ``well_idx``.

    ``well_idx`` must be sorted/contiguous per well (every row-level bank/OOF
    array in this repo is built by a single sequential per-well loop, so this
    always holds; violated only by a corrupt/hand-edited npz, which is why
    this raises rather than silently mis-slicing).
    """
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
    disagrees with the master eval-zone length for that well, is left as
    ``NaN`` for its whole block (never a partial/garbled copy) -- downstream
    SSE computation treats an all-NaN block as "candidate unavailable for
    this well" (SSE=inf), never a crash.
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


# --------------------------------------------------------------------------- #
# 2. data assembly: RouterV2Data
# --------------------------------------------------------------------------- #


@dataclass
class RouterV2Data:
    y_true: np.ndarray
    well_idx: np.ndarray
    used_wells: np.ndarray
    boundaries: np.ndarray
    candidate_rows: dict[str, np.ndarray]
    candidates: tuple[str, ...]
    family_reps: tuple[str, ...]
    pfbma_rep: str
    beam_rep: str
    backtest_by_well: dict[str, np.ndarray]
    backtest_feature_names: list[str]
    n_wells: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_wells = len(self.used_wells)


def _closest_scale_name(scales: list[float], target: float = PFBMA_REP_TARGET_SCALE) -> str:
    best = min(scales, key=lambda s: abs(s - target))
    return f"pf_bma_scale{best:g}"


def _load_backtest_features(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=True) as npz:
        matrix = npz["features"].astype(np.float32)
        feature_names = [str(x) for x in npz["feature_names"]]
        well_names = [str(x) for x in npz["well_names"]]
    by_well = {w: matrix[i] for i, w in enumerate(well_names)}
    return by_well, feature_names


def build_router_v2_data(bank_dir: Path) -> RouterV2Data:
    """Load + inner-join stack_v2_oof / backtest_features / path_bank_pfbma / path_bank_beam."""
    with np.load(STACK_OOF_PATH, allow_pickle=True) as npz:
        stack_y_true = npz["y_true"].astype(np.float64)
        stack_anchor = npz["anchor"].astype(np.float32)
        stack_pf_blend = npz["pf_blend"].astype(np.float32)
        stack_well_idx = npz["well_idx"]
        stack_used_wells = npz["used_wells"]
        config_labels = list(npz["config_labels"])
        all_in_i = config_labels.index(ALL_IN_CONFIG_LABEL)
        stack_pred = npz[f"oof_{all_in_i}"].astype(np.float32)
    stack_slice = block_slices(stack_well_idx, stack_used_wells)
    stack_wells = [str(w) for w in stack_used_wells]

    backtest_by_well, backtest_feature_names = _load_backtest_features(
        bank_dir / BACKTEST_FEATURES_FILENAME
    )

    with np.load(bank_dir / PFBMA_BANK_FILENAME, allow_pickle=True) as npz:
        pfbma_well_idx = npz["well_idx"]
        pfbma_used_wells = npz["used_wells"]
        pfbma_y_true = npz["y_true"].astype(np.float64)
        pfbma_scales = [float(s) for s in npz["bma_scales"]]
        pfbma_arrays = {
            s: npz[f"pf_bma_scale{s:g}_tvt"].astype(np.float32) for s in pfbma_scales
        }
    pfbma_slice = block_slices(pfbma_well_idx, pfbma_used_wells)
    pfbma_wells = [str(w) for w in pfbma_used_wells]

    with np.load(bank_dir / BEAM_BANK_FILENAME, allow_pickle=True) as npz:
        beam_well_idx = npz["well_idx"]
        beam_used_wells = npz["used_wells"]
        beam_y_true = npz["y_true"].astype(np.float64)
        beam_labels = [str(x) for x in npz["config_labels"]]
        beam_arrays = {
            label: npz[f"beam_tvt_{i}"].astype(np.float32) for i, label in enumerate(beam_labels)
        }
    beam_slice = block_slices(beam_well_idx, beam_used_wells)
    beam_wells = [str(w) for w in beam_used_wells]

    master_wells = join_well_names(
        stack_wells, list(backtest_by_well.keys()), pfbma_wells, beam_wells
    )
    for name, wells in (
        ("backtest_features", list(backtest_by_well.keys())),
        ("path_bank_pfbma", pfbma_wells),
        ("path_bank_beam", beam_wells),
    ):
        missing = sorted(set(stack_wells) - set(wells))
        if missing:
            print(
                f"  [align] {len(missing)} well(s) in stack_v2_oof missing from {name} -- "
                f"excluded from master join (e.g. {missing[:5]})"
            )
    n_dropped = len(stack_wells) - len(master_wells)
    print(
        f"  [align] master wells (inner join, stack order): {len(master_wells)}/"
        f"{len(stack_wells)} ({n_dropped} dropped)"
    )

    well_lengths = [stack_slice[w][1] - stack_slice[w][0] for w in master_wells]
    master_boundaries = np.concatenate([[0], np.cumsum(well_lengths)]).astype(np.int64)
    master_well_idx = np.repeat(
        np.arange(len(master_wells), dtype=np.int32), np.asarray(well_lengths, dtype=np.int32)
    )
    master_y_true = np.concatenate(
        [stack_y_true[stack_slice[w][0] : stack_slice[w][1]] for w in master_wells]
    ).astype(np.float32)

    # ---- regression-safety cross-check: bank y_true must agree with stack's ----
    for name, y_arr, sl in (("path_bank_pfbma", pfbma_y_true, pfbma_slice),
                             ("path_bank_beam", beam_y_true, beam_slice)):
        bank_y = np.concatenate([y_arr[sl[w][0] : sl[w][1]] for w in master_wells])
        if not np.allclose(bank_y, master_y_true, atol=1e-2):
            raise AssertionError(
                f"{name}'s y_true disagrees with stack_v2_oof's for the joined master "
                "wells -- alignment bug (row-order mismatch), not a leak, but must not "
                "be silently ignored"
            )

    candidate_rows: dict[str, np.ndarray] = {
        "stack": reindex_to_master(stack_pred, stack_slice, master_wells, master_boundaries),
        "carry_last": reindex_to_master(stack_anchor, stack_slice, master_wells, master_boundaries),
        "pf_blend": reindex_to_master(stack_pf_blend, stack_slice, master_wells, master_boundaries),
    }

    spatial_out = np.full(int(master_boundaries[-1]), np.nan, dtype=np.float32)
    n_spatial_missing = 0
    for i, well in enumerate(master_wells):
        ms, me = int(master_boundaries[i]), int(master_boundaries[i + 1])
        try:
            with np.load(SPATIAL_CACHE_DIR / f"{well}.npz") as npz:
                seg = npz["spatial_tvt"].astype(np.float32)
        except (FileNotFoundError, KeyError):
            n_spatial_missing += 1
            continue
        if seg.shape[0] == (me - ms):
            spatial_out[ms:me] = seg
        else:
            n_spatial_missing += 1
    if n_spatial_missing:
        print(f"  [align] spatial cache unavailable/mismatched for {n_spatial_missing} well(s)")
    candidate_rows["spatial"] = spatial_out

    for scale, arr in pfbma_arrays.items():
        candidate_rows[f"pf_bma_scale{scale:g}"] = reindex_to_master(
            arr, pfbma_slice, master_wells, master_boundaries
        )
    for label, arr in beam_arrays.items():
        candidate_rows[f"beam:{label}"] = reindex_to_master(
            arr, beam_slice, master_wells, master_boundaries
        )

    candidates = (
        BASE_CANDIDATES
        + tuple(f"pf_bma_scale{s:g}" for s in sorted(pfbma_scales))
        + tuple(f"beam:{lab}" for lab in beam_labels)
    )
    pfbma_rep = _closest_scale_name(pfbma_scales)
    beam_rep = f"beam:{beam_labels[0]}"
    family_reps = (*BASE_CANDIDATES, pfbma_rep, beam_rep)

    return RouterV2Data(
        y_true=master_y_true,
        well_idx=master_well_idx,
        used_wells=np.array(master_wells),
        boundaries=master_boundaries,
        candidate_rows=candidate_rows,
        candidates=candidates,
        family_reps=family_reps,
        pfbma_rep=pfbma_rep,
        beam_rep=beam_rep,
        backtest_by_well=backtest_by_well,
        backtest_feature_names=backtest_feature_names,
    )


# --------------------------------------------------------------------------- #
# 3. cross-candidate agreement features (never read y_true -- leak-boundary
#    critical, see tests/test_router_v2.py)
# --------------------------------------------------------------------------- #


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, NaN (never an exception/warning) on degenerate input.

    ``carry_last`` is exactly constant for every well by construction, so a
    pair involving it triggers scipy's ``ConstantInputWarning`` on every
    single well -- expected, not a bug, and silenced here rather than left to
    spam stdout for hundreds of wells on the real run.
    """
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            result = stats.spearmanr(a, b)
        rho = float(getattr(result, "statistic", result[0]))
    except Exception:
        return float("nan")
    return rho if np.isfinite(rho) else float("nan")


def _usable(seg_by_candidate: dict[str, np.ndarray], name: str) -> bool:
    """``name``'s segment exists, is non-empty, and is fully finite."""
    seg = seg_by_candidate.get(name)
    return seg is not None and seg.size > 0 and bool(np.all(np.isfinite(seg)))


def compute_agreement_features(
    seg_by_candidate: dict[str, np.ndarray],
    family_reps: tuple[str, ...],
    pfbma_rep: str,
    beam_rep: str,
) -> dict[str, float]:
    """Cross-candidate agreement features for one well's eval-zone rows.

    Takes only already-computed candidate *predictions* -- never ``TVT``/the
    eval-zone ``TVT_input`` -- so every feature here is legal at hidden-test
    inference time.
    """

    def _finite_stack(names: list[str]) -> np.ndarray | None:
        arrs = [seg_by_candidate[n] for n in names if _usable(seg_by_candidate, n)]
        return np.stack(arrs, axis=0) if arrs else None

    out: dict[str, float] = dict.fromkeys(AGREE_FEATURE_COLUMNS, float("nan"))

    pfbma_names = [n for n in seg_by_candidate if n.startswith("pf_bma_scale")]
    pfbma_stack = _finite_stack(pfbma_names)
    if pfbma_stack is not None and pfbma_stack.shape[0] >= 2 and pfbma_stack.shape[1] > 0:
        row_std = pfbma_stack.std(axis=0)
        out["agree_pfbma_std_final"] = float(row_std[-1])
        out["agree_pfbma_std_mean"] = float(row_std.mean())

    beam_names = [n for n in seg_by_candidate if n.startswith("beam:")]
    beam_stack = _finite_stack(beam_names)
    if beam_stack is not None and beam_stack.shape[0] >= 2 and beam_stack.shape[1] > 0:
        row_std = beam_stack.std(axis=0)
        out["agree_beam_std_final"] = float(row_std[-1])
        out["agree_beam_std_mean"] = float(row_std.mean())

    fam_arrs = {
        n: seg_by_candidate[n] for n in family_reps if _usable(seg_by_candidate, n)
    }
    if len(fam_arrs) >= 2:
        stacked = np.stack(list(fam_arrs.values()), axis=0)
        row_std = stacked.std(axis=0)
        out["agree_family_std_final"] = float(row_std[-1])
        out["agree_family_std_mean"] = float(row_std.mean())

        rms_vals: list[float] = []
        rank_corrs: list[float] = []
        names = list(fam_arrs.keys())
        for a, b in combinations(names, 2):
            va, vb = fam_arrs[a], fam_arrs[b]
            rms_vals.append(float(np.sqrt(np.mean((va - vb) ** 2))))
            rc = _safe_spearman(va, vb)
            if np.isfinite(rc):
                rank_corrs.append(rc)
        if rms_vals:
            out["agree_family_rms_max"] = float(max(rms_vals))
        if rank_corrs:
            out["agree_family_rank_corr_min"] = float(min(rank_corrs))

    if pfbma_rep in fam_arrs and beam_rep in fam_arrs:
        va, vb = fam_arrs[pfbma_rep], fam_arrs[beam_rep]
        out["agree_pfbma_beam_rms"] = float(np.sqrt(np.mean((va - vb) ** 2)))

    if "stack" in fam_arrs:
        stack_seg = fam_arrs["stack"]
        cand_rms = [
            float(np.sqrt(np.mean((stack_seg - fam_arrs[rep]) ** 2)))
            for rep in (pfbma_rep, beam_rep)
            if rep in fam_arrs
        ]
        if cand_rms:
            out["agree_stack_bank_best_rms"] = float(min(cand_rms))

    return out


# --------------------------------------------------------------------------- #
# 4. per-well SSE matrix + agreement feature matrix (single streaming pass)
# --------------------------------------------------------------------------- #


def build_well_matrices(rd: RouterV2Data) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Stream over wells once: per-well-per-candidate SSE + agreement features.

    Only ever holds one well's row slices in local scope at a time -- the big
    per-candidate row-level arrays (``rd.candidate_rows``) stay resident (a
    few hundred MB at 773 wells, not an all-pairs matrix), but nothing
    O(K^2 x N) or O(n_wells x N) is ever materialized.
    """
    n_wells, k = rd.n_wells, len(rd.candidates)
    sse_matrix = np.full((n_wells, k), np.inf, dtype=np.float64)
    well_n = np.zeros(n_wells, dtype=np.int64)
    agree_rows: list[dict[str, float]] = []

    for i in range(n_wells):
        s, e = int(rd.boundaries[i]), int(rd.boundaries[i + 1])
        n = e - s
        well_n[i] = n
        y = rd.y_true[s:e].astype(np.float64)

        seg_by_candidate: dict[str, np.ndarray] = {}
        for j, cand in enumerate(rd.candidates):
            seg = rd.candidate_rows[cand][s:e]
            seg_by_candidate[cand] = seg
            if seg.shape[0] == n and np.all(np.isfinite(seg)):
                sse_matrix[i, j] = float(np.sum((y - seg.astype(np.float64)) ** 2))

        agree_rows.append(
            compute_agreement_features(seg_by_candidate, rd.family_reps, rd.pfbma_rep, rd.beam_rep)
        )

    agree_df = pd.DataFrame(agree_rows, index=[str(w) for w in rd.used_wells])
    return sse_matrix, well_n, agree_df[list(AGREE_FEATURE_COLUMNS)]


# --------------------------------------------------------------------------- #
# 5. R3-v1 well-diagnostic features (reused verbatim from run_router_cv.py)
# --------------------------------------------------------------------------- #


def _prefix_slope(h: pd.DataFrame, k: int = PREFIX_SLOPE_K) -> float:
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
    if "Z" not in h.columns:
        return 0.0
    z = h["Z"].to_numpy(dtype=np.float64)[mask]
    z = z[np.isfinite(z)]
    return float(np.max(z) - np.min(z)) if z.size else 0.0


def build_diagnostic_features(used_wells: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for well in used_wells:
        well = str(well)
        with np.load(TRACKER_CACHE_DIR / f"{well}.npz") as npz:
            pf_tvt = npz["pf_tvt"].astype(np.float64)
            pf_std = npz["pf_std"].astype(np.float64)
            beam_tvt = npz["beam_tvt"].astype(np.float64)
            beam_margin = npz["beam_margin"].astype(np.float64)
            anchor = float(npz["anchor"])
        with np.load(SPATIAL_CACHE_DIR / f"{well}.npz") as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
            prefix_rmse = float(npz["prefix_rmse"])
            nn_dist_median = float(npz["nn_dist_median"])

        h = D.load_horizontal(well, "train")
        gr_nan_frac = float(h["GR"].isna().mean()) if "GR" in h.columns else 1.0
        mask = D.eval_mask(h)

        well_agg = build_well_aggregate_features(
            anchor, pf_tvt, pf_std, beam_tvt, beam_margin, gr_nan_frac
        )
        spatial_feats = build_spatial_features(anchor, spatial_tvt, prefix_rmse, nn_dist_median)

        beam_d = beam_tvt - anchor
        beam_d_final = float(beam_d[-1]) if beam_d.size and np.isfinite(beam_d[-1]) else 0.0
        pf_d_final = float(well_agg["pf_d_final"].iloc[-1])
        spatial_d_final = float(spatial_feats["spatial_d"].iloc[-1])
        disagree_std = float(np.std([pf_d_final, beam_d_final, spatial_d_final]))

        rows.append(
            {
                "well": well,
                "pf_d_final": pf_d_final,
                "pf_d_mean": float(well_agg["pf_d_mean"].iloc[-1]),
                "pf_std_well_mean": float(well_agg["pf_std_well_mean"].iloc[-1]),
                "pf_std_well_max": float(well_agg["pf_std_well_max"].iloc[-1]),
                "beam_margin_well_mean": float(well_agg["beam_margin_well_mean"].iloc[-1]),
                "pf_beam_abs_diff_well_mean": float(
                    well_agg["pf_beam_abs_diff_well_mean"].iloc[-1]
                ),
                "beam_d_final": beam_d_final,
                "eval_len": float(well_agg["eval_len"].iloc[-1]),
                "gr_nan_frac": gr_nan_frac,
                "spatial_d_final": spatial_d_final,
                "spatial_prefix_rmse": float(spatial_feats["spatial_prefix_rmse"].iloc[-1]),
                "spatial_nn_dist_median": float(spatial_feats["spatial_nn_dist_median"].iloc[-1]),
                "spatial_gated_d_final": float(spatial_feats["spatial_gated_d"].iloc[-1]),
                "z_span": _z_span(h, mask),
                "prefix_slope": _prefix_slope(h),
                "disagree_std": disagree_std,
            }
        )
    df = pd.DataFrame(rows).set_index("well")
    return df[list(ROUTER_FEATURE_COLUMNS)]


# --------------------------------------------------------------------------- #
# 6. oracle / greedy forward selection (reference-only, ground truth used for
#    *selection*, exactly as R3-v1's oracle guards -- never a feature)
# --------------------------------------------------------------------------- #


def hard_well_oracle_guard(
    candidates: tuple[str, ...], n_wells: int, sse_matrix: np.ndarray, well_n: np.ndarray
) -> float:
    """Reproduce run_router_cv.py's top-10% hard-well 4-way oracle substitution.

    Decoupled from :class:`RouterV2Data` (only ``candidates``/``n_wells`` are
    needed) so it is directly unit-testable against a synthetic SSE matrix.
    """
    cand_to_idx = {c: j for j, c in enumerate(candidates)}
    base_idx = [cand_to_idx[c] for c in BASE_CANDIDATES]
    stack_j = cand_to_idx["stack"]
    stack_sse = sse_matrix[:, stack_j]

    n_top = max(1, round(n_wells * TOP_FRAC))
    top_positions = set(np.argsort(-stack_sse)[:n_top].tolist())

    total_sse, total_n = 0.0, 0
    for i in range(n_wells):
        n = int(well_n[i])
        total_n += n
        if i in top_positions:
            sub = sse_matrix[i, base_idx]
            best = float(np.min(sub))
            total_sse += best if np.isfinite(best) else float(stack_sse[i])
        else:
            total_sse += float(stack_sse[i])
    return pooled_rmse_from_sse(total_sse, total_n)


def full_oracle_from_sse(
    sse_matrix: np.ndarray, well_n: np.ndarray, candidate_idx: list[int] | None = None
) -> float:
    """Reference-only ceiling: every well gets its own best (of ``candidate_idx``) candidate."""
    sub = sse_matrix if candidate_idx is None else sse_matrix[:, candidate_idx]
    best_per_well = np.min(sub, axis=1)
    total_n = int(well_n.sum())
    return pooled_rmse_from_sse(float(np.sum(best_per_well)), total_n)


def greedy_forward_selection(
    sse_matrix: np.ndarray, well_n: np.ndarray, candidates: tuple[str, ...]
) -> list[tuple[str, float]]:
    """Marginal contribution of each candidate to the full-switchable oracle.

    Greedily adds, one at a time, whichever remaining candidate most reduces
    the current all-wells oracle pooled RMSE; returns the pick order with the
    cumulative oracle RMSE after each addition (task: "候補追加の限界寄与").
    """
    n_wells, k = sse_matrix.shape
    total_n = int(well_n.sum())
    remaining = list(range(k))
    current_best = np.full(n_wells, np.inf)
    history: list[tuple[str, float]] = []

    while remaining:
        best_j, best_rmse, best_next = None, float("inf"), None
        for j in remaining:
            candidate_best = np.minimum(current_best, sse_matrix[:, j])
            rmse = pooled_rmse_from_sse(float(np.sum(candidate_best)), total_n)
            if rmse < best_rmse:
                best_rmse, best_j, best_next = rmse, j, candidate_best
        history.append((candidates[best_j], best_rmse))
        current_best = best_next
        remaining.remove(best_j)
    return history


# --------------------------------------------------------------------------- #
# 7. regression router: per-candidate RMSE regressors + trust-gate decision
# --------------------------------------------------------------------------- #


def rmse_from_sse(sse_matrix: np.ndarray, well_n: np.ndarray) -> np.ndarray:
    n = well_n.astype(np.float64)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(sse_matrix / n)


def sanitize_labels(rmse_matrix: np.ndarray, sentinel: float = SENTINEL_RMSE_FT) -> np.ndarray:
    out = rmse_matrix.copy()
    out[~np.isfinite(out)] = sentinel
    return out


def fit_predict_rmse_matrix(
    X_train: pd.DataFrame, Y_train: np.ndarray, X_val: pd.DataFrame, model_params: dict
) -> np.ndarray:
    """Fit one small LightGBM regressor per candidate; return ``(len(X_val), K)`` predictions.

    Never raises (playbook 00): any single candidate's fit/predict failure
    falls back to that candidate's training-fold mean RMSE (a neutral
    "about average" guess that rarely wins an argmin unless every model
    failed).
    """
    n_val, k = len(X_val), Y_train.shape[1]
    out = np.empty((n_val, k), dtype=np.float64)
    params = {**LGB_FIXED, **model_params}
    for j in range(k):
        y_col = Y_train[:, j]
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_col)
            out[:, j] = model.predict(X_val)
        except Exception:
            out[:, j] = float(np.mean(y_col)) if y_col.size else SENTINEL_RMSE_FT
    return out


def decide(pred_rmse_row: np.ndarray, candidates: tuple[str, ...], tau: float) -> str:
    """Argmin the predicted-RMSE row; switch away from ``stack`` only if the
    predicted improvement clears the trust-gate ``tau`` (feet)."""
    stack_j = candidates.index("stack")
    j = int(np.argmin(pred_rmse_row))
    if candidates[j] == "stack":
        return "stack"
    improvement = float(pred_rmse_row[stack_j] - pred_rmse_row[j])
    return candidates[j] if improvement >= tau else "stack"


def pooled_rmse_from_decisions(
    decisions: dict[str, str],
    well_to_idx: dict[str, int],
    sse_matrix: np.ndarray,
    well_n: np.ndarray,
    cand_to_idx: dict[str, int],
) -> float:
    total_sse, total_n = 0.0, 0
    stack_j = cand_to_idx["stack"]
    for well, cand in decisions.items():
        i = well_to_idx[well]
        j = cand_to_idx[cand]
        sse = sse_matrix[i, j]
        if not np.isfinite(sse):  # picked candidate unavailable for this well -- fall back
            sse = sse_matrix[i, stack_j]
        total_sse += float(sse)
        total_n += int(well_n[i])
    return pooled_rmse_from_sse(total_sse, total_n)


def effective_n_splits(n_wells: int, desired: int = N_SPLITS_DEFAULT) -> int:
    """Auto-degrade the outer fold count for small (e.g. smoke) well sets."""
    if n_wells < 4:
        return max(1, n_wells)
    return max(2, min(desired, n_wells // 2))


@dataclass
class RouterV2Result:
    nested_pred: np.ndarray
    chosen: dict[str, str]
    tau_by_fold: dict[int, float]
    model_by_fold: dict[int, int]
    well_fold: np.ndarray


def run_nested_router_v2(
    rd: RouterV2Data,
    features: pd.DataFrame,
    label_matrix: np.ndarray,
    sse_matrix: np.ndarray,
    well_n: np.ndarray,
    n_splits: int,
) -> RouterV2Result:
    wells = [str(w) for w in rd.used_wells]
    well_to_idx = {w: i for i, w in enumerate(wells)}
    cand_to_idx = {c: j for j, c in enumerate(rd.candidates)}

    folds = well_folds(wells, n_splits=n_splits, seed=RANDOM_STATE)
    well_to_fold = {w: f for f, group in enumerate(folds) for w in group}
    well_fold = np.array([well_to_fold[w] for w in wells], dtype=np.int32)

    chosen: dict[str, str] = {}
    tau_by_fold: dict[int, float] = {}
    model_by_fold: dict[int, int] = {}

    for f in range(n_splits):
        train_mask = well_fold != f
        test_mask = well_fold == f
        train_positions = np.where(train_mask)[0]
        test_positions = np.where(test_mask)[0]
        if train_positions.size == 0 or test_positions.size == 0:
            for pos in test_positions:
                chosen[wells[pos]] = "stack"
            continue

        # ---- inner leave-one-subfold-out: nested (model_config, tau) selection ----
        inner_pred_by_model: dict[int, dict[str, np.ndarray]] = {
            m: {} for m in range(len(MODEL_GRID))
        }
        for g in range(n_splits):
            if g == f:
                continue
            inner_val_positions = np.where(train_mask & (well_fold == g))[0]
            inner_fit_positions = np.where(train_mask & (well_fold != g))[0]
            if inner_val_positions.size == 0 or inner_fit_positions.size == 0:
                continue
            for m, model_params in enumerate(MODEL_GRID):
                pred = fit_predict_rmse_matrix(
                    features.iloc[inner_fit_positions],
                    label_matrix[inner_fit_positions],
                    features.iloc[inner_val_positions],
                    model_params,
                )
                for row_i, pos in enumerate(inner_val_positions):
                    inner_pred_by_model[m][wells[pos]] = pred[row_i]

        best_model, best_tau, best_rmse = 0, TAU_GRID_FT[0], float("inf")
        for m in range(len(MODEL_GRID)):
            covered = list(inner_pred_by_model[m].keys())
            if not covered:
                continue
            preds_m = inner_pred_by_model[m]
            for tau in TAU_GRID_FT:
                decisions = {w: decide(preds_m[w], rd.candidates, tau) for w in covered}
                rmse = pooled_rmse_from_decisions(
                    decisions, well_to_idx, sse_matrix, well_n, cand_to_idx
                )
                if rmse < best_rmse:
                    best_rmse, best_model, best_tau = rmse, m, tau
        model_by_fold[f] = best_model
        tau_by_fold[f] = best_tau

        # ---- final: fit on ALL train-fold wells, predict test-fold wells ----
        pred_test = fit_predict_rmse_matrix(
            features.iloc[train_positions],
            label_matrix[train_positions],
            features.iloc[test_positions],
            MODEL_GRID[best_model],
        )
        for row_i, pos in enumerate(test_positions):
            chosen[wells[pos]] = decide(pred_test[row_i], rd.candidates, best_tau)

        print(
            f"  outer fold {f}: train={len(train_positions)} test={len(test_positions)} "
            f"model*={best_model} tau*={best_tau:.2f} (inner pooled={best_rmse:.4f})"
        )

    nested_pred = np.empty_like(rd.y_true)
    for i, well in enumerate(wells):
        s, e = int(rd.boundaries[i]), int(rd.boundaries[i + 1])
        cand = chosen.get(well, "stack")
        seg = rd.candidate_rows[cand][s:e]
        if seg.shape[0] != (e - s) or not np.all(np.isfinite(seg)):
            seg = rd.candidate_rows["stack"][s:e]  # picked candidate unusable -- fall back
        nested_pred[s:e] = seg

    return RouterV2Result(nested_pred, chosen, tau_by_fold, model_by_fold, well_fold)


def helped_hurt_report(
    rd: RouterV2Data, sse_matrix: np.ndarray, well_n: np.ndarray, chosen: dict[str, str]
) -> None:
    wells = [str(w) for w in rd.used_wells]
    cand_to_idx = {c: j for j, c in enumerate(rd.candidates)}
    stack_j = cand_to_idx["stack"]

    switched = {w: c for w, c in chosen.items() if c != "stack"}
    print(f"\nswitched wells: {len(switched)}/{len(chosen)}")
    if switched:
        counts = pd.Series(list(switched.values())).value_counts()
        print(f"  breakdown (top 10): {counts.head(10).to_dict()}")

    helped = hurt = tied = 0
    helped_deltas: list[float] = []
    hurt_deltas: list[float] = []
    for i, well in enumerate(wells):
        cand = chosen.get(well, "stack")
        j = cand_to_idx[cand]
        n = well_n[i]
        before = np.sqrt(sse_matrix[i, stack_j] / n)
        after_sse = sse_matrix[i, j] if np.isfinite(sse_matrix[i, j]) else sse_matrix[i, stack_j]
        after = np.sqrt(after_sse / n)
        if after < before - 1e-9:
            helped += 1
            helped_deltas.append(before - after)
        elif after > before + 1e-9:
            hurt += 1
            hurt_deltas.append(after - before)
        else:
            tied += 1
    print(f"  helped={helped} hurt={hurt} tied={tied}")
    if helped_deltas:
        print(f"  median RMSE improvement (helped wells): {np.median(helped_deltas):.4f} ft")
    if hurt_deltas:
        print(f"  median RMSE regression (hurt wells):    {np.median(hurt_deltas):.4f} ft")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        type=str,
        default=None,
        help="Directory holding a bank_builder.py --smoke output "
        "(backtest_features.npz / path_bank_pfbma.npz / path_bank_beam.npz). "
        "Omit to read the full bank from outputs/.",
    )
    parser.add_argument(
        "--skip-oracle-guard",
        action="store_true",
        help="Skip the 4-way hard-well-oracle regression-guard assert (smoke banks "
        "are a well subset, so the ledger's 7.7732 value cannot reproduce exactly).",
    )
    args = parser.parse_args()
    t_start = time.time()
    is_smoke = args.smoke is not None
    bank_dir = Path(args.smoke) if is_smoke else OUTPUTS_DIR

    print(f"loading stack_v2_oof.npz + bank npz (bank_dir={bank_dir}, smoke={is_smoke}) ...")
    rd = build_router_v2_data(bank_dir)
    print(f"  master wells: {rd.n_wells}  candidates: {len(rd.candidates)}")
    print(f"  candidates: {rd.candidates}")
    print(f"  total eval rows: {rd.y_true.shape[0]:,}")

    baseline_pooled = pooled_rmse(rd.y_true, rd.candidate_rows["stack"])
    carry_pooled = pooled_rmse(rd.y_true, rd.candidate_rows["carry_last"])
    pf_pooled = pooled_rmse(rd.y_true, rd.candidate_rows["pf_blend"])
    print(f"  baseline (stack) pooled RMSE over master wells  = {baseline_pooled:.4f}")

    print("\nbuilding per-well SSE matrix + cross-candidate agreement features (streaming) ...")
    t0 = time.time()
    sse_matrix, well_n, agree_df = build_well_matrices(rd)
    print(f"  done ({time.time() - t0:.1f}s)")

    print("\nreproducing R3-v1's 4-way hard-well oracle regression guard ...")
    oracle4 = hard_well_oracle_guard(rd.candidates, rd.n_wells, sse_matrix, well_n)
    if args.skip_oracle_guard:
        print(f"  [skip-oracle-guard] hard-well oracle (top10%) = {oracle4:.4f} (assert skipped)")
    else:
        delta = oracle4 - HARD_WELL_ORACLE_POOLED_RMSE
        assert abs(delta) < GUARD_TOL, (
            f"hard-well oracle (top-10%) pooled RMSE {oracle4:.4f} drifted from ledger "
            f"{HARD_WELL_ORACLE_POOLED_RMSE} by {delta:+.4f}"
        )
        print(
            f"  guard PASSED: {oracle4:.4f} "
            f"(ledger {HARD_WELL_ORACLE_POOLED_RMSE}, delta={delta:+.4f})"
        )

    full_k = full_oracle_from_sse(sse_matrix, well_n)
    print(
        f"\n{len(rd.candidates)}-way full-switchable oracle (reference only, NOT deployable) "
        f"= {full_k:.4f}  (delta vs baseline {full_k - baseline_pooled:+.4f})"
    )

    print("\ngreedy forward candidate selection (oracle marginal contribution) ...")
    history = greedy_forward_selection(sse_matrix, well_n, rd.candidates)
    for rank, (name, rmse) in enumerate(history, start=1):
        print(f"  {rank:2d}. +{name:<28} cumulative oracle pooled RMSE = {rmse:.4f}")

    print("\nbuilding backtest(48) + R3-v1 diagnostic feature blocks ...")
    bt_df = pd.DataFrame(
        {w: rd.backtest_by_well[w] for w in [str(w) for w in rd.used_wells]}
    ).T
    bt_df.columns = rd.backtest_feature_names
    diag_df = build_diagnostic_features(rd.used_wells)
    features = pd.concat([bt_df, agree_df, diag_df], axis=1).astype(np.float64)
    print(f"  feature matrix: {features.shape} (48 backtest + {len(AGREE_FEATURE_COLUMNS)} "
          f"agreement + {len(ROUTER_FEATURE_COLUMNS)} diagnostic)")

    n_splits = effective_n_splits(rd.n_wells)
    print(
        f"\nrunning nested well-GroupKFold({n_splits}) router "
        f"(regression + trust-gate, n_wells={rd.n_wells}) ..."
    )
    label_matrix = sanitize_labels(rmse_from_sse(sse_matrix, well_n))
    result = run_nested_router_v2(rd, features, label_matrix, sse_matrix, well_n, n_splits)
    nested_pooled = pooled_rmse(rd.y_true, result.nested_pred)

    row_fold_master = result.well_fold[rd.well_idx]
    print("\nper-fold nested pooled RMSE:")
    for f in range(n_splits):
        mask = row_fold_master == f
        if not np.any(mask):
            continue
        fold_rmse = pooled_rmse(rd.y_true[mask], result.nested_pred[mask])
        print(f"  fold {f}: n_rows={int(mask.sum()):,} pooled_rmse={fold_rmse:.4f}")

    def _row(label: str, rmse: float) -> str:
        return f"{label:<42}{rmse:>13.4f}{rmse - baseline_pooled:>+13.4f}"

    print("\n" + "=" * 92)
    print(f"{'predictor':<42}{'pooled RMSE':>13}{'delta base':>13}")
    print("-" * 92)
    print(_row("carry_last (floor)", carry_pooled))
    print(_row("PF blend w0.7", pf_pooled))
    print(_row("stack v2 all-in (baseline)", baseline_pooled))
    print(_row("hard-well oracle (top10%, guard)", oracle4))
    print(_row(f"{len(rd.candidates)}-way full oracle (reference)", full_k))
    print(_row("R3-v2 router (nested)", nested_pooled))
    print("=" * 92)

    print(f"\ntau by outer fold: {result.tau_by_fold}")
    print(f"model config by outer fold: {result.model_by_fold}")

    helped_hurt_report(rd, sse_matrix, well_n, result.chosen)

    improvement = baseline_pooled - nested_pooled
    if is_smoke:
        print(
            f"\n{'=' * 92}\n[smoke] pipeline check only -- verdict below is NOT a real "
            f"experimental result ({rd.n_wells}-well subset).\n{'=' * 92}"
        )
    verdict = "採用" if nested_pooled <= ADOPTION_GATE_RMSE else "却下（非改善）"
    print(f"\n{'=' * 92}")
    print(
        f"判定: {verdict} -- nested router={nested_pooled:.4f}, baseline={baseline_pooled:.4f}, "
        f"改善={improvement:+.4f}ft (gate: <= {ADOPTION_GATE_RMSE:.4f}ft)"
    )
    print(f"{'=' * 92}")

    del rd.candidate_rows, sse_matrix, features, label_matrix
    gc.collect()
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\npeak RSS (whole-process high-water mark): {peak_mb:.1f} MB")
    print(f"total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
