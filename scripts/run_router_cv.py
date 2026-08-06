"""Per-well candidate router with a trust-gate (design.md §11 R3-a+R3-b combined).

Hypothesis under test (1 hypothesis / 1 measurement, design.md §11 R3): a
**well-level classifier that picks the best of 4 already-computed candidate
predictors** -- stack v2 all-in / carry_last / PF blend w0.7 / spatial (raw
leave-self-out KNN) -- switching away from the stack v2 baseline only when
the classifier's confidence clears a trust-gate threshold ``tau``, can beat
the stack v2 baseline (pooled **9.2309**) by >= 0.10ft, honestly measured
with **nested** well-level GroupKFold(5): ``tau`` itself is selected per
outer fold using only the other four folds' *inner*-OOF probabilities (never
the fold it is scored on), mirroring ``scripts/run_postproc_cv.py``'s
``_nested_select``.

The 4-way oracle ceiling this hypothesis chases was measured by
``scripts/analyze_hard_wells.py`` (NOT modified/re-run here, only its
top-10%-hard-well oracle-substitution number is *reproduced* below as a
regression guard): substituting the SSE-top-10% "hard" wells with a per-well
oracle pick among the same 4 candidates (using the true label to choose --
not deployable) gives pooled RMSE **7.7732** (ledger 2026-07-08), a
**-1.4577ft** ceiling relative to the 9.2309 baseline.

No retraining of stack v2 itself: every candidate is read from an existing,
already leave-self-out / OOF artifact --

  - ``stack``      : ``outputs/stack_v2_oof.npz`` "all-in" config's OOF column
  - ``carry_last`` : the same npz's ``anchor`` column (flat extension)
  - ``pf_blend``   : the same npz's ``pf_blend`` column (PF blend w0.7)
  - ``spatial``    : ``data/processed/spatial_cache/{well}.npz``'s raw
                      ``spatial_tvt`` (leave-self-out KNN, ungated -- the same
                      definition ``analyze_hard_wells.py`` uses for its
                      "spatial" oracle-substitution candidate)

Router features (``ROUTER_FEATURE_COLUMNS``) are well-level scalars
computable from the known-zone prefix alone: tracker-cache / spatial-cache
diagnostics (already used as leak-safe stack v2 features, see
``rogii.stack_features``), GR-log NaN rate (GR is present in both train/test
schemas), a local slope of ``TVT_input`` fit only over the *known* zone, and
eval-zone/Z-survey geometry. None of these read ``TVT``, the eval-zone
``TVT_input`` (which is NaN there by definition), or any train-only formation
column. The classifier's *training labels* (which candidate had the lowest
SSE for a well) do use the true label -- exactly like any supervised model's
training target -- but the leak boundary being defended is that no eval-zone
value ever becomes a *feature seen at decision time*, and every outer test
fold is scored by a classifier (and a trust-gate ``tau``) that never saw
that fold's wells.

Usage::

    uv run python scripts/run_router_cv.py                 # full 773-well run
    uv run python scripts/run_router_cv.py --n-wells 80    # health-check subset
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rogii import data as D
from rogii.stack_features import build_spatial_features, build_well_aggregate_features

_REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = _REPO_ROOT / "outputs" / "stack_v2_oof.npz"
TRACKER_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "tracker_cache"
TRACKER_MANIFEST_PATH = TRACKER_CACHE_DIR / "manifest.csv"
SPATIAL_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "spatial_cache"

ALL_IN_CONFIG_LABEL = "all-in (levers 1+2+3)"
STACK_V2_ALL_IN_POOLED_RMSE = 9.2309  # ledger 2026-07-07, sanity anchor for this script
CARRY_LAST_POOLED_RMSE = 15.9099  # ledger 2026-06-17
PF_BLEND_POOLED_RMSE = 11.0621  # ledger 2026-07-06
HARD_WELL_ORACLE_POOLED_RMSE = 7.7732  # ledger 2026-07-08, analyze_hard_wells.py top-10% oracle
TOP_FRAC = 0.10
GUARD_TOL = 0.01

CANDIDATES: tuple[str, ...] = ("stack", "carry_last", "pf_blend", "spatial")
CANDIDATE_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CANDIDATES)}

N_SPLITS = 5
IMPROVEMENT_GATE_FT = 0.10
TAU_GRID: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
PREFIX_SLOPE_K = 50  # last-K known-zone points used for the local TVT_input slope feature
RANDOM_STATE = 42

# Every feature here is computable from the known-zone prefix alone (pre-inference
# safe): tracker/spatial-cache scalars are leave-self-out by construction (see
# docs/playbooks/00_common.md), GR is present in both train/test schemas, MD/Z are
# always known, and TVT_input is only read over its non-NaN (known) prefix.
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


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pooled_sse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


# --------------------------------------------------------------------------- #
# 1. load OOF + all 4 candidate row-level arrays
# --------------------------------------------------------------------------- #


class RouterData:
    """Row-level arrays + well-block bookkeeping for all 4 routing candidates."""

    def __init__(
        self,
        y_true: np.ndarray,
        well_idx: np.ndarray,
        used_wells: np.ndarray,
        boundaries: np.ndarray,
        row_fold: np.ndarray,
        candidate_rows: dict[str, np.ndarray],
    ) -> None:
        self.y_true = y_true
        self.well_idx = well_idx
        self.used_wells = used_wells
        self.boundaries = boundaries
        self.row_fold = row_fold
        self.candidate_rows = candidate_rows
        self.n_wells = len(used_wells)


def load_router_data() -> RouterData:
    with np.load(OOF_PATH, allow_pickle=True) as npz:
        y_true = npz["y_true"].astype(np.float64)
        anchor = npz["anchor"].astype(np.float64)
        pf_blend = npz["pf_blend"].astype(np.float64)
        well_idx = npz["well_idx"]
        row_fold = npz["row_fold"]
        used_wells = npz["used_wells"]
        config_labels = list(npz["config_labels"])
        all_in_i = config_labels.index(ALL_IN_CONFIG_LABEL)
        stack_pred = npz[f"oof_{all_in_i}"].astype(np.float64)

    n_wells = len(used_wells)
    boundaries = np.searchsorted(well_idx, np.arange(n_wells + 1))
    assert boundaries[-1] == well_idx.shape[0], "well_idx blocks are not contiguous/sorted"

    baseline = pooled_rmse(y_true, stack_pred)
    assert abs(baseline - STACK_V2_ALL_IN_POOLED_RMSE) < GUARD_TOL, (
        f"stack v2 all-in pooled RMSE {baseline:.4f} drifted from ledger "
        f"{STACK_V2_ALL_IN_POOLED_RMSE} -- npz may be stale"
    )
    pf_pooled = pooled_rmse(y_true, pf_blend)
    assert abs(pf_pooled - PF_BLEND_POOLED_RMSE) < GUARD_TOL, (
        f"pf_blend column pooled RMSE {pf_pooled:.4f} drifted from ledger "
        f"{PF_BLEND_POOLED_RMSE} -- npz may be stale"
    )

    print(f"  loading spatial cache for {n_wells} wells ...")
    spatial_pred = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    t0 = time.time()
    for i, well in enumerate(used_wells):
        well = str(well)
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        with np.load(SPATIAL_CACHE_DIR / f"{well}.npz") as npz2:
            spatial_tvt = npz2["spatial_tvt"].astype(np.float64)
        if spatial_tvt.size != (e - s):
            raise ValueError(f"{well}: spatial cache length mismatch vs OOF block")
        spatial_pred[s:e] = spatial_tvt
    print(f"  spatial cache loaded ({time.time() - t0:.1f}s)")

    candidate_rows = {
        "stack": stack_pred,
        "carry_last": anchor,
        "pf_blend": pf_blend,
        "spatial": spatial_pred,
    }
    return RouterData(y_true, well_idx, used_wells, boundaries, row_fold, candidate_rows)


def _stratified_health_check_wells(
    used_wells: np.ndarray, n: int, seed: int = RANDOM_STATE
) -> list[str]:
    """Deterministic small subset for a fast crash/NaN sanity pass before the full run."""
    if not TRACKER_MANIFEST_PATH.exists():
        return sorted(used_wells.tolist())[:n]
    manifest = pd.read_csv(TRACKER_MANIFEST_PATH)
    manifest = manifest[manifest["well"].isin(used_wells.tolist())].copy()
    manifest["decile"] = pd.qcut(manifest["pf_std_mean"], 10, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    per_stratum = max(1, n // manifest["decile"].nunique())
    picked: list[str] = []
    for _, group in manifest.groupby("decile"):
        wells_in_group = sorted(group["well"].tolist())
        rng.shuffle(wells_in_group)
        picked.extend(wells_in_group[:per_stratum])
    return sorted(picked[:n])


def _subset_router_data(rd: RouterData, subset_wells: set[str]) -> RouterData:
    keep_positions = [i for i, w in enumerate(rd.used_wells) if str(w) in subset_wells]
    used_wells = rd.used_wells[keep_positions]
    row_mask = np.isin(rd.well_idx, keep_positions)
    remap = {old: new for new, old in enumerate(keep_positions)}
    well_idx = np.array([remap[w] for w in rd.well_idx[row_mask]], dtype=np.int32)
    y_true = rd.y_true[row_mask]
    row_fold = rd.row_fold[row_mask]
    candidate_rows = {c: arr[row_mask] for c, arr in rd.candidate_rows.items()}
    n_wells = len(used_wells)
    boundaries = np.searchsorted(well_idx, np.arange(n_wells + 1))
    return RouterData(y_true, well_idx, used_wells, boundaries, row_fold, candidate_rows)


# --------------------------------------------------------------------------- #
# 2. per-well candidate SSE / best-candidate label (uses true y -- label only)
# --------------------------------------------------------------------------- #


def build_well_labels(rd: RouterData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(rd.n_wells):
        s, e = int(rd.boundaries[i]), int(rd.boundaries[i + 1])
        n = e - s
        y = rd.y_true[s:e]
        row: dict[str, object] = {"well": str(rd.used_wells[i]), "n": n}
        sse_by_cand: dict[str, float] = {}
        for cand in CANDIDATES:
            seg = rd.candidate_rows[cand][s:e]
            sse = pooled_sse(y, seg) if seg.size and np.all(np.isfinite(seg)) else float("inf")
            row[f"sse_{cand}"] = sse
            sse_by_cand[cand] = sse
        row["best_candidate"] = min(sse_by_cand, key=lambda c: sse_by_cand[c])
        rows.append(row)
    return pd.DataFrame(rows).set_index("well")


def hard_well_oracle_guard(rd: RouterData, well_labels: pd.DataFrame) -> float:
    """Reproduce analyze_hard_wells.py's top-10% per-well oracle substitution.

    Ranking key = each well's own stack-candidate SSE (matches that script's
    ``per_well_error_decomposition``'s ``sse_total``, computed from the same
    all-in stack OOF column loaded here). Only the top decile's rows are
    substituted with the true-label-selected best candidate; every other
    well keeps its stack prediction -- an exact behavioral match to
    ``analyze_hard_wells.py::oracle_substitution``'s per-well-oracle branch.
    """
    stack_sse = well_labels["sse_stack"]
    n_top = max(1, round(rd.n_wells * TOP_FRAC))
    top_wells = set(stack_sse.sort_values(ascending=False).index[:n_top])

    pred = rd.candidate_rows["stack"].copy()
    for i in range(rd.n_wells):
        well = str(rd.used_wells[i])
        if well not in top_wells:
            continue
        s, e = int(rd.boundaries[i]), int(rd.boundaries[i + 1])
        best = str(well_labels.loc[well, "best_candidate"])
        pred[s:e] = rd.candidate_rows[best][s:e]

    return pooled_rmse(rd.y_true, pred)


def full_oracle_rmse(rd: RouterData, well_labels: pd.DataFrame) -> float:
    """Reference-only ceiling: every well (not just the top decile) gets the
    true-label-selected best candidate. NOT deployable (uses ground truth to
    pick) -- reported only to bound the router's own headroom since, unlike
    ``analyze_hard_wells.py``, the router is free to switch any well.
    """
    pred = np.empty_like(rd.y_true)
    for i in range(rd.n_wells):
        well = str(rd.used_wells[i])
        s, e = int(rd.boundaries[i]), int(rd.boundaries[i + 1])
        best = str(well_labels.loc[well, "best_candidate"])
        pred[s:e] = rd.candidate_rows[best][s:e]
    return pooled_rmse(rd.y_true, pred)


# --------------------------------------------------------------------------- #
# 3. router features (well-level, pre-inference safe)
# --------------------------------------------------------------------------- #


def _prefix_slope(h: pd.DataFrame, k: int = PREFIX_SLOPE_K) -> float:
    """Local linear slope of ``TVT_input`` vs ``MD`` over the last ``k`` known rows.

    Uses only the non-NaN (known) prefix, i.e. rows strictly before the eval
    zone -- no leakage, this is available at hidden-test inference time.
    """
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
    """Z (TVD) range across the eval zone -- a geometric survey column, always known."""
    if "Z" not in h.columns:
        return 0.0
    z = h["Z"].to_numpy(dtype=np.float64)[mask]
    z = z[np.isfinite(z)]
    return float(np.max(z) - np.min(z)) if z.size else 0.0


def build_router_features(used_wells: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    t0 = time.time()
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
    print(f"  router features built for {len(used_wells)} wells ({time.time() - t0:.1f}s)")
    df = pd.DataFrame(rows).set_index("well")
    return df[list(ROUTER_FEATURE_COLUMNS)]


# --------------------------------------------------------------------------- #
# 4. nested trust-gate router
# --------------------------------------------------------------------------- #


def _well_fold_array(rd: RouterData) -> np.ndarray:
    well_fold = rd.row_fold[rd.boundaries[:-1]]
    for i in range(rd.n_wells):
        s, e = int(rd.boundaries[i]), int(rd.boundaries[i + 1])
        if not np.all(rd.row_fold[s:e] == well_fold[i]):
            raise ValueError(f"well {rd.used_wells[i]} rows span multiple folds")
    return well_fold


def _fit_predict_proba(
    X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame
) -> np.ndarray:
    """Fit a small logistic-regression router; return a ``(len(X_val), 4)`` proba matrix.

    Columns always align to ``CANDIDATES`` order regardless of which classes
    were present in ``y_train`` (a class absent from the training fold gets
    probability 0 -- it can never be selected). Never raises (playbook 00:
    predictors must not leak exceptions): any fit/predict failure falls back
    to "always stack" (the no-switch floor, i.e. behaviorally identical to
    not routing that well at all).
    """
    try:
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        raw_proba = pipe.predict_proba(X_val)
        classes_present = pipe.named_steps["clf"].classes_
    except Exception:
        proba = np.zeros((len(X_val), len(CANDIDATES)), dtype=np.float64)
        proba[:, CANDIDATE_TO_IDX["stack"]] = 1.0
        return proba

    proba = np.zeros((len(X_val), len(CANDIDATES)), dtype=np.float64)
    for j, cls in enumerate(classes_present):
        proba[:, int(cls)] = raw_proba[:, j]
    return proba


def _decide(proba_row: np.ndarray, tau: float) -> str:
    idx = int(np.argmax(proba_row))
    cand = CANDIDATES[idx]
    if cand != "stack" and proba_row[idx] >= tau:
        return cand
    return "stack"


def _pooled_rmse_from_decisions(
    well_labels: pd.DataFrame, wells: list[str], decisions: dict[str, str]
) -> float:
    sse, n = 0.0, 0
    for well in wells:
        cand = decisions[well]
        sse += float(well_labels.loc[well, f"sse_{cand}"])
        n += int(well_labels.loc[well, "n"])
    return float(np.sqrt(sse / n)) if n else float("nan")


@dataclass
class RouterResult:
    nested_pred: np.ndarray
    chosen: dict[str, str]  # well -> candidate, nested-selected tau (headline)
    tau_by_fold: dict[int, float]
    test_proba: dict[str, np.ndarray]  # well -> proba row (len 4), for the tau sweep


def run_nested_router(
    rd: RouterData, well_labels: pd.DataFrame, features: pd.DataFrame
) -> RouterResult:
    """Outer well-GroupKFold(5) + inner leave-one-subfold-out tau selection.

    For outer fold ``f``: the 4 other folds' wells get an inner-OOF proba
    (each inner sub-fold ``g`` scored by a classifier fit on the remaining 3,
    so it never saw the wells it scores). ``tau`` is chosen by minimizing
    pooled RMSE over those inner-OOF decisions -- fold ``f`` never enters this
    selection. A final classifier is then fit on *all* of fold ``f``'s
    complementary wells and applied (with the selected ``tau``) to fold
    ``f``'s wells. Every well is scored exactly once, by a classifier (and a
    tau) that never saw it during fitting or tuning.
    """
    well_fold = _well_fold_array(rd)
    y_codes = well_labels["best_candidate"].map(CANDIDATE_TO_IDX).to_numpy()
    wells = [str(w) for w in rd.used_wells]

    test_proba: dict[str, np.ndarray] = {}
    chosen: dict[str, str] = {}
    tau_by_fold: dict[int, float] = {}

    for f in range(N_SPLITS):
        t_fold = time.time()
        train_mask = well_fold != f
        test_mask = well_fold == f
        train_positions = np.where(train_mask)[0]
        test_positions = np.where(test_mask)[0]
        train_wells = [wells[i] for i in train_positions]

        # ---- inner leave-one-subfold-out OOF decisions on the train wells ----
        inner_decisions_by_tau: dict[float, dict[str, str]] = {tau: {} for tau in TAU_GRID}
        for g in range(N_SPLITS):
            if g == f:
                continue
            inner_val_mask = train_mask & (well_fold == g)
            inner_fit_mask = train_mask & (well_fold != g)
            inner_val_positions = np.where(inner_val_mask)[0]
            inner_fit_positions = np.where(inner_fit_mask)[0]
            if inner_val_positions.size == 0 or inner_fit_positions.size == 0:
                continue
            proba = _fit_predict_proba(
                features.iloc[inner_fit_positions],
                y_codes[inner_fit_positions],
                features.iloc[inner_val_positions],
            )
            for row_i, pos in enumerate(inner_val_positions):
                well = wells[pos]
                for tau in TAU_GRID:
                    inner_decisions_by_tau[tau][well] = _decide(proba[row_i], tau)

        best_tau, best_rmse = TAU_GRID[0], float("inf")
        for tau in TAU_GRID:
            decisions = inner_decisions_by_tau[tau]
            covered = [w for w in train_wells if w in decisions]
            if not covered:
                continue
            rmse = _pooled_rmse_from_decisions(well_labels, covered, decisions)
            if rmse < best_rmse:
                best_rmse, best_tau = rmse, tau
        tau_by_fold[f] = best_tau

        # ---- final classifier: fit on ALL train-fold wells, predict test-fold wells ----
        proba_test = _fit_predict_proba(
            features.iloc[train_positions],
            y_codes[train_positions],
            features.iloc[test_positions],
        )
        for row_i, pos in enumerate(test_positions):
            well = wells[pos]
            test_proba[well] = proba_test[row_i]
            chosen[well] = _decide(proba_test[row_i], best_tau)

        print(
            f"  outer fold {f}: train={len(train_positions)} test={len(test_positions)} "
            f"tau*={best_tau:.2f} (inner train-fold pooled={best_rmse:.4f}) "
            f"({time.time() - t_fold:.1f}s)"
        )

    nested_pred = np.empty_like(rd.y_true)
    for i in range(rd.n_wells):
        well = wells[i]
        s, e = int(rd.boundaries[i]), int(rd.boundaries[i + 1])
        nested_pred[s:e] = rd.candidate_rows[chosen[well]][s:e]

    return RouterResult(nested_pred, chosen, tau_by_fold, test_proba)


def tau_sensitivity_table(
    well_labels: pd.DataFrame, wells: list[str], test_proba: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Fixed-tau sweep over the SAME nested (never-saw-this-well) test_proba.

    Distinct from the headline number: here ``tau`` is applied uniformly
    across all outer folds rather than nested-selected per fold, so this is
    a diagnostic sensitivity curve, not an alternate honest estimate (a
    single tau picked by eye off this table would be the naive/leaky
    counterpart to the nested-selected number, same distinction
    ``run_postproc_cv.py`` draws between "nested" and "reference/optimistic").
    """
    rows = []
    for tau in TAU_GRID:
        decisions = {w: _decide(test_proba[w], tau) for w in wells}
        rmse = _pooled_rmse_from_decisions(well_labels, wells, decisions)
        n_switch = sum(1 for c in decisions.values() if c != "stack")
        rows.append({"tau": tau, "pooled_rmse": rmse, "n_switched": n_switch})
    return pd.DataFrame(rows)


def helped_hurt_report(well_labels: pd.DataFrame, chosen: dict[str, str]) -> None:
    switched = {w: c for w, c in chosen.items() if c != "stack"}
    print(f"\nswitched wells: {len(switched)}/{len(chosen)}")
    if switched:
        counts = pd.Series(list(switched.values())).value_counts().to_dict()
        print(f"  breakdown by target candidate: {counts}")

    helped = hurt = tied = 0
    helped_deltas: list[float] = []
    hurt_deltas: list[float] = []
    per_target: dict[str, dict[str, int]] = {c: {"helped": 0, "hurt": 0} for c in CANDIDATES}
    for well, cand in chosen.items():
        n = float(well_labels.loc[well, "n"])
        before = np.sqrt(float(well_labels.loc[well, "sse_stack"]) / n)
        after = np.sqrt(float(well_labels.loc[well, f"sse_{cand}"]) / n)
        if after < before:
            helped += 1
            helped_deltas.append(before - after)
            if cand != "stack":
                per_target[cand]["helped"] += 1
        elif after > before:
            hurt += 1
            hurt_deltas.append(after - before)
            if cand != "stack":
                per_target[cand]["hurt"] += 1
        else:
            tied += 1

    print(f"  helped: {helped}  hurt: {hurt}  tied: {tied}")
    if helped_deltas:
        print(f"  median RMSE improvement (helped wells): {np.median(helped_deltas):.4f} ft")
    if hurt_deltas:
        print(f"  median RMSE regression (hurt wells):    {np.median(hurt_deltas):.4f} ft")
    for cand in CANDIDATES:
        if cand == "stack":
            continue
        print(f"  {cand}: helped={per_target[cand]['helped']} hurt={per_target[cand]['hurt']}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wells",
        type=int,
        default=None,
        help="Health-check subset size (stratified by tracker-manifest pf_std_mean "
        "decile). Omit for the full 773-well run.",
    )
    args = parser.parse_args()
    t_start = time.time()

    print(f"loading {OOF_PATH.name} + spatial cache ...")
    rd_full = load_router_data()
    print(
        f"  {rd_full.y_true.shape[0]:,} rows, {rd_full.n_wells} wells, "
        f"baseline pooled RMSE = "
        f"{pooled_rmse(rd_full.y_true, rd_full.candidate_rows['stack']):.4f}"
    )

    if args.n_wells is not None:
        subset = set(_stratified_health_check_wells(rd_full.used_wells, args.n_wells))
        rd = _subset_router_data(rd_full, subset)
        print(
            f"health-check subset: {rd.n_wells}/{rd_full.n_wells} wells, "
            f"{rd.y_true.shape[0]:,} rows"
        )
    else:
        rd = rd_full

    baseline_pooled = pooled_rmse(rd.y_true, rd.candidate_rows["stack"])
    carry_pooled = pooled_rmse(rd.y_true, rd.candidate_rows["carry_last"])
    pf_pooled = pooled_rmse(rd.y_true, rd.candidate_rows["pf_blend"])

    print("\nbuilding well-level candidate SSE / best-candidate labels ...")
    well_labels = build_well_labels(rd)
    label_counts = well_labels["best_candidate"].value_counts().to_dict()
    print(f"  best-candidate distribution (per-well oracle, {rd.n_wells} wells): {label_counts}")

    oracle_top10 = hard_well_oracle_guard(rd, well_labels)
    if args.n_wells is None:
        delta = oracle_top10 - HARD_WELL_ORACLE_POOLED_RMSE
        assert abs(delta) < GUARD_TOL, (
            f"hard-well oracle (top-10%) pooled RMSE {oracle_top10:.4f} drifted from ledger "
            f"{HARD_WELL_ORACLE_POOLED_RMSE} by {delta:+.4f} -- candidate definitions diverged "
            "from analyze_hard_wells.py"
        )
        print(
            f"  guard PASSED: hard-well oracle (top-10%) = {oracle_top10:.4f} "
            f"(ledger {HARD_WELL_ORACLE_POOLED_RMSE}, delta={delta:+.4f})"
        )
    else:
        print(
            f"  hard-well oracle (top-10%, health-check subset) = {oracle_top10:.4f} "
            "(guard skipped -- subset run)"
        )

    full_oracle = full_oracle_rmse(rd, well_labels)
    print(
        f"  reference only (NOT deployable): full per-well oracle "
        f"(all {rd.n_wells} wells switchable) = {full_oracle:.4f}"
    )

    print("\nbuilding router features (tracker + spatial cache + GR/Z/prefix-slope) ...")
    features = build_router_features(rd.used_wells)
    features = features.loc[[str(w) for w in rd.used_wells]]

    print(
        f"\nrunning nested well-GroupKFold({N_SPLITS}) router "
        "(outer + inner leave-one-fold-out tau selection) ..."
    )
    result = run_nested_router(rd, well_labels, features)
    nested_pooled = pooled_rmse(rd.y_true, result.nested_pred)

    print("\n" + "=" * 92)
    print(f"{'predictor':<42}{'pooled RMSE':>13}{'delta base':>13}")
    print("-" * 92)
    print(
        f"{'carry_last (floor)':<42}"
        f"{carry_pooled:>13.4f}{carry_pooled - baseline_pooled:>+13.4f}"
    )
    print(f"{'PF blend w0.7':<42}{pf_pooled:>13.4f}{pf_pooled - baseline_pooled:>+13.4f}")
    print(f"{'stack v2 all-in (baseline)':<42}{baseline_pooled:>13.4f}{0.0:>+13.4f}")
    print(
        f"{'hard-well oracle (top10%, guard)':<42}"
        f"{oracle_top10:>13.4f}{oracle_top10 - baseline_pooled:>+13.4f}"
    )
    print(
        f"{'full oracle (all wells, reference)':<42}"
        f"{full_oracle:>13.4f}{full_oracle - baseline_pooled:>+13.4f}"
    )
    print(
        f"{'router (nested, tau selected)':<42}"
        f"{nested_pooled:>13.4f}{nested_pooled - baseline_pooled:>+13.4f}"
    )
    print("=" * 92)

    print(f"\ntau by outer fold: {result.tau_by_fold}")

    wells_list = [str(w) for w in rd.used_wells]
    sens = tau_sensitivity_table(well_labels, wells_list, result.test_proba)
    print("\ntau sensitivity (fixed tau applied to the nested test_proba, diagnostic only):")
    print(sens.to_string(index=False))

    helped_hurt_report(well_labels, result.chosen)

    improvement = baseline_pooled - nested_pooled
    verdict = "採用" if improvement >= IMPROVEMENT_GATE_FT else "却下（非改善）"
    print(f"\n{'=' * 92}")
    print(
        f"判定: {verdict} -- nested router={nested_pooled:.4f}, baseline={baseline_pooled:.4f}, "
        f"改善={improvement:+.4f}ft (gate: >= {IMPROVEMENT_GATE_FT}ft)"
    )
    print(f"{'=' * 92}")

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
