"""Per-well slope-predictability probe (task: slope 回復レバーの実現可能性).

Motivation (see ``analysis/experiment_ledger.md`` 2026-07-08 hard-well decomposition
and ``scripts/analyze_hard_wells.py``): the stack v2 all-in OOF
(``outputs/stack_v2_oof.npz``, pooled RMSE 9.2309) has its per-well residual error
decomposed into offset 55% / slope 20% / shape 25% of pooled SSE. This script asks
whether the slope component specifically -- a per-well *linear drift* the current
stack does not model -- is (a) predictable from signals available before inference
and (b) worth correcting for, i.e. whether "per-well slope direct estimation" is a
credible next design lever toward the gold band (oracle-line 6.59 vs the current
oracle-const ceiling 9.07).

Method
------
1. **Target** ``b_true`` -- for every one of the 773 train wells, fit
   ``r(t) = TVT_true(t) - stack_v2_pred(t) ~ a + b*t`` by OLS over the well's
   evaluation-zone rows (``t = 0, 1, 2, ...`` = row index within the eval zone,
   which equals MD elapsed since the anchor because MD advances in exact 1ft steps
   within a well -- same convention as
   ``scripts/analyze_hard_wells.py::per_well_error_decomposition``, whose
   ``slope_b`` is the negative of this quantity since that script fits
   ``pred - true`` instead of ``true - pred``). This is the "true" per-well linear
   drift the stack under/over-shoots by, in ft per row (= ft per MD-ft).
2. **Candidate features** ``X`` -- 16 scalars per well, every one computable from
   the known-zone prefix / GR log / leave-self-out spatial prior alone (see the
   "Leak boundary" section below): a known-``TVT_input``-prefix slope, the eval
   zone's own trajectory dip (``dZ/dMD``), OLS slopes of the PF/beam/spatial
   tracker outputs over the eval zone, two-point drift-rate estimates for each
   tracker, and tracker/spatial/GR quality diagnostics reused from
   ``scripts/analyze_hard_wells.py``'s well-signal table.
3. **Ridge regression** (median-imputed + standardized features, light L2) predicts
   ``b_true`` from ``X`` under the same well-level GroupKFold(5) partition already
   saved in the OOF npz's ``row_fold`` (built by ``scripts/run_stack_v2_cv.py`` via
   ``rogii.cv.well_folds(wells, 5, seed=42)``), giving an out-of-fold ``b_hat`` per
   well and its OOF R^2. Ridge's alpha is fixed (light regularization, not tuned --
   this is an exploratory feasibility probe, not a model-selection exercise).
4. **Real-world payoff** -- correcting the stack prediction by
   ``pred(t) + shrink * b_hat * t`` for ``shrink in {0.3, 0.5, 0.7, 1.0}``, with the
   shrink value itself chosen **nested** (fold f's shrink is picked by minimizing
   pooled RMSE on the *other* four folds, mirroring
   ``scripts/run_postproc_cv.py::_nested_select`` exactly) to keep the reported
   pooled RMSE honest -- ``b_hat`` is already an OOF quantity from step 3, and the
   outer shrink choice never touches the fold it is applied to. Compared against:
   no correction (baseline) and an **oracle** upper bound that corrects with the
   true ``b_true`` at shrink=1.0 (exactly cancels the slope component of every
   well's residual -- not a deployable predictor, see the "oracle" label on that
   number, same convention as ``analyze_hard_wells.py``'s oracle probes).

Leak boundary
-------------
``b_true`` is computed **only** from ``outputs/stack_v2_oof.npz``'s cached
``y_true``/``pred`` arrays (training-time-only, exactly like
``analyze_hard_wells.py``'s SSE decomposition) and is used **only** as the Ridge
regression target and for the oracle upper-bound probe -- never as a feature. Every
feature in ``X`` is computable from data available at hidden-test inference time:
  - the known-``TVT_input`` prefix and ``MD`` (never the eval-zone ``TVT_input``,
    which does not exist in test),
  - ``Z`` over the eval zone (a schema column in both train and test -- wellbore
    trajectory coordinate, not a stratigraphic/TVT label),
  - the PF/beam tracker cache (``data/processed/tracker_cache``, built only from
    the known-zone prefix + typewell GR -- see ``registration/particle.py`` /
    ``registration/beam.py``),
  - the spatial-prior cache (``data/processed/spatial_cache``, leave-self-out KNN
    over *other* wells' formation surfaces -- see ``scripts/build_spatial_cache.py``),
  - the well's overall GR NaN fraction (``GR`` is a schema column in both splits).
No feature reads ``TVT`` or the eval-zone ``TVT_input``.

Usage::

    uv run python scripts/analyze_slope_predictability.py                # full 773-well run
    uv run python scripts/analyze_slope_predictability.py --n-wells 80   # health-check subset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rogii import data as D

_REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = _REPO_ROOT / "outputs" / "stack_v2_oof.npz"
TRACKER_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "tracker_cache"
SPATIAL_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "spatial_cache"

ALL_IN_CONFIG_LABEL = "all-in (levers 1+2+3)"
STACK_V2_ALL_IN_POOLED_RMSE = 9.2309  # ledger 2026-07-07, sanity anchor for this script
N_SPLITS = 5
PREFIX_SLOPE_WINDOW = 200  # last N known-TVT_input rows used for the prefix-slope feature
RIDGE_ALPHA = 1.0  # "light regularization" per task spec, fixed (not tuned -- exploratory)
SEED = 42
SHRINK_GRID: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
IMPROVEMENT_GATE_FT = 0.10  # same adoption gate as run_postproc_cv.py

FEATURE_COLUMNS: tuple[str, ...] = (
    "prefix_slope200",
    "traj_dZdMD_mean",
    "spatial_slope",
    "pf_slope",
    "beam_slope",
    "pf_d_final_rate",
    "beam_d_final_rate",
    "spatial_d_final_rate",
    "pf_std_mean",
    "pf_std_max",
    "beam_margin_mean",
    "spatial_prefix_rmse",
    "spatial_nn_dist_median",
    "gr_nan_frac",
    "eval_len",
    "n_known_prefix",
)


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pooled RMSE: sqrt(mean squared error) over every row, never per-well average."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _ols_slope(y: np.ndarray, t: np.ndarray) -> float:
    """OLS slope of ``y`` on ``t`` (returns 0.0 for degenerate/too-short input)."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(t)
    if finite.sum() < 2:
        return 0.0
    yy, tt = y[finite], t[finite]
    t_bar = float(np.mean(tt))
    denom = float(np.sum((tt - t_bar) ** 2))
    if denom <= 0.0:
        return 0.0
    y_bar = float(np.mean(yy))
    return float(np.sum((tt - t_bar) * (yy - y_bar)) / denom)


# --------------------------------------------------------------------------- #
# 1. load stack v2 OOF + target b_true
# --------------------------------------------------------------------------- #


def load_stack_oof() -> dict[str, np.ndarray]:
    """Load the all-in stack v2 OOF arrays (read-only; not modified by this script)."""
    with np.load(OOF_PATH, allow_pickle=True) as npz:
        y_true = npz["y_true"].astype(np.float64)
        pred_anchor = npz["anchor"].astype(np.float64)
        well_idx = npz["well_idx"]
        row_fold = npz["row_fold"]
        used_wells = npz["used_wells"]
        config_labels = list(npz["config_labels"])
        all_in_i = config_labels.index(ALL_IN_CONFIG_LABEL)
        pred = npz[f"oof_{all_in_i}"].astype(np.float64)

    n_wells = len(used_wells)
    boundaries = np.searchsorted(well_idx, np.arange(n_wells + 1))
    assert boundaries[-1] == well_idx.shape[0], "well_idx blocks are not contiguous/sorted"

    baseline = pooled_rmse(y_true, pred)
    delta = baseline - STACK_V2_ALL_IN_POOLED_RMSE
    assert abs(delta) < 0.01, (
        f"stack v2 all-in pooled RMSE {baseline:.4f} drifted from ledger "
        f"{STACK_V2_ALL_IN_POOLED_RMSE} by {delta:+.4f} -- npz may be stale"
    )

    return {
        "y_true": y_true,
        "anchor": pred_anchor,
        "pred": pred,
        "well_idx": well_idx,
        "row_fold": row_fold,
        "used_wells": used_wells,
        "boundaries": boundaries,
        "n_wells": n_wells,
    }


def compute_b_true(oof: dict[str, np.ndarray]) -> np.ndarray:
    """Per-well OLS slope of ``r(t) = y_true - pred`` vs ``t`` (row index in eval zone).

    ``b_true`` is training-time-only (needs ``y_true``) and is used solely as the
    Ridge regression target and the oracle upper-bound correction -- never as a
    model feature.
    """
    y_true, pred, boundaries = oof["y_true"], oof["pred"], oof["boundaries"]
    n_wells = oof["n_wells"]
    b_true = np.zeros(n_wells, dtype=np.float64)
    for i in range(n_wells):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        t = np.arange(e - s, dtype=np.float64)
        r = y_true[s:e] - pred[s:e]
        b_true[i] = _ols_slope(r, t)
    return b_true


# --------------------------------------------------------------------------- #
# 2. candidate features (all pre-inference / leave-self-out safe -- see module doc)
# --------------------------------------------------------------------------- #


def build_feature_matrix(oof: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the 16-column feature matrix, one row per well in ``oof['used_wells']`` order.

    Every well in the OOF must yield a full feature row (this experiment is
    773-well, no sampling); a length/cache mismatch is a hard failure rather than
    a silent skip, since prior scripts (``run_stack_v2_cv.py``) already proved all
    773 train wells have matching tracker+spatial caches. ``counts`` is still
    tracked and reported (expected all-zero) per playbook discipline.
    """
    used_wells, boundaries = oof["used_wells"], oof["boundaries"]
    n_wells = oof["n_wells"]
    counts = {"n_tracker_missing": 0, "n_spatial_missing": 0, "n_len_mismatch": 0}

    rows: list[dict[str, float]] = []
    t0 = time.time()
    for i in range(n_wells):
        well = str(used_wells[i])
        n = int(boundaries[i + 1] - boundaries[i])

        h = D.load_horizontal(well, "train")
        mask = D.eval_mask(h)
        if int(mask.sum()) != n:
            counts["n_len_mismatch"] += 1
            raise ValueError(f"{well}: eval-zone length {int(mask.sum())} != OOF block {n}")

        md = h["MD"].to_numpy(dtype=np.float64)
        tvt_input = h["TVT_input"].to_numpy(dtype=np.float64)
        known_idx = np.where(~mask)[0]
        window_idx = known_idx[-min(PREFIX_SLOPE_WINDOW, known_idx.size) :]
        prefix_slope = _ols_slope(tvt_input[window_idx], md[window_idx])

        z_eval = h["Z"].to_numpy(dtype=np.float64)[mask]
        t = np.arange(n, dtype=np.float64)
        traj_slope = _ols_slope(z_eval, t)

        gr = h["GR"].to_numpy(dtype=np.float64) if "GR" in h.columns else np.array([])
        gr_nan_frac = float(np.isnan(gr).mean()) if gr.size else 1.0

        trk_path = TRACKER_CACHE_DIR / f"{well}.npz"
        if not trk_path.exists():
            counts["n_tracker_missing"] += 1
            raise FileNotFoundError(f"{well}: missing tracker cache {trk_path}")
        with np.load(trk_path) as npz:
            pf_tvt = npz["pf_tvt"].astype(np.float64)
            pf_std = npz["pf_std"].astype(np.float64)
            beam_tvt = npz["beam_tvt"].astype(np.float64)
            beam_margin = npz["beam_margin"].astype(np.float64)
            cached_anchor = float(npz["anchor"])
        if pf_tvt.size != n:
            counts["n_len_mismatch"] += 1
            raise ValueError(f"{well}: tracker cache length {pf_tvt.size} != OOF block {n}")

        sp_path = SPATIAL_CACHE_DIR / f"{well}.npz"
        if not sp_path.exists():
            counts["n_spatial_missing"] += 1
            raise FileNotFoundError(f"{well}: missing spatial cache {sp_path}")
        with np.load(sp_path) as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
            prefix_rmse = float(npz["prefix_rmse"])
            nn_dist_median = float(npz["nn_dist_median"])
        if spatial_tvt.size != n:
            counts["n_len_mismatch"] += 1
            raise ValueError(f"{well}: spatial cache length {spatial_tvt.size} != OOF block {n}")

        pf_finite = pf_std[np.isfinite(pf_std)]
        beam_finite = beam_margin[np.isfinite(beam_margin)]

        rows.append(
            {
                "well": well,
                "prefix_slope200": prefix_slope,
                "traj_dZdMD_mean": traj_slope,
                "spatial_slope": _ols_slope(spatial_tvt, t),
                "pf_slope": _ols_slope(pf_tvt, t),
                "beam_slope": _ols_slope(beam_tvt, t),
                "pf_d_final_rate": (float(pf_tvt[-1]) - cached_anchor) / n,
                "beam_d_final_rate": (float(beam_tvt[-1]) - cached_anchor) / n,
                "spatial_d_final_rate": (float(spatial_tvt[-1]) - cached_anchor) / n,
                "pf_std_mean": float(np.mean(pf_finite)) if pf_finite.size else np.nan,
                "pf_std_max": float(np.max(pf_finite)) if pf_finite.size else np.nan,
                "beam_margin_mean": float(np.mean(beam_finite)) if beam_finite.size else np.nan,
                "spatial_prefix_rmse": prefix_rmse,
                "spatial_nn_dist_median": nn_dist_median,
                "gr_nan_frac": gr_nan_frac,
                "eval_len": float(n),
                "n_known_prefix": float(known_idx.size),
            }
        )
        if (i + 1) % 200 == 0 or (i + 1) == n_wells:
            print(f"  features: {i + 1}/{n_wells} wells ({time.time() - t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows).set_index("well")
    return df.loc[[str(w) for w in used_wells], list(FEATURE_COLUMNS)], counts


# --------------------------------------------------------------------------- #
# 3. Ridge OOF b_hat under the stack v2 well-level GroupKFold(5)
# --------------------------------------------------------------------------- #


def well_fold_from_oof(oof: dict[str, np.ndarray]) -> np.ndarray:
    """One fold id per well, read from the OOF's row-level ``row_fold`` (same folds
    ``run_stack_v2_cv.py`` trained on -- a well's rows are entirely inside one fold
    by construction of ``rogii.cv.well_folds``, so the first row of each block is
    representative)."""
    boundaries, row_fold = oof["boundaries"], oof["row_fold"]
    n_wells = oof["n_wells"]
    return np.array([int(row_fold[int(boundaries[i])]) for i in range(n_wells)], dtype=np.int32)


def ridge_oof_b_hat(
    X: pd.DataFrame, b_true: np.ndarray, well_fold: np.ndarray, n_splits: int, alpha: float
) -> tuple[np.ndarray, float]:
    """Well-level GroupKFold(5) Ridge OOF prediction of ``b_true`` from ``X``.

    Each fold's pipeline (median-impute -> standardize -> Ridge) is fit only on
    wells outside that fold, so ``b_hat`` is a genuine out-of-fold estimate for
    every well.
    """
    n = b_true.shape[0]
    b_hat = np.full(n, np.nan, dtype=np.float64)
    for f in range(n_splits):
        train_mask = well_fold != f
        test_mask = well_fold == f
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=SEED)),
            ]
        )
        pipe.fit(X.loc[train_mask], b_true[train_mask])
        b_hat[test_mask] = pipe.predict(X.loc[test_mask])
    assert not np.isnan(b_hat).any(), "every well must be scored by exactly one fold"
    return b_hat, float(r2_score(b_true, b_hat))


def print_feature_correlations(X: pd.DataFrame, b_true: np.ndarray) -> None:
    """Per-feature Pearson/Spearman correlation with ``b_true``, sorted by |Pearson|."""
    b_series = pd.Series(b_true, index=X.index)
    rows = []
    for col in X.columns:
        x = X[col]
        valid = x.notna()
        n_valid = int(valid.sum())
        if n_valid > 2:
            pearson = float(x[valid].corr(b_series[valid], method="pearson"))
            spearman = float(x[valid].corr(b_series[valid], method="spearman"))
        else:
            pearson, spearman = np.nan, np.nan
        rows.append(
            {"feature": col, "pearson_r": pearson, "spearman_rho": spearman, "n": n_valid}
        )
    raw = pd.DataFrame(rows)
    table = raw.reindex(raw["pearson_r"].abs().sort_values(ascending=False).index)
    print("\n" + "=" * 78)
    print("② 特徴 x b_true (真slope) の単相関（|pearson| 降順）")
    print("-" * 78)
    print(f"{'feature':<26}{'pearson_r':>12}{'spearman_rho':>14}{'n':>8}")
    print("-" * 78)
    for _, r in table.iterrows():
        print(f"{r['feature']:<26}{r['pearson_r']:>12.4f}{r['spearman_rho']:>14.4f}{int(r['n']):>8}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 4. real-world payoff: slope correction, nested shrink selection
# --------------------------------------------------------------------------- #


def apply_slope_correction(
    pred_all: np.ndarray, b_per_well: np.ndarray, boundaries: np.ndarray, shrink: float
) -> np.ndarray:
    """``pred(t) + shrink * b_per_well[i] * t`` applied per well (t = row index in eval zone)."""
    out = pred_all.copy()
    for i in range(boundaries.shape[0] - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        t = np.arange(e - s, dtype=np.float64)
        out[s:e] = pred_all[s:e] + shrink * b_per_well[i] * t
    return out


def _nested_select_shrink(
    y_true: np.ndarray, row_fold: np.ndarray, n_splits: int, candidates: dict[float, np.ndarray]
) -> tuple[np.ndarray, dict[int, float], float]:
    """Fold-honest shrink selection (identical pattern to
    ``scripts/run_postproc_cv.py::_nested_select``): fold f's shrink is chosen by
    minimizing pooled RMSE on the OTHER four folds' rows, then applied only to
    fold f's held-out rows -- so the reported pooled RMSE never uses a fold's own
    rows to pick the hyperparameter applied to it."""
    n = y_true.shape[0]
    nested_pred = np.empty(n, dtype=np.float64)
    chosen: dict[int, float] = {}
    for f in range(n_splits):
        other_mask = row_fold != f
        held_mask = row_fold == f
        best_shrink, best_rmse = None, np.inf
        for shrink, cand in candidates.items():
            rmse = pooled_rmse(y_true[other_mask], cand[other_mask])
            if rmse < best_rmse:
                best_rmse, best_shrink = rmse, shrink
        assert best_shrink is not None
        chosen[f] = best_shrink
        nested_pred[held_mask] = candidates[best_shrink][held_mask]
    return nested_pred, chosen, pooled_rmse(y_true, nested_pred)


def _per_well_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, well_idx: np.ndarray, n_wells: int
) -> np.ndarray:
    out = np.full(n_wells, np.nan, dtype=np.float64)
    for w_i in range(n_wells):
        rows = well_idx == w_i
        if rows.any():
            out[w_i] = pooled_rmse(y_true[rows], y_pred[rows])
    return out


def _row(
    label: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    well_idx: np.ndarray,
    n_wells: int,
    baseline: float,
) -> str:
    pooled = pooled_rmse(y_true, pred)
    well_arr = _per_well_rmse(y_true, pred, well_idx, n_wells)
    valid = well_arr[np.isfinite(well_arr)]
    return (
        f"{label:<42}{pooled:>13.4f}{pooled - baseline:>+11.4f}"
        f"{np.median(valid):>10.4f}{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wells",
        type=int,
        default=None,
        help="Health-check subset size (first N wells by used_wells order). Omit for the full run.",
    )
    args = parser.parse_args()
    t_start = time.time()

    print(f"loading {OOF_PATH.name} ...")
    oof = load_stack_oof()
    baseline_pooled = pooled_rmse(oof["y_true"], oof["pred"])
    print(
        f"  {int(oof['y_true'].shape[0]):,} rows, {oof['n_wells']} wells, "
        f"baseline pooled RMSE = {baseline_pooled:.4f}"
    )

    if args.n_wells is not None:
        keep = min(args.n_wells, oof["n_wells"])
        boundaries = oof["boundaries"][: keep + 1]
        n_rows = int(boundaries[-1])
        oof = {
            "y_true": oof["y_true"][:n_rows],
            "anchor": oof["anchor"][:n_rows],
            "pred": oof["pred"][:n_rows],
            "well_idx": oof["well_idx"][:n_rows],
            "row_fold": oof["row_fold"][:n_rows],
            "used_wells": oof["used_wells"][:keep],
            "boundaries": boundaries,
            "n_wells": keep,
        }
        baseline_pooled = pooled_rmse(oof["y_true"], oof["pred"])
        print(f"health-check subset: {keep} wells, {n_rows:,} rows")

    b_true = compute_b_true(oof)
    print(f"\nb_true (真slope) summary: mean={b_true.mean():+.5f} std={b_true.std():.5f} "
          f"median={np.median(b_true):+.5f} ft/row")

    print("\nbuilding feature matrix (16 cols, pre-inference-safe -- see module docstring) ...")
    X, counts = build_feature_matrix(oof)
    print(f"  feature matrix: {X.shape}, counts={counts}")

    well_fold = well_fold_from_oof(oof)
    b_hat, oof_r2 = ridge_oof_b_hat(X, b_true, well_fold, N_SPLITS, RIDGE_ALPHA)
    print("\n" + "=" * 78)
    print(f"① Ridge OOF R^2 (b_true 予測, well-level GroupKFold({N_SPLITS}), alpha={RIDGE_ALPHA})")
    print("-" * 78)
    print(f"  OOF R^2 = {oof_r2:.4f}")
    print(f"  b_hat  : mean={b_hat.mean():+.5f} std={b_hat.std():.5f}")
    print("=" * 78)

    print_feature_correlations(X, b_true)

    # ---- payoff: baseline / oracle-slope ceiling / predicted-slope (nested) ----
    y_true, pred, boundaries, well_idx, row_fold = (
        oof["y_true"],
        oof["pred"],
        oof["boundaries"],
        oof["well_idx"],
        oof["row_fold"],
    )
    n_wells = oof["n_wells"]

    oracle_pred = apply_slope_correction(pred, b_true, boundaries, shrink=1.0)
    oracle_rmse = pooled_rmse(y_true, oracle_pred)

    predicted_candidates = {
        s: apply_slope_correction(pred, b_hat, boundaries, shrink=s) for s in SHRINK_GRID
    }
    nested_pred, chosen, nested_rmse = _nested_select_shrink(
        y_true, row_fold, N_SPLITS, predicted_candidates
    )
    ref_shrink, ref_rmse = min(
        ((s, pooled_rmse(y_true, cand)) for s, cand in predicted_candidates.items()),
        key=lambda kv: kv[1],
    )

    def row(label: str, pred_arr: np.ndarray) -> str:
        return _row(label, pred_arr, y_true, well_idx, n_wells, baseline_pooled)

    print("\n" + "=" * 96)
    print(
        f"{'predictor':<42}{'pooled RMSE':>13}{'Δ base':>11}{'median':>10}{'p90':>10}{'max':>10}"
    )
    print("-" * 96)
    print(row("baseline (stack v2 all-in, no corr.)", pred))
    print(row("oracle-slope 上限 (b_true, shrink=1.0)", oracle_pred))
    for s, cand in predicted_candidates.items():
        print(row(f"  predicted-slope (non-nested, shrink={s})", cand))
    ref_pred = predicted_candidates[ref_shrink]
    print(row(f"  predicted-slope ref/optimistic (s={ref_shrink})", ref_pred))
    print(row("predicted-slope (nested shrink)", nested_pred))
    print("=" * 96)
    print(f"  nested per-fold shrink picks: {chosen}")
    print(f"  reference/optimistic pooled RMSE = {ref_rmse:.4f} (non-honest, kept for comparison)")

    # ---- helped/hurt vs baseline for the nested predicted-slope corrector ------
    baseline_well_rmse = _per_well_rmse(y_true, pred, well_idx, n_wells)
    nested_well_rmse = _per_well_rmse(y_true, nested_pred, well_idx, n_wells)
    valid = np.isfinite(baseline_well_rmse) & np.isfinite(nested_well_rmse)
    helped = valid & (nested_well_rmse < baseline_well_rmse)
    hurt = valid & (nested_well_rmse > baseline_well_rmse)
    tied = valid & (nested_well_rmse == baseline_well_rmse)
    print("\nhelped/hurt vs baseline -- (predicted-slope nested shrink):")
    print(f"  helped: {int(helped.sum())}  hurt: {int(hurt.sum())}  tied: {int(tied.sum())}")
    if helped.any():
        print(
            "  median RMSE improvement (helped wells): "
            f"{float(np.median(baseline_well_rmse[helped] - nested_well_rmse[helped])):.4f} ft"
        )
    if hurt.any():
        print(
            "  median RMSE regression (hurt wells):    "
            f"{float(np.median(nested_well_rmse[hurt] - baseline_well_rmse[hurt])):.4f} ft"
        )

    # ---- judgement --------------------------------------------------------------
    improvement = baseline_pooled - nested_rmse
    oracle_headroom = baseline_pooled - oracle_rmse
    verdict = "採用" if improvement >= IMPROVEMENT_GATE_FT else "却下（非改善）"
    print(f"\n{'=' * 96}")
    print(
        f"判定: {verdict} -- nested predicted-slope={nested_rmse:.4f}, "
        f"baseline={baseline_pooled:.4f}, "
        f"改善={improvement:+.4f}ft (gate: >= {IMPROVEMENT_GATE_FT}ft)"
    )
    print(
        f"oracle 上限が示す slope レバーの理論天井 = {oracle_headroom:.4f}ft "
        f"(oracle {oracle_rmse:.4f} vs baseline {baseline_pooled:.4f})"
    )
    if oracle_headroom > 1e-9:
        capture = 100.0 * improvement / oracle_headroom
        print(f"予測モデルが回収した天井の割合 = {capture:.1f}%")
    print(f"{'=' * 96}")

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
