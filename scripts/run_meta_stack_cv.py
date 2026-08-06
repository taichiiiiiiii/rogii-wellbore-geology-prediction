"""R3-v3 per-sample meta-stack: LightGBM over 18 candidate trajectories as
*row-level* features (design.md queue R3-v3, ledger 2026-07-09 R3-v2 rejection
follow-up).

**Why R3-v3.** R3-v1/v2 (``scripts/run_router_cv.py`` / ``run_router_v2_cv.py``,
both read-only references, never modified here) tried to pick *one* candidate
per well and lost twice: nested pooled RMSE 9.2161 and 9.3765, both worse than
the 9.2309 ``stack v2 all-in`` baseline. The common failure mode was
extrapolation transfer -- a well-level classifier/regressor trained on
known-zone backtest features does not reliably predict *eval-zone* candidate
quality. R3-v3 sidesteps discrete per-well selection entirely: instead of
choosing a winning trajectory, it feeds all 18 candidate *predictions* for
each row into a LightGBM regressor that predicts ``TVT`` directly, so the
meta-model can blend/interpolate row-by-row rather than switch. K16's part 3
("disagreement-feature meta-stack", v7->v8, claimed LOO -1.7) is the same
shape of idea (ledger 2026-07-09 K16 dissection).

**Candidate alignment: reused, not reimplemented.** ``run_router_v2_cv.py``'s
:func:`build_router_v2_data` already does the exact well-name inner-join +
row-order reindex this task needs across the three bank ``.npz`` files
(``stack_v2_oof.npz`` / ``path_bank_pfbma.npz`` / ``path_bank_beam.npz``) and
returns all 18 candidates' per-row predictions in a single master row order.
It is imported here via the same ``sys.path`` + module-import pattern
``tests/test_router_v2.py`` and ``scripts/run_beam_grid_cv.py`` already use
for cross-script reuse (``scripts/`` is not a package) -- never copy-pasted.
Verified empirically (not merely assumed) that this master row order is
*bit-identical* to ``stack_v2_oof.npz``'s own row order for the real bank (all
773 wells present in all three sources, zero drops), which is what licenses
reusing that file's saved ``row_fold`` directly (see
:func:`load_row_fold_aligned` -- it re-derives and asserts the alignment
rather than trusting the docstring).

**Feature set (28 columns, all leak-safe -- no ``TVT``/``TVT_input``, no
train-only formation columns):**

  1. the 18 candidate predictions themselves (columns named after
     ``rd.candidates``, e.g. ``stack``, ``carry_last``, ``beam:mid_cal_sm0_mm3``)
  2. 6 per-row cross-candidate agreement stats: median, std, min, max, range,
     ``|stack - median|``
  3. 2 anchor-relative deltas: ``median - anchor``, ``stack - anchor``
     (``anchor`` = ``carry_last``, which is the flat-extended known-zone
     anchor value by construction, see ``run_stack_v2_cv.py``)
  4. 2 position features: ``frac_into_zone`` (row's 0-indexed position in its
     well's eval zone / zone length) and ``log1p(zone_len)``

**CV protocol.** Well-level GroupKFold(5) reusing ``stack_v2_oof.npz``'s
saved ``row_fold`` (verified well-constant and identical row order to this
script's candidate matrix, see :func:`load_row_fold_aligned` and
:func:`derive_well_fold`). Two fixed LightGBM configs (``num_leaves``
63/127), each with a *fixed* ``n_estimators`` -- **no early stopping on the
scoring fold** (playbook 02's optimism warning; v1/v2's ES-holdout dance is
unnecessary here since there is no held-out validation split to leak into,
just a fixed tree count).

**Leak sanity check.** A third, single-feature ("stack" column only) meta run
uses the identical CV protocol; its pooled OOF RMSE must land within +/-0.3ft
of the 9.2309 baseline. A large unexplained improvement there would indicate
fold misalignment (the meta-model effectively seeing same-fold information),
not real signal, since the only feature available is a value already itself
an honest OOF prediction under the *same* fold split.

**Anchor-relative parameterization (added after a measured abort).** The first
full run aborted exactly as designed: with *absolute*-TVT features/target the
stack-only sanity meta scored pooled OOF 28.6421 vs the 9.2309 baseline
(per-fold RMSE 14.9-53.8, log: outputs/logs_archive/meta_stack_abs_abort.log).
Cause: TVT lives on a per-well absolute depth scale and tree models cannot
extrapolate outside the training folds' output range, so held-out wells whose
depth band lies outside it collapse. Fix: subtract the per-row anchor
(``carry_last``) from every candidate column and from the target, train the
meta-model entirely in drift-from-anchor space (a small, cross-well-comparable
range), and add the anchor back before scoring/saving. Per-fold RMSEs printed
inside :func:`train_meta_oof` are unchanged by the shift (the anchor cancels
in ``y_rel - pred_rel``), so they remain directly comparable to absolute-space
numbers. Two engineered columns (``median_minus_anchor``/``stack_minus_anchor``)
become duplicates of ``cand_median``/``stack`` in relative space and the
``carry_last`` column becomes all-zero -- harmless for LightGBM, kept for
column-order stability with the unit tests.

**Memory discipline** (3.8GB dev box). The 28-column float32 feature frame
for all 3,783,989 rows is ~424MB, well under the <2.5GB target; the 18-array
``candidate_rows`` dict from :func:`build_router_v2_data` is deleted as soon
as the (N,18) matrix is built. Peak RSS printed via ``resource.getrusage``.

Usage::

    uv run python scripts/run_meta_stack_cv.py
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_router_v2_cv as RV2  # noqa: E402  (reused: build_router_v2_data + alignment helpers)

OUTPUTS_DIR = _REPO_ROOT / "outputs"
STACK_OOF_PATH = OUTPUTS_DIR / "stack_v2_oof.npz"
OOF_OUT_PATH = OUTPUTS_DIR / "meta_stack_oof.npz"

BASELINE_POOLED_RMSE = 9.2309  # ledger 2026-07-07, stack v2 all-in, sanity anchor
IMPROVEMENT_GATE_FT = 0.10
ADOPTION_GATE_RMSE = BASELINE_POOLED_RMSE - IMPROVEMENT_GATE_FT  # 9.1309
SANITY_TOL_FT = 0.3  # single-feature ("stack" only) meta run must stay within this of baseline

N_SPLITS = 5
RANDOM_STATE = 42

# Two fixed LightGBM configs (task spec). No early stopping on the scoring
# fold -- n_estimators is fixed, never chosen by watching score-fold loss.
MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "leaves63": {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 600},
    "leaves127": {"num_leaves": 127, "learning_rate": 0.05, "n_estimators": 600},
}
LGB_FIXED: dict[str, object] = {
    "objective": "regression",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 100,
    "max_bin": 127,
    "force_col_wise": True,
    "n_jobs": 4,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}

STAT_FEATURE_COLUMNS: tuple[str, ...] = (
    "cand_median",
    "cand_std",
    "cand_min",
    "cand_max",
    "cand_range",
    "abs_stack_minus_median",
    "median_minus_anchor",
    "stack_minus_anchor",
    "frac_into_zone",
    "log1p_zone_len",
)


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pooled_rmse_from_sse(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


# --------------------------------------------------------------------------- #
# 1. candidate matrix + feature construction (pure, unit-tested)
# --------------------------------------------------------------------------- #


def build_candidate_matrix(
    candidate_rows: dict[str, np.ndarray], candidates: tuple[str, ...]
) -> np.ndarray:
    """Stack the K candidate per-row arrays into an ``(N, K)`` float32 matrix.

    Column order = ``candidates`` (fixed, reproducible importances/columns).
    """
    return np.stack([candidate_rows[c] for c in candidates], axis=1).astype(np.float32)


def sanitize_feature_name(name: str) -> str:
    """LightGBM rejects JSON-special characters (``:``, ``"``, ``,``, ``{``,
    ``}``, ``[``, ``]``) in feature names; bank candidate names like
    ``"beam:mid_cal_sm0_mm3"`` contain a colon, so every column name is routed
    through this before becoming a DataFrame column.
    """
    out = name
    for ch in ':",{}[]':
        out = out.replace(ch, "_")
    return out


def build_meta_features(
    cand_matrix: np.ndarray,
    candidates: tuple[str, ...],
    well_idx: np.ndarray,
    boundaries: np.ndarray,
) -> pd.DataFrame:
    """28-column per-row feature frame: 18 candidates + 6 agreement stats +
    2 anchor-relative deltas + 2 zone-position features.

    Leak-safe: every input is a candidate *prediction* (never ``TVT``/eval-zone
    ``TVT_input``), the anchor (``carry_last``, the known-zone flat-extended
    value), or a position derived purely from row index + zone length (both
    knowable at hidden-test inference time, since the eval zone's row count is
    fixed by ``TVT_input``'s NaN-count, not by any future value).

    Candidate column names are passed through :func:`sanitize_feature_name`
    (LightGBM rejects raw ``"beam:..."`` names); the 10 engineered stat/
    position columns never contain special characters and pass through as-is.
    """
    if "stack" not in candidates or "carry_last" not in candidates:
        raise ValueError("candidates must include 'stack' and 'carry_last'")
    stack_i = candidates.index("stack")
    anchor_i = candidates.index("carry_last")
    stack_col = cand_matrix[:, stack_i]
    anchor_col = cand_matrix[:, anchor_i]

    median = np.median(cand_matrix, axis=1)
    std = cand_matrix.std(axis=1)
    cmin = cand_matrix.min(axis=1)
    cmax = cand_matrix.max(axis=1)

    n = cand_matrix.shape[0]
    well_start = boundaries[well_idx].astype(np.float64)
    well_end = boundaries[well_idx + 1].astype(np.float64)
    zone_len = well_end - well_start
    local_idx = np.arange(n, dtype=np.float64) - well_start
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_into_zone = np.where(zone_len > 0, local_idx / zone_len, 0.0)

    sanitized_names = [sanitize_feature_name(c) for c in candidates]
    if len(set(sanitized_names)) != len(candidates):
        raise ValueError(
            "sanitize_feature_name produced colliding column names -- a candidate column "
            f"would silently vanish (last-write-wins): {sanitized_names!r}"
        )
    data: dict[str, np.ndarray] = {
        s: cand_matrix[:, j] for j, s in enumerate(sanitized_names)
    }
    data["cand_median"] = median.astype(np.float32)
    data["cand_std"] = std.astype(np.float32)
    data["cand_min"] = cmin.astype(np.float32)
    data["cand_max"] = cmax.astype(np.float32)
    data["cand_range"] = (cmax - cmin).astype(np.float32)
    data["abs_stack_minus_median"] = np.abs(stack_col - median).astype(np.float32)
    data["median_minus_anchor"] = (median - anchor_col).astype(np.float32)
    data["stack_minus_anchor"] = (stack_col - anchor_col).astype(np.float32)
    data["frac_into_zone"] = frac_into_zone.astype(np.float32)
    data["log1p_zone_len"] = np.log1p(zone_len).astype(np.float32)

    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# 2. fold reuse: load stack_v2_oof.npz's saved row_fold, asserting alignment
# --------------------------------------------------------------------------- #


def load_row_fold_aligned(
    y_true: np.ndarray, well_idx: np.ndarray, used_wells: np.ndarray, stack_oof_path: Path
) -> np.ndarray:
    """Load ``stack_v2_oof.npz``'s well-level GroupKFold(5) ``row_fold``, but
    only after asserting the caller's row order is identical to that file's --
    reusing a fold assignment across a *different* row order would silently
    reintroduce the well-mixing leak the whole pooled-RMSE discipline exists
    to prevent, so this never trusts alignment without checking it.
    """
    with np.load(stack_oof_path, allow_pickle=True) as npz:
        stack_y_true = npz["y_true"].astype(np.float32)
        stack_well_idx = npz["well_idx"]
        stack_used_wells = npz["used_wells"]
        stack_row_fold = npz["row_fold"]

    wells_match = list(str(w) for w in used_wells) == [str(w) for w in stack_used_wells]
    well_idx_match = np.array_equal(well_idx, stack_well_idx)
    y_match = np.allclose(np.asarray(y_true, dtype=np.float32), stack_y_true, atol=1e-2)
    if not (wells_match and well_idx_match and y_match):
        raise AssertionError(
            "candidate-matrix row order disagrees with stack_v2_oof.npz's row order -- "
            "reusing its row_fold would be an invalid (leaky) fold assignment; "
            f"wells_match={wells_match!r} well_idx_match={well_idx_match!r} y_match={y_match!r}"
        )
    return stack_row_fold.astype(np.int32)


def derive_well_fold(row_fold: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Collapse a row-level fold array to one fold id per well, asserting
    every well's rows share a single fold (a violated well-GroupKFold split
    would silently corrupt every downstream pooled-RMSE number, so this
    raises rather than averaging/majority-voting through the inconsistency).
    """
    n_wells = boundaries.shape[0] - 1
    well_fold = np.empty(n_wells, dtype=np.int32)
    for i in range(n_wells):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        seg = row_fold[s:e]
        first = seg[0]
        if not np.all(seg == first):
            raise AssertionError(f"row_fold is not well-constant for well index {i}")
        well_fold[i] = first
    return well_fold


# --------------------------------------------------------------------------- #
# 3. train + OOF (fixed n_estimators, no early stopping on the scoring fold)
# --------------------------------------------------------------------------- #


def train_meta_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    row_fold: np.ndarray,
    n_splits: int,
    model_params: dict[str, object],
) -> tuple[np.ndarray, list[dict[str, object]], pd.Series]:
    """Well-GroupKFold(``n_splits``) OOF: fit on the other folds' rows with a
    fixed tree count, predict the held-out fold. Never uses the scoring
    fold's rows for early stopping or any other model-selection decision.
    """
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importances = np.zeros(X.shape[1], dtype=np.float64)
    params = {**LGB_FIXED, **model_params}

    for f in range(n_splits):
        t0 = time.time()
        train_mask = row_fold != f
        score_mask = row_fold == f

        model = lgb.LGBMRegressor(**params)
        model.fit(X.loc[train_mask], y[train_mask])
        pred = model.predict(X.loc[score_mask])
        oof[score_mask] = pred

        fold_rmse = pooled_rmse(y[score_mask], pred)
        importances += model.feature_importances_.astype(np.float64) / n_splits
        fold_rows.append(
            {
                "fold": f,
                "n_train_rows": int(train_mask.sum()),
                "n_score_rows": int(score_mask.sum()),
                "rmse": fold_rmse,
                "seconds": time.time() - t0,
            }
        )
        print(
            f"    fold {f}: n_train={int(train_mask.sum()):,} n_score={int(score_mask.sum()):,} "
            f"rmse={fold_rmse:.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        del model
        gc.collect()

    assert not np.isnan(oof).any(), "every row must be scored by exactly one fold"
    return oof, fold_rows, pd.Series(importances, index=X.columns)


# --------------------------------------------------------------------------- #
# 4. reporting helpers
# --------------------------------------------------------------------------- #


def per_well_rmse_from_boundaries(
    y: np.ndarray, pred: np.ndarray, boundaries: np.ndarray
) -> np.ndarray:
    n_wells = boundaries.shape[0] - 1
    out = np.empty(n_wells, dtype=np.float64)
    for i in range(n_wells):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        out[i] = pooled_rmse(y[s:e], pred[s:e])
    return out


def helped_hurt_report(
    label: str, stack_well_rmse: np.ndarray, meta_well_rmse: np.ndarray, used_wells: np.ndarray
) -> None:
    valid = np.isfinite(stack_well_rmse) & np.isfinite(meta_well_rmse)
    helped = valid & (meta_well_rmse < stack_well_rmse - 1e-9)
    hurt = valid & (meta_well_rmse > stack_well_rmse + 1e-9)
    tied = valid & ~helped & ~hurt

    print(f"\nhelped/hurt vs stack v2 -- ({label}):")
    print(f"  helped: {int(helped.sum())}  hurt: {int(hurt.sum())}  tied: {int(tied.sum())}")
    if helped.any():
        print(
            "  median RMSE improvement (helped wells): "
            f"{float(np.median(stack_well_rmse[helped] - meta_well_rmse[helped])):.4f} ft"
        )
    if hurt.any():
        print(
            "  median RMSE regression (hurt wells):    "
            f"{float(np.median(meta_well_rmse[hurt] - stack_well_rmse[hurt])):.4f} ft"
        )

    order = np.argsort(-stack_well_rmse)
    worst10 = order[:10]
    print("\n  worst-10 wells by stack v2 per-well RMSE (before -> after, delta):")
    for i in worst10:
        before, after = stack_well_rmse[i], meta_well_rmse[i]
        print(
            f"    {str(used_wells[i]):<10} before={before:8.4f} after={after:8.4f} "
            f"delta={after - before:+8.4f}"
        )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-oof",
        action="store_true",
        default=True,
        help="Save the best config's OOF prediction to outputs/meta_stack_oof.npz (default on).",
    )
    args = parser.parse_args()
    t_start = time.time()

    print("loading stack_v2_oof.npz + bank npz via run_router_v2_cv.build_router_v2_data ...")
    rd = RV2.build_router_v2_data(OUTPUTS_DIR)
    print(f"  master wells: {rd.n_wells}  candidates: {len(rd.candidates)}")
    print(f"  candidates: {rd.candidates}")
    print(f"  total eval rows: {rd.y_true.shape[0]:,}")

    baseline_pred = rd.candidate_rows["stack"].astype(np.float64)
    baseline_pooled = pooled_rmse(rd.y_true, baseline_pred)
    print(f"  baseline (stack v2 all-in) pooled RMSE = {baseline_pooled:.4f}")

    print("\nverifying row order against stack_v2_oof.npz and reusing its row_fold ...")
    row_fold = load_row_fold_aligned(rd.y_true, rd.well_idx, rd.used_wells, STACK_OOF_PATH)
    well_fold = derive_well_fold(row_fold, rd.boundaries)
    print(
        f"  [fold check] row order identical to stack_v2_oof.npz "
        f"(wells/well_idx/y_true all matched); row_fold is well-constant for all "
        f"{well_fold.shape[0]} wells; fold sizes (rows): "
        f"{[int((row_fold == f).sum()) for f in range(N_SPLITS)]}"
    )

    print("\nbuilding candidate matrix + 28-column feature frame ...")
    t0 = time.time()
    cand_matrix = build_candidate_matrix(rd.candidate_rows, rd.candidates)
    del rd.candidate_rows
    gc.collect()
    anchor_col = cand_matrix[:, rd.candidates.index("carry_last")].astype(np.float64)
    cand_matrix -= anchor_col.astype(np.float32)[:, None]  # anchor-relative space (see docstring)
    features = build_meta_features(cand_matrix, rd.candidates, rd.well_idx, rd.boundaries)
    del cand_matrix
    gc.collect()
    y = rd.y_true.astype(np.float64)
    y_rel = y - anchor_col
    print(f"  feature matrix: {features.shape} ({time.time() - t0:.1f}s)")
    assert not features.isna().any().any(), "meta feature frame must be NaN-free"

    boundaries = rd.boundaries
    used_wells = rd.used_wells

    print("\n" + "=" * 96)
    print(f"{'run':<28}{'pooled RMSE':>13}{'delta base':>13}{'gate':>10}")
    print("-" * 96)

    def _row(label: str, rmse: float, gated: bool) -> str:
        if not gated:
            verdict = "--"
        else:
            verdict = "PASS" if rmse <= ADOPTION_GATE_RMSE else "FAIL"
        return f"{label:<28}{rmse:>13.4f}{rmse - baseline_pooled:>+13.4f}{verdict:>10}"

    print(_row("stack v2 all-in (baseline)", baseline_pooled, gated=False))

    # ---- sanity: single feature ("stack" column only), identical CV protocol ----
    print("\nsanity run: meta model restricted to 'stack' column only ...")
    sanity_X = features[["stack"]]
    sanity_oof, _sanity_fold_rows, _ = train_meta_oof(
        sanity_X, y_rel, row_fold, N_SPLITS, MODEL_CONFIGS["leaves63"]
    )
    sanity_pooled = pooled_rmse(y, sanity_oof + anchor_col)
    sanity_delta = sanity_pooled - baseline_pooled
    sanity_ok = abs(sanity_delta) <= SANITY_TOL_FT
    print(
        f"  sanity (stack-only) pooled RMSE = {sanity_pooled:.4f} "
        f"(delta={sanity_delta:+.4f}, tolerance=+/-{SANITY_TOL_FT}) "
        "-> "
        + ("PASS" if sanity_ok else "FAIL -- STOP: fold leak or scale/extrapolation issue")
    )
    if not sanity_ok:
        print(
            "\n[ABORT] sanity check failed -- pooled OOF RMSE drifted more than the leak-sanity "
            "tolerance from baseline. Not proceeding to the full 28-feature runs; this would be "
            "reporting numbers built on a suspect fold assignment."
        )
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"\npeak RSS (whole-process high-water mark): {peak_mb:.1f} MB")
        print(f"total runtime: {time.time() - t_start:.1f}s")
        return
    del sanity_X, sanity_oof
    gc.collect()

    # ---- full 28-feature runs, both model configs ----
    results: dict[str, np.ndarray] = {}
    fold_reports: dict[str, list[dict[str, object]]] = {}
    importances: dict[str, pd.Series] = {}

    for name, params in MODEL_CONFIGS.items():
        print(f"\nfull 28-feature run: config={name} params={params} ...")
        oof_rel, fold_rows, imp = train_meta_oof(features, y_rel, row_fold, N_SPLITS, params)
        results[name] = oof_rel + anchor_col
        fold_reports[name] = fold_rows
        importances[name] = imp

    print()
    print("=" * 96)
    print(f"{'run':<28}{'pooled RMSE':>13}{'delta base':>13}{'gate':>10}")
    print("-" * 96)
    print(_row("stack v2 all-in (baseline)", baseline_pooled, gated=False))
    print(_row("sanity (stack-only)", sanity_pooled, gated=False))
    best_name, best_pooled = None, float("inf")
    for name in MODEL_CONFIGS:
        pooled = pooled_rmse(y, results[name])
        print(_row(f"meta-stack ({name})", pooled, gated=True))
        if pooled < best_pooled:
            best_name, best_pooled = name, pooled
    print("=" * 96)

    print("\nper-fold pooled RMSE by config:")
    for name, fold_rows in fold_reports.items():
        print(f"  {name}:")
        for row in fold_rows:
            print(
                f"    fold {row['fold']}: n_score={row['n_score_rows']:,} "
                f"rmse={row['rmse']:.4f} ({row['seconds']:.1f}s)"
            )

    stack_well_rmse = per_well_rmse_from_boundaries(y, baseline_pred, boundaries)
    for name in MODEL_CONFIGS:
        meta_well_rmse = per_well_rmse_from_boundaries(y, results[name], boundaries)
        helped_hurt_report(name, stack_well_rmse, meta_well_rmse, used_wells)

        top15 = importances[name].sort_values(ascending=False).head(15)
        print(f"\nfeature importance top15 -- {name}:")
        print(top15.to_string())

    assert best_name is not None
    improvement = baseline_pooled - best_pooled
    verdict = "採用" if best_pooled <= ADOPTION_GATE_RMSE else "却下（非改善）"
    print(f"\n{'=' * 96}")
    print(
        f"判定: {verdict} -- best config={best_name} pooled={best_pooled:.4f}, "
        f"baseline={baseline_pooled:.4f}, 改善={improvement:+.4f}ft "
        f"(gate: <= {ADOPTION_GATE_RMSE:.4f}ft)"
    )
    print(f"{'=' * 96}")

    if args.save_oof and best_name is not None:
        OOF_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OOF_OUT_PATH,
            y_true=y.astype(np.float32),
            well_idx=rd.well_idx,
            used_wells=np.array([str(w) for w in used_wells]),
            best_config=np.array(best_name),
            **{f"oof_{name}": results[name].astype(np.float32) for name in MODEL_CONFIGS},
        )
        print(f"\nOOF arrays saved: {OOF_OUT_PATH}")

    del features, results
    gc.collect()
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\npeak RSS (whole-process high-water mark): {peak_mb:.1f} MB")
    print(f"total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
