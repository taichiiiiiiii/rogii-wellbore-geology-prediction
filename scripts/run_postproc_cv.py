"""P4 post-processing validation on the saved stack v2 OOF (no retraining).

Reads ``outputs/stack_v2_oof.npz`` (5 stack-v2 ablation configs' OOF predictions,
built by ``scripts/run_stack_v2_cv.py``, NOT modified here) and applies the three
public-cluster post-processing families from ``docs/design.md`` §9 --
Savitzky-Golay smoothing, robust IRLS degree-4 polyfit blend, and warm-up damping
toward the anchor -- to the **all-in** config (current best, pooled 9.2309), plus
the top-2 families composed together.

Honesty protocol (mandatory, see task spec / playbook 02): a post-processing
family's hyperparameter is never chosen using the fold it is scored on. For each
of the 5 existing well-level folds, the grid value is selected by minimizing
pooled RMSE over the OTHER four folds' rows, then applied to the held-out fold;
concatenating the 5 held-out predictions gives the **nested** pooled RMSE. A
separate **reference (optimistic)** number selects the single best grid value
using ALL rows (no held-out fold) -- this is the naive/leaky number public
notebooks often report and is kept clearly labeled apart from the honest one.

Each transform is a pure per-well function (``rogii.postproc``): it only reads
that well's own predicted curve, anchor, and MD -- never another well's data or
the true TVT -- so it is valid to apply identically at inference time on hidden
test wells (the same MD-since-anchor and normalized-MD quantities used here are
computable from ``TVT_input``/``MD`` alone, no leak).

Usage::

    uv run python scripts/run_postproc_cv.py                  # full 773-well OOF
    uv run python scripts/run_postproc_cv.py --n-wells 80     # health-check subset
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from rogii import data as D
from rogii import postproc as PP

_REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = _REPO_ROOT / "outputs" / "stack_v2_oof.npz"
MANIFEST_PATH = _REPO_ROOT / "data" / "processed" / "tracker_cache" / "manifest.csv"
ALL_IN_CONFIG_LABEL = "all-in (levers 1+2+3)"

STACK_V2_ALL_IN_POOLED_RMSE = 9.2309  # ledger 2026-07-07, sanity anchor for this script
N_SPLITS = 5
IMPROVEMENT_GATE_FT = 0.10  # success criterion: nested pooled RMSE must beat baseline by >= this

SAVGOL_WINDOWS = [9, 17, 33]
SAVGOL_ORDER = 3
IRLS_ALPHAS = [0.25, 0.5, 0.75]
IRLS_DEGREE = 4
WARMUP_TAUS = [20.0, 50.0, 85.0]

PARAM_GRIDS: dict[str, list[float]] = {
    "savgol": SAVGOL_WINDOWS,
    "irls": IRLS_ALPHAS,
    "warmup": WARMUP_TAUS,
}
PARAM_NAMES: dict[str, str] = {"savgol": "window", "irls": "alpha", "warmup": "tau"}
TRANSFORM_FUNCS: dict[str, Callable[..., np.ndarray]] = {
    "savgol": PP.savgol_transform,
    "irls": PP.irls_transform,
    "warmup": PP.warmup_transform,
}
FAMILY_LABELS: dict[str, str] = {
    "savgol": "Savitzky-Golay",
    "irls": "robust IRLS polyfit",
    "warmup": "warm-up damping",
}


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


def _load_md_since(used_wells: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Per-row MD elapsed since the anchor, in ``used_wells``/``well_idx`` block order.

    ``md_since = MD - MD_at_anchor`` where the anchor is the last row with a known
    (non-NaN) ``TVT_input`` -- the same anchor row ``D.last_known_tvt`` reads.
    """
    parts: list[np.ndarray] = []
    for i, well in enumerate(used_wells):
        h = D.load_horizontal(str(well), "train")
        mask = D.eval_mask(h)
        md_full = h["MD"].to_numpy(dtype=np.float64)
        known_idx = np.where(~mask)[0]
        if known_idx.size == 0:
            raise ValueError(f"well {well} has no known TVT_input anchor row")
        anchor_md = md_full[known_idx[-1]]
        md_since = md_full[mask] - anchor_md
        expected_len = int(boundaries[i + 1] - boundaries[i])
        if md_since.shape[0] != expected_len:
            raise ValueError(
                f"well {well}: MD eval-zone length {md_since.shape[0]} != "
                f"OOF block length {expected_len} (row-order misalignment)"
            )
        parts.append(md_since)
    return np.concatenate(parts)


def _apply_per_well(
    transform: Callable[..., np.ndarray],
    pred_all: np.ndarray,
    anchor_all: np.ndarray,
    md_since_all: np.ndarray,
    boundaries: np.ndarray,
    **params: float,
) -> np.ndarray:
    out = np.empty_like(pred_all)
    for i in range(boundaries.shape[0] - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        out[s:e] = transform(pred_all[s:e], float(anchor_all[s]), md_since_all[s:e], **params)
    return out


def _apply_family(
    name: str,
    pred_all: np.ndarray,
    anchor_all: np.ndarray,
    md_since_all: np.ndarray,
    boundaries: np.ndarray,
    param: float,
) -> np.ndarray:
    return _apply_per_well(
        TRANSFORM_FUNCS[name],
        pred_all,
        anchor_all,
        md_since_all,
        boundaries,
        **{PARAM_NAMES[name]: param},
    )


def _nested_select(
    y_true: np.ndarray,
    row_fold: np.ndarray,
    n_splits: int,
    candidates: dict[object, np.ndarray],
) -> tuple[np.ndarray, dict[int, object], float]:
    """Fold-honest hyperparameter selection: fold f's param is chosen on folds != f."""
    n = y_true.shape[0]
    nested_pred = np.empty(n, dtype=np.float64)
    chosen: dict[int, object] = {}
    for f in range(n_splits):
        other_mask = row_fold != f
        held_mask = row_fold == f
        best_param, best_rmse = None, np.inf
        for param, cand in candidates.items():
            rmse = pooled_rmse(y_true[other_mask], cand[other_mask])
            if rmse < best_rmse:
                best_rmse, best_param = rmse, param
        chosen[f] = best_param
        nested_pred[held_mask] = candidates[best_param][held_mask]
    return nested_pred, chosen, pooled_rmse(y_true, nested_pred)


def _global_best(
    y_true: np.ndarray, candidates: dict[object, np.ndarray]
) -> tuple[object, np.ndarray, float]:
    """Reference/optimistic selection: single best param chosen using ALL rows."""
    best_param, best_pred, best_rmse = None, None, np.inf
    for param, cand in candidates.items():
        rmse = pooled_rmse(y_true, cand)
        if rmse < best_rmse:
            best_rmse, best_param, best_pred = rmse, param, cand
    assert best_pred is not None
    return best_param, best_pred, best_rmse


def _stratified_health_check_wells(used_wells: np.ndarray, n: int, seed: int = 42) -> list[str]:
    """Deterministic small subset for a fast crash/NaN sanity pass before the full run."""
    if not MANIFEST_PATH.exists():
        return sorted(used_wells.tolist())[:n]
    manifest = pd.read_csv(MANIFEST_PATH)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wells",
        type=int,
        default=None,
        help="Health-check subset size (stratified by tracker-manifest pf_std_mean "
        "decile, subset of the OOF's used_wells). Omit for the full run.",
    )
    args = parser.parse_args()
    t_start = time.time()

    with np.load(OOF_PATH, allow_pickle=True) as npz:
        y_true_full = npz["y_true"].astype(np.float64)
        anchor_full = npz["anchor"].astype(np.float64)
        well_idx_full = npz["well_idx"]
        row_fold_full = npz["row_fold"]
        used_wells_full = npz["used_wells"]
        config_labels = list(npz["config_labels"])
        all_in_i = config_labels.index(ALL_IN_CONFIG_LABEL)
        pred_full = npz[f"oof_{all_in_i}"].astype(np.float64)

    n_wells_full = len(used_wells_full)
    print(f"loaded {OOF_PATH.name}: {len(y_true_full):,} rows, {n_wells_full} wells")
    print(f"base config: '{ALL_IN_CONFIG_LABEL}' (oof_{all_in_i})")

    if args.n_wells is not None:
        subset_wells = set(_stratified_health_check_wells(used_wells_full, args.n_wells))
        row_mask = np.isin(used_wells_full[well_idx_full], list(subset_wells))
        keep_well_ids = sorted({wid for wid in well_idx_full[row_mask]})
        remap = {old: new for new, old in enumerate(keep_well_ids)}
        well_idx = np.array([remap[w] for w in well_idx_full[row_mask]], dtype=np.int32)
        used_wells = used_wells_full[keep_well_ids]
        y_true = y_true_full[row_mask]
        anchor_all = anchor_full[row_mask]
        pred_all = pred_full[row_mask]
        row_fold = row_fold_full[row_mask]
        print(
            f"health-check subset: {len(used_wells)}/{n_wells_full} wells, "
            f"{row_mask.sum():,} rows"
        )
    else:
        well_idx = well_idx_full
        used_wells = used_wells_full
        y_true = y_true_full
        anchor_all = anchor_full
        pred_all = pred_full
        row_fold = row_fold_full

    n_wells = len(used_wells)
    boundaries = np.searchsorted(well_idx, np.arange(n_wells + 1))
    assert boundaries[-1] == well_idx.shape[0], "well_idx blocks are not contiguous/sorted"

    t0 = time.time()
    md_since_all = _load_md_since(used_wells, boundaries)
    print(f"MD-since-anchor loaded for {n_wells} wells ({time.time() - t0:.1f}s)")

    baseline_pooled = pooled_rmse(y_true, pred_all)
    print(f"baseline (all-in stack, no postproc) pooled RMSE: {baseline_pooled:.4f}")
    if args.n_wells is None:
        delta = baseline_pooled - STACK_V2_ALL_IN_POOLED_RMSE
        assert abs(delta) < 0.01, (
            f"baseline pooled RMSE {baseline_pooled:.4f} drifted from ledger "
            f"{STACK_V2_ALL_IN_POOLED_RMSE} by {delta:+.4f} -- npz may be stale"
        )

    # ---- individual families -------------------------------------------------
    family_results: dict[str, dict[str, object]] = {}
    for name in ("savgol", "irls", "warmup"):
        t0 = time.time()
        candidates = {
            p: _apply_family(name, pred_all, anchor_all, md_since_all, boundaries, p)
            for p in PARAM_GRIDS[name]
        }
        nested_pred, chosen, nested_rmse = _nested_select(y_true, row_fold, N_SPLITS, candidates)
        ref_param, ref_pred, ref_rmse = _global_best(y_true, candidates)
        family_results[name] = {
            "nested_pred": nested_pred,
            "nested_rmse": nested_rmse,
            "chosen": chosen,
            "ref_param": ref_param,
            "ref_pred": ref_pred,
            "ref_rmse": ref_rmse,
        }
        print(
            f"[{FAMILY_LABELS[name]}] nested={nested_rmse:.4f} "
            f"(Δ{nested_rmse - baseline_pooled:+.4f})  "
            f"reference(optimistic)={ref_rmse:.4f} ({PARAM_NAMES[name]}={ref_param})  "
            f"per-fold picks={chosen}  ({time.time() - t0:.1f}s)"
        )

    ranking = sorted(family_results.items(), key=lambda kv: kv[1]["nested_rmse"])
    top2_names = [name for name, _ in ranking[:2]]
    print(f"\nfamily ranking by nested pooled RMSE: {[n for n, _ in ranking]}")
    print(f"top-2 for composition: {top2_names}")

    # ---- top-2 composition (both orders) --------------------------------------
    combo_results: dict[tuple[str, str], dict[str, object]] = {}
    for order in (tuple(top2_names), tuple(reversed(top2_names))):
        fam1, fam2 = order
        t0 = time.time()
        stage1_by_p1 = {
            p1: _apply_family(fam1, pred_all, anchor_all, md_since_all, boundaries, p1)
            for p1 in PARAM_GRIDS[fam1]
        }
        candidates = {
            (p1, p2): _apply_family(fam2, stage1, anchor_all, md_since_all, boundaries, p2)
            for p1, stage1 in stage1_by_p1.items()
            for p2 in PARAM_GRIDS[fam2]
        }
        nested_pred, chosen, nested_rmse = _nested_select(y_true, row_fold, N_SPLITS, candidates)
        ref_param, ref_pred, ref_rmse = _global_best(y_true, candidates)
        combo_results[order] = {
            "nested_pred": nested_pred,
            "nested_rmse": nested_rmse,
            "chosen": chosen,
            "ref_param": ref_param,
            "ref_pred": ref_pred,
            "ref_rmse": ref_rmse,
        }
        print(
            f"[{fam1}->{fam2}] nested={nested_rmse:.4f} (Δ{nested_rmse - baseline_pooled:+.4f})  "
            f"reference(optimistic)={ref_rmse:.4f} "
            f"(({PARAM_NAMES[fam1]},{PARAM_NAMES[fam2]})={ref_param})  ({time.time() - t0:.1f}s)"
        )

    # ---- summary table (playbook 02 format) ------------------------------------
    baseline_well_rmse = _per_well_rmse(y_true, pred_all, well_idx, n_wells)

    def _row(label: str, pooled: float, pred: np.ndarray) -> str:
        well_arr = _per_well_rmse(y_true, pred, well_idx, n_wells)
        valid = well_arr[np.isfinite(well_arr)]
        return (
            f"{label:<38}{pooled:>13.4f}{pooled - baseline_pooled:>+11.4f}"
            f"{np.median(valid):>10.4f}{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )

    print()
    print("=" * 92)
    print(
        f"{'predictor':<38}{'pooled RMSE':>13}{'Δ base':>11}{'median':>10}{'p90':>10}{'max':>10}"
    )
    print("-" * 92)
    print(_row("stack v2 all-in (baseline)", baseline_pooled, pred_all))
    for name, res in family_results.items():
        print(_row(f"+{FAMILY_LABELS[name]} (nested)", res["nested_rmse"], res["nested_pred"]))
        print(
            _row(
                f"  {FAMILY_LABELS[name]} ref/optimistic",
                res["ref_rmse"],
                res["ref_pred"],
            )
        )
    for order, res in combo_results.items():
        fam1, fam2 = order
        label = f"+{fam1}->{fam2} combo (nested)"
        print(_row(label, res["nested_rmse"], res["nested_pred"]))
        print(_row(f"  {fam1}->{fam2} ref/optimistic", res["ref_rmse"], res["ref_pred"]))
    print("=" * 92)

    # ---- best-variant helped/hurt diagnostics -----------------------------------
    all_variants: dict[str, tuple[float, np.ndarray]] = {
        f"+{FAMILY_LABELS[n]}": (r["nested_rmse"], r["nested_pred"])
        for n, r in family_results.items()
    }
    for order, r in combo_results.items():
        all_variants[f"+{order[0]}->{order[1]} combo"] = (r["nested_rmse"], r["nested_pred"])
    best_label, (best_nested_rmse, best_pred) = min(all_variants.items(), key=lambda kv: kv[1][0])

    print(f"\nbest nested variant: {best_label}  pooled={best_nested_rmse:.4f}  "
          f"Δ vs baseline={best_nested_rmse - baseline_pooled:+.4f}")

    best_well_rmse = _per_well_rmse(y_true, best_pred, well_idx, n_wells)
    valid = np.isfinite(baseline_well_rmse) & np.isfinite(best_well_rmse)
    helped = valid & (best_well_rmse < baseline_well_rmse)
    hurt = valid & (best_well_rmse > baseline_well_rmse)
    tied = valid & (best_well_rmse == baseline_well_rmse)
    print(f"helped/hurt vs baseline -- ({best_label}):")
    print(f"  helped: {int(helped.sum())}  hurt: {int(hurt.sum())}  tied: {int(tied.sum())}")
    if helped.any():
        print(
            "  median RMSE improvement (helped wells): "
            f"{float(np.median(baseline_well_rmse[helped] - best_well_rmse[helped])):.4f} ft"
        )
    if hurt.any():
        print(
            "  median RMSE regression (hurt wells):    "
            f"{float(np.median(best_well_rmse[hurt] - baseline_well_rmse[hurt])):.4f} ft"
        )

    if MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH).set_index("well")
        delta_by_well = baseline_well_rmse - best_well_rmse  # positive = postproc helped
        well_to_pf_std = np.array(
            [
                manifest.loc[str(w), "pf_std_mean"] if str(w) in manifest.index else np.nan
                for w in used_wells
            ]
        )
        ok = valid & np.isfinite(well_to_pf_std)
        if ok.sum() > 10:
            corr = float(np.corrcoef(well_to_pf_std[ok], delta_by_well[ok])[0, 1])
            print(
                f"  diagnostic: corr(pf_std_mean, postproc RMSE improvement) = {corr:+.3f} "
                f"(n={int(ok.sum())} wells; does PF uncertainty predict where postproc helps?)"
            )

    # ---- judgement ----------------------------------------------------------------
    improvement = baseline_pooled - best_nested_rmse
    verdict = "採用" if improvement >= IMPROVEMENT_GATE_FT else "却下（非改善）"
    print(f"\n{'=' * 92}")
    print(
        f"判定: {verdict} -- nested best={best_nested_rmse:.4f}, baseline={baseline_pooled:.4f}, "
        f"改善={improvement:+.4f}ft (gate: >= {IMPROVEMENT_GATE_FT}ft)"
    )
    print(f"{'=' * 92}")

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
