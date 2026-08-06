"""Hard-well error decomposition and pre-inference routing-feasibility diagnostic.

Analysis-only script (task: hard-well 誤差分解診断). Reads the saved stack v2
all-in OOF (``outputs/stack_v2_oof.npz``, NOT modified here -- built by
``scripts/run_stack_v2_cv.py --save-oof``) and answers, purely by measurement:

  1. **3-way SSE decomposition** -- for every well, the per-row error curve
     ``e(t) = pred - true`` (``t`` = MD elapsed since the eval-zone anchor,
     which is exactly ``0, 1, 2, ...`` because MD is documented to advance in
     strict 1ft steps within a well -- verified directly against the raw CSVs
     for a handful of wells before relying on it here) is split via the
     orthogonal-projection identity for an OLS fit with an intercept:

         SSE_total  = sum(e^2)
         SSE_offset = n * mean(e)^2                       (constant bias)
         SSE_slope  = b^2 * sum((t - mean(t))^2)           (linear drift)
         SSE_shape  = SSE_total - SSE_offset - SSE_slope   (everything else)

     where ``b`` is the least-squares slope of ``e`` on ``t``. This is exact
     (not approximate) because ``e_hat = a + b*t`` is the orthogonal
     projection of ``e`` onto span([1, t]): the residual ``e - e_hat`` is
     orthogonal to that span by the OLS normal equations, and the offset/slope
     terms of ``e_hat`` are mutually orthogonal because ``t - mean(t)`` sums
     to zero. Pooling ``SSE_offset``/``SSE_slope``/``SSE_shape`` across all
     wells and dividing by the pooled ``SSE_total`` gives the fraction of the
     competition metric each error mode is responsible for.

  2. **Hard-well identification** -- wells ranked by their own ``SSE_total``;
     the top decile's share of pooled SSE, plus a characteristics table
     (top-10% vs bottom-90% medians) against tracker/spatial-cache signals,
     carry-drift magnitude, eval-zone length, and GR NaN rate.

  3. **Pre-inference routing feasibility** -- for signals computable purely
     from the known-zone prefix (i.e. available at hidden-test inference
     time -- tracker/spatial cache scalars, eval length, GR NaN rate; NOT the
     true-label-derived carry-drift, which step 2 uses only for
     characterization), report top-decile recall/precision against the
     ground-truth hard-well set, plus a rank correlation.

  4. **Oracle substitution ceilings** -- for the hard-well row set only,
     what pooled RMSE would result from swapping those wells' predictions to
     carry_last / PF-blend / spatial (uniformly), and from a per-well oracle
     pick among {stack, carry_last, PF blend, spatial} using the true label.
     This uses ground truth to choose and is explicitly an upper-bound probe,
     not a deployable predictor -- see the "oracle" label on every number it
     produces.

No retraining, no OOF re-computation, no leakage into the stack v2 pipeline:
this script only reads existing artifacts (``outputs/stack_v2_oof.npz``,
``data/processed/tracker_cache/*.npz``+``manifest.csv``,
``data/processed/spatial_cache/*.npz``, and the raw per-well CSVs' ``GR``
column via ``rogii.data.load_horizontal``, used only to compute a well-level
GR-NaN-rate diagnostic).

Usage::

    uv run python scripts/analyze_hard_wells.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from rogii import data as D

_REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = _REPO_ROOT / "outputs" / "stack_v2_oof.npz"
TRACKER_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "tracker_cache"
TRACKER_MANIFEST_PATH = TRACKER_CACHE_DIR / "manifest.csv"
SPATIAL_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "spatial_cache"

ALL_IN_CONFIG_LABEL = "all-in (levers 1+2+3)"
STACK_V2_ALL_IN_POOLED_RMSE = 9.2309  # ledger 2026-07-07, sanity anchor for this script
PF_BLEND_POOLED_RMSE = 11.0621  # ledger, sanity anchor for the pf_blend column in the npz
TOP_FRAC = 0.10


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pooled_sse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


# --------------------------------------------------------------------------- #
# 1. load stack v2 OOF + per-well error decomposition
# --------------------------------------------------------------------------- #


def load_stack_oof() -> dict[str, np.ndarray]:
    with np.load(OOF_PATH, allow_pickle=True) as npz:
        y_true = npz["y_true"].astype(np.float64)
        anchor = npz["anchor"].astype(np.float64)
        pf_blend = npz["pf_blend"].astype(np.float64)
        well_idx = npz["well_idx"]
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
    pf_pooled = pooled_rmse(y_true, pf_blend)
    delta_pf = pf_pooled - PF_BLEND_POOLED_RMSE
    assert abs(delta_pf) < 0.01, (
        f"pf_blend column pooled RMSE {pf_pooled:.4f} drifted from ledger "
        f"{PF_BLEND_POOLED_RMSE} by {delta_pf:+.4f} -- npz may be stale"
    )

    return {
        "y_true": y_true,
        "anchor": anchor,
        "pf_blend": pf_blend,
        "pred": pred,
        "well_idx": well_idx,
        "used_wells": used_wells,
        "boundaries": boundaries,
        "n_wells": n_wells,
    }


def per_well_error_decomposition(oof: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-well SSE decomposition ``e(t) = pred - true`` ~ ``a + b*t``.

    ``t`` = row index within the well's eval zone (``0, 1, 2, ...``), which
    equals MD elapsed since the anchor because MD advances in exact 1ft steps
    within a well (verified against raw CSVs for a sample of wells; see
    module docstring). Returns one row per well, indexed by well id.
    """
    y_true, pred = oof["y_true"], oof["pred"]
    boundaries, used_wells = oof["boundaries"], oof["used_wells"]
    n_wells = oof["n_wells"]

    rows: list[dict[str, object]] = []
    for i in range(n_wells):
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        n = e_ - s
        err = pred[s:e_] - y_true[s:e_]
        sse_total = float(np.sum(err**2))
        mean_e = float(np.mean(err))
        sse_offset = n * mean_e**2

        t = np.arange(n, dtype=np.float64)
        if n >= 2 and np.var(t) > 0:
            t_bar = float(np.mean(t))
            centered_t = t - t_bar
            centered_e = err - mean_e
            denom = float(np.sum(centered_t**2))
            b = float(np.sum(centered_t * centered_e) / denom)
            sse_slope = b**2 * denom
        else:
            b = 0.0
            sse_slope = 0.0

        sse_shape = max(sse_total - sse_offset - sse_slope, 0.0)

        rows.append(
            {
                "well": str(used_wells[i]),
                "n": n,
                "sse_total": sse_total,
                "sse_offset": sse_offset,
                "sse_slope": sse_slope,
                "sse_shape": sse_shape,
                "rmse": float(np.sqrt(sse_total / n)) if n > 0 else np.nan,
                "mean_e": mean_e,
                "slope_b": b,
                "anchor": float(oof["anchor"][s]),
                "true_tail": float(y_true[e_ - 1]),
            }
        )
    return pd.DataFrame(rows).set_index("well")


# --------------------------------------------------------------------------- #
# 2/3. per-well signals available at (or before) inference time
# --------------------------------------------------------------------------- #


def load_well_signals(used_wells: np.ndarray) -> pd.DataFrame:
    """Well-level diagnostic signals: tracker cache, spatial cache, GR NaN rate.

    All of these are computable from the known-zone prefix alone (no ground
    truth), so they are legitimate candidates for a pre-inference routing
    rule -- as opposed to the carry-drift magnitude computed separately in
    ``main()`` from the OOF's true labels, which is diagnostic-only.
    """
    manifest = pd.read_csv(TRACKER_MANIFEST_PATH).set_index("well")

    rows: list[dict[str, object]] = []
    t0 = time.time()
    for well in used_wells:
        well = str(well)
        with np.load(TRACKER_CACHE_DIR / f"{well}.npz") as npz:
            pf_tvt = npz["pf_tvt"].astype(np.float64)
            pf_std = npz["pf_std"].astype(np.float64)
            beam_tvt = npz["beam_tvt"].astype(np.float64)
            anchor = float(npz["anchor"])
        with np.load(SPATIAL_CACHE_DIR / f"{well}.npz") as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
            prefix_rmse = float(npz["prefix_rmse"])
            nn_dist_median = float(npz["nn_dist_median"])

        h = D.load_horizontal(well, "train")
        gr_nan_frac = float(h["GR"].isna().mean()) if "GR" in h.columns else 1.0

        pf_d = pf_tvt - anchor
        beam_d = beam_tvt - anchor
        spatial_d = spatial_tvt - anchor
        pf_finite = pf_std[np.isfinite(pf_std)]

        rows.append(
            {
                "well": well,
                "pf_std_mean": float(manifest.loc[well, "pf_std_mean"]),
                "beam_margin_mean": float(manifest.loc[well, "beam_margin_mean"]),
                "pf_std_max": float(np.max(pf_finite)) if pf_finite.size else np.nan,
                "pf_d_final": float(pf_d[-1]) if pf_d.size else np.nan,
                "beam_d_final": float(beam_d[-1]) if beam_d.size else np.nan,
                "spatial_d_final": float(spatial_d[-1]) if spatial_d.size else np.nan,
                "spatial_prefix_rmse": prefix_rmse,
                "spatial_nn_dist_median": nn_dist_median,
                "gr_nan_frac": gr_nan_frac,
                "eval_len": int(pf_tvt.size),
            }
        )
    print(f"  well signals loaded for {len(used_wells)} wells ({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows).set_index("well")


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #


def _fmt_pct(x: float) -> str:
    return f"{x:6.2f}%"


def print_decomposition_summary(per_well: pd.DataFrame) -> None:
    total = per_well["sse_total"].sum()
    offset = per_well["sse_offset"].sum()
    slope = per_well["sse_slope"].sum()
    shape = per_well["sse_shape"].sum()

    print("\n" + "=" * 78)
    print("① 誤差の3成分分解（pooled, all-in stack, N={:,} rows, {} wells）".format(
        int(per_well["n"].sum()), len(per_well)
    ))
    print("-" * 78)
    check_rmse = np.sqrt(total / per_well["n"].sum())
    print(f"  SSE_total  = {total:,.1f}  (pooled RMSE check = {check_rmse:.4f})")
    print(f"  offset (定数バイアス) : {offset:>18,.1f}  {_fmt_pct(100 * offset / total)}")
    print(f"  slope  (線形ドリフト) : {slope:>18,.1f}  {_fmt_pct(100 * slope / total)}")
    print(f"  shape  (残差/高周波)  : {shape:>18,.1f}  {_fmt_pct(100 * shape / total)}")
    print("=" * 78)


def print_hard_well_share(per_well: pd.DataFrame, top_frac: float) -> pd.Index:
    n_wells = len(per_well)
    n_top = max(1, round(n_wells * top_frac))
    ranked = per_well.sort_values("sse_total", ascending=False)
    top_wells = ranked.index[:n_top]

    total_sse = per_well["sse_total"].sum()
    top_sse = per_well.loc[top_wells, "sse_total"].sum()
    top_rows = per_well.loc[top_wells, "n"].sum()
    total_rows = per_well["n"].sum()

    print("\n" + "=" * 78)
    print(f"② hard-well 特定: SSE 上位 {top_frac:.0%}（{n_top}/{n_wells} well）")
    print("-" * 78)
    print(f"  行数シェア : {top_rows:,}/{total_rows:,} = {_fmt_pct(100 * top_rows / total_rows)}")
    print(f"  SSE シェア : {top_sse:,.1f}/{total_sse:,.1f} = {_fmt_pct(100 * top_sse / total_sse)}")
    bottom_pct = 100 * (1 - top_frac)
    bottom_share = 100 * (total_sse - top_sse) / total_sse
    print(f"  下位{bottom_pct:.0f}% の SSE シェア: {_fmt_pct(bottom_share)}")
    print("=" * 78)
    return top_wells


def print_characteristics_table(
    per_well: pd.DataFrame, signals: pd.DataFrame, top_wells: pd.Index
) -> None:
    joined = per_well.join(signals, how="left")
    joined["carry_drift_abs"] = (joined["anchor"] - joined["true_tail"]).abs()
    joined["offset_ratio"] = joined["sse_offset"] / joined["sse_total"].replace(0, np.nan)
    joined["slope_ratio"] = joined["sse_slope"] / joined["sse_total"].replace(0, np.nan)
    joined["shape_ratio"] = joined["sse_shape"] / joined["sse_total"].replace(0, np.nan)

    is_top = joined.index.isin(top_wells)
    cols = [
        "rmse",
        "n",
        "offset_ratio",
        "slope_ratio",
        "shape_ratio",
        "carry_drift_abs",
        "gr_nan_frac",
        "pf_std_mean",
        "pf_std_max",
        "beam_margin_mean",
        "spatial_prefix_rmse",
        "spatial_nn_dist_median",
        "pf_d_final",
        "beam_d_final",
        "spatial_d_final",
    ]

    print("\n" + "=" * 92)
    print("③ hard-well（上位10%） vs easy-well（下位90%） 特性中央値")
    print("-" * 92)
    print(f"{'signal':<26}{'hard median':>14}{'easy median':>14}{'hard/easy':>12}")
    print("-" * 92)
    for c in cols:
        hard_med = joined.loc[is_top, c].median()
        easy_med = joined.loc[~is_top, c].median()
        easy_valid = easy_med != 0 and not np.isnan(easy_med)
        ratio = hard_med / easy_med if easy_valid else np.nan
        print(f"{c:<26}{hard_med:>14.4f}{easy_med:>14.4f}{ratio:>12.2f}")
    print("=" * 92)

    for stat, label in [(0.90, "p90"), (1.00, "max")]:
        val = joined["rmse"].quantile(stat)
        print(f"  参考: 全体 per-well RMSE {label} = {val:.4f}ft")


def print_routing_feasibility(
    per_well: pd.DataFrame, signals: pd.DataFrame, top_wells: pd.Index, top_frac: float
) -> None:
    """Recall/precision of ranking wells by pre-inference-available signals."""
    joined = per_well.join(signals, how="left")
    n_wells = len(joined)
    n_top = max(1, round(n_wells * top_frac))
    true_hard = set(top_wells)

    # direction: True = "higher signal -> harder"; each candidate is a signal
    # computable from the known-zone prefix / GR log alone (no ground truth).
    candidate_signals: list[tuple[str, bool]] = [
        ("pf_std_mean", True),
        ("pf_std_max", True),
        ("spatial_prefix_rmse", True),
        ("spatial_nn_dist_median", True),
        ("eval_len", True),
        ("gr_nan_frac", True),
        ("beam_margin_mean", False),  # higher margin = more decisive DP -> expect easier
        ("pf_d_final", None),  # sign-agnostic drift magnitude, use abs
        ("beam_d_final", None),
        ("spatial_d_final", None),
    ]

    print("\n" + "=" * 92)
    print(f"④ 事前識別可能性: 予測時に分かる信号で hard-well 上位{top_frac:.0%}を")
    print("   ランキングした場合")
    print("-" * 92)
    print(f"{'signal':<26}{'direction':>10}{'spearman':>11}{'recall':>10}{'precision':>11}")
    print("-" * 92)
    for name, higher_is_harder in candidate_signals:
        vals = joined[name].to_numpy(dtype=np.float64)
        if higher_is_harder is None:
            vals = np.abs(vals)
            higher_is_harder = True
        valid = np.isfinite(vals)
        rank_vals = np.where(valid, vals, -np.inf if higher_is_harder else np.inf)

        order = np.argsort(-rank_vals if higher_is_harder else rank_vals)
        predicted_hard = set(joined.index[order[:n_top]])
        tp = len(predicted_hard & true_hard)
        recall = tp / len(true_hard)
        precision = tp / len(predicted_hard)

        sse = joined.loc[valid, "sse_total"].to_numpy()
        sig = vals[valid]
        if valid.sum() > 2:
            spearman = float(pd.Series(sig).corr(pd.Series(sse), method="spearman"))
        else:
            spearman = np.nan
        direction = "+" if higher_is_harder else "-"
        print(f"{name:<26}{direction:>10}{spearman:>11.3f}{recall:>10.3f}{precision:>11.3f}")
    print("-" * 92)
    print(f"  baseline (ランダム選択期待値): recall=precision={top_frac:.3f}")
    print("=" * 92)


def oracle_substitution(
    oof: dict[str, np.ndarray], per_well: pd.DataFrame, top_wells: pd.Index
) -> None:
    """Upper-bound probes: swap hard wells' predictions to alternative sources.

    Every number here is an ORACLE ceiling (uses the true label to decide, or
    to select), not a deployable predictor -- reported to bound how much
    headroom exists if a routing rule could perfectly identify + fix hard
    wells, never as an adoption recommendation.
    """
    y_true, pred, boundaries, used_wells = (
        oof["y_true"],
        oof["pred"],
        oof["boundaries"],
        oof["used_wells"],
    )
    well_pos = {str(w): i for i, w in enumerate(used_wells)}
    total_rows = y_true.shape[0]
    baseline_sse = pooled_sse(y_true, pred)
    baseline_rmse = np.sqrt(baseline_sse / total_rows)

    top_set = set(str(w) for w in top_wells)
    top_positions = [well_pos[w] for w in top_set]
    hard_rows_mask = np.zeros(total_rows, dtype=bool)
    for i in top_positions:
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        hard_rows_mask[s:e_] = True

    # ---- load per-row alternative predictions for the hard wells only -------
    pf_blend_pred = oof["pf_blend"]  # already full-length, aligned

    anchor_pred = oof["anchor"].copy()  # carry_last equivalent (constant per well)

    spatial_pred = np.full(total_rows, np.nan, dtype=np.float64)
    for i in top_positions:
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        well = str(used_wells[i])
        with np.load(SPATIAL_CACHE_DIR / f"{well}.npz") as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
        assert spatial_tvt.size == (e_ - s), f"{well}: spatial cache length mismatch"
        spatial_pred[s:e_] = spatial_tvt

    print("\n" + "=" * 92)
    print("⑤ oracle 差し替え上限（hard-well 上位10%のみ, 真値使用 = 採用可否ではなく上限測定）")
    print("-" * 92)
    print(f"  baseline (stack v2 all-in, 全 well)      pooled RMSE = {baseline_rmse:.4f}")

    def _substitute_uniform(name: str, alt: np.ndarray) -> None:
        pred_sub = pred.copy()
        pred_sub[hard_rows_mask] = alt[hard_rows_mask]
        rmse_sub = pooled_rmse(y_true, pred_sub)
        print(
            f"  hard-well を一律 {name} に差し替え          pooled RMSE = {rmse_sub:.4f}  "
            f"(Δ{rmse_sub - baseline_rmse:+.4f})"
        )

    _substitute_uniform("carry_last", anchor_pred)
    _substitute_uniform("PF blend w0.7", pf_blend_pred)
    _substitute_uniform("spatial", spatial_pred)

    # ---- per-well oracle pick among {stack, carry_last, PF blend, spatial} --
    pred_oracle = pred.copy()
    picks: dict[str, str] = {}
    candidates = {
        "stack": pred,
        "carry_last": anchor_pred,
        "PF blend": pf_blend_pred,
        "spatial": spatial_pred,
    }
    for i in top_positions:
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        well = str(used_wells[i])
        best_name, best_sse = "stack", np.inf
        for name, arr in candidates.items():
            seg = arr[s:e_]
            if not np.all(np.isfinite(seg)):
                continue
            sse = float(np.sum((seg - y_true[s:e_]) ** 2))
            if sse < best_sse:
                best_sse, best_name = sse, name
        picks[well] = best_name
        pred_oracle[s:e_] = candidates[best_name][s:e_]

    oracle_rmse = pooled_rmse(y_true, pred_oracle)
    pick_counts = pd.Series(list(picks.values())).value_counts().to_dict()
    print(
        f"  hard-well を per-well oracle 選択に差し替え pooled RMSE = {oracle_rmse:.4f}  "
        f"(Δ{oracle_rmse - baseline_rmse:+.4f})  内訳={pick_counts}"
    )
    print("=" * 92)


def main() -> None:
    t_start = time.time()

    print(f"loading {OOF_PATH.name} ...")
    oof = load_stack_oof()
    print(
        f"  {int(oof['y_true'].shape[0]):,} rows, {oof['n_wells']} wells, "
        f"baseline pooled RMSE = {pooled_rmse(oof['y_true'], oof['pred']):.4f}"
    )

    per_well = per_well_error_decomposition(oof)
    print_decomposition_summary(per_well)

    top_wells = print_hard_well_share(per_well, TOP_FRAC)

    print("\nloading well-level signals (tracker cache + spatial cache + GR NaN rate) ...")
    signals = load_well_signals(oof["used_wells"])

    print_characteristics_table(per_well, signals, top_wells)
    print_routing_feasibility(per_well, signals, top_wells, TOP_FRAC)
    oracle_substitution(oof, per_well, top_wells)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
