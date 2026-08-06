"""Self-referential (within-well) GR tie-point decisive pretest (cheap, no build).

Background (see task brief / docs/design.md:296-339, host comment 698825
"lateral GR self-correlation > typewell", CLAUDE.md's typewell-NCC rejection
at 47.08 pooled RMSE). Every registration method we ship (PF, beam, GR-affine
typewell match) aligns the horizontal (lateral) GR log against the *vertical
typewell's* GR(TVT) curve -- a cross-well match. That match must survive two
confounders: (a) cross-well character mismatch (different tool calibration,
different local geology even at the same TVT) and (b) a ~25-100x MD:TVT
stretch factor in the lateral (CLAUDE.md: dTVT/dMD median 0.01-0.04 ft/row),
which is why naive fixed-window NCC against the typewell was rejected
(47.08, CLAUDE.md scoreboard).

This pretest asks a narrower, cheaper question: instead of matching the eval
zone's GR against a *different well's* typewell, match it against **the same
well's own known (build/heel) prefix** -- rows where `TVT_input` is already
known (`data.eval_mask` complement). This is still a cross-*section* match
(vertical build section vs near-horizontal lateral -> the same MD:TVT stretch
confound as (b) applies within one well too), but confound (a) is structurally
gone: there is no tool-calibration or local-geology mismatch, because it is
the same borehole. If the near-flat lateral (TVT wanders ~median 11 / p90 32ft
around the anchor, CLAUDE.md) re-crosses a GR level already recorded in the
well's own known section, that known section's `TVT_input` at the matching
position is an exact, leak-free tie-point candidate for the eval row.

Method: multi-scale (SCALES) x 2-channel (raw GR, GR gradient) normalized
cross-correlation (NCC) of each eval-zone window against every same-length
window in the well's own known-zone GR trace. For each eval row (on a common,
edge-trimmed grid), take the (channel, scale, known-offset) triple with the
single highest peak correlation as the honest tie-point prediction; record
that peak correlation and its non-max-suppressed runner-up gap as a
confidence covariate. This mirrors ``run_donor_pretest.py``'s honest/oracle
duality: an **oracle** predictor (closest known TVT_input to the true y_true,
searched over the *entire* known array, no GR involved) upper-bounds what any
selector over this well's known section could ever achieve; the **honest**
predictor is what a GR-only selector, blind to y_true, actually achieves. A
**random-offset control** (uniformly random known position, same grid) and a
**known-median-constant control** isolate how much of any honest gain is
genuine GR-driven selection vs. the trivial fact that a near-flat lateral
tends to sit close to *any* known TVT_input value already.

Leak boundary (see CLAUDE.md's 5 leak rules -- checked against each):
  - Matching uses only `GR` (imputed via each well's *own* full-log
    interpolation, forward+backward -- a legitimate input-feature fill, not a
    TVT/TVT_input read) and `TVT_input`'s known (non-NaN) prefix. Never reads
    `h["TVT"]` except to build `y_true` for scoring (never fed into matching).
  - Tie-point candidates are drawn only from the *same* well's own known
    zone -- no other train well's data is touched (this pretest measures
    within-well signal only, orthogonal to the donor-well pretest).
  - The eval zone's own `TVT_input` (always NaN by `eval_mask`'s definition)
    is never read; only `GR` (present, ~30% NaN, in both zones per CLAUDE.md)
    is used for matching, imputed from the well's own full-length GR series
    (both known- and eval-zone GR values -- legitimate at submission time,
    since GR, unlike TVT, is a raw measurement present in test's eval zone
    too).
  - The feature-value regression (Sec. 3 of main()) reuses
    ``outputs/stack_v26_oof.npz``'s ``row_fold`` column -- the exact
    well-GroupKFold(5, seed=42) split the 9.1456 ledger number itself used --
    so the reported R2/counterfactual pooled RMSE is directly comparable, not
    a differently-drawn split.

Runtime/memory: dense per-row matching (stride=1) benchmarked at ~0.1-0.3s for
773 wells worth of well sizes (see task-note in ledger/agent report) -> ~2-4
minutes end to end. Peak transient array per (well, channel, scale) is the
correlation surface (eval_len x known_len), <=~96MB worst-case (10,052 x 2,392
float32), freed every iteration -- never holds all 773 wells' surfaces at
once. numpy/pandas/scipy only, no NN, no training beyond a closed-form OLS
(``numpy.linalg.lstsq``) for the tiny feature-value check.

Usage::

    uv run python scripts/run_selfref_tiepoint_pretest.py [--n-wells N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rogii import data as D  # noqa: E402

OOF_PATH = REPO_ROOT / "outputs" / "stack_v26_oof.npz"

# --- ledger sanity targets (must reproduce exactly -- confirms OOF load is correct) ---
POOLED_CARRY_LEDGER = 15.9099
POOLED_PFBLEND_LEDGER = 11.0621
POOLED_OOF_MEAN_LEDGER = 9.1456
SANITY_TOL = 0.01

# --- matching config ---
SCALES = [8, 20, 50]  # rows(=ft, 1ft/row) -- PF's (8,12,20,30) + host's "直前50点" hint
CHANNELS = ("raw", "deriv")
# Degenerate-window threshold is RELATIVE to each well's own channel std, not absolute: GR is
# ~18-265 (well std ~20-40), so a fixed absolute epsilon (e.g. 1e-3) never fires and lets
# near-flat (low-information) windows get z-scored into huge spurious correlations (empirically
# confirmed: eval-zone windows are systematically flatter than known-zone windows at the same row
# scale -- direct evidence of the MD:TVT stretch mismatch the task brief hypothesizes around).
REL_STD_FRAC = 0.15
STD_EPS_ABS = 1e-6  # floor to avoid pure division-by-zero on a truly constant window
N_RANDOM_REPS = 3
RANDOM_SEED = 42
MIN_KNOWN_LEN = max(SCALES) + 10  # guard: well skipped (falls back to anchor) below this length

# --- pre-declared decision gates (task brief) ---
GO_R2_THRESHOLD = 0.1
GO_HONEST_BEATS_OOF_MARGIN = 0.05  # ft; "明確に下回る" needs >0.05ft per playbook 02's noise floor


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def unweighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def impute_gr(gr_full: np.ndarray) -> np.ndarray:
    """Fill NaNs in a well's *full-length* GR trace via same-well interpolation.

    Legitimate: GR (unlike TVT/TVT_input) is a raw measurement present, with
    gaps, in both the known and eval zones at submission time too -- filling
    gaps from the well's own other GR readings is not a label read.
    """
    s = pd.Series(gr_full).interpolate(limit_direction="both")
    if s.isna().any():  # entire column NaN (degenerate well) -> fall back to 0
        s = s.fillna(0.0)
    return s.to_numpy(dtype=np.float64)


def zscore_rows(win: np.ndarray, deg_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise z-score a (n, w) window matrix; flag degenerate (near-flat) rows.

    ``deg_threshold`` is an absolute std cutoff already scaled by the caller
    (``REL_STD_FRAC`` x the well/channel's own reference std -- see module
    docstring note on why a fixed absolute epsilon is wrong for this data).
    """
    mu = win.mean(axis=1, keepdims=True)
    sd = win.std(axis=1, keepdims=True)
    degenerate = (sd < deg_threshold).ravel()
    sd_safe = np.where(sd < STD_EPS_ABS, 1.0, sd)
    z = (win - mu) / sd_safe
    return z.astype(np.float32), degenerate


@dataclass
class MatchResult:
    pred_tvt: np.ndarray  # (T,) honest tie-point pred (argmax-correlation combo), T=trimmed grid
    peak_corr: np.ndarray  # (T,) winning (channel,scale) peak correlation
    margin: np.ndarray  # (T,) peak - NMS runner-up correlation
    pred_tvt_median: np.ndarray  # (T,) robust variant: median across all (channel,scale) combos
    margin_left: int  # rows trimmed off the eval-zone start (edge effect of largest scale)
    margin_right: int  # rows trimmed off the eval-zone end


def match_well(
    known_channels: dict[str, np.ndarray],
    eval_channels: dict[str, np.ndarray],
    known_tvt: np.ndarray,
    scales: list[int],
) -> MatchResult:
    """Multi-scale x multi-channel NCC self-match: eval zone vs. this well's known zone.

    Returns per-row (trimmed grid) honest tie-point predictions + confidence
    covariates. The grid is common across all (channel, scale) combos (defined
    by the *largest* scale's edge margin) so results are directly comparable
    and combinable via elementwise argmax over peak correlation.
    """
    eval_len = len(next(iter(eval_channels.values())))
    max_scale = max(scales)
    margin_left = max_scale // 2
    margin_right = max_scale - margin_left
    centers = np.arange(margin_left, eval_len - margin_right)
    t = len(centers)

    best_pred = np.full(t, np.nan)
    best_corr = np.full(t, -np.inf)
    best_margin = np.zeros(t)
    all_combo_preds: list[np.ndarray] = []  # for the median-of-combos robustified variant

    for chan in known_channels:
        k_arr = known_channels[chan]
        e_arr = eval_channels[chan]
        # relative degeneracy threshold: this well's own known-zone channel std (large, stable
        # sample) is the reference "typical variability" -- see REL_STD_FRAC note above.
        ref_std = float(np.std(k_arr))
        deg_threshold = max(STD_EPS_ABS, REL_STD_FRAC * ref_std)
        for sc in scales:
            half = sc // 2
            starts = centers - half
            ew = e_arr[starts[:, None] + np.arange(sc)[None, :]]  # (T, sc) materialized
            kw = sliding_window_view(k_arr, sc)  # (K, sc) view, K = known_len - sc + 1
            ewz, e_deg = zscore_rows(ew, deg_threshold)
            kwz, k_deg = zscore_rows(kw, deg_threshold)
            corr = ewz @ kwz.T / sc  # (T, K)
            corr[e_deg, :] = -np.inf
            corr[:, k_deg] = -np.inf

            best_idx = np.argmax(corr, axis=1)
            peak = corr[np.arange(t), best_idx]

            # non-max-suppressed runner-up: mask offsets within one window length of the peak
            offsets = np.arange(corr.shape[1])
            nms_mask = np.abs(offsets[None, :] - best_idx[:, None]) <= sc
            corr_masked = np.where(nms_mask, -np.inf, corr)
            second = corr_masked.max(axis=1)
            margin_cs = np.where(np.isfinite(second), peak - second, peak)

            centers_known = best_idx + half  # matched position within known array
            pred_cs = known_tvt[centers_known]
            all_combo_preds.append(np.where(np.isfinite(peak), pred_cs, np.nan))

            improve = peak > best_corr
            best_pred = np.where(improve, pred_cs, best_pred)
            best_margin = np.where(improve, margin_cs, best_margin)
            best_corr = np.where(improve, peak, best_corr)

    median_pred = np.nanmedian(np.stack(all_combo_preds), axis=0)
    return MatchResult(best_pred, best_corr, best_margin, median_pred, margin_left, margin_right)


def oracle_tvt(known_tvt: np.ndarray, y_true_well: np.ndarray) -> np.ndarray:
    """Closest known TVT_input value to the true y_true, searched over the *whole*
    known array (no GR involved) -- upper bound of what any within-well
    selector could achieve. Uses y_true for *selection only* (a theoretical
    ceiling probe), never as a submission-time predictor.
    """
    order = np.argsort(known_tvt)
    kt_sorted = known_tvt[order]
    idx = np.searchsorted(kt_sorted, y_true_well)
    idx = np.clip(idx, 1, len(kt_sorted) - 1)
    left = kt_sorted[idx - 1]
    right = kt_sorted[idx]
    pick_left = np.abs(y_true_well - left) <= np.abs(right - y_true_well)
    return np.where(pick_left, left, right)


def random_control_tvt(known_tvt: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(known_tvt), size=n)
    return known_tvt[idx]


def extend_to_full(
    grid_arr: np.ndarray, eval_len: int, margin_left: int, margin_right: int
) -> np.ndarray:
    """Edge-extend a trimmed-grid array to the full eval-zone length (nearest value)."""
    full = np.empty(eval_len)
    full[margin_left : eval_len - margin_right] = grid_arr
    if margin_left > 0:
        full[:margin_left] = grid_arr[0]
    if margin_right > 0:
        full[eval_len - margin_right :] = grid_arr[-1]
    return full


def process_well(
    well: str, rng: np.random.Generator
) -> dict[str, np.ndarray] | None:
    h = D.load_horizontal(well, "train")
    mask = D.eval_mask(h)
    known_mask = ~mask
    eval_len = int(mask.sum())
    known_len = int(known_mask.sum())
    if eval_len == 0 or known_len < MIN_KNOWN_LEN:
        return None

    gr_full = h["GR"].to_numpy(dtype=np.float64)
    gr_imp = impute_gr(gr_full)
    deriv_imp = np.gradient(gr_imp)

    known_channels = {"raw": gr_imp[known_mask], "deriv": deriv_imp[known_mask]}
    eval_channels = {"raw": gr_imp[mask], "deriv": deriv_imp[mask]}
    known_tvt = h["TVT_input"].to_numpy(dtype=np.float64)[known_mask]
    y_true_well = h["TVT"].to_numpy(dtype=np.float64)[mask]
    anchor_well = D.last_known_tvt(h)

    mr = match_well(known_channels, eval_channels, known_tvt, SCALES)

    # rows where every (channel, scale) combo was degenerate (best_corr stayed -inf, no genuine
    # match found anywhere) fall back to the anchor -- same "no signal -> flat carry" convention
    # CLAUDE.md mandates for every predictor's exception path. Tracked as no_match_frac.
    no_match = ~np.isfinite(mr.peak_corr)
    pred_tvt = np.where(no_match, anchor_well, mr.pred_tvt)
    peak_corr = np.where(no_match, 0.0, mr.peak_corr)
    match_margin = np.where(no_match, 0.0, mr.margin)
    pred_tvt_median = np.where(np.isnan(mr.pred_tvt_median), anchor_well, mr.pred_tvt_median)

    ml, mrr = mr.margin_left, mr.margin_right
    honest_full = extend_to_full(pred_tvt, eval_len, ml, mrr)
    honest_median_full = extend_to_full(pred_tvt_median, eval_len, ml, mrr)
    corr_full = extend_to_full(peak_corr, eval_len, ml, mrr)
    margin_full = extend_to_full(match_margin, eval_len, ml, mrr)
    no_match_full = extend_to_full(no_match.astype(np.float64), eval_len, ml, mrr)

    oracle_full = oracle_tvt(known_tvt, y_true_well)

    rand_preds = np.stack(
        [random_control_tvt(known_tvt, eval_len, rng) for _ in range(N_RANDOM_REPS)]
    )
    random_full = rand_preds.mean(axis=0)  # for the pointwise table; SE pooled per-rep too
    random_reps_se = np.mean((y_true_well[None, :] - rand_preds) ** 2, axis=1)  # per-rep well MSE

    median_const_full = np.full(eval_len, float(np.median(known_tvt)))

    return {
        "well": well,
        "y_true": y_true_well,
        "anchor": np.full(eval_len, anchor_well),
        "honest": honest_full,
        "honest_median": honest_median_full,
        "confidence": corr_full,
        "margin": margin_full,
        "no_match": no_match_full,
        "oracle": oracle_full,
        "random": random_full,
        "random_reps_se": random_reps_se,
        "median_const": median_const_full,
        "known_len": known_len,
        "eval_len": eval_len,
    }


def well_groupkfold_ols(
    features: np.ndarray, target: np.ndarray, row_fold: np.ndarray, n_splits: int
) -> np.ndarray:
    """Closed-form OLS (with intercept) fit per well-GroupKFold fold, OOF predictions."""
    n = len(target)
    oof = np.full(n, np.nan)
    design = np.column_stack([np.ones(n), features])
    for f in range(n_splits):
        train_mask = row_fold != f
        valid_mask = row_fold == f
        if not valid_mask.any() or not train_mask.any():
            continue
        beta, *_ = np.linalg.lstsq(design[train_mask], target[train_mask], rcond=None)
        oof[valid_mask] = design[valid_mask] @ beta
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wells", type=int, default=0, help="0 = all 773 (full run)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    # --- 0. load OOF for sanity + the feature-value test's row_fold split ---
    with np.load(OOF_PATH, allow_pickle=True) as npz:
        y_true_oof = npz["y_true"].astype(np.float64)
        anchor_oof = npz["anchor"].astype(np.float64)
        pf_blend_oof = npz["pf_blend"].astype(np.float64)
        oof_mean_oof = npz["oof_mean"].astype(np.float64)
        well_idx_oof = npz["well_idx"].astype(np.int64)
        row_fold_oof = npz["row_fold"].astype(np.int64)
        used_wells_oof = [str(w) for w in npz["used_wells"]]
    n_wells_oof = len(used_wells_oof)
    print(f"loaded {OOF_PATH.relative_to(REPO_ROOT)}: rows={len(y_true_oof):,} wells={n_wells_oof}")

    print("\n=== sanity: reproduce ledger pooled RMSEs from stack_v26_oof.npz ===")
    p_carry = pooled_rmse(y_true_oof, anchor_oof)
    p_pf = pooled_rmse(y_true_oof, pf_blend_oof)
    p_oof = pooled_rmse(y_true_oof, oof_mean_oof)
    print(f"  carry_last (anchor) = {p_carry:.4f}  (ledger {POOLED_CARRY_LEDGER})")
    print(f"  pf_blend             = {p_pf:.4f}  (ledger {POOLED_PFBLEND_LEDGER})")
    print(f"  oof_mean (stack v2.6)= {p_oof:.4f}  (ledger {POOLED_OOF_MEAN_LEDGER})")
    for label, val, ledger in [
        ("carry_last", p_carry, POOLED_CARRY_LEDGER),
        ("pf_blend", p_pf, POOLED_PFBLEND_LEDGER),
        ("oof_mean", p_oof, POOLED_OOF_MEAN_LEDGER),
    ]:
        if abs(val - ledger) > SANITY_TOL:
            raise SystemExit(f"STOP: {label}={val:.4f} does not match ledger {ledger} -- aborting.")
    print("  -> PASS, all three ledger numbers reproduced from the OOF file.")

    boundaries: dict[str, tuple[int, int]] = {}
    counts_per_well = np.bincount(well_idx_oof, minlength=n_wells_oof)
    assert np.array_equal(well_idx_oof, np.repeat(np.arange(n_wells_oof), counts_per_well)), (
        "well_idx is not sorted/contiguous per well -- alignment assumption broken"
    )
    starts = np.concatenate(([0], np.cumsum(counts_per_well)))
    for i, w in enumerate(used_wells_oof):
        boundaries[w] = (int(starts[i]), int(starts[i + 1]))

    # --- 1. run self-ref matching, well by well (streamed) ---
    wells = used_wells_oof if args.n_wells == 0 else used_wells_oof[: args.n_wells]
    print(f"\n=== self-ref matching: {len(wells)} wells, scales={SCALES}, channels={CHANNELS} ===")
    t0 = time.time()
    n_skipped = 0
    n_failed = 0
    per_well_results: list[dict] = []
    for i, w in enumerate(wells):
        try:
            res = process_well(w, rng)
        except Exception as e:  # noqa: BLE001 -- pretest must not crash on one bad well
            n_failed += 1
            print(f"  [FAIL] well={w}: {type(e).__name__}: {e}")
            continue
        if res is None:
            n_skipped += 1
            continue
        per_well_results.append(res)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(wells)} wells ({time.time() - t0:.1f}s)")
    print(
        f"matching done: {len(per_well_results)} ok, {n_skipped} skipped "
        f"(known_len<{MIN_KNOWN_LEN}), {n_failed} failed ({time.time() - t0:.1f}s)"
    )

    if not per_well_results:
        raise SystemExit("STOP: 0 wells produced results -- cannot proceed.")

    # --- 2. concat pooled arrays (row order == used_wells order, matches OOF file) ---
    y_true = np.concatenate([r["y_true"] for r in per_well_results])
    anchor = np.concatenate([r["anchor"] for r in per_well_results])
    honest = np.concatenate([r["honest"] for r in per_well_results])
    honest_median = np.concatenate([r["honest_median"] for r in per_well_results])
    confidence = np.concatenate([r["confidence"] for r in per_well_results])
    margin = np.concatenate([r["margin"] for r in per_well_results])
    no_match = np.concatenate([r["no_match"] for r in per_well_results]).astype(bool)
    oracle = np.concatenate([r["oracle"] for r in per_well_results])
    random_pred = np.concatenate([r["random"] for r in per_well_results])
    median_const = np.concatenate([r["median_const"] for r in per_well_results])
    well_names = [r["well"] for r in per_well_results]
    well_lens = np.array([r["eval_len"] for r in per_well_results])
    row_well_idx = np.repeat(np.arange(len(per_well_results)), well_lens)

    # subset of the *original* OOF arrays restricted to the wells we actually matched
    # (aligns pf_blend/oof_mean for the feature-value test in Sec 3 & the pooled table in Sec 1)
    used_slices = [boundaries[w] for w in well_names]
    pf_blend = np.concatenate([pf_blend_oof[s:e] for s, e in used_slices])
    oof_mean = np.concatenate([oof_mean_oof[s:e] for s, e in used_slices])
    row_fold = np.concatenate([row_fold_oof[s:e] for s, e in used_slices])
    y_true_check = np.concatenate([y_true_oof[s:e] for s, e in used_slices])
    assert np.allclose(y_true, y_true_check), "row alignment mismatch: y_true differs from OOF file"

    # --- 3. primary pooled RMSE table ---
    print("\n" + "=" * 100)
    print("SECTION 1: 3系 pooled RMSE (carry_last / oof_mean / oracle tie / honest tie)")
    print("=" * 100)
    p_carry_sub = pooled_rmse(y_true, anchor)
    p_oof_sub = pooled_rmse(y_true, oof_mean)
    p_pf_sub = pooled_rmse(y_true, pf_blend)
    p_oracle = pooled_rmse(y_true, oracle)
    p_honest = pooled_rmse(y_true, honest)
    p_random = pooled_rmse(y_true, random_pred)
    p_median_const = pooled_rmse(y_true, median_const)
    print(f"  {'predictor':<32}{'pooled RMSE (ft)':>18}")
    print(f"  {'carry_last (subset)':<32}{p_carry_sub:>18.4f}")
    print(f"  {'pf_blend (subset)':<32}{p_pf_sub:>18.4f}")
    print(f"  {'oof_mean / 前ベスト (subset)':<32}{p_oof_sub:>18.4f}")
    print(f"  {'known-median-const (control)':<32}{p_median_const:>18.4f}")
    print(f"  {'random-offset tie (control)':<32}{p_random:>18.4f}")
    print(f"  {'HONEST tie-point':<32}{p_honest:>18.4f}")
    p_honest_med = pooled_rmse(y_true, honest_median)
    print(f"  {'HONEST tie-point (median-of-6 robust)':<32}{p_honest_med:>18.4f}")
    print(f"  {'ORACLE tie-point (upper bound)':<32}{p_oracle:>18.4f}")

    abs_err = np.abs(honest - y_true)
    bad = abs_err > 50.0
    sse_total = float(np.sum((honest - y_true) ** 2))
    sse_bad = float(np.sum((honest[bad] - y_true[bad]) ** 2))
    print(
        f"\n  diagnostic: median |honest-y_true|={np.median(abs_err):.3f}ft  "
        f"p90={np.percentile(abs_err, 90):.3f}ft  "
        f"catastrophic(err>50ft) frac={float(bad.mean()):.4f}  "
        f"SSE share of catastrophic rows={sse_bad / sse_total:.4f}"
    )
    corr_vs_err = float(np.corrcoef(confidence, abs_err)[0, 1])
    margin_vs_err = float(np.corrcoef(margin, abs_err)[0, 1])
    print(
        f"  diagnostic: corr(peak_corr,|err|)={corr_vs_err:+.4f}  "
        f"corr(margin,|err|)={margin_vs_err:+.4f}  "
        f"(near-zero -> confidence does not discriminate correct from wrong "
        f"matches, cf. CLAUDE.md's typewell-NCC finding r=+0.054 p=0.30)"
    )

    # --- 4. feature-value: tie_pred - pf_blend (+ confidence) predicts y_true - pf_blend ---
    print("\n" + "=" * 100)
    print("SECTION 2: tie_pred-pf_blend 特徴の well-GroupKFold R^2 + 反実仮想 pooled")
    print("=" * 100)
    tie_delta = honest - pf_blend
    target = y_true - pf_blend
    feats = np.column_stack([tie_delta, confidence, tie_delta * confidence, margin])
    resid_oof = well_groupkfold_ols(feats, target, row_fold, n_splits=int(row_fold.max()) + 1)
    valid = ~np.isnan(resid_oof)
    r2_resid = unweighted_r2(target[valid], resid_oof[valid])
    counterfactual_pred = pf_blend[valid] + resid_oof[valid]
    p_counterfactual = pooled_rmse(y_true[valid], counterfactual_pred)
    print(f"  rows used in OLS OOF: {valid.sum():,}/{len(target):,}")
    print(f"  R^2(residual, well-GroupKFold OOF)      = {r2_resid:.4f}")
    print(f"  counterfactual pooled RMSE (pf_blend+resid_hat) = {p_counterfactual:.4f}")
    print(f"  vs oof_mean (subset)                    = {p_oof_sub:.4f}")
    print(f"  vs pf_blend alone (subset)               = {p_pf_sub:.4f}")

    # --- 5. coverage + tail-well stratification ---
    print("\n" + "=" * 100)
    print("SECTION 3: カバレッジ・tail well(SSE上位)層別")
    print("=" * 100)
    print(
        f"  no-match rows (all channel/scale combos degenerate -> fell back to anchor): "
        f"{float(no_match.mean()):.4f} ({int(no_match.sum()):,}/{len(no_match):,})"
    )
    for thresh in (0.2, 0.3, 0.5):
        cov = float(np.mean(confidence >= thresh))
        if cov > 0:
            sub = confidence >= thresh
            p_hi = pooled_rmse(y_true[sub], honest[sub])
            p_lo = pooled_rmse(y_true[~sub], honest[~sub]) if (~sub).any() else float("nan")
            print(
                f"  conf>={thresh:.1f}: coverage={cov:.3f}  "
                f"honest pooled(covered)={p_hi:.4f}  honest pooled(uncovered)={p_lo:.4f}"
            )
        else:
            print(f"  conf>={thresh:.1f}: coverage=0.000 (no rows)")

    per_well_sse_oof = np.zeros(len(per_well_results))
    for i in range(len(per_well_results)):
        m = row_well_idx == i
        per_well_sse_oof[i] = np.sum((y_true[m] - oof_mean[m]) ** 2)
    tail_thresh = np.percentile(per_well_sse_oof, 90)
    tail_well_ids = np.where(per_well_sse_oof >= tail_thresh)[0]
    tail_row_mask = np.isin(row_well_idx, tail_well_ids)
    n_tail = len(tail_well_ids)
    print(f"\n  tail wells (top decile SSE under oof_mean): {n_tail}/{len(per_well_results)}")
    print(
        f"    tail:     honest={pooled_rmse(y_true[tail_row_mask], honest[tail_row_mask]):.4f}  "
        f"oof_mean={pooled_rmse(y_true[tail_row_mask], oof_mean[tail_row_mask]):.4f}  "
        f"oracle={pooled_rmse(y_true[tail_row_mask], oracle[tail_row_mask]):.4f}  "
        f"coverage(conf>=0.3)={float(np.mean(confidence[tail_row_mask] >= 0.3)):.3f}"
    )
    print(
        f"    non-tail: honest={pooled_rmse(y_true[~tail_row_mask], honest[~tail_row_mask]):.4f}  "
        f"oof_mean={pooled_rmse(y_true[~tail_row_mask], oof_mean[~tail_row_mask]):.4f}  "
        f"oracle={pooled_rmse(y_true[~tail_row_mask], oracle[~tail_row_mask]):.4f}  "
        f"coverage(conf>=0.3)={float(np.mean(confidence[~tail_row_mask] >= 0.3)):.3f}"
    )

    # --- 6. controls: self-ref-specific increment over pseudo-correlation ---
    print("\n" + "=" * 100)
    print("SECTION 4: 対照との差（self-ref 固有増分）")
    print("=" * 100)
    print(f"  honest tie pooled           = {p_honest:.4f}")
    print(
        f"  random-offset control pooled= {p_random:.4f}   "
        f"(delta honest-random = {p_honest - p_random:+.4f})"
    )
    print(
        f"  known-median-const control  = {p_median_const:.4f}   "
        f"(delta honest-medianconst = {p_honest - p_median_const:+.4f})"
    )
    print(
        f"  carry_last (subset)         = {p_carry_sub:.4f}   "
        f"(delta honest-carry = {p_honest - p_carry_sub:+.4f})"
    )

    # --- 7. pre-declared verdict ---
    print("\n" + "=" * 100)
    print("SECTION 5: 事前宣言ゲート判定")
    print("=" * 100)
    honest_beats_oof = (p_oof_sub - p_honest) > GO_HONEST_BEATS_OOF_MARGIN
    feature_go = r2_resid > GO_R2_THRESHOLD
    tail_honest = pooled_rmse(y_true[tail_row_mask], honest[tail_row_mask])
    tail_oof = pooled_rmse(y_true[tail_row_mask], oof_mean[tail_row_mask])
    tail_cov = float(np.mean(confidence[tail_row_mask] >= 0.3))
    tail_go = (tail_oof - tail_honest) > GO_HONEST_BEATS_OOF_MARGIN and tail_cov > 0.2

    verdict = "GO" if (honest_beats_oof or feature_go or tail_go) else "NO-GO"
    print(f"  honest tie beats oof_mean by >{GO_HONEST_BEATS_OOF_MARGIN}ft? {honest_beats_oof} "
          f"({p_oof_sub:.4f} -> {p_honest:.4f}, delta={p_oof_sub - p_honest:+.4f})")
    print(f"  feature R^2 > {GO_R2_THRESHOLD}? {feature_go} (R^2={r2_resid:.4f})")
    print(
        f"  tail-well clear improvement "
        f"(delta>{GO_HONEST_BEATS_OOF_MARGIN}ft, coverage>0.2)? {tail_go} "
        f"(oof_mean={tail_oof:.4f} -> honest={tail_honest:.4f}, "
        f"delta={tail_oof - tail_honest:+.4f}, coverage={tail_cov:.3f})"
    )
    print(f"\n  >>> 判定: {verdict} <<<")
    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
