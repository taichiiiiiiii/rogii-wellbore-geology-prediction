"""Track B: from-scratch feature module for a GBDT drift-residual predictor.

Track B is a parallel, independent pipeline (separate from the PF/beam/
spatial-tracker stack ``rogii.features``/``rogii.stack_features`` targets)
built to be auditable end-to-end: every feature here is either (a) a raw
column present in the **test** schema (``MD, X, Y, Z, GR, TVT_input``) or
(b) a statistic fit exclusively on a well's own *known* zone (rows where
``TVT_input`` is not NaN). Nothing here ever reads ``h["TVT"]`` or a
train-only formation column (``ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA``).

Target definition (verified against the data, see module docstring footer)
----------------------------------------------------------------------------
``drift = TVT_true - carry_last``, where ``carry_last`` is the single scalar
``rogii.data.last_known_tvt(h)`` (the last non-NaN ``TVT_input`` value just
before the eval zone), broadcast across every eval-zone row of that well.
This is *exactly* the quantity ``rogii.baseline.predict_carry_last`` already
predicts as a constant zero-drift baseline -- Track B's job is to predict
the residual on top of it:

    tvt_pred = carry_last + predicted_drift

Verification of this definition against ``sample_submission.csv`` / the eval
zone: submission ids are ``{well}_{row_index}`` (``rogii.data.submission_ids``),
one id per eval-zone row, and eval-zone row order is exactly
``rogii.data.eval_mask(h)`` order (``np.where(mask)[0]``, ascending row
index). ``build_trackb_features`` and ``track_b_target`` both return arrays
in that same order, so ``carry_last(h) + model.predict(build_trackb_features(h,
tw))`` lines up with ``rogii.data.submission_ids(well, h)`` element-for-element
with no re-sorting needed. Also verified directly against the data (see
``tests/test_trackb_features.py`` and the local 30-well sanity run): in the
known zone, ``TVT == TVT_input`` exactly (checked over multiple wells), so
``carry_last`` is a legitimate, leak-free anchor.

Leak boundary
-------------
Only ``MD, X, Y, Z, GR, TVT_input`` are ever read from ``h``. Critically,
``MD/X/Y/Z`` (the wellbore *trajectory*, from the drilling survey) are known
for the **entire** lateral in both train and test, eval zone included --
only the *geology* (``TVT``/``TVT_input`` in the eval zone) is hidden. This
was confirmed by inspecting a test well file directly: ``X, Y, Z`` are
non-NaN through the last row. So trajectory features below use the full
``MD`` range (including centered rolling windows that look at both sides of
an eval row) without touching geology -- this mirrors the precedent already
set by ``rogii.features.build_features`` (its ``row_frac`` feature uses
``len(h)``, i.e. the whole file, the same way). ``GR`` is also present
(mostly) in the eval zone -- it is a measured-while-drilling log, not a
geological interpretation -- so raw/derived GR features at eval rows are
legitimate too; occasional NaN GR rows (observed in real test data) are
handled via the ``gr_missing`` flag rather than silently dropped.

Self-containment (for Kaggle inlining)
---------------------------------------
This module deliberately depends on **nothing but numpy/pandas** (not even
``rogii.data``) -- ``eval_mask``/``carry_last`` below duplicate
``rogii.data.eval_mask``/``last_known_tvt`` verbatim. Code competitions run
with internet disabled and cannot ``pip install`` or ``import rogii``, so
the plan is to copy this entire file's source into a single cell of
``notebooks/submission/kernel_trackb_train`` and
``notebooks/submission/kernel_trackb_infer`` (see those directories) rather
than reference it as a package. Keeping this file free of other project
imports means that copy-paste is exact and does not silently drag in
unrelated modules.

STARTER feature set -- replaceable by design
----------------------------------------------
Every feature family below (:data:`FEATURE_FAMILIES`) is a first-pass
guess, not a tuned or validated set. This module exists to prove the
plumbing (build features -> train GBDT -> honest well-level CV) works
end-to-end; an information audit running in parallel is expected to decide
which families are trustworthy signal vs. noise vs. accidentally-leaky, and
this file's feature list should be edited (families added/removed/replaced)
based on that verdict before Track B is trusted for a real submission.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable constants -- all first-pass guesses (STARTER values, not tuned).
# ---------------------------------------------------------------------------
_TRAJ_ROLL_WINDOW_SHORT = 11  # rows == ft (MD step is always 1.0 ft, verified in tests)
_TRAJ_ROLL_WINDOW_LONG = 51
_GR_ROLL_WINDOW_SHORT = 11
_GR_ROLL_WINDOW_LONG = 51
_TAIL_FIT_ROWS_SHORT = 50  # known-zone tail window for the linear MD-trend fits
_TAIL_FIT_ROWS_LONG = 200
_MIN_FIT_ROWS = 2  # below this many finite (x, y) pairs, a slope is unreliable
_MIN_FIT_X_STD = 1e-9  # below this x-spread, the slope is undefined (flat/degenerate)
_MIN_GR_FIT_ROWS = 20  # below this many known-zone rows, the GR affine fit is unreliable
_MIN_GR_STD = 1e-6  # below this typewell-GR spread at known depths, the affine fit is degenerate

FEATURE_FAMILIES: dict[str, list[str]] = {
    # Per-row, derived from MD/X/Y/Z only (known across the whole lateral,
    # eval zone included -- see module docstring's leak-boundary section).
    "trajectory": [
        "md_from_anchor",
        "dz_dmd",
        "dz_dmd_roll11",
        "dz_dmd_roll51",
        "dz_dmd_roll51_std",
        "dx_dmd_roll51",
        "dy_dmd_roll51",
        "curvature_roll51",
        "cum_dz_since_anchor",
        "cum_dz_since_eval_start",
        "z_dev_from_tail_linear_extrap",
        "z_dev_from_tail_linear_extrap_long",
        "dist3d_from_anchor",
    ],
    # Per-row GR features, calibrated against the typewell via a per-well
    # affine (gain, offset) fit over the known zone ("heel calibration" --
    # see _fit_affine_gr's docstring for the fit itself).
    "gr_heel_calibration": [
        "gr",
        "gr_missing",
        "gr_diff1",
        "gr_roll11_mean",
        "gr_roll51_mean",
        "gr_roll51_std",
        "gr_resid_vs_carry_last",
        "gr_resid_vs_carry_last_roll11",
        "gr_resid_vs_carry_last_roll51",
    ],
    # Constant-per-well summaries of the known zone, broadcast to every eval
    # row (except linear_trend_extrap_delta, which varies per row -- see
    # its own docstring note below).
    "known_zone_summary": [
        "known_zone_n_rows",
        "known_zone_md_span",
        "tail_slope_z_md",
        "tail_slope_tvt_md",
        "tail_slope_tvt_md_long",
        "known_tvt_std",
        "known_tvt_range",
        "gr_affine_gain",
        "gr_affine_offset",
        "gr_affine_fit_residual_rms",
        "gr_affine_n_fit_rows",
        "linear_trend_extrap_delta",
    ],
}

FEATURE_COLUMNS: list[str] = [
    c for cols in FEATURE_FAMILIES.values() for c in cols
]


# ---------------------------------------------------------------------------
# Target / anchor plumbing (self-contained duplicates of rogii.data, see
# module docstring's "Self-containment" section for why they're duplicated
# rather than imported).
# ---------------------------------------------------------------------------


def eval_mask(h: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the evaluation zone (rows where ``TVT_input`` is NaN)."""
    return h["TVT_input"].isna().to_numpy()


def carry_last(h: pd.DataFrame) -> float:
    """The anchor: last known (non-NaN) ``TVT_input`` value.

    Raises ``ValueError`` if the well has no known-zone row at all -- there
    is then no anchor to define ``drift`` or most features against, matching
    ``rogii.data.last_known_tvt``'s contract.
    """
    known = h["TVT_input"].to_numpy(dtype=float)
    valid = known[~np.isnan(known)]
    if valid.size == 0:
        raise ValueError("well has no known TVT_input anchor")
    return float(valid[-1])


def track_b_target(h: pd.DataFrame) -> np.ndarray:
    """``drift = TVT_true - carry_last`` over the eval zone. TRAIN ONLY.

    Requires ``h["TVT"]`` (ground truth), so this must never be called on a
    test well or anywhere near ``build_trackb_features`` (which never reads
    ``TVT``) -- callers assemble the label separately, exactly like
    ``rogii.features.build_features``'s docstring documents for its own
    P1-GBDT target.
    """
    if "TVT" not in h.columns:
        raise KeyError("track_b_target needs h['TVT'] (train-only ground truth)")
    mask = eval_mask(h)
    return h["TVT"].to_numpy(dtype=float)[mask] - carry_last(h)


# ---------------------------------------------------------------------------
# Small numeric helpers (degenerate-input guards keep everything NaN/inf-free)
# ---------------------------------------------------------------------------


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)`` of ``y ~ x``.

    Falls back to ``(0.0, mean(y))`` (or ``(0.0, 0.0)`` if ``y`` is empty)
    when there are fewer than :data:`_MIN_FIT_ROWS` finite pairs or ``x`` has
    near-zero spread (undefined slope) -- mirrors
    ``rogii.features._robust_slope``'s degenerate-case handling, extended to
    also return an intercept for extrapolation.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0
    if n < _MIN_FIT_ROWS:
        return 0.0, float(y[mask].mean())
    xs, ys = x[mask], y[mask]
    if np.std(xs) < _MIN_FIT_X_STD:
        return 0.0, float(np.mean(ys))
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _tw_gr_lookup(tw: pd.DataFrame):
    """Build a ``TVT -> GR`` interpolator from the typewell (clipped at the ends)."""
    if not {"TVT", "GR"}.issubset(tw.columns) or tw.empty:
        tvt_ref = np.array([], dtype=float)
        gr_ref = np.array([], dtype=float)
    else:
        valid = tw.dropna(subset=["GR"]).sort_values("TVT")
        tvt_ref = valid["TVT"].to_numpy(dtype=float)
        gr_ref = valid["GR"].to_numpy(dtype=float)

    def lookup(query_tvt: np.ndarray) -> np.ndarray:
        query_tvt = np.asarray(query_tvt, dtype=float)
        if tvt_ref.size == 0:
            return np.zeros_like(query_tvt)
        return np.interp(query_tvt, tvt_ref, gr_ref)

    return lookup


def _fit_affine_gr(
    known_tvt_input: np.ndarray, known_gr: np.ndarray, tw: pd.DataFrame
) -> tuple[float, float, float, int]:
    """Fit ``h.GR ~= gain * twGR(TVT_input) + offset`` over the known zone.

    "Heel calibration": aligns the horizontal well's own GR tool response to
    the typewell's GR scale using only known-zone rows (``TVT_input`` and
    ``GR`` both present there in both train and test). Returns
    ``(gain, offset, residual_rms, n_fit_rows)`` -- ``residual_rms`` is the
    fit's own RMS error in GR units, a cheap per-well confidence signal for
    every other GR-calibration feature (a large residual means the well's GR
    tool doesn't track the typewell well, so calibrated-GR features should be
    trusted less for that well; the GBDT can learn this itself given the
    feature). Falls back to the identity transform ``(1.0, 0.0, 0.0, 0)``
    when there are too few finite (TVT_input, GR) pairs or the typewell GR
    sampled at those depths is degenerate (near-constant).
    """
    mask = np.isfinite(known_tvt_input) & np.isfinite(known_gr)
    n_fit = int(mask.sum())
    if n_fit < _MIN_GR_FIT_ROWS:
        return 1.0, 0.0, 0.0, n_fit

    lookup = _tw_gr_lookup(tw)
    gr_tw = lookup(known_tvt_input[mask])
    gr_h = known_gr[mask]

    if np.std(gr_tw) < _MIN_GR_STD:
        return 1.0, 0.0, 0.0, n_fit

    gain, offset = np.polyfit(gr_tw, gr_h, 1)
    resid = gr_h - (gain * gr_tw + offset)
    resid_rms = float(np.sqrt(np.mean(resid**2)))
    return float(gain), float(offset), resid_rms, n_fit


def _roll(a: np.ndarray, window: int, *, stat: str = "mean") -> np.ndarray:
    """Centered rolling mean/std over the FULL array (min_periods=1)."""
    s = pd.Series(a).rolling(window, min_periods=1, center=True)
    return (s.mean() if stat == "mean" else s.std()).to_numpy()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_trackb_features(h: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    """Build the Track B feature matrix for the eval-zone rows of ``h``.

    Returns a ``float32`` DataFrame with columns exactly :data:`FEATURE_COLUMNS`
    (fixed order) and one row per eval-zone row, in ``eval_mask(h)`` order.
    Never reads ``h["TVT"]`` or any train-only formation column. Raises
    ``ValueError`` if the well has no known-zone anchor (mirrors
    :func:`carry_last`'s contract -- almost every feature here is
    anchor-relative, so there is nothing meaningful to compute without one).
    """
    mask = eval_mask(h)
    idx = np.where(mask)[0]
    n_eval = int(idx.size)
    known_mask = ~mask
    n_known = int(known_mask.sum())
    if n_known == 0:
        raise ValueError("well has no known TVT_input anchor")

    md = h["MD"].to_numpy(dtype=float)
    x = h["X"].to_numpy(dtype=float)
    y = h["Y"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    gr = h["GR"].to_numpy(dtype=float) if "GR" in h.columns else np.full(len(h), np.nan)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    known_md = md[known_mask]
    known_x = x[known_mask]
    known_y = y[known_mask]
    known_z = z[known_mask]
    known_gr = gr[known_mask]
    known_tvt = tvt_input[known_mask]

    anchor_idx = int(np.where(known_mask)[0][-1])
    anchor_md = float(known_md[-1])
    anchor_x = float(known_x[-1])
    anchor_y = float(known_y[-1])
    anchor_z = float(known_z[-1])
    anchor_tvt = float(known_tvt[-1])  # == carry_last(h)
    eval_start_idx = int(idx[0]) if n_eval > 0 else anchor_idx

    # ---- trajectory family --------------------------------------------------
    dmd = np.diff(md, prepend=md[0] if len(md) else 0.0)
    dmd_safe = np.where(np.abs(dmd) < 1e-9, np.nan, dmd)
    dz_dmd = np.diff(z, prepend=z[0] if len(z) else 0.0) / dmd_safe
    dx_dmd = np.diff(x, prepend=x[0] if len(x) else 0.0) / dmd_safe
    dy_dmd = np.diff(y, prepend=y[0] if len(y) else 0.0) / dmd_safe
    # curvature: rate of change of the inclination dz_dmd (2nd derivative of Z wrt MD)
    d2z_dmd2 = np.diff(np.nan_to_num(dz_dmd, nan=0.0), prepend=0.0) / dmd_safe

    dz_dmd_roll11 = _roll(dz_dmd, _TRAJ_ROLL_WINDOW_SHORT)
    dz_dmd_roll51 = _roll(dz_dmd, _TRAJ_ROLL_WINDOW_LONG)
    dz_dmd_roll51_std = _roll(dz_dmd, _TRAJ_ROLL_WINDOW_LONG, stat="std")
    dx_dmd_roll51 = _roll(dx_dmd, _TRAJ_ROLL_WINDOW_LONG)
    dy_dmd_roll51 = _roll(dy_dmd, _TRAJ_ROLL_WINDOW_LONG)
    curvature_roll51 = _roll(d2z_dmd2, _TRAJ_ROLL_WINDOW_LONG)

    cum_dz_since_anchor = z - anchor_z
    cum_dz_since_eval_start = z - z[eval_start_idx]

    tail_short = min(_TAIL_FIT_ROWS_SHORT, n_known)
    tail_long = min(_TAIL_FIT_ROWS_LONG, n_known)
    slope_z_short, intercept_z_short = _fit_line(known_md[-tail_short:], known_z[-tail_short:])
    slope_z_long, intercept_z_long = _fit_line(known_md[-tail_long:], known_z[-tail_long:])
    z_dev_from_tail_linear_extrap = z - (intercept_z_short + slope_z_short * md)
    z_dev_from_tail_linear_extrap_long = z - (intercept_z_long + slope_z_long * md)

    dist3d_from_anchor = np.sqrt(
        (x - anchor_x) ** 2 + (y - anchor_y) ** 2 + (z - anchor_z) ** 2
    )

    # ---- known-zone summary family (well-constant, computed once) ----------
    known_zone_n_rows = float(n_known)
    known_zone_md_span = float(known_md.max() - known_md.min()) if n_known > 0 else 0.0
    tail_slope_tvt_short, _ = _fit_line(known_md[-tail_short:], known_tvt[-tail_short:])
    tail_slope_tvt_long, _ = _fit_line(known_md[-tail_long:], known_tvt[-tail_long:])
    known_tvt_valid = known_tvt[np.isfinite(known_tvt)]
    known_tvt_std = float(known_tvt_valid.std()) if known_tvt_valid.size >= 2 else 0.0
    known_tvt_range = (
        float(known_tvt_valid.max() - known_tvt_valid.min()) if known_tvt_valid.size else 0.0
    )
    gain, offset, gr_fit_resid_rms, n_gr_fit_rows = _fit_affine_gr(known_tvt, known_gr, tw)

    # linear extrapolation of the known-zone TVT trend, as a candidate drift
    # prediction: how far TVT would move from the anchor if the last-known
    # dip rate (tail_slope_tvt_short) simply continued in a straight line.
    # NOTE this is the one "known_zone_summary" entry that varies per row
    # (it multiplies the well-constant slope by each row's own MD distance).
    linear_trend_extrap_delta = tail_slope_tvt_short * (md - anchor_md)

    # ---- GR heel-calibration family -----------------------------------------
    gr_series_missing = (~np.isfinite(gr)).astype(np.float64)
    gr_filled = pd.Series(gr).ffill().bfill().fillna(0.0).to_numpy()
    gr_diff1 = pd.Series(gr).diff(1).to_numpy()
    gr_roll11_mean = _roll(gr, _GR_ROLL_WINDOW_SHORT)
    gr_roll51_mean = _roll(gr, _GR_ROLL_WINDOW_LONG)
    gr_roll51_std = _roll(gr, _GR_ROLL_WINDOW_LONG, stat="std")

    lookup = _tw_gr_lookup(tw)
    tw_gr_at_carry_last = float(lookup(np.array([anchor_tvt]))[0])
    tw_gr_at_carry_last_calibrated = gain * tw_gr_at_carry_last + offset
    gr_resid_vs_carry_last = gr_filled - tw_gr_at_carry_last_calibrated
    gr_resid_vs_carry_last_roll11 = _roll(gr_resid_vs_carry_last, _GR_ROLL_WINDOW_SHORT)
    gr_resid_vs_carry_last_roll51 = _roll(gr_resid_vs_carry_last, _GR_ROLL_WINDOW_LONG)

    feats: dict[str, np.ndarray] = {
        # trajectory
        "md_from_anchor": md[idx] - anchor_md,
        "dz_dmd": dz_dmd[idx],
        "dz_dmd_roll11": dz_dmd_roll11[idx],
        "dz_dmd_roll51": dz_dmd_roll51[idx],
        "dz_dmd_roll51_std": dz_dmd_roll51_std[idx],
        "dx_dmd_roll51": dx_dmd_roll51[idx],
        "dy_dmd_roll51": dy_dmd_roll51[idx],
        "curvature_roll51": curvature_roll51[idx],
        "cum_dz_since_anchor": cum_dz_since_anchor[idx],
        "cum_dz_since_eval_start": cum_dz_since_eval_start[idx],
        "z_dev_from_tail_linear_extrap": z_dev_from_tail_linear_extrap[idx],
        "z_dev_from_tail_linear_extrap_long": z_dev_from_tail_linear_extrap_long[idx],
        "dist3d_from_anchor": dist3d_from_anchor[idx],
        # gr heel calibration
        "gr": gr[idx],
        "gr_missing": gr_series_missing[idx],
        "gr_diff1": gr_diff1[idx],
        "gr_roll11_mean": gr_roll11_mean[idx],
        "gr_roll51_mean": gr_roll51_mean[idx],
        "gr_roll51_std": gr_roll51_std[idx],
        "gr_resid_vs_carry_last": gr_resid_vs_carry_last[idx],
        "gr_resid_vs_carry_last_roll11": gr_resid_vs_carry_last_roll11[idx],
        "gr_resid_vs_carry_last_roll51": gr_resid_vs_carry_last_roll51[idx],
        # known-zone summary (well-constant, broadcast)
        "known_zone_n_rows": np.full(n_eval, known_zone_n_rows),
        "known_zone_md_span": np.full(n_eval, known_zone_md_span),
        "tail_slope_z_md": np.full(n_eval, slope_z_short),
        "tail_slope_tvt_md": np.full(n_eval, tail_slope_tvt_short),
        "tail_slope_tvt_md_long": np.full(n_eval, tail_slope_tvt_long),
        "known_tvt_std": np.full(n_eval, known_tvt_std),
        "known_tvt_range": np.full(n_eval, known_tvt_range),
        "gr_affine_gain": np.full(n_eval, gain),
        "gr_affine_offset": np.full(n_eval, offset),
        "gr_affine_fit_residual_rms": np.full(n_eval, gr_fit_resid_rms),
        "gr_affine_n_fit_rows": np.full(n_eval, float(n_gr_fit_rows)),
        "linear_trend_extrap_delta": linear_trend_extrap_delta[idx],
    }

    out = pd.DataFrame(feats, columns=FEATURE_COLUMNS)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)
