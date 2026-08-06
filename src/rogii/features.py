"""Layer2 tabular features for the residual GBDT (Issue P1).

``build_features(h, tw)`` builds a feature matrix aligned to the eval-zone
rows of a horizontal-well DataFrame (``data.eval_mask(h)`` order), for use as
inputs to a GBDT that predicts the residual ``U = TVT - anchor``.

Leak boundary (see ``docs/design.md`` sec 6 / ``CLAUDE.md``): only columns
that exist in the **test** schema are ever read from ``h`` -- ``MD, X, Y, Z,
GR, TVT_input``. The ground-truth ``TVT`` column and the train-only formation
columns (``ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA``) are never read here; the
label (``TVT``) is only ever assembled by the *caller* (e.g.
``scripts/run_p1_gbdt_cv.py``), never inside this module. Known-segment
statistics come exclusively from the non-NaN prefix of ``TVT_input``, which
is available in both train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import typewell as TW

# GR rolling-window sizes (rows == ft, since MD steps are always 1ft).
_GR_ROLL_WINDOWS = (11, 51, 151)
# Trajectory-derivative rolling window and known-segment tail window (host
# hint: "the last 50 points before prediction start are the most important").
_TRAJ_ROLL_WINDOW = 51
_KNOWN_TAIL_LONG = 200
_KNOWN_TAIL_SHORT = 50
# Below this many valid (x, y) pairs, a least-squares slope is unreliable.
_MIN_SLOPE_ROWS = 2
# Below this x-spread, the slope is undefined (degenerate/flat); fall back to 0.
_MIN_SLOPE_X_STD = 1e-9


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y ~ x``; ``0.0`` when degenerate.

    Degenerate cases: fewer than 2 finite (x, y) pairs, or ``x`` has
    near-zero spread (undefined gain).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < _MIN_SLOPE_ROWS:
        return 0.0
    xs, ys = x[mask], y[mask]
    if np.std(xs) < _MIN_SLOPE_X_STD:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def _stat_block(values: np.ndarray) -> dict[str, float]:
    """min/max/range/mean/std over finite ``values``; all-``0.0`` when degenerate."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"min": 0.0, "max": 0.0, "range": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "range": float(valid.max() - valid.min()),
        "mean": float(valid.mean()),
        "std": float(valid.std()) if valid.size >= 2 else 0.0,
    }


def _typewell_gr_at(tw: pd.DataFrame, tvt_query: float) -> float:
    """``twGR(tvt_query)``, or ``0.0`` when the typewell has no usable GR."""
    if not {"TVT", "GR"}.issubset(tw.columns) or tw.empty or not np.isfinite(tvt_query):
        return 0.0
    lookup = TW.tw_gr_lookup(tw)
    return float(lookup(np.array([tvt_query]))[0])


def build_features(h: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    """Build the Layer2 feature matrix for the eval-zone rows of ``h``.

    Returns a ``float32`` DataFrame with one row per eval-zone row (same
    order as ``data.eval_mask(h)``). Never reads ``h["TVT"]`` or any
    train-only formation column.
    """
    mask = D.eval_mask(h)
    idx = np.where(mask)[0]
    n_eval = int(idx.size)
    known_mask = ~mask
    n_known = int(known_mask.sum())

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

    # anchor == the last known TVT_input value (data.last_known_tvt, without
    # raising when a well has no known rows at all).
    anchor_tvt = float(known_tvt[-1]) if n_known > 0 else float("nan")
    ps_md = float(known_md[-1]) if n_known > 0 else float(md[0]) if len(md) else 0.0
    ps_x = float(known_x[-1]) if n_known > 0 else float(x[0]) if len(x) else 0.0
    ps_y = float(known_y[-1]) if n_known > 0 else float(y[0]) if len(y) else 0.0
    ps_z = float(known_z[-1]) if n_known > 0 else float(z[0]) if len(z) else 0.0
    ps_gr = float(known_gr[-1]) if n_known > 0 and np.isfinite(known_gr[-1]) else float("nan")

    # --- known-prefix TVT_input-vs-MD/Z slopes & stats (§6 "既知セグメント統計") ---
    slope_md_all = _robust_slope(known_md, known_tvt)
    tail200 = min(_KNOWN_TAIL_LONG, n_known)
    slope_md_last200 = _robust_slope(known_md[-tail200:], known_tvt[-tail200:]) if tail200 else 0.0
    tail50 = min(_KNOWN_TAIL_SHORT, n_known)
    slope_md_last50 = _robust_slope(known_md[-tail50:], known_tvt[-tail50:]) if tail50 else 0.0
    slope_z_all = _robust_slope(known_z, known_tvt)
    tvt_stats = _stat_block(known_tvt)

    # --- trajectory derivatives dZ/dMD, dX/dMD, dY/dMD (§6 "軌跡") ---------------
    dmd = np.diff(md, prepend=md[0] if len(md) else 0.0)
    dmd_safe = np.where(np.abs(dmd) < 1e-9, np.nan, dmd)
    dz_dmd = np.diff(z, prepend=z[0] if len(z) else 0.0) / dmd_safe
    dx_dmd = np.diff(x, prepend=x[0] if len(x) else 0.0) / dmd_safe
    dy_dmd = np.diff(y, prepend=y[0] if len(y) else 0.0) / dmd_safe

    def _roll_mean(a: np.ndarray) -> np.ndarray:
        return (
            pd.Series(a)
            .rolling(_TRAJ_ROLL_WINDOW, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )

    dz_roll = _roll_mean(dz_dmd)
    dx_roll = _roll_mean(dx_dmd)
    dy_roll = _roll_mean(dy_dmd)

    known_tail_n = min(_TRAJ_ROLL_WINDOW, n_known)
    if known_tail_n > 0:
        dz_dmd_known_tail_mean = float(np.nanmean(dz_dmd[known_mask][-known_tail_n:]))
        dx_dmd_known_tail_mean = float(np.nanmean(dx_dmd[known_mask][-known_tail_n:]))
        dy_dmd_known_tail_mean = float(np.nanmean(dy_dmd[known_mask][-known_tail_n:]))
    else:
        dz_dmd_known_tail_mean = 0.0
        dx_dmd_known_tail_mean = 0.0
        dy_dmd_known_tail_mean = 0.0
    dz_dmd_known_tail_mean = dz_dmd_known_tail_mean if np.isfinite(dz_dmd_known_tail_mean) else 0.0
    dx_dmd_known_tail_mean = dx_dmd_known_tail_mean if np.isfinite(dx_dmd_known_tail_mean) else 0.0
    dy_dmd_known_tail_mean = dy_dmd_known_tail_mean if np.isfinite(dy_dmd_known_tail_mean) else 0.0

    # --- GR rolling stats (§6 "GR") ----------------------------------------------
    gr_series = pd.Series(gr)
    roll_feats: dict[str, np.ndarray] = {}
    for w in _GR_ROLL_WINDOWS:
        roll = gr_series.rolling(w, min_periods=1, center=True)
        roll_feats[f"gr_roll_mean_{w}"] = roll.mean().to_numpy()
        roll_feats[f"gr_roll_std_{w}"] = roll.std().to_numpy()
    gr_diff1 = gr_series.diff(1).to_numpy()
    gr_diff10 = gr_series.diff(10).to_numpy()

    # --- typewell GR at anchor, raw and affine-calibrated ------------------------
    twgr_anchor = _typewell_gr_at(tw, anchor_tvt)
    if "GR" in h.columns:
        gain, offset = TW.fit_affine_gr(h, tw)
    else:
        gain, offset = 1.0, 0.0
    twgr_anchor_calibrated = float(TW.apply_affine(np.array([twgr_anchor]), gain, offset)[0])
    known_tail_gr_minus_twgr_anchor = ps_gr - twgr_anchor if np.isfinite(ps_gr) else 0.0

    # --- per-eval-row position features (§6 "位置") -------------------------------
    n_rows_total = len(h)
    row_frac = idx / max(n_rows_total - 1, 1)
    md_from_ps = md[idx] - ps_md
    dx_ps = x[idx] - ps_x
    dy_ps = y[idx] - ps_y
    dz_ps = z[idx] - ps_z
    dist3d_ps = np.sqrt(dx_ps**2 + dy_ps**2 + dz_ps**2)

    gr_eval = gr[idx]
    gr_missing = (~np.isfinite(gr_eval)).astype(np.float64)
    gr_minus_twgr_anchor_raw = gr_eval - twgr_anchor
    gr_minus_twgr_anchor_calibrated = gr_eval - twgr_anchor_calibrated

    feats: dict[str, np.ndarray] = {
        # position
        "md_from_ps": md_from_ps,
        "row_frac": row_frac,
        "dx_from_ps": dx_ps,
        "dy_from_ps": dy_ps,
        "dz_from_ps": dz_ps,
        "dist3d_from_ps": dist3d_ps,
        # trajectory / incidence angle
        "dz_dmd_roll51": dz_roll[idx],
        "dx_dmd_roll51": dx_roll[idx],
        "dy_dmd_roll51": dy_roll[idx],
        "dz_dmd_known_tail_mean": np.full(n_eval, dz_dmd_known_tail_mean),
        "dx_dmd_known_tail_mean": np.full(n_eval, dx_dmd_known_tail_mean),
        "dy_dmd_known_tail_mean": np.full(n_eval, dy_dmd_known_tail_mean),
        # GR
        "gr": gr_eval,
        "gr_missing": gr_missing,
        "gr_diff_1": gr_diff1[idx],
        "gr_diff_10": gr_diff10[idx],
        "gr_minus_twgr_anchor_raw": gr_minus_twgr_anchor_raw,
        "gr_minus_twgr_anchor_calibrated": gr_minus_twgr_anchor_calibrated,
        # known-prefix stats (constant across a well's eval rows)
        "slope_md_all": np.full(n_eval, slope_md_all),
        "slope_md_last200": np.full(n_eval, slope_md_last200),
        "slope_md_last50": np.full(n_eval, slope_md_last50),
        "slope_z_all": np.full(n_eval, slope_z_all),
        "known_tvt_min": np.full(n_eval, tvt_stats["min"]),
        "known_tvt_max": np.full(n_eval, tvt_stats["max"]),
        "known_tvt_range": np.full(n_eval, tvt_stats["range"]),
        "known_tvt_mean": np.full(n_eval, tvt_stats["mean"]),
        "known_tvt_std": np.full(n_eval, tvt_stats["std"]),
        "n_known_rows": np.full(n_eval, float(n_known)),
        # anchor info (§6 "anchor 情報")
        "anchor_tvt": np.full(n_eval, anchor_tvt if np.isfinite(anchor_tvt) else 0.0),
        "known_tail_gr_minus_twgr_anchor": np.full(n_eval, known_tail_gr_minus_twgr_anchor),
    }
    for name, values in roll_feats.items():
        feats[name] = values[idx]

    out = pd.DataFrame(feats)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)
