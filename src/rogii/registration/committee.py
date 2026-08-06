"""Near-strike committee gate for the spatial-prior predictor.

Source / inspiration
---------------------
The gate idea ("committee") is adapted from the public Kaggle notebook
``connortynan/rogii-k16-spline-kernel-knn-adaptive-kappa``
(``connortynan/k16_spline_kernel_knn`` on Kaggle, functions
``committee_inputs`` / ``theta_loc_at`` / ``segment_geometry``). That
notebook assigns each well's lateral a *global* regional dip azimuth
(``THETA0``, a single fixed constant fit once across the whole field) and
converts a donor-field dip-rate estimate into a per-segment TVT drift via
``rate * cos(heading - THETA0)``. When a segment's heading runs close to
*parallel* to the regional strike (``|cos(heading - THETA0)| < GATE =
0.35``), that projection is close to singular -- a small error in the
assumed *global* direction swings the along-heading drift estimate wildly --
so the notebook substitutes a *locally* fit dip azimuth (a weighted plane
fit of nearby formation-top samples, excluding the well's own) for that
segment instead of trusting the fixed global one, but only when the local
estimate is itself plausible (its direction does not deviate from the
global consensus by more than ``ROT_MAX = 60`` degrees -- otherwise the
local fit is presumed too noisy/data-starved to trust, and the segment is
left alone). On the notebook's own worst-well audit this repaired a 46 ft
well down to 13 ft.

This module is an independent reimplementation of that *gate mechanism*
against this repository's own spatial-prior data structures
(:class:`~rogii.spatial.SurfaceBank`, :func:`~rogii.spatial.predict_spatial`)
-- it does not reuse K16's basis-spline TVT model, its per-segment kernel
field ``F_raw``/``F_sm``, or any of its code verbatim. Concretely:

* :func:`strike_alignment` computes, per K16-style contiguous segment of the
  evaluation zone, ``|cos(heading_azimuth - regional_dip_azimuth)|`` where
  the regional dip azimuth is fit fresh from the bank's own formation-top
  samples (leave-self-out), rather than hardcoded as a constant. This is
  the ``|proj|`` gate signal.
* :func:`committee_correction` gates on that alignment (``< threshold``),
  further requires the *locally* fit dip azimuth near that segment to not
  differ from the regional one by more than ``min_rot_deg`` (the K16
  ``ROT_MAX`` plausibility guard), and for segments that pass both checks,
  replaces the caller-supplied ``base_path`` with
  ``predict_spatial(..., method="plane")`` -- this repository's own
  local-plane-fit spatial interpolator, which (unlike the caller's
  presumably IDW-based ``base_path``) resolves the true local gradient
  direction directly per query point instead of projecting a single fixed
  heading onto a fixed regional direction. Segments that do not gate keep
  ``base_path`` untouched, and if the gate never fires anywhere in the
  well, ``committee_correction`` returns ``base_path`` unchanged (identity)
  without paying for the extra plane-fit computation at all.

Leak-safety notes
------------------
* Both public functions take ``exclude_well`` and use it consistently: the
  regional (global) dip-azimuth fit, the local dip-azimuth fit, and the
  substituted ``predict_spatial(method="plane")`` call all exclude
  ``exclude_well``'s own bank samples (leave-self-out), matching
  ``rogii.spatial``'s own contract.
* Neither function reads a well's own formation-top columns or ``TVT``
  column -- only ``X``, ``Y``, and (via ``eval_mask``/``predict_spatial``)
  ``TVT_input``, all present in test.
* Like every other tracker/predictor in this package, the public entry
  points never raise: any failure falls back to the caller's own
  ``base_path`` (or, for :func:`strike_alignment`, to an "everything is
  aligned" (no-gate) array) so a bad well cannot abort a CV/submission run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data as D
from ..spatial import SurfaceBank, predict_spatial

# K16's GATE / ROT_MAX (see module docstring); reused as this module's
# defaults since they are the empirically validated values, not refit here.
_DEFAULT_THRESHOLD = 0.35
_DEFAULT_MIN_ROT_DEG = 60.0

# K16 segments each well's evaluation zone into 16 contiguous chunks for the
# heading/gate geometry; matched here for the same reason.
_DEFAULT_N_SEGMENTS = 16

# Gaussian kernel bandwidth (ft) for the local dip-azimuth plane fit, and the
# search radius as a multiple of it -- both match K16's ``theta_loc_at``
# defaults (``h=1500.0``, radius ``4*h``).
_DEFAULT_BANDWIDTH_FT = 1500.0
_DEFAULT_RADIUS_MULT = 4.0

# Below this many (leave-self-out) neighbors within the search radius, the
# local plane fit is not trusted; K16 uses the same threshold (``m.sum() <
# 30``) and falls back to the global/regional direction in that case.
_DEFAULT_MIN_LOCAL_POINTS = 30

_DEFAULT_K = 10


def _pick_formation(bank: SurfaceBank, formation: str | None) -> str | None:
    """Formation to build the direction field from: caller override, else ANCC, else first."""
    if formation is not None and formation in bank.formations:
        return formation
    if "ANCC" in bank.formations:
        return "ANCC"
    return bank.formations[0] if bank.formations else None


def _segment_geometry(
    x: np.ndarray, y: np.ndarray, n_segments: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``n = len(x)`` rows into up to ``n_segments`` contiguous chunks.

    Returns ``(seg_id, az, mid)``: ``seg_id`` (shape ``(n,)``) maps each row
    to its segment; ``az`` (shape ``(n_seg,)``) is each segment's heading
    azimuth in radians (``arctan2`` of the displacement between its first and
    last row); ``mid`` (shape ``(n_seg, 2)``) is each segment's ``(X, Y)``
    midpoint. A segment whose first and last row coincide (degenerate, e.g. a
    single-row segment) gets azimuth ``0.0``.
    """
    n = x.shape[0]
    n_seg = max(1, min(n_segments, n))
    edges = np.linspace(0, n, n_seg + 1)
    seg_id = np.clip(np.searchsorted(edges[1:], np.arange(n), side="left"), 0, n_seg - 1)

    az = np.zeros(n_seg)
    mid = np.empty((n_seg, 2))
    for j in range(n_seg):
        rows = np.flatnonzero(seg_id == j)
        i0, i1 = int(rows[0]), int(rows[-1])
        dx = x[i1] - x[i0]
        dy = y[i1] - y[i0]
        if dx != 0.0 or dy != 0.0:
            az[j] = np.arctan2(dy, dx)
        mid[j] = ((x[i0] + x[i1]) / 2.0, (y[i0] + y[i1]) / 2.0)
    return seg_id, az, mid


def _global_dip_azimuth(bank: SurfaceBank, formation: str, exclude_code: int) -> float:
    """Regional dip-azimuth: ``atan2`` of an unweighted plane fit over every non-self sample."""
    tree = bank.trees[formation]
    xy = np.asarray(tree.data, dtype=float)
    depths = bank.depths[formation]
    codes = bank.well_codes[formation]

    keep = codes != exclude_code
    if int(np.sum(keep)) < 3:
        return 0.0

    x, y, d = xy[keep, 0], xy[keep, 1], depths[keep]
    xm, ym = x.mean(), y.mean()
    design = np.column_stack([np.ones_like(x), x - xm, y - ym])
    try:
        beta, *_ = np.linalg.lstsq(design, d, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    if not (np.isfinite(beta[1]) and np.isfinite(beta[2])):
        return 0.0
    return float(np.arctan2(beta[2], beta[1]))


def _local_dip_azimuth(
    bank: SurfaceBank,
    formation: str,
    mids: np.ndarray,
    exclude_code: int,
    fallback_az: float,
    bandwidth: float,
    radius_mult: float,
    min_points: int,
) -> np.ndarray:
    """Per-``mids`` row weighted local-plane dip azimuth (K16's ``theta_loc_at``).

    For each query point, fits a Gaussian-weighted plane to bank samples
    (formation ``formation``, leave-self-out) within ``radius_mult *
    bandwidth`` of the query. Falls back to ``fallback_az`` where fewer than
    ``min_points`` such neighbors exist or the weighted normal equations are
    singular.
    """
    tree = bank.trees[formation]
    xy = np.asarray(tree.data, dtype=float)
    depths = bank.depths[formation]
    codes = bank.well_codes[formation]
    radius = radius_mult * bandwidth

    out = np.full(mids.shape[0], fallback_az, dtype=float)
    for q in range(mids.shape[0]):
        idx = tree.query_ball_point(mids[q], r=radius)
        if not idx:
            continue
        idx = np.asarray(idx, dtype=int)
        idx = idx[codes[idx] != exclude_code]
        if idx.size < min_points:
            continue

        dx = xy[idx, 0] - mids[q, 0]
        dy = xy[idx, 1] - mids[q, 1]
        d2 = dx * dx + dy * dy
        w = np.exp(np.maximum(-d2 / (2.0 * bandwidth * bandwidth), -700.0))
        design = np.column_stack([np.ones_like(dx), dx, dy])
        a_mat = (design * w[:, None]).T @ design
        b_vec = (design * w[:, None]).T @ depths[idx]
        try:
            beta = np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(beta[1]) and np.isfinite(beta[2]):
            out[q] = float(np.arctan2(beta[2], beta[1]))
    return out


def _angular_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest-angle |a - b| in degrees, both given in radians (handles wraparound)."""
    d = a - b
    return np.degrees(np.abs(np.arctan2(np.sin(d), np.cos(d))))


def strike_alignment(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    n_segments: int = _DEFAULT_N_SEGMENTS,
    formation: str | None = None,
) -> np.ndarray:
    """Per-eval-row heading/strike alignment, K16's ``|proj|`` gate signal.

    The evaluation zone is split into up to ``n_segments`` contiguous
    segments (see :func:`_segment_geometry`); each segment's heading azimuth
    is compared against the bank's regional (leave-self-out) dip azimuth via
    ``|cos(heading - regional_dip_azimuth)|``. A value near ``1`` means the
    well is heading roughly along the dip direction (safe: kNN spatial
    interpolation has good along-heading resolving power there); a value
    near ``0`` means the well is heading roughly along strike (risky: see
    the module docstring). The result is broadcast back to per-row
    resolution, aligned to ``data.eval_mask(h)`` order.

    Never raises: on any failure returns an all-``1.0`` ("everything
    aligned", i.e. no gate should fire) array of the correct length where
    computable, else an empty array.
    """
    try:
        eval_idx = np.flatnonzero(D.eval_mask(h))
        n = eval_idx.size
        if n == 0:
            return np.zeros(0, dtype=float)

        formation_used = _pick_formation(bank, formation)
        if formation_used is None:
            return np.ones(n, dtype=float)

        x = h["X"].to_numpy(dtype=float)[eval_idx]
        y = h["Y"].to_numpy(dtype=float)[eval_idx]
        seg_id, az, _mid = _segment_geometry(x, y, n_segments)

        exclude_code = bank.well_to_code.get(exclude_well, -1)
        g_dir = _global_dip_azimuth(bank, formation_used, exclude_code)

        alignment_seg = np.abs(np.cos(az - g_dir))
        return alignment_seg[seg_id]
    except Exception:
        try:
            return np.ones(int(D.eval_mask(h).sum()), dtype=float)
        except Exception:
            return np.zeros(0, dtype=float)


def committee_correction(
    h: pd.DataFrame,
    bank: SurfaceBank,
    base_path: np.ndarray,
    exclude_well: str,
    threshold: float = _DEFAULT_THRESHOLD,
    min_rot_deg: float = _DEFAULT_MIN_ROT_DEG,
    n_segments: int = _DEFAULT_N_SEGMENTS,
    k: int = _DEFAULT_K,
    formation: str | None = None,
    bandwidth: float = _DEFAULT_BANDWIDTH_FT,
    radius_mult: float = _DEFAULT_RADIUS_MULT,
    min_local_points: int = _DEFAULT_MIN_LOCAL_POINTS,
) -> np.ndarray:
    """Near-strike committee gate: substitute a local-plane fit on gated segments only.

    For each segment (see :func:`_segment_geometry`) whose alignment (see
    :func:`strike_alignment`) is below ``threshold`` *and* whose locally fit
    dip azimuth (:func:`_local_dip_azimuth`, leave-self-out) does not differ
    from the regional one by more than ``min_rot_deg`` (the K16 ``ROT_MAX``
    plausibility guard -- an untrustworthy/noisy local fit is not used to
    override the caller's prediction), every row in that segment is replaced
    with ``predict_spatial(h, bank, exclude_well, k=k,
    method="plane").tvt`` at that row. Rows in non-gated segments (or gated
    segments whose local fit fails the rotation guard) keep ``base_path``.

    If the gate never fires anywhere in the well, ``base_path`` is returned
    unchanged (identity) without computing the plane-fit fallback at all.

    Never raises: on any failure returns ``base_path`` (or, if that itself
    is unusable, a zero array of the eval-zone's length, or an empty array).
    """
    try:
        base = np.asarray(base_path, dtype=float)
        eval_idx = np.flatnonzero(D.eval_mask(h))
        n = eval_idx.size
        if n == 0 or base.shape[0] != n:
            return base

        formation_used = _pick_formation(bank, formation)
        if formation_used is None:
            return base

        x = h["X"].to_numpy(dtype=float)[eval_idx]
        y = h["Y"].to_numpy(dtype=float)[eval_idx]
        seg_id, az, mid = _segment_geometry(x, y, n_segments)

        exclude_code = bank.well_to_code.get(exclude_well, -1)
        g_dir = _global_dip_azimuth(bank, formation_used, exclude_code)

        alignment_seg = np.abs(np.cos(az - g_dir))
        gated_seg = alignment_seg < threshold
        if not np.any(gated_seg):
            return base

        local_az = _local_dip_azimuth(
            bank,
            formation_used,
            mid,
            exclude_code,
            fallback_az=g_dir,
            bandwidth=bandwidth,
            radius_mult=radius_mult,
            min_points=min_local_points,
        )
        rot_deg = _angular_diff_deg(local_az, g_dir)
        substitute_seg = gated_seg & (rot_deg < min_rot_deg)
        if not np.any(substitute_seg):
            return base

        plane_result = predict_spatial(h, bank, exclude_well=exclude_well, k=k, method="plane")
        plane_pred = np.asarray(plane_result.tvt, dtype=float)
        if plane_pred.shape[0] != n:
            return base

        row_mask = substitute_seg[seg_id]
        return np.where(row_mask, plane_pred, base)
    except Exception:
        try:
            return np.asarray(base_path, dtype=float)
        except Exception:
            try:
                return np.zeros(int(D.eval_mask(h).sum()), dtype=float)
            except Exception:
                return np.zeros(0, dtype=float)
