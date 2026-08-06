"""Multi-scale normalized cross-correlation (NCC) GR registration tracker.

Source / inspiration
---------------------
The "multi-scale NCC with softmax-weighted scale ensembling" idea (window
half-widths ~8/15/25 ft, temperature-softmax combination across scales) is
adapted from the public Kaggle notebook ``lightningv08/lb-7-776-rogii-ridge-sp``
(function ``multi_scale_ncc`` in ``lightningv08/lb-7-776-rogii-ridge-sp`` on Kaggle),
which the notebook ``pixiux/rogii-dual-pipeline-blend`` reuses verbatim. Those
notebooks state no open-source license, so no code from them is reproduced here:
the implementation below is our own and only the algorithm design is adapted.

That reference implementation matches windows of the horizontal GR log against
a *dictionary of known-zone GR windows* (whose TVT is already known via
``TVT_input``), all in one batched, non-sequential pass. This module instead
implements a **sequential, anchor-seeded tracker**: starting from
``last_known_tvt``, it walks the evaluation zone one row at a time, matching a
horizontal GR window against *typewell GR segments* directly (not the
known-zone dictionary), searching only a small neighborhood around the
previous row's estimate and capping how far the estimate may move per row.
This is an independent reimplementation built around the same scale/softmax
pattern, not a copy of the reference's matching target or control flow.

Honest empirical note: on real train wells, short-window normalized
cross-correlation against the typewell turns out to be a weak, often
near-ambiguous localization signal (thin-bed cyclicity and lateral facies
variation between the typewell and the lateral make many nearby candidate
depths look similarly plausible) -- see ``scripts/run_ncc_cv.py`` for the
measured pooled RMSE against ``carry_last`` across every train well. This
module implements the algorithm faithfully; it does not claim the raw
signal beats the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import data as D

_EPS = 1e-6
_SOFTMAX_TEMPERATURE = 3.0
_SMOOTH_WINDOW = 5

# Typewell-side window resolution, in TVT feet per grid sample.
#
# The horizontal window for scale `hw` spans `2*hw` *MD* feet (native 1 ft/row
# spacing). If the typewell window spanned the same `2*hw` feet of *TVT*, the
# two windows would represent wildly different physical scales: empirically
# (measured on the known zone's tail, right before the anchor -- see
# scripts/run_ncc_cv.py / project notes) the true local TVT step size is
# usually tiny (median ~0.01-0.04 ft per MD row near the anchor), because a
# near-horizontal well covers many MD feet while barely moving in TVT. A
# typewell window built at 1 ft/sample would therefore span a stratigraphic
# range roughly an order of magnitude wider than what the corresponding MD
# window could physically have sampled, making the two windows' *shapes*
# incomparable (confirmed by a synthetic-data regression: with a 1 ft/sample
# typewell grid, the tracker consistently walked the *wrong* direction).
# `_GRID_STEP_FT` shrinks the typewell grid instead, so a window of `2*hw+1`
# samples spans only `2*hw*_GRID_STEP_FT` TVT feet -- a scale calibrated to
# that near-flat local regime while keeping both windows the same length
# (required for the dot-product correlation).
_GRID_STEP_FT = 0.2

# Typewell candidate grid is built only within `anchor +/- _GRID_RANGE_FT`
# (clipped to the typewell's own TVT extent), not the whole typewell -- this
# keeps the per-well correlation matrices small. The margin is generous
# relative to the drift actually observed in this dataset (median ~11 ft, max
# ~104 ft per well), so it should not bind in practice; if the tracker's
# cumulative move ever reaches this margin it is clipped (degrades gracefully
# rather than raising or wrapping around).
_GRID_RANGE_FT = 200.0


@dataclass
class NCCResult:
    """Per-row TVT estimate and match confidence for the evaluation zone.

    Both arrays are aligned to ``data.eval_mask(h)`` order (length = number of
    evaluation-zone rows). ``confidence`` is a softmax-weighted correlation
    score clipped to ``[0, 1]``; it is exactly ``0.0`` for rows where the
    update was skipped (missing GR) and for the exception-fallback path.
    """

    tvt: np.ndarray
    confidence: np.ndarray


def _rolling_fill(values: np.ndarray, window: int = _SMOOTH_WINDOW) -> np.ndarray:
    """NaN-tolerant rolling-mean smoothing, fully filling residual NaN gaps.

    A centered rolling mean with ``min_periods=1`` averages over whichever
    samples in the window are finite, closing isolated NaN gaps. Any position
    where *every* sample in the window is NaN (a gap wider than ``window``)
    falls back to the column mean (or ``0.0`` if the whole column is NaN),
    guaranteeing a fully finite array so correlation windows never see NaN.
    """
    s = pd.Series(values, dtype="float64").rolling(window, center=True, min_periods=1).mean()
    finite = values[np.isfinite(values)]
    fallback = float(finite.mean()) if finite.size else 0.0
    return s.fillna(fallback).to_numpy(dtype=np.float64)


def _sliding_windows(arr: np.ndarray, centers: np.ndarray, half_width: int) -> np.ndarray:
    """Extract ``(len(centers), 2*half_width+1)`` windows, edge-padded at the ends."""
    padded = np.pad(arr, half_width, mode="edge")
    offsets = np.arange(-half_width, half_width + 1)
    idx = centers[:, None] + offsets[None, :] + half_width
    return padded[idx]


def _normalize_windows(mat: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalize each row (window) of ``mat``."""
    mean = mat.mean(axis=-1, keepdims=True)
    std = mat.std(axis=-1, keepdims=True)
    return (mat - mean) / (std + _EPS)


def _softmax_weights(scores: np.ndarray, temperature: float = _SOFTMAX_TEMPERATURE) -> np.ndarray:
    z = temperature * (scores - scores.max())
    w = np.exp(z)
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1.0 / w.size)


def _flat_fallback(n: int, anchor: float) -> NCCResult:
    return NCCResult(tvt=np.full(n, anchor, dtype=float), confidence=np.zeros(n, dtype=float))


def _prepare_typewell_grid(
    tw: pd.DataFrame, anchor: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Resample the typewell GR log onto a uniform, fine-resolution TVT grid.

    The grid only covers ``anchor +/- _GRID_RANGE_FT`` (clipped to the
    typewell's own TVT extent) at ``_GRID_STEP_FT`` resolution -- see the
    module-level comments for why the resolution is far finer than the
    horizontal log's native 1 ft/row spacing.

    Returns ``(grid_tvt, grid_gr, lo, hi)`` where ``[lo, hi]`` is the clipped
    TVT range actually covered by the grid. Raises if the typewell has fewer
    than 3 finite TVT samples, or its TVT range is degenerate (caller falls
    back to flat anchor).
    """
    tw_sorted = tw.sort_values("TVT")
    tw_tvt = tw_sorted["TVT"].to_numpy(dtype=float)
    tw_gr = tw_sorted["GR"].to_numpy(dtype=float)
    finite = np.isfinite(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[finite], tw_gr[finite]
    if tw_tvt.size < 3:
        raise ValueError("typewell has fewer than 3 finite TVT samples")
    tw_gr = _rolling_fill(tw_gr)

    tw_min, tw_max = float(tw_tvt.min()), float(tw_tvt.max())
    if not (np.isfinite(tw_min) and np.isfinite(tw_max)) or tw_max <= tw_min:
        raise ValueError("typewell TVT range is degenerate")

    lo = max(tw_min, anchor - _GRID_RANGE_FT)
    hi = min(tw_max, anchor + _GRID_RANGE_FT)
    if hi <= lo:  # anchor sits entirely outside the typewell's range
        lo, hi = tw_min, tw_max

    glen = max(int(round((hi - lo) / _GRID_STEP_FT)) + 1, 3)
    grid_tvt = lo + np.arange(glen, dtype=float) * _GRID_STEP_FT
    grid_gr = np.interp(grid_tvt, tw_tvt, tw_gr)
    return grid_tvt, grid_gr, lo, hi


def _scale_scores(
    grid_gr: np.ndarray, gr_smooth: np.ndarray, eval_idx: np.ndarray, half_width: int
) -> np.ndarray:
    """Full ``(n_eval, glen)`` normalized-cross-correlation surface for one scale."""
    win_len = 2 * half_width + 1
    grid_centers = np.arange(grid_gr.size)
    cand_norm = _normalize_windows(_sliding_windows(grid_gr, grid_centers, half_width))
    h_norm = _normalize_windows(_sliding_windows(gr_smooth, eval_idx, half_width))
    return (h_norm @ cand_norm.T / win_len).astype(np.float32)


def _track_ncc_impl(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    half_widths: tuple[int, ...],
    search_halfwidth_ft: float,
    max_step_ft: float,
) -> NCCResult:
    n_eval = int(D.eval_mask(h).sum())
    if n_eval == 0:
        return NCCResult(tvt=np.zeros(0, dtype=float), confidence=np.zeros(0, dtype=float))

    anchor = D.last_known_tvt(h)
    eval_idx = np.flatnonzero(D.eval_mask(h))

    raw_gr = h["GR"].to_numpy(dtype=float)
    gr_smooth = _rolling_fill(raw_gr)
    raw_valid = ~np.isnan(raw_gr)

    try:
        grid_tvt, grid_gr, grid_lo, grid_hi = _prepare_typewell_grid(tw, anchor)
    except ValueError:
        return _flat_fallback(n_eval, anchor)

    n_half = max(int(round(search_halfwidth_ft / _GRID_STEP_FT)), 1)
    glen = grid_gr.size
    scale_scores = [_scale_scores(grid_gr, gr_smooth, eval_idx, hw) for hw in half_widths]

    tvt_out = np.empty(n_eval, dtype=float)
    conf_out = np.empty(n_eval, dtype=float)
    prev = float(np.clip(anchor, grid_lo, grid_hi))

    for k in range(n_eval):
        row = int(eval_idx[k])
        if not raw_valid[row]:
            tvt_out[k] = prev
            conf_out[k] = 0.0
            continue

        prev_idx = int(round((prev - grid_lo) / _GRID_STEP_FT))
        lo = max(prev_idx - n_half, 0)
        hi = min(prev_idx + n_half, glen - 1)

        best_tvt = np.empty(len(scale_scores))
        best_score = np.empty(len(scale_scores))
        for si, scores in enumerate(scale_scores):
            local = scores[k, lo : hi + 1]
            j = int(np.argmax(local))
            best_score[si] = float(local[j])
            best_tvt[si] = grid_tvt[lo + j]

        weights = _softmax_weights(best_score)
        combined_tvt = float(np.dot(weights, best_tvt))
        combined_score = float(np.dot(weights, best_score))

        delta = float(np.clip(combined_tvt - prev, -max_step_ft, max_step_ft))
        prev = float(np.clip(prev + delta, grid_lo, grid_hi))

        tvt_out[k] = prev
        conf_out[k] = float(np.clip(combined_score, 0.0, 1.0))

    return NCCResult(tvt=tvt_out, confidence=conf_out)


def track_ncc(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    half_widths: tuple[int, ...] = (8, 15, 25),
    search_halfwidth_ft: float = 30.0,
    max_step_ft: float = 0.5,
) -> NCCResult:
    """Sequentially register a horizontal well's GR log onto the typewell TVT axis.

    Starting from ``data.last_known_tvt(h)``, walks the evaluation zone
    (``data.eval_mask(h)`` order) one row at a time. At each row it builds a
    normalized GR window (of half-width ``hw`` for every ``hw`` in
    ``half_widths``) centered on that row, and normalized-cross-correlates it
    against every candidate typewell-GR window within
    ``prev_estimate +/- search_halfwidth_ft`` (typewell TVT range beyond the
    log is clipped, not extrapolated). Per scale, the best-matching candidate
    TVT and its score are kept; the ``half_widths`` proposals are combined via
    softmax-over-score weighting (temperature ``3.0``, matching the reference
    notebook's scale ensembling). The proposed move is then clamped to
    ``+/- max_step_ft`` to respect the near-flat, slowly-drifting physical
    trajectory.

    Rows with missing (``NaN``) GR skip the update entirely (the previous
    estimate is carried forward, ``confidence = 0.0`` for that row).

    This function never raises: any failure (malformed input, a typewell too
    short to build a grid, etc.) falls back to a flat anchor prediction with
    ``confidence`` all zero, matching ``baseline.predict_carry_last``.

    Returns an :class:`NCCResult` with arrays of length equal to the number of
    evaluation-zone rows, aligned to ``data.eval_mask(h)`` order.
    """
    try:
        return _track_ncc_impl(h, tw, half_widths, search_halfwidth_ft, max_step_ft)
    except Exception:
        try:
            n = int(D.eval_mask(h).sum())
        except Exception:
            n = 0
        try:
            anchor = D.last_known_tvt(h)
        except Exception:
            anchor = 0.0
        return _flat_fallback(n, anchor)
