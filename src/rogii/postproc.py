"""Layer3 post-processing: per-well smoothing/damping of an existing TVT prediction.

Design rationale (``docs/design.md`` §9): public top-tier notebooks apply a small
family of post-hoc curve-shaping steps on top of a regression prediction --
Savitzky-Golay smoothing, a robust (IRLS) low-degree polyfit blend, and a warm-up
damping toward the anchor at the start of the eval zone. Every transform here is a
**pure function of one well's own prediction curve** (plus its anchor and MD) -- it
never reads another well's data or the ground-truth ``TVT`` column, so it is safe to
apply identically at train-time validation and at inference on hidden test wells.

Contract: each ``*_transform`` function takes ``(pred, anchor, md_since, **params)``
where ``pred``/``md_since`` are 1-D arrays for a single well's evaluation-zone rows
(in ``eval_mask`` order) and ``anchor`` is that well's scalar anchor TVT, and returns
an array of the same shape. This shared signature lets a CV harness (or, later,
``predict.py``) apply any transform -- or compose several -- via one generic
per-well loop (see ``scripts/run_postproc_cv.py``).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth_well(pred: np.ndarray, window: int, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing of one well's prediction curve.

    ``window`` is clamped to the largest value <= ``len(pred)`` (and forced odd);
    if the well is too short for a valid window (``window <= polyorder``), the raw
    prediction is returned unchanged rather than raising.
    """
    pred = np.asarray(pred, dtype=np.float64)
    n = pred.shape[0]
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w <= polyorder or w < 1:
        return pred.copy()
    return savgol_filter(pred, window_length=w, polyorder=polyorder, mode="interp")


def irls_polyfit_blend(
    pred: np.ndarray,
    anchor: float,
    md_since: np.ndarray,
    degree: int = 4,
    alpha: float = 0.5,
    n_iter: int = 5,
    huber_c: float = 1.345,
) -> np.ndarray:
    """Blend the raw prediction with a robust (Huber-IRLS) low-degree polyfit.

    Fits ``U = pred - anchor`` as a degree-``degree`` polynomial of normalized MD
    (``(md_since - md_since.min()) / range``) using iteratively-reweighted least
    squares (Huber weights recomputed from the residual MAD each iteration), then
    returns ``anchor + alpha*U + (1-alpha)*fit`` -- ``alpha=1`` reproduces the raw
    prediction untouched, ``alpha=0`` is the fully-smoothed polynomial trend.

    Falls back to the raw prediction for wells too short to support the requested
    degree (``degree`` is clamped to ``len(pred) - 2``; if that is < 1 the well is
    left untouched).
    """
    pred = np.asarray(pred, dtype=np.float64)
    md_since = np.asarray(md_since, dtype=np.float64)
    n = pred.shape[0]
    eff_degree = min(degree, max(n - 2, 0))
    if eff_degree < 1 or n < 3:
        return pred.copy()

    md_range = md_since.max() - md_since.min()
    x = (md_since - md_since.min()) / md_range if md_range > 0 else np.zeros(n)
    y = pred - anchor

    weights = np.ones(n)
    coeffs = np.polyfit(x, y, eff_degree, w=weights)
    for _ in range(n_iter):
        fit = np.polyval(coeffs, x)
        resid = y - fit
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scale = 1.4826 * mad
        if scale < 1e-6:
            break
        r = resid / scale
        weights = np.where(np.abs(r) <= huber_c, 1.0, huber_c / np.abs(r))
        coeffs = np.polyfit(x, y, eff_degree, w=weights)

    fit = np.polyval(coeffs, x)
    blended_u = alpha * y + (1.0 - alpha) * fit
    return anchor + blended_u


def warmup_damping(pred: np.ndarray, anchor: float, md_since: np.ndarray, tau: float) -> np.ndarray:
    """Damp the prediction toward the anchor near the start of the eval zone.

    ``factor = 1 - exp(-md_since/tau)`` grows from ~0 (pure anchor) at
    ``md_since=0`` toward 1 (raw prediction) as MD advances past the anchor.
    """
    pred = np.asarray(pred, dtype=np.float64)
    md_since = np.asarray(md_since, dtype=np.float64)
    factor = 1.0 - np.exp(-md_since / tau)
    return anchor + factor * (pred - anchor)


def savgol_transform(
    pred: np.ndarray, anchor: float, md_since: np.ndarray, window: int, polyorder: int = 3
) -> np.ndarray:
    """``(pred, anchor, md_since, **params) -> pred`` wrapper around :func:`savgol_smooth_well`."""
    del anchor, md_since  # unused by this family, kept for a uniform call signature
    return savgol_smooth_well(pred, window=window, polyorder=polyorder)


def irls_transform(
    pred: np.ndarray,
    anchor: float,
    md_since: np.ndarray,
    alpha: float,
    degree: int = 4,
    n_iter: int = 5,
) -> np.ndarray:
    """``(pred, anchor, md_since, **params) -> pred`` wrapper around :func:`irls_polyfit_blend`."""
    return irls_polyfit_blend(pred, anchor, md_since, degree=degree, alpha=alpha, n_iter=n_iter)


def warmup_transform(
    pred: np.ndarray, anchor: float, md_since: np.ndarray, tau: float
) -> np.ndarray:
    """``(pred, anchor, md_since, **params) -> pred`` wrapper around :func:`warmup_damping`."""
    return warmup_damping(pred, anchor, md_since, tau=tau)
