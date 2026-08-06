"""Spatial-prior TVT predictor: interpolate formation-top depth from nearby wells.

Source / motivation
--------------------
Host (Igor Kuvaev, forum 708167) claims formation-top depth at a given
``(X, Y)`` can be interpolated from the ``k`` nearest offset wells at
``R^2 > 0.99``. The train-only ``horizontal_well.csv`` columns ``ANCC``,
``ASTNU``, ``ASTNL``, ``EGFDU``, ``EGFDL``, ``BUDA`` give exactly that: the
(true vertical, sea-level-referenced) depth of six named formation tops at
every logged ``(X, Y, Z)`` along every train well. A quick check across
sampled wells confirms the geometric identity implied by "TVT" (true vertical
thickness) and "formation depth measured from the same origin as Z"::

    TVT_input + Z - formation_depth  ~=  constant (per well, per formation)

i.e. ``TVT = -Z + formation_depth(X, Y) + offset``, where ``offset`` absorbs
the (per-well) datum shift between the wellbore's ``TVT`` frame and the
formation-depth frame, and is nearly exactly constant within a well (observed
std ~0.008 ft across a well's known zone). This module never uses a well's
*own* formation-depth columns (a train-only leak) or its ``TVT`` column --
only the formation-depth samples of *other* wells (spatially interpolated via
inverse-distance-weighted KNN) plus the well's own ``Z`` and known-zone
``TVT_input`` (present in both train and test), which are used only to
calibrate the per-well ``offset`` and to judge fit quality (``prefix_rmse``).

Leak-safety notes
------------------
* ``build_surface_bank`` only ever reads *other* wells' formation-depth
  columns; at query time ``predict_spatial`` additionally excludes the target
  well's own samples from the KNN search via ``exclude_well``
  (leave-self-out), so a well is never interpolated against itself.
* ``predict_spatial`` never reads the target well's own formation-depth
  columns or ``TVT`` column -- only ``X``, ``Y``, ``Z``, and ``TVT_input``
  (all present in test).
* Like ``registration/ncc.py`` and ``registration/particle.py``, the public
  entry point never raises: any failure falls back to a flat carry-last
  anchor with ``prefix_rmse = inf`` so a bad well cannot abort a CV/submission
  run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import data as D

# The six train-only formation-top depth columns (see module docstring).
FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")

_DEFAULT_STRIDE = 10
_DEFAULT_K = 10

# KNN-with-exclusion is implemented by over-fetching neighbors from the KD-tree
# and then filtering out the excluded well's own samples. Near a well's own
# trajectory its *own* samples genuinely are the nearest ones, so the fetch
# size must account for all of them: fetching ``k + n_self`` neighbors (where
# ``n_self`` is the excluded well's total sample count in that formation's
# bank) guarantees at least ``k`` valid neighbors in a single query, because
# self samples can crowd out at most ``n_self`` slots.

# Number of known-zone rows (closest to the eval-zone boundary) used to
# calibrate each formation's per-well offset and to measure fit quality.
_PREFIX_WINDOW = 500

# Inverse-distance-weighting: avoid division by zero for a coincident sample.
_IDW_EPS = 1e-3

# Composite across formations: avoid division by zero for a perfect prefix fit.
_COMPOSITE_EPS = 1e-6

# Local-plane fit: ridge regularization applied to the two gradient
# coefficients (never the intercept), relative to the neighborhood's mean
# squared distance. Guards against near-collinear neighborhoods -- laterals
# are long straight lines, so all k neighbors can fall on a single line in
# (X, Y), which would make an unregularized plane fit singular.
_PLANE_RIDGE = 1e-4

# Use every core for KD-tree queries (they release the GIL).
_QUERY_WORKERS = -1


@dataclass(frozen=True)
class SurfaceBank:
    """Per-formation spatial index of ``(X, Y) -> formation-top depth`` samples.

    Built once from every well's formation-depth columns (stride-subsampled).
    Contains *every* well handed to :func:`build_surface_bank`, including
    whichever one will later be held out -- callers must pass ``exclude_well``
    to :func:`predict_spatial` to leave a well's own samples out at query time.
    """

    formations: tuple[str, ...]
    trees: dict[str, cKDTree] = field(default_factory=dict)
    depths: dict[str, np.ndarray] = field(default_factory=dict)
    well_codes: dict[str, np.ndarray] = field(default_factory=dict)
    well_to_code: dict[str, int] = field(default_factory=dict)


@dataclass
class SpatialResult:
    """Per-row TVT estimate for the evaluation zone, plus fit diagnostics.

    ``tvt`` is aligned to ``data.eval_mask(h)`` order (length = number of
    evaluation-zone rows) and comes from the single best formation (lowest
    ``prefix_rmse``). ``tvt_composite`` is the inverse-squared-prefix-residual
    weighted average across *all* usable formations (identical to ``tvt`` on
    the fallback path or when only one formation is usable). ``prefix_rmse``
    is the residual of the winning formation's calibrated fit over the
    known-zone prefix window (``inf`` on the exception-fallback path, or if
    no formation produced a usable candidate). ``n_formations_used`` counts
    how many formations produced a usable (finite-offset) candidate, out of
    ``len(bank.formations)``.
    """

    tvt: np.ndarray
    prefix_rmse: float
    n_formations_used: int
    tvt_composite: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        if self.tvt_composite.size == 0:
            self.tvt_composite = self.tvt


def build_surface_bank(
    wells: list[str],
    split: str = "train",
    stride: int = _DEFAULT_STRIDE,
    root: str | None = None,
    formations: tuple[str, ...] = FORMATIONS,
) -> SurfaceBank:
    """Collect stride-subsampled ``(X, Y, depth)`` samples for each formation.

    Every ``stride``-th row of each well is kept (after dropping rows with a
    non-finite ``X``/``Y``/formation-depth value), tagged with an integer well
    code, and indexed with a per-formation :class:`~scipy.spatial.cKDTree`.
    Wells whose horizontal-well log fails to load are skipped rather than
    aborting the whole bank build. A formation missing from every well (e.g.
    an unexpected schema) is simply absent from ``bank.formations``.
    """
    well_to_code = {well: i for i, well in enumerate(wells)}
    xs: dict[str, list[np.ndarray]] = {f: [] for f in formations}
    ys: dict[str, list[np.ndarray]] = {f: [] for f in formations}
    ds: dict[str, list[np.ndarray]] = {f: [] for f in formations}
    codes: dict[str, list[np.ndarray]] = {f: [] for f in formations}

    for well in wells:
        code = well_to_code[well]
        try:
            h = D.load_horizontal(well, split, root)
        except Exception:
            continue

        sub = h.iloc[::stride]
        if "X" not in sub.columns or "Y" not in sub.columns:
            continue
        x = sub["X"].to_numpy(dtype=float)
        y = sub["Y"].to_numpy(dtype=float)
        finite_xy = np.isfinite(x) & np.isfinite(y)

        for f in formations:
            if f not in sub.columns:
                continue
            depth = sub[f].to_numpy(dtype=float)
            valid = finite_xy & np.isfinite(depth)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            xs[f].append(x[valid])
            ys[f].append(y[valid])
            ds[f].append(depth[valid])
            codes[f].append(np.full(n_valid, code, dtype=np.int32))

    trees: dict[str, cKDTree] = {}
    depths: dict[str, np.ndarray] = {}
    well_codes: dict[str, np.ndarray] = {}
    used_formations: list[str] = []

    for f in formations:
        if not xs[f]:
            continue
        xf = np.concatenate(xs[f])
        yf = np.concatenate(ys[f])
        trees[f] = cKDTree(np.column_stack([xf, yf]))
        depths[f] = np.concatenate(ds[f])
        well_codes[f] = np.concatenate(codes[f])
        used_formations.append(f)

    return SurfaceBank(
        formations=tuple(used_formations),
        trees=trees,
        depths=depths,
        well_codes=well_codes,
        well_to_code=well_to_code,
    )


def _knn_weights(
    tree: cKDTree,
    coords: np.ndarray,
    depths: np.ndarray,
    codes: np.ndarray,
    exclude_code: int,
    k: int,
    weight_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-self-out KNN: per-query neighbor indices and IDW weights.

    Fetches ``k + n_self`` neighbors in one query (see the module-level
    comment: self samples can occupy at most ``n_self`` of the nearest slots,
    so at least ``k`` valid neighbors are guaranteed whenever the bank holds
    that many non-self samples), filters out samples tagged with
    ``exclude_code`` (leave-self-out), and returns ``(idx, weights)`` of shape
    ``(n_query, k_query)`` where non-kept slots have weight ``0``. Weights are
    inverse-distance (to the power ``weight_power``: ``1`` = standard IDW,
    ``0`` = uniform among kept neighbors, ``2`` = tighter/more local) and
    *not* normalized. ``weight_power=1.0`` (the default, used by every
    existing caller) reproduces the original ``1.0 / (dist + _IDW_EPS)``
    expression exactly (no ``**`` applied), so this parameter is purely
    additive -- R7-R16 callers that never pass it see byte-identical output
    (see ``docs/playbooks`` R17: opt-in distance-weighting power for the
    SurfaceBank precision pass).
    """
    n_total = depths.size
    n_self = int(np.sum(codes == exclude_code))
    k_query = min(k + n_self, n_total)
    dist, idx = tree.query(coords, k=k_query, workers=_QUERY_WORKERS)
    if k_query == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    valid = codes[idx] != exclude_code

    rank_within_valid = np.cumsum(valid, axis=1) - 1
    keep = valid & (rank_within_valid < k)

    inv_dist = 1.0 / (dist + _IDW_EPS)
    raw_weights = inv_dist if weight_power == 1.0 else inv_dist**weight_power
    weights = np.where(keep, raw_weights, 0.0)
    return idx, weights


def _knn_idw_depth(
    tree: cKDTree,
    coords: np.ndarray,
    depths: np.ndarray,
    codes: np.ndarray,
    exclude_code: int,
    k: int,
) -> np.ndarray:
    """Inverse-distance-weighted KNN interpolation, excluding ``exclude_code``.

    Rows with zero valid neighbors (only possible in a near-empty bank) come
    back as ``nan``.
    """
    n_query = coords.shape[0]
    if depths.size == 0:
        return np.full(n_query, np.nan)

    idx, weights = _knn_weights(tree, coords, depths, codes, exclude_code, k)
    weight_sum = weights.sum(axis=1)
    weighted_depth = (weights * depths[idx]).sum(axis=1)

    safe_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    return np.where(weight_sum > 0, weighted_depth / safe_sum, np.nan)


def _knn_plane_depth(
    tree: cKDTree,
    coords: np.ndarray,
    depths: np.ndarray,
    codes: np.ndarray,
    exclude_code: int,
    k: int,
) -> np.ndarray:
    """Weighted local-plane (linear) KNN interpolation, excluding ``exclude_code``.

    For each query point, fits ``depth ~= a + b*dx + c*dy`` over its ``k``
    nearest non-self neighbors by inverse-distance-weighted least squares
    (coordinates centered on the query point, so the prediction is the
    intercept ``a``). Unlike plain IDW -- which is systematically dragged
    toward whichever adjacent lateral supplies the nearest samples -- a local
    plane cancels the first-order gradient error, which dominates because the
    formation surfaces have significant regional dip relative to the well
    spacing. The gradient coefficients are ridge-regularized (see
    ``_PLANE_RIDGE``) because all ``k`` neighbors can lie on one straight
    lateral (collinear in ``(X, Y)``), which would otherwise be singular.
    Falls back to plain IDW if the batched solve still fails.
    """
    n_query = coords.shape[0]
    if depths.size == 0:
        return np.full(n_query, np.nan)

    idx, weights = _knn_weights(tree, coords, depths, codes, exclude_code, k)
    bank_xy = np.asarray(tree.data)

    dx = bank_xy[idx, 0] - coords[:, 0:1]
    dy = bank_xy[idx, 1] - coords[:, 1:2]
    d = depths[idx]

    w_sum = weights.sum(axis=1)
    empty = w_sum <= 0

    # Weighted normal equations for [1, dx, dy] per query: A beta = b.
    features = np.stack([np.ones_like(dx), dx, dy], axis=-1)  # (nq, kq, 3)
    a_mat = np.einsum("qk,qki,qkj->qij", weights, features, features)
    b_vec = np.einsum("qk,qk,qki->qi", weights, d, features)

    # Ridge on the gradient terms only, scaled by the neighborhood extent.
    r2 = (weights * (dx**2 + dy**2)).sum(axis=1)
    lam = _PLANE_RIDGE * np.maximum(r2, 1.0)
    a_mat[:, 1, 1] += lam
    a_mat[:, 2, 2] += lam
    # Keep the batched solve non-singular for empty rows; they are NaN-ed below.
    a_mat[empty] = np.eye(3)

    try:
        # NumPy 2 batched-solve semantics: b must be (..., m, n) when batched.
        beta = np.linalg.solve(a_mat, b_vec[:, :, None])[:, :, 0]
        result = beta[:, 0]
    except np.linalg.LinAlgError:
        return _knn_idw_depth(tree, coords, depths, codes, exclude_code, k)

    result = np.where(empty | ~np.isfinite(result), np.nan, result)
    return result


def _flat_fallback(h: pd.DataFrame) -> SpatialResult:
    try:
        n = int(D.eval_mask(h).sum())
    except Exception:
        n = 0
    try:
        anchor = D.last_known_tvt(h)
    except Exception:
        anchor = 0.0
    return SpatialResult(
        tvt=np.full(n, anchor, dtype=float), prefix_rmse=float("inf"), n_formations_used=0
    )


_INTERPOLATORS = {"idw": _knn_idw_depth, "plane": _knn_plane_depth}


def _predict_spatial_impl(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    k: int,
    method: str,
) -> SpatialResult:
    known = h["TVT_input"].notna().to_numpy()
    eval_m = D.eval_mask(h)
    eval_idx = np.flatnonzero(eval_m)
    known_idx = np.flatnonzero(known)

    if eval_idx.size == 0 or known_idx.size == 0:
        raise ValueError("empty eval zone or no known-zone anchor")

    prefix_idx = known_idx[-min(_PREFIX_WINDOW, known_idx.size) :]

    x = h["X"].to_numpy(dtype=float)
    y = h["Y"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    active_idx = np.concatenate([prefix_idx, eval_idx])
    coords = np.column_stack([x[active_idx], y[active_idx]])
    n_prefix = prefix_idx.size

    exclude_code = bank.well_to_code.get(exclude_well, -1)
    fallback_anchor = float(tvt_input[known_idx[-1]])

    interpolate = _INTERPOLATORS[method]
    eval_z = z[eval_idx]
    candidates: list[np.ndarray] = []
    candidate_rmses: list[float] = []

    for f in bank.formations:
        depth_interp = interpolate(
            bank.trees[f], coords, bank.depths[f], bank.well_codes[f], exclude_code, k
        )
        prefix_depth = depth_interp[:n_prefix]
        eval_depth = depth_interp[n_prefix:]

        valid_prefix = np.isfinite(prefix_depth)
        if not np.any(valid_prefix):
            continue

        prefix_z = z[prefix_idx][valid_prefix]
        prefix_tvt = tvt_input[prefix_idx][valid_prefix]
        offset = float(np.median(prefix_tvt + prefix_z - prefix_depth[valid_prefix]))

        prefix_pred = -prefix_z + prefix_depth[valid_prefix] + offset
        resid = prefix_tvt - prefix_pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        if not np.isfinite(rmse):
            continue

        eval_tvt = -eval_z + eval_depth + offset
        valid_eval = np.isfinite(eval_tvt)
        if not np.all(valid_eval):
            eval_tvt = np.where(valid_eval, eval_tvt, fallback_anchor)
        candidates.append(eval_tvt)
        candidate_rmses.append(rmse)

    if not candidates:
        raise ValueError("no formation produced a usable candidate")

    rmses = np.array(candidate_rmses)
    best = int(np.argmin(rmses))

    # Inverse-squared-residual weighted composite across usable formations.
    inv_weights = 1.0 / (rmses**2 + _COMPOSITE_EPS)
    composite = np.average(np.stack(candidates), axis=0, weights=inv_weights)

    return SpatialResult(
        tvt=candidates[best],
        prefix_rmse=float(rmses[best]),
        n_formations_used=len(candidates),
        tvt_composite=composite,
    )


def predict_spatial(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    k: int = _DEFAULT_K,
    method: str = "idw",
) -> SpatialResult:
    """Spatial-prior TVT prediction for the evaluation zone of ``h``.

    For each formation in ``bank``, the target well's known-zone prefix
    (last ``_PREFIX_WINDOW`` rows with a known ``TVT_input``) and evaluation
    zone are interpolated via leave-self-out KNN against ``bank`` (``method``:
    ``"plane"`` = IDW-weighted local-plane fit, ``"idw"`` = plain inverse
    distance weighting), a per-well offset is calibrated on the prefix, and
    the formation whose calibrated prefix residual (``prefix_rmse``) is lowest
    is used for the evaluation-zone prediction. Never raises: any failure
    (missing anchor, empty eval zone, an unknown ``method``, or an internal
    error) falls back to a flat carry-last prediction with
    ``prefix_rmse = inf`` and ``n_formations_used = 0``.
    """
    try:
        return _predict_spatial_impl(h, bank, exclude_well, k, method)
    except Exception:
        return _flat_fallback(h)


# --------------------------------------------------------------------------- #
# R17 (SurfaceBank precision pass): opt-in quadratic-surface fit + per-row
# local-fit diagnostics, exposed only via predict_spatial_v2 below.
# predict_spatial / build_surface_bank / SpatialResult stay byte-unchanged;
# the only touched *existing* symbol is _knn_weights (new `weight_power`
# kwarg, default-preserving -- see its docstring). Everything else here is a
# new function/dataclass, so every pre-R17 caller sees identical numbers.
# --------------------------------------------------------------------------- #

# A quadratic surface needs 6 params (1, dx, dy, dx^2, dy^2, dx*dy); with
# fewer than this many valid (non-self) donor neighbors in a row's
# neighborhood the fit is under-determined even with ridge, so that row
# falls back to the degree-1 (plane) fit instead.
_QUADRATIC_MIN_DONORS = 6


def _knn_surface_fit(
    tree: cKDTree,
    coords: np.ndarray,
    depths: np.ndarray,
    codes: np.ndarray,
    exclude_code: int,
    k: int,
    degree: int = 1,
    weight_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted local surface fit (``degree=1`` plane / ``degree=2`` quadratic).

    Generalizes ``_knn_plane_depth`` to also report per-query diagnostics:
    the fitted first-order gradient (``grad_x``, ``grad_y`` -- present for
    both degrees; a per-row, local analogue of the well-level
    ``spatial_slope`` feature from the 2026-07-11 slope-predictability
    analysis, r=0.49 with the eval-zone true slope), the weighted in-sample
    residual std of the fit over its own donor neighbors (``resid_std`` -- a
    per-row surface-fit confidence/roughness diagnostic: large values mean
    the local geology is not well described by a smooth surface here), and
    the donor count actually used (``n_donors``).

    Returns ``(depth, grad_x, grad_y, resid_std, n_donors)``, each shape
    ``(n_query,)``. Rows with zero valid neighbors come back ``nan``
    (``n_donors`` -> ``0``). At ``degree=1``, ``weight_power=1.0`` this is a
    strict superset of ``_knn_plane_depth`` (identical ``depth``, plus the
    three new diagnostics) -- same features, same ridge, same batched solve.
    For ``degree=2``, rows with fewer than ``_QUADRATIC_MIN_DONORS`` valid
    donors fall back to the ``degree=1`` fit for that row only (mirrors
    ``_knn_plane_depth``'s ridge-then-fallback pattern, one level up: a
    LinAlgError on the ``degree=1`` batched solve falls back further, to a
    plain weighted mean).
    """
    n_query = coords.shape[0]
    if depths.size == 0:
        nan_arr = np.full(n_query, np.nan)
        return nan_arr, nan_arr.copy(), nan_arr.copy(), nan_arr.copy(), np.zeros(n_query)

    idx, weights = _knn_weights(tree, coords, depths, codes, exclude_code, k, weight_power)
    bank_xy = np.asarray(tree.data)

    dx = bank_xy[idx, 0] - coords[:, 0:1]
    dy = bank_xy[idx, 1] - coords[:, 1:2]
    d = depths[idx]

    n_donors = (weights > 0).sum(axis=1).astype(np.float64)
    w_sum = weights.sum(axis=1)
    empty = w_sum <= 0

    if degree == 2:
        features = np.stack([np.ones_like(dx), dx, dy, dx * dx, dy * dy, dx * dy], axis=-1)
        ridge_terms: tuple[int, ...] = (1, 2, 3, 4, 5)
    else:
        features = np.stack([np.ones_like(dx), dx, dy], axis=-1)
        ridge_terms = (1, 2)

    a_mat = np.einsum("qk,qki,qkj->qij", weights, features, features)
    b_vec = np.einsum("qk,qk,qki->qi", weights, d, features)

    r2 = (weights * (dx**2 + dy**2)).sum(axis=1)
    lam = _PLANE_RIDGE * np.maximum(r2, 1.0)
    for term in ridge_terms:
        a_mat[:, term, term] += lam
    # Keep the batched solve non-singular for empty rows; they are NaN-ed below.
    a_mat[empty] = np.eye(features.shape[-1])

    try:
        beta = np.linalg.solve(a_mat, b_vec[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        if degree == 2:
            return _knn_surface_fit(
                tree, coords, depths, codes, exclude_code, k, degree=1, weight_power=weight_power
            )
        # Degree-1 batched solve failed as a whole -> fall back to a plain
        # weighted mean, reusing the neighbors/weights already fetched above
        # (mirrors _knn_plane_depth's IDW fallback without a redundant query).
        safe_w_sum = np.where(w_sum > 0, w_sum, 1.0)
        depth = np.where(w_sum > 0, (weights * d).sum(axis=1) / safe_w_sum, np.nan)
        nan_arr = np.full(n_query, np.nan)
        return depth, nan_arr.copy(), nan_arr.copy(), nan_arr.copy(), n_donors

    depth = beta[:, 0]
    grad_x = beta[:, 1]
    grad_y = beta[:, 2]

    pred_at_donors = np.einsum("qki,qi->qk", features, beta)
    resid = d - pred_at_donors
    weighted_sse = (weights * resid**2).sum(axis=1)
    safe_w_sum = np.where(w_sum > 0, w_sum, 1.0)
    resid_std = np.sqrt(np.maximum(weighted_sse / safe_w_sum, 0.0))

    depth = np.where(empty | ~np.isfinite(depth), np.nan, depth)
    grad_x = np.where(empty | ~np.isfinite(grad_x), np.nan, grad_x)
    grad_y = np.where(empty | ~np.isfinite(grad_y), np.nan, grad_y)
    resid_std = np.where(empty | ~np.isfinite(resid_std), np.nan, resid_std)

    if degree == 2:
        insufficient = n_donors < _QUADRATIC_MIN_DONORS
        if np.any(insufficient):
            p_depth, p_gx, p_gy, p_resid, p_n = _knn_surface_fit(
                tree, coords, depths, codes, exclude_code, k, degree=1, weight_power=weight_power
            )
            depth = np.where(insufficient, p_depth, depth)
            grad_x = np.where(insufficient, p_gx, grad_x)
            grad_y = np.where(insufficient, p_gy, grad_y)
            resid_std = np.where(insufficient, p_resid, resid_std)

    return depth, grad_x, grad_y, resid_std, n_donors


@dataclass
class SpatialResultV2:
    """Extended per-row spatial estimate + local-fit diagnostics (R17).

    Same ``tvt`` / ``tvt_composite`` / ``prefix_rmse`` / ``n_formations_used``
    contract as :class:`SpatialResult` (computed via :func:`_knn_surface_fit`
    with ``degree=1`` for ``method="plane"`` or ``degree=2`` for
    ``method="quadratic"``), plus four new per-row diagnostic columns for the
    winning formation, eval-zone-aligned: ``resid_std`` (local surface-fit
    confidence), ``n_donors`` (non-self neighbors used), and ``grad_x`` /
    ``grad_y`` (the fitted first-order surface gradient at the query point).
    See :func:`predict_spatial_v2` for the full contract.
    """

    tvt: np.ndarray
    prefix_rmse: float
    n_formations_used: int
    resid_std: np.ndarray
    n_donors: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    tvt_composite: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        if self.tvt_composite.size == 0:
            self.tvt_composite = self.tvt


def _flat_fallback_v2(h: pd.DataFrame) -> SpatialResultV2:
    try:
        n = int(D.eval_mask(h).sum())
    except Exception:
        n = 0
    try:
        anchor = D.last_known_tvt(h)
    except Exception:
        anchor = 0.0
    zeros = np.zeros(n, dtype=float)
    return SpatialResultV2(
        tvt=np.full(n, anchor, dtype=float),
        prefix_rmse=float("inf"),
        n_formations_used=0,
        resid_std=zeros.copy(),
        n_donors=zeros.copy(),
        grad_x=zeros.copy(),
        grad_y=zeros.copy(),
    )


_V2_DEGREE_BY_METHOD = {"plane": 1, "quadratic": 2}


def _predict_spatial_v2_impl(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    k: int,
    method: str,
    weight_power: float,
) -> SpatialResultV2:
    if method not in _V2_DEGREE_BY_METHOD:
        raise ValueError(f"predict_spatial_v2: unknown method {method!r}")
    degree = _V2_DEGREE_BY_METHOD[method]

    known = h["TVT_input"].notna().to_numpy()
    eval_m = D.eval_mask(h)
    eval_idx = np.flatnonzero(eval_m)
    known_idx = np.flatnonzero(known)

    if eval_idx.size == 0 or known_idx.size == 0:
        raise ValueError("empty eval zone or no known-zone anchor")

    prefix_idx = known_idx[-min(_PREFIX_WINDOW, known_idx.size) :]

    x = h["X"].to_numpy(dtype=float)
    y = h["Y"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    active_idx = np.concatenate([prefix_idx, eval_idx])
    coords = np.column_stack([x[active_idx], y[active_idx]])
    n_prefix = prefix_idx.size

    exclude_code = bank.well_to_code.get(exclude_well, -1)
    fallback_anchor = float(tvt_input[known_idx[-1]])

    eval_z = z[eval_idx]
    candidates: list[np.ndarray] = []
    candidate_rmses: list[float] = []
    candidate_extra: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for f in bank.formations:
        depth_interp, gx, gy, resid, ndon = _knn_surface_fit(
            bank.trees[f],
            coords,
            bank.depths[f],
            bank.well_codes[f],
            exclude_code,
            k,
            degree=degree,
            weight_power=weight_power,
        )
        prefix_depth = depth_interp[:n_prefix]
        eval_depth = depth_interp[n_prefix:]
        eval_gx = gx[n_prefix:]
        eval_gy = gy[n_prefix:]
        eval_resid = resid[n_prefix:]
        eval_ndon = ndon[n_prefix:]

        valid_prefix = np.isfinite(prefix_depth)
        if not np.any(valid_prefix):
            continue

        prefix_z = z[prefix_idx][valid_prefix]
        prefix_tvt = tvt_input[prefix_idx][valid_prefix]
        offset = float(np.median(prefix_tvt + prefix_z - prefix_depth[valid_prefix]))

        prefix_pred = -prefix_z + prefix_depth[valid_prefix] + offset
        resid_prefix = prefix_tvt - prefix_pred
        rmse = float(np.sqrt(np.mean(resid_prefix**2)))
        if not np.isfinite(rmse):
            continue

        eval_tvt = -eval_z + eval_depth + offset
        valid_eval = np.isfinite(eval_tvt)
        if not np.all(valid_eval):
            eval_tvt = np.where(valid_eval, eval_tvt, fallback_anchor)
            eval_gx = np.where(valid_eval, eval_gx, 0.0)
            eval_gy = np.where(valid_eval, eval_gy, 0.0)
            eval_resid = np.where(valid_eval, eval_resid, np.nan)
            eval_ndon = np.where(valid_eval, eval_ndon, 0.0)

        candidates.append(eval_tvt)
        candidate_rmses.append(rmse)
        candidate_extra.append((eval_gx, eval_gy, eval_resid, eval_ndon))

    if not candidates:
        raise ValueError("no formation produced a usable candidate")

    rmses = np.array(candidate_rmses)
    best = int(np.argmin(rmses))

    inv_weights = 1.0 / (rmses**2 + _COMPOSITE_EPS)
    composite = np.average(np.stack(candidates), axis=0, weights=inv_weights)

    best_gx, best_gy, best_resid, best_ndon = candidate_extra[best]

    return SpatialResultV2(
        tvt=candidates[best],
        prefix_rmse=float(rmses[best]),
        n_formations_used=len(candidates),
        resid_std=np.where(np.isfinite(best_resid), best_resid, 0.0),
        n_donors=best_ndon,
        grad_x=np.where(np.isfinite(best_gx), best_gx, 0.0),
        grad_y=np.where(np.isfinite(best_gy), best_gy, 0.0),
        tvt_composite=composite,
    )


def predict_spatial_v2(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    k: int = _DEFAULT_K,
    method: str = "plane",
    weight_power: float = 1.0,
) -> SpatialResultV2:
    """R17: spatial-prior prediction with an opt-in quadratic-surface fit
    (``method="quadratic"``), opt-in IDW-power distance weighting
    (``weight_power``), and four new per-row local-fit diagnostic columns
    (see :class:`SpatialResultV2`).

    Sibling of :func:`predict_spatial`, not a replacement: at
    ``method="plane"``, any ``k``, ``weight_power=1.0``, its ``tvt`` /
    ``prefix_rmse`` / ``n_formations_used`` are numerically identical to
    ``predict_spatial(h, bank, exclude_well, k, method="plane")`` (both
    ultimately run the same weighted-least-squares plane fit via
    ``_knn_surface_fit(degree=1)`` / ``_knn_plane_depth``) -- it does **not**
    reproduce ``predict_spatial``'s own *default* method (``"idw"``), since
    plain IDW has no local gradient to report as a diagnostic.

    ``method="quadratic"`` fits ``depth ~= a + b*dx + c*dy + d*dx^2 + e*dy^2 +
    f*dx*dy`` per query point instead of a plane, falling back to the plane
    fit row-by-row wherever fewer than ``_QUADRATIC_MIN_DONORS`` non-self
    donors are available (an under-determined quadratic system) -- see
    :func:`_knn_surface_fit`.

    Never raises: any failure (missing anchor, empty eval zone, an unknown
    ``method``, or an internal error) falls back to a flat carry-last
    prediction with ``prefix_rmse = inf``, ``n_formations_used = 0``, and
    zero-filled diagnostic columns.
    """
    try:
        return _predict_spatial_v2_impl(h, bank, exclude_well, k, method, weight_power)
    except Exception:
        return _flat_fallback_v2(h)


# --------------------------------------------------------------------------- #
# R28 (per-well offset time-series diagnostics): opt-in prefix-only
# offset-drift diagnostics for the downstream LGBM stack. Mirrors
# ``_predict_spatial_impl``'s best-formation selection exactly (identical
# ``prefix_idx`` slicing, ``valid_prefix`` mask, offset-median calibration,
# and lowest-``prefix_rmse`` formation selection) but only interpolates the
# known-zone prefix rows -- never the eval zone, since these diagnostics only
# ever look at the known-zone offset series. Every symbol above this banner
# is untouched; ``offset_diagnostics`` never raises (see its docstring).
# --------------------------------------------------------------------------- #


@dataclass
class OffsetDiag:
    """Per-well offset time-series diagnostics over the known-zone prefix.

    ``predict_spatial``/``_predict_spatial_impl`` calibrate a single per-well
    ``offset`` (the median of ``off_j = tvt_input_j + z_j - depth_j`` over the
    known-zone prefix, for the formation with the lowest prefix RMSE) and use
    it as a flat constant. This dataclass instead reports how that per-row
    ``off_j`` series *evolves* across the prefix, for the same winning
    formation, so a downstream GBDT can learn from prefix-internal offset
    drift instead of only its median.

    All fields describe the winning formation's *valid* prefix rows (rows
    whose interpolated formation depth was finite), in time (row-index)
    order:

    * ``offset_med`` -- median of the full valid-prefix ``off`` series.
      Identical definition/value to the per-well calibration constant used
      by ``predict_spatial`` for the same well/bank/``k``/``method``.
    * ``offset_early`` / ``offset_mid`` / ``offset_late`` -- medians of the
      first / middle / last third of the ``off`` series, split via
      ``np.array_split`` (a length not divisible by 3 puts the larger chunks
      first). ``nan`` if fewer than 3 valid prefix rows.
    * ``wls_slope_per100`` -- the slope (``b``) of a tail-weighted weighted
      least-squares linear fit ``off_j ~= a + b * t_j``, scaled to ft per 100
      rows (``b * 100``). ``t_j`` is the row position relative to the last
      valid prefix row (``t_j <= 0``, increasing toward ``0`` at the
      prefix's end). Weights ramp linearly from ``1`` (oldest row) to ``3``
      (newest row), so the fit favors recent drift without discarding the
      rest of the prefix. Positive means the offset is drifting *upward*
      (larger TVT relative to the interpolated formation depth) toward the
      end of the known zone. ``nan`` if fewer than 2 valid prefix rows or the
      weighted normal equations are degenerate.
    * ``wls_end`` -- the same fit's intercept ``a``: the offset extrapolated
      to the prefix's last row. ``nan`` under the same conditions as
      ``wls_slope_per100``.
    * ``n_valid_prefix`` -- valid-prefix row count for the winning formation.
    * ``prefix_rmse_best`` -- the winning formation's calibrated prefix RMSE
      (identical definition/value to ``SpatialResult.prefix_rmse`` for the
      same inputs).

    Never constructed with a mix of valid and NaN fields: on any failure
    (see :func:`offset_diagnostics`), every field is ``nan`` except
    ``n_valid_prefix`` (``0``) and ``prefix_rmse_best`` (``inf``).
    """

    offset_med: float
    offset_early: float
    offset_mid: float
    offset_late: float
    wls_slope_per100: float
    wls_end: float
    n_valid_prefix: int
    prefix_rmse_best: float


def _tail_weighted_wls(off: np.ndarray) -> tuple[float, float]:
    """Tail-weighted WLS linear fit ``off ~= a + b*t`` (see ``OffsetDiag``).

    ``t`` is ``0`` at the last element of ``off`` and decreases by ``1`` per
    element moving backward (``t <= 0``); weights ramp linearly from ``1``
    (first/oldest element) to ``3`` (last/newest element). Returns
    ``(slope_per100, intercept)`` where ``slope_per100 = b * 100``; both
    ``nan`` if ``off`` has fewer than 2 elements or the weighted normal
    equations are degenerate (near-singular or non-finite).
    """
    n = off.size
    if n < 2:
        return float("nan"), float("nan")

    t = np.arange(n, dtype=float) - (n - 1)
    w = np.linspace(1.0, 3.0, n)

    sw = float(np.sum(w))
    swt = float(np.sum(w * t))
    swtt = float(np.sum(w * t * t))
    swo = float(np.sum(w * off))
    swto = float(np.sum(w * t * off))

    det = sw * swtt - swt * swt
    if not np.isfinite(det) or abs(det) < 1e-12:
        return float("nan"), float("nan")

    a = (swo * swtt - swt * swto) / det
    b = (sw * swto - swt * swo) / det
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan"), float("nan")
    return float(b * 100.0), float(a)


def _offset_diag_fallback() -> OffsetDiag:
    nan = float("nan")
    return OffsetDiag(
        offset_med=nan,
        offset_early=nan,
        offset_mid=nan,
        offset_late=nan,
        wls_slope_per100=nan,
        wls_end=nan,
        n_valid_prefix=0,
        prefix_rmse_best=float("inf"),
    )


def _offset_diagnostics_impl(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    k: int,
    method: str,
) -> OffsetDiag:
    known = h["TVT_input"].notna().to_numpy()
    known_idx = np.flatnonzero(known)
    if known_idx.size == 0:
        raise ValueError("no known-zone anchor")

    prefix_idx = known_idx[-min(_PREFIX_WINDOW, known_idx.size) :]

    x = h["X"].to_numpy(dtype=float)
    y = h["Y"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    coords = np.column_stack([x[prefix_idx], y[prefix_idx]])
    exclude_code = bank.well_to_code.get(exclude_well, -1)
    interpolate = _INTERPOLATORS[method]

    best_off: np.ndarray | None = None
    best_rmse = float("inf")

    for f in bank.formations:
        depth_interp = interpolate(
            bank.trees[f], coords, bank.depths[f], bank.well_codes[f], exclude_code, k
        )
        valid_prefix = np.isfinite(depth_interp)
        if not np.any(valid_prefix):
            continue

        prefix_z = z[prefix_idx][valid_prefix]
        prefix_tvt = tvt_input[prefix_idx][valid_prefix]
        prefix_depth = depth_interp[valid_prefix]
        off = prefix_tvt + prefix_z - prefix_depth
        offset = float(np.median(off))

        prefix_pred = -prefix_z + prefix_depth + offset
        resid = prefix_tvt - prefix_pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        if not np.isfinite(rmse):
            continue

        if rmse < best_rmse:
            best_rmse = rmse
            best_off = off

    if best_off is None:
        raise ValueError("no formation produced a usable candidate")

    n_valid = int(best_off.size)
    offset_med = float(np.median(best_off))

    if n_valid >= 3:
        thirds = np.array_split(best_off, 3)
        offset_early = float(np.median(thirds[0]))
        offset_mid = float(np.median(thirds[1]))
        offset_late = float(np.median(thirds[2]))
    else:
        offset_early = offset_mid = offset_late = float("nan")

    wls_slope_per100, wls_end = _tail_weighted_wls(best_off)

    return OffsetDiag(
        offset_med=offset_med,
        offset_early=offset_early,
        offset_mid=offset_mid,
        offset_late=offset_late,
        wls_slope_per100=wls_slope_per100,
        wls_end=wls_end,
        n_valid_prefix=n_valid,
        prefix_rmse_best=best_rmse,
    )


def offset_diagnostics(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    k: int = _DEFAULT_K,
    method: str = "idw",
) -> OffsetDiag:
    """Per-well offset time-series diagnostics (R28: prefix-drift features).

    For the known-zone prefix (last ``_PREFIX_WINDOW`` rows with a known
    ``TVT_input``, exactly as in :func:`predict_spatial`), interpolates each
    ``bank`` formation's depth via leave-self-out KNN at *only* the prefix
    rows (the eval zone is never touched -- this function produces no
    prediction, only diagnostics, so skipping the eval-zone interpolation is
    a pure speedup), calibrates each formation's per-well offset exactly as
    :func:`predict_spatial` does, and picks the formation with the lowest
    prefix RMSE. Returns an :class:`OffsetDiag` describing how that winning
    formation's per-row offset evolves across the prefix (median, early/mid/
    late thirds, and a tail-weighted linear drift fit).

    Never raises: any failure (missing anchor, an unknown ``method``, no
    formation producing a usable candidate, or an internal error) falls back
    to an all-``nan`` :class:`OffsetDiag` (``n_valid_prefix=0``,
    ``prefix_rmse_best=inf``).
    """
    try:
        return _offset_diagnostics_impl(h, bank, exclude_well, k, method)
    except Exception:
        return _offset_diag_fallback()
