"""Particle-filter (PF) GR registration tracker.

Source / inspiration
---------------------
The state-space design (particles carry a position in the ``TVT + Z`` frame
plus a ``rate`` -- i.e. ``d(TVT + Z)/dMD`` -- with ``pos += rate * dMD`` motion,
GR likelihood ``exp(-0.5 * ((GR - expected_GR) / gs) ** 2)`` against a typewell
GR lookup, and systematic resampling on low effective sample size) is adapted
from the public Kaggle notebook ``lightningv08/lb-7-776-rogii-ridge-sp``
(functions ``_pf_ancc`` / ``run_pf_ancc`` in
``lightningv08/lb-7-776-rogii-ridge-sp`` on Kaggle), which the notebook
``pixiux/rogii-dual-pipeline-blend`` reuses. The upstream notebook states no open-source license, so no code from it is
reproduced here: the implementation below is our own and only the algorithm
design is adapted.

The ``TVT + Z`` state choice is not cosmetic: empirically (see
``analysis/experiment_ledger.md`` and this module's own checks) the raw ``TVT``
trajectory can locally reverse direction near the known/eval-zone boundary
while the trajectory's ``Z`` (wellbore elevation) absorbs most of that
reversal, so ``TVT + Z`` drifts far more smoothly and is the more trackable
quantity for a slowly-varying ``rate`` state -- exactly the reference's design.

This reimplementation differs from the reference in a few ways: it is a
from-scratch, vectorized-over-particles NumPy port (no Numba dependency, to
avoid adding a build-time dependency to the submission notebook); the
per-well GR likelihood sigma is *not* recalibrated from the known zone
(``_gr_sig`` in the reference) but instead spans a fixed grid of scales
handed out to particles directly (see ``_assign_particle_scales``), mirroring
how ``registration/ncc.py``'s multi-scale ensembling assigns a scale per
candidate rather than per well; and multiple independent filter runs
(different seeds) are averaged at the output level in :func:`track_pf_multi`
for variance reduction, rather than simply growing a single filter's particle
count.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import data as D
from .. import typewell as TW

# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #
# Rate (d(TVT+Z)/dMD) calibration: fit over the last _RATE_CAL_ROWS known-zone
# rows via linear regression against MD. Empirically (see module docstring /
# experiment ledger) this local slope is tiny -- median ~0.01-0.04 ft/row --
# so a small number of rows is enough and using too many would blend in the
# (differently sloped) build section far from the eval-zone boundary.
_RATE_CAL_ROWS = 200

# Floor/fallback for the rate's initial spread (std) when the known-zone tail
# is too short or degenerate (e.g. constant MD) to estimate a local rate
# variability directly from finite differences.
_MIN_RATE_STD = 1e-4
_DEFAULT_RATE_STD = 0.02

# Particle process model, kept close to the reference notebook's calibrated
# ANCC particle-filter constants (``ANCC_ALPHA``/``ANCC_RN``/``ANCC_PN`` and
# ``ANCC_IS``/``ANCC_RP``/``ANCC_RR`` in the reference's cell 7).
_INIT_POS_STD = 0.3  # initial position (TVT+Z) spread across particles, ft
_RATE_MOMENTUM = 0.998  # AR(1)-style momentum applied to `rate` each row
_RATE_NOISE_STD = 0.002  # per-row Gaussian noise injected into `rate`
_POS_PROCESS_NOISE = 0.005  # per-row Gaussian noise injected into `pos`
_RESAMPLE_ROUGHEN_POS = 0.1  # jitter added to `pos` after resampling
_RESAMPLE_ROUGHEN_RATE = 0.001  # jitter added to `rate` after resampling

# GR likelihood: clip the squared, scaled residual before `exp` to avoid
# underflow/overflow for particles that have drifted far from a plausible
# match (mirrors the reference's `d*d < 600.` guard).
_MAX_SQUARED_RESIDUAL = 600.0
_MIN_SCALE = 1e-6

# Default GR-likelihood scale grid. Tuned empirically on a 129-well stratified
# train subset (see analysis/experiment_ledger.md): the affine-calibrated GR
# residual std over the known zone is ~10-16 (median ~12.7) on real wells, so
# a grid centered near/above that noise level -- (8, 12, 20, 30), pooled
# blend-0.7 RMSE 11.17 on the subset -- decisively beats both the sharper
# (3, 5, 8, 12) grid (13.54; treats GR noise as signal and chases it) and the
# wider (10, 15, 25, 40) grid (12.67; likelihood too flat, motion prior
# drifts). The reference notebook's per-well calibrated sigma (its `_gr_sig`,
# clipped to [10, 60]) was also measured and does not beat this fixed grid.
_DEFAULT_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)

# R6-b(4): opt-in per-well adaptive GR-likelihood scale grid (see
# ``_adaptive_gr_scale_grid``). ``a`` is the std of the known-zone
# affine-calibrated GR residual, clipped to ``_ADAPTIVE_SCALE_CLIP`` (cf. the
# sunnywu27 reference's per-well ``_gr_sig`` clip ``[10, 60]`` and this
# repo's measured median calibrated residual std ~12.7); the factor grid
# ``(0.6a, a, 1.6a)`` keeps :func:`_assign_particle_scales`'s multi-scale
# diversity while centering it on the well's own GR noise level instead of
# the one-size-fits-all ``_DEFAULT_SCALES``.
_ADAPTIVE_SCALE_FACTORS: tuple[float, ...] = (0.6, 1.0, 1.6)
_ADAPTIVE_SCALE_CLIP: tuple[float, float] = (8.0, 60.0)
# Below this many known-zone rows with both TVT_input and GR present, a
# residual-std estimate is too unreliable; fall back to the fixed grid.
_MIN_ADAPTIVE_RESID_ROWS = 10

# R19: opt-in spatial-prior position prior (see ``track_pf``'s
# ``spatial_prior_tvt`` / ``spatial_prior_sigma``). The default sigma (ft) is
# a conservative starting point for the caller-supplied, anchor-aligned
# formation-surface TVT curve; callers with a per-well fit-quality estimate
# (e.g. ``spatial_cache``'s ``prefix_rmse``) are expected to pass a tighter or
# looser sigma per well rather than rely on this default in production use.
_DEFAULT_SPATIAL_PRIOR_SIGMA = 10.0


@dataclass
class PFResult:
    """Per-row TVT estimate and particle-cloud uncertainty for the eval zone.

    Both arrays are aligned to ``data.eval_mask(h)`` order (length = number of
    evaluation-zone rows). ``std`` is the particle-weighted standard deviation
    of the TVT estimate (an uncertainty proxy, not an error bound); it is
    ``inf`` for every row on the exception-fallback path. ``n_updates`` counts
    how many eval-zone rows had a finite GR reading and therefore contributed
    a GR-likelihood reweighting step (rows with missing GR only propagate the
    motion model).
    """

    tvt: np.ndarray
    std: np.ndarray
    n_updates: int


def _flat_fallback(n: int, anchor: float) -> PFResult:
    return PFResult(
        tvt=np.full(n, anchor, dtype=float),
        std=np.full(n, np.inf, dtype=float),
        n_updates=0,
    )


def _safe_n_and_anchor(h: pd.DataFrame) -> tuple[int, float]:
    try:
        n = int(D.eval_mask(h).sum())
    except Exception:
        n = 0
    try:
        anchor = D.last_known_tvt(h)
    except Exception:
        anchor = 0.0
    return n, anchor


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized systematic resampling: draw ``N`` indices proportional to ``weights``."""
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating-point round-off leaving < 1.0
    return np.searchsorted(cumulative, positions)


def _assign_particle_scales(n_particles: int, scales: tuple[float, ...]) -> np.ndarray:
    """Round-robin assignment of a fixed GR-likelihood scale to each particle.

    This is the "multi-scale ensembling" analogue of ``registration/ncc.py``'s
    per-candidate scale scoring, but applied per-particle instead: particles
    seeded with a small ``gs`` commit to sharp, decisive GR matches, while
    particles with a large ``gs`` tolerate more GR mismatch. Resampling then
    lets whichever scales are actually working for this well's GR signal
    dominate the particle population over time, without having to pick one
    scale up front.
    """
    clipped = np.maximum(np.asarray(scales, dtype=float), _MIN_SCALE)
    return clipped[np.arange(n_particles) % clipped.size]


def _adaptive_gr_scale_grid(
    h: pd.DataFrame,
    lookup: Callable[[np.ndarray], np.ndarray],
    gain: float,
    offset: float,
) -> tuple[float, ...] | None:
    """Per-well GR-likelihood scale grid from the known zone's calibrated residual.

    Computes the residual series ``h.GR - apply_affine(twGR(TVT_input))`` over
    the known zone (rows where both ``TVT_input`` and ``GR`` are present --
    the same leak-safe row set :func:`typewell.fit_affine_gr` fits on, and
    always against the *real* GR, never a gap-filled series), takes its std
    as the well's own GR noise level ``a`` (clipped to
    ``_ADAPTIVE_SCALE_CLIP``; because the affine fit includes an offset term,
    the residual is mean-zero on these rows and its std equals its RMS), and
    returns the grid ``(0.6 * a, a, 1.6 * a)`` (``_ADAPTIVE_SCALE_FACTORS``).

    Returns ``None`` (caller keeps the fixed grid) when there are fewer than
    ``_MIN_ADAPTIVE_RESID_ROWS`` usable rows or the std is not finite.
    """
    known = h["TVT_input"].notna() & h["GR"].notna()
    if int(known.sum()) < _MIN_ADAPTIVE_RESID_ROWS:
        return None

    tvt_known = h.loc[known, "TVT_input"].to_numpy(dtype=float)
    gr_known = h.loc[known, "GR"].to_numpy(dtype=float)
    expected = TW.apply_affine(lookup(tvt_known), gain, offset)
    resid_std = float(np.std(gr_known - expected))
    if not np.isfinite(resid_std):
        return None

    a = float(np.clip(resid_std, *_ADAPTIVE_SCALE_CLIP))
    return tuple(factor * a for factor in _ADAPTIVE_SCALE_FACTORS)


def _fill_gr_gaps(raw_gr: np.ndarray, tw: pd.DataFrame) -> np.ndarray:
    """Interpolate NaN gaps in a well's raw GR series (opt-in, see ``fill_gr_gaps``).

    Ported from the public Kaggle notebook
    ``sunnywu27/rogii-wellbore-tvt-physical-model`` (``run_particle_filter``'s
    ``gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())``),
    which reports this halves its local score on wells with heavy eval-zone
    GR NaN (``000d7d20`` at 47% NaN is called out by name in that notebook).

    This module's trackers otherwise skip the GR-likelihood reweighting step
    entirely on a NaN row (the ``np.isfinite(gr_row)`` guard in
    :func:`_track_pf_impl` / :func:`_simulate_bma_seed`) -- the particle
    cloud free-diffuses through any GR gap under the motion model alone. This
    helper builds an alternative, gap-filled observation array so those rows
    *do* get a reweighting step instead: interior gaps are linearly
    interpolated against real neighbouring GR values (``limit_direction``
    ``'both'`` also extrapolates flat across any leading/trailing gap), and
    any remaining NaN (e.g. an entirely-NaN series) falls back to the
    typewell's own mean GR.

    Must be applied to the full ``h["GR"]`` series (known zone + eval zone,
    not just the eval-zone slice) so that interior eval-zone gaps interpolate
    against real known-zone GR values on both sides, matching the reference.
    Callers must apply this only to the *post-calibration* observation array
    fed to the likelihood loop -- :func:`typewell.fit_affine_gr` is always
    fit against ``h`` directly (real GR only), so gap-filling here can never
    leak into the affine calibration itself.
    """
    filled = pd.Series(raw_gr).interpolate(limit_direction="both")
    tw_gr_valid = tw["GR"].dropna()
    tw_mean = float(tw_gr_valid.mean()) if not tw_gr_valid.empty else np.nan
    return filled.fillna(tw_mean).to_numpy(dtype=float)


def _calibrate_rate(md: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    """Fit the local ``rate = d(pos)/dMD`` and its variability from a known-zone tail.

    ``pos`` is ``TVT_input + Z`` restricted to the last ``_RATE_CAL_ROWS`` known
    rows (``md`` likewise). Returns ``(rate_init, rate_init_std)``; falls back
    to ``(0.0, _DEFAULT_RATE_STD)`` when there are too few rows or ``md`` is
    constant (degenerate regression).
    """
    if md.size < 2 or np.ptp(md) <= 0:
        return 0.0, _DEFAULT_RATE_STD

    slope, _ = np.polyfit(md, pos, 1)

    d_md = np.diff(md)
    valid = d_md > 0
    if valid.sum() >= 2:
        local_rates = np.diff(pos)[valid] / d_md[valid]
        rate_std = float(np.std(local_rates))
    else:
        rate_std = _DEFAULT_RATE_STD

    return float(slope), max(rate_std, _MIN_RATE_STD)


def _track_pf_impl(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int,
    scales: tuple[float, ...],
    seed: int,
    resample_ess: float,
    fill_gr_gaps: bool = False,
    init_pos_std: float = _INIT_POS_STD,
    adaptive_gr_scales: bool = False,
    spatial_prior_tvt: np.ndarray | None = None,
    spatial_prior_sigma: float = _DEFAULT_SPATIAL_PRIOR_SIGMA,
) -> PFResult:
    mask = D.eval_mask(h)
    n_eval = int(mask.sum())
    if n_eval == 0:
        return PFResult(tvt=np.zeros(0, dtype=float), std=np.zeros(0, dtype=float), n_updates=0)

    eval_idx = np.flatnonzero(mask)
    anchor = D.last_known_tvt(h)

    known_notna = h["TVT_input"].notna().to_numpy()
    known_idx = np.flatnonzero(known_notna)
    if known_idx.size == 0:
        raise ValueError("well has no known TVT_input anchor")
    last_known_row = int(known_idx[-1])

    md = h["MD"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    raw_gr = h["GR"].to_numpy(dtype=float)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    z_last = float(z[last_known_row])
    pos_anchor = float(anchor) + z_last

    tail_n = min(_RATE_CAL_ROWS, known_idx.size)
    tail_idx = known_idx[-tail_n:]
    tail_md = md[tail_idx]
    tail_pos = tvt_input[tail_idx] + z[tail_idx]
    rate_init, rate_init_std = _calibrate_rate(tail_md, tail_pos)

    lookup = TW.tw_gr_lookup(tw)
    gain, offset = TW.fit_affine_gr(h, tw)  # calibration fit on real (unfilled) GR only

    if adaptive_gr_scales:
        adaptive = _adaptive_gr_scale_grid(h, lookup, gain, offset)  # real GR only
        if adaptive is not None:
            scales = adaptive

    if fill_gr_gaps:
        raw_gr = _fill_gr_gaps(raw_gr, tw)  # post-calibration observation stream only

    tw_valid = tw.dropna(subset=["TVT", "GR"])
    if tw_valid.empty:
        raise ValueError("typewell has no usable TVT/GR rows")
    tw_min = float(tw_valid["TVT"].min())
    tw_max = float(tw_valid["TVT"].max())
    if not (np.isfinite(tw_min) and np.isfinite(tw_max)) or tw_max <= tw_min:
        raise ValueError("typewell TVT range is degenerate")

    rng = np.random.default_rng(seed)
    n = int(n_particles)

    pos = np.full(n, pos_anchor) + init_pos_std * rng.standard_normal(n)
    rate = np.full(n, rate_init) + rate_init_std * rng.standard_normal(n)
    particle_scale = _assign_particle_scales(n, scales)
    weights = np.full(n, 1.0 / n)

    tvt_out = np.empty(n_eval, dtype=float)
    std_out = np.empty(n_eval, dtype=float)
    n_updates = 0
    prev_md = md[last_known_row]
    prior_sigma = max(float(spatial_prior_sigma), _MIN_SCALE)

    for k in range(n_eval):
        row = int(eval_idx[k])
        dm = md[row] - prev_md
        if not np.isfinite(dm) or dm <= 0:
            dm = 1.0
        prev_md = md[row]

        rate = _RATE_MOMENTUM * rate + _RATE_NOISE_STD * rng.standard_normal(n)
        pos = pos + rate * dm + _POS_PROCESS_NOISE * rng.standard_normal(n)

        z_row = z[row]
        tvt_particles = np.clip(pos - z_row, tw_min, tw_max)
        pos = tvt_particles + z_row

        gr_row = raw_gr[row]
        gr_valid = np.isfinite(gr_row)
        prior_valid = spatial_prior_tvt is not None and np.isfinite(spatial_prior_tvt[k])

        if gr_valid or prior_valid:
            log_lik = np.zeros(n)
            if gr_valid:
                expected_gr = TW.apply_affine(lookup(tvt_particles), gain, offset)
                resid = (gr_row - expected_gr) / particle_scale
                sq_resid = np.clip(resid * resid, 0.0, _MAX_SQUARED_RESIDUAL)
                log_lik += -0.5 * sq_resid
            if prior_valid:
                prior_resid = (tvt_particles - spatial_prior_tvt[k]) / prior_sigma
                prior_sq_resid = np.clip(prior_resid * prior_resid, 0.0, _MAX_SQUARED_RESIDUAL)
                log_lik += -0.5 * prior_sq_resid
            likelihood = np.exp(log_lik)
            weights = weights * likelihood
            weight_sum = weights.sum()
            if weight_sum > 0 and np.isfinite(weight_sum):
                weights = weights / weight_sum
            else:
                weights = np.full(n, 1.0 / n)
            if gr_valid:
                n_updates += 1

        mean_tvt = float(np.dot(weights, tvt_particles))
        var_tvt = float(np.dot(weights, (tvt_particles - mean_tvt) ** 2))
        tvt_out[k] = mean_tvt
        std_out[k] = float(np.sqrt(max(var_tvt, 0.0)))

        ess = 1.0 / float(np.sum(weights * weights))
        if ess < resample_ess * n:
            idx = _systematic_resample(weights, rng)
            pos = pos[idx] + _RESAMPLE_ROUGHEN_POS * rng.standard_normal(n)
            rate = rate[idx] + _RESAMPLE_ROUGHEN_RATE * rng.standard_normal(n)
            particle_scale = particle_scale[idx]
            weights = np.full(n, 1.0 / n)

    return PFResult(tvt=tvt_out, std=std_out, n_updates=n_updates)


def track_pf(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = 512,
    scales: tuple[float, ...] = _DEFAULT_SCALES,
    seed: int = 0,
    resample_ess: float = 0.5,
    fill_gr_gaps: bool = False,
    init_pos_std: float = _INIT_POS_STD,
    adaptive_gr_scales: bool = False,
    spatial_prior_tvt: np.ndarray | None = None,
    spatial_prior_sigma: float = _DEFAULT_SPATIAL_PRIOR_SIGMA,
) -> PFResult:
    """Sequential Monte Carlo (particle filter) GR registration tracker.

    Each particle carries a position in the ``TVT + Z`` frame and a ``rate``
    (``d(TVT + Z)/dMD``). Starting from ``data.last_known_tvt(h)`` (converted
    to the ``TVT + Z`` frame via the last known row's ``Z``), the tracker
    walks the evaluation zone (``data.eval_mask(h)`` order) one row at a time:

    1. **Propagate**: ``rate`` follows an AR(1)-style random walk (momentum
       ``0.998`` plus small Gaussian noise); ``pos`` advances by
       ``rate * dMD`` plus small Gaussian process noise. The resulting TVT
       candidate (``pos - Z``) is clipped to the typewell's TVT range.
    2. **Reweight**: if the row's GR reading is finite, each particle's TVT
       candidate is mapped through the typewell GR lookup and the fitted
       ``(gain, offset)`` affine calibration (:func:`typewell.fit_affine_gr`)
       to get an expected GR in the horizontal log's units; the Gaussian
       likelihood ``exp(-0.5 * ((GR - expected_GR) / gs) ** 2)`` reweights the
       particle, where ``gs`` is a per-particle scale drawn round-robin from
       ``scales`` (see :func:`_assign_particle_scales`). Rows with missing GR
       skip reweighting entirely (motion-model-only propagation).
    3. **Resample**: when the effective sample size (``1 / sum(w**2)``) drops
       below ``resample_ess * n_particles``, particles are redrawn via
       systematic resampling and lightly jittered ("roughened") to maintain
       diversity.

    The initial ``rate`` and its spread are calibrated from a linear
    regression of the last ``_RATE_CAL_ROWS`` known-zone rows' ``TVT_input + Z``
    against ``MD`` (see :func:`_calibrate_rate`) -- this is the "known-zone
    slope" prior described in the project's design notes, not a fixed
    constant.

    ``fill_gr_gaps`` (default ``False``, opt-in): when ``True``, the GR
    series fed to the per-row likelihood step is gap-filled (interpolated,
    remaining NaN filled with the typewell's mean GR; see
    :func:`_fill_gr_gaps`) *before* the tracking loop, so rows with missing
    raw GR still get a likelihood reweighting step instead of propagating
    motion-model-only. Calibration (:func:`typewell.fit_affine_gr`) always
    runs on the real, unfilled GR first regardless of this flag. Default
    ``False`` leaves every existing call site's behaviour unchanged.

    ``init_pos_std`` (default ``_INIT_POS_STD = 0.3``, the adopted value) is
    the standard deviation (ft, in the ``TVT + Z`` frame) of the particles'
    initial position spread around the anchor. The public reference notebook
    ``sunnywu27/rogii-wellbore-tvt-physical-model`` uses ``2.0`` here,
    arguing a wider initial spread helps wells with an abrupt TVT shift right
    at the known/eval boundary; the default keeps this module's adopted
    behaviour bit-identical.

    ``adaptive_gr_scales`` (default ``False``, opt-in): when ``True``,
    ``scales`` is replaced per well by :func:`_adaptive_gr_scale_grid` --
    a ``(0.6a, a, 1.6a)`` grid centered on the std ``a`` (clipped to
    ``[8, 60]``) of the known zone's affine-calibrated GR residual --
    whenever that estimate is available; otherwise the passed ``scales``
    grid is kept. Default ``False`` leaves every existing call site's
    behaviour unchanged.

    ``spatial_prior_tvt`` (default ``None``, opt-in, R19): an optional
    per-row spatial position prior over the evaluation zone, aligned to
    ``data.eval_mask(h)`` order (length must equal the eval-zone length --
    see the ``ValueError`` note below). Callers are expected to have already
    anchor-aligned this curve against the well's own known-zone anchor (e.g.
    ``spatial_cache``'s ``spatial_tvt - spatial_tvt[0] + anchor``) so it only
    contributes shape/slope information, not an absolute-position bias --
    this function performs no such alignment itself. At every eval-zone row
    ``i`` whose ``spatial_prior_tvt[i]`` is finite, a Gaussian log-likelihood
    term ``-0.5 * ((tvt_particle - spatial_prior_tvt[i]) / spatial_prior_sigma) ** 2``
    is added (in log-space, i.e. multiplied in probability space) to that
    row's GR log-likelihood before reweighting -- including rows whose GR
    reading is missing, where it is the *only* reweighting signal (this is
    the point: it lets the particle cloud stay anchored to the geological
    surface trend through GR gaps instead of free-diffusing). A non-finite
    ``spatial_prior_tvt[i]`` skips the prior term for that row only (matching
    a missing-GR row's existing skip behaviour). Reweighting from the prior
    alone does **not** increment ``PFResult.n_updates``, which keeps counting
    only GR-driven updates (its documented meaning). Default ``None`` leaves
    every existing call site's behaviour bit-identical (verified by test).

    ``spatial_prior_sigma`` (default ``10.0`` ft, only meaningful when
    ``spatial_prior_tvt`` is not ``None``): the prior's Gaussian width in ft,
    clipped to a small positive floor to avoid a division by zero. Callers
    with a per-well fit-quality estimate (e.g. ``spatial_cache``'s
    ``prefix_rmse``) should scale this per well rather than rely on the
    default in production use.

    This function never raises: any failure (malformed input, missing
    columns, a degenerate/too-short typewell, no known anchor, etc.) falls
    back to a flat anchor prediction with ``std`` set to ``inf`` for every
    row (signalling maximal uncertainty), matching
    ``baseline.predict_carry_last``. The one exception is a length mismatch
    between ``spatial_prior_tvt`` and the evaluation zone when
    ``spatial_prior_tvt`` is not ``None``: that is a caller/programmer error
    (not a data-quality issue), so it raises ``ValueError`` immediately
    instead of being swallowed into the flat-anchor fallback.

    Returns a :class:`PFResult` with arrays of length equal to the number of
    evaluation-zone rows, aligned to ``data.eval_mask(h)`` order.
    """
    if spatial_prior_tvt is not None:
        spatial_prior_tvt = np.asarray(spatial_prior_tvt, dtype=float)
        try:
            n_eval: int | None = int(D.eval_mask(h).sum())
        except Exception:
            # A malformed ``h`` is a data-quality issue, not a caller error:
            # defer to the flat-anchor fallback inside the try below.
            n_eval = None
        if n_eval is not None and spatial_prior_tvt.shape[0] != n_eval:
            raise ValueError(
                f"spatial_prior_tvt length ({spatial_prior_tvt.shape[0]}) does not match "
                f"the evaluation-zone length ({n_eval})"
            )

    try:
        return _track_pf_impl(
            h, tw, n_particles, scales, seed, resample_ess, fill_gr_gaps, init_pos_std,
            adaptive_gr_scales, spatial_prior_tvt, spatial_prior_sigma,
        )
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _flat_fallback(n, anchor)


def track_pf_multi(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_seeds: int = 8,
    fill_gr_gaps: bool = False,
    **kwargs: object,
) -> PFResult:
    """Average ``n_seeds`` independent :func:`track_pf` runs for variance reduction.

    Each seed is an independent particle cloud realization (independent
    resampling trajectories, not just more particles in one cloud), so
    averaging their outputs smooths out seed-specific mode collapse in a way
    a single larger filter would not. The combined ``tvt`` is the plain mean
    across seeds; the combined ``std`` follows the law of total variance --
    the mean of the per-seed (within-seed) variances plus the variance of the
    per-seed means (between-seed variance) -- so it reflects both each run's
    internal particle-cloud spread and how much the runs disagree with each
    other.

    ``n_updates`` is taken from the first run (identical across seeds: it only
    depends on which rows have finite GR, not on the random draws).

    ``fill_gr_gaps`` (default ``False``, opt-in) is forwarded unchanged to
    every per-seed :func:`track_pf` call -- see that function's docstring.

    ``spatial_prior_tvt`` / ``spatial_prior_sigma`` (R19, opt-in) are not
    named parameters here but flow through ``**kwargs`` to every per-seed
    :func:`track_pf` call unchanged (identical prior curve/sigma for every
    seed) -- see that function's docstring for the full contract, including
    that a ``spatial_prior_tvt`` length mismatch raises ``ValueError`` (from
    the first seed's call, propagating out of this function too, since it is
    a caller error rather than a data-quality issue).

    Never raises for data-quality reasons (each underlying :func:`track_pf`
    call has its own exception-safe fallback); an empty eval zone or an
    ``n_seeds <= 0`` yields an empty (or flat-fallback via the first call's
    fallback) result.
    """
    kwargs.pop("seed", None)  # track_pf_multi owns per-run seeding; ignore a stray override
    results = [
        track_pf(h, tw, seed=seed, fill_gr_gaps=fill_gr_gaps, **kwargs)
        for seed in range(max(int(n_seeds), 1))
    ]

    tvts = np.stack([r.tvt for r in results], axis=0)
    stds = np.stack([r.std for r in results], axis=0)

    mean_tvt = tvts.mean(axis=0)
    with np.errstate(invalid="ignore"):
        combined_var = np.mean(stds**2, axis=0) + np.var(tvts, axis=0)
    combined_std = np.sqrt(combined_var)

    return PFResult(tvt=mean_tvt, std=combined_std, n_updates=results[0].n_updates)


# --------------------------------------------------------------------------- #
# PF-BMA: likelihood-weighted Bayesian model averaging over independent seeds
# --------------------------------------------------------------------------- #
# Source / inspiration
# ---------------------
# This is a from-scratch, independent reimplementation of the *method* used by
# the public Kaggle notebook
# ``bernubritz/rogii-lb7295-public-rebuild`` (cell 27's ``_pf_lik_allseeds`` /
# ``lik_pf``, driven by cell 17/23's ``CFG.PF_SEEDS = 128`` and
# ``CFG.PF_SCALES = (3., 5., 8., 12.)``), not a port of its code (the
# reference is Numba-JIT'd; this module intentionally avoids a Numba
# dependency, same rationale as :func:`track_pf` above). The upstream notebook states no open-source license, so no code from it is
# reproduced here: the implementation below is our own and only the algorithm
# design is adapted.
#
# The reference's design, reimplemented here: run ``n_seeds`` *independent*
# particle-filter realizations over the eval zone (different RNG streams, not
# just more particles in one cloud -- as in :func:`track_pf_multi`). Each
# seed accumulates its own cumulative log-likelihood -- the standard
# sequential-Monte-Carlo incremental marginal-likelihood estimator,
# ``log p(y_1..t) - log p(y_1..t-1) = log( sum_j w_j(t-1) * lik_j(t) )`` at
# every row with a finite GR reading -- alongside its per-row TVT path. A
# grid of "scales" (here ``(3, 5, 8, 12)``, the reference's ``PF_SCALES``) is
# then used as a softmax temperature over each seed's *final* cumulative
# log-likelihood, producing one likelihood-weighted consensus path per scale:
# seeds whose particle cloud tracked the GR signal well end up with higher
# cumulative log-likelihood and dominate the softmax-weighted average, while
# seeds that lost the signal (mode collapse onto the wrong branch) are
# down-weighted instead of diluting a plain mean (as :func:`track_pf_multi`
# does).
#
# Two deliberate differences from the reference (beyond the Numba-free port):
#
# 1. **Naming disambiguation.** The reference reuses one name ("scale") for
#    two different things: the per-row GR-likelihood Gaussian width (its
#    ``gs``, calibrated once per well from the known zone) and the softmax
#    temperature over seeds' cumulative log-likelihood (its ``PF_SCALES``).
#    Those are unrelated quantities at unrelated magnitudes (GR units vs.
#    cumulative log-probability units), so this module keeps them as
#    separate, clearly-named parameters: ``gr_scales`` (per-particle
#    GR-likelihood scale) and ``bma_scales`` (the softmax-temperature grid --
#    the actual hyperparameter under test here).
# 2. **Within-seed GR-likelihood scale.** Rather than the reference's
#    single per-well calibrated ``gs`` (its ``_gr_sig``, clipped to
#    ``[10, 60]``), each seed's particle swarm reuses :func:`track_pf`'s
#    already-adopted per-particle round-robin scale grid
#    (``_DEFAULT_SCALES = (8, 12, 20, 30)``, see that constant's docstring
#    for the empirical justification) via :func:`_assign_particle_scales`.
#    This keeps the within-seed particle dynamics identical to the
#    already-measured-and-adopted :func:`track_pf`, isolating this module's
#    novel contribution to exactly the outer seed-BMA layer being tested.
#
# Each seed's TVT estimate is read off *before* the row's ESS-triggered
# resampling step (matching :func:`track_pf`'s own estimate ordering above,
# for internal consistency), a minor deliberate deviation from the reference
# (which estimates after resampling).

# Softmax-temperature grid for seed-BMA over cumulative log-likelihood
# (mirrors the reference's ``CFG.PF_SCALES``). This is *not* a GR-likelihood
# scale -- see the module-docstring block above.
_DEFAULT_BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
_DEFAULT_BMA_N_PARTICLES = 512
_DEFAULT_BMA_N_SEEDS = 8

# Floor applied to a row's predictive likelihood (``sum_j w_j * lik_j``)
# before taking ``log`` -- avoids ``-inf`` when every particle's GR match is
# implausible for a given seed (mirrors the reference's ``if avg_lk<1e-300``
# guard).
_MIN_AVG_LIKELIHOOD = 1e-300


@dataclass
class PFBMAResult:
    """Per-scale seed-BMA consensus paths + per-seed diagnostics for the eval zone.

    ``tvt_by_scale`` and ``std_by_scale`` are keyed by the softmax-temperature
    values in ``bma_scales`` (see :func:`track_pf_bma`); every array is
    aligned to ``data.eval_mask(h)`` order (length = number of
    evaluation-zone rows), matching the :class:`PFResult` contract. For a
    given scale, ``tvt_by_scale[scale]`` is the softmax(``log_lik / scale``)
    weighted average of the ``n_seeds`` independent seeds' full paths, and
    ``std_by_scale[scale]`` is the corresponding weighted between-seed
    standard deviation at each row (an uncertainty proxy driven by how much
    the seeds disagree once weighted by that scale, not a particle-cloud
    spread). ``log_lik`` holds each seed's *final* cumulative
    log-likelihood (length ``n_seeds``, diagnostic only -- e.g. a small
    spread across seeds signals the GR signal is decisive and any scale in
    the grid should agree; a large spread signals scale choice matters more).
    ``n_updates`` counts eval-zone rows with a finite GR reading (identical
    across seeds, like :class:`PFResult`).

    On the exception-fallback path, every scale's ``tvt`` is a flat anchor,
    every scale's ``std`` is ``inf``, and ``log_lik`` is ``-inf`` for every
    (nominal) seed.
    """

    tvt_by_scale: dict[float, np.ndarray]
    std_by_scale: dict[float, np.ndarray]
    log_lik: np.ndarray
    n_updates: int


def _flat_fallback_bma(
    n: int, anchor: float, scales: tuple[float, ...], n_seeds: int
) -> PFBMAResult:
    flat = np.full(n, anchor, dtype=float)
    inf_arr = np.full(n, np.inf, dtype=float)
    return PFBMAResult(
        tvt_by_scale={scale: flat.copy() for scale in scales},
        std_by_scale={scale: inf_arr.copy() for scale in scales},
        log_lik=np.full(max(int(n_seeds), 1), -np.inf, dtype=float),
        n_updates=0,
    )


@dataclass
class _BMAWellPrep:
    """Once-per-well setup shared by every PF-BMA seed (does not depend on the
    per-seed random draws): rate calibration, GR lookup/affine calibration,
    typewell TVT range. Computing this once (rather than ``n_seeds`` times,
    as a naive per-seed call to :func:`track_pf`'s internals would) keeps
    PF-BMA's wall-clock cost close to ``n_seeds`` times a single
    :func:`track_pf` run, not ``n_seeds`` times a well's full setup+run cost.
    """

    md: np.ndarray
    z: np.ndarray
    raw_gr: np.ndarray
    eval_idx: np.ndarray
    n_eval: int
    last_known_row: int
    pos_anchor: float
    rate_init: float
    rate_init_std: float
    lookup: object
    gain: float
    offset: float
    tw_min: float
    tw_max: float


def _prepare_bma_well(
    h: pd.DataFrame, tw: pd.DataFrame, fill_gr_gaps: bool = False
) -> _BMAWellPrep:
    mask = D.eval_mask(h)
    n_eval = int(mask.sum())
    eval_idx = np.flatnonzero(mask)
    anchor = D.last_known_tvt(h)

    known_notna = h["TVT_input"].notna().to_numpy()
    known_idx = np.flatnonzero(known_notna)
    if known_idx.size == 0:
        raise ValueError("well has no known TVT_input anchor")
    last_known_row = int(known_idx[-1])

    md = h["MD"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    raw_gr = h["GR"].to_numpy(dtype=float)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    z_last = float(z[last_known_row])
    pos_anchor = float(anchor) + z_last

    tail_n = min(_RATE_CAL_ROWS, known_idx.size)
    tail_idx = known_idx[-tail_n:]
    tail_md = md[tail_idx]
    tail_pos = tvt_input[tail_idx] + z[tail_idx]
    rate_init, rate_init_std = _calibrate_rate(tail_md, tail_pos)

    lookup = TW.tw_gr_lookup(tw)
    gain, offset = TW.fit_affine_gr(h, tw)  # calibration fit on real (unfilled) GR only

    if fill_gr_gaps:
        raw_gr = _fill_gr_gaps(raw_gr, tw)  # post-calibration observation stream only

    tw_valid = tw.dropna(subset=["TVT", "GR"])
    if tw_valid.empty:
        raise ValueError("typewell has no usable TVT/GR rows")
    tw_min = float(tw_valid["TVT"].min())
    tw_max = float(tw_valid["TVT"].max())
    if not (np.isfinite(tw_min) and np.isfinite(tw_max)) or tw_max <= tw_min:
        raise ValueError("typewell TVT range is degenerate")

    return _BMAWellPrep(
        md=md,
        z=z,
        raw_gr=raw_gr,
        eval_idx=eval_idx,
        n_eval=n_eval,
        last_known_row=last_known_row,
        pos_anchor=pos_anchor,
        rate_init=rate_init,
        rate_init_std=rate_init_std,
        lookup=lookup,
        gain=gain,
        offset=offset,
        tw_min=tw_min,
        tw_max=tw_max,
    )


def _simulate_bma_seed(
    prep: _BMAWellPrep,
    seed: int,
    n_particles: int,
    gr_scales: tuple[float, ...],
    resample_ess: float,
    init_pos_std: float = _INIT_POS_STD,
    pos_process_noise: float = _POS_PROCESS_NOISE,
    rate_noise_std: float = _RATE_NOISE_STD,
    rate_momentum: float = _RATE_MOMENTUM,
    resample_roughen_pos: float = _RESAMPLE_ROUGHEN_POS,
) -> tuple[np.ndarray, float, int]:
    """One independent particle-cloud realization: TVT path + cumulative log-likelihood.

    Mechanically identical to a single :func:`track_pf` run (same motion
    model, same per-particle round-robin GR-likelihood scale via
    :func:`_assign_particle_scales`, same systematic resampling), with one
    addition: at every finite-GR row, the *pre-update* weights' dot product
    with that row's likelihood vector (``avg_lk = sum_j w_j * lik_j``, with
    ``sum_j w_j == 1`` going in) is accumulated in log-space as this seed's
    running evidence -- see the module-docstring block above for why this is
    the standard SMC incremental marginal-likelihood identity.
    """
    n_eval = prep.n_eval
    if n_eval == 0:
        return np.zeros(0, dtype=float), 0.0, 0

    rng = np.random.default_rng(seed)
    n = int(n_particles)

    pos = np.full(n, prep.pos_anchor) + init_pos_std * rng.standard_normal(n)
    rate = np.full(n, prep.rate_init) + prep.rate_init_std * rng.standard_normal(n)
    particle_scale = _assign_particle_scales(n, gr_scales)
    weights = np.full(n, 1.0 / n)

    tvt_out = np.empty(n_eval, dtype=float)
    log_lik = 0.0
    n_updates = 0
    prev_md = prep.md[prep.last_known_row]

    for k in range(n_eval):
        row = int(prep.eval_idx[k])
        dm = prep.md[row] - prev_md
        if not np.isfinite(dm) or dm <= 0:
            dm = 1.0
        prev_md = prep.md[row]

        rate = rate_momentum * rate + rate_noise_std * rng.standard_normal(n)
        pos = pos + rate * dm + pos_process_noise * rng.standard_normal(n)

        z_row = prep.z[row]
        tvt_particles = np.clip(pos - z_row, prep.tw_min, prep.tw_max)
        pos = tvt_particles + z_row

        gr_row = prep.raw_gr[row]
        if np.isfinite(gr_row):
            expected_gr = TW.apply_affine(prep.lookup(tvt_particles), prep.gain, prep.offset)
            resid = (gr_row - expected_gr) / particle_scale
            sq_resid = np.clip(resid * resid, 0.0, _MAX_SQUARED_RESIDUAL)
            likelihood = np.exp(-0.5 * sq_resid)

            avg_lk = float(np.dot(weights, likelihood))  # weights sum to 1.0 going in
            log_lik += float(np.log(max(avg_lk, _MIN_AVG_LIKELIHOOD)))

            unnorm = weights * likelihood
            weight_sum = unnorm.sum()
            if weight_sum > 0 and np.isfinite(weight_sum):
                weights = unnorm / weight_sum
            else:
                weights = np.full(n, 1.0 / n)
            n_updates += 1

        tvt_out[k] = float(np.dot(weights, tvt_particles))

        ess = 1.0 / float(np.sum(weights * weights))
        if ess < resample_ess * n:
            idx = _systematic_resample(weights, rng)
            pos = pos[idx] + resample_roughen_pos * rng.standard_normal(n)
            rate = rate[idx] + _RESAMPLE_ROUGHEN_RATE * rng.standard_normal(n)
            particle_scale = particle_scale[idx]
            weights = np.full(n, 1.0 / n)

    return tvt_out, log_lik, n_updates


def _track_pf_bma_impl(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int,
    n_seeds: int,
    bma_scales: tuple[float, ...],
    gr_scales: tuple[float, ...],
    seed_base: int,
    resample_ess: float,
    fill_gr_gaps: bool = False,
    init_pos_std: float = _INIT_POS_STD,
    adaptive_gr_scales: bool = False,
    pos_process_noise: float = _POS_PROCESS_NOISE,
    rate_noise_std: float = _RATE_NOISE_STD,
    rate_momentum: float = _RATE_MOMENTUM,
    resample_roughen_pos: float = _RESAMPLE_ROUGHEN_POS,
) -> PFBMAResult:
    prep = _prepare_bma_well(h, tw, fill_gr_gaps)
    n_seeds = max(int(n_seeds), 1)

    if adaptive_gr_scales:
        # Computed once per well (shared by every seed), against the *real* GR
        # (`h` directly), never the (optionally) gap-filled stream in `prep`.
        adaptive = _adaptive_gr_scale_grid(h, prep.lookup, prep.gain, prep.offset)
        if adaptive is not None:
            gr_scales = adaptive

    if prep.n_eval == 0:
        return PFBMAResult(
            tvt_by_scale={scale: np.zeros(0, dtype=float) for scale in bma_scales},
            std_by_scale={scale: np.zeros(0, dtype=float) for scale in bma_scales},
            log_lik=np.zeros(n_seeds, dtype=float),
            n_updates=0,
        )

    paths = np.empty((n_seeds, prep.n_eval), dtype=float)
    log_liks = np.empty(n_seeds, dtype=float)
    n_updates = 0
    for s in range(n_seeds):
        tvt_path, log_lik, n_upd = _simulate_bma_seed(
            prep,
            seed_base + s,
            n_particles,
            gr_scales,
            resample_ess,
            init_pos_std,
            pos_process_noise=pos_process_noise,
            rate_noise_std=rate_noise_std,
            rate_momentum=rate_momentum,
            resample_roughen_pos=resample_roughen_pos,
        )
        paths[s] = tvt_path
        log_liks[s] = log_lik
        if s == 0:
            n_updates = n_upd

    ll_shifted = log_liks - log_liks.max()  # softmax numerical stability, cancels in the ratio

    tvt_by_scale: dict[float, np.ndarray] = {}
    std_by_scale: dict[float, np.ndarray] = {}
    for scale in bma_scales:
        w = np.exp(ll_shifted / float(scale))
        w_sum = w.sum()
        if w_sum > 0 and np.isfinite(w_sum):
            w = w / w_sum
        else:
            w = np.full(n_seeds, 1.0 / n_seeds)
        weighted_mean = w @ paths
        weighted_var = w @ ((paths - weighted_mean) ** 2)
        tvt_by_scale[scale] = weighted_mean
        std_by_scale[scale] = np.sqrt(np.maximum(weighted_var, 0.0))

    return PFBMAResult(
        tvt_by_scale=tvt_by_scale,
        std_by_scale=std_by_scale,
        log_lik=log_liks,
        n_updates=n_updates,
    )


def track_pf_bma(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = _DEFAULT_BMA_N_PARTICLES,
    n_seeds: int = _DEFAULT_BMA_N_SEEDS,
    bma_scales: tuple[float, ...] = _DEFAULT_BMA_SCALES,
    gr_scales: tuple[float, ...] = _DEFAULT_SCALES,
    seed_base: int = 0,
    resample_ess: float = 0.5,
    fill_gr_gaps: bool = False,
    init_pos_std: float = _INIT_POS_STD,
    adaptive_gr_scales: bool = False,
    pos_process_noise: float = _POS_PROCESS_NOISE,
    rate_noise_std: float = _RATE_NOISE_STD,
    rate_momentum: float = _RATE_MOMENTUM,
    resample_roughen_pos: float = _RESAMPLE_ROUGHEN_POS,
) -> PFBMAResult:
    """Likelihood-weighted PF-BMA: seed-softmax consensus over independent PF runs.

    Runs ``n_seeds`` independent :func:`track_pf`-style particle filters
    (different RNG streams; see :func:`_simulate_bma_seed`), each
    accumulating its own cumulative GR-likelihood evidence alongside its
    per-row TVT path. For every temperature in ``bma_scales``, the seeds are
    combined via ``softmax(log_lik / scale)`` weights into one consensus
    path (and a between-seed weighted std) -- see the module-docstring block
    above (search ``PF-BMA``) for the full method description, its relation
    to the public reference notebook, and the two deliberate naming/design
    differences from it.

    ``fill_gr_gaps`` (default ``False``, opt-in): when ``True``, every seed's
    GR observation stream is gap-filled (see :func:`_fill_gr_gaps`) once per
    well (in :func:`_prepare_bma_well`, shared across all ``n_seeds`` runs)
    before the tracking loops, so rows with missing raw GR still contribute a
    likelihood-reweighting step for every seed instead of propagating
    motion-model-only. Calibration (:func:`typewell.fit_affine_gr`) always
    runs on the real, unfilled GR first regardless of this flag. Default
    ``False`` leaves every existing call site's behaviour unchanged.

    ``init_pos_std`` (default ``_INIT_POS_STD = 0.3``, the adopted value) is
    the standard deviation (ft, in the ``TVT + Z`` frame) of every seed's
    initial particle-position spread around the anchor -- forwarded unchanged
    to each :func:`_simulate_bma_seed` run; see :func:`track_pf`'s docstring
    for the reference-notebook rationale (its value is ``2.0``). The default
    keeps existing behaviour bit-identical.

    ``adaptive_gr_scales`` (default ``False``, opt-in): when ``True``,
    ``gr_scales`` is replaced per well by :func:`_adaptive_gr_scale_grid`
    (a ``(0.6a, a, 1.6a)`` grid centered on the known zone's calibrated GR
    residual std ``a``, clipped to ``[8, 60]``; computed once per well and
    shared by every seed) whenever that estimate is available; otherwise the
    passed ``gr_scales`` grid is kept. This only affects the *within-seed*
    per-particle likelihood scales -- the seed-BMA softmax over
    ``bma_scales`` is untouched. Default ``False`` leaves every existing
    call site's behaviour unchanged.

    ``pos_process_noise`` / ``rate_noise_std`` / ``rate_momentum`` /
    ``resample_roughen_pos`` (R6-b(5)): opt-in overrides for the particle
    process-model constants (defaults are the module-level
    ``_POS_PROCESS_NOISE`` / ``_RATE_NOISE_STD`` / ``_RATE_MOMENTUM`` /
    ``_RESAMPLE_ROUGHEN_POS`` inherited from the common-ancestor reference
    notebook, never systematically tuned before). Forwarded unchanged to
    every :func:`_simulate_bma_seed` run; the defaults keep existing
    behaviour bit-identical.

    This function never raises: any failure (malformed input, missing
    columns, a degenerate/too-short typewell, no known anchor, etc.) falls
    back to a flat-anchor prediction for every scale, with ``std`` set to
    ``inf`` and ``log_lik`` set to ``-inf`` for every seed (maximal
    uncertainty), matching :func:`track_pf`'s fallback contract.

    Returns a :class:`PFBMAResult`; every per-row array is aligned to
    ``data.eval_mask(h)`` order.
    """
    try:
        return _track_pf_bma_impl(
            h, tw, n_particles, n_seeds, bma_scales, gr_scales, seed_base, resample_ess,
            fill_gr_gaps, init_pos_std, adaptive_gr_scales,
            pos_process_noise=pos_process_noise,
            rate_noise_std=rate_noise_std,
            rate_momentum=rate_momentum,
            resample_roughen_pos=resample_roughen_pos,
        )
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _flat_fallback_bma(n, anchor, bma_scales, n_seeds)
