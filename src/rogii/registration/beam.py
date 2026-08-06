"""Anchor-seeded, full-grid Viterbi-DP GR registration tracker ("beam" tracker).

Source / inspiration
---------------------
The cost shape -- a per-row squared GR-mismatch term plus a linear
move-penalty term, tracked one horizontal-log row at a time from the
``last_known_tvt`` anchor over a typewell-derived candidate grid -- is
adapted from the public Kaggle notebook ``lightningv08/lb-7-776-rogii-ridge-sp``
(functions ``beam_search`` / ``_beam_jit`` in
``lightningv08/lb-7-776-rogii-ridge-sp`` on Kaggle) and the closely
related ``nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based``
(same cost shape; that notebook's ``_grid`` helper resamples the typewell GR
onto a uniform ``step=0.2`` ft grid, which is where this module's default
``grid_step_ft`` comes from). Those notebooks state no open-source license, so no
code from them is reproduced here: the implementation below is our own and only
the algorithm design is adapted.

Those references keep only a small, pruned *beam* of the best ``bs``
candidate states per row (an approximate, top-k-truncated search) over that
grid, with transitions restricted to +/-2 index steps and cost
``mc * |d| + (gr - tw_gr[state])**2 / es``.

This module reimplements the same cost shape but keeps the **full** state
grid at every row instead of a pruned top-k beam: an exact forward Viterbi
dynamic program, not an approximate/truncated beam search. The grid (anchor
+/- ``grid_range_ft`` at ``grid_step_ft`` resolution) has on the order of
1,000-2,000 states, small enough that the full DP -- vectorized over states
with numpy at every row -- runs in a few seconds per well even for the
longest evaluation zones in this dataset (see ``scripts/run_beam_cv.py`` for
measured timing). GR is calibrated onto the typewell's scale via
``typewell.fit_affine_gr`` (leak-safe: only the known zone's ``TVT_input`` is
read), never raw.

Honest empirical note: an earlier, unrelated sequential tracker
(``registration/ncc.py``, windowed normalized cross-correlation) found GR to
be a weak, often-ambiguous localization signal on this dataset -- see that
module's docstring and ``scripts/run_ncc_cv.py`` for the measured pooled
RMSE, which came in well above ``carry_last``. This DP is a different
algorithm (a global per-row-mismatch + move-penalty optimization over a full
state grid, not a local windowed correlation), but it draws on the same
underlying GR signal, so the default move penalty is kept comparatively
strong -- biasing the tracker toward "don't move without GR evidence" rather
than chasing GR's weak per-row discriminative power. Whether that is enough
to beat ``carry_last`` on this dataset is an empirical question answered
honestly in ``scripts/run_beam_cv.py``'s pooled-RMSE report, not asserted
here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import data as D
from .. import typewell as TW

# Grid defaults: anchor +/- grid_range_ft at grid_step_ft resolution. The
# 0.2 ft step matches both reference notebooks' typewell-grid resolution
# (``_grid(..., step=0.2)``); the 120 ft range comfortably covers the
# observed drift (median ~11 ft, max ~104 ft per well -- see
# ``analysis/experiment_ledger.md``) with headroom.
_DEFAULT_GRID_STEP_FT = 0.2
_DEFAULT_GRID_RANGE_FT = 120.0

# Move-penalty / mismatch-scale defaults. A structural property of the DP
# cost ``mismatch/mismatch_scale + move_penalty*|d|``: multiplying the whole
# per-well cost by ``mismatch_scale`` leaves the argmin path unchanged, so
# the only effective stiffness knob is the *product*
# ``move_penalty * mismatch_scale`` (confirmed empirically: 60-well configs
# with equal products, e.g. 20*150 vs 25*120, produced pooled RMSE identical
# to ~1e-8). Subset scans put the optimum somewhere in products ~3000-4500
# (a 60-well subset favored ~3000; a 194-well stratified sample favored
# ~4500 -- subsample-based tuning is unstable here), but the decisive
# measurements are the two *full 773-well* runs in
# ``scripts/run_beam_cv.py``: the {2400, 3000, 3750}-product ensemble
# measured pooled 15.8000 raw / 15.7232 at blend w=0.7, strictly better
# than the {3750, 4500, 5250} ensemble (15.8627 raw / 15.8002 blend), so
# the former is kept. Both scan tails degrade as expected: GR-trusting
# products collapse (640 -> 13.28, 200 -> 24.06 on the easy subset) and
# very stiff products converge to carry_last. The 194-well scan confirmed
# the affine GR calibration helps (no-cal 15.071 vs 15.042) and that grid
# step is a secondary knob. ``max_move_per_row`` in 2-4 was nearly
# irrelevant (true motion is sub-ft/row).
_DEFAULT_MOVE_PENALTY = 20.0
_DEFAULT_MISMATCH_SCALE = 150.0
_DEFAULT_MAX_MOVE_PER_ROW = 3

# Anything below this many finite typewell TVT samples cannot define a
# usable grid.
_MIN_TYPEWELL_SAMPLES = 2


@dataclass
class BeamResult:
    """Per-row TVT estimate and DP diagnostics for the evaluation zone.

    Both ``tvt`` and ``margin`` are aligned to ``data.eval_mask(h)`` order
    (length = number of evaluation-zone rows).

    ``margin`` is, for each row, the gap between the best and second-best
    accumulated cost across the *entire* state grid at that row (not
    necessarily the two states on the final backtracked path) -- a measure
    of how sharply the DP's cost surface picks out a state at that row.
    Larger margin means a more decisive frontier. Note: an earlier NCC
    tracker's analogous confidence score empirically did *not* predict
    per-row correctness on this dataset, so this margin should likewise be
    read as a DP-internal diagnostic, not asserted to correlate with error.

    ``total_cost`` is the minimum accumulated cost at the final row (the
    Viterbi path's total cost); it is not comparable across wells with
    different eval-zone lengths.
    """

    tvt: np.ndarray
    margin: np.ndarray
    total_cost: float


@dataclass(frozen=True)
class BeamConfig:
    """One parameterization of :func:`track_beam`, plus its ensemble weight."""

    grid_step_ft: float = _DEFAULT_GRID_STEP_FT
    grid_range_ft: float = _DEFAULT_GRID_RANGE_FT
    move_penalty: float = _DEFAULT_MOVE_PENALTY
    mismatch_scale: float = _DEFAULT_MISMATCH_SCALE
    max_move_per_row: int = _DEFAULT_MAX_MOVE_PER_ROW
    weight: float = 1.0


# A small 3-config default ensemble at stiffness products 2400 / 3000 /
# 3750 -- the best *measured full-773-well* configuration (see the comment
# above _DEFAULT_MOVE_PENALTY; the stiffer {3750, 4500, 5250} alternative
# suggested by a 194-well subsample measured worse on the full set). Since
# only the product matters, mismatch_scale is held at 150 and move_penalty
# varied. In the spirit of the reference's 14-config ``BEAM_CONFIGS``
# ensemble but sized down per the task's "3-7 configs to start" guidance;
# a wider conservative-heavy 5-config spread measured *worse* than a tight
# bracket around the optimum on the 60-well subset, so the ensemble is
# kept tight.
DEFAULT_BEAM_CONFIGS: tuple[BeamConfig, ...] = (
    BeamConfig(move_penalty=16.0, mismatch_scale=150.0, max_move_per_row=3, weight=1.0),
    BeamConfig(move_penalty=20.0, mismatch_scale=150.0, max_move_per_row=3, weight=1.0),
    BeamConfig(move_penalty=25.0, mismatch_scale=150.0, max_move_per_row=3, weight=1.0),
)


def _empty_result() -> BeamResult:
    return BeamResult(tvt=np.zeros(0, dtype=float), margin=np.zeros(0, dtype=float), total_cost=0.0)


def _flat_fallback(n: int, anchor: float) -> BeamResult:
    return BeamResult(
        tvt=np.full(n, anchor, dtype=float), margin=np.zeros(n, dtype=float), total_cost=0.0
    )


def _build_grid(
    tw: pd.DataFrame, anchor: float, grid_step_ft: float, grid_range_ft: float
) -> tuple[np.ndarray, float, float]:
    """Anchor-centered TVT grid, clipped to the typewell's finite TVT extent.

    Returns ``(grid_tvt, lo, hi)``. Raises ``ValueError`` if the typewell has
    too few finite ``TVT`` samples or a degenerate (zero-width) range; the
    caller (``track_beam``) treats that as any other failure and falls back
    to a flat anchor prediction.
    """
    tw_tvt = tw["TVT"].to_numpy(dtype=float)
    tw_tvt = tw_tvt[np.isfinite(tw_tvt)]
    if tw_tvt.size < _MIN_TYPEWELL_SAMPLES:
        raise ValueError("typewell has too few finite TVT samples to build a grid")

    tw_min, tw_max = float(tw_tvt.min()), float(tw_tvt.max())
    if tw_max <= tw_min:
        raise ValueError("typewell TVT range is degenerate")

    lo = max(tw_min, anchor - grid_range_ft)
    hi = min(tw_max, anchor + grid_range_ft)
    if hi <= lo:  # anchor sits entirely outside the typewell's TVT extent
        lo, hi = tw_min, tw_max

    n_states = max(int(round((hi - lo) / grid_step_ft)) + 1, 3)
    grid_tvt = lo + np.arange(n_states, dtype=float) * grid_step_ft
    return grid_tvt, lo, hi


def _forward_viterbi(
    gr_eval: np.ndarray,
    grid_gr: np.ndarray,
    init_cost: np.ndarray,
    move_penalty: float,
    mismatch_scale: float,
    max_move: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Full-grid forward Viterbi pass.

    ``cost[i, s] = min_{|d| <= max_move}(cost[i-1, s-d] + move_penalty*|d|) + mismatch(i, s)``,
    with ``cost[-1, :] = init_cost`` (the anchor prior) and
    ``mismatch(i, s) = (gr_eval[i] - grid_gr[s])**2 / mismatch_scale`` (or
    ``0`` when ``gr_eval[i]`` is not finite, so NaN GR rows only pay the
    move penalty). Returns ``(path_idx, row_margin, total_cost)`` where
    ``path_idx`` (length ``n_rows``) indexes into ``grid_gr`` / the grid's
    TVT axis, and ``row_margin[i]`` is the best-vs-second-best cost gap in
    row ``i``'s full frontier (see ``BeamResult.margin``).
    """
    n_rows = gr_eval.shape[0]
    n_states = grid_gr.shape[0]

    # penalty[k] = move_penalty * |d| where d = max_move - k, for the k-th row
    # of the sliding-window view built below (see inline comment at its use).
    penalty = move_penalty * np.abs(np.arange(-max_move, max_move + 1, dtype=float))

    backptr = np.empty((n_rows, n_states), dtype=np.int32)
    margin = np.zeros(n_rows, dtype=float)

    prev_cost = init_cost
    padded = np.empty(n_states + 2 * max_move, dtype=float)
    state_idx = np.arange(n_states)

    for i in range(n_rows):
        padded[:] = np.inf
        padded[max_move : max_move + n_states] = prev_cost
        # window[k] = padded[k : k + n_states]; padded[max_move + j] =
        # prev_cost[j], so window[k][s] = prev_cost[s - (max_move - k)] --
        # i.e. row k supplies the candidate for move d = max_move - k.
        window = np.lib.stride_tricks.sliding_window_view(padded, n_states)
        candidates = window + penalty[:, None]
        best_k = np.argmin(candidates, axis=0)
        best_cost = candidates[best_k, state_idx]
        backptr[i] = state_idx - (max_move - best_k)

        gv = gr_eval[i]
        row_mismatch = (gv - grid_gr) ** 2 / mismatch_scale if np.isfinite(gv) else 0.0

        cost_i = best_cost + row_mismatch
        if n_states >= 2:
            part = np.partition(cost_i, 1)
            margin[i] = float(part[1] - part[0])
        prev_cost = cost_i

    path = np.empty(n_rows, dtype=np.int64)
    path[-1] = int(np.argmin(prev_cost))
    total_cost = float(prev_cost[path[-1]])
    for i in range(n_rows - 1, 0, -1):
        path[i - 1] = backptr[i, path[i]]

    return path, margin, total_cost


def _track_beam_impl(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    grid_step_ft: float,
    grid_range_ft: float,
    move_penalty: float,
    mismatch_scale: float,
    max_move_per_row: int,
) -> BeamResult:
    n_eval = int(D.eval_mask(h).sum())
    if n_eval == 0:
        return _empty_result()

    anchor = D.last_known_tvt(h)
    eval_idx = np.flatnonzero(D.eval_mask(h))

    grid_tvt, lo, hi = _build_grid(tw, anchor, grid_step_ft, grid_range_ft)
    n_states = grid_tvt.size

    grid_gr = TW.tw_gr_lookup(tw)(grid_tvt)

    gain, offset = TW.fit_affine_gr(h, tw)
    gr_cal = TW.apply_affine(h["GR"].to_numpy(dtype=float), gain, offset)
    gr_eval = gr_cal[eval_idx]

    anchor_idx = int(round((float(np.clip(anchor, lo, hi)) - lo) / grid_step_ft))
    anchor_idx = int(np.clip(anchor_idx, 0, n_states - 1))
    init_cost = move_penalty * np.abs(np.arange(n_states, dtype=float) - anchor_idx)

    max_move = max(int(max_move_per_row), 1)
    path, margin, total_cost = _forward_viterbi(
        gr_eval, grid_gr, init_cost, move_penalty, mismatch_scale, max_move
    )

    return BeamResult(tvt=grid_tvt[path], margin=margin, total_cost=total_cost)


def track_beam(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    grid_step_ft: float = _DEFAULT_GRID_STEP_FT,
    grid_range_ft: float = _DEFAULT_GRID_RANGE_FT,
    move_penalty: float = _DEFAULT_MOVE_PENALTY,
    mismatch_scale: float = _DEFAULT_MISMATCH_SCALE,
    max_move_per_row: int = _DEFAULT_MAX_MOVE_PER_ROW,
) -> BeamResult:
    """Register a horizontal well's GR log onto the typewell TVT axis via full-grid DP.

    Builds a TVT state grid centered on ``data.last_known_tvt(h)`` spanning
    ``+/- grid_range_ft`` at ``grid_step_ft`` resolution (clipped to the
    typewell's finite TVT extent), then runs an exact forward Viterbi DP
    (see ``_forward_viterbi``) over ``data.eval_mask(h)``'s rows in order:
    each row's cost is a squared mismatch between that row's affine-
    calibrated GR (``typewell.fit_affine_gr``, fit leak-safely on the known
    zone only) and the grid state's typewell GR, plus a move penalty
    linear in how many grid steps (up to ``max_move_per_row``) the state
    shifted from the previous row. Rows with non-finite GR pay only the
    move penalty (mismatch cost 0). The path is chosen by backtracking from
    the minimum-cost final state.

    This function never raises: any failure (malformed input, a typewell
    too short to build a grid, no eval zone, no anchor, etc.) falls back to
    a flat-anchor prediction with ``margin`` all zero and ``total_cost``
    ``0.0``, matching ``baseline.predict_carry_last``.

    Returns a :class:`BeamResult` whose arrays have length equal to the
    number of evaluation-zone rows, aligned to ``data.eval_mask(h)`` order.
    """
    try:
        return _track_beam_impl(
            h, tw, grid_step_ft, grid_range_ft, move_penalty, mismatch_scale, max_move_per_row
        )
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


def _track_beam_multi_impl(
    h: pd.DataFrame, tw: pd.DataFrame, configs: Sequence[BeamConfig]
) -> BeamResult:
    results = [
        track_beam(
            h,
            tw,
            grid_step_ft=c.grid_step_ft,
            grid_range_ft=c.grid_range_ft,
            move_penalty=c.move_penalty,
            mismatch_scale=c.mismatch_scale,
            max_move_per_row=c.max_move_per_row,
        )
        for c in configs
    ]

    weights = np.array([c.weight for c in configs], dtype=float)
    wsum = weights.sum()
    weights = weights / wsum if wsum > 0 else np.full(len(configs), 1.0 / len(configs))

    tvt_stack = np.stack([r.tvt for r in results])
    margin_stack = np.stack([r.margin for r in results])
    tvt = np.tensordot(weights, tvt_stack, axes=(0, 0))
    margin = np.tensordot(weights, margin_stack, axes=(0, 0))
    total_cost = float(np.dot(weights, np.array([r.total_cost for r in results])))

    return BeamResult(tvt=tvt, margin=margin, total_cost=total_cost)


def track_beam_multi(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    configs: Sequence[BeamConfig] | None = None,
) -> BeamResult:
    """Weighted-average ensemble of :func:`track_beam` over several ``configs``.

    Defaults to :data:`DEFAULT_BEAM_CONFIGS` (a small conservative/loose,
    GR-trusting/distrusting spread) when ``configs`` is ``None``. Each
    config's :class:`BeamResult` is combined via its ``weight`` (normalized
    to sum to 1; equal weight if all weights are non-positive).

    Never raises: any failure (including inside an individual config's
    ``track_beam`` call bubbling up from something outside its own
    fallback, or a malformed ``configs`` sequence) falls back to a flat
    anchor prediction, same as :func:`track_beam`.
    """
    try:
        resolved = configs if configs is not None else DEFAULT_BEAM_CONFIGS
        return _track_beam_multi_impl(h, tw, resolved)
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
