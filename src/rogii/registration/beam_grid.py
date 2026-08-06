"""Parameterized beam-DP configs for a candidate-path bank (design.md R2).

Extends ``registration/beam.py`` with two axes that module does not expose:
GR affine calibration on/off, and a pre-match smoothing radius on the
horizontal GR series. Both are genuine *observation-changing* axes -- unlike
``move_penalty``/``mismatch_scale``, whose product is the DP's only effective
stiffness knob (see ``beam.py``'s module docstring: rescaling the whole
per-well cost by ``mismatch_scale`` leaves the argmin path unchanged, so
varying the ratio alone is a redundant, path-identical config). Calibration
and smoothing instead change the *observation sequence itself* fed to the
same cost formula, so they produce genuinely different Viterbi paths even at
a fixed cost product.

This module does not modify ``beam.py`` (existing functions there are frozen
per task instructions). It reuses the two private helpers that do the actual
work -- ``_build_grid`` (anchor-centered TVT state grid) and
``_forward_viterbi`` (the exact full-grid DP) -- so the DP kernel itself is
not duplicated, only the observation-preparation step around it.

Used by ``scripts/run_beam_grid_cv.py`` to build a multi-configuration
candidate-path bank (``outputs/path_bank_beam.npz``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import data as D
from .. import typewell as TW
from .beam import BeamResult, _build_grid, _forward_viterbi

_DEFAULT_GRID_STEP_FT = 0.2
_DEFAULT_GRID_RANGE_FT = 120.0


@dataclass(frozen=True)
class BeamGridConfig:
    """One candidate-bank beam parameterization.

    ``move_penalty`` * ``mismatch_scale`` is the DP's effective stiffness
    (see module docstring); ``gr_calibration`` and ``smooth_radius`` are the
    two additional observation-changing axes this module adds on top of
    ``registration.beam``.
    """

    label: str
    move_penalty: float
    mismatch_scale: float
    max_move_per_row: int = 3
    gr_calibration: bool = True
    smooth_radius: int = 0
    grid_step_ft: float = _DEFAULT_GRID_STEP_FT
    grid_range_ft: float = _DEFAULT_GRID_RANGE_FT

    @property
    def product(self) -> float:
        return self.move_penalty * self.mismatch_scale


def _smooth_gr(gr: np.ndarray, radius: int) -> np.ndarray:
    """Centered rolling-mean smoothing with a ``2*radius+1`` window.

    ``min_periods=1`` means a position is only left ``NaN`` if *every*
    sample in its window is ``NaN`` -- an isolated missing GR reading gets
    filled from its smoothed neighbors instead of staying a "free" (cost-0)
    row in the DP, which is the intended behavior for a smoothing axis (it
    also mildly gap-fills, unlike the un-smoothed path where
    ``_forward_viterbi`` treats each non-finite GR row as paying only the
    move penalty). ``radius <= 0`` is a no-op passthrough.
    """
    if radius <= 0:
        return gr
    window = 2 * radius + 1
    return (
        pd.Series(gr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _empty_result() -> BeamResult:
    return BeamResult(tvt=np.zeros(0, dtype=float), margin=np.zeros(0, dtype=float), total_cost=0.0)


def _flat_fallback(n: int, anchor: float) -> BeamResult:
    return BeamResult(
        tvt=np.full(n, anchor, dtype=float), margin=np.zeros(n, dtype=float), total_cost=0.0
    )


def _track_beam_grid_impl(h: pd.DataFrame, tw: pd.DataFrame, config: BeamGridConfig) -> BeamResult:
    n_eval = int(D.eval_mask(h).sum())
    if n_eval == 0:
        return _empty_result()

    anchor = D.last_known_tvt(h)
    eval_idx = np.flatnonzero(D.eval_mask(h))

    grid_tvt, lo, hi = _build_grid(tw, anchor, config.grid_step_ft, config.grid_range_ft)
    n_states = grid_tvt.size
    grid_gr = TW.tw_gr_lookup(tw)(grid_tvt)

    raw_gr = h["GR"].to_numpy(dtype=float)
    if config.gr_calibration:
        gain, offset = TW.fit_affine_gr(h, tw)
        gr_cal = TW.apply_affine(raw_gr, gain, offset)
    else:
        gr_cal = raw_gr

    gr_cal = _smooth_gr(gr_cal, config.smooth_radius)
    gr_eval = gr_cal[eval_idx]

    anchor_idx = int(round((float(np.clip(anchor, lo, hi)) - lo) / config.grid_step_ft))
    anchor_idx = int(np.clip(anchor_idx, 0, n_states - 1))
    init_cost = config.move_penalty * np.abs(np.arange(n_states, dtype=float) - anchor_idx)

    max_move = max(int(config.max_move_per_row), 1)
    path, margin, total_cost = _forward_viterbi(
        gr_eval, grid_gr, init_cost, config.move_penalty, config.mismatch_scale, max_move
    )

    return BeamResult(tvt=grid_tvt[path], margin=margin, total_cost=total_cost)


def track_beam_grid(h: pd.DataFrame, tw: pd.DataFrame, config: BeamGridConfig) -> BeamResult:
    """Run one :class:`BeamGridConfig` of the full-grid Viterbi DP.

    Never raises: any failure falls back to a flat-anchor prediction with
    ``margin`` all zero and ``total_cost`` ``0.0``, matching
    ``registration.beam.track_beam``.
    """
    try:
        return _track_beam_grid_impl(h, tw, config)
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
