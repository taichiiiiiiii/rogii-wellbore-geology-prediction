"""Typewell GR lookup and affine GR calibration between horizontal well and typewell.

These helpers stay strictly on the "safe" side of the leak boundary: only
``TVT_input`` (the known-zone depth-registration column, present in both
train and test) is ever read from the horizontal-well DataFrame. The
ground-truth ``TVT`` column must never be touched here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# Below this many known-zone rows, an affine fit is considered unreliable.
_MIN_VALID_ROWS = 50

# If the typewell GR sampled at the known TVT_input values is essentially
# constant, a least-squares fit is degenerate (undefined gain); fall back.
_MIN_GR_STD = 1e-6


def tw_gr_lookup(tw: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
    """Build a ``TVT -> GR`` interpolator from a typewell DataFrame.

    Rows with a missing ``GR`` are dropped and the remainder is sorted by
    ``TVT`` before interpolation. The returned callable uses ``np.interp``,
    so queries outside the typewell's TVT range are clipped to the nearest
    endpoint value rather than extrapolated.
    """
    valid = tw.dropna(subset=["GR"]).sort_values("TVT")
    tvt = valid["TVT"].to_numpy(dtype=float)
    gr = valid["GR"].to_numpy(dtype=float)

    def lookup(query_tvt: np.ndarray) -> np.ndarray:
        return np.interp(query_tvt, tvt, gr)

    return lookup


def fit_affine_gr(h: pd.DataFrame, tw: pd.DataFrame) -> tuple[float, float]:
    """Fit ``h.GR ~= gain * twGR(TVT_input) + offset`` over the known zone.

    Only rows where ``TVT_input`` and ``GR`` are both known (i.e. outside the
    eval zone) are used, and only ``TVT_input`` is read from ``h`` -- never
    ``TVT``. Falls back to the identity transform ``(1.0, 0.0)`` when there
    are too few known rows or the typewell GR sampled at those depths is
    degenerate (near-constant), since a least-squares fit would be unreliable
    or undefined in those cases.
    """
    known = h["TVT_input"].notna() & h["GR"].notna()
    if int(known.sum()) < _MIN_VALID_ROWS:
        return 1.0, 0.0

    tw_valid = tw.dropna(subset=["GR"])
    if tw_valid.empty:
        return 1.0, 0.0

    tvt_input = h.loc[known, "TVT_input"].to_numpy(dtype=float)
    gr_h = h.loc[known, "GR"].to_numpy(dtype=float)

    lookup = tw_gr_lookup(tw)
    gr_tw = lookup(tvt_input)

    if np.std(gr_tw) < _MIN_GR_STD:
        return 1.0, 0.0

    gain, offset = np.polyfit(gr_tw, gr_h, 1)
    return float(gain), float(offset)


def apply_affine(gr: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """Apply the ``(gain, offset)`` affine transform to a GR array."""
    return gain * np.asarray(gr, dtype=float) + offset
