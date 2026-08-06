"""Baseline TVT predictors for the eval zone of a horizontal well.

All predictors take the horizontal-well DataFrame and return an array of TVT
predictions aligned to the eval-zone rows (``data.eval_mask(h)`` order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D


def predict_carry_last(h: pd.DataFrame) -> np.ndarray:
    """Carry the last known TVT (``TVT_input`` anchor) flat through the eval zone.

    Strong because the lateral TVT is nearly flat; it misses the slow drift of
    the wellbore relative to the geology (which registration methods recover).
    """
    n = int(D.eval_mask(h).sum())
    return np.full(n, D.last_known_tvt(h), dtype=float)
