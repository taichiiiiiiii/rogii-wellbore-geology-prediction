"""Tests for the Layer2 GBDT feature builder in ``rogii.features``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rogii import data as D
from rogii.features import build_features


def make_typewell(tvt: list[float], gr: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def make_h(
    n_known: int,
    n_eval: int,
    *,
    gr_known: list[float] | None = None,
    gr_eval: list[float] | None = None,
    start_md: float = 0.0,
) -> pd.DataFrame:
    """Synthetic horizontal-well DataFrame with only the test-time schema.

    Only ``MD, X, Y, Z, GR, TVT_input`` are present -- deliberately no
    ``TVT`` column and no train-only formation columns, so that any
    accidental read of ground truth or formation depths inside
    ``build_features`` raises a ``KeyError`` rather than silently leaking.
    """
    n = n_known + n_eval
    md = start_md + np.arange(n, dtype=float)
    x = np.arange(n, dtype=float) * 0.5
    y = np.arange(n, dtype=float) * 0.3
    z = -md + 100.0

    if gr_known is None:
        gr_known = list(100.0 + 0.05 * np.arange(n_known))
    if gr_eval is None:
        gr_eval = list(100.0 + 0.05 * np.arange(n_known, n))
    gr = np.array(list(gr_known) + list(gr_eval), dtype=float)

    known_tvt = 10_000.0 + 0.1 * np.arange(n_known)
    tvt_input = np.concatenate([known_tvt, np.full(n_eval, np.nan)])

    return pd.DataFrame({"MD": md, "X": x, "Y": y, "Z": z, "GR": gr, "TVT_input": tvt_input})


# --------------------------------------------------------------------------- #
# (a) no TVT / formation columns -- leak-structure guard
# --------------------------------------------------------------------------- #


def test_build_features_works_without_tvt_or_formation_columns() -> None:
    h = make_h(n_known=300, n_eval=50)
    tw = make_typewell(
        tvt=list(10_000.0 + 0.1 * np.arange(400)), gr=list(100.0 + 0.05 * np.arange(400))
    )

    assert "TVT" not in h.columns
    for formation_col in ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]:
        assert formation_col not in h.columns

    out = build_features(h, tw)  # must not raise KeyError

    assert len(out) == 50
    assert "TVT" not in out.columns


# --------------------------------------------------------------------------- #
# (b) row count == eval row count
# --------------------------------------------------------------------------- #


def test_build_features_row_count_matches_eval_mask() -> None:
    h = make_h(n_known=150, n_eval=37)
    tw = make_typewell(
        tvt=list(10_000.0 + 0.1 * np.arange(200)), gr=list(100.0 + 0.05 * np.arange(200))
    )

    out = build_features(h, tw)

    assert len(out) == int(D.eval_mask(h).sum())
    assert len(out) == 37


# --------------------------------------------------------------------------- #
# (c) no inf values
# --------------------------------------------------------------------------- #


def test_build_features_has_no_infinite_values() -> None:
    # degenerate typewell (constant GR -> degenerate affine fit) and a flat
    # known MD/TVT segment, both of which are prone to producing inf/nan from
    # divide-by-zero if not guarded.
    h = make_h(n_known=80, n_eval=20)
    tw = make_typewell(tvt=list(10_000.0 + 0.1 * np.arange(100)), gr=[100.0] * 100)

    out = build_features(h, tw)

    values = out.to_numpy(dtype=float)
    assert not np.isinf(values).any()


# --------------------------------------------------------------------------- #
# (d) boundary: eval zone of 1 row, known zone of 2 rows
# --------------------------------------------------------------------------- #


def test_build_features_boundary_one_eval_row_two_known_rows() -> None:
    h = make_h(n_known=2, n_eval=1)
    tw = make_typewell(tvt=[9_999.0, 10_000.0, 10_001.0], gr=[95.0, 100.0, 105.0])

    out = build_features(h, tw)  # must not raise

    assert len(out) == 1
    assert not np.isinf(out.to_numpy(dtype=float)).any()


# --------------------------------------------------------------------------- #
# (e) gr_missing correctly flags NaN GR at eval rows
# --------------------------------------------------------------------------- #


def test_build_features_gr_missing_flags_nan_gr_rows() -> None:
    n_known, n_eval = 100, 10
    gr_eval = [100.0] * n_eval
    gr_eval[2] = float("nan")
    gr_eval[7] = float("nan")

    h = make_h(n_known=n_known, n_eval=n_eval, gr_eval=gr_eval)
    tw = make_typewell(
        tvt=list(10_000.0 + 0.1 * np.arange(n_known + n_eval)),
        gr=list(100.0 + 0.05 * np.arange(n_known + n_eval)),
    )

    out = build_features(h, tw)

    expected_missing = np.zeros(n_eval)
    expected_missing[[2, 7]] = 1.0
    np.testing.assert_array_equal(out["gr_missing"].to_numpy(), expected_missing)
    # the raw GR column itself should carry NaN through (not silently 0-filled)
    assert np.isnan(out["gr"].to_numpy()[2])
    assert np.isnan(out["gr"].to_numpy()[7])
