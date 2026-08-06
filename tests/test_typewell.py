"""Tests for typewell GR lookup and affine GR calibration in ``rogii.typewell``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii import typewell as TW


def make_typewell(tvt: list[float], gr: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": [None] * len(tvt)})


def make_known_zone_h(tvt_input: np.ndarray, gr: np.ndarray, n_eval_rows: int = 0) -> pd.DataFrame:
    """Horizontal-well DataFrame with only ``TVT_input``/``GR`` -- no ``TVT`` column at all.

    This intentionally omits the ``TVT`` column so that any accidental read of
    ground truth inside ``fit_affine_gr`` raises a ``KeyError`` rather than
    silently leaking.
    """
    tvt_input_full = np.concatenate([tvt_input, np.full(n_eval_rows, np.nan)])
    gr_full = np.concatenate([gr, np.zeros(n_eval_rows)])
    return pd.DataFrame({"TVT_input": tvt_input_full, "GR": gr_full})


# --------------------------------------------------------------------------- #
# tw_gr_lookup
# --------------------------------------------------------------------------- #


def test_tw_gr_lookup_interpolates_between_known_points() -> None:
    tw = make_typewell(tvt=[0.0, 10.0, 20.0], gr=[100.0, 110.0, 120.0])

    lookup = TW.tw_gr_lookup(tw)

    result = lookup(np.array([5.0, 15.0]))
    np.testing.assert_allclose(result, [105.0, 115.0])


def test_tw_gr_lookup_clips_out_of_range_queries_to_endpoints() -> None:
    tw = make_typewell(tvt=[0.0, 10.0, 20.0], gr=[100.0, 110.0, 120.0])

    lookup = TW.tw_gr_lookup(tw)

    result = lookup(np.array([-50.0, -5.0, 0.0, 20.0, 30.0, 1000.0]))
    np.testing.assert_allclose(result, [100.0, 100.0, 100.0, 120.0, 120.0, 120.0])


def test_tw_gr_lookup_drops_nan_gr_rows() -> None:
    tw = make_typewell(tvt=[0.0, 10.0, 20.0], gr=[100.0, float("nan"), 120.0])

    lookup = TW.tw_gr_lookup(tw)

    # with the NaN row dropped, interpolation spans 0 -> 20 directly
    result = lookup(np.array([10.0]))
    np.testing.assert_allclose(result, [110.0])


def test_tw_gr_lookup_sorts_unsorted_input() -> None:
    tw = make_typewell(tvt=[20.0, 0.0, 10.0], gr=[120.0, 100.0, 110.0])

    lookup = TW.tw_gr_lookup(tw)

    result = lookup(np.array([5.0, 15.0]))
    np.testing.assert_allclose(result, [105.0, 115.0])


# --------------------------------------------------------------------------- #
# fit_affine_gr
# --------------------------------------------------------------------------- #


def test_fit_affine_gr_falls_back_when_fewer_than_50_known_rows() -> None:
    tvt_grid = np.linspace(0.0, 200.0, 201)
    tw = make_typewell(tvt=list(tvt_grid), gr=list(100.0 + 0.1 * tvt_grid))

    n_known = 30  # below the 50-row threshold
    tvt_input = np.linspace(10.0, 190.0, n_known)
    tw_gr_at_known = 100.0 + 0.1 * tvt_input
    true_gain, true_offset = 2.0, 5.0
    gr_h = true_gain * tw_gr_at_known + true_offset

    h = make_known_zone_h(tvt_input, gr_h, n_eval_rows=5)

    gain, offset = TW.fit_affine_gr(h, tw)

    assert (gain, offset) == (1.0, 0.0)


def test_fit_affine_gr_recovers_gain_and_offset_with_enough_rows() -> None:
    tvt_grid = np.linspace(0.0, 200.0, 201)
    tw = make_typewell(tvt=list(tvt_grid), gr=list(100.0 + 0.1 * tvt_grid))

    n_known = 60  # at/above the 50-row threshold
    tvt_input = np.linspace(10.0, 190.0, n_known)
    tw_gr_at_known = 100.0 + 0.1 * tvt_input
    true_gain, true_offset = 2.0, 5.0
    gr_h = true_gain * tw_gr_at_known + true_offset

    h = make_known_zone_h(tvt_input, gr_h, n_eval_rows=5)

    gain, offset = TW.fit_affine_gr(h, tw)

    assert gain == pytest.approx(true_gain, abs=1e-6)
    assert offset == pytest.approx(true_offset, abs=1e-6)


def test_fit_affine_gr_falls_back_when_typewell_gr_is_degenerate() -> None:
    # typewell GR is constant everywhere -> twGR(TVT_input) has ~zero variance
    tvt_grid = np.linspace(0.0, 200.0, 201)
    tw = make_typewell(tvt=list(tvt_grid), gr=[100.0] * len(tvt_grid))

    n_known = 60
    tvt_input = np.linspace(10.0, 190.0, n_known)
    # h.GR varies, but that variation cannot be explained by the (constant) typewell GR
    gr_h = 100.0 + np.linspace(0.0, 10.0, n_known)

    h = make_known_zone_h(tvt_input, gr_h, n_eval_rows=5)

    gain, offset = TW.fit_affine_gr(h, tw)

    assert (gain, offset) == (1.0, 0.0)


def test_fit_affine_gr_falls_back_when_typewell_has_no_valid_gr_rows() -> None:
    tw = make_typewell(tvt=list(np.linspace(0.0, 200.0, 21)), gr=[float("nan")] * 21)

    n_known = 60
    tvt_input = np.linspace(10.0, 190.0, n_known)
    gr_h = 100.0 + np.linspace(0.0, 10.0, n_known)
    h = make_known_zone_h(tvt_input, gr_h, n_eval_rows=5)

    gain, offset = TW.fit_affine_gr(h, tw)

    assert (gain, offset) == (1.0, 0.0)


def test_fit_affine_gr_boundary_exactly_50_known_rows_does_not_fall_back() -> None:
    tvt_grid = np.linspace(0.0, 200.0, 201)
    tw = make_typewell(tvt=list(tvt_grid), gr=[100.0 + 0.1 * v for v in tvt_grid])

    n_known = 50  # exactly at the threshold
    tvt_input = np.linspace(10.0, 190.0, n_known)
    tw_gr_at_known = 100.0 + 0.1 * tvt_input
    true_gain, true_offset = 3.0, -2.0
    gr_h = true_gain * tw_gr_at_known + true_offset

    h = make_known_zone_h(tvt_input, gr_h, n_eval_rows=1)

    gain, offset = TW.fit_affine_gr(h, tw)

    assert gain == pytest.approx(true_gain, abs=1e-6)
    assert offset == pytest.approx(true_offset, abs=1e-6)


def test_fit_affine_gr_never_reads_tvt_column() -> None:
    # h has no "TVT" column at all; if the implementation ever touched it,
    # this would raise a KeyError instead of returning a result.
    tvt_grid = np.linspace(0.0, 200.0, 201)
    tw = make_typewell(tvt=list(tvt_grid), gr=list(100.0 + 0.1 * tvt_grid))

    n_known = 60
    tvt_input = np.linspace(10.0, 190.0, n_known)
    gr_h = 2.0 * (100.0 + 0.1 * tvt_input) + 5.0
    h = make_known_zone_h(tvt_input, gr_h, n_eval_rows=5)

    assert "TVT" not in h.columns
    gain, offset = TW.fit_affine_gr(h, tw)  # must not raise KeyError

    assert gain == pytest.approx(2.0, abs=1e-6)
    assert offset == pytest.approx(5.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# apply_affine
# --------------------------------------------------------------------------- #


def test_apply_affine_applies_gain_and_offset() -> None:
    gr = np.array([0.0, 10.0, 20.0])

    result = TW.apply_affine(gr, gain=2.0, offset=3.0)

    np.testing.assert_allclose(result, [3.0, 23.0, 43.0])


def test_apply_affine_identity_transform_is_a_no_op() -> None:
    gr = np.array([1.0, 2.0, 3.0])

    result = TW.apply_affine(gr, gain=1.0, offset=0.0)

    np.testing.assert_allclose(result, gr)
