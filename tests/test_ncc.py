"""Tests for the multi-scale NCC registration tracker in ``rogii.registration.ncc``.

Note on the "recovers drift" tests: a real typewell GR curve is rich, noisy,
and often locally repetitive (thin-bed cyclicity), which makes short-window
normalized cross-correlation a genuinely weak, ambiguous localization signal in
practice (confirmed empirically against real train wells while building this
module -- see ``scripts/run_ncc_cv.py``'s pooled-RMSE report). To keep these
unit tests meaningful without depending on that hard, ambiguous regime, the
synthetic typewell here uses one clean, isolated landmark (a single Gaussian
bump on a flat baseline) that the drifting horizontal log passes near/through.
This validates the tracker's *mechanics* (it correctly locks onto an
unambiguous, identifiable feature and follows it) independently of whether raw
NCC beats ``carry_last`` on real, ambiguous geology -- which it typically does
not, and which is reported honestly rather than asserted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.registration.ncc import NCCResult, track_ncc


def _typewell_gr(tvt: np.ndarray, spike_center: float = 260.0) -> np.ndarray:
    """Flat baseline plus one isolated, unambiguous Gaussian landmark.

    A single well-separated bump (no periodic structure) gives normalized
    cross-correlation a unique, identifiable target: away from the bump the
    log is flat and uninformative (matches are low-confidence and roughly
    equally poor everywhere), but near the bump there is exactly one place
    that looks like it.
    """
    base = np.full_like(np.asarray(tvt, dtype=float), 60.0)
    return base + 40.0 * np.exp(-0.5 * ((tvt - spike_center) / 3.0) ** 2)


def _make_synthetic_well(
    n_known: int = 200,
    n_eval: int = 200,
    anchor: float = 250.0,
    drift_max_ft: float = 12.0,
    tw_noise_std: float = 0.3,
    gr_noise_std: float = 0.5,
    nan_frac: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build a synthetic (horizontal, typewell) pair with a known TVT drift curve.

    The known zone sits flat at ``anchor`` (matching ``TVT_input``); the eval
    zone ramps linearly toward (and through) the typewell's landmark bump.
    Returns ``(h, tw, true_eval_tvt)``.
    """
    rng = np.random.default_rng(seed)

    tw_tvt = np.arange(0.0, 500.0, 0.5)
    tw_gr = _typewell_gr(tw_tvt) + rng.normal(0.0, tw_noise_std, size=tw_tvt.size)
    tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})

    i_eval = np.arange(n_eval)
    drift = drift_max_ft * (i_eval / max(n_eval - 1, 1))
    true_eval_tvt = anchor + drift

    known_tvt = np.full(n_known, anchor)
    all_true_tvt = np.concatenate([known_tvt, true_eval_tvt])
    gr_clean = _typewell_gr(all_true_tvt) + rng.normal(0.0, gr_noise_std, size=all_true_tvt.size)

    if nan_frac > 0:
        drop = rng.random(gr_clean.size) < nan_frac
        gr_clean = np.where(drop, np.nan, gr_clean)

    tvt_input = np.concatenate([known_tvt, np.full(n_eval, np.nan)])
    md = np.arange(n_known + n_eval, dtype=float)
    h = pd.DataFrame({"MD": md, "GR": gr_clean, "TVT_input": tvt_input})

    return h, tw, true_eval_tvt


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- #
# core recovery behaviour
# --------------------------------------------------------------------------- #


def test_track_ncc_recovers_known_drift_better_than_carry_last() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well()
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result = track_ncc(h, tw)

    assert isinstance(result, NCCResult)
    assert result.tvt.shape == true_eval_tvt.shape
    assert result.confidence.shape == true_eval_tvt.shape

    ncc_rmse = _rmse(result.tvt, true_eval_tvt)
    carry_last_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)

    assert carry_last_rmse > 1.0  # sanity: the synthetic drift is non-trivial
    assert ncc_rmse < 0.5 * carry_last_rmse


def test_track_ncc_tolerates_missing_gr_and_still_beats_carry_last() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(nan_frac=0.3, seed=1)
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result = track_ncc(h, tw)

    ncc_rmse = _rmse(result.tvt, true_eval_tvt)
    carry_last_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)

    assert ncc_rmse < carry_last_rmse
    assert np.all(np.isfinite(result.tvt))
    assert np.all((result.confidence >= 0.0) & (result.confidence <= 1.0))


def test_track_ncc_max_step_bounds_row_to_row_movement() -> None:
    h, tw, _ = _make_synthetic_well()
    max_step_ft = 0.2

    result = track_ncc(h, tw, max_step_ft=max_step_ft)

    steps = np.abs(np.diff(result.tvt))
    assert np.all(steps <= max_step_ft + 1e-9)


def test_track_ncc_confidence_zero_when_gr_missing_at_row() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=50)
    # Force every eval-zone GR reading to NaN so every row must skip its update.
    h = h.copy()
    known_count = h["TVT_input"].notna().sum()
    h.loc[known_count:, "GR"] = np.nan

    result = track_ncc(h, tw)

    assert np.all(result.confidence == 0.0)
    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert np.all(result.tvt == anchor)


# --------------------------------------------------------------------------- #
# fallback / exception-safety behaviour
# --------------------------------------------------------------------------- #


def test_track_ncc_all_nan_gr_falls_back_to_flat_anchor() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=40)
    h = h.copy()
    h["GR"] = np.nan

    result = track_ncc(h, tw)

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert np.all(result.tvt == anchor)
    assert np.all(result.confidence == 0.0)


def test_track_ncc_never_raises_on_degenerate_typewell() -> None:
    h, _, _ = _make_synthetic_well(n_eval=20)
    tiny_tw = pd.DataFrame({"TVT": [100.0], "GR": [55.0]})

    result = track_ncc(h, tiny_tw)  # must not raise

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert result.tvt.shape == (20,)
    np.testing.assert_allclose(result.tvt, anchor)
    np.testing.assert_allclose(result.confidence, 0.0)


def test_track_ncc_never_raises_on_missing_columns() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=20)
    h_missing_gr = h.drop(columns=["GR"])

    result = track_ncc(h_missing_gr, tw)  # must not raise despite missing GR column

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert result.tvt.shape == (20,)
    np.testing.assert_allclose(result.tvt, anchor)
    np.testing.assert_allclose(result.confidence, 0.0)


def test_track_ncc_never_raises_on_completely_malformed_input() -> None:
    empty_h = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    empty_tw = pd.DataFrame({"also_wrong": [1, 2, 3]})

    result = track_ncc(empty_h, empty_tw)  # must not raise

    assert isinstance(result, NCCResult)
    assert result.tvt.shape == result.confidence.shape


def test_track_ncc_empty_eval_zone_returns_empty_arrays() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=0)

    result = track_ncc(h, tw)

    assert result.tvt.shape == (0,)
    assert result.confidence.shape == (0,)


@pytest.mark.parametrize("half_widths", [(8, 15, 25), (5,), (3, 10)])
def test_track_ncc_accepts_various_half_width_configs(half_widths: tuple[int, ...]) -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(n_eval=80)

    result = track_ncc(h, tw, half_widths=half_widths)

    assert result.tvt.shape == true_eval_tvt.shape
    assert np.all(np.isfinite(result.tvt))
