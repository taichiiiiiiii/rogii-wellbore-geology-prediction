"""Tests for the full-grid Viterbi-DP tracker in ``rogii.registration.beam``.

Note on the "recovers drift" test: as with ``tests/test_ncc.py``, real
typewell GR is a genuinely weak, ambiguous localization signal in practice
(see ``scripts/run_beam_cv.py`` for the honest pooled-RMSE report against
real train wells). To validate the DP's *mechanics* independently of that
hard, ambiguous real-data regime, the synthetic typewell here uses one
clean, isolated Gaussian landmark on a flat baseline. The recovery test
explicitly passes a GR-trusting parameterization (lower move penalty /
mismatch scale than the module's real-data-tuned defaults) because the
module's defaults are deliberately conservative -- see
``scripts/run_beam_cv.py``'s parameter scan, which is what actually justifies
the default values, not this synthetic scenario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.registration.beam import BeamConfig, BeamResult, track_beam, track_beam_multi


def _typewell_gr(tvt: np.ndarray, spike_center: float = 260.0) -> np.ndarray:
    """Flat baseline plus one isolated, unambiguous Gaussian landmark."""
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
    """Build a synthetic (horizontal, typewell) pair with a known TVT drift curve."""
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


# A deliberately GR-trusting parameterization used only to validate DP
# mechanics against the clean synthetic landmark above -- not the module's
# real-data-tuned defaults (see module docstring at the top of this file).
_TRUSTING_KW = {"move_penalty": 4.0, "mismatch_scale": 100.0, "max_move_per_row": 3}


# --------------------------------------------------------------------------- #
# core recovery behaviour
# --------------------------------------------------------------------------- #


def test_track_beam_recovers_known_drift_better_than_flat_anchor() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well()
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result = track_beam(h, tw, **_TRUSTING_KW)

    assert isinstance(result, BeamResult)
    assert result.tvt.shape == true_eval_tvt.shape
    assert result.margin.shape == true_eval_tvt.shape

    beam_rmse = _rmse(result.tvt, true_eval_tvt)
    carry_last_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)

    assert carry_last_rmse > 1.0  # sanity: the synthetic drift is non-trivial
    assert beam_rmse < 0.5 * carry_last_rmse


def test_track_beam_tolerates_missing_gr_and_still_beats_flat_anchor() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(nan_frac=0.3, seed=1)
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result = track_beam(h, tw, **_TRUSTING_KW)

    beam_rmse = _rmse(result.tvt, true_eval_tvt)
    carry_last_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)

    assert beam_rmse < carry_last_rmse
    assert np.all(np.isfinite(result.tvt))


def test_track_beam_max_move_bounds_row_to_row_movement() -> None:
    h, tw, _ = _make_synthetic_well()
    grid_step_ft = 0.2
    max_move_per_row = 2

    result = track_beam(
        h,
        tw,
        grid_step_ft=grid_step_ft,
        max_move_per_row=max_move_per_row,
        **{k: v for k, v in _TRUSTING_KW.items() if k != "max_move_per_row"},
    )

    steps = np.abs(np.diff(result.tvt))
    assert np.all(steps <= max_move_per_row * grid_step_ft + 1e-6)


def test_track_beam_all_nan_gr_holds_anchor_via_move_penalty() -> None:
    """With zero GR evidence, the cheapest path is to never move (move penalty > 0)."""
    h, tw, _ = _make_synthetic_well(n_eval=60)
    h = h.copy()
    known_count = int(h["TVT_input"].notna().sum())
    h.loc[known_count:, "GR"] = np.nan

    result = track_beam(h, tw)

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert np.all(result.tvt == anchor)


def test_track_beam_margin_is_nonnegative_and_shaped_like_eval_zone() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(n_eval=80)

    result = track_beam(h, tw, **_TRUSTING_KW)

    assert result.margin.shape == (80,)
    assert np.all(result.margin >= -1e-9)
    assert np.all(np.isfinite(result.margin))


# --------------------------------------------------------------------------- #
# fallback / exception-safety behaviour
# --------------------------------------------------------------------------- #


def test_track_beam_falls_back_to_zero_margin_on_genuine_exception() -> None:
    """A typewell missing its ``TVT`` column can't build a grid -- a real exception,
    unlike the "all GR is NaN" case above (which is a legitimate DP outcome, not a
    caught failure, so its margin is a real move-penalty gap, not zero)."""
    h, _, _ = _make_synthetic_well(n_eval=40)
    tw_missing_tvt = pd.DataFrame({"not_tvt": [1.0, 2.0, 3.0], "GR": [10.0, 20.0, 30.0]})

    result = track_beam(h, tw_missing_tvt)

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert np.all(result.tvt == anchor)
    assert np.all(result.margin == 0.0)
    assert result.total_cost == 0.0


def test_track_beam_never_raises_on_degenerate_typewell() -> None:
    h, _, _ = _make_synthetic_well(n_eval=20)
    tiny_tw = pd.DataFrame({"TVT": [100.0], "GR": [55.0]})

    result = track_beam(h, tiny_tw)  # must not raise

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert result.tvt.shape == (20,)
    np.testing.assert_allclose(result.tvt, anchor)


def test_track_beam_never_raises_on_missing_gr_column() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=20)
    h_missing_gr = h.drop(columns=["GR"])

    result = track_beam(h_missing_gr, tw)  # must not raise despite missing GR column

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert result.tvt.shape == (20,)
    np.testing.assert_allclose(result.tvt, anchor)


def test_track_beam_never_raises_on_completely_malformed_input() -> None:
    empty_h = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    empty_tw = pd.DataFrame({"also_wrong": [1, 2, 3]})

    result = track_beam(empty_h, empty_tw)  # must not raise

    assert isinstance(result, BeamResult)
    assert result.tvt.shape == result.margin.shape


def test_track_beam_empty_eval_zone_returns_empty_arrays() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=0)

    result = track_beam(h, tw)

    assert result.tvt.shape == (0,)
    assert result.margin.shape == (0,)


def _trusting_kw_no_max_move() -> dict:
    return {k: v for k, v in _TRUSTING_KW.items() if k != "max_move_per_row"}


@pytest.mark.parametrize("max_move_per_row", [1, 2, 5])
def test_track_beam_accepts_various_max_move_configs(max_move_per_row: int) -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(n_eval=80)

    result = track_beam(h, tw, max_move_per_row=max_move_per_row, **_trusting_kw_no_max_move())

    assert result.tvt.shape == true_eval_tvt.shape
    assert np.all(np.isfinite(result.tvt))


# --------------------------------------------------------------------------- #
# track_beam_multi
# --------------------------------------------------------------------------- #


def test_track_beam_multi_returns_weighted_average_of_configs() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(n_eval=60)

    configs = (
        BeamConfig(move_penalty=100.0, mismatch_scale=150.0, max_move_per_row=1, weight=1.0),
        BeamConfig(move_penalty=4.0, mismatch_scale=100.0, max_move_per_row=3, weight=1.0),
    )
    result = track_beam_multi(h, tw, configs=configs)

    r0 = track_beam(
        h,
        tw,
        move_penalty=configs[0].move_penalty,
        mismatch_scale=configs[0].mismatch_scale,
        max_move_per_row=configs[0].max_move_per_row,
    )
    r1 = track_beam(
        h,
        tw,
        move_penalty=configs[1].move_penalty,
        mismatch_scale=configs[1].mismatch_scale,
        max_move_per_row=configs[1].max_move_per_row,
    )
    expected = 0.5 * r0.tvt + 0.5 * r1.tvt

    assert result.tvt.shape == true_eval_tvt.shape
    np.testing.assert_allclose(result.tvt, expected, atol=1e-8)


def test_track_beam_multi_uses_default_configs_when_none_given() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(n_eval=50)

    result = track_beam_multi(h, tw)

    assert result.tvt.shape == true_eval_tvt.shape
    assert np.all(np.isfinite(result.tvt))


def test_track_beam_multi_never_raises_on_malformed_input() -> None:
    empty_h = pd.DataFrame({"nope": [1, 2, 3]})
    empty_tw = pd.DataFrame({"also_nope": [1, 2, 3]})

    result = track_beam_multi(empty_h, empty_tw)  # must not raise

    assert isinstance(result, BeamResult)
    assert result.tvt.shape == result.margin.shape
