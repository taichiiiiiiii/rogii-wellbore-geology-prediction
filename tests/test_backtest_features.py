"""Tests for scripts/build_backtest_features.py -- truncation + scoring helpers.

Focused on the leak-safety-critical piece (``truncate_known_prefix``: the
pseudo eval mask must land exactly on the cut region, and that region's
``TVT_input`` must be genuinely NaN-ed out, not just logically ignored) plus
the pure scoring/diversity helpers used to build the well-level feature row.
Does not exercise the real ``track_pf``/``track_beam_multi`` candidates
(slow, and covered by their own module tests) -- only this script's own
truncation and aggregation logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_backtest_features as BF  # noqa: E402

from rogii import data as D  # noqa: E402


def _make_well(n_known: int = 20, n_eval: int = 6, start_tvt: float = 100.0) -> pd.DataFrame:
    """A synthetic train-split well: contiguous known prefix + NaN eval-zone tail."""
    known_tvt = start_tvt + np.arange(n_known, dtype=float) * 0.5
    eval_tvt_true = known_tvt[-1] + np.arange(1, n_eval + 1, dtype=float) * 0.5
    tvt_input = np.concatenate([known_tvt, np.full(n_eval, np.nan)])
    tvt_true = np.concatenate([known_tvt, eval_tvt_true])  # TVT known everywhere in train
    n = n_known + n_eval
    return pd.DataFrame(
        {
            "MD": np.arange(n, dtype=float),
            "X": np.zeros(n),
            "Y": np.zeros(n),
            "Z": np.zeros(n),
            "GR": np.full(n, 50.0),
            "TVT_input": tvt_input,
            "TVT": tvt_true,
        }
    )


# --------------------------------------------------------------------------- #
# truncate_known_prefix
# --------------------------------------------------------------------------- #


def test_truncate_known_prefix_pseudo_eval_mask_matches_the_cut_region() -> None:
    h = _make_well(n_known=20, n_eval=6)

    result = BF.truncate_known_prefix(h, cut_frac=0.5)

    assert result is not None
    assert result.prefix_len == 20
    assert result.cut_idx == 10
    assert result.pseudo_eval_len == 10

    mask = D.eval_mask(result.h)
    assert mask.sum() == result.pseudo_eval_len
    assert np.array_equal(np.flatnonzero(mask), np.arange(10, 20))
    # the real (downstream) eval zone must be dropped entirely, not merely masked
    assert len(result.h) == result.prefix_len


def test_truncate_known_prefix_nans_out_the_cut_region_no_leak() -> None:
    h = _make_well(n_known=20, n_eval=6)

    result = BF.truncate_known_prefix(h, cut_frac=0.7)

    tvt_input = result.h["TVT_input"].to_numpy()
    assert np.all(np.isfinite(tvt_input[: result.cut_idx]))
    assert np.all(np.isnan(tvt_input[result.cut_idx :]))
    # ground truth for the pseudo zone is captured for scoring, but never left
    # reachable via TVT_input (the only column predictors are allowed to read
    # for "is this known" logic)
    expected_true = h["TVT"].to_numpy()[result.cut_idx : result.prefix_len]
    np.testing.assert_allclose(result.y_true, expected_true)


def test_truncate_known_prefix_y_true_matches_original_tvt_at_cut_positions() -> None:
    h = _make_well(n_known=30, n_eval=5)

    result = BF.truncate_known_prefix(h, cut_frac=0.85)

    original_positions = np.arange(result.cut_idx, result.prefix_len)
    np.testing.assert_allclose(
        result.y_true, h["TVT"].to_numpy()[original_positions]
    )


@pytest.mark.parametrize("cut_frac", [0.5, 0.7, 0.85])
def test_truncate_known_prefix_cut_idx_within_valid_bounds(cut_frac: float) -> None:
    h = _make_well(n_known=20, n_eval=6)

    result = BF.truncate_known_prefix(h, cut_frac=cut_frac)

    assert result is not None
    assert 1 <= result.cut_idx <= result.prefix_len - 1
    assert result.pseudo_eval_len >= 1


def test_truncate_known_prefix_returns_none_for_too_short_prefix() -> None:
    h = _make_well(n_known=2, n_eval=6)

    result = BF.truncate_known_prefix(h, cut_frac=0.5, min_prefix_rows=4)

    assert result is None


def test_truncate_known_prefix_returns_none_without_tvt_column() -> None:
    h = _make_well(n_known=20, n_eval=6).drop(columns=["TVT"])

    result = BF.truncate_known_prefix(h, cut_frac=0.5)

    assert result is None


def test_truncate_known_prefix_rejects_out_of_range_cut_frac() -> None:
    h = _make_well(n_known=20, n_eval=6)

    with pytest.raises(ValueError, match="cut_frac"):
        BF.truncate_known_prefix(h, cut_frac=1.0)
    with pytest.raises(ValueError, match="cut_frac"):
        BF.truncate_known_prefix(h, cut_frac=0.0)


def test_truncate_known_prefix_known_zone_before_cut_is_unchanged() -> None:
    h = _make_well(n_known=20, n_eval=6)

    result = BF.truncate_known_prefix(h, cut_frac=0.5)

    np.testing.assert_allclose(
        result.h["TVT_input"].to_numpy()[: result.cut_idx],
        h["TVT_input"].to_numpy()[: result.cut_idx],
    )


# --------------------------------------------------------------------------- #
# _rank_from_values
# --------------------------------------------------------------------------- #


def test_rank_from_values_lowest_gets_rank_one() -> None:
    ranks = BF._rank_from_values({"carry_last": 5.0, "pf": 2.0, "beam": 8.0})

    assert ranks["pf"] == 1.0
    assert ranks["carry_last"] == 2.0
    assert ranks["beam"] == 3.0


def test_rank_from_values_nan_ranks_last() -> None:
    ranks = BF._rank_from_values({"carry_last": 5.0, "pf": float("nan"), "beam": 2.0})

    assert ranks["beam"] == 1.0
    assert ranks["carry_last"] == 2.0
    assert ranks["pf"] == 3.0


def test_rank_from_values_missing_key_treated_as_nan() -> None:
    ranks = BF._rank_from_values({"carry_last": 1.0})

    assert ranks["carry_last"] == 1.0
    assert ranks["pf"] in (2.0, 3.0)
    assert ranks["beam"] in (2.0, 3.0)
    assert ranks["pf"] != ranks["beam"]


# --------------------------------------------------------------------------- #
# _best_second_margin
# --------------------------------------------------------------------------- #


def test_best_second_margin_computes_gap_between_top_two() -> None:
    margin = BF._best_second_margin({"carry_last": 5.0, "pf": 2.0, "beam": 3.0})

    assert margin == pytest.approx(1.0)


def test_best_second_margin_nan_when_fewer_than_two_finite() -> None:
    assert np.isnan(BF._best_second_margin({"carry_last": 5.0, "pf": float("nan")}))
    assert np.isnan(BF._best_second_margin({}))


# --------------------------------------------------------------------------- #
# _pairwise_agreement
# --------------------------------------------------------------------------- #


def test_pairwise_agreement_computes_mean_abs_diff() -> None:
    preds = {
        "carry_last": np.array([1.0, 2.0, 3.0]),
        "pf": np.array([1.0, 2.0, 5.0]),
    }

    agree = BF._pairwise_agreement(preds)

    assert agree[("carry_last", "pf")] == pytest.approx(2.0 / 3.0)
    assert np.isnan(agree[("carry_last", "beam")])
    assert np.isnan(agree[("pf", "beam")])


def test_pairwise_agreement_nan_on_shape_mismatch() -> None:
    preds = {"carry_last": np.array([1.0, 2.0]), "pf": np.array([1.0, 2.0, 3.0])}

    agree = BF._pairwise_agreement(preds)

    assert np.isnan(agree[("carry_last", "pf")])


# --------------------------------------------------------------------------- #
# build_feature_names
# --------------------------------------------------------------------------- #


def test_build_feature_names_length_matches_schema() -> None:
    cut_fracs = (0.5, 0.7, 0.85)
    names = BF.build_feature_names(cut_fracs)

    assert len(names) == len(set(names))  # no duplicate column names
    assert "prefix_len" in names
    for cand in BF.CANDIDATES:
        for cut in cut_fracs:
            assert f"bt_rmse_{cand}_cut{cut:g}" in names
        assert f"bt_rmse_{cand}_mean" in names
        assert f"bt_rmse_{cand}_worst" in names


def test_build_feature_names_shrinks_with_degraded_cut_fracs() -> None:
    names_full = BF.build_feature_names(BF.DEFAULT_CUT_FRACS)
    names_degraded = BF.build_feature_names(BF.DEGRADED_CUT_FRACS)

    assert len(names_degraded) < len(names_full)


# --------------------------------------------------------------------------- #
# compute_well_row: keys always match the declared schema
# --------------------------------------------------------------------------- #


def test_compute_well_row_keys_match_build_feature_names() -> None:
    h = _make_well(n_known=40, n_eval=10)
    tw = pd.DataFrame({"TVT": np.linspace(90.0, 130.0, 50), "GR": np.full(50, 50.0)})
    cut_fracs = (0.5, 0.85)

    row, diag = BF.compute_well_row(h, tw, cut_fracs)

    assert set(row.keys()) == set(BF.build_feature_names(cut_fracs))
    assert diag["truncate_fail"] == 0
    assert row["prefix_len"] == 40.0


def test_compute_well_row_handles_degenerate_well_without_raising() -> None:
    h = pd.DataFrame({"TVT_input": [np.nan, np.nan], "TVT": [1.0, 2.0]})
    tw = pd.DataFrame({"TVT": [1.0], "GR": [1.0]})

    row, diag = BF.compute_well_row(h, tw, (0.5, 0.85))

    assert set(row.keys()) == set(BF.build_feature_names((0.5, 0.85)))
    assert diag["truncate_fail"] == 2  # both cuts skipped: prefix too short
    assert np.isnan(row["bt_rmse_carry_last_cut0.5"])
