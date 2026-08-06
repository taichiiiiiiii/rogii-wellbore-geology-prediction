"""Tests for scripts/build_aug_caches.py -- prefix-cut pseudo-well helpers (R22-lite).

Focused on the leak-safety-critical piece (the pseudo well's evaluation zone
must land exactly on ``[cut_idx, known_len)``, that region's ``TVT_input``
must be genuinely NaN-ed out and the ``TVT`` ground-truth column dropped
entirely -- not just logically ignored -- so anything built downstream from
the pseudo well, e.g. ``rogii.features.build_features``, structurally cannot
see the pseudo evaluation zone's true values) plus the cut-point selection
logic (``compute_cut_point``'s cap-at-3000 / skip-below-800-or-40% rules).

Does not exercise the real ``track_pf_multi``/``track_beam_multi``/
``rogii.spatial.predict_spatial`` trackers (slow, covered by their own
module tests) -- only this script's own cut-point and truncation logic, plus
one integration check that a real leak-safe feature builder
(``rogii.features.build_features``) runs cleanly on a pseudo well and only
ever reflects prefix (``[0, cut_idx)``) information.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_aug_caches as AC  # noqa: E402

from rogii import data as D  # noqa: E402
from rogii.features import build_features  # noqa: E402


def _make_well(n_known: int, n_eval: int, start_tvt: float = 100.0) -> pd.DataFrame:
    """A synthetic train-split well: contiguous known prefix + NaN eval-zone tail.

    Same shape/convention as ``tests/test_backtest_features.py``'s helper.
    """
    known_tvt = start_tvt + np.arange(n_known, dtype=float) * 0.5
    eval_tvt_true = known_tvt[-1] + np.arange(1, n_eval + 1, dtype=float) * 0.5
    tvt_input = np.concatenate([known_tvt, np.full(n_eval, np.nan)])
    tvt_true = np.concatenate([known_tvt, eval_tvt_true])  # TVT known everywhere in train
    n = n_known + n_eval
    return pd.DataFrame(
        {
            "MD": np.arange(n, dtype=float),
            "X": np.arange(n, dtype=float) * 0.1,
            "Y": np.arange(n, dtype=float) * 0.2,
            "Z": -np.arange(n, dtype=float) * 0.3,
            "GR": 50.0 + np.sin(np.arange(n, dtype=float) / 5.0),
            "TVT_input": tvt_input,
            "TVT": tvt_true,
        }
    )


# --------------------------------------------------------------------------- #
# known_zone_length
# --------------------------------------------------------------------------- #


def test_known_zone_length_matches_last_known_index_plus_one() -> None:
    h = _make_well(n_known=20, n_eval=6)

    assert AC.known_zone_length(h) == 20


def test_known_zone_length_zero_when_no_known_rows() -> None:
    h = pd.DataFrame({"TVT_input": [np.nan, np.nan, np.nan]})

    assert AC.known_zone_length(h) == 0


# --------------------------------------------------------------------------- #
# compute_cut_point: cap-at-3000 and skip-below-threshold rules
# --------------------------------------------------------------------------- #


def test_compute_cut_point_uses_full_orig_eval_len_when_under_cap() -> None:
    # known_len=2000, orig_eval_len=500 (< 3000 cap) -> pseudo_eval_len=500
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2000 + 500)

    assert plan is not None
    assert plan.pseudo_eval_len == 500
    assert plan.cut_idx == 2000 - 500


def test_compute_cut_point_caps_pseudo_eval_len_at_3000() -> None:
    # known_len=5000, orig_eval_len=4000 (> 3000 cap) -> pseudo_eval_len=3000
    plan = AC.compute_cut_point("w", known_len=5000, orig_n=5000 + 4000)

    assert plan is not None
    assert plan.pseudo_eval_len == AC.MAX_PSEUDO_EVAL_LEN
    assert plan.cut_idx == 5000 - AC.MAX_PSEUDO_EVAL_LEN


def test_compute_cut_point_skips_below_800_row_absolute_floor() -> None:
    # known_len=1000, orig_eval_len=300 -> cut_idx=700 < 800 -> skip
    plan = AC.compute_cut_point("w", known_len=1000, orig_n=1000 + 300)

    assert plan is None


def test_compute_cut_point_skips_below_40pct_relative_floor() -> None:
    # known_len=3000 (40% = 1200), orig_eval_len=2000 -> cut_idx=1000:
    # >= 800 absolute floor but < 1200 relative floor -> skip
    plan = AC.compute_cut_point("w", known_len=3000, orig_n=3000 + 2000)

    assert plan is None


def test_compute_cut_point_allows_when_both_floors_satisfied() -> None:
    # known_len=3000 (40% = 1200), orig_eval_len=1500 -> cut_idx=1500:
    # >= 800 and >= 1200 -> allowed
    plan = AC.compute_cut_point("w", known_len=3000, orig_n=3000 + 1500)

    assert plan is not None
    assert plan.cut_idx == 1500
    assert plan.pseudo_eval_len == 1500
    assert plan.known_len == 3000
    assert plan.orig_n == 4500
    assert plan.orig_eval_len == 1500


def test_compute_cut_point_none_when_no_real_eval_zone() -> None:
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2000)

    assert plan is None


def test_compute_cut_point_none_when_no_known_rows() -> None:
    plan = AC.compute_cut_point("w", known_len=0, orig_n=500)

    assert plan is None


# --------------------------------------------------------------------------- #
# build_pseudo_well: eval-mask position, NaN-out, anchor, ground-truth, no-leak
# --------------------------------------------------------------------------- #


def test_build_pseudo_well_eval_mask_matches_the_cut_region() -> None:
    h = _make_well(n_known=2000, n_eval=100)
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2100)
    assert plan is not None

    h_pseudo, y_true = AC.build_pseudo_well(h, plan.known_len, plan.cut_idx)

    mask = D.eval_mask(h_pseudo)
    assert mask.sum() == plan.pseudo_eval_len
    assert np.array_equal(np.flatnonzero(mask), np.arange(plan.cut_idx, plan.known_len))
    # the real (downstream, [known_len, orig_n)) eval zone must be dropped
    # entirely, not merely masked -- no row duplication vs. the real eval zone
    assert len(h_pseudo) == plan.known_len


def test_build_pseudo_well_nans_out_only_the_cut_region() -> None:
    h = _make_well(n_known=2000, n_eval=100)
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2100)
    assert plan is not None

    h_pseudo, _ = AC.build_pseudo_well(h, plan.known_len, plan.cut_idx)

    tvt_input = h_pseudo["TVT_input"].to_numpy()
    assert np.all(np.isfinite(tvt_input[: plan.cut_idx]))
    assert np.all(np.isnan(tvt_input[plan.cut_idx :]))
    # residual known zone [0, cut_idx) is byte-identical to the source well
    np.testing.assert_allclose(
        tvt_input[: plan.cut_idx], h["TVT_input"].to_numpy()[: plan.cut_idx]
    )


def test_build_pseudo_well_drops_the_tvt_column_structurally() -> None:
    """Defense in depth: no feature builder can read ``TVT`` even by accident."""
    h = _make_well(n_known=2000, n_eval=100)
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2100)
    assert plan is not None

    h_pseudo, _ = AC.build_pseudo_well(h, plan.known_len, plan.cut_idx)

    assert "TVT" not in h_pseudo.columns


def test_build_pseudo_well_anchor_is_the_row_before_the_cut() -> None:
    h = _make_well(n_known=2000, n_eval=100)
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2100)
    assert plan is not None

    h_pseudo, _ = AC.build_pseudo_well(h, plan.known_len, plan.cut_idx)

    expected_anchor = float(h["TVT_input"].to_numpy()[plan.cut_idx - 1])
    assert D.last_known_tvt(h_pseudo) == pytest.approx(expected_anchor)


def test_build_pseudo_well_y_true_matches_original_tvt_at_cut_positions() -> None:
    h = _make_well(n_known=2000, n_eval=100)
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2100)
    assert plan is not None

    _, y_true = AC.build_pseudo_well(h, plan.known_len, plan.cut_idx)

    expected = h["TVT"].to_numpy()[plan.cut_idx : plan.known_len]
    np.testing.assert_allclose(y_true, expected)
    assert y_true.size == plan.pseudo_eval_len


def test_build_pseudo_well_mdxyzgr_unchanged_in_pseudo_eval_zone() -> None:
    """MD/X/Y/Z/GR at [cut_idx, known_len) are real (known at test time too) --
    only TVT/TVT_input are hidden. No column beyond TVT_input/TVT is touched.
    """
    h = _make_well(n_known=2000, n_eval=100)
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2100)
    assert plan is not None

    h_pseudo, _ = AC.build_pseudo_well(h, plan.known_len, plan.cut_idx)

    for col in ("MD", "X", "Y", "Z", "GR"):
        np.testing.assert_allclose(
            h_pseudo[col].to_numpy()[plan.cut_idx : plan.known_len],
            h[col].to_numpy()[plan.cut_idx : plan.known_len],
        )


# --------------------------------------------------------------------------- #
# Integration: a real leak-safe feature builder only ever reflects [0, cut_idx)
# --------------------------------------------------------------------------- #


def _typewell(n: int = 400) -> pd.DataFrame:
    tvt = np.linspace(50.0, 250.0, n)
    return pd.DataFrame({"TVT": tvt, "GR": 50.0 + np.sin(tvt / 7.0)})


def test_pseudo_well_features_depend_only_on_prefix_information() -> None:
    """``build_features``'s known-prefix-derived columns (anchor_tvt,
    slope_md_last200, ...) must be identical whether or not the ORIGINAL
    (pre-cut) well's tail ``[cut_idx, known_len)`` values are corrupted --
    because ``build_pseudo_well`` NaNs that region out before any feature
    builder ever sees it.
    """
    h_a = _make_well(n_known=2000, n_eval=100)
    h_b = h_a.copy()
    # Corrupt the region that will become the pseudo eval zone in h_b only
    # (both TVT_input and TVT, i.e. the ground truth the pseudo well must
    # never leak); the residual known prefix [0, cut_idx) stays identical.
    plan = AC.compute_cut_point("w", known_len=2000, orig_n=2100)
    assert plan is not None
    rng = np.random.default_rng(0)
    corrupt = rng.uniform(-500.0, 500.0, size=plan.known_len - plan.cut_idx)
    tvt_input_b = h_b["TVT_input"].to_numpy(copy=True)
    tvt_input_b[plan.cut_idx : plan.known_len] = corrupt
    h_b["TVT_input"] = tvt_input_b
    tvt_b = h_b["TVT"].to_numpy(copy=True)
    tvt_b[plan.cut_idx : plan.known_len] = corrupt
    h_b["TVT"] = tvt_b

    tw = _typewell()
    h_pseudo_a, _ = AC.build_pseudo_well(h_a, plan.known_len, plan.cut_idx)
    h_pseudo_b, _ = AC.build_pseudo_well(h_b, plan.known_len, plan.cut_idx)

    feats_a = build_features(h_pseudo_a, tw)
    feats_b = build_features(h_pseudo_b, tw)

    assert len(feats_a) == plan.pseudo_eval_len
    pd.testing.assert_frame_equal(feats_a, feats_b)

    # anchor_tvt must equal the source well's row (cut_idx - 1) value exactly
    expected_anchor = float(h_a["TVT_input"].to_numpy()[plan.cut_idx - 1])
    assert feats_a["anchor_tvt"].iloc[0] == pytest.approx(expected_anchor)
