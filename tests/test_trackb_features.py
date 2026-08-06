"""Tests for the Track B from-scratch feature builder in ``rogii.trackb_features``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.trackb_features import (
    FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    build_trackb_features,
    carry_last,
    eval_mask,
    track_b_target,
)


def make_typewell(tvt: list[float], gr: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def make_h(
    n_known: int,
    n_eval: int,
    *,
    gr_known: list[float] | None = None,
    gr_eval: list[float] | None = None,
    start_md: float = 0.0,
    with_tvt: bool = False,
) -> pd.DataFrame:
    """Synthetic horizontal-well DataFrame with only the test-time schema.

    No train-only formation columns, and no ``TVT`` unless ``with_tvt=True``
    -- mirrors ``tests/test_features.py``'s ``make_h`` so an accidental read
    of ground truth inside ``build_trackb_features`` raises ``KeyError``
    rather than silently leaking.
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

    cols = {"MD": md, "X": x, "Y": y, "Z": z, "GR": gr, "TVT_input": tvt_input}
    if with_tvt:
        # ground truth continues the same known-zone trend plus a drift ramp,
        # so track_b_target has a nontrivial (nonzero) signal to check.
        eval_tvt = known_tvt[-1] + 0.1 * np.arange(1, n_eval + 1) + 2.0
        cols["TVT"] = np.concatenate([known_tvt, eval_tvt])
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# (a) no TVT / formation columns needed -- leak-structure guard
# --------------------------------------------------------------------------- #


def test_build_features_works_without_tvt_or_formation_columns() -> None:
    h = make_h(n_known=300, n_eval=50)
    tw = make_typewell(
        tvt=list(10_000.0 + 0.1 * np.arange(400)), gr=list(100.0 + 0.05 * np.arange(400))
    )

    assert "TVT" not in h.columns
    for formation_col in ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]:
        assert formation_col not in h.columns

    out = build_trackb_features(h, tw)  # must not raise KeyError

    assert len(out) == 50
    assert "TVT" not in out.columns
    assert list(out.columns) == FEATURE_COLUMNS


# --------------------------------------------------------------------------- #
# (b) row count == eval row count, column order fixed
# --------------------------------------------------------------------------- #


def test_build_features_row_count_matches_eval_mask() -> None:
    h = make_h(n_known=150, n_eval=37)
    tw = make_typewell(
        tvt=list(10_000.0 + 0.1 * np.arange(200)), gr=list(100.0 + 0.05 * np.arange(200))
    )

    out = build_trackb_features(h, tw)

    assert len(out) == int(eval_mask(h).sum())
    assert len(out) == 37
    assert out.dtypes.eq(np.float32).all()


# --------------------------------------------------------------------------- #
# (c) no inf/nan values on a degenerate typewell + flat known segment
# --------------------------------------------------------------------------- #


def test_build_features_has_no_infinite_or_nan_values() -> None:
    h = make_h(n_known=80, n_eval=20)
    tw = make_typewell(tvt=list(10_000.0 + 0.1 * np.arange(100)), gr=[100.0] * 100)

    out = build_trackb_features(h, tw)

    values = out.to_numpy(dtype=float)
    assert not np.isinf(values).any()
    assert not np.isnan(values).any()


# --------------------------------------------------------------------------- #
# (d) boundary: eval zone of 1 row, known zone of 2 rows
# --------------------------------------------------------------------------- #


def test_build_features_boundary_one_eval_row_two_known_rows() -> None:
    h = make_h(n_known=2, n_eval=1)
    tw = make_typewell(tvt=[9_999.0, 10_000.0, 10_001.0], gr=[95.0, 100.0, 105.0])

    out = build_trackb_features(h, tw)  # must not raise

    assert len(out) == 1
    values = out.to_numpy(dtype=float)
    assert not np.isinf(values).any()
    assert not np.isnan(values).any()


# --------------------------------------------------------------------------- #
# (e) no known-zone anchor at all -> ValueError, matching carry_last's contract
# --------------------------------------------------------------------------- #


def test_build_features_raises_without_known_zone_anchor() -> None:
    h = make_h(n_known=0, n_eval=10)
    tw = make_typewell(tvt=list(10_000.0 + 0.1 * np.arange(10)), gr=[100.0] * 10)

    try:
        build_trackb_features(h, tw)
        raised = False
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# (f) gr_missing correctly flags NaN GR at eval rows, and gr carries NaN through
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

    out = build_trackb_features(h, tw)

    expected_missing = np.zeros(n_eval)
    expected_missing[[2, 7]] = 1.0
    np.testing.assert_array_equal(out["gr_missing"].to_numpy(), expected_missing)
    assert np.isnan(out["gr"].to_numpy()[2])
    assert np.isnan(out["gr"].to_numpy()[7])
    # ``gr`` and ``gr_diff1`` deliberately carry raw NaN through (same
    # convention as rogii.features's gr_diff_1/gr_diff_10 -- missingness is
    # real information, flagged separately by gr_missing rather than
    # silently filled). Every OTHER column must still be finite: the
    # calibrated-GR features (gr_resid_vs_carry_last*) are built from a
    # ffill/bfill'd copy of GR specifically so a stray missing tick doesn't
    # propagate NaN into them, and the rolling stats use min_periods=1 so a
    # single NaN in a window doesn't blank the whole window.
    other_cols = [c for c in FEATURE_COLUMNS if c not in {"gr", "gr_diff1"}]
    assert not out[other_cols].isna().to_numpy().any()


# --------------------------------------------------------------------------- #
# (g) GR affine fit recovers a clean synthetic gain/offset relationship
# --------------------------------------------------------------------------- #


def test_build_features_recovers_clean_affine_gr_fit() -> None:
    n_known, n_eval = 200, 20
    tw_tvt = 10_000.0 + 0.1 * np.arange(n_known + n_eval)
    tw_gr = 50.0 + 0.2 * np.arange(n_known + n_eval)  # varies enough to be non-degenerate
    tw = make_typewell(tvt=list(tw_tvt), gr=list(tw_gr))

    true_gain, true_offset = 2.0, 5.0
    gr_known = list(true_gain * tw_gr[:n_known] + true_offset)
    h = make_h(n_known=n_known, n_eval=n_eval, gr_known=gr_known)

    out = build_trackb_features(h, tw)

    np.testing.assert_allclose(out["gr_affine_gain"].to_numpy(), true_gain, rtol=1e-3)
    np.testing.assert_allclose(out["gr_affine_offset"].to_numpy(), true_offset, atol=1e-2)
    np.testing.assert_allclose(
        out["gr_affine_fit_residual_rms"].to_numpy(), 0.0, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# (h) target definition: track_b_target == TVT - carry_last, eval-mask order
# --------------------------------------------------------------------------- #


def test_track_b_target_matches_definition() -> None:
    h = make_h(n_known=50, n_eval=15, with_tvt=True)

    target = track_b_target(h)
    anchor = carry_last(h)
    expected = h["TVT"].to_numpy()[eval_mask(h)] - anchor

    np.testing.assert_allclose(target, expected)
    assert len(target) == 15
    # anchor equals the last known TVT_input, per the module's target definition
    assert anchor == h.loc[~eval_mask(h), "TVT_input"].to_numpy()[-1]


def test_track_b_target_requires_tvt_column() -> None:
    h = make_h(n_known=50, n_eval=15, with_tvt=False)
    try:
        track_b_target(h)
        raised = False
    except KeyError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# (i) FEATURE_FAMILIES flattens to exactly FEATURE_COLUMNS, no drift between them
# --------------------------------------------------------------------------- #


def test_feature_families_matches_feature_columns() -> None:
    flattened = [c for cols in FEATURE_FAMILIES.values() for c in cols]
    assert flattened == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))  # no duplicate names
