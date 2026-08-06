"""Tests for the tracker-feature builders in ``rogii.stack_features``."""

from __future__ import annotations

import numpy as np
import pytest

from rogii.stack_features import (
    FEATURE_COLUMNS,
    SPATIAL_FEATURE_COLUMNS,
    WELL_FEATURE_COLUMNS,
    build_spatial_features,
    build_tracker_features,
    build_well_aggregate_features,
)

# --------------------------------------------------------------------------- #
# (a) normal case: known values -> exact arithmetic checked by hand
# --------------------------------------------------------------------------- #


def test_build_tracker_features_normal_case_exact_values() -> None:
    anchor = 100.0
    pf_tvt = np.array([102.0, 98.0, 100.0], dtype=np.float32)
    pf_std = np.array([1.0, 4.0, 0.0], dtype=np.float32)
    beam_tvt = np.array([103.0, 99.0, 100.0], dtype=np.float32)
    beam_margin = np.array([5.0, 10.0, 0.0], dtype=np.float32)

    out = build_tracker_features(anchor, pf_tvt, pf_std, beam_tvt, beam_margin)

    assert list(out.columns) == list(FEATURE_COLUMNS)
    assert len(out) == 3

    np.testing.assert_allclose(out["pf_d"].to_numpy(), [2.0, -2.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out["beam_d"].to_numpy(), [3.0, -1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out["pf_std"].to_numpy(), [1.0, 4.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out["beam_margin"].to_numpy(), [5.0, 10.0, 0.0], atol=1e-5)
    # |pf_d - beam_d|
    np.testing.assert_allclose(out["pf_beam_abs_diff"].to_numpy(), [1.0, 1.0, 0.0], atol=1e-5)
    # pf_d / (1 + pf_std)
    np.testing.assert_allclose(out["pf_d_damp"].to_numpy(), [1.0, -0.4, 0.0], atol=1e-5)
    np.testing.assert_allclose(out["pf_std_is_inf"].to_numpy(), [0.0, 0.0, 0.0], atol=1e-5)
    # row0: pf_d>0, beam_d>0 -> agree. row1: pf_d<0, beam_d<0 -> agree.
    # row2: both zero -> tie counts as agree.
    np.testing.assert_allclose(out["pf_beam_sign_agree"].to_numpy(), [1.0, 1.0, 1.0], atol=1e-5)


def test_build_tracker_features_sign_disagreement() -> None:
    anchor = 100.0
    pf_tvt = np.array([105.0], dtype=np.float32)  # pf_d = +5
    pf_std = np.array([1.0], dtype=np.float32)
    beam_tvt = np.array([95.0], dtype=np.float32)  # beam_d = -5
    beam_margin = np.array([1.0], dtype=np.float32)

    out = build_tracker_features(anchor, pf_tvt, pf_std, beam_tvt, beam_margin)

    assert out["pf_beam_sign_agree"].to_numpy()[0] == 0.0
    assert out["pf_beam_abs_diff"].to_numpy()[0] == 10.0


# --------------------------------------------------------------------------- #
# (b) inf pf_std (PF exception-fallback path) -- must not leak inf/produce NaN
#     in the damp feature, and must set the explicit failure flag
# --------------------------------------------------------------------------- #


def test_build_tracker_features_inf_pf_std_handled() -> None:
    anchor = 50.0
    pf_tvt = np.array([50.0, 60.0], dtype=np.float32)
    pf_std = np.array([np.inf, np.inf], dtype=np.float32)
    beam_tvt = np.array([55.0, 58.0], dtype=np.float32)
    beam_margin = np.array([2.0, 3.0], dtype=np.float32)

    out = build_tracker_features(anchor, pf_tvt, pf_std, beam_tvt, beam_margin)

    values = out.to_numpy(dtype=float)
    assert not np.isinf(values).any()
    assert np.all(out["pf_std_is_inf"].to_numpy() == 1.0)
    # finite / inf == 0.0: damped pf_d collapses to 0 (no PF confidence)
    np.testing.assert_allclose(out["pf_d_damp"].to_numpy(), [0.0, 0.0], atol=1e-5)
    # the raw pf_std column itself is NaN (inf replaced, matching
    # rogii.features' inf->nan convention; LightGBM handles NaN natively)
    assert out["pf_std"].isna().all()


# --------------------------------------------------------------------------- #
# (c) boundary: empty arrays (n_eval == 0) must not raise
# --------------------------------------------------------------------------- #


def test_build_tracker_features_empty_arrays() -> None:
    out = build_tracker_features(
        100.0,
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    )

    assert list(out.columns) == list(FEATURE_COLUMNS)
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# (d) boundary: non-finite anchor falls back to 0.0 rather than raising or
#     propagating NaN into every row
# --------------------------------------------------------------------------- #


def test_build_tracker_features_non_finite_anchor_falls_back_to_zero() -> None:
    pf_tvt = np.array([10.0], dtype=np.float32)
    pf_std = np.array([1.0], dtype=np.float32)
    beam_tvt = np.array([20.0], dtype=np.float32)
    beam_margin = np.array([1.0], dtype=np.float32)

    out = build_tracker_features(float("nan"), pf_tvt, pf_std, beam_tvt, beam_margin)

    assert not out.isna().any().any()
    assert out["pf_d"].to_numpy()[0] == 10.0
    assert out["beam_d"].to_numpy()[0] == 20.0


# --------------------------------------------------------------------------- #
# (e) mismatched array lengths must raise (fail loud, not silently misalign)
# --------------------------------------------------------------------------- #


def test_build_tracker_features_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        build_tracker_features(
            100.0,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(2, dtype=np.float32),  # mismatched
            np.zeros(3, dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# (f) dtype / no-inf sanity over a larger randomized array
# --------------------------------------------------------------------------- #


def test_build_tracker_features_no_inf_randomized() -> None:
    rng = np.random.default_rng(0)
    n = 500
    pf_tvt = rng.normal(1000.0, 20.0, n).astype(np.float32)
    pf_std = np.abs(rng.normal(5.0, 3.0, n)).astype(np.float32)
    pf_std[::50] = np.inf  # sprinkle some fallback rows
    beam_tvt = rng.normal(1000.0, 20.0, n).astype(np.float32)
    beam_margin = np.abs(rng.normal(10.0, 5.0, n)).astype(np.float32)

    out = build_tracker_features(1000.0, pf_tvt, pf_std, beam_tvt, beam_margin)

    assert out.dtypes.apply(lambda d: d == np.float32).all()
    values = out.to_numpy(dtype=float)
    assert not np.isinf(values).any()


# =========================================================================== #
# build_well_aggregate_features (stack v2, issue: stack v2)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# (a) normal case: known values -> exact arithmetic checked by hand
# --------------------------------------------------------------------------- #


def test_build_well_aggregate_features_normal_case_exact_values() -> None:
    anchor = 100.0
    pf_tvt = np.array([102.0, 98.0, 100.0], dtype=np.float32)  # pf_d = [2, -2, 0]
    pf_std = np.array([1.0, 4.0, 0.0], dtype=np.float32)
    beam_tvt = np.array([103.0, 99.0, 100.0], dtype=np.float32)  # beam_d = [3, -1, 0]
    beam_margin = np.array([5.0, 10.0, 0.0], dtype=np.float32)
    gr_nan_frac = 0.25

    out = build_well_aggregate_features(anchor, pf_tvt, pf_std, beam_tvt, beam_margin, gr_nan_frac)

    assert list(out.columns) == list(WELL_FEATURE_COLUMNS)
    assert len(out) == 3
    # every row repeats the same well-level scalars
    assert out.nunique().eq(1).all()

    row0 = out.iloc[0]
    assert row0["pf_d_final"] == pytest.approx(0.0, abs=1e-5)  # last pf_d value
    assert row0["pf_d_mean"] == pytest.approx(0.0, abs=1e-5)  # mean([2, -2, 0])
    assert row0["pf_std_well_mean"] == pytest.approx(5.0 / 3.0, abs=1e-4)
    assert row0["pf_std_well_max"] == pytest.approx(4.0, abs=1e-5)
    assert row0["beam_margin_well_mean"] == pytest.approx(5.0, abs=1e-5)
    # |pf_d - beam_d| = [1, 1, 0]
    assert row0["pf_beam_abs_diff_well_mean"] == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert row0["eval_len"] == pytest.approx(3.0, abs=1e-5)
    assert row0["gr_nan_frac"] == pytest.approx(0.25, abs=1e-5)


# --------------------------------------------------------------------------- #
# (b) all-inf pf_std -> mean/max fall back to 0.0, not NaN/inf; NaN
#     gr_nan_frac also falls back to 0.0
# --------------------------------------------------------------------------- #


def test_build_well_aggregate_features_all_inf_pf_std_and_nan_gr_frac_fallback() -> None:
    anchor = 50.0
    pf_tvt = np.array([50.0, 60.0], dtype=np.float32)  # pf_d = [0, 10]
    pf_std = np.array([np.inf, np.inf], dtype=np.float32)
    beam_tvt = np.array([55.0, 58.0], dtype=np.float32)
    beam_margin = np.array([2.0, 3.0], dtype=np.float32)

    out = build_well_aggregate_features(
        anchor, pf_tvt, pf_std, beam_tvt, beam_margin, gr_nan_frac=float("nan")
    )

    values = out.to_numpy(dtype=float)
    assert not np.isnan(values).any()
    assert not np.isinf(values).any()

    row0 = out.iloc[0]
    assert row0["pf_std_well_mean"] == 0.0
    assert row0["pf_std_well_max"] == 0.0
    assert row0["gr_nan_frac"] == 0.0
    assert row0["pf_d_final"] == pytest.approx(10.0, abs=1e-5)  # last (finite) pf_d
    assert row0["pf_d_mean"] == pytest.approx(5.0, abs=1e-5)  # mean([0, 10])


# --------------------------------------------------------------------------- #
# (c) boundary: empty arrays (n_eval == 0) must not raise
# --------------------------------------------------------------------------- #


def test_build_well_aggregate_features_empty_arrays() -> None:
    out = build_well_aggregate_features(
        100.0,
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        gr_nan_frac=0.1,
    )

    assert list(out.columns) == list(WELL_FEATURE_COLUMNS)
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# (d) boundary: non-finite anchor falls back to 0.0
# --------------------------------------------------------------------------- #


def test_build_well_aggregate_features_non_finite_anchor_falls_back_to_zero() -> None:
    pf_tvt = np.array([10.0], dtype=np.float32)
    pf_std = np.array([1.0], dtype=np.float32)
    beam_tvt = np.array([20.0], dtype=np.float32)
    beam_margin = np.array([1.0], dtype=np.float32)

    out = build_well_aggregate_features(
        float("nan"), pf_tvt, pf_std, beam_tvt, beam_margin, gr_nan_frac=0.0
    )

    assert not out.isna().any().any()
    assert out["pf_d_final"].to_numpy()[0] == pytest.approx(10.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# (e) mismatched array lengths must raise (fail loud, not silently misalign)
# --------------------------------------------------------------------------- #


def test_build_well_aggregate_features_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        build_well_aggregate_features(
            100.0,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(2, dtype=np.float32),  # mismatched
            np.zeros(3, dtype=np.float32),
            gr_nan_frac=0.0,
        )


# --------------------------------------------------------------------------- #
# (f) pf_d_final uses the last *finite* value, not blindly index -1, when the
#     tail of the well's PF track happens to be non-finite
# --------------------------------------------------------------------------- #


def test_build_well_aggregate_features_pf_d_final_skips_trailing_non_finite() -> None:
    anchor = 100.0
    pf_tvt = np.array([102.0, np.nan], dtype=np.float64)  # pf_d = [2.0, nan]
    pf_std = np.array([1.0, 1.0], dtype=np.float32)
    beam_tvt = np.array([103.0, 99.0], dtype=np.float32)
    beam_margin = np.array([5.0, 10.0], dtype=np.float32)

    out = build_well_aggregate_features(anchor, pf_tvt, pf_std, beam_tvt, beam_margin, 0.0)

    assert out["pf_d_final"].to_numpy()[0] == pytest.approx(2.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# (g) dtype / no-inf-or-nan sanity over a larger randomized well
# --------------------------------------------------------------------------- #


def test_build_well_aggregate_features_no_inf_randomized() -> None:
    rng = np.random.default_rng(1)
    n = 500
    pf_tvt = rng.normal(1000.0, 20.0, n).astype(np.float32)
    pf_std = np.abs(rng.normal(5.0, 3.0, n)).astype(np.float32)
    beam_tvt = rng.normal(1000.0, 20.0, n).astype(np.float32)
    beam_margin = np.abs(rng.normal(10.0, 5.0, n)).astype(np.float32)

    out = build_well_aggregate_features(1000.0, pf_tvt, pf_std, beam_tvt, beam_margin, 0.3)

    assert out.dtypes.apply(lambda d: d == np.float32).all()
    assert len(out) == n
    values = out.to_numpy(dtype=float)
    assert not np.isinf(values).any()
    assert not np.isnan(values).any()
    assert out["eval_len"].to_numpy()[0] == pytest.approx(float(n), abs=1e-5)


# =========================================================================== #
# build_spatial_features (stack v2, issue: stack v2)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# (a) normal case, gate open (prefix_rmse <= 2.0): exact arithmetic
# --------------------------------------------------------------------------- #


def test_build_spatial_features_gate_open_exact_values() -> None:
    anchor = 100.0
    spatial_tvt = np.array([105.0, 95.0, 100.0], dtype=np.float32)
    prefix_rmse = 1.5
    nn_dist_median = 250.0

    out = build_spatial_features(anchor, spatial_tvt, prefix_rmse, nn_dist_median)

    assert list(out.columns) == list(SPATIAL_FEATURE_COLUMNS)
    assert len(out) == 3
    np.testing.assert_allclose(out["spatial_d"].to_numpy(), [5.0, -5.0, 0.0], atol=1e-4)
    # gate open -> gated equals raw delta
    np.testing.assert_allclose(out["spatial_gated_d"].to_numpy(), [5.0, -5.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(out["spatial_prefix_rmse"].to_numpy(), [1.5, 1.5, 1.5], atol=1e-4)
    np.testing.assert_allclose(
        out["spatial_nn_dist_median"].to_numpy(), [250.0, 250.0, 250.0], atol=1e-3
    )


# --------------------------------------------------------------------------- #
# (b) gate closed (prefix_rmse > 2.0): spatial_gated_d is zeroed, but the raw
#     spatial_d feature is left untouched (GBDT can still see it)
# --------------------------------------------------------------------------- #


def test_build_spatial_features_gate_closed_zeros_gated_delta_only() -> None:
    anchor = 100.0
    spatial_tvt = np.array([120.0, 80.0], dtype=np.float32)
    prefix_rmse = 5.0  # > SPATIAL_GATE_THETA (2.0)
    nn_dist_median = 400.0

    out = build_spatial_features(anchor, spatial_tvt, prefix_rmse, nn_dist_median)

    np.testing.assert_allclose(out["spatial_d"].to_numpy(), [20.0, -20.0], atol=1e-4)
    np.testing.assert_allclose(out["spatial_gated_d"].to_numpy(), [0.0, 0.0], atol=1e-9)


# --------------------------------------------------------------------------- #
# (c) boundary: empty spatial_tvt must not raise
# --------------------------------------------------------------------------- #


def test_build_spatial_features_empty_array() -> None:
    out = build_spatial_features(100.0, np.zeros(0, dtype=np.float32), 1.0, 100.0)

    assert list(out.columns) == list(SPATIAL_FEATURE_COLUMNS)
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# (d) boundary: non-finite anchor falls back to 0.0
# --------------------------------------------------------------------------- #


def test_build_spatial_features_non_finite_anchor_falls_back_to_zero() -> None:
    out = build_spatial_features(
        float("nan"), np.array([10.0], dtype=np.float32), 1.0, 100.0
    )

    assert out["spatial_d"].to_numpy()[0] == pytest.approx(10.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# (e) non-finite prefix_rmse: gate treated as closed (worst case), and the
#     broadcast prefix_rmse column itself becomes NaN (inf -> NaN convention)
# --------------------------------------------------------------------------- #


def test_build_spatial_features_non_finite_prefix_rmse_closes_gate() -> None:
    out = build_spatial_features(
        0.0, np.array([10.0, 20.0], dtype=np.float32), float("nan"), 100.0
    )

    assert out["spatial_prefix_rmse"].isna().all()
    np.testing.assert_allclose(out["spatial_gated_d"].to_numpy(), [0.0, 0.0], atol=1e-9)
    # raw spatial_d is unaffected by the gate
    np.testing.assert_allclose(out["spatial_d"].to_numpy(), [10.0, 20.0], atol=1e-4)


# --------------------------------------------------------------------------- #
# (f) non-finite nn_dist_median passes through as NaN (LightGBM-native)
# --------------------------------------------------------------------------- #


def test_build_spatial_features_non_finite_nn_dist_median_passthrough_nan() -> None:
    out = build_spatial_features(
        0.0, np.array([10.0], dtype=np.float32), 1.0, float("nan")
    )

    assert out["spatial_nn_dist_median"].isna().all()


# --------------------------------------------------------------------------- #
# (g) mismatched length is impossible by construction (single array input),
#     but dtype / no-unexpected-inf sanity over a larger randomized well
# --------------------------------------------------------------------------- #


def test_build_spatial_features_no_unexpected_inf_randomized() -> None:
    rng = np.random.default_rng(2)
    n = 500
    spatial_tvt = rng.normal(1000.0, 30.0, n).astype(np.float32)

    out = build_spatial_features(1000.0, spatial_tvt, 1.2, 180.0)

    assert out.dtypes.apply(lambda d: d == np.float32).all()
    assert len(out) == n
    values = out.to_numpy(dtype=float)
    assert not np.isinf(values).any()
