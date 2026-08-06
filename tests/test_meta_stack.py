"""Tests for scripts/run_meta_stack_cv.py -- R3-v3 per-sample meta-stack's
pure/testable core (feature construction, fold-reuse alignment guard,
per-well RMSE helper). Everything here runs on small synthetic arrays, per
``docs/playbooks/01_implement_module.md`` (real-data integration is covered
by the CLI's own row-order assertion against ``stack_v2_oof.npz``, not by
pytest).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_meta_stack_cv as MS  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. candidate matrix construction
# --------------------------------------------------------------------------- #


def test_build_candidate_matrix_preserves_column_order() -> None:
    candidates = ("stack", "carry_last", "beam:x")
    candidate_rows = {
        "carry_last": np.array([10.0, 10.0, 10.0], dtype=np.float32),
        "stack": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "beam:x": np.array([5.0, 6.0, 7.0], dtype=np.float32),
    }

    mat = MS.build_candidate_matrix(candidate_rows, candidates)

    assert mat.shape == (3, 3)
    assert mat.dtype == np.float32
    np.testing.assert_allclose(mat[:, 0], [1.0, 2.0, 3.0])  # "stack" column 0
    np.testing.assert_allclose(mat[:, 1], [10.0, 10.0, 10.0])  # "carry_last" column 1
    np.testing.assert_allclose(mat[:, 2], [5.0, 6.0, 7.0])  # "beam:x" column 2


# --------------------------------------------------------------------------- #
# 2. feature construction: shape, NaN-free, candidate order, agreement stats
# --------------------------------------------------------------------------- #


def _synthetic_two_well_setup() -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """Two wells: well 0 has 3 rows, well 1 has 2 rows. 3 candidates
    (stack/carry_last/other), constructed so every stat is hand-checkable."""
    candidates = ("stack", "carry_last", "other")
    # well 0 rows: stack=[1,2,3], carry_last=[10,10,10], other=[3,3,3]
    # well 1 rows: stack=[5,7],   carry_last=[0,0],      other=[1,3]
    cand_matrix = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 10.0, 3.0],
            [3.0, 10.0, 3.0],
            [5.0, 0.0, 1.0],
            [7.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    well_idx = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    boundaries = np.array([0, 3, 5], dtype=np.int64)
    return cand_matrix, candidates, well_idx, boundaries


def test_build_meta_features_shape_and_no_nan() -> None:
    cand_matrix, candidates, well_idx, boundaries = _synthetic_two_well_setup()

    feats = MS.build_meta_features(cand_matrix, candidates, well_idx, boundaries)

    n_candidates = len(candidates)
    n_stats = len(MS.STAT_FEATURE_COLUMNS)
    assert feats.shape == (5, n_candidates + n_stats)
    assert not feats.isna().any().any()


def test_build_meta_features_candidate_columns_match_input_order() -> None:
    cand_matrix, candidates, well_idx, boundaries = _synthetic_two_well_setup()

    feats = MS.build_meta_features(cand_matrix, candidates, well_idx, boundaries)

    for j, name in enumerate(candidates):
        np.testing.assert_allclose(feats[name].to_numpy(), cand_matrix[:, j])


def test_build_meta_features_agreement_stats_correct() -> None:
    cand_matrix, candidates, well_idx, boundaries = _synthetic_two_well_setup()

    feats = MS.build_meta_features(cand_matrix, candidates, well_idx, boundaries)

    # row 0: values [1, 10, 3] -> median=3, min=1, max=10, range=9
    row0 = feats.iloc[0]
    assert row0["cand_median"] == pytest.approx(3.0)
    assert row0["cand_min"] == pytest.approx(1.0)
    assert row0["cand_max"] == pytest.approx(10.0)
    assert row0["cand_range"] == pytest.approx(9.0)
    assert row0["abs_stack_minus_median"] == pytest.approx(abs(1.0 - 3.0))
    expected_std = np.std([1.0, 10.0, 3.0])
    assert row0["cand_std"] == pytest.approx(expected_std, rel=1e-5)

    # anchor-relative: carry_last=10 for row0 -> median - anchor = 3-10=-7, stack-anchor=1-10=-9
    assert row0["median_minus_anchor"] == pytest.approx(-7.0)
    assert row0["stack_minus_anchor"] == pytest.approx(-9.0)


def test_build_meta_features_position_columns() -> None:
    cand_matrix, candidates, well_idx, boundaries = _synthetic_two_well_setup()

    feats = MS.build_meta_features(cand_matrix, candidates, well_idx, boundaries)

    # well 0 has 3 rows (zone_len=3): local idx 0,1,2 -> frac 0/3, 1/3, 2/3
    np.testing.assert_allclose(
        feats["frac_into_zone"].to_numpy()[:3], [0.0, 1.0 / 3, 2.0 / 3], atol=1e-6
    )
    # well 1 has 2 rows (zone_len=2): local idx 0,1 -> frac 0/2, 1/2
    np.testing.assert_allclose(feats["frac_into_zone"].to_numpy()[3:], [0.0, 0.5], atol=1e-6)

    expected_log_len = np.array([np.log1p(3.0)] * 3 + [np.log1p(2.0)] * 2)
    np.testing.assert_allclose(feats["log1p_zone_len"].to_numpy(), expected_log_len, atol=1e-5)


def test_build_meta_features_sanitizes_lightgbm_unsafe_candidate_names() -> None:
    """Bank candidate names like ``"beam:mid_cal_sm0_mm3"`` contain a colon,
    which LightGBM's ``LGBM_GetLastError`` rejects as a "special JSON
    character" -- verified empirically against the real bank npz files before
    this fix (``lightgbm.basic.LightGBMError: Do not support special JSON
    characters in feature name.``)."""
    candidates = ("stack", "carry_last", "beam:cfg0")
    cand_matrix = np.array(
        [[1.0, 10.0, 3.0], [2.0, 10.0, 4.0]],
        dtype=np.float32,
    )
    well_idx = np.array([0, 0], dtype=np.int32)
    boundaries = np.array([0, 2], dtype=np.int64)

    feats = MS.build_meta_features(cand_matrix, candidates, well_idx, boundaries)

    assert "beam:cfg0" not in feats.columns
    sanitized = MS.sanitize_feature_name("beam:cfg0")
    assert sanitized in feats.columns
    for ch in ':",{}[]':
        assert all(ch not in str(col) for col in feats.columns)
    np.testing.assert_allclose(feats[sanitized].to_numpy(), cand_matrix[:, 2])


def test_build_meta_features_requires_stack_and_carry_last() -> None:
    cand_matrix = np.zeros((2, 1), dtype=np.float32)
    well_idx = np.array([0, 0], dtype=np.int32)
    boundaries = np.array([0, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="stack.*carry_last"):
        MS.build_meta_features(cand_matrix, ("only_one",), well_idx, boundaries)


# --------------------------------------------------------------------------- #
# 3. fold-reuse alignment guard (leak-safety critical)
# --------------------------------------------------------------------------- #


def test_load_row_fold_aligned_accepts_matching_order(tmp_path: Path) -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    well_idx = np.array([0, 0, 1, 1], dtype=np.int32)
    used_wells = np.array(["w0", "w1"])
    row_fold = np.array([0, 0, 1, 1], dtype=np.int32)

    npz_path = tmp_path / "stack_v2_oof.npz"
    np.savez(
        npz_path,
        y_true=y_true,
        well_idx=well_idx,
        used_wells=used_wells,
        row_fold=row_fold,
    )

    out = MS.load_row_fold_aligned(y_true, well_idx, used_wells, npz_path)

    np.testing.assert_array_equal(out, row_fold)


def test_load_row_fold_aligned_rejects_mismatched_row_order(tmp_path: Path) -> None:
    stack_y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    stack_well_idx = np.array([0, 0, 1, 1], dtype=np.int32)
    used_wells = np.array(["w0", "w1"])
    row_fold = np.array([0, 0, 1, 1], dtype=np.int32)

    npz_path = tmp_path / "stack_v2_oof.npz"
    np.savez(
        npz_path,
        y_true=stack_y_true,
        well_idx=stack_well_idx,
        used_wells=used_wells,
        row_fold=row_fold,
    )

    # caller's y_true disagrees (row order/content mismatch) -- must raise, not
    # silently reuse a fold assignment that no longer corresponds to these rows.
    caller_y_true = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)

    with pytest.raises(AssertionError, match="row order disagrees"):
        MS.load_row_fold_aligned(caller_y_true, stack_well_idx, used_wells, npz_path)


# --------------------------------------------------------------------------- #
# 4. well-fold collapse
# --------------------------------------------------------------------------- #


def test_derive_well_fold_collapses_row_fold_per_well() -> None:
    row_fold = np.array([2, 2, 2, 4, 4], dtype=np.int32)
    boundaries = np.array([0, 3, 5], dtype=np.int64)

    well_fold = MS.derive_well_fold(row_fold, boundaries)

    np.testing.assert_array_equal(well_fold, [2, 4])


def test_derive_well_fold_raises_on_inconsistent_well() -> None:
    row_fold = np.array([0, 0, 1], dtype=np.int32)  # well 0's rows split across folds 0 and 1
    boundaries = np.array([0, 3], dtype=np.int64)

    with pytest.raises(AssertionError, match="well-constant"):
        MS.derive_well_fold(row_fold, boundaries)


# --------------------------------------------------------------------------- #
# 5. per-well pooled RMSE helper
# --------------------------------------------------------------------------- #


def test_per_well_rmse_from_boundaries_matches_manual_computation() -> None:
    y = np.array([1.0, 2.0, 3.0, 10.0, 10.0], dtype=np.float64)
    pred = np.array([1.0, 2.0, 5.0, 10.0, 12.0], dtype=np.float64)
    boundaries = np.array([0, 3, 5], dtype=np.int64)

    out = MS.per_well_rmse_from_boundaries(y, pred, boundaries)

    expected_well0 = np.sqrt(np.mean(((y - pred)[:3]) ** 2))
    expected_well1 = np.sqrt(np.mean(((y - pred)[3:]) ** 2))
    np.testing.assert_allclose(out, [expected_well0, expected_well1])


# --------------------------------------------------------------------------- #
# 6. train_meta_oof: fixed n_estimators, well-GroupKFold OOF, no NaN
# --------------------------------------------------------------------------- #


def test_train_meta_oof_produces_full_oof_coverage_no_nan() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n).astype(np.float32)
    y = (2.0 * x + rng.normal(scale=0.1, size=n)).astype(np.float64)
    X = pd.DataFrame({"f0": x})
    row_fold = np.array([i % 4 for i in range(n)], dtype=np.int32)

    oof, fold_rows, importances = MS.train_meta_oof(
        X,
        y,
        row_fold,
        n_splits=4,
        model_params={"num_leaves": 7, "n_estimators": 20, "min_child_samples": 5},
    )

    assert oof.shape == (n,)
    assert not np.isnan(oof).any()
    assert len(fold_rows) == 4
    assert list(importances.index) == ["f0"]
    # a near-linear signal should be recoverable far better than the naive-mean floor
    naive_rmse = MS.pooled_rmse(y, np.full(n, y.mean()))
    oof_rmse = MS.pooled_rmse(y, oof)
    assert oof_rmse < naive_rmse


# --------------------------------------------------------------------------- #
# 6. anchor-relative round trip (reviewer MEDIUM: main()'s reparametrization)
# --------------------------------------------------------------------------- #


def test_anchor_relative_round_trip_matches_absolute_space() -> None:
    """Replicates main()'s anchor-relative transform on synthetic data and
    checks the two properties the whole fix rests on: (1) extracting the
    anchor with ``.astype(np.float64)`` copies, so the in-place candidate
    mutation cannot corrupt it (the aliasing-bug class the reviewer flagged),
    and (2) absolute-space residuals equal relative-space residuals exactly
    (``y - (pred_rel + anchor) == y_rel - pred_rel``), which is what makes the
    per-fold RMSEs printed in relative space directly comparable to the
    absolute-space baseline."""
    rng = np.random.default_rng(7)
    n, candidates = 64, ("stack", "carry_last", "other")
    anchor_i = candidates.index("carry_last")
    cand_matrix = rng.normal(10_000.0, 500.0, size=(n, len(candidates))).astype(np.float32)
    cand_before = cand_matrix.copy()

    anchor_col = cand_matrix[:, anchor_i].astype(np.float64)
    cand_matrix -= anchor_col.astype(np.float32)[:, None]

    # (1) anchor survived the in-place mutation (true copy, no aliasing)
    np.testing.assert_array_equal(anchor_col, cand_before[:, anchor_i].astype(np.float64))
    # carry_last column is exactly zero in relative space
    np.testing.assert_array_equal(cand_matrix[:, anchor_i], np.zeros(n, dtype=np.float32))

    # (2) residual identity: absolute and relative spaces give identical errors
    y = rng.normal(10_000.0, 500.0, size=n)
    y_rel = y - anchor_col
    pred_rel = rng.normal(0.0, 5.0, size=n)
    np.testing.assert_allclose(y - (pred_rel + anchor_col), y_rel - pred_rel, rtol=0, atol=1e-9)
    assert MS.pooled_rmse(y, pred_rel + anchor_col) == pytest.approx(
        MS.pooled_rmse(y_rel, pred_rel), abs=1e-12
    )
