"""Tests for scripts/run_router_v2_cv.py -- R3-v2 router's pure/testable core.

Scoped to the alignment/leak-safety/oracle/trust-gate primitives that do not
require the real (not-yet-recovered) bank .npz files -- everything here runs
on small synthetic arrays, per ``docs/playbooks/01_implement_module.md``'s
TDD guidance (real-data integration is covered separately by the smoke-run
CLI check, not by pytest).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_router_v2_cv as RV2  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. well-name alignment: robust to shuffled/mismatched source ordering
# --------------------------------------------------------------------------- #


def test_join_well_names_preserves_first_list_order_and_drops_missing() -> None:
    a = ["w1", "w2", "w3"]
    b = ["w3", "w1", "w2"]  # same set, shuffled
    c = ["w2", "w1"]  # w3 missing from this source

    joined = RV2.join_well_names(a, b, c)

    assert joined == ["w1", "w2"]  # order = a's order; w3 dropped (not in c)


def test_block_slices_basic_contiguous_blocks() -> None:
    well_idx = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    used_wells = np.array(["a", "b", "c"])

    slices = RV2.block_slices(well_idx, used_wells)

    assert slices == {"a": (0, 3), "b": (3, 5), "c": (5, 9)}


def test_block_slices_rejects_non_contiguous_well_idx() -> None:
    well_idx = np.array([0, 1, 0], dtype=np.int32)  # well 0's rows are split
    used_wells = np.array(["a", "b"])

    with pytest.raises(ValueError, match="contiguous"):
        RV2.block_slices(well_idx, used_wells)


def test_alignment_survives_shuffled_source_well_order() -> None:
    """End-to-end: join + reindex must land each well's data in the right
    master rows even when a source's own physical well-block order is
    unrelated to the master (stack) order, and a well missing from one
    source is dropped from the whole join (task: "well名で内部結合...欠けて
    いる well は警告して除外")."""
    # Source A ("stack"): defines master order w1, w2, w3 with lengths 2,3,4.
    a_wells = ["w1", "w2", "w3"]
    a_lengths = [2, 3, 4]
    a_well_idx = np.repeat([0, 1, 2], a_lengths).astype(np.int32)
    a_slice = RV2.block_slices(a_well_idx, np.array(a_wells))

    # Source B ("bank"): same 3 wells, stored in a shuffled physical order,
    # each well's segment is a constant marker so row placement is checkable.
    b_wells = ["w3", "w1", "w2"]
    b_lengths = {"w3": 4, "w1": 2, "w2": 3}
    b_well_idx = np.repeat([0, 1, 2], [b_lengths[w] for w in b_wells]).astype(np.int32)
    b_slice = RV2.block_slices(b_well_idx, np.array(b_wells))
    marker = {"w1": 10.0, "w2": 20.0, "w3": 30.0}
    b_array = np.concatenate(
        [np.full(b_lengths[w], marker[w], dtype=np.float32) for w in b_wells]
    )

    # Source C: only has w1/w2 -- exercises join_well_names' drop behavior.
    c_wells = ["w2", "w1"]

    master_wells = RV2.join_well_names(a_wells, b_wells, c_wells)
    assert master_wells == ["w1", "w2"]

    master_lengths = [a_slice[w][1] - a_slice[w][0] for w in master_wells]
    master_boundaries = np.concatenate([[0], np.cumsum(master_lengths)])

    reindexed = RV2.reindex_to_master(b_array, b_slice, master_wells, master_boundaries)

    assert reindexed.shape[0] == sum(master_lengths) == 5
    np.testing.assert_allclose(reindexed[:2], np.full(2, 10.0))  # w1's marker
    np.testing.assert_allclose(reindexed[2:5], np.full(3, 20.0))  # w2's marker


def test_reindex_to_master_leaves_nan_for_missing_or_mismatched_well() -> None:
    master_wells = ["w1", "w2"]
    master_boundaries = np.array([0, 2, 5])
    source_slices = {"w1": (0, 2)}  # w2 absent from this source entirely
    source_array = np.array([1.0, 2.0], dtype=np.float32)

    out = RV2.reindex_to_master(source_array, source_slices, master_wells, master_boundaries)

    np.testing.assert_allclose(out[:2], [1.0, 2.0])
    assert np.all(np.isnan(out[2:]))


# --------------------------------------------------------------------------- #
# 2. cross-candidate agreement features: leak-safety (never touch y_true)
# --------------------------------------------------------------------------- #


def test_agreement_features_signature_has_no_truth_argument() -> None:
    params = set(inspect.signature(RV2.compute_agreement_features).parameters)

    assert "y_true" not in params
    assert "y" not in params
    assert "tvt" not in params


def test_agreement_features_are_deterministic_pure_function_of_predictions() -> None:
    rng = np.random.default_rng(0)
    n = 20
    seg_by_candidate = {
        "stack": rng.normal(100, 1, n).astype(np.float32),
        "carry_last": np.full(n, 100.0, dtype=np.float32),
        "pf_blend": rng.normal(100, 1, n).astype(np.float32),
        "spatial": rng.normal(100, 1, n).astype(np.float32),
        "pf_bma_scale8": rng.normal(100, 1, n).astype(np.float32),
        "beam:cfg0": rng.normal(100, 1, n).astype(np.float32),
    }
    family_reps = ("stack", "carry_last", "pf_blend", "spatial", "pf_bma_scale8", "beam:cfg0")

    feats1 = RV2.compute_agreement_features(
        seg_by_candidate, family_reps, "pf_bma_scale8", "beam:cfg0"
    )
    feats2 = RV2.compute_agreement_features(
        dict(seg_by_candidate), family_reps, "pf_bma_scale8", "beam:cfg0"
    )

    assert set(feats1.keys()) == set(RV2.AGREE_FEATURE_COLUMNS)
    v1 = np.array([feats1[k] for k in RV2.AGREE_FEATURE_COLUMNS])
    v2 = np.array([feats2[k] for k in RV2.AGREE_FEATURE_COLUMNS])
    np.testing.assert_allclose(v1, v2, equal_nan=True)


def test_agreement_features_detect_zero_disagreement_when_candidates_identical() -> None:
    n = 10
    same = np.full(n, 42.0, dtype=np.float32)
    seg_by_candidate = {
        "stack": same,
        "pf_bma_scale8": same,
        "beam:cfg0": same,
    }
    feats = RV2.compute_agreement_features(
        seg_by_candidate, ("stack", "pf_bma_scale8", "beam:cfg0"), "pf_bma_scale8", "beam:cfg0"
    )

    assert feats["agree_pfbma_beam_rms"] == pytest.approx(0.0)
    assert feats["agree_stack_bank_best_rms"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3. oracle computation: known solution on hand-built SSE matrices
# --------------------------------------------------------------------------- #


def test_full_oracle_and_greedy_selection_known_solution() -> None:
    well_n = np.array([2, 2, 2])
    candidates = ("stack", "pf_blend", "spatial")
    # well0/well1 best=pf_blend(4.0); well2 best=spatial(1.0); stack always worst.
    sse = np.array(
        [
            [100.0, 4.0, 50.0],
            [100.0, 4.0, 50.0],
            [100.0, 50.0, 1.0],
        ]
    )

    full = RV2.full_oracle_from_sse(sse, well_n)
    expected_rmse = np.sqrt((4.0 + 4.0 + 1.0) / well_n.sum())
    assert full == pytest.approx(expected_rmse)

    history = RV2.greedy_forward_selection(sse, well_n, candidates)
    assert [name for name, _ in history] == sorted(candidates)  # every candidate appears once
    assert len(history) == 3
    # best single candidate (pf_blend: 4+4+50=58 sse) must be picked first.
    assert history[0][0] == "pf_blend"
    # adding every candidate must reach the same ceiling as full_oracle_from_sse.
    assert history[-1][1] == pytest.approx(full)
    # pooled RMSE must be non-increasing as candidates are added.
    values = [v for _, v in history]
    assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1))


def test_hard_well_oracle_guard_substitutes_only_the_hardest_well() -> None:
    candidates = ("stack", "carry_last", "pf_blend", "spatial")
    n_wells = 10  # top 10% -> exactly 1 well substituted
    well_n = np.full(10, 2)
    sse = np.full((10, 4), 4.0)
    sse[3, 0] = 1000.0  # well 3 is by far the hardest for "stack"
    sse[3, 1] = 1.0  # ...but "carry_last" is much better there

    result = RV2.hard_well_oracle_guard(candidates, n_wells, sse, well_n)

    expected_total_sse = 1.0 + 4.0 * 9  # well3 substituted, the other 9 keep stack's 4.0
    expected = np.sqrt(expected_total_sse / int(well_n.sum()))
    assert result == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 4. trust-gate fallback behavior
# --------------------------------------------------------------------------- #


def test_decide_falls_back_to_stack_below_tau() -> None:
    candidates = ("stack", "carry_last", "pf_blend")
    # argmin=carry_last; improvement = 5.0 - 4.5 = 0.5ft exactly (binary-exact fractions
    # throughout, to avoid float-rounding ambiguity right at the >= boundary).
    pred_row = np.array([5.0, 4.5, 10.0])

    assert RV2.decide(pred_row, candidates, tau=0.75) == "stack"  # 0.5 < tau -> fallback
    assert RV2.decide(pred_row, candidates, tau=0.5) == "carry_last"  # exact boundary: >= switches
    assert RV2.decide(pred_row, candidates, tau=0.25) == "carry_last"


def test_decide_stays_on_stack_when_stack_is_already_the_argmin() -> None:
    candidates = ("stack", "carry_last")
    pred_row = np.array([1.0, 5.0])

    assert RV2.decide(pred_row, candidates, tau=0.0) == "stack"


def test_pooled_rmse_from_decisions_falls_back_when_chosen_candidate_unavailable() -> None:
    candidates = ("stack", "beam:x")
    cand_to_idx = {c: i for i, c in enumerate(candidates)}
    sse = np.array([[4.0, np.inf]])  # beam:x unavailable (e.g. shape mismatch) for well 0
    well_n = np.array([2])
    well_to_idx = {"w0": 0}
    decisions = {"w0": "beam:x"}

    rmse = RV2.pooled_rmse_from_decisions(decisions, well_to_idx, sse, well_n, cand_to_idx)

    assert rmse == pytest.approx(np.sqrt(4.0 / 2))  # silently falls back to stack's SSE


# --------------------------------------------------------------------------- #
# 5. misc pure helpers
# --------------------------------------------------------------------------- #


def test_sanitize_labels_replaces_inf_and_nan_with_sentinel() -> None:
    m = np.array([[1.0, np.inf], [np.nan, 2.0]])

    out = RV2.sanitize_labels(m, sentinel=99.0)

    np.testing.assert_allclose(out, [[1.0, 99.0], [99.0, 2.0]])


def test_effective_n_splits_degrades_for_small_well_counts() -> None:
    assert RV2.effective_n_splits(773) == 5
    assert RV2.effective_n_splits(8) == 4
    assert RV2.effective_n_splits(3) == 3
    assert RV2.effective_n_splits(1) == 1
