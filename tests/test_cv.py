"""Tests for well-level fold splitting and pooled-RMSE scoring in ``rogii.cv``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rogii import baseline
from rogii import cv as cv_module
from rogii import data as D

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "data" / "raw" / "train"


def make_horizontal(
    well_id: str, known_tvt: list[float], eval_true_tvt: list[float]
) -> pd.DataFrame:
    """Build a minimal synthetic horizontal-well DataFrame.

    ``known_tvt`` becomes the known zone (``TVT_input`` present); ``eval_true_tvt``
    becomes the eval zone (``TVT_input`` is NaN there, ``TVT`` holds the ground truth).
    """
    tvt_input = known_tvt + [float("nan")] * len(eval_true_tvt)
    tvt = known_tvt + eval_true_tvt
    n = len(tvt)
    return pd.DataFrame(
        {
            "MD": np.arange(n, dtype=float),
            "GR": np.zeros(n, dtype=float),
            "TVT_input": tvt_input,
            "TVT": tvt,
            "_well_id": [well_id] * n,
        }
    )


def make_typewell() -> pd.DataFrame:
    return pd.DataFrame({"TVT": [0.0, 100.0], "GR": [100.0, 110.0], "Geology": [None, None]})


class FakeWells:
    """Patches ``cv.D.load_horizontal``/``load_typewell`` to serve in-memory wells."""

    def __init__(self, horizontals: dict[str, pd.DataFrame]) -> None:
        self.horizontals = horizontals
        self.tw = make_typewell()

    def load_horizontal(self, well: str, split: str, root: str | None = None) -> pd.DataFrame:
        return self.horizontals[well]

    def load_typewell(self, well: str, split: str, root: str | None = None) -> pd.DataFrame:
        return self.tw


@pytest.fixture
def patch_wells(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _patch(horizontals: dict[str, pd.DataFrame]) -> FakeWells:
        fake = FakeWells(horizontals)
        monkeypatch.setattr(cv_module.D, "load_horizontal", fake.load_horizontal)
        monkeypatch.setattr(cv_module.D, "load_typewell", fake.load_typewell)
        return fake

    return _patch


def predict_by_well_id(pred_map: dict[str, np.ndarray]) -> Any:
    def _predict(h: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
        well_id = h["_well_id"].iloc[0]
        return pred_map[well_id]

    return _predict


# --------------------------------------------------------------------------- #
# well_folds
# --------------------------------------------------------------------------- #


def test_well_folds_is_deterministic() -> None:
    wells = [f"well{i:02d}" for i in range(23)]

    first = cv_module.well_folds(wells, n_splits=5, seed=42)
    second = cv_module.well_folds(wells, n_splits=5, seed=42)

    assert first == second


def test_well_folds_different_seed_can_differ() -> None:
    wells = [f"well{i:02d}" for i in range(23)]

    a = cv_module.well_folds(wells, n_splits=5, seed=1)
    b = cv_module.well_folds(wells, n_splits=5, seed=2)

    assert a != b


def test_well_folds_no_duplicates_and_full_coverage() -> None:
    wells = [f"well{i:02d}" for i in range(17)]

    folds = cv_module.well_folds(wells, n_splits=4, seed=42)

    flat = [w for fold in folds for w in fold]
    assert sorted(flat) == sorted(wells)
    assert len(flat) == len(set(flat))


def test_well_folds_returns_n_splits_folds_reasonably_balanced() -> None:
    wells = [f"well{i:02d}" for i in range(17)]

    folds = cv_module.well_folds(wells, n_splits=4, seed=42)

    assert len(folds) == 4
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1


def test_well_folds_does_not_mutate_input_list() -> None:
    wells = [f"well{i:02d}" for i in range(6)]
    original = list(wells)

    cv_module.well_folds(wells, n_splits=3, seed=42)

    assert wells == original


# --------------------------------------------------------------------------- #
# score_predictor
# --------------------------------------------------------------------------- #


def test_score_predictor_pooled_rmse_matches_manual_calculation(patch_wells: Any) -> None:
    h_a = make_horizontal("A", known_tvt=[10.0] * 5, eval_true_tvt=[12.0, 14.0])
    h_b = make_horizontal("B", known_tvt=[5.0] * 3, eval_true_tvt=[6.0, 8.0, 10.0])
    patch_wells({"A": h_a, "B": h_b})

    predict_fn = predict_by_well_id(
        {
            "A": np.array([11.0, 13.0]),
            "B": np.array([6.0, 8.0, 9.0]),
        }
    )

    result = cv_module.score_predictor(predict_fn, ["A", "B"], split="train")

    # errors: A -> [1.0, 1.0]; B -> [0.0, 0.0, 1.0] => SSE = 2 + 1 = 3, N = 5
    expected_pooled_rmse = math.sqrt(3.0 / 5.0)
    assert result["pooled_rmse"] == pytest.approx(expected_pooled_rmse)
    assert result["n_rows"] == 5
    assert result["n_wells_scored"] == 2
    assert result["n_wells_skipped"] == 0
    assert result["n_wells_failed"] == 0


def test_score_predictor_per_well_rmse_summary_stats(patch_wells: Any) -> None:
    h_a = make_horizontal("A", known_tvt=[10.0] * 5, eval_true_tvt=[10.0, 10.0])
    h_b = make_horizontal("B", known_tvt=[5.0] * 5, eval_true_tvt=[7.0, 7.0])
    patch_wells({"A": h_a, "B": h_b})

    # A: perfect prediction (rmse=0), B: off by 2 everywhere (rmse=2)
    predict_fn = predict_by_well_id(
        {
            "A": np.array([10.0, 10.0]),
            "B": np.array([5.0, 5.0]),
        }
    )

    result = cv_module.score_predictor(predict_fn, ["A", "B"], split="train")

    assert result["median_well_rmse"] == pytest.approx(1.0)
    assert result["max_well_rmse"] == pytest.approx(2.0)
    assert result["p90_well_rmse"] == pytest.approx(np.percentile([0.0, 2.0], 90))


def test_score_predictor_marks_nan_prediction_as_failed_and_continues(patch_wells: Any) -> None:
    h_a = make_horizontal("A", known_tvt=[10.0] * 5, eval_true_tvt=[12.0, 14.0])
    h_b = make_horizontal("B", known_tvt=[5.0] * 3, eval_true_tvt=[6.0, 8.0, 10.0])
    patch_wells({"A": h_a, "B": h_b})

    predict_fn = predict_by_well_id(
        {
            "A": np.array([np.nan, 13.0]),
            "B": np.array([6.0, 8.0, 9.0]),
        }
    )

    result = cv_module.score_predictor(predict_fn, ["A", "B"], split="train")

    assert result["n_wells_failed"] == 1
    assert result["n_wells_scored"] == 1
    # only well B contributes: SSE = 0 + 0 + 1 = 1, N = 3
    assert result["pooled_rmse"] == pytest.approx(math.sqrt(1.0 / 3.0))
    assert result["n_rows"] == 3


def test_score_predictor_marks_inf_prediction_as_failed(patch_wells: Any) -> None:
    h_a = make_horizontal("A", known_tvt=[10.0] * 5, eval_true_tvt=[12.0, 14.0])
    patch_wells({"A": h_a})

    predict_fn = predict_by_well_id({"A": np.array([np.inf, 13.0])})

    result = cv_module.score_predictor(predict_fn, ["A"], split="train")

    assert result["n_wells_failed"] == 1
    assert result["n_wells_scored"] == 0
    assert result["n_rows"] == 0
    assert math.isnan(result["pooled_rmse"])


def test_score_predictor_marks_raising_predict_fn_as_failed_and_continues(
    patch_wells: Any,
) -> None:
    h_a = make_horizontal("A", known_tvt=[10.0] * 5, eval_true_tvt=[12.0, 14.0])
    h_b = make_horizontal("B", known_tvt=[5.0] * 3, eval_true_tvt=[6.0, 8.0, 10.0])
    patch_wells({"A": h_a, "B": h_b})

    def flaky_predict(h: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
        if h["_well_id"].iloc[0] == "A":
            raise RuntimeError("boom")
        return np.array([6.0, 8.0, 9.0])

    result = cv_module.score_predictor(flaky_predict, ["A", "B"], split="train")

    assert result["n_wells_failed"] == 1
    assert result["n_wells_scored"] == 1
    assert result["pooled_rmse"] == pytest.approx(math.sqrt(1.0 / 3.0))


def test_score_predictor_skips_well_with_no_eval_zone(patch_wells: Any) -> None:
    # entirely known zone -> eval_mask is all False
    h_a = make_horizontal("A", known_tvt=[10.0, 11.0, 12.0], eval_true_tvt=[])
    patch_wells({"A": h_a})

    predict_fn = predict_by_well_id({"A": np.array([])})

    result = cv_module.score_predictor(predict_fn, ["A"], split="train")

    assert result["n_wells_skipped"] == 1
    assert result["n_wells_scored"] == 0
    assert result["n_wells_failed"] == 0
    assert result["n_rows"] == 0
    assert math.isnan(result["pooled_rmse"])


def test_score_predictor_skips_well_with_no_anchor(patch_wells: Any) -> None:
    # entirely eval zone -> no known TVT_input anchor at all
    h_a = make_horizontal("A", known_tvt=[], eval_true_tvt=[12.0, 14.0])
    patch_wells({"A": h_a})

    predict_fn = predict_by_well_id({"A": np.array([12.0, 14.0])})

    result = cv_module.score_predictor(predict_fn, ["A"], split="train")

    assert result["n_wells_skipped"] == 1
    assert result["n_wells_scored"] == 0
    assert result["n_rows"] == 0


def test_score_predictor_boundary_eval_zone_of_one_row(patch_wells: Any) -> None:
    h_a = make_horizontal("A", known_tvt=[10.0] * 5, eval_true_tvt=[11.0])
    patch_wells({"A": h_a})

    predict_fn = predict_by_well_id({"A": np.array([11.5])})

    result = cv_module.score_predictor(predict_fn, ["A"], split="train")

    assert result["n_wells_scored"] == 1
    assert result["n_rows"] == 1
    assert result["pooled_rmse"] == pytest.approx(0.5)


def test_score_predictor_boundary_anchor_of_one_point(patch_wells: Any) -> None:
    h_a = make_horizontal("A", known_tvt=[10.0], eval_true_tvt=[12.0, 13.0])
    patch_wells({"A": h_a})

    predict_fn = predict_by_well_id({"A": np.array([12.0, 13.0])})

    result = cv_module.score_predictor(predict_fn, ["A"], split="train")

    assert result["n_wells_skipped"] == 0
    assert result["n_wells_scored"] == 1
    assert result["pooled_rmse"] == pytest.approx(0.0)


@pytest.mark.skipif(
    not TRAIN_DIR.exists(),
    reason="requires local competition data under data/raw/train (not committed to the repo)",
)
def test_score_predictor_integration_with_real_train_data() -> None:
    wells = D.list_wells("train")[:3]
    assert wells, "expected at least one train well for the integration test"

    def predict_fn(h: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
        return baseline.predict_carry_last(h)

    result = cv_module.score_predictor(predict_fn, wells, split="train")

    assert result["n_rows"] >= 0
    assert result["n_wells_scored"] + result["n_wells_skipped"] + result["n_wells_failed"] == len(
        wells
    )
    if result["n_rows"] > 0:
        assert result["pooled_rmse"] >= 0.0
        assert not math.isnan(result["pooled_rmse"])
