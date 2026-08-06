"""Tests for scripts/make_submission.py -- carry_last submission pipeline + validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import make_submission as MS  # noqa: E402


def _write_well(root: Path, well: str, tvt_input: list[float]) -> None:
    """Write a minimal test-split horizontal-well csv (MD,X,Y,Z,GR,TVT_input)."""
    n = len(tvt_input)
    df = pd.DataFrame(
        {
            "MD": np.arange(n, dtype=float),
            "X": np.zeros(n),
            "Y": np.zeros(n),
            "Z": np.zeros(n),
            "GR": np.full(n, 100.0),
            "TVT_input": tvt_input,
        }
    )
    test_dir = root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_dir / f"{well}__horizontal_well.csv", index=False)


def _write_sample_submission(root: Path, rows: list[tuple[str, int]]) -> None:
    ids = [f"{well}_{idx}" for well, idx in rows]
    pd.DataFrame({"id": ids, "tvt": [0.0] * len(ids)}).to_csv(
        root / "sample_submission.csv", index=False
    )


def _make_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data_root"
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# parse_sample_submission
# --------------------------------------------------------------------------- #


def test_parse_sample_submission_splits_well_and_row_index(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    _write_sample_submission(root, [("aaaa1111", 5), ("aaaa1111", 6), ("bbbb2222", 4)])

    sample = MS.parse_sample_submission(root / "sample_submission.csv")

    assert sample["well"].tolist() == ["aaaa1111", "aaaa1111", "bbbb2222"]
    assert sample["row_index"].tolist() == [5, 6, 4]


def test_parse_sample_submission_rejects_malformed_ids(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    pd.DataFrame({"id": ["noUnderscoreHere"], "tvt": [0.0]}).to_csv(
        root / "sample_submission.csv", index=False
    )

    with pytest.raises(ValueError, match="well}_{row_index"):
        MS.parse_sample_submission(root / "sample_submission.csv")


# --------------------------------------------------------------------------- #
# build_predictions (carry_last)
# --------------------------------------------------------------------------- #


def test_build_predictions_carries_last_known_value_and_preserves_id_order(
    tmp_path: Path,
) -> None:
    root = _make_data_root(tmp_path)
    _write_well(root, "aaaa1111", tvt_input=[10.0, 11.0, 12.0, 13.0, 14.0, np.nan, np.nan, np.nan])
    _write_well(root, "bbbb2222", tvt_input=[20.0, 21.0, 22.0, 23.0, np.nan, np.nan])
    _write_sample_submission(
        root,
        [("aaaa1111", 5), ("aaaa1111", 6), ("aaaa1111", 7), ("bbbb2222", 4), ("bbbb2222", 5)],
    )
    sample = MS.parse_sample_submission(root / "sample_submission.csv")

    out, warnings = MS.build_predictions(sample, root, "carry_last")

    assert list(out["id"]) == list(sample["id"])  # row order preserved
    np.testing.assert_allclose(
        out.loc[out["id"].str.startswith("aaaa"), "tvt"].to_numpy(), [14.0, 14.0, 14.0]
    )
    np.testing.assert_allclose(
        out.loc[out["id"].str.startswith("bbbb"), "tvt"].to_numpy(), [23.0, 23.0]
    )
    assert warnings == []


def test_build_predictions_handles_single_row_eval_zone_boundary(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    _write_well(root, "cccc3333", tvt_input=[30.0, 31.0, 32.0, np.nan])  # eval zone = 1 row
    _write_sample_submission(root, [("cccc3333", 3)])
    sample = MS.parse_sample_submission(root / "sample_submission.csv")

    out, warnings = MS.build_predictions(sample, root, "carry_last")

    assert out["tvt"].tolist() == [32.0]
    assert warnings == []


def test_build_predictions_warns_when_id_row_index_outside_eval_zone(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    _write_well(root, "dddd4444", tvt_input=[40.0, 41.0, np.nan, np.nan])
    # row 1 is in the *known* zone, not the eval zone -- and the well only has 2 eval
    # rows, so we ask for exactly 2 ids to keep the eval-zone-size check satisfied.
    _write_sample_submission(root, [("dddd4444", 1), ("dddd4444", 3)])
    sample = MS.parse_sample_submission(root / "sample_submission.csv")

    out, warnings = MS.build_predictions(sample, root, "carry_last")

    assert len(warnings) == 1
    assert "dddd4444" in warnings[0]
    # row_index=1 falls back to last_known_tvt (41.0); row_index=3 is genuine eval zone
    assert out["tvt"].tolist() == [41.0, 41.0]


def test_build_predictions_raises_when_sample_id_count_disagrees_with_eval_zone(
    tmp_path: Path,
) -> None:
    root = _make_data_root(tmp_path)
    _write_well(root, "eeee5555", tvt_input=[50.0, 51.0, np.nan, np.nan])  # eval zone = 2 rows
    _write_sample_submission(root, [("eeee5555", 2)])  # sample only expects 1 id
    sample = MS.parse_sample_submission(root / "sample_submission.csv")

    with pytest.raises(ValueError, match="eval zone"):
        MS.build_predictions(sample, root, "carry_last")


# --------------------------------------------------------------------------- #
# validate_submission
# --------------------------------------------------------------------------- #


def _sample_and_out(ids: list[str], tvt: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.DataFrame({"id": ids, "tvt": [0.0] * len(ids)})
    out = pd.DataFrame({"id": ids, "tvt": tvt})
    return sample, out


def test_validate_submission_passes_for_well_formed_output() -> None:
    sample, out = _sample_and_out(["a_0", "a_1", "b_0"], [1.0, 2.0, 3.0])

    assert MS.validate_submission(out, sample) == []


def test_validate_submission_fails_on_missing_id() -> None:
    sample, out = _sample_and_out(["a_0", "a_1"], [1.0, 2.0])
    out = out.iloc[[0]]  # drop "a_1"

    errors = MS.validate_submission(out, sample)

    assert any("missing" in e for e in errors)


def test_validate_submission_fails_on_extra_id() -> None:
    sample, out = _sample_and_out(["a_0", "a_1"], [1.0, 2.0])
    extra_row = pd.DataFrame({"id": ["a_99"], "tvt": [9.0]})
    out = pd.concat([out, extra_row], ignore_index=True)

    errors = MS.validate_submission(out, sample)

    assert any("unexpected" in e or "not in sample_submission" in e for e in errors)


def test_validate_submission_fails_on_duplicate_id() -> None:
    sample, out = _sample_and_out(["a_0", "a_1"], [1.0, 2.0])
    dup_row = pd.DataFrame({"id": ["a_0"], "tvt": [1.0]})
    out = pd.concat([out, dup_row], ignore_index=True)
    sample = pd.concat([sample, dup_row[["id"]].assign(tvt=0.0)], ignore_index=True)

    errors = MS.validate_submission(out, sample)

    assert any("duplicate" in e for e in errors)


def test_validate_submission_fails_on_nan_tvt() -> None:
    sample, out = _sample_and_out(["a_0", "a_1"], [1.0, float("nan")])

    errors = MS.validate_submission(out, sample)

    assert any("NaN" in e for e in errors)


def test_validate_submission_fails_on_inf_tvt() -> None:
    sample, out = _sample_and_out(["a_0", "a_1"], [1.0, float("inf")])

    errors = MS.validate_submission(out, sample)

    assert any("inf" in e for e in errors)


def test_validate_submission_fails_on_row_count_mismatch() -> None:
    sample, out = _sample_and_out(["a_0", "a_1", "a_2"], [1.0, 2.0, 3.0])
    out = out.iloc[[0, 1]]
    sample = sample.iloc[[0, 1]]  # keep id sets equal, only vary the underlying frame length
    extra = pd.DataFrame({"id": ["a_2"], "tvt": [0.0]})
    sample_full = pd.concat([sample, extra], ignore_index=True)

    errors = MS.validate_submission(out, sample_full)

    assert any("row count mismatch" in e for e in errors)


# --------------------------------------------------------------------------- #
# main() end-to-end
# --------------------------------------------------------------------------- #


def test_main_writes_valid_submission_and_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_data_root(tmp_path)
    _write_well(root, "ffff6666", tvt_input=[60.0, 61.0, np.nan, np.nan, np.nan])
    _write_sample_submission(root, [("ffff6666", 2), ("ffff6666", 3), ("ffff6666", 4)])
    out_path = tmp_path / "submission.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_submission.py",
            "--data-root",
            str(root),
            "--out",
            str(out_path),
        ],
    )

    MS.main()  # should not raise / should not sys.exit

    written = pd.read_csv(out_path)
    assert list(written.columns) == ["id", "tvt"]
    assert written["id"].tolist() == ["ffff6666_2", "ffff6666_3", "ffff6666_4"]
    np.testing.assert_allclose(written["tvt"].to_numpy(), [61.0, 61.0, 61.0])


def test_main_exits_nonzero_when_sample_ids_disagree_with_eval_zone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_data_root(tmp_path)
    _write_well(root, "aabb0011", tvt_input=[1.0, 2.0, np.nan, np.nan])  # eval zone = 2 rows
    _write_sample_submission(root, [("aabb0011", 2)])  # sample only expects 1 id -> mismatch
    out_path = tmp_path / "submission.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_submission.py",
            "--data-root",
            str(root),
            "--out",
            str(out_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        MS.main()

    assert exc_info.value.code != 0
    assert not out_path.exists()
