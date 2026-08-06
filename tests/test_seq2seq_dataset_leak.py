"""Leak-safety tests for scripts/build_seq2seq_dataset.py (full-well seq2seq builder).

Guards under test (analysis/seq2seq_design.md "leak 厳禁" + task leak-guard list a-e):
  (a) eval rows have ch6 (prefix_rel) == 0.
  (b) eval rows have ch7 (is_known) == 0.
  (c) ``h["TVT"]`` (ground truth) is read in exactly one place in the builder source
      -- the ``q_target`` line. Verified by source-grep (design C-1/C-3).
  (d) formation columns (ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA) and the typewell
      ``Geology`` column are never read -- enforced structurally via ``usecols``
      at CSV-read time, verified both behaviorally and by source-grep.
  (e) no cross-well/global normalization statistics are computed anywhere in the
      builder -- every stat used by a well's features is derived from that well
      alone (R34's global-GR-stats pass is deliberately dropped, not reused).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_seq2seq_dataset as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REAL_TRAIN_DIR = ROOT / "data" / "raw" / "train"

_FORBIDDEN_HORIZONTAL_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def _make_typewell(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    tvt = np.linspace(0.0, 300.0, n)
    gr = 80.0 + 20.0 * np.sin(tvt / 30.0) + rng.normal(0, 1.0, n)
    return pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": ["shale"] * n})


def _make_well(n_known: int = 40, n_eval: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = n_known + n_eval
    md = np.arange(n, dtype=float) * 0.5
    z = -9000.0 - np.arange(n, dtype=float) * 0.1
    known_tvt = 100.0 + np.cumsum(rng.normal(0.2, 0.05, n_known))
    drift_eval = known_tvt[-1] + np.cumsum(rng.normal(0.2, 0.1, n_eval))
    tvt = np.concatenate([known_tvt, drift_eval])
    tvt_input = np.concatenate([known_tvt, np.full(n_eval, np.nan)])
    gr = 80.0 + 20.0 * np.sin(tvt / 30.0) + rng.normal(0, 1.0, n)
    return pd.DataFrame({"MD": md, "Z": z, "GR": gr, "TVT_input": tvt_input, "TVT": tvt})


# --------------------------------------------------------------------------- #
# (a)/(b) eval rows carry zero prefix_rel / is_known
# --------------------------------------------------------------------------- #


def test_eval_rows_have_zero_prefix_rel_and_zero_is_known() -> None:
    h = _make_well()
    tw = _make_typewell()

    wq = B.build_well_query(h, tw)

    assert wq is not None
    assert np.all(wq.q_prefix_rel[wq.q_eval_mask] == 0.0)
    assert np.all(wq.q_is_known[wq.q_eval_mask] == 0.0)


def test_known_rows_have_populated_is_known_and_prefix_rel() -> None:
    h = _make_well()
    tw = _make_typewell()

    wq = B.build_well_query(h, tw)

    assert wq is not None
    known_mask = ~wq.q_eval_mask
    assert np.all(wq.q_is_known[known_mask] == 1.0)
    assert not np.all(wq.q_prefix_rel[known_mask] == 0.0)


def test_skip_condition_too_few_eval_rows_returns_none() -> None:
    h = _make_well(n_known=40, n_eval=5)
    tw = _make_typewell()

    assert B.build_well_query(h, tw) is None


def test_skip_condition_too_few_known_rows_returns_none() -> None:
    h = _make_well(n_known=5, n_eval=40)
    tw = _make_typewell()

    assert B.build_well_query(h, tw) is None


def test_full_well_length_matches_input_rows() -> None:
    h = _make_well(n_known=50, n_eval=35)
    tw = _make_typewell()

    wq = B.build_well_query(h, tw)

    assert wq is not None
    assert wq.q_gr.shape == (85,)
    assert int(wq.q_eval_mask.sum()) == 35
    assert int((~wq.q_eval_mask).sum()) == 50


def test_anchor_equals_last_known_tvt_input() -> None:
    h = _make_well(n_known=40, n_eval=30)
    tw = _make_typewell()

    wq = B.build_well_query(h, tw)

    known = h["TVT_input"].notna().to_numpy()
    expected_anchor = float(h["TVT_input"].to_numpy()[known][-1])
    assert wq is not None
    assert wq.anchor == pytest.approx(expected_anchor)


# --------------------------------------------------------------------------- #
# (c) ground-truth TVT column is read in exactly one place
# --------------------------------------------------------------------------- #


def test_ground_truth_tvt_column_is_read_in_exactly_one_place() -> None:
    source = (SCRIPTS_DIR / "build_seq2seq_dataset.py").read_text()
    # h["TVT"] specifically (the horizontal-well ground truth) -- variable-
    # qualified so it does not also match the typewell's legitimate,
    # repeatedly-read tw["TVT"]/twg["TVT"] reference column.
    hits = re.findall(r'\bh\[\s*["\']TVT["\']\s*\]', source)
    assert len(hits) == 1, f'expected exactly one h["TVT"] read, found {len(hits)}: {hits}'


# --------------------------------------------------------------------------- #
# (d) formation / Geology columns never read (usecols enforced structurally)
# --------------------------------------------------------------------------- #


def test_horizontal_read_excludes_formation_and_xy_columns(tmp_path: Path) -> None:
    n = 10
    df = pd.DataFrame(
        {
            "MD": np.arange(n, dtype=float),
            "X": np.zeros(n),
            "Y": np.zeros(n),
            "Z": np.zeros(n),
            **{c: np.zeros(n) for c in _FORBIDDEN_HORIZONTAL_COLS},
            "TVT": np.zeros(n),
            "GR": np.zeros(n),
            "TVT_input": np.zeros(n),
        }
    )
    p = tmp_path / "well__horizontal_well.csv"
    df.to_csv(p, index=False)

    h = B._read_horizontal(p)

    for col in [*_FORBIDDEN_HORIZONTAL_COLS, "X", "Y"]:
        assert col not in h.columns
    assert set(h.columns) == {"MD", "Z", "GR", "TVT_input", "TVT"}


def test_typewell_read_excludes_geology_column(tmp_path: Path) -> None:
    n = 10
    df = pd.DataFrame({"TVT": np.arange(n, dtype=float), "GR": np.zeros(n), "Geology": ["x"] * n})
    p = tmp_path / "well__typewell.csv"
    df.to_csv(p, index=False)

    tw = B._read_typewell(p)

    assert "Geology" not in tw.columns
    assert set(tw.columns) == {"TVT", "GR"}


def test_usecols_allowlists_exclude_forbidden_columns() -> None:
    assert set(B._HORIZONTAL_USECOLS) == {"MD", "Z", "GR", "TVT_input", "TVT"}
    assert set(B._TYPEWELL_USECOLS) == {"TVT", "GR"}
    for col in _FORBIDDEN_HORIZONTAL_COLS:
        assert col not in B._HORIZONTAL_USECOLS
    assert "Geology" not in B._TYPEWELL_USECOLS


def test_source_never_uses_forbidden_columns_as_a_string_literal() -> None:
    # A plain-English mention in a comment/docstring is fine; an actual
    # quoted string literal (usecols=[...], df["COL"], etc.) is not -- that
    # would be the code path that could load/read the column.
    source = (SCRIPTS_DIR / "build_seq2seq_dataset.py").read_text()
    for col in [*_FORBIDDEN_HORIZONTAL_COLS, "Geology"]:
        assert f'"{col}"' not in source
        assert f"'{col}'" not in source


# --------------------------------------------------------------------------- #
# (e) no cross-well / global normalization statistics
# --------------------------------------------------------------------------- #


def test_no_global_cross_well_gr_stats_computed() -> None:
    source = (SCRIPTS_DIR / "build_seq2seq_dataset.py").read_text()
    # R34's build_registration_dataset.py computed a global GR mean/std across
    # ALL wells before the per-well loop; this builder deliberately drops that
    # (unused) pass since every feature here is normalized per-well only.
    assert "gr_sum" not in source
    assert "gr_sq" not in source


# --------------------------------------------------------------------------- #
# real-data smoke integration: (a)/(b) across the actual first few wells
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not REAL_TRAIN_DIR.is_dir(), reason="competition data not present locally")
def test_smoke_build_on_real_wells_satisfies_leak_guards(tmp_path: Path) -> None:
    out = tmp_path / "smoke.npz"

    B.build(smoke=3, out_path=out)

    d = np.load(out, allow_pickle=True)
    assert len(d["used_wells"]) >= 1
    eval_mask = d["q_eval_mask"]
    assert np.all(d["q_prefix_rel"][eval_mask] == 0.0)
    assert np.all(d["q_is_known"][eval_mask] == 0.0)
    assert eval_mask.sum() > 0
    assert (~eval_mask).sum() > 0
    # fold assignment covers every used well with a valid fold id
    assert d["fold"].shape[0] == len(d["used_wells"])
    assert np.all((d["fold"] >= 0) & (d["fold"] < 5))
