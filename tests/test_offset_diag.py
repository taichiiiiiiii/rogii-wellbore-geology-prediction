"""Tests for R28's per-well offset time-series diagnostics in ``rogii.spatial``.

Source / motivation: forum notebook (karnakbaevarthur, vote107) observed that
splitting the known-zone prefix offset (``off = tvt_input + z - donor
formation depth``) into early/mid/late thirds plus a tail-weighted linear
fit reveals whether the offset is drifting within the prefix -- a signal the
single flat median (used by ``predict_spatial``'s own calibration) throws
away. This module precomputes those diagnostics per well as a candidate
feature for the downstream LGBM stack; see ``rogii.spatial.offset_diagnostics``.

Synthetic geometry mirrors ``tests/test_spatial.py``: a single formation
whose top depth is an exact plane ``depth(X, Y) = A*X + B*Y + C``. The
target well's known-zone ``TVT_input`` is constructed so its per-row offset
``off_j = tvt_input_j + z_j - depth_j`` follows a known (possibly drifting)
series, so a correct leave-self-out KNN interpolation must recover that
series from the prefix rows alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii import data as D
from rogii import spatial as SP

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "data" / "raw" / "train"

# Plane: depth(X, Y) = A * X + B * Y + C
A, B, C = 0.5, 0.2, 100.0
TRUE_OFFSET = 7.0

TARGET = "wt"
FORMATION = ("ANCC",)


def plane_depth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return A * x + B * y + C


def make_bank_well(y_line: float) -> pd.DataFrame:
    """A straight offset well along X at ``y = y_line`` sampling the plane."""
    x = np.arange(0.0, 101.0)
    y = np.full_like(x, y_line)
    return pd.DataFrame(
        {
            "MD": x,
            "X": x,
            "Y": y,
            "Z": np.full_like(x, -50.0),
            "ANCC": plane_depth(x, y),
            "TVT": np.zeros_like(x),
            "GR": np.zeros_like(x),
            "TVT_input": np.zeros_like(x),
        }
    )


def bank_frames() -> dict[str, pd.DataFrame]:
    """Offset wells on lines y = 0, 10, ..., 100."""
    return {f"w{int(y):02d}": make_bank_well(float(y)) for y in range(0, 101, 10)}


def install_fake_loader(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> None:
    def fake_load(well: str, split: str, root: str | None = None) -> pd.DataFrame:
        return frames[well]

    monkeypatch.setattr("rogii.data.load_horizontal", fake_load)


def make_target_well(
    n_known: int = 60, n_eval: int = 41, slope_per_row: float = 0.0
) -> pd.DataFrame:
    """Target well along ``y = 45`` with **no TVT / formation columns** (test schema).

    ``off_j = tvt_input_j + z_j - depth_j`` is exactly ``TRUE_OFFSET +
    slope_per_row * j`` by construction (``j`` = row index). Omitting ``TVT``
    and the formation columns means any accidental read of them inside
    ``offset_diagnostics`` raises a ``KeyError`` instead of silently leaking
    (same pattern as ``tests/test_spatial.py``).
    """
    n = n_known + n_eval
    x = np.arange(0.0, float(n))
    y = np.full(n, 45.0)
    z = -50.0 - 0.1 * x
    depth = plane_depth(x, y)
    offset_series = TRUE_OFFSET + slope_per_row * np.arange(n)
    tvt_input = -z + depth + offset_series
    tvt_input[n_known:] = np.nan
    return pd.DataFrame(
        {"MD": x, "X": x, "Y": y, "Z": z, "GR": np.zeros(n), "TVT_input": tvt_input}
    )


# --------------------------------------------------------------------------- #
# drift recovery (the headline behavior this module exists for)
# --------------------------------------------------------------------------- #


def test_drifting_offset_detected_via_late_minus_early_and_wls_slope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    # off drifts from TRUE_OFFSET to TRUE_OFFSET + 5 across the 60 known rows.
    h = make_target_well(n_known=60, slope_per_row=5.0 / 59.0)
    assert "TVT" not in h.columns and "ANCC" not in h.columns

    diag = SP.offset_diagnostics(h, bank, exclude_well=TARGET, k=10, method="plane")

    assert diag.n_valid_prefix == 60
    assert np.isfinite(diag.prefix_rmse_best)
    assert diag.offset_late - diag.offset_early > 3.0
    assert diag.wls_slope_per100 > 0.0
    # the fit's end-of-prefix estimate should land near the true end value.
    assert diag.wls_end == pytest.approx(TRUE_OFFSET + 5.0, abs=1.0)


def test_zero_drift_well_has_near_zero_slope_and_flat_thirds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h = make_target_well(n_known=60, slope_per_row=0.0)

    diag = SP.offset_diagnostics(h, bank, exclude_well=TARGET, k=10, method="plane")

    assert diag.n_valid_prefix == 60
    assert abs(diag.wls_slope_per100) < 0.5
    assert abs(diag.offset_late - diag.offset_early) < 0.5
    assert diag.offset_med == pytest.approx(TRUE_OFFSET, abs=0.1)


def test_prefix_rmse_matches_predict_spatial_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    """offset_diagnostics must mirror predict_spatial's own prefix-fit logic
    exactly: same bank/k/method/well must produce the same calibrated
    prefix RMSE (the eval-zone-skipping optimization must not change the
    prefix-side computation at all)."""
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)
    h = make_target_well(n_known=60, slope_per_row=0.3)

    diag = SP.offset_diagnostics(h, bank, exclude_well=TARGET, k=10, method="plane")
    result = SP.predict_spatial(h, bank, exclude_well=TARGET, k=10, method="plane")

    assert diag.prefix_rmse_best == pytest.approx(result.prefix_rmse, abs=1e-9)


# --------------------------------------------------------------------------- #
# boundaries (never raise)
# --------------------------------------------------------------------------- #


def test_single_valid_prefix_row_leaves_split_and_wls_fields_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h = make_target_well(n_known=1, n_eval=41)

    diag = SP.offset_diagnostics(h, bank, exclude_well=TARGET, k=10, method="plane")

    assert diag.n_valid_prefix == 1
    assert np.isfinite(diag.offset_med)
    assert np.isfinite(diag.prefix_rmse_best)
    assert np.isnan(diag.offset_early)
    assert np.isnan(diag.offset_mid)
    assert np.isnan(diag.offset_late)
    assert np.isnan(diag.wls_slope_per100)
    assert np.isnan(diag.wls_end)


def test_zero_valid_prefix_rows_falls_back_to_all_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    # No formations at all -> the formation loop never runs -> no candidate
    # -> zero valid prefix rows in effect -> exception-fallback path.
    bank = SP.SurfaceBank(formations=())

    h = make_target_well(n_known=60)

    diag = SP.offset_diagnostics(h, bank, exclude_well=TARGET, k=10)

    assert diag.n_valid_prefix == 0
    assert diag.prefix_rmse_best == float("inf")
    assert np.isnan(diag.offset_med)
    assert np.isnan(diag.offset_early)
    assert np.isnan(diag.offset_mid)
    assert np.isnan(diag.offset_late)
    assert np.isnan(diag.wls_slope_per100)
    assert np.isnan(diag.wls_end)


def test_no_known_prefix_falls_back_to_all_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h = make_target_well(n_known=60)
    h["TVT_input"] = np.nan  # no anchor at all

    diag = SP.offset_diagnostics(h, bank, exclude_well=TARGET, k=10, method="plane")

    assert diag.n_valid_prefix == 0
    assert diag.prefix_rmse_best == float("inf")
    assert np.isnan(diag.offset_med)


def test_never_raises_on_malformed_input() -> None:
    bank = SP.SurfaceBank(formations=())
    h = pd.DataFrame({"GR": [1.0, 2.0]})  # no TVT_input / X / Y / Z at all

    diag = SP.offset_diagnostics(h, bank, exclude_well="nope", k=10)

    assert diag.n_valid_prefix == 0
    assert diag.prefix_rmse_best == float("inf")
    assert np.isnan(diag.offset_med)


def test_exclude_well_not_in_bank_still_computes(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()  # target NOT in bank
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h = make_target_well(n_known=60, slope_per_row=0.0)
    diag = SP.offset_diagnostics(h, bank, exclude_well=TARGET, k=10, method="plane")

    assert diag.n_valid_prefix == 60
    assert np.isfinite(diag.prefix_rmse_best)


# --------------------------------------------------------------------------- #
# real-data integration (skipped without local competition data)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not TRAIN_DIR.exists(),
    reason="requires local competition data under data/raw/train (not committed to the repo)",
)
def test_offset_diagnostics_integration_with_real_train_data() -> None:
    wells = D.list_wells("train")[:30]
    assert len(wells) == 30

    bank = SP.build_surface_bank(wells, split="train", stride=10)

    target = wells[0]
    h = D.load_horizontal(target, "train")
    diag = SP.offset_diagnostics(h, bank, exclude_well=target, k=10, method="idw")

    assert diag.n_valid_prefix > 0
    assert np.isfinite(diag.prefix_rmse_best)
