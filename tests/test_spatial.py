"""Tests for the spatial-prior TVT predictor in ``rogii.spatial``.

Synthetic geometry: a single formation whose top depth is an exact plane
``depth(X, Y) = a*X + b*Y + c``. Offset ("bank") wells sample that plane
along straight lines; the target well's true TVT then satisfies
``TVT = -Z + depth(X, Y) + offset`` by construction, so a correct
leave-self-out KNN interpolation must recover the eval-zone TVT to high
accuracy while ``carry_last`` cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii import spatial as SP

# Plane: depth(X, Y) = A * X + B * Y + C
A, B, C = 0.5, 0.2, 100.0
TRUE_OFFSET = 7.0

TARGET = "wt"
FORMATION = ("ANCC",)


def plane_depth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return A * x + B * y + C


def make_bank_well(y_line: float, depth_bias: np.ndarray | float = 0.0) -> pd.DataFrame:
    """A straight offset well along X at ``y = y_line`` sampling the plane."""
    x = np.arange(0.0, 101.0)
    y = np.full_like(x, y_line)
    return pd.DataFrame(
        {
            "MD": x,
            "X": x,
            "Y": y,
            "Z": np.full_like(x, -50.0),
            "ANCC": plane_depth(x, y) + depth_bias,
            "TVT": np.zeros_like(x),
            "GR": np.zeros_like(x),
            "TVT_input": np.zeros_like(x),
        }
    )


def make_target_well(n_known: int = 60, n_eval: int = 41) -> tuple[pd.DataFrame, np.ndarray]:
    """Target well along ``y = 45`` with **no TVT / formation columns** (test schema).

    Returns ``(h, tvt_true_eval)``. Omitting ``TVT`` and the formation columns
    means any accidental read of them inside ``predict_spatial`` raises a
    ``KeyError`` instead of silently leaking (same pattern as
    ``tests/test_typewell.py``).
    """
    n = n_known + n_eval
    x = np.arange(0.0, float(n))
    y = np.full(n, 45.0)
    z = -50.0 - 0.1 * x  # varying Z so the -Z term actually matters
    tvt_true = -z + plane_depth(x, y) + TRUE_OFFSET
    tvt_input = tvt_true.copy()
    tvt_input[n_known:] = np.nan
    h = pd.DataFrame(
        {"MD": x, "X": x, "Y": y, "Z": z, "GR": np.zeros(n), "TVT_input": tvt_input}
    )
    return h, tvt_true[n_known:]


def install_fake_loader(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> None:
    def fake_load(well: str, split: str, root: str | None = None) -> pd.DataFrame:
        return frames[well]

    monkeypatch.setattr("rogii.data.load_horizontal", fake_load)


def bank_frames(target_bias: np.ndarray | float | None = None) -> dict[str, pd.DataFrame]:
    """Offset wells on lines y = 0, 10, ..., 100; optionally the target's own samples."""
    frames = {f"w{int(y):02d}": make_bank_well(float(y)) for y in range(0, 101, 10)}
    if target_bias is not None:
        frames[TARGET] = make_bank_well(45.0, depth_bias=target_bias)
    return frames


# --------------------------------------------------------------------------- #
# plane recovery (leave-self-out end to end)
# --------------------------------------------------------------------------- #


def test_recovers_plane_and_beats_carry_last(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames(target_bias=0.0)  # target present in bank, unbiased
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h, tvt_true = make_target_well()
    assert "TVT" not in h.columns and "ANCC" not in h.columns

    result = SP.predict_spatial(h, bank, exclude_well=TARGET, k=10)

    err = result.tvt - tvt_true
    rmse = float(np.sqrt(np.mean(err**2)))
    assert rmse < 1.0
    assert float(np.max(np.abs(err))) < 2.0
    assert result.prefix_rmse < 1.0
    assert result.n_formations_used == 1

    # carry_last (flat anchor) is far worse on this sloped plane
    anchor = h["TVT_input"].dropna().iloc[-1]
    carry_rmse = float(np.sqrt(np.mean((anchor - tvt_true) ** 2)))
    assert carry_rmse > 10 * rmse


def test_result_is_aligned_to_eval_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h, tvt_true = make_target_well(n_known=70, n_eval=31)
    result = SP.predict_spatial(h, bank, exclude_well=TARGET, k=10)

    assert result.tvt.shape == (31,)
    assert np.all(np.isfinite(result.tvt))
    np.testing.assert_allclose(result.tvt, tvt_true, atol=2.0)

    # with a single formation the composite must equal the best-formation tvt
    assert result.tvt_composite.shape == (31,)
    np.testing.assert_allclose(result.tvt_composite, result.tvt)


# --------------------------------------------------------------------------- #
# exclude_well (leave-self-out) actually excludes the target's samples
# --------------------------------------------------------------------------- #


def test_knn_excludes_self_samples_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Target's own bank samples carry a +500 bias; every query point lies
    # exactly on a target sample (distance 0), so *any* self leakage would
    # dominate the IDW weights and drag the result ~500 off the plane.
    frames = bank_frames(target_bias=500.0)
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    code = bank.well_to_code[TARGET]
    x = np.arange(0.0, 101.0)
    coords = np.column_stack([x, np.full_like(x, 45.0)])
    interp = SP._knn_idw_depth(
        bank.trees["ANCC"], coords, bank.depths["ANCC"], bank.well_codes["ANCC"], code, k=10
    )

    true = plane_depth(x, np.full_like(x, 45.0))
    assert float(np.max(np.abs(interp - true))) < 5.0  # ~500 if the self samples leaked

    # the local-plane interpolator must exclude self samples too, and being
    # exact on a plane it recovers the surface essentially perfectly
    interp_plane = SP._knn_plane_depth(
        bank.trees["ANCC"], coords, bank.depths["ANCC"], bank.well_codes["ANCC"], code, k=10
    )
    assert float(np.max(np.abs(interp_plane - true))) < 0.1


def test_predict_ignores_self_samples_with_sloped_bias(monkeypatch: pytest.MonkeyPatch) -> None:
    # A *sloped* self bias (5 * X) cannot be absorbed by the per-well constant
    # offset calibration, so if the target's own samples leaked into the KNN
    # the eval predictions would be off by hundreds of feet.
    x = np.arange(0.0, 101.0)
    frames = bank_frames(target_bias=5.0 * x)
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h, tvt_true = make_target_well()
    result = SP.predict_spatial(h, bank, exclude_well=TARGET, k=10)

    rmse = float(np.sqrt(np.mean((result.tvt - tvt_true) ** 2)))
    assert rmse < 1.0
    assert result.prefix_rmse < 1.0


# --------------------------------------------------------------------------- #
# fallbacks (never raise)
# --------------------------------------------------------------------------- #


def test_fallback_when_no_known_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h, _ = make_target_well()
    h["TVT_input"] = np.nan  # no anchor at all

    result = SP.predict_spatial(h, bank, exclude_well=TARGET, k=10)

    assert result.tvt.shape[0] == len(h)
    assert np.all(np.isfinite(result.tvt))
    assert result.prefix_rmse == float("inf")
    assert result.n_formations_used == 0


def test_fallback_when_bank_has_no_formations(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {w: f.assign(ANCC=np.nan) for w, f in bank_frames().items()}
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    assert bank.formations == ()

    h, _ = make_target_well()
    anchor = h["TVT_input"].dropna().iloc[-1]
    result = SP.predict_spatial(h, bank, exclude_well=TARGET, k=10)

    assert result.prefix_rmse == float("inf")
    assert result.n_formations_used == 0
    np.testing.assert_allclose(result.tvt, anchor)


def test_fallback_on_malformed_input_never_raises() -> None:
    bank = SP.SurfaceBank(formations=())
    h = pd.DataFrame({"GR": [1.0, 2.0]})  # no TVT_input / X / Y / Z at all

    result = SP.predict_spatial(h, bank, exclude_well="nope", k=10)

    assert result.prefix_rmse == float("inf")
    assert result.tvt.shape == (0,)


def test_exclude_well_not_in_bank_still_predicts(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()  # target NOT in bank
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    h, tvt_true = make_target_well()
    result = SP.predict_spatial(h, bank, exclude_well=TARGET, k=10)

    np.testing.assert_allclose(result.tvt, tvt_true, atol=2.0)


# --------------------------------------------------------------------------- #
# bank construction details
# --------------------------------------------------------------------------- #


def test_stride_subsamples_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)

    dense = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)
    sparse = SP.build_surface_bank(sorted(frames), stride=10, formations=FORMATION)

    assert sparse.depths["ANCC"].size < dense.depths["ANCC"].size
    # ceil(101 / 10) = 11 samples per 101-row well
    assert sparse.depths["ANCC"].size == 11 * len(frames)


def test_bank_skips_wells_that_fail_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()

    def flaky_load(well: str, split: str, root: str | None = None) -> pd.DataFrame:
        if well == "w50":
            raise OSError("corrupt file")
        return frames[well]

    monkeypatch.setattr("rogii.data.load_horizontal", flaky_load)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)

    assert bank.formations == FORMATION
    assert bank.depths["ANCC"].size == 101 * (len(frames) - 1)


# --------------------------------------------------------------------------- #
# R17: predict_spatial_v2 (opt-in quadratic surface + weight_power +
# per-row diagnostics). All new code paths -- predict_spatial /
# build_surface_bank / SpatialResult are untouched (see test_spatial.py's
# original tests above, all still passing unmodified).
# --------------------------------------------------------------------------- #

# Curved surface with real curvature terms, for the quadratic-vs-plane tests.
QA, QB, QC = 0.5, 0.2, 100.0
QXX, QYY, QXY = 0.01, 0.005, 0.002


def quad_depth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return QA * x + QB * y + QC + QXX * x**2 + QYY * y**2 + QXY * x * y


def make_bank_well_quad(y_line: float) -> pd.DataFrame:
    x = np.arange(0.0, 101.0)
    y = np.full_like(x, y_line)
    return pd.DataFrame(
        {
            "MD": x,
            "X": x,
            "Y": y,
            "Z": np.full_like(x, -50.0),
            "ANCC": quad_depth(x, y),
            "TVT": np.zeros_like(x),
            "GR": np.zeros_like(x),
            "TVT_input": np.zeros_like(x),
        }
    )


def bank_frames_quad() -> dict[str, pd.DataFrame]:
    return {f"w{int(y):02d}": make_bank_well_quad(float(y)) for y in range(0, 101, 5)}


def make_target_well_quad(n_known: int = 60, n_eval: int = 41) -> tuple[pd.DataFrame, np.ndarray]:
    n = n_known + n_eval
    x = np.arange(0.0, float(n))
    y = np.full(n, 45.0)
    z = -50.0 - 0.1 * x
    tvt_true = -z + quad_depth(x, y) + TRUE_OFFSET
    tvt_input = tvt_true.copy()
    tvt_input[n_known:] = np.nan
    h = pd.DataFrame(
        {"MD": x, "X": x, "Y": y, "Z": z, "GR": np.zeros(n), "TVT_input": tvt_input}
    )
    return h, tvt_true[n_known:]


def test_v2_plane_matches_predict_spatial_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    """At method="plane", weight_power=1.0, predict_spatial_v2 must exactly
    reproduce predict_spatial(method="plane")'s tvt/prefix_rmse (same
    underlying weighted-least-squares fit) -- proof that R17 changed nothing
    about the pre-existing computation."""
    frames = bank_frames(target_bias=0.0)
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)
    h, _ = make_target_well()

    v1 = SP.predict_spatial(h, bank, exclude_well=TARGET, k=10, method="plane")
    v2 = SP.predict_spatial_v2(h, bank, exclude_well=TARGET, k=10, method="plane")

    np.testing.assert_allclose(v2.tvt, v1.tvt)
    assert v2.prefix_rmse == pytest.approx(v1.prefix_rmse)
    assert v2.n_formations_used == v1.n_formations_used
    assert v2.resid_std.shape == v1.tvt.shape
    assert v2.n_donors.shape == v1.tvt.shape


def test_v2_grad_recovers_true_plane_slope(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames(target_bias=0.0)
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)
    h, _ = make_target_well()

    result = SP.predict_spatial_v2(h, bank, exclude_well=TARGET, k=10, method="plane")

    # ridge regularization (see _PLANE_RIDGE) admits a small bias
    np.testing.assert_allclose(result.grad_x, A, atol=1e-3)
    np.testing.assert_allclose(result.grad_y, B, atol=1e-3)
    # exact plane -> the local fit's own in-sample residual is ~0
    assert float(np.max(result.resid_std)) < 2e-3
    assert np.all(result.n_donors == 10)


def test_v2_quadratic_beats_plane_on_curved_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames_quad()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)
    h, tvt_true = make_target_well_quad()

    plane = SP.predict_spatial_v2(h, bank, exclude_well=TARGET, k=12, method="plane")
    quad = SP.predict_spatial_v2(h, bank, exclude_well=TARGET, k=12, method="quadratic")

    plane_rmse = float(np.sqrt(np.mean((plane.tvt - tvt_true) ** 2)))
    quad_rmse = float(np.sqrt(np.mean((quad.tvt - tvt_true) ** 2)))
    assert quad_rmse < plane_rmse
    assert quad_rmse < 0.5


def test_v2_quadratic_falls_back_to_plane_with_scarce_donors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """k=3 < _QUADRATIC_MIN_DONORS(6) -> every row's quadratic fit is
    under-determined and must fall back row-by-row to the plane fit."""
    frames = bank_frames(target_bias=0.0)
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)
    h, _ = make_target_well()

    plane = SP.predict_spatial_v2(h, bank, exclude_well=TARGET, k=3, method="plane")
    quad = SP.predict_spatial_v2(h, bank, exclude_well=TARGET, k=3, method="quadratic")

    assert np.all(quad.n_donors < SP._QUADRATIC_MIN_DONORS)
    np.testing.assert_allclose(quad.tvt, plane.tvt)


def test_v2_weight_power_zero_is_uniform_weighting() -> None:
    """Unit-test _knn_weights in isolation (no loader/bank machinery needed)."""
    from scipy.spatial import cKDTree

    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    tree = cKDTree(pts)
    depths = np.array([1.0, 2.0, 3.0, 4.0])
    codes = np.array([0, 0, 0, 0])
    coords = np.array([[0.5, 0.0]])

    _, w_idw = SP._knn_weights(tree, coords, depths, codes, exclude_code=-1, k=4, weight_power=1.0)
    _, w_uniform = SP._knn_weights(
        tree, coords, depths, codes, exclude_code=-1, k=4, weight_power=0.0
    )

    assert len(set(np.round(w_uniform[0], 9))) == 1  # all kept weights identical
    assert len(set(np.round(w_idw[0], 9))) > 1  # IDW weights vary with distance


def test_v2_never_raises_on_malformed_input() -> None:
    bank = SP.SurfaceBank(formations=())
    h = pd.DataFrame({"GR": [1.0, 2.0]})

    result = SP.predict_spatial_v2(h, bank, exclude_well="nope", k=10)

    assert result.prefix_rmse == float("inf")
    assert result.tvt.shape == (0,)
    assert result.resid_std.shape == (0,)
    assert result.n_donors.shape == (0,)


def test_v2_unknown_method_falls_back_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = bank_frames()
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)
    h, _ = make_target_well()

    result = SP.predict_spatial_v2(h, bank, exclude_well=TARGET, k=10, method="bogus")

    assert result.prefix_rmse == float("inf")
    assert result.n_formations_used == 0
