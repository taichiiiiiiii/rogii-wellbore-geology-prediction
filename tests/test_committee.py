"""Tests for the near-strike committee gate in ``rogii.registration.committee``.

Synthetic geometry mirrors ``tests/test_spatial.py``: a formation top whose
depth is an exact plane, sampled by donor ("bank") wells; a target well's
true TVT then satisfies ``TVT = -Z + depth(X, Y) + offset`` by construction.

Two bank shapes are used:

* "line" donors (straight lines at fixed Y, matching ``test_spatial.py``) for
  the alignment-gate tests (1 and 2) -- a clean single global plane, so the
  locally fit and regionally fit dip azimuths coincide everywhere (rotation
  guard trivially passes).
* a "hollowed grid + local cluster" donor set for the rotation-guard test
  (3) -- a broad grid follows one gradient everywhere except a small patch
  near the target well's own location, which is exclusively populated by a
  cluster following a very different gradient. This lets the *global*
  (all-bank) fit and the *local* (small-bandwidth, near-query) fit disagree
  by design, so the ``min_rot_deg`` guard has something real to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii import spatial as SP
from rogii.registration import committee as C

# --------------------------------------------------------------------------- #
# shared plane / donor-well helpers
# --------------------------------------------------------------------------- #

A, B, PLANE_C = 0.5, 0.2, 100.0
TRUE_OFFSET = 7.0
TARGET = "wt"
FORMATION = ("ANCC",)


def plane_depth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return A * x + B * y + PLANE_C


def make_line_well(y_line: float, depth_bias: np.ndarray | float = 0.0) -> pd.DataFrame:
    """A straight donor well along X at ``y = y_line`` sampling the plane."""
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


def line_bank_frames() -> dict[str, pd.DataFrame]:
    return {f"w{int(y):02d}": make_line_well(float(y)) for y in range(0, 101, 10)}


def install_fake_loader(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> None:
    def fake_load(well: str, split: str, root: str | None = None) -> pd.DataFrame:
        return frames[well]

    monkeypatch.setattr("rogii.data.load_horizontal", fake_load)


def build_line_bank(
    monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]
) -> SP.SurfaceBank:
    install_fake_loader(monkeypatch, frames)
    return SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)


def make_target_well(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, n_known: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Target well with **no TVT / formation columns** (matches the test schema).

    Returns ``(h, tvt_true_eval)``.
    """
    tvt_true = -z + plane_depth(x, y) + TRUE_OFFSET
    tvt_input = tvt_true.copy()
    tvt_input[n_known:] = np.nan
    n = len(x)
    h = pd.DataFrame(
        {
            "MD": np.arange(n, dtype=float),
            "X": x,
            "Y": y,
            "Z": z,
            "GR": np.zeros(n),
            "TVT_input": tvt_input,
        }
    )
    return h, tvt_true[n_known:]


def perpendicular_path(
    n: int, start: tuple[float, float], scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """A path moving exactly perpendicular to the plane's gradient (A, B) -- gate fires."""
    grad = np.array([A, B])
    perp = np.array([-B, A]) / np.linalg.norm(grad)
    t = np.arange(n, dtype=float)
    x = start[0] + perp[0] * t * scale
    y = start[1] + perp[1] * t * scale
    return x, y


def aligned_path(n: int, start: tuple[float, float], scale: float) -> tuple[np.ndarray, np.ndarray]:
    """A path moving exactly along the plane's gradient (A, B) -- gate never fires."""
    grad = np.array([A, B])
    unit = grad / np.linalg.norm(grad)
    t = np.arange(n, dtype=float)
    x = start[0] + unit[0] * t * scale
    y = start[1] + unit[1] * t * scale
    return x, y


# --------------------------------------------------------------------------- #
# 1. gate fires -> segments are substituted with the local-plane fit
# --------------------------------------------------------------------------- #


def test_gate_fires_and_substitutes_with_plane_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = line_bank_frames()
    bank = build_line_bank(monkeypatch, frames)

    n, n_known = 101, 60
    x, y = perpendicular_path(n, start=(60.0, 10.0), scale=0.6)
    z = -50.0 - 0.1 * x
    h, tvt_true_eval = make_target_well(x, y, z, n_known)
    assert "TVT" not in h.columns and "ANCC" not in h.columns

    alignment = C.strike_alignment(h, bank, exclude_well=TARGET, n_segments=16)
    assert alignment.shape == (n - n_known,)
    assert float(alignment.max()) < 0.35  # gate condition holds for every segment

    base = np.zeros(n - n_known)  # deliberately wrong ("carry a flat zero") baseline
    corrected = C.committee_correction(
        h, bank, base, exclude_well=TARGET, threshold=0.35, min_rot_deg=60.0
    )

    base_rmse = float(np.sqrt(np.mean((base - tvt_true_eval) ** 2)))
    corrected_rmse = float(np.sqrt(np.mean((corrected - tvt_true_eval) ** 2)))
    assert corrected_rmse < 1.0
    assert base_rmse > 50 * corrected_rmse  # committee clearly repairs the bad baseline


# --------------------------------------------------------------------------- #
# 2. gate never fires -> identity (base_path returned unchanged)
# --------------------------------------------------------------------------- #


def test_gate_never_fires_is_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = line_bank_frames()
    bank = build_line_bank(monkeypatch, frames)

    n, n_known = 101, 60
    x, y = aligned_path(n, start=(10.0, 10.0), scale=0.6)
    z = -50.0 - 0.1 * x
    h, _ = make_target_well(x, y, z, n_known)

    alignment = C.strike_alignment(h, bank, exclude_well=TARGET, n_segments=16)
    assert float(alignment.min()) > 0.9  # heading closely tracks the gradient everywhere

    base = np.arange(n - n_known, dtype=float) * 3.3 + 11.0  # arbitrary distinctive baseline
    corrected = C.committee_correction(
        h, bank, base, exclude_well=TARGET, threshold=0.35, min_rot_deg=60.0
    )
    np.testing.assert_array_equal(corrected, base)


# --------------------------------------------------------------------------- #
# 3. rotation guard blocks substitution when the local fit disagrees with the
#    regional one, even though the (global-azimuth) alignment gate condition
#    is met
# --------------------------------------------------------------------------- #


def _rot_guard_bank(monkeypatch: pytest.MonkeyPatch) -> SP.SurfaceBank:
    """Broad grid (gradient along X) hollowed out near (50, 45); a local cluster
    (gradient along Y) fills that hole exclusively."""
    a_broad, b_broad, c_broad = 0.5, 0.0, 100.0

    xs = np.arange(0.0, 101.0, 5.0)
    ys = np.arange(0.0, 101.0, 5.0)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    hole_r = 15.0
    keep = (gx - 50.0) ** 2 + (gy - 45.0) ** 2 > hole_r**2
    gx, gy = gx[keep], gy[keep]
    gdepth = a_broad * gx + b_broad * gy + c_broad

    rng = np.random.default_rng(0)
    n_local, local_r = 20, 8.0
    ang = rng.uniform(0, 2 * np.pi, n_local)
    rad = rng.uniform(0, local_r, n_local)
    lx = 50.0 + rad * np.cos(ang)
    ly = 45.0 + rad * np.sin(ang)
    ldepth = 2.0 * ly + 50.0  # gradient purely along Y -- ~90 deg from the broad grid's 0 deg

    def make_point_well(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> pd.DataFrame:
        n = len(x)
        return pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "X": x,
                "Y": y,
                "Z": np.zeros(n),
                "ANCC": depth,
                "TVT": np.zeros(n),
                "GR": np.zeros(n),
                "TVT_input": np.zeros(n),
            }
        )

    frames = {
        "donor_broad": make_point_well(gx, gy, gdepth),
        "donor_local": make_point_well(lx, ly, ldepth),
    }
    return build_line_bank(monkeypatch, frames)


def _rot_guard_target() -> pd.DataFrame:
    n, n_known = 40, 20
    t = np.arange(n, dtype=float)
    x = np.full(n, 50.0)
    y = 45.0 + (t - 20.0) * 0.05  # short path near the local-cluster hole, heading ~90 deg
    z = np.zeros(n)
    tvt_input = np.where(t < n_known, 0.0, np.nan)
    return pd.DataFrame(
        {"MD": t, "X": x, "Y": y, "Z": z, "GR": np.zeros(n), "TVT_input": tvt_input}
    )


def test_rot_guard_blocks_substitution_when_local_direction_diverges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = _rot_guard_bank(monkeypatch)
    h = _rot_guard_target()

    alignment = C.strike_alignment(h, bank, exclude_well=TARGET, n_segments=4, formation="ANCC")
    assert float(alignment.max()) < 0.35  # the (global-azimuth) gate condition is met everywhere

    base = np.arange(20, dtype=float) * 3.3 + 11.0

    # default-ish rotation guard (60 deg, matching K16's ROT_MAX): local vs regional
    # direction disagree by ~90 deg here, so the guard must block substitution.
    guarded = C.committee_correction(
        h,
        bank,
        base,
        exclude_well=TARGET,
        threshold=0.35,
        min_rot_deg=60.0,
        n_segments=4,
        bandwidth=3.0,
        radius_mult=4.0,
        min_local_points=10,
        formation="ANCC",
    )
    np.testing.assert_array_equal(guarded, base)

    # sanity: with the guard effectively disabled (a huge min_rot_deg), the same
    # gated segments *do* get substituted -- proving the guard above was load-bearing.
    unguarded = C.committee_correction(
        h,
        bank,
        base,
        exclude_well=TARGET,
        threshold=0.35,
        min_rot_deg=180.0,
        n_segments=4,
        bandwidth=3.0,
        radius_mult=4.0,
        min_local_points=10,
        formation="ANCC",
    )
    assert not np.array_equal(unguarded, base)


# --------------------------------------------------------------------------- #
# 4. leave-self-out: the excluded well's own (heavily biased) bank samples
#    must not influence the result
# --------------------------------------------------------------------------- #


def test_leave_self_out_ignores_own_biased_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = line_bank_frames()

    n, n_known = 101, 60
    x, y = perpendicular_path(n, start=(60.0, 10.0), scale=0.6)
    z = -50.0 - 0.1 * x

    # inject the target's own samples into the bank, biased by +100000 ft --
    # any leak would blow the result up by orders of magnitude.
    frames[TARGET] = pd.DataFrame(
        {
            "MD": np.arange(n, dtype=float),
            "X": x,
            "Y": y,
            "Z": z,
            "ANCC": plane_depth(x, y) + 100_000.0,
            "TVT": np.zeros(n),
            "GR": np.zeros(n),
            "TVT_input": np.zeros(n),
        }
    )
    bank = build_line_bank(monkeypatch, frames)
    assert TARGET in bank.well_to_code

    h, tvt_true_eval = make_target_well(x, y, z, n_known)
    base = np.zeros(n - n_known)
    corrected = C.committee_correction(
        h, bank, base, exclude_well=TARGET, threshold=0.35, min_rot_deg=60.0
    )

    rmse = float(np.sqrt(np.mean((corrected - tvt_true_eval) ** 2)))
    assert rmse < 1.0  # unaffected by the +100000 self-bias


# --------------------------------------------------------------------------- #
# 5. output length always matches the evaluation-zone length
# --------------------------------------------------------------------------- #


def test_output_length_matches_eval_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = line_bank_frames()
    bank = build_line_bank(monkeypatch, frames)

    n, n_known = 101, 70
    x, y = perpendicular_path(n, start=(60.0, 10.0), scale=0.6)
    z = -50.0 - 0.1 * x
    h, _ = make_target_well(x, y, z, n_known)

    alignment = C.strike_alignment(h, bank, exclude_well=TARGET)
    assert alignment.shape == (n - n_known,)

    base = np.zeros(n - n_known)
    corrected = C.committee_correction(h, bank, base, exclude_well=TARGET)
    assert corrected.shape == (n - n_known,)


# --------------------------------------------------------------------------- #
# 6. leak-structure guard: works with no TVT / formation columns on ``h``
# --------------------------------------------------------------------------- #


def test_runs_without_tvt_or_formation_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = line_bank_frames()
    bank = build_line_bank(monkeypatch, frames)

    n, n_known = 101, 60
    x, y = perpendicular_path(n, start=(60.0, 10.0), scale=0.6)
    z = -50.0 - 0.1 * x
    h, _ = make_target_well(x, y, z, n_known)
    assert "TVT" not in h.columns
    for formation in SP.FORMATIONS:
        assert formation not in h.columns

    # must not raise (would surface as a KeyError if either function read a
    # train-only column instead of only X/Y/Z/TVT_input plus the bank).
    C.strike_alignment(h, bank, exclude_well=TARGET)
    C.committee_correction(h, bank, np.zeros(n - n_known), exclude_well=TARGET)


# --------------------------------------------------------------------------- #
# 7. boundary: a single-row evaluation zone
# --------------------------------------------------------------------------- #


def test_single_row_eval_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = line_bank_frames()
    bank = build_line_bank(monkeypatch, frames)

    n, n_known = 101, 100
    x, y = perpendicular_path(n, start=(60.0, 10.0), scale=0.6)
    z = -50.0 - 0.1 * x
    h, _ = make_target_well(x, y, z, n_known)

    alignment = C.strike_alignment(h, bank, exclude_well=TARGET)
    assert alignment.shape == (1,)

    base = np.array([123.4])
    corrected = C.committee_correction(h, bank, base, exclude_well=TARGET)
    assert corrected.shape == (1,)


# --------------------------------------------------------------------------- #
# 8. exception safety: malformed inputs never raise
# --------------------------------------------------------------------------- #


def test_never_raises_on_malformed_input() -> None:
    empty_bank = SP.SurfaceBank(formations=())
    h = pd.DataFrame({"GR": [1.0, 2.0]})  # no X / Y / Z / TVT_input at all

    alignment = C.strike_alignment(h, empty_bank, exclude_well="nope")
    assert isinstance(alignment, np.ndarray)

    base = np.array([1.0, 2.0, 3.0])
    corrected = C.committee_correction(h, empty_bank, base, exclude_well="nope")
    assert isinstance(corrected, np.ndarray)


def test_never_raises_with_bank_but_no_formation_columns_and_mismatched_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = line_bank_frames()
    bank = build_line_bank(monkeypatch, frames)

    n, n_known = 101, 60
    x, y = perpendicular_path(n, start=(60.0, 10.0), scale=0.6)
    z = -50.0 - 0.1 * x
    h, _ = make_target_well(x, y, z, n_known)

    # base_path length mismatched with the eval zone -> must fall back safely,
    # not raise or silently broadcast into a wrong-length array.
    wrong_base = np.zeros(3)
    result = C.committee_correction(h, bank, wrong_base, exclude_well=TARGET)
    assert isinstance(result, np.ndarray)


# --------------------------------------------------------------------------- #
# 9. bank with no usable formations -> alignment reports "no gate" (all 1.0)
# --------------------------------------------------------------------------- #


def test_no_formations_in_bank_reports_full_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {w: f.assign(ANCC=np.nan) for w, f in line_bank_frames().items()}
    install_fake_loader(monkeypatch, frames)
    bank = SP.build_surface_bank(sorted(frames), stride=1, formations=FORMATION)
    assert bank.formations == ()

    n, n_known = 101, 60
    x, y = perpendicular_path(n, start=(60.0, 10.0), scale=0.6)
    z = -50.0 - 0.1 * x
    h, _ = make_target_well(x, y, z, n_known)

    alignment = C.strike_alignment(h, bank, exclude_well=TARGET)
    np.testing.assert_array_equal(alignment, np.ones(n - n_known))

    base = np.zeros(n - n_known)
    corrected = C.committee_correction(h, bank, base, exclude_well=TARGET)
    np.testing.assert_array_equal(corrected, base)
