"""Tests for R19's opt-in spatial-position prior on ``rogii.registration.particle.track_pf``.

R19's idea: the surface-based spatial predictor (``rogii.spatial``, cached
per-well in ``data/processed/spatial_cache``) gives a per-row TVT curve that
is available even where GR is NaN (~30% of eval-zone rows median). Callers
anchor-align that curve against the well's own known-zone anchor before
passing it in (this module performs no such alignment itself), then
``track_pf`` folds it in as a weak Gaussian position prior alongside the
existing GR likelihood, so GR-NaN rows still get a reweighting step instead
of pure motion-model diffusion -- the mechanism this project's error
dissection blames for "step" mistracks (a mid-zone jump the tracker misses,
then carries as a fixed per-well offset for the rest of the well).

Synthetic wells here largely reuse ``tests/test_particle.py``'s
``_make_synthetic_well`` conventions (own local copy, per this repo's
per-test-file convention -- see ``test_beam.py`` / ``test_ncc.py``), plus one
new helper (``_make_synthetic_well_with_jump``) that injects a sudden
mid-eval-zone TVT level shift the GR-only tracker cannot instantly correct
for (tiny per-row process noise), which is the scenario the spatial prior is
meant to help with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.registration.particle import track_pf, track_pf_multi


def _typewell_gr(tvt: np.ndarray, spike_center: float = 260.0) -> np.ndarray:
    """Flat baseline plus one isolated, unambiguous Gaussian landmark."""
    base = np.full_like(np.asarray(tvt, dtype=float), 60.0)
    return base + 40.0 * np.exp(-0.5 * ((tvt - spike_center) / 3.0) ** 2)


def _make_synthetic_well(
    n_known: int = 300,
    n_eval: int = 300,
    anchor: float = 200.0,
    rate: float = 0.03,
    z_rate: float = 0.0,
    tw_noise_std: float = 0.3,
    gr_noise_std: float = 0.5,
    nan_frac: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build a synthetic (horizontal, typewell) pair with a known, constant TVT+Z rate.

    Identical construction to ``tests/test_particle.py``'s helper of the same
    name: the known zone's ``TVT_input`` (and ``Z``) already follow the same
    linear rate that continues into the eval zone, so the tracker's
    known-zone-tail rate calibration should recover ``rate`` almost exactly.
    Returns ``(h, tw, true_eval_tvt)``.
    """
    rng = np.random.default_rng(seed)

    tw_tvt = np.arange(0.0, 500.0, 0.5)
    tw_gr = _typewell_gr(tw_tvt) + rng.normal(0.0, tw_noise_std, size=tw_tvt.size)
    tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})

    n_total = n_known + n_eval
    md = np.arange(n_total, dtype=float)
    z = z_rate * md
    true_tvt = anchor + rate * md  # TVT + Z frame collapses to `rate` when z_rate=0

    tvt_input = np.concatenate([true_tvt[:n_known], np.full(n_eval, np.nan)])
    gr_clean = _typewell_gr(true_tvt) + rng.normal(0.0, gr_noise_std, size=n_total)

    if nan_frac > 0:
        drop = rng.random(gr_clean.size) < nan_frac
        gr_clean = np.where(drop, np.nan, gr_clean)

    h = pd.DataFrame({"MD": md, "Z": z, "GR": gr_clean, "TVT_input": tvt_input})
    true_eval_tvt = true_tvt[n_known:]
    return h, tw, true_eval_tvt


def _make_synthetic_well_with_jump(
    n_known: int = 200,
    n_eval: int = 200,
    anchor: float = 200.0,
    rate: float = 0.02,
    jump: float = 15.0,
    jump_frac: float = 0.5,
    gr_noise_std: float = 25.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Linear-drift well with a sudden mid-eval-zone TVT level shift.

    The known zone (and the eval zone *before* the jump) follow a constant,
    correctly-calibratable rate; at ``jump_frac`` through the eval zone the
    true path steps by ``jump`` ft instantly (simulating the kind of abrupt
    formation-level mistrack this project's error dissection identifies as
    the dominant source of per-well constant offset). The per-row process
    noise on ``pos`` (``_POS_PROCESS_NOISE = 0.005`` ft/row) is far too small
    for the particle cloud to diffuse across a 15 ft jump within the
    remaining eval-zone rows, so a GR-only tracker (especially with a noisy,
    only weakly-informative GR series -- ``gr_noise_std=25`` swamps the
    typewell landmark's own amplitude) systematically lags the jump for the
    rest of the well, exactly the scenario a well-aligned spatial prior
    (which "sees" the jump directly) should correct.
    """
    rng = np.random.default_rng(seed)

    tw_tvt = np.arange(0.0, 500.0, 0.5)
    tw_gr = _typewell_gr(tw_tvt) + rng.normal(0.0, 0.3, size=tw_tvt.size)
    tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})

    n_total = n_known + n_eval
    md = np.arange(n_total, dtype=float)
    z = np.zeros(n_total)

    true_tvt = anchor + rate * md
    jump_row = n_known + int(jump_frac * n_eval)
    true_tvt = true_tvt.copy()
    true_tvt[jump_row:] += jump

    tvt_input = np.concatenate([true_tvt[:n_known], np.full(n_eval, np.nan)])
    gr_clean = _typewell_gr(true_tvt) + rng.normal(0.0, gr_noise_std, size=n_total)

    h = pd.DataFrame({"MD": md, "Z": z, "GR": gr_clean, "TVT_input": tvt_input})
    true_eval_tvt = true_tvt[n_known:]
    return h, tw, true_eval_tvt


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- #
# 1. OFF (default) is bit-identical to the pre-R19 tracker
# --------------------------------------------------------------------------- #


def test_spatial_prior_off_matches_default_bit_for_bit() -> None:
    """``spatial_prior_tvt`` defaults to ``None``; passing it explicitly as
    ``None`` must be numerically identical to omitting it entirely, i.e. the
    R19 change is a strict no-op for every existing (pre-R19) call site."""
    h, tw, _ = _make_synthetic_well(nan_frac=0.3, seed=1)

    default_result = track_pf(h, tw, seed=1)
    explicit_none_result = track_pf(h, tw, seed=1, spatial_prior_tvt=None)

    np.testing.assert_array_equal(default_result.tvt, explicit_none_result.tvt)
    np.testing.assert_array_equal(default_result.std, explicit_none_result.std)
    assert default_result.n_updates == explicit_none_result.n_updates

    # track_pf_multi threads spatial_prior_tvt through **kwargs; same guarantee.
    default_multi = track_pf_multi(h, tw, n_seeds=4)
    explicit_none_multi = track_pf_multi(h, tw, n_seeds=4, spatial_prior_tvt=None)
    np.testing.assert_array_equal(default_multi.tvt, explicit_none_multi.tvt)
    np.testing.assert_array_equal(default_multi.std, explicit_none_multi.std)


def test_spatial_prior_off_reproduces_pre_r19_call_shape() -> None:
    """A plain call with no spatial_prior kwargs at all (the exact call shape
    every pre-R19 call site uses) must still work and be deterministic run to
    run for the same seed -- guards against the new opt-in parameters having
    accidentally changed the function's positional/keyword contract."""
    h, tw, true_eval_tvt = _make_synthetic_well(seed=2)

    result_a = track_pf(h, tw, seed=2)
    result_b = track_pf(h, tw, seed=2)

    np.testing.assert_array_equal(result_a.tvt, result_b.tvt)
    np.testing.assert_array_equal(result_a.std, result_b.std)
    assert result_a.tvt.shape == true_eval_tvt.shape


# --------------------------------------------------------------------------- #
# 2. ON: an accurate prior reduces RMSE vs. OFF on a mid-zone-jump well
# --------------------------------------------------------------------------- #


def test_spatial_prior_on_reduces_rmse_vs_off_for_a_mid_zone_jump() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well_with_jump(seed=1)

    result_off = track_pf(h, tw, seed=1)
    result_on = track_pf(
        h, tw, seed=1, spatial_prior_tvt=true_eval_tvt, spatial_prior_sigma=3.0
    )

    rmse_off = _rmse(result_off.tvt, true_eval_tvt)
    rmse_on = _rmse(result_on.tvt, true_eval_tvt)

    assert rmse_off > 5.0  # sanity: the jump genuinely breaks the GR-only tracker
    assert rmse_on < 0.75 * rmse_off  # accurate prior meaningfully closes the gap
    assert np.all(np.isfinite(result_on.tvt))


def test_spatial_prior_on_reduces_rmse_vs_off_via_track_pf_multi() -> None:
    """Same effect must hold through the ``track_pf_multi`` threading path
    (``track_pf_multi`` -- the function the R19 A/B harness actually drives
    in production -- forwards spatial_prior_tvt/spatial_prior_sigma via
    ``**kwargs`` to every per-seed :func:`track_pf` call)."""
    h, tw, true_eval_tvt = _make_synthetic_well_with_jump(seed=4)

    result_off = track_pf_multi(h, tw, n_seeds=4)
    result_on = track_pf_multi(
        h, tw, n_seeds=4, spatial_prior_tvt=true_eval_tvt, spatial_prior_sigma=3.0
    )

    rmse_off = _rmse(result_off.tvt, true_eval_tvt)
    rmse_on = _rmse(result_on.tvt, true_eval_tvt)

    assert rmse_on < rmse_off
    assert np.all(np.isfinite(result_on.tvt))


# --------------------------------------------------------------------------- #
# 3. GR entirely NaN: prior ON still tracks the prior curve (not the flat carry)
# --------------------------------------------------------------------------- #


def test_spatial_prior_tracks_prior_curve_not_flat_carry_when_gr_all_nan() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(seed=8)
    h = h.copy()
    known_count = int(h["TVT_input"].notna().sum())
    h.loc[known_count:, "GR"] = np.nan  # eval-zone GR entirely missing
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result_no_prior = track_pf(h, tw, seed=8)
    result_prior = track_pf(
        h, tw, seed=8, spatial_prior_tvt=true_eval_tvt, spatial_prior_sigma=5.0
    )

    flat_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)
    no_prior_rmse = _rmse(result_no_prior.tvt, true_eval_tvt)
    prior_rmse = _rmse(result_prior.tvt, true_eval_tvt)

    # Neither arm sees a single GR-likelihood update: the prior is the only
    # thing distinguishing the two runs.
    assert result_no_prior.n_updates == 0
    assert result_prior.n_updates == 0

    # The prior-guided path tracks the (accurate) spatial curve closely --
    # far closer than a flat carry-last extension of the anchor -- and beats
    # the same well's own GR-only (pure motion-model diffusion) run too.
    assert prior_rmse < 0.5
    assert prior_rmse < 0.2 * flat_rmse
    assert prior_rmse < no_prior_rmse
    assert np.all(np.isfinite(result_prior.tvt))
    assert np.all(np.isfinite(result_prior.std))


def test_spatial_prior_skips_non_finite_rows_like_missing_gr() -> None:
    """A ``spatial_prior_tvt`` row that is itself non-finite (NaN/inf) must be
    skipped for that row only (mirrors the existing missing-GR skip
    behaviour), not propagate NaN into the output or raise."""
    h, tw, true_eval_tvt = _make_synthetic_well(n_eval=40, seed=9)
    prior = true_eval_tvt.copy()
    prior[10:15] = np.nan
    prior[20] = np.inf

    result = track_pf(h, tw, seed=9, spatial_prior_tvt=prior, spatial_prior_sigma=5.0)

    assert result.tvt.shape == true_eval_tvt.shape
    assert np.all(np.isfinite(result.tvt))
    assert np.all(np.isfinite(result.std))


# --------------------------------------------------------------------------- #
# 4. Length mismatch raises ValueError (caller/programmer error, not
#    swallowed into the flat-anchor fallback)
# --------------------------------------------------------------------------- #


def test_spatial_prior_length_mismatch_raises_value_error() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(n_eval=50, seed=10)
    wrong_length_prior = true_eval_tvt[:-5]  # 45 rows vs. 50 eval-zone rows

    with pytest.raises(ValueError, match="spatial_prior_tvt"):
        track_pf(h, tw, seed=10, spatial_prior_tvt=wrong_length_prior)


def test_spatial_prior_length_mismatch_raises_value_error_via_track_pf_multi() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(n_eval=30, seed=11)
    too_long_prior = np.concatenate([true_eval_tvt, [123.0]])

    with pytest.raises(ValueError, match="spatial_prior_tvt"):
        track_pf_multi(h, tw, n_seeds=3, spatial_prior_tvt=too_long_prior)


def test_spatial_prior_empty_array_matches_empty_eval_zone() -> None:
    """An empty ``spatial_prior_tvt`` against an empty eval zone is a length
    match (0 == 0), not a mismatch -- must not raise."""
    h, tw, _ = _make_synthetic_well(n_eval=0, seed=12)

    result = track_pf(h, tw, spatial_prior_tvt=np.array([]))  # must not raise

    assert result.tvt.shape == (0,)


def test_spatial_prior_with_malformed_h_does_not_raise() -> None:
    """A malformed ``h`` (missing ``TVT_input``) with ``spatial_prior_tvt``
    set is a data-quality issue, not a caller error: ``track_pf`` must keep
    its never-raises contract and fall back instead of leaking an exception
    from the length-validation path."""
    malformed_h = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    _, tw, _ = _make_synthetic_well(seed=13)

    result = track_pf(malformed_h, tw, spatial_prior_tvt=np.zeros(5))  # must not raise

    assert np.all(np.isfinite(result.tvt)) or result.tvt.shape == (0,)
