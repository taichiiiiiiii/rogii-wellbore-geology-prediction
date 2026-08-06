"""Tests for the particle-filter (PF) registration tracker in ``rogii.registration.particle``.

Synthetic wells here use a *known, constant* rate (``d(TVT + Z)/dMD``) that
continues unchanged from the known zone into the eval zone, plus one
isolated, unambiguous GR landmark in the typewell (see ``test_ncc.py`` for the
same rationale: real typewell GR is locally repetitive/ambiguous, which is not
what these tests are meant to probe). This validates the tracker's core
mechanics -- rate calibration from the known-zone tail, GR-guided correction,
and graceful degradation -- independent of how ambiguous real GR signals are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.registration.particle import (
    PFBMAResult,
    PFResult,
    track_pf,
    track_pf_bma,
    track_pf_multi,
)


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

    The known zone's ``TVT_input`` (and ``Z``) already follow the same linear
    rate that continues into the eval zone, so the tracker's known-zone-tail
    rate calibration should recover ``rate`` almost exactly. Returns
    ``(h, tw, true_eval_tvt)``.
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


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- #
# core recovery behaviour
# --------------------------------------------------------------------------- #


def test_track_pf_recovers_known_drift_better_than_flat_anchor() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well()
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result = track_pf(h, tw, seed=0)

    assert isinstance(result, PFResult)
    assert result.tvt.shape == true_eval_tvt.shape
    assert result.std.shape == true_eval_tvt.shape

    pf_rmse = _rmse(result.tvt, true_eval_tvt)
    flat_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)

    assert flat_rmse > 1.0  # sanity: the synthetic drift is non-trivial
    assert pf_rmse < 0.5 * flat_rmse


def test_track_pf_multi_recovers_known_drift_better_than_flat_anchor() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(seed=3)
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result = track_pf_multi(h, tw, n_seeds=4)

    pf_rmse = _rmse(result.tvt, true_eval_tvt)
    flat_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)

    assert pf_rmse < 0.5 * flat_rmse
    assert np.all(np.isfinite(result.std))


def test_track_pf_tolerates_missing_gr_and_still_beats_flat_anchor() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(nan_frac=0.3, seed=1)
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result = track_pf(h, tw, seed=1)

    pf_rmse = _rmse(result.tvt, true_eval_tvt)
    flat_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)

    assert pf_rmse < flat_rmse
    assert np.all(np.isfinite(result.tvt))
    assert result.n_updates < true_eval_tvt.size  # some rows really were skipped


# --------------------------------------------------------------------------- #
# fill_gr_gaps opt-in flag (R6-b①: eval-zone GR NaN interpolation before PF)
# --------------------------------------------------------------------------- #


def test_track_pf_fill_gr_gaps_off_matches_default_bit_for_bit() -> None:
    """``fill_gr_gaps`` defaults to False; passing it explicitly as False must
    be numerically identical to omitting it, i.e. adding the flag changes
    nothing for every existing (pre-flag) call site."""
    h, tw, _ = _make_synthetic_well(nan_frac=0.3, seed=1)

    default_result = track_pf(h, tw, seed=1)
    explicit_off_result = track_pf(h, tw, seed=1, fill_gr_gaps=False)

    np.testing.assert_array_equal(default_result.tvt, explicit_off_result.tvt)
    np.testing.assert_array_equal(default_result.std, explicit_off_result.std)
    assert default_result.n_updates == explicit_off_result.n_updates

    default_multi = track_pf_multi(h, tw, n_seeds=4)
    explicit_off_multi = track_pf_multi(h, tw, n_seeds=4, fill_gr_gaps=False)
    np.testing.assert_array_equal(default_multi.tvt, explicit_off_multi.tvt)
    np.testing.assert_array_equal(default_multi.std, explicit_off_multi.std)


def test_track_pf_fill_gr_gaps_on_updates_previously_skipped_gr_rows() -> None:
    """With ``fill_gr_gaps=True`` on a well with scattered eval-zone GR NaN,
    every eval-zone row must contribute a likelihood update (the interpolated
    / typewell-mean-filled GR is always finite), whereas the flag-off path
    genuinely skips the NaN rows (as asserted by the sibling
    ``test_track_pf_tolerates_missing_gr_and_still_beats_flat_anchor`` test)."""
    h, tw, true_eval_tvt = _make_synthetic_well(nan_frac=0.3, seed=1)

    result_off = track_pf(h, tw, seed=1, fill_gr_gaps=False)
    result_on = track_pf(h, tw, seed=1, fill_gr_gaps=True)

    assert result_off.n_updates < true_eval_tvt.size
    assert result_on.n_updates == true_eval_tvt.size  # gap-filled: no row skipped
    assert np.all(np.isfinite(result_on.tvt))
    assert np.all(np.isfinite(result_on.std))
    # the two paths must differ (the flag actually changed the likelihood inputs)
    assert not np.array_equal(result_off.tvt, result_on.tvt)


def test_track_pf_fill_gr_gaps_never_raises_on_all_nan_typewell_gr() -> None:
    """A typewell with entirely-NaN GR is degenerate for ``_fill_gr_gaps``'s own
    final fallback too (its typewell-mean fill becomes NaN itself); combined
    with ``fill_gr_gaps=True`` the tracker must still not raise and must fall
    back to the flat-anchor path like any other degenerate-typewell case."""
    h, _, _ = _make_synthetic_well(n_eval=20, nan_frac=0.3, seed=1)
    tw_all_nan_gr = pd.DataFrame({"TVT": [100.0, 200.0], "GR": [np.nan, np.nan]})

    result = track_pf(h, tw_all_nan_gr, fill_gr_gaps=True)  # must not raise

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert result.tvt.shape == (20,)
    np.testing.assert_allclose(result.tvt, anchor)
    np.testing.assert_allclose(result.std, np.inf)


# --------------------------------------------------------------------------- #
# uncertainty behaviour
# --------------------------------------------------------------------------- #


def test_track_pf_std_grows_when_gr_never_updates() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=400, seed=2)
    h = h.copy()
    known_count = int(h["TVT_input"].notna().sum())
    h.loc[known_count:, "GR"] = np.nan  # eval zone GR entirely missing

    result = track_pf(h, tw, seed=2)

    assert result.n_updates == 0
    assert np.all(np.isfinite(result.tvt))
    # No reweighting ever happens, so weights stay uniform and resampling never
    # triggers (ESS == n_particles at all times); the particle cloud undergoes
    # a pure random walk, so its spread should trend upward overall. Individual
    # steps can wobble slightly due to sampling noise, so check the trend
    # robustly (correlation with row index + a healthy start-vs-end margin)
    # rather than requiring strict elementwise monotonicity.
    idx = np.arange(result.std.size)
    corr = np.corrcoef(idx, result.std)[0, 1]
    assert corr > 0.8
    assert result.std[-1] > result.std[0] * 1.5


def test_track_pf_all_nan_gr_never_raises_and_stays_finite() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=50, seed=4)
    h = h.copy()
    h["GR"] = np.nan

    result = track_pf(h, tw, seed=4)

    assert result.tvt.shape == (50,)
    assert result.n_updates == 0
    assert np.all(np.isfinite(result.tvt))
    assert np.all(result.std >= 0.0)


# --------------------------------------------------------------------------- #
# fallback / exception-safety behaviour
# --------------------------------------------------------------------------- #


def test_track_pf_never_raises_on_degenerate_typewell() -> None:
    h, _, _ = _make_synthetic_well(n_eval=20)
    tiny_tw = pd.DataFrame({"TVT": [100.0], "GR": [55.0]})

    result = track_pf(h, tiny_tw)  # must not raise

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert result.tvt.shape == (20,)
    np.testing.assert_allclose(result.tvt, anchor)
    np.testing.assert_allclose(result.std, np.inf)


def test_track_pf_never_raises_on_missing_columns() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=20)
    h_missing_z = h.drop(columns=["Z"])

    result = track_pf(h_missing_z, tw)  # must not raise despite missing Z column

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    assert result.tvt.shape == (20,)
    np.testing.assert_allclose(result.tvt, anchor)
    np.testing.assert_allclose(result.std, np.inf)


def test_track_pf_never_raises_on_completely_malformed_input() -> None:
    empty_h = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    empty_tw = pd.DataFrame({"also_wrong": [1, 2, 3]})

    result = track_pf(empty_h, empty_tw)  # must not raise

    assert isinstance(result, PFResult)
    assert result.tvt.shape == result.std.shape


def test_track_pf_never_raises_when_no_known_anchor() -> None:
    h, tw, _ = _make_synthetic_well(n_known=0, n_eval=20)

    result = track_pf(h, tw)  # must not raise despite no TVT_input anchor at all

    assert result.tvt.shape == (20,)
    np.testing.assert_allclose(result.std, np.inf)


def test_track_pf_empty_eval_zone_returns_empty_arrays() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=0)

    result = track_pf(h, tw)

    assert result.tvt.shape == (0,)
    assert result.std.shape == (0,)
    assert result.n_updates == 0


def test_track_pf_multi_never_raises_on_completely_malformed_input() -> None:
    empty_h = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    empty_tw = pd.DataFrame({"also_wrong": [1, 2, 3]})

    result = track_pf_multi(empty_h, empty_tw, n_seeds=3)

    assert isinstance(result, PFResult)
    assert result.tvt.shape == result.std.shape


# --------------------------------------------------------------------------- #
# track_pf_bma: PF-BMA (seed-softmax over cumulative log-likelihood)
# --------------------------------------------------------------------------- #

_BMA_SCALES = (3.0, 5.0, 8.0, 12.0)
_BMA_N_SEEDS = 6
_BMA_N_PARTICLES = 128  # small for test speed; mechanics don't depend on particle count


def test_track_pf_bma_recovers_known_drift_better_than_flat_anchor() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well()
    anchor = float(h["TVT_input"].dropna().iloc[-1])

    result = track_pf_bma(
        h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=_BMA_N_SEEDS, bma_scales=_BMA_SCALES
    )

    assert isinstance(result, PFBMAResult)
    assert set(result.tvt_by_scale) == set(_BMA_SCALES)
    assert set(result.std_by_scale) == set(_BMA_SCALES)
    flat_rmse = _rmse(np.full_like(true_eval_tvt, anchor), true_eval_tvt)
    assert flat_rmse > 1.0  # sanity: the synthetic drift is non-trivial

    for scale in _BMA_SCALES:
        tvt = result.tvt_by_scale[scale]
        std = result.std_by_scale[scale]
        assert tvt.shape == true_eval_tvt.shape  # aligned to eval_mask(h) order/length
        assert std.shape == true_eval_tvt.shape
        assert np.all(np.isfinite(tvt))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)
        assert _rmse(tvt, true_eval_tvt) < 0.5 * flat_rmse


def test_track_pf_bma_log_lik_shape_and_finiteness() -> None:
    h, tw, _ = _make_synthetic_well(seed=5)

    result = track_pf_bma(
        h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=_BMA_N_SEEDS, bma_scales=_BMA_SCALES
    )

    assert result.log_lik.shape == (_BMA_N_SEEDS,)
    assert np.all(np.isfinite(result.log_lik))
    assert result.n_updates > 0  # GR is fully present in this synthetic well


def test_track_pf_bma_leak_guard_no_tvt_or_formation_columns() -> None:
    """Must run fine on an `h` with no `TVT`/formation columns at all (no KeyError)."""
    h, tw, _ = _make_synthetic_well(n_eval=40, seed=6)
    assert "TVT" not in h.columns
    for formation_col in ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "Geology"):
        assert formation_col not in h.columns

    result = track_pf_bma(h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=3, bma_scales=_BMA_SCALES)

    assert isinstance(result, PFBMAResult)
    for scale in _BMA_SCALES:
        assert np.all(np.isfinite(result.tvt_by_scale[scale]))


def test_track_pf_bma_log_lik_prefers_matching_gr_signal() -> None:
    """Sanity check on the likelihood machinery: a well whose eval-zone GR is
    consistent with the typewell signature should yield a higher cumulative
    log-likelihood (averaged across scales' softmax input, i.e. the raw
    per-seed `log_lik`) than an otherwise-identical well whose eval-zone GR is
    replaced with signal-free noise unrelated to the typewell.
    """
    h_clean, tw, _ = _make_synthetic_well(gr_noise_std=0.5, seed=7)

    h_noisy = h_clean.copy()
    known_count = int(h_clean["TVT_input"].notna().sum())
    rng = np.random.default_rng(7)
    h_noisy.loc[known_count:, "GR"] = rng.normal(60.0, 25.0, size=len(h_noisy) - known_count)

    result_clean = track_pf_bma(
        h_clean, tw, n_particles=_BMA_N_PARTICLES, n_seeds=_BMA_N_SEEDS, bma_scales=_BMA_SCALES
    )
    result_noisy = track_pf_bma(
        h_noisy, tw, n_particles=_BMA_N_PARTICLES, n_seeds=_BMA_N_SEEDS, bma_scales=_BMA_SCALES
    )

    assert np.mean(result_clean.log_lik) > np.mean(result_noisy.log_lik)


def test_track_pf_bma_tolerates_all_nan_gr_and_stays_finite() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=30, seed=8)
    h = h.copy()
    known_count = int(h["TVT_input"].notna().sum())
    h.loc[known_count:, "GR"] = np.nan  # eval zone GR entirely missing

    result = track_pf_bma(h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=4, bma_scales=_BMA_SCALES)

    assert result.n_updates == 0
    # No likelihood evidence at all -> log_lik stays at its initial value (0.0) for every seed
    np.testing.assert_allclose(result.log_lik, 0.0)
    for scale in _BMA_SCALES:
        assert np.all(np.isfinite(result.tvt_by_scale[scale]))
        assert np.all(np.isfinite(result.std_by_scale[scale]))


def test_track_pf_bma_boundary_single_eval_row_and_minimal_prefix() -> None:
    h, tw, true_eval_tvt = _make_synthetic_well(n_known=2, n_eval=1, seed=9)

    result = track_pf_bma(h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=3, bma_scales=_BMA_SCALES)

    assert true_eval_tvt.shape == (1,)
    for scale in _BMA_SCALES:
        assert result.tvt_by_scale[scale].shape == (1,)
        assert np.all(np.isfinite(result.tvt_by_scale[scale]))


def test_track_pf_bma_empty_eval_zone_returns_empty_arrays() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=0, seed=10)

    result = track_pf_bma(h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=3, bma_scales=_BMA_SCALES)

    for scale in _BMA_SCALES:
        assert result.tvt_by_scale[scale].shape == (0,)
        assert result.std_by_scale[scale].shape == (0,)
    assert result.n_updates == 0


def test_track_pf_bma_never_raises_on_degenerate_typewell() -> None:
    h, _, _ = _make_synthetic_well(n_eval=20, seed=11)
    tiny_tw = pd.DataFrame({"TVT": [100.0], "GR": [55.0]})

    result = track_pf_bma(h, tiny_tw, n_particles=_BMA_N_PARTICLES, bma_scales=_BMA_SCALES)

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    for scale in _BMA_SCALES:
        assert result.tvt_by_scale[scale].shape == (20,)
        np.testing.assert_allclose(result.tvt_by_scale[scale], anchor)
        np.testing.assert_allclose(result.std_by_scale[scale], np.inf)
    np.testing.assert_allclose(result.log_lik, -np.inf)


def test_track_pf_bma_never_raises_on_missing_columns() -> None:
    h, tw, _ = _make_synthetic_well(n_eval=20, seed=12)
    h_missing_z = h.drop(columns=["Z"])

    result = track_pf_bma(h_missing_z, tw, n_particles=_BMA_N_PARTICLES, bma_scales=_BMA_SCALES)

    anchor = float(h["TVT_input"].dropna().iloc[-1])
    for scale in _BMA_SCALES:
        assert result.tvt_by_scale[scale].shape == (20,)
        np.testing.assert_allclose(result.tvt_by_scale[scale], anchor)


def test_track_pf_bma_never_raises_on_completely_malformed_input() -> None:
    empty_h = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    empty_tw = pd.DataFrame({"also_wrong": [1, 2, 3]})

    result = track_pf_bma(empty_h, empty_tw, n_seeds=3, bma_scales=_BMA_SCALES)

    assert isinstance(result, PFBMAResult)
    for scale in _BMA_SCALES:
        assert result.tvt_by_scale[scale].shape == result.std_by_scale[scale].shape


def test_track_pf_bma_never_raises_when_no_known_anchor() -> None:
    h, tw, _ = _make_synthetic_well(n_known=0, n_eval=20, seed=13)

    result = track_pf_bma(h, tw, n_particles=_BMA_N_PARTICLES, bma_scales=_BMA_SCALES)

    for scale in _BMA_SCALES:
        assert result.tvt_by_scale[scale].shape == (20,)
        np.testing.assert_allclose(result.std_by_scale[scale], np.inf)


# --------------------------------------------------------------------------- #
# fill_gr_gaps opt-in flag on track_pf_bma (R6-b①: same eval-zone GR NaN
# interpolation, wired through the PF-BMA path used by the bank-builder's
# Phase B / the pf_bma_scale3 candidate)
# --------------------------------------------------------------------------- #


def test_track_pf_bma_fill_gr_gaps_off_matches_default_bit_for_bit() -> None:
    """``fill_gr_gaps`` defaults to False; passing it explicitly as False must
    be numerically identical to omitting it for every scale's path/std and for
    ``log_lik``/``n_updates`` -- i.e. adding the flag changes nothing for
    every existing (pre-flag) ``track_pf_bma`` call site."""
    h, tw, _ = _make_synthetic_well(nan_frac=0.3, seed=14)

    default_result = track_pf_bma(
        h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=_BMA_N_SEEDS, bma_scales=_BMA_SCALES
    )
    explicit_off_result = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        fill_gr_gaps=False,
    )

    for scale in _BMA_SCALES:
        np.testing.assert_array_equal(
            default_result.tvt_by_scale[scale], explicit_off_result.tvt_by_scale[scale]
        )
        np.testing.assert_array_equal(
            default_result.std_by_scale[scale], explicit_off_result.std_by_scale[scale]
        )
    np.testing.assert_array_equal(default_result.log_lik, explicit_off_result.log_lik)
    assert default_result.n_updates == explicit_off_result.n_updates


def test_track_pf_bma_fill_gr_gaps_on_updates_previously_skipped_gr_rows() -> None:
    """With ``fill_gr_gaps=True`` on a well with scattered eval-zone GR NaN,
    every eval-zone row must contribute a likelihood update for every seed
    (``n_updates`` -- taken from the first seed, identical across seeds --
    must equal the eval-zone length), whereas the flag-off path genuinely
    skips the NaN rows."""
    h, tw, true_eval_tvt = _make_synthetic_well(nan_frac=0.3, seed=14)

    result_off = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        fill_gr_gaps=False,
    )
    result_on = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        fill_gr_gaps=True,
    )

    assert result_off.n_updates < true_eval_tvt.size
    assert result_on.n_updates == true_eval_tvt.size  # gap-filled: no row skipped
    for scale in _BMA_SCALES:
        assert np.all(np.isfinite(result_on.tvt_by_scale[scale]))
        assert np.all(np.isfinite(result_on.std_by_scale[scale]))
        # the two paths must differ (the flag actually changed the likelihood inputs)
        assert not np.array_equal(
            result_off.tvt_by_scale[scale], result_on.tvt_by_scale[scale]
        )


# --------------------------------------------------------------------------- #
# init_pos_std parameter (R6-b(3): initial particle-position spread A/B)
# --------------------------------------------------------------------------- #


def test_track_pf_bma_init_pos_std_default_matches_bit_for_bit_and_wider_differs() -> None:
    """``init_pos_std`` defaults to the adopted 0.3; passing it explicitly as
    0.3 must be numerically identical to omitting it (existing call sites
    unchanged), and a wider spread (the reference notebook's 2.0) must
    actually change the output (the parameter is really wired through)."""
    h, tw, _ = _make_synthetic_well(seed=16)

    default_result = track_pf_bma(
        h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=_BMA_N_SEEDS, bma_scales=_BMA_SCALES
    )
    explicit_result = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        init_pos_std=0.3,
    )
    wider_result = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        init_pos_std=2.0,
    )

    for scale in _BMA_SCALES:
        np.testing.assert_array_equal(
            default_result.tvt_by_scale[scale], explicit_result.tvt_by_scale[scale]
        )
        np.testing.assert_array_equal(
            default_result.std_by_scale[scale], explicit_result.std_by_scale[scale]
        )
        assert not np.array_equal(
            default_result.tvt_by_scale[scale], wider_result.tvt_by_scale[scale]
        )
        assert np.all(np.isfinite(wider_result.tvt_by_scale[scale]))
    np.testing.assert_array_equal(default_result.log_lik, explicit_result.log_lik)

    # track_pf shares the same init site; its default must also be bit-identical.
    pf_default = track_pf(h, tw, seed=0)
    pf_explicit = track_pf(h, tw, seed=0, init_pos_std=0.3)
    np.testing.assert_array_equal(pf_default.tvt, pf_explicit.tvt)
    np.testing.assert_array_equal(pf_default.std, pf_explicit.std)


# --------------------------------------------------------------------------- #
# adaptive_gr_scales parameter (R6-b(4): per-well adaptive GR-likelihood scale)
# --------------------------------------------------------------------------- #


def test_track_pf_bma_adaptive_gr_scales_default_matches_bit_for_bit_and_on_differs() -> None:
    """``adaptive_gr_scales`` defaults to False; passing it explicitly as
    False must be numerically identical to omitting it (existing call sites
    unchanged), and True must actually change the output (the synthetic
    well's calibrated residual std ~0.5 clips to a=8 -> grid (4.8, 8, 12.8),
    different from the fixed (8, 12, 20, 30))."""
    h, tw, _ = _make_synthetic_well(seed=17)

    default_result = track_pf_bma(
        h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=_BMA_N_SEEDS, bma_scales=_BMA_SCALES
    )
    explicit_result = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        adaptive_gr_scales=False,
    )
    adaptive_result = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        adaptive_gr_scales=True,
    )

    for scale in _BMA_SCALES:
        np.testing.assert_array_equal(
            default_result.tvt_by_scale[scale], explicit_result.tvt_by_scale[scale]
        )
        np.testing.assert_array_equal(
            default_result.std_by_scale[scale], explicit_result.std_by_scale[scale]
        )
        assert not np.array_equal(
            default_result.tvt_by_scale[scale], adaptive_result.tvt_by_scale[scale]
        )
        assert np.all(np.isfinite(adaptive_result.tvt_by_scale[scale]))
    np.testing.assert_array_equal(default_result.log_lik, explicit_result.log_lik)

    # track_pf shares the same mechanism; its default must also be bit-identical.
    pf_default = track_pf(h, tw, seed=0)
    pf_explicit = track_pf(h, tw, seed=0, adaptive_gr_scales=False)
    pf_adaptive = track_pf(h, tw, seed=0, adaptive_gr_scales=True)
    np.testing.assert_array_equal(pf_default.tvt, pf_explicit.tvt)
    np.testing.assert_array_equal(pf_default.std, pf_explicit.std)
    assert not np.array_equal(pf_default.tvt, pf_adaptive.tvt)


# --------------------------------------------------------------------------- #
# PF process-model constant overrides (R6-b(5): coarse constant sweep support)
# --------------------------------------------------------------------------- #


def test_track_pf_bma_process_const_defaults_match_bit_for_bit_and_overrides_differ() -> None:
    """The four R6-b(5) process-model overrides (pos_process_noise,
    rate_noise_std, rate_momentum, resample_roughen_pos) default to the
    module constants; passing them explicitly at those values must be
    numerically identical to omitting them, and changing any one of them
    must actually change the output (each parameter is really wired
    through)."""
    h, tw, _ = _make_synthetic_well(seed=18)

    default_result = track_pf_bma(
        h, tw, n_particles=_BMA_N_PARTICLES, n_seeds=_BMA_N_SEEDS, bma_scales=_BMA_SCALES
    )
    explicit_result = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        pos_process_noise=0.005,
        rate_noise_std=0.002,
        rate_momentum=0.998,
        resample_roughen_pos=0.1,
    )

    for scale in _BMA_SCALES:
        np.testing.assert_array_equal(
            default_result.tvt_by_scale[scale], explicit_result.tvt_by_scale[scale]
        )
        np.testing.assert_array_equal(
            default_result.std_by_scale[scale], explicit_result.std_by_scale[scale]
        )
    np.testing.assert_array_equal(default_result.log_lik, explicit_result.log_lik)

    for override in (
        {"pos_process_noise": 0.02},
        {"rate_noise_std": 0.008},
        {"rate_momentum": 0.995},
    ):
        changed = track_pf_bma(
            h,
            tw,
            n_particles=_BMA_N_PARTICLES,
            n_seeds=_BMA_N_SEEDS,
            bma_scales=_BMA_SCALES,
            **override,
        )
        assert not np.array_equal(
            default_result.tvt_by_scale[_BMA_SCALES[0]],
            changed.tvt_by_scale[_BMA_SCALES[0]],
        ), f"override {override} did not change the output"
        for scale in _BMA_SCALES:
            assert np.all(np.isfinite(changed.tvt_by_scale[scale]))

    # resample_roughen_pos only acts when ESS-triggered resampling fires, which
    # this synthetic well's near-flat GR baseline rarely does at the default
    # resample_ess=0.5 -- force per-row resampling (resample_ess=1.0) in both
    # runs so the roughening path is exercised and the wiring is observable.
    roughen_base = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        resample_ess=1.0,
    )
    roughen_changed = track_pf_bma(
        h,
        tw,
        n_particles=_BMA_N_PARTICLES,
        n_seeds=_BMA_N_SEEDS,
        bma_scales=_BMA_SCALES,
        resample_ess=1.0,
        resample_roughen_pos=0.3,
    )
    assert not np.array_equal(
        roughen_base.tvt_by_scale[_BMA_SCALES[0]],
        roughen_changed.tvt_by_scale[_BMA_SCALES[0]],
    ), "resample_roughen_pos override did not change the output under forced resampling"
    for scale in _BMA_SCALES:
        assert np.all(np.isfinite(roughen_changed.tvt_by_scale[scale]))
