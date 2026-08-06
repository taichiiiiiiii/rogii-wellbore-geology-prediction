"""Tracker-derived features for the stack-integration GBDT (issue: stack GBDT).

``build_tracker_features(anchor, pf_tvt, pf_std, beam_tvt, beam_margin)`` turns
the four per-row arrays cached by ``scripts/build_tracker_cache.py`` (one
``.npz`` per well, see that script's docstring for the on-disk layout) into a
small feature block that a downstream GBDT (``scripts/run_stack_cv.py``) can
concatenate onto the Layer2 within-well features from ``rogii.features``.

Leak boundary: every input here is already leak-safe by construction --
``pf_tvt``/``beam_tvt`` come from the unsupervised trackers (which only ever
read the known-zone prefix + typewell GR, see ``registration/particle.py`` and
``registration/beam.py``), and ``anchor`` is ``data.last_known_tvt(h)``. This
module does no I/O and reads no ground-truth column; it is pure arithmetic
over already-computed arrays.

Semantics recap (see the tracker modules' docstrings for the full story):
  - ``pf_std``: particle-cloud uncertainty. Higher = PF trusts its own
    estimate less. ``inf`` on the PF's exception-fallback path.
  - ``beam_margin``: best-vs-second-best DP cost gap. Higher = the DP's cost
    surface picks out a state more decisively (not asserted to correlate with
    per-row error -- see ``BeamResult.margin`` docstring).

Stack v2 (issue: stack v2) adds two more pure feature blocks, still with no
I/O and no ground-truth reads, used by ``scripts/run_stack_v2_cv.py``:

  - :func:`build_well_aggregate_features` -- per-well scalar summaries of the
    same tracker cache arrays (final-row drift, well-wide means/maxes, log
    quality), broadcast to every eval-zone row of that well. These
    complement the *per-row* ``build_tracker_features`` block with a
    well-wide view a GBDT cannot otherwise reconstruct row-by-row (e.g. "how
    much did the PF drift over the *whole* well", not just "at this row").
  - :func:`build_spatial_features` -- turns the per-well spatial-prior cache
    (``data/processed/spatial_cache/*.npz``, built by
    ``scripts/build_spatial_cache.py`` from ``rogii.spatial``) into a
    per-row feature block, gated by the same ``prefix_rmse <= 2.0`` quality
    threshold as that script's regression-guard predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature-column order (fixed, for reproducible importances across runs).
FEATURE_COLUMNS: tuple[str, ...] = (
    "pf_d",
    "pf_std",
    "beam_d",
    "beam_margin",
    "pf_beam_abs_diff",
    "pf_d_damp",
    "pf_std_is_inf",
    "pf_beam_sign_agree",
)


def build_tracker_features(
    anchor: float,
    pf_tvt: np.ndarray,
    pf_std: np.ndarray,
    beam_tvt: np.ndarray,
    beam_margin: np.ndarray,
) -> pd.DataFrame:
    """Build the tracker feature block for one well's eval-zone rows.

    All four arrays must be aligned to ``data.eval_mask(h)`` order (same order
    and length as the tracker cache's per-well ``.npz`` arrays); ``anchor`` is
    the scalar ``data.last_known_tvt(h)`` for that well. Returns a ``float32``
    DataFrame with one row per eval-zone row and columns ``FEATURE_COLUMNS``.

    Never raises on degenerate input: a non-finite ``anchor`` is treated as
    ``0.0``, and empty arrays yield a zero-row DataFrame with the same
    columns (no length assumption beyond "all four arrays agree").
    """
    pf_tvt = np.asarray(pf_tvt, dtype=np.float64)
    pf_std = np.asarray(pf_std, dtype=np.float64)
    beam_tvt = np.asarray(beam_tvt, dtype=np.float64)
    beam_margin = np.asarray(beam_margin, dtype=np.float64)

    lengths = {pf_tvt.size, pf_std.size, beam_tvt.size, beam_margin.size}
    if len(lengths) > 1:
        raise ValueError(
            f"tracker arrays must share one length, got pf_tvt={pf_tvt.size} "
            f"pf_std={pf_std.size} beam_tvt={beam_tvt.size} beam_margin={beam_margin.size}"
        )
    n = pf_tvt.size

    anchor_f = float(anchor) if np.isfinite(anchor) else 0.0

    if n == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in FEATURE_COLUMNS})

    pf_d = pf_tvt - anchor_f
    beam_d = beam_tvt - anchor_f

    pf_std_is_inf = (~np.isfinite(pf_std)).astype(np.float64)
    # finite / inf == 0.0 in IEEE-754 numpy arithmetic -- no NaN produced, and
    # it is the intended reading: "no PF confidence -> don't trust pf_d at all".
    pf_d_damp = pf_d / (1.0 + pf_std)

    pf_beam_abs_diff = np.abs(pf_d - beam_d)

    # Sign agreement: +1 both-positive/both-negative, 0 disagreement, and a
    # tie (either side exactly 0) counts as "agree" (no evidence of conflict).
    pf_sign = np.sign(pf_d)
    beam_sign = np.sign(beam_d)
    pf_beam_sign_agree = ((pf_sign == beam_sign) | (pf_sign == 0) | (beam_sign == 0)).astype(
        np.float64
    )

    out = pd.DataFrame(
        {
            "pf_d": pf_d,
            "pf_std": pf_std,
            "beam_d": beam_d,
            "beam_margin": beam_margin,
            "pf_beam_abs_diff": pf_beam_abs_diff,
            "pf_d_damp": pf_d_damp,
            "pf_std_is_inf": pf_std_is_inf,
            "pf_beam_sign_agree": pf_beam_sign_agree,
        }
    )
    out = out[list(FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Stack v2: well-aggregate tracker features (issue: stack v2)
# --------------------------------------------------------------------------- #

WELL_FEATURE_COLUMNS: tuple[str, ...] = (
    "pf_d_final",
    "pf_d_mean",
    "pf_std_well_mean",
    "pf_std_well_max",
    "beam_margin_well_mean",
    "pf_beam_abs_diff_well_mean",
    "eval_len",
    "gr_nan_frac",
)


def build_well_aggregate_features(
    anchor: float,
    pf_tvt: np.ndarray,
    pf_std: np.ndarray,
    beam_tvt: np.ndarray,
    beam_margin: np.ndarray,
    gr_nan_frac: float,
) -> pd.DataFrame:
    """Well-level scalar tracker/quality summaries, broadcast to every eval row.

    A GBDT trained on :func:`build_tracker_features` alone only ever sees a
    *single row's* ``pf_d``/``beam_margin`` -- it cannot ask "how far did the
    PF drift over this well's *entire* evaluation zone" or "how noisy was
    this well's GR log overall". These eight scalars answer exactly that,
    computed once per well and repeated across all of that well's rows (a
    GBDT can split on them the same way it splits on any other feature).

    ``pf_tvt``/``pf_std``/``beam_tvt``/``beam_margin`` are the same
    ``eval_mask``-aligned arrays :func:`build_tracker_features` takes;
    ``anchor`` is ``data.last_known_tvt(h)``; ``gr_nan_frac`` is the caller-
    computed fraction of NaN in the well's ``GR`` column (this module does no
    I/O, so it cannot read ``h`` itself).

    Never raises on degenerate input: a non-finite ``anchor`` falls back to
    ``0.0``, a non-finite ``gr_nan_frac`` falls back to ``0.0`` (unknown
    quality treated as "no evidence of missingness"), all-non-finite tracker
    arrays fall back to ``0.0`` per statistic, and empty arrays yield a
    zero-row DataFrame with the same columns.
    """
    pf_tvt = np.asarray(pf_tvt, dtype=np.float64)
    pf_std = np.asarray(pf_std, dtype=np.float64)
    beam_tvt = np.asarray(beam_tvt, dtype=np.float64)
    beam_margin = np.asarray(beam_margin, dtype=np.float64)

    lengths = {pf_tvt.size, pf_std.size, beam_tvt.size, beam_margin.size}
    if len(lengths) > 1:
        raise ValueError(
            f"tracker arrays must share one length, got pf_tvt={pf_tvt.size} "
            f"pf_std={pf_std.size} beam_tvt={beam_tvt.size} beam_margin={beam_margin.size}"
        )
    n = pf_tvt.size

    if n == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in WELL_FEATURE_COLUMNS})

    anchor_f = float(anchor) if np.isfinite(anchor) else 0.0
    gr_nan_frac_f = float(gr_nan_frac) if np.isfinite(gr_nan_frac) else 0.0

    pf_d = pf_tvt - anchor_f
    beam_d = beam_tvt - anchor_f
    abs_diff = np.abs(pf_d - beam_d)

    def _last_finite_or(values: np.ndarray, default: float) -> float:
        finite = values[np.isfinite(values)]
        return float(finite[-1]) if finite.size else default

    def _mean_finite_or(values: np.ndarray, default: float) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else default

    def _max_finite_or(values: np.ndarray, default: float) -> float:
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else default

    pf_d_final = _last_finite_or(pf_d, 0.0)
    pf_d_mean = _mean_finite_or(pf_d, 0.0)
    pf_std_well_mean = _mean_finite_or(pf_std, 0.0)
    pf_std_well_max = _max_finite_or(pf_std, 0.0)
    beam_margin_well_mean = _mean_finite_or(beam_margin, 0.0)
    abs_diff_well_mean = _mean_finite_or(abs_diff, 0.0)

    out = pd.DataFrame(
        {
            "pf_d_final": np.full(n, pf_d_final, dtype=np.float64),
            "pf_d_mean": np.full(n, pf_d_mean, dtype=np.float64),
            "pf_std_well_mean": np.full(n, pf_std_well_mean, dtype=np.float64),
            "pf_std_well_max": np.full(n, pf_std_well_max, dtype=np.float64),
            "beam_margin_well_mean": np.full(n, beam_margin_well_mean, dtype=np.float64),
            "pf_beam_abs_diff_well_mean": np.full(n, abs_diff_well_mean, dtype=np.float64),
            "eval_len": np.full(n, float(n), dtype=np.float64),
            "gr_nan_frac": np.full(n, gr_nan_frac_f, dtype=np.float64),
        }
    )
    out = out[list(WELL_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Stack v2: spatial-prior features (issue: stack v2)
# --------------------------------------------------------------------------- #

SPATIAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "spatial_d",
    "spatial_prefix_rmse",
    "spatial_nn_dist_median",
    "spatial_gated_d",
)

# Same quality gate as scripts/build_spatial_cache.py's regression-guard
# predictor (GATE_THETA): above this prefix_rmse, the spatial prior's own
# known-zone fit is unreliable, so its row-wise delta is zeroed out rather
# than trusted.
SPATIAL_GATE_THETA = 2.0


def build_spatial_features(
    anchor: float,
    spatial_tvt: np.ndarray,
    prefix_rmse: float,
    nn_dist_median: float,
) -> pd.DataFrame:
    """Per-row spatial-prior feature block from one well's spatial cache.

    ``spatial_tvt`` is the ``eval_mask``-aligned array cached by
    ``scripts/build_spatial_cache.py`` (``rogii.spatial.predict_spatial``,
    leave-self-out KNN over other wells' formation-depth surfaces);
    ``prefix_rmse``/``nn_dist_median`` are that same cache's per-well scalar
    quality diagnostics; ``anchor`` is ``data.last_known_tvt(h)``.

    ``spatial_gated_d`` is ``spatial_d`` when ``prefix_rmse <=
    SPATIAL_GATE_THETA`` (the spatial prior's known-zone fit is trustworthy)
    and ``0.0`` otherwise -- the same gate ``build_spatial_cache.py`` uses for
    its own regression-guard predictor, exposed here as a feature rather than
    baked into a single blended prediction so the GBDT can still use the raw
    ``spatial_d`` on the ungated rows.

    Never raises: a non-finite ``anchor`` falls back to ``0.0``, a
    non-finite ``prefix_rmse`` is treated as "gate closed" (worst case), a
    non-finite ``nn_dist_median`` is passed through as ``NaN`` (LightGBM
    handles NaN natively), and an empty ``spatial_tvt`` yields a zero-row
    DataFrame with the same columns.
    """
    spatial_tvt = np.asarray(spatial_tvt, dtype=np.float64)
    n = spatial_tvt.size

    if n == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in SPATIAL_FEATURE_COLUMNS})

    anchor_f = float(anchor) if np.isfinite(anchor) else 0.0
    prefix_rmse_f = float(prefix_rmse) if np.isfinite(prefix_rmse) else float("inf")
    nn_dist_median_f = float(nn_dist_median) if np.isfinite(nn_dist_median) else float("nan")

    spatial_d = spatial_tvt - anchor_f
    gate_open = prefix_rmse_f <= SPATIAL_GATE_THETA
    spatial_gated_d = spatial_d if gate_open else np.zeros(n, dtype=np.float64)

    out = pd.DataFrame(
        {
            "spatial_d": spatial_d,
            "spatial_prefix_rmse": np.full(n, prefix_rmse_f, dtype=np.float64),
            "spatial_nn_dist_median": np.full(n, nn_dist_median_f, dtype=np.float64),
            "spatial_gated_d": spatial_gated_d,
        }
    )
    out = out[list(SPATIAL_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)
