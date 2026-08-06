"""Self-contained Kaggle Notebook submission: PF blend w0.7.

Paste this entire script into a single Kaggle Notebook cell for the ROGII
Wellbore Geology Prediction **code competition** (internet disabled, <=9h). It
has NO project imports (no ``import rogii``) -- only numpy/pandas/stdlib -- so
it is fully self-contained and runs unmodified both:

- on Kaggle, reading the mounted competition dataset under ``/kaggle/input``
  (see :func:`find_data_root` for the exact discovery strategy), and
- locally against ``data/raw/`` for validation::

    uv run python notebooks/submission/kernel_pf/pf_blend_submission.py

Predictor
---------
``anchor + 0.7 * (PF_mean - anchor)`` -- the current local best (pooled RMSE
**11.0621 ft** on all 773 train wells; see
``analysis/experiment_ledger.md``, 2026-07-06 "Particle Filter" row), where:

- ``anchor`` is ``carry_last``: the last known (non-NaN) ``TVT_input`` value
  extended flat across the evaluation zone
  (:func:`predict_carry_last`, ported from ``src/rogii/baseline.py``).
- ``PF_mean`` is the mean of 8 independent particle-filter runs (512
  particles each) over a ``TVT + Z`` state-space with a slowly-varying
  ``rate`` (``d(TVT + Z)/dMD``), reweighted against a typewell GR lookup
  (:func:`track_pf_multi` / :func:`track_pf`, ported verbatim from
  ``src/rogii/registration/particle.py``, itself adapted from the public
  Kaggle notebook ``lightningv08/lb-7-776-rogii-ridge-sp`` -- see that
  module's docstring for the full attribution).
- The typewell GR lookup and the affine GR calibration
  (:func:`tw_gr_lookup`, :func:`fit_affine_gr`, :func:`apply_affine`) are
  ported verbatim from ``src/rogii/typewell.py``.

Contract
--------
- The set of required ``(well, row_index)`` pairs is derived from
  ``sample_submission.csv``'s ``id`` column (``{well}_{row_index}``), NOT by
  listing ``test/`` directly. Locally ``test/`` holds only 3 example wells;
  the real Kaggle rerun substitutes a ~200-well ``test/`` set. Driving
  everything off ``sample_submission.csv`` keeps this script correct in both
  cases.
- Eval zone = the contiguous tail of a well's horizontal-well rows where
  ``TVT_input`` is NaN (ground truth is hidden there in ``test/``).
- Per-well prediction **never raises**: :func:`track_pf` /
  :func:`track_pf_multi` already carry their own internal exception-safe
  fallback (flat anchor, ``std=inf``); :func:`predict_well_blend` wraps the
  blend itself in a second try/except so any failure (malformed typewell,
  shape mismatch, non-finite output, ...) for a single well degrades to
  ``carry_last`` for that well only, never aborting the whole submission.
- Writes ``submission.csv`` with columns ``id,tvt`` and validates it against
  ``sample_submission.csv`` before writing (exits non-zero on failure).
- Prints progress every ``PROGRESS_EVERY`` wells and running elapsed time, to
  monitor the <=9h Kaggle CPU budget while the ~200-well rerun is in flight.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_INPUT = Path("/kaggle/input/rogii-wellbore-geology-prediction")

# PF tuning: identical to the calibrated defaults in
# ``src/rogii/registration/particle.py`` (8 seeds x 512 particles, scales
# tuned on a 129-well stratified train subset -- see that module's docstring
# and ``analysis/experiment_ledger.md``, 2026-07-06 "Particle Filter" row).
N_PARTICLES = 512
N_SEEDS = 8
SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)
BLEND_WEIGHT = 0.7

PROGRESS_EVERY = 10


# --------------------------------------------------------------------------- #
# Data root / I-O discovery
# --------------------------------------------------------------------------- #


def find_data_root() -> Path:
    """Locate the directory containing ``train/``, ``test/``, ``sample_submission.csv``.

    On Kaggle the competition dataset is not reliably mounted at
    ``/kaggle/input/{competition-slug}`` -- observed in practice (2026-07-06
    run of ``notebooks/submission/probe_carry_last.py``) to 404 there, while
    proven public kernels for this competition locate it via
    ``Path('/kaggle/input').rglob('sample_submission.csv')`` instead. So: try
    the conventional path first, then fall back to an ``/kaggle/input``-wide
    rglob for ``sample_submission.csv`` before giving up.
    """
    candidates = [KAGGLE_INPUT]
    try:
        here = Path(__file__).resolve()
        # Locally this file sits at notebooks/submission/kernel_pf/<name>.py, so
        # the repo root is parents[3]. On Kaggle the script runs from the shallow
        # /kaggle/src/script.py where parents[3] does not exist (raises
        # IndexError, observed in the v1 kernel run) -- the /kaggle/input
        # candidates above/below cover that case instead.
        candidates.append(here.parents[3] / "data" / "raw")
    except (NameError, IndexError):
        pass  # __file__ undefined inside a Jupyter kernel; parents[3] absent on Kaggle
    candidates.append(Path.cwd() / "data" / "raw")

    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
        if root.is_dir():
            for sub in root.glob("*/train"):  # zip sometimes extracts a nested folder
                return sub.parent

    if Path("/kaggle/input").is_dir():
        for hit in sorted(Path("/kaggle/input").rglob("sample_submission.csv")):
            root = hit.parent
            if (root / "train").is_dir() and (root / "test").is_dir():
                return root

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate train/ and test/ under any of: {tried}, "
        "nor via /kaggle/input rglob for sample_submission.csv"
    )


def output_path(root: Path) -> Path:
    """Kaggle expects ``submission.csv`` in the working directory; keep local runs tidy."""
    if str(root).startswith("/kaggle"):
        return Path("submission.csv")
    try:
        repo_root = Path(__file__).resolve().parents[3]
    except (NameError, IndexError):
        repo_root = Path.cwd()
    return repo_root / "outputs" / "submission_pf_blend.csv"


# --------------------------------------------------------------------------- #
# Ported from src/rogii/data.py (eval-zone / anchor helpers only)
# --------------------------------------------------------------------------- #


def eval_mask(h: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the evaluation zone (rows where ``TVT_input`` is NaN)."""
    return h["TVT_input"].isna().to_numpy()


def last_known_tvt(h: pd.DataFrame) -> float:
    """The last non-NaN ``TVT_input`` value (the anchor just before the eval zone)."""
    known = h["TVT_input"].to_numpy()
    valid = known[~np.isnan(known)]
    if valid.size == 0:
        raise ValueError("well has no known TVT_input anchor")
    return float(valid[-1])


# --------------------------------------------------------------------------- #
# Ported from src/rogii/baseline.py
# --------------------------------------------------------------------------- #


def predict_carry_last(h: pd.DataFrame) -> np.ndarray:
    """Carry the last known TVT (``TVT_input`` anchor) flat through the eval zone."""
    n = int(eval_mask(h).sum())
    return np.full(n, last_known_tvt(h), dtype=float)


# --------------------------------------------------------------------------- #
# Ported from src/rogii/typewell.py
# --------------------------------------------------------------------------- #

# Below this many known-zone rows, an affine fit is considered unreliable.
_MIN_VALID_ROWS = 50

# If the typewell GR sampled at the known TVT_input values is essentially
# constant, a least-squares fit is degenerate (undefined gain); fall back.
_MIN_GR_STD = 1e-6


def tw_gr_lookup(tw: pd.DataFrame):
    """Build a ``TVT -> GR`` interpolator from a typewell DataFrame.

    Rows with a missing ``GR`` are dropped and the remainder is sorted by
    ``TVT`` before interpolation. The returned callable uses ``np.interp``,
    so queries outside the typewell's TVT range are clipped to the nearest
    endpoint value rather than extrapolated.
    """
    valid = tw.dropna(subset=["GR"]).sort_values("TVT")
    tvt = valid["TVT"].to_numpy(dtype=float)
    gr = valid["GR"].to_numpy(dtype=float)

    def lookup(query_tvt: np.ndarray) -> np.ndarray:
        return np.interp(query_tvt, tvt, gr)

    return lookup


def fit_affine_gr(h: pd.DataFrame, tw: pd.DataFrame) -> tuple[float, float]:
    """Fit ``h.GR ~= gain * twGR(TVT_input) + offset`` over the known zone.

    Only rows where ``TVT_input`` and ``GR`` are both known (i.e. outside the
    eval zone) are used, and only ``TVT_input`` is read from ``h`` -- never
    ``TVT``. Falls back to the identity transform ``(1.0, 0.0)`` when there
    are too few known rows or the typewell GR sampled at those depths is
    degenerate (near-constant).
    """
    known = h["TVT_input"].notna() & h["GR"].notna()
    if int(known.sum()) < _MIN_VALID_ROWS:
        return 1.0, 0.0

    tw_valid = tw.dropna(subset=["GR"])
    if tw_valid.empty:
        return 1.0, 0.0

    tvt_input = h.loc[known, "TVT_input"].to_numpy(dtype=float)
    gr_h = h.loc[known, "GR"].to_numpy(dtype=float)

    lookup = tw_gr_lookup(tw)
    gr_tw = lookup(tvt_input)

    if np.std(gr_tw) < _MIN_GR_STD:
        return 1.0, 0.0

    gain, offset = np.polyfit(gr_tw, gr_h, 1)
    return float(gain), float(offset)


def apply_affine(gr: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """Apply the ``(gain, offset)`` affine transform to a GR array."""
    return gain * np.asarray(gr, dtype=float) + offset


# --------------------------------------------------------------------------- #
# Ported from src/rogii/registration/particle.py
# --------------------------------------------------------------------------- #

# Rate (d(TVT+Z)/dMD) calibration: fit over the last _RATE_CAL_ROWS known-zone
# rows via linear regression against MD.
_RATE_CAL_ROWS = 200

# Floor/fallback for the rate's initial spread (std).
_MIN_RATE_STD = 1e-4
_DEFAULT_RATE_STD = 0.02

# Particle process model constants.
_INIT_POS_STD = 0.3
_RATE_MOMENTUM = 0.998
_RATE_NOISE_STD = 0.002
_POS_PROCESS_NOISE = 0.005
_RESAMPLE_ROUGHEN_POS = 0.1
_RESAMPLE_ROUGHEN_RATE = 0.001

# GR likelihood: clip the squared, scaled residual before `exp`.
_MAX_SQUARED_RESIDUAL = 600.0
_MIN_SCALE = 1e-6


@dataclass
class PFResult:
    """Per-row TVT estimate and particle-cloud uncertainty for the eval zone."""

    tvt: np.ndarray
    std: np.ndarray
    n_updates: int


def _flat_fallback(n: int, anchor: float) -> PFResult:
    return PFResult(
        tvt=np.full(n, anchor, dtype=float),
        std=np.full(n, np.inf, dtype=float),
        n_updates=0,
    )


def _safe_n_and_anchor(h: pd.DataFrame) -> tuple[int, float]:
    try:
        n = int(eval_mask(h).sum())
    except Exception:
        n = 0
    try:
        anchor = last_known_tvt(h)
    except Exception:
        anchor = 0.0
    return n, anchor


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized systematic resampling: draw ``N`` indices proportional to ``weights``."""
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating-point round-off leaving < 1.0
    return np.searchsorted(cumulative, positions)


def _assign_particle_scales(n_particles: int, scales: tuple[float, ...]) -> np.ndarray:
    """Round-robin assignment of a fixed GR-likelihood scale to each particle."""
    clipped = np.maximum(np.asarray(scales, dtype=float), _MIN_SCALE)
    return clipped[np.arange(n_particles) % clipped.size]


def _calibrate_rate(md: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    """Fit the local ``rate = d(pos)/dMD`` and its variability from a known-zone tail."""
    if md.size < 2 or np.ptp(md) <= 0:
        return 0.0, _DEFAULT_RATE_STD

    slope, _ = np.polyfit(md, pos, 1)

    d_md = np.diff(md)
    valid = d_md > 0
    if valid.sum() >= 2:
        local_rates = np.diff(pos)[valid] / d_md[valid]
        rate_std = float(np.std(local_rates))
    else:
        rate_std = _DEFAULT_RATE_STD

    return float(slope), max(rate_std, _MIN_RATE_STD)


def _track_pf_impl(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int,
    scales: tuple[float, ...],
    seed: int,
    resample_ess: float,
) -> PFResult:
    mask = eval_mask(h)
    n_eval = int(mask.sum())
    if n_eval == 0:
        return PFResult(tvt=np.zeros(0, dtype=float), std=np.zeros(0, dtype=float), n_updates=0)

    eval_idx = np.flatnonzero(mask)
    anchor = last_known_tvt(h)

    known_notna = h["TVT_input"].notna().to_numpy()
    known_idx = np.flatnonzero(known_notna)
    if known_idx.size == 0:
        raise ValueError("well has no known TVT_input anchor")
    last_known_row = int(known_idx[-1])

    md = h["MD"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    raw_gr = h["GR"].to_numpy(dtype=float)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    z_last = float(z[last_known_row])
    pos_anchor = float(anchor) + z_last

    tail_n = min(_RATE_CAL_ROWS, known_idx.size)
    tail_idx = known_idx[-tail_n:]
    tail_md = md[tail_idx]
    tail_pos = tvt_input[tail_idx] + z[tail_idx]
    rate_init, rate_init_std = _calibrate_rate(tail_md, tail_pos)

    lookup = tw_gr_lookup(tw)
    gain, offset = fit_affine_gr(h, tw)

    tw_valid = tw.dropna(subset=["TVT", "GR"])
    if tw_valid.empty:
        raise ValueError("typewell has no usable TVT/GR rows")
    tw_min = float(tw_valid["TVT"].min())
    tw_max = float(tw_valid["TVT"].max())
    if not (np.isfinite(tw_min) and np.isfinite(tw_max)) or tw_max <= tw_min:
        raise ValueError("typewell TVT range is degenerate")

    rng = np.random.default_rng(seed)
    n = int(n_particles)

    pos = np.full(n, pos_anchor) + _INIT_POS_STD * rng.standard_normal(n)
    rate = np.full(n, rate_init) + rate_init_std * rng.standard_normal(n)
    particle_scale = _assign_particle_scales(n, scales)
    weights = np.full(n, 1.0 / n)

    tvt_out = np.empty(n_eval, dtype=float)
    std_out = np.empty(n_eval, dtype=float)
    n_updates = 0
    prev_md = md[last_known_row]

    for k in range(n_eval):
        row = int(eval_idx[k])
        dm = md[row] - prev_md
        if not np.isfinite(dm) or dm <= 0:
            dm = 1.0
        prev_md = md[row]

        rate = _RATE_MOMENTUM * rate + _RATE_NOISE_STD * rng.standard_normal(n)
        pos = pos + rate * dm + _POS_PROCESS_NOISE * rng.standard_normal(n)

        z_row = z[row]
        tvt_particles = np.clip(pos - z_row, tw_min, tw_max)
        pos = tvt_particles + z_row

        gr_row = raw_gr[row]
        if np.isfinite(gr_row):
            expected_gr = apply_affine(lookup(tvt_particles), gain, offset)
            resid = (gr_row - expected_gr) / particle_scale
            sq_resid = np.clip(resid * resid, 0.0, _MAX_SQUARED_RESIDUAL)
            likelihood = np.exp(-0.5 * sq_resid)
            weights = weights * likelihood
            weight_sum = weights.sum()
            if weight_sum > 0 and np.isfinite(weight_sum):
                weights = weights / weight_sum
            else:
                weights = np.full(n, 1.0 / n)
            n_updates += 1

        mean_tvt = float(np.dot(weights, tvt_particles))
        var_tvt = float(np.dot(weights, (tvt_particles - mean_tvt) ** 2))
        tvt_out[k] = mean_tvt
        std_out[k] = float(np.sqrt(max(var_tvt, 0.0)))

        ess = 1.0 / float(np.sum(weights * weights))
        if ess < resample_ess * n:
            idx = _systematic_resample(weights, rng)
            pos = pos[idx] + _RESAMPLE_ROUGHEN_POS * rng.standard_normal(n)
            rate = rate[idx] + _RESAMPLE_ROUGHEN_RATE * rng.standard_normal(n)
            particle_scale = particle_scale[idx]
            weights = np.full(n, 1.0 / n)

    return PFResult(tvt=tvt_out, std=std_out, n_updates=n_updates)


def track_pf(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = 512,
    scales: tuple[float, ...] = SCALES,
    seed: int = 0,
    resample_ess: float = 0.5,
) -> PFResult:
    """Sequential Monte Carlo (particle filter) GR registration tracker.

    Never raises: any failure (malformed input, missing columns, a
    degenerate/too-short typewell, no known anchor, etc.) falls back to a
    flat anchor prediction with ``std`` set to ``inf`` for every row.
    """
    try:
        return _track_pf_impl(h, tw, n_particles, scales, seed, resample_ess)
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _flat_fallback(n, anchor)


def track_pf_multi(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_seeds: int = 8,
    **kwargs: object,
) -> PFResult:
    """Average ``n_seeds`` independent :func:`track_pf` runs for variance reduction."""
    kwargs.pop("seed", None)  # track_pf_multi owns per-run seeding; ignore a stray override
    results = [track_pf(h, tw, seed=seed, **kwargs) for seed in range(max(int(n_seeds), 1))]

    tvts = np.stack([r.tvt for r in results], axis=0)
    stds = np.stack([r.std for r in results], axis=0)

    mean_tvt = tvts.mean(axis=0)
    with np.errstate(invalid="ignore"):
        combined_var = np.mean(stds**2, axis=0) + np.var(tvts, axis=0)
    combined_std = np.sqrt(combined_var)

    return PFResult(tvt=mean_tvt, std=combined_std, n_updates=results[0].n_updates)


# --------------------------------------------------------------------------- #
# Submission predictor: anchor + 0.7 * (PF_mean - anchor)
# --------------------------------------------------------------------------- #


def predict_well_blend(h: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    """PF blend w0.7 predictor for a single well; never raises.

    ``track_pf_multi``/``track_pf`` already carry their own internal
    exception-safe fallback, so this function's own try/except is a second,
    belt-and-suspenders safety net: any failure while computing the blend
    (shape mismatch, non-finite output, ...) degrades to ``carry_last`` for
    this well only. If even ``carry_last`` fails (no anchor at all), falls
    back to a constant-filled array via ``_safe_n_and_anchor`` so the whole
    submission can never abort on a single malformed well.
    """
    try:
        anchor_pred = predict_carry_last(h)
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return np.full(n, anchor, dtype=float)

    try:
        pf = track_pf_multi(h, tw, n_seeds=N_SEEDS, n_particles=N_PARTICLES, scales=SCALES)
        if pf.tvt.shape != anchor_pred.shape:
            raise ValueError(
                f"PF output shape {pf.tvt.shape} != anchor shape {anchor_pred.shape}"
            )
        blended = anchor_pred + BLEND_WEIGHT * (pf.tvt - anchor_pred)
        if not np.all(np.isfinite(blended)):
            raise ValueError("PF blend produced non-finite values")
        return blended
    except Exception:
        return anchor_pred


# --------------------------------------------------------------------------- #
# sample_submission-driven prediction / validation (pattern from
# notebooks/submission/probe_carry_last.py and scripts/make_submission.py)
# --------------------------------------------------------------------------- #


def parse_sample_submission(path: Path) -> pd.DataFrame:
    """Load ``sample_submission.csv`` and split ``id`` into ``well``/``row_index``."""
    sample = pd.read_csv(path)
    parts = sample["id"].str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2 or parts[1].isna().any():
        raise ValueError("sample_submission ids must be formatted as '{well}_{row_index}'")
    return sample.assign(well=parts[0], row_index=parts[1].astype(int)).reset_index(drop=True)


def build_predictions(
    sample: pd.DataFrame, root: Path, t_start: float
) -> tuple[pd.DataFrame, list[str], list[dict[str, object]]]:
    """Predict ``tvt`` for every id in ``sample``, grouped by well.

    Returns the submission DataFrame (same row order as ``sample``), a list
    of human-readable mismatch warnings, and a per-well summary (for the
    local validation printout: anchor vs. PF-blend prediction stats).
    """
    mismatch_warnings: list[str] = []
    well_summaries: list[dict[str, object]] = []
    tvt = np.full(len(sample), np.nan, dtype=float)

    wells = list(sample.groupby("well", sort=False))
    n_wells = len(wells)

    for i, (well, well_rows) in enumerate(wells):
        try:
            h = pd.read_csv(root / "test" / f"{well}__horizontal_well.csv")
            tw = pd.read_csv(root / "test" / f"{well}__typewell.csv")
            eval_idx = np.where(eval_mask(h))[0]
            preds = predict_well_blend(h, tw)
            anchor_pred = predict_carry_last(h)
        except Exception as exc:  # last-resort: never let a single well abort the run
            print(f"[WARN] {well}: prediction failed ({exc!r}), fallback to constant anchor")
            eval_idx = well_rows["row_index"].to_numpy()
            n = eval_idx.shape[0]
            preds = np.zeros(n, dtype=float)
            anchor_pred = preds
            for pos in well_rows.index.to_numpy():
                tvt[pos] = 0.0
            mismatch_warnings.append(f"{well}: total prediction failure, wrote 0.0 fallback")
            continue

        if preds.shape[0] != eval_idx.shape[0]:
            print(
                f"[WARN] {well}: predictor produced {preds.shape[0]} rows, "
                f"expected {eval_idx.shape[0]}; padding/truncating with carry_last"
            )
            fallback_val = last_known_tvt(h) if eval_idx.shape[0] else 0.0
            preds = np.full(eval_idx.shape[0], fallback_val, dtype=float)

        pred_by_row = dict(zip(eval_idx.tolist(), preds.tolist(), strict=True))
        fallback = last_known_tvt(h) if eval_idx.shape[0] else 0.0

        want_rows = well_rows["row_index"].to_numpy()
        missing = sorted(set(want_rows.tolist()) - set(pred_by_row))
        if missing:
            mismatch_warnings.append(
                f"{well}: {len(missing)} row_index(es) outside eval zone, e.g. {missing[:5]}"
            )

        for pos, row_idx in zip(well_rows.index.to_numpy(), want_rows, strict=True):
            tvt[pos] = pred_by_row.get(int(row_idx), fallback)

        well_preds = np.array([pred_by_row.get(int(r), fallback) for r in want_rows])
        well_summaries.append(
            {
                "well": well,
                "n_rows": int(want_rows.shape[0]),
                "anchor_mean": float(np.mean(anchor_pred)) if anchor_pred.size else float("nan"),
                "blend_mean": float(np.mean(well_preds)) if well_preds.size else float("nan"),
                "mean_abs_delta": (
                    float(np.mean(np.abs(well_preds - anchor_pred)))
                    if anchor_pred.size == well_preds.size and anchor_pred.size
                    else float("nan")
                ),
            }
        )

        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == n_wells:
            elapsed = time.time() - t_start
            per_well = elapsed / (i + 1)
            remaining = (n_wells - (i + 1)) * per_well
            print(
                f"[PROGRESS] {i + 1}/{n_wells} wells done | "
                f"elapsed={elapsed:.1f}s | avg={per_well:.3f}s/well | "
                f"est. remaining={remaining:.1f}s"
            )

    out = pd.DataFrame({"id": sample["id"].to_numpy(), "tvt": tvt})
    return out, mismatch_warnings, well_summaries


def validate(out: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    """Hard-fail validation checks. Returns error messages; empty means OK."""
    errors: list[str] = []

    if len(out) != len(sample):
        errors.append(f"row count mismatch: submission={len(out)} sample={len(sample)}")

    n_dup = int(out["id"].duplicated().sum())
    if n_dup:
        errors.append(f"{n_dup} duplicate id(s) in submission")

    out_ids, sample_ids = set(out["id"]), set(sample["id"])
    missing_ids = sample_ids - out_ids
    extra_ids = out_ids - sample_ids
    if missing_ids:
        errors.append(f"missing {len(missing_ids)} id(s), e.g. {sorted(missing_ids)[:5]}")
    if extra_ids:
        errors.append(f"{len(extra_ids)} unexpected id(s), e.g. {sorted(extra_ids)[:5]}")

    tvt = out["tvt"].to_numpy(dtype=float)
    n_nan = int(np.isnan(tvt).sum())
    n_inf = int(np.isinf(tvt).sum())
    if n_nan:
        errors.append(f"{n_nan} NaN value(s) in tvt")
    if n_inf:
        errors.append(f"{n_inf} inf value(s) in tvt")

    return errors


def main() -> None:
    t_start = time.time()
    root = find_data_root()
    print(f"[INFO] data root: {root}")
    print(
        f"[INFO] predictor: anchor + {BLEND_WEIGHT} * (PF_mean - anchor) | "
        f"n_particles={N_PARTICLES} n_seeds={N_SEEDS} scales={SCALES}"
    )

    sample = parse_sample_submission(root / "sample_submission.csv")
    n_wells_in = sample["well"].nunique()
    print(f"[INFO] sample_submission: {len(sample)} id(s) across {n_wells_in} well(s)")

    out, mismatch_warnings, well_summaries = build_predictions(sample, root, t_start)

    errors = validate(out, sample)
    if errors:
        print("[FAIL] submission validation failed:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    for w in mismatch_warnings:
        print(f"[WARN] {w}")

    out_path = output_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[["id", "tvt"]].to_csv(out_path, index=False)

    elapsed_total = time.time() - t_start
    n_wells = out["id"].str.rsplit("_", n=1).str[0].nunique()
    print(f"[PASS] wrote {out_path.resolve()} ({len(out)} rows, {n_wells} well(s))")
    print("[PASS] id set matches sample_submission exactly; no NaN/inf in tvt; row counts match")
    print(f"[INFO] total elapsed: {elapsed_total:.1f}s ({elapsed_total / 60:.2f} min)")

    print("\n[SUMMARY] per-well anchor vs. PF-blend prediction:")
    for s in well_summaries:
        print(
            f"  {s['well']}: n={s['n_rows']:d} "
            f"anchor_mean={s['anchor_mean']:.4f} blend_mean={s['blend_mean']:.4f} "
            f"mean_abs_delta={s['mean_abs_delta']:.4f}"
        )


if __name__ == "__main__":
    main()
