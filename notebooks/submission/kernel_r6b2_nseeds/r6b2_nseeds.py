"""Self-contained Kaggle Notebook **compute-only** kernel: candidate-path bank builder.

This is NOT a submission kernel (it writes no ``submission.csv``). It exists to
offload three CPU-heavy batch computations -- listed below -- to a Kaggle
kernel (4-core / 30GB RAM / <=12h) that repeatedly got OOM-killed on this
project's local dev box (3.8GB RAM). Paste this entire script into a single
Kaggle Notebook cell (script kernel), run it with internet disabled, and
recover its three ``.npz`` outputs afterwards via
``kaggle kernels output <user>/rogii-bank-builder -p <outdir>``.

It has NO project imports (no ``import rogii``) -- only numpy/pandas/stdlib --
so it is fully self-contained and runs unmodified both:

- on Kaggle, reading the mounted competition dataset under ``/kaggle/input``
  (see :func:`find_data_root` for the exact discovery strategy), writing to
  ``/kaggle/working/``, and
- locally against ``data/raw/`` for a smoke test::

    ROGII_BANK_LIMIT=8 uv run python notebooks/submission/kernel_bank_builder/bank_builder.py

  (``ROGII_BANK_LIMIT=<N>`` restricts every phase to the first ``N`` train
  wells, sorted deterministically, and redirects output from
  ``/kaggle/working`` to ``./outputs_smoke/`` so a local smoke run never
  touches ``/kaggle/working`` or the project's real ``outputs/`` bank files.
  **Do not run this locally without the env var set** -- a full 773-well run
  is exactly the OOM-killed workload this kernel exists to move off this box.)

Three phases, each independently save-checkpointed (see "Time budget" below)
and each reproducing an existing local script's bank-building logic --
ancestry kept 1:1 so downstream analysis code (the oracle/ranking scripts
that already read these bank files) needs zero changes:

  **Phase A** -- ``scripts/build_backtest_features.py``'s prefix-cut backtest
  feature matrix (48 columns x N wells): for each well, cut the known
  ``TVT_input`` prefix short at ``cut_fracs=(0.5, 0.7, 0.85)``, treat the
  cut-off tail as a pseudo eval zone with known ground truth, and score
  ``carry_last`` / ``track_pf`` (seed=0) / ``track_beam_multi`` against it.
  Writes ``backtest_features.npz``.

  **Phase B** -- ``scripts/run_pf_bma_cv.py``'s PF-BMA candidate-path bank:
  ``track_pf_bma`` (likelihood-weighted seed-BMA over independent particle
  filters) over every well at ``n_seeds=12`` (fixed here -- the source
  script's adaptive time-gate probe is not needed on Kaggle's much larger
  compute budget; 12 was itself a time-gate-selected value on this repo's dev
  box). Writes ``path_bank_pfbma.npz``.

  **Phase C** -- ``scripts/run_beam_grid_cv.py``'s 10-configuration beam-DP
  candidate-path bank (``CONFIG_GRID``: product/calibration/smoothing/
  transition-width axes). Writes ``path_bank_beam.npz``.

**Deliberately NOT ported**: the oracle/ranking analysis tails of
``run_pf_bma_cv.py`` (its "5-way oracle" phase) and ``run_beam_grid_cv.py``
(its "(4+K)-way oracle" and error-correlation-matrix sections), and each
source script's own adaptive time-gate probe. Those all read
``outputs/stack_v2_oof.npz`` (an OOF cache built by a separate, un-portable
CV pipeline that has no equivalent on hidden Kaggle test wells) and/or
``data/processed/spatial_cache/`` -- neither exists on Kaggle, and neither is
needed here: this kernel's only job is to *compute and persist the raw
per-well candidate paths*; the oracle/ranking analysis is re-run locally
afterwards against the recovered bank files, against the real (local) OOF
cache. ``build_backtest_features.py``'s well ordering also originally came
from that OOF cache's ``used_wells`` array (purely for a *consistent row
order* across scripts, not a leak) -- here it instead comes from
:func:`list_wells` directly (sorted train-well hashes), and every phase's
``.npz`` always carries its own well-name array so the local analysis side
can re-align by name rather than assuming a shared row order.

Leak boundary
-------------
Every predictor here only ever reads ``MD, X, Y, Z, GR, TVT_input`` from a
well's own horizontal-well log for *inference*. The ground-truth ``TVT``
column is read only for **scoring bookkeeping** (pooled-RMSE summaries
printed at the end of each phase, and Phase A's pseudo-eval-zone labels,
which are themselves cut from the *known*, already-non-secret prefix -- see
``truncate_known_prefix``) -- train-only, never fed to a predictor, and never
written to the output banks as a feature. This mirrors the leak-safety
analysis in ``src/rogii/registration/particle.py`` / ``beam.py`` /
``beam_grid.py`` and ``scripts/build_backtest_features.py``'s own docstrings.

Attribution
-----------
Ported/adapted from (see each source module's own docstring for the full
attribution trail):

- ``src/rogii/data.py`` (well listing / eval-zone / anchor helpers)
- ``src/rogii/baseline.py`` (``predict_carry_last``)
- ``src/rogii/typewell.py`` (GR lookup + affine calibration)
- ``src/rogii/registration/particle.py`` (``track_pf`` / ``track_pf_bma``;
  the particle-filter design is adapted from the public Kaggle notebook
  ``lightningv08/lb-7-776-rogii-ridge-sp``; PF-BMA reimplements the *method*
  of ``bernubritz/rogii-lb7295-public-rebuild``)
- ``src/rogii/registration/beam.py`` (``track_beam`` / ``track_beam_multi``;
  adapted from ``lightningv08/lb-7-776-rogii-ridge-sp`` and
  ``nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based``)
- ``src/rogii/registration/beam_grid.py`` (``BeamGridConfig`` /
  ``track_beam_grid``, the calibration/smoothing observation-axis extension)
- ``scripts/build_backtest_features.py`` (Phase A: prefix-cut backtest
  feature schema)
- ``scripts/run_pf_bma_cv.py`` (Phase B: PF-BMA bank-building config)
- ``scripts/run_beam_grid_cv.py`` (Phase C: ``CONFIG_GRID`` + bank-building)

Algorithms are adapted at the design level from the public notebooks cited
above. Those notebooks state no open-source license, so no upstream code is
reproduced here.

Time budget
-----------
Local dev-box measurements (this repo's ledger): PF-BMA ~4.2s/well
(n_seeds=12) => ~55min/773 wells; beam grid ~0.41s/well summed across the
10-config grid => ~53min/773 wells; backtest features ~0.66s/well => ~9min/
773 wells. Combined ~2h, comfortably inside Kaggle's <=12h budget even at
Kaggle's measured ~1.5x-slower-than-local execution
(``docs/playbooks/03_kaggle_ops.md``). A ``GLOBAL_TIME_BUDGET_S`` guard
(10.5h) is still enforced at every phase boundary *and* inside every phase's
per-well loop (:func:`budget_left`): if exceeded, the current phase stops
early (whatever it already accumulated is still saved) and every remaining
phase is skipped, so the kernel always exits normally with whatever partial
output it has -- Kaggle only ever commits kernel outputs from a *normal*
process exit, so a mid-run crash here would silently lose everything
computed so far.

v2 (2026-07-10, R6-b(3))
-------------------------
This revision's ``RUN_PHASES = ("B",)`` restricts a push to *only* rebuild
Phase B, at ``PFBMA_INIT_POS_STD = 2.0`` (v1 used 0.3, mirroring
``src/rogii/registration/particle.py``'s adopted ``init_pos_std`` default --
see commit f35249d and ``analysis/experiment_ledger.md``). Phase B's output
filename changed to ``path_bank_pfbma_v2.npz`` so this never overwrites v1's
``path_bank_pfbma.npz``. Phase A/C's code and defaults (``_INIT_POS_STD =
0.3`` for ``track_pf``/``track_beam_multi``) are untouched and simply skipped
via ``RUN_PHASES``; set it back to ``("A", "B", "C")`` for a full v1-parity
run.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #

KAGGLE_INPUT = Path("/kaggle/input/rogii-wellbore-geology-prediction")

GLOBAL_TIME_BUDGET_S = 10.5 * 3600.0
PROGRESS_EVERY = 50

# R6-b(3) (analysis/experiment_ledger.md, 2026-07-10): this run only rebuilds
# Phase B (PF-BMA bank at PFBMA_INIT_POS_STD=2.0, see that constant below).
# Phase A/C are unaffected by that change and are skipped to avoid
# re-running ~62min of unrelated compute; set back to ("A", "B", "C") for a
# full v1-parity run.
RUN_PHASES: tuple[str, ...] = ("B",)


def _bank_limit() -> int | None:
    """``ROGII_BANK_LIMIT=<N>`` env var: restrict every phase to the first N train wells."""
    raw = os.environ.get("ROGII_BANK_LIMIT")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def budget_left(t_program_start: float) -> float:
    """Seconds remaining in the global 10.5h budget (negative once exceeded)."""
    return GLOBAL_TIME_BUDGET_S - (time.perf_counter() - t_program_start)


# --------------------------------------------------------------------------- #
# Data root / output directory discovery (ported from
# notebooks/submission/kernel_stack_v2/stack_v2_submission.py's proven
# discovery strategy).
# --------------------------------------------------------------------------- #


def find_data_root() -> Path:
    """Locate the directory containing ``train/``, ``test/``, ``sample_submission.csv``."""
    candidates = [KAGGLE_INPUT]
    try:
        here = Path(__file__).resolve()
        # Locally this file sits at notebooks/submission/kernel_bank_builder/<name>.py,
        # so the repo root is parents[3]. On Kaggle the script runs from the
        # shallow /kaggle/src/script.py where parents[3] does not exist
        # (raises IndexError) -- the /kaggle/input candidates cover that case.
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


def _repo_root_safe() -> Path:
    try:
        return Path(__file__).resolve().parents[3]
    except (NameError, IndexError):
        return Path.cwd()


def output_dir() -> Path:
    """Where to write the three bank ``.npz`` files.

    ``ROGII_BANK_LIMIT`` set (smoke mode) -> ``./outputs_smoke/`` (never
    touches ``/kaggle/working`` or the project's real ``outputs/``). Kaggle
    (``/kaggle/working`` exists) -> that directory (Kaggle's kernel-output
    mechanism only persists files written there). Otherwise (a bare local
    full run, discouraged -- see module docstring) -> a scratch subdirectory
    under the repo's git-ignored ``outputs/``.
    """
    if _bank_limit() is not None:
        d = Path.cwd() / "outputs_smoke"
        d.mkdir(parents=True, exist_ok=True)
        return d
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    d = _repo_root_safe() / "outputs" / "bank_builder_local"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Ported from src/rogii/data.py
# --------------------------------------------------------------------------- #


def list_wells(split: str, root: Path) -> list[str]:
    """Sorted unique well hashes in ``train`` or ``test``."""
    d = root / split
    return sorted({p.name.split("__")[0] for p in d.glob("*__horizontal_well.csv")})


def load_horizontal(well: str, split: str, root: Path) -> pd.DataFrame:
    return pd.read_csv(root / split / f"{well}__horizontal_well.csv")


def load_typewell(well: str, split: str, root: Path) -> pd.DataFrame:
    return pd.read_csv(root / split / f"{well}__typewell.csv")


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

_MIN_VALID_ROWS = 50
_MIN_GR_STD = 1e-6


def tw_gr_lookup(tw: pd.DataFrame):
    """Build a ``TVT -> GR`` interpolator from a typewell DataFrame."""
    valid = tw.dropna(subset=["GR"]).sort_values("TVT")
    tvt = valid["TVT"].to_numpy(dtype=float)
    gr = valid["GR"].to_numpy(dtype=float)

    def lookup(query_tvt: np.ndarray) -> np.ndarray:
        return np.interp(query_tvt, tvt, gr)

    return lookup


def fit_affine_gr(h: pd.DataFrame, tw: pd.DataFrame) -> tuple[float, float]:
    """Fit ``h.GR ~= gain * twGR(TVT_input) + offset`` over the known zone."""
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
    return gain * np.asarray(gr, dtype=float) + offset


# --------------------------------------------------------------------------- #
# Ported from src/rogii/registration/particle.py (PF + PF-BMA trackers).
# Adapted from the public Kaggle notebook lightningv08/lb-7-776-rogii-ridge-sp
# (PF) and reimplements the method of bernubritz/rogii-lb7295-public-rebuild
# (PF-BMA seed-softmax). See that module's own docstring for full attribution
# and empirical tuning rationale.
# --------------------------------------------------------------------------- #

_RATE_CAL_ROWS = 200
_MIN_RATE_STD = 1e-4
_DEFAULT_RATE_STD = 0.02
_INIT_POS_STD = 0.3
_RATE_MOMENTUM = 0.998
_RATE_NOISE_STD = 0.002
_POS_PROCESS_NOISE = 0.005
_RESAMPLE_ROUGHEN_POS = 0.1
_RESAMPLE_ROUGHEN_RATE = 0.001
_MAX_SQUARED_RESIDUAL = 600.0
_MIN_SCALE = 1e-6

# GR-likelihood scale grid, empirically tuned (analysis/experiment_ledger.md):
# affine-calibrated GR residual std over the known zone is ~10-16 (median
# ~12.7) on real wells.
_DEFAULT_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)

# PF-BMA seed-softmax temperature grid (mirrors the reference's CFG.PF_SCALES).
_DEFAULT_BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
_DEFAULT_BMA_N_PARTICLES = 512
_DEFAULT_BMA_N_SEEDS = 8
_MIN_AVG_LIKELIHOOD = 1e-300


@dataclass
class PFResult:
    tvt: np.ndarray
    std: np.ndarray
    n_updates: int


def _pf_flat_fallback(n: int, anchor: float) -> PFResult:
    return PFResult(
        tvt=np.full(n, anchor, dtype=float), std=np.full(n, np.inf, dtype=float), n_updates=0
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
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def _assign_particle_scales(n_particles: int, scales: tuple[float, ...]) -> np.ndarray:
    clipped = np.maximum(np.asarray(scales, dtype=float), _MIN_SCALE)
    return clipped[np.arange(n_particles) % clipped.size]


def _calibrate_rate(md: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
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
    scales: tuple[float, ...] = _DEFAULT_SCALES,
    seed: int = 0,
    resample_ess: float = 0.5,
) -> PFResult:
    """Sequential Monte Carlo (particle filter) GR registration tracker.

    Never raises: any failure falls back to a flat anchor prediction with
    ``std`` set to ``inf`` for every row.
    """
    try:
        return _track_pf_impl(h, tw, n_particles, scales, seed, resample_ess)
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _pf_flat_fallback(n, anchor)


@dataclass
class PFBMAResult:
    tvt_by_scale: dict[float, np.ndarray]
    std_by_scale: dict[float, np.ndarray]
    log_lik: np.ndarray
    n_updates: int


def _flat_fallback_bma(
    n: int, anchor: float, scales: tuple[float, ...], n_seeds: int
) -> PFBMAResult:
    flat = np.full(n, anchor, dtype=float)
    inf_arr = np.full(n, np.inf, dtype=float)
    return PFBMAResult(
        tvt_by_scale={scale: flat.copy() for scale in scales},
        std_by_scale={scale: inf_arr.copy() for scale in scales},
        log_lik=np.full(max(int(n_seeds), 1), -np.inf, dtype=float),
        n_updates=0,
    )


@dataclass
class _BMAWellPrep:
    md: np.ndarray
    z: np.ndarray
    raw_gr: np.ndarray
    eval_idx: np.ndarray
    n_eval: int
    last_known_row: int
    pos_anchor: float
    rate_init: float
    rate_init_std: float
    lookup: object
    gain: float
    offset: float
    tw_min: float
    tw_max: float


def _prepare_bma_well(h: pd.DataFrame, tw: pd.DataFrame) -> _BMAWellPrep:
    mask = eval_mask(h)
    n_eval = int(mask.sum())
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

    return _BMAWellPrep(
        md=md,
        z=z,
        raw_gr=raw_gr,
        eval_idx=eval_idx,
        n_eval=n_eval,
        last_known_row=last_known_row,
        pos_anchor=pos_anchor,
        rate_init=rate_init,
        rate_init_std=rate_init_std,
        lookup=lookup,
        gain=gain,
        offset=offset,
        tw_min=tw_min,
        tw_max=tw_max,
    )


def _simulate_bma_seed(
    prep: _BMAWellPrep,
    seed: int,
    n_particles: int,
    gr_scales: tuple[float, ...],
    resample_ess: float,
    init_pos_std: float = _INIT_POS_STD,
) -> tuple[np.ndarray, float, int]:
    n_eval = prep.n_eval
    if n_eval == 0:
        return np.zeros(0, dtype=float), 0.0, 0

    rng = np.random.default_rng(seed)
    n = int(n_particles)

    pos = np.full(n, prep.pos_anchor) + init_pos_std * rng.standard_normal(n)
    rate = np.full(n, prep.rate_init) + prep.rate_init_std * rng.standard_normal(n)
    particle_scale = _assign_particle_scales(n, gr_scales)
    weights = np.full(n, 1.0 / n)

    tvt_out = np.empty(n_eval, dtype=float)
    log_lik = 0.0
    n_updates = 0
    prev_md = prep.md[prep.last_known_row]

    for k in range(n_eval):
        row = int(prep.eval_idx[k])
        dm = prep.md[row] - prev_md
        if not np.isfinite(dm) or dm <= 0:
            dm = 1.0
        prev_md = prep.md[row]

        rate = _RATE_MOMENTUM * rate + _RATE_NOISE_STD * rng.standard_normal(n)
        pos = pos + rate * dm + _POS_PROCESS_NOISE * rng.standard_normal(n)

        z_row = prep.z[row]
        tvt_particles = np.clip(pos - z_row, prep.tw_min, prep.tw_max)
        pos = tvt_particles + z_row

        gr_row = prep.raw_gr[row]
        if np.isfinite(gr_row):
            expected_gr = apply_affine(prep.lookup(tvt_particles), prep.gain, prep.offset)
            resid = (gr_row - expected_gr) / particle_scale
            sq_resid = np.clip(resid * resid, 0.0, _MAX_SQUARED_RESIDUAL)
            likelihood = np.exp(-0.5 * sq_resid)

            avg_lk = float(np.dot(weights, likelihood))  # weights sum to 1.0 going in
            log_lik += float(np.log(max(avg_lk, _MIN_AVG_LIKELIHOOD)))

            unnorm = weights * likelihood
            weight_sum = unnorm.sum()
            if weight_sum > 0 and np.isfinite(weight_sum):
                weights = unnorm / weight_sum
            else:
                weights = np.full(n, 1.0 / n)
            n_updates += 1

        tvt_out[k] = float(np.dot(weights, tvt_particles))

        ess = 1.0 / float(np.sum(weights * weights))
        if ess < resample_ess * n:
            idx = _systematic_resample(weights, rng)
            pos = pos[idx] + _RESAMPLE_ROUGHEN_POS * rng.standard_normal(n)
            rate = rate[idx] + _RESAMPLE_ROUGHEN_RATE * rng.standard_normal(n)
            particle_scale = particle_scale[idx]
            weights = np.full(n, 1.0 / n)

    return tvt_out, log_lik, n_updates


def _track_pf_bma_impl(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int,
    n_seeds: int,
    bma_scales: tuple[float, ...],
    gr_scales: tuple[float, ...],
    seed_base: int,
    resample_ess: float,
    init_pos_std: float = _INIT_POS_STD,
) -> PFBMAResult:
    prep = _prepare_bma_well(h, tw)
    n_seeds = max(int(n_seeds), 1)

    if prep.n_eval == 0:
        return PFBMAResult(
            tvt_by_scale={scale: np.zeros(0, dtype=float) for scale in bma_scales},
            std_by_scale={scale: np.zeros(0, dtype=float) for scale in bma_scales},
            log_lik=np.zeros(n_seeds, dtype=float),
            n_updates=0,
        )

    paths = np.empty((n_seeds, prep.n_eval), dtype=float)
    log_liks = np.empty(n_seeds, dtype=float)
    n_updates = 0
    for s in range(n_seeds):
        tvt_path, log_lik, n_upd = _simulate_bma_seed(
            prep, seed_base + s, n_particles, gr_scales, resample_ess, init_pos_std
        )
        paths[s] = tvt_path
        log_liks[s] = log_lik
        if s == 0:
            n_updates = n_upd

    ll_shifted = log_liks - log_liks.max()  # softmax numerical stability, cancels in the ratio

    tvt_by_scale: dict[float, np.ndarray] = {}
    std_by_scale: dict[float, np.ndarray] = {}
    for scale in bma_scales:
        w = np.exp(ll_shifted / float(scale))
        w_sum = w.sum()
        if w_sum > 0 and np.isfinite(w_sum):
            w = w / w_sum
        else:
            w = np.full(n_seeds, 1.0 / n_seeds)
        weighted_mean = w @ paths
        weighted_var = w @ ((paths - weighted_mean) ** 2)
        tvt_by_scale[scale] = weighted_mean
        std_by_scale[scale] = np.sqrt(np.maximum(weighted_var, 0.0))

    return PFBMAResult(
        tvt_by_scale=tvt_by_scale,
        std_by_scale=std_by_scale,
        log_lik=log_liks,
        n_updates=n_updates,
    )


def track_pf_bma(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = _DEFAULT_BMA_N_PARTICLES,
    n_seeds: int = _DEFAULT_BMA_N_SEEDS,
    bma_scales: tuple[float, ...] = _DEFAULT_BMA_SCALES,
    gr_scales: tuple[float, ...] = _DEFAULT_SCALES,
    seed_base: int = 0,
    resample_ess: float = 0.5,
    init_pos_std: float = _INIT_POS_STD,
) -> PFBMAResult:
    """Likelihood-weighted PF-BMA: seed-softmax consensus over independent PF runs.

    ``init_pos_std`` (default ``_INIT_POS_STD = 0.3``, v1's bit-identical
    behaviour) is the standard deviation (ft, TVT+Z frame) of every seed's
    initial particle-position spread around the anchor -- mirrors
    ``src/rogii/registration/particle.py::track_pf_bma``'s ``init_pos_std``
    param (commit f35249d). ``run_phase_b`` below passes
    ``PFBMA_INIT_POS_STD = 2.0`` (the R6-b(3) adopted value).

    Never raises: any failure falls back to a flat-anchor prediction for every
    scale, with ``std`` set to ``inf`` and ``log_lik`` set to ``-inf`` for
    every seed.
    """
    try:
        return _track_pf_bma_impl(
            h, tw, n_particles, n_seeds, bma_scales, gr_scales, seed_base, resample_ess,
            init_pos_std,
        )
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _flat_fallback_bma(n, anchor, bma_scales, n_seeds)


# --------------------------------------------------------------------------- #
# Ported from src/rogii/registration/beam.py (full-grid Viterbi DP tracker).
# Adapted from the public Kaggle notebooks lightningv08/lb-7-776-rogii-
# ridge-sp and nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-
# dwt-based. See that module's own docstring for full attribution.
# --------------------------------------------------------------------------- #

_DEFAULT_GRID_STEP_FT = 0.2
_DEFAULT_GRID_RANGE_FT = 120.0
_DEFAULT_MOVE_PENALTY = 20.0
_DEFAULT_MISMATCH_SCALE = 150.0
_DEFAULT_MAX_MOVE_PER_ROW = 3
_MIN_TYPEWELL_SAMPLES = 2


@dataclass
class BeamResult:
    tvt: np.ndarray
    margin: np.ndarray
    total_cost: float


@dataclass(frozen=True)
class BeamConfig:
    grid_step_ft: float = _DEFAULT_GRID_STEP_FT
    grid_range_ft: float = _DEFAULT_GRID_RANGE_FT
    move_penalty: float = _DEFAULT_MOVE_PENALTY
    mismatch_scale: float = _DEFAULT_MISMATCH_SCALE
    max_move_per_row: int = _DEFAULT_MAX_MOVE_PER_ROW
    weight: float = 1.0


# Best measured full-773-well configuration (analysis/experiment_ledger.md,
# 2026-07-06 "Beam/DTW" row): products 2400/3000/3750 at mismatch_scale=150.
DEFAULT_BEAM_CONFIGS: tuple[BeamConfig, ...] = (
    BeamConfig(move_penalty=16.0, mismatch_scale=150.0, max_move_per_row=3, weight=1.0),
    BeamConfig(move_penalty=20.0, mismatch_scale=150.0, max_move_per_row=3, weight=1.0),
    BeamConfig(move_penalty=25.0, mismatch_scale=150.0, max_move_per_row=3, weight=1.0),
)


def _empty_result() -> BeamResult:
    return BeamResult(tvt=np.zeros(0, dtype=float), margin=np.zeros(0, dtype=float), total_cost=0.0)


def _beam_flat_fallback(n: int, anchor: float) -> BeamResult:
    return BeamResult(
        tvt=np.full(n, anchor, dtype=float), margin=np.zeros(n, dtype=float), total_cost=0.0
    )


def _build_grid(
    tw: pd.DataFrame, anchor: float, grid_step_ft: float, grid_range_ft: float
) -> tuple[np.ndarray, float, float]:
    tw_tvt = tw["TVT"].to_numpy(dtype=float)
    tw_tvt = tw_tvt[np.isfinite(tw_tvt)]
    if tw_tvt.size < _MIN_TYPEWELL_SAMPLES:
        raise ValueError("typewell has too few finite TVT samples to build a grid")

    tw_min, tw_max = float(tw_tvt.min()), float(tw_tvt.max())
    if tw_max <= tw_min:
        raise ValueError("typewell TVT range is degenerate")

    lo = max(tw_min, anchor - grid_range_ft)
    hi = min(tw_max, anchor + grid_range_ft)
    if hi <= lo:  # anchor sits entirely outside the typewell's TVT extent
        lo, hi = tw_min, tw_max

    n_states = max(int(round((hi - lo) / grid_step_ft)) + 1, 3)
    grid_tvt = lo + np.arange(n_states, dtype=float) * grid_step_ft
    return grid_tvt, lo, hi


def _forward_viterbi(
    gr_eval: np.ndarray,
    grid_gr: np.ndarray,
    init_cost: np.ndarray,
    move_penalty: float,
    mismatch_scale: float,
    max_move: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_rows = gr_eval.shape[0]
    n_states = grid_gr.shape[0]

    penalty = move_penalty * np.abs(np.arange(-max_move, max_move + 1, dtype=float))

    backptr = np.empty((n_rows, n_states), dtype=np.int32)
    margin = np.zeros(n_rows, dtype=float)

    prev_cost = init_cost
    padded = np.empty(n_states + 2 * max_move, dtype=float)
    state_idx = np.arange(n_states)

    for i in range(n_rows):
        padded[:] = np.inf
        padded[max_move : max_move + n_states] = prev_cost
        window = np.lib.stride_tricks.sliding_window_view(padded, n_states)
        candidates = window + penalty[:, None]
        best_k = np.argmin(candidates, axis=0)
        best_cost = candidates[best_k, state_idx]
        backptr[i] = state_idx - (max_move - best_k)

        gv = gr_eval[i]
        row_mismatch = (gv - grid_gr) ** 2 / mismatch_scale if np.isfinite(gv) else 0.0

        cost_i = best_cost + row_mismatch
        if n_states >= 2:
            part = np.partition(cost_i, 1)
            margin[i] = float(part[1] - part[0])
        prev_cost = cost_i

    path = np.empty(n_rows, dtype=np.int64)
    path[-1] = int(np.argmin(prev_cost))
    total_cost = float(prev_cost[path[-1]])
    for i in range(n_rows - 1, 0, -1):
        path[i - 1] = backptr[i, path[i]]

    return path, margin, total_cost


def _track_beam_impl(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    grid_step_ft: float,
    grid_range_ft: float,
    move_penalty: float,
    mismatch_scale: float,
    max_move_per_row: int,
) -> BeamResult:
    n_eval = int(eval_mask(h).sum())
    if n_eval == 0:
        return _empty_result()

    anchor = last_known_tvt(h)
    eval_idx = np.flatnonzero(eval_mask(h))

    grid_tvt, lo, hi = _build_grid(tw, anchor, grid_step_ft, grid_range_ft)
    n_states = grid_tvt.size

    grid_gr = tw_gr_lookup(tw)(grid_tvt)

    gain, offset = fit_affine_gr(h, tw)
    gr_cal = apply_affine(h["GR"].to_numpy(dtype=float), gain, offset)
    gr_eval = gr_cal[eval_idx]

    anchor_idx = int(round((float(np.clip(anchor, lo, hi)) - lo) / grid_step_ft))
    anchor_idx = int(np.clip(anchor_idx, 0, n_states - 1))
    init_cost = move_penalty * np.abs(np.arange(n_states, dtype=float) - anchor_idx)

    max_move = max(int(max_move_per_row), 1)
    path, margin, total_cost = _forward_viterbi(
        gr_eval, grid_gr, init_cost, move_penalty, mismatch_scale, max_move
    )

    return BeamResult(tvt=grid_tvt[path], margin=margin, total_cost=total_cost)


def track_beam(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    grid_step_ft: float = _DEFAULT_GRID_STEP_FT,
    grid_range_ft: float = _DEFAULT_GRID_RANGE_FT,
    move_penalty: float = _DEFAULT_MOVE_PENALTY,
    mismatch_scale: float = _DEFAULT_MISMATCH_SCALE,
    max_move_per_row: int = _DEFAULT_MAX_MOVE_PER_ROW,
) -> BeamResult:
    """Register a horizontal well's GR log onto the typewell TVT axis via full-grid DP.

    Never raises: any failure falls back to a flat-anchor prediction.
    """
    try:
        return _track_beam_impl(
            h, tw, grid_step_ft, grid_range_ft, move_penalty, mismatch_scale, max_move_per_row
        )
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _beam_flat_fallback(n, anchor)


def _track_beam_multi_impl(h: pd.DataFrame, tw: pd.DataFrame, configs) -> BeamResult:
    results = [
        track_beam(
            h,
            tw,
            grid_step_ft=c.grid_step_ft,
            grid_range_ft=c.grid_range_ft,
            move_penalty=c.move_penalty,
            mismatch_scale=c.mismatch_scale,
            max_move_per_row=c.max_move_per_row,
        )
        for c in configs
    ]

    weights = np.array([c.weight for c in configs], dtype=float)
    wsum = weights.sum()
    weights = weights / wsum if wsum > 0 else np.full(len(configs), 1.0 / len(configs))

    tvt_stack = np.stack([r.tvt for r in results])
    margin_stack = np.stack([r.margin for r in results])
    tvt = np.tensordot(weights, tvt_stack, axes=(0, 0))
    margin = np.tensordot(weights, margin_stack, axes=(0, 0))
    total_cost = float(np.dot(weights, np.array([r.total_cost for r in results])))

    return BeamResult(tvt=tvt, margin=margin, total_cost=total_cost)


def track_beam_multi(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    configs: tuple[BeamConfig, ...] | None = None,
) -> BeamResult:
    """Weighted-average ensemble of :func:`track_beam` over several ``configs``.

    Defaults to :data:`DEFAULT_BEAM_CONFIGS`. Never raises.
    """
    try:
        resolved = configs if configs is not None else DEFAULT_BEAM_CONFIGS
        return _track_beam_multi_impl(h, tw, resolved)
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _beam_flat_fallback(n, anchor)


# --------------------------------------------------------------------------- #
# Ported from src/rogii/registration/beam_grid.py (parameterized beam-DP
# configs for the multi-configuration candidate-path bank). Reuses this
# file's own ``_build_grid``/``_forward_viterbi`` (identical to
# ``registration/beam_grid.py`` reusing ``registration/beam.py``'s private
# helpers rather than duplicating the DP kernel).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BeamGridConfig:
    label: str
    move_penalty: float
    mismatch_scale: float
    max_move_per_row: int = 3
    gr_calibration: bool = True
    smooth_radius: int = 0
    grid_step_ft: float = _DEFAULT_GRID_STEP_FT
    grid_range_ft: float = _DEFAULT_GRID_RANGE_FT

    @property
    def product(self) -> float:
        return self.move_penalty * self.mismatch_scale


def _smooth_gr(gr: np.ndarray, radius: int) -> np.ndarray:
    """Centered rolling-mean smoothing with a ``2*radius+1`` window. ``radius<=0`` is a no-op."""
    if radius <= 0:
        return gr
    window = 2 * radius + 1
    return (
        pd.Series(gr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _track_beam_grid_impl(h: pd.DataFrame, tw: pd.DataFrame, config: BeamGridConfig) -> BeamResult:
    n_eval = int(eval_mask(h).sum())
    if n_eval == 0:
        return _empty_result()

    anchor = last_known_tvt(h)
    eval_idx = np.flatnonzero(eval_mask(h))

    grid_tvt, lo, hi = _build_grid(tw, anchor, config.grid_step_ft, config.grid_range_ft)
    n_states = grid_tvt.size
    grid_gr = tw_gr_lookup(tw)(grid_tvt)

    raw_gr = h["GR"].to_numpy(dtype=float)
    if config.gr_calibration:
        gain, offset = fit_affine_gr(h, tw)
        gr_cal = apply_affine(raw_gr, gain, offset)
    else:
        gr_cal = raw_gr

    gr_cal = _smooth_gr(gr_cal, config.smooth_radius)
    gr_eval = gr_cal[eval_idx]

    anchor_idx = int(round((float(np.clip(anchor, lo, hi)) - lo) / config.grid_step_ft))
    anchor_idx = int(np.clip(anchor_idx, 0, n_states - 1))
    init_cost = config.move_penalty * np.abs(np.arange(n_states, dtype=float) - anchor_idx)

    max_move = max(int(config.max_move_per_row), 1)
    path, margin, total_cost = _forward_viterbi(
        gr_eval, grid_gr, init_cost, config.move_penalty, config.mismatch_scale, max_move
    )

    return BeamResult(tvt=grid_tvt[path], margin=margin, total_cost=total_cost)


def track_beam_grid(h: pd.DataFrame, tw: pd.DataFrame, config: BeamGridConfig) -> BeamResult:
    """Run one :class:`BeamGridConfig` of the full-grid Viterbi DP. Never raises."""
    try:
        return _track_beam_grid_impl(h, tw, config)
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _beam_flat_fallback(n, anchor)


# Ported verbatim from scripts/run_beam_grid_cv.py's CONFIG_GRID (module
# docstring there has the full axis rationale + the live 20-well time-gate
# measurement that sized it at K=10). mismatch_scale held fixed at 150.0
# throughout (only the move_penalty*mismatch_scale product matters -- see
# registration/beam.py's module docstring).
_MS = 150.0

CONFIG_GRID: tuple[BeamGridConfig, ...] = (
    BeamGridConfig("mid_cal_sm0_mm3", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("loose_cal_sm0_mm3", move_penalty=2000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("stiff_cal_sm0_mm3", move_penalty=4500 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("mid_nocal_sm0_mm3", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=False, smooth_radius=0),
    BeamGridConfig("mid_cal_sm2_mm3", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=2),
    BeamGridConfig("mid_cal_sm5_mm3", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=5),
    BeamGridConfig("mid_cal_sm0_mm1", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=1, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("mid_cal_sm0_mm2", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=2, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("loose_nocal_sm5_mm2", move_penalty=2000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=2, gr_calibration=False, smooth_radius=5),
    BeamGridConfig("stiff_cal_sm5_mm1", move_penalty=4500 / _MS, mismatch_scale=_MS,
                    max_move_per_row=1, gr_calibration=True, smooth_radius=5),
)


# --------------------------------------------------------------------------- #
# Phase A: prefix-cut backtest features (ported from
# scripts/build_backtest_features.py). 48 columns, cut_fracs=(0.5,0.7,0.85),
# candidates = carry_last / track_pf(seed=0) / track_beam_multi. The
# spatial-prior candidate is omitted here too (same reason as the source
# script: its leave-self-out SurfaceBank build is a ~35min fixed cost
# unrelated to this bank's per-well loop, and its cache directory does not
# exist on Kaggle at all).
# --------------------------------------------------------------------------- #

BT_CUT_FRACS: tuple[float, ...] = (0.5, 0.7, 0.85)
BT_CANDIDATES: tuple[str, ...] = ("carry_last", "pf", "beam")
BT_CANDIDATE_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(BT_CANDIDATES, 2))
BT_MIN_PREFIX_ROWS = 4


@dataclass
class TruncatedWell:
    h: pd.DataFrame
    y_true: np.ndarray
    prefix_len: int
    cut_idx: int
    pseudo_eval_len: int


def truncate_known_prefix(
    h: pd.DataFrame, cut_frac: float, min_prefix_rows: int = BT_MIN_PREFIX_ROWS
) -> TruncatedWell | None:
    """Cut ``h``'s known ``TVT_input`` prefix at ``cut_frac`` and build a pseudo eval zone.

    Leak-safe: the real, downstream eval-zone rows are dropped entirely (not
    just masked), and the pseudo ground truth is read once, before the
    ``TVT_input`` NaN-out, purely for scoring.
    """
    if not 0.0 < cut_frac < 1.0:
        raise ValueError(f"cut_frac must be in (0, 1), got {cut_frac}")

    tvt_input = h["TVT_input"].to_numpy(dtype=float)
    known_idx = np.flatnonzero(np.isfinite(tvt_input))
    if known_idx.size < min_prefix_rows:
        return None

    prefix_len = int(known_idx[-1]) + 1
    if prefix_len < min_prefix_rows or "TVT" not in h.columns:
        return None

    cut_idx = int(round(cut_frac * prefix_len))
    cut_idx = max(1, min(cut_idx, prefix_len - 1))

    h_trunc = h.iloc[:prefix_len].reset_index(drop=True).copy()
    y_true = h_trunc["TVT"].to_numpy(dtype=float)[cut_idx:prefix_len].copy()

    tvt_input_trunc = h_trunc["TVT_input"].to_numpy(dtype=float, copy=True)
    tvt_input_trunc[cut_idx:] = np.nan
    h_trunc["TVT_input"] = tvt_input_trunc

    return TruncatedWell(
        h=h_trunc,
        y_true=y_true,
        prefix_len=prefix_len,
        cut_idx=cut_idx,
        pseudo_eval_len=prefix_len - cut_idx,
    )


def _bt_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _bt_rank_from_values(values: dict[str, float]) -> dict[str, float]:
    arr = np.array([values.get(c, float("nan")) for c in BT_CANDIDATES])
    key = np.where(np.isfinite(arr), arr, np.inf)
    order = np.argsort(key, kind="stable")
    ranks = np.empty(len(BT_CANDIDATES))
    ranks[order] = np.arange(1, len(BT_CANDIDATES) + 1)
    return {c: float(ranks[i]) for i, c in enumerate(BT_CANDIDATES)}


def _bt_best_second_margin(values: dict[str, float]) -> float:
    arr = np.array([v for v in values.values() if np.isfinite(v)])
    if arr.size < 2:
        return float("nan")
    arr_sorted = np.sort(arr)
    return float(arr_sorted[1] - arr_sorted[0])


def _bt_pairwise_agreement(preds: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for a, b in BT_CANDIDATE_PAIRS:
        pa, pb = preds.get(a), preds.get(b)
        if pa is not None and pb is not None and pa.shape == pb.shape and pa.size:
            result[(a, b)] = float(np.mean(np.abs(pa - pb)))
        else:
            result[(a, b)] = float("nan")
    return result


def _bt_predict_candidate(name: str, h_trunc: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray | None:
    """Run one candidate predictor on a truncated well; ``None`` on any failure."""
    try:
        if name == "carry_last":
            return predict_carry_last(h_trunc)
        if name == "pf":
            return track_pf(h_trunc, tw).tvt
        if name == "beam":
            return track_beam_multi(h_trunc, tw).tvt
        raise ValueError(f"unknown candidate {name!r}")
    except Exception:
        return None


def bt_feature_names(cut_fracs: tuple[float, ...]) -> list[str]:
    """Deterministic 48-column order for the well x feature matrix."""
    names: list[str] = ["prefix_len"]
    names += [f"pseudo_eval_len_cut{cut:g}" for cut in cut_fracs]

    names += [f"bt_rmse_{cand}_cut{cut:g}" for cand in BT_CANDIDATES for cut in cut_fracs]
    for cand in BT_CANDIDATES:
        names += [f"bt_rmse_{cand}_mean", f"bt_rmse_{cand}_worst"]

    names += [f"bt_rank_{cand}_cut{cut:g}" for cut in cut_fracs for cand in BT_CANDIDATES]
    names += [f"bt_rank_{cand}_mean" for cand in BT_CANDIDATES]

    names += [f"bt_margin_best_second_cut{cut:g}" for cut in cut_fracs]
    names.append("bt_margin_best_second_mean")

    names += [
        f"bt_agree_{a}_{b}_cut{cut:g}" for cut in cut_fracs for a, b in BT_CANDIDATE_PAIRS
    ]
    names += [f"bt_agree_mean_cut{cut:g}" for cut in cut_fracs]
    names.append("bt_agree_mean")
    return names


def bt_compute_well_row(
    h: pd.DataFrame, tw: pd.DataFrame, cut_fracs: tuple[float, ...]
) -> tuple[dict[str, float], dict[str, int]]:
    """Build one well's feature dict. Never raises: failures degrade to NaN entries."""
    diag = {"truncate_fail": 0, "candidate_fail": 0}
    row: dict[str, float] = {}

    tvt_input_full = h["TVT_input"].to_numpy(dtype=float)
    known_idx = np.flatnonzero(np.isfinite(tvt_input_full))
    prefix_len = int(known_idx[-1]) + 1 if known_idx.size else 0
    row["prefix_len"] = float(prefix_len)

    rmse_by_cut: dict[float, dict[str, float]] = {}
    preds_by_cut: dict[float, dict[str, np.ndarray]] = {}
    pseudo_len_by_cut: dict[float, int] = {}

    for cut in cut_fracs:
        tw_row = truncate_known_prefix(h, cut)
        if tw_row is None:
            diag["truncate_fail"] += 1
            pseudo_len_by_cut[cut] = 0
            rmse_by_cut[cut] = {c: float("nan") for c in BT_CANDIDATES}
            preds_by_cut[cut] = {}
            continue

        pseudo_len_by_cut[cut] = tw_row.pseudo_eval_len
        cand_rmse: dict[str, float] = {}
        cand_preds: dict[str, np.ndarray] = {}
        for cand in BT_CANDIDATES:
            pred = _bt_predict_candidate(cand, tw_row.h, tw)
            if pred is None or pred.shape != tw_row.y_true.shape or not np.all(np.isfinite(pred)):
                diag["candidate_fail"] += 1
                cand_rmse[cand] = float("nan")
                continue
            cand_rmse[cand] = _bt_rmse(tw_row.y_true, pred)
            cand_preds[cand] = pred
        rmse_by_cut[cut] = cand_rmse
        preds_by_cut[cut] = cand_preds
        del tw_row

    for cut in cut_fracs:
        row[f"pseudo_eval_len_cut{cut:g}"] = float(pseudo_len_by_cut[cut])

    for cand in BT_CANDIDATES:
        for cut in cut_fracs:
            row[f"bt_rmse_{cand}_cut{cut:g}"] = rmse_by_cut[cut].get(cand, float("nan"))
        vals = np.array([rmse_by_cut[cut].get(cand, float("nan")) for cut in cut_fracs])
        finite = vals[np.isfinite(vals)]
        row[f"bt_rmse_{cand}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        row[f"bt_rmse_{cand}_worst"] = float(np.max(finite)) if finite.size else float("nan")

    rank_history: dict[str, list[float]] = {c: [] for c in BT_CANDIDATES}
    for cut in cut_fracs:
        ranks = _bt_rank_from_values(rmse_by_cut[cut])
        for cand in BT_CANDIDATES:
            row[f"bt_rank_{cand}_cut{cut:g}"] = ranks[cand]
            rank_history[cand].append(ranks[cand])
    for cand in BT_CANDIDATES:
        row[f"bt_rank_{cand}_mean"] = float(np.mean(rank_history[cand]))

    margins: list[float] = []
    for cut in cut_fracs:
        m = _bt_best_second_margin(rmse_by_cut[cut])
        row[f"bt_margin_best_second_cut{cut:g}"] = m
        margins.append(m)
    finite_margins = np.array([m for m in margins if np.isfinite(m)])
    row["bt_margin_best_second_mean"] = (
        float(np.mean(finite_margins)) if finite_margins.size else float("nan")
    )

    agree_means: list[float] = []
    for cut in cut_fracs:
        pair_vals = _bt_pairwise_agreement(preds_by_cut[cut])
        for a, b in BT_CANDIDATE_PAIRS:
            row[f"bt_agree_{a}_{b}_cut{cut:g}"] = pair_vals[(a, b)]
        finite_pairs = [v for v in pair_vals.values() if np.isfinite(v)]
        cut_mean = float(np.mean(finite_pairs)) if finite_pairs else float("nan")
        row[f"bt_agree_mean_cut{cut:g}"] = cut_mean
        agree_means.append(cut_mean)
    finite_agree = np.array([v for v in agree_means if np.isfinite(v)])
    row["bt_agree_mean"] = float(np.mean(finite_agree)) if finite_agree.size else float("nan")

    return row, diag


def run_phase_a(wells: list[str], root: Path, out_path: Path, t_program_start: float) -> None:
    feature_names = bt_feature_names(BT_CUT_FRACS)
    n_wells, n_features = len(wells), len(feature_names)
    print(
        f"\n{'=' * 78}\n# Phase A: backtest features ({n_wells} wells, "
        f"{n_features} features, cut_fracs={BT_CUT_FRACS})\n{'=' * 78}"
    )

    matrix = np.full((n_wells, n_features), np.nan, dtype=np.float32)
    col_pos = {name: i for i, name in enumerate(feature_names)}

    timings: list[float] = []
    n_well_failures = 0
    n_truncate_fail = 0
    n_candidate_fail = 0
    n_processed = 0
    n_stopped_early = 0

    t_start = time.perf_counter()
    for i, well in enumerate(wells, start=1):
        if budget_left(t_program_start) <= 0:
            n_stopped_early = n_wells - i + 1
            print(
                f"  [budget] exhausted at well {i}/{n_wells} -- stopping Phase A early "
                f"({n_stopped_early} wells left unprocessed, rows stay NaN)"
            )
            break
        t0 = time.perf_counter()
        try:
            h = load_horizontal(well, "train", root)
            tw = load_typewell(well, "train", root)
            row, diag = bt_compute_well_row(h, tw, BT_CUT_FRACS)
            for name, value in row.items():
                matrix[i - 1, col_pos[name]] = value
            n_truncate_fail += diag["truncate_fail"]
            n_candidate_fail += diag["candidate_fail"]
            n_processed += 1
            del h, tw
        except Exception as exc:  # a single well must never abort the whole run
            n_well_failures += 1
            print(f"  [WARN] well {well} failed entirely: {exc!r}")
        timings.append(time.perf_counter() - t0)

        if i % PROGRESS_EVERY == 0 or i == n_wells:
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{i}/{n_wells}] elapsed={elapsed:.1f}s well_failures={n_well_failures} "
                f"truncate_fail={n_truncate_fail} candidate_fail={n_candidate_fail}",
                flush=True,
            )

    timings_arr = np.array(timings)
    n_nan = int(np.isnan(matrix).sum())
    total_cells = matrix.size
    print(
        f"\n## Phase A summary: processed={n_processed}/{n_wells} "
        f"well_failures={n_well_failures} truncate_fail={n_truncate_fail} "
        f"candidate_fail={n_candidate_fail} stopped_early={n_stopped_early}"
    )
    print(f"## NaN rate: {n_nan}/{total_cells} ({100 * n_nan / total_cells:.2f}%)")
    if timings_arr.size:
        print(
            f"## per-well time: mean={timings_arr.mean():.3f}s "
            f"p95={np.percentile(timings_arr, 95):.3f}s total={timings_arr.sum():.1f}s"
        )

    np.savez(
        out_path,
        features=matrix,
        feature_names=np.array(feature_names),
        well_names=np.array(wells),
        cut_fracs=np.array(BT_CUT_FRACS, dtype=np.float64),
        candidates=np.array(BT_CANDIDATES),
    )
    print(f"## saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


# --------------------------------------------------------------------------- #
# Phase B: PF-BMA candidate-path bank (ported from scripts/run_pf_bma_cv.py's
# Phase 1 bank-building; its Phase 0 time-gate probe and Phase 2 OOF-oracle
# are intentionally not ported -- see module docstring).
# --------------------------------------------------------------------------- #

PFBMA_N_PARTICLES = 512
PFBMA_N_SEEDS = 12  # fixed: this repo's dev-box time-gate already chose 12 for a 60min budget
PFBMA_GR_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)
PFBMA_BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)

# R6-b(3) adopted value (analysis/experiment_ledger.md, 2026-07-10): widening
# the initial particle-position spread from v1's 0.3 (== _INIT_POS_STD --
# still used unchanged by Phase A's track_pf/track_beam_multi candidates) to
# 2.0 dropped 150-well pooled pf_bma_scale3 RMSE by -0.59 (12.1848 ->
# 11.5933). This is the *only* behavioural change from v1's Phase B in this
# file -- mirrors src/rogii/registration/particle.py's init_pos_std param
# (commit f35249d).
PFBMA_INIT_POS_STD = 2.0


def run_phase_b(wells: list[str], root: Path, out_path: Path, t_program_start: float) -> None:
    print(
        f"\n{'=' * 78}\n# Phase B: PF-BMA bank ({len(wells)} wells, "
        f"n_seeds={PFBMA_N_SEEDS})\n{'=' * 78}"
    )

    used_wells: list[str] = []
    well_idx_parts: list[np.ndarray] = []
    y_true_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    scale_tvt_parts: dict[float, list[np.ndarray]] = {s: [] for s in PFBMA_BMA_SCALES}
    scale_std_parts: dict[float, list[np.ndarray]] = {s: [] for s in PFBMA_BMA_SCALES}
    sse = {s: 0.0 for s in PFBMA_BMA_SCALES}
    n_rows_total = {s: 0 for s in PFBMA_BMA_SCALES}

    n_skipped = 0
    n_well_failures = 0
    n_stopped_early = 0
    timings: list[float] = []
    t_start = time.perf_counter()

    for i, well in enumerate(wells, start=1):
        if budget_left(t_program_start) <= 0:
            n_stopped_early = len(wells) - i + 1
            print(f"  [budget] exhausted at well {i}/{len(wells)} -- stopping Phase B early")
            break
        try:
            h = load_horizontal(well, "train", root)
            tw = load_typewell(well, "train", root)
            ez = eval_mask(h)
            if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
                n_skipped += 1
                continue

            y_true = h["TVT"].to_numpy(dtype=float)[ez]
            anchor_pred = predict_carry_last(h)

            t0 = time.perf_counter()
            result = track_pf_bma(
                h,
                tw,
                n_particles=PFBMA_N_PARTICLES,
                n_seeds=PFBMA_N_SEEDS,
                bma_scales=PFBMA_BMA_SCALES,
                gr_scales=PFBMA_GR_SCALES,
                init_pos_std=PFBMA_INIT_POS_STD,
            )
            timings.append(time.perf_counter() - t0)

            well_i = len(used_wells)
            used_wells.append(well)
            well_idx_parts.append(np.full(y_true.shape[0], well_i, dtype=np.int32))
            y_true_parts.append(y_true.astype(np.float32))
            anchor_parts.append(anchor_pred.astype(np.float32))

            for scale in PFBMA_BMA_SCALES:
                tvt = result.tvt_by_scale[scale]
                std = result.std_by_scale[scale]
                scale_tvt_parts[scale].append(tvt.astype(np.float32))
                scale_std_parts[scale].append(std.astype(np.float32))
                sse[scale] += float(np.sum((y_true - tvt) ** 2))
                n_rows_total[scale] += y_true.size
            del h, tw
        except Exception as exc:  # a single well must never abort the whole run
            n_well_failures += 1
            print(f"  [WARN] well {well} failed entirely: {exc!r}")

        if i % PROGRESS_EVERY == 0 or i == len(wells):
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{i}/{len(wells)}] used={len(used_wells)} skipped={n_skipped} "
                f"well_failures={n_well_failures} elapsed={elapsed:.1f}s",
                flush=True,
            )

    timings_arr = np.array(timings)
    print(
        f"\n## Phase B summary: used={len(used_wells)}/{len(wells)} skipped={n_skipped} "
        f"well_failures={n_well_failures} stopped_early={n_stopped_early}"
    )
    if timings_arr.size:
        print(
            f"## track_pf_bma per-well time: mean={timings_arr.mean():.3f}s "
            f"p95={np.percentile(timings_arr, 95):.3f}s total={timings_arr.sum():.1f}s"
        )
    print("## pooled RMSE (raw path) per bma_scale:")
    for scale in PFBMA_BMA_SCALES:
        r = np.sqrt(sse[scale] / n_rows_total[scale]) if n_rows_total[scale] else float("nan")
        print(f"    scale={scale:g}: {r:.4f}")

    payload: dict[str, np.ndarray] = {
        "well_idx": np.concatenate(well_idx_parts) if well_idx_parts else np.zeros(0, np.int32),
        "used_wells": np.array(used_wells),
        "y_true": np.concatenate(y_true_parts) if y_true_parts else np.zeros(0, np.float32),
        "anchor": np.concatenate(anchor_parts) if anchor_parts else np.zeros(0, np.float32),
        "bma_scales": np.array(PFBMA_BMA_SCALES, dtype=np.float64),
        "gr_scales": np.array(PFBMA_GR_SCALES, dtype=np.float64),
        "n_particles": np.int64(PFBMA_N_PARTICLES),
        "n_seeds": np.int64(PFBMA_N_SEEDS),
        "init_pos_std": np.float64(PFBMA_INIT_POS_STD),
    }
    for scale in PFBMA_BMA_SCALES:
        payload[f"pf_bma_scale{scale:g}_tvt"] = (
            np.concatenate(scale_tvt_parts[scale])
            if scale_tvt_parts[scale]
            else np.zeros(0, np.float32)
        )
        payload[f"pf_bma_scale{scale:g}_std"] = (
            np.concatenate(scale_std_parts[scale])
            if scale_std_parts[scale]
            else np.zeros(0, np.float32)
        )

    np.savez(out_path, **payload)
    print(f"## saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


# --------------------------------------------------------------------------- #
# Phase C: beam-grid candidate-path bank (ported from
# scripts/run_beam_grid_cv.py's Phase 2 bank-building; its own time-gate
# probe and the OOF-dependent (4+K)-way oracle / correlation-matrix sections
# are intentionally not ported -- see module docstring).
# --------------------------------------------------------------------------- #


def run_phase_c(wells: list[str], root: Path, out_path: Path, t_program_start: float) -> None:
    print(
        f"\n{'=' * 78}\n# Phase C: beam grid bank ({len(wells)} wells x "
        f"{len(CONFIG_GRID)} configs)\n{'=' * 78}"
    )

    n_configs = len(CONFIG_GRID)
    tvt_parts: list[list[np.ndarray]] = [[] for _ in range(n_configs)]
    margin_parts: list[list[np.ndarray]] = [[] for _ in range(n_configs)]
    well_idx_parts: list[np.ndarray] = []
    y_true_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    used_wells: list[str] = []
    eval_len_per_well: list[int] = []

    n_skipped = 0
    n_well_failures = 0
    n_config_shape_fail = 0
    n_stopped_early = 0
    t_start = time.perf_counter()

    for i, well in enumerate(wells, start=1):
        if budget_left(t_program_start) <= 0:
            n_stopped_early = len(wells) - i + 1
            print(f"  [budget] exhausted at well {i}/{len(wells)} -- stopping Phase C early")
            break
        try:
            h = load_horizontal(well, "train", root)
            tw = load_typewell(well, "train", root)
            ez = eval_mask(h)
            if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
                n_skipped += 1
                continue

            y_true = h["TVT"].to_numpy(dtype=np.float64)[ez]
            anchor_pred = predict_carry_last(h).astype(np.float64)
            n = y_true.size

            well_i = len(used_wells)
            for c_i, cfg in enumerate(CONFIG_GRID):
                result = track_beam_grid(h, tw, cfg)
                tvt = result.tvt
                margin = result.margin
                if tvt.shape != (n,) or margin.shape != (n,) or not np.all(np.isfinite(tvt)):
                    n_config_shape_fail += 1
                    tvt = anchor_pred.copy()
                    margin = np.zeros(n, dtype=float)
                tvt_parts[c_i].append(tvt.astype(np.float32))
                margin_parts[c_i].append(margin.astype(np.float32))

            well_idx_parts.append(np.full(n, well_i, dtype=np.int32))
            y_true_parts.append(y_true.astype(np.float32))
            anchor_parts.append(anchor_pred.astype(np.float32))
            used_wells.append(well)
            eval_len_per_well.append(n)
            del h, tw
        except Exception as exc:  # a single well must never abort the whole run
            n_well_failures += 1
            print(f"  [WARN] well {well} failed entirely: {exc!r}")

        if i % PROGRESS_EVERY == 0 or i == len(wells):
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{i}/{len(wells)}] used={len(used_wells)} skipped={n_skipped} "
                f"well_failures={n_well_failures} config_shape_fail={n_config_shape_fail} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(
        f"\n## Phase C summary: used={len(used_wells)}/{len(wells)} skipped={n_skipped} "
        f"well_failures={n_well_failures} config_shape_fail={n_config_shape_fail} "
        f"stopped_early={n_stopped_early}"
    )

    if used_wells:
        y_true_all = np.concatenate(y_true_parts)
        n_total = y_true_all.size
        print("## per-config raw pooled RMSE (diversity check):")
        for c_i, cfg in enumerate(CONFIG_GRID):
            tvt_all = np.concatenate(tvt_parts[c_i]) if tvt_parts[c_i] else np.zeros(0, np.float32)
            raw_rmse = (
                float(np.sqrt(np.sum((y_true_all - tvt_all) ** 2) / n_total))
                if n_total
                else float("nan")
            )
            print(f"    {cfg.label:24s} product={cfg.product:.0f} raw_pooled_rmse={raw_rmse:.4f}")

    bank: dict[str, np.ndarray] = {
        "y_true": np.concatenate(y_true_parts) if y_true_parts else np.zeros(0, np.float32),
        "anchor": np.concatenate(anchor_parts) if anchor_parts else np.zeros(0, np.float32),
        "well_idx": np.concatenate(well_idx_parts) if well_idx_parts else np.zeros(0, np.int32),
        "used_wells": np.array(used_wells),
        "eval_len_per_well": np.array(eval_len_per_well, dtype=np.int32),
        "config_labels": np.array([c.label for c in CONFIG_GRID]),
        "product_per_config": np.array([c.product for c in CONFIG_GRID], dtype=np.float64),
        "max_move_per_config": np.array(
            [c.max_move_per_row for c in CONFIG_GRID], dtype=np.int32
        ),
        "gr_calibration_per_config": np.array(
            [c.gr_calibration for c in CONFIG_GRID], dtype=np.int8
        ),
        "smooth_radius_per_config": np.array(
            [c.smooth_radius for c in CONFIG_GRID], dtype=np.int32
        ),
    }
    for c_i in range(n_configs):
        bank[f"beam_tvt_{c_i}"] = (
            np.concatenate(tvt_parts[c_i]) if tvt_parts[c_i] else np.zeros(0, np.float32)
        )
        bank[f"beam_margin_{c_i}"] = (
            np.concatenate(margin_parts[c_i]) if margin_parts[c_i] else np.zeros(0, np.float32)
        )

    np.savez_compressed(out_path, **bank)
    print(f"## saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    t_program_start = time.perf_counter()
    root = find_data_root()
    out_dir = output_dir()
    print(f"# data root: {root}")
    print(f"# output dir: {out_dir}")
    print(f"# global time budget: {GLOBAL_TIME_BUDGET_S / 3600:.1f}h")

    all_wells = list_wells("train", root)
    limit = _bank_limit()
    wells = all_wells[:limit] if limit is not None else all_wells
    print(
        f"# train wells: {len(all_wells)} total, using {len(wells)} "
        f"(ROGII_BANK_LIMIT={limit})"
    )

    print(f"# RUN_PHASES: {RUN_PHASES}")

    if "A" not in RUN_PHASES:
        print("\n# Phase A skipped (RUN_PHASES does not include 'A')")
    elif budget_left(t_program_start) <= 0:
        print("[budget] exhausted before Phase A -- skipping all phases")
    else:
        run_phase_a(wells, root, out_dir / "backtest_features.npz", t_program_start)

    if "B" not in RUN_PHASES:
        print("\n# Phase B skipped (RUN_PHASES does not include 'B')")
    elif budget_left(t_program_start) <= 0:
        print("\n[budget] exhausted before Phase B -- skipping")
    else:
        run_phase_b(wells, root, out_dir / "path_bank_pfbma_v2.npz", t_program_start)

    if "C" not in RUN_PHASES:
        print("\n# Phase C skipped (RUN_PHASES does not include 'C')")
    elif budget_left(t_program_start) <= 0:
        print("\n[budget] exhausted before Phase C -- skipping")
    else:
        run_phase_c(wells, root, out_dir / "path_bank_beam.npz", t_program_start)

    total = time.perf_counter() - t_program_start
    print(f"\n# bank_builder total runtime: {total:.1f}s ({total / 60:.1f}min)")




# ======================================================================
# R6-b(2) appendix -- PF seed-count sensitivity for the PRODUCTION pf_blend.
#
# Base above = bank_builder.py verbatim (guard stripped at build time).
# Question: does averaging more track_pf seeds (8 -> 16 -> 32) improve the
# pf_blend candidate's pooled RMSE enough to justify bumping N_SEEDS in the
# submission kernel (aggressive-slot candidate (b), analysis/r5_endgame.md)?
#
# Design: run track_pf once per seed (32 seeds), then average nested subsets
# {first 8, first 16, all 32} of the SAME runs -- paired by construction,
# 1x compute instead of 3x. Seeds 0..31 match track_pf_multi's range(n)
# seeding, so the 8-subset equals production pf_blend exactly.
#
# Pre-declared decision rule: adopt-as-aggressive-candidate only if pooled
# RMSE improves by >= 0.05 at 16 or 32 vs 8; otherwise close R6-b(2) for good.
# (Runtime if adopted: 16 seeds ~= +3.1s/well -> hidden rerun ~2.9h, fine;
# 32 seeds ~= +9.3s/well -> ~4.6h, still inside 9h.)
# ======================================================================

R6B2_MAX_SEEDS = 32
R6B2_SUBSETS = (8, 16, 32)


def r6b2_main():
    t0 = time.perf_counter()
    root = find_data_root()
    out_dir = output_dir()
    wells = list_wells("train", root)
    limit = _bank_limit()
    if limit is not None:
        wells = wells[:limit]
    print(f"# R6-b(2) pf seed sensitivity: {len(wells)} wells, "
          f"{R6B2_MAX_SEEDS} seeds, subsets {R6B2_SUBSETS}")

    rows_y = []
    rows_pred = {k: [] for k in R6B2_SUBSETS}
    w_name, w_nrows = [], []
    failures = 0
    for wi, well in enumerate(wells):
        try:
            h = load_horizontal(well, "train", root)
            tw = load_typewell(well, "train", root)
            m = eval_mask(h)
            n_eval = int(m.sum())
            if n_eval == 0:
                continue
            y_true = h["TVT"].to_numpy(dtype=float)[np.flatnonzero(m)]
            fin = np.isfinite(y_true)
            if not fin.any():
                continue
            per_seed = np.stack(
                [track_pf(h, tw, seed=s).tvt for s in range(R6B2_MAX_SEEDS)]
            )  # (32, n_eval)
            rows_y.append(y_true[fin])
            for k in R6B2_SUBSETS:
                rows_pred[k].append(per_seed[:k].mean(axis=0)[fin])
            w_name.append(well)
            w_nrows.append(int(fin.sum()))
        except Exception as e:  # noqa: BLE001 -- sweep must survive odd wells
            failures += 1
            print(f"  [fail] {well}: {type(e).__name__}: {e}")
        if (wi + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {wi + 1}/{len(wells)} wells | {el:.0f}s | "
                  f"{el / (wi + 1):.1f}s/well", flush=True)

    y = np.concatenate(rows_y)
    print(f"\nwells scored={len(w_name)} failures={failures} rows={y.size}")
    print("pooled RMSE of pf_blend by seed count (paired, nested subsets):")
    base = None
    summary = {}
    for k in R6B2_SUBSETS:
        pred = np.concatenate(rows_pred[k])
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        summary[k] = rmse
        if base is None:
            base = rmse
        print(f"  n_seeds={k:2d}: {rmse:8.4f}  (delta vs 8: {rmse - base:+.4f})")

    np.savez_compressed(
        out_dir / "r6b2_nseeds.npz",
        used_wells=np.asarray(w_name), n_rows=np.asarray(w_nrows, dtype=np.int32),
        y_true=y.astype(np.float32),
        **{f"pf_mean_{k}": np.concatenate(rows_pred[k]).astype(np.float32)
           for k in R6B2_SUBSETS},
        subset_keys=np.asarray(list(summary.keys()), dtype=np.int32),
        subset_rmse=np.asarray(list(summary.values())),
    )
    print(f"\nsaved {out_dir / 'r6b2_nseeds.npz'}")
    print(f"# r6b2 total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    r6b2_main()
