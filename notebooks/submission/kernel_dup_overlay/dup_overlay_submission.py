"""Self-contained Kaggle Notebook submission: stack v2.6 / R13 (59 features + 5-seed).

Paste this entire script into a single Kaggle Notebook cell for the ROGII
Wellbore Geology Prediction **code competition** (internet disabled, <=9h). It
has NO project imports (no ``import rogii``) -- only numpy/pandas/scipy/
lightgbm/stdlib -- so it is fully self-contained and runs unmodified both:

- on Kaggle, reading the mounted competition dataset under ``/kaggle/input``
  (see :func:`find_data_root` for the exact discovery strategy), and
- locally against ``data/raw/`` for validation::

    uv run python notebooks/submission/kernel_stack_v2_blend/stack_v2_blend_submission.py
    ROGII_SMOKE=20 uv run python \
        notebooks/submission/kernel_stack_v2_blend/stack_v2_blend_submission.py

Method: sub-v8 = stack v2.6 (R13: 56 + H3 = 59 features, 5-seed LGBM mean)
----------------------------------------------------------------------------
This kernel is ``notebooks/submission/kernel_stack_v2`` (sub-v3, LB 9.022)
extended per ``scripts/run_stack_v26_cv.py`` (R13, pooled OOF RMSE **9.1456**
with the 5-seed mean; ``analysis/experiment_ledger.md``, ledger commit
baf82a7). R13 is the best **transfer-safe** configuration to date: every
feature group in it is either LB-calibrated or raw-data-derived. The
bank-candidate feature family (R7 + G1 + G2, CV 8.8573 in sub-v7) is
deliberately NOT included here -- its CV->LB transfer is unproven, and the
one bank-derived construct that HAS been LB-tested (sub-v4's output blend,
honest CV 9.1104) came back **LB 9.456**, worse than stack v2 alone
(LB 9.022). What did transfer is stack v2's feature mechanism itself
(well-CV 9.2309 -> LB 9.022). sub-v8 therefore ships:

  - stack v2's 56 columns (P1 + tracker + well-agg + spatial), unchanged;
  - **H3**: 3 raw-data columns (:data:`H3_FEATURE_COLUMNS`, ported
    byte-for-byte from ``run_stack_v26_cv.py::build_h3_features``, itself a
    byte-copy of R12's ``run_stack_v25_cv.py``);
  - the R9 5-seed LGBM mean (unchanged from sub-v7).

H3 reads only ``MD``, the known-zone ``TVT_input``, and ``Z`` (all present
in the test schema; computed directly from the raw CSVs -- no bank npz, no
extra trackers):

  - ``known_linear_trend_extrap_minus_anchor``: known-zone tail-200
    ``TVT_input ~ MD`` linear-trend slope extrapolated to each eval row's
    MD, relative to the anchor (``slope * (MD[i] - anchor_MD)``).
  - ``known_tvt_z_identity_resid_std``: residual std of a linear fit of
    ``(TVT_input + Z) ~ MD`` over the FULL known zone (leak-safe
    ANCC=TVT+Z identity proxy; never reads the train-only ``ANCC`` column).
  - ``zone_len_known_len_ratio``: ``n_eval_rows / n_known_rows``.

**Prediction = stack v2.6's own output**: ``pf_blend + mean over
ENSEMBLE_SEEDS = (42, 101, 202, 303, 404) of each member's LGBM residual``.
``random_state`` is the only member-to-member difference (LightGBM derives
its bagging/feature/data seeds from it); the fit/ES-holdout well split stays
fixed at seed 42 for every member -- both exactly the R9/R13 recipe.

The PF-BMA (:func:`track_pf_bma`) and beam-grid (:func:`track_beam_grid`)
tracker implementations from sub-v6/v7 are RETAINED below as library code
but are NOT called anywhere in sub-v8 (their feature columns left with the
bank-candidate family) -- Phase 2 is back to the sub-v3 cost profile
(PF + beam-multi + spatial only, ~5.7s/well measured locally).

Fallback design (never crash)
-------------------------------
:func:`predict_well_stack` wraps the whole per-well prediction: any failure
(missing tracker cache entry, a tracker length mismatch, non-finite output)
degrades that well to ``carry_last``; a further outer try/except in
:func:`build_predictions` degrades a total failure to ``0.0`` with a printed
warning -- never aborting the whole submission. On the training side,
:func:`build_train_matrix` skips (and counts) any well whose tracker arrays
mismatch its eval-zone length. H3 itself never raises (degenerate wells
yield 0.0-filled columns, mirroring the CV script).

Runtime budget (<=9h Kaggle CPU)
----------------------------------
Actual per-well timing (this file's own local measurement, 10 train wells,
full PF(8-seed) + beam-multi(3-config) + spatial + the 2 new candidates
combined) is printed at the end of this docstring's revision history in
``analysis/experiment_ledger.md`` once measured; the in-code
``[PROGRESS]``/``[TIMING]`` prints report it live during every run so the 9h
budget can be monitored without relying on a stale docstring number.

Leak boundary
-------------
Only columns present in BOTH train and test schemas are ever read from a
target well's own horizontal-well log: ``MD, X, Y, Z, GR, TVT_input``. The
ground-truth ``TVT`` column is read exactly once per **train** well (never a
test well) to build the regression target/label; it is never a feature. The
train-only formation-depth columns (``ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA``)
and the typewell's ``Geology`` column are never read anywhere in this file.
The 3 H3 columns read only ``MD``, the known-zone ``TVT_input`` prefix, and
``Z`` -- all present in the test schema, all disjoint from the eval zone's
unknowns (``known_tvt_z_identity_resid_std`` is a leak-safe proxy for the
ANCC=TVT+Z identity fitted over the KNOWN zone only; the train-only ``ANCC``
column itself is never read) -- no new leak surface (see
``run_stack_v25_cv.py``/``run_stack_v26_cv.py``'s own leak-boundary notes).

The spatial-prior surface bank (:func:`build_surface_bank`) is built ONLY
from the 773 **train** wells' formation-depth columns (test wells have no
such columns to contribute -- ``build_surface_bank`` silently skips any well
whose horizontal-well log lacks a given formation column, so passing it a
test-schema DataFrame would contribute zero samples even if attempted). At
*query* time (:func:`predict_spatial`), a train well's own bank samples are
excluded via ``exclude_well`` (leave-self-out KNN); a test well's
``exclude_well`` simply never matches anything in the bank (it was never
added), so the "exclusion" is a structural no-op there -- there is nothing of
the test well's own to leak out of a bank that never contained it. The
query's own known-zone ``TVT_input`` prefix (used to calibrate the per-well
offset and to measure ``prefix_rmse`` fit quality) is read from ``h`` for
BOTH train and test wells -- this is safe because it is disjoint from the
evaluation zone by construction (``known = TVT_input.notna()``, ``eval_m =
TVT_input.isna()``; see ``_predict_spatial_impl``) and is present in the test
schema too.

Predictor
---------
``target = TVT - pf_blend`` where ``pf_blend = anchor + 0.7 * (PF_mean -
anchor)`` (identical formulation to ``scripts/run_stack_cv.py``'s (b)
PF-residual target, ``scripts/run_stack_v2_cv.py`` and
``scripts/run_stack_v26_cv.py``). Each of the 5 LGBM members (59 features)
predicts the residual; the final submitted prediction is ``pf_blend +
mean(residual_pred over members)`` -- stack v2.6's own output, no blending.
Ported verbatim from (attribution kept per-module below):

- ``src/rogii/data.py`` (eval-zone / anchor helpers)
- ``src/rogii/baseline.py`` (``predict_carry_last``, the per-well fallback)
- ``src/rogii/typewell.py`` (GR lookup + affine calibration)
- ``src/rogii/features.py`` (P1 Layer2 within-well features, 36 columns)
- ``src/rogii/stack_features.py`` (tracker / well-aggregate / spatial feature
  blocks, 8 + 8 + 4 columns)
- ``src/rogii/registration/particle.py`` (PF tracker; adapted from the public
  Kaggle notebook ``lightningv08/lb-7-776-rogii-ridge-sp``, see that module's
  own docstring for full attribution)
- ``src/rogii/registration/beam.py`` (beam/DP tracker; adapted from the same
  public notebook plus ``nihilisticneuralnet/9-251-rogii-wellbore-geology-
  prediction-dwt-based``, see that module's own docstring)
- ``src/rogii/spatial.py`` (spatial-prior predictor; host forum idea, see
  that module's own docstring)
- ``scripts/build_tracker_cache.py`` / ``scripts/build_spatial_cache.py``
  (the exact PF/beam/spatial call configuration used to build the cache that
  produced the 9.2309 OOF number, reproduced here so the kernel's own
  from-scratch tracker pass matches it)
- ``scripts/run_stack_v2_cv.py`` (feature-matrix assembly, LGB_PARAMS, the
  ES-holdout well-split recipe, and the (b) PF-residual target formulation)
- ``notebooks/submission/kernel_bank_builder/bank_builder.py`` (v2 Phase B:
  PF-BMA seed-softmax; Phase C: beam-grid -- both RETAINED as library code
  below but not called in sub-v8, see the Method section)
- ``scripts/run_stack_v23_cv.py`` (R9: the 5-seed ensemble recipe --
  :data:`ENSEMBLE_SEEDS`, ``_lgb_params_for_seed``)
- ``scripts/run_stack_v25_cv.py`` / ``scripts/run_stack_v26_cv.py``
  (R12/R13: :data:`H3_FEATURE_COLUMNS` and :func:`build_h3_features`'s exact
  definitions, and the adoption evidence -- pooled OOF RMSE 9.1456, honest
  well-GroupKFold(5) + ES-holdout protocol, ledger commit baf82a7)

Contract
--------
- The set of required ``(well, row_index)`` pairs for output is derived from
  ``sample_submission.csv``'s ``id`` column (``{well}_{row_index}``), NOT by
  listing ``test/`` directly (locally ``test/`` holds only 3 example wells;
  the real Kaggle rerun substitutes a ~200-well ``test/`` set).
- Per-well prediction **never raises**: the trackers already carry their own
  internal exception-safe fallback (flat anchor); :func:`predict_well_stack`
  wraps the stack-model prediction itself in a second try/except so any
  failure (missing tracker/spatial cache entry, shape mismatch, non-finite
  output, ...) for a single well degrades to ``carry_last`` for that well
  only -- never aborting the whole submission.
- ``ROGII_SMOKE=<N>`` (env var) restricts training to the first ``N`` train
  wells (sorted, deterministic) for a fast local smoke test; the full test
  set is always predicted. Unset/absent on Kaggle -> full 773 train wells.
- Writes ``submission.csv`` with columns ``id,tvt`` and validates it against
  ``sample_submission.csv`` before writing (exits non-zero on failure).

ROGII_DUP duplicate-twin overlay (new; strictly additive on top of sub-v8)
---------------------------------------------------------------------------
This file is a byte-for-byte copy of
``notebooks/submission/kernel_stack_v2_blend/stack_v2_blend_submission.py``
(the proven sub-v8, LB 9.014) with one overlay bolted on top, never
replacing or editing any of the logic above: a visible-test-well audit found
that some test wells are literal train-well duplicates -- identical well ID,
identical MD range, byte-identical ``GR``/``X``/``Y``/``TVT_input`` over the
whole well (see :func:`find_duplicate_twin`). When a hidden test well has
such a twin in train, its evaluation-zone ``TVT`` can be copied directly from
the twin's own ``TVT`` column (RMSE 0 for that well) instead of relying on
the model.

- :func:`build_train_dup_signatures` loads a lightweight
  (``MD``, ``GR``, ``X``, ``Y``, ``TVT_input``, ``TVT``) cache for every
  **train** well once, regardless of ``ROGII_SMOKE`` (a hidden test well's
  twin can be any of the 773 train wells, not just the smoke-restricted
  subset used to train the LGBM ensemble).
- :func:`find_duplicate_twin` searches that cache for a train well whose MD
  range overlaps a given test well by >=500 rows AND passes three strict
  gates (known-zone ``TVT_input`` agreement, ``GR`` agreement, ``X``/``Y``
  agreement); it never raises (any internal failure returns ``None``).
- Toggle: ``ROGII_DUP`` env var, default ``"1"`` (on); ``ROGII_DUP=0``
  disables the overlay entirely -- the rest of this file's behaviour and
  ``submission.csv`` output are then byte-identical to running
  ``stack_v2_blend_submission.py`` under the same conditions.
- The existing sub-v8 pipeline below is **never skipped**: every train/test
  well still goes through Phase 1-6 unchanged. The overlay only overwrites
  individual evaluation-zone rows in the final ``out`` DataFrame, and only
  where a matched twin's ``TVT`` is finite for that row -- unmatched wells
  and unmatched rows keep the pipeline's own prediction untouched. Any
  overlay failure degrades to the unmodified pipeline output with a
  ``[DUP][WARN]`` log line, never aborting the run.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# Configuration (identical to scripts/run_stack_v2_cv.py "all-in" row +
# scripts/build_tracker_cache.py + scripts/build_spatial_cache.py's call
# configuration, so this from-scratch kernel pass reproduces the same
# features that measured pooled OOF RMSE 9.2309).
# --------------------------------------------------------------------------- #

KAGGLE_INPUT = Path("/kaggle/input/rogii-wellbore-geology-prediction")

SEED = 42

# PF tracker (src/rogii/registration/particle.py defaults == build_tracker_cache.py's
# bare `track_pf_multi(h, tw)` call).
N_PARTICLES = 512
N_SEEDS = 8
PF_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)

# Beam tracker: build_tracker_cache.py calls track_beam_multi(h, tw) with no
# override -> DEFAULT_BEAM_CONFIGS (3-config move_penalty x mismatch_scale
# ensemble at products 2400/3000/3750, see registration/beam.py).

# PF-residual target / blend weight (scripts/run_stack_cv.py (b), reused by v2).
PF_BLEND_W = 0.7

# Spatial-prior predictor (scripts/build_spatial_cache.py's STRIDE/K_NEIGHBORS/METHOD,
# the exact config that produced data/processed/spatial_cache/*.npz).
SPATIAL_STRIDE = 10
SPATIAL_K = 10
SPATIAL_METHOD = "idw"

# PF-BMA tracker config (kernel_bank_builder/bank_builder.py Phase B).
# *** UNUSED in sub-v8 *** -- track_pf_bma is retained below as library code
# (see module docstring) but not called; these constants stay with it.
PFBMA_N_PARTICLES = 512
PFBMA_N_SEEDS = 12
PFBMA_GR_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)
PFBMA_BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
PFBMA_BMA_SCALE = 3.0
PFBMA_INIT_POS_STD = 2.0  # v2 bank behaviour; the stack's own PF stays at 0.3

# Beam-grid tracker config (bank_builder.py Phase C's "mid_cal_sm2_mm3").
# *** UNUSED in sub-v8 *** -- track_beam_grid is retained below as library
# code but not called; these constants stay with it.
BEAM_GRID_LABEL = "mid_cal_sm2_mm3"
BEAM_GRID_MOVE_PENALTY = 20.0
BEAM_GRID_MISMATCH_SCALE = 150.0
BEAM_GRID_MAX_MOVE_PER_ROW = 3
BEAM_GRID_GR_CALIBRATION = True
BEAM_GRID_SMOOTH_RADIUS = 2

# sub-v8 = stack v2.6 (R13, scripts/run_stack_v26_cv.py): 59-column feature
# set = stack v2's 56 + H3's 3 raw-data columns, exact order of that script's
# well matrix (p1 + tracker + well-agg + spatial + H3). Pooled OOF RMSE
# 9.1456 with the 5-seed LGBM mean (ledger commit baf82a7). The bank-candidate
# feature family (R7/G1/G2, CV 8.8573) is deliberately NOT shipped here --
# its CV->LB transfer is uncalibrated (transfer-safe positioning; see the
# module docstring). H3 needs no bank npz: all 3 columns derive from MD /
# known-zone TVT_input / Z directly.
H3_FEATURE_COLUMNS: tuple[str, ...] = (
    "known_linear_trend_extrap_minus_anchor",
    "known_tvt_z_identity_resid_std",
    "zone_len_known_len_ratio",
)
_H3_TREND_TAIL = 200  # matches rogii.features._KNOWN_TAIL_LONG (and _KNOWN_TAIL_LONG above)
N_FEATURES_EXPECTED = 59  # 36 P1 + 8 tracker + 8 well-agg + 4 spatial + 3 H3

# LGBM: identical to scripts/run_stack_v23/v26_cv.py's LGB_PARAMS_BASE
# (min_child_samples=50 -- R13 measured mcs=200 as no better). sub-v8 trains
# ENSEMBLE_SEEDS members (random_state is the ONLY thing that varies;
# LightGBM derives bagging_seed/feature_fraction_seed/data_random_seed from
# it) and averages their test predictions -- the R9 recipe, kept by R13. The
# ES-holdout well split stays fixed at SEED=42 for every member.
ENSEMBLE_SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404)
LGB_PARAMS_BASE: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50


def _lgb_params_for_seed(seed: int) -> dict[str, object]:
    """R9's single-knob member configuration (run_stack_v23_cv.py::_lgb_params_for_seed)."""
    return {**LGB_PARAMS_BASE, "random_state": seed}


ES_FRAC = 0.15  # fraction of train wells carved out as an ES-holdout (well-level)

PROGRESS_EVERY = 50


def _smoke_n() -> int | None:
    """``ROGII_SMOKE=<N>`` env var: restrict training to the first N train wells."""
    raw = os.environ.get("ROGII_SMOKE")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _dup_enabled() -> bool:
    """``ROGII_DUP`` env var toggle for the duplicate-twin overlay (default ON).

    ``ROGII_DUP=0`` disables the overlay entirely, in which case this file's
    behaviour and output are byte-identical to the unmodified sub-v8 kernel.
    """
    return os.environ.get("ROGII_DUP", "1") != "0"


# --------------------------------------------------------------------------- #
# Data root / I-O discovery (ported from notebooks/submission/kernel_pf, the
# proven LB 10.559 kernel's discovery strategy).
# --------------------------------------------------------------------------- #


def find_data_root() -> Path:
    """Locate the directory containing ``train/``, ``test/``, ``sample_submission.csv``.

    On Kaggle the competition dataset is not reliably mounted at
    ``/kaggle/input/{competition-slug}`` -- observed in practice to 404
    there, while proven public kernels for this competition locate it via
    ``Path('/kaggle/input').rglob('sample_submission.csv')`` instead. So: try
    the conventional path first, then fall back to an ``/kaggle/input``-wide
    rglob for ``sample_submission.csv`` before giving up.
    """
    candidates = [KAGGLE_INPUT]
    try:
        here = Path(__file__).resolve()
        # Locally this file sits at notebooks/submission/kernel_stack_v2/<name>.py,
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


def output_path(root: Path) -> Path:
    """Kaggle expects ``submission.csv`` in the working directory; keep local runs tidy.

    The local filename is distinct from ``kernel_stack_v2``'s
    ``submission_stack_v2.csv`` so side-by-side local validation runs of the
    two kernels never overwrite each other's output.
    """
    if str(root).startswith("/kaggle"):
        return Path("submission.csv")
    try:
        repo_root = Path(__file__).resolve().parents[3]
    except (NameError, IndexError):
        repo_root = Path.cwd()
    return repo_root / "outputs" / "submission_stack_v2_blend.csv"


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
# Ported from src/rogii/features.py (P1 Layer2 within-well features, 36 cols)
# --------------------------------------------------------------------------- #

_GR_ROLL_WINDOWS = (11, 51, 151)
_TRAJ_ROLL_WINDOW = 51
_KNOWN_TAIL_LONG = 200
_KNOWN_TAIL_SHORT = 50
_MIN_SLOPE_ROWS = 2
_MIN_SLOPE_X_STD = 1e-9


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < _MIN_SLOPE_ROWS:
        return 0.0
    xs, ys = x[mask], y[mask]
    if np.std(xs) < _MIN_SLOPE_X_STD:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def _stat_block(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"min": 0.0, "max": 0.0, "range": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "range": float(valid.max() - valid.min()),
        "mean": float(valid.mean()),
        "std": float(valid.std()) if valid.size >= 2 else 0.0,
    }


def _typewell_gr_at(tw: pd.DataFrame, tvt_query: float) -> float:
    if not {"TVT", "GR"}.issubset(tw.columns) or tw.empty or not np.isfinite(tvt_query):
        return 0.0
    lookup = tw_gr_lookup(tw)
    return float(lookup(np.array([tvt_query]))[0])


def build_features(h: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    """Layer2 feature matrix for the eval-zone rows of ``h`` (P1, 36 columns).

    Never reads ``h["TVT"]`` or any train-only formation column. Only
    ``MD, X, Y, Z, GR, TVT_input`` (all present in the test schema).
    """
    mask = eval_mask(h)
    idx = np.where(mask)[0]
    n_eval = int(idx.size)
    known_mask = ~mask
    n_known = int(known_mask.sum())

    md = h["MD"].to_numpy(dtype=float)
    x = h["X"].to_numpy(dtype=float)
    y = h["Y"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    gr = h["GR"].to_numpy(dtype=float) if "GR" in h.columns else np.full(len(h), np.nan)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    known_md = md[known_mask]
    known_x = x[known_mask]
    known_y = y[known_mask]
    known_z = z[known_mask]
    known_gr = gr[known_mask]
    known_tvt = tvt_input[known_mask]

    anchor_tvt = float(known_tvt[-1]) if n_known > 0 else float("nan")
    ps_md = float(known_md[-1]) if n_known > 0 else float(md[0]) if len(md) else 0.0
    ps_x = float(known_x[-1]) if n_known > 0 else float(x[0]) if len(x) else 0.0
    ps_y = float(known_y[-1]) if n_known > 0 else float(y[0]) if len(y) else 0.0
    ps_z = float(known_z[-1]) if n_known > 0 else float(z[0]) if len(z) else 0.0
    ps_gr = float(known_gr[-1]) if n_known > 0 and np.isfinite(known_gr[-1]) else float("nan")

    slope_md_all = _robust_slope(known_md, known_tvt)
    tail200 = min(_KNOWN_TAIL_LONG, n_known)
    slope_md_last200 = _robust_slope(known_md[-tail200:], known_tvt[-tail200:]) if tail200 else 0.0
    tail50 = min(_KNOWN_TAIL_SHORT, n_known)
    slope_md_last50 = _robust_slope(known_md[-tail50:], known_tvt[-tail50:]) if tail50 else 0.0
    slope_z_all = _robust_slope(known_z, known_tvt)
    tvt_stats = _stat_block(known_tvt)

    dmd = np.diff(md, prepend=md[0] if len(md) else 0.0)
    dmd_safe = np.where(np.abs(dmd) < 1e-9, np.nan, dmd)
    dz_dmd = np.diff(z, prepend=z[0] if len(z) else 0.0) / dmd_safe
    dx_dmd = np.diff(x, prepend=x[0] if len(x) else 0.0) / dmd_safe
    dy_dmd = np.diff(y, prepend=y[0] if len(y) else 0.0) / dmd_safe

    def _roll_mean(a: np.ndarray) -> np.ndarray:
        return pd.Series(a).rolling(_TRAJ_ROLL_WINDOW, min_periods=1, center=True).mean().to_numpy()

    dz_roll = _roll_mean(dz_dmd)
    dx_roll = _roll_mean(dx_dmd)
    dy_roll = _roll_mean(dy_dmd)

    known_tail_n = min(_TRAJ_ROLL_WINDOW, n_known)
    if known_tail_n > 0:
        dz_dmd_known_tail_mean = float(np.nanmean(dz_dmd[known_mask][-known_tail_n:]))
        dx_dmd_known_tail_mean = float(np.nanmean(dx_dmd[known_mask][-known_tail_n:]))
        dy_dmd_known_tail_mean = float(np.nanmean(dy_dmd[known_mask][-known_tail_n:]))
    else:
        dz_dmd_known_tail_mean = 0.0
        dx_dmd_known_tail_mean = 0.0
        dy_dmd_known_tail_mean = 0.0
    dz_dmd_known_tail_mean = dz_dmd_known_tail_mean if np.isfinite(dz_dmd_known_tail_mean) else 0.0
    dx_dmd_known_tail_mean = dx_dmd_known_tail_mean if np.isfinite(dx_dmd_known_tail_mean) else 0.0
    dy_dmd_known_tail_mean = dy_dmd_known_tail_mean if np.isfinite(dy_dmd_known_tail_mean) else 0.0

    gr_series = pd.Series(gr)
    roll_feats: dict[str, np.ndarray] = {}
    for w in _GR_ROLL_WINDOWS:
        roll = gr_series.rolling(w, min_periods=1, center=True)
        roll_feats[f"gr_roll_mean_{w}"] = roll.mean().to_numpy()
        roll_feats[f"gr_roll_std_{w}"] = roll.std().to_numpy()
    gr_diff1 = gr_series.diff(1).to_numpy()
    gr_diff10 = gr_series.diff(10).to_numpy()

    twgr_anchor = _typewell_gr_at(tw, anchor_tvt)
    if "GR" in h.columns:
        gain, offset = fit_affine_gr(h, tw)
    else:
        gain, offset = 1.0, 0.0
    twgr_anchor_calibrated = float(apply_affine(np.array([twgr_anchor]), gain, offset)[0])
    known_tail_gr_minus_twgr_anchor = ps_gr - twgr_anchor if np.isfinite(ps_gr) else 0.0

    n_rows_total = len(h)
    row_frac = idx / max(n_rows_total - 1, 1)
    md_from_ps = md[idx] - ps_md
    dx_ps = x[idx] - ps_x
    dy_ps = y[idx] - ps_y
    dz_ps = z[idx] - ps_z
    dist3d_ps = np.sqrt(dx_ps**2 + dy_ps**2 + dz_ps**2)

    gr_eval = gr[idx]
    gr_missing = (~np.isfinite(gr_eval)).astype(np.float64)
    gr_minus_twgr_anchor_raw = gr_eval - twgr_anchor
    gr_minus_twgr_anchor_calibrated = gr_eval - twgr_anchor_calibrated

    feats: dict[str, np.ndarray] = {
        "md_from_ps": md_from_ps,
        "row_frac": row_frac,
        "dx_from_ps": dx_ps,
        "dy_from_ps": dy_ps,
        "dz_from_ps": dz_ps,
        "dist3d_from_ps": dist3d_ps,
        "dz_dmd_roll51": dz_roll[idx],
        "dx_dmd_roll51": dx_roll[idx],
        "dy_dmd_roll51": dy_roll[idx],
        "dz_dmd_known_tail_mean": np.full(n_eval, dz_dmd_known_tail_mean),
        "dx_dmd_known_tail_mean": np.full(n_eval, dx_dmd_known_tail_mean),
        "dy_dmd_known_tail_mean": np.full(n_eval, dy_dmd_known_tail_mean),
        "gr": gr_eval,
        "gr_missing": gr_missing,
        "gr_diff_1": gr_diff1[idx],
        "gr_diff_10": gr_diff10[idx],
        "gr_minus_twgr_anchor_raw": gr_minus_twgr_anchor_raw,
        "gr_minus_twgr_anchor_calibrated": gr_minus_twgr_anchor_calibrated,
        "slope_md_all": np.full(n_eval, slope_md_all),
        "slope_md_last200": np.full(n_eval, slope_md_last200),
        "slope_md_last50": np.full(n_eval, slope_md_last50),
        "slope_z_all": np.full(n_eval, slope_z_all),
        "known_tvt_min": np.full(n_eval, tvt_stats["min"]),
        "known_tvt_max": np.full(n_eval, tvt_stats["max"]),
        "known_tvt_range": np.full(n_eval, tvt_stats["range"]),
        "known_tvt_mean": np.full(n_eval, tvt_stats["mean"]),
        "known_tvt_std": np.full(n_eval, tvt_stats["std"]),
        "n_known_rows": np.full(n_eval, float(n_known)),
        "anchor_tvt": np.full(n_eval, anchor_tvt if np.isfinite(anchor_tvt) else 0.0),
        "known_tail_gr_minus_twgr_anchor": np.full(n_eval, known_tail_gr_minus_twgr_anchor),
    }
    for name, values in roll_feats.items():
        feats[name] = values[idx]

    out = pd.DataFrame(feats)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Ported from src/rogii/stack_features.py (tracker / well-agg / spatial blocks)
# --------------------------------------------------------------------------- #

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

SPATIAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "spatial_d",
    "spatial_prefix_rmse",
    "spatial_nn_dist_median",
    "spatial_gated_d",
)
SPATIAL_GATE_THETA = 2.0


def build_tracker_features(
    anchor: float,
    pf_tvt: np.ndarray,
    pf_std: np.ndarray,
    beam_tvt: np.ndarray,
    beam_margin: np.ndarray,
) -> pd.DataFrame:
    """Per-row tracker feature block (8 columns), aligned to ``eval_mask(h)`` order."""
    pf_tvt = np.asarray(pf_tvt, dtype=np.float64)
    pf_std = np.asarray(pf_std, dtype=np.float64)
    beam_tvt = np.asarray(beam_tvt, dtype=np.float64)
    beam_margin = np.asarray(beam_margin, dtype=np.float64)

    n = pf_tvt.size
    anchor_f = float(anchor) if np.isfinite(anchor) else 0.0

    if n == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in FEATURE_COLUMNS})

    pf_d = pf_tvt - anchor_f
    beam_d = beam_tvt - anchor_f

    pf_std_is_inf = (~np.isfinite(pf_std)).astype(np.float64)
    pf_d_damp = pf_d / (1.0 + pf_std)

    pf_beam_abs_diff = np.abs(pf_d - beam_d)

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


def build_well_aggregate_features(
    anchor: float,
    pf_tvt: np.ndarray,
    pf_std: np.ndarray,
    beam_tvt: np.ndarray,
    beam_margin: np.ndarray,
    gr_nan_frac: float,
) -> pd.DataFrame:
    """Well-level scalar tracker/quality summaries (8 columns), broadcast per row."""
    pf_tvt = np.asarray(pf_tvt, dtype=np.float64)
    pf_std = np.asarray(pf_std, dtype=np.float64)
    beam_tvt = np.asarray(beam_tvt, dtype=np.float64)
    beam_margin = np.asarray(beam_margin, dtype=np.float64)
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


def build_spatial_features(
    anchor: float,
    spatial_tvt: np.ndarray,
    prefix_rmse: float,
    nn_dist_median: float,
) -> pd.DataFrame:
    """Per-row spatial-prior feature block (4 columns)."""
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


# --------------------------------------------------------------------------- #
# Ported from src/rogii/registration/particle.py (PF tracker). Adapted from
# the public Kaggle notebook lightningv08/lb-7-776-rogii-ridge-sp (functions
# _pf_ancc / run_pf_ancc), reused under the competition's code-sharing terms.
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
    scales: tuple[float, ...] = PF_SCALES,
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


def track_pf_multi(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_seeds: int = 8,
    **kwargs: object,
) -> PFResult:
    """Average ``n_seeds`` independent :func:`track_pf` runs for variance reduction."""
    kwargs.pop("seed", None)
    results = [track_pf(h, tw, seed=seed, **kwargs) for seed in range(max(int(n_seeds), 1))]

    tvts = np.stack([r.tvt for r in results], axis=0)
    stds = np.stack([r.std for r in results], axis=0)

    mean_tvt = tvts.mean(axis=0)
    with np.errstate(invalid="ignore"):
        combined_var = np.mean(stds**2, axis=0) + np.var(tvts, axis=0)
    combined_std = np.sqrt(combined_var)

    return PFResult(tvt=mean_tvt, std=combined_std, n_updates=results[0].n_updates)


# --------------------------------------------------------------------------- #
# Ported from notebooks/submission/kernel_bank_builder/bank_builder.py's Phase
# B (PF-BMA: likelihood-weighted Bayesian Model Averaging over independent PF
# seed-runs via a softmax of each seed's total log-likelihood). Reimplements
# the method of the public Kaggle notebook
# bernubritz/rogii-lb7295-public-rebuild; reuses this file's own
# ``_systematic_resample``/``_assign_particle_scales``/``_calibrate_rate``/
# ``tw_gr_lookup``/``fit_affine_gr``/``apply_affine`` helpers (identical PF
# per-step dynamics to :func:`_track_pf_impl` above -- only the multi-seed
# log-likelihood bookkeeping and the softmax consensus step are new).
# --------------------------------------------------------------------------- #

_MIN_AVG_LIKELIHOOD = 1e-300


@dataclass
class PFBMAResult:
    tvt_by_scale: dict[float, np.ndarray]
    std_by_scale: dict[float, np.ndarray]
    log_lik: np.ndarray
    n_updates: int


def _flat_fallback_bma(
    n: int, anchor: float, bma_scales: tuple[float, ...], n_seeds: int
) -> PFBMAResult:
    flat = np.full(n, anchor, dtype=float)
    inf_arr = np.full(n, np.inf, dtype=float)
    return PFBMAResult(
        tvt_by_scale={scale: flat.copy() for scale in bma_scales},
        std_by_scale={scale: inf_arr.copy() for scale in bma_scales},
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
    n_particles: int = PFBMA_N_PARTICLES,
    n_seeds: int = PFBMA_N_SEEDS,
    bma_scales: tuple[float, ...] = (PFBMA_BMA_SCALE,),
    gr_scales: tuple[float, ...] = PFBMA_GR_SCALES,
    seed_base: int = 0,
    resample_ess: float = 0.5,
    init_pos_std: float = _INIT_POS_STD,
) -> PFBMAResult:
    """Likelihood-weighted PF-BMA: seed-softmax consensus over independent PF runs.

    ``init_pos_std`` (default ``_INIT_POS_STD = 0.3``, the v1 bank's
    bit-identical behaviour) is the standard deviation (ft, TVT+Z frame) of
    every seed's initial particle-position spread around the anchor --
    mirrors ``bank_builder.py``'s v2 wiring (commit 9152035). The sub-v7
    bank-delta feature inputs are computed at ``PFBMA_INIT_POS_STD = 2.0``,
    reproducing ``outputs/path_bank_pfbma_v2.npz``.

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
# dwt-based, reused under the competition's code-sharing terms.
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


def _beam_empty_result() -> BeamResult:
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
    if hi <= lo:
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
        return _beam_empty_result()

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
        try:
            n = int(eval_mask(h).sum())
        except Exception:
            n = 0
        try:
            anchor = last_known_tvt(h)
        except Exception:
            anchor = 0.0
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

    Defaults to :data:`DEFAULT_BEAM_CONFIGS`. Never raises: falls back to a
    flat anchor prediction, same as :func:`track_beam`.
    """
    try:
        resolved = configs if configs is not None else DEFAULT_BEAM_CONFIGS
        return _track_beam_multi_impl(h, tw, resolved)
    except Exception:
        try:
            n = int(eval_mask(h).sum())
        except Exception:
            n = 0
        try:
            anchor = last_known_tvt(h)
        except Exception:
            anchor = 0.0
        return _beam_flat_fallback(n, anchor)


# --------------------------------------------------------------------------- #
# Ported from notebooks/submission/kernel_bank_builder/bank_builder.py's Phase
# C (parameterized beam-DP grid tracker; reuses this file's own
# ``_build_grid``/``_forward_viterbi``/``tw_gr_lookup``/``fit_affine_gr``/
# ``apply_affine`` helpers -- identical DP kernel to :func:`_track_beam_impl`
# above -- only the ``gr_calibration`` on/off switch and the optional
# rolling-mean GR smoothing (:func:`_smooth_gr`) are new).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BeamGridConfig:
    label: str
    move_penalty: float
    mismatch_scale: float
    max_move_per_row: int = _DEFAULT_MAX_MOVE_PER_ROW
    gr_calibration: bool = True
    smooth_radius: int = 0
    grid_step_ft: float = _DEFAULT_GRID_STEP_FT
    grid_range_ft: float = _DEFAULT_GRID_RANGE_FT


# bank_builder.py's CONFIG_GRID entry "mid_cal_sm2_mm3" (index 4 of 10),
# verified bit-for-bit against outputs/path_bank_beam.npz's config_labels /
# product_per_config (3000.0) / max_move_per_config (3) /
# gr_calibration_per_config (1) / smooth_radius_per_config (2).
BEAM_MID_CAL_SM2_MM3 = BeamGridConfig(
    label=BEAM_GRID_LABEL,
    move_penalty=BEAM_GRID_MOVE_PENALTY,
    mismatch_scale=BEAM_GRID_MISMATCH_SCALE,
    max_move_per_row=BEAM_GRID_MAX_MOVE_PER_ROW,
    gr_calibration=BEAM_GRID_GR_CALIBRATION,
    smooth_radius=BEAM_GRID_SMOOTH_RADIUS,
)


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
        return _beam_empty_result()

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


# --------------------------------------------------------------------------- #
# Ported from src/rogii/spatial.py (spatial-prior predictor). Host forum idea
# (Igor Kuvaev, forum 708167): formation-top depth is spatially smooth, so it
# can be interpolated from nearby train wells' formation-depth columns. See
# module docstring in src/rogii/spatial.py for the full leak-safety analysis
# reproduced in this file's own module docstring above.
# --------------------------------------------------------------------------- #

FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
_SPATIAL_PREFIX_WINDOW = 500
_IDW_EPS = 1e-3
_COMPOSITE_EPS = 1e-6
_PLANE_RIDGE = 1e-4
_QUERY_WORKERS = -1


@dataclass(frozen=True)
class SurfaceBank:
    formations: tuple[str, ...]
    trees: dict[str, cKDTree] = field(default_factory=dict)
    depths: dict[str, np.ndarray] = field(default_factory=dict)
    well_codes: dict[str, np.ndarray] = field(default_factory=dict)
    well_to_code: dict[str, int] = field(default_factory=dict)


@dataclass
class SpatialResult:
    tvt: np.ndarray
    prefix_rmse: float
    n_formations_used: int
    tvt_composite: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        if self.tvt_composite.size == 0:
            self.tvt_composite = self.tvt


def build_surface_bank(
    wells: list[str],
    split: str,
    root: Path,
    stride: int = SPATIAL_STRIDE,
    formations: tuple[str, ...] = FORMATIONS,
) -> SurfaceBank:
    """Collect stride-subsampled ``(X, Y, depth)`` samples for each formation.

    Only ever reads *train*-split wells (the only schema with formation-depth
    columns) -- called with ``split="train"`` and the full train well list, so
    this bank never contains a hidden test well's own data.
    """
    well_to_code = {well: i for i, well in enumerate(wells)}
    xs: dict[str, list[np.ndarray]] = {f: [] for f in formations}
    ys: dict[str, list[np.ndarray]] = {f: [] for f in formations}
    ds: dict[str, list[np.ndarray]] = {f: [] for f in formations}
    codes: dict[str, list[np.ndarray]] = {f: [] for f in formations}

    for well in wells:
        code = well_to_code[well]
        try:
            h = load_horizontal(well, split, root)
        except Exception:
            continue

        sub = h.iloc[::stride]
        if "X" not in sub.columns or "Y" not in sub.columns:
            continue
        x = sub["X"].to_numpy(dtype=float)
        y = sub["Y"].to_numpy(dtype=float)
        finite_xy = np.isfinite(x) & np.isfinite(y)

        for f in formations:
            if f not in sub.columns:
                continue
            depth = sub[f].to_numpy(dtype=float)
            valid = finite_xy & np.isfinite(depth)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            xs[f].append(x[valid])
            ys[f].append(y[valid])
            ds[f].append(depth[valid])
            codes[f].append(np.full(n_valid, code, dtype=np.int32))

    trees: dict[str, cKDTree] = {}
    depths: dict[str, np.ndarray] = {}
    well_codes: dict[str, np.ndarray] = {}
    used_formations: list[str] = []

    for f in formations:
        if not xs[f]:
            continue
        xf = np.concatenate(xs[f])
        yf = np.concatenate(ys[f])
        trees[f] = cKDTree(np.column_stack([xf, yf]))
        depths[f] = np.concatenate(ds[f])
        well_codes[f] = np.concatenate(codes[f])
        used_formations.append(f)

    return SurfaceBank(
        formations=tuple(used_formations),
        trees=trees,
        depths=depths,
        well_codes=well_codes,
        well_to_code=well_to_code,
    )


def _knn_weights(
    tree: cKDTree,
    coords: np.ndarray,
    depths: np.ndarray,
    codes: np.ndarray,
    exclude_code: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_total = depths.size
    n_self = int(np.sum(codes == exclude_code))
    k_query = min(k + n_self, n_total)
    dist, idx = tree.query(coords, k=k_query, workers=_QUERY_WORKERS)
    if k_query == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    valid = codes[idx] != exclude_code

    rank_within_valid = np.cumsum(valid, axis=1) - 1
    keep = valid & (rank_within_valid < k)

    weights = np.where(keep, 1.0 / (dist + _IDW_EPS), 0.0)
    return idx, weights


def _knn_idw_depth(
    tree: cKDTree,
    coords: np.ndarray,
    depths: np.ndarray,
    codes: np.ndarray,
    exclude_code: int,
    k: int,
) -> np.ndarray:
    n_query = coords.shape[0]
    if depths.size == 0:
        return np.full(n_query, np.nan)

    idx, weights = _knn_weights(tree, coords, depths, codes, exclude_code, k)
    weight_sum = weights.sum(axis=1)
    weighted_depth = (weights * depths[idx]).sum(axis=1)

    safe_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    return np.where(weight_sum > 0, weighted_depth / safe_sum, np.nan)


def _knn_plane_depth(
    tree: cKDTree,
    coords: np.ndarray,
    depths: np.ndarray,
    codes: np.ndarray,
    exclude_code: int,
    k: int,
) -> np.ndarray:
    n_query = coords.shape[0]
    if depths.size == 0:
        return np.full(n_query, np.nan)

    idx, weights = _knn_weights(tree, coords, depths, codes, exclude_code, k)
    bank_xy = np.asarray(tree.data)

    dx = bank_xy[idx, 0] - coords[:, 0:1]
    dy = bank_xy[idx, 1] - coords[:, 1:2]
    d = depths[idx]

    w_sum = weights.sum(axis=1)
    empty = w_sum <= 0

    features = np.stack([np.ones_like(dx), dx, dy], axis=-1)
    a_mat = np.einsum("qk,qki,qkj->qij", weights, features, features)
    b_vec = np.einsum("qk,qk,qki->qi", weights, d, features)

    r2 = (weights * (dx**2 + dy**2)).sum(axis=1)
    lam = _PLANE_RIDGE * np.maximum(r2, 1.0)
    a_mat[:, 1, 1] += lam
    a_mat[:, 2, 2] += lam
    a_mat[empty] = np.eye(3)

    try:
        beta = np.linalg.solve(a_mat, b_vec[:, :, None])[:, :, 0]
        result = beta[:, 0]
    except np.linalg.LinAlgError:
        return _knn_idw_depth(tree, coords, depths, codes, exclude_code, k)

    result = np.where(empty | ~np.isfinite(result), np.nan, result)
    return result


def _spatial_flat_fallback(h: pd.DataFrame) -> SpatialResult:
    try:
        n = int(eval_mask(h).sum())
    except Exception:
        n = 0
    try:
        anchor = last_known_tvt(h)
    except Exception:
        anchor = 0.0
    return SpatialResult(
        tvt=np.full(n, anchor, dtype=float), prefix_rmse=float("inf"), n_formations_used=0
    )


_INTERPOLATORS = {"idw": _knn_idw_depth, "plane": _knn_plane_depth}


def _predict_spatial_impl(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    k: int,
    method: str,
) -> SpatialResult:
    known = h["TVT_input"].notna().to_numpy()
    eval_m = eval_mask(h)
    eval_idx = np.flatnonzero(eval_m)
    known_idx = np.flatnonzero(known)

    if eval_idx.size == 0 or known_idx.size == 0:
        raise ValueError("empty eval zone or no known-zone anchor")

    prefix_idx = known_idx[-min(_SPATIAL_PREFIX_WINDOW, known_idx.size) :]

    x = h["X"].to_numpy(dtype=float)
    y = h["Y"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    active_idx = np.concatenate([prefix_idx, eval_idx])
    coords = np.column_stack([x[active_idx], y[active_idx]])
    n_prefix = prefix_idx.size

    exclude_code = bank.well_to_code.get(exclude_well, -1)
    fallback_anchor = float(tvt_input[known_idx[-1]])

    interpolate = _INTERPOLATORS[method]
    eval_z = z[eval_idx]
    candidates: list[np.ndarray] = []
    candidate_rmses: list[float] = []

    for f in bank.formations:
        depth_interp = interpolate(
            bank.trees[f], coords, bank.depths[f], bank.well_codes[f], exclude_code, k
        )
        prefix_depth = depth_interp[:n_prefix]
        eval_depth = depth_interp[n_prefix:]

        valid_prefix = np.isfinite(prefix_depth)
        if not np.any(valid_prefix):
            continue

        prefix_z = z[prefix_idx][valid_prefix]
        prefix_tvt = tvt_input[prefix_idx][valid_prefix]
        offset = float(np.median(prefix_tvt + prefix_z - prefix_depth[valid_prefix]))

        prefix_pred = -prefix_z + prefix_depth[valid_prefix] + offset
        resid = prefix_tvt - prefix_pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        if not np.isfinite(rmse):
            continue

        eval_tvt = -eval_z + eval_depth + offset
        valid_eval = np.isfinite(eval_tvt)
        if not np.all(valid_eval):
            eval_tvt = np.where(valid_eval, eval_tvt, fallback_anchor)
        candidates.append(eval_tvt)
        candidate_rmses.append(rmse)

    if not candidates:
        raise ValueError("no formation produced a usable candidate")

    rmses = np.array(candidate_rmses)
    best = int(np.argmin(rmses))

    inv_weights = 1.0 / (rmses**2 + _COMPOSITE_EPS)
    composite = np.average(np.stack(candidates), axis=0, weights=inv_weights)

    return SpatialResult(
        tvt=candidates[best],
        prefix_rmse=float(rmses[best]),
        n_formations_used=len(candidates),
        tvt_composite=composite,
    )


def predict_spatial(
    h: pd.DataFrame,
    bank: SurfaceBank,
    exclude_well: str,
    k: int = SPATIAL_K,
    method: str = SPATIAL_METHOD,
) -> SpatialResult:
    """Spatial-prior TVT prediction for the evaluation zone of ``h``.

    Never raises: any failure falls back to a flat carry-last prediction with
    ``prefix_rmse = inf``.
    """
    try:
        return _predict_spatial_impl(h, bank, exclude_well, k, method)
    except Exception:
        return _spatial_flat_fallback(h)


def nn_distance_per_row(h: pd.DataFrame, bank: SurfaceBank, well: str) -> np.ndarray:
    """Per-row nearest non-self bank-sample distance for every eval-zone row.

    Ported from ``scripts/build_spatial_cache.py::_nn_distance_per_row`` --
    feeds ``spatial_nn_dist_median`` (one of the 4 spatial feature columns),
    computed against ``bank.formations[0]`` as a single representative
    spatial-density diagnostic (not per-formation).
    """
    if not bank.formations:
        return np.array([], dtype=float)
    formation = bank.formations[0]
    tree = bank.trees[formation]
    codes = bank.well_codes[formation]
    exclude_code = bank.well_to_code.get(well, -1)

    ez = eval_mask(h)
    x = h["X"].to_numpy(dtype=float)[ez]
    y = h["Y"].to_numpy(dtype=float)[ez]
    if x.size == 0:
        return np.array([], dtype=float)

    coords = np.column_stack([x, y])
    n_self = int(np.sum(codes == exclude_code))
    k = min(1 + n_self, codes.size)
    dist, idx = tree.query(coords, k=k, workers=-1)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    valid = codes[idx] != exclude_code
    first_valid = np.where(valid.any(axis=1), valid.argmax(axis=1), -1)
    rows = np.arange(coords.shape[0])
    return np.where(first_valid >= 0, dist[rows, np.maximum(first_valid, 0)], np.nan)


# --------------------------------------------------------------------------- #
# Orchestration: per-well tracker+spatial computation, feature assembly,
# ES-holdout split, LGBM training, and the sample_submission-driven predict
# loop (structure ported from notebooks/submission/kernel_pf's proven
# build_predictions/validate pattern).
# --------------------------------------------------------------------------- #


@dataclass
class WellTrackerSpatial:
    """Cached PF + beam + spatial-prior results for one well (train or test)."""

    anchor: float
    pf_tvt: np.ndarray
    pf_std: np.ndarray
    beam_tvt: np.ndarray
    beam_margin: np.ndarray
    spatial_tvt: np.ndarray
    prefix_rmse: float
    nn_dist_median: float
    gr_nan_frac: float


def compute_well_tracker_spatial(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    well: str,
    bank: SurfaceBank,
) -> WellTrackerSpatial | None:
    """PF + beam + spatial-prior computation for one well; never raises.

    Each underlying tracker already carries its own internal exception-safe
    fallback (flat anchor, ``std=inf``/``prefix_rmse=inf`` on failure), so a
    well with e.g. a degenerate typewell still yields finite numbers here.
    This wrapper's own try/except only guards the outer scaffolding
    (``last_known_tvt``, GR-NaN fraction, the nn-distance diagnostic) -- if
    even that fails, returns ``None`` so the caller can skip (train) / fall
    back to carry_last (test). sub-v8 does NOT call the retained PF-BMA /
    beam-grid trackers (see module docstring) -- Phase 2 is back to the
    sub-v3 cost profile.
    """
    try:
        anchor = last_known_tvt(h)
        pf = track_pf_multi(h, tw, n_seeds=N_SEEDS, n_particles=N_PARTICLES, scales=PF_SCALES)
        beam = track_beam_multi(h, tw)
        spatial = predict_spatial(h, bank, exclude_well=well, k=SPATIAL_K, method=SPATIAL_METHOD)
        nn_dist = nn_distance_per_row(h, bank, well)
        finite_nn = nn_dist[np.isfinite(nn_dist)] if nn_dist.size else np.array([])
        nn_dist_median = float(np.median(finite_nn)) if finite_nn.size else float("nan")
        gr_nan_frac = float(h["GR"].isna().mean()) if "GR" in h.columns else 1.0
        return WellTrackerSpatial(
            anchor=anchor,
            pf_tvt=pf.tvt,
            pf_std=pf.std,
            beam_tvt=beam.tvt,
            beam_margin=beam.margin,
            spatial_tvt=spatial.tvt,
            prefix_rmse=spatial.prefix_rmse,
            nn_dist_median=nn_dist_median,
            gr_nan_frac=gr_nan_frac,
        )
    except Exception:
        return None


def build_h3_features(h: pd.DataFrame) -> pd.DataFrame:
    """H3: 3 columns from ``MD``, the known-zone ``TVT_input``, and ``Z`` only.

    Byte-copied from ``run_stack_v26_cv.py::build_h3_features`` (itself a
    byte-copy of R12's ``run_stack_v25_cv.py``); only the ``D.eval_mask`` /
    ``_robust_slope`` references are swapped for this file's own identical
    ports. See the module docstring for the 3 column definitions. Never
    raises; one row per eval-zone row (``eval_mask(h)`` order).
    """
    mask = eval_mask(h)
    n_eval = int(mask.sum())
    known_mask = ~mask
    n_known = int(known_mask.sum())
    if n_eval == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in H3_FEATURE_COLUMNS})

    md = h["MD"].to_numpy(dtype=float)
    idx = np.where(mask)[0]

    trend_delta = np.zeros(n_eval, dtype=float)
    identity_resid_std = 0.0

    if n_known > 0:
        known_md = h.loc[known_mask, "MD"].to_numpy(dtype=float)
        known_tvt = h.loc[known_mask, "TVT_input"].to_numpy(dtype=float)
        anchor_md = float(known_md[-1])

        tail_n = min(_H3_TREND_TAIL, n_known)
        tail_md = known_md[-tail_n:]
        tail_tvt = known_tvt[-tail_n:]
        slope = _robust_slope(tail_md, tail_tvt)
        trend_delta = slope * (md[idx] - anchor_md)

        z = h["Z"].to_numpy(dtype=float)
        known_z = z[known_mask]
        identity_val = known_tvt + known_z
        finite = np.isfinite(known_md) & np.isfinite(identity_val)
        if int(finite.sum()) >= 2 and np.std(known_md[finite]) > 1e-9:
            coeffs = np.polyfit(known_md[finite], identity_val[finite], 1)
            fitted = np.polyval(coeffs, known_md[finite])
            identity_resid_std = float(np.std(identity_val[finite] - fitted))

    ratio = float(n_eval) / float(n_known) if n_known > 0 else 0.0

    trend_delta = np.where(np.isfinite(trend_delta), trend_delta, 0.0)
    identity_resid_std = identity_resid_std if np.isfinite(identity_resid_std) else 0.0
    ratio = ratio if np.isfinite(ratio) else 0.0

    out = pd.DataFrame(
        {
            "known_linear_trend_extrap_minus_anchor": trend_delta,
            "known_tvt_z_identity_resid_std": np.full(
                n_eval, identity_resid_std, dtype=np.float64
            ),
            "zone_len_known_len_ratio": np.full(n_eval, ratio, dtype=np.float64),
        }
    )
    out = out[list(H3_FEATURE_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


def build_well_features(h: pd.DataFrame, tw: pd.DataFrame, wts: WellTrackerSpatial) -> pd.DataFrame:
    """Concatenate P1 + tracker + well-agg + spatial + H3 blocks (59 cols).

    Column order mirrors ``run_stack_v26_cv.py``'s well matrix exactly:
    ``p1_cols + tracker + well-agg + spatial + H3``
    (``N_FEATURES_EXPECTED`` = 59).
    """
    p1 = build_features(h, tw)
    trk = build_tracker_features(wts.anchor, wts.pf_tvt, wts.pf_std, wts.beam_tvt, wts.beam_margin)
    wellagg = build_well_aggregate_features(
        wts.anchor, wts.pf_tvt, wts.pf_std, wts.beam_tvt, wts.beam_margin, wts.gr_nan_frac
    )
    spat = build_spatial_features(wts.anchor, wts.spatial_tvt, wts.prefix_rmse, wts.nn_dist_median)
    h3 = build_h3_features(h)
    return pd.concat(
        [
            p1.reset_index(drop=True),
            trk.reset_index(drop=True),
            wellagg.reset_index(drop=True),
            spat.reset_index(drop=True),
            h3.reset_index(drop=True),
        ],
        axis=1,
    )


def split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level split into fit vs. ES-holdout (ported from run_stack_v2_cv.py::_split_fit_es)."""
    ordered = sorted(train_wells)
    random.Random(seed).shuffle(ordered)
    n_es = max(1, round(len(ordered) * es_frac))
    es_wells = sorted(ordered[:n_es])
    fit_wells = sorted(ordered[n_es:])
    return fit_wells, es_wells


def build_train_matrix(
    train_wells: list[str],
    root: Path,
    wts_cache: dict[str, WellTrackerSpatial],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Build the all-in feature matrix + PF-residual target for the usable train wells."""
    feat_parts: list[pd.DataFrame] = []
    y_true_parts: list[np.ndarray] = []
    pf_blend_parts: list[np.ndarray] = []
    well_idx_parts: list[np.ndarray] = []
    used_wells: list[str] = []
    counts = {"no_eval_or_anchor": 0, "tracker_spatial_failed": 0, "length_mismatch": 0}

    for well in train_wells:
        try:
            h = load_horizontal(well, "train", root)
            tw = load_typewell(well, "train", root)
        except Exception:
            counts["no_eval_or_anchor"] += 1
            continue

        mask = eval_mask(h)
        if mask.sum() == 0:
            counts["no_eval_or_anchor"] += 1
            continue
        try:
            last_known_tvt(h)
        except ValueError:
            counts["no_eval_or_anchor"] += 1
            continue

        wts = wts_cache.get(well)
        if wts is None:
            counts["tracker_spatial_failed"] += 1
            continue
        if wts.pf_tvt.size != int(mask.sum()):
            counts["length_mismatch"] += 1
            continue

        feats = build_well_features(h, tw, wts)
        y_true = h["TVT"].to_numpy(dtype=np.float32)[mask].astype(np.float64)  # label only
        pf_blend = wts.anchor + PF_BLEND_W * (wts.pf_tvt - wts.anchor)

        well_i = len(used_wells)
        used_wells.append(well)
        feat_parts.append(feats)
        y_true_parts.append(y_true)
        pf_blend_parts.append(pf_blend)
        well_idx_parts.append(np.full(y_true.shape[0], well_i, dtype=np.int32))

    if not feat_parts:
        raise RuntimeError("no train wells produced usable features; cannot train a model")

    X = pd.concat(feat_parts, ignore_index=True)
    y_true_arr = np.concatenate(y_true_parts)
    pf_blend_arr = np.concatenate(pf_blend_parts)
    well_idx = np.concatenate(well_idx_parts)
    well_to_idx = {w: i for i, w in enumerate(used_wells)}

    print(
        f"[INFO] train feature matrix: {X.shape}, rows={len(y_true_arr):,}, "
        f"wells_used={len(used_wells)}/{len(train_wells)} counts={counts}"
    )
    return X, y_true_arr, pf_blend_arr, well_idx, used_wells, well_to_idx


def predict_well_stack(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    well: str,
    wts_cache: dict[str, WellTrackerSpatial],
    feature_cols: list[str],
    models: list[lgb.LGBMRegressor],
) -> np.ndarray:
    """Stack v2.6 predictor (5-seed LGBM mean) for one well's eval zone; never raises.

    ``pred = pf_blend + mean_over_seeds(model_s.predict(feats))`` -- the R9
    ensemble recipe kept by R13. Falls back to carry_last (this well only)
    on any failure: missing tracker/spatial cache entry, a length mismatch,
    or a non-finite output.
    """
    try:
        anchor_pred = predict_carry_last(h)
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return np.full(n, anchor, dtype=float)

    try:
        mask = eval_mask(h)
        wts = wts_cache.get(well)
        if wts is None:
            raise ValueError("no cached tracker/spatial result for this well")
        n_eval = int(mask.sum())
        if wts.pf_tvt.size != n_eval:
            raise ValueError(f"tracker length mismatch: {wts.pf_tvt.size} != {n_eval}")
        feats = build_well_features(h, tw, wts)[feature_cols]
        pf_blend = wts.anchor + PF_BLEND_W * (wts.pf_tvt - wts.anchor)
        residual = np.mean([m.predict(feats) for m in models], axis=0)
        pred = pf_blend + residual
        if pred.shape != anchor_pred.shape or not np.all(np.isfinite(pred)):
            raise ValueError("stack prediction shape/finite check failed")
        return pred
    except Exception:
        return anchor_pred


# --------------------------------------------------------------------------- #
# sample_submission-driven prediction / validation (ported from
# notebooks/submission/kernel_pf's proven build_predictions/validate).
# --------------------------------------------------------------------------- #


def parse_sample_submission(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path)
    parts = sample["id"].str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2 or parts[1].isna().any():
        raise ValueError("sample_submission ids must be formatted as '{well}_{row_index}'")
    return sample.assign(well=parts[0], row_index=parts[1].astype(int)).reset_index(drop=True)


def build_predictions(
    sample: pd.DataFrame,
    root: Path,
    wts_cache: dict[str, WellTrackerSpatial],
    feature_cols: list[str],
    models: list[lgb.LGBMRegressor],
    t_start: float,
) -> tuple[pd.DataFrame, list[str]]:
    mismatch_warnings: list[str] = []
    tvt = np.full(len(sample), np.nan, dtype=float)

    wells = list(sample.groupby("well", sort=False))
    n_wells = len(wells)

    for i, (well, well_rows) in enumerate(wells):
        try:
            h = load_horizontal(well, "test", root)
            tw = load_typewell(well, "test", root)
            eval_idx = np.where(eval_mask(h))[0]
            preds = predict_well_stack(h, tw, well, wts_cache, feature_cols, models)
        except Exception as exc:  # last-resort: never let a single well abort the run
            print(f"[WARN] {well}: total prediction failure ({exc!r}), fallback to 0.0")
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

        if (i + 1) % max(n_wells // 10, 1) == 0 or (i + 1) == n_wells:
            elapsed = time.time() - t_start
            print(f"[PROGRESS] test predict {i + 1}/{n_wells} wells | elapsed={elapsed:.1f}s")

    out = pd.DataFrame({"id": sample["id"].to_numpy(), "tvt": tvt})
    return out, mismatch_warnings


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


# --------------------------------------------------------------------------- #
# ROGII_DUP: duplicate-twin overlay (new; self-contained, purely additive).
#
# Background (measured, see this repo's memory/ledger, not re-verified here):
# the 3 locally-visible test wells are byte-identical duplicates of train
# wells with the same well ID (same MD range, GR match ratio 1.0000, XY
# diff 0, known-zone TVT_input diff 0.00e+00). Train's own TVT column
# covers 100% of the corresponding test eval zone for those 3 wells. This
# overlay searches for the same pattern among the hidden ~200 test wells;
# any well that does NOT match falls through unchanged to the sub-v8
# prediction above -- this overlay can only ever help or no-op, never hurt,
# by construction (finite overrides only replace rows already produced by
# the untouched pipeline; everything else is left as-is).
# --------------------------------------------------------------------------- #

_DUP_MIN_OVERLAP_ROWS = 500
_DUP_MIN_TVT_INPUT_ROWS = 200
_DUP_TVT_INPUT_TOL = 0.01
_DUP_MIN_GR_ROWS = 500
_DUP_GR_TOL = 0.01
_DUP_GR_MATCH_FRAC = 0.999
_DUP_XY_TOL = 1.0
_DUP_HEAD_N = 200  # size of the head window used for the signature roundsum
_DUP_COLUMNS: tuple[str, ...] = ("MD", "GR", "X", "Y", "TVT_input", "TVT")


@dataclass
class TrainDupRecord:
    """Cached per-train-well signature + full arrays for duplicate-twin matching.

    The 5-field signature (``n_rows``, ``md_first``, ``md_last``,
    ``n_finite_gr``, ``head200_roundsum``) is computed once per train well
    and is cheap to keep around; the full ``md``/``gr``/``x``/``y``/
    ``tvt_input``/``tvt`` arrays are also cached here (loaded once, shared
    across every test well's search) so matching never re-reads a train
    well's CSV from disk.
    """

    well: str
    n_rows: int
    md_first: float
    md_last: float
    n_finite_gr: int
    head200_roundsum: float
    md: np.ndarray
    gr: np.ndarray
    x: np.ndarray
    y: np.ndarray
    tvt_input: np.ndarray
    tvt: np.ndarray


def _dup_head_roundsum(gr: np.ndarray) -> float:
    """Deterministic cheap digest of a well's first ``_DUP_HEAD_N`` finite GR values."""
    finite = gr[np.isfinite(gr)]
    if finite.size == 0:
        return 0.0
    return float(np.round(finite[:_DUP_HEAD_N], 2).sum())


def build_train_dup_signatures(train_wells: list[str], root: Path) -> dict[str, TrainDupRecord]:
    """Load and cache (once) the duplicate-twin matching signature for every train well.

    Called once per submission run, over the FULL train well population
    (independent of ``ROGII_SMOKE``, which only restricts the LGBM training
    subset elsewhere in this file) -- a hidden test well's twin can be any
    of the 773 train wells. A single well whose CSV fails to load is
    skipped (logged) rather than aborting the whole cache build.
    """
    signatures: dict[str, TrainDupRecord] = {}
    for well in train_wells:
        try:
            path = root / "train" / f"{well}__horizontal_well.csv"
            df = pd.read_csv(path, usecols=list(_DUP_COLUMNS))
            md = df["MD"].to_numpy(dtype=float)
            if md.size == 0:
                continue
            gr = df["GR"].to_numpy(dtype=float)
            signatures[well] = TrainDupRecord(
                well=well,
                n_rows=int(md.size),
                md_first=float(md[0]),
                md_last=float(md[-1]),
                n_finite_gr=int(np.isfinite(gr).sum()),
                head200_roundsum=_dup_head_roundsum(gr),
                md=md,
                gr=gr,
                x=df["X"].to_numpy(dtype=float),
                y=df["Y"].to_numpy(dtype=float),
                tvt_input=df["TVT_input"].to_numpy(dtype=float),
                tvt=df["TVT"].to_numpy(dtype=float),
            )
        except Exception as exc:  # never let one bad train well abort the cache build
            print(f"[DUP][WARN] {well}: failed to build train signature ({exc!r})")
            continue
    return signatures


def find_duplicate_twin(
    h_test: pd.DataFrame,
    train_signatures: dict[str, TrainDupRecord],
    root: Path,  # noqa: ARG001 -- kept for API symmetry with build_train_dup_signatures
) -> dict[str, object] | None:
    """Search ``train_signatures`` for a byte-identical twin of a test well ``h_test``.

    Because every well's ``MD`` column is a strict 1ft-step arithmetic
    sequence, a candidate train well corresponds to the test well via a
    single integer row offset ``offset = md_first_test - md_first_train``:
    row ``i`` of the test well lines up with row ``i + offset`` of the
    candidate (whenever both are in range). Candidates are first narrowed by
    the MD-overlap row count implied by that offset (>= 500 rows, no exact
    row-count / MD-range match required -- this deliberately also catches
    truncated/extended duplicates), then must pass ALL of:

      1. known-zone ``TVT_input`` agreement: >=200 rows both finite, and
         max abs diff over those rows <= 0.01 ft.
      2. ``GR`` agreement: >=500 rows both finite, and >=99.9% of those rows
         have abs diff <= 0.01.
      3. ``X``/``Y`` agreement (when present): max abs diff <= 1.0 on both.

    Among all candidates passing every gate, the one with the largest
    MD-overlap row count wins. Returns ``None`` if no candidate passes, or
    on ANY internal failure (never raises -- callers can always fall back to
    the existing pipeline's own prediction for this well).

    Returns (on a match)::

        {
            "twin_well": <train well name>,
            "override": np.ndarray,  # aligned to eval_mask(h_test) row order;
                                     # finite where the twin's TVT covers that
                                     # row, NaN elsewhere (leave to caller)
            "diagnostics": {...},    # overlap_rows, tvt_input_max_diff,
                                     # gr_match_ratio, eval_row_coverage, ...
        }
    """
    try:
        md_test = h_test["MD"].to_numpy(dtype=float)
        n_test = int(md_test.size)
        if n_test == 0:
            return None
        md_first_test = float(md_test[0])
        gr_test = h_test["GR"].to_numpy(dtype=float)
        tvt_input_test = h_test["TVT_input"].to_numpy(dtype=float)
        has_xy = "X" in h_test.columns and "Y" in h_test.columns
        x_test = h_test["X"].to_numpy(dtype=float) if has_xy else None
        y_test = h_test["Y"].to_numpy(dtype=float) if has_xy else None

        best: dict[str, object] | None = None
        best_overlap = -1
        for well, rec in train_signatures.items():
            if rec.n_finite_gr < _DUP_MIN_GR_ROWS:
                continue  # cheap signature-only reject: GR gate can never pass

            offset_f = md_first_test - rec.md_first
            offset = int(round(offset_f))
            if abs(offset_f - offset) > 1e-6:
                continue  # MD grids not phase-aligned on the 1ft grid; skip

            i_lo = max(0, -offset)
            i_hi = min(n_test, rec.n_rows - offset)
            overlap_rows = i_hi - i_lo
            if overlap_rows < _DUP_MIN_OVERLAP_ROWS:
                continue

            test_sl = slice(i_lo, i_hi)
            train_sl = slice(i_lo + offset, i_hi + offset)

            # Defensive re-check that the two MD grids actually line up over
            # the claimed overlap window (should always hold given the 1ft
            # grid + integer-offset construction above; cheap insurance
            # against a malformed/irregular MD column).
            if not np.allclose(md_test[test_sl], rec.md[train_sl], atol=1e-6):
                continue

            ti_test = tvt_input_test[test_sl]
            ti_train = rec.tvt_input[train_sl]
            both_ti = np.isfinite(ti_test) & np.isfinite(ti_train)
            n_ti = int(both_ti.sum())
            if n_ti < _DUP_MIN_TVT_INPUT_ROWS:
                continue
            tvt_input_max_diff = float(np.max(np.abs(ti_test[both_ti] - ti_train[both_ti])))
            if tvt_input_max_diff > _DUP_TVT_INPUT_TOL:
                continue

            gr_t = gr_test[test_sl]
            gr_r = rec.gr[train_sl]
            both_gr = np.isfinite(gr_t) & np.isfinite(gr_r)
            n_gr = int(both_gr.sum())
            if n_gr < _DUP_MIN_GR_ROWS:
                continue
            gr_match_ratio = float(np.mean(np.abs(gr_t[both_gr] - gr_r[both_gr]) <= _DUP_GR_TOL))
            if gr_match_ratio < _DUP_GR_MATCH_FRAC:
                continue

            max_dx = max_dy = 0.0
            if has_xy:
                max_dx = float(np.max(np.abs(x_test[test_sl] - rec.x[train_sl])))
                max_dy = float(np.max(np.abs(y_test[test_sl] - rec.y[train_sl])))
                if max_dx > _DUP_XY_TOL or max_dy > _DUP_XY_TOL:
                    continue

            if overlap_rows > best_overlap:
                best_overlap = overlap_rows
                best = {
                    "twin_well": well,
                    "offset": offset,
                    "overlap_rows": int(overlap_rows),
                    "tvt_input_max_diff": tvt_input_max_diff,
                    "n_tvt_input_matched": n_ti,
                    "gr_match_ratio": gr_match_ratio,
                    "n_gr_matched": n_gr,
                    "max_dx": max_dx,
                    "max_dy": max_dy,
                }

        if best is None:
            return None

        twin_well = str(best["twin_well"])
        offset = int(best["offset"])
        rec = train_signatures[twin_well]
        eval_idx = np.where(eval_mask(h_test))[0]
        n_eval = int(eval_idx.size)
        if n_eval == 0:
            return None

        train_rows = eval_idx + offset
        valid = (train_rows >= 0) & (train_rows < rec.n_rows)
        override = np.full(n_eval, np.nan, dtype=float)
        override[valid] = rec.tvt[train_rows[valid]]
        eval_row_coverage = float(np.mean(valid))

        diagnostics = {k: v for k, v in best.items() if k not in ("twin_well", "offset")}
        diagnostics["eval_row_coverage"] = eval_row_coverage
        diagnostics["n_eval_rows"] = n_eval

        return {"twin_well": twin_well, "override": override, "diagnostics": diagnostics}
    except Exception:  # never-raise: caller falls back to the pipeline's own prediction
        return None


def merge_duplicate_overrides(
    out: pd.DataFrame, sample: pd.DataFrame, dup_results: dict[str, dict[str, object]]
) -> int:
    """Overwrite ``out['tvt']`` rows in place with matched twins' finite overrides.

    ``dup_results[well]`` is expected to carry ``"override"`` (aligned to
    ``eval_mask`` row order, as returned by :func:`find_duplicate_twin`) and
    ``"eval_idx"`` (the corresponding row-index array, cached by the caller
    when it called :func:`find_duplicate_twin` so this function never needs
    to reload the test well). Only rows where the matched twin's override is
    finite are touched; everything else (unmatched wells, and rows within a
    matched well whose override is NaN) is left exactly as the existing
    pipeline produced it. A single malformed entry is skipped (logged)
    rather than aborting the whole merge. Returns the number of rows
    overridden.
    """
    n_overridden = 0
    well_col = sample["well"].to_numpy()
    row_index_col = sample["row_index"].to_numpy()
    for well, twin in dup_results.items():
        try:
            override = np.asarray(twin["override"], dtype=float)
            eval_idx = np.asarray(twin["eval_idx"])
            override_map = pd.Series(override, index=eval_idx)
            well_positions = np.where(well_col == well)[0]
            if well_positions.size == 0:
                continue
            mapped_vals = override_map.reindex(row_index_col[well_positions]).to_numpy(dtype=float)
            finite_mask = np.isfinite(mapped_vals)
            if not finite_mask.any():
                continue
            positions_to_update = well_positions[finite_mask]
            out.loc[positions_to_update, "tvt"] = mapped_vals[finite_mask]
            n_overridden += int(finite_mask.sum())
        except Exception as exc:  # never let one malformed entry break the merge
            print(f"[DUP][WARN] {well}: overlay merge failed for this well ({exc!r})")
            continue
    return n_overridden


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    t_start = time.time()
    root = find_data_root()
    print(f"[INFO] data root: {root}")

    sample = parse_sample_submission(root / "sample_submission.csv")
    test_wells = sorted(sample["well"].unique().tolist())
    print(f"[INFO] sample_submission: {len(sample)} id(s) across {len(test_wells)} test well(s)")

    train_wells = list_wells("train", root)
    # Full train population for the ROGII_DUP duplicate-twin search below --
    # captured BEFORE any ROGII_SMOKE truncation of `train_wells`, since a
    # hidden test well's twin can be any of the 773 train wells regardless
    # of how many wells the LGBM ensemble itself trains on.
    dup_all_train_wells = list(train_wells)
    smoke = _smoke_n()
    if smoke is not None:
        train_wells = train_wells[:smoke]
        print(
            f"[INFO] ROGII_SMOKE={smoke}: restricting training to {len(train_wells)} train well(s)"
        )
    print(f"[INFO] train wells: {len(train_wells)}")

    # --- ROGII_DUP: duplicate-twin search (see the module docstring and the
    # function docstrings of build_train_dup_signatures/find_duplicate_twin
    # for the full rationale). Runs BEFORE Phase 2 as required; every train
    # AND test well below is still processed exactly as sub-v8 would --
    # nothing here skips or alters that pipeline, it only prepares
    # `dup_results` for a final per-row merge applied after build_predictions.
    dup_enabled = _dup_enabled()
    dup_results: dict[str, dict[str, object]] = {}
    if dup_enabled:
        t_dup = time.time()
        try:
            dup_signatures = build_train_dup_signatures(dup_all_train_wells, root)
            print(
                f"[DUP] signature cache built for {len(dup_signatures)}/"
                f"{len(dup_all_train_wells)} train well(s) ({time.time() - t_dup:.1f}s)"
            )
            n_matched = 0
            coverage_weighted_sum = 0.0
            coverage_row_total = 0
            for well in test_wells:
                try:
                    h_test_dup = load_horizontal(well, "test", root)
                except Exception as exc:
                    print(f"[DUP][WARN] {well}: failed to load test well ({exc!r})")
                    continue
                twin = find_duplicate_twin(h_test_dup, dup_signatures, root)
                if twin is None:
                    print(f"[DUP] {well}: unmatched")
                    continue
                twin["eval_idx"] = np.where(eval_mask(h_test_dup))[0]
                dup_results[well] = twin
                n_matched += 1
                diag = twin["diagnostics"]
                coverage_weighted_sum += diag["eval_row_coverage"] * diag["n_eval_rows"]
                coverage_row_total += diag["n_eval_rows"]
                print(
                    f"[DUP] {well}: matched twin={twin['twin_well']} "
                    f"overlap_rows={diag['overlap_rows']} "
                    f"tvt_input_max_diff={diag['tvt_input_max_diff']:.6f} "
                    f"gr_match_ratio={diag['gr_match_ratio']:.4f} "
                    f"eval_row_coverage={diag['eval_row_coverage']:.4f} "
                    f"n_eval_rows={diag['n_eval_rows']}"
                )
            overall_coverage = (
                coverage_weighted_sum / coverage_row_total if coverage_row_total else 0.0
            )
            print(
                f"[DUP] summary: matched {n_matched}/{len(test_wells)} test well(s), "
                f"eval-row coverage (matched wells only) = {overall_coverage:.4f} "
                f"({time.time() - t_dup:.1f}s total)"
            )
        except Exception as exc:  # never let the overlay abort the submission
            print(
                f"[DUP][WARN] duplicate-twin overlay failed entirely ({exc!r}); "
                "falling back to the unmodified pipeline for every well"
            )
            dup_results = {}
    else:
        print("[DUP] ROGII_DUP=0: overlay disabled, pipeline output left unmodified")

    # --- Phase 1: spatial-prior surface bank (train formations only; see the
    # leak-boundary discussion in this file's module docstring). ---
    t0 = time.time()
    bank = build_surface_bank(train_wells, split="train", root=root, stride=SPATIAL_STRIDE)
    sizes = {f: int(bank.depths[f].size) for f in bank.formations}
    print(
        f"[PROGRESS] spatial surface bank built: formations={bank.formations} "
        f"samples={sizes} ({time.time() - t0:.1f}s)"
    )

    # --- Phase 2: PF + beam + spatial-prior computation for every train AND
    # test well (the dominant cost, ~5.7s/well measured locally with the
    # full bank -- back to the sub-v3 profile; sub-v8 does not call the
    # retained PF-BMA/beam-grid trackers, see the module docstring). H3
    # needs no tracker: it is computed later from the raw h in
    # build_well_features. ---
    all_wells = [(w, "train") for w in train_wells] + [(w, "test") for w in test_wells]
    wts_cache: dict[str, WellTrackerSpatial] = {}
    n_tracker_failed = 0
    for i, (well, split) in enumerate(all_wells, start=1):
        try:
            h = load_horizontal(well, split, root)
            tw = load_typewell(well, split, root)
        except Exception as exc:
            print(f"[WARN] {well} ({split}): failed to load ({exc!r})")
            n_tracker_failed += 1
            continue
        wts = compute_well_tracker_spatial(h, tw, well, bank)
        if wts is None:
            n_tracker_failed += 1
        else:
            wts_cache[well] = wts

        if i % PROGRESS_EVERY == 0 or i == len(all_wells):
            elapsed = time.time() - t_start
            per_well = elapsed / i
            remaining = (len(all_wells) - i) * per_well
            print(
                f"[PROGRESS] tracker/spatial {i}/{len(all_wells)} | "
                f"elapsed={elapsed:.1f}s | avg={per_well:.3f}s/well | "
                f"est. remaining={remaining:.1f}s | failed={n_tracker_failed}"
            )
    print(
        f"[INFO] tracker/spatial cache: {len(wts_cache)}/{len(all_wells)} wells ok, "
        f"{n_tracker_failed} failed ({time.time() - t_start:.1f}s elapsed total)"
    )

    # --- Phase 3: train feature matrix + PF-residual target ---
    X_train, y_true, pf_blend_train, well_idx, used_train_wells, _well_to_idx = build_train_matrix(
        train_wells, root, wts_cache
    )
    target = y_true - pf_blend_train
    feature_cols = list(X_train.columns)
    print(f"[INFO] feature columns ({len(feature_cols)}): {feature_cols}")
    if len(feature_cols) != N_FEATURES_EXPECTED:
        print(
            f"[WARN] expected {N_FEATURES_EXPECTED} feature columns (run_stack_v23_cv.py "
            f"config (c)), got {len(feature_cols)} -- porting drift, investigate"
        )

    # --- Phase 4: well-level ES-holdout split (same carving recipe as
    # run_stack_v2_cv.py's all-in row, applied over all usable train wells) ---
    fit_wells, es_wells = split_fit_es(used_train_wells, ES_FRAC, SEED)
    degenerate_es = not fit_wells
    if degenerate_es:
        # Only reachable when there are <2 usable train wells (ROGII_SMOKE=1-
        # style local smoke runs; the full 773-well run always has both sides
        # non-empty). Fit and early-stop on the same rows rather than crash.
        print(
            "[WARN] <2 usable train wells: fit set would be empty; "
            "fitting and early-stopping on the same rows (smoke-run-only path)"
        )
        fit_wells = list(es_wells)
    well_to_idx = {w: i for i, w in enumerate(used_train_wells)}
    fit_idx = np.array([well_to_idx[w] for w in fit_wells], dtype=np.int32)
    es_idx = np.array([well_to_idx[w] for w in es_wells], dtype=np.int32)
    fit_mask = np.isin(well_idx, fit_idx)
    es_mask = np.isin(well_idx, es_idx)
    if not degenerate_es:
        assert not (fit_mask & es_mask).any(), "fit/ES-holdout rows must not overlap"
    assert bool((fit_mask | es_mask).all()), "every train row must be fit or ES-holdout"
    print(
        f"[INFO] ES-holdout split: fit_wells={len(fit_wells)} ({int(fit_mask.sum()):,} rows) "
        f"es_wells={len(es_wells)} ({int(es_mask.sum()):,} rows)"
    )

    # --- Phase 5: train the ENSEMBLE_SEEDS LGBM members on all usable train
    # data (R9: random_state is the only member-to-member difference; the
    # fit/ES-holdout row split is identical for every member) ---
    t0 = time.time()
    models: list[lgb.LGBMRegressor] = []
    es_residual_sum = np.zeros(int(es_mask.sum()), dtype=np.float64)
    for seed_i, member_seed in enumerate(ENSEMBLE_SEEDS, start=1):
        t_m = time.time()
        model = lgb.LGBMRegressor(**_lgb_params_for_seed(member_seed))
        model.fit(
            X_train.loc[fit_mask],
            target[fit_mask],
            eval_set=[(X_train.loc[es_mask], target[es_mask])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        models.append(model)
        member_res = model.predict(X_train.loc[es_mask])
        es_residual_sum += member_res
        member_rmse = float(
            np.sqrt(np.mean((y_true[es_mask] - (pf_blend_train[es_mask] + member_res)) ** 2))
        )
        print(
            f"[INFO] LGBM member {seed_i}/{len(ENSEMBLE_SEEDS)} (seed={member_seed}): "
            f"best_iteration={model.best_iteration_} ES-holdout RMSE={member_rmse:.4f} "
            f"({time.time() - t_m:.1f}s)"
        )
    es_pred_tvt = pf_blend_train[es_mask] + es_residual_sum / len(ENSEMBLE_SEEDS)
    es_rmse = float(np.sqrt(np.mean((y_true[es_mask] - es_pred_tvt) ** 2)))
    print(
        f"[INFO] {len(ENSEMBLE_SEEDS)}-seed ensemble ES-holdout pooled RMSE={es_rmse:.4f} "
        f"(diagnostic only, not a CV score) ({time.time() - t0:.1f}s total fit)"
    )

    # --- Phase 6: predict every test well (well-level try/except ->
    # carry_last -> 0.0, in that degrading order; sub-v8 = stack v2.6's own
    # 5-seed-mean output, no blend) ---
    out, mismatch_warnings = build_predictions(
        sample, root, wts_cache, feature_cols, models, t_start
    )

    # --- ROGII_DUP: final per-row merge. Only overwrites rows whose matched
    # twin has a finite TVT for that row; everything else keeps the
    # unmodified pipeline's own prediction from build_predictions above. With
    # ROGII_DUP=0 (dup_enabled False) or no matches, `out` is untouched here
    # and this file's submission.csv is byte-identical to the plain sub-v8
    # kernel run under the same conditions. ---
    if dup_enabled and dup_results:
        try:
            n_overridden = merge_duplicate_overrides(out, sample, dup_results)
            print(
                f"[DUP] overlay applied: {n_overridden} row(s) overridden with "
                f"twin TVT across {len(dup_results)} matched well(s)"
            )
        except Exception as exc:  # never let the overlay merge break the submission
            print(
                f"[DUP][WARN] overlay merge failed entirely ({exc!r}); "
                "submission left unmodified"
            )

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
    print(f"[PASS] wrote {out_path.resolve()} ({len(out)} rows, {len(test_wells)} well(s))")
    print("[PASS] id set matches sample_submission exactly; no NaN/inf in tvt; row counts match")
    print(f"[INFO] total elapsed: {elapsed_total:.1f}s ({elapsed_total / 60:.2f} min)")


if __name__ == "__main__":
    main()
