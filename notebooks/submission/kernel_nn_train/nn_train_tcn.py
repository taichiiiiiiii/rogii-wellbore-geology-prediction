"""Self-contained Kaggle **GPU training** kernel: R10 Step 2+3 -- TCN on the NN sequence dataset.

This is NOT a submission kernel (it writes no ``submission.csv``). It is
implementation Steps 2+3 of ``docs/r10_nn_design.md``: a PyTorch Dataset/
collate layer with fold-safe scalers (Step 2) plus the design's main-line
TCN (dilated 1D-CNN) model, masked pooled-weighted MSE loss, AdamW + cosine
schedule, and ES-holdout early stopping (Step 3), trained on the output of
Step 1 (``notebooks/submission/kernel_nn_dataset/nn_dataset_builder.py``:
``nn_seq_channels.npz`` + ``nn_static_channels.npz``).

It has NO project imports (no ``import rogii``) -- only numpy/torch/stdlib --
so it runs unmodified both:

- on Kaggle (``enable_gpu: true``), reading the Step-1 npz pair from the
  attached dataset ``taichiiiii/rogii-nn-dataset`` under
  ``/kaggle/input/rogii-nn-dataset/`` (see :func:`find_data_dir` -- an rglob
  fallback covers a nested upload layout), writing checkpoints + OOF npz to
  ``/kaggle/working/``. **The no-argument Kaggle run is the iter3 sweep**:
  configs E+F x all 5 folds (see "Config sweep" below); and
- locally on CPU against a small Step-1 smoke build for a fast check::

    uv run python notebooks/submission/kernel_nn_train/nn_train_tcn.py \
        --data-dir <dir with the two Step-1 npz files> \
        --fold 1 --configs EF --epochs 2 --smoke 10 --device cpu \
        --batch-rows 6000 --out-dir <scratch dir>

Design contract (docs/r10_nn_design.md)
-----------------------------------------
- **Input channels (full set: 66)**: Step 1's 44 per-row sequence channels
  + its 21 well-static channels broadcast to every row of that well
  (Sec.3.2) = 65, plus 1 ``any_nan`` indicator channel (1.0 on rows where
  ANY kept sequence channel was NaN before imputation -- in this dataset
  NaN only ever comes from the GR-derived channels). All real channels are
  standardized with **fit-well-only masked mean/std** (Sec.1.3's fold-safe
  scaler contract, cdeotte ``compute_feature_scaler`` pattern), then NaN
  positions are imputed to 0 (== the post-standardization mean) and values
  are clipped to +-20 std units (:data:`CLIP_STD_UNITS` -- guards against
  fit-degenerate channels exploding on unseen score wells); the indicator
  channel is appended raw. Config D trains on a REDUCED channel set (see
  "Config sweep").
- **Target**: Step 1's ``target = TVT - pf_blend`` (Sec.1.2, PF-blend
  residual), standardized with fit-well-only mean/std over eval rows
  (cdeotte ``compute_target_scaler`` pattern).
- **Loss** (Sec.1.3): masked MSE over eval rows only (plus, under the
  prefix-cut augmentation, the pseudo-eval rows -- see below), flattened
  across the batch x time axes -- every unmasked ROW carries equal weight,
  matching the pooled-RMSE metric definition (never per-well-averaged).
  Padding rows and (non-pseudo) known-zone rows are excluded via the mask.
- **Model** (Sec.2.1): the cdeotte-reference TCN, hidden=128, kernel=5,
  ``--blocks`` B in {7, 8} (default 7), dilation 2^i, symmetric padding,
  BatchNorm+SiLU+Dropout ResidualBlocks. Parameter count ~=1.2M at B=7;
  receptive field = 1 + 8*(2^B - 1) rows (B=7: +-508) -- printed per run.
- **CV** (Sec.4.1): score folds come from the ``well_fold`` array Step 1
  embedded in ``nn_static_channels.npz`` (the byte-identical
  ``cv.well_folds(seed=42)`` assignment recovered from
  ``outputs/stack_v2_oof.npz``, so this script's OOF is directly
  pooled-RMSE-comparable to stack v2/v2.1/v2.2/v2.3's OOF numbers). Within
  each fold-run the remaining wells are split well-level into fit /
  ES-holdout (``--es-frac`` 0.15, shuffle seed FIXED at 42 -- the
  ``_split_fit_es`` / ``ES_SPLIT_SEED`` recipe from
  ``run_stack_v23_cv.py``); the score fold is never touched during training
  or early stopping.
- **All-fold default**: ``--fold -1`` (default) trains every fold present
  serially in ONE process; pass ``--fold f`` for a single fold.

Config sweep (iter2)
----------------------
Iter1 measured A (lr 2e-3) = 9.5953 and B (lr/3 + warmup) = 10.0083 pooled
5-fold OOF, with EVERY fold showing best-epoch=1-then-degrade -- an
immediate-overfit (generalization) signature, not an LR one. Iter2 keeps
A's optimizer and attacks generalization directly:

- **config C** = A + (a) per-epoch random prefix-cut augmentation
  (:data:`AUG_CUT_FRAC_MAX`, see "Prefix-cut augmentation") + (b) dropout
  0.10 -> 0.25 + (c) weight_decay 1e-4 -> 1e-2 + (d) patience 5 -> 7.
- **config D** = C on a reduced channel set: DROPS the pfbma_v2/beam_grid
  bank-delta sequence channels (8) + the spatial-prior block (4 seq) + the
  2 bank-derived static channels (:data:`NOBANK_DROP_SEQ` /
  :data:`NOBANK_DROP_STATIC`) -> 32 seq + 19 static + 1 indicator = 52
  input channels. Rationale: the LB calibration ledger (2026-07-10 sub-v6
  row) established that pf_bma_v2/beam_grid-derived information is HARMFUL
  on the hidden test set, and the field-CV audit (2026-07-09) flagged the
  spatial prior's near-well dependence -- D measures a transfer-safe NN
  that cannot depend on the suspect channels. The core PF/beam tracker
  channels (``pf_d``/``beam_d``/... 8 seq + 6 static aggregates) are KEPT:
  they are the pf_blend floor's own ingredients and demonstrably
  transferred in stack v2 (well-CV 9.2309 -> LB 9.022).

Config sweep (iter3: insurance architecture, design doc Sec.2.2)
------------------------------------------------------------------
Iter2 measured C (full 66ch + generalization pack) and D (bank channels
removed, 52ch) = 10.7657 -- D degraded heavily BUT D's channel set is the
transfer-safe one (the dropped channels are the LB-toxic suspects). Iter3
asks whether a DIFFERENT architecture recovers the gap on D's safe
channels -- the design doc's insurance BiGRU (Sec.2.2), whose unbounded
receptive field is the structural complement to the TCN's fixed +-508
rows:

- **config E** = config D verbatim (TCN + generalization pack + nobank
  52ch), re-run in the same kernel as the head-to-head reference.
- **config F** = E's exact data/regularization/optimizer settings with the
  architecture swapped to :class:`GRUEncDecModel`: a 2-layer BiGRU encoder
  (hidden 64/direction) consumes the known-zone prefix (already truncated
  to the last ``--known-tail`` rows, matching the design doc's 3-4k-row
  encoder window); its last-layer forward+backward final hidden states
  form a context vector, which is broadcast-concatenated onto every
  eval-zone row's input channels and decoded by a 1-layer unidirectional
  GRU (hidden 96, NON-autoregressive -- no predicted TVT is ever fed back)
  + small MLP head. ~0.25M parameters (:data:`GRU_ENC_HIDDEN` etc.).
  Packed sequences keep padding out of both RNNs. Under the prefix-cut
  augmentation the encoder/decoder split follows the AUGMENTED boundary
  (the pseudo-eval rows are decoded, not encoded) -- the split point is
  derived from the loss mask, which is a contiguous tail by construction.

Simplifications (iter3, documented): GRU hyperparameters are fixed module
constants (no CLI sweep); the decoder consumes ALL kept channels + context
(a superset of the design doc's "位置/幾何/トラッカー系" listing); LR/
schedule/clip stay identical to E so the comparison isolates architecture.

``--configs`` selects {EF (default), CD, AB, A, B, C, D, E, F}; earlier
iterations stay reproducible from this same file (A/B take their
dropout/weight-decay/patience from the CLI flags, which default to iter1's
values). ``--arch`` can force every selected config onto one architecture
(default ``auto`` = each config's own). A final table compares every
trained config's combined pooled OOF against the pf_blend floor, stack v2
(9.2309), the TCN iter1 config-A (9.5953) and iter2 config-D (10.7657)
references, and the design gates (<=9.5 / <=8.8, Sec.5).

Prefix-cut augmentation (config C/D; simplified -- explicitly documented)
---------------------------------------------------------------------------
The design doc's full prefix-cut (Sec.4.3, "保留(stretch)") would re-run
every physical tracker at a synthetic anchor -- a heavy Kaggle batch job.
This implements the sanctioned SIMPLIFIED variant (the design doc's own
"トラッカーchannel 0埋め" fallback), entirely inside the Dataset layer:

For each FIT-well sample, each epoch, draw ``c ~ Uniform{0..c_max}``
(``c_max = min(0.4 * n_known_rows, n_known_rows - 1)``,
:data:`AUG_CUT_FRAC_MAX`); if ``c > 0``, treat the last ``c`` known rows
as a *pseudo eval zone*:

1. Pseudo rows' ``known_tvt_mask`` and ``tvt_input_delta_last`` inputs are
   zeroed (they must look like eval rows -- otherwise the target below is
   readable straight off the input, a shortcut that would teach nothing).
   Their tracker/spatial/bank channels are ALREADY all-zero in the Step-1
   layout (known-zone zero-fill), which is exactly the design doc's
   simplified-variant semantics.
2. Remaining known rows' ``tvt_input_delta_last`` is re-based to the
   pseudo-anchor (subtract the pseudo-anchor row's stored delta), and the
   anchor-relative position channels (``md/dx/dy/dz_from_ps`` +
   recomputed ``dist3d_from_ps``) are re-based likewise for every
   pre-boundary row. REAL eval rows keep their true-anchor channels and
   true PF-residual targets untouched (the exact test-time distribution).
3. Pseudo rows' target = ``TVT - pseudo_pf_blend`` where ``pseudo_pf_blend``
   is the flat carry from the pseudo-anchor -- computable exactly from the
   stored delta channel (``target = stored_delta - delta_at_pseudo_anchor``)
   because ``TVT == TVT_input`` on known rows (verified in Step 1). This is
   the zone-start approximation of the PF blend (PF drift ~= 0 at the
   anchor, design doc Sec.1.2); no tracker re-run.
4. The loss mask becomes ``real eval rows + pseudo rows``.

Documented simplifications: GR-vs-typewell anchor channels
(``gr_minus_twgr_anchor_calibrated``/``tw_gr_at_carry``) and ALL static
channels stay true-anchor-based (not recomputable without raw data);
pseudo rows carry zero tracker signal where real eval rows have live
trackers. Both are accepted augmentation noise. ES-holdout and score
predictions always use CLEAN (non-augmented) samples.

Outputs (to ``--out-dir`` / ``/kaggle/working``)
--------------------------------------------------
Per config c and fold f: ``tcn_{c}_fold{f}.pt`` (best-ES-epoch state_dict
+ scalers + kept-channel names + full config) and
``nn_tcn_oof_{c}_fold{f}.npz`` (per-row OOF + epoch histories + config).
Per config c: ``nn_tcn_oof_{c}_all.npz`` (all folds' score rows
concatenated + combined pooled RMSE) -- the single file Step-5 local
analysis downloads.

Timing
------
Per-epoch wall time, running mean, projected fold time, and a final
per-run + total summary are printed from the LIVE run's own measurements
(design doc Sec.4.4; CPU-smoke numbers do NOT transfer to Kaggle GPU).

Leak boundary
-------------
This script never opens a competition CSV -- every input value comes from
Step 1's npz pair (leak boundary documented in ``nn_dataset_builder.py``).
Fold discipline: scalers see fit wells only, early stopping sees ES wells
only, score-fold rows are used exactly once per fold-run (final OOF). The
prefix-cut augmentation only ever reads KNOWN-zone values (``TVT_input``'s
own prefix, which is public input) -- pseudo targets are derived from the
``tvt_input_delta_last`` channel, never from eval-zone ground truth.

Attribution
-----------
Model/Dataset/collate/scaler/masked-loss structure adapted from the public
Kaggle notebook ``cdeotte/nn-starter-cv-15-5`` (mirrored at
``cdeotte/nn-starter-cv-15-5`` on Kaggle). Channel design,
PF-residual target, fold reuse, ES protocol and the prefix-cut idea per
``docs/r10_nn_design.md`` (this repo). ES-split recipe ported from
``scripts/run_stack_v23_cv.py::_split_fit_es``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Constants / expected dataset schema (must match nn_dataset_builder.py)
# --------------------------------------------------------------------------- #

KAGGLE_DATA_DIR = Path("/kaggle/input/rogii-nn-dataset")
SEQ_NPZ = "nn_seq_channels.npz"
STATIC_NPZ = "nn_static_channels.npz"

N_SEQ_CHANNELS = 44
N_STATIC_CHANNELS = 21

N_FOLDS = 5
ES_SPLIT_SEED = 42  # frozen, run_stack_v23_cv.py::ES_SPLIT_SEED

# References for the final report table (analysis/experiment_ledger.md /
# docs/r10_nn_design.md Sec.5 / TCN iter1 T4 measurement).
STACK_V2_OOF_RMSE = 9.2309  # stack v2 (LB 9.022)
STACK_V23_OOF_RMSE = 8.8573  # stack v2.3 (R9, current best)
TCN_ITER1_A_RMSE = 9.5953  # TCN iter1 config A, 5-fold pooled (T4)
TCN_ITER2_D_RMSE = 10.7657  # TCN iter2 config D (nobank 52ch), 5-fold pooled (T4)
GATE_ENSEMBLE_RMSE = 9.5  # design gate: ensemble-material
GATE_SOLO_RMSE = 8.8  # design gate: solo-competitive

DEFAULT_EPOCHS = 15
DEFAULT_KNOWN_TAIL = 4000
DEFAULT_BATCH_ROWS = 16000
DEFAULT_HIDDEN = 128
DEFAULT_BLOCKS = 7
DEFAULT_DROPOUT = 0.10  # configs A/B (iter1); C/D use C_DROPOUT
DEFAULT_LR = 2e-3
DEFAULT_WEIGHT_DECAY = 1e-4  # configs A/B (iter1); C/D use C_WEIGHT_DECAY
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_PATIENCE = 5  # configs A/B (iter1); C/D use C_PATIENCE
DEFAULT_ES_FRAC = 0.15
DEFAULT_SEED = 42

WARMUP_START_FACTOR = 0.1  # config B: first-epoch LR ramps from lr*0.1 to lr

# iter2 generalization package (configs C/D/E/F -- module docstring).
C_DROPOUT = 0.25
C_WEIGHT_DECAY = 1e-2
C_PATIENCE = 7
AUG_CUT_FRAC_MAX = 0.4  # prefix-cut: up to 40% of the known rows become pseudo-eval

# iter3 insurance architecture (config F -- design doc Sec.2.2's BiGRU
# encoder + non-autoregressive decoder; fixed constants, no CLI sweep).
GRU_ENC_HIDDEN = 64  # per direction; bidirectional context = 128
GRU_ENC_LAYERS = 2
GRU_DEC_HIDDEN = 96
GRU_DEC_LAYERS = 1

# Post-standardization clip (std units). Guards against a channel that is
# near-constant across the FIT wells (its std collapses to the 1e-3 floor)
# standardizing an unseen score-well value to ~1e3-1e5 and exploding the
# network output -- observed live as a 33,537ft OOF blow-up on one fold of a
# 10-well CPU smoke (tiny fit sets make degenerate channels likely; at the
# full 618-fit-well scale this clip should be inert for real signals, which
# stay well inside +-20 std).
CLIP_STD_UNITS = 20.0

# Config D's dropped channels (module docstring "Config sweep"): the
# pfbma_v2/beam_grid bank-delta family (ledger 2026-07-10 sub-v6: harmful on
# the hidden test set) + the spatial-prior block (field-CV audit 2026-07-09:
# near-well dependence) + the 2 bank-derived static aggregates. Names must
# match nn_dataset_builder.py's channel lists exactly (validated at load).
NOBANK_DROP_SEQ: tuple[str, ...] = (
    "spatial_d",
    "spatial_prefix_rmse",
    "spatial_nn_dist_median",
    "spatial_gated_d",
    "pfbma_v2_d",
    "beam_grid_d",
    "pfbma_v2_minus_pf",
    "pfbma_v2_s5_d",
    "pfbma_v2_s8_d",
    "pfbma_v2_s12_d",
    "pfbma_v2_scale_range",
    "pfbma_v2_s3_std",
)
NOBANK_DROP_STATIC: tuple[str, ...] = (
    "pfbma_v2_d_well_mean",
    "pfbma_v2_s3_std_well_mean",
)

# Channels the prefix-cut augmentation edits (must exist in the seq layout).
AUG_CH_KNOWN_MASK = "known_tvt_mask"
AUG_CH_DELTA = "tvt_input_delta_last"
AUG_CH_POS = ("md_from_ps", "dx_from_ps", "dy_from_ps", "dz_from_ps")
AUG_CH_DIST = "dist3d_from_ps"


# --------------------------------------------------------------------------- #
# Data directory / output directory discovery
# --------------------------------------------------------------------------- #


def find_data_dir(explicit: str | None) -> Path:
    """Locate the directory containing the two Step-1 npz files."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("ROGII_NN_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(KAGGLE_DATA_DIR)
    candidates.append(Path.cwd())

    for c in candidates:
        if (c / SEQ_NPZ).is_file() and (c / STATIC_NPZ).is_file():
            return c

    # Kaggle dataset uploads sometimes nest their files one level down.
    if Path("/kaggle/input").is_dir():
        for hit in sorted(Path("/kaggle/input").rglob(SEQ_NPZ)):
            if (hit.parent / STATIC_NPZ).is_file():
                return hit.parent

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate {SEQ_NPZ} + {STATIC_NPZ} under any of: {tried}, "
        f"nor via /kaggle/input rglob"
    )


def resolve_out_dir(explicit: str | None) -> Path:
    if explicit:
        d = Path(explicit)
        d.mkdir(parents=True, exist_ok=True)
        return d
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    d = Path.cwd() / "outputs_nn_train_local"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Dataset loading (Step-1 npz pair -> per-well records)
# --------------------------------------------------------------------------- #


@dataclass
class WellRecord:
    """One well after known-tail truncation. Arrays are row-aligned (length T')."""

    well: str
    fold: int
    x_seq: np.ndarray  # (T', 44) float32
    static: np.ndarray  # (21,) float32
    target: np.ndarray  # (T',) float32   TVT - pf_blend on eval rows, 0 elsewhere
    mask: np.ndarray  # (T',) bool       True on eval rows
    y_true: np.ndarray  # (n_eval,) float32  ground-truth TVT, eval rows only
    pf_blend: np.ndarray  # (n_eval,) float32  PF-blend floor, eval rows only

    @property
    def n_rows(self) -> int:
        return int(self.x_seq.shape[0])

    @property
    def n_eval(self) -> int:
        return int(self.y_true.shape[0])

    @property
    def first_eval(self) -> int:
        """Row index of the first eval row (eval zone is a contiguous tail)."""
        return self.n_rows - self.n_eval


def load_records(
    data_dir: Path, known_tail: int, smoke: int | None
) -> tuple[list[WellRecord], list[str], list[str]]:
    """Load + validate the Step-1 npz pair and slice it into per-well records.

    Applies the known-zone tail truncation (module docstring) during
    slicing. ``smoke`` keeps only the first N wells (file order). Returns
    ``(records, seq_channel_names, static_channel_names)``.
    """
    with np.load(data_dir / SEQ_NPZ, allow_pickle=True) as seq:
        channel_names = [str(c) for c in seq["channel_names"]]
        x_all = seq["X_seq"]
        boundaries = seq["boundaries"].astype(np.int64)
        well_names = [str(w) for w in seq["well_names"]]
        target_all = seq["target"].astype(np.float32)
        mask_all = seq["target_mask"].astype(bool)
        y_true_all = seq["y_true"].astype(np.float32)
        pf_blend_all = seq["pf_blend"].astype(np.float32)

    with np.load(data_dir / STATIC_NPZ, allow_pickle=True) as stat:
        static_names = [str(c) for c in stat["static_channel_names"]]
        x_static = stat["X_static"].astype(np.float32)
        static_wells = [str(w) for w in stat["well_names"]]
        well_fold = stat["well_fold"].astype(np.int64)

    if len(channel_names) != N_SEQ_CHANNELS:
        raise AssertionError(f"expected {N_SEQ_CHANNELS} seq channels, got {len(channel_names)}")
    if len(static_names) != N_STATIC_CHANNELS:
        raise AssertionError(
            f"expected {N_STATIC_CHANNELS} static channels, got {len(static_names)}"
        )
    if well_names != static_wells:
        raise AssertionError("well_names disagree between the seq and static npz files")
    if int(boundaries[-1]) != int(x_all.shape[0]):
        raise AssertionError("boundaries[-1] != X_seq row count -- corrupt dataset")
    if not np.isin(well_fold, np.arange(N_FOLDS)).all():
        raise AssertionError(f"well_fold values outside 0..{N_FOLDS - 1}")

    n_wells = len(well_names)
    keep = min(smoke, n_wells) if smoke is not None else n_wells

    records: list[WellRecord] = []
    for i in range(keep):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        mask_w = mask_all[s:e]
        n_eval = int(mask_w.sum())
        if n_eval == 0:
            continue
        eval_idx = np.flatnonzero(mask_w)
        first_eval = int(eval_idx[0])
        if not mask_w[first_eval:].all():
            raise AssertionError(f"{well_names[i]}: eval zone is not a contiguous tail block")

        start = max(0, first_eval - max(int(known_tail), 0))
        records.append(
            WellRecord(
                well=well_names[i],
                fold=int(well_fold[i]),
                x_seq=np.ascontiguousarray(x_all[s + start : e]),
                static=x_static[i].copy(),
                target=target_all[s + start : e].copy(),
                mask=mask_w[start:].copy(),
                y_true=y_true_all[s:e][mask_w].copy(),
                pf_blend=pf_blend_all[s:e][mask_w].copy(),
            )
        )

    if not records:
        raise RuntimeError("0 usable wells after loading -- nothing to train on")
    return records, channel_names, static_names


# --------------------------------------------------------------------------- #
# Channel plan (full vs "nobank" reduced set) + fold-safe scalers
# --------------------------------------------------------------------------- #


@dataclass
class ChannelPlan:
    """Kept-channel indices + the augmentation's edit-channel indices.

    ``seq_keep``/``static_keep`` index into the FULL Step-1 layouts (44/21);
    the augmentation indices (``i_*``) also refer to the full seq layout
    because prefix-cut edits happen on the raw arrays BEFORE subsetting.
    """

    seq_keep: np.ndarray
    static_keep: np.ndarray
    keep_names: list[str]
    i_known_mask: int
    i_delta: int
    i_pos: tuple[int, ...]  # md/dx/dy/dz_from_ps
    i_dist: int

    @property
    def n_features(self) -> int:
        return len(self.seq_keep) + len(self.static_keep) + 1  # + any_nan indicator


def build_channel_plan(
    channel_names: list[str], static_names: list[str], channel_set: str
) -> ChannelPlan:
    if channel_set == "nobank":
        drop_seq = set(NOBANK_DROP_SEQ)
        drop_static = set(NOBANK_DROP_STATIC)
        unknown = (drop_seq - set(channel_names)) | (drop_static - set(static_names))
        if unknown:
            raise AssertionError(f"nobank drop lists name unknown channels: {sorted(unknown)}")
    elif channel_set == "full":
        drop_seq = set()
        drop_static = set()
    else:
        raise ValueError(f"unknown channel_set {channel_set!r}")

    seq_keep = np.array(
        [i for i, c in enumerate(channel_names) if c not in drop_seq], dtype=np.int64
    )
    static_keep = np.array(
        [i for i, c in enumerate(static_names) if c not in drop_static], dtype=np.int64
    )
    keep_names = (
        [channel_names[i] for i in seq_keep]
        + [static_names[i] for i in static_keep]
        + ["any_nan"]
    )
    return ChannelPlan(
        seq_keep=seq_keep,
        static_keep=static_keep,
        keep_names=keep_names,
        i_known_mask=channel_names.index(AUG_CH_KNOWN_MASK),
        i_delta=channel_names.index(AUG_CH_DELTA),
        i_pos=tuple(channel_names.index(c) for c in AUG_CH_POS),
        i_dist=channel_names.index(AUG_CH_DIST),
    )


@dataclass
class Scalers:
    feat_mean: np.ndarray  # (65,) float32 -- FULL layout; datasets subset it
    feat_std: np.ndarray  # (65,) float32
    y_mean: float
    y_std: float


def compute_scalers(records: list[WellRecord]) -> Scalers:
    """Masked (finite-only) per-channel mean/std over the full 65 real channels.

    Static channels enter row-weighted (each well's scalar counted once per
    row), exactly as if they had been broadcast first -- so standardization
    is identical whether broadcasting happens before or after.
    """
    n_real = N_SEQ_CHANNELS + N_STATIC_CHANNELS
    sums = np.zeros(n_real, dtype=np.float64)
    sums_sq = np.zeros(n_real, dtype=np.float64)
    counts = np.zeros(n_real, dtype=np.float64)
    y_vals: list[np.ndarray] = []

    for rec in records:
        x = rec.x_seq.astype(np.float64)
        finite = np.isfinite(x)
        x0 = np.where(finite, x, 0.0)
        sums[:N_SEQ_CHANNELS] += x0.sum(axis=0)
        sums_sq[:N_SEQ_CHANNELS] += (x0**2).sum(axis=0)
        counts[:N_SEQ_CHANNELS] += finite.sum(axis=0)

        t = float(rec.n_rows)
        sv = rec.static.astype(np.float64)
        sums[N_SEQ_CHANNELS:] += sv * t
        sums_sq[N_SEQ_CHANNELS:] += (sv**2) * t
        counts[N_SEQ_CHANNELS:] += t

        y_vals.append(rec.target[rec.mask].astype(np.float64))

    counts = np.maximum(counts, 1.0)
    mean = sums / counts
    var = sums_sq / counts - mean**2
    std = np.sqrt(np.maximum(var, 1e-6))

    y_cat = np.concatenate(y_vals)
    y_mean = float(y_cat.mean())
    y_std = float(y_cat.std() + 1e-6)

    return Scalers(
        feat_mean=mean.astype(np.float32),
        feat_std=std.astype(np.float32),
        y_mean=y_mean,
        y_std=y_std,
    )


# --------------------------------------------------------------------------- #
# PyTorch Dataset / collate (Step 2) + prefix-cut augmentation (iter2)
# --------------------------------------------------------------------------- #


class WellSequenceDataset(torch.utils.data.Dataset):
    """1 item = 1 well: assembled (T, n_features) input, scaled target, loss mask.

    Assembly happens lazily per __getitem__ (broadcast static + indicator +
    standardize + impute + clip), so the resident memory stays at Step 1's
    compact (T, 44) + (21,) layout. ``augment=True`` (fit wells of configs
    C/D only) applies the per-epoch random prefix-cut (module docstring);
    ES/score datasets are always constructed with ``augment=False``.
    """

    def __init__(
        self,
        records: list[WellRecord],
        scalers: Scalers,
        plan: ChannelPlan,
        augment: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.records = records
        self.plan = plan
        self.augment = augment and rng is not None
        self.rng = rng
        keep = np.concatenate([plan.seq_keep, N_SEQ_CHANNELS + plan.static_keep])
        self.feat_mean = scalers.feat_mean[keep]
        self.feat_std = scalers.feat_std[keep]
        self.y_mean = scalers.y_mean
        self.y_std = scalers.y_std
        self.n_aug_applied = 0  # diagnostics: prefix-cuts actually applied (c > 0)

    def __len__(self) -> int:
        return len(self.records)

    def _prefix_cut(self, rec: WellRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random prefix-cut on one sample; returns (x_raw, target, mask) copies.

        See the module docstring's "Prefix-cut augmentation" for the exact
        contract (pseudo-eval window, re-based deltas/positions, flat-carry
        pseudo target, extended loss mask).
        """
        assert self.rng is not None
        first_eval = rec.first_eval
        c_max = min(int(AUG_CUT_FRAC_MAX * first_eval), first_eval - 1)
        c = int(self.rng.integers(0, c_max + 1)) if c_max > 0 else 0
        if c <= 0:
            return rec.x_seq, rec.target, rec.mask

        x = rec.x_seq.copy()
        target = rec.target.copy()
        mask = rec.mask.copy()
        plan = self.plan

        pa = first_eval - c - 1  # pseudo-anchor row (>= 0 by c_max construction)
        pseudo = slice(first_eval - c, first_eval)
        pre = slice(0, first_eval)  # all pre-boundary rows (known + pseudo)

        delta = x[:, plan.i_delta].copy()
        d_shift = float(delta[pa])

        # (3) pseudo target BEFORE the input edits below overwrite the delta:
        # TVT - flat-carry-from-pseudo-anchor == stored_delta - d_shift.
        target[pseudo] = delta[pseudo] - d_shift
        mask[pseudo] = True

        # (1) pseudo rows must look like eval rows on the boundary channels.
        # (2) remaining known rows re-based to the pseudo-anchor.
        x[pre, plan.i_delta] = delta[pre] - d_shift
        x[pseudo, plan.i_delta] = 0.0
        x[pseudo, plan.i_known_mask] = 0.0

        # (2) anchor-relative positions re-based for pre-boundary rows only
        # (real eval rows keep the exact test-time distribution).
        pos_shift = x[pa, list(plan.i_pos)].copy()
        for j, i_ch in enumerate(plan.i_pos):
            x[pre, i_ch] = x[pre, i_ch] - pos_shift[j]
        dx = x[pre, plan.i_pos[1]]
        dy = x[pre, plan.i_pos[2]]
        dz = x[pre, plan.i_pos[3]]
        x[pre, plan.i_dist] = np.sqrt(dx**2 + dy**2 + dz**2)

        self.n_aug_applied += 1
        return x, target, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        rec = self.records[idx]
        t = rec.n_rows
        plan = self.plan

        if self.augment:
            x_raw, target, mask = self._prefix_cut(rec)
        else:
            x_raw, target, mask = rec.x_seq, rec.target, rec.mask

        x_seq_kept = x_raw[:, plan.seq_keep]
        any_nan = (~np.isfinite(x_seq_kept)).any(axis=1).astype(np.float32)

        n_seq = len(plan.seq_keep)
        n_real = n_seq + len(plan.static_keep)
        x_real = np.empty((t, n_real), dtype=np.float32)
        x_real[:, :n_seq] = x_seq_kept
        x_real[:, n_seq:] = rec.static[plan.static_keep][None, :]
        x_real = (x_real - self.feat_mean) / self.feat_std
        x_real = np.nan_to_num(x_real, nan=0.0, posinf=0.0, neginf=0.0)
        x_real = np.clip(x_real, -CLIP_STD_UNITS, CLIP_STD_UNITS)  # see CLIP_STD_UNITS

        x = np.concatenate([x_real, any_nan[:, None]], axis=1)

        y = (target - self.y_mean) / self.y_std
        y = np.where(mask, y, 0.0).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "idx": idx,
        }


def collate_wells(
    batch: list[dict[str, torch.Tensor | int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Pad a list of per-well items to the batch max length.

    Padding rows get ``mask=False`` so they never reach the loss (cdeotte
    ``collate_wells`` convention). Returns ``(x, y, mask, lengths, indices)``.
    """
    bsz = len(batch)
    max_len = max(int(item["x"].shape[0]) for item in batch)
    n_features = int(batch[0]["x"].shape[1])

    x = torch.zeros(bsz, max_len, n_features, dtype=torch.float32)
    y = torch.zeros(bsz, max_len, dtype=torch.float32)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    lengths: list[int] = []
    indices: list[int] = []

    for i, item in enumerate(batch):
        length = int(item["x"].shape[0])
        x[i, :length] = item["x"]
        y[i, :length] = item["y"]
        mask[i, :length] = item["mask"]
        lengths.append(length)
        indices.append(int(item["idx"]))

    return x, y, mask, lengths, indices


def make_batches(lengths: list[int], row_budget: int, rng: random.Random) -> list[list[int]]:
    """Length-bucketed batches under a total padded-row budget.

    Wells sorted by length descending, greedily packed while
    ``max_len * (n+1) <= row_budget`` (every batch keeps >=1 well even if a
    single well exceeds the budget). Batch ORDER is shuffled; batch contents
    are deterministic (see module docstring's documented trade-off).
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        if not cur:
            cur = [i]
            cur_max = lengths[i]
            continue
        if cur_max * (len(cur) + 1) <= row_budget:
            cur.append(i)
        else:
            batches.append(cur)
            cur = [i]
            cur_max = lengths[i]
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------- #
# Model (Step 3) -- cdeotte TCNResidualModel, hidden/blocks per design Sec.2.1
# --------------------------------------------------------------------------- #


class ResidualBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TCNResidualModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = DEFAULT_HIDDEN,
        num_blocks: int = DEFAULT_BLOCKS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_size, kernel_size=5, dilation=2**i, dropout=dropout)
                for i in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_size // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: list[int] | None = None,
    ) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, seq_len).
        # mask/lengths are part of the shared model interface (the GRU
        # architecture needs them for its encoder/decoder split); the TCN
        # is translation-invariant over the padded axis and ignores them.
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)


class GRUEncDecModel(nn.Module):
    """Design doc Sec.2.2 insurance architecture (config F).

    BiGRU encoder over each well's known-zone prefix -> last-layer
    forward+backward final hidden states = context vector -> broadcast onto
    every eval-zone row's input channels -> 1-layer unidirectional GRU
    decoder + MLP head, emitting the whole eval zone in one pass
    (NON-autoregressive: predictions are never fed back). Unbounded
    receptive field (the encoder summarizes the entire known prefix; the
    decoder state runs the entire eval zone) -- the structural complement
    to the TCN's fixed +-508-row window. Packed sequences keep padding out
    of both RNNs. The encoder/decoder split per sample = ``length -
    mask.sum()`` (the loss mask is a contiguous tail by construction, so
    under prefix-cut augmentation pseudo-eval rows are decoded, not
    encoded).
    """

    def __init__(
        self,
        n_features: int,
        enc_hidden: int = GRU_ENC_HIDDEN,
        enc_layers: int = GRU_ENC_LAYERS,
        dec_hidden: int = GRU_DEC_HIDDEN,
        dec_layers: int = GRU_DEC_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.encoder = nn.GRU(
            n_features,
            enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )
        ctx_dim = 2 * enc_hidden
        self.decoder = nn.GRU(
            n_features + ctx_dim,
            dec_hidden,
            num_layers=dec_layers,
            batch_first=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(dec_hidden, dec_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: list[int] | None = None,
    ) -> torch.Tensor:
        if mask is None or lengths is None:
            raise ValueError("GRUEncDecModel.forward requires mask and lengths")
        bsz, t_max, _n_feat = x.shape
        lens = torch.as_tensor(lengths, device=x.device, dtype=torch.long)
        n_dec = mask.sum(dim=1).to(torch.long)  # eval(+pseudo) rows per sample
        enc_len = (lens - n_dec).clamp(min=1)

        # --- encoder over the known prefixes (packed: padding never enters) ---
        packed_enc = nn.utils.rnn.pack_padded_sequence(
            x, enc_len.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder(packed_enc)  # (layers*2, B, H); [-2]=last fwd, [-1]=last bwd
        ctx = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)

        # --- gather eval segments into their own padded tensor ---
        dec_max = int(n_dec.max().clamp(min=1))
        dec_in = x.new_zeros(bsz, dec_max, x.shape[2])
        for i in range(bsz):
            s = int(enc_len[i])
            e = int(lens[i])
            if e > s:
                dec_in[i, : e - s] = x[i, s:e]
        dec_in = torch.cat([dec_in, ctx[:, None, :].expand(-1, dec_max, -1)], dim=2)

        packed_dec = nn.utils.rnn.pack_padded_sequence(
            dec_in, n_dec.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        dec_out, _ = self.decoder(packed_dec)
        dec_out, _ = nn.utils.rnn.pad_packed_sequence(
            dec_out, batch_first=True, total_length=dec_max
        )
        pred_eval = self.head(dec_out).squeeze(-1)  # (B, dec_max)

        # --- scatter predictions back to the full padded layout ---
        out = x.new_zeros(bsz, t_max)
        for i in range(bsz):
            s = int(enc_len[i])
            e = int(lens[i])
            if e > s:
                out[i, s:e] = pred_eval[i, : e - s]
        return out


def build_model(cfg: TrainConfig, n_features: int, args: argparse.Namespace) -> nn.Module:
    """Model factory: config's own arch, unless ``--arch`` forces one."""
    arch = cfg.arch if args.arch == "auto" else args.arch
    if arch == "gru":
        return GRUEncDecModel(n_features, dropout=cfg.dropout)
    return TCNResidualModel(
        n_features=n_features,
        hidden_size=args.hidden,
        num_blocks=args.blocks,
        dropout=cfg.dropout,
    )


def receptive_field_rows(num_blocks: int, kernel_size: int = 5) -> int:
    """Symmetric receptive field of the block stack: 1 + 2*(k-1)*sum(2^i)."""
    return 1 + 2 * (kernel_size - 1) * (2**num_blocks - 1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """Flat masked MSE: every unmasked row weighs equally (pooled-RMSE aligned)."""
    if int(mask.sum()) == 0:
        return None
    diff = pred[mask] - target[mask]
    return torch.mean(diff**2)


# --------------------------------------------------------------------------- #
# Optimizer/regularization configs (iter2 sweep) + schedule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainConfig:
    """One configuration of the sweep (module docstring "Config sweep")."""

    label: str
    lr: float
    warmup_epochs: int
    dropout: float
    weight_decay: float
    patience: int
    prefix_cut: bool  # per-epoch random prefix-cut augmentation on fit wells
    channels: str  # "full" | "nobank"
    arch: str = "tcn"  # "tcn" | "gru" (design doc Sec.2.1 vs Sec.2.2)


def build_configs(args: argparse.Namespace) -> list[TrainConfig]:
    cfg_a = TrainConfig(
        "A", args.lr, 0, args.dropout, args.weight_decay, args.patience, False, "full"
    )
    cfg_b = TrainConfig(
        "B", args.lr / 3.0, 1, args.dropout, args.weight_decay, args.patience, False, "full"
    )
    cfg_c = TrainConfig("C", args.lr, 0, C_DROPOUT, C_WEIGHT_DECAY, C_PATIENCE, True, "full")
    cfg_d = TrainConfig("D", args.lr, 0, C_DROPOUT, C_WEIGHT_DECAY, C_PATIENCE, True, "nobank")
    # iter3 (module docstring "Config sweep (iter3)"): E = D re-run under its
    # own label (same-kernel head-to-head reference), F = E with the
    # architecture swapped to the Sec.2.2 BiGRU encoder-decoder.
    cfg_e = TrainConfig("E", args.lr, 0, C_DROPOUT, C_WEIGHT_DECAY, C_PATIENCE, True, "nobank")
    cfg_f = TrainConfig(
        "F", args.lr, 0, C_DROPOUT, C_WEIGHT_DECAY, C_PATIENCE, True, "nobank", arch="gru"
    )
    mapping: dict[str, list[TrainConfig]] = {
        "EF": [cfg_e, cfg_f],
        "CD": [cfg_c, cfg_d],
        "AB": [cfg_a, cfg_b],
        "A": [cfg_a],
        "B": [cfg_b],
        "C": [cfg_c],
        "D": [cfg_d],
        "E": [cfg_e],
        "F": [cfg_f],
    }
    return mapping[args.configs]


def build_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Epoch-stepped cosine decay, optionally preceded by a linear warmup."""
    if warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=WARMUP_START_FACTOR,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, cosine], milestones=[warmup_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


# --------------------------------------------------------------------------- #
# Train / evaluate one (config, fold) run
# --------------------------------------------------------------------------- #


def split_fit_es(train_wells: list[str], es_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Well-level fit vs ES-holdout split (run_stack_v23_cv.py::_split_fit_es, verbatim)."""
    ordered = sorted(train_wells)
    random.Random(seed).shuffle(ordered)
    n_es = max(1, round(len(ordered) * es_frac))
    es_wells = sorted(ordered[:n_es])
    fit_wells = sorted(ordered[n_es:])
    return fit_wells, es_wells


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


@torch.no_grad()
def predict_wells(
    model: nn.Module,
    dataset: WellSequenceDataset,
    device: torch.device,
    row_budget: int,
) -> list[np.ndarray]:
    """Predicted TVT (ft) for every well's eval rows, in dataset order."""
    model.eval()
    lengths = [rec.n_rows for rec in dataset.records]
    batches = make_batches(lengths, row_budget, random.Random(0))
    preds: list[np.ndarray | None] = [None] * len(dataset)

    for batch_idx in batches:
        items = [dataset[i] for i in batch_idx]
        x, _y, mask, lens, indices = collate_wells(items)
        x = x.to(device)
        out = model(x, mask.to(device), lens).float().cpu().numpy()
        for row_i, (ds_i, length) in enumerate(zip(indices, lens, strict=True)):
            rec = dataset.records[ds_i]
            mask_np = mask[row_i, :length].numpy()
            resid_scaled = out[row_i, :length][mask_np]
            resid = resid_scaled * dataset.y_std + dataset.y_mean
            preds[ds_i] = (rec.pf_blend.astype(np.float64) + resid).astype(np.float32)

    missing = [i for i, p in enumerate(preds) if p is None]
    if missing:
        raise AssertionError(f"{len(missing)} well(s) got no prediction -- batching bug")
    return [p for p in preds if p is not None]


def eval_pooled_rmse(
    model: nn.Module,
    dataset: WellSequenceDataset,
    device: torch.device,
    row_budget: int,
) -> float:
    preds = predict_wells(model, dataset, device, row_budget)
    y_true = np.concatenate([rec.y_true for rec in dataset.records])
    y_pred = np.concatenate(preds)
    return pooled_rmse(y_true, y_pred)


def run_fold(
    records: list[WellRecord],
    channel_names: list[str],
    static_names: list[str],
    fold: int,
    cfg: TrainConfig,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    out_dir: Path,
) -> dict[str, object]:
    """Train one (config, fold) run to completion; save its checkpoint + OOF npz."""
    t_run0 = time.perf_counter()
    tag = f"{cfg.label}/f{fold}"
    seed_everything(args.seed)

    score_recs = [r for r in records if r.fold == fold]
    train_recs = [r for r in records if r.fold != fold]
    if not score_recs:
        raise RuntimeError(
            f"fold {fold} has no wells in this (possibly --smoke-limited) subset -- "
            f"folds present: {sorted({r.fold for r in records})}"
        )
    if len(train_recs) < 2:
        raise RuntimeError(f"fold {fold}: need >=2 train wells, got {len(train_recs)}")

    fit_wells, es_wells = split_fit_es([r.well for r in train_recs], args.es_frac, ES_SPLIT_SEED)
    fit_set = set(fit_wells)
    es_set = set(es_wells)
    fit_recs = [r for r in train_recs if r.well in fit_set]
    es_recs = [r for r in train_recs if r.well in es_set]
    assert not (fit_set & es_set), "fit/ES wells overlap"
    assert not ({r.well for r in score_recs} & (fit_set | es_set)), "score fold leaked into train"

    plan = build_channel_plan(channel_names, static_names, cfg.channels)
    arch = cfg.arch if args.arch == "auto" else args.arch
    print(
        f"\n=== [{tag}] arch={arch} lr={cfg.lr:g} warmup={cfg.warmup_epochs}ep "
        f"do={cfg.dropout:g} wd={cfg.weight_decay:g} pat={cfg.patience} "
        f"aug={'prefix-cut' if cfg.prefix_cut else 'off'} "
        f"ch={cfg.channels}({plan.n_features}): "
        f"fit={len(fit_recs)} es={len(es_recs)} score={len(score_recs)} wells ===",
        flush=True,
    )

    scalers = compute_scalers(fit_recs)  # FIT wells only -- fold-safe
    print(f"[{tag}] target scaler (fit wells): mean={scalers.y_mean:.4f} std={scalers.y_std:.4f}")

    aug_rng = np.random.default_rng(args.seed + fold) if cfg.prefix_cut else None
    fit_ds = WellSequenceDataset(fit_recs, scalers, plan, augment=cfg.prefix_cut, rng=aug_rng)
    es_ds = WellSequenceDataset(es_recs, scalers, plan)
    score_ds = WellSequenceDataset(score_recs, scalers, plan)

    model = build_model(cfg, plan.n_features, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    rf_note = (
        "unbounded (recurrent)"
        if arch == "gru"
        else f"{receptive_field_rows(args.blocks)} rows"
    )
    print(f"[{tag}] model: params={n_params:,} receptive field={rf_note}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, cfg.warmup_epochs)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    fit_lengths = [r.n_rows for r in fit_recs]
    batch_rng = random.Random(args.seed)

    best_es = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_no_improve = 0
    history_train_loss: list[float] = []
    history_es_rmse: list[float] = []
    epoch_seconds: list[float] = []

    for epoch in range(1, args.epochs + 1):
        t_ep0 = time.perf_counter()
        lr_used = float(optimizer.param_groups[0]["lr"])  # LR in effect DURING this epoch
        model.train()
        losses: list[float] = []

        for batch_idx in make_batches(fit_lengths, args.batch_rows, batch_rng):
            items = [fit_ds[i] for i in batch_idx]
            x, y, mask, lens, _idx = collate_wells(items)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x, mask, lens)
                loss = masked_mse(pred, y, mask)
            if loss is None:
                continue
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            losses.append(float(loss.detach().cpu()))

        scheduler.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        es_rmse = eval_pooled_rmse(model, es_ds, device, args.batch_rows)
        history_train_loss.append(train_loss)
        history_es_rmse.append(es_rmse)
        ep_s = time.perf_counter() - t_ep0
        epoch_seconds.append(ep_s)

        improved = es_rmse < best_es
        if improved:
            best_es = es_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        mean_ep = float(np.mean(epoch_seconds))
        aug_note = f" aug_applied={fit_ds.n_aug_applied}" if cfg.prefix_cut else ""
        print(
            f"[{tag}] epoch {epoch:>3}/{args.epochs}: train_loss={train_loss:.5f} "
            f"lr={lr_used:.2e} "
            f"ES pooled RMSE={es_rmse:.4f}ft{' *' if improved else ''}  "
            f"({ep_s:.1f}s, mean {mean_ep:.1f}s/ep, projected full fold "
            f"~{mean_ep * args.epochs / 60:.1f}min){aug_note}",
            flush=True,
        )

        if epochs_no_improve >= cfg.patience:
            print(f"[{tag}] early stop: no ES improvement for {cfg.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{tag}] best epoch: {best_epoch} (ES pooled RMSE {best_es:.4f}ft)")

    # --- final score-fold OOF prediction (the ONLY use of score-fold rows) ---
    t_pred0 = time.perf_counter()
    preds = predict_wells(model, score_ds, device, args.batch_rows)
    t_pred = time.perf_counter() - t_pred0

    y_true = np.concatenate([r.y_true for r in score_recs])
    pf_blend = np.concatenate([r.pf_blend for r in score_recs])
    pred_flat = np.concatenate(preds)
    assert pred_flat.shape == y_true.shape, "OOF prediction misaligned with y_true"
    assert np.isfinite(pred_flat).all(), "non-finite values in OOF prediction"

    fold_rmse = pooled_rmse(y_true, pred_flat)
    floor_rmse = pooled_rmse(y_true, pf_blend)
    print(
        f"[{tag}] OOF: pooled RMSE={fold_rmse:.4f}ft "
        f"(pf_blend floor on same rows: {floor_rmse:.4f}ft)  predict: {t_pred:.1f}s"
    )

    config = vars(args).copy()
    config["config_label"] = cfg.label
    config["config_arch"] = arch
    config["config_lr"] = cfg.lr
    config["config_warmup_epochs"] = cfg.warmup_epochs
    config["config_dropout"] = cfg.dropout
    config["config_weight_decay"] = cfg.weight_decay
    config["config_patience"] = cfg.patience
    config["config_prefix_cut"] = cfg.prefix_cut
    config["config_channels"] = cfg.channels
    config["n_input_channels"] = plan.n_features
    config["kept_channel_names"] = plan.keep_names
    config["n_params"] = n_params
    config["receptive_field_rows"] = receptive_field_rows(args.blocks)
    config["n_prefix_cuts_applied"] = fit_ds.n_aug_applied

    ckpt_path = out_dir / f"tcn_{cfg.label}_fold{fold}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feat_mean": fit_ds.feat_mean,  # already subset to the kept channels
            "feat_std": fit_ds.feat_std,
            "y_mean": scalers.y_mean,
            "y_std": scalers.y_std,
            "n_input_channels": plan.n_features,
            "kept_channel_names": plan.keep_names,
            "config": config,
            "best_epoch": best_epoch,
            "best_es_rmse": best_es,
        },
        ckpt_path,
    )

    eval_lens = [r.n_eval for r in score_recs]
    eval_boundaries = np.concatenate(([0], np.cumsum(eval_lens))).astype(np.int64)
    oof_path = out_dir / f"nn_tcn_oof_{cfg.label}_fold{fold}.npz"
    np.savez_compressed(
        oof_path,
        well_names=np.array([r.well for r in score_recs]),
        eval_boundaries=eval_boundaries,
        pred_tvt=pred_flat.astype(np.float32),
        y_true=y_true.astype(np.float32),
        pf_blend=pf_blend.astype(np.float32),
        fold=np.int64(fold),
        pooled_rmse=np.float64(fold_rmse),
        pf_blend_rmse=np.float64(floor_rmse),
        history_train_loss=np.array(history_train_loss, dtype=np.float64),
        history_es_rmse=np.array(history_es_rmse, dtype=np.float64),
        best_epoch=np.int64(best_epoch),
        epoch_seconds=np.array(epoch_seconds, dtype=np.float64),
        config_json=json.dumps(config),
    )
    print(f"[{tag}] saved: {ckpt_path.name}, {oof_path.name}")

    return {
        "config": cfg.label,
        "arch": arch,
        "fold": fold,
        "well_names": [r.well for r in score_recs],
        "eval_lens": eval_lens,
        "pred": pred_flat,
        "y_true": y_true,
        "pf_blend": pf_blend,
        "oof_rmse": fold_rmse,
        "floor_rmse": floor_rmse,
        "best_epoch": best_epoch,
        "best_es_rmse": best_es,
        "epochs_ran": len(epoch_seconds),
        "mean_epoch_s": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
        "wall_s": time.perf_counter() - t_run0,
    }


# --------------------------------------------------------------------------- #
# Per-config aggregation + final report
# --------------------------------------------------------------------------- #


def aggregate_config(
    cfg: TrainConfig, fold_results: list[dict[str, object]], out_dir: Path
) -> dict[str, float]:
    """Combine one config's fold OOFs into a pooled RMSE + save the merged npz."""
    y_true = np.concatenate([np.asarray(r["y_true"]) for r in fold_results])
    pred = np.concatenate([np.asarray(r["pred"]) for r in fold_results])
    pf_blend = np.concatenate([np.asarray(r["pf_blend"]) for r in fold_results])
    pooled = pooled_rmse(y_true, pred)
    floor = pooled_rmse(y_true, pf_blend)

    well_names: list[str] = []
    fold_per_well: list[int] = []
    eval_lens: list[int] = []
    for r in fold_results:
        names = [str(w) for w in r["well_names"]]  # type: ignore[union-attr]
        well_names.extend(names)
        fold_per_well.extend([int(r["fold"])] * len(names))  # type: ignore[arg-type]
        eval_lens.extend(int(n) for n in r["eval_lens"])  # type: ignore[union-attr]
    eval_boundaries = np.concatenate(([0], np.cumsum(eval_lens))).astype(np.int64)

    all_path = out_dir / f"nn_tcn_oof_{cfg.label}_all.npz"
    folds_arr = np.array(sorted({int(r["fold"]) for r in fold_results}), dtype=np.int64)  # type: ignore[arg-type]
    np.savez_compressed(
        all_path,
        well_names=np.array(well_names),
        fold_per_well=np.array(fold_per_well, dtype=np.int32),
        eval_boundaries=eval_boundaries,
        pred_tvt=pred.astype(np.float32),
        y_true=y_true.astype(np.float32),
        pf_blend=pf_blend.astype(np.float32),
        pooled_rmse=np.float64(pooled),
        pf_blend_rmse=np.float64(floor),
        folds=folds_arr,
        config_label=cfg.label,
    )

    n_folds = len(fold_results)
    arch = str(fold_results[0]["arch"])
    print(f"\n{'=' * 92}")
    print(
        f"config {cfg.label} (arch={arch} lr={cfg.lr:g} do={cfg.dropout:g} "
        f"wd={cfg.weight_decay:g} aug={'on' if cfg.prefix_cut else 'off'} "
        f"ch={cfg.channels}): {n_folds} fold(s), {len(y_true):,} OOF rows -> {all_path.name}"
    )
    print(f"{'fold':>5}{'OOF rmse':>11}{'floor':>9}{'best_ep':>9}{'epochs':>8}{'min':>7}")
    for r in sorted(fold_results, key=lambda d: int(d["fold"])):  # type: ignore[arg-type]
        print(
            f"{int(r['fold']):>5}{float(r['oof_rmse']):>11.4f}{float(r['floor_rmse']):>9.4f}"  # type: ignore[arg-type]
            f"{int(r['best_epoch']):>9}{int(r['epochs_ran']):>8}"  # type: ignore[arg-type]
            f"{float(r['wall_s']) / 60:>7.1f}"  # type: ignore[arg-type]
        )
    print("-" * 92)
    partial = "" if n_folds == N_FOLDS else f" (PARTIAL: {n_folds}/{N_FOLDS} folds)"
    print(f"combined pooled OOF RMSE: {pooled:.4f}ft{partial}")
    print(f"  pf_blend floor (same rows):       {floor:.4f}ft ({pooled - floor:+.4f})")
    print(f"  TCN iter1 A reference (9.5953):   {pooled - TCN_ITER1_A_RMSE:+.4f}")
    print(f"  TCN iter2 D reference (10.7657):  {pooled - TCN_ITER2_D_RMSE:+.4f}")
    print(f"  stack v2 reference (9.2309):      {pooled - STACK_V2_OOF_RMSE:+.4f}")
    print(f"  stack v2.3 reference (8.8573):    {pooled - STACK_V23_OOF_RMSE:+.4f}")
    print(
        f"  gate <= {GATE_ENSEMBLE_RMSE} (ensemble-material): "
        f"{'PASS' if pooled <= GATE_ENSEMBLE_RMSE else 'FAIL'}"
        f"   gate <= {GATE_SOLO_RMSE} (solo-competitive): "
        f"{'PASS' if pooled <= GATE_SOLO_RMSE else 'FAIL'}"
    )
    return {"pooled": pooled, "floor": floor, "n_folds": float(n_folds)}


def print_final_report(
    configs: list[TrainConfig],
    summaries: dict[str, dict[str, float]],
    all_results: list[dict[str, object]],
    t_start: float,
) -> None:
    if len(configs) > 1:
        print(f"\n{'=' * 92}")
        print("config comparison (combined pooled OOF RMSE):")
        print(
            f"{'config':>7}{'arch':>6}{'aug':>5}{'ch':>8}{'folds':>7}{'pooled':>10}"
            f"{'vs floor':>10}{'vs A':>9}{'vs D':>9}{'<=9.5':>7}"
        )
        for cfg in configs:
            s = summaries[cfg.label]
            arch = next(
                str(r["arch"]) for r in all_results if r["config"] == cfg.label
            )
            print(
                f"{cfg.label:>7}{arch:>6}{'on' if cfg.prefix_cut else 'off':>5}"
                f"{cfg.channels:>8}{int(s['n_folds']):>7}{s['pooled']:>10.4f}"
                f"{s['pooled'] - s['floor']:>+10.4f}"
                f"{s['pooled'] - TCN_ITER1_A_RMSE:>+9.4f}"
                f"{s['pooled'] - TCN_ITER2_D_RMSE:>+9.4f}"
                f"{'PASS' if s['pooled'] <= GATE_ENSEMBLE_RMSE else 'FAIL':>7}"
            )
        print(
            f"reference rows: TCN iter1 A = {TCN_ITER1_A_RMSE} / "
            f"TCN iter2 D = {TCN_ITER2_D_RMSE} (5-fold pooled, T4)"
        )

    print(f"\n{'=' * 92}")
    print("timing summary (measured on THIS device -- CPU numbers do NOT transfer to GPU):")
    print(f"{'run':>8}{'epochs':>8}{'s/epoch':>9}{'wall min':>10}")
    for r in all_results:
        print(
            f"{str(r['config']) + '/f' + str(r['fold']):>8}{int(r['epochs_ran']):>8}"  # type: ignore[arg-type]
            f"{float(r['mean_epoch_s']):>9.1f}{float(r['wall_s']) / 60:>10.1f}"  # type: ignore[arg-type]
        )
    total_s = time.perf_counter() - t_start
    print(f"total wall time: {total_s / 60:.1f}min ({total_s:.0f}s)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def resolve_fp16(arg: str, device: torch.device) -> bool:
    if arg == "on":
        if device.type != "cuda":
            raise ValueError("--fp16 on requires a CUDA device")
        return True
    if arg == "off":
        return False
    return device.type == "cuda"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(-1, 0, 1, 2, 3, 4), default=-1,
                        help="score fold id; -1 (default) trains ALL folds serially")
    parser.add_argument("--configs", choices=("EF", "CD", "AB", "A", "B", "C", "D", "E", "F"),
                        default="EF",
                        help="config sweep: E=TCN-D rerun, F=GRU on D's channels (iter3); "
                        "C/D = iter2; A/B = iter1 (default EF)")
    parser.add_argument("--arch", choices=("auto", "tcn", "gru"), default="auto",
                        help="force one architecture for every selected config "
                        "(default auto = each config's own)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="restrict to the first N wells (file order) for a smoke run")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fp16", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"directory containing {SEQ_NPZ} + {STATIC_NPZ} (default: auto)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="checkpoint/OOF output directory (default: /kaggle/working or CWD)")
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS,
                        help="total padded-row budget per batch (bucketing)")
    parser.add_argument("--known-tail", type=int, default=DEFAULT_KNOWN_TAIL,
                        help="max known-zone rows kept per well (design Sec.4.3)")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT,
                        help="configs A/B only (C/D use the fixed iter2 value 0.25)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="configs A/C/D base LR; config B derives LR/3 from it")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
                        help="configs A/B only (C/D use the fixed iter2 value 1e-2)")
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="configs A/B only (C/D use the fixed iter2 value 7)")
    parser.add_argument("--es-frac", type=float, default=DEFAULT_ES_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    t_start = time.perf_counter()

    device = resolve_device(args.device)
    use_amp = resolve_fp16(args.fp16, device)
    print(f"device: {device}  fp16: {use_amp}  torch: {torch.__version__}")

    data_dir = find_data_dir(args.data_dir)
    print(f"data dir: {data_dir}")
    t_load0 = time.perf_counter()
    records, channel_names, static_names = load_records(data_dir, args.known_tail, args.smoke)
    total_rows = sum(r.n_rows for r in records)
    total_eval = sum(r.n_eval for r in records)
    print(
        f"wells: {len(records)}  rows(after known-tail {args.known_tail}): {total_rows:,}  "
        f"eval rows: {total_eval:,}  load: {time.perf_counter() - t_load0:.1f}s"
    )

    folds_present = sorted({r.fold for r in records})
    folds = folds_present if args.fold == -1 else [args.fold]
    if args.fold == -1 and len(folds_present) < N_FOLDS:
        print(
            f"WARNING: only folds {folds_present} present in this "
            f"(--smoke-limited?) subset -- combined pooled OOF will be PARTIAL"
        )

    configs = build_configs(args)
    out_dir = resolve_out_dir(args.out_dir)
    print(
        f"plan: {len(configs)} config(s) x {len(folds)} fold(s) = "
        f"{len(configs) * len(folds)} runs; out dir: {out_dir}"
    )

    all_results: list[dict[str, object]] = []
    for cfg in configs:
        for fold in folds:
            all_results.append(
                run_fold(
                    records, channel_names, static_names, fold, cfg, args,
                    device, use_amp, out_dir,
                )
            )

    summaries: dict[str, dict[str, float]] = {}
    for cfg in configs:
        fold_results = [r for r in all_results if r["config"] == cfg.label]
        summaries[cfg.label] = aggregate_config(cfg, fold_results, out_dir)

    print_final_report(configs, summaries, all_results, t_start)


if __name__ == "__main__":
    main()
