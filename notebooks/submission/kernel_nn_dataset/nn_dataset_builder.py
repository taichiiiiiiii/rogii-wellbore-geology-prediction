"""Self-contained Kaggle Notebook **compute-only** kernel: R10 NN sequence dataset builder.

This is NOT a submission kernel (it writes no ``submission.csv``). It is
implementation Step 1 of ``docs/r10_nn_design.md`` (the NN second-model
design, R10): it builds the per-row + well-static feature channels and
target described in that document's Sec.3 for every usable train well, and
persists them as two ``.npz`` files so a later step (the design's Step
2/3/4, out of this script's scope) can build a PyTorch ``Dataset`` /
``DataLoader`` without re-running any of this compute.

Paste this entire script into a single Kaggle Notebook cell (script kernel),
run it with internet disabled, and recover its two ``.npz`` outputs
afterwards via ``kaggle kernels output <user>/rogii-nn-dataset-builder -p
<outdir>``. It has NO project imports (no ``import rogii``) -- only numpy/
pandas/scipy/stdlib -- so it is fully self-contained and runs unmodified
both:

- on Kaggle, reading the mounted competition dataset under ``/kaggle/input``
  (see :func:`find_data_root` for the exact discovery strategy), writing to
  ``/kaggle/working/``, and
- locally against ``data/raw/`` for a smoke test::

    ROGII_NN_DATASET_LIMIT=5 uv run python \
        notebooks/submission/kernel_nn_dataset/nn_dataset_builder.py

  (``ROGII_NN_DATASET_LIMIT=<N>`` restricts the well list to the first ``N``
  train wells, sorted deterministically, **and** restricts the spatial-prior
  surface bank (:func:`build_surface_bank`) to that same subset -- a full
  773-well bank build is a fixed ~tens-of-minutes-scale cost unrelated to
  the per-well loop this smoke mode exists to check, see
  ``scripts/build_spatial_cache.py``'s docstring for the measured ~35min
  figure on the full well list. Smoke mode also redirects output from
  ``/kaggle/working``/``outputs/`` to ``./outputs_smoke_nn/`` so a local run
  never touches ``/kaggle/working`` or this repo's real ``outputs/``.
  **Do not run this locally without the env var set** -- a full 773-well
  run (surface bank + ~10s/well tracker compute, see Sec.4.4 of the design
  doc) is exactly the kind of batch job this project's memory-constrained
  local dev box (3.8GB RAM) cannot safely run; it belongs on Kaggle.)

What this builds (``docs/r10_nn_design.md`` Sec.3)
----------------------------------------------------
For every usable train well (eval zone non-empty + known-zone anchor
present -- the well-level acceptance rule below is otherwise identical to
``run_stack_v23_cv.py::build_well_matrix_v2``'s):

1. **Per-row sequence channels** (Sec.3.1, one value per row of the well's
   *entire* ``MD`` axis -- known-zone context rows AND eval-zone rows, both
   in one contiguous sequence, mirroring
   ``cdeotte/nn-starter-cv-15-5`` on Kaggle's ``build_sequence`` cell
   8 I/O shape): position (anchor-relative only, no absolute X/Y/Z -- see
   design doc Sec.2.1's R3-v3-derived rationale), trajectory derivatives,
   GR + rolling stats, typewell GR calibration/carry-forward, the boundary
   markers ``known_tvt_mask``/``tvt_input_delta_last``, and the physical
   tracker deltas (PF / beam / spatial-prior / PF-BMA-v2 / beam-grid, all
   vs. the same ``pf_blend`` basis the stack GBDT uses). :data:`SEQ_CHANNEL_NAMES`
   is the fixed 44-column order (24 position/GR/typewell/boundary columns
   from :func:`build_seq_features` + 20 tracker/spatial/bank-delta columns
   from :func:`build_eval_delta_block`, broadcast into the full row range
   -- see "Tracker channels are known-zone zero-filled" below).
2. **Well-level static channels** (Sec.3.2, one value per well, *not*
   broadcast to every row here -- see "Storage format" below):
   :data:`STATIC_CHANNEL_NAMES`, 21 columns = the design doc's listed 19
   (known-prefix slopes/stats + tracker well-aggregates) plus 2 more of
   stack v2.3's own well-level bank-delta columns
   (``pfbma_v2_d_well_mean``/``pfbma_v2_s3_std_well_mean``) that the design
   doc's Sec.3.2 prose omits but Sec.3's "ほぼ全量" (translate stack v2.3's
   66 features almost entirely) intent covers -- a deliberate, documented
   interpretation call (see the module-level report this script's caller
   produces), not a design deviation of substance.
3. **Target** = ``TVT - pf_blend`` (design doc Sec.1.2's "(b) PF残差"
   target, byte-identical formula to ``run_stack_v23_cv.py``:
   ``pf_blend = anchor + 0.7 * (pf_tvt - anchor)``), defined **only** on
   eval-zone rows (``target_mask``); known-zone rows get ``target=0.0``
   (never used in a masked loss, and verified elsewhere in this docstring's
   attribution trail that ``TVT == TVT_input`` exactly on every known-zone
   row of every sampled well, so a masked-out placeholder of 0 is the
   correct value even if the mask were dropped by a downstream bug).
4. **Meta**: well names, per-well row counts, and each well's CV fold id,
   recovered from a **773-well well->fold table embedded as a source-code
   constant** (:data:`_WELL_FOLD_NAMES`/:data:`_WELL_FOLD_DIGITS` ->
   :data:`WELL_FOLD_MAP`) -- Kaggle has no access to this repo's
   ``outputs/stack_v2_oof.npz`` (which is git-ignored competition-adjacent
   analysis output, never uploaded to Kaggle), so the fold assignment is
   baked into this script's source instead of read from that file at
   runtime. The table was generated **once, locally**, by loading
   ``outputs/stack_v2_oof.npz``'s ``used_wells``/``well_idx``/``row_fold``
   arrays, verifying every one of the 773 wells' rows carry a single
   consistent fold id (0..4), and sorting by well name for a deterministic
   embedding order -- see this file's own :func:`_parse_well_fold_map` for
   the exact reconstruction and the module-level assertion right after it
   that re-derives and checks the same invariants at import time. This
   table is the byte-identical fold assignment ``run_stack_v2_cv.py`` /
   ``run_stack_v21/v22/v23_cv.py`` all trained and scored against
   (``cv.well_folds(list_wells("train"), n_splits=5, seed=42)``), so a
   downstream NN OOF computed against these fold ids is directly
   pooled-RMSE-comparable to those scripts' OOF numbers (design doc Sec.4.1).

Tracker channels are known-zone zero-filled
--------------------------------------------
The physical trackers (PF/beam/PF-BMA-v2/beam-grid) and the spatial-prior
predictor only ever produce a value for a well's **evaluation-zone** rows
(that is their entire purpose -- estimating the unknown continuation past
the anchor); they are never run against known-zone rows, where the true
``TVT_input`` is already exact and running a tracker there would be a
tautology. So the 20 tracker/spatial/bank-delta columns in
:data:`SEQ_CHANNEL_NAMES` are **zero-filled on every known-zone row** and
carry each tracker's real per-row delta-vs-``pf_blend`` (or raw
uncertainty/margin) on eval-zone rows only -- exactly mirroring how the
boundary marker ``known_tvt_mask`` already tells a downstream model "this
row's ``tvt_input_delta_last`` is real signal, ignore the (deliberately
degenerate) tracker columns" and vice versa for eval rows. This is a
concrete, load-bearing design decision this script makes on the design
doc's behalf (Sec.3.1's table does not spell out the known-zone fill
value) -- 0.0 was chosen because it is the *only* value consistent with
each tracker column's own zero-point (a delta of 0 vs. ``pf_blend``, an
"agree" sign-agreement flag, an uncertainty of 0), not because it is
necessarily optimal for training; a downstream ablation could swap this for
NaN + a per-tracker-block missingness flag if the zero-fill hurts.

Storage format ("npz群" -- CSR/offsets-style, not per-well files)
--------------------------------------------------------------------
Design doc Sec.6 Step 1 names two acceptable formats -- "well毎npz" (one
``.npz`` per well) or "offsets+flat配列のCSR風単一npz" (a single CSR-style
npz of concatenated flat arrays + row offsets). This script uses the
latter, split across **two** files (a small "group", matching this task's
"npz群" instruction) rather than one, purely so the (large, ~4.9M-row)
per-row block and the (tiny, 773-row) well-static block can each use the
storage layout that fits them, without wasting ~590MB (773 wells x 21
static cols x 4.9M/773 avg rows x 4 bytes) broadcasting every well-static
scalar across that well's own rows on disk:

- ``nn_seq_channels.npz``: :data:`SEQ_CHANNEL_NAMES` (44,), ``X_seq``
  (total_rows, 44) float32, ``boundaries`` (n_wells+1,) int64 (CSR row
  offsets into ``X_seq``/``target``/... for well ``i`` ==
  ``X_seq[boundaries[i]:boundaries[i+1]]``, bank_builder.py's own
  convention), ``well_names`` (n_wells,), ``target``/``target_mask``/
  ``y_true``/``pf_blend`` (total_rows,).
- ``nn_static_channels.npz``: :data:`STATIC_CHANNEL_NAMES` (21,),
  ``X_static`` (n_wells, 21) float32, ``well_names`` (n_wells,, duplicated
  for self-containment), ``well_fold`` (n_wells,) int32, ``n_rows_per_well``
  / ``n_eval_rows_per_well`` (n_wells,) int64.

A downstream Dataset (design doc Sec.6 Step 2, not this script) broadcasts
``X_static[i]`` across well ``i``'s own rows at load time -- cheap (one
well at a time, not the whole 773-well matrix at once) and keeps this
script's own peak memory + on-disk footprint far under the design doc's
<2GB budget (see the module docstring's own well-count x 44-channel
estimate, ~1GB at full scale for ``X_seq`` alone; the static block adds
<1MB).

Leak boundary
-------------
Identical to ``notebooks/submission/kernel_stack_v2_blend/
stack_v2_blend_submission.py`` (this script ports the bulk of its tracker/
spatial/feature code verbatim from that file and from
``notebooks/submission/kernel_bank_builder/bank_builder.py``, see the
per-section attribution comments below): only ``MD, X, Y, Z, GR,
TVT_input`` are ever read from a well's own horizontal-well log for
per-row/per-well *features*. ``h["TVT"]`` is read exactly once per well,
only to build the ``target``/``y_true`` output arrays (never a feature
input to anything computed here). The spatial-prior surface bank reads the
train-only formation-depth columns (``ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA``)
of *other* wells only (leave-self-out KNN, ``exclude_well``), never the
target well's own -- see :func:`build_surface_bank`'s docstring for the
unchanged leak-safety analysis. Typewell ``Geology`` is never read.

Time budget
-----------
A ``GLOBAL_TIME_BUDGET_S`` guard (10.5h) is checked at the start of every
well's processing in the main loop: if exceeded, the loop stops early
(whatever wells were already accumulated are still written) rather than
losing all progress on a mid-run cutoff -- mirrors ``bank_builder.py``'s
per-phase budget guard. See this script's own printed per-well timing (and
the calling report) for the actual measured throughput; design doc Sec.4.4
independently estimated ~10s/well x 773 ~= 2.2h for this exact tracker
workload, comfortably inside the 10.5h guard even with the one-time
surface-bank build (``scripts/build_spatial_cache.py``: ~35min at full
773-well scale) added on top.

Attribution
-----------
Ported/adapted (byte-identical where noted) from:

- ``notebooks/submission/kernel_bank_builder/bank_builder.py`` (data-root
  discovery, ``list_wells``/``load_horizontal``/``load_typewell``/
  ``eval_mask``/``last_known_tvt``, ``tw_gr_lookup``/``fit_affine_gr``/
  ``apply_affine``, the PF/PF-BMA/beam/beam-grid tracker implementations)
- ``notebooks/submission/kernel_stack_v2_blend/stack_v2_blend_submission.py``
  (``track_pf_multi``, the spatial-prior predictor
  (``SurfaceBank``/``build_surface_bank``/``predict_spatial``/
  ``nn_distance_per_row``, IDW branch only -- see "Simplifications" below),
  ``WellTrackerSpatial``/``compute_well_tracker_spatial``,
  ``build_bank_delta_features``, and this file's config constants for
  ``N_PARTICLES``/``N_SEEDS``/``PF_SCALES``/``SPATIAL_STRIDE``/
  ``SPATIAL_K``/``PFBMA_*``/``BEAM_GRID_*``)
- ``src/rogii/features.py`` (``_robust_slope``/``_stat_block``/
  ``_typewell_gr_at``, and the position/trajectory/GR feature math this
  script's :func:`build_seq_features` generalizes from eval-zone-only rows
  to the full row range)
- ``src/rogii/stack_features.py`` (``build_tracker_features``/
  ``build_spatial_features``/well-aggregate math, byte-identical column
  definitions)
- ``cdeotte/nn-starter-cv-15-5`` on Kaggle (the full-row
  known+eval sequence I/O shape, ``known_tvt_mask``/``TVT_input_delta_last``
  boundary-marker convention this script's own ``tvt_input_delta_last``
  channel follows)

Simplifications vs. the ported sources (all documented, none silent)
------------------------------------------------------------------------
- Spatial-prior interpolation: only the ``"idw"`` method is ported (the
  ``"plane"`` method exists in ``stack_v2_blend_submission.py`` but
  ``SPATIAL_METHOD = "idw"`` is the only value any adopted config has ever
  used -- ``scripts/build_spatial_cache.py``'s own module constant).
- The "composite" (inverse-RMSE-weighted average across formations)
  spatial-prior candidate is not computed -- only the single
  lowest-``prefix_rmse`` formation's path (``SpatialResult.tvt``), matching
  what :func:`build_spatial_features`'s ``spatial_d``/``spatial_gated_d``
  actually consume in the ported stack scripts (``tvt_composite`` is unused
  there too).
- The beam-DP grid tracker only computes the single ``"mid_cal_sm2_mm3"``
  configuration (the one R7's ``beam_grid_d`` feature actually uses), not
  ``bank_builder.py``'s full 10-configuration ``CONFIG_GRID``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #

KAGGLE_INPUT = Path("/kaggle/input/rogii-wellbore-geology-prediction")

GLOBAL_TIME_BUDGET_S = 10.5 * 3600.0
PROGRESS_EVERY = 50

PF_BLEND_W = 0.7

# PF tracker (src/rogii/registration/particle.py defaults ==
# scripts/build_tracker_cache.py's bare track_pf_multi(h, tw) call -- the
# same PF this well's pf_d/pf_std/anchor/pf_blend channels are built from).
N_PARTICLES = 512
N_SEEDS = 8
PF_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)

# Beam tracker: DEFAULT_BEAM_CONFIGS (3-config move_penalty x mismatch_scale
# ensemble, products 2400/3000/3750 at mismatch_scale=150).

# Spatial-prior predictor (scripts/build_spatial_cache.py's config, the
# exact one data/processed/spatial_cache/*.npz was built with).
SPATIAL_STRIDE = 10
SPATIAL_K = 10

# PF-BMA tracker (bank_builder.py v2 Phase B config, verbatim -- builds the
# same pf_bma_scale{3,5,8,12}_{tvt,std} arrays outputs/path_bank_pfbma_v2.npz
# holds, which run_stack_v23_cv.py trained R7/G1/G2 on).
PFBMA_N_PARTICLES = 512
PFBMA_N_SEEDS = 12
PFBMA_GR_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)
PFBMA_BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
PFBMA_INIT_POS_STD = 2.0  # v2 bank value; the well's OWN PF tracker keeps 0.3

# Beam-grid tracker: bank_builder.py CONFIG_GRID's "mid_cal_sm2_mm3" entry
# (index 4), the one R7's beam_grid_d feature uses. mismatch_scale=150 ->
# move_penalty = 3000 / 150 = 20.0.
BEAM_GRID_MOVE_PENALTY = 20.0
BEAM_GRID_MISMATCH_SCALE = 150.0
BEAM_GRID_MAX_MOVE_PER_ROW = 3
BEAM_GRID_GR_CALIBRATION = True
BEAM_GRID_SMOOTH_RADIUS = 2


def _dataset_limit() -> int | None:
    """``ROGII_NN_DATASET_LIMIT=<N>`` env var: restrict to the first N train wells."""
    raw = os.environ.get("ROGII_NN_DATASET_LIMIT")
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
# notebooks/submission/kernel_bank_builder/bank_builder.py).
# --------------------------------------------------------------------------- #


def find_data_root() -> Path:
    """Locate the directory containing ``train/``, ``test/``, ``sample_submission.csv``."""
    candidates = [KAGGLE_INPUT]
    try:
        here = Path(__file__).resolve()
        # Locally this file sits at notebooks/submission/kernel_nn_dataset/<name>.py,
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
    """Where to write the two dataset ``.npz`` files.

    ``ROGII_NN_DATASET_LIMIT`` set (smoke mode) -> ``./outputs_smoke_nn/``
    (never touches ``/kaggle/working`` or this repo's real ``outputs/``).
    Kaggle (``/kaggle/working`` exists) -> that directory. Otherwise (a bare
    local full run, discouraged -- see module docstring) -> a scratch
    subdirectory under this repo's git-ignored ``outputs/``.
    """
    if _dataset_limit() is not None:
        d = Path.cwd() / "outputs_smoke_nn"
        d.mkdir(parents=True, exist_ok=True)
        return d
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    d = _repo_root_safe() / "outputs" / "nn_dataset_local"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Embedded 773-well well->fold table (see module docstring point 4). Sorted
# by well name; _WELL_FOLD_NAMES is the 773 8-char well hashes concatenated
# 8-per-line as adjacent string literals (auto-concatenated by Python),
# _WELL_FOLD_DIGITS is the parallel 773 single-digit (0..4) fold ids.
# Regeneration recipe (run locally, NEVER on Kaggle -- outputs/stack_v2_oof.npz
# does not exist there): load that npz's used_wells/well_idx/row_fold, verify
# every well's row block carries one consistent fold id, sort by well name,
# re-emit these two constants.
# --------------------------------------------------------------------------- #

_WELL_FOLD_NAMES: str = (
    "000d7d20" "00bbac68" "00e12e8b" "015fe0d2" "01869cd4" "01982c1d" "028d7b28" "02e7fe5a"
    "0390d174" "03a935ae" "044af7d1" "0498acab" "052d64df" "05948241" "059c8f24" "05a0ee4d"
    "060ab2b8" "06df5958" "071d7b45" "0849b4e2" "084d66f9" "08fecf00" "09053135" "09441b8d"
    "0955cd6c" "09ec2ca9" "0a57a29c" "0bbf5e67" "0be01c24" "0c1b160c" "0d99f8e4" "0dc5e64d"
    "0dc835f3" "0dd99dc5" "0e5e560d" "109d9d70" "10a1281a" "10b89021" "10be2420" "113011eb"
    "1131525f" "11d0f5ac" "12203f2a" "122bd617" "125be846" "1285a37b" "1295a25b" "12f0cd90"
    "137d1e44" "13ce113d" "13f598d9" "14a53cb3" "14ab73fb" "14ad5efb" "14fee784" "153552f5"
    "1550491d" "155887bb" "1590af81" "15e154a1" "1619e2ca" "16e4a047" "173ed841" "1770d7c6"
    "177622d7" "18d24f7b" "19137a89" "1944513c" "197f8a5a" "19871e7f" "198a5607" "19fb4f7b"
    "1a518997" "1a68190b" "1a730ac2" "1aaf1da0" "1b08ed48" "1b1c5372" "1b1eba53" "1b6ba517"
    "1b82b665" "1bcb2fcf" "1beca81d" "1d3fbf02" "1d78281a" "1e30dedf" "1e5e3573" "1f9afacb"
    "1fdbba44" "1fee2b62" "20096916" "200d233d" "201661b5" "203721ab" "20402d71" "2043abc5"
    "204cc64b" "206b6193" "2075c696" "22c5f93f" "230eaaa3" "2364716c" "23b9beb0" "24d8997e"
    "25050f63" "25939962" "25fd32b3" "261785ee" "264feceb" "26623a50" "26d3a96a" "272abef3"
    "27300fd5" "276b012a" "27c3155b" "27ebb9b9" "283269ac" "28473855" "28f4eda9" "29b24409"
    "29de15c8" "2ab29395" "2ac7cb6b" "2acc78d4" "2b034622" "2b06ad65" "2b547943" "2be5c96a"
    "2bff06cf" "2c0c4a4e" "2cee0cba" "2d0c268a" "2d179ded" "2d196986" "2d2d0c6b" "2d35f86d"
    "2d9f6cb9" "2ddad940" "2e29e833" "2e43c2ea" "2e63d9de" "2ee0235b" "2f19d536" "2f8e53c3"
    "2fa01aa6" "2fd68f7b" "2fe023be" "30d75030" "310c71a5" "32fe84c9" "33c40295" "3407f5bf"
    "3417285d" "347242c8" "353e5502" "357c12f9" "3584a6eb" "35b30c7f" "35b3ef6a" "367456ce"
    "368131f9" "368fe682" "3690a146" "37344c2a" "373ef48b" "374be387" "37d36812" "38991fd4"
    "389ae58f" "3932faa6" "398dce4b" "39b51819" "3a0af893" "3a223519" "3a7dd95d" "3a824aa7"
    "3a86fe8d" "3aaa2b30" "3ac16ad0" "3b21ea64" "3bbe1f5d" "3bfee975" "3c3dbcbc" "3d96d897"
    "3df4b19e" "3e011332" "3e678ea7" "3ef6372a" "3ef98781" "3f603129" "404c4384" "4121c517"
    "41fb0192" "42669188" "42c538a1" "43b06c70" "43c7f85a" "43e16325" "44441e54" "445af6c2"
    "4463446c" "44a94013" "45fb3e21" "460f859e" "4630ab92" "463c38f2" "4654e7de" "466fc788"
    "46a301e9" "46a7190b" "46d24a09" "46d41ff6" "46dfcfca" "47222616" "47ee27ab" "481de5fc"
    "48c1f673" "4936af16" "493b5b31" "49dbc14c" "4a035ec2" "4a335117" "4a8ecc0b" "4b462989"
    "4b831d20" "4c2208f5" "4c3168cf" "4c3df468" "4caa7289" "4cd1ac61" "4cecf7b3" "4ddde4c1"
    "4e050c92" "4ed93db6" "4f3eb9e9" "4f4ac5ce" "4f4afcc6" "4f69febf" "4fbe9c06" "504c8b08"
    "511e1db0" "512943b2" "5138a660" "516088b7" "521a7819" "5254b6db" "529e88ca" "52f1e77a"
    "5305524b" "53134831" "5397ceb1" "53f23031" "543198e8" "54753541" "54a7e3c2" "5542c301"
    "5567d551" "55efee7f" "5663b2b7" "5693dde2" "56b00794" "57796579" "577eec7c" "57f05c51"
    "58850d2c" "58fa8486" "591cc951" "593876b6" "59f8b2e8" "5a1a8fd8" "5a5269a9" "5a5d1982"
    "5a63b5d1" "5aa03df7" "5aa403d0" "5aef5c6c" "5b7b6324" "5bd25f59" "5beded95" "5cb483ac"
    "5cdd8dbc" "5d11dfb5" "5d7198fd" "5def1ce5" "5eae34a8" "5f4d2a52" "5f6756c5" "5f75f796"
    "5fb1c15f" "5fffa282" "60e37807" "61f27424" "62569ae3" "62a6f539" "633774dc" "63559d57"
    "647f2a41" "65156b96" "653612bc" "656a067b" "65a5466a" "660d9546" "6767c111" "67847288"
    "67a8da4a" "684a6fc1" "698930d1" "699ea575" "69c1bdad" "69e41fa6" "6a0ff78e" "6a8fa194"
    "6ae68655" "6d1d74e1" "6d590e26" "6d6d93af" "6dbe4b60" "6e6d122c" "6e9ccd38" "6ea22614"
    "6ede926a" "708caea9" "70925e23" "70e1788b" "71642c7d" "71ccf778" "7224331b" "722cf0d8"
    "7256229b" "726f3dc2" "7271dd80" "727a3a10" "729665c0" "729e9750" "729f4217" "72dd8501"
    "73348914" "739a914f" "73b1447d" "74b6186b" "75c20ec7" "75cd5f11" "75d20a82" "76201745"
    "764072a5" "77b0d905" "77e4821c" "7850c72e" "78a4a386" "79507629" "796dbd4c" "7987f2f2"
    "7993a768" "7a5660b0" "7acda2df" "7b2c7b3c" "7b38844c" "7bb17b96" "7c607683" "7cbbd047"
    "7cd4bb31" "7d50373e" "7d57c75d" "7e208414" "7e721392" "7ff89f8f" "8050c789" "807f0298"
    "811f52b3" "81bf5923" "81fc01c6" "8255c867" "82aab6d2" "82fcf37c" "833af382" "8382f4b6"
    "84633df4" "8478df29" "84815d5a" "84c3b497" "85380836" "8612b37f" "861e71df" "86454a6f"
    "8648abae" "865033f8" "87695469" "877ed19c" "87aa3730" "884ecb5f" "8902c3f6" "896d15b9"
    "8995c945" "89969c7a" "89f1085d" "89f36adf" "89fdb2f2" "8a3da6d1" "8a89da23" "8a9c8932"
    "8ac2f237" "8b12bab6" "8b5be31b" "8b95d6d1" "8b9d8326" "8bb9c1e6" "8bda1a11" "8bfa881d"
    "8c167025" "8c8348e8" "8cc21f01" "8cc800b3" "8d5d46d7" "8d92cb49" "8f201368" "90386ee4"
    "91abc4c7" "91b301ce" "91db7070" "925a7cd1" "9283ae69" "9298ad5b" "9314ff13" "93209a3d"
    "939d9c34" "93f5d2e6" "940a48d9" "940e709a" "9426ec1e" "94467f50" "944f36b9" "94d813a4"
    "95b559e7" "95c8427f" "96936c22" "96ae5806" "96b7eea1" "9719aa04" "974802e7" "97cd5bf9"
    "98177ced" "9896fc0b" "98bc6f66" "992c99e8" "992ce078" "99529c45" "995ff498" "999daf80"
    "9a6b0392" "9a8ae0d6" "9a95e33f" "9ab94eeb" "9ad5410a" "9ae248be" "9af1ffd0" "9b5b61c1"
    "9b9e3b49" "9bd2bba3" "9c211c53" "9c350a6a" "9c8375f7" "9d072e4c" "9d3ec64c" "9d4272e3"
    "9d5e14d4" "9da441ba" "9dede370" "9def2eb2" "9df1b1f9" "9dfff011" "9e3e364d" "9efc812e"
    "9f0c7bae" "9ffce529" "a0383629" "a056d01e" "a08d9e63" "a0b83014" "a0e92ed6" "a15bc102"
    "a247e7cf" "a2e8e7f6" "a3518960" "a3f71395" "a4719920" "a48640d9" "a4cdbb0f" "a4f989c2"
    "a5aa7973" "a5e6f7dc" "a612d5a0" "a622ce4d" "a645da9a" "a692b17e" "a6b8ac67" "a6ed2eff"
    "a6f967fb" "a76db406" "a7744f5b" "a783cc24" "a79fd2e7" "a7fc3e6b" "a85bb86f" "a85e4bc3"
    "a87433c9" "a8ed028a" "a959858c" "a9c9b150" "aa95015b" "aaaf3c03" "aaeffccb" "ab112491"
    "ab18d3fc" "ab3ced07" "ab6fe95d" "ab94878e" "abf460d2" "ac6f01d5" "ad2133a8" "add9c322"
    "ae069086" "ae0784a8" "ae35f33a" "ae8959c3" "aed44918" "aee6393a" "af7a59ce" "aff4e561"
    "b04b58a3" "b0d42b0d" "b0f53bf4" "b19b0395" "b1efb8a4" "b32902c0" "b32e4903" "b3388334"
    "b37fd114" "b38e3116" "b4d8dd4f" "b4f37e6e" "b56bc0aa" "b585b4fa" "b5aa9634" "b7974e66"
    "b7cdcecb" "b7fea9ef" "b85f7b06" "b881d27e" "b8c49c1a" "b95e7121" "b977be4a" "ba48188d"
    "bb2cdd83" "bb337fa0" "bb682ebd" "bc4381e2" "bcebcc5f" "bd2c2f34" "bd4ae5da" "bd6f0e19"
    "bd8f847e" "be063506" "be35c8f2" "be83e781" "beda8696" "bf39bc20" "bf70b410" "bffe3082"
    "c03b9305" "c03d4aed" "c03fba65" "c07fd0cd" "c1708b88" "c1d046f4" "c2717d4f" "c2c4db09"
    "c2c6fc71" "c36625df" "c3957531" "c42fe306" "c43812e2" "c472c0b5" "c47e0767" "c50b42f6"
    "c593337f" "c59b6c4a" "c66be2b8" "c6797dd7" "c6b782a4" "c6c96179" "c70c0e01" "c84c65be"
    "c8d9680c" "c908edd0" "c9578d27" "c96c018a" "c99f9fae" "c9e6734f" "c9e980e8" "c9fedcf3"
    "cad13e8f" "cb34231f" "cb879f40" "cbb406b5" "cbe51de1" "cbe62450" "cc08aa63" "cc8a1d57"
    "ccedb12b" "cd7f1687" "cda7eb37" "cdc31d65" "cdc60a53" "ce55ba43" "ce8399b7" "cf50c9d1"
    "d00e7eb9" "d011f41b" "d07aed8f" "d085e611" "d0a0e7b5" "d0a43760" "d0cb63ae" "d114d9c2"
    "d1457cc5" "d1ecf309" "d1ee630d" "d217e0c5" "d24ff243" "d273674e" "d2f3b1ab" "d3df146b"
    "d40f61d5" "d41beaf2" "d463e527" "d4f60a28" "d50c649a" "d543916f" "d5df0ff5" "d60430e6"
    "d63196fe" "d6c320d9" "d6dd7330" "d7ba4f9d" "d7eb0be8" "d84b621b" "d90aa14c" "d924e971"
    "d9ae59b2" "d9d6d94d" "d9eca87b" "da2d4d7c" "da3bfe4e" "da3cf798" "da940160" "dafc88c3"
    "dbee6c86" "dc5fbe29" "dc7da28f" "dc7f9757" "dd168de0" "dd64382d" "dd6bc5b2" "dd7d638e"
    "ddc790ff" "df71d1ff" "df73b8f3" "df8290be" "e0079ed1" "e03b45fd" "e0f36d98" "e149204f"
    "e14d641e" "e161bd0c" "e201fd6d" "e25f1537" "e2fc7745" "e32da4ec" "e3b34f5b" "e45a2a62"
    "e46f4ef4" "e4726608" "e5864244" "e5c92e59" "e5ff9fd2" "e6c55748" "e730c32b" "e737965a"
    "e748dd88" "e7818f7a" "e7c36e81" "e7d0c1de" "e85074c5" "e97032ad" "e9949a23" "ea0b99b7"
    "ea3a0e38" "ea41324e" "eabc711b" "eac10e86" "eafb863e" "eb06a4e7" "eb640a35" "eb934281"
    "eba6605e" "ebe69318" "ec0d597b" "ec13af42" "ecdab904" "ecdaf714" "ed6436d6" "ed6e6e54"
    "edf885a3" "ee0300f7" "ee288beb" "ee4aecac" "eeda36f5" "eeed308d" "ef04658e" "ef8e3ed0"
    "efde6ac3" "efe96181" "f0188a48" "f021b650" "f03f10fe" "f053a1b9" "f074d277" "f07d7037"
    "f08774c3" "f140c2fa" "f16e25e3" "f2d4c8c9" "f321a31c" "f40cd091" "f439a621" "f49fdea3"
    "f4d12d23" "f56d3ace" "f5859199" "f5d8aab6" "f631da6e" "f6824bb0" "f6bc699b" "f6d009f4"
    "f72b268f" "f8662c22" "f88ddb26" "f8afa78a" "f8f59188" "f9fc81aa" "fa16d114" "fa31da94"
    "fa667be2" "fae0c593" "fb03ae90" "fb0904bd" "fb3848a1" "fb73b13a" "fba7683c" "fbd68d27"
    "fc03f21f" "fc0d20b2" "fcfcc902" "fd3b4faa" "fd710aea" "fd8f77fa" "fde20ecf" "fdfd57da"
    "febb4411" "fef8af96" "ff0aea78" "ff8bb73a" "ffefef30"
)

_WELL_FOLD_DIGITS: str = (
    "323241224432323120204041304431021141301434000013103230111320044120124342122340000231423104"
    "020224344001103034242102112121120001222231304430020001143101331440443104413141233332140134"
    "024142341040434030410132320202043211130001130122133314241422344311422121040134210223444232"
    "121433331313133214131341323142322201441014432003213240440004024121001134314312330030033001"
    "040214240203301144430021120141322343404433023144344133130132313404023221200340011132004432"
    "011440223002142213444430411321140433034022121310211230433402220232012344240441320200431310"
    "102433321313022231033042043130111434323403344210221424024411443433043414200402211341004030"
    "231142203320324021031211221204030300420112420100332114240434212220034122204031422411020444"
    "11202242432131203120232303344340134014143023012244321"
)


def _parse_well_fold_map() -> dict[str, int]:
    # _WELL_FOLD_NAMES is a single concatenated string (adjacent string
    # literals auto-concatenate in Python, see the constant above) of 773
    # 8-char well hashes back-to-back; slice it into individual hashes.
    flat = _WELL_FOLD_NAMES
    names = tuple(flat[i : i + 8] for i in range(0, len(flat), 8))
    if len(names) != 773 or len(_WELL_FOLD_DIGITS) != 773:
        raise AssertionError(
            f"embedded well->fold table must have 773 entries, got "
            f"{len(names)} names / {len(_WELL_FOLD_DIGITS)} digits"
        )
    return {w: int(d) for w, d in zip(names, _WELL_FOLD_DIGITS, strict=True)}


WELL_FOLD_MAP: dict[str, int] = _parse_well_fold_map()
assert len(WELL_FOLD_MAP) == 773, f"expected 773 unique well hashes, got {len(WELL_FOLD_MAP)}"
assert set(WELL_FOLD_MAP.values()) == {0, 1, 2, 3, 4}, "fold ids must be exactly {0,1,2,3,4}"


# --------------------------------------------------------------------------- #
# Ported from src/rogii/data.py (bank_builder.py's copy).
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
# Ported from src/rogii/typewell.py.
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


def _safe_tw_gr_lookup(tw: pd.DataFrame):
    """Like :func:`tw_gr_lookup` but never raises on a typewell with no usable GR.

    ``np.interp`` requires a non-empty sample-point array; a typewell whose
    ``GR`` column is entirely NaN (or absent) would otherwise crash the
    per-row :func:`build_seq_features` carry-forward lookup. Returns NaN for
    every query in that case (kept as NaN, not 0.0 -- this is a brand-new
    channel with no established fallback convention, unlike
    ``_typewell_gr_at`` below which matches ``src/rogii/features.py``'s
    existing 0.0 default for parity with the already-audited stack feature).
    """
    if not {"TVT", "GR"}.issubset(tw.columns):
        return lambda q: np.full(np.shape(q), np.nan, dtype=float)
    valid = tw.dropna(subset=["GR"]).sort_values("TVT")
    if valid.empty:
        return lambda q: np.full(np.shape(q), np.nan, dtype=float)
    tvt = valid["TVT"].to_numpy(dtype=float)
    gr = valid["GR"].to_numpy(dtype=float)
    return lambda q: np.interp(q, tvt, gr)


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
# Ported from src/rogii/features.py.
# --------------------------------------------------------------------------- #

_GR_ROLL_WINDOWS = (11, 51, 151)
_TRAJ_ROLL_WINDOW = 51
_KNOWN_TAIL_LONG = 200
_KNOWN_TAIL_SHORT = 50
_MIN_SLOPE_ROWS = 2
_MIN_SLOPE_X_STD = 1e-9


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y ~ x``; ``0.0`` when degenerate."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < _MIN_SLOPE_ROWS:
        return 0.0
    xs, ys = x[mask], y[mask]
    if np.std(xs) < _MIN_SLOPE_X_STD:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def _stat_block(values: np.ndarray) -> dict[str, float]:
    """min/max/range/mean/std over finite ``values``; all-``0.0`` when degenerate."""
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
    """``twGR(tvt_query)``, or ``0.0`` when the typewell has no usable GR.

    Byte-identical to ``src/rogii/features.py::_typewell_gr_at`` -- kept as
    the anchor-scalar lookup so ``gr_minus_twgr_anchor_calibrated`` matches
    the already-audited stack v2.3 feature exactly; see
    :func:`_safe_tw_gr_lookup` above for the array-valued, NaN-fallback
    sibling this script's new ``tw_gr_at_carry`` channel uses instead.
    """
    if not {"TVT", "GR"}.issubset(tw.columns) or tw.empty or not np.isfinite(tvt_query):
        return 0.0
    lookup = tw_gr_lookup(tw)
    return float(lookup(np.array([tvt_query]))[0])


# --------------------------------------------------------------------------- #
# Full-row (known-zone + eval-zone) sequence channel builder. Generalizes
# src/rogii/features.py::build_features from "one row per eval-zone row" to
# "one row per row of the well", per docs/r10_nn_design.md Sec.1.1/3.1 (the
# NN sees known-zone rows as encoder context, not just eval-zone rows).
# --------------------------------------------------------------------------- #


@dataclass
class WellStaticStats:
    """Known-prefix scalar stats -- shared by the per-row typewell-carry
    channel (``slope_md_last200``) and the well-level static block."""

    anchor_tvt: float
    slope_md_all: float
    slope_md_last200: float
    slope_md_last50: float
    slope_z_all: float
    known_tvt_min: float
    known_tvt_max: float
    known_tvt_range: float
    known_tvt_mean: float
    known_tvt_std: float
    n_known_rows: float


# Fixed column order for build_seq_features's output (24 columns).
SEQ_P1_COLUMNS: tuple[str, ...] = (
    "known_tvt_mask",
    "tvt_input_delta_last",
    "md_from_ps",
    "dx_from_ps",
    "dy_from_ps",
    "dz_from_ps",
    "dist3d_from_ps",
    "row_frac",
    "dz_dmd_roll51",
    "dx_dmd_roll51",
    "dy_dmd_roll51",
    "gr",
    "gr_missing",
    "gr_diff_1",
    "gr_diff_10",
    "gr_minus_twgr_anchor_calibrated",
    "tw_gr_at_carry",
    "gr_minus_tw_gr_at_carry",
    "gr_roll_mean_11",
    "gr_roll_std_11",
    "gr_roll_mean_51",
    "gr_roll_std_51",
    "gr_roll_mean_151",
    "gr_roll_std_151",
)


def build_seq_features(h: pd.DataFrame, tw: pd.DataFrame) -> tuple[pd.DataFrame, WellStaticStats]:
    """Build the 24 :data:`SEQ_P1_COLUMNS` for EVERY row of ``h`` (not just eval-zone).

    Returns ``(channels_df, static_stats)`` -- ``static_stats`` is reused
    verbatim by the well-level static block (:func:`build_static_block`) so
    the known-prefix slope/stat computation happens exactly once per well.
    """
    n = len(h)
    known_mask = h["TVT_input"].notna().to_numpy()

    md = h["MD"].to_numpy(dtype=float)
    x = h["X"].to_numpy(dtype=float)
    y = h["Y"].to_numpy(dtype=float)
    z = h["Z"].to_numpy(dtype=float)
    gr = h["GR"].to_numpy(dtype=float) if "GR" in h.columns else np.full(n, np.nan)
    tvt_input = h["TVT_input"].to_numpy(dtype=float)

    known_md = md[known_mask]
    known_x = x[known_mask]
    known_y = y[known_mask]
    known_z = z[known_mask]
    known_tvt = tvt_input[known_mask]
    n_known = int(known_mask.sum())

    anchor_tvt = float(known_tvt[-1]) if n_known > 0 else float("nan")
    ps_md = float(known_md[-1]) if n_known > 0 else (float(md[0]) if n else 0.0)
    ps_x = float(known_x[-1]) if n_known > 0 else (float(x[0]) if n else 0.0)
    ps_y = float(known_y[-1]) if n_known > 0 else (float(y[0]) if n else 0.0)
    ps_z = float(known_z[-1]) if n_known > 0 else (float(z[0]) if n else 0.0)

    slope_md_all = _robust_slope(known_md, known_tvt)
    tail200 = min(_KNOWN_TAIL_LONG, n_known)
    slope_md_last200 = _robust_slope(known_md[-tail200:], known_tvt[-tail200:]) if tail200 else 0.0
    tail50 = min(_KNOWN_TAIL_SHORT, n_known)
    slope_md_last50 = _robust_slope(known_md[-tail50:], known_tvt[-tail50:]) if tail50 else 0.0
    slope_z_all = _robust_slope(known_z, known_tvt)
    tvt_stats = _stat_block(known_tvt)

    # --- trajectory derivatives, over ALL rows (not just eval) ---
    dmd = np.diff(md, prepend=md[0] if n else 0.0)
    dmd_safe = np.where(np.abs(dmd) < 1e-9, np.nan, dmd)
    dz_dmd = np.diff(z, prepend=z[0] if n else 0.0) / dmd_safe
    dx_dmd = np.diff(x, prepend=x[0] if n else 0.0) / dmd_safe
    dy_dmd = np.diff(y, prepend=y[0] if n else 0.0) / dmd_safe

    def _roll_mean(a: np.ndarray) -> np.ndarray:
        return pd.Series(a).rolling(_TRAJ_ROLL_WINDOW, min_periods=1, center=True).mean().to_numpy()

    dz_roll = _roll_mean(dz_dmd)
    dx_roll = _roll_mean(dx_dmd)
    dy_roll = _roll_mean(dy_dmd)

    # --- GR rolling stats, over ALL rows ---
    gr_series = pd.Series(gr)
    roll_feats: dict[str, np.ndarray] = {}
    for w in _GR_ROLL_WINDOWS:
        roll = gr_series.rolling(w, min_periods=1, center=True)
        roll_feats[f"gr_roll_mean_{w}"] = roll.mean().to_numpy()
        roll_feats[f"gr_roll_std_{w}"] = roll.std().to_numpy()
    gr_diff1 = gr_series.diff(1).to_numpy()
    gr_diff10 = gr_series.diff(10).to_numpy()

    # --- typewell GR at anchor, raw and affine-calibrated (matches
    # src/rogii/features.py::build_features exactly) ---
    twgr_anchor = _typewell_gr_at(tw, anchor_tvt)
    if "GR" in h.columns:
        gain, offset = fit_affine_gr(h, tw)
    else:
        gain, offset = 1.0, 0.0
    twgr_anchor_calibrated = float(apply_affine(np.array([twgr_anchor]), gain, offset)[0])
    gr_minus_twgr_anchor_calibrated = gr - twgr_anchor_calibrated

    # --- NEW: tw_gr_at_carry (docs/r10_nn_design.md Sec.3.1) -- per-row
    # single-slope carry-forward TVT (same _RATE_CAL_ROWS=200 window the PF
    # motion-model init uses), typewell-GR-looked-up ---
    carry_tvt = anchor_tvt + slope_md_last200 * (md - ps_md)
    tw_gr_at_carry = _safe_tw_gr_lookup(tw)(carry_tvt)
    gr_minus_tw_gr_at_carry = gr - tw_gr_at_carry

    # --- position (anchor-relative only), over ALL rows ---
    row_frac = np.arange(n, dtype=float) / max(n - 1, 1)
    md_from_ps = md - ps_md
    dx_from_ps = x - ps_x
    dy_from_ps = y - ps_y
    dz_from_ps = z - ps_z
    dist3d_from_ps = np.sqrt(dx_from_ps**2 + dy_from_ps**2 + dz_from_ps**2)

    # --- boundary markers (cdeotte cell 8 convention) ---
    known_tvt_mask_f = known_mask.astype(np.float64)
    tvt_input_delta_last = np.where(known_mask, tvt_input - anchor_tvt, 0.0)

    feats: dict[str, np.ndarray] = {
        "known_tvt_mask": known_tvt_mask_f,
        "tvt_input_delta_last": tvt_input_delta_last,
        "md_from_ps": md_from_ps,
        "dx_from_ps": dx_from_ps,
        "dy_from_ps": dy_from_ps,
        "dz_from_ps": dz_from_ps,
        "dist3d_from_ps": dist3d_from_ps,
        "row_frac": row_frac,
        "dz_dmd_roll51": dz_roll,
        "dx_dmd_roll51": dx_roll,
        "dy_dmd_roll51": dy_roll,
        "gr": gr,
        "gr_missing": (~np.isfinite(gr)).astype(np.float64),
        "gr_diff_1": gr_diff1,
        "gr_diff_10": gr_diff10,
        "gr_minus_twgr_anchor_calibrated": gr_minus_twgr_anchor_calibrated,
        "tw_gr_at_carry": tw_gr_at_carry,
        "gr_minus_tw_gr_at_carry": gr_minus_tw_gr_at_carry,
    }
    feats.update(roll_feats)

    out = pd.DataFrame(feats)[list(SEQ_P1_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan).astype(np.float32)

    static = WellStaticStats(
        anchor_tvt=anchor_tvt,
        slope_md_all=slope_md_all,
        slope_md_last200=slope_md_last200,
        slope_md_last50=slope_md_last50,
        slope_z_all=slope_z_all,
        known_tvt_min=tvt_stats["min"],
        known_tvt_max=tvt_stats["max"],
        known_tvt_range=tvt_stats["range"],
        known_tvt_mean=tvt_stats["mean"],
        known_tvt_std=tvt_stats["std"],
        n_known_rows=float(n_known),
    )
    return out, static


# --------------------------------------------------------------------------- #
# Ported from src/rogii/registration/particle.py (PF + PF-BMA trackers) and
# notebooks/submission/kernel_stack_v2_blend/stack_v2_blend_submission.py's
# track_pf_multi. See those modules' docstrings for full attribution
# (adapted from the public Kaggle notebook lightningv08/lb-7-776-rogii-
# ridge-sp; PF-BMA reimplements bernubritz/rogii-lb7295-public-rebuild).
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
    n_particles: int = N_PARTICLES,
    scales: tuple[float, ...] = PF_SCALES,
    seed: int = 0,
    resample_ess: float = 0.5,
) -> PFResult:
    """Sequential Monte Carlo (particle filter) GR registration tracker. Never raises."""
    try:
        return _track_pf_impl(h, tw, n_particles, scales, seed, resample_ess)
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _pf_flat_fallback(n, anchor)


def track_pf_multi(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    n_seeds: int = N_SEEDS,
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
    init_pos_std: float,
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

            avg_lk = float(np.dot(weights, likelihood))
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
    init_pos_std: float,
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

    ll_shifted = log_liks - log_liks.max()

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
    bma_scales: tuple[float, ...] = PFBMA_BMA_SCALES,
    gr_scales: tuple[float, ...] = PFBMA_GR_SCALES,
    seed_base: int = 0,
    resample_ess: float = 0.5,
    init_pos_std: float = PFBMA_INIT_POS_STD,
) -> PFBMAResult:
    """Likelihood-weighted PF-BMA: seed-softmax consensus over independent PF runs. Never raises."""
    try:
        return _track_pf_bma_impl(
            h, tw, n_particles, n_seeds, bma_scales, gr_scales, seed_base, resample_ess,
            init_pos_std,
        )
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _flat_fallback_bma(n, anchor, bma_scales, n_seeds)


# --------------------------------------------------------------------------- #
# Ported from src/rogii/registration/beam.py + beam_grid.py (full-grid
# Viterbi DP tracker). Adapted from lightningv08/lb-7-776-rogii-ridge-sp and
# nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based.
# --------------------------------------------------------------------------- #

_DEFAULT_GRID_STEP_FT = 0.2
_DEFAULT_GRID_RANGE_FT = 120.0
_DEFAULT_MAX_MOVE_PER_ROW = 3
_MIN_TYPEWELL_SAMPLES = 2


@dataclass
class BeamResult:
    tvt: np.ndarray
    margin: np.ndarray
    total_cost: float


@dataclass(frozen=True)
class BeamConfig:
    move_penalty: float
    mismatch_scale: float
    max_move_per_row: int = _DEFAULT_MAX_MOVE_PER_ROW
    weight: float = 1.0


# Best measured full-773-well configuration (analysis/experiment_ledger.md):
# products 2400/3000/3750 at mismatch_scale=150.
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
    move_penalty: float = 20.0,
    mismatch_scale: float = 150.0,
    max_move_per_row: int = _DEFAULT_MAX_MOVE_PER_ROW,
) -> BeamResult:
    """Register a well's GR log onto the typewell TVT axis via full-grid DP. Never raises."""
    try:
        return _track_beam_impl(
            h, tw, grid_step_ft, grid_range_ft, move_penalty, mismatch_scale, max_move_per_row
        )
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _beam_flat_fallback(n, anchor)


def track_beam_multi(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    configs: tuple[BeamConfig, ...] | None = None,
) -> BeamResult:
    """Weighted-average ensemble of :func:`track_beam` over several ``configs``. Never raises."""
    try:
        resolved = configs if configs is not None else DEFAULT_BEAM_CONFIGS
        results = [
            track_beam(
                h,
                tw,
                move_penalty=c.move_penalty,
                mismatch_scale=c.mismatch_scale,
                max_move_per_row=c.max_move_per_row,
            )
            for c in resolved
        ]
        weights = np.array([c.weight for c in resolved], dtype=float)
        wsum = weights.sum()
        weights = weights / wsum if wsum > 0 else np.full(len(resolved), 1.0 / len(resolved))

        tvt_stack = np.stack([r.tvt for r in results])
        margin_stack = np.stack([r.margin for r in results])
        tvt = np.tensordot(weights, tvt_stack, axes=(0, 0))
        margin = np.tensordot(weights, margin_stack, axes=(0, 0))
        total_cost = float(np.dot(weights, np.array([r.total_cost for r in results])))
        return BeamResult(tvt=tvt, margin=margin, total_cost=total_cost)
    except Exception:
        n, anchor = _safe_n_and_anchor(h)
        return _beam_flat_fallback(n, anchor)


@dataclass(frozen=True)
class BeamGridConfig:
    move_penalty: float
    mismatch_scale: float
    max_move_per_row: int = 3
    gr_calibration: bool = True
    smooth_radius: int = 0
    grid_step_ft: float = _DEFAULT_GRID_STEP_FT
    grid_range_ft: float = _DEFAULT_GRID_RANGE_FT


# bank_builder.py CONFIG_GRID's "mid_cal_sm2_mm3" entry -- the one R7's
# beam_grid_d feature actually uses (see module docstring's Simplifications).
BEAM_MID_CAL_SM2_MM3 = BeamGridConfig(
    move_penalty=BEAM_GRID_MOVE_PENALTY,
    mismatch_scale=BEAM_GRID_MISMATCH_SCALE,
    max_move_per_row=BEAM_GRID_MAX_MOVE_PER_ROW,
    gr_calibration=BEAM_GRID_GR_CALIBRATION,
    smooth_radius=BEAM_GRID_SMOOTH_RADIUS,
)


def _smooth_gr(gr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return gr
    window = 2 * radius + 1
    roll = pd.Series(gr).rolling(window=window, center=True, min_periods=1)
    return roll.mean().to_numpy(dtype=float)


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
# Ported from src/rogii/spatial.py (spatial-prior predictor, IDW branch
# only, see module docstring's Simplifications). Host forum idea (Igor
# Kuvaev, forum 708167): formation-top depth is spatially smooth, so it can
# be interpolated from nearby train wells' formation-depth columns.
# --------------------------------------------------------------------------- #

FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
_SPATIAL_PREFIX_WINDOW = 500
_IDW_EPS = 1e-3
_COMPOSITE_EPS = 1e-6
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


def build_surface_bank(
    wells: list[str],
    split: str,
    root: Path,
    stride: int = SPATIAL_STRIDE,
    formations: tuple[str, ...] = FORMATIONS,
) -> SurfaceBank:
    """Collect stride-subsampled ``(X, Y, depth)`` samples for each formation.

    Only ever reads *train*-split wells (the only schema with formation-depth
    columns). Leak-safe: this bank is built from the FULL train well list
    (the target well's own samples are excluded only at *query* time via
    ``exclude_well``, see :func:`predict_spatial`).
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


def _predict_spatial_impl(
    h: pd.DataFrame, bank: SurfaceBank, exclude_well: str, k: int
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

    eval_z = z[eval_idx]
    candidates: list[np.ndarray] = []
    candidate_rmses: list[float] = []

    for f in bank.formations:
        depth_interp = _knn_idw_depth(
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
    return SpatialResult(
        tvt=candidates[best], prefix_rmse=float(rmses[best]), n_formations_used=len(candidates)
    )


def predict_spatial(
    h: pd.DataFrame, bank: SurfaceBank, exclude_well: str, k: int = SPATIAL_K
) -> SpatialResult:
    """Spatial-prior TVT prediction for the evaluation zone of ``h``. Never raises."""
    try:
        return _predict_spatial_impl(h, bank, exclude_well, k)
    except Exception:
        return _spatial_flat_fallback(h)


def nn_distance_per_row(h: pd.DataFrame, bank: SurfaceBank, well: str) -> np.ndarray:
    """Per-row nearest non-self bank-sample distance for every eval-zone row."""
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
# Per-well orchestration: tracker/spatial computation (eval-zone length) ->
# eval-row 20-column delta block -> broadcast into the full-T sequence with
# known-zone rows zero-filled (see module docstring) + well-level static
# aggregates. Ported from stack_v2_blend_submission.py's WellTrackerSpatial /
# compute_well_tracker_spatial / build_tracker_features / build_spatial_
# features / build_well_aggregate_features / build_bank_delta_features
# (byte-identical column math, restructured to also emit the static-block
# scalars instead of a broadcast-to-every-eval-row DataFrame).
# --------------------------------------------------------------------------- #

TRACKER_COLUMNS: tuple[str, ...] = (
    "pf_d",
    "pf_std",
    "beam_d",
    "beam_margin",
    "pf_beam_abs_diff",
    "pf_d_damp",
    "pf_std_is_inf",
    "pf_beam_sign_agree",
)
SPATIAL_COLUMNS: tuple[str, ...] = (
    "spatial_d",
    "spatial_prefix_rmse",
    "spatial_nn_dist_median",
    "spatial_gated_d",
)
SPATIAL_GATE_THETA = 2.0
DELTA_SEQ_COLUMNS: tuple[str, ...] = (
    "pfbma_v2_d",
    "beam_grid_d",
    "pfbma_v2_minus_pf",
    "pfbma_v2_s5_d",
    "pfbma_v2_s8_d",
    "pfbma_v2_s12_d",
    "pfbma_v2_scale_range",
    "pfbma_v2_s3_std",
)

# Fixed order for the 20-col eval-zone tracker/spatial/delta block, which
# feeds the LAST 20 of the 44 SEQ_CHANNEL_NAMES (see below).
SEQ_TRACKER_COLUMNS: tuple[str, ...] = TRACKER_COLUMNS + SPATIAL_COLUMNS + DELTA_SEQ_COLUMNS

SEQ_CHANNEL_NAMES: tuple[str, ...] = SEQ_P1_COLUMNS + SEQ_TRACKER_COLUMNS
N_SEQ_CHANNELS = len(SEQ_CHANNEL_NAMES)

WELL_AGG_COLUMNS: tuple[str, ...] = (
    "pf_d_final",
    "pf_d_mean",
    "pf_std_well_mean",
    "pf_std_well_max",
    "beam_margin_well_mean",
    "pf_beam_abs_diff_well_mean",
    "gr_nan_frac",
    "eval_len",
)

STATIC_CHANNEL_NAMES: tuple[str, ...] = (
    "anchor_tvt",
    "slope_md_all",
    "slope_md_last200",
    "slope_md_last50",
    "slope_z_all",
    "known_tvt_min",
    "known_tvt_max",
    "known_tvt_range",
    "known_tvt_mean",
    "known_tvt_std",
    "n_known_rows",
) + WELL_AGG_COLUMNS + (
    "pfbma_v2_d_well_mean",
    "pfbma_v2_s3_std_well_mean",
)
N_STATIC_CHANNELS = len(STATIC_CHANNEL_NAMES)


@dataclass
class WellTrackerSpatial:
    """Cached PF + beam + spatial-prior + bank-delta inputs, eval-zone length."""

    anchor: float
    pf_tvt: np.ndarray
    pf_std: np.ndarray
    beam_tvt: np.ndarray
    beam_margin: np.ndarray
    spatial_tvt: np.ndarray
    prefix_rmse: float
    nn_dist_median: float
    gr_nan_frac: float
    pf_bma_scale3_tvt: np.ndarray
    pf_bma_scale5_tvt: np.ndarray
    pf_bma_scale8_tvt: np.ndarray
    pf_bma_scale12_tvt: np.ndarray
    pf_bma_scale3_std: np.ndarray
    beam_mid_cal_sm2_mm3_tvt: np.ndarray


def _delta_input_arrays(wts: WellTrackerSpatial) -> tuple[np.ndarray, ...]:
    """Every eval-zone-length tracker array that must agree on length."""
    return (
        wts.pf_tvt,
        wts.pf_bma_scale3_tvt,
        wts.pf_bma_scale5_tvt,
        wts.pf_bma_scale8_tvt,
        wts.pf_bma_scale12_tvt,
        wts.pf_bma_scale3_std,
        wts.beam_mid_cal_sm2_mm3_tvt,
    )


def compute_well_tracker_spatial(
    h: pd.DataFrame, tw: pd.DataFrame, well: str, bank: SurfaceBank
) -> WellTrackerSpatial | None:
    """PF + beam + spatial-prior + bank-delta computation; never raises (None on failure)."""
    try:
        anchor = last_known_tvt(h)
        pf = track_pf_multi(h, tw, n_seeds=N_SEEDS, n_particles=N_PARTICLES, scales=PF_SCALES)
        beam = track_beam_multi(h, tw)
        spatial = predict_spatial(h, bank, exclude_well=well, k=SPATIAL_K)
        nn_dist = nn_distance_per_row(h, bank, well)
        finite_nn = nn_dist[np.isfinite(nn_dist)] if nn_dist.size else np.array([])
        nn_dist_median = float(np.median(finite_nn)) if finite_nn.size else float("nan")
        gr_nan_frac = float(h["GR"].isna().mean()) if "GR" in h.columns else 1.0

        pfbma = track_pf_bma(
            h,
            tw,
            n_particles=PFBMA_N_PARTICLES,
            n_seeds=PFBMA_N_SEEDS,
            bma_scales=PFBMA_BMA_SCALES,
            gr_scales=PFBMA_GR_SCALES,
            init_pos_std=PFBMA_INIT_POS_STD,
        )
        beam_mid_cal_sm2_mm3_tvt = track_beam_grid(h, tw, BEAM_MID_CAL_SM2_MM3).tvt

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
            pf_bma_scale3_tvt=pfbma.tvt_by_scale[3.0],
            pf_bma_scale5_tvt=pfbma.tvt_by_scale[5.0],
            pf_bma_scale8_tvt=pfbma.tvt_by_scale[8.0],
            pf_bma_scale12_tvt=pfbma.tvt_by_scale[12.0],
            pf_bma_scale3_std=pfbma.std_by_scale[3.0],
            beam_mid_cal_sm2_mm3_tvt=beam_mid_cal_sm2_mm3_tvt,
        )
    except Exception:
        return None


def build_tracker_features(
    anchor: float,
    pf_tvt: np.ndarray,
    pf_std: np.ndarray,
    beam_tvt: np.ndarray,
    beam_margin: np.ndarray,
) -> pd.DataFrame:
    """Byte-identical port of ``src/rogii/stack_features.py::build_tracker_features`` (8 cols)."""
    pf_tvt = np.asarray(pf_tvt, dtype=np.float64)
    pf_std = np.asarray(pf_std, dtype=np.float64)
    beam_tvt = np.asarray(beam_tvt, dtype=np.float64)
    beam_margin = np.asarray(beam_margin, dtype=np.float64)
    n = pf_tvt.size

    anchor_f = float(anchor) if np.isfinite(anchor) else 0.0
    if n == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in TRACKER_COLUMNS})

    pf_d = pf_tvt - anchor_f
    beam_d = beam_tvt - anchor_f
    pf_std_is_inf = (~np.isfinite(pf_std)).astype(np.float64)
    pf_d_damp = pf_d / (1.0 + pf_std)
    pf_beam_abs_diff = np.abs(pf_d - beam_d)
    pf_sign = np.sign(pf_d)
    beam_sign = np.sign(beam_d)
    sign_agree = (pf_sign == beam_sign) | (pf_sign == 0) | (beam_sign == 0)
    pf_beam_sign_agree = sign_agree.astype(np.float64)

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
    )[list(TRACKER_COLUMNS)]
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def build_spatial_features(
    anchor: float, spatial_tvt: np.ndarray, prefix_rmse: float, nn_dist_median: float
) -> pd.DataFrame:
    """Byte-identical port of ``src/rogii/stack_features.py::build_spatial_features`` (4 cols)."""
    spatial_tvt = np.asarray(spatial_tvt, dtype=np.float64)
    n = spatial_tvt.size
    if n == 0:
        return pd.DataFrame({c: np.zeros(0, dtype=np.float32) for c in SPATIAL_COLUMNS})

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
    )[list(SPATIAL_COLUMNS)]
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def build_bank_delta_features(
    trk: pd.DataFrame, wts: WellTrackerSpatial
) -> tuple[pd.DataFrame, float, float]:
    """Port of ``run_stack_v23_cv.py::add_config_c_columns`` (R7+G1+G2), split into
    (8-col per-row DataFrame, pfbma_v2_d_well_mean, pfbma_v2_s3_std_well_mean).
    """
    anchor_f64 = float(wts.anchor) if np.isfinite(wts.anchor) else 0.0
    pf_blend = anchor_f64 + PF_BLEND_W * (wts.pf_tvt.astype(np.float64) - anchor_f64)
    pfbma = wts.pf_bma_scale3_tvt.astype(np.float64)
    beam_grid = wts.beam_mid_cal_sm2_mm3_tvt.astype(np.float64)
    pf_from_feature = trk["pf_d"].to_numpy(dtype=np.float64) + anchor_f64
    n = pfbma.shape[0]

    pfbma_v2_d = pfbma - pf_blend
    beam_grid_d = beam_grid - pf_blend
    pfbma_v2_minus_pf = pfbma - pf_from_feature
    well_mean = float(pd.Series(pfbma_v2_d).mean()) if n else 0.0

    pfbma_v2_d_f32 = pfbma_v2_d.astype(np.float32)
    scale3_tvt = pfbma_v2_d_f32.astype(np.float64) + pf_blend
    scale5_tvt = wts.pf_bma_scale5_tvt.astype(np.float64)
    scale8_tvt = wts.pf_bma_scale8_tvt.astype(np.float64)
    scale12_tvt = wts.pf_bma_scale12_tvt.astype(np.float64)
    all_scales = np.stack([scale3_tvt, scale5_tvt, scale8_tvt, scale12_tvt], axis=1)
    pfbma_v2_scale_range = all_scales.max(axis=1) - all_scales.min(axis=1)

    s3_std = wts.pf_bma_scale3_std.astype(np.float64)
    finite_or_nan = np.where(np.isfinite(s3_std), s3_std, np.nan)
    std_mean = float(pd.Series(finite_or_nan).mean()) if n else 0.0

    out = pd.DataFrame(
        {
            "pfbma_v2_d": pfbma_v2_d,
            "beam_grid_d": beam_grid_d,
            "pfbma_v2_minus_pf": pfbma_v2_minus_pf,
            "pfbma_v2_s5_d": scale5_tvt - pf_blend,
            "pfbma_v2_s8_d": scale8_tvt - pf_blend,
            "pfbma_v2_s12_d": scale12_tvt - pf_blend,
            "pfbma_v2_scale_range": pfbma_v2_scale_range,
            "pfbma_v2_s3_std": s3_std,
        }
    )[list(DELTA_SEQ_COLUMNS)]
    out = out.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return out, well_mean, std_mean


def well_aggregate_scalars(
    anchor: float,
    pf_tvt: np.ndarray,
    pf_std: np.ndarray,
    beam_tvt: np.ndarray,
    beam_margin: np.ndarray,
    gr_nan_frac: float,
) -> dict[str, float]:
    """Scalar port of ``src/rogii/stack_features.py::build_well_aggregate_features``."""
    pf_tvt = np.asarray(pf_tvt, dtype=np.float64)
    pf_std = np.asarray(pf_std, dtype=np.float64)
    beam_tvt = np.asarray(beam_tvt, dtype=np.float64)
    beam_margin = np.asarray(beam_margin, dtype=np.float64)
    n = pf_tvt.size
    if n == 0:
        return dict.fromkeys(WELL_AGG_COLUMNS, 0.0)

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

    return {
        "pf_d_final": _last_finite_or(pf_d, 0.0),
        "pf_d_mean": _mean_finite_or(pf_d, 0.0),
        "pf_std_well_mean": _mean_finite_or(pf_std, 0.0),
        "pf_std_well_max": _max_finite_or(pf_std, 0.0),
        "beam_margin_well_mean": _mean_finite_or(beam_margin, 0.0),
        "pf_beam_abs_diff_well_mean": _mean_finite_or(abs_diff, 0.0),
        "gr_nan_frac": gr_nan_frac_f,
        "eval_len": float(n),
    }


def _broadcast_eval_to_full(n: int, known_mask: np.ndarray, eval_block: pd.DataFrame) -> np.ndarray:
    """Zero-fill known-zone rows, place ``eval_block`` values at eval-zone rows.

    Returns a ``(n, len(eval_block.columns))`` float32 array in
    ``eval_block``'s column order. See module docstring "Tracker channels
    are known-zone zero-filled".
    """
    eval_idx = np.flatnonzero(~known_mask)
    out = np.zeros((n, len(eval_block.columns)), dtype=np.float32)
    for j, col in enumerate(eval_block.columns):
        vals = eval_block[col].to_numpy(dtype=np.float32, copy=False)
        if vals.shape[0] != eval_idx.shape[0]:
            raise AssertionError(
                f"{col}: eval_block length {vals.shape[0]} != "
                f"eval-zone row count {eval_idx.shape[0]}"
            )
        out[eval_idx, j] = vals
    return out


# --------------------------------------------------------------------------- #
# Per-well record + main dataset-building loop.
# --------------------------------------------------------------------------- #


@dataclass
class WellRecord:
    well: str
    n_rows: int
    n_eval_rows: int
    x_seq: np.ndarray  # (n_rows, N_SEQ_CHANNELS) float32
    x_static: np.ndarray  # (N_STATIC_CHANNELS,) float32
    target: np.ndarray  # (n_rows,) float32
    target_mask: np.ndarray  # (n_rows,) bool
    y_true: np.ndarray  # (n_rows,) float32
    pf_blend: np.ndarray  # (n_rows,) float32


def process_well(
    well: str, root: Path, bank: SurfaceBank, counts: dict[str, int]
) -> WellRecord | None:
    """Build one well's full :class:`WellRecord`, or ``None`` (+ increment ``counts``) on skip.

    Never raises: any well-level failure is caught, counted, and skipped --
    matching every ``track_*``/``predict_spatial`` function's own
    never-raise contract (``docs/playbooks/01_implement_module.md``).
    """
    try:
        h = load_horizontal(well, "train", root)
        tw = load_typewell(well, "train", root)
    except Exception:
        counts["load_failed"] += 1
        return None

    n = len(h)
    if n == 0:
        counts["empty_well"] += 1
        return None

    mask = eval_mask(h)
    n_eval = int(mask.sum())
    if n_eval == 0:
        counts["no_eval_zone"] += 1
        return None
    try:
        last_known_tvt(h)
    except ValueError:
        counts["no_anchor"] += 1
        return None
    if "TVT" not in h.columns:
        counts["no_tvt_label"] += 1
        return None

    try:
        seq_p1, static = build_seq_features(h, tw)
    except Exception:
        counts["seq_features_failed"] += 1
        return None

    wts = compute_well_tracker_spatial(h, tw, well, bank)
    if wts is None:
        counts["tracker_spatial_failed"] += 1
        return None
    bad = [a.size for a in _delta_input_arrays(wts) if a.size != n_eval]
    if bad:
        counts["tracker_length_mismatch"] += 1
        return None

    trk_df = build_tracker_features(
        wts.anchor, wts.pf_tvt, wts.pf_std, wts.beam_tvt, wts.beam_margin
    )
    spat_df = build_spatial_features(
        wts.anchor, wts.spatial_tvt, wts.prefix_rmse, wts.nn_dist_median
    )
    delta_df, pfbma_v2_d_well_mean, pfbma_v2_s3_std_well_mean = build_bank_delta_features(
        trk_df, wts
    )
    eval_block = pd.concat(
        [
            trk_df.reset_index(drop=True),
            spat_df.reset_index(drop=True),
            delta_df.reset_index(drop=True),
        ],
        axis=1,
    )[list(SEQ_TRACKER_COLUMNS)]

    known_mask = h["TVT_input"].notna().to_numpy()
    tracker_full = _broadcast_eval_to_full(n, known_mask, eval_block)

    x_seq = np.concatenate([seq_p1.to_numpy(dtype=np.float32), tracker_full], axis=1)
    if x_seq.shape != (n, N_SEQ_CHANNELS):
        raise AssertionError(f"{well}: x_seq shape {x_seq.shape} != ({n}, {N_SEQ_CHANNELS})")

    agg = well_aggregate_scalars(
        wts.anchor, wts.pf_tvt, wts.pf_std, wts.beam_tvt, wts.beam_margin, wts.gr_nan_frac
    )
    static_values = (
        static.anchor_tvt,
        static.slope_md_all,
        static.slope_md_last200,
        static.slope_md_last50,
        static.slope_z_all,
        static.known_tvt_min,
        static.known_tvt_max,
        static.known_tvt_range,
        static.known_tvt_mean,
        static.known_tvt_std,
        static.n_known_rows,
        agg["pf_d_final"],
        agg["pf_d_mean"],
        agg["pf_std_well_mean"],
        agg["pf_std_well_max"],
        agg["beam_margin_well_mean"],
        agg["pf_beam_abs_diff_well_mean"],
        agg["gr_nan_frac"],
        agg["eval_len"],
        pfbma_v2_d_well_mean,
        pfbma_v2_s3_std_well_mean,
    )
    x_static = np.array(
        [v if np.isfinite(v) else 0.0 for v in static_values], dtype=np.float32
    )
    if x_static.shape != (N_STATIC_CHANNELS,):
        raise AssertionError(f"{well}: x_static shape {x_static.shape} != ({N_STATIC_CHANNELS},)")

    # target = TVT - pf_blend (docs/r10_nn_design.md Sec.1.2), eval-zone only;
    # known-zone rows are target=0.0 / target_mask=False (never used in a
    # masked loss -- see module docstring point 3).
    anchor_f64 = float(wts.anchor) if np.isfinite(wts.anchor) else 0.0
    pf_blend_eval = anchor_f64 + PF_BLEND_W * (wts.pf_tvt.astype(np.float64) - anchor_f64)
    y_true_full = h["TVT"].to_numpy(dtype=np.float32)
    eval_idx = np.flatnonzero(~known_mask)

    target = np.zeros(n, dtype=np.float32)
    pf_blend_full = np.zeros(n, dtype=np.float32)
    target[eval_idx] = (y_true_full[eval_idx].astype(np.float64) - pf_blend_eval).astype(np.float32)
    pf_blend_full[eval_idx] = pf_blend_eval.astype(np.float32)

    return WellRecord(
        well=well,
        n_rows=n,
        n_eval_rows=n_eval,
        x_seq=x_seq,
        x_static=x_static,
        target=target,
        target_mask=mask,
        y_true=y_true_full,
        pf_blend=pf_blend_full,
    )


def build_dataset(
    wells: list[str], root: Path, bank: SurfaceBank, t_program_start: float
) -> tuple[list[WellRecord], dict[str, int], list[float]]:
    records: list[WellRecord] = []
    counts: dict[str, int] = {
        "load_failed": 0,
        "empty_well": 0,
        "no_eval_zone": 0,
        "no_anchor": 0,
        "no_tvt_label": 0,
        "seq_features_failed": 0,
        "tracker_spatial_failed": 0,
        "tracker_length_mismatch": 0,
        "budget_exceeded_skip": 0,
    }
    per_well_seconds: list[float] = []

    for i, well in enumerate(wells, start=1):
        if budget_left(t_program_start) <= 0:
            counts["budget_exceeded_skip"] += len(wells) - i + 1
            print(
                f"[BUDGET] {GLOBAL_TIME_BUDGET_S / 3600:.1f}h exceeded at well {i}/{len(wells)} "
                "-- stopping early, writing what was already built."
            )
            break

        t0 = time.perf_counter()
        rec = process_well(well, root, bank, counts)
        per_well_seconds.append(time.perf_counter() - t0)
        if rec is not None:
            records.append(rec)

        if i % PROGRESS_EVERY == 0 or i == len(wells):
            mean_s = float(np.mean(per_well_seconds)) if per_well_seconds else 0.0
            print(
                f"[PROGRESS] {i}/{len(wells)} wells (used={len(records)}) "
                f"mean={mean_s:.2f}s/well elapsed={time.perf_counter() - t_program_start:.0f}s",
                flush=True,
            )

    return records, counts, per_well_seconds


def write_outputs(records: list[WellRecord], out_dir: Path) -> None:
    if not records:
        raise RuntimeError("build_dataset produced 0 usable wells -- nothing to write")

    well_names = np.array([r.well for r in records])
    n_rows_per_well = np.array([r.n_rows for r in records], dtype=np.int64)
    n_eval_rows_per_well = np.array([r.n_eval_rows for r in records], dtype=np.int64)
    boundaries = np.concatenate(([0], np.cumsum(n_rows_per_well))).astype(np.int64)

    x_seq = np.concatenate([r.x_seq for r in records], axis=0)
    target = np.concatenate([r.target for r in records])
    target_mask = np.concatenate([r.target_mask for r in records])
    y_true = np.concatenate([r.y_true for r in records])
    pf_blend = np.concatenate([r.pf_blend for r in records])
    x_static = np.stack([r.x_static for r in records], axis=0)

    missing_fold = [w for w in well_names if w not in WELL_FOLD_MAP]
    if missing_fold:
        raise AssertionError(
            f"{len(missing_fold)} well(s) missing from the embedded WELL_FOLD_MAP "
            f"(first few: {missing_fold[:5]}) -- table is stale or a non-train well leaked in"
        )
    well_fold = np.array([WELL_FOLD_MAP[w] for w in well_names], dtype=np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    seq_path = out_dir / "nn_seq_channels.npz"
    static_path = out_dir / "nn_static_channels.npz"

    np.savez_compressed(
        seq_path,
        channel_names=np.array(SEQ_CHANNEL_NAMES),
        X_seq=x_seq,
        boundaries=boundaries,
        well_names=well_names,
        target=target,
        target_mask=target_mask,
        y_true=y_true,
        pf_blend=pf_blend,
    )
    np.savez_compressed(
        static_path,
        static_channel_names=np.array(STATIC_CHANNEL_NAMES),
        X_static=x_static,
        well_names=well_names,
        well_fold=well_fold,
        n_rows_per_well=n_rows_per_well,
        n_eval_rows_per_well=n_eval_rows_per_well,
    )
    print(f"\nwrote {seq_path} ({seq_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {static_path} ({static_path.stat().st_size / 1e6:.1f} MB)")
    print(
        f"n_wells={len(records)} total_rows={x_seq.shape[0]:,} "
        f"total_eval_rows={int(target_mask.sum()):,} n_seq_channels={x_seq.shape[1]} "
        f"n_static_channels={x_static.shape[1]}"
    )


def main() -> None:
    t_start = time.perf_counter()
    root = find_data_root()
    print(f"data root: {root}")

    all_wells = list_wells("train", root)
    limit = _dataset_limit()
    wells = all_wells[:limit] if limit is not None else all_wells
    if limit is not None:
        print(
            f"[SMOKE] ROGII_NN_DATASET_LIMIT={limit} -- {len(wells)}/{len(all_wells)} train wells"
        )
    else:
        print(f"train wells: {len(wells)}")

    t_bank0 = time.perf_counter()
    # Surface bank is built from the SAME `wells` subset in smoke mode (see
    # module docstring's smoke-mode caveat: a full-773-well bank build is a
    # fixed cost this smoke test intentionally does not pay).
    bank = build_surface_bank(wells, "train", root)
    t_bank = time.perf_counter() - t_bank0
    print(f"surface bank built: {len(bank.formations)} formations, {t_bank:.1f}s")

    records, counts, per_well_seconds = build_dataset(wells, root, bank, t_start)
    write_outputs(records, output_dir())

    n_done = len(per_well_seconds)
    if n_done:
        mean_s = float(np.mean(per_well_seconds))
        median_s = float(np.median(per_well_seconds))
        print(
            f"\nper-well timing (n={n_done}, excludes one-time {t_bank:.1f}s bank build): "
            f"mean={mean_s:.3f}s median={median_s:.3f}s"
        )
        print(
            f"773-well projection (linear extrapolation from this run's mean, "
            f"NOT an actual 773-well measurement): {mean_s * 773:.0f}s "
            f"({mean_s * 773 / 3600:.2f}h) + one-time bank build"
        )
    print(f"\ncounts (skipped wells by reason): {counts}")
    print(f"total runtime: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
