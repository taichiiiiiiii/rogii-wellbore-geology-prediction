"""Build the two Track B Kaggle kernel scripts by inlining ``rogii.trackb_features``.

Code competitions run with internet disabled and cannot ``import rogii`` (no
pip install from this repo), so ``notebooks/submission/kernel_trackb_train``
and ``kernel_trackb_infer`` each need the ENTIRE feature module pasted in as
plain source. Hand-copying that risks silent drift between the tested
module and what actually runs on Kaggle, so this script does the copy
mechanically: read ``src/rogii/trackb_features.py``, strip its module
docstring and (duplicate) ``from __future__ import annotations`` line (a
second ``from __future__`` statement mid-file is a ``SyntaxError``), and
splice the remainder between a driver header and a driver body for each
kernel. Mirrors the same "builder script assembles the real kernel"
philosophy as ``scripts/harness/build_levelling_probe.py``.

Usage::

    uv run python scripts/harness/build_trackb_kernels.py

Regenerate this after every edit to ``trackb_features.py`` -- the kernel
scripts are generated output, not meant to be hand-edited directly (a header
comment in each says so).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_SRC = REPO_ROOT / "src" / "rogii" / "trackb_features.py"
TRAIN_DIR = REPO_ROOT / "notebooks" / "submission" / "kernel_trackb_train"
INFER_DIR = REPO_ROOT / "notebooks" / "submission" / "kernel_trackb_infer"


def _inlined_feature_module() -> str:
    """``trackb_features.py`` source, minus its module docstring and the
    ``from __future__ import annotations`` line (kept once, at the top of
    each generated kernel file, by the driver headers below instead)."""
    tree = ast.parse(FEATURES_SRC.read_text())
    lines = FEATURES_SRC.read_text().splitlines(keepends=True)

    # module docstring is the first statement, if it's a bare string Expr
    body = tree.body
    docstring_end = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        docstring_end = body[0].end_lineno  # 1-indexed, inclusive

    kept = lines[docstring_end:]
    out = []
    for line in kept:
        if line.strip() == "from __future__ import annotations":
            continue
        out.append(line)
    text = "".join(out).strip("\n") + "\n"
    ast.parse(text)  # must still be valid on its own
    return text


TRAIN_HEADER = '''\
"""Track B TRAINING kernel (Kaggle, CPU-only -- GPU quota is exhausted; a CPU
kernel does not consume it). GENERATED FILE: do not hand-edit -- regenerate
via ``uv run python scripts/harness/build_trackb_kernels.py`` after changing
``src/rogii/trackb_features.py``.

Self-contained: no project imports (Kaggle code competitions run with
internet disabled and cannot ``pip install``/``import rogii``). The block
between the BEGIN/END markers below is a byte-for-byte copy of
``src/rogii/trackb_features.py`` (module docstring and its own
``from __future__ import annotations`` stripped, since a second future-import
statement mid-file is a SyntaxError) -- see that file for the target
definition, feature-family docs, and leak-boundary discussion; nothing here
duplicates that reasoning.

What this does
---------------
1. Reads every TRAIN well (``/kaggle/input/.../train``), builds Track B
   features + the ``drift = TVT - carry_last`` target streamingly (one well
   at a time, float32, dropped before the next is loaded).
2. Well-level 5-fold GroupKFold (``well_folds`` below, identical algorithm to
   ``rogii.cv.well_folds`` -- sorted well list, seeded shuffle, round-robin
   deal -- so fold membership is reproducible and comparable to any other
   script in this repo that also calls ``rogii.cv.well_folds(wells,
   n_splits=5, seed=42)``).
3. Per fold: trains a LightGBM regressor on ``drift`` with an honest
   early-stopping holdout carved from the fold's OWN training wells (not the
   score fold -- see ``_split_fit_es`` below), predicts the score fold's OOF
   drift, and reports that fold's pooled RMSE of ``carry_last + pred`` vs
   ground truth TVT next to plain ``carry_last``.
4. Prints the POOLED (all 5 folds' score rows accumulated, squared-error
   pooled then sqrt'd -- the competition's own metric, never a per-well
   average) OOF RMSE.
5. Saves each fold's model (LightGBM text format, portable) plus one npz of
   OOF predictions/metadata to ``/kaggle/working`` for download.

An elapsed-time guard (``TRACKB_GUARD_HOURS``) stops starting new folds once
the budget is close to spent, printing what got skipped rather than running
past the target <2h -- CPU training here is far cheaper than 9h, but the
guard exists so a future accidental scale-up (more wells, more trees) fails
safe instead of timing out mid-kernel.

Smoke test locally (a handful of wells, NOT a strength claim, just "does it
run"): ``TRACKB_TRAIN_LIMIT=30 uv run python
notebooks/submission/kernel_trackb_train/trackb_train.py`` -- writes to
``outputs/trackb_train_smoke/`` instead of ``/kaggle/working`` when it detects
it isn't running on Kaggle. See ``scripts/harness/trackb_sanity_30well.py``
for a from-scratch (non-kernel) version of the same smoke test plus a
carry_last comparison, which is the actual validated 30-well sanity result
this scaffold was built and checked against.
"""

from __future__ import annotations

import glob
import os
import random
import time
from pathlib import Path

import lightgbm as lgb

# ============================================================================
# BEGIN inlined rogii.trackb_features (verbatim copy, see that file for docs)
# (also provides this file's only `numpy`/`pandas` imports)
# ============================================================================
__TRACKB_FEATURE_MODULE_PLACEHOLDER__
# ============================================================================
# END inlined rogii.trackb_features
# ============================================================================

'''

TRAIN_DRIVER = '''
# ============================================================================
# Track B training driver
# ============================================================================

N_SPLITS = 5
SEED = 42
ES_FRAC = 0.15  # fraction of each fold's training wells carved out as an honest ES holdout
TRACKB_GUARD_HOURS = float(_os.environ.get("TRACKB_GUARD_HOURS", "1.7"))  # target <2h total

LGB_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50


def find_data_root() -> Path:
    """Same discovery strategy as the repo's other self-contained kernels
    (notebooks/submission/probe_carry_last.py) -- try the conventional path,
    then rglob /kaggle/input for sample_submission.csv."""
    candidates = [Path("/kaggle/input/rogii-wellbore-geology-prediction")]
    try:
        here = Path(__file__).resolve()
        candidates.append(here.parents[2] / "data" / "raw")
    except NameError:
        pass
    candidates.append(Path.cwd() / "data" / "raw")

    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
        if root.is_dir():
            for sub in root.glob("*/train"):
                return sub.parent

    if Path("/kaggle/input").is_dir():
        for hit in sorted(Path("/kaggle/input").rglob("sample_submission.csv")):
            root = hit.parent
            if (root / "train").is_dir() and (root / "test").is_dir():
                return root

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not locate train/ and test/ under any of: {tried}")


def output_dir() -> Path:
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working")
    out = Path.cwd() / "outputs" / "trackb_train_smoke"
    out.mkdir(parents=True, exist_ok=True)
    return out


def well_folds(wells: list[str], n_splits: int = N_SPLITS, seed: int = SEED) -> list[list[str]]:
    """Identical algorithm to rogii.cv.well_folds -- duplicated here for the
    same self-containment reason as the inlined feature module."""
    ordered = sorted(wells)
    random.Random(seed).shuffle(ordered)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for i, well in enumerate(ordered):
        folds[i % n_splits].append(well)
    return folds


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _train_limit() -> int | None:
    v = _os.environ.get("TRACKB_TRAIN_LIMIT")
    return int(v) if v else None


def build_matrix(wells: list[str], root: Path, split: str):
    """Stream-build features/target/carry_last/well_id for `wells` -- one
    well's CSVs read, featurized, and dropped before the next is loaded."""
    x_parts, y_drift_parts, y_tvt_parts, carry_parts, well_id_parts = [], [], [], [], []
    n_skipped = 0
    for well in wells:
        h = pd.read_csv(root / split / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(root / split / f"{well}__typewell.csv")
        mask = eval_mask(h)
        if mask.sum() == 0:
            n_skipped += 1
            del h, tw
            continue
        try:
            anchor = carry_last(h)
        except ValueError:
            n_skipped += 1
            del h, tw
            continue

        feats = build_trackb_features(h, tw)
        y_true = h["TVT"].to_numpy(dtype=np.float32)[mask]
        y_drift = (y_true - anchor).astype(np.float32)

        x_parts.append(feats)
        y_drift_parts.append(y_drift)
        y_tvt_parts.append(y_true)
        carry_parts.append(np.full(len(feats), anchor, dtype=np.float32))
        well_id_parts.append(np.full(len(feats), well))
        del h, tw, feats, y_true, y_drift

    x = pd.concat(x_parts, ignore_index=True) if x_parts else pd.DataFrame(columns=FEATURE_COLUMNS)
    y_drift_arr = np.concatenate(y_drift_parts) if y_drift_parts else np.array([], dtype=np.float32)
    y_tvt_arr = np.concatenate(y_tvt_parts) if y_tvt_parts else np.array([], dtype=np.float32)
    carry_arr = np.concatenate(carry_parts) if carry_parts else np.array([], dtype=np.float32)
    well_id_arr = np.concatenate(well_id_parts) if well_id_parts else np.array([], dtype=object)
    print(f"[TRACKB] build_matrix: {len(wells) - n_skipped}/{len(wells)} wells usable "
          f"({n_skipped} skipped: empty eval zone or no anchor) n_rows={len(x)}")
    return x, y_drift_arr, y_tvt_arr, carry_arr, well_id_arr


def _split_fit_es(train_wells: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Carve an honest early-stopping holdout OUT of a fold's training wells
    (well-level, never the score fold itself) -- matches
    scripts/run_stack_v2_cv.py's ES-holdout lever."""
    ordered = sorted(train_wells)
    random.Random(seed).shuffle(ordered)
    n_es = max(1, int(round(len(ordered) * ES_FRAC)))
    return ordered[n_es:], ordered[:n_es]


def main() -> None:
    t0 = time.time()
    root = find_data_root()
    out_dir = output_dir()
    print(f"[TRACKB] data_root={root} out_dir={out_dir}")

    all_train_wells = sorted(
        {Path(p).name.split("__")[0] for p in glob.glob(str(root / "train" / "*__horizontal_well.csv"))}
    )
    limit = _train_limit()
    wells = all_train_wells[:limit] if limit else all_train_wells
    if limit:
        print(f"[TRACKB] TRACKB_TRAIN_LIMIT={limit} -- {len(wells)}/{len(all_train_wells)} train wells")
    print(f"[TRACKB] n_wells={len(wells)}")

    x, y_drift, y_tvt, carry, well_id = build_matrix(wells, root, "train")
    usable_wells = sorted(set(well_id.tolist()))
    n_rows = len(x)
    n_features = len(FEATURE_COLUMNS)
    print(f"[TRACKB] n_wells_usable={len(usable_wells)} n_rows={n_rows} n_features={n_features}")

    folds = well_folds(usable_wells, N_SPLITS, SEED)
    well_to_fold = {w: i for i, fw in enumerate(folds) for w in fw}
    fold_id = np.array([well_to_fold[w] for w in well_id])
    x_arr = x.to_numpy(dtype=np.float32)

    oof_drift = np.full(n_rows, np.nan, dtype=np.float64)
    fold_models: list[lgb.Booster] = []
    n_folds_run = 0

    for fold in range(N_SPLITS):
        elapsed_hours = (time.time() - t0) / 3600.0
        if elapsed_hours > TRACKB_GUARD_HOURS:
            print(f"[TRACKB] fold={fold} SKIPPED_TIME_GUARD elapsed_hours={elapsed_hours:.3f} "
                  f"guard_hours={TRACKB_GUARD_HOURS:.3f}")
            continue

        valid_wells = set(folds[fold])
        train_wells_fold = [w for w in usable_wells if w not in valid_wells]
        fit_wells, es_wells = _split_fit_es(train_wells_fold, SEED + fold)

        fit_mask = np.isin(well_id, fit_wells)
        es_mask = np.isin(well_id, es_wells)
        valid_mask = fold_id == fold
        if fit_mask.sum() == 0 or valid_mask.sum() == 0:
            print(f"[TRACKB] fold={fold} skipped (empty fit or valid set)")
            continue

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        fit_kwargs: dict[str, object] = {}
        if es_mask.sum() > 0:
            fit_kwargs["eval_set"] = [(x_arr[es_mask], y_drift[es_mask])]
            fit_kwargs["callbacks"] = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
        model.fit(x_arr[fit_mask], y_drift[fit_mask], **fit_kwargs)

        pred = model.predict(x_arr[valid_mask])
        oof_drift[valid_mask] = pred
        fold_models.append(model.booster_)
        n_folds_run += 1

        pred_tvt = carry[valid_mask] + pred
        rmse_model = pooled_rmse(y_tvt[valid_mask], pred_tvt)
        rmse_carry = pooled_rmse(y_tvt[valid_mask], carry[valid_mask])
        print(f"[TRACKB] fold={fold} n_fit_wells={len(fit_wells)} n_es_wells={len(es_wells)} "
              f"n_valid_wells={len(valid_wells)} n_valid_rows={int(valid_mask.sum())} "
              f"best_iteration={getattr(model, 'best_iteration_', None)} "
              f"rmse_carry_last={rmse_carry:.4f} rmse_trackb={rmse_model:.4f} "
              f"delta={rmse_model - rmse_carry:+.4f}")

        model.booster_.save_model(str(out_dir / f"trackb_fold{fold}.txt"))

    scored = ~np.isnan(oof_drift)
    if scored.any():
        pooled_pred_tvt = carry[scored] + oof_drift[scored]
        pooled_model_rmse = pooled_rmse(y_tvt[scored], pooled_pred_tvt)
        pooled_carry_rmse = pooled_rmse(y_tvt[scored], carry[scored])
    else:
        pooled_model_rmse = float("nan")
        pooled_carry_rmse = float("nan")

    np.savez_compressed(
        out_dir / "trackb_oof.npz",
        feature_columns=np.array(FEATURE_COLUMNS),
        well_id=well_id,
        fold_id=fold_id,
        y_drift_true=y_drift,
        y_tvt_true=y_tvt,
        carry_last=carry,
        oof_drift_pred=oof_drift,
    )

    elapsed_hours = (time.time() - t0) / 3600.0
    print(f"\\n[TRACKB] === POOLED OOF (n_wells={len(usable_wells)}, n_rows={int(scored.sum())}, "
          f"n_features={n_features}, n_folds_run={n_folds_run}/{N_SPLITS}) ===")
    print(f"[TRACKB] pooled_rmse_carry_last = {pooled_carry_rmse:.4f}")
    print(f"[TRACKB] pooled_rmse_trackb_oof = {pooled_model_rmse:.4f}")
    print(f"[TRACKB] model_count={len(fold_models)} elapsed_hours={elapsed_hours:.3f} "
          f"guard_hours={TRACKB_GUARD_HOURS:.3f}")
    print(f"[TRACKB] wrote {len(fold_models)} fold model file(s) + trackb_oof.npz to {out_dir}")


if __name__ == "__main__":
    import os as _os

    main()
'''

INFER_HEADER = '''\
"""Track B INFERENCE kernel (Kaggle, CPU-only). GENERATED FILE: do not
hand-edit -- regenerate via ``uv run python
scripts/harness/build_trackb_kernels.py`` after changing
``src/rogii/trackb_features.py``.

Self-contained: no project imports. The block between the BEGIN/END markers
is a byte-for-byte copy of ``src/rogii/trackb_features.py`` (see that file
for the target definition and leak-boundary discussion).

*** PLACEHOLDER: fill in TRACKB_MODEL_DATASET_SLUG below before running on
Kaggle. *** It must point at a Kaggle Dataset containing the fold model
files (``trackb_fold{0..4}.txt``) produced by
``notebooks/submission/kernel_trackb_train`` -- upload that training
kernel's ``/kaggle/working`` output as a new private Dataset (or via
``kaggle datasets version``) and put its slug
(``<your-kaggle-username>/<dataset-name>``) here; ``kernel-metadata.json``'s
``dataset_sources`` must list the same slug so it's mounted at
``/kaggle/input/<dataset-name>``.

What this does
---------------
1. Loads the attached fold models (``lgb.Booster(model_file=...)`` per fold
   -- 1 to 5 of them, however many the training kernel produced/uploaded).
2. Reads every TEST well, builds the same Track B features, and predicts
   ``drift`` per fold, averaging fold predictions (simple mean ensemble --
   a first-pass choice, not tuned; see the training kernel's per-fold OOF
   RMSE if you want to weight folds instead).
3. ``tvt_pred = carry_last(h) + mean_fold_predicted_drift`` for every
   eval-zone row, written out in ``eval_mask(h)`` order --exactly the id
   order ``rogii.data.submission_ids``/this kernel's own id-building uses,
   per the feature module's target-definition docstring.
4. [TRACKB] print discipline (n_wells, n_rows, feature_count, model_count,
   prediction stats), an elapsed-time guard (``TRACKB_INFER_GUARD_HOURS``,
   pattern copied from ``scripts/harness/build_levelling_probe.py``'s
   ``LEVELLING_GUARD_HOURS``) that aborts to a pure-``carry_last`` fallback
   submission if inference is somehow still running dangerously close to the
   9h ceiling, and a hard validation of ``submission.csv`` against
   ``sample_submission.csv`` before exit (id-set exact match, no NaN/inf).
"""

from __future__ import annotations

import glob
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ============================================================================
# BEGIN inlined rogii.trackb_features (verbatim copy, see that file for docs)
# ============================================================================
__TRACKB_FEATURE_MODULE_PLACEHOLDER__
# ============================================================================
# END inlined rogii.trackb_features
# ============================================================================

'''

INFER_DRIVER = '''
# ============================================================================
# Track B inference driver
# ============================================================================

# *** FILL IN before running on Kaggle *** -- see module docstring above.
TRACKB_MODEL_DATASET_SLUG = "taichiiiii/rogii-trackb-models"  # PLACEHOLDER

TRACKB_INFER_GUARD_HOURS = float(_os.environ.get("TRACKB_INFER_GUARD_HOURS", "8.6"))


def find_data_root() -> Path:
    candidates = [Path("/kaggle/input/rogii-wellbore-geology-prediction")]
    try:
        here = Path(__file__).resolve()
        candidates.append(here.parents[2] / "data" / "raw")
    except NameError:
        pass
    candidates.append(Path.cwd() / "data" / "raw")

    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
        if root.is_dir():
            for sub in root.glob("*/train"):
                return sub.parent

    if Path("/kaggle/input").is_dir():
        for hit in sorted(Path("/kaggle/input").rglob("sample_submission.csv")):
            root = hit.parent
            if (root / "train").is_dir() and (root / "test").is_dir():
                return root

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not locate train/ and test/ under any of: {tried}")


def find_model_files() -> list[Path]:
    """Locate the trained fold model text files under /kaggle/input.

    Tries the documented dataset slug first, then falls back to an
    /kaggle/input-wide glob for trackb_fold*.txt so a renamed dataset still
    works without editing this file (mirrors find_data_root's own
    rglob-fallback philosophy).
    """
    named = Path(f"/kaggle/input/{TRACKB_MODEL_DATASET_SLUG.split('/')[-1]}")
    hits = sorted(named.glob("trackb_fold*.txt")) if named.is_dir() else []
    if not hits and Path("/kaggle/input").is_dir():
        hits = sorted(Path(p) for p in glob.glob("/kaggle/input/**/trackb_fold*.txt", recursive=True))
    if not hits:
        local = Path.cwd() / "outputs" / "trackb_train_smoke"
        hits = sorted(local.glob("trackb_fold*.txt"))
    return hits


def output_path(root: Path) -> Path:
    if str(root).startswith("/kaggle"):
        return Path("submission.csv")
    try:
        repo_root = Path(__file__).resolve().parents[2]
    except NameError:
        repo_root = Path.cwd()
    return repo_root / "outputs" / "submission_trackb_infer.csv"


def parse_sample_submission(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path)
    parts = sample["id"].str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2 or parts[1].isna().any():
        raise ValueError("sample_submission ids must be formatted as '{well}_{row_index}'")
    return sample.assign(well=parts[0], row_index=parts[1].astype(int)).reset_index(drop=True)


def validate(out: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
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
    t0 = time.time()
    root = find_data_root()
    print(f"[TRACKB] data_root={root}")

    sample = parse_sample_submission(root / "sample_submission.csv")
    test_wells = sorted(sample["well"].unique().tolist())
    print(f"[TRACKB] n_wells={len(test_wells)} n_ids={len(sample)}")

    model_files = find_model_files()
    print(f"[TRACKB] model_count={len(model_files)} model_files={[p.name for p in model_files]}")
    if not model_files:
        raise RuntimeError(
            f"no trackb_fold*.txt model files found -- attach the training kernel's output "
            f"dataset (see TRACKB_MODEL_DATASET_SLUG={TRACKB_MODEL_DATASET_SLUG!r} placeholder "
            f"in this file's module docstring) as a dataset_source"
        )
    boosters = [lgb.Booster(model_file=str(p)) for p in model_files]

    tvt_by_id: dict[str, float] = {}
    n_rows_total = 0
    pred_drift_all: list[np.ndarray] = []
    fallback_wells = 0

    for well in test_wells:
        elapsed_hours = (time.time() - t0) / 3600.0
        if elapsed_hours > TRACKB_INFER_GUARD_HOURS:
            print(f"[TRACKB] TIME_GUARD_HIT elapsed_hours={elapsed_hours:.3f} "
                  f"guard_hours={TRACKB_INFER_GUARD_HOURS:.3f} -- falling back to carry_last "
                  f"for all remaining wells")
            break

        h = pd.read_csv(root / "test" / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(root / "test" / f"{well}__typewell.csv")
        mask = eval_mask(h)
        idx = np.where(mask)[0]
        if idx.size == 0:
            del h, tw
            continue

        try:
            anchor = carry_last(h)
        except ValueError:
            fallback_wells += 1
            del h, tw
            continue

        try:
            feats = build_trackb_features(h, tw)
            x_arr = feats.to_numpy(dtype=np.float32)
            fold_preds = np.stack([b.predict(x_arr) for b in boosters], axis=0)
            pred_drift = fold_preds.mean(axis=0)
            pred_tvt = anchor + pred_drift
        except Exception as exc:  # noqa: BLE001 -- any feature/predict failure falls back safely
            print(f"[TRACKB] well={well} FEATURE_OR_PREDICT_ERROR ({exc!r}) -- falling back to carry_last")
            pred_tvt = np.full(idx.size, anchor, dtype=np.float64)
            pred_drift = np.zeros(idx.size, dtype=np.float64)
            fallback_wells += 1

        for row_idx, val in zip(idx.tolist(), pred_tvt.tolist(), strict=True):
            tvt_by_id[f"{well}_{row_idx}"] = float(val)
        n_rows_total += idx.size
        pred_drift_all.append(np.asarray(pred_drift))
        del h, tw

    # any id the loop above never reached (time-guard break) falls back to carry_last
    n_ids_covered = len(tvt_by_id)
    if n_ids_covered < len(sample):
        print(f"[TRACKB] {len(sample) - n_ids_covered} id(s) uncovered (time guard or error) "
              f"-- filling with per-well carry_last fallback")
        for well, well_rows in sample.groupby("well", sort=False):
            missing = [r for r in well_rows["row_index"].tolist() if f"{well}_{r}" not in tvt_by_id]
            if not missing:
                continue
            h = pd.read_csv(root / "test" / f"{well}__horizontal_well.csv")
            try:
                anchor = carry_last(h)
            except ValueError:
                anchor = 0.0
            for row_idx in missing:
                tvt_by_id[f"{well}_{row_idx}"] = float(anchor)
            del h

    tvt = np.array([tvt_by_id[i] for i in sample["id"]], dtype=float)
    out = pd.DataFrame({"id": sample["id"].to_numpy(), "tvt": tvt})

    errors = validate(out, sample)
    if errors:
        print("[TRACKB] FAIL submission validation failed:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    out_path = output_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[["id", "tvt"]].to_csv(out_path, index=False)

    all_drift = np.concatenate(pred_drift_all) if pred_drift_all else np.array([])
    elapsed_hours = (time.time() - t0) / 3600.0
    print(f"\\n[TRACKB] n_wells={len(test_wells)} n_rows={n_rows_total} "
          f"n_features={len(FEATURE_COLUMNS)} model_count={len(boosters)} "
          f"fallback_wells={fallback_wells}")
    if all_drift.size:
        print(f"[TRACKB] pred_drift stats: mean={all_drift.mean():.4f} std={all_drift.std():.4f} "
              f"min={all_drift.min():.4f} max={all_drift.max():.4f}")
    print(f"[TRACKB] tvt stats: mean={tvt.mean():.4f} std={tvt.std():.4f} "
          f"min={tvt.min():.4f} max={tvt.max():.4f}")
    print(f"[TRACKB] elapsed_hours={elapsed_hours:.3f} guard_hours={TRACKB_INFER_GUARD_HOURS:.3f}")
    print(f"[TRACKB] PASS wrote {out_path.resolve()} ({len(out)} rows); "
          f"id set matches sample_submission exactly; no NaN/inf in tvt")


if __name__ == "__main__":
    import os as _os

    main()
'''

TRAIN_METADATA = """{
 "id": "taichiiiii/rogii-trackb-train",
 "title": "ROGII Track B Train",
 "code_file": "trackb_train.py",
 "language": "python",
 "kernel_type": "script",
 "is_private": true,
 "enable_gpu": false,
 "enable_tpu": false,
 "enable_internet": false,
 "docker_image_pinning_type": "latest",
 "keywords": ["trackb", "lightgbm", "cpu"],
 "dataset_sources": [],
 "kernel_sources": [],
 "competition_sources": ["rogii-wellbore-geology-prediction"],
 "model_sources": []
}
"""

INFER_METADATA = """{
 "id": "taichiiiii/rogii-trackb-infer",
 "title": "ROGII Track B Infer",
 "code_file": "trackb_infer.py",
 "language": "python",
 "kernel_type": "script",
 "is_private": true,
 "enable_gpu": false,
 "enable_tpu": false,
 "enable_internet": false,
 "docker_image_pinning_type": "latest",
 "keywords": ["trackb", "lightgbm", "cpu", "submission"],
 "dataset_sources": ["taichiiiii/rogii-trackb-models"],
 "kernel_sources": [],
 "competition_sources": ["rogii-wellbore-geology-prediction"],
 "model_sources": []
}
"""


def main() -> None:
    feature_module = _inlined_feature_module()

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    INFER_DIR.mkdir(parents=True, exist_ok=True)

    placeholder = "__TRACKB_FEATURE_MODULE_PLACEHOLDER__"
    train_src = TRAIN_HEADER.replace(placeholder, feature_module) + TRAIN_DRIVER
    infer_src = INFER_HEADER.replace(placeholder, feature_module) + INFER_DRIVER

    ast.parse(train_src)
    ast.parse(infer_src)

    (TRAIN_DIR / "trackb_train.py").write_text(train_src)
    (TRAIN_DIR / "kernel-metadata.json").write_text(TRAIN_METADATA)
    (INFER_DIR / "trackb_infer.py").write_text(infer_src)
    (INFER_DIR / "kernel-metadata.json").write_text(INFER_METADATA)

    print(f"wrote {TRAIN_DIR / 'trackb_train.py'} ({len(train_src.splitlines())} lines)")
    print(f"wrote {INFER_DIR / 'trackb_infer.py'} ({len(infer_src.splitlines())} lines)")


if __name__ == "__main__":
    main()
