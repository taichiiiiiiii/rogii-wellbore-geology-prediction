"""Self-contained Kaggle Notebook submission: carry_last probe.

Paste this entire script into a single Kaggle Notebook cell for the ROGII
Wellbore Geology Prediction **code competition** (internet disabled, <=9h). It has
NO project imports -- only pandas/numpy -- so it is fully self-contained and runs
unmodified both:

- on Kaggle, reading ``/kaggle/input/rogii-wellbore-geology-prediction``, and
- locally against ``data/raw/`` for validation::

    uv run python notebooks/submission/probe_carry_last.py

Contract
--------
- The set of required ``(well, row_index)`` pairs is derived from
  ``sample_submission.csv``'s ``id`` column (``{well}_{row_index}``), NOT by
  listing ``test/`` directly. Locally ``test/`` holds only 3 example wells; the
  real Kaggle rerun substitutes a ~200-well ``test/`` set. Driving everything off
  ``sample_submission.csv`` keeps this script correct in both cases.
- Eval zone = the contiguous tail of a well's horizontal-well rows where
  ``TVT_input`` is NaN (ground truth is hidden there in ``test/``).
- Predictor: ``carry_last`` -- extend the last known (non-NaN) ``TVT_input``
  value flat across the eval zone.
- Writes ``submission.csv`` with columns ``id,tvt`` and validates it against
  ``sample_submission.csv`` before writing (exits non-zero on failure).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_INPUT = Path("/kaggle/input/rogii-wellbore-geology-prediction")


def find_data_root() -> Path:
    """Locate the directory containing ``train/``, ``test/``, ``sample_submission.csv``.

    On Kaggle the competition dataset is not reliably mounted at
    ``/kaggle/input/{competition-slug}`` -- observed in practice (2026-07-06 run of
    this very script) to 404 there while proven public kernels for this competition
    locate it via ``Path('/kaggle/input').rglob('sample_submission.csv')`` instead.
    So: try the conventional path first, then fall back to an ``/kaggle/input``-wide
    rglob for ``sample_submission.csv`` before giving up.
    """
    candidates = [KAGGLE_INPUT]
    try:
        here = Path(__file__).resolve()
        candidates.append(here.parents[2] / "data" / "raw")
    except NameError:
        pass  # __file__ is undefined when this cell runs inside a Jupyter kernel
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
        repo_root = Path(__file__).resolve().parents[2]
    except NameError:
        repo_root = Path.cwd()
    return repo_root / "outputs" / "submission_probe.csv"


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


def predict_carry_last(h: pd.DataFrame) -> np.ndarray:
    """Carry the last known TVT_input flat across the eval zone."""
    n = int(eval_mask(h).sum())
    return np.full(n, last_known_tvt(h), dtype=float)


def parse_sample_submission(path: Path) -> pd.DataFrame:
    """Load ``sample_submission.csv`` and split ``id`` into ``well``/``row_index``."""
    sample = pd.read_csv(path)
    parts = sample["id"].str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2 or parts[1].isna().any():
        raise ValueError("sample_submission ids must be formatted as '{well}_{row_index}'")
    return sample.assign(well=parts[0], row_index=parts[1].astype(int)).reset_index(drop=True)


def build_predictions(sample: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Predict ``tvt`` for every id in ``sample``, grouped by well.

    Returns the submission DataFrame (same row order as ``sample``) plus warnings
    about ids whose ``row_index`` falls outside the well's actual eval zone.
    """
    mismatch_warnings: list[str] = []
    tvt = np.full(len(sample), np.nan, dtype=float)

    for well, well_rows in sample.groupby("well", sort=False):
        h = pd.read_csv(root / "test" / f"{well}__horizontal_well.csv")
        idx = np.where(eval_mask(h))[0]
        preds = predict_carry_last(h)
        if preds.shape[0] != idx.shape[0]:
            raise ValueError(
                f"{well}: predictor produced {preds.shape[0]} rows, expected {idx.shape[0]}"
            )
        pred_by_row = dict(zip(idx.tolist(), preds.tolist(), strict=True))
        fallback = last_known_tvt(h)

        want_rows = well_rows["row_index"].to_numpy()
        if want_rows.shape[0] != idx.shape[0]:
            raise ValueError(
                f"{well}: sample_submission has {want_rows.shape[0]} id(s), "
                f"eval zone has {idx.shape[0]} row(s)"
            )

        missing = sorted(set(want_rows.tolist()) - set(pred_by_row))
        if missing:
            mismatch_warnings.append(
                f"{well}: {len(missing)} row_index(es) outside eval zone, e.g. {missing[:5]}"
            )

        for pos, row_idx in zip(well_rows.index.to_numpy(), want_rows, strict=True):
            tvt[pos] = pred_by_row.get(int(row_idx), fallback)

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


def main() -> None:
    root = find_data_root()
    print(f"[INFO] data root: {root}")

    sample = parse_sample_submission(root / "sample_submission.csv")
    n_wells_in = sample["well"].nunique()
    print(f"[INFO] sample_submission: {len(sample)} id(s) across {n_wells_in} well(s)")

    out, mismatch_warnings = build_predictions(sample, root)

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

    n_wells = out["id"].str.rsplit("_", n=1).str[0].nunique()
    print(f"[PASS] wrote {out_path.resolve()} ({len(out)} rows, {n_wells} well(s))")
    print("[PASS] id set matches sample_submission exactly; no NaN/inf in tvt; row counts match")


if __name__ == "__main__":
    main()
