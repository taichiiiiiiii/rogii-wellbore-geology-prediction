"""Build a Kaggle ``submission.csv`` for the ROGII Wellbore Geology Prediction competition.

The required ``(well, row_index)`` pairs are derived from ``sample_submission.csv``'s
``id`` column (``{well}_{row_index}``) rather than by listing ``test/`` directly. This
matters because the local ``data/raw/test/`` only holds 3 example wells, while the
real Kaggle rerun swaps in a ~200-well ``test/`` set -- deriving from the sample keeps
this script correct in both cases.

For each well this predicts TVT for the eval-zone rows (``TVT_input`` is NaN, per
``rogii.data.eval_mask``) and validates the result against ``sample_submission.csv``
before writing, exiting non-zero on any validation failure.

    uv run python scripts/make_submission.py
    uv run python scripts/make_submission.py --predictor carry_last --out outputs/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import baseline  # noqa: E402
from rogii import data as D  # noqa: E402

PREDICTORS = {
    "carry_last": baseline.predict_carry_last,
}


def parse_sample_submission(path: Path) -> pd.DataFrame:
    """Load ``sample_submission.csv`` and split ``id`` into ``well``/``row_index``."""
    sample = pd.read_csv(path)
    if "id" not in sample.columns:
        raise ValueError(f"{path} is missing the required 'id' column")

    parts = sample["id"].str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2 or parts[1].isna().any():
        raise ValueError("sample_submission ids must be formatted as '{well}_{row_index}'")

    return sample.assign(well=parts[0], row_index=parts[1].astype(int)).reset_index(drop=True)


def build_predictions(
    sample: pd.DataFrame, root: Path, predictor_name: str
) -> tuple[pd.DataFrame, list[str]]:
    """Predict ``tvt`` for every id in ``sample``, grouped by well.

    Returns the submission DataFrame (same row order as ``sample``) plus a list of
    human-readable warnings about ids whose ``row_index`` falls outside the well's
    actual eval zone (``TVT_input`` NaN tail).

    Raises ``ValueError`` if a well's predicted row count disagrees with the number
    of eval-zone rows the predictor's contract promises, or with the number of ids
    ``sample_submission`` expects for that well.
    """
    predict_fn = PREDICTORS[predictor_name]
    mismatch_warnings: list[str] = []
    tvt = np.full(len(sample), np.nan, dtype=float)

    for well, well_rows in sample.groupby("well", sort=False):
        h = pd.read_csv(root / "test" / f"{well}__horizontal_well.csv")
        eval_idx = np.where(D.eval_mask(h))[0]
        preds = predict_fn(h)
        if preds.shape[0] != eval_idx.shape[0]:
            raise ValueError(
                f"{well}: predictor '{predictor_name}' returned {preds.shape[0]} rows, "
                f"expected {eval_idx.shape[0]} (eval-zone size)"
            )
        pred_by_row = dict(zip(eval_idx.tolist(), preds.tolist(), strict=True))
        fallback = D.last_known_tvt(h)

        want_rows = well_rows["row_index"].to_numpy()
        if want_rows.shape[0] != eval_idx.shape[0]:
            raise ValueError(
                f"{well}: sample_submission has {want_rows.shape[0]} id(s) but the eval "
                f"zone has {eval_idx.shape[0]} row(s)"
            )

        missing = sorted(set(want_rows.tolist()) - set(pred_by_row))
        if missing:
            mismatch_warnings.append(
                f"{well}: {len(missing)} row_index(es) fall outside the TVT_input-NaN "
                f"eval zone, e.g. {missing[:5]}"
            )

        for pos, row_idx in zip(well_rows.index.to_numpy(), want_rows, strict=True):
            tvt[pos] = pred_by_row.get(int(row_idx), fallback)

    out = pd.DataFrame({"id": sample["id"].to_numpy(), "tvt": tvt})
    return out, mismatch_warnings


def validate_submission(out: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    """Hard-fail validation checks. Returns error messages; empty means OK."""
    errors: list[str] = []

    if len(out) != len(sample):
        errors.append(f"row count mismatch: submission has {len(out)}, sample has {len(sample)}")

    n_dup = int(out["id"].duplicated().sum())
    if n_dup:
        errors.append(f"submission has {n_dup} duplicate id(s)")

    out_ids, sample_ids = set(out["id"]), set(sample["id"])
    missing_ids = sample_ids - out_ids
    extra_ids = out_ids - sample_ids
    if missing_ids:
        errors.append(
            f"missing {len(missing_ids)} id(s) required by sample_submission, "
            f"e.g. {sorted(missing_ids)[:5]}"
        )
    if extra_ids:
        errors.append(
            f"submission has {len(extra_ids)} id(s) not in sample_submission, "
            f"e.g. {sorted(extra_ids)[:5]}"
        )

    tvt = out["tvt"].to_numpy(dtype=float)
    n_nan = int(np.isnan(tvt).sum())
    n_inf = int(np.isinf(tvt).sum())
    if n_nan:
        errors.append(f"{n_nan} NaN value(s) in 'tvt'")
    if n_inf:
        errors.append(f"{n_inf} inf value(s) in 'tvt'")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", default="carry_last", choices=sorted(PREDICTORS))
    parser.add_argument(
        "--data-root", default=None, help="Override data root (default: auto-detect)"
    )
    parser.add_argument("--out", default="outputs/submission.csv")
    args = parser.parse_args()

    root = D.data_root(args.data_root)
    print(f"[INFO] data root: {root}")

    sample = parse_sample_submission(root / "sample_submission.csv")
    n_wells_in = sample["well"].nunique()
    print(f"[INFO] sample_submission: {len(sample)} id(s) across {n_wells_in} well(s)")

    try:
        out, mismatch_warnings = build_predictions(sample, root, args.predictor)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    errors = validate_submission(out, sample)
    if errors:
        print("[FAIL] submission validation failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    for w in mismatch_warnings:
        print(f"[WARN] {w}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[["id", "tvt"]].to_csv(out_path, index=False)

    n_wells = out["id"].str.rsplit("_", n=1).str[0].nunique()
    print(f"[PASS] wrote {out_path} ({len(out)} rows, {n_wells} well(s))")
    print("[PASS] id set matches sample_submission exactly; no NaN/inf in 'tvt'; row counts match")


if __name__ == "__main__":
    main()
