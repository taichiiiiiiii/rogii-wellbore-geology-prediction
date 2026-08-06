"""First-pass EDA for ROGII Wellbore Geology Prediction.

Assumption-light: inspects the actual files on disk, prints schema / shapes /
NaN structure, and maps the submission format. Run after the data is extracted:

    uv run python scripts/eda_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def find_data_root() -> Path:
    """Locate the directory that actually contains train/ and test/."""
    candidates = [RAW, *[p for p in RAW.rglob("train") if p.is_dir()]]
    for c in candidates:
        train = c if c.name == "train" else c / "train"
        if (train).is_dir():
            return train.parent
    print("ERROR: could not locate train/ under", RAW)
    sys.exit(1)


def well_ids(split_dir: Path) -> list[str]:
    """Unique 8-char well hashes from *__horizontal_well.csv files."""
    return sorted({p.name.split("__")[0] for p in split_dir.glob("*__horizontal_well.csv")})


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)

    data_root = find_data_root()
    print(f"# data root: {data_root}")
    train_dir, test_dir = data_root / "train", data_root / "test"

    tr_wells = well_ids(train_dir)
    te_wells = well_ids(test_dir)
    print(f"# train wells: {len(tr_wells)} | test wells: {len(te_wells)}")
    print(f"# example train ids: {tr_wells[:5]}")

    # top-level files
    print("\n# top-level files in data root:")
    for p in sorted(data_root.iterdir()):
        if p.is_file():
            print(f"  {p.name:40s} {p.stat().st_size:>12,} bytes")

    # ---- horizontal well ----
    w = tr_wells[0]
    hw = pd.read_csv(train_dir / f"{w}__horizontal_well.csv")
    print(f"\n{'='*70}\n# HORIZONTAL WELL  ({w})  shape={hw.shape}\n{'='*70}")
    print("columns:", list(hw.columns))
    print("\ndtypes:\n", hw.dtypes)
    print("\nhead:\n", hw.head())
    print("\ndescribe:\n", hw.describe(include="all").T)
    print("\nNaN counts per column:\n", hw.isna().sum())

    # evaluation zone = where TVT_input is NaN but (train) TVT is known
    if "TVT_input" in hw.columns:
        evalzone = hw["TVT_input"].isna()
        print(f"\n# eval-zone rows (TVT_input NaN): {evalzone.sum()} / {len(hw)}")
        if "TVT" in hw.columns:
            print("# TVT known in eval zone? ",
                  hw.loc[evalzone, "TVT"].notna().mean(), "fraction non-NaN")
            # contiguity: is the eval zone a single contiguous block?
            idx = hw.index[evalzone].to_numpy()
            if len(idx):
                gaps = (idx[1:] - idx[:-1])
                print(f"# eval-zone index range: [{idx.min()}, {idx.max()}], "
                      f"contiguous={bool((gaps == 1).all())}")

    # ---- type well ----
    tw = pd.read_csv(train_dir / f"{w}__typewell.csv")
    print(f"\n{'='*70}\n# TYPE WELL  ({w})  shape={tw.shape}\n{'='*70}")
    print("columns:", list(tw.columns))
    print("dtypes:\n", tw.dtypes)
    print("\nhead:\n", tw.head())
    if "Geology" in tw.columns:
        print("\nGeology value counts:\n", tw["Geology"].value_counts())

    # ---- per-well size variation ----
    print(f"\n{'='*70}\n# size variation across first 20 train wells\n{'='*70}")
    rows = []
    for wi in tr_wells[:20]:
        h = pd.read_csv(train_dir / f"{wi}__horizontal_well.csv")
        ez = h["TVT_input"].isna().sum() if "TVT_input" in h.columns else -1
        rows.append({"well": wi, "n_rows": len(h), "eval_zone": ez})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- submission format ----
    sub_path = data_root / "sample_submission.csv"
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        print(f"\n{'='*70}\n# sample_submission  shape={sub.shape}\n{'='*70}")
        print("columns:", list(sub.columns))
        print(sub.head())
        # map id -> well
        sub["well"] = sub["id"].str.split("_").str[0]
        print("\n# submission rows per well (head):\n",
              sub.groupby("well").size().head())
        print(f"# unique wells in submission: {sub['well'].nunique()}")
        print(f"# total prediction rows: {len(sub)}")


if __name__ == "__main__":
    main()
