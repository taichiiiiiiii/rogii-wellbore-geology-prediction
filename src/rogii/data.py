"""Data access for the ROGII Wellbore Geology Prediction competition.

The on-disk layout (after extraction) is::

    <data_root>/
        train/{well}__horizontal_well.csv   # MD,X,Y,Z,[formations],[TVT],GR,TVT_input
        train/{well}__typewell.csv           # TVT,GR,Geology
        train/{well}.png
        test/{well}__horizontal_well.csv     # MD,X,Y,Z,GR,TVT_input (TVT hidden)
        test/{well}__typewell.csv
        sample_submission.csv                # id={well}_{row_index}, tvt

Key invariant: the **evaluation zone** of a well is exactly the contiguous tail of
rows where ``TVT_input`` is NaN. In train, ground-truth ``TVT`` is still present
there (so we can score locally); in test the ``TVT`` column is absent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Local repo default; on Kaggle this is "/kaggle/input/rogii-wellbore-geology-prediction".
_REPO_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


_KAGGLE_INPUT = Path("/kaggle/input/rogii-wellbore-geology-prediction")


@lru_cache(maxsize=1)
def data_root(override: str | None = None) -> Path:
    """Locate the directory that contains ``train/`` and ``test/``."""
    roots = [Path(override)] if override else [_REPO_RAW, _KAGGLE_INPUT]
    for r in roots:
        if (r / "train").is_dir() and (r / "test").is_dir():
            return r
        # zip sometimes extracts into a nested folder
        for sub in r.glob("*/train") if r.is_dir() else []:
            return sub.parent
    raise FileNotFoundError(f"Could not locate train/ and test/ under any of: {roots}")


def list_wells(split: str, root: str | None = None) -> list[str]:
    """Return sorted unique well hashes in ``train`` or ``test``."""
    d = data_root(root) / split
    return sorted({p.name.split("__")[0] for p in d.glob("*__horizontal_well.csv")})


def load_horizontal(well: str, split: str, root: str | None = None) -> pd.DataFrame:
    """Load a well's horizontal-well log."""
    return pd.read_csv(data_root(root) / split / f"{well}__horizontal_well.csv")


def load_typewell(well: str, split: str, root: str | None = None) -> pd.DataFrame:
    """Load a well's vertical type-well reference log."""
    return pd.read_csv(data_root(root) / split / f"{well}__typewell.csv")


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


def submission_ids(well: str, h: pd.DataFrame) -> list[str]:
    """Submission ids (``{well}_{row_index}``) for the eval-zone rows of ``h``."""
    idx = np.where(eval_mask(h))[0]
    return [f"{well}_{i}" for i in idx]
