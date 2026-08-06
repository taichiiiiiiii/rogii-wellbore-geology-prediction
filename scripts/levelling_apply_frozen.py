"""Apply the FROZEN recipe (w=0.30, cap=kappa*MAD, no refit) to a rows_{tag} file.

The frozen recipe is the one ported into the submission kernel
(scripts/harness/build_levelling_probe.py, gate_and_blend + the module
docstring's revision-history note): a fixed blend weight, and a clip cap that
rescales with THIS RUN's own pooled |spat-base| MAD (fallback/no-coverage
rows included in the MAD pool as an explicit zero contribution -- they
already equal base after the upstream script's own fallback, so this just
means "don't exclude them from the pool").

    cap   = KAPPA * MAD_ft,  MAD_ft = median(|spat - base|) over ALL judged
            eval rows this tag contributes (fallback rows included)
    final = base + W * clip(spat - base, +-cap)

Reproduces the pair450 reference exactly: base 8.1723 -> final 7.0903
(Delta -1.0820), 24/40 wells improved, worst well +3.38 ft (see the run
report).  No parameter here is refit; W and KAPPA are frozen constants
copied verbatim from the submission kernel.

Usage:
    uv run python scripts/levelling_apply_frozen.py <tag> [anchor]
    # e.g. a_pair450 tail   |   tops_none tail   |   tops_pair450 tail
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE  # noqa: E402

W = 0.30
KAPPA = 3.3523203402896797


def apply_frozen(pipe: np.ndarray, spat: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float, float]:
    diff = spat - pipe
    mad = float(np.median(np.abs(diff)))
    cap = KAPPA * mad
    final = pipe + W * np.clip(diff, -cap, cap)
    return final, mad, cap


def pooled_rmse(p: np.ndarray, t: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - t) ** 2)))


def main(argv: list[str]) -> int:
    tag = argv[1] if len(argv) > 1 else "a_pair450"
    anchor = argv[2] if len(argv) > 2 else "tail"
    df = pd.read_parquet(CACHE / f"rows_{tag}.parquet")
    pipe = df["pipe"].to_numpy(float)
    truth = df["truth"].to_numpy(float)
    spat = df[f"sp_{anchor}"].to_numpy(float)

    final, mad, cap = apply_frozen(pipe, spat, truth)
    base_rmse = pooled_rmse(pipe, truth)
    final_rmse = pooled_rmse(final, truth)

    per = df.assign(well=df["well"], e0=(pipe - truth) ** 2, e1=(final - truth) ** 2).groupby(
        "well").agg(n=("truth", "size"),
                    rmse_pipe=("e0", lambda x: float(np.sqrt(x.mean()))),
                    rmse_final=("e1", lambda x: float(np.sqrt(x.mean())))).reset_index()
    per["delta"] = per["rmse_final"] - per["rmse_pipe"]
    n_improved = int((per["delta"] < 0).sum())
    n_wells = len(per)

    print(f"=== frozen recipe (W={W}, kappa={KAPPA:.4f}) on {tag} / anchor={anchor} ===")
    print(f"  MAD_ft={mad:.4f}  cap_ft={cap:.4f}")
    print(f"  base   {base_rmse:8.4f} ft")
    print(f"  final  {final_rmse:8.4f} ft   delta {final_rmse - base_rmse:+.4f}")
    print(f"  wells improved {n_improved}/{n_wells}")
    print(f"  worst well delta {per['delta'].max():+.4f} ft ({per.loc[per['delta'].idxmax(), 'well']})")
    print(f"  best  well delta {per['delta'].min():+.4f} ft ({per.loc[per['delta'].idxmin(), 'well']})")

    out = CACHE / f"FROZEN_perwell_{tag}_{anchor}.csv"
    per.sort_values("delta").to_csv(out, index=False)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
