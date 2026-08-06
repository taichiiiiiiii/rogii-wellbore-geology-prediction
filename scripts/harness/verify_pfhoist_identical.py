"""Prove the hoisted particle filter is bit-identical to the original.

The whole point of hoisting the seed-independent preparation out of the 128-seed
loop is that it buys runtime without changing anything, so the refactor is only
worth shipping if the outputs match exactly — not approximately.

Both versions of the function are extracted from their notebooks and executed in
isolated namespaces (the PF body needs nothing but numpy and the two DataFrames),
then run over real wells across several seeds and compared with an exact equality
check on the predictions and the log-likelihood.

Usage:
    uv run python scripts/harness/verify_pfhoist_identical.py [n_wells] [n_seeds]
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(os.environ.get("ROGII_WORK", "work"))
ORIG = SCRATCH / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
NEW = SCRATCH / "pfhoist" / "rogii-gs145-pfhoist" / "rogii-gs145-pfhoist.ipynb"
DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "train"


def pf_source(nb_path: Path) -> str:
    """The particle-filter block, from its definition to the next top-level def."""
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "def run_particle_filter" in src:
            start = src.index("def _pf_prepare") if "def _pf_prepare" in src \
                else src.index("def run_particle_filter")
            end = src.index("\ndef run_pf_lik_ensemble", start)
            return src[start:end]
    raise RuntimeError(f"no particle filter in {nb_path}")


def load(nb_path: Path) -> dict:
    ns: dict = {"np": np, "pd": pd}
    exec(compile(pf_source(nb_path), str(nb_path), "exec"), ns)
    return ns


def _compare_one(old: dict, new: dict, hw: pd.DataFrame, tw: pd.DataFrame,
                  prepared, wid: str, n_seeds: int, tag: str) -> int | None:
    """Run n_seeds direct _pf_core comparisons plus one wrapper comparison.

    Returns the number of (well, seed) pairs checked, or None on mismatch (after
    printing the diagnostic) so the caller can abort with a non-zero exit code.
    """
    for seed in range(n_seeds):
        p_old, ll_old = old["run_particle_filter"](hw, tw, n_particles=500, seed=seed)
        p_new, ll_new = new["_pf_core"](prepared, seed, 500)

        same_pred = np.array_equal(p_old, p_new, equal_nan=True)
        same_ll = (ll_old == ll_new) or (np.isnan(ll_old) and np.isnan(ll_new))
        if not (same_pred and same_ll):
            bad = np.nanmax(np.abs(np.asarray(p_old) - np.asarray(p_new)))
            print(f"MISMATCH well={wid} seed={seed} max|delta|={bad:.6g} "
                  f"ll_old={ll_old!r} ll_new={ll_new!r}{tag}")
            return None

    # The wrapper must also stay bit-identical for any untouched call site.
    p_w, ll_w = new["run_particle_filter"](hw, tw, n_particles=500, seed=0)
    p_o, ll_o = old["run_particle_filter"](hw, tw, n_particles=500, seed=0)
    same_pred_w = np.array_equal(p_w, p_o, equal_nan=True)
    same_ll_w = (ll_w == ll_o) or (np.isnan(ll_w) and np.isnan(ll_o))
    if not (same_pred_w and same_ll_w):
        print(f"MISMATCH via wrapper on well={wid}{tag}")
        return None
    print(f"  {wid}: {n_seeds} seeds identical (+ wrapper){tag}")
    return n_seeds


def main(argv: list[str]) -> int:
    n_wells = int(argv[1]) if len(argv) > 1 else 6
    n_seeds = int(argv[2]) if len(argv) > 2 else 8

    old, new = load(ORIG), load(NEW)
    wells = sorted(p.name.split("__")[0] for p in DATA.glob("*__horizontal_well.csv"))[:n_wells]
    if not wells:
        print("FAIL — no wells found under", DATA)
        return 1

    checked = 0
    empty_zone_checked = 0
    for wid in wells:
        hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(DATA / f"{wid}__typewell.csv")
        # The pipeline hides the eval zone; reproduce that here.
        hw = hw.drop(columns=[c for c in ("TVT",) if c in hw.columns])
        if hw["TVT_input"].notna().sum() < 10:
            continue
        is_empty_zone = hw["TVT_input"].isna().sum() == 0

        prepared = new["_pf_prepare"](hw, tw, 500)
        tag = " [EMPTY eval zone]" if is_empty_zone else ""
        n = _compare_one(old, new, hw, tw, prepared, wid, n_seeds, tag)
        if n is None:
            return 1
        checked += n
        if is_empty_zone:
            empty_zone_checked += n

    # Real wells in this dataset may never have a fully-known TVT_input (the eval
    # zone is the whole point of the task), so the empty-eval-zone branch that Fix 2
    # is about is not guaranteed to be exercised by the loop above. Synthesize one by
    # filling every gap, so the guard is checked regardless of what the sample holds.
    if empty_zone_checked == 0:
        wid = wells[0]
        hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(DATA / f"{wid}__typewell.csv")
        hw = hw.drop(columns=[c for c in ("TVT",) if c in hw.columns])
        hw["TVT_input"] = hw["TVT_input"].interpolate(limit_direction="both")
        if hw["TVT_input"].isna().sum() != 0:
            print(f"FAIL — could not synthesize a fully-known well from {wid}")
            return 1

        prepared = new["_pf_prepare"](hw, tw, 500)
        n = _compare_one(old, new, hw, tw, prepared, wid, n_seeds,
                          " [SYNTHETIC, EMPTY eval zone]")
        if n is None:
            return 1
        checked += n
        empty_zone_checked += n

    if checked == 0:
        print("FAIL — zero (well, seed) pairs were checked; the comparison never ran")
        return 1
    if empty_zone_checked == 0:
        print("FAIL — no well with an EMPTY eval zone was exercised")
        return 1

    print(f"\nOK — {checked} (well, seed) pairs bit-identical across {len(wells)} wells "
          f"({empty_zone_checked} from an EMPTY eval zone)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
