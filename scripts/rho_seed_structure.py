"""Structural test for a GLOBAL run effect, using row-level predictions.

The variance route (scripts/estimate_rho_seed.py) answers the decision question directly
but has weak power with only a handful of runs, because the global variance is estimated
on (n_runs - 1) degrees of freedom. This script adds two higher-power tests whose power
comes from the WELL dimension instead:

1. AGREEMENT / ALIGNMENT (discrete). If run-to-run differences come from discrete branch
   flips inside each well, then for every well the runs partition into groups with
   byte-identical predictions. Under pure locality those partitions are INDEPENDENT
   across wells, so the pairwise run-agreement matrix has no structure. Under a global
   driver the same runs flip together in many wells and the matrix acquires block
   structure. Permuting run labels independently within each well gives an exact null.

2. SHARED DIRECTION (continuous). Let v_w be the R-vector of well w's per-run deviations,
   standardised to unit norm. Under locality the v_w point in independent random
   directions; under a global effect they share a direction. The leading eigenvalue of
   the mean outer product (1/W) sum_w v_w v_w^T detects this, and it tolerates a global
   effect whose per-well loading varies (which the constant-loading variance estimator
   does not). Same within-well permutation null.

Neither test replaces the variance decomposition -- they establish PRESENCE of a global
effect, not its decision-relevant magnitude -- but a clean null here at high power is
strong evidence for locality.

Usage
-----
    uv run python scripts/rho_seed_structure.py RUN_DIR [RUN_DIR ...] [--out OUT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_PERM = 20000
RNG_SEED = 99
ATOL = 0.0  # exact byte-level equality by default


def load_predictions(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (pred[runs, rows], well_index[rows], run_names) aligned on submission id."""
    series = {}
    for d in run_dirs:
        p = d / "submission.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing")
        s = pd.read_csv(p)
        series[d.name] = s.set_index("id")["tvt"].astype(float)

    names = list(series)
    common = series[names[0]].index
    for nm in names[1:]:
        common = common.intersection(series[nm].index)
    common = common.sort_values()
    pred = np.vstack([series[nm].loc[common].to_numpy() for nm in names])
    wells = pd.Index(common).str.rsplit("_", n=1).str[0].to_numpy()
    return pred, wells, names


def well_partitions(pred: np.ndarray, wells: np.ndarray, atol: float) -> tuple[np.ndarray, dict]:
    """For each well, label the runs by which identical-prediction group they fall into."""
    uniq = sorted(set(wells.tolist()))
    n_runs = pred.shape[0]
    labels = np.zeros((len(uniq), n_runs), dtype=int)
    info = {}
    for i, w in enumerate(uniq):
        block = pred[:, wells == w]
        lab = -np.ones(n_runs, dtype=int)
        nxt = 0
        for r in range(n_runs):
            if lab[r] >= 0:
                continue
            lab[r] = nxt
            for r2 in range(r + 1, n_runs):
                if lab[r2] < 0:
                    same = (
                        np.array_equal(block[r], block[r2])
                        if atol == 0
                        else np.allclose(block[r], block[r2], rtol=0, atol=atol)
                    )
                    if same:
                        lab[r2] = nxt
            nxt += 1
        labels[i] = lab
        info[w] = int(nxt)
    return labels, info


def agreement_matrix(labels: np.ndarray) -> np.ndarray:
    """A[r, r'] = fraction of wells where runs r and r' produced identical predictions."""
    n_runs = labels.shape[1]
    a = np.zeros((n_runs, n_runs))
    for r in range(n_runs):
        for r2 in range(n_runs):
            a[r, r2] = float(np.mean(labels[:, r] == labels[:, r2]))
    return a


def agreement_stat(labels: np.ndarray) -> float:
    """Spread of off-diagonal agreement; 0 when every run pair agrees equally often."""
    a = agreement_matrix(labels)
    n = a.shape[0]
    off = a[~np.eye(n, dtype=bool)]
    return float(off.var())


def perm_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute run labels within each well."""
    n_wells, n_runs = labels.shape
    idx = np.argsort(rng.random((n_wells, n_runs)), axis=1)
    return np.take_along_axis(labels, idx, axis=1)


def eig_stat(v: np.ndarray) -> float:
    """Leading eigenvalue of the mean outer product of unit-norm per-well deviation vectors."""
    c = v.T @ v / v.shape[0]
    return float(np.linalg.eigvalsh(c)[-1])


def unit_deviation_vectors(mse: np.ndarray) -> np.ndarray:
    """Per-well run-deviation vectors, standardised to unit norm (wells x runs)."""
    u = (mse - mse.mean(axis=0, keepdims=True)).T  # wells x runs
    nrm = np.linalg.norm(u, axis=1, keepdims=True)
    keep = (nrm[:, 0] > 0)
    return u[keep] / nrm[keep]


def perm_rows(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.argsort(rng.random(v.shape), axis=1)
    return np.take_along_axis(v, idx, axis=1)


def permutation_test(obs: float, sample_fn, n_perm: int) -> dict:
    null = np.array([sample_fn() for _ in range(n_perm)])
    p = float(((null >= obs).sum() + 1) / (n_perm + 1))
    return {
        "observed": float(obs),
        "null_median": float(np.median(null)),
        "null_p95": float(np.percentile(null, 95)),
        "null_p99": float(np.percentile(null, 99)),
        "p_one_sided": p,
    }


def per_well_mse(run_dirs: list[Path]) -> tuple[np.ndarray, list[str], list[str]]:
    frames = {d.name: pd.read_csv(d / "cv_report.csv").set_index("well_id").sort_index()
              for d in run_dirs}
    names = list(frames)
    wells = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    mse = np.vstack([frames[nm].loc[wells, "rmse"].to_numpy(dtype=float) ** 2 for nm in names])
    return mse, wells, names


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural test for a global run effect.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--atol", type=float, default=ATOL)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    pred, wells, names = load_predictions(args.run_dirs)
    n_runs = len(names)
    print(f"runs={n_runs}  rows={pred.shape[1]}  wells={len(set(wells.tolist()))}")

    labels, nvar = well_partitions(pred, wells, args.atol)
    n_identical = sum(1 for v in nvar.values() if v == 1)
    print(f"\nwells bit-identical across all runs: {n_identical} / {len(nvar)}")
    print(f"distribution of distinct prediction variants per well: "
          f"{pd.Series(list(nvar.values())).value_counts().sort_index().to_dict()}")

    amat = agreement_matrix(labels)
    print("\npairwise run agreement (fraction of wells with identical predictions):")
    print(pd.DataFrame(amat, index=names, columns=names).round(3).to_string())

    varying = labels[[i for i, w in enumerate(sorted(nvar)) if nvar[w] > 1]]
    align = None
    if varying.shape[0] >= 2 and n_runs >= 3:
        obs = agreement_stat(varying)
        align = permutation_test(
            obs, lambda: agreement_stat(perm_labels(varying, rng)), args.n_perm
        )
        print(f"\nALIGNMENT test on the {varying.shape[0]} wells that vary:")
        for k, v in align.items():
            print(f"  {k}: {v}")
    else:
        print("\nALIGNMENT test skipped (need >=2 varying wells and >=3 runs)")

    mse, wl, nm2 = per_well_mse(args.run_dirs)
    v = unit_deviation_vectors(mse)
    shared = None
    if v.shape[0] >= 2 and n_runs >= 3:
        obs_e = eig_stat(v)
        shared = permutation_test(obs_e, lambda: eig_stat(perm_rows(v, rng)), args.n_perm)
        print(f"\nSHARED-DIRECTION test on {v.shape[0]} wells with nonzero deviation:")
        for k, val in shared.items():
            print(f"  {k}: {val}")

    result = {
        "runs": names,
        "n_wells": len(nvar),
        "n_wells_bit_identical": n_identical,
        "variants_per_well": nvar,
        "agreement_matrix": {n: dict(zip(names, amat[i].tolist(), strict=True))
                             for i, n in enumerate(names)},
        "alignment_test": align,
        "shared_direction_test": shared,
        "wells_order": wl,
        "mse_runs": nm2,
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
