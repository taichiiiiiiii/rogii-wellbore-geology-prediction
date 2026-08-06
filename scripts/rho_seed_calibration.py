"""Detect a global run effect and put an upper bound on it, from repeated harness runs.

Why two statistics
------------------
The decision-relevant quantity is metric-weighted (pooled RMSE weights each well by its
row count). But in this pipeline the metric-weighted run variance is dominated by ~3 long,
unstable wells out of 40, so a test built on it has very little power.

A GLOBAL effect, by definition, appears in EVERY well. So a statistic that gives each well
equal say aggregates 40 near-independent pieces of evidence instead of 3. Measured power at
n_runs=5 under the real per-well variance structure:

    g (global share of public variance)   0.25   0.50   0.75
    metric-weighted statistic             0.17   0.31   0.61
    standardised statistic (const load)   1.00   1.00   1.00
    standardised statistic (load ~ sigma) 0.32   0.61   0.90

so the standardised statistic is used for DETECTION and for the upper bound, and the
metric-weighted one for reporting the decision-relevant magnitude.

Both use the same exact permutation null: under "pure per-well independent noise" the run
labels are exchangeable WITHIN each well, independently across wells. Permuting run labels
independently per well destroys any global alignment while preserving each well's marginal
noise distribution exactly (heavy tails and all).

Usage
-----
    uv run python scripts/rho_seed_calibration.py RESULT_JSON [--n-perm 20000] [--out OUT]

RESULT_JSON is the --out file written by scripts/estimate_rho_seed.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_PERM = 20000
N_SIM = 400
RNG_SEED = 12345
DEFAULT_N_PUBLIC = 52
DEFAULT_N_PRIVATE = 150


def rebuild(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild centred per-well MSE deviations u[runs, wells] and row counts."""
    wells = result["wells"]
    names = result["runs"]
    n_rows = np.array([result["n_rows_per_well"][w] for w in wells], dtype=float)
    mse = np.vstack(
        [
            np.array([result["per_well_rmse"][nm][w] for w in wells], dtype=float) ** 2
            for nm in names
        ]
    )
    return mse - mse.mean(axis=0, keepdims=True), n_rows


def mean_pair_corr(u: np.ndarray) -> float:
    """Mean pairwise correlation across wells of the run-deviation vectors (equal say)."""
    v = u.T
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    keep = nrm[:, 0] > 0
    v = v[keep] / nrm[keep]
    c = v @ v.T
    off = ~np.eye(c.shape[0], dtype=bool)
    return float(c[off].mean())


def weighted_var_a(u: np.ndarray, n_rows: np.ndarray) -> float:
    """Metric-weighted mean off-diagonal covariance == Var(a) in pooled-MSE units."""
    w = n_rows / n_rows.sum()
    cov = np.cov(u, rowvar=False, ddof=1)
    off = ~np.eye(u.shape[1], dtype=bool)
    ww = np.outer(w, w)
    return float((ww * cov)[off].sum() / ww[off].sum())


def permute_within_wells(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.argsort(rng.random(u.shape), axis=0)
    return np.take_along_axis(u, idx, axis=0)


def perm_test(u: np.ndarray, statfn, n_perm: int, rng: np.random.Generator) -> dict:
    obs = statfn(u)
    null = np.array([statfn(permute_within_wells(u, rng)) for _ in range(n_perm)])
    return {
        "observed": float(obs),
        "null_median": float(np.median(null)),
        "null_p95": float(np.percentile(null, 95)),
        "p_one_sided": float(((null >= obs).sum() + 1) / (n_perm + 1)),
    }


def v_local(u: np.ndarray, n_rows: np.ndarray, var_a: float) -> float:
    """Local part of the run-level pooled-MSE variance on this well set."""
    w = n_rows / n_rows.sum()
    well_var = np.var(u, axis=0, ddof=1)
    return float((w**2 * np.maximum(well_var - var_a, 0.0)).sum())


def rho_and_g(var_a: float, v_loc_here: float, n_here: int, n_pub: int, n_priv: int) -> dict:
    """Project to the real split. Local variance of a subset mean scales as 1/|S|."""
    var_a = max(var_a, 0.0)
    v_pub = var_a + v_loc_here * n_here / n_pub
    v_priv = var_a + v_loc_here * n_here / n_priv
    return {
        "rho_seed": float(var_a / np.sqrt(v_pub * v_priv)) if v_pub > 0 and v_priv > 0 else 0.0,
        "g_global_share_public": float(var_a / v_pub) if v_pub > 0 else 0.0,
    }


def simulate(u: np.ndarray, var_a_true: float, loading: np.ndarray, n_sim: int,
             rng: np.random.Generator) -> np.ndarray:
    """Distribution of the standardised statistic when the truth has Var(a)=var_a_true."""
    n_runs = u.shape[0]
    out = np.empty(n_sim)
    sd = np.sqrt(max(var_a_true, 0.0))
    for i in range(n_sim):
        eps = permute_within_wells(u, rng)  # resampled local noise, alignment destroyed
        a = rng.normal(0.0, sd, n_runs) if sd > 0 else np.zeros(n_runs)
        a -= a.mean()
        m = eps + a[:, None] * loading[None, :]
        out[i] = mean_pair_corr(m - m.mean(axis=0, keepdims=True))
    return out


def upper_bound(u: np.ndarray, n_rows: np.ndarray, obs: float, n_sim: int,
                rng: np.random.Generator, n_pub: int, n_priv: int) -> dict:
    """Largest true Var(a) whose simulated 2.5th percentile still sits below the observation."""
    sigma = np.sqrt(np.maximum(np.var(u, axis=0, ddof=1), 1e-12))
    w = n_rows / n_rows.sum()
    v_loc_naive = float((w**2 * sigma**2).sum())
    results = {}
    # Loadings are normalised so that sum_w w_w * load_w == 1. Without this, the same
    # Var(a) would mean a different metric-weighted global variance under each loading
    # and the resulting bounds on g would not be comparable.
    for name, loading in (
        ("constant", np.ones(u.shape[1])),
        ("prop_sigma", sigma / float((w * sigma).sum())),
    ):
        grid = np.concatenate([[0.0], np.geomspace(v_loc_naive * 1e-3, v_loc_naive * 20, 30)])
        hi = 0.0
        for gval in grid:
            sim = simulate(u, gval, loading, n_sim, rng)
            if np.percentile(sim, 2.5) <= obs:
                hi = gval
        proj = rho_and_g(hi, v_local(u, n_rows, hi), u.shape[1], n_pub, n_priv)
        results[name] = {"var_a_upper": float(hi), **proj}
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect and bound a global run effect.")
    ap.add_argument("result_json", type=Path)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-sim", type=int, default=N_SIM)
    ap.add_argument("--n-public", type=int, default=DEFAULT_N_PUBLIC)
    ap.add_argument("--n-private", type=int, default=DEFAULT_N_PRIVATE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    u, n_rows = rebuild(result)
    rng = np.random.default_rng(RNG_SEED)
    print(f"runs={u.shape[0]} wells={u.shape[1]}")

    std_test = perm_test(u, mean_pair_corr, args.n_perm, rng)
    wt_test = perm_test(u, lambda x: weighted_var_a(x, n_rows), args.n_perm, rng)
    print("\nSTANDARDISED statistic (mean pairwise correlation, equal say per well):")
    for k, v in std_test.items():
        print(f"  {k}: {v:.6g}")
    print("\nMETRIC-WEIGHTED statistic (Var(a) in pooled-MSE units):")
    for k, v in wt_test.items():
        print(f"  {k}: {v:.6g}")

    var_a_pt = max(wt_test["observed"], 0.0)
    point = rho_and_g(var_a_pt, v_local(u, n_rows, var_a_pt), u.shape[1],
                      args.n_public, args.n_private)
    print(f"\npoint estimate: rho_seed={point['rho_seed']:.4f}  "
          f"g={point['g_global_share_public']:.4f}")

    bounds = upper_bound(u, n_rows, std_test["observed"], args.n_sim, rng,
                         args.n_public, args.n_private)
    print("\n95% upper bounds from the standardised statistic:")
    for name, b in bounds.items():
        print(f"  loading={name:11s} Var(a)<={b['var_a_upper']:.5g}  "
              f"rho_seed<={b['rho_seed']:.4f}  g<={b['g_global_share_public']:.4f}")

    out = {"standardised_test": std_test, "weighted_test": wt_test,
           "point_estimate": point, "upper_bounds": bounds,
           "n_runs": int(u.shape[0]), "n_wells": int(u.shape[1])}
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
