"""Estimate rho_seed: correlation between run-level noise on disjoint well subsets.

Background
----------
Byte-identical reruns of the submission kernel score differently on the public LB.
The particle-filter seeds are deterministic loop indices, so the spread is *numerical*
nondeterminism (thread scheduling, float reduction order, GPU atomics). The strategic
question is whether that noise is

  (a) PER-WELL LOCAL   -- generated independently inside each well's computation, or
  (b) GLOBAL           -- a shared per-run effect hitting every well coherently.

Under (a) the noise on the public wells is independent of the noise on the private
wells, so picking the best public draw buys nothing on private. Under (b) a lucky run
is lucky on both, and best-of-N selection transfers.

This script consumes several independent runs of the SAME configuration of the local CV
harness (K held-out wells) and decomposes the run-to-run variation.

Estimator
---------
For run r and well w let m[r,w] be the well's mean squared error and n[w] its row count.
Pooled MSE on a well subset S is the n-weighted mean of m over S; pooled RMSE is its
square root (the competition metric).

Centre each well across runs: u[r,w] = m[r,w] - mean_r m[r,w], and model

    u[r,w] = a[r] + eps[r,w],     eps independent across wells.

For two DISJOINT subsets A and B the local parts are independent, hence

    Cov_r(D_A, D_B) = Var(a)          where D_S is the weighted mean of u over S,

so the split-half covariance isolates the global variance, and

    rho_seed = Cov(D_A, D_B) / sqrt(Var(D_A) Var(D_B)).

The same identity averaged over all distinct well pairs gives a higher-precision
estimate of Var(a) (the "mean off-diagonal covariance" estimator), which is then
projected onto the real public/private split sizes.

Usage
-----
    uv run python scripts/estimate_rho_seed.py RUN_DIR [RUN_DIR ...] [--out OUT.json]

Each RUN_DIR must contain cv_report.csv (well_id, n_rows, rmse). submission.csv is used
when present for a row-level view of where the noise actually lives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_N_PUBLIC = 52
DEFAULT_N_PRIVATE = 150
N_SPLITS = 4000
RNG_SEED = 0


def load_runs(run_dirs: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Return (well_ids, n_rows, mse[runs, wells], run_names) aligned across runs."""
    frames: dict[str, pd.DataFrame] = {}
    for d in run_dirs:
        rep = d / "cv_report.csv"
        if not rep.exists():
            raise FileNotFoundError(f"{rep} missing")
        frames[d.name] = pd.read_csv(rep).set_index("well_id").sort_index()

    names = list(frames)
    wells = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    if not wells:
        raise ValueError("no common wells across runs")

    n_rows = frames[names[0]].loc[wells, "n_rows"].to_numpy(dtype=float)
    for nm, f in frames.items():
        if not np.array_equal(n_rows, f.loc[wells, "n_rows"].to_numpy(dtype=float)):
            raise ValueError(f"run {nm} has different per-well row counts; runs not comparable")

    mse = np.vstack([frames[nm].loc[wells, "rmse"].to_numpy(dtype=float) ** 2 for nm in names])
    return wells, n_rows, mse, names


def pooled_rmse(mse: np.ndarray, n_rows: np.ndarray) -> np.ndarray:
    """Pooled RMSE per run over all wells."""
    return np.sqrt((mse * n_rows).sum(axis=1) / n_rows.sum())


def _weighted_dev(u: np.ndarray, n_rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Run-level pooled-MSE deviation on the well subset idx."""
    n = n_rows[idx]
    return (u[:, idx] * n).sum(axis=1) / n.sum()


def split_half_rho(
    u: np.ndarray, n_rows: np.ndarray, n_splits: int, rng: np.random.Generator
) -> dict:
    """Correlate run-level deviations on random disjoint halves; return the distribution."""
    n_wells = u.shape[1]
    half = n_wells // 2
    rhos, covs, betas = [], [], []
    for _ in range(n_splits):
        perm = rng.permutation(n_wells)
        a_idx, b_idx = perm[:half], perm[half : 2 * half]
        d_a = _weighted_dev(u, n_rows, a_idx)
        d_b = _weighted_dev(u, n_rows, b_idx)
        va, vb = d_a.var(ddof=1), d_b.var(ddof=1)
        cab = float(np.cov(d_a, d_b, ddof=1)[0, 1])
        covs.append(cab)
        if va > 0 and vb > 0:
            rhos.append(cab / np.sqrt(va * vb))
            betas.append(cab / va)
    rhos = np.asarray(rhos)
    covs = np.asarray(covs)
    betas = np.asarray(betas)
    qs = [2.5, 10, 25, 50, 75, 90, 97.5]
    return {
        "n_splits_used": int(rhos.size),
        "rho_mean": float(rhos.mean()),
        "rho_median": float(np.median(rhos)),
        "rho_quantiles": {str(q): float(np.percentile(rhos, q)) for q in qs},
        "rho_frac_positive": float((rhos > 0).mean()),
        "cov_mean": float(covs.mean()),
        "cov_frac_positive": float((covs > 0).mean()),
        "beta_mean": float(betas.mean()),
        "beta_median": float(np.median(betas)),
    }


def variance_decomposition(u: np.ndarray, n_rows: np.ndarray) -> dict:
    """Split run-to-run variation into a global run effect and per-well local noise."""
    n_runs, n_wells = u.shape
    cov = np.cov(u, rowvar=False, ddof=1)  # wells x wells, across runs
    w = n_rows / n_rows.sum()
    off = ~np.eye(n_wells, dtype=bool)
    ww = np.outer(w, w)
    # Weighted mean off-diagonal covariance == unbiased estimate of Var(a).
    var_a = float((ww * cov)[off].sum() / ww[off].sum())
    well_var = np.diag(cov)
    # Pure-local prediction from the RAW per-well variances. This is the H0 benchmark and
    # must not subtract var_a, otherwise the comparison against v_obs becomes circular.
    v_loc_naive = float((w**2 * well_var).sum())
    # Local part after removing the estimated global component (used for projection).
    sigma_local_w = np.maximum(well_var - var_a, 0.0)
    v_loc_here = float((w**2 * sigma_local_w).sum())
    d_all = _weighted_dev(u, n_rows, np.arange(n_wells))
    v_obs = float(d_all.var(ddof=1))
    return {
        "n_runs": int(n_runs),
        "n_wells": int(n_wells),
        "var_a_hat": var_a,
        "var_a_hat_clamped": max(var_a, 0.0),
        "mean_per_well_variance": float(well_var.mean()),
        "v_local_this_wellset": v_loc_here,
        "observed_run_var_pooled_mse": v_obs,
        "pure_local_predicted_run_var_pooled_mse": v_loc_naive,
        "ratio_observed_over_pure_local": (
            float(v_obs / v_loc_naive) if v_loc_naive > 0 else float("inf")
        ),
        "global_share_of_run_variance": (
            float(max(var_a, 0.0) / v_obs) if v_obs > 0 else float("nan")
        ),
    }


def anova_two_way(u: np.ndarray, n_rows: np.ndarray) -> dict:
    """Weighted two-way ANOVA (runs x wells, no replication) testing for a global run effect."""
    n_runs, n_wells = u.shape
    w = n_rows / n_rows.mean()  # relative weights, mean 1
    run_eff = (u * w).sum(axis=1) / w.sum()  # weighted per-run mean deviation
    ss_run = float(w.sum() * (run_eff**2).sum())
    resid = u - run_eff[:, None]
    ss_res = float((w[None, :] * resid**2).sum())
    df_run = n_runs - 1
    df_res = (n_runs - 1) * (n_wells - 1)
    ms_run = ss_run / df_run if df_run else float("nan")
    ms_res = ss_res / df_res if df_res else float("nan")
    f_stat = ms_run / ms_res if ms_res > 0 else float("nan")
    return {
        "df_run": df_run,
        "df_resid": df_res,
        "ms_run": ms_run,
        "ms_resid": ms_res,
        "F": f_stat,
        "p_value": float(stats.f.sf(f_stat, df_run, df_res)),
    }


def project_split(var_a: float, v_loc_here: float, n_here: int, n_pub: int, n_priv: int) -> dict:
    """rho and transfer beta for a public/private split, assuming exchangeable wells.

    Local variance of a subset mean scales as 1/|S|, so V_loc(m) = v_loc_here * n_here / m.
    """
    v_loc_pub = v_loc_here * n_here / n_pub
    v_loc_priv = v_loc_here * n_here / n_priv
    v_pub = var_a + v_loc_pub
    v_priv = var_a + v_loc_priv
    rho = var_a / np.sqrt(v_pub * v_priv) if v_pub > 0 and v_priv > 0 else float("nan")
    beta = var_a / v_pub if v_pub > 0 else float("nan")
    return {
        "n_public": n_pub,
        "n_private": n_priv,
        "rho_seed": float(rho),
        "transfer_beta": float(beta),
        "v_local_public": float(v_loc_pub),
        "v_local_private": float(v_loc_priv),
        "var_public": float(v_pub),
        "var_private": float(v_priv),
    }


def rowlevel_report(run_dirs: list[Path]) -> dict | None:
    """Where does the run-to-run difference actually live, row by row?"""
    subs = {}
    for d in run_dirs:
        p = d / "submission.csv"
        if not p.exists():
            return None
        s = pd.read_csv(p)
        if not {"id", "tvt"}.issubset(s.columns):
            return None
        subs[d.name] = s.set_index("id")["tvt"].astype(float)

    names = list(subs)
    common = subs[names[0]].index
    for nm in names[1:]:
        common = common.intersection(subs[nm].index)
    mat = np.vstack([subs[nm].loc[common].to_numpy() for nm in names])
    well_of = pd.Index(common).str.rsplit("_", n=1).str[0]

    spread = mat.max(axis=0) - mat.min(axis=0)
    changed = spread > 0
    per_well = pd.DataFrame({"well": well_of, "spread": spread, "changed": changed})
    g = per_well.groupby("well")
    frac_changed = g["changed"].mean()
    max_spread = g["spread"].max()
    return {
        "n_rows_compared": int(mat.shape[1]),
        "frac_rows_differing_between_runs": float(changed.mean()),
        "n_wells_with_any_difference": int((frac_changed > 0).sum()),
        "n_wells_total": int(len(frac_changed)),
        "wells_bit_identical": sorted(frac_changed.index[frac_changed == 0].tolist()),
        "max_abs_row_spread": float(spread.max()),
        "median_spread_on_changed_rows": (
            float(np.median(spread[changed])) if changed.any() else 0.0
        ),
        "per_well_frac_changed": {k: float(v) for k, v in frac_changed.items()},
        "per_well_max_spread": {k: float(v) for k, v in max_spread.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate rho_seed from repeated harness runs.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-public", type=int, default=DEFAULT_N_PUBLIC)
    ap.add_argument("--n-private", type=int, default=DEFAULT_N_PRIVATE)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = ap.parse_args()

    wells, n_rows, mse, names = load_runs(args.run_dirs)
    n_runs = len(names)
    rmse_runs = pooled_rmse(mse, n_rows)

    print(f"runs: {n_runs}   wells: {len(wells)}   rows: {int(n_rows.sum())}")
    for nm, v in zip(names, rmse_runs, strict=True):
        print(f"  {nm:28s} pooled RMSE = {v:.6f}")
    sd = rmse_runs.std(ddof=1) if n_runs > 1 else float("nan")
    print(f"  mean={rmse_runs.mean():.6f}  sd={sd:.6f}  rel_sd={sd / rmse_runs.mean():.5%}")
    if np.ptp(rmse_runs) == 0:
        print("  !! runs are numerically identical -- harness does NOT reproduce the LB spread")

    u = mse - mse.mean(axis=0, keepdims=True)  # centre each well across runs
    rng = np.random.default_rng(RNG_SEED)

    var = variance_decomposition(u, n_rows)
    anova = anova_two_way(u, n_rows)
    halves = split_half_rho(u, n_rows, args.n_splits, rng)
    proj = project_split(
        var["var_a_hat"], var["v_local_this_wellset"], len(wells), args.n_public, args.n_private
    )

    print("\n--- variance decomposition (units: pooled MSE) ---")
    for k, v in var.items():
        print(f"  {k}: {v}")
    print("\n--- two-way ANOVA for a global run effect ---")
    for k, v in anova.items():
        print(f"  {k}: {v}")
    print(f"\n--- split-half rho over {halves['n_splits_used']} random disjoint halves ---")
    for k, v in halves.items():
        print(f"  {k}: {v}")
    print(f"\n--- projection to {args.n_public} public / {args.n_private} private ---")
    for k, v in proj.items():
        print(f"  {k}: {v}")

    row = rowlevel_report(args.run_dirs)
    if row is not None:
        print("\n--- row-level view ---")
        for k, v in row.items():
            if not k.startswith("per_well") and k != "wells_bit_identical":
                print(f"  {k}: {v}")
        print(f"  wells bit-identical across all runs: {len(row['wells_bit_identical'])}")

    result = {
        "runs": names,
        "pooled_rmse": {nm: float(v) for nm, v in zip(names, rmse_runs, strict=True)},
        "pooled_rmse_mean": float(rmse_runs.mean()),
        "pooled_rmse_sd": float(sd) if n_runs > 1 else None,
        "wells": wells,
        "n_rows_per_well": {w: int(n) for w, n in zip(wells, n_rows, strict=True)},
        "variance_decomposition": var,
        "anova": anova,
        "split_half": halves,
        "projection": proj,
        "row_level": row,
        "per_well_rmse": {
            nm: {w: float(np.sqrt(mse[i, j])) for j, w in enumerate(wells)}
            for i, nm in enumerate(names)
        },
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
