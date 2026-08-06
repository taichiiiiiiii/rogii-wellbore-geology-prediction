"""Fixed-weight (non-learned) candidate blends over the 18-candidate bank.

**Why after R3-v3.** Learned oracle-extraction lost 3 straight (R3-v1 well
classifier 9.2161 / R3-v2 well regressor 9.3765 / R3-v3 row-level meta-stack
9.8108 vs the 9.2309 stack-v2 baseline; ledger 2026-07-09/10): cross-well
models cannot predict *which* candidate is better where. A fixed convex/affine
blend sidesteps that entirely -- the same global weights apply to every row,
so there is no per-well decision to mistransfer. The question is only whether
candidate errors are decorrelated enough for averaging to help.

**Method: error Gram matrices, closed form.** With per-row errors
``e_k = pred_k - y`` for candidate ``k``, any weights ``w`` (``sum w = 1``)
give blend SSE ``w^T G w`` where ``G[i, j] = sum_rows e_i * e_j``. So:

  * best pair weight is analytic: ``w* = (Gjj - Gij) / (Gii + Gjj - 2 Gij)``;
  * best sum-1 affine blend solves ``G w = lam * 1`` -> ``w ~ G^-1 1``
    (tiny ridge added for conditioning);
  * a non-negativity constraint is handled by active-set: drop negative
    coordinates, re-solve on the survivors (fine at this scale);
  * greedy forward blending re-uses the same algebra per step.

**Honest CV.** All weights are *fitted* quantities, so they are chosen
per-fold from the other folds' Gram sum and scored on the held-out fold
(well-level GroupKFold(5) reusing ``stack_v2_oof.npz``'s ``row_fold`` via
:func:`run_meta_stack_cv.load_row_fold_aligned`, which asserts row-order
identity before trusting it). The in-sample (all-rows) optimum is also
printed, clearly labelled, as an upper-bound reference only -- adoption is
judged on the honest CV number against the usual gate (<= 9.1309 =
baseline 9.2309 - 0.10).

Usage::

    uv run python scripts/run_fixed_blend_grid.py

**Optional: ``--extra-pfbma PATH``.** Adds a second, independently-built
pf_bma bank npz (same schema as ``outputs/path_bank_pfbma.npz``, e.g. a
bank-builder v2 rerun at a different ``init_pos_std``) as 4 extra candidates
``pf_bma_scale{3,5,8,12}_v2``, aligned to the existing 18-way master row
order via the same ``block_slices``/``reindex_to_master`` convention
``run_router_v2_cv.build_router_v2_data`` uses (see
:func:`load_extra_pfbma_candidates`). When given, a second 22-way report
(in-sample optima / honest CV / non-negative weights) is printed *alongside*
the unchanged 18-way one, gated against the 18-way honest CV non-negative
result minus 0.05 (blend-vs-blend comparisons use a smaller -0.05 step, not
the -0.10 baseline gate)::

    uv run python scripts/run_fixed_blend_grid.py --extra-pfbma outputs/path_bank_pfbma_v2.npz
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_meta_stack_cv as MS  # noqa: E402  (load_row_fold_aligned, pooled_rmse)
import run_router_v2_cv as RV2  # noqa: E402  (build_router_v2_data)

OUTPUTS_DIR = _REPO_ROOT / "outputs"
BASELINE_POOLED_RMSE = 9.2309
ADOPTION_GATE_RMSE = BASELINE_POOLED_RMSE - 0.10
N_SPLITS = 5
RIDGE_EPS = 1e-8  # relative ridge on G for conditioning (candidates are highly correlated)
EXTRA_GATE_STEP = 0.05  # blend-vs-blend (22-way vs 18-way) gate is a smaller step than -0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra-pfbma",
        type=Path,
        default=None,
        help=(
            "Path to an additional pf_bma bank npz (same schema as "
            "outputs/path_bank_pfbma.npz). Adds pf_bma_scale{3,5,8,12}_v2 "
            "candidates and prints a second 22-way report alongside the "
            "unchanged 18-way one."
        ),
    )
    return parser.parse_args()


def load_extra_pfbma_candidates(
    path: Path,
    master_wells: list[str],
    master_boundaries: np.ndarray,
    master_y_true: np.ndarray,
    master_anchor: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load an extra pf_bma bank npz (same schema as ``path_bank_pfbma.npz``,
    e.g. bank-builder v2 rerun at a different ``init_pos_std``) and reindex
    its ``pf_bma_scale{S}_tvt`` arrays into the existing 18-way master row
    order as new candidates ``pf_bma_scale{S}_v2``.

    Reuses ``run_router_v2_cv``'s ``block_slices``/``reindex_to_master``
    (the exact convention :func:`RV2.build_router_v2_data` uses for its own
    ``path_bank_pfbma`` block) rather than reimplementing well-name
    alignment. ``y_true``/``anchor`` are cross-checked against the master
    arrays (the same style of check RV2 does for its ``path_bank_pfbma``
    ``y_true``) so a well-order or content mismatch fails loudly instead of
    silently poisoning the error Gram matrix.
    """
    with np.load(path, allow_pickle=True) as npz:
        well_idx = npz["well_idx"]
        used_wells = npz["used_wells"]
        y_true = npz["y_true"].astype(np.float64)
        anchor = npz["anchor"].astype(np.float32)
        scales = [float(s) for s in npz["bma_scales"]]
        arrays = {s: npz[f"pf_bma_scale{s:g}_tvt"].astype(np.float32) for s in scales}

    sl = RV2.block_slices(well_idx, used_wells)
    missing = sorted(set(master_wells) - {str(w) for w in used_wells})
    if missing:
        raise AssertionError(
            f"--extra-pfbma bank ({path}) is missing {len(missing)} of the "
            f"{len(master_wells)} master well(s) -- cannot align to the 18-way "
            f"master row order (e.g. {missing[:5]})"
        )

    bank_y = np.concatenate([y_true[sl[w][0] : sl[w][1]] for w in master_wells])
    if not np.allclose(bank_y, master_y_true.astype(np.float64), atol=1e-2):
        raise AssertionError(
            f"--extra-pfbma bank ({path})'s y_true disagrees with the 18-way "
            "master y_true for the joined wells -- alignment bug, not a leak, "
            "must not be ignored"
        )
    bank_anchor = np.concatenate([anchor[sl[w][0] : sl[w][1]] for w in master_wells])
    if not np.allclose(bank_anchor, master_anchor.astype(np.float32), atol=1e-2):
        raise AssertionError(
            f"--extra-pfbma bank ({path})'s anchor disagrees with the 18-way "
            "master anchor for the joined wells -- alignment bug, must not be ignored"
        )

    return {
        f"pf_bma_scale{s:g}_v2": RV2.reindex_to_master(arr, sl, master_wells, master_boundaries)
        for s, arr in arrays.items()
    }


def solve_sum1(g: np.ndarray) -> np.ndarray:
    """Minimise ``w^T G w`` s.t. ``sum w = 1`` (affine, weights may be negative)."""
    k = g.shape[0]
    ridge = RIDGE_EPS * np.trace(g) / k
    ginv_one = np.linalg.solve(g + ridge * np.eye(k), np.ones(k))
    return ginv_one / ginv_one.sum()


def solve_sum1_nonneg(g: np.ndarray) -> np.ndarray:
    """Simplified active-set non-negative version of :func:`solve_sum1` (drop
    negative coordinates, re-solve on survivors until all remaining weights
    are >= 0; dropped coordinates are never re-admitted). NOT full
    Lawson-Hanson: the 2026-07-10 review measured mild KKT violations on 3/5
    per-fold train Grams (slack -3e-3..-5e-3 relative) -- **conservative**
    direction only (exact NNLS gives CV 9.1044 < the 9.1104 this reports),
    and the all-data deployment weights pass the KKT check exactly (global
    optimum), so the shortcut understates rather than inflates performance."""
    k = g.shape[0]
    active = np.arange(k)
    for _ in range(k):
        w_sub = solve_sum1(g[np.ix_(active, active)])
        if (w_sub >= -1e-12).all():
            w = np.zeros(k)
            w[active] = np.clip(w_sub, 0.0, None)
            return w / w.sum()
        active = active[w_sub > 0]
        if active.size == 1:
            w = np.zeros(k)
            w[active[0]] = 1.0
            return w
    raise RuntimeError("active-set failed to converge")  # k iterations always suffice


def best_pair(g: np.ndarray) -> tuple[int, int, float, float]:
    """Analytic best 2-candidate convex blend: returns (i, j, w_i, sse)."""
    k = g.shape[0]
    best = (0, 0, 1.0, np.inf)
    for i in range(k):
        for j in range(i + 1, k):
            denom = g[i, i] + g[j, j] - 2.0 * g[i, j]
            w = 0.5 if denom <= 0 else float(np.clip((g[j, j] - g[i, j]) / denom, 0.0, 1.0))
            sse = w * w * g[i, i] + 2 * w * (1 - w) * g[i, j] + (1 - w) ** 2 * g[j, j]
            if sse < best[3]:
                best = (i, j, w, float(sse))
    return best


def greedy_forward(g: np.ndarray, max_k: int) -> list[np.ndarray]:
    """Greedy forward blend: start from the best single candidate, repeatedly
    add the candidate whose sum-1 non-negative re-solve over the chosen subset
    lowers SSE most. Returns the weight vector after each step (1..max_k)."""
    k = g.shape[0]
    chosen = [int(np.argmin(np.diag(g)))]
    out: list[np.ndarray] = []
    w = np.zeros(k)
    w[chosen[0]] = 1.0
    out.append(w)
    while len(chosen) < max_k:
        best_sse, best_c, best_w = np.inf, -1, None
        for c in range(k):
            if c in chosen:
                continue
            sub = chosen + [c]
            w_sub = solve_sum1_nonneg(g[np.ix_(sub, sub)])
            sse = float(w_sub @ g[np.ix_(sub, sub)] @ w_sub)
            if sse < best_sse:
                best_sse, best_c, best_w = sse, c, w_sub
        chosen.append(best_c)
        w = np.zeros(k)
        w[chosen] = best_w
        out.append(w)
    return out


def cv_score(per_fold_g: np.ndarray, n_rows_fold: np.ndarray, solver) -> tuple[float, list[float]]:
    """Honest CV: weights from the *other* folds' Gram sum, scored on the
    held-out fold. Returns (pooled RMSE, per-fold RMSE list)."""
    total_sse, total_n = 0.0, 0
    fold_rmse: list[float] = []
    for f in range(per_fold_g.shape[0]):
        g_train = per_fold_g.sum(axis=0) - per_fold_g[f]
        w = solver(g_train)
        sse_f = float(w @ per_fold_g[f] @ w)
        total_sse += sse_f
        total_n += int(n_rows_fold[f])
        fold_rmse.append(float(np.sqrt(sse_f / n_rows_fold[f])))
    return float(np.sqrt(total_sse / total_n)), fold_rmse


def run_gram_report(
    cands: tuple[str, ...],
    candidate_rows: dict[str, np.ndarray],
    y: np.ndarray,
    row_fold: np.ndarray,
    baseline_rmse: float,
    gate_rmse: float,
) -> dict[str, tuple[float, list[float]]]:
    """Full error-Gram blend report (in-sample optima + honest CV + verdict)
    for one candidate set. Parameterised over ``cands``/``candidate_rows`` so
    the identical code path serves both the 18-way baseline and any wider
    (e.g. 22-way) candidate set -- for the unmodified 18-way call (``cands``
    = ``rd.candidates``, ``baseline_rmse``/``gate_rmse`` = the module
    constants) this produces byte-identical output to the pre-refactor
    18-way-only script. Returns the honest-CV ``results`` dict (label ->
    (pooled RMSE, per-fold RMSE list)) so a caller can pull out a specific
    number (e.g. the non-negative solve) to gate a subsequent wider run.
    """
    k = len(cands)
    err = np.empty((y.shape[0], k), dtype=np.float64)
    for j, c in enumerate(cands):
        err[:, j] = candidate_rows[c].astype(np.float64) - y
    assert np.isfinite(err).all(), "candidate errors must be finite (bank NaN would poison G)"

    per_fold_g = np.zeros((N_SPLITS, k, k), dtype=np.float64)
    n_rows_fold = np.zeros(N_SPLITS, dtype=np.int64)
    for f in range(N_SPLITS):
        mask = row_fold == f
        e_f = err[mask]
        per_fold_g[f] = e_f.T @ e_f
        n_rows_fold[f] = int(mask.sum())
    del err
    assert int(n_rows_fold.sum()) == y.shape[0], "folds must partition all rows exactly"
    g_all = per_fold_g.sum(axis=0)
    n_all = int(n_rows_fold.sum())

    def rmse_all(w: np.ndarray) -> float:
        return float(np.sqrt(w @ g_all @ w / n_all))

    stack_i = cands.index("stack")
    w_stack = np.zeros(k)
    w_stack[stack_i] = 1.0
    print(f"\nbaseline (stack alone) pooled RMSE = {rmse_all(w_stack):.4f}")

    singles = np.sqrt(np.diag(g_all) / n_all)
    order = np.argsort(singles)
    print("\nsingle candidates (in-sample, reference):")
    for i in order[:6]:
        print(f"  {cands[i]:<28} {singles[i]:.4f}")

    # ---- in-sample optima (upper bounds, NOT adoption evidence) ----
    i, j, w_ij, sse_ij = best_pair(g_all)
    print("\nin-sample optima (reference only -- weights fitted on all rows):")
    print(
        f"  best pair : {w_ij:.3f}*{cands[i]} + {1 - w_ij:.3f}*{cands[j]}"
        f"  -> {np.sqrt(sse_ij / n_all):.4f}"
    )
    w_aff = solve_sum1(g_all)
    w_nn = solve_sum1_nonneg(g_all)
    print(f"  {k}-way affine (sum=1)      -> {rmse_all(w_aff):.4f}")
    print(f"  {k}-way non-negative        -> {rmse_all(w_nn):.4f}")
    for step_w in greedy_forward(g_all, max_k=4)[1:]:
        used = " + ".join(f"{step_w[c]:.3f}*{cands[c]}" for c in np.flatnonzero(step_w))
        print(f"  greedy k={np.count_nonzero(step_w)}: {used} -> {rmse_all(step_w):.4f}")

    # ---- honest CV (weights from other folds, scored on held-out fold) ----
    print(f"\nhonest CV (per-fold refit weights, gate <= {gate_rmse:.4f}):")

    def pair_solver(g_train: np.ndarray) -> np.ndarray:
        bi, bj, bw, _ = best_pair(g_train)
        w = np.zeros(k)
        w[bi], w[bj] = bw, 1.0 - bw
        return w

    results = {
        "best pair (refit/fold)": cv_score(per_fold_g, n_rows_fold, pair_solver),
        f"{k}-way affine (sum=1)": cv_score(per_fold_g, n_rows_fold, solve_sum1),
        f"{k}-way non-negative": cv_score(per_fold_g, n_rows_fold, solve_sum1_nonneg),
    }
    best_label, (best_rmse, _) = min(results.items(), key=lambda kv: kv[1][0])
    for label, (pooled, fold_rmse) in results.items():
        verdict = "PASS" if pooled <= gate_rmse else "FAIL"
        folds = " ".join(f"{r:.3f}" for r in fold_rmse)
        print(f"  {label:<26} pooled={pooled:.4f} [{verdict}]  folds: {folds}")

    verdict = "採用" if best_rmse <= gate_rmse else "却下（非改善）"
    print(f"\n判定: {verdict} -- best='{best_label}' pooled={best_rmse:.4f}, "
          f"baseline={baseline_rmse:.4f}, gate<={gate_rmse:.4f}")
    return results


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print("loading bank via run_router_v2_cv.build_router_v2_data ...")
    rd = RV2.build_router_v2_data(OUTPUTS_DIR)
    y = rd.y_true.astype(np.float64)
    cands = rd.candidates
    k = len(cands)
    print(f"  wells={rd.n_wells} candidates={k} rows={y.shape[0]:,}")

    row_fold = MS.load_row_fold_aligned(
        rd.y_true, rd.well_idx, rd.used_wells, OUTPUTS_DIR / "stack_v2_oof.npz"
    )

    merged_rows: dict[str, np.ndarray] | None = None
    extra_cands: tuple[str, ...] = ()
    if args.extra_pfbma is not None:
        master_wells = [str(w) for w in rd.used_wells]
        extra_candidate_rows = load_extra_pfbma_candidates(
            args.extra_pfbma,
            master_wells,
            rd.boundaries,
            rd.y_true,
            rd.candidate_rows["carry_last"],
        )
        extra_cands = tuple(extra_candidate_rows)
        print(
            f"  [extra-pfbma] {args.extra_pfbma}: +{len(extra_cands)} candidate(s) "
            f"({', '.join(extra_cands)})"
        )
        merged_rows = dict(rd.candidate_rows)
        merged_rows.update(extra_candidate_rows)

    # ---- unchanged 18-way report (identical output to the pre-refactor script) ----
    results_18 = run_gram_report(
        cands, rd.candidate_rows, y, row_fold, BASELINE_POOLED_RMSE, ADOPTION_GATE_RMSE
    )
    del rd.candidate_rows

    # ---- optional 22-way report (18-way bank + extra pf_bma_v2 candidates), printed alongside ----
    if merged_rows is not None:
        cands22 = cands + extra_cands
        baseline_22 = results_18[f"{k}-way non-negative"][0]
        gate_22 = baseline_22 - EXTRA_GATE_STEP
        print("\n" + "=" * 78)
        print(
            f"{len(cands22)}-way ({k}-way bank + {len(extra_cands)} extra pf_bma_v2 candidate(s))"
        )
        print("=" * 78)
        print(f"  wells={rd.n_wells} candidates={len(cands22)} rows={y.shape[0]:,}")
        run_gram_report(cands22, merged_rows, y, row_fold, baseline_22, gate_22)
        del merged_rows

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\npeak RSS: {peak_mb:.1f} MB   runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
