"""Prefix-cut backtest features per well (design.md, R3-v1 follow-up).

**Motivation.** The per-well candidate router (R3-v1, ``scripts/run_router_cv.py``)
failed because none of its features directly measured *relative candidate
quality* -- they were well-level diagnostics (PF std, beam margin, GR NaN
rate, ...) correlated with difficulty, not with which candidate actually wins
on that well. This script builds a feature that *is* a direct relative-quality
measurement and is legal at hidden-test time: for each well, take the known
prefix (the contiguous block where ``TVT_input`` is non-NaN), cut it short at
a few points, treat the cut-off tail of the *known* prefix as a pseudo
evaluation zone (its true ``TVT`` is known -- it is the never-visible-at-
inference-time ground truth of a still-legal, already-known stretch of the
well), and actually run each candidate predictor on the truncated well. The
resulting backtest RMSE is a genuine, per-candidate, per-well quality signal
computable identically on a hidden test well (it only ever reads that well's
own known-zone ``TVT_input`` prefix).

**Leak safety.** ``truncate_known_prefix`` builds the truncated well by
slicing ``h`` down to exactly the known prefix (``h.iloc[:prefix_len]`` --
the real, downstream eval-zone rows are dropped entirely, not just masked)
and then NaN-ing out ``TVT_input`` for the tail slice ``[cut_idx:]``. The
pseudo ground truth (``TVT`` at those same positions) is read once, before
the NaN-ing, purely for scoring -- it is never placed anywhere a predictor
can see it, exactly mirroring the real leak boundary (``TVT`` vs
``TVT_input``, see ``docs/playbooks/00_common.md``). ``tests/test_backtest_features.py``
asserts both the pseudo-eval mask position and the NaN-out.

**Candidates** (task-scoped exactly to these three; ``particle.py`` is
read-only -- untouched, imported only):

1. ``carry_last`` -- ``baseline.predict_carry_last`` (flat anchor extension).
2. ``pf`` -- ``registration.particle.track_pf`` (single run, default
   ``n_particles=512``, ``seed=0``; *not* ``track_pf_multi``'s 8-seed
   ensemble -- the task names ``track_pf`` explicitly, and a single run keeps
   the 3-cuts-x-3-candidates-per-well cost down given this is a 3 GB-RAM box
   with another CV job already running).
3. ``beam`` -- ``registration.beam.track_beam_multi`` with its module
   default ``DEFAULT_BEAM_CONFIGS`` (the already-adopted 3-config ensemble,
   see ``docs/playbooks/00_common.md``'s "beam 採用設定") -- chosen over a
   single ``track_beam`` call because that ensemble is the canonical "beam"
   candidate used everywhere else in this repo (``run_router_cv.py``,
   ``run_beam_grid_cv.py``), so features built from it are representative of
   the beam candidate a router would actually pick between.

**spatial candidate: omitted.** ``rogii.spatial.predict_spatial`` needs a
leave-self-out ``SurfaceBank`` built from *every* train well's formation-depth
columns; per ``scripts/build_spatial_cache.py``'s own docstring that bank
build is a **~35-minute fixed cost independent of how many wells are then
queried**. Paying that once per script invocation (smoke or full) is
disproportionate to a feature-building pass that is supposed to run cheaply
per well, and it does not shrink with ``--limit``. Logged here, as required,
rather than silently dropped: this is the "may omit if heavy" branch the
task allows.

**Time gate.** A probe over ``TIME_GATE_N_WELLS`` wells measures per-well
wall-clock time for the full 3-candidate x 3-cut pipeline and projects it to
773 wells. If the probe's per-well time exceeds ``PER_WELL_TIME_BUDGET_S``
(5s), the cut-fraction grid degrades from 3 points (0.5/0.7/0.85) to 2
(0.5/0.85, dropping the middle point) and the probe is re-run once to confirm
-- exactly the degradation strategy the task names as an example. The chosen
grid is then used for every well in the actual run (``--limit`` or full).

**Memory.** 3 GB RAM box, another CV job already resident. Every well's
``DataFrame``s (``h``, ``tw``, and each cut's truncated ``h``) are processed
and dropped immediately; only a small per-well feature dict survives into a
preallocated ``float32`` matrix.

Usage::

    uv run python scripts/build_backtest_features.py --limit 30 \\
        --out outputs/backtest_features_smoke.npz
    uv run python scripts/build_backtest_features.py   # full 773-well run
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from rogii import baseline  # noqa: E402
from rogii import data as D  # noqa: E402
from rogii.registration.beam import track_beam_multi  # noqa: E402
from rogii.registration.particle import track_pf  # noqa: E402

OOF_PATH = _REPO_ROOT / "outputs" / "stack_v2_oof.npz"
DEFAULT_OUT_PATH = _REPO_ROOT / "outputs" / "backtest_features.npz"

CANDIDATES: tuple[str, ...] = ("carry_last", "pf", "beam")
CANDIDATE_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(CANDIDATES, 2))
DEFAULT_CUT_FRACS: tuple[float, ...] = (0.5, 0.7, 0.85)
DEGRADED_CUT_FRACS: tuple[float, ...] = (0.5, 0.85)  # drop the middle point if too slow

MIN_PREFIX_ROWS = 4  # below this a prefix cannot be meaningfully split

TIME_GATE_N_WELLS = 20
PER_WELL_TIME_BUDGET_S = 5.0
FULL_N_WELLS_PROJECTION = 773
PROGRESS_EVERY = 10


# --------------------------------------------------------------------------- #
# truncation helper (unit-tested in tests/test_backtest_features.py)
# --------------------------------------------------------------------------- #


@dataclass
class TruncatedWell:
    """A known-prefix-only well DataFrame cut at ``cut_idx``, plus its pseudo ground truth.

    ``h`` has exactly ``prefix_len`` rows (the well's real eval zone is
    dropped, not masked) with ``TVT_input`` NaN-ed out for positions
    ``[cut_idx, prefix_len)`` -- i.e. ``data.eval_mask(h)`` is ``True`` on
    exactly that pseudo eval zone. ``y_true`` is the corresponding true
    ``TVT`` slice (same positions), captured before the NaN-out so it is
    never reachable from ``h`` itself.
    """

    h: pd.DataFrame
    y_true: np.ndarray
    prefix_len: int
    cut_idx: int
    pseudo_eval_len: int


def truncate_known_prefix(
    h: pd.DataFrame, cut_frac: float, min_prefix_rows: int = MIN_PREFIX_ROWS
) -> TruncatedWell | None:
    """Cut ``h``'s known ``TVT_input`` prefix at ``cut_frac`` and build a pseudo eval zone.

    ``cut_idx = round(cut_frac * prefix_len)``, clipped to ``[1, prefix_len - 1]``
    so at least one known row remains (an anchor for the predictors) and the
    pseudo eval zone is non-empty. Returns ``None`` (a well/cut combination to
    skip, not a fatal error) when the well's known prefix has fewer than
    ``min_prefix_rows`` rows or has no ``TVT`` column (train-only; needed to
    score the pseudo eval zone).
    """
    if not 0.0 < cut_frac < 1.0:
        raise ValueError(f"cut_frac must be in (0, 1), got {cut_frac}")

    tvt_input = h["TVT_input"].to_numpy(dtype=float)
    known_idx = np.flatnonzero(np.isfinite(tvt_input))
    if known_idx.size < min_prefix_rows:
        return None

    prefix_len = int(known_idx[-1]) + 1
    if prefix_len < min_prefix_rows or "TVT" not in h.columns:
        return None

    cut_idx = int(round(cut_frac * prefix_len))
    cut_idx = max(1, min(cut_idx, prefix_len - 1))

    h_trunc = h.iloc[:prefix_len].reset_index(drop=True).copy()
    y_true = h_trunc["TVT"].to_numpy(dtype=float)[cut_idx:prefix_len].copy()

    tvt_input_trunc = h_trunc["TVT_input"].to_numpy(dtype=float, copy=True)
    tvt_input_trunc[cut_idx:] = np.nan
    h_trunc["TVT_input"] = tvt_input_trunc

    return TruncatedWell(
        h=h_trunc,
        y_true=y_true,
        prefix_len=prefix_len,
        cut_idx=cut_idx,
        pseudo_eval_len=prefix_len - cut_idx,
    )


# --------------------------------------------------------------------------- #
# pure scoring / diversity helpers (unit-tested)
# --------------------------------------------------------------------------- #


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _rank_from_values(values: dict[str, float]) -> dict[str, float]:
    """Rank ``CANDIDATES`` by ascending value (1 = best); NaN ranks last, ties stable."""
    arr = np.array([values.get(c, float("nan")) for c in CANDIDATES])
    key = np.where(np.isfinite(arr), arr, np.inf)
    order = np.argsort(key, kind="stable")
    ranks = np.empty(len(CANDIDATES))
    ranks[order] = np.arange(1, len(CANDIDATES) + 1)
    return {c: float(ranks[i]) for i, c in enumerate(CANDIDATES)}


def _best_second_margin(values: dict[str, float]) -> float:
    """Gap between the best and second-best finite value, or NaN if <2 are finite."""
    arr = np.array([v for v in values.values() if np.isfinite(v)])
    if arr.size < 2:
        return float("nan")
    arr_sorted = np.sort(arr)
    return float(arr_sorted[1] - arr_sorted[0])


def _pairwise_agreement(preds: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """Mean absolute difference between each candidate pair's predicted paths."""
    result: dict[tuple[str, str], float] = {}
    for a, b in CANDIDATE_PAIRS:
        pa, pb = preds.get(a), preds.get(b)
        if pa is not None and pb is not None and pa.shape == pb.shape and pa.size:
            result[(a, b)] = float(np.mean(np.abs(pa - pb)))
        else:
            result[(a, b)] = float("nan")
    return result


def _predict_candidate(name: str, h_trunc: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray | None:
    """Run one candidate predictor on a truncated well; ``None`` on any failure.

    ``track_pf``/``track_beam_multi`` already never raise (module contracts),
    but ``baseline.predict_carry_last`` can raise ``ValueError`` if there is
    no known anchor -- wrapped here too so a single bad well/cut never aborts
    the run (task requirement: 1-well failures degrade to NaN, not a crash).
    """
    try:
        if name == "carry_last":
            return baseline.predict_carry_last(h_trunc)
        if name == "pf":
            return track_pf(h_trunc, tw).tvt
        if name == "beam":
            return track_beam_multi(h_trunc, tw).tvt
        raise ValueError(f"unknown candidate {name!r}")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# feature schema + per-well row builder
# --------------------------------------------------------------------------- #


def build_feature_names(cut_fracs: tuple[float, ...]) -> list[str]:
    """Deterministic column order for the well x feature matrix (fixed for the whole run)."""
    names: list[str] = ["prefix_len"]
    names += [f"pseudo_eval_len_cut{cut:g}" for cut in cut_fracs]

    names += [f"bt_rmse_{cand}_cut{cut:g}" for cand in CANDIDATES for cut in cut_fracs]
    for cand in CANDIDATES:
        names += [f"bt_rmse_{cand}_mean", f"bt_rmse_{cand}_worst"]

    names += [f"bt_rank_{cand}_cut{cut:g}" for cut in cut_fracs for cand in CANDIDATES]
    names += [f"bt_rank_{cand}_mean" for cand in CANDIDATES]

    names += [f"bt_margin_best_second_cut{cut:g}" for cut in cut_fracs]
    names.append("bt_margin_best_second_mean")

    names += [
        f"bt_agree_{a}_{b}_cut{cut:g}" for cut in cut_fracs for a, b in CANDIDATE_PAIRS
    ]
    names += [f"bt_agree_mean_cut{cut:g}" for cut in cut_fracs]
    names.append("bt_agree_mean")
    return names


def compute_well_row(
    h: pd.DataFrame, tw: pd.DataFrame, cut_fracs: tuple[float, ...]
) -> tuple[dict[str, float], dict[str, int]]:
    """Build one well's feature dict (keys == ``build_feature_names(cut_fracs)``).

    Returns ``(row, diag)``; ``diag`` counts truncation/candidate failures for
    this well (used by the caller to aggregate a NaN-rate / failure report).
    Never raises: candidate/truncation failures degrade to NaN entries, not
    an exception (the caller still wraps the whole call for well-load
    failures, e.g. a corrupt CSV).
    """
    diag = {"truncate_fail": 0, "candidate_fail": 0}
    row: dict[str, float] = {}

    tvt_input_full = h["TVT_input"].to_numpy(dtype=float)
    known_idx = np.flatnonzero(np.isfinite(tvt_input_full))
    prefix_len = int(known_idx[-1]) + 1 if known_idx.size else 0
    row["prefix_len"] = float(prefix_len)

    rmse_by_cut: dict[float, dict[str, float]] = {}
    preds_by_cut: dict[float, dict[str, np.ndarray]] = {}
    pseudo_len_by_cut: dict[float, int] = {}

    for cut in cut_fracs:
        tw_row = truncate_known_prefix(h, cut)
        if tw_row is None:
            diag["truncate_fail"] += 1
            pseudo_len_by_cut[cut] = 0
            rmse_by_cut[cut] = {c: float("nan") for c in CANDIDATES}
            preds_by_cut[cut] = {}
            continue

        pseudo_len_by_cut[cut] = tw_row.pseudo_eval_len
        cand_rmse: dict[str, float] = {}
        cand_preds: dict[str, np.ndarray] = {}
        for cand in CANDIDATES:
            pred = _predict_candidate(cand, tw_row.h, tw)
            if pred is None or pred.shape != tw_row.y_true.shape or not np.all(np.isfinite(pred)):
                diag["candidate_fail"] += 1
                cand_rmse[cand] = float("nan")
                continue
            cand_rmse[cand] = _rmse(tw_row.y_true, pred)
            cand_preds[cand] = pred
        rmse_by_cut[cut] = cand_rmse
        preds_by_cut[cut] = cand_preds
        del tw_row  # drop the truncated DataFrame as soon as this cut is scored

    for cut in cut_fracs:
        row[f"pseudo_eval_len_cut{cut:g}"] = float(pseudo_len_by_cut[cut])

    for cand in CANDIDATES:
        for cut in cut_fracs:
            row[f"bt_rmse_{cand}_cut{cut:g}"] = rmse_by_cut[cut].get(cand, float("nan"))
        vals = np.array([rmse_by_cut[cut].get(cand, float("nan")) for cut in cut_fracs])
        finite = vals[np.isfinite(vals)]
        row[f"bt_rmse_{cand}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        row[f"bt_rmse_{cand}_worst"] = float(np.max(finite)) if finite.size else float("nan")

    rank_history: dict[str, list[float]] = {c: [] for c in CANDIDATES}
    for cut in cut_fracs:
        ranks = _rank_from_values(rmse_by_cut[cut])
        for cand in CANDIDATES:
            row[f"bt_rank_{cand}_cut{cut:g}"] = ranks[cand]
            rank_history[cand].append(ranks[cand])
    for cand in CANDIDATES:
        row[f"bt_rank_{cand}_mean"] = float(np.mean(rank_history[cand]))

    margins: list[float] = []
    for cut in cut_fracs:
        m = _best_second_margin(rmse_by_cut[cut])
        row[f"bt_margin_best_second_cut{cut:g}"] = m
        margins.append(m)
    finite_margins = np.array([m for m in margins if np.isfinite(m)])
    row["bt_margin_best_second_mean"] = (
        float(np.mean(finite_margins)) if finite_margins.size else float("nan")
    )

    agree_means: list[float] = []
    for cut in cut_fracs:
        pair_vals = _pairwise_agreement(preds_by_cut[cut])
        for a, b in CANDIDATE_PAIRS:
            row[f"bt_agree_{a}_{b}_cut{cut:g}"] = pair_vals[(a, b)]
        finite_pairs = [v for v in pair_vals.values() if np.isfinite(v)]
        cut_mean = float(np.mean(finite_pairs)) if finite_pairs else float("nan")
        row[f"bt_agree_mean_cut{cut:g}"] = cut_mean
        agree_means.append(cut_mean)
    finite_agree = np.array([v for v in agree_means if np.isfinite(v)])
    row["bt_agree_mean"] = float(np.mean(finite_agree)) if finite_agree.size else float("nan")

    return row, diag


# --------------------------------------------------------------------------- #
# Phase 0: time gate
# --------------------------------------------------------------------------- #


def _time_probe(wells: list[str], cut_fracs: tuple[float, ...]) -> float:
    t0 = time.perf_counter()
    for well in wells:
        h = D.load_horizontal(well, "train")
        tw = D.load_typewell(well, "train")
        compute_well_row(h, tw, cut_fracs)
        del h, tw
    return time.perf_counter() - t0


def time_gate(probe_wells: list[str]) -> tuple[float, ...]:
    print(f"\n{'=' * 78}\n# Phase 0: time gate ({len(probe_wells)}-well probe)\n{'=' * 78}")
    print(f"candidates={CANDIDATES} cut_fracs={DEFAULT_CUT_FRACS}")

    dt = _time_probe(probe_wells, DEFAULT_CUT_FRACS)
    per_well = dt / len(probe_wells)
    projected_min = per_well * FULL_N_WELLS_PROJECTION / 60.0
    print(
        f"  3-cut probe: total={dt:.2f}s  per_well={per_well:.4f}s  "
        f"773-well projection={projected_min:.2f}min"
    )

    if per_well <= PER_WELL_TIME_BUDGET_S:
        print(
            f"  per_well={per_well:.4f}s <= budget {PER_WELL_TIME_BUDGET_S:.1f}s -- "
            f"keeping cut_fracs={DEFAULT_CUT_FRACS}"
        )
        return DEFAULT_CUT_FRACS

    print(
        f"  per_well={per_well:.4f}s > budget {PER_WELL_TIME_BUDGET_S:.1f}s -- "
        f"degrading cut_fracs {DEFAULT_CUT_FRACS} -> {DEGRADED_CUT_FRACS} (drop the middle point)"
    )
    dt2 = _time_probe(probe_wells, DEGRADED_CUT_FRACS)
    per_well2 = dt2 / len(probe_wells)
    projected_min2 = per_well2 * FULL_N_WELLS_PROJECTION / 60.0
    print(
        f"  2-cut probe (confirm): total={dt2:.2f}s  per_well={per_well2:.4f}s  "
        f"773-well projection={projected_min2:.2f}min"
    )
    return DEGRADED_CUT_FRACS


# --------------------------------------------------------------------------- #
# Phase 1: build the well x feature matrix
# --------------------------------------------------------------------------- #


def build_features(wells: list[str], cut_fracs: tuple[float, ...]) -> dict[str, object]:
    feature_names = build_feature_names(cut_fracs)
    n_wells, n_features = len(wells), len(feature_names)
    print(
        f"\n{'=' * 78}\n# Phase 1: feature build ({n_wells} wells, "
        f"{n_features} features, cut_fracs={cut_fracs})\n{'=' * 78}"
    )

    matrix = np.full((n_wells, n_features), np.nan, dtype=np.float32)
    col_pos = {name: i for i, name in enumerate(feature_names)}

    timings: list[float] = []
    n_well_failures = 0
    n_truncate_fail = 0
    n_candidate_fail = 0

    t_start = time.perf_counter()
    for i, well in enumerate(wells, start=1):
        t0 = time.perf_counter()
        try:
            h = D.load_horizontal(well, "train")
            tw = D.load_typewell(well, "train")
            row, diag = compute_well_row(h, tw, cut_fracs)
            for name, value in row.items():
                matrix[i - 1, col_pos[name]] = value
            n_truncate_fail += diag["truncate_fail"]
            n_candidate_fail += diag["candidate_fail"]
            del h, tw
        except Exception as exc:  # a single well must never abort the whole run
            n_well_failures += 1
            print(f"  [WARN] well {well} failed entirely: {exc!r}")
        timings.append(time.perf_counter() - t0)

        if i % PROGRESS_EVERY == 0 or i == n_wells:
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{i}/{n_wells}] elapsed={elapsed:.1f}s "
                f"well_failures={n_well_failures} truncate_fail={n_truncate_fail} "
                f"candidate_fail={n_candidate_fail}",
                flush=True,
            )

    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "well_names": np.array(wells),
        "timings": np.array(timings),
        "n_well_failures": n_well_failures,
        "n_truncate_fail": n_truncate_fail,
        "n_candidate_fail": n_candidate_fail,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _print_nan_report(matrix: np.ndarray, feature_names: list[str]) -> None:
    total = matrix.size
    n_nan = int(np.isnan(matrix).sum())
    print(f"\n## NaN rate: {n_nan}/{total} ({100 * n_nan / total:.2f}%) across the full matrix")
    per_col_nan = np.isnan(matrix).mean(axis=0)
    worst = np.argsort(per_col_nan)[::-1][:10]
    print("  10 highest-NaN-rate columns:")
    for j in worst:
        if per_col_nan[j] > 0:
            print(f"    {feature_names[j]:38s} {100 * per_col_nan[j]:6.2f}%")


def _print_rmse_distribution(
    matrix: np.ndarray, feature_names: list[str], cut_fracs: tuple[float, ...]
) -> None:
    print("\n## backtest RMSE distribution per candidate (pooled across cuts, this run)\n")
    print(f"{'candidate':14s} {'n':>6s} {'median':>9s} {'p10':>9s} {'p90':>9s} {'mean':>9s}")
    print("-" * 60)
    col_pos = {name: i for i, name in enumerate(feature_names)}
    for cand in CANDIDATES:
        cols = [col_pos[f"bt_rmse_{cand}_cut{cut:g}"] for cut in cut_fracs]
        vals = matrix[:, cols].ravel()
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            print(f"{cand:14s} {'0':>6s}  (all NaN)")
            continue
        print(
            f"{cand:14s} {finite.size:6d} {np.median(finite):9.4f} "
            f"{np.percentile(finite, 10):9.4f} {np.percentile(finite, 90):9.4f} "
            f"{finite.mean():9.4f}"
        )


def _print_timing(timings: np.ndarray) -> None:
    print("\n## per-well wall-clock time (this run's build phase)\n")
    mean_s = float(timings.mean())
    projected_min = mean_s * FULL_N_WELLS_PROJECTION / 60.0
    print(
        f"mean={mean_s:.4f}s  p50={np.median(timings):.4f}s  "
        f"p95={np.percentile(timings, 95):.4f}s  max={timings.max():.4f}s  "
        f"total={timings.sum():.1f}s  n={timings.size}"
    )
    print(
        f"773-well projection (from this run's measured per-well average) = "
        f"{projected_min:.2f}min"
    )


def save_features(out_path: Path, result: dict[str, object], cut_fracs: tuple[float, ...]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        features=result["matrix"],
        feature_names=np.array(result["feature_names"]),
        well_names=result["well_names"],
        cut_fracs=np.array(cut_fracs, dtype=np.float64),
        candidates=np.array(CANDIDATES),
        spatial_omitted=np.bool_(True),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n# saved {out_path} ({size_mb:.2f} MB)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _load_target_wells(limit: int | None) -> list[str]:
    """Same well set + order as ``outputs/stack_v2_oof.npz``'s ``used_wells``."""
    with np.load(OOF_PATH, allow_pickle=True) as npz:
        used_wells = [str(w) for w in npz["used_wells"]]
    return used_wells[:limit] if limit is not None else used_wells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N wells (smoke run)."
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH), help="Output npz path.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    target_wells = _load_target_wells(args.limit)
    print(f"# target wells: {len(target_wells)} (limit={args.limit})")
    print(
        "# spatial candidate: OMITTED -- rogii.spatial's leave-self-out SurfaceBank build is a "
        "~35min fixed cost independent of --limit (see scripts/build_spatial_cache.py docstring), "
        "disproportionate to this feature-building pass. See module docstring."
    )

    probe_wells = target_wells[: min(TIME_GATE_N_WELLS, len(target_wells))]
    cut_fracs = time_gate(probe_wells)

    result = build_features(target_wells, cut_fracs)
    matrix = result["matrix"]
    feature_names = result["feature_names"]

    print(f"\n# well-level total failures: {result['n_well_failures']}/{len(target_wells)}")
    print(f"# truncation failures (well,cut pairs): {result['n_truncate_fail']}")
    print(f"# candidate-prediction failures (well,cut,candidate triples): "
          f"{result['n_candidate_fail']}")

    _print_nan_report(matrix, feature_names)
    _print_rmse_distribution(matrix, feature_names, cut_fracs)
    _print_timing(result["timings"])

    save_features(Path(args.out), result, cut_fracs)

    print(f"\ntotal runtime: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
