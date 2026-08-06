"""Evaluate PF-BMA (likelihood-weighted particle-filter Bayesian model averaging)
as a new candidate-path source (design.md section 11 R2, notebook-dissection
migration candidate #1: ``rogii.registration.particle.track_pf_bma``).

**Hypothesis under test**: independent-seed PF runs combined via
``softmax(cumulative_log_lik / scale)`` (seed-BMA) produce a *stronger
candidate path* than the current adopted PF (:func:`track_pf_multi`, plain
seed-averaging), where "stronger" is measured two ways:

1. its own **pooled RMSE** (raw and anchor-blended) against the adopted PF
   blend w0.7 (11.0621, ``analysis/experiment_ledger.md``), and
2. whether adding it as a 5th candidate to the existing **hard-well
   per-well oracle** (``scripts/analyze_hard_wells.py``'s mechanism: rank
   wells by stack-v2 all-in OOF SSE, take the top decile, and for those wells
   only, per-well-pick the best of several candidate predictors) improves the
   oracle ceiling beyond the measured 4-way number, **7.7732**
   (``analysis/experiment_ledger.md``, 2026-07-08).

This script does **not** retrain the stack model and does **not** touch
``outputs/stack_v2_oof.npz`` (read-only dependency, built by
``scripts/run_stack_v2_cv.py --save-oof``); the oracle-substitution logic
below re-derives the same hard-well set and per-well-SSE-argmin mechanism as
``scripts/analyze_hard_wells.py`` (reused, not imported -- that script is a
fixed diagnostic artifact, off-limits to modify/import-couple with) and
simply extends its candidate set with ``pf_bma``.

Phases:

  0. **Time gate** (20-well probe): measure ``track_pf_bma`` wall-clock cost
     at two seed counts, extrapolate the marginal per-seed cost, and pick the
     largest seed count (from a small candidate grid) whose projected
     773-well runtime fits a 60-minute budget (with a safety margin) --
     confirmed with one more direct 20-well measurement at the chosen count.
  1. **Full 773-well run**: compute ``track_pf_bma`` for every train well at
     the chosen config, accumulate pooled RMSE for the raw per-scale paths
     and their w in {0.5, 0.7} anchor-blends, and persist every well's
     per-scale predicted TVT (+ uncertainty) to
     ``outputs/path_bank_pfbma.npz`` (row-aligned to
     ``outputs/stack_v2_oof.npz``'s ``well_idx``/``used_wells`` -- verified,
     not assumed).
  2. **5-way oracle**: re-derive the stack-v2 hard-well top-decile set,
     extend {stack, carry_last, PF blend w0.7, spatial} with a 5th
     ``pf_bma`` candidate (the best-scale path, anchor-blended w0.7 to match
     how "PF blend" is already represented in the OOF), and report the
     resulting per-well-oracle pooled RMSE against the 7.7732 baseline.

Usage::

    uv run python scripts/run_pf_bma_cv.py > outputs/pf_bma_cv.log
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import baseline  # noqa: E402
from rogii import data as D  # noqa: E402
from rogii.registration.particle import track_pf_bma  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = REPO_ROOT / "outputs" / "stack_v2_oof.npz"
SPATIAL_CACHE_DIR = REPO_ROOT / "data" / "processed" / "spatial_cache"
BANK_PATH = REPO_ROOT / "outputs" / "path_bank_pfbma.npz"

# ---- ledger sanity anchors (analysis/experiment_ledger.md) -----------------
CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
STACK_V2_ALL_IN_POOLED_RMSE = 9.2309
HARD_WELL_4WAY_ORACLE_RMSE = 7.7732  # analyze_hard_wells.py oracle_substitution(), 2026-07-08
LEDGER_TOLERANCE = 0.01

ALL_IN_CONFIG_LABEL = "all-in (levers 1+2+3)"
TOP_FRAC = 0.10

# ---- PF-BMA config -----------------------------------------------------
N_PARTICLES = 512
GR_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)  # reused from track_pf's calibrated grid
BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)  # reference notebook's softmax-temp grid
BLEND_WEIGHTS: tuple[float, ...] = (0.5, 0.7)

# ---- time-gate config ----------------------------------------------------
TIME_GATE_N_WELLS = 20
TIME_BUDGET_S = 60 * 60
TIME_SAFETY_MARGIN = 0.85  # only commit to a candidate below 85% of the raw budget
PROBE_SEED_COUNTS = (4, 8)
CANDIDATE_N_SEEDS = (8, 10, 12, 14, 16, 20, 24, 32)

PROGRESS_EVERY = 100


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pooled_rmse_from_sse(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


def well_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# --------------------------------------------------------------------------- #
# Phase 0: time gate
# --------------------------------------------------------------------------- #


def _time_probe(wells: list[str], n_seeds: int) -> float:
    """Total wall-clock time (s) for ``track_pf_bma`` over ``wells`` at ``n_seeds``."""
    t0 = time.perf_counter()
    for well in wells:
        h = D.load_horizontal(well, "train")
        tw = D.load_typewell(well, "train")
        track_pf_bma(
            h, tw, n_particles=N_PARTICLES, n_seeds=n_seeds, bma_scales=BMA_SCALES,
            gr_scales=GR_SCALES,
        )
    return time.perf_counter() - t0


def time_gate(all_wells: list[str]) -> int:
    probe_wells = all_wells[:TIME_GATE_N_WELLS]
    print(f"\n{'=' * 78}\n# Phase 0: time gate ({TIME_GATE_N_WELLS}-well probe)\n{'=' * 78}")
    print(f"config: n_particles={N_PARTICLES} gr_scales={GR_SCALES} bma_scales={BMA_SCALES}")

    probe_times: dict[int, float] = {}
    for n_seeds in PROBE_SEED_COUNTS:
        dt = _time_probe(probe_wells, n_seeds)
        probe_times[n_seeds] = dt
        print(
            f"  probe n_seeds={n_seeds:2d}: total={dt:7.2f}s  "
            f"per_well={dt / TIME_GATE_N_WELLS:.4f}s"
        )

    lo, hi = PROBE_SEED_COUNTS
    per_seed_cost = (probe_times[hi] - probe_times[lo]) / (hi - lo) / TIME_GATE_N_WELLS
    fixed_cost = probe_times[lo] / TIME_GATE_N_WELLS - lo * per_seed_cost
    print(
        f"\n  marginal per-seed cost = {per_seed_cost:.4f}s/well/seed  "
        f"(fixed per-well overhead ~{fixed_cost:.4f}s: data load + one-time BMA prep)"
    )

    print(f"\n  {'n_seeds':>8s}  {'projected 773-well time':>26s}  {'fits 60min?':>12s}")
    budget_s = TIME_BUDGET_S * TIME_SAFETY_MARGIN
    chosen = PROBE_SEED_COUNTS[0]
    for n_seeds in CANDIDATE_N_SEEDS:
        projected = len(all_wells) * (fixed_cost + n_seeds * per_seed_cost)
        fits = projected <= budget_s
        verdict = "OK" if fits else "over"
        print(f"  {n_seeds:8d}  {projected:22.1f}s ({projected / 60:5.1f}min)  {verdict:>12s}")
        if fits:
            chosen = n_seeds

    print(
        f"\n  => choosing n_seeds={chosen} (largest candidate whose projected 773-well "
        f"time stays within {TIME_SAFETY_MARGIN:.0%} of the {TIME_BUDGET_S / 60:.0f}-minute budget)"
    )

    confirm_dt = _time_probe(probe_wells, chosen)
    confirm_per_well = confirm_dt / TIME_GATE_N_WELLS
    confirm_projected = confirm_per_well * len(all_wells)
    print(
        f"\n  confirm n_seeds={chosen}: measured per_well={confirm_per_well:.4f}s  "
        f"projected 773-well={confirm_projected:.1f}s ({confirm_projected / 60:.2f}min)"
    )
    return chosen


# --------------------------------------------------------------------------- #
# Phase 1: full 773-well run
# --------------------------------------------------------------------------- #


def _predictor_names() -> list[str]:
    names = ["carry_last (floor, from OOF)", "PF blend w0.7 (前ベスト, from OOF)"]
    for scale in BMA_SCALES:
        names.append(f"pf_bma_scale{scale:g}_raw")
        for w in BLEND_WEIGHTS:
            names.append(f"pf_bma_scale{scale:g}_blend_w{w:.1f}")
    return names


def run_full(wells: list[str], n_seeds: int) -> dict[str, object]:
    print(
        f"\n{'=' * 78}\n# Phase 1: full train set ({len(wells)} wells, "
        f"n_seeds={n_seeds})\n{'=' * 78}"
    )

    names = _predictor_names()
    sse = {name: 0.0 for name in names}
    n_rows = {name: 0 for name in names}
    per_well_rmse: dict[str, list[float]] = {name: [] for name in names}
    timings: list[float] = []
    skipped = 0

    used_wells: list[str] = []
    well_idx_parts: list[np.ndarray] = []
    y_true_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    scale_tvt_parts: dict[float, list[np.ndarray]] = {s: [] for s in BMA_SCALES}
    scale_std_parts: dict[float, list[np.ndarray]] = {s: [] for s in BMA_SCALES}

    t_start = time.perf_counter()
    for i, well in enumerate(wells, start=1):
        h = D.load_horizontal(well, "train")
        tw = D.load_typewell(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            skipped += 1
            continue

        y_true = h["TVT"].to_numpy(dtype=float)[ez]
        anchor_pred = baseline.predict_carry_last(h)

        t0 = time.perf_counter()
        result = track_pf_bma(
            h, tw, n_particles=N_PARTICLES, n_seeds=n_seeds, bma_scales=BMA_SCALES,
            gr_scales=GR_SCALES,
        )
        timings.append(time.perf_counter() - t0)

        well_i = len(used_wells)
        used_wells.append(well)
        well_idx_parts.append(np.full(y_true.shape[0], well_i, dtype=np.int32))
        y_true_parts.append(y_true.astype(np.float32))
        anchor_parts.append(anchor_pred.astype(np.float32))

        for scale in BMA_SCALES:
            tvt = result.tvt_by_scale[scale]
            std = result.std_by_scale[scale]
            scale_tvt_parts[scale].append(tvt.astype(np.float32))
            scale_std_parts[scale].append(std.astype(np.float32))

            delta = tvt - anchor_pred
            preds = {f"pf_bma_scale{scale:g}_raw": tvt}
            for w in BLEND_WEIGHTS:
                preds[f"pf_bma_scale{scale:g}_blend_w{w:.1f}"] = anchor_pred + w * delta

            for name, pred in preds.items():
                err = y_true - pred
                sse[name] += float(np.sum(err**2))
                n_rows[name] += err.size
                per_well_rmse[name].append(well_rmse(y_true, pred))

        if i % PROGRESS_EVERY == 0 or i == len(wells):
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{i}/{len(wells)}] elapsed={elapsed:.1f}s ({elapsed / 60:.1f}min)",
                flush=True,
            )

    print(f"\n# skipped wells (no eval zone / no anchor): {skipped}")

    return {
        "names": names,
        "sse": sse,
        "n_rows": n_rows,
        "per_well_rmse": per_well_rmse,
        "timings": np.array(timings),
        "used_wells": np.array(used_wells),
        "well_idx": np.concatenate(well_idx_parts) if well_idx_parts else np.zeros(0, np.int32),
        "y_true": np.concatenate(y_true_parts) if y_true_parts else np.zeros(0, np.float32),
        "anchor": np.concatenate(anchor_parts) if anchor_parts else np.zeros(0, np.float32),
        "scale_tvt": {
            s: np.concatenate(v) if v else np.zeros(0, np.float32)
            for s, v in scale_tvt_parts.items()
        },
        "scale_std": {
            s: np.concatenate(v) if v else np.zeros(0, np.float32)
            for s, v in scale_std_parts.items()
        },
    }


def _print_pooled_table(run: dict[str, object]) -> None:
    print("\n## Pooled RMSE (all eval-zone rows, all wells) -- the competition metric\n")
    print(f"{'predictor':38s} {'pooled RMSE (ft)':>18s} {'median':>9s} {'p90':>9s} {'max':>9s}")
    print("-" * 88)
    print(f"{'carry_last (floor, ledger)':38s} {CARRY_LAST_POOLED_RMSE:18.4f}")
    print(f"{'PF blend w0.7 (前ベスト, ledger)':38s} {PF_BLEND_POOLED_RMSE:18.4f}")
    for name in run["names"][2:]:
        sse = run["sse"][name]
        n = run["n_rows"][name]
        arr = np.array(run["per_well_rmse"][name])
        print(
            f"{name:38s} {pooled_rmse_from_sse(sse, n):18.4f} "
            f"{np.median(arr):9.3f} {np.percentile(arr, 90):9.3f} {arr.max():9.3f}"
        )


def _print_timing(timings: np.ndarray) -> None:
    print("\n## track_pf_bma() wall-clock time per well (s)\n")
    print(
        f"mean={timings.mean():.4f}  p50={np.median(timings):.4f}  "
        f"p95={np.percentile(timings, 95):.4f}  max={timings.max():.4f}  "
        f"total={timings.sum():.1f}s ({timings.sum() / 60:.2f}min)  n={timings.size}"
    )


def save_bank(run: dict[str, object], n_seeds: int) -> None:
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "well_idx": run["well_idx"],
        "used_wells": run["used_wells"],
        "y_true": run["y_true"],
        "anchor": run["anchor"],
        "bma_scales": np.array(BMA_SCALES, dtype=np.float64),
        "gr_scales": np.array(GR_SCALES, dtype=np.float64),
        "n_particles": np.int64(N_PARTICLES),
        "n_seeds": np.int64(n_seeds),
    }
    for scale in BMA_SCALES:
        payload[f"pf_bma_scale{scale:g}_tvt"] = run["scale_tvt"][scale]
        payload[f"pf_bma_scale{scale:g}_std"] = run["scale_std"][scale]

    np.savez(BANK_PATH, **payload)
    print(f"\n# bank saved: {BANK_PATH} ({BANK_PATH.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- #
# Phase 2: 5-way oracle (reuses scripts/analyze_hard_wells.py's mechanism)
# --------------------------------------------------------------------------- #


def load_stack_oof() -> dict[str, np.ndarray]:
    """Same loading + sanity asserts as ``analyze_hard_wells.py::load_stack_oof``."""
    with np.load(OOF_PATH, allow_pickle=True) as npz:
        y_true = npz["y_true"].astype(np.float64)
        anchor = npz["anchor"].astype(np.float64)
        pf_blend = npz["pf_blend"].astype(np.float64)
        well_idx = npz["well_idx"]
        used_wells = npz["used_wells"]
        config_labels = list(npz["config_labels"])
        all_in_i = config_labels.index(ALL_IN_CONFIG_LABEL)
        pred = npz[f"oof_{all_in_i}"].astype(np.float64)

    n_wells = len(used_wells)
    boundaries = np.searchsorted(well_idx, np.arange(n_wells + 1))
    assert boundaries[-1] == well_idx.shape[0], "well_idx blocks are not contiguous/sorted"

    baseline_rmse = pooled_rmse(y_true, pred)
    assert abs(baseline_rmse - STACK_V2_ALL_IN_POOLED_RMSE) < LEDGER_TOLERANCE, (
        f"stack v2 all-in pooled RMSE {baseline_rmse:.4f} drifted from ledger "
        f"{STACK_V2_ALL_IN_POOLED_RMSE} -- npz may be stale"
    )
    pf_pooled = pooled_rmse(y_true, pf_blend)
    assert abs(pf_pooled - PF_BLEND_POOLED_RMSE) < LEDGER_TOLERANCE, (
        f"pf_blend column pooled RMSE {pf_pooled:.4f} drifted from ledger "
        f"{PF_BLEND_POOLED_RMSE} -- npz may be stale"
    )

    return {
        "y_true": y_true,
        "anchor": anchor,
        "pf_blend": pf_blend,
        "pred": pred,
        "well_idx": well_idx,
        "used_wells": used_wells,
        "boundaries": boundaries,
        "n_wells": n_wells,
    }


def per_well_sse(oof: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-well SSE only (the subset of analyze_hard_wells.py's decomposition
    needed here to reproduce its top-decile hard-well ranking)."""
    y_true, pred = oof["y_true"], oof["pred"]
    boundaries, used_wells, n_wells = oof["boundaries"], oof["used_wells"], oof["n_wells"]
    rows = []
    for i in range(n_wells):
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        err = pred[s:e_] - y_true[s:e_]
        rows.append({"well": str(used_wells[i]), "sse_total": float(np.sum(err**2))})
    return pd.DataFrame(rows).set_index("well")


def identify_hard_wells(per_well: pd.DataFrame, top_frac: float) -> pd.Index:
    n_wells = len(per_well)
    n_top = max(1, round(n_wells * top_frac))
    ranked = per_well.sort_values("sse_total", ascending=False)
    top_wells = ranked.index[:n_top]
    top_sse = per_well.loc[top_wells, "sse_total"].sum()
    total_sse = per_well["sse_total"].sum()
    print(
        f"\n## hard-well set: SSE top {top_frac:.0%} ({n_top}/{n_wells} wells), "
        f"SSE share = {100 * top_sse / total_sse:.2f}%"
    )
    return top_wells


def five_way_oracle(
    oof: dict[str, np.ndarray],
    top_wells: pd.Index,
    bank_well_idx: np.ndarray,
    bank_used_wells: np.ndarray,
    pf_bma_blend_by_scale: dict[float, np.ndarray],
) -> None:
    y_true, pred, boundaries, used_wells = (
        oof["y_true"],
        oof["pred"],
        oof["boundaries"],
        oof["used_wells"],
    )
    assert bank_well_idx.shape == oof["well_idx"].shape and np.array_equal(
        bank_well_idx, oof["well_idx"]
    ), "path_bank_pfbma.npz row layout does not match stack_v2_oof.npz -- cannot align"
    assert list(bank_used_wells) == list(used_wells), (
        "path_bank_pfbma.npz used_wells order does not match stack_v2_oof.npz"
    )

    total_rows = y_true.shape[0]
    baseline_sse = float(np.sum((y_true - pred) ** 2))
    baseline_rmse = np.sqrt(baseline_sse / total_rows)

    well_pos = {str(w): i for i, w in enumerate(used_wells)}
    top_set = {str(w) for w in top_wells}
    top_positions = [well_pos[w] for w in top_set]
    hard_rows_mask = np.zeros(total_rows, dtype=bool)
    for i in top_positions:
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        hard_rows_mask[s:e_] = True

    anchor_pred = oof["anchor"].copy()
    pf_blend_pred = oof["pf_blend"]

    spatial_pred = np.full(total_rows, np.nan, dtype=np.float64)
    for i in top_positions:
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        well = str(used_wells[i])
        with np.load(SPATIAL_CACHE_DIR / f"{well}.npz") as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
        assert spatial_tvt.size == (e_ - s), f"{well}: spatial cache length mismatch"
        spatial_pred[s:e_] = spatial_tvt

    print(
        f"\n{'=' * 92}\n# Phase 2: 5-way oracle (hard-well top {TOP_FRAC:.0%} "
        f"substitution)\n{'=' * 92}"
    )
    print(f"  baseline (stack v2 all-in, all wells)      pooled RMSE = {baseline_rmse:.4f}")
    print(
        f"  4-way oracle (ledger, 2026-07-08)          pooled RMSE = "
        f"{HARD_WELL_4WAY_ORACLE_RMSE:.4f}"
    )

    def _substitute_uniform(name: str, alt: np.ndarray) -> None:
        pred_sub = pred.copy()
        pred_sub[hard_rows_mask] = alt[hard_rows_mask]
        rmse_sub = pooled_rmse(y_true, pred_sub)
        print(
            f"  hard-well を一律 {name} に差し替え          pooled RMSE = {rmse_sub:.4f}  "
            f"(Δ{rmse_sub - baseline_rmse:+.4f})"
        )

    _substitute_uniform("carry_last", anchor_pred)
    _substitute_uniform("PF blend w0.7", pf_blend_pred)
    _substitute_uniform("spatial", spatial_pred)

    for scale, pf_bma_pred in pf_bma_blend_by_scale.items():
        _substitute_uniform(f"pf_bma scale{scale:g} (blend w0.7)", pf_bma_pred)

    # per-well oracle over the *best* single pf_bma scale (chosen by the caller)
    best_scale = next(iter(pf_bma_blend_by_scale))
    pf_bma_best = pf_bma_blend_by_scale[best_scale]

    pred_oracle4 = pred.copy()
    pred_oracle5 = pred.copy()
    picks4: dict[str, str] = {}
    picks5: dict[str, str] = {}
    candidates4 = {
        "stack": pred,
        "carry_last": anchor_pred,
        "PF blend": pf_blend_pred,
        "spatial": spatial_pred,
    }
    candidates5 = {**candidates4, "pf_bma": pf_bma_best}

    for i in top_positions:
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        well = str(used_wells[i])
        for candidates, picks, pred_out in (
            (candidates4, picks4, pred_oracle4),
            (candidates5, picks5, pred_oracle5),
        ):
            best_name, best_sse = "stack", np.inf
            for name, arr in candidates.items():
                seg = arr[s:e_]
                if not np.all(np.isfinite(seg)):
                    continue
                sse = float(np.sum((seg - y_true[s:e_]) ** 2))
                if sse < best_sse:
                    best_sse, best_name = sse, name
            picks[well] = best_name
            pred_out[s:e_] = candidates[best_name][s:e_]

    oracle4_rmse = pooled_rmse(y_true, pred_oracle4)
    oracle5_rmse = pooled_rmse(y_true, pred_oracle5)
    counts4 = pd.Series(list(picks4.values())).value_counts().to_dict()
    counts5 = pd.Series(list(picks5.values())).value_counts().to_dict()

    print(
        f"\n  re-derived 4-way oracle (this script, sanity check vs ledger) "
        f"pooled RMSE = {oracle4_rmse:.4f}  内訳={counts4}"
    )
    assert abs(oracle4_rmse - HARD_WELL_4WAY_ORACLE_RMSE) < LEDGER_TOLERANCE, (
        f"re-derived 4-way oracle {oracle4_rmse:.4f} does not match ledger "
        f"{HARD_WELL_4WAY_ORACLE_RMSE} -- hard-well set or candidate arrays diverged"
    )
    print(
        f"  5-way oracle (+ pf_bma scale{best_scale:g} blend w0.7, this script) "
        f"pooled RMSE = {oracle5_rmse:.4f}  (Δ vs 4-way {oracle5_rmse - oracle4_rmse:+.4f})  "
        f"内訳={counts5}"
    )
    n_pf_bma_picks = counts5.get("pf_bma", 0)
    print(
        f"  pf_bma was the per-well oracle pick for {n_pf_bma_picks}/{len(top_positions)} "
        "hard wells"
    )


def main() -> None:
    t_start = time.perf_counter()
    all_wells = D.list_wells("train")
    print(f"# train wells: {len(all_wells)}")

    n_seeds = time_gate(all_wells)

    run = run_full(all_wells, n_seeds)
    _print_pooled_table(run)
    _print_timing(run["timings"])
    save_bank(run, n_seeds)

    # pick the best-scale variant (by standalone pooled RMSE at blend w0.7) for the oracle
    scale_rmse = {
        scale: pooled_rmse_from_sse(
            run["sse"][f"pf_bma_scale{scale:g}_blend_w0.7"],
            run["n_rows"][f"pf_bma_scale{scale:g}_blend_w0.7"],
        )
        for scale in BMA_SCALES
    }
    ranked_scales = sorted(scale_rmse, key=lambda s: scale_rmse[s])
    print("\n## scale ranking by standalone pooled RMSE (blend w0.7)\n")
    for scale in ranked_scales:
        print(f"  scale={scale:g}: {scale_rmse[scale]:.4f}")
    best_scale = ranked_scales[0]
    print(f"\n  => best scale for the oracle stage: {best_scale:g}")

    oof = load_stack_oof()
    print(f"\n# stack v2 OOF loaded: {oof['n_wells']} wells, baseline pooled RMSE = "
          f"{pooled_rmse(oof['y_true'], oof['pred']):.4f}")
    per_well = per_well_sse(oof)
    top_wells = identify_hard_wells(per_well, TOP_FRAC)

    anchor_full = run["anchor"].astype(np.float64)
    pf_bma_blend_by_scale = {}
    for scale in [best_scale, *[s for s in BMA_SCALES if s != best_scale]]:
        tvt = run["scale_tvt"][scale].astype(np.float64)
        pf_bma_blend_by_scale[scale] = anchor_full + 0.7 * (tvt - anchor_full)

    five_way_oracle(
        oof, top_wells, run["well_idx"], run["used_wells"], pf_bma_blend_by_scale
    )

    print(f"\ntotal runtime: {time.perf_counter() - t_start:.1f}s "
          f"({(time.perf_counter() - t_start) / 60:.1f}min)")


if __name__ == "__main__":
    main()
