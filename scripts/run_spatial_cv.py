"""Evaluate the spatial-prior predictor (``rogii.spatial``) on all train wells.

Builds the formation-depth surface bank once from every train well, then
scores wells leave-self-out (the target well's own bank samples are excluded
from the KNN at query time -- see ``rogii/spatial.py``) in two phases: an
80-well health-check subset first, then the full train set. Reports, for
(a) ``carry_last``, (b) raw ``predict_spatial`` output (best formation),
(c) the inverse-squared-prefix-residual composite across formations,
(d) blends ``anchor + w * (spatial - anchor)`` for ``w`` in ``{0.5, 0.7,
1.0}``, and (e) prefix-gated variants (wells whose known-zone ``prefix_rmse``
exceeds a threshold theta fall back to the flat anchor) for theta in
``{1, 2, 5}`` ft:

* pooled RMSE (the competition metric: squared error summed across every
  eval-zone row of every well, then a single sqrt at the end),
* the per-well RMSE distribution (median / p90 / max),
* bank-build time and ``predict_spatial()`` wall-clock time per well,
* the distribution of nearest-neighbor distances from each well's eval-zone
  points to the (self-excluded) bank -- a sanity check that offset wells
  actually exist nearby, and
* the ``prefix_rmse`` distribution plus how many wells each gate trips on.

    uv run python scripts/run_spatial_cv.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import baseline  # noqa: E402
from rogii import data as D  # noqa: E402
from rogii import spatial as SP  # noqa: E402

BLEND_WEIGHTS: tuple[float, ...] = (0.5, 0.7, 1.0)
GATE_THRESHOLDS_FT: tuple[float, ...] = (1.0, 2.0, 5.0)

STRIDE = 10
K_NEIGHBORS = 10
# "idw" per the module default. "plane" (local plane fit) was measured on a
# 20-well subset and had a *worse* heavy tail (pooled 26.8 vs 23.0), so IDW
# is the headline configuration.
METHOD = "idw"

HEALTH_CHECK_N_WELLS = 80

# Subsample eval-zone rows for the nearest-neighbor-distance sanity check.
_NN_SUBSAMPLE = 20


def pooled_rmse(sq_err_sum: float, n: int) -> float:
    return float(np.sqrt(sq_err_sum / n)) if n else float("nan")


def well_rmse(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def _predictor_names() -> list[str]:
    return [
        "carry_last",
        "spatial_raw",
        "spatial_comp",
        *[f"blend_w{w:.1f}" for w in BLEND_WEIGHTS],
        *[f"gate_t{theta:g}" for theta in GATE_THRESHOLDS_FT],
        *[f"gateblend_t{theta:g}" for theta in GATE_THRESHOLDS_FT],
    ]


def _predictions_for_well(
    anchor_pred: np.ndarray, result: SP.SpatialResult
) -> dict[str, np.ndarray]:
    preds = {
        "carry_last": anchor_pred,
        "spatial_raw": result.tvt,
        "spatial_comp": result.tvt_composite,
    }
    delta = result.tvt - anchor_pred
    for w in BLEND_WEIGHTS:
        preds[f"blend_w{w:.1f}"] = anchor_pred + w * delta
    blend_07 = anchor_pred + 0.7 * delta
    for theta in GATE_THRESHOLDS_FT:
        gated = result.prefix_rmse <= theta
        preds[f"gate_t{theta:g}"] = result.tvt if gated else anchor_pred
        preds[f"gateblend_t{theta:g}"] = blend_07 if gated else anchor_pred
    return preds


def _nn_distance_for_well(h, bank: SP.SurfaceBank, well: str) -> float:
    """Median distance from (subsampled) eval-zone points to the nearest non-self sample."""
    formation = bank.formations[0]
    tree = bank.trees[formation]
    codes = bank.well_codes[formation]
    exclude_code = bank.well_to_code.get(well, -1)

    ez = D.eval_mask(h)
    x = h["X"].to_numpy(dtype=float)[ez][::_NN_SUBSAMPLE]
    y = h["Y"].to_numpy(dtype=float)[ez][::_NN_SUBSAMPLE]
    if x.size == 0:
        return float("nan")

    coords = np.column_stack([x, y])
    # Fetch enough neighbors that the excluded well's own samples (which are
    # the nearest ones along its own trajectory) cannot fill every slot.
    n_self = int(np.sum(codes == exclude_code))
    k = min(1 + n_self, codes.size)
    dist, idx = tree.query(coords, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    valid = codes[idx] != exclude_code
    first_valid = np.where(valid.any(axis=1), valid.argmax(axis=1), -1)
    rows = np.arange(coords.shape[0])
    nn = np.where(first_valid >= 0, dist[rows, np.maximum(first_valid, 0)], np.nan)
    if not np.any(np.isfinite(nn)):
        return float("nan")
    return float(np.nanmedian(nn))


def _print_pooled_table(names: list[str], sse: dict[str, float], n_rows: dict[str, int]) -> None:
    print("## Pooled RMSE (all eval-zone rows, all wells) -- the competition metric\n")
    print(f"{'predictor':16s} {'pooled RMSE (ft)':>18s} {'rows':>12s}")
    print("-" * 48)
    for name in names:
        print(f"{name:16s} {pooled_rmse(sse[name], n_rows[name]):18.4f} {n_rows[name]:12,d}")


def _print_per_well_table(names: list[str], per_well: dict[str, list[float]]) -> None:
    print("\n## Per-well RMSE distribution (ft)\n")
    print(f"{'predictor':16s} {'median':>10s} {'p90':>10s} {'max':>10s} {'n_wells':>10s}")
    print("-" * 60)
    for name in names:
        arr = np.array(per_well[name])
        print(
            f"{name:16s} {np.median(arr):10.3f} {np.percentile(arr, 90):10.3f} "
            f"{arr.max():10.3f} {arr.size:10,d}"
        )


def _print_timing(bank_secs: float, timings: np.ndarray) -> None:
    print("\n## Timing\n")
    print(f"bank build: {bank_secs:.1f} s (once)")
    print(
        f"predict_spatial() per well (s): mean={timings.mean():.4f}  "
        f"p50={np.median(timings):.4f}  p95={np.percentile(timings, 95):.4f}  "
        f"max={timings.max():.4f}  n={timings.size}"
    )


def _print_nn_distances(nn_dists: np.ndarray) -> None:
    finite = nn_dists[np.isfinite(nn_dists)]
    print("\n## Nearest non-self bank sample distance, per-well median (ft in X/Y units)\n")
    if finite.size == 0:
        print("(no finite distances)")
        return
    print(
        f"median={np.median(finite):.1f}  p90={np.percentile(finite, 90):.1f}  "
        f"max={finite.max():.1f}  n={finite.size}"
    )


def _print_prefix_diag(prefix_rmses: np.ndarray, n_formations: np.ndarray) -> None:
    finite = prefix_rmses[np.isfinite(prefix_rmses)]
    print("\n## prefix_rmse (known-zone fit quality of the winning formation, ft)\n")
    if finite.size:
        print(
            f"median={np.median(finite):.3f}  p90={np.percentile(finite, 90):.3f}  "
            f"max={finite.max():.3f}  (finite for {finite.size} wells; "
            f"inf/fallback for {prefix_rmses.size - finite.size})"
        )
    for theta in GATE_THRESHOLDS_FT:
        n_gated = int(np.sum(~(prefix_rmses <= theta)))
        print(f"gate theta={theta:g} ft -> {n_gated} wells fall back to anchor")
    print(f"n_formations_used: median={np.median(n_formations):.0f}  min={n_formations.min()}")


def _print_extremes(deltas: list[tuple[float, str, float, float]]) -> None:
    """Wells where spatial helps / hurts the most vs carry_last (per-well RMSE)."""
    ordered = sorted(deltas)
    print("\n## spatial_raw vs carry_last, per-well RMSE delta (negative = spatial better)\n")
    print("best 5 improvements:")
    for d, well, carry, spat in ordered[:5]:
        print(f"  {well}  carry={carry:8.3f}  spatial={spat:8.3f}  delta={d:+9.3f}")
    print("worst 5 regressions:")
    for d, well, carry, spat in ordered[-5:]:
        print(f"  {well}  carry={carry:8.3f}  spatial={spat:8.3f}  delta={d:+9.3f}")


def run_cv(wells: list[str], bank: SP.SurfaceBank, bank_secs: float, label: str) -> None:
    print(f"\n{'=' * 70}\n# {label}: {len(wells)} wells\n{'=' * 70}")

    names = _predictor_names()
    sse = {name: 0.0 for name in names}
    n_rows = {name: 0 for name in names}
    per_well: dict[str, list[float]] = {name: [] for name in names}
    timings: list[float] = []
    nn_dists: list[float] = []
    prefix_rmses: list[float] = []
    n_formations: list[int] = []
    deltas: list[tuple[float, str, float, float]] = []
    skipped = 0

    for i, well in enumerate(wells):
        h = D.load_horizontal(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            skipped += 1
            continue

        y_true = h["TVT"].to_numpy()[ez]
        anchor_pred = baseline.predict_carry_last(h)

        t1 = time.perf_counter()
        result = SP.predict_spatial(h, bank, exclude_well=well, k=K_NEIGHBORS, method=METHOD)
        timings.append(time.perf_counter() - t1)

        nn_dists.append(_nn_distance_for_well(h, bank, well))
        prefix_rmses.append(result.prefix_rmse)
        n_formations.append(result.n_formations_used)

        for name, pred in _predictions_for_well(anchor_pred, result).items():
            err = y_true - pred
            sse[name] += float(np.sum(err**2))
            n_rows[name] += err.size
            per_well[name].append(well_rmse(y_true, pred))

        carry = per_well["carry_last"][-1]
        spat = per_well["spatial_raw"][-1]
        deltas.append((spat - carry, well, carry, spat))

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(wells)} wells", flush=True)

    print(f"# skipped wells (no eval zone / no anchor): {skipped}\n")
    _print_pooled_table(names, sse, n_rows)
    _print_per_well_table(names, per_well)
    _print_timing(bank_secs, np.array(timings))
    _print_nn_distances(np.array(nn_dists))
    _print_prefix_diag(np.array(prefix_rmses), np.array(n_formations))
    _print_extremes(deltas)
    _print_helped_hurt(deltas)


def _print_helped_hurt(deltas: list[tuple[float, str, float, float]]) -> None:
    """Win/loss counts and whether prefix quality separates them (playbook 02)."""
    helped = [d for d, *_ in deltas if d < 0]
    hurt = [d for d, *_ in deltas if d > 0]
    print("\n## helped / hurt (spatial_raw vs carry_last, per-well RMSE)\n")
    print(
        f"helped: {len(helped)} wells (median improvement {np.median(np.abs(helped)):.3f} ft)"
        if helped
        else "helped: 0 wells"
    )
    print(
        f"hurt:   {len(hurt)} wells (median regression {np.median(hurt):.3f} ft)"
        if hurt
        else "hurt:   0 wells"
    )


def main() -> None:
    wells = D.list_wells("train")
    print(f"# train wells: {len(wells)}")
    print(f"# config: stride={STRIDE} k={K_NEIGHBORS} method={METHOD} formations={SP.FORMATIONS}")

    t0 = time.perf_counter()
    bank = SP.build_surface_bank(wells, split="train", stride=STRIDE)
    bank_secs = time.perf_counter() - t0
    sizes = {f: bank.depths[f].size for f in bank.formations}
    print(f"# bank built in {bank_secs:.1f} s; samples per formation: {sizes}", flush=True)

    subset = wells[:HEALTH_CHECK_N_WELLS]
    run_cv(subset, bank, bank_secs, f"Phase 1: health-check subset ({len(subset)} wells)")

    run_cv(wells, bank, bank_secs, "Phase 2: full train set")


if __name__ == "__main__":
    main()
