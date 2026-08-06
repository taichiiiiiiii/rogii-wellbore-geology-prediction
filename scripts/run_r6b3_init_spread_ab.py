"""R6-b(3) A/B: does a wider initial particle spread improve pf_bma_scale3?

Hypothesis under test (analysis/experiment_ledger.md, 2026-07-10 R6-b dissection,
improvement queue item (3)): the adopted PF/PF-BMA initializes each particle
cloud with ``init_pos_std = 0.3`` ft around the anchor in the ``TVT + Z``
frame; the public reference notebook ``sunnywu27/rogii-wellbore-tvt-physical-model``
uses ``2.0`` ft, with the inline comment "wider init spread helps wells with
abrupt TVT shift at PS" (i.e. wells whose true path jumps right at the
known/eval-zone boundary, which a tight cloud around the anchor cannot reach
before mode collapse).

``rogii.registration.particle`` now exposes ``init_pos_std`` (default 0.3,
bit-identical to before) on ``track_pf`` / ``track_pf_bma``.

This script measures OFF (init_pos_std=0.3, adopted) vs ON (init_pos_std=2.0,
reference value) pooled RMSE for the exact ``pf_bma_scale3`` config used by
the bank-builder's Phase B (n_particles=512, n_seeds=12,
bma_scales=(3,5,8,12), gr_scales=(8,12,20,30), fill_gr_gaps=False on BOTH
arms -- R6-b(1) was measured and rejected), on the same deterministic
150-well subset as ``scripts/run_r6b1_fill_gr_gaps_ab.py`` (evenly spaced
indices into ``sorted(list_wells("train"))``), and cross-checks the OFF side
against ``outputs/path_bank_pfbma.npz``'s ``pf_bma_scale3_tvt`` column
(implementation-regression guard).

Usage::

    nohup uv run python scripts/run_r6b3_init_spread_ab.py \
        > outputs/r6b3_init_spread_ab.log 2>&1 &
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import data as D  # noqa: E402
from rogii.registration.particle import track_pf_bma  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = REPO_ROOT / "outputs" / "path_bank_pfbma.npz"

N_WELLS_SAMPLE = 150
N_PARTICLES = 512
N_SEEDS = 12
BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
GR_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)
TARGET_SCALE = 3.0
INIT_POS_STD_OFF = 0.3  # adopted default (must reproduce the bank)
INIT_POS_STD_ON = 2.0  # sunnywu27 reference value
BANK_TOLERANCE = 1e-3  # float32 round-trip tolerance for the OFF==bank cross-check

PROGRESS_EVERY = 25


def pooled_rmse_from_sse(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


def well_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def deterministic_sample(all_wells: list[str], n: int) -> list[str]:
    """Evenly spaced indices into the sorted well list (deterministic, no RNG).

    Identical to scripts/run_r6b1_fill_gr_gaps_ab.py's sample so the A/B
    subsets are directly comparable across the R6-b experiments.
    """
    if n >= len(all_wells):
        return list(all_wells)
    idx = sorted({round(i * (len(all_wells) - 1) / (n - 1)) for i in range(n)})
    return [all_wells[i] for i in idx]


def load_bank_scale3(used_wells_wanted: list[str]) -> dict[str, np.ndarray]:
    """Per-well pf_bma_scale3_tvt row-blocks from outputs/path_bank_pfbma.npz,
    keyed by well name, restricted to `used_wells_wanted`."""
    with np.load(BANK_PATH, allow_pickle=True) as npz:
        bank_used_wells = [str(w) for w in npz["used_wells"]]
        well_idx = npz["well_idx"]
        tvt = npz["pf_bma_scale3_tvt"]

    wanted = set(used_wells_wanted)
    pos = {w: i for i, w in enumerate(bank_used_wells) if w in wanted}
    n_wells = len(bank_used_wells)
    boundaries = np.searchsorted(well_idx, np.arange(n_wells + 1))

    out: dict[str, np.ndarray] = {}
    for w, i in pos.items():
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        out[w] = tvt[s:e_].astype(np.float64)
    return out


def main() -> None:
    t_program_start = time.perf_counter()

    all_wells = D.list_wells("train")
    print(f"# train wells (total): {len(all_wells)}")
    sample = deterministic_sample(all_wells, N_WELLS_SAMPLE)
    print(f"# deterministic subset: {len(sample)} wells (evenly spaced over sorted list)")
    print(f"  first5={sample[:5]}  last3={sample[-3:]}")

    print(
        f"\n# config: n_particles={N_PARTICLES} n_seeds={N_SEEDS} bma_scales={BMA_SCALES} "
        f"gr_scales={GR_SCALES} target_scale={TARGET_SCALE} fill_gr_gaps=False (both arms)"
    )
    print(f"# OFF: init_pos_std={INIT_POS_STD_OFF}   ON: init_pos_std={INIT_POS_STD_ON}")

    bank_scale3 = load_bank_scale3(sample)
    print(f"# loaded bank pf_bma_scale3_tvt for {len(bank_scale3)}/{len(sample)} sample wells")

    rows_off: list[dict[str, object]] = []
    rows_on: list[dict[str, object]] = []
    sse_off, n_off = 0.0, 0
    sse_on, n_on = 0.0, 0
    bank_max_abs_diff = 0.0
    n_bank_checked = 0
    t_off_total = 0.0
    t_on_total = 0.0
    skipped = 0

    t_loop_start = time.perf_counter()
    for i, well in enumerate(sample, start=1):
        h = D.load_horizontal(well, "train")
        tw = D.load_typewell(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            skipped += 1
            continue

        y_true = h["TVT"].to_numpy(dtype=np.float64)[ez]
        gr = h["GR"].to_numpy(dtype=np.float64)
        gr_nan_frac = float(np.isnan(gr[ez]).mean())

        t0 = time.perf_counter()
        result_off = track_pf_bma(
            h,
            tw,
            n_particles=N_PARTICLES,
            n_seeds=N_SEEDS,
            bma_scales=BMA_SCALES,
            gr_scales=GR_SCALES,
            init_pos_std=INIT_POS_STD_OFF,
        )
        t_off = time.perf_counter() - t0
        t_off_total += t_off

        t0 = time.perf_counter()
        result_on = track_pf_bma(
            h,
            tw,
            n_particles=N_PARTICLES,
            n_seeds=N_SEEDS,
            bma_scales=BMA_SCALES,
            gr_scales=GR_SCALES,
            init_pos_std=INIT_POS_STD_ON,
        )
        t_on = time.perf_counter() - t0
        t_on_total += t_on

        tvt_off = result_off.tvt_by_scale[TARGET_SCALE].astype(np.float64)
        tvt_on = result_on.tvt_by_scale[TARGET_SCALE].astype(np.float64)

        if well in bank_scale3:
            bank_tvt = bank_scale3[well]
            if bank_tvt.shape == tvt_off.shape:
                diff = float(np.max(np.abs(bank_tvt - tvt_off)))
                bank_max_abs_diff = max(bank_max_abs_diff, diff)
                n_bank_checked += 1
            else:
                print(
                    f"  [WARN] {well}: bank shape {bank_tvt.shape} != OFF shape "
                    f"{tvt_off.shape}, skipping bank cross-check for this well"
                )

        rmse_off = well_rmse(y_true, tvt_off)
        rmse_on = well_rmse(y_true, tvt_on)
        sse_off += float(np.sum((y_true - tvt_off) ** 2))
        n_off += y_true.size
        sse_on += float(np.sum((y_true - tvt_on) ** 2))
        n_on += y_true.size

        rows_off.append({"well": well, "rmse": rmse_off, "gr_nan_frac": gr_nan_frac})
        rows_on.append({"well": well, "rmse": rmse_on, "gr_nan_frac": gr_nan_frac})

        del h, tw
        gc.collect()

        if i % PROGRESS_EVERY == 0 or i == len(sample):
            elapsed = time.perf_counter() - t_loop_start
            print(
                f"  [{i}/{len(sample)}] elapsed={elapsed:.1f}s peak_rss={peak_rss_mb():.0f}MB",
                flush=True,
            )

    print(f"\n# skipped wells (no eval zone / no anchor): {skipped}")
    print(f"# used wells: {len(rows_off)}")

    pooled_off = pooled_rmse_from_sse(sse_off, n_off)
    pooled_on = pooled_rmse_from_sse(sse_on, n_on)

    print(f"\n{'=' * 78}\n# pooled RMSE (pf_bma_scale3, {len(rows_off)} wells)\n{'=' * 78}")
    print(f"  OFF (init_pos_std={INIT_POS_STD_OFF}, existing behaviour): {pooled_off:.4f}")
    print(f"  ON  (init_pos_std={INIT_POS_STD_ON}):                      {pooled_on:.4f}")
    print(f"  delta (ON - OFF):                                {pooled_on - pooled_off:+.4f}")

    helped = hurt = tied = 0
    deltas: list[float] = []
    nan_fracs: list[float] = []
    for r_off, r_on in zip(rows_off, rows_on, strict=True):
        assert r_off["well"] == r_on["well"]
        d = float(r_on["rmse"]) - float(r_off["rmse"])
        deltas.append(d)
        nan_fracs.append(float(r_off["gr_nan_frac"]))
        if d < -0.01:
            helped += 1
        elif d > 0.01:
            hurt += 1
        else:
            tied += 1

    print("\n## per-well win/loss (well RMSE, ON - OFF, |delta|<=0.01 = tied)")
    print(f"  helped(ON better)={helped}  hurt(ON worse)={hurt}  tied={tied}")
    deltas_arr = np.array(deltas)
    print(
        f"  per-well delta: median={np.median(deltas_arr):+.4f}  "
        f"p10={np.percentile(deltas_arr, 10):+.4f}  p90={np.percentile(deltas_arr, 90):+.4f}"
    )

    nan_arr = np.array(nan_fracs)
    if np.std(nan_arr) > 0 and np.std(deltas_arr) > 0:
        corr = float(np.corrcoef(nan_arr, deltas_arr)[0, 1])
    else:
        corr = float("nan")
    print(
        f"\n## correlation(gr_nan_frac, per-well RMSE delta ON-OFF) = {corr:+.4f}  "
        "(negative = higher NaN rate wells improve more under ON)"
    )
    print(
        f"  gr_nan_frac over sample: median={np.median(nan_arr):.3f} "
        f"p90={np.percentile(nan_arr, 90):.3f} max={nan_arr.max():.3f}"
    )

    print("\n## bank cross-check (OFF vs outputs/path_bank_pfbma.npz pf_bma_scale3_tvt)")
    print(f"  wells checked: {n_bank_checked}/{len(rows_off)}")
    print(f"  max |diff|: {bank_max_abs_diff:.6f}  (tolerance {BANK_TOLERANCE:g}; float32 bank)")
    print(f"  MATCH: {bank_max_abs_diff <= BANK_TOLERANCE}")

    print("\n## timing")
    print(f"  OFF: total={t_off_total:.1f}s mean/well={t_off_total / len(rows_off):.3f}s")
    print(f"  ON:  total={t_on_total:.1f}s mean/well={t_on_total / len(rows_on):.3f}s")

    print(f"\n## memory: peak RSS = {peak_rss_mb():.0f}MB")
    print("\n## verdict threshold: ON pooled - OFF pooled <= -0.30 => proceed to bank rebuild")
    verdict = "PROCEED" if (pooled_on - pooled_off) <= -0.30 else "REJECT"
    print(f"## verdict: {verdict}  (delta={pooled_on - pooled_off:+.4f})")

    print(f"\ntotal runtime: {time.perf_counter() - t_program_start:.1f}s")


if __name__ == "__main__":
    main()
