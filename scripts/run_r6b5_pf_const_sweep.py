"""R6-b(5): coarse one-at-a-time sweep of the PF process-model constants.

Background (analysis/experiment_ledger.md, R6-b series): the particle filter's
process-model constants (``_POS_PROCESS_NOISE=0.005``, ``_RATE_NOISE_STD=0.002``,
``_RATE_MOMENTUM=0.998``, ``_RESAMPLE_ROUGHEN_POS=0.1``) are inherited from the
common-ancestor reference notebook and have never been systematically tuned.
R6-b(3) moved a single constant (``init_pos_std`` 0.3 -> 2.0) and produced a
-0.59 candidate improvement, so this space may hold more.

This script sweeps each of the four constants one-at-a-time (coarse, one-sided
grids; 10 variants + baseline = 11 runs) for ``pf_bma_scale3`` at the
production config (n_particles=512, n_seeds=12, init_pos_std=2.0,
fill_gr_gaps=False, fixed gr_scales) on the same deterministic 150-well
subset as the other R6-b A/Bs:

- pos_process_noise:    0.002 / 0.010 / 0.020   (current 0.005)
- rate_noise_std:       0.001 / 0.004 / 0.008   (current 0.002)
- rate_momentum:        0.995 / 0.999           (current 0.998)
- resample_roughen_pos: 0.05  / 0.3             (current 0.1)

The baseline run is cross-checked against ``outputs/path_bank_pfbma_v2.npz``'s
``pf_bma_scale3_tvt`` (implementation-regression guard) and against the
expected 150-well pooled RMSE 11.5933 (the r6b2/r6b4 OFF arm; deterministic,
so it must reproduce). **Checkpointing**: each configuration's full result
block is printed (flushed) the moment that configuration finishes, so a
killed run still leaves every completed configuration's numbers in the log.

Verdicts are left to the coordinator (promising directions feed a second
sweep or a combined A/B); this script only reports the measured table.

Usage::

    nohup uv run python scripts/run_r6b5_pf_const_sweep.py \
        > outputs/r6b5_pf_const_sweep.log 2>&1 &
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
BANK_PATH = REPO_ROOT / "outputs" / "path_bank_pfbma_v2.npz"

N_WELLS_SAMPLE = 150
N_PARTICLES = 512
N_SEEDS = 12
BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
GR_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)
TARGET_SCALE = 3.0
INIT_POS_STD = 2.0  # production setting (R6-b(3) adopted), all configs

# Expected baseline pooled RMSE on this exact 150-well subset (the OFF arm of
# scripts/run_r6b2_nseeds_ab.py / run_r6b4_adaptive_scale_ab.py -- the runs
# are deterministic, so the baseline must reproduce this to ~4 decimals).
EXPECTED_BASELINE_POOLED = 11.5933
BASELINE_TOLERANCE = 0.01
BANK_TOLERANCE = 1e-3  # float32 round-trip tolerance for the baseline==bank check

PROGRESS_EVERY = 50

# (label, track_pf_bma keyword overrides) -- baseline first (bank-checked).
SWEEP: tuple[tuple[str, dict[str, float]], ...] = (
    ("baseline (current constants)", {}),
    ("pos_process_noise=0.002", {"pos_process_noise": 0.002}),
    ("pos_process_noise=0.010", {"pos_process_noise": 0.010}),
    ("pos_process_noise=0.020", {"pos_process_noise": 0.020}),
    ("rate_noise_std=0.001", {"rate_noise_std": 0.001}),
    ("rate_noise_std=0.004", {"rate_noise_std": 0.004}),
    ("rate_noise_std=0.008", {"rate_noise_std": 0.008}),
    ("rate_momentum=0.995", {"rate_momentum": 0.995}),
    ("rate_momentum=0.999", {"rate_momentum": 0.999}),
    ("resample_roughen_pos=0.05", {"resample_roughen_pos": 0.05}),
    ("resample_roughen_pos=0.3", {"resample_roughen_pos": 0.3}),
)


def pooled_rmse_from_sse(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


def well_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def deterministic_sample(all_wells: list[str], n: int) -> list[str]:
    """Evenly spaced indices into the sorted well list (deterministic, no RNG).

    Identical to the other R6-b A/B scripts' sample so results are directly
    comparable across the R6-b experiments.
    """
    if n >= len(all_wells):
        return list(all_wells)
    idx = sorted({round(i * (len(all_wells) - 1) / (n - 1)) for i in range(n)})
    return [all_wells[i] for i in idx]


def load_bank_scale3(used_wells_wanted: list[str]) -> dict[str, np.ndarray]:
    """Per-well pf_bma_scale3_tvt row-blocks from outputs/path_bank_pfbma_v2.npz,
    keyed by well name, restricted to `used_wells_wanted`. Asserts the bank
    really is the init_pos_std=2.0 / n_seeds=12 v2 build."""
    with np.load(BANK_PATH, allow_pickle=True) as npz:
        assert float(npz["init_pos_std"]) == INIT_POS_STD, (
            f"bank v2 init_pos_std {float(npz['init_pos_std'])} != expected {INIT_POS_STD}"
        )
        assert int(npz["n_seeds"]) == N_SEEDS, (
            f"bank v2 n_seeds {int(npz['n_seeds'])} != {N_SEEDS}"
        )
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


def run_config(
    sample: list[str],
    overrides: dict[str, float],
    bank_scale3: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    """One full 150-well pass at the given constant overrides.

    Returns pooled RMSE, per-well RMSEs (well order = sample order), wall
    time, and (when ``bank_scale3`` is given, i.e. the baseline config) the
    max |diff| against the bank."""
    sse, n_rows = 0.0, 0
    per_well: dict[str, float] = {}
    bank_max_abs_diff = 0.0
    n_bank_checked = 0
    skipped = 0

    t0 = time.perf_counter()
    for i, well in enumerate(sample, start=1):
        h = D.load_horizontal(well, "train")
        tw = D.load_typewell(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            skipped += 1
            continue

        y_true = h["TVT"].to_numpy(dtype=np.float64)[ez]
        result = track_pf_bma(
            h,
            tw,
            n_particles=N_PARTICLES,
            n_seeds=N_SEEDS,
            bma_scales=BMA_SCALES,
            gr_scales=GR_SCALES,
            init_pos_std=INIT_POS_STD,
            **overrides,
        )
        tvt = result.tvt_by_scale[TARGET_SCALE].astype(np.float64)

        if bank_scale3 is not None and well in bank_scale3:
            bank_tvt = bank_scale3[well]
            if bank_tvt.shape == tvt.shape:
                bank_max_abs_diff = max(
                    bank_max_abs_diff, float(np.max(np.abs(bank_tvt - tvt)))
                )
                n_bank_checked += 1

        sse += float(np.sum((y_true - tvt) ** 2))
        n_rows += y_true.size
        per_well[well] = well_rmse(y_true, tvt)

        del h, tw
        gc.collect()

        if i % PROGRESS_EVERY == 0:
            print(
                f"    [{i}/{len(sample)}] elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    return {
        "pooled": pooled_rmse_from_sse(sse, n_rows),
        "per_well": per_well,
        "time_s": time.perf_counter() - t0,
        "skipped": skipped,
        "bank_max_abs_diff": bank_max_abs_diff,
        "n_bank_checked": n_bank_checked,
    }


def main() -> None:
    t_program_start = time.perf_counter()

    all_wells = D.list_wells("train")
    print(f"# train wells (total): {len(all_wells)}")
    sample = deterministic_sample(all_wells, N_WELLS_SAMPLE)
    print(f"# deterministic subset: {len(sample)} wells (evenly spaced over sorted list)")
    print(f"  first5={sample[:5]}  last3={sample[-3:]}")

    print(
        f"\n# config (all runs): n_particles={N_PARTICLES} n_seeds={N_SEEDS} "
        f"bma_scales={BMA_SCALES} gr_scales={GR_SCALES} target_scale={TARGET_SCALE} "
        f"init_pos_std={INIT_POS_STD} fill_gr_gaps=False"
    )
    print(f"# sweep: {len(SWEEP)} configurations (baseline + {len(SWEEP) - 1} variants)")

    bank_scale3 = load_bank_scale3(sample)
    print(
        f"# loaded bank v2 pf_bma_scale3_tvt for {len(bank_scale3)}/{len(sample)} "
        "sample wells (baseline cross-check)",
        flush=True,
    )

    results: list[tuple[str, dict[str, object]]] = []
    baseline_per_well: dict[str, float] | None = None
    baseline_pooled = float("nan")

    for k, (label, overrides) in enumerate(SWEEP, start=1):
        print(f"\n{'=' * 78}\n# [{k}/{len(SWEEP)}] {label}\n{'=' * 78}", flush=True)
        is_baseline = not overrides
        res = run_config(sample, overrides, bank_scale3 if is_baseline else None)
        results.append((label, res))

        pooled = float(res["pooled"])  # type: ignore[arg-type]
        if is_baseline:
            baseline_per_well = dict(res["per_well"])  # type: ignore[arg-type]
            baseline_pooled = pooled
            print(f"  pooled RMSE = {pooled:.4f}  (expected {EXPECTED_BASELINE_POOLED})")
            if abs(pooled - EXPECTED_BASELINE_POOLED) > BASELINE_TOLERANCE:
                print(
                    "  [WARN] baseline pooled RMSE drifted from the r6b2/r6b4 OFF-arm "
                    "value -- implementation regression suspected"
                )
            print(
                f"  bank cross-check: {res['n_bank_checked']}/{len(sample)} wells, "
                f"max |diff| = {float(res['bank_max_abs_diff']):.6f} "  # type: ignore[arg-type]
                f"(tolerance {BANK_TOLERANCE:g})  "
                f"MATCH: {float(res['bank_max_abs_diff']) <= BANK_TOLERANCE}"  # type: ignore[arg-type]
            )
        else:
            assert baseline_per_well is not None
            per_well = res["per_well"]  # type: ignore[assignment]
            deltas = np.array(
                [per_well[w] - baseline_per_well[w] for w in per_well]  # type: ignore[index]
            )
            helped = int(np.sum(deltas < -0.01))
            hurt = int(np.sum(deltas > 0.01))
            tied = deltas.size - helped - hurt
            print(
                f"  pooled RMSE = {pooled:.4f}  delta_vs_baseline = "
                f"{pooled - baseline_pooled:+.4f}"
            )
            print(
                f"  helped={helped} hurt={hurt} tied={tied}  "
                f"per-well delta median={np.median(deltas):+.4f} "
                f"p10={np.percentile(deltas, 10):+.4f} p90={np.percentile(deltas, 90):+.4f}"
            )
        print(
            f"  time = {float(res['time_s']):.1f}s "  # type: ignore[arg-type]
            f"({float(res['time_s']) / len(sample):.2f}s/well)  "  # type: ignore[arg-type]
            f"skipped={res['skipped']}  peak_rss={peak_rss_mb():.0f}MB",
            flush=True,
        )

    print(f"\n{'=' * 78}\n# final summary (sorted by pooled RMSE)\n{'=' * 78}")
    print(f"{'configuration':34s} {'pooled RMSE':>12s} {'delta vs baseline':>18s}")
    print("-" * 68)
    for label, res in sorted(results, key=lambda t: float(t[1]["pooled"])):  # type: ignore[arg-type]
        pooled = float(res["pooled"])  # type: ignore[arg-type]
        delta = pooled - baseline_pooled
        marker = " <= baseline" if label.startswith("baseline") else ""
        print(f"{label:34s} {pooled:12.4f} {delta:+18.4f}{marker}")

    print(f"\n## memory: peak RSS = {peak_rss_mb():.0f}MB")
    print(
        f"\ntotal runtime: {time.perf_counter() - t_program_start:.1f}s "
        f"({(time.perf_counter() - t_program_start) / 60:.1f}min)"
    )


if __name__ == "__main__":
    main()
