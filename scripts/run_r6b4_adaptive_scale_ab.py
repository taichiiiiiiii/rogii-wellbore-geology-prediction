"""R6-b(4) A/B: does a per-well adaptive GR-likelihood scale improve pf_bma_scale3?

Hypothesis under test (analysis/experiment_ledger.md, 2026-07-10 R6-b dissection,
improvement queue item (4)): the adopted PF-BMA hands every well the same
fixed per-particle GR-likelihood scale grid ``gr_scales=(8, 12, 20, 30)``
(round-robin, :func:`_assign_particle_scales`); the sunnywu27 reference
instead calibrates a single per-well sigma from the known zone (its
``_gr_sig``, clipped [10, 60]) so wells with clean GR get a sharp likelihood
and noisy wells a tolerant one. This experiment tests the per-well idea
*while keeping this repo's affine-calibration advantage*: the well's noise
level ``a`` is the std of the known zone's **affine-calibrated** residual
(``h.GR - apply_affine(twGR(TVT_input))``, clipped [8, 60]) and the grid
becomes ``(0.6a, a, 1.6a)`` -- centered on the well's own noise but still
multi-scale (see ``particle._adaptive_gr_scale_grid``).

Note the historical caution baked into ``_DEFAULT_SCALES``'s docstring: a
per-well calibrated sigma was already measured once (pre-BMA, pre-init2.0)
and did not beat the fixed grid. This A/B re-tests it under the *new*
production config (PF-BMA, init_pos_std=2.0), where the answer may differ.

This script measures OFF (fixed grid, adopted) vs ON (adaptive_gr_scales=True)
pooled RMSE for ``pf_bma_scale3`` with **both arms at the production setting**
``init_pos_std=2.0`` / ``n_seeds=12`` (fill_gr_gaps=False on both arms), on
the same deterministic 150-well subset as the other R6-b A/Bs, and
cross-checks the OFF side against ``outputs/path_bank_pfbma_v2.npz``'s
``pf_bma_scale3_tvt`` column (implementation-regression guard).

Adoption threshold is ``delta <= -0.30`` (same strictness as R6-b(3): a
likelihood-design change is invasive).

Usage::

    nohup uv run python scripts/run_r6b4_adaptive_scale_ab.py \
        > outputs/r6b4_adaptive_scale_ab.log 2>&1 &
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
from rogii import typewell as TW  # noqa: E402
from rogii.registration.particle import track_pf_bma  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = REPO_ROOT / "outputs" / "path_bank_pfbma_v2.npz"

N_WELLS_SAMPLE = 150
N_PARTICLES = 512
N_SEEDS = 12
BMA_SCALES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
GR_SCALES: tuple[float, ...] = (8.0, 12.0, 20.0, 30.0)  # OFF arm's fixed grid
TARGET_SCALE = 3.0
INIT_POS_STD = 2.0  # production setting (R6-b(3) adopted), BOTH arms
ADAPTIVE_CLIP = (8.0, 60.0)  # mirrors particle._ADAPTIVE_SCALE_CLIP (diagnostic only)
ADOPTION_DELTA = -0.30  # likelihood-design change => same strictness as R6-b(3)
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

    Identical to the other R6-b A/B scripts' sample so results are directly
    comparable across the R6-b experiments.
    """
    if n >= len(all_wells):
        return list(all_wells)
    idx = sorted({round(i * (len(all_wells) - 1) / (n - 1)) for i in range(n)})
    return [all_wells[i] for i in idx]


def adaptive_a_diag(h: object, tw: object) -> float:
    """Diagnostic-only replica of particle._adaptive_gr_scale_grid's ``a``
    (clipped calibrated-residual std), via the public typewell API."""
    known = h["TVT_input"].notna() & h["GR"].notna()  # type: ignore[index]
    if int(known.sum()) < 10:
        return float("nan")
    lookup = TW.tw_gr_lookup(tw)
    gain, offset = TW.fit_affine_gr(h, tw)
    tvt_known = h.loc[known, "TVT_input"].to_numpy(dtype=float)  # type: ignore[attr-defined]
    gr_known = h.loc[known, "GR"].to_numpy(dtype=float)  # type: ignore[attr-defined]
    resid_std = float(np.std(gr_known - TW.apply_affine(lookup(tvt_known), gain, offset)))
    if not np.isfinite(resid_std):
        return float("nan")
    return float(np.clip(resid_std, *ADAPTIVE_CLIP))


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


def main() -> None:
    t_program_start = time.perf_counter()

    all_wells = D.list_wells("train")
    print(f"# train wells (total): {len(all_wells)}")
    sample = deterministic_sample(all_wells, N_WELLS_SAMPLE)
    print(f"# deterministic subset: {len(sample)} wells (evenly spaced over sorted list)")
    print(f"  first5={sample[:5]}  last3={sample[-3:]}")

    print(
        f"\n# config: n_particles={N_PARTICLES} n_seeds={N_SEEDS} bma_scales={BMA_SCALES} "
        f"target_scale={TARGET_SCALE} init_pos_std={INIT_POS_STD} (both arms) "
        "fill_gr_gaps=False (both arms)"
    )
    print(
        f"# OFF: fixed gr_scales={GR_SCALES}   "
        "ON: adaptive_gr_scales=True (grid (0.6a, a, 1.6a), a=calibrated resid std clip[8,60])"
    )

    bank_scale3 = load_bank_scale3(sample)
    print(
        f"# loaded bank v2 pf_bma_scale3_tvt for {len(bank_scale3)}/{len(sample)} sample wells"
    )

    rows_off: list[dict[str, object]] = []
    rows_on: list[dict[str, object]] = []
    sse_off, n_off = 0.0, 0
    sse_on, n_on = 0.0, 0
    bank_max_abs_diff = 0.0
    n_bank_checked = 0
    t_off_total = 0.0
    t_on_total = 0.0
    skipped = 0
    a_values: list[float] = []

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
        a_well = adaptive_a_diag(h, tw)
        a_values.append(a_well)

        t0 = time.perf_counter()
        result_off = track_pf_bma(
            h,
            tw,
            n_particles=N_PARTICLES,
            n_seeds=N_SEEDS,
            bma_scales=BMA_SCALES,
            gr_scales=GR_SCALES,
            init_pos_std=INIT_POS_STD,
            adaptive_gr_scales=False,
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
            init_pos_std=INIT_POS_STD,
            adaptive_gr_scales=True,
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
    print(f"  OFF (fixed gr_scales, existing behaviour): {pooled_off:.4f}")
    print(f"  ON  (adaptive_gr_scales=True):             {pooled_on:.4f}")
    print(f"  delta (ON - OFF):                          {pooled_on - pooled_off:+.4f}")

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

    a_arr = np.array(a_values)
    a_finite = a_arr[np.isfinite(a_arr)]
    if a_finite.size:
        n_lo = int(np.sum(a_finite <= ADAPTIVE_CLIP[0] + 1e-9))
        n_hi = int(np.sum(a_finite >= ADAPTIVE_CLIP[1] - 1e-9))
        print(
            f"\n## adaptive scale a (clipped calibrated-resid std): "
            f"median={np.median(a_finite):.2f} p10={np.percentile(a_finite, 10):.2f} "
            f"p90={np.percentile(a_finite, 90):.2f}  at_lower_clip={n_lo} "
            f"at_upper_clip={n_hi} fallback_nan={int(np.isnan(a_arr).sum())}"
        )
        both_ok = np.isfinite(a_arr) & np.isfinite(deltas_arr)
        if both_ok.sum() >= 3 and np.std(a_arr[both_ok]) > 0:
            corr_a = float(np.corrcoef(a_arr[both_ok], deltas_arr[both_ok])[0, 1])
            print(f"  correlation(a, per-well RMSE delta ON-OFF) = {corr_a:+.4f}")

    print("\n## bank cross-check (OFF vs outputs/path_bank_pfbma_v2.npz pf_bma_scale3_tvt)")
    print(f"  wells checked: {n_bank_checked}/{len(rows_off)}")
    print(f"  max |diff|: {bank_max_abs_diff:.6f}  (tolerance {BANK_TOLERANCE:g}; float32 bank)")
    print(f"  MATCH: {bank_max_abs_diff <= BANK_TOLERANCE}")

    print("\n## timing")
    print(f"  OFF: total={t_off_total:.1f}s mean/well={t_off_total / len(rows_off):.3f}s")
    print(f"  ON:  total={t_on_total:.1f}s mean/well={t_on_total / len(rows_on):.3f}s")

    print(f"\n## memory: peak RSS = {peak_rss_mb():.0f}MB")
    print(
        f"\n## verdict threshold: ON pooled - OFF pooled <= {ADOPTION_DELTA} "
        "=> proceed to bank rebuild (likelihood-design change: same strictness as R6-b(3))"
    )
    verdict = "PROCEED" if (pooled_on - pooled_off) <= ADOPTION_DELTA else "REJECT"
    print(f"## verdict: {verdict}  (delta={pooled_on - pooled_off:+.4f})")

    print(f"\ntotal runtime: {time.perf_counter() - t_program_start:.1f}s")


if __name__ == "__main__":
    main()
