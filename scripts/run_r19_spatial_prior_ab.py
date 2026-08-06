"""R19 A/B: does a spatial-cache position prior improve track_pf_multi (the
mainline PF that feeds the stack)?

Background (analysis/experiment_ledger.md, R17 close-out / R19 open item):
stack v2.6's error is dominated (~55%) by a per-well *constant offset* --
traced to a mid-zone "step" mistrack that the tracker then carries flat for
the rest of the eval zone. ``track_pf`` / ``track_pf_multi``
(``rogii.registration.particle``) only observe GR, whose eval-zone median
NaN rate is ~30%: rows with missing GR skip the likelihood reweighting step
entirely and the particle cloud pure-diffuses through them under the motion
model alone, so a genuine step near a GR gap goes uncorrected. Meanwhile
``data/processed/spatial_cache/{well}.npz`` holds a leave-self-out,
eval-zone-aligned per-row TVT prediction from nearby wells' formation-depth
surfaces (``spatial_tvt``, built by ``scripts/build_spatial_cache.py`` via
``rogii.spatial.predict_spatial``) plus a per-well fit-quality scalar
(``prefix_rmse``) -- available regardless of GR NaN, because it never reads
this well's own GR at all.

R19 (``src/rogii/registration/particle.py``, this project's implementer) now
exposes ``spatial_prior_tvt`` / ``spatial_prior_sigma`` as opt-in ``track_pf``
kwargs: a per-row Gaussian log-likelihood term added alongside GR's, applied
on *every* row where the prior is finite (including GR-NaN rows). Since the
raw spatial surface prediction carries a per-well constant bias of its own
(pooled RMSE ~25.2 unblended, per the experiment ledger), this script
anchor-aligns it before handing it to the tracker: ``prior_curve =
spatial_tvt - spatial_tvt[0] + anchor`` shifts the whole curve so its first
eval-zone value coincides with the well's own known-zone anchor, injecting
only the *shape/slope* the surface predicts, not its absolute-position bias.

This script measures OFF (plain ``track_pf_multi(h, tw)``, the exact call
``scripts/build_tracker_cache.py`` uses to build the tracker cache the stack
consumes) vs. ON (same call plus the anchor-aligned prior, sigma =
``clip(2.0 * prefix_rmse, 5.0, 40.0)``) on a 150-well deterministic subset
(the same evenly-spaced-over-sorted-wells sample as
``scripts/run_r6b1_fill_gr_gaps_ab.py`` / ``run_r6b3_init_spread_ab.py``, so
results are directly comparable to those R6-b experiments). Wells whose
spatial cache is missing, whose ``spatial_tvt`` length doesn't match the
eval zone, or whose ``prefix_rmse`` exceeds 25 ft get the prior disabled
(``prior_used=False``): their ON-arm call is identical to OFF (results are
reused directly rather than recomputed, since ``track_pf_multi`` is
deterministic given identical arguments -- this also roughly halves the
wasted compute on those wells).

Two pooled-RMSE metrics are reported per arm: the raw PF mean (``pf_tvt``)
and the "blend w0.7" anchor blend (``anchor + 0.7 * (pf_tvt - anchor)``) --
the latter matches the experiment ledger's current-best headline number
(PF blend w0.7 = 11.0621, from ``scripts/run_pf_cv.py`` / consumed by
``scripts/build_tracker_cache.py``'s own regression guard), so this script's
OFF-arm blend number should reproduce it closely (same tracker call,
different well subset -- 150/773 wells, not identical, so exact equality
is not expected, only the same ballpark).

Usage::

    nohup setsid uv run python scripts/run_r19_spatial_prior_ab.py \
        > outputs/r19_spatial_prior_ab.log 2>&1 &
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import baseline  # noqa: E402
from rogii import data as D  # noqa: E402
from rogii.registration.particle import track_pf_multi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPATIAL_CACHE_DIR = REPO_ROOT / "data" / "processed" / "spatial_cache"

N_WELLS_SAMPLE = 150
BLEND_W = 0.7  # matches the ledger's "PF blend w0.7" headline definition

# Prior eligibility / calibration (per the task spec). Env overrides let a
# re-run test a different sigma calibration without touching arm-1 defaults.
PREFIX_RMSE_MAX = float(os.environ.get("R19_PREFIX_RMSE_MAX", "25.0"))
SIGMA_MULT = float(os.environ.get("R19_SIGMA_MULT", "2.0"))
SIGMA_MIN = float(os.environ.get("R19_SIGMA_MIN", "5.0"))
SIGMA_MAX = float(os.environ.get("R19_SIGMA_MAX", "40.0"))

PROGRESS_EVERY = 10


def pooled_rmse_from_sse(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


def well_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def deterministic_sample(all_wells: list[str], n: int) -> list[str]:
    """Evenly spaced indices into the sorted well list (deterministic, no RNG).

    Identical to ``scripts/run_r6b1_fill_gr_gaps_ab.py`` / ``run_r6b3_init_spread_ab.py``'s
    sample so this A/B's 150-well subset is directly comparable to those R6-b
    experiments.
    """
    if n >= len(all_wells):
        return list(all_wells)
    idx = sorted({round(i * (len(all_wells) - 1) / (n - 1)) for i in range(n)})
    return [all_wells[i] for i in idx]


def read_spatial_cache(well: str) -> tuple[np.ndarray | None, float]:
    """Raw ``spatial_tvt`` + ``prefix_rmse`` for ``well``, or ``(None, nan)`` if the
    well has no cached npz (build_spatial_cache.py covers all 773 train wells,
    so this is a defensive guard, not an expected path)."""
    path = SPATIAL_CACHE_DIR / f"{well}.npz"
    if not path.exists():
        return None, float("nan")
    with np.load(path) as npz:
        spatial_tvt = npz["spatial_tvt"].astype(np.float64)
        prefix_rmse = float(npz["prefix_rmse"])
    return spatial_tvt, prefix_rmse


def anchor_aligned_prior(
    spatial_tvt: np.ndarray | None,
    n_eval: int,
    anchor: float,
    prefix_rmse: float,
) -> tuple[np.ndarray | None, float | None]:
    """Anchor-aligned prior curve + sigma, or ``(None, None)`` if unusable.

    Disables the prior (returns ``(None, None)``) when: the cache is missing
    (``spatial_tvt is None``), its length doesn't match this well's eval
    zone, its first row is non-finite (alignment is impossible), or its
    ``prefix_rmse`` exceeds ``PREFIX_RMSE_MAX`` (the well's spatial fit is
    judged too unreliable to trust as a position prior).
    """
    if spatial_tvt is None or spatial_tvt.shape[0] != n_eval:
        return None, None
    if not np.isfinite(prefix_rmse) or prefix_rmse > PREFIX_RMSE_MAX:
        return None, None
    if not np.isfinite(spatial_tvt[0]):
        return None, None

    prior_curve = spatial_tvt - spatial_tvt[0] + anchor
    sigma = float(np.clip(SIGMA_MULT * prefix_rmse, SIGMA_MIN, SIGMA_MAX))
    return prior_curve, sigma


def main() -> None:
    t_program_start = time.perf_counter()

    all_wells = D.list_wells("train")
    print(f"# train wells (total): {len(all_wells)}")
    sample = deterministic_sample(all_wells, N_WELLS_SAMPLE)
    print(f"# deterministic subset: {len(sample)} wells (evenly spaced over sorted list)")
    print(f"  first5={sample[:5]}  last3={sample[-3:]}")

    print(
        "\n# config: track_pf_multi default (n_seeds=8, n_particles=512, "
        "scales=(8,12,20,30), fill_gr_gaps=False) -- both arms, matching "
        "scripts/build_tracker_cache.py's mainline call"
    )
    print(
        f"# prior: prefix_rmse_max={PREFIX_RMSE_MAX}  "
        f"sigma=clip({SIGMA_MULT}*prefix_rmse, {SIGMA_MIN}, {SIGMA_MAX})  "
        "anchor-aligned (spatial_tvt - spatial_tvt[0] + anchor)"
    )
    print(f"# blend_w={BLEND_W}  (pred = anchor + blend_w * (pf_tvt - anchor))")

    rows: list[dict[str, object]] = []
    sse_pf_off, sse_pf_on = 0.0, 0.0
    sse_blend_off, sse_blend_on = 0.0, 0.0
    n_total = 0
    skipped = 0
    prior_used_count = 0
    t_off_total = 0.0
    t_on_total = 0.0

    t_loop_start = time.perf_counter()
    for i, well in enumerate(sample, start=1):
        h = D.load_horizontal(well, "train")
        tw = D.load_typewell(well, "train")
        ez = D.eval_mask(h)
        n_eval = int(ez.sum())
        if n_eval == 0 or h["TVT_input"].notna().sum() == 0:
            skipped += 1
            del h, tw
            continue

        y_true = h["TVT"].to_numpy(dtype=np.float64)[ez]
        anchor = D.last_known_tvt(h)
        anchor_pred = baseline.predict_carry_last(h)

        t0 = time.perf_counter()
        result_off = track_pf_multi(h, tw)
        t_off = time.perf_counter() - t0
        t_off_total += t_off

        spatial_tvt, prefix_rmse = read_spatial_cache(well)
        prior_curve, sigma = anchor_aligned_prior(spatial_tvt, n_eval, anchor, prefix_rmse)
        prior_used = prior_curve is not None

        if prior_used:
            prior_used_count += 1
            t0 = time.perf_counter()
            result_on = track_pf_multi(
                h, tw, spatial_prior_tvt=prior_curve, spatial_prior_sigma=sigma
            )
            t_on = time.perf_counter() - t0
            t_on_total += t_on
        else:
            result_on = result_off  # identical call to OFF -- reuse, don't recompute
            t_on = 0.0

        pf_off = result_off.tvt.astype(np.float64)
        pf_on = result_on.tvt.astype(np.float64)
        blend_off = anchor_pred + BLEND_W * (pf_off - anchor_pred)
        blend_on = anchor_pred + BLEND_W * (pf_on - anchor_pred)

        sse_pf_off += float(np.sum((y_true - pf_off) ** 2))
        sse_pf_on += float(np.sum((y_true - pf_on) ** 2))
        sse_blend_off += float(np.sum((y_true - blend_off) ** 2))
        sse_blend_on += float(np.sum((y_true - blend_on) ** 2))
        n_total += y_true.size

        rows.append(
            {
                "well": well,
                "n_eval": n_eval,
                "prior_used": prior_used,
                "prefix_rmse": prefix_rmse,
                "blend_rmse_off": well_rmse(y_true, blend_off),
                "blend_rmse_on": well_rmse(y_true, blend_on),
                "pf_rmse_off": well_rmse(y_true, pf_off),
                "pf_rmse_on": well_rmse(y_true, pf_on),
                "t_off": t_off,
                "t_on": t_on,
            }
        )

        del h, tw
        gc.collect()

        if i % PROGRESS_EVERY == 0 or i == len(sample):
            elapsed = time.perf_counter() - t_loop_start
            n_used_so_far = sum(1 for r in rows if r["prior_used"])
            print(
                f"  [{i}/{len(sample)}] elapsed={elapsed:.1f}s "
                f"prior_used={n_used_so_far}/{len(rows)} "
                f"peak_rss={peak_rss_mb():.0f}MB",
                flush=True,
            )

    print(f"\n# skipped wells (no eval zone / no anchor): {skipped}")
    print(f"# used wells: {len(rows)}")
    print(f"# prior_used wells: {prior_used_count}/{len(rows)}")

    pf_off_pooled = pooled_rmse_from_sse(sse_pf_off, n_total)
    pf_on_pooled = pooled_rmse_from_sse(sse_pf_on, n_total)
    blend_off_pooled = pooled_rmse_from_sse(sse_blend_off, n_total)
    blend_on_pooled = pooled_rmse_from_sse(sse_blend_on, n_total)

    print(f"\n{'=' * 78}\n# pooled RMSE ({len(rows)} wells)\n{'=' * 78}")
    print(f"  pf_mean   OFF: {pf_off_pooled:.4f}   ON: {pf_on_pooled:.4f}   "
          f"delta: {pf_on_pooled - pf_off_pooled:+.4f}")
    print(f"  blend w{BLEND_W}  OFF: {blend_off_pooled:.4f}   ON: {blend_on_pooled:.4f}   "
          f"delta: {blend_on_pooled - blend_off_pooled:+.4f}")

    helped = hurt = tied = 0
    deltas: list[float] = []
    prefix_rmses: list[float] = []
    for r in rows:
        if not r["prior_used"]:
            continue
        d = float(r["blend_rmse_on"]) - float(r["blend_rmse_off"])
        deltas.append(d)
        prefix_rmses.append(float(r["prefix_rmse"]))
        if d < -0.01:
            helped += 1
        elif d > 0.01:
            hurt += 1
        else:
            tied += 1

    print(
        "\n## per-well win/loss (blend-w0.7 well RMSE, ON - OFF, |delta|<=0.01 = tied, "
        f"restricted to prior_used wells, n={len(deltas)})"
    )
    print(f"  helped(ON better)={helped}  hurt(ON worse)={hurt}  tied={tied}")
    if deltas:
        deltas_arr = np.array(deltas)
        print(
            f"  per-well delta: median={np.median(deltas_arr):+.4f}  "
            f"p10={np.percentile(deltas_arr, 10):+.4f}  p90={np.percentile(deltas_arr, 90):+.4f}"
        )

        prefix_arr = np.array(prefix_rmses)
        if np.std(prefix_arr) > 0 and np.std(deltas_arr) > 0:
            corr = float(np.corrcoef(prefix_arr, deltas_arr)[0, 1])
        else:
            corr = float("nan")
        print(
            f"\n## correlation(prefix_rmse, per-well blend RMSE delta ON-OFF) = {corr:+.4f}  "
            "(positive = worse spatial fit quality predicts worse/less-helpful prior)"
        )
        print(
            f"  prefix_rmse over prior_used wells: median={np.median(prefix_arr):.3f} "
            f"p90={np.percentile(prefix_arr, 90):.3f} max={prefix_arr.max():.3f}"
        )
    else:
        print("  (no prior_used wells in this sample -- cannot compute delta stats)")

    print("\n## timing")
    print(f"  OFF: total={t_off_total:.1f}s mean/well={t_off_total / max(len(rows), 1):.3f}s")
    n_on_computed = max(prior_used_count, 1)
    print(
        f"  ON (prior_used wells only): total={t_on_total:.1f}s "
        f"mean/well={t_on_total / n_on_computed:.3f}s"
    )

    print(f"\n## memory: peak RSS = {peak_rss_mb():.0f}MB")

    print(
        "\n## verdict threshold: ON blend-w0.7 pooled - OFF blend-w0.7 pooled <= -0.05 "
        "=> improvement (see docs/playbooks/02_run_experiment.md's +/-0.05ft noise floor)"
    )
    blend_delta = blend_on_pooled - blend_off_pooled
    if blend_delta <= -0.05:
        verdict = "IMPROVED"
    elif blend_delta >= 0.05:
        verdict = "REGRESSED"
    else:
        verdict = "NEUTRAL (within noise floor)"
    print(f"## verdict: {verdict}  (delta={blend_delta:+.4f})")

    print(f"\ntotal runtime: {time.perf_counter() - t_program_start:.1f}s")


if __name__ == "__main__":
    main()
