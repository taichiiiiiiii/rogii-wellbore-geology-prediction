"""Build the per-well offset time-series diagnostics cache (R28).

Builds the formation-depth surface bank once from every train well (see
``rogii.spatial.build_surface_bank``, same ``STRIDE``/``K_NEIGHBORS``/
``METHOD`` as ``build_spatial_cache.py``), then computes
``offset_diagnostics(h, bank, exclude_well=well)`` (leave-self-out KNN) for
every train well and stores the result in a single ``.npz`` file so a
downstream GBDT stack can pick up the prefix-drift features
(``offset_early``/``offset_mid``/``offset_late``/``wls_slope_per100``/
``wls_end``) without re-running the bank-build-plus-773-well cost.

Unlike ``build_spatial_cache.py`` (one ``.npz`` per well, since its arrays
are per-eval-row), every ``offset_diagnostics`` field is a well-level scalar,
so this script writes exactly one file: ``data/processed/offset_diag.npz``,
with all arrays aligned to the same ``wells`` order.

Usage::

    uv run python scripts/build_offset_diag_cache.py             # full 773-well run
    uv run python scripts/build_offset_diag_cache.py --limit 8    # smoke test (first 8 wells)

``--limit`` writes to ``outputs_smoke/offset_diag.npz`` instead of the real
cache path, so a smoke run can never clobber the full cache. ``--limit``
only restricts which wells are *diagnosed* -- the bank is always built from
every train well (matching ``build_spatial_cache.py``'s own bank, which has
no ``--limit``), since building it is cheap (~10-13s at ``stride=10`` per
``outputs/logs_archive/stack_v31_cv.log``, not the ~35min figure quoted
elsewhere for a bank-build-plus-773-well-*query* pass) and a bank built from
only the first few wells would give unrealistically sparse KNN neighbors,
making a smoke run's per-well timing/quality non-representative.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import data as D  # noqa: E402
from rogii import spatial as SP  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / "data" / "processed" / "offset_diag.npz"
SMOKE_CACHE_PATH = REPO_ROOT / "outputs_smoke" / "offset_diag.npz"

# Bank/query config matching build_spatial_cache.py's defaults (the
# experiment ledger's measured spatial-predictor run), so this cache's
# offset diagnostics are computed against the same bank as the spatial
# predictor's own calibration.
STRIDE = 10
K_NEIGHBORS = 10
METHOD = "idw"

PROGRESS_EVERY = 25

# npz array names, in OffsetDiag field order (plus "wells").
FIELDS = [
    "offset_med",
    "offset_early",
    "offset_mid",
    "offset_late",
    "wls_slope_per100",
    "wls_end",
    "prefix_rmse_best",
    "n_valid_prefix",
]


def _diagnose_well(well: str, bank: SP.SurfaceBank) -> SP.OffsetDiag:
    h = D.load_horizontal(well, "train")
    return SP.offset_diagnostics(h, bank, exclude_well=well, k=K_NEIGHBORS, method=METHOD)


def build(bank_wells: list[str], wells: list[str], out_path: Path) -> None:
    print(f"# bank wells: {len(bank_wells)}; wells to diagnose: {len(wells)}")
    print(f"# config: stride={STRIDE} k={K_NEIGHBORS} method={METHOD}")

    t0 = time.perf_counter()
    bank = SP.build_surface_bank(bank_wells, split="train", stride=STRIDE)
    bank_secs = time.perf_counter() - t0
    sizes = {f: bank.depths[f].size for f in bank.formations}
    print(f"# bank built in {bank_secs:.1f}s; samples per formation: {sizes}", flush=True)

    n = len(wells)
    offset_med = np.full(n, np.nan, dtype=np.float64)
    offset_early = np.full(n, np.nan, dtype=np.float64)
    offset_mid = np.full(n, np.nan, dtype=np.float64)
    offset_late = np.full(n, np.nan, dtype=np.float64)
    wls_slope_per100 = np.full(n, np.nan, dtype=np.float64)
    wls_end = np.full(n, np.nan, dtype=np.float64)
    prefix_rmse_best = np.full(n, np.inf, dtype=np.float64)
    n_valid_prefix = np.zeros(n, dtype=np.int32)

    t_start = time.perf_counter()
    for i, well in enumerate(wells):
        diag = _diagnose_well(well, bank)
        offset_med[i] = diag.offset_med
        offset_early[i] = diag.offset_early
        offset_mid[i] = diag.offset_mid
        offset_late[i] = diag.offset_late
        wls_slope_per100[i] = diag.wls_slope_per100
        wls_end[i] = diag.wls_end
        prefix_rmse_best[i] = diag.prefix_rmse_best
        n_valid_prefix[i] = diag.n_valid_prefix

        done = i + 1
        if done % PROGRESS_EVERY == 0 or done == n:
            elapsed = time.perf_counter() - t_start
            rate = elapsed / done
            print(
                f"[{done}/{n}] elapsed={elapsed:.1f}s ({rate:.3f}s/well)",
                flush=True,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        wells=np.array(wells, dtype=str),
        offset_med=offset_med,
        offset_early=offset_early,
        offset_mid=offset_mid,
        offset_late=offset_late,
        wls_slope_per100=wls_slope_per100,
        wls_end=wls_end,
        prefix_rmse_best=prefix_rmse_best,
        n_valid_prefix=n_valid_prefix,
    )
    print(f"# wrote {out_path}")

    _print_summary(
        n_valid_prefix,
        prefix_rmse_best,
        wls_slope_per100,
    )


def _print_summary(
    n_valid_prefix: np.ndarray,
    prefix_rmse_best: np.ndarray,
    wls_slope_per100: np.ndarray,
) -> None:
    n = n_valid_prefix.size
    n_nan_well = int(np.sum(n_valid_prefix == 0))
    finite_slope = wls_slope_per100[np.isfinite(wls_slope_per100)]
    finite_rmse = prefix_rmse_best[np.isfinite(prefix_rmse_best)]

    print(f"\n## Summary ({n} wells)\n")
    print(f"  wells with n_valid_prefix == 0 (all-NaN fallback): {n_nan_well}")
    if finite_rmse.size:
        print(
            f"  prefix_rmse_best (finite, n={finite_rmse.size}): "
            f"median={np.median(finite_rmse):.4f} p90={np.percentile(finite_rmse, 90):.4f}"
        )
    if finite_slope.size:
        print(
            f"  wls_slope_per100 (finite, n={finite_slope.size}): "
            f"median={np.median(finite_slope):.4f} p90={np.percentile(finite_slope, 90):.4f} "
            f"abs-median={np.median(np.abs(finite_slope)):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N wells; writes to outputs_smoke/offset_diag.npz instead "
        "of the real cache path",
    )
    args = parser.parse_args()

    bank_wells = D.list_wells("train")
    print(f"# train wells available: {len(bank_wells)}")

    if args.limit is not None:
        wells = bank_wells[: args.limit]
        out_path = SMOKE_CACHE_PATH
    else:
        wells = bank_wells
        out_path = CACHE_PATH

    build(bank_wells, wells, out_path)


if __name__ == "__main__":
    main()
