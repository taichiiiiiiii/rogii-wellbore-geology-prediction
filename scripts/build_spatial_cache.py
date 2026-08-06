"""Build a per-well spatial-prior feature cache for fast stack-model iteration.

Builds the formation-depth surface bank once from every train well (see
``rogii.spatial.build_surface_bank``), then computes ``predict_spatial(h,
bank, exclude_well=well)`` (leave-self-out KNN, default ``stride``/``k``/
``method``) for every train well and caches the result to disk so downstream
stack-model experiments (GBDT combining tracker + spatial features) don't
have to re-run the ~35-minute bank-build-plus-773-well cost on every
iteration.

Layout::

    data/processed/spatial_cache/{well}.npz   # spatial_tvt, prefix_rmse,
                                               # nn_dist_median, nn_dist
    data/processed/spatial_cache/manifest.csv # well, eval_len, prefix_rmse,
                                               # nn_dist_median, time_s

``spatial_tvt`` (float32) and ``nn_dist`` (float32, per-row nearest
non-self bank-sample distance) are aligned to ``data.eval_mask(h)`` order
(length = number of evaluation-zone rows for that well). ``prefix_rmse`` and
``nn_dist_median`` are scalars.

Usage::

    uv run python scripts/build_spatial_cache.py           # build (resume: skip cached) + verify
    uv run python scripts/build_spatial_cache.py --force   # rebuild every well from scratch
    uv run python scripts/build_spatial_cache.py --verify  # skip build, verify existing cache only

The verification step is a regression guard, not a formality: it recomputes
the pooled RMSE of the spatial gateblend-theta=2 predictor *from the cached
arrays* (``anchor + 0.7 * (spatial_tvt - anchor)`` where ``prefix_rmse <=
2.0``, else the flat anchor -- see ``scripts/run_spatial_cv.py``'s
``gateblend_t2``) and checks it matches the experiment ledger's measured
value (``analysis/experiment_ledger.md``: gateblend theta=2 = 14.8724, from
the full-773-well run in ``scripts/run_spatial_cv.py``). A mismatch beyond
the tolerance means the cache was built with a different stride/k/method than
the ledger's numbers and must not be trusted for downstream experiments.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import baseline  # noqa: E402
from rogii import data as D  # noqa: E402
from rogii import spatial as SP  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "spatial_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.csv"

MANIFEST_FIELDS = [
    "well",
    "eval_len",
    "prefix_rmse",
    "nn_dist_median",
    "time_s",
]

PROGRESS_EVERY = 50

# Bank/query config matching the experiment ledger's measured run
# (scripts/run_spatial_cv.py's module-level defaults).
STRIDE = 10
K_NEIGHBORS = 10
METHOD = "idw"

# Gate-blend predictor that reproduces the experiment ledger's headline
# spatial number (scripts/run_spatial_cv.py's gateblend_t2.0).
GATE_THETA = 2.0
BLEND_W = 0.7
EXPECTED_GATEBLEND_RMSE = 14.8724
RMSE_TOLERANCE = 0.01

VERIFY_RELOAD_N_WELLS = 5


def _well_npz_path(well: str) -> Path:
    return CACHE_DIR / f"{well}.npz"


def _load_manifest_rows() -> dict[str, dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open(newline="") as f:
        return {row["well"]: row for row in csv.DictReader(f)}


def _write_manifest(rows: dict[str, dict[str, str]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for well in sorted(rows):
            writer.writerow(rows[well])


def _manifest_row_from_cache(well: str) -> dict[str, str]:
    """Reconstruct a manifest row from an already-cached npz (timing unknown)."""
    with np.load(_well_npz_path(well)) as npz:
        eval_len = int(npz["spatial_tvt"].size)
        prefix_rmse = float(npz["prefix_rmse"])
        nn_dist_median = float(npz["nn_dist_median"])
    return {
        "well": well,
        "eval_len": str(eval_len),
        "prefix_rmse": f"{prefix_rmse:.6f}",
        "nn_dist_median": f"{nn_dist_median:.6f}",
        "time_s": "",
    }


def _nn_distance_per_row(h: pd.DataFrame, bank: SP.SurfaceBank, well: str) -> np.ndarray:
    """Per-row nearest non-self bank-sample distance for every eval-zone row.

    Same leave-self-out KD-tree logic as ``run_spatial_cv.py``'s
    ``_nn_distance_for_well``, but over every eval-zone row (not a subsample)
    so the result can be cached per-row, and against ``bank.formations[0]``
    (a single representative formation, matching that function's sanity-check
    role -- this is a spatial-density diagnostic, not per-formation).
    """
    formation = bank.formations[0]
    tree = bank.trees[formation]
    codes = bank.well_codes[formation]
    exclude_code = bank.well_to_code.get(well, -1)

    ez = D.eval_mask(h)
    x = h["X"].to_numpy(dtype=float)[ez]
    y = h["Y"].to_numpy(dtype=float)[ez]
    if x.size == 0:
        return np.array([], dtype=float)

    coords = np.column_stack([x, y])
    n_self = int(np.sum(codes == exclude_code))
    k = min(1 + n_self, codes.size)
    dist, idx = tree.query(coords, k=k, workers=-1)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    valid = codes[idx] != exclude_code
    first_valid = np.where(valid.any(axis=1), valid.argmax(axis=1), -1)
    rows = np.arange(coords.shape[0])
    return np.where(first_valid >= 0, dist[rows, np.maximum(first_valid, 0)], np.nan)


def _build_well_cache(well: str, bank: SP.SurfaceBank) -> dict[str, str]:
    """Compute and persist one well's spatial cache; returns its manifest row."""
    h = D.load_horizontal(well, "train")

    t0 = time.perf_counter()
    result = SP.predict_spatial(h, bank, exclude_well=well, k=K_NEIGHBORS, method=METHOD)
    nn_dist = _nn_distance_per_row(h, bank, well)
    time_s = time.perf_counter() - t0

    finite_nn = nn_dist[np.isfinite(nn_dist)] if nn_dist.size else np.array([])
    nn_dist_median = float(np.median(finite_nn)) if finite_nn.size else float("nan")
    eval_len = int(result.tvt.size)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _well_npz_path(well),
        spatial_tvt=result.tvt.astype(np.float32),
        prefix_rmse=np.float64(result.prefix_rmse),
        nn_dist_median=np.float64(nn_dist_median),
        nn_dist=nn_dist.astype(np.float32),
    )

    return {
        "well": well,
        "eval_len": str(eval_len),
        "prefix_rmse": f"{result.prefix_rmse:.6f}",
        "nn_dist_median": f"{nn_dist_median:.6f}",
        "time_s": f"{time_s:.4f}",
    }


def build_cache(wells: list[str], bank: SP.SurfaceBank, force: bool) -> None:
    manifest_rows = _load_manifest_rows()
    n_built = 0
    n_skipped = 0
    t_start = time.perf_counter()

    for i, well in enumerate(wells, start=1):
        npz_path = _well_npz_path(well)
        if npz_path.exists() and not force:
            n_skipped += 1
            if well not in manifest_rows:
                manifest_rows[well] = _manifest_row_from_cache(well)
        else:
            manifest_rows[well] = _build_well_cache(well, bank)
            n_built += 1

        if i % PROGRESS_EVERY == 0 or i == len(wells):
            elapsed = time.perf_counter() - t_start
            print(
                f"[{i}/{len(wells)}] built={n_built} skipped={n_skipped} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            _write_manifest(manifest_rows)  # incremental save: survive an interruption

    _write_manifest(manifest_rows)
    print(f"# build done: built={n_built} skipped={n_skipped} total={len(wells)}")


def _pooled_rmse(sq_err_sum: float, n: int) -> float:
    return float(np.sqrt(sq_err_sum / n)) if n else float("nan")


def _check_reload_lengths(wells: list[str]) -> None:
    sample = wells[:VERIFY_RELOAD_N_WELLS]
    print(f"\n## Reload length check ({len(sample)} wells)\n")
    for well in sample:
        h = D.load_horizontal(well, "train")
        eval_len = int(D.eval_mask(h).sum())
        with np.load(_well_npz_path(well)) as npz:
            spatial_tvt = npz["spatial_tvt"]
            nn_dist = npz["nn_dist"]
        lengths = {"spatial_tvt": spatial_tvt.size, "nn_dist": nn_dist.size}
        bad = {k: v for k, v in lengths.items() if v != eval_len}
        if bad:
            raise SystemExit(
                f"[len-check] FAIL well={well}: eval_len={eval_len} mismatched arrays={bad}"
            )
        print(f"  well={well}: eval_len={eval_len} -- both arrays match OK")


def _regression_guard(wells: list[str]) -> float:
    """Recompute pooled RMSE of the spatial gateblend-theta=2 predictor from the cache."""
    sse, n = 0.0, 0

    for well in wells:
        h = D.load_horizontal(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            continue

        y_true = h["TVT"].to_numpy(dtype=float)[ez]
        anchor_pred = baseline.predict_carry_last(h)

        with np.load(_well_npz_path(well)) as npz:
            spatial_tvt = npz["spatial_tvt"].astype(float)
            prefix_rmse = float(npz["prefix_rmse"])

        delta = spatial_tvt - anchor_pred
        blend = anchor_pred + BLEND_W * delta
        pred = blend if prefix_rmse <= GATE_THETA else anchor_pred

        sse += float(np.sum((y_true - pred) ** 2))
        n += y_true.size

    return _pooled_rmse(sse, n)


def verify_cache(wells: list[str]) -> None:
    missing = [w for w in wells if not _well_npz_path(w).exists()]
    if missing:
        raise SystemExit(
            f"[verify] {len(missing)} well(s) have no cached npz (e.g. {missing[:5]}); "
            "run without --verify first to build the cache."
        )

    _check_reload_lengths(wells)

    print("\n## Regression guard: pooled RMSE from cache vs. experiment ledger\n")
    rmse = _regression_guard(wells)
    ok = abs(rmse - EXPECTED_GATEBLEND_RMSE) <= RMSE_TOLERANCE

    print(
        f"  spatial gateblend theta=2: {rmse:.4f}  (expected {EXPECTED_GATEBLEND_RMSE:.4f} "
        f"+/- {RMSE_TOLERANCE}) -- {'PASS' if ok else 'FAIL'}"
    )

    if not ok:
        raise SystemExit(
            "\nREGRESSION GUARD FAILED: cached spatial output does not match the "
            "experiment ledger's number within tolerance. Suspect a stride/k/method "
            "mismatch between this script's default calls and scripts/run_spatial_cv.py. "
            "Stopping."
        )

    print("\nREGRESSION GUARD PASSED")


def _print_manifest_summary() -> None:
    df = pd.read_csv(MANIFEST_PATH)
    print(f"\n## Manifest summary ({len(df)} wells, {MANIFEST_PATH})\n")
    cols = ["eval_len", "prefix_rmse", "nn_dist_median", "time_s"]
    print(df[cols].describe())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing cached npz files (default: skip)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Skip the build step; only run verification against an existing cache",
    )
    args = parser.parse_args()

    wells = D.list_wells("train")
    print(f"# train wells: {len(wells)}")
    print(f"# cache dir: {CACHE_DIR}")

    if not args.verify:
        print(f"# config: stride={STRIDE} k={K_NEIGHBORS} method={METHOD}")
        t0 = time.perf_counter()
        bank = SP.build_surface_bank(wells, split="train", stride=STRIDE)
        bank_secs = time.perf_counter() - t0
        sizes = {f: bank.depths[f].size for f in bank.formations}
        print(f"# bank built in {bank_secs:.1f}s; samples per formation: {sizes}", flush=True)
        build_cache(wells, bank, force=args.force)

    verify_cache(wells)
    _print_manifest_summary()


if __name__ == "__main__":
    main()
