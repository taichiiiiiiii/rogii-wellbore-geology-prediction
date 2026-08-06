"""Build a per-well cache of R17 spatial-fit diagnostics (issue: R17).

Companion to ``scripts/build_spatial_cache.py``: that script caches
``rogii.spatial.predict_spatial`` (leave-self-out IDW, ``k=10``) output,
which stays completely untouched by R17. This script instead caches
``rogii.spatial.predict_spatial_v2`` output for one of two named configs --
a local-plane fit with the untouched ``k=10`` (config ``b``), or a
quadratic-surface fit with a wider, more tightly-distance-weighted
neighborhood (config ``c``) -- and persists **only the four new V4
diagnostic columns** (``resid_std``, ``n_donors``, ``grad_x``, ``grad_y``),
since the existing ``spatial_d`` / ``spatial_prefix_rmse`` /
``spatial_nn_dist_median`` / ``spatial_gated_d`` features (sourced from the
original IDW cache) are kept as-is in the R31 well matrix -- R17's designs
(b)/(c) *add* four columns rather than replace the existing four (the
original ``run_spatial_cv.py`` finding that a raw local-plane fit has a
worse heavy tail than IDW, pooled 26.8 vs 23.0 on a 20-well subset, is the
reason: replacing the proven IDW-based ``spatial_d`` outright is a bigger,
unvalidated bet than adding diagnostic columns alongside it).

**Donor fold boundary (as-is, not changed by R17).** Like
``build_spatial_cache.py``, the bank is built once from *every* train well
(``rogii.spatial.build_surface_bank(D.list_wells("train"))``); at query time
only the target well's own samples are excluded (``exclude_well``), not its
CV-fold siblings. This is the same (small, already-diagnosed) CV-vs-fold-
train-only mismatch flagged by the 2026-07-07 field-CV v2 audit and
empirically absolved by the sub-v2 LB submission (calibration point #3: LB
9.022 fell inside the "transfers" band, not the "field-isolated" 10.69
pessimistic band) -- see this repo's R17 task report for the full argument.
R17 keeps the identical donor policy so its new diagnostic columns are
apples-to-apples with the existing ``spatial_d`` cache.

Layout::

    data/processed/spatial_cache_v2_{config}/{well}.npz
        # resid_std, n_donors, grad_x, grad_y (eval_mask-aligned float32)
    data/processed/spatial_cache_v2_{config}/manifest.csv
        # well, eval_len, prefix_rmse, mean_resid_std, mean_n_donors, time_s

Usage::

    uv run python scripts/build_spatial_cache_v2.py --config b --n-wells 30   # smoke
    uv run python scripts/build_spatial_cache_v2.py --config b                # full 773
    uv run python scripts/build_spatial_cache_v2.py --config c
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

from rogii import data as D  # noqa: E402
from rogii import spatial as SP  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_MANIFEST_PATH = (
    _REPO_ROOT / "data" / "processed" / "tracker_cache" / "manifest.csv"
)

# name -> (method, k, weight_power). See module docstring for rationale.
CONFIGS: dict[str, tuple[str, int, float]] = {
    "b": ("plane", 10, 1.0),
    "c": ("quadratic", 12, 2.0),
}

STRIDE = 10  # matches build_spatial_cache.py's bank-build stride

MANIFEST_FIELDS = [
    "well",
    "eval_len",
    "prefix_rmse",
    "mean_resid_std",
    "mean_n_donors",
    "time_s",
]

PROGRESS_EVERY = 50
N_STRATA = 10


def _cache_dir(config: str) -> Path:
    return _REPO_ROOT / "data" / "processed" / f"spatial_cache_v2_{config}"


def _well_npz_path(config: str, well: str) -> Path:
    return _cache_dir(config) / f"{well}.npz"


def _manifest_path(config: str) -> Path:
    return _cache_dir(config) / "manifest.csv"


def _stratified_sample(all_wells: list[str], n: int, seed: int) -> list[str]:
    """Deterministic stratified subset, byte-copied recipe from run_stack_v26/29_cv.py."""
    import random

    manifest = pd.read_csv(TRACKER_MANIFEST_PATH)
    manifest = manifest[manifest["well"].isin(all_wells)].copy()
    manifest["decile"] = pd.qcut(manifest["pf_std_mean"], N_STRATA, labels=False, duplicates="drop")

    rng = random.Random(seed)
    per_stratum = max(1, n // manifest["decile"].nunique())
    picked: list[str] = []
    for _, group in manifest.groupby("decile"):
        wells_in_group = sorted(group["well"].tolist())
        rng.shuffle(wells_in_group)
        picked.extend(wells_in_group[:per_stratum])

    rng.shuffle(picked)
    return sorted(picked[:n]) if len(picked) >= n else sorted(picked)


def _load_manifest_rows(config: str) -> dict[str, dict[str, str]]:
    path = _manifest_path(config)
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["well"]: row for row in csv.DictReader(f)}


def _write_manifest(config: str, rows: dict[str, dict[str, str]]) -> None:
    _cache_dir(config).mkdir(parents=True, exist_ok=True)
    with _manifest_path(config).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for well in sorted(rows):
            writer.writerow(rows[well])


def _build_well_cache(
    config: str, well: str, bank: SP.SurfaceBank, method: str, k: int, weight_power: float
) -> dict[str, str]:
    h = D.load_horizontal(well, "train")

    t0 = time.perf_counter()
    result = SP.predict_spatial_v2(
        h, bank, exclude_well=well, k=k, method=method, weight_power=weight_power
    )
    time_s = time.perf_counter() - t0

    _cache_dir(config).mkdir(parents=True, exist_ok=True)
    np.savez(
        _well_npz_path(config, well),
        resid_std=result.resid_std.astype(np.float32),
        n_donors=result.n_donors.astype(np.float32),
        grad_x=result.grad_x.astype(np.float32),
        grad_y=result.grad_y.astype(np.float32),
    )

    finite_resid = result.resid_std[np.isfinite(result.resid_std)]
    mean_resid = float(np.mean(finite_resid)) if finite_resid.size else float("nan")
    return {
        "well": well,
        "eval_len": str(result.tvt.size),
        "prefix_rmse": f"{result.prefix_rmse:.6f}",
        "mean_resid_std": f"{mean_resid:.6f}",
        "mean_n_donors": f"{float(np.mean(result.n_donors)):.6f}",
        "time_s": f"{time_s:.4f}",
    }


def build_cache(config: str, wells: list[str], bank: SP.SurfaceBank, force: bool) -> None:
    method, k, weight_power = CONFIGS[config]
    manifest_rows = _load_manifest_rows(config)
    n_built = 0
    n_skipped = 0
    t_start = time.perf_counter()

    for i, well in enumerate(wells, start=1):
        npz_path = _well_npz_path(config, well)
        if npz_path.exists() and not force and well in manifest_rows:
            n_skipped += 1
        else:
            manifest_rows[well] = _build_well_cache(config, well, bank, method, k, weight_power)
            n_built += 1

        if i % PROGRESS_EVERY == 0 or i == len(wells):
            elapsed = time.perf_counter() - t_start
            print(
                f"[{config}][{i}/{len(wells)}] built={n_built} skipped={n_skipped} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            _write_manifest(config, manifest_rows)

    _write_manifest(config, manifest_rows)
    print(f"# [{config}] build done: built={n_built} skipped={n_skipped} total={len(wells)}")


def _check_reload_lengths(config: str, wells: list[str], n_check: int = 5) -> None:
    print(f"\n## [{config}] Reload length check ({min(n_check, len(wells))} wells)\n")
    for well in wells[:n_check]:
        h = D.load_horizontal(well, "train")
        eval_len = int(D.eval_mask(h).sum())
        with np.load(_well_npz_path(config, well)) as npz:
            lengths = {k: npz[k].size for k in ("resid_std", "n_donors", "grad_x", "grad_y")}
        bad = {k: v for k, v in lengths.items() if v != eval_len}
        if bad:
            raise SystemExit(f"[len-check] FAIL well={well}: eval_len={eval_len} mismatched={bad}")
        print(f"  well={well}: eval_len={eval_len} -- all 4 arrays match OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument(
        "--n-wells", type=int, default=None, help="Health-check subset size (stratified sample)."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached npz files")
    args = parser.parse_args()

    all_wells = D.list_wells("train")
    if args.n_wells is not None:
        wells = _stratified_sample(all_wells, args.n_wells, seed=42)
        print(f"health-check subset: {len(wells)}/{len(all_wells)} wells (stratified)")
    else:
        wells = all_wells
        print(f"train wells: {len(wells)}")

    method, k, weight_power = CONFIGS[args.config]
    print(
        f"# config {args.config}: method={method} k={k} weight_power={weight_power} "
        f"stride={STRIDE}"
    )

    t0 = time.perf_counter()
    # Bank always built from ALL 773 train wells regardless of --n-wells (matches
    # build_spatial_cache.py's donor policy exactly -- see module docstring).
    bank = SP.build_surface_bank(all_wells, split="train", stride=STRIDE)
    bank_secs = time.perf_counter() - t0
    sizes = {f: bank.depths[f].size for f in bank.formations}
    print(f"# bank built in {bank_secs:.1f}s; samples per formation: {sizes}", flush=True)

    build_cache(args.config, wells, bank, force=args.force)
    _check_reload_lengths(args.config, wells)

    if args.n_wells is None:
        df = pd.read_csv(_manifest_path(args.config))
        print(f"\n## [{args.config}] Manifest summary ({len(df)} wells)\n")
        cols = ["eval_len", "prefix_rmse", "mean_resid_std", "mean_n_donors", "time_s"]
        print(df[cols].describe())


if __name__ == "__main__":
    main()
