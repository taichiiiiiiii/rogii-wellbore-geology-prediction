"""Build a per-well tracker-feature cache for fast stack-model iteration.

Computes ``track_pf_multi(h, tw)`` (particle filter, default config) and
``track_beam_multi(h, tw)`` (full-grid Viterbi DP, default config) for every
train well and caches the results to disk so downstream stack-model
experiments (GBDT/blend combining PF + beam + other features) don't have to
re-run the ~1h combined tracker cost on every iteration.

Layout::

    data/processed/tracker_cache/{well}.npz   # pf_tvt, pf_std, beam_tvt, beam_margin, anchor
    data/processed/tracker_cache/manifest.csv # well, eval_len, pf_time_s, beam_time_s,
                                               # pf_std_mean, beam_margin_mean

All four per-row arrays are aligned to ``data.eval_mask(h)`` order (length =
number of evaluation-zone rows for that well), matching the tracker contract
used everywhere else in this repo (see ``docs/playbooks/00_common.md``).

Usage::

    uv run python scripts/build_tracker_cache.py           # build (resume: skip cached) + verify
    uv run python scripts/build_tracker_cache.py --force   # rebuild every well from scratch
    uv run python scripts/build_tracker_cache.py --verify  # skip build, verify existing cache only

The verification step is a regression guard, not a formality: it recomputes
the pooled RMSE of the PF blend-w0.7 and beam blend-w0.7 predictors *from the
cached arrays* and checks they match the experiment ledger's measured values
(``analysis/experiment_ledger.md``: PF blend w0.7 = 11.0621, beam blend w0.7 =
15.7232, both from the full-773-well runs in ``scripts/run_pf_cv.py`` /
``scripts/run_beam_cv.py``). A mismatch beyond the tolerance means the cache
was built with a different seed/config than the ledger's numbers and must not
be trusted for downstream experiments.
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
from rogii.registration.beam import track_beam_multi  # noqa: E402
from rogii.registration.particle import track_pf_multi  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "tracker_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.csv"

MANIFEST_FIELDS = [
    "well",
    "eval_len",
    "pf_time_s",
    "beam_time_s",
    "pf_std_mean",
    "beam_margin_mean",
]

PROGRESS_EVERY = 50

# Anchor-blend weights that reproduce the experiment ledger's current-best
# numbers (see scripts/run_pf_cv.py / scripts/run_beam_cv.py's blend_w0.7).
PF_BLEND_W = 0.7
BEAM_BLEND_W = 0.7
EXPECTED_PF_BLEND_RMSE = 11.0621
EXPECTED_BEAM_BLEND_RMSE = 15.7232
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
    """Reconstruct a manifest row from an already-cached npz (timings unknown)."""
    with np.load(_well_npz_path(well)) as npz:
        pf_std = npz["pf_std"]
        beam_margin = npz["beam_margin"]
        eval_len = int(pf_std.size)
    return {
        "well": well,
        "eval_len": str(eval_len),
        "pf_time_s": "",
        "beam_time_s": "",
        "pf_std_mean": f"{float(np.mean(pf_std)):.6f}" if eval_len else "",
        "beam_margin_mean": f"{float(np.mean(beam_margin)):.6f}" if eval_len else "",
    }


def _build_well_cache(well: str) -> dict[str, str]:
    """Compute and persist one well's tracker cache; returns its manifest row."""
    h = D.load_horizontal(well, "train")
    tw = D.load_typewell(well, "train")
    anchor = D.last_known_tvt(h)

    t0 = time.perf_counter()
    pf_result = track_pf_multi(h, tw)
    pf_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    beam_result = track_beam_multi(h, tw)
    beam_time_s = time.perf_counter() - t0

    eval_len = int(pf_result.tvt.size)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _well_npz_path(well),
        pf_tvt=pf_result.tvt.astype(np.float32),
        pf_std=pf_result.std.astype(np.float32),
        beam_tvt=beam_result.tvt.astype(np.float32),
        beam_margin=beam_result.margin.astype(np.float32),
        anchor=np.float64(anchor),
    )

    return {
        "well": well,
        "eval_len": str(eval_len),
        "pf_time_s": f"{pf_time_s:.4f}",
        "beam_time_s": f"{beam_time_s:.4f}",
        "pf_std_mean": f"{float(np.mean(pf_result.std)):.6f}" if eval_len else "",
        "beam_margin_mean": f"{float(np.mean(beam_result.margin)):.6f}" if eval_len else "",
    }


def build_cache(wells: list[str], force: bool) -> None:
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
            manifest_rows[well] = _build_well_cache(well)
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
            pf_tvt = npz["pf_tvt"]
            pf_std = npz["pf_std"]
            beam_tvt = npz["beam_tvt"]
            beam_margin = npz["beam_margin"]
        lengths = {
            "pf_tvt": pf_tvt.size,
            "pf_std": pf_std.size,
            "beam_tvt": beam_tvt.size,
            "beam_margin": beam_margin.size,
        }
        bad = {k: v for k, v in lengths.items() if v != eval_len}
        if bad:
            raise SystemExit(
                f"[len-check] FAIL well={well}: eval_len={eval_len} mismatched arrays={bad}"
            )
        print(f"  well={well}: eval_len={eval_len} -- all 4 arrays match OK")


def _regression_guard(wells: list[str]) -> tuple[float, float]:
    """Recompute pooled RMSE of the PF/beam blend-w0.7 predictors from the cache."""
    pf_sse, pf_n = 0.0, 0
    beam_sse, beam_n = 0.0, 0

    for well in wells:
        h = D.load_horizontal(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            continue

        y_true = h["TVT"].to_numpy(dtype=float)[ez]
        anchor_pred = baseline.predict_carry_last(h)

        with np.load(_well_npz_path(well)) as npz:
            pf_tvt = npz["pf_tvt"].astype(float)
            beam_tvt = npz["beam_tvt"].astype(float)

        pf_pred = anchor_pred + PF_BLEND_W * (pf_tvt - anchor_pred)
        beam_pred = anchor_pred + BEAM_BLEND_W * (beam_tvt - anchor_pred)

        pf_sse += float(np.sum((y_true - pf_pred) ** 2))
        pf_n += y_true.size
        beam_sse += float(np.sum((y_true - beam_pred) ** 2))
        beam_n += y_true.size

    return _pooled_rmse(pf_sse, pf_n), _pooled_rmse(beam_sse, beam_n)


def verify_cache(wells: list[str]) -> None:
    missing = [w for w in wells if not _well_npz_path(w).exists()]
    if missing:
        raise SystemExit(
            f"[verify] {len(missing)} well(s) have no cached npz (e.g. {missing[:5]}); "
            "run without --verify first to build the cache."
        )

    _check_reload_lengths(wells)

    print("\n## Regression guard: pooled RMSE from cache vs. experiment ledger\n")
    pf_rmse, beam_rmse = _regression_guard(wells)
    pf_ok = abs(pf_rmse - EXPECTED_PF_BLEND_RMSE) <= RMSE_TOLERANCE
    beam_ok = abs(beam_rmse - EXPECTED_BEAM_BLEND_RMSE) <= RMSE_TOLERANCE

    print(
        f"  PF blend w0.7:   {pf_rmse:.4f}  (expected {EXPECTED_PF_BLEND_RMSE:.4f} "
        f"+/- {RMSE_TOLERANCE}) -- {'PASS' if pf_ok else 'FAIL'}"
    )
    print(
        f"  beam blend w0.7: {beam_rmse:.4f}  (expected {EXPECTED_BEAM_BLEND_RMSE:.4f} "
        f"+/- {RMSE_TOLERANCE}) -- {'PASS' if beam_ok else 'FAIL'}"
    )

    if not (pf_ok and beam_ok):
        raise SystemExit(
            "\nREGRESSION GUARD FAILED: cached tracker output does not match the "
            "experiment ledger's numbers within tolerance. Suspect a seed/config "
            "mismatch between this script's default calls and "
            "scripts/run_pf_cv.py / scripts/run_beam_cv.py. Stopping."
        )

    print("\nREGRESSION GUARD PASSED")


def _print_manifest_summary() -> None:
    df = pd.read_csv(MANIFEST_PATH)
    print(f"\n## Manifest summary ({len(df)} wells, {MANIFEST_PATH})\n")
    cols = ["eval_len", "pf_time_s", "beam_time_s", "pf_std_mean", "beam_margin_mean"]
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
        build_cache(wells, force=args.force)

    verify_cache(wells)
    _print_manifest_summary()


if __name__ == "__main__":
    main()
