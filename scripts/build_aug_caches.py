"""Build the R22-lite prefix-cut pseudo-well augmentation cache (issue: R22-lite).

**Hypothesis.** cross-well generalization is limited by the effective sample
count = 773 train wells, not by row count within a well. If a train well's
known region is cut short partway through, the newly-hidden tail (which is
still real, still has ground-truth ``TVT``) becomes a *pseudo* evaluation
zone for a *pseudo* well -- a second, earlier-anchored training example
carved out of the same well. Because it is built purely from information
available before the real anchor, it is exactly as leak-safe as the real
evaluation zone; it just adds more (well, anchor-position) training signal
without touching inference (a submission notebook never calls this script).

**150-well subset.** Same deterministic sample as
``scripts/run_r6b1_fill_gr_gaps_ab.py`` (:func:`deterministic_sample`,
byte-copied here) -- evenly spaced indices into ``sorted(list_wells("train"))``,
so this cache's well set is reproducible without an RNG and matches the one
already used elsewhere in the repo's R6-b analysis.

**Cut-point selection** (:func:`compute_cut_point`). For a well with known-
zone length ``known_len`` (= ``k``, the row index where the real evaluation
zone starts) and total row count ``orig_n`` (= ``n``), the pseudo evaluation
zone ``[cut_idx, known_len)`` is sized to ``min(orig_eval_len, 3000)`` rows
(``orig_eval_len = orig_n - known_len``), per the R22-lite task spec. The
well is **skipped entirely** (no pseudo well built) if the residual known
prefix ``[0, cut_idx)`` would fall below 800 rows or below 40% of the
original known-zone length -- not enough context left to calibrate the
trackers'/spatial predictor's own known-zone fits (``_calibrate_rate``'s
200-row tail, ``predict_spatial``'s 500-row prefix window, ...).

**IMPORTANT FINDING (measured, not assumed).** With these three spec
thresholds at their literal default values, this cut-point rule is
essentially infeasible on the real data: across ALL 773 train wells,
``known_len`` (the horizontal log's known-zone row count) is **median 1703,
max 2392** -- i.e. it NEVER reaches the 3000-row pseudo-eval-zone cap, while
``orig_eval_len`` is **median 4840** (p10 already 3263 > 3000). So for the
overwhelming majority of wells ``pseudo_eval_len`` saturates at 3000 >
``known_len``, which forces ``cut_idx`` negative -- always below the 800-row
floor. Measured: **0/150 of the deterministic sample, 1/773 of the full
train set** pass :func:`compute_cut_point` at the spec defaults. A
data-informed cap of roughly 500-800 rows recovers most of the population
(500 -> 146/150, 800 -> 103/150, see this script's own diagnostic run) but
choosing that cap is a design decision beyond this task's scope (R22-lite:
"実装+スモークのみ") -- flagged here for the next design pass rather than
silently changed. The three ``--max-pseudo-eval-len``/
``--min-residual-known-rows``/``--min-residual-known-frac`` CLI flags let a
caller override the thresholds (defaulting to the spec's 3000/800/0.4) to
smoke-test the build MECHANISM below a feasible cap without changing the
shipped default.

**Pseudo well construction** (:func:`build_pseudo_well`). ``h`` is sliced
down to exactly its known-zone rows ``[0, known_len)`` -- the real
downstream eval-zone rows ``[known_len, orig_n)`` are DROPPED, not masked,
so a pseudo well's rows never duplicate the real eval zone's rows in a
downstream training matrix. ``TVT_input[cut_idx:known_len]`` is then NaN-ed
out (``data.eval_mask`` on the result returns exactly ``[cut_idx,
known_len)``), and the ``TVT`` ground-truth column is dropped from the
returned frame entirely (defense in depth: no feature builder can read it
even by accident, on top of the fact that none of them do -- see the leak
boundary note below). The corresponding true-``TVT`` slice is captured
BEFORE the NaN-out and returned separately as ``y_true`` for the *caller* to
use as a training label; it is never written into any cached ``.npz`` (see
"On-disk layout").

**What gets computed per pseudo well.** The exact same tracker/spatial calls
``scripts/build_tracker_cache.py``/``scripts/build_spatial_cache.py`` make
for real wells, run on the pseudo well instead:

  - ``registration.particle.track_pf_multi(h_pseudo, tw)`` (module defaults)
  - ``registration.beam.track_beam_multi(h_pseudo, tw)`` (module defaults)
  - ``spatial.predict_spatial(h_pseudo, bank, exclude_well=well, k=10,
    method="idw")`` -- ``bank`` is built ONCE from every REAL (untruncated)
    train well's formation-depth columns (:func:`rogii.spatial.build_surface_bank`,
    same ``stride=10`` as ``build_spatial_cache.py``); ``exclude_well=well``
    keeps leave-self-out semantics identical to the real-well cache (the
    pseudo well is still "this well", just earlier-anchored -- its bank
    samples must still be excluded from its own KNN queries).

**Leak boundary.** Identical to the real-well caches, applied to the pseudo
well instead of the real one: ``track_pf_multi``/``track_beam_multi`` only
ever read ``MD``/``Z``/``GR``/the known-zone (non-NaN) prefix of
``TVT_input``; ``predict_spatial`` only ever reads ``X``/``Y``/``Z``/the
known-zone prefix of ``TVT_input`` from ``h_pseudo`` (never the target
well's own formation-depth columns). Because ``h_pseudo`` physically ends at
row ``known_len`` and has ``TVT_input[cut_idx:known_len]`` NaN-ed out (and
``TVT`` dropped outright), none of these calls can see anything from the
pseudo evaluation zone ``[cut_idx, known_len)`` -- structurally, not just by
convention. ``tests/test_aug_prefix_cut.py`` asserts this directly (the
eval-mask position, the NaN-out, the dropped ``TVT`` column, and that a real
feature builder -- ``rogii.features.build_features`` -- produces identical
output regardless of what the corrupted/real tail values would have been).

**True ``TVT`` is never cached.** Per this script's contract, only features
(tracker/spatial arrays) are persisted to disk; ground-truth targets are
reconstructed by the *caller* (``scripts/run_stack_v33_cv.py``) from the
real, untruncated well's own ``TVT`` column at ``[cut_idx, known_len)`` --
identical to how the real evaluation zone's target is always assembled
outside the leak-safe feature-building layer (see
``docs/playbooks/00_common.md``'s "リーク鉄則").

On-disk layout::

    data/processed/aug_cache_lite/{well}.npz
        pf_tvt, pf_std, beam_tvt, beam_margin, anchor   # tracker block
        spatial_tvt, prefix_rmse, nn_dist, nn_dist_median  # spatial block
        known_len, cut_idx, orig_n, pseudo_eval_len      # cut-plan metadata
    data/processed/aug_cache_lite/manifest.csv   # one row per BUILT well
    data/processed/aug_cache_lite/skipped.csv    # one row per SKIPPED well
        # (well, known_len, orig_n, orig_eval_len, reason)

All arrays are aligned to ``data.eval_mask(h_pseudo)`` order (length =
``pseudo_eval_len`` = ``known_len - cut_idx``), matching the tracker/spatial
contract used everywhere else in this repo.

Usage::

    # 8-well smoke (first 8 of the 150-well deterministic sample)
    uv run python scripts/build_aug_caches.py --n-wells 8

    # full 150-well build (NOT launched by this task -- ~7-8s/well projected,
    # see the module-level PROGRESS_EVERY prints for the live measurement)
    uv run python scripts/build_aug_caches.py
"""

from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import data as D  # noqa: E402
from rogii import spatial as SP  # noqa: E402
from rogii.registration.beam import track_beam_multi  # noqa: E402
from rogii.registration.particle import track_pf_multi  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = _REPO_ROOT / "data" / "processed" / "aug_cache_lite"
MANIFEST_PATH = CACHE_DIR / "manifest.csv"
SKIPPED_PATH = CACHE_DIR / "skipped.csv"

N_WELLS_SAMPLE = 150
PROGRESS_EVERY = 10

# Cut-point selection thresholds (module docstring's "Cut-point selection").
MAX_PSEUDO_EVAL_LEN = 3000
MIN_RESIDUAL_KNOWN_ROWS = 800
MIN_RESIDUAL_KNOWN_FRAC = 0.4

# Spatial bank/query config -- byte-identical to build_spatial_cache.py's
# module defaults, so this cache's spatial features are directly comparable
# to the real-well spatial cache built by that script.
SPATIAL_STRIDE = 10
SPATIAL_K = 10
SPATIAL_METHOD = "idw"

MANIFEST_FIELDS = [
    "well",
    "known_len",
    "cut_idx",
    "orig_n",
    "orig_eval_len",
    "pseudo_eval_len",
    "pf_time_s",
    "beam_time_s",
    "spatial_time_s",
    "pf_std_mean",
    "beam_margin_mean",
    "prefix_rmse",
    "nn_dist_median",
]
SKIPPED_FIELDS = ["well", "known_len", "orig_n", "orig_eval_len", "reason"]


def deterministic_sample(all_wells: list[str], n: int) -> list[str]:
    """Evenly spaced indices into the sorted well list (deterministic, no RNG).

    Byte-copied from ``scripts/run_r6b1_fill_gr_gaps_ab.py`` -- same
    function body, same output for the same inputs, so this cache's 150-well
    set matches the deterministic subset already used elsewhere in the
    repo's R6-b analysis (task requirement: "決定論サンプル150 well … と同一
    関数・同一150").
    """
    if n >= len(all_wells):
        return list(all_wells)
    idx = sorted({round(i * (len(all_wells) - 1) / (n - 1)) for i in range(n)})
    return [all_wells[i] for i in idx]


# --------------------------------------------------------------------------- #
# Cut-point selection + pseudo-well construction (unit-tested in
# tests/test_aug_prefix_cut.py).
# --------------------------------------------------------------------------- #


@dataclass
class CutPlan:
    """One well's chosen prefix-cut point."""

    well: str
    known_len: int  # k: original known-zone row count (real eval zone starts here)
    orig_n: int  # n: total row count of the original (untruncated) well
    orig_eval_len: int  # n - k: the real evaluation zone's length
    cut_idx: int  # c: pseudo eval zone is rows [c, k)
    pseudo_eval_len: int  # k - c


def known_zone_length(h: pd.DataFrame) -> int:
    """Row count of ``h``'s known (non-eval) zone prefix.

    Assumes the known zone is exactly the contiguous block of rows before
    the evaluation zone's NaN tail (per this repo's verified data invariant,
    0/773 violations -- ``docs/playbooks/00_common.md``). Returns ``0``
    defensively if that invariant does not hold for a given well (a
    non-contiguous "known" region), or if the well has no known rows at all
    -- either way the caller's :func:`compute_cut_point` then skips it via
    its ``known_len <= 0`` guard.
    """
    known_mask = h["TVT_input"].notna().to_numpy()
    if not known_mask.any():
        return 0
    known_len = int(np.flatnonzero(known_mask)[-1]) + 1
    if int(known_mask.sum()) != known_len:
        return 0  # non-contiguous known zone: pathological, skip defensively
    return known_len


def compute_cut_point(
    well: str,
    known_len: int,
    orig_n: int,
    max_pseudo_eval_len: int = MAX_PSEUDO_EVAL_LEN,
    min_residual_known_rows: int = MIN_RESIDUAL_KNOWN_ROWS,
    min_residual_known_frac: float = MIN_RESIDUAL_KNOWN_FRAC,
) -> CutPlan | None:
    """Choose a prefix-cut point for one well, or ``None`` if it can't be safely cut.

    The pseudo evaluation zone ``[cut_idx, known_len)`` is sized to
    ``min(orig_eval_len, max_pseudo_eval_len)`` rows. The well is skipped
    (``None``) if there's no real evaluation zone to size against, no known
    zone at all, or if the residual known prefix ``[0, cut_idx)`` would fall
    below ``min_residual_known_rows`` rows or below
    ``min_residual_known_frac`` of the original known-zone length -- not
    enough known-zone context left to calibrate the trackers'/spatial
    predictor's own known-zone fits.

    The three threshold parameters default to the R22-lite task spec's
    values (``MAX_PSEUDO_EVAL_LEN``/``MIN_RESIDUAL_KNOWN_ROWS``/
    ``MIN_RESIDUAL_KNOWN_FRAC``) and are only ever overridden by
    ``main``'s CLI flags -- see the module docstring's "IMPORTANT FINDING"
    section for why an override is needed to get any built wells at all out
    of this dataset's actual known/eval-zone row-count distribution.
    """
    orig_eval_len = orig_n - known_len
    if known_len <= 0 or orig_eval_len <= 0:
        return None

    pseudo_eval_len = min(orig_eval_len, max_pseudo_eval_len)
    cut_idx = known_len - pseudo_eval_len
    if cut_idx < min_residual_known_rows or cut_idx < min_residual_known_frac * known_len:
        return None

    return CutPlan(
        well=well,
        known_len=known_len,
        orig_n=orig_n,
        orig_eval_len=orig_eval_len,
        cut_idx=cut_idx,
        pseudo_eval_len=pseudo_eval_len,
    )


def build_pseudo_well(
    h: pd.DataFrame, known_len: int, cut_idx: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the pseudo well: rows ``[0, known_len)`` with ``TVT_input``
    NaN-ed out on ``[cut_idx, known_len)``.

    Returns ``(h_pseudo, y_true)``: ``h_pseudo`` has exactly ``known_len``
    rows (the well's real downstream eval zone is dropped, not masked), its
    ``TVT`` column removed entirely (defense in depth), and
    ``data.eval_mask(h_pseudo)`` true on exactly ``[cut_idx, known_len)``.
    ``y_true`` is the corresponding true-``TVT`` slice, captured before the
    NaN-out -- the caller's training label, never written back into
    ``h_pseudo`` or any cached array.
    """
    h_trunc = h.iloc[:known_len].reset_index(drop=True).copy()
    y_true = h_trunc["TVT"].to_numpy(dtype=float)[cut_idx:known_len].copy()

    tvt_input = h_trunc["TVT_input"].to_numpy(dtype=float, copy=True)
    tvt_input[cut_idx:known_len] = np.nan
    h_trunc["TVT_input"] = tvt_input

    h_pseudo = h_trunc.drop(columns=["TVT"]) if "TVT" in h_trunc.columns else h_trunc
    return h_pseudo, y_true


# --------------------------------------------------------------------------- #
# Per-well cache build (tracker + spatial on the pseudo well)
# --------------------------------------------------------------------------- #


def _npz_path(well: str) -> Path:
    return CACHE_DIR / f"{well}.npz"


def _nn_distance_per_row(h: pd.DataFrame, bank: SP.SurfaceBank, well: str) -> np.ndarray:
    """Per-row nearest non-self bank-sample distance for every pseudo eval-zone row.

    Byte-copied from ``scripts/build_spatial_cache.py``'s
    ``_nn_distance_for_well`` / ``_nn_distance_per_row`` (same leave-self-out
    KD-tree logic), applied to the pseudo well ``h`` instead of a real one.
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


def _build_well_cache(
    well: str,
    bank: SP.SurfaceBank,
    max_pseudo_eval_len: int = MAX_PSEUDO_EVAL_LEN,
    min_residual_known_rows: int = MIN_RESIDUAL_KNOWN_ROWS,
    min_residual_known_frac: float = MIN_RESIDUAL_KNOWN_FRAC,
) -> tuple[dict[str, str], str | None]:
    """Compute + persist one well's aug cache. Returns ``(manifest_row, skip_reason)``.

    ``skip_reason`` is ``None`` on success (``manifest_row`` populated), else
    a short label and ``manifest_row == {}``. Never raises: every failure
    mode this well can hit (unsafe cut point, no anchor, a tracker/spatial
    length mismatch) degrades to a skip, matching this repo's "予測器・
    トラッカーは絶対に例外を外に出さない" convention. The three threshold
    kwargs default to the task spec's values -- see ``main``'s
    ``--max-pseudo-eval-len`` etc. overrides and the module docstring's
    "IMPORTANT FINDING" section.
    """
    h = D.load_horizontal(well, "train")
    known_len = known_zone_length(h)
    orig_n = len(h)

    plan = compute_cut_point(
        well, known_len, orig_n, max_pseudo_eval_len, min_residual_known_rows,
        min_residual_known_frac,
    )
    if plan is None:
        return {}, "cut_infeasible"

    tw = D.load_typewell(well, "train")
    h_pseudo, y_true = build_pseudo_well(h, plan.known_len, plan.cut_idx)
    if y_true.size != plan.pseudo_eval_len:
        return {}, "size_mismatch"

    try:
        anchor = D.last_known_tvt(h_pseudo)
    except ValueError:
        return {}, "no_anchor"

    t0 = time.perf_counter()
    pf_result = track_pf_multi(h_pseudo, tw)
    pf_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    beam_result = track_beam_multi(h_pseudo, tw)
    beam_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    spatial_result = SP.predict_spatial(
        h_pseudo, bank, exclude_well=well, k=SPATIAL_K, method=SPATIAL_METHOD
    )
    nn_dist = _nn_distance_per_row(h_pseudo, bank, well)
    spatial_time_s = time.perf_counter() - t0

    n = plan.pseudo_eval_len
    sizes = {
        "pf_tvt": pf_result.tvt.size,
        "beam_tvt": beam_result.tvt.size,
        "spatial_tvt": spatial_result.tvt.size,
        "nn_dist": nn_dist.size,
    }
    if any(v != n for v in sizes.values()):
        return {}, f"tracker_len_mismatch({sizes})"

    finite_nn = nn_dist[np.isfinite(nn_dist)] if nn_dist.size else np.array([])
    nn_dist_median = float(np.median(finite_nn)) if finite_nn.size else float("nan")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _npz_path(well),
        pf_tvt=pf_result.tvt.astype(np.float32),
        pf_std=pf_result.std.astype(np.float32),
        beam_tvt=beam_result.tvt.astype(np.float32),
        beam_margin=beam_result.margin.astype(np.float32),
        anchor=np.float64(anchor),
        spatial_tvt=spatial_result.tvt.astype(np.float32),
        prefix_rmse=np.float64(spatial_result.prefix_rmse),
        nn_dist=nn_dist.astype(np.float32),
        nn_dist_median=np.float64(nn_dist_median),
        known_len=np.int64(plan.known_len),
        cut_idx=np.int64(plan.cut_idx),
        orig_n=np.int64(plan.orig_n),
        pseudo_eval_len=np.int64(plan.pseudo_eval_len),
    )

    row = {
        "well": well,
        "known_len": str(plan.known_len),
        "cut_idx": str(plan.cut_idx),
        "orig_n": str(plan.orig_n),
        "orig_eval_len": str(plan.orig_eval_len),
        "pseudo_eval_len": str(plan.pseudo_eval_len),
        "pf_time_s": f"{pf_time_s:.4f}",
        "beam_time_s": f"{beam_time_s:.4f}",
        "spatial_time_s": f"{spatial_time_s:.4f}",
        "pf_std_mean": f"{float(np.mean(pf_result.std)):.6f}",
        "beam_margin_mean": f"{float(np.mean(beam_result.margin)):.6f}",
        "prefix_rmse": f"{spatial_result.prefix_rmse:.6f}",
        "nn_dist_median": f"{nn_dist_median:.6f}",
    }
    return row, None


# --------------------------------------------------------------------------- #
# Build loop + reporting
# --------------------------------------------------------------------------- #


def _peak_rss_mb() -> float:
    """Current process's peak (high-water-mark) RSS in MB, Linux ``ru_maxrss`` is KB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_cache(
    wells: list[str],
    force: bool,
    max_pseudo_eval_len: int = MAX_PSEUDO_EVAL_LEN,
    min_residual_known_rows: int = MIN_RESIDUAL_KNOWN_ROWS,
    min_residual_known_frac: float = MIN_RESIDUAL_KNOWN_FRAC,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    print("# building surface bank from ALL real (untruncated) train wells ...", flush=True)
    t0 = time.perf_counter()
    all_train_wells = D.list_wells("train")
    bank = SP.build_surface_bank(all_train_wells, split="train", stride=SPATIAL_STRIDE)
    print(f"# bank built in {time.perf_counter() - t0:.1f}s", flush=True)

    manifest_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    n_built = n_skipped_existing = 0
    t_start = time.perf_counter()

    for i, well in enumerate(wells, start=1):
        npz_path = _npz_path(well)
        if npz_path.exists() and not force:
            n_skipped_existing += 1
        else:
            row, skip_reason = _build_well_cache(
                well, bank, max_pseudo_eval_len, min_residual_known_rows,
                min_residual_known_frac,
            )
            if skip_reason is None:
                manifest_rows.append(row)
                n_built += 1
            else:
                h = D.load_horizontal(well, "train")
                known_len = known_zone_length(h)
                skipped_rows.append(
                    {
                        "well": well,
                        "known_len": str(known_len),
                        "orig_n": str(len(h)),
                        "orig_eval_len": str(len(h) - known_len),
                        "reason": skip_reason,
                    }
                )

        if i % PROGRESS_EVERY == 0 or i == len(wells):
            elapsed = time.perf_counter() - t_start
            print(
                f"[{i}/{len(wells)}] built={n_built} skipped_existing={n_skipped_existing} "
                f"skipped_infeasible={len(skipped_rows)} elapsed={elapsed:.1f}s "
                f"peak_rss={_peak_rss_mb():.0f}MB",
                flush=True,
            )

    return manifest_rows, skipped_rows


def _print_cut_diagnostics(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("\n## cut-point diagnostics: no wells built this run")
        return
    df = pd.DataFrame(rows).astype(
        {
            "known_len": int,
            "cut_idx": int,
            "orig_n": int,
            "orig_eval_len": int,
            "pseudo_eval_len": int,
        }
    )
    print(f"\n## cut-point diagnostics ({len(df)} wells built this run)\n")
    print(f"{'well':>10}{'known_len(k)':>14}{'cut_idx(c)':>12}{'orig_n(n)':>11}"
          f"{'orig_eval_len':>15}{'pseudo_eval_len':>17}")
    for _, r in df.iterrows():
        print(
            f"{r['well']:>10}{r['known_len']:>14}{r['cut_idx']:>12}{r['orig_n']:>11}"
            f"{r['orig_eval_len']:>15}{r['pseudo_eval_len']:>17}"
        )
    print("\nsummary:")
    print(df[["known_len", "cut_idx", "orig_eval_len", "pseudo_eval_len"]].describe())


def _verify_anchor_consistency(wells: list[str]) -> None:
    """Re-derive each built well's cached anchor from the SOURCE (untruncated)
    well's ``TVT_input[cut_idx - 1]`` and check it matches the npz's cached
    value exactly -- the concrete "anchor整合" check the task asks for.
    """
    print("\n## anchor consistency check (cached anchor vs. source h[cut_idx-1])\n")
    n_checked = n_ok = 0
    for well in wells:
        npz_path = _npz_path(well)
        if not npz_path.exists():
            continue
        with np.load(npz_path) as npz:
            cached_anchor = float(npz["anchor"])
            cut_idx = int(npz["cut_idx"])
            pseudo_eval_len = int(npz["pseudo_eval_len"])
            known_len = int(npz["known_len"])
            pf_tvt_size = int(npz["pf_tvt"].shape[0])
        h = D.load_horizontal(well, "train")
        expected_anchor = float(h["TVT_input"].to_numpy()[cut_idx - 1])
        ok = (
            abs(cached_anchor - expected_anchor) < 1e-9
            and pseudo_eval_len == known_len - cut_idx
            and pf_tvt_size == pseudo_eval_len
        )
        n_checked += 1
        n_ok += int(ok)
        status = "OK" if ok else "FAIL"
        print(
            f"  well={well}: cached_anchor={cached_anchor:.4f} "
            f"expected={expected_anchor:.4f} pseudo_eval_len={pseudo_eval_len} "
            f"pf_tvt_size={pf_tvt_size} -- {status}"
        )
    print(f"\n  {n_ok}/{n_checked} wells: anchor + length consistency OK")
    if n_checked and n_ok != n_checked:
        raise SystemExit("anchor consistency check FAILED for at least one well")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wells",
        type=int,
        default=None,
        help="Restrict to the first N of the 150-well deterministic sample (smoke run).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing cached npz files (default: skip)"
    )
    parser.add_argument(
        "--max-pseudo-eval-len",
        type=int,
        default=MAX_PSEUDO_EVAL_LEN,
        help=(
            "Override MAX_PSEUDO_EVAL_LEN (default: the R22-lite task spec's 3000). "
            "NOTE: with the spec default, this dataset's known-zone rows top out at "
            "2392 (< 3000) across all 773 train wells, so the spec-literal cut-point "
            "rule skips essentially every well (measured: 0/150 of the deterministic "
            "sample, 1/773 overall) -- see the module docstring's IMPORTANT FINDING. "
            "This flag exists to smoke-test the build MECHANISM (tracker/spatial/npz "
            "pipeline) with a feasible cap; it does not change the shipped default."
        ),
    )
    parser.add_argument(
        "--min-residual-known-rows",
        type=int,
        default=MIN_RESIDUAL_KNOWN_ROWS,
        help="Override MIN_RESIDUAL_KNOWN_ROWS (default: the task spec's 800).",
    )
    parser.add_argument(
        "--min-residual-known-frac",
        type=float,
        default=MIN_RESIDUAL_KNOWN_FRAC,
        help="Override MIN_RESIDUAL_KNOWN_FRAC (default: the task spec's 0.4).",
    )
    args = parser.parse_args()

    t_start = time.time()

    all_wells = D.list_wells("train")
    sample = deterministic_sample(all_wells, N_WELLS_SAMPLE)
    wells = sample[: args.n_wells] if args.n_wells is not None else sample
    print(f"# train wells (total): {len(all_wells)}")
    print(f"# deterministic 150-well sample, using {len(wells)} (n_wells={args.n_wells})")
    print(f"# cache dir: {CACHE_DIR}")
    print(
        f"# cut-point config: max_pseudo_eval_len={args.max_pseudo_eval_len} "
        f"min_residual_known_rows={args.min_residual_known_rows} "
        f"min_residual_known_frac={args.min_residual_known_frac}"
        + (
            "  [OFF-SPEC OVERRIDE -- task default is 3000/800/0.4]"
            if (
                args.max_pseudo_eval_len != MAX_PSEUDO_EVAL_LEN
                or args.min_residual_known_rows != MIN_RESIDUAL_KNOWN_ROWS
                or args.min_residual_known_frac != MIN_RESIDUAL_KNOWN_FRAC
            )
            else ""
        )
    )

    built_rows, skipped_rows = build_cache(
        wells,
        force=args.force,
        max_pseudo_eval_len=args.max_pseudo_eval_len,
        min_residual_known_rows=args.min_residual_known_rows,
        min_residual_known_frac=args.min_residual_known_frac,
    )

    # Merge with any pre-existing manifest/skipped rows (resume-safe, mirrors
    # build_tracker_cache.py / build_spatial_cache.py's incremental-save style).
    existing_manifest = (
        {r["well"]: r for r in csv.DictReader(MANIFEST_PATH.open(newline=""))}
        if MANIFEST_PATH.exists()
        else {}
    )
    for row in built_rows:
        existing_manifest[row["well"]] = row
    all_manifest_rows = [existing_manifest[w] for w in sorted(existing_manifest)]
    _write_rows(MANIFEST_PATH, MANIFEST_FIELDS, all_manifest_rows)

    existing_skipped = (
        {r["well"]: r for r in csv.DictReader(SKIPPED_PATH.open(newline=""))}
        if SKIPPED_PATH.exists()
        else {}
    )
    for row in skipped_rows:
        existing_skipped[row["well"]] = row
    all_skipped_rows = [existing_skipped[w] for w in sorted(existing_skipped)]
    _write_rows(SKIPPED_PATH, SKIPPED_FIELDS, all_skipped_rows)

    print(
        f"\n# build done: built_this_run={len(built_rows)} "
        f"skipped_this_run={len(skipped_rows)} "
        f"manifest_total={len(all_manifest_rows)} skipped_total={len(all_skipped_rows)}"
    )

    _print_cut_diagnostics(built_rows)
    _verify_anchor_consistency(wells)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")
    print(f"this process's peak RSS: {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
