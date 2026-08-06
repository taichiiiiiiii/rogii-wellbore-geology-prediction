"""Limited evaluation of the near-strike committee gate (K16 part 1) on hard wells.

Scope (see ``docs/playbooks/00_common.md`` / task instructions): a **≤25-well**
diagnostic, not a full-773-well CV. Compares the current spatial predictor
(``rogii.spatial.predict_spatial(..., method="idw")``, read straight from the
existing ``data/processed/spatial_cache`` -- not recomputed) against
``rogii.registration.committee.committee_correction`` applied on top of that
same cached baseline, for:

* the 4 community-flagged "hard wells" named in the K16 dissection
  (``86454a6f``, ``389ae58f``, ``fb03ae90``, ``4c2208f5``),
* the worst 10 train wells by per-well RMSE under the current best predictor
  (stack v2 all-in, ``outputs/stack_v2_oof.npz``'s ``oof_4`` config),
  excluding any of the 4 community wells above,
* the 6 train wells that a full leak-safe ``strike_alignment`` scan (see
  ``GATE_FIRE_WELLS`` below) found actually trip the gate (>=1 evaluation-zone
  segment with alignment < 0.35) -- **none of the 4 community-flagged wells or
  the worst-10 wells trip it even once**, so without this category the
  evaluation would silently exercise ``committee_correction``'s identity path
  on every well and never the substitution path, and
* 5 randomly sampled wells (seed 42) excluding the 20 wells above, as a
  regression/side-effect check (K16's own dissection found the gate helps on
  some wells and hurts on others -- see the module docstring in
  ``src/rogii/registration/committee.py``).

**Empirical note on gate rarity** (see :func:`main`'s printed diagnostic and
the task report): a full leak-safe scan of all 773 train wells found only 6
(0.8%) ever trip the |proj| < 0.35 gate, and those 6 fire on effectively every
segment of their own lateral (near-strike-parallel for their whole length),
not a handful of segments -- this is a bimodal, not a gradual, property.
K16's own 3-well Kaggle smoke log (kept for reference at
``/tmp/claude-.../scratchpad/connortynan_output/*.log``) likewise shows
"gate fires 0/16" on every well it printed, consistent with rarity rather
than an implementation bug here.

The formation-depth surface bank is built once (773 train wells, stride=10,
matching ``scripts/build_spatial_cache.py``'s config) -- this is the "full
SurfaceBank rebuild" the task instructions flag as a possible 35-minute/
several-GB cost; measured here at ~9-11s / ~360-900MB peak RSS (see the
docstring of :func:`main`), well under the 3.8GB local budget, so no Kaggle
offload is needed for this limited run.

Usage::

    uv run python scripts/eval_committee_hardwells.py
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import data as D  # noqa: E402
from rogii import spatial as SP  # noqa: E402
from rogii.registration import committee as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPATIAL_CACHE_DIR = REPO_ROOT / "data" / "processed" / "spatial_cache"
STACK_V2_OOF_PATH = REPO_ROOT / "outputs" / "stack_v2_oof.npz"

# Bank config matching scripts/build_spatial_cache.py (so the cached
# spatial_tvt baseline and the freshly built bank are apples-to-apples).
BANK_STRIDE = 10
K_NEIGHBORS = 10

COMMUNITY_HARD_WELLS: tuple[str, ...] = ("86454a6f", "389ae58f", "fb03ae90", "4c2208f5")

# Found via a one-off full-773-well ``strike_alignment`` scan (leak-safe,
# no predict_spatial calls, ~48s): the *only* train wells with any
# evaluation-zone segment below the 0.35 gate threshold. None of the 4
# community-flagged wells or the stack-v2 worst-10 wells are in this set
# (see module docstring) -- without deliberately including them, this
# evaluation would never exercise the gate's substitution path at all.
GATE_FIRE_WELLS: tuple[str, ...] = (
    "43e16325",
    "896d15b9",
    "3a7dd95d",
    "14ab73fb",
    "eba6605e",
    "42c538a1",
)

N_WORST = 10
N_RANDOM = 5
RANDOM_SEED = 42

GATE_THRESHOLD = 0.35
MIN_ROT_DEG = 60.0
# A couple of alternate thresholds probed for sensitivity (task: "threshold の
# 感度を1-2点試す"), bracketing the K16 default.
SENSITIVITY_THRESHOLDS: tuple[float, ...] = (0.20, 0.50)


def _pooled_rmse(sq_err_sum: float, n: int) -> float:
    return float(np.sqrt(sq_err_sum / n)) if n else float("nan")


def _well_rmse(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def _select_target_wells() -> tuple[list[str], dict[str, str]]:
    """Community hard + worst-10 (stack v2 OOF) + gate-fire-actual + random-5; (wells, category)."""
    category: dict[str, str] = dict.fromkeys(COMMUNITY_HARD_WELLS, "community")

    with np.load(STACK_V2_OOF_PATH, allow_pickle=True) as z:
        y_true = z["y_true"].astype(np.float64)
        pred = z["oof_4"].astype(np.float64)  # "all-in (levers 1+2+3)", the current-best config
        well_idx = z["well_idx"]
        used_wells = z["used_wells"]

    n_wells = len(used_wells)
    sse = np.zeros(n_wells)
    cnt = np.zeros(n_wells)
    np.add.at(sse, well_idx, (y_true - pred) ** 2)
    np.add.at(cnt, well_idx, 1)
    rmse = np.sqrt(sse / np.maximum(cnt, 1))
    order = np.argsort(-rmse)

    worst: list[str] = []
    for i in order:
        w = str(used_wells[i])
        if w in category:
            continue
        worst.append(w)
        if len(worst) == N_WORST:
            break
    for w in worst:
        category[w] = "worst10"

    for w in GATE_FIRE_WELLS:
        category.setdefault(w, "gate_fire_actual")

    rng = np.random.default_rng(RANDOM_SEED)
    pool = [str(w) for w in used_wells if w not in category]
    random_wells = list(rng.choice(pool, size=N_RANDOM, replace=False))
    for w in random_wells:
        category[w] = "random"

    wells = list(COMMUNITY_HARD_WELLS) + worst + list(GATE_FIRE_WELLS) + random_wells
    return wells, category


def _cached_spatial_tvt(well: str) -> np.ndarray:
    with np.load(SPATIAL_CACHE_DIR / f"{well}.npz") as npz:
        return npz["spatial_tvt"].astype(np.float64)


def _run_one_threshold(
    wells: list[str], bank: SP.SurfaceBank, threshold: float, min_rot_deg: float
) -> dict[str, dict[str, float]]:
    """Per-well {current_rmse, committee_rmse, fire_rate} for one gate threshold."""
    out: dict[str, dict[str, float]] = {}
    for well in wells:
        h = D.load_horizontal(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or "TVT" not in h.columns:
            continue
        y_true = h["TVT"].to_numpy(dtype=float)[ez]

        base = _cached_spatial_tvt(well)
        if base.shape[0] != y_true.shape[0]:
            continue

        alignment = C.strike_alignment(h, bank, exclude_well=well)
        fire_rate = float(np.mean(alignment < threshold)) if alignment.size else float("nan")

        corrected = C.committee_correction(
            h,
            bank,
            base,
            exclude_well=well,
            threshold=threshold,
            min_rot_deg=min_rot_deg,
            k=K_NEIGHBORS,
        )

        out[well] = {
            "current_rmse": _well_rmse(y_true, base),
            "committee_rmse": _well_rmse(y_true, corrected),
            "fire_rate": fire_rate,
            "n_rows": float(y_true.size),
        }
    return out


def _print_main_table(
    wells: list[str], category: dict[str, str], results: dict[str, dict[str, float]]
) -> None:
    print("\n## 主評価 (threshold=0.35, min_rot_deg=60.0)\n")
    print(
        f"{'well':10s} {'category':10s} {'current':>10s} {'committee':>10s} "
        f"{'delta':>9s} {'fire%':>7s} {'rows':>8s}"
    )
    print("-" * 68)
    for well in wells:
        if well not in results:
            print(f"{well:10s} (skipped: no eval zone / cache mismatch)")
            continue
        r = results[well]
        delta = r["committee_rmse"] - r["current_rmse"]
        print(
            f"{well:10s} {category.get(well, '?'):10s} {r['current_rmse']:10.3f} "
            f"{r['committee_rmse']:10.3f} {delta:+9.3f} {100 * r['fire_rate']:6.1f}% "
            f"{int(r['n_rows']):8d}"
        )


def _print_pooled_and_category_summary(
    wells: list[str], category: dict[str, str], results: dict[str, dict[str, float]]
) -> None:
    print(f"\n## 部分集合 pooled RMSE ({len(results)} wells; 全773件CVではない)\n")
    sse_cur = sse_comm = 0.0
    n_rows = 0
    for well in wells:
        if well not in results:
            continue
        r = results[well]
        n = r["n_rows"]
        sse_cur += (r["current_rmse"] ** 2) * n
        sse_comm += (r["committee_rmse"] ** 2) * n
        n_rows += int(n)
    print(f"current spatial (idw):    pooled RMSE = {_pooled_rmse(sse_cur, n_rows):.4f}")
    print(f"spatial_committee:        pooled RMSE = {_pooled_rmse(sse_comm, n_rows):.4f}")

    print("\n## カテゴリ別 helped/hurt (per-well RMSE)\n")
    for cat in ("community", "worst10", "gate_fire_actual", "random"):
        deltas = [
            results[w]["committee_rmse"] - results[w]["current_rmse"]
            for w in wells
            if category.get(w) == cat and w in results
        ]
        if not deltas:
            continue
        helped = sum(1 for d in deltas if d < -1e-9)
        hurt = sum(1 for d in deltas if d > 1e-9)
        flat = len(deltas) - helped - hurt
        print(
            f"  {cat:10s}: n={len(deltas):2d}  helped={helped}  hurt={hurt}  flat={flat}  "
            f"mean_delta={np.mean(deltas):+.3f}  median_delta={np.median(deltas):+.3f}"
        )


def _print_sensitivity(
    wells: list[str], bank: SP.SurfaceBank, base_results: dict[str, dict[str, float]]
) -> None:
    print("\n## threshold 感度\n")
    print(
        f"{'threshold':>10s} {'mean_fire%':>11s} {'mean_delta':>11s} {'n_hurt':>7s} "
        f"{'n_helped':>9s}"
    )
    print("-" * 55)

    base_deltas = [
        base_results[w]["committee_rmse"] - base_results[w]["current_rmse"]
        for w in wells
        if w in base_results
    ]
    base_fire = [base_results[w]["fire_rate"] for w in wells if w in base_results]
    n_hurt0 = sum(1 for d in base_deltas if d > 1e-9)
    n_helped0 = sum(1 for d in base_deltas if d < -1e-9)
    print(
        f"{GATE_THRESHOLD:10.2f} {100 * float(np.mean(base_fire)):10.1f}% "
        f"{float(np.mean(base_deltas)):11.3f} {n_hurt0:7d} {n_helped0:9d}   (主評価と同一)"
    )

    for thr in SENSITIVITY_THRESHOLDS:
        res = _run_one_threshold(wells, bank, threshold=thr, min_rot_deg=MIN_ROT_DEG)
        deltas = [res[w]["committee_rmse"] - res[w]["current_rmse"] for w in wells if w in res]
        fires = [res[w]["fire_rate"] for w in wells if w in res]
        n_hurt = sum(1 for d in deltas if d > 1e-9)
        n_helped = sum(1 for d in deltas if d < -1e-9)
        print(
            f"{thr:10.2f} {100 * float(np.mean(fires)):10.1f}% "
            f"{float(np.mean(deltas)):11.3f} {n_hurt:7d} {n_helped:9d}"
        )


def main() -> None:
    t_start = time.time()
    wells, category = _select_target_wells()
    print(f"# target wells: {len(wells)}")
    for cat in ("community", "worst10", "gate_fire_actual", "random"):
        print(f"  {cat}: {[w for w in wells if category[w] == cat]}")

    train_wells = D.list_wells("train")
    print(f"\n# building SurfaceBank: {len(train_wells)} train wells, stride={BANK_STRIDE}")
    t0 = time.time()
    bank = SP.build_surface_bank(train_wells, split="train", stride=BANK_STRIDE)
    bank_secs = time.time() - t0
    sizes = {f: bank.depths[f].size for f in bank.formations}
    print(f"# bank built in {bank_secs:.1f}s; samples per formation: {sizes}")

    results = _run_one_threshold(wells, bank, threshold=GATE_THRESHOLD, min_rot_deg=MIN_ROT_DEG)

    _print_main_table(wells, category, results)
    _print_pooled_and_category_summary(wells, category, results)
    _print_sensitivity(wells, bank, results)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\npeak RSS (whole-process high-water mark): {peak_mb:.1f} MB")
    print(f"total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
