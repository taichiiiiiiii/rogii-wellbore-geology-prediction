"""Cross-fit the write-up's physics post-process against our own pipeline output.

Our surfacepp is a crude version of a lever the award write-up reports as real-LB
confirmed and part of their best submission. Three refinements we skipped:

  * a Tukey-biweight IRLS degree-4 fit of U = TVT + Z, not ordinary least squares,
    so the fault rows that dominate this task's error stop dragging the fit;
  * a warm-up ramp — full trust in the raw track at the anchor, easing to the smooth
    fit over the next 500 ft — instead of one blend weight everywhere. Right after the
    anchor the tracker is accurate, and a constant weight damages exactly that region;
  * light Savitzky-Golay smoothing afterwards.

They report both cross-fit halves independently choosing the same hyperparameters
(beta 0.75, 500 ft warm-up, 51-point window) and 10.826 -> 10.522 on their proxy.

Judged here the only way that has meant anything on this task: cross-fit across wells,
never in-sample, since three attractive in-sample numbers have already been overturned.

A single hash-based 2-fold split is one draw among many, so it is repeated 200 times
over random well partitions. Since the hyperparameter "fit" here is really just
picking the best of a small fixed grid, the expensive per-(well, config) Tukey-IRLS
fit is computed once and cached; each of the 200 splits only sums the cached squared
errors over its well subset and re-derives the argmin, never rerunning physics_pp.

Usage:
    uv run python scripts/physics_pp_crossfit.py <harness_output_dir>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
N_SPLITS = 200


def tukey_polyfit(x: np.ndarray, y: np.ndarray, deg: int, iters: int = 4) -> np.ndarray:
    """Degree-`deg` fit reweighted by Tukey's biweight, evaluated back on x."""
    w = np.ones_like(y)
    fit = np.polyval(np.polyfit(x, y, deg), x)
    for _ in range(iters):
        r = y - fit
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if s <= 1e-9:
            break
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if w.sum() < deg + 2:
            break
        fit = np.polyval(np.polyfit(x, y, deg, w=np.sqrt(w)), x)
    return fit


def physics_pp(md: np.ndarray, z: np.ndarray, tvt: np.ndarray,
               beta: float, warmup: float, window: int, deg: int = 4) -> np.ndarray:
    order = np.argsort(md, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    md_s, z_s, tvt_s = md[order], z[order], tvt[order]

    u = tvt_s + z_s
    xs = (md_s - md_s.mean()) / max(md_s.std(), 1e-9)
    fit = tukey_polyfit(xs, u, deg)

    # Ramp: trust the raw track at the anchor, ease into the fit over `warmup` feet.
    d = md_s - md_s[0]
    lam = beta * np.clip(d / max(warmup, 1e-9), 0.0, 1.0)
    u_new = (1.0 - lam) * u + lam * fit

    if window >= 5 and len(u_new) > window:
        w = window if window % 2 == 1 else window - 1
        u_new = savgol_filter(u_new, w, 3, mode="interp")

    return (u_new - z_s)[inv]


def collect(d: Path) -> list[pd.DataFrame]:
    sub = pd.read_csv(d / "submission.csv")
    sub["id"] = sub["id"].astype(str)
    out = []
    for wid in sorted({i.rsplit("_", 1)[0] for i in sub["id"]}):
        hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
        hw["id"] = [f"{wid}_{i}" for i in hw.index]
        ev = hw[hw["TVT_input"].isna()]
        g = ev.merge(sub[["id", "tvt"]], on="id", how="left")
        if g["tvt"].isna().any() or len(g) < 60:
            continue
        out.append(g.assign(well=wid))
    return out


def _precompute_well_config_sse(
    wells: list[pd.DataFrame], grid: list[tuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-well squared error for every grid config, plus the baseline, computed once.

    The "fit" step in the repeated-split loop is just picking the best column of this
    table on a well subset, so the expensive Tukey-IRLS polyfit inside physics_pp is
    never rerun per split -- only summed and argmin'd, both cheap.
    """
    n_arr = np.array([len(g) for g in wells], dtype=float)
    base_sse = np.array([
        float(np.sum((g["tvt"].to_numpy(float) - g["TVT"].to_numpy(float)) ** 2)) for g in wells
    ])
    sse_table = np.empty((len(wells), len(grid)))
    for wi, g in enumerate(wells):
        md, z, tvt = g["MD"].to_numpy(float), g["Z"].to_numpy(float), g["tvt"].to_numpy(float)
        truth = g["TVT"].to_numpy(float)
        for ci, cfg in enumerate(grid):
            p = physics_pp(md, z, tvt, *cfg)
            sse_table[wi, ci] = float(np.sum((p - truth) ** 2))
    return sse_table, base_sse, n_arr


def _random_group(rng: np.random.Generator, n_wells: int) -> np.ndarray:
    """A random near-even partition of well indices into two groups (True=A, False=B)."""
    order = rng.permutation(n_wells)
    group_a = np.zeros(n_wells, dtype=bool)
    group_a[order[: n_wells // 2]] = True
    return group_a


def _split_delta(sse_table: np.ndarray, base_sse: np.ndarray, n_arr: np.ndarray,
                  group_a: np.ndarray) -> float:
    """Pooled out-of-sample RMSE delta (best grid cfg vs the raw pipeline) for one split."""
    sse = bse = n = 0.0
    for grp in (group_a, ~group_a):
        src, dst = grp, ~grp
        best_ci = int(np.argmin(sse_table[src].sum(axis=0)))
        sse += float(sse_table[dst, best_ci].sum())
        bse += float(base_sse[dst].sum())
        n += float(n_arr[dst].sum())
    return float(np.sqrt(sse / n) - np.sqrt(bse / n))


def repeated_splits(sse_table: np.ndarray, base_sse: np.ndarray, n_arr: np.ndarray,
                     n_wells: int, n_splits: int = N_SPLITS, seed: int = 0) -> np.ndarray:
    """Pooled out-of-sample delta for n_splits independent random 2-fold splits by well."""
    rng = np.random.default_rng(seed)
    return np.array([
        _split_delta(sse_table, base_sse, n_arr, _random_group(rng, n_wells))
        for _ in range(n_splits)
    ])


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    wells = collect(Path(argv[1]))
    print(f"{Path(argv[1]).name}: {len(wells)} wells")

    grid = [(b, wu, win) for b in (0.5, 0.75, 1.0)
            for wu in (250.0, 500.0, 1000.0) for win in (0, 51)]

    def score(cfg, subset) -> float:
        sse = n = 0
        for g in subset:
            p = physics_pp(g["MD"].to_numpy(float), g["Z"].to_numpy(float),
                           g["tvt"].to_numpy(float), *cfg)
            sse += float(np.sum((p - g["TVT"].to_numpy(float)) ** 2))
            n += len(g)
        return float(np.sqrt(sse / n))

    def base(subset) -> float:
        sse = n = 0
        for g in subset:
            sse += float(np.sum((g["tvt"].to_numpy(float) - g["TVT"].to_numpy(float)) ** 2))
            n += len(g)
        return float(np.sqrt(sse / n))

    print(f"  baseline {base(wells):.4f} ft")
    print("\n  in-sample sweep (beta, warmup, sg-window):")
    scored = sorted((score(c, wells), c) for c in grid)
    for s, c in scored[:5]:
        print(f"    {c}  {s:.4f}  ({s - base(wells):+.4f})")

    print("\n  cross-fit (single sha256 split, for comparison with the historical number):")
    half = {g['well'].iloc[0]: int(hashlib.sha256(g['well'].iloc[0].encode()).hexdigest(), 16) % 2
            for g in wells}
    a = [g for g in wells if half[g["well"].iloc[0]] == 0]
    b = [g for g in wells if half[g["well"].iloc[0]] == 1]
    print(f"    split: {len(a)} / {len(b)} wells")
    sse = bse = 0.0
    n = 0
    for src, dst, name in ((a, b, "A->B"), (b, a, "B->A")):
        cfg = min(grid, key=lambda c: score(c, src))
        got, bas = score(cfg, dst), base(dst)
        rows = sum(len(g) for g in dst)
        print(f"    {name}: {cfg}  {bas:.4f} -> {got:.4f}  ({got - bas:+.4f})")
        sse += got**2 * rows
        bse += bas**2 * rows
        n += rows
    print(f"    pooled out-of-sample: {np.sqrt(bse/n):.4f} -> {np.sqrt(sse/n):.4f} "
          f"({np.sqrt(sse/n) - np.sqrt(bse/n):+.4f})")

    print(f"\n  repeated random 2-fold splits by well (N={N_SPLITS}, seed=0):")
    sse_table, base_sse, n_arr = _precompute_well_config_sse(wells, grid)
    deltas = repeated_splits(sse_table, base_sse, n_arr, len(wells))
    print(f"    median {np.median(deltas):+.4f}  sd {deltas.std(ddof=1):.4f}  "
          f"p5 {np.percentile(deltas, 5):+.4f}  p95 {np.percentile(deltas, 95):+.4f}  "
          f"helps(delta<0) {100 * float((deltas < 0).mean()):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
