"""Leave-spatial-block-out validation of the clip rule.

Both Working Note Award write-ups rejected cross-well spatial features after a
well-grouped CV gain evaporated or reversed under a leave-spatial-block-out audit.
Random well-half splits scatter neighbouring wells across both halves, so a rule
whose parameters are really being tuned to one patch of the field still looks
transferable.  K-means blocks over well centroids put whole neighbourhoods on one
side of the split, which is the test that killed those features.

Reported next to the random-split number, never instead of it.

Usage:
    uv run python scripts/levelling_lso.py [variant] [levelling] [anchor] [n_blocks]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE, load  # noqa: E402
from levelling_gate_blend import pooled  # noqa: E402
from levelling_rules import CLIP_GRID, KAPPA_GRID, W_GRID  # noqa: E402


def fit_clip(pipe, spat, truth, caps) -> tuple[float, float]:
    best, arg = np.inf, (0.0, np.inf)
    for w in W_GRID:
        for L in caps:
            p = (1 - w) * pipe + w * (pipe + np.clip(spat - pipe, -L, L))
            r = pooled(p, truth)
            if r < best:
                best, arg = r, (float(w), float(L))
    return arg


def apply_clip(p, pipe, spat):
    w, L = p
    return (1 - w) * pipe + w * (pipe + np.clip(spat - pipe, -L, L))


def main(argv: list[str]) -> int:
    variant = argv[1] if len(argv) > 1 else "a"
    lev = argv[2] if len(argv) > 2 else "pair450"
    anchor = argv[3] if len(argv) > 3 else "tail"
    nblocks = int(argv[4]) if len(argv) > 4 else 6
    tag = f"{variant}_{lev}"
    rows = pd.read_parquet(CACHE / f"rows_{tag}.parquet")
    df = rows[["well", "pipe", "truth"]].copy()
    df["spat"] = rows[f"sp_{anchor}"].to_numpy()
    df = df[np.isfinite(df["spat"])].reset_index(drop=True)

    w = load()
    wells = sorted(df["well"].unique())
    cent = np.array([[w.X[w.slice(x)].mean(), w.Y[w.slice(x)].mean()] for x in wells])
    z = (cent - cent.mean(0)) / cent.std(0)
    _, lab = kmeans2(z, nblocks, minit="++", seed=0)
    rw = df["well"].map({x: i for i, x in enumerate(wells)}).to_numpy()
    pipe, spat, truth = (df[c].to_numpy(float) for c in ("pipe", "spat", "truth"))
    mad = float(np.median(np.abs(spat - pipe)))

    print(f"=== LSO  variant {variant} / levelling {lev} / anchor {anchor} ===")
    print(f"rows {len(df)}  wells {len(wells)}  blocks {nblocks} "
          f"(sizes {list(np.bincount(lab, minlength=nblocks))})")
    base_all = pooled(pipe, truth)
    print(f"  pipeline alone {base_all:.4f} ft\n")

    for name, caps, unit in (("clip (absolute ft)", CLIP_GRID, 1.0),
                             ("cliprel (x MAD)", KAPPA_GRID * mad, mad)):
        sse = base = 0.0
        n = 0
        detail = []
        for b in range(nblocks):
            dst = lab[rw] == b
            src = ~dst
            if dst.sum() == 0 or src.sum() == 0:
                continue
            p = fit_clip(pipe[src], spat[src], truth[src], caps)
            got = pooled(apply_clip(p, pipe[dst], spat[dst]), truth[dst])
            bas = pooled(pipe[dst], truth[dst])
            detail.append((b, int((lab == b).sum()), int(dst.sum()), p[0],
                           p[1] / unit, bas, got, got - bas))
            sse += got**2 * dst.sum()
            base += bas**2 * dst.sum()
            n += int(dst.sum())
        d = pd.DataFrame(detail, columns=["block", "wells", "rows", "w", "cap",
                                          "base", "blend", "delta"])
        print(f"  {name}")
        print("   " + d.round(4).to_string(index=False).replace("\n", "\n   "))
        print(f"   POOLED leave-spatial-block-out: {np.sqrt(base / n):.4f} -> "
              f"{np.sqrt(sse / n):.4f}  ({np.sqrt(sse / n) - np.sqrt(base / n):+.4f})\n")

    # does the benefit concentrate on wells with very close donors?
    ft = pd.read_csv(CACHE / f"feats_{tag}.csv")
    p_all = fit_clip(pipe, spat, truth, CLIP_GRID)
    bl = apply_clip(p_all, pipe, spat)
    per = df.assign(e0=(pipe - truth) ** 2, e1=(bl - truth) ** 2).groupby("well").agg(
        n=("truth", "size"), rmse_pipe=("e0", lambda x: float(np.sqrt(x.mean()))),
        rmse_blend=("e1", lambda x: float(np.sqrt(x.mean())))).reset_index()
    per = per.merge(ft[["well", "dnn_ev", "nwells_ev", "hb_rmse"]], on="well")
    per["delta"] = per["rmse_blend"] - per["rmse_pipe"]
    per["block"] = [lab[wells.index(x)] for x in per["well"]]
    per = per.sort_values("delta")
    per.to_csv(CACHE / f"lso_perwell_{tag}_{anchor}.csv", index=False)
    print(f"  in-sample clip pick w={p_all[0]:.2f} cap={p_all[1]:.2f} ft "
          f"({p_all[1] / mad:.2f} x MAD {mad:.2f})")
    print(f"  spearman(delta, dnn_ev)   = "
          f"{per['delta'].corr(per['dnn_ev'], method='spearman'):+.3f}")
    print(f"  spearman(delta, nwells_ev)= "
          f"{per['delta'].corr(per['nwells_ev'], method='spearman'):+.3f}")
    print(f"  wells helped {int((per['delta'] < 0).sum())}/{len(per)}; "
          f"median dnn_ev among helped {per.loc[per['delta'] < 0, 'dnn_ev'].median():.0f} ft, "
          f"among harmed {per.loc[per['delta'] >= 0, 'dnn_ev'].median():.0f} ft")
    print(f"\n{per.round(3).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
