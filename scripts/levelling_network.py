"""Step 2: mis-tie network levelling of the per-well datum constants.

Model: every well w carries an unknown datum constant c_w, and the levelled surfaces
S_w - c_w all sample one true structural surface.  Where two wells pass near each
other their levelled surfaces must agree, which gives one observation per pair:

    c_i - c_j  ~=  d_ij  =  robust median of (S_i(p) - S_j(nearest q))  over matches

Matching is nearest-point, computed both directions and averaged, so the surface
gradient across the ~300 ft well spacing enters with opposite sign in the two
directions and largely cancels instead of biasing the pair.

The network is solved by IRLS with a Huber loss (a handful of pairs are matched
across a fault or between wells targeting different zones and would otherwise drag
the whole component).  Gauge freedom is fixed by centring each connected component.

Usage:
    uv run python scripts/levelling_network.py [radius_ft] [pair_stride]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import CACHE, load  # noqa: E402

PAIR_RADIUS = 450.0
PAIR_STRIDE = 16
MIN_MATCHES = 20
HUBER = 6.0
IRLS_ITERS = 8
NN_K = 48
CHUNK = 20000


def pair_observations(w, *, radius: float, point_mask: np.ndarray) -> pd.DataFrame:
    """Nearest-point cross-well surface differences, aggregated per well pair.

    `point_mask` already carries the stride; do not re-stride here, or a per-well
    stride and a global stride will intersect on almost nothing.
    """
    from scipy.spatial import cKDTree

    well_of = np.repeat(np.arange(len(w.wells)), np.diff(w.starts))
    idx = np.where(point_mask)[0]
    xy = np.stack([w.X[idx], w.Y[idx]], axis=1)
    s = (w.TVT + w.Z)[idx]
    wo = well_of[idx]
    tree = cKDTree(xy)
    print(f"  pair points: {len(idx)}")

    src, dst, diff, dist_l = [], [], [], []
    k = min(NN_K, len(idx))
    for a in range(0, len(idx), CHUNK):
        b = min(a + CHUNK, len(idx))
        dd, ii = tree.query(xy[a:b], k=k, distance_upper_bound=radius, workers=-1)
        valid = np.isfinite(dd) & (ii < len(idx))
        other = np.zeros_like(valid)
        other[valid] = wo[ii[valid]] != wo[a:b, None].repeat(k, 1)[valid]
        good = valid & other
        if not good.any():
            continue
        rr, cc = np.where(good)
        j = ii[rr, cc]
        # keep only the nearest match per (source point, other well)
        key = wo[j].astype(np.int64) * len(idx) + (a + rr)
        order = np.lexsort((dd[rr, cc], key))
        rr, cc, j, key = rr[order], cc[order], j[order], key[order]
        first = np.ones(len(key), dtype=bool)
        first[1:] = key[1:] != key[:-1]
        rr, cc, j = rr[first], cc[first], j[first]
        src.append(wo[a + rr])
        dst.append(wo[j])
        diff.append(s[a + rr] - s[j])
        dist_l.append(dd[rr, cc])

    obs = pd.DataFrame({
        "i": np.concatenate(src), "j": np.concatenate(dst),
        "d": np.concatenate(diff), "dist": np.concatenate(dist_l),
    })
    print(f"  raw matched points: {len(obs)}")
    # symmetrise: fold (j,i) onto (i,j) with a sign flip so both directions average
    flip = obs["i"] > obs["j"]
    lo = np.where(flip, obs["j"], obs["i"])
    hi = np.where(flip, obs["i"], obs["j"])
    obs["d"] = np.where(flip, -obs["d"], obs["d"])
    obs["i"], obs["j"] = lo, hi
    g = obs.groupby(["i", "j"]).agg(
        d_ij=("d", "median"), n=("d", "size"),
        spread=("d", lambda x: float(np.percentile(np.abs(x - np.median(x)), 50))),
        dist=("dist", "median"),
    ).reset_index()
    return g[g["n"] >= MIN_MATCHES].reset_index(drop=True)


def solve_network(pairs: pd.DataFrame, n_wells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IRLS Huber solve for c, returning (c, comp_labels, final_residuals)."""
    i = pairs["i"].to_numpy()
    j = pairs["j"].to_numpy()
    d = pairs["d_ij"].to_numpy(float)
    # pair weight: more matches and a closer approach is a better tie
    w0 = np.sqrt(pairs["n"].to_numpy(float)) / (1.0 + pairs["dist"].to_numpy(float) / 300.0)
    w0 = w0 / w0.mean()

    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n_wells, n_wells))
    ncomp, comp = connected_components(adj, directed=False)

    m = len(i)
    rows = np.concatenate([np.arange(m), np.arange(m)])
    cols = np.concatenate([i, j])
    c = np.zeros(n_wells)
    weight = w0.copy()
    for _ in range(IRLS_ITERS):
        vals = np.concatenate([weight, -weight])
        A = coo_matrix((vals, (rows, cols)), shape=(m, n_wells)).tocsr()
        c = lsqr(A, d * weight, damp=1e-3, atol=1e-10, btol=1e-10, iter_lim=2000)[0]
        r = (c[i] - c[j]) - d
        scale = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-3)
        u = np.abs(r) / (HUBER * scale)
        weight = w0 * np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    for k in range(ncomp):
        sel = comp == k
        c[sel] -= c[sel].mean()
    return c, comp, (c[i] - c[j]) - d


def trend_levelling(w, point_mask: np.ndarray, *, degree: int = 2, iters: int = 8,
                    stride: int = 16) -> np.ndarray:
    """Alternative gauge: a smooth global trend plus per-well constants.

    The pair graph ties a well only to wells it physically passes; 170 of 773 wells
    have no such neighbour and stay unlevelled.  A global polynomial trend ties every
    well to every other, at the cost of absorbing genuine local structure into c_w
    wherever the polynomial cannot bend.  Prior work measured the true surface as
    degree-2 with R^2=0.9925, which is what makes this defensible here.
    """
    well_of = np.repeat(np.arange(len(w.wells)), np.diff(w.starts))
    sel = point_mask & (np.arange(len(w.X)) % stride == 0)
    x, y = w.X[sel], w.Y[sel]
    s = (w.TVT + w.Z)[sel]
    wo = well_of[sel]
    x0 = (x - x.mean()) / 1e4
    y0 = (y - y.mean()) / 1e4
    terms = [np.ones_like(x0)]
    for dg in range(1, degree + 1):
        for a in range(dg + 1):
            terms.append(x0 ** (dg - a) * y0 ** a)
    A = np.stack(terms, axis=1)

    c = np.zeros(len(w.wells))
    for _ in range(iters):
        beta = np.linalg.lstsq(A, s - c[wo], rcond=None)[0]
        resid = s - A @ beta
        upd = np.zeros(len(w.wells))
        order = np.argsort(wo, kind="stable")
        wo_s, r_s = wo[order], resid[order]
        bnd = np.flatnonzero(np.r_[True, wo_s[1:] != wo_s[:-1]])
        for st, en in zip(bnd, np.r_[bnd[1:], len(wo_s)]):
            upd[wo_s[st]] = np.median(r_s[st:en])
        c = upd - upd[np.unique(wo)].mean()
    return c


def main(argv: list[str]) -> int:
    radius = float(argv[1]) if len(argv) > 1 else PAIR_RADIUS
    stride = int(argv[2]) if len(argv) > 2 else PAIR_STRIDE
    w = load()
    print(f"radius={radius} stride={stride}")
    from levelling_common import donor_mask
    pairs = pair_observations(w, radius=radius,
                              point_mask=donor_mask(w, exclude=set(), stride=stride))
    print(f"  well pairs (n>={MIN_MATCHES}): {len(pairs)}")
    c, comp, res = solve_network(pairs, len(w.wells))

    sizes = pd.Series(comp).value_counts()
    print(f"\nconnected components: {len(sizes)}  "
          f"largest {sizes.iloc[0]}  singletons {int((sizes == 1).sum())}")
    print(f"  component sizes (top 6): {list(sizes.head(6))}")
    d = pairs["d_ij"].to_numpy()
    print(f"\nraw mis-tie d_ij (ft):      median|d| {np.median(np.abs(d)):7.2f}  "
          f"p90 {np.percentile(np.abs(d), 90):8.2f}  max {np.abs(d).max():9.2f}  "
          f"rms {np.sqrt(np.mean(d**2)):8.2f}")
    print(f"  residual after adjustment: median|r| {np.median(np.abs(res)):7.2f}  "
          f"p90 {np.percentile(np.abs(res), 90):8.2f}  max {np.abs(res).max():9.2f}  "
          f"rms {np.sqrt(np.mean(res**2)):8.2f}")
    print(f"\nfitted c_w (ft): sd {c.std():.2f}  "
          f"p5 {np.percentile(c, 5):.2f}  p95 {np.percentile(c, 95):.2f}  "
          f"range [{c.min():.1f}, {c.max():.1f}]")

    out = CACHE / f"network_r{int(radius)}_s{stride}.npz"
    np.savez(out, wells=w.wells, c=c, comp=comp,
             pairs_i=pairs["i"].to_numpy(), pairs_j=pairs["j"].to_numpy(),
             pairs_d=d, pairs_n=pairs["n"].to_numpy(),
             pairs_dist=pairs["dist"].to_numpy(), res=res)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
