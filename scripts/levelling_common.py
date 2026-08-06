"""Shared loading / IDW machinery for the mis-tie levelling experiments.

Two deliberate departures from scripts/anchored_spatial_surface.py:

* the donor tree never contains the query well (we rebuild it per variant with the
  held-out wells stripped), so a k-nearest query is legal here where it was not
  there.  Radius-only `query_ball_point` over 200k eval rows against 500k donors
  returns ~1e8 indices at once and will not fit in 3.8 GB, so we take the K nearest
  and additionally drop anything past RADIUS.  K=256 at 8 ft donor spacing spans
  several neighbouring wells, which is the averaging the radius form was buying.
* queries are chunked, so peak memory is bounded by CHUNK * K.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CACHE = Path("/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/levelling")
DONOR_STRIDE = 8
RADIUS = 1500.0
POWER = 2.0
EPS = 1.0
KNN = 256
CHUNK = 4096


@dataclass
class Wells:
    wells: np.ndarray
    starts: np.ndarray
    row_idx: np.ndarray
    MD: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    TVT: np.ndarray
    known: np.ndarray

    def index(self, wid: str) -> int:
        return int(self._lookup[wid])

    def slice(self, wid: str) -> slice:
        i = self.index(wid)
        return slice(int(self.starts[i]), int(self.starts[i + 1]))

    def surface(self) -> np.ndarray:
        return self.TVT + self.Z


def load() -> Wells:
    z = np.load(CACHE / "wells.npz", allow_pickle=False)
    w = Wells(**{k: z[k] for k in
                 ["wells", "starts", "row_idx", "MD", "X", "Y", "Z", "TVT", "known"]})
    w.wells = w.wells.astype(str)
    w._lookup = {name: i for i, name in enumerate(w.wells)}  # type: ignore[attr-defined]
    return w


def donor_mask(w: Wells, *, exclude: set[str], known_only: set[str] | None = None,
               stride: int = DONOR_STRIDE) -> np.ndarray:
    """Row mask selecting donor points.

    `exclude` wells contribute nothing.  `known_only` wells contribute their known
    zone only -- that is exactly what a mounted-but-unlabelled test well offers at
    real inference time.
    """
    m = np.zeros(len(w.X), dtype=bool)
    ko = known_only or set()
    for i, wid in enumerate(w.wells):
        if wid in exclude:
            continue
        a, b = int(w.starts[i]), int(w.starts[i + 1])
        sel = np.zeros(b - a, dtype=bool)
        sel[::stride] = True
        if wid in ko:
            sel &= w.known[a:b]
        m[a:b] = sel
    return m


class Field:
    """IDW interpolator over a donor point cloud."""

    def __init__(self, xy: np.ndarray, s: np.ndarray, well_of: np.ndarray | None = None):
        self.xy = np.ascontiguousarray(xy, dtype=np.float64)
        self.s = np.ascontiguousarray(s, dtype=np.float64)
        self.well_of = well_of
        self.tree = cKDTree(self.xy)

    def __len__(self) -> int:
        return len(self.s)

    def query(self, xy_q: np.ndarray, *, k: int = KNN, radius: float = RADIUS,
              offsets: np.ndarray | None = None,
              exclude_well: int | None = None,
              k_eff: int | None = None) -> dict[str, np.ndarray]:
        """IDW estimate plus the geometry features the gate needs.

        `offsets` (per donor well, indexed by well_of) is subtracted from the donor
        surface before averaging -- that is how a levelled field is evaluated.
        `exclude_well` drops one well's donors after the k-nearest query, which lets a
        single shared tree serve every query well.  `k_eff` then keeps only that many
        nearest survivors, so a field that contains the query well gives the identical
        answer to a field that never contained it -- without rebuilding the tree.
        """
        xy_q = np.ascontiguousarray(xy_q, dtype=np.float64)
        n = len(xy_q)
        out = np.full(n, np.nan)
        nw = np.zeros(n, dtype=np.int32)
        dnn = np.full(n, np.inf)
        kk = min(k, len(self.s))
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            dist, idx = self.tree.query(xy_q[a:b], k=kk, workers=-1)
            if dist.ndim == 1:
                dist, idx = dist[:, None], idx[:, None]
            ok = np.isfinite(dist) & (dist <= radius)
            idx_safe = np.where(ok, idx, 0)
            if exclude_well is not None and self.well_of is not None:
                ok &= self.well_of[idx_safe] != exclude_well
            if not ok.any():
                continue
            if k_eff is not None and k_eff < kk:
                dsort = np.where(ok, dist, np.inf)
                keep = np.argsort(dsort, axis=1)[:, :k_eff]
                rowi = np.arange(b - a)[:, None]
                dist = dist[rowi, keep]
                idx_safe = idx_safe[rowi, keep]
                ok = ok[rowi, keep]
            sv = self.s[idx_safe]
            if offsets is not None and self.well_of is not None:
                sv = sv - offsets[self.well_of[idx_safe]]
            wgt = np.where(ok, 1.0 / np.power(np.maximum(dist, EPS), POWER), 0.0)
            den = wgt.sum(axis=1)
            good = den > 0
            res = np.full(b - a, np.nan)
            res[good] = (wgt * sv).sum(axis=1)[good] / den[good]
            out[a:b] = res
            dd = np.where(ok, dist, np.inf)
            dnn[a:b] = dd.min(axis=1)
            if self.well_of is not None:
                wid = np.where(ok, self.well_of[idx_safe], -1)
                ws = np.sort(wid, axis=1)
                chg = np.ones_like(ws, dtype=bool)
                chg[:, 1:] = ws[:, 1:] != ws[:, :-1]
                nw[a:b] = ((ws >= 0) & chg).sum(axis=1)
        return {"s": out, "nwells": nw, "dnn": dnn}


def build_field(w: Wells, mask: np.ndarray) -> Field:
    well_of = np.repeat(np.arange(len(w.wells)), np.diff(w.starts))
    return Field(np.stack([w.X[mask], w.Y[mask]], axis=1),
                 (w.TVT + w.Z)[mask], well_of[mask])


def pooled_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))
