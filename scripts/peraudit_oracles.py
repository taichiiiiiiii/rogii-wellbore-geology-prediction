"""Oracle decomposition + wavelength analysis of the eval-zone drift.

Answers, for d[i] = TVT_true[i] - TVT_anchor over the eval zone:
  A. How much of the drift is a per-well CONSTANT / LINEAR-in-MD / LINEAR-in-dZ?
     (upper bounds: the coefficients are fitted on the truth)
  B. Is that structure PERSISTENT (fit on the first half of the eval zone,
     score on the second half) or just fit flexibility?
  C. Can the coefficients be recovered LEGALLY from the known zone
     (per-well OLS of TVT_input on MD and Z over the known lateral, then
     extrapolated)?  This is a real model, not an oracle.
  D. At which WAVELENGTHS does the drift variance live, and how much of each
     band does the learned model capture?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CACHE = Path(
    "/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/peraudit"
)
K40 = Path(
    "/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/"
    "harness_audit/k40/cv_summary.json"
)


def rmse(a, b=0.0):
    return float(np.sqrt(np.mean((np.asarray(a) - b) ** 2)))


def smooth(a: np.ndarray, w: int) -> np.ndarray:
    """O(n) centred box filter with edge padding."""
    n = len(a)
    if w <= 1 or w >= n:
        return np.full(n, a.mean())
    pad = w // 2
    ap = np.concatenate([np.full(pad, a[0]), a, np.full(w - 1 - pad, a[-1])])
    c = np.concatenate([[0.0], np.cumsum(ap)])
    return (c[w : w + n] - c[:n]) / w


def ols_pred(A: np.ndarray, y: np.ndarray, A2: np.ndarray) -> np.ndarray:
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A2 @ c


def main() -> None:
    from rogii.data import data_root

    meta = json.loads((CACHE / "meta.json").read_text())
    feats = meta["features"]
    fi = {f: i for i, f in enumerate(feats)}
    wells = [s["well"] for s in meta["wells"]]
    X = np.load(CACHE / "X.npy", mmap_mode="r")
    y = np.load(CACHE / "y.npy").astype(np.float64)
    g = np.load(CACHE / "well.npy")
    dmd = np.asarray(X[:, fi["dmd"]], np.float64)
    dz = np.asarray(X[:, fi["dz"]], np.float64)
    oof = None
    if (CACHE / "oof_L4.npy").exists():
        oof = np.load(CACHE / "oof_L4.npy").astype(np.float64)

    k40 = set(json.loads(K40.read_text())["well_ids"])
    starts = np.searchsorted(g, np.arange(len(wells)))
    ends = np.searchsorted(g, np.arange(len(wells)), side="right")

    # accumulators: name -> [sse, n]
    acc: dict[str, list[float]] = {}
    acc40: dict[str, list[float]] = {}

    def add(name: str, resid: np.ndarray, is40: bool) -> None:
        s = float(np.sum(resid**2))
        acc.setdefault(name, [0.0, 0.0])
        acc[name][0] += s
        acc[name][1] += len(resid)
        if is40:
            acc40.setdefault(name, [0.0, 0.0])
            acc40[name][0] += s
            acc40[name][1] += len(resid)

    # second-half accumulators (persistence test)
    acc_h: dict[str, list[float]] = {}

    def addh(name: str, resid: np.ndarray) -> None:
        acc_h.setdefault(name, [0.0, 0.0])
        acc_h[name][0] += float(np.sum(resid**2))
        acc_h[name][1] += len(resid)

    # wavelength bands
    BANDS = [("gt4000ft", 4001), ("1000-4000", 1001), ("250-1000", 251), ("lt250", 1)]
    band_var = np.zeros(len(BANDS))
    band_cov = np.zeros(len(BANDS))  # explained SS by oof in each band
    band_n = 0.0

    rows = []
    for w, well in enumerate(wells):
        a, b = starts[w], ends[w]
        if b - a < 50:
            continue
        yy = y[a:b]
        md = dmd[a:b]
        zz = dz[a:b]
        n = len(yy)
        is40 = well in k40
        ones = np.ones(n)

        add("carry_last", yy, is40)
        A_c = ones[:, None]
        A_m = np.c_[md, ones]
        A_z = np.c_[zz, ones]
        A_mz = np.c_[md, zz, ones]
        add("O1_const", yy - ols_pred(A_c, yy, A_c), is40)
        add("O2_lin_md", yy - ols_pred(A_m, yy, A_m), is40)
        add("O3_lin_dz", yy - ols_pred(A_z, yy, A_z), is40)
        add("O4_lin_md_dz", yy - ols_pred(A_mz, yy, A_mz), is40)
        # quadratic in MD as a richer smooth-shape oracle
        A_q = np.c_[md, md**2, ones]
        add("O6_quad_md", yy - ols_pred(A_q, yy, A_q), is40)
        A_qz = np.c_[md, md**2, zz, ones]
        add("O7_quad_md_dz", yy - ols_pred(A_qz, yy, A_qz), is40)

        # ---- B: persistence (fit on the first half, score on the second) ----
        h = n // 2
        y2 = yy[h:]
        addh("carry_last", y2)
        for nm, A in (("O1_const", A_c), ("O2_lin_md", A_m), ("O4_lin_md_dz", A_mz)):
            addh(nm + "_half", y2 - ols_pred(A[:h], yy[:h], A[h:]))
            addh(nm + "_full", y2 - ols_pred(A, yy, A)[h:])

        # ---- D: wavelength bands ----
        if n > 400:
            s4 = smooth(yy, min(4001, n - 1))
            s1 = smooth(yy, min(1001, n - 1))
            s0 = smooth(yy, min(251, n - 1))
            parts = [s4, s1 - s4, s0 - s1, yy - s0]
            for j, p in enumerate(parts):
                band_var[j] += float(np.sum(p**2))
                if oof is not None:
                    band_cov[j] += float(np.sum(p * oof[a:b]))
            band_n += n

        rows.append(
            dict(
                well=well,
                n=n,
                carry=rmse(yy),
                o4=rmse(yy - ols_pred(A_mz, yy, A_mz)),
                beta_md=float(np.linalg.lstsq(A_m, yy, rcond=None)[0][0]),
                beta_dz=float(np.linalg.lstsq(A_z, yy, rcond=None)[0][0]),
                r_ydz=float(np.corrcoef(yy, zz)[0, 1]) if zz.std() > 0 else 0.0,
                k40=is40,
            )
        )

    # ---- C: legal per-well known-zone regression, extrapolated ------------
    print("legal known-zone extrapolation ...", flush=True)
    legal: dict[str, list[float]] = {}

    def addl(name: str, resid: np.ndarray, is40: bool) -> None:
        legal.setdefault(name, [0.0, 0.0, 0.0, 0.0])
        legal[name][0] += float(np.sum(resid**2))
        legal[name][1] += len(resid)
        if is40:
            legal[name][2] += float(np.sum(resid**2))
            legal[name][3] += len(resid)

    for w, well in enumerate(wells):
        p = data_root() / "train" / f"{well}__horizontal_well.csv"
        h = pd.read_csv(p, usecols=["MD", "Z", "TVT", "TVT_input"])
        ti = h["TVT_input"].to_numpy(np.float64)
        ev = np.isnan(ti)
        kn = np.where(~ev)[0]
        if len(kn) == 0:
            continue
        anc = int(kn[-1])
        ei = np.where(ev)[0]
        ei = ei[ei > anc]
        if len(ei) < 10:
            continue
        md = h["MD"].to_numpy(np.float64)
        z = h["Z"].to_numpy(np.float64)
        tv = h["TVT"].to_numpy(np.float64)
        tvt_a = ti[anc]
        yy = tv[ei] - tvt_a
        is40 = well in k40
        # lateral known rows only (|dZ/dMD| small)
        sl = np.gradient(z, md)
        lat = kn[np.abs(sl[kn]) < 0.25]
        for L in (500, 1000, 2000, 100000):
            sel = lat[lat > anc - L]
            if len(sel) < 30:
                addl(f"legal_MDZ_{L}", yy, is40)
                addl(f"legal_MD_{L}", yy, is40)
                continue
            dm = md[sel] - md[anc]
            dzk = z[sel] - z[anc]
            tk = ti[sel] - tvt_a
            dme = md[ei] - md[anc]
            dze = z[ei] - z[anc]
            for nm, A, A2 in (
                (f"legal_MD_{L}", np.c_[dm, np.ones(len(sel))], np.c_[dme, np.ones(len(ei))]),
                (
                    f"legal_MDZ_{L}",
                    np.c_[dm, dzk, np.ones(len(sel))],
                    np.c_[dme, dze, np.ones(len(ei))],
                ),
            ):
                try:
                    pr = ols_pred(A, tk, A2)
                except np.linalg.LinAlgError:
                    pr = np.zeros(len(ei))
                pr = np.clip(pr, -200, 200)
                addl(nm, yy - pr, is40)

    def show(d, label, idx=(0, 1)):
        print(f"\n--- {label} ---")
        for k, v in sorted(d.items(), key=lambda t: np.sqrt(t[1][idx[0]] / t[1][idx[1]])):
            print(f"  {k:24s} {np.sqrt(v[idx[0]] / v[idx[1]]):8.3f}")

    print(f"\nwells used: {len(rows)}")
    show(acc, "pooled (all 773 wells, in-sample oracles)")
    show(acc40, "k40 subset")
    show(acc_h, "second-half-only (persistence test)")
    show(legal, "LEGAL known-zone extrapolation (pooled)")
    show(legal, "LEGAL known-zone extrapolation (k40)", idx=(2, 3))

    bv = band_var / max(band_n, 1)
    print("\n--- wavelength decomposition of the drift (variance share) ---")
    tot = band_var.sum()
    for (nm, _), v in zip(BANDS, band_var, strict=False):
        print(f"  {nm:12s} rms={np.sqrt(v / band_n):7.3f}  share={v / tot:6.1%}")
    if oof is not None:
        print("  explained-SS share of L4 OOF per band (cov/|band|):")
        for (nm, _), c, v in zip(BANDS, band_cov, band_var, strict=False):
            print(f"    {nm:12s} cov/var = {c / max(v, 1e-9):7.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "per_well_oracle.csv", index=False)
    q = df["carry"].to_numpy()
    sse = df["n"].to_numpy() * q**2
    o = np.argsort(-sse)
    top = int(np.ceil(0.05 * len(df)))
    print(
        f"\nworst 5% of wells ({top}) carry {sse[o[:top]].sum() / sse.sum():.1%} "
        f"of the total carry-last SSE"
    )
    print(
        "median per-well |corr(d, dZ)| = "
        f"{np.nanmedian(np.abs(df['r_ydz'])):.3f}; "
        f"beta_dz median={np.nanmedian(df['beta_dz']):.3f} "
        f"IQR=({np.nanpercentile(df['beta_dz'], 25):.3f},"
        f"{np.nanpercentile(df['beta_dz'], 75):.3f})"
    )
    print("saved ->", CACHE / "per_well_oracle.csv")


if __name__ == "__main__":
    main()
