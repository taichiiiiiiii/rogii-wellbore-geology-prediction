"""Does the steering coupling estimated on the known zone transfer to the eval zone?

steering_signal_probe.py found the coupling is real and large inside a well — 80.6%
of wells have |r| > 0.5 between the surface residual and the steering residual, and
fitting the coefficient per well cuts the linear-extrapolation residual from 43.4 ft
to 15.3 ft. But that fit uses the answer, and pooled across wells the correlation
collapses to +0.046 because the sign is well-specific.

The coefficient describes how hard the driller chased the formation. If that habit is
stable along a well, it can be measured where TVT_input is known and carried into the
eval zone — which would turn an oracle into a predictor.

The known zone is split: an earlier window estimates beta, a later held-out window
plays the role of the eval zone. Everything is anchored at the boundary between them,
so nothing from the held-out side leaks into the estimate.

Usage:
    uv run python scripts/steering_beta_transfer.py [n_wells]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
MIN_KNOWN = 600
TAIL = 200


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    d = float(np.sum(x * x))
    return float(np.sum(x * (y - y.mean())) / d) if d > 0 else 0.0


def residuals(md, surf, z, md_a, s_a, z_a, b_s, b_z):
    d = md - md_a
    return surf - (s_a + b_s * d), z - (z_a + b_z * d)


def main(argv: list[str]) -> int:
    limit = int(argv[1]) if len(argv) > 1 else 773
    rows = []
    for p in sorted(DATA.glob("*__horizontal_well.csv"))[:limit]:
        df = pd.read_csv(p, usecols=["MD", "Z", "TVT", "TVT_input"])
        kn = df[df["TVT_input"].notna()].dropna(subset=["Z", "MD"])
        if len(kn) < MIN_KNOWN:
            continue
        # Split the known zone in half: fit on the first, pretend the second is eval.
        cut = len(kn) // 2
        fit, held = kn.iloc[:cut], kn.iloc[cut:]
        if len(fit) < TAIL + 50 or len(held) < 100:
            continue

        tail = fit.tail(TAIL)
        md_a = float(tail["MD"].iloc[-1])
        s_tail = tail["TVT_input"].to_numpy(float) + tail["Z"].to_numpy(float)
        z_tail = tail["Z"].to_numpy(float)
        md_tail = tail["MD"].to_numpy(float)
        b_s, b_z = slope(md_tail, s_tail), slope(md_tail, z_tail)
        s_a, z_a = float(s_tail[-1]), float(z_tail[-1])

        # beta measured on the WHOLE fit window, not just the anchor tail: a longer
        # baseline identifies the coefficient far better, which is the steelman for
        # this idea. Slopes still come from the tail, so the anchor is unchanged.
        md_f = fit["MD"].to_numpy(float)
        s_f = fit["TVT_input"].to_numpy(float) + fit["Z"].to_numpy(float)
        z_f = fit["Z"].to_numpy(float)
        sr_f, tr_f = residuals(md_f, s_f, z_f, md_a, s_a, z_a, b_s, b_z)
        if np.sum(tr_f**2) < 1e-9:
            continue
        beta_known = float(np.sum(tr_f * sr_f) / np.sum(tr_f**2))

        md_h = held["MD"].to_numpy(float)
        s_h = held["TVT_input"].to_numpy(float) + held["Z"].to_numpy(float)
        z_h = held["Z"].to_numpy(float)
        sr_h, tr_h = residuals(md_h, s_h, z_h, md_a, s_a, z_a, b_s, b_z)
        if np.sum(tr_h**2) < 1e-9 or not np.isfinite(sr_h).all():
            continue
        beta_held = float(np.sum(tr_h * sr_h) / np.sum(tr_h**2))

        before = float(np.sqrt(np.mean(sr_h**2)))
        with_known = float(np.sqrt(np.mean((sr_h - beta_known * tr_h) ** 2)))
        with_oracle = float(np.sqrt(np.mean((sr_h - beta_held * tr_h) ** 2)))
        with_unit = float(np.sqrt(np.mean((sr_h - 1.0 * tr_h) ** 2)))
        rows.append({"n": len(held), "beta_known": beta_known, "beta_held": beta_held,
                     "before": before, "transfer": with_known,
                     "oracle": with_oracle, "unit": with_unit})

    r = pd.DataFrame(rows)
    print(f"wells usable: {len(r)}")
    ok = np.isfinite(r["beta_known"]) & np.isfinite(r["beta_held"])
    print("\nbeta agreement between the fit window and the held-out window:")
    rho = r.loc[ok, "beta_known"].corr(r.loc[ok, "beta_held"])
    print(f"  corr(beta_known, beta_held) = {rho:+.4f}")
    print(f"  beta_known  median {r['beta_known'].median():+.3f}  IQR "
          f"[{r['beta_known'].quantile(.25):+.3f}, {r['beta_known'].quantile(.75):+.3f}]")
    print(f"  beta_held   median {r['beta_held'].median():+.3f}  IQR "
          f"[{r['beta_held'].quantile(.25):+.3f}, {r['beta_held'].quantile(.75):+.3f}]")
    print(f"  same sign: {(np.sign(r['beta_known']) == np.sign(r['beta_held'])).mean()*100:.1f}%")

    def pooled(col):
        return float(np.sqrt(np.average(r[col] ** 2, weights=r["n"])))

    print("\nheld-out surface-residual RMS (row-weighted):")
    print(f"  no correction              {pooled('before'):8.3f} ft")
    print(f"  beta from the fit window   {pooled('transfer'):8.3f} ft   "
          f"({pooled('transfer') - pooled('before'):+.3f})")
    print(f"  beta = 1 (hold TVT flat)   {pooled('unit'):8.3f} ft   "
          f"({pooled('unit') - pooled('before'):+.3f})")
    print(f"  beta oracle (uses truth)   {pooled('oracle'):8.3f} ft   "
          f"({pooled('oracle') - pooled('before'):+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
