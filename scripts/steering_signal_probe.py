"""Does the driller's steering encode the formation trend the gamma ray cannot?

Both award write-ups establish the same wall: the eval-zone dip is effectively an
unobserved constant, and the gamma ray does not carry it. Both looked for the signal
in the log. Neither looked at the trajectory.

But geosteering is a closed loop — a human watched the formation and steered to stay
in zone — so the path itself may encode what they inferred. And the trajectory
(MD, X, Y, Z) is known exactly in the eval zone, which is precisely where the label
is not.

The test removes what is already known at the anchor from both sides and asks
whether what is left co-varies:

    surface residual = (TVT + Z) - [anchor level + known-zone surface slope * dMD]
    steering residual =      Z    - [anchor Z     + known-zone Z slope       * dMD]

A driller holding the well in zone against a dipping surface produces a positive
association; no association means the path carries nothing beyond its own inertia.

Usage:
    uv run python scripts/steering_signal_probe.py [n_wells]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
TAIL = 200          # rows of the known zone used to estimate the incoming slopes
MIN_KNOWN = 300
MIN_EVAL = 200


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = float(np.sum(x * x))
    return float(np.sum(x * (y - y.mean())) / denom) if denom > 0 else 0.0


def main(argv: list[str]) -> int:
    limit = int(argv[1]) if len(argv) > 1 else 773
    paths = sorted(DATA.glob("*__horizontal_well.csv"))[:limit]

    per_well, pooled_s, pooled_t = [], [], []
    for p in paths:
        df = pd.read_csv(p, usecols=["MD", "Z", "TVT", "TVT_input"])
        kn = df[df["TVT_input"].notna()]
        ev = df[df["TVT_input"].isna()].dropna(subset=["TVT", "Z", "MD"])
        if len(kn) < MIN_KNOWN or len(ev) < MIN_EVAL:
            continue

        tail = kn.tail(TAIL)
        md_a = float(tail["MD"].iloc[-1])
        surf_kn = tail["TVT_input"].to_numpy(float) + tail["Z"].to_numpy(float)
        z_kn = tail["Z"].to_numpy(float)
        md_kn = tail["MD"].to_numpy(float)

        b_s, b_z = slope(md_kn, surf_kn), slope(md_kn, z_kn)
        s_a, z_a = float(surf_kn[-1]), float(z_kn[-1])

        md = ev["MD"].to_numpy(float)
        d = md - md_a
        surf_res = (ev["TVT"].to_numpy(float) + ev["Z"].to_numpy(float)) - (s_a + b_s * d)
        steer_res = ev["Z"].to_numpy(float) - (z_a + b_z * d)

        if not (np.isfinite(surf_res).all() and np.isfinite(steer_res).all()):
            continue
        if surf_res.std() < 1e-9 or steer_res.std() < 1e-9:
            continue

        r = float(np.corrcoef(surf_res, steer_res)[0, 1])
        # How much of the surface residual a least-squares fit on steering removes.
        beta = float(np.sum(steer_res * surf_res) / max(np.sum(steer_res**2), 1e-12))
        before = float(np.sqrt(np.mean(surf_res**2)))
        after = float(np.sqrt(np.mean((surf_res - beta * steer_res) ** 2)))
        per_well.append({"well": p.name.split("__")[0], "n": len(ev), "r": r,
                         "beta": beta, "rmse_before": before, "rmse_after": after})
        pooled_s.append(surf_res)
        pooled_t.append(steer_res)

    res = pd.DataFrame(per_well)
    print(f"wells usable: {len(res)} / {len(paths)}")
    print("\nper-well correlation between surface residual and steering residual:")
    for q in (10, 25, 50, 75, 90):
        print(f"  r p{q:<2d} = {np.percentile(res['r'], q):+.3f}")
    strong = (res["r"].abs() > 0.5).mean() * 100
    print(f"  mean r = {res['r'].mean():+.3f}   share |r|>0.5: {strong:.1f}%")

    s = np.concatenate(pooled_s)
    t = np.concatenate(pooled_t)
    r_pool = float(np.corrcoef(s, t)[0, 1])
    beta = float(np.sum(t * s) / max(np.sum(t * t), 1e-12))
    print(f"\npooled over {len(s)} eval rows:  r = {r_pool:+.4f}   beta = {beta:+.4f}")
    print(f"  surface-residual RMS before = {np.sqrt(np.mean(s**2)):8.3f} ft")
    print(f"  after a single global fit   = {np.sqrt(np.mean((s - beta*t)**2)):8.3f} ft")

    # A per-well fit is the optimistic bound: it uses the answer to pick beta.
    ora = np.sqrt(np.average(res["rmse_after"]**2, weights=res["n"]))
    base = np.sqrt(np.average(res["rmse_before"]**2, weights=res["n"]))
    print("\nper-well ORACLE beta (upper bound, uses the truth):")
    print(f"  before = {base:8.3f} ft   after = {ora:8.3f} ft   gain = {base - ora:6.3f} ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
