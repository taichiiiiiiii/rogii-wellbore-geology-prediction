"""Does the steering signal explain anything the pipeline has not already used?

R93 measured the steering coupling against a linear-extrapolation baseline of 43.4 ft.
Our pipeline reaches 6.4, and its particle filter tracks locally along the same
trajectory, so most of that coupling may already be spent. The question that decides
whether the route has any headroom left is therefore not "does steering explain the
surface" but "does steering explain the pipeline's remaining error".

Held-out harness wells give real predictions with realistic error against known truth,
which is exactly what this needs. The harness cannot rank pipeline variants (R81), but
that failure is about variants; these are its actual predictions on wells it never saw.

Usage:
    uv run python scripts/steering_vs_pipeline_residual.py <harness_output_dir> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
TAIL = 200


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    d = float(np.sum(x * x))
    return float(np.sum(x * (y - y.mean())) / d) if d > 0 else 0.0


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv[1:]]
    if not dirs:
        print(__doc__)
        return 2

    for d in dirs:
        sub = pd.read_csv(d / "submission.csv")
        sub["id"] = sub["id"].astype(str)
        wells = sorted({i.rsplit("_", 1)[0] for i in sub["id"]})

        rows, pooled_e, pooled_t = [], [], []
        for wid in wells:
            hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
            hw["id"] = [f"{wid}_{i}" for i in hw.index]
            kn = hw[hw["TVT_input"].notna()]
            ev = hw[hw["TVT_input"].isna()]
            if len(kn) < TAIL + 10 or len(ev) < 50:
                continue

            tail = kn.tail(TAIL)
            md_a = float(tail["MD"].iloc[-1])
            z_a = float(tail["Z"].iloc[-1])
            b_z = slope(tail["MD"].to_numpy(float), tail["Z"].to_numpy(float))

            g = ev.merge(sub[["id", "tvt"]], on="id", how="left")
            if g["tvt"].isna().any():
                continue
            err = g["tvt"].to_numpy(float) - g["TVT"].to_numpy(float)
            steer = g["Z"].to_numpy(float) - (z_a + b_z * (g["MD"].to_numpy(float) - md_a))
            if steer.std() < 1e-9 or err.std() < 1e-9:
                continue

            r = float(np.corrcoef(err, steer)[0, 1])
            beta = float(np.sum(steer * err) / max(np.sum(steer**2), 1e-12))
            before = float(np.sqrt(np.mean(err**2)))
            after = float(np.sqrt(np.mean((err - beta * steer) ** 2)))
            rows.append({"well": wid, "n": len(g), "r": r, "beta": beta,
                         "before": before, "after": after})
            pooled_e.append(err)
            pooled_t.append(steer)

        res = pd.DataFrame(rows)
        e, t = np.concatenate(pooled_e), np.concatenate(pooled_t)
        beta_g = float(np.sum(t * e) / max(np.sum(t * t), 1e-12))
        base = float(np.sqrt(np.mean(e**2)))
        glob = float(np.sqrt(np.mean((e - beta_g * t) ** 2)))
        ora = float(np.sqrt(np.average(res["after"] ** 2, weights=res["n"])))

        print(f"\n=== {d.name} ===  wells={len(res)}  rows={len(e)}")
        print(f"  pipeline pooled RMSE            {base:8.4f} ft")
        print(f"  pooled r(error, steering)       {np.corrcoef(e, t)[0, 1]:+8.4f}")
        print(f"  after one global steering fit   {glob:8.4f} ft  ({glob - base:+.4f})")
        print(f"  after per-well ORACLE beta      {ora:8.4f} ft  ({ora - base:+.4f})")
        print(f"  per-well |r| > 0.5              {(res['r'].abs() > 0.5).mean()*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
