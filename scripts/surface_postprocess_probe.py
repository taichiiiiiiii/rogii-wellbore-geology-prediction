"""Measure the surface reparametrisation as a post-process on harness predictions.

TVT(MD) = surface(MD) - Z(MD), and Z is measured exactly in the eval zone, so the
high-frequency detail of the target is handed to us for free; only the smooth
per-well surface trend has to be predicted. If our pipeline's implied surface
carries high-frequency error, projecting it onto a smooth basis and subtracting
the known Z removes that error structurally.

This is a post-process on a fixed prediction file, so the comparison is paired:
the pipeline's own run-to-run nondeterminism cancels exactly and a gain of any
size is measurable, unlike a knob that needs a fresh pipeline run.

Usage:
    uv run python scripts/surface_postprocess_probe.py <harness_output_dir> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def load_eval_frame(well: str) -> pd.DataFrame:
    """Eval-zone rows for one well, with the ids the pipeline emits."""
    hw = pd.read_csv(DATA / f"{well}__horizontal_well.csv")
    hw["id"] = [f"{well}_{i}" for i in hw.index]
    ev = hw[hw["TVT_input"].isna()].copy()
    return ev[["id", "MD", "Z", "TVT"]].assign(well=well)


def poly_surface(md: np.ndarray, surf: np.ndarray, deg: int) -> np.ndarray:
    """Least-squares polynomial refit of the implied surface."""
    x = (md - md.mean()) / max(md.std(), 1e-9)
    return np.polyval(np.polyfit(x, surf, deg), x)


def anchored_poly_surface(md: np.ndarray, surf: np.ndarray, deg: int) -> np.ndarray:
    """Polynomial refit forced through the pipeline's own value at the eval start.

    A free fit can slide the whole curve off the anchor the tracker established;
    pinning the first point keeps the correction to shape only.
    """
    fit = poly_surface(md, surf, deg)
    return fit + (surf[0] - fit[0])


def blend(orig: np.ndarray, smooth: np.ndarray, w: float) -> np.ndarray:
    return (1.0 - w) * orig + w * smooth


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv[1:]]
    if not dirs:
        print(__doc__)
        return 2

    for d in dirs:
        sub = pd.read_csv(d / "submission.csv")
        sub["id"] = sub["id"].astype(str)
        wells = sorted({i.rsplit("_", 1)[0] for i in sub["id"]})
        truth = pd.concat([load_eval_frame(w) for w in wells], ignore_index=True)
        df = truth.merge(sub[["id", "tvt"]], on="id", how="left")
        assert df["tvt"].notna().all(), f"{d}: submission does not cover all eval ids"

        base = rmse(df["tvt"], df["TVT"])
        print(f"\n=== {d.name} ===   wells={len(wells)}  rows={len(df)}  baseline={base:.4f}")

        # How smooth is the true surface, and how smooth is ours?
        r2_true, r2_pred = [], []
        for _, g in df.groupby("well", sort=False):
            md = g["MD"].to_numpy(float)
            for surf, acc in ((g["TVT"] + g["Z"], r2_true), (g["tvt"] + g["Z"], r2_pred)):
                s = surf.to_numpy(float)
                fit = poly_surface(md, s, 2)
                ss_res = float(np.sum((s - fit) ** 2))
                ss_tot = float(np.sum((s - s.mean()) ** 2))
                acc.append(1.0 - ss_res / max(ss_tot, 1e-12))
        print(f"  deg-2 R^2 of surface   true={np.mean(r2_true):.4f}   ours={np.mean(r2_pred):.4f}")

        # Oracle: smooth the TRUE surface, subtract known Z.
        for deg in (1, 2, 3):
            pred = np.concatenate([
                poly_surface(g["MD"].to_numpy(float), (g["TVT"] + g["Z"]).to_numpy(float), deg)
                - g["Z"].to_numpy(float)
                for _, g in df.groupby("well", sort=False)
            ])
            order = np.concatenate([g.index.to_numpy() for _, g in df.groupby("well", sort=False)])
            got = rmse(pred, df["TVT"].to_numpy()[order])
            print(f"  ORACLE deg-{deg} smooth-surface minus known Z: {got:.4f}")

        # Post-process on OUR predictions.
        rows = []
        for deg in (1, 2, 3, 4):
            for anchored in (False, True):
                fn = anchored_poly_surface if anchored else poly_surface
                for w in (0.25, 0.5, 0.75, 1.0):
                    out, idx = [], []
                    for _, g in df.groupby("well", sort=False):
                        md = g["MD"].to_numpy(float)
                        z = g["Z"].to_numpy(float)
                        surf = g["tvt"].to_numpy(float) + z
                        out.append(blend(surf, fn(md, surf, deg), w) - z)
                        idx.append(g.index.to_numpy())
                    pred = np.concatenate(out)
                    y = df["TVT"].to_numpy()[np.concatenate(idx)]
                    rows.append({
                        "deg": deg, "anchored": anchored, "w": w,
                        "rmse": rmse(pred, y), "delta": rmse(pred, y) - base,
                    })
        res = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
        print(res.head(12).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
