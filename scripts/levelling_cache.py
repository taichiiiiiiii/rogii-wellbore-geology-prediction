"""Cache the 773 train wells into one compact npz for the levelling experiments.

Reading 773 CSVs with pandas takes minutes; every downstream script needs the same
five columns, so pay the cost once and write a single concatenated array set to the
scratchpad (never into the repo).

Layout: flat arrays over all rows of all wells, plus `starts` offsets and `wells`
names.  X/Y are Texas State Plane feet (~3e6) so they must stay float64; a float32
would quantise them to ~0.25 ft and destroy the 300 ft well spacing we rely on.

Usage:
    uv run python scripts/levelling_cache.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train"
CACHE = Path("/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/levelling")
COLS = ["MD", "X", "Y", "Z", "TVT", "TVT_input"]


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "wells.npz"
    if out.exists():
        print(f"exists: {out}")
        return 0

    wells: list[str] = []
    starts: list[int] = []
    row_idx: list[np.ndarray] = []
    chunks: dict[str, list[np.ndarray]] = {c: [] for c in ["MD", "X", "Y", "Z", "TVT"]}
    known: list[np.ndarray] = []
    n = 0
    mismatch = 0

    for p in sorted(DATA.glob("*__horizontal_well.csv")):
        wid = p.name.split("__")[0]
        df = pd.read_csv(p, usecols=COLS)
        orig = df.index.to_numpy()
        m = df[["X", "Y", "Z", "TVT"]].notna().all(axis=1).to_numpy()
        df = df[m]
        orig = orig[m]
        if len(df) < 100:
            continue
        kn = df["TVT_input"].notna().to_numpy()
        # TVT_input should be TVT wherever it is present; verify rather than assume.
        d = np.abs(df["TVT_input"].to_numpy(float)[kn] - df["TVT"].to_numpy(float)[kn])
        if d.size and float(np.nanmax(d)) > 1e-6:
            mismatch += 1
        wells.append(wid)
        starts.append(n)
        row_idx.append(orig.astype(np.int32))
        for c in chunks:
            chunks[c].append(df[c].to_numpy(np.float64))
        known.append(kn)
        n += len(df)

    starts.append(n)
    np.savez(
        out,
        wells=np.array(wells),
        starts=np.array(starts, dtype=np.int64),
        row_idx=np.concatenate(row_idx),
        MD=np.concatenate(chunks["MD"]),
        X=np.concatenate(chunks["X"]),
        Y=np.concatenate(chunks["Y"]),
        Z=np.concatenate(chunks["Z"]),
        TVT=np.concatenate(chunks["TVT"]),
        known=np.concatenate(known),
    )
    print(f"wells={len(wells)} rows={n} TVT_input!=TVT wells={mismatch}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
