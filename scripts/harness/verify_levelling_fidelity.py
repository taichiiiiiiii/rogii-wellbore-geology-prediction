"""Fidelity proof for the ported levelling cell (build_levelling_probe.py).

Extracts the levelling cell VERBATIM from the built notebook (same discipline
as verify_levelling_smoke.py: firing is demonstrated by exec()ing the actual
in-kernel code, never a re-implementation) with LEVELLING_ENABLED=False so
only the function definitions land in the namespace -- the driver block
never runs (it reads a real submission.csv from disk, which this harness
does not have; the reference inputs here are wells.npz + the k40 audit's
own submission.csv, read directly).

Reference: the 40-well k40 held-out audit (scripts/levelling_{cache,common,
network,predict,rules}.py), donor variant "a" (train wells only for the
spatial field), network radius 450 ft / stride 16, tail-300 anchor, blended
with the CURRENT rule -- w=LEVELLING_BLEND_W, cap=LEVELLING_CLIP_KAPPA *
MAD_ft(pooled over ALL judged eval rows, fallback rows included as a zero
contribution). This reproduces
scratchpad/levelling/REVIEW_perwell_a_pair450_w030_kappamad.csv, derived
directly from scratchpad/levelling/rows_a_pair450.parquet's sp_tail column
(the reference's own tail-anchor spatial prediction) -- see the module
docstring in build_levelling_probe.py for why this superseded the earlier
w=0.45 / frozen-cap rule that reproduces FINAL_perwell_a_pair450_clip.csv.

Usage:
    uv run python scripts/harness/verify_levelling_fidelity.py
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("ROGII_WORK", "work"))
NB_PATH = SCRATCH / "levelling_kernel" / "rogii-gs145-levelling" / "rogii-gs145-levelling.ipynb"
CACHE = SCRATCH / "levelling"
K40 = SCRATCH / "harness_audit" / "k40"
REFERENCE_CSV = CACHE / "REVIEW_perwell_a_pair450_w030_kappamad.csv"

TOLERANCE = 1e-6


def extract_levelling_cell(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "[LEVELLING] params enabled=" in src:
            return src
    raise RuntimeError(f"could not locate the levelling cell in {nb_path}")


def load_functions() -> dict:
    """exec() the levelling cell with LEVELLING_ENABLED=False so only the
    function definitions (and the frozen param defaults) land in the
    namespace -- the driver block short-circuits to the DISABLED print."""
    src = extract_levelling_cell(NB_PATH)
    ns: dict = {"LEVELLING_ENABLED": False}
    exec(compile(src, "<levelling_cell>", "exec"), ns)
    for name in ("solve_network_offsets", "predict_levelled_surface",
                 "fit_query_anchor", "gate_and_blend",
                 "LEVELLING_BLEND_W", "LEVELLING_CLIP_KAPPA",
                 "LEVELLING_PAIR_RADIUS_FT", "LEVELLING_PAIR_STRIDE",
                 "LEVELLING_PAIR_MIN_MATCHES", "LEVELLING_PAIR_NN_K",
                 "LEVELLING_PAIR_HUBER", "LEVELLING_PAIR_IRLS_ITERS",
                 "LEVELLING_RADIUS_FT", "LEVELLING_KNN", "LEVELLING_DONOR_STRIDE",
                 "LEVELLING_ANCHOR_TAIL_ROWS", "LEVELLING_MIN_KNOWN_ROWS"):
        if name not in ns:
            raise RuntimeError(f"expected {name!r} in the extracted cell namespace")
    return ns


def load_wells() -> dict:
    z = np.load(CACHE / "wells.npz", allow_pickle=False)
    d = {k: z[k] for k in ["wells", "starts", "row_idx", "MD", "X", "Y", "Z", "TVT", "known"]}
    d["wells"] = d["wells"].astype(str)
    return d


def build_table(w: dict, *, exclude: set, known_only: set, stride: int) -> dict:
    """donor_mask()'s exact row selection (stride first, 0-indexed from each
    well's own start, THEN AND with `known` for known_only wells), as flat
    float64 arrays -- the "same inputs the reference used"."""
    wells_out, x_out, y_out, s_out = [], [], [], []
    starts = w["starts"]
    for i, wid in enumerate(w["wells"]):
        if wid in exclude:
            continue
        a, b = int(starts[i]), int(starts[i + 1])
        sel = np.zeros(b - a, dtype=bool)
        sel[::stride] = True
        if wid in known_only:
            sel &= w["known"][a:b]
        if not sel.any():
            continue
        idx = np.where(sel)[0] + a
        wells_out.append(np.full(len(idx), wid))
        x_out.append(w["X"][idx])
        y_out.append(w["Y"][idx])
        s_out.append(w["TVT"][idx] + w["Z"][idx])
    return {
        "well": np.concatenate(wells_out),
        "x": np.concatenate(x_out).astype(np.float64),
        "y": np.concatenate(y_out).astype(np.float64),
        "s": np.concatenate(s_out).astype(np.float64),
    }


def main() -> int:
    t0 = time.time()
    ns = load_functions()
    print(f"extracted levelling cell functions from {NB_PATH}")
    print(f"  LEVELLING_BLEND_W={ns['LEVELLING_BLEND_W']}  "
          f"LEVELLING_CLIP_KAPPA={ns['LEVELLING_CLIP_KAPPA']}")

    w = load_wells()
    sub = pd.read_csv(K40 / "submission.csv")
    sub["id"] = sub["id"].astype(str)
    parts = sub["id"].str.rsplit("_", n=1)
    sub["well"] = parts.str[0]
    sub["ri"] = parts.str[1].astype(int)
    held = sorted(sub["well"].unique())
    heldset = set(held)
    print(f"held (query) wells: {len(held)}  eval rows: {len(sub)}")

    # network donors: every mounted well's legally visible points (train in
    # full, held-out by known prefix only) -- "true for both variants" in
    # the reference.
    net_table = build_table(w, exclude=set(), known_only=heldset,
                             stride=ns["LEVELLING_PAIR_STRIDE"])
    # spatial-field donors: donor variant "a" -- train wells only.
    field_table = build_table(w, exclude=heldset, known_only=set(),
                               stride=ns["LEVELLING_DONOR_STRIDE"])
    print(f"network donor points: {len(net_table['s'])}  "
          f"field donor points: {len(field_table['s'])} "
          f"({len(np.unique(field_table['well']))} train wells)")

    offsets, info = ns["solve_network_offsets"](
        net_table, radius=ns["LEVELLING_PAIR_RADIUS_FT"],
        min_matches=ns["LEVELLING_PAIR_MIN_MATCHES"], nn_k=ns["LEVELLING_PAIR_NN_K"],
        huber=ns["LEVELLING_PAIR_HUBER"], irls_iters=ns["LEVELLING_PAIR_IRLS_ITERS"])
    print(f"network solve: n_pairs={info['n_pairs']} n_components={info['n_components']} "
          f"residual_rms_ft={info['residual_rms_ft']:.3f}  ({time.time() - t0:.1f}s)")

    starts = w["starts"]
    lookup = {wid: i for i, wid in enumerate(w["wells"])}
    rows, all_diff = [], []
    for wid in held:
        i = lookup[wid]
        a, b = int(starts[i]), int(starts[i + 1])
        kn = w["known"][a:b]
        X, Y, Z, TV, RI = w["X"][a:b], w["Y"][a:b], w["Z"][a:b], w["TVT"][a:b], w["row_idx"][a:b]
        S = TV + Z
        known_rows = {"x": X[kn], "y": Y[kn], "s": S[kn]}
        c_q = ns["fit_query_anchor"](known_rows, field_table, offsets, wid,
                                      tail_rows=ns["LEVELLING_ANCHOR_TAIL_ROWS"],
                                      min_known_rows=ns["LEVELLING_MIN_KNOWN_ROWS"],
                                      radius=ns["LEVELLING_RADIUS_FT"])

        ev = ~kn
        eval_x, eval_y, eval_z, eval_tv, eval_ri = X[ev], Y[ev], Z[ev], TV[ev], RI[ev]
        if np.isfinite(c_q):
            s_hat = ns["predict_levelled_surface"](eval_x, eval_y, field_table, offsets,
                                                     exclude_well=wid,
                                                     radius=ns["LEVELLING_RADIUS_FT"],
                                                     k=ns["LEVELLING_KNN"])
            spat = (s_hat + c_q) - eval_z
        else:
            spat = np.full(len(eval_x), np.nan)

        sp = sub[sub["well"] == wid].set_index("ri")["tvt"]
        pipe = np.array([sp.get(int(r), np.nan) for r in eval_ri])
        m = np.isfinite(pipe)
        diff = spat[m] - pipe[m]
        all_diff.append(np.where(np.isfinite(diff), diff, 0.0))
        rows.append(pd.DataFrame({"well": wid, "pipe": pipe[m], "truth": eval_tv[m],
                                   "spat": spat[m]}))

    df = pd.concat(rows, ignore_index=True)
    mad_ft = float(np.median(np.abs(np.concatenate(all_diff))))
    cap_ft = ns["LEVELLING_CLIP_KAPPA"] * mad_ft
    print(f"\nrows {len(df)}  wells {df['well'].nunique()}  "
          f"MAD_ft={mad_ft:.6f}  cap_ft={cap_ft:.6f}  ({time.time() - t0:.1f}s total)")

    pipe = df["pipe"].to_numpy(float)
    spat = df["spat"].to_numpy(float)
    truth = df["truth"].to_numpy(float)
    final = ns["gate_and_blend"](pipe, spat, cap_ft, blend_w=ns["LEVELLING_BLEND_W"])

    base_rmse = float(np.sqrt(np.mean((pipe - truth) ** 2)))
    pooled_rmse = float(np.sqrt(np.mean((final - truth) ** 2)))
    print(f"pooled base RMSE     {base_rmse:.4f}")
    print(f"pooled blended RMSE  {pooled_rmse:.4f}  (delta {pooled_rmse - base_rmse:+.4f})")

    per = df.assign(e1=(final - truth) ** 2).groupby("well").agg(
        n=("truth", "size"),
        rmse_blend=("e1", lambda x: float(np.sqrt(x.mean())))).reset_index()

    if not REFERENCE_CSV.exists():
        print(f"\nREFERENCE CSV NOT FOUND: {REFERENCE_CSV}")
        return 1
    ref = pd.read_csv(REFERENCE_CSV)
    mrg = per.merge(ref[["well", "rmse_blend"]], on="well", suffixes=("_mine", "_ref"))
    if len(mrg) != len(ref) or len(mrg) != len(per):
        print(f"\nWELL SET MISMATCH: mine={len(per)} ref={len(ref)} merged={len(mrg)}")
        return 1
    mrg["diff"] = (mrg["rmse_blend_mine"] - mrg["rmse_blend_ref"]).abs()
    max_diff = float(mrg["diff"].max())

    print(f"\nper-well |RMSE_ported - RMSE_reference| over {len(mrg)} wells:")
    print(mrg["diff"].describe().to_string())
    print("\nworst 5 wells by diff:")
    print(mrg.sort_values("diff", ascending=False).head(5).to_string(index=False))
    print(f"\nMAX PER-WELL |RMSE_ported - RMSE_reference| = {max_diff:.3e}  "
          f"(tolerance {TOLERANCE:.0e})")
    ok = max_diff < TOLERANCE
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
