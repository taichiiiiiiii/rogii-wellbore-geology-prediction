"""Price the numba particle filter against the pure-Python one it could replace.

R89 put the main line at ~8.3 h against a 9 h cap, with the 128-seed selector PF
costing ~3.26 h of that. R90 found the notebook already ships an njit twin of the
same filter — identical state transition, likelihood, resampling and estimator, and
identical MOM/VN/PN/RP/RR/RESAMP constants — used by a different stage.

This measures what swapping would actually buy, on a real well, at the seed count
the pipeline uses. It does not check agreement: the two differ by construction in
their random stream and in resampling the typewell GR onto a regular grid, so a swap
is a new config to be judged on the leaderboard, not a refactor.

Usage:
    uv run python scripts/harness/bench_pf_numba_vs_python.py [well] [n_seeds]
"""

from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(os.environ.get("ROGII_WORK", "work"))
NB = SCRATCH / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "train"

# Constants the pipeline passes to the njit filter, read off the lik_pf call site.
PF_CONSTS = dict(MOM=0.998, VN=0.002, PN=0.005, RP=0.1, RR=0.001, RESAMP=0.5, init_spr=4.5)


def cell_source(idx: int) -> str:
    return "".join(json.loads(NB.read_text())["cells"][idx].get("source", []))


def load_numba_pf() -> dict:
    """Exec just the njit helpers from the fast cell, stopping before its pandas glue.

    _grid lives one cell earlier, so pull it in too.
    """
    prev = cell_source(45)
    helpers = []
    for name in ("_interp1", "_grid"):
        i = prev.index(f"def {name}")
        i = prev.rfind("@njit", 0, i) if f"def {name}" != "def _grid" else i
        helpers.append(prev[i:prev.index("\ndef ", prev.index(f"def {name}") + 5)])
    grid_src = "\n".join(helpers)
    src = cell_source(46)
    start = src.index("@njit")
    end = src.index("def lik_pf")
    # cache=True needs a real source file on disk; drop it, it only affects the
    # one-off compile which is reported separately.
    def uncached(text: str) -> str:
        for form in ("cache=True, ", ", cache=True", "cache=True"):
            text = text.replace(form, "")
        return text

    ns: dict = {"np": np, "pd": pd}
    exec("from numba import njit\n" + uncached(grid_src) + "\n" + uncached(src[start:end]), ns)
    return ns


def load_python_pf() -> dict:
    src = cell_source(20)
    start = src.index("def run_particle_filter")
    end = src.index("\ndef run_pf_lik_ensemble_scales")
    ns: dict = {"np": np, "pd": pd}
    exec(src[start:end], ns)
    return ns


def main(argv: list[str]) -> int:
    wid = argv[1] if len(argv) > 1 else "000d7d20"
    n_seeds = int(argv[2]) if len(argv) > 2 else 128
    n_particles = 500

    hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv").drop(columns=["TVT"], errors="ignore")
    tw = pd.read_csv(DATA / f"{wid}__typewell.csv")

    nb_ns, py_ns = load_numba_pf(), load_python_pf()
    grid, allseeds = nb_ns["_grid"], nb_ns["_pf_lik_allseeds"]

    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    last = kn.iloc[-1]
    ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10.0, 60.0))
    tail = kn.tail(30)
    dt, dz, dm = (np.diff(tail.TVT_input.values), np.diff(tail.Z.values), np.diff(tail.MD.values))
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = grid(tw_tvt, tw_gr)
    gr_full = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_full.values.astype(float)[ev.index]
    md_v = ev.MD.values.astype(float)
    z_v = ev.Z.values.astype(float)

    args = (md_v, z_v, gr_v, gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, 0,
            PF_CONSTS["MOM"], PF_CONSTS["VN"], PF_CONSTS["PN"], PF_CONSTS["RP"],
            PF_CONSTS["RR"], PF_CONSTS["RESAMP"], PF_CONSTS["init_spr"])

    t0 = time.perf_counter()
    allseeds(md_v[:20], z_v[:20], gr_v[:20], gg, gmin, gst, gs, ls, ir, 64, 2, 0,
             *[PF_CONSTS[k] for k in ("MOM", "VN", "PN", "RP", "RR", "RESAMP", "init_spr")])
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    allseeds(*args)
    t_numba = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in range(n_seeds):
        py_ns["run_particle_filter"](hw, tw, n_particles=n_particles, seed=s)
    t_python = time.perf_counter() - t0

    print(f"well={wid}  eval_rows={len(md_v)}  seeds={n_seeds}  particles={n_particles}")
    print(f"  pure Python : {t_python:8.2f} s")
    print(f"  numba       : {t_numba:8.2f} s   (one-off compile {compile_s:.1f} s)")
    print(f"  speedup     : x{t_python / max(t_numba, 1e-9):.1f}")
    hidden = 200
    print(f"\nextrapolated over {hidden} hidden wells (marginal PF only):")
    print(f"  pure Python : {t_python * hidden / 3600:6.2f} h")
    print(f"  numba       : {t_numba * hidden / 3600:6.2f} h")
    print(f"  freed       : {(t_python - t_numba) * hidden / 3600:6.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
