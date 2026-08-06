"""Hoist the seed-independent half of the particle filter out of the 128-seed loop.

run12 was rejected for exceeding the runtime, and it is a byte-identical replicate of
the main line, so the baseline itself sits at roughly 8.3 h against a 9 h cap. Every
draw carries an ~8% chance of scoring nothing, and there is no room to add the second
particle cloud that the real decorrelation lever needs.

`run_particle_filter` does a block of DataFrame work before it ever touches the seed:
sorting the typewell, interpolating the full-length GR trace, computing gs and the
initial rate, and materialising md/z/gr arrays for the eval zone. None of it depends
on `seed`, and the ensemble runs it 128 times per well.

Splitting the function at `rng = np.random.default_rng(seed)` into a prepare step and
a seed-dependent core, then calling prepare once per well, changes nothing about the
result — same arrays, same seeds, same arithmetic — and removes 127 redundant passes.
`run_particle_filter` stays as a wrapper so every existing call site is untouched.

The kernel prints the realised prepare/core call counts, because neither output
deltas nor elapsed time can establish firing in this pipeline.
"""

from __future__ import annotations

import os
import ast
import json
import shutil
from pathlib import Path

HERE = Path(os.environ.get("ROGII_WORK", "work"))
SRC = HERE / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
OUT = HERE / "pfhoist"
SLUG = "rogii-gs145-pfhoist"

PF_OFF = ("PF_SCALE_PARTNER_WEIGHT = 0.35", "PF_SCALE_PARTNER_WEIGHT = 0.0")
SPLIT_AT = "    rng = np.random.default_rng(seed)"

PREPARE_HEAD = "def _pf_prepare(hw, tw, n_particles=500):"
# An empty eval zone has to keep returning the original contract, i.e. the same value
# `run_particle_filter` returned before the split: (hw['TVT_input'] as float, 0.0). The
# core itself never sees `hw`, so `_pf_prepare` packs that fallback into `_P` as
# `(None, fallback)` instead of bare `None`, and the core checks and returns *before*
# ever touching the 12-field unpack below (the original bug spliced this check in after
# the unpack, where `_P is None` could never be reached: unpacking None raises first).
CORE_HEAD = """def _pf_core(_P, seed, n_particles=500):
    globals()['_PF_CORE_CALLS'] = int(globals().get('_PF_CORE_CALLS', 0)) + 1
    if _P[0] is None:
        return _P[1], 0.0
    (tw_tvt, tw_gr, last_tvt, last_Z, last_MD, gs, ir,
     md_v, z_v, gr_v, out_vals_src, ev_index) = _P
    out_vals = out_vals_src.copy()
    N = n_particles"""

WRAPPER = """

def run_particle_filter(hw, tw, n_particles=500, seed=42):
    \"\"\"Unchanged contract; the per-well preparation is just no longer inlined.\"\"\"
    return _pf_core(_pf_prepare(hw, tw, n_particles), seed, n_particles)
"""

LOOP_OLD = """    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)"""
LOOP_NEW = """    _P = _pf_prepare(hw, tw, n_particles)
    for s in range(n_seeds):
        p, ll = _pf_core(_P, s, n_particles)"""

REPORT = """
_prep = int(globals().get('_PF_PREPARE_CALLS', 0))
_core = int(globals().get('_PF_CORE_CALLS', 0))
if _core == 0:
    raise RuntimeError('pfhoist: the PF core never ran')
print(f'pfhoist: prepare={_prep} core={_core} ratio={_core / max(_prep, 1):.1f} '
      f'(unhoisted would be prepare==core)')
if _core <= _prep:
    raise RuntimeError('pfhoist: preparation was not hoisted out of the seed loop')
"""


def edit(nb: dict, old: str, new: str, expect: int = 1) -> None:
    hits = [i for i, c in enumerate(nb["cells"]) if old in "".join(c.get("source", []))]
    assert len(hits) == 1, f"expected 1 cell, got {hits} for {old[:60]!r}"
    i = hits[0]
    src = "".join(nb["cells"][i].get("source", []))
    assert src.count(old) == expect, f"cell {i}: {src.count(old)} occurrences"
    src = src.replace(old, new)
    ast.parse(src)
    nb["cells"][i]["source"] = src.splitlines(keepends=True)


def rewrite_pf(src: str) -> str:
    start = src.index("def run_particle_filter")
    end = src.index("\ndef ", start + 10)
    fn = src[start:end]

    k = fn.index(SPLIT_AT)
    head = fn[:k]           # signature + all seed-independent preparation
    tail = fn[k:]           # rng onwards

    # The head keeps its body but returns the prepared state instead of falling through.
    # The empty-eval-zone branch packs its fallback value into the returned tuple
    # (None, fallback) rather than returning bare None, so `_pf_core` -- which never
    # sees `hw` -- can still honour the original contract on its own.
    body = head.split("\n", 1)[1]
    body = body.replace(
        "        return hw['TVT_input'].values.astype(float).copy(), 0.0",
        "        return None, hw['TVT_input'].values.astype(float).copy()",
    )
    body = body.rstrip()
    assert "N   = n_particles" in body, "unexpected preparation body"
    body = body.replace("    N   = n_particles", "")

    prepare = (
        PREPARE_HEAD
        + "\n    globals()['_PF_PREPARE_CALLS'] = int(globals().get('_PF_PREPARE_CALLS', 0)) + 1\n"
        + body.rstrip()
        + "\n\n    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())\n"
        + "    return (tw_tvt, tw_gr, last_tvt, last_Z, last_MD, gs, ir,\n"
        + "            ev['MD'].values.astype(float), ev['Z'].values.astype(float),\n"
        + "            gr_interp.values.astype(float)[ev.index],\n"
        + "            hw['TVT_input'].values.astype(float).copy(), list(ev.index))\n"
    )

    core = tail
    for drop in (
        "    md_v = ev['MD'].values.astype(float)\n",
        "    z_v  = ev['Z'].values.astype(float)\n",
        "    # Interpolate GR gaps before tracking\n",
        "    gr_interp = hw['GR'].interpolate(limit_direction='both')"
        ".fillna(tw_gr.mean())\n",
        "    gr_v = gr_interp.values.astype(float)[ev.index]\n",
        "    out_vals = hw['TVT_input'].values.astype(float).copy()\n",
    ):
        core = core.replace(drop, "")
    core = core.replace("    out_vals[list(ev.index)] = res", "    out_vals[ev_index] = res")
    core = core.replace("    res = np.empty(len(ev))", "    res = np.empty(len(md_v))")
    core = core.replace("    for i in range(len(ev)):", "    for i in range(len(md_v)):")
    # CORE_HEAD already contains the empty-eval-zone guard, positioned before the
    # 12-field unpack (see the comment above CORE_HEAD's definition).
    core = CORE_HEAD + "\n" + core.rstrip() + "\n"

    new_fn = prepare + "\n\n" + core + WRAPPER
    ast.parse(new_fn)
    return src[:start] + new_fn + src[end:]


def main() -> int:
    nb = json.loads(SRC.read_text())
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())

    hits = [i for i, c in enumerate(nb["cells"])
            if "def run_particle_filter" in "".join(c.get("source", []))]
    assert len(hits) == 1, f"run_particle_filter in cells {hits}"
    src = "".join(nb["cells"][hits[0]].get("source", []))
    src = rewrite_pf(src)
    assert src.count(LOOP_OLD) == 2, f"seed loops found: {src.count(LOOP_OLD)}"
    src = src.replace(LOOP_OLD, LOOP_NEW)
    ast.parse(src)
    nb["cells"][hits[0]]["source"] = src.splitlines(keepends=True)

    edit(nb, *PF_OFF)

    ast.parse(REPORT)
    nb["cells"].append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": REPORT.splitlines(keepends=True),
    })

    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            ast.parse("".join(c.get("source", [])))

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / SLUG).mkdir(parents=True)
    (OUT / SLUG / f"{SLUG}.ipynb").write_text(json.dumps(nb))
    m = dict(meta)
    m["id"] = f"taichiiiii/{SLUG}"
    m["title"] = SLUG
    m["code_file"] = f"{SLUG}.ipynb"
    m.pop("id_no", None)
    (OUT / SLUG / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
    print(f"built {OUT / SLUG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
