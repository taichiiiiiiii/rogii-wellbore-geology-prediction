"""Split the particle-filter seed cloud across two process-noise regimes.

The award write-up's one lever that transferred to the real leaderboard (-0.134)
was a decorrelated particle-filter ensemble: a second filter at *lower process
noise*, averaged with the base at 0.65/0.35. The members make independent errors,
so averaging cancels variance.

Our pipeline has every likelihood temperature and the plain seed mean already, but
all of them are reweightings of one cloud run at one noise setting, so they are
strongly correlated with each other — pfmean is at best a weak echo of that lever.

Running a second full cloud would buy real decorrelation, but the timing says no:
the log shows 48.5 / 74.1 / 53.4 s per well for the 128-seed ensemble, so ~58.7 s
per well, which is ~3.26 h over the ~200 hidden wells. Doubling it does not fit
under the 9-hour cap.

Splitting the existing 128 seeds does. Half run at the current noise and half at
`PF_STIFF_SCALE` times it, so the cloud spans two regimes at exactly the same cost,
and both the likelihood-weighted averages and pf_mean become genuine decorrelated
ensembles. The knob defaults to 1.0, which is a provable no-op.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

HERE = Path("/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad")
SRC = HERE / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
OUT = HERE / "stiffsplit"

SIG_OLD = "def run_particle_filter(hw, tw, n_particles=500, seed=42):"
SIG_NEW = "def run_particle_filter(hw, tw, n_particles=500, seed=42, noise_scale=1.0):"

CONST_OLD = "    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5"
CONST_NEW = """    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5
    # A stiffer partner regime: same trajectories cost, independent errors.
    _ns = float(noise_scale)
    VN *= _ns; PN *= _ns; RP *= _ns; RR *= _ns"""

LOOP_OLD = """    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)"""
LOOP_NEW = """    _stiff = float(globals().get('PF_STIFF_SCALE', 1.0))
    _frac = float(globals().get('PF_STIFF_FRACTION', 0.5))
    _cut = n_seeds - int(round(n_seeds * _frac)) if _stiff != 1.0 else n_seeds
    _seen = globals().setdefault('_PF_SPLIT_SEEN', {})
    _key = (int(n_seeds), int(_cut), int(n_seeds - _cut))
    _seen[_key] = _seen.get(_key, 0) + 1
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s,
                                    noise_scale=1.0 if s < _cut else _stiff)"""

# Firing has to be provable from the log. Output deltas are contaminated by the
# pipeline's nondeterminism, and so is elapsed time: the identical half-and-half
# build measured 204.1 s of PF on one run and 172.9 s on the next, against a
# 156.4-180.5 s control band across five unmodified runs. Neither signal can carry
# the claim, so the kernel reports the split it actually used.
REPORT = """
# The pipeline calls the PF ensemble with different seed counts per stage, so report
# every distinct split that ran, not just the last one.
_seen = globals().get('_PF_SPLIT_SEEN')
if not _seen:
    raise RuntimeError('stiffsplit: the seed loop never ran -- the probe is a no-op')
print(f'stiffsplit: PF_STIFF_SCALE={PF_STIFF_SCALE} PF_STIFF_FRACTION={PF_STIFF_FRACTION}')
for (_n, _b, _st), _calls in sorted(_seen.items()):
    print(f'  n_seeds={_n:4d}: {_b} base + {_st} stiff, {_calls} call(s)')
if all(_st == 0 for (_n, _b, _st) in _seen):
    raise RuntimeError('stiffsplit: no seeds ran stiff -- the probe is a no-op')
if not any(_n >= 128 for (_n, _b, _st) in _seen):
    raise RuntimeError('stiffsplit: the 128-seed main stage never split -- check the patch')
"""

# Keep this probe isolated: no pf_mean blend on top (corrections do not compose).
PF_OFF = ("PF_SCALE_PARTNER_WEIGHT = 0.35", "PF_SCALE_PARTNER_WEIGHT = 0.0")
CFG_ANCHOR = "SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'"

# slug -> (noise scale for the stiff half, fraction of seeds run stiff)
# The 0.25-fraction build is the runtime hedge: the visible run puts the half-and-half
# split at 204.1 s of PF against 156.4-180.5 s for every unmodified run, i.e. ~13% over
# the highest of them, which is ~0.5 h more across the ~200 hidden wells.
VARIANTS = {
    "rogii-gs145-stiffsplit05": (0.5, 0.50),
    "rogii-gs145-stiffsplit03": (0.3, 0.50),
    "rogii-gs145-stiffsplit05q": (0.5, 0.25),
}


def edit(nb: dict, old: str, new: str, expect: int = 1) -> None:
    hits = [i for i, c in enumerate(nb["cells"]) if old in "".join(c.get("source", []))]
    assert len(hits) == 1, f"expected 1 cell, got {hits} for {old[:60]!r}"
    i = hits[0]
    src = "".join(nb["cells"][i].get("source", []))
    assert src.count(old) == expect, f"cell {i}: {src.count(old)} occurrences, expected {expect}"
    src = src.replace(old, new)
    ast.parse(src)
    nb["cells"][i]["source"] = src.splitlines(keepends=True)


def main() -> int:
    base = json.loads(SRC.read_text())
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())

    for slug, (stiff, frac) in VARIANTS.items():
        nb = json.loads(json.dumps(base))
        edit(nb, SIG_OLD, SIG_NEW)
        edit(nb, CONST_OLD, CONST_NEW)
        # both run_pf_lik_ensemble and run_pf_lik_ensemble_scales share the loop
        edit(nb, LOOP_OLD, LOOP_NEW, expect=2)
        edit(nb, *PF_OFF)
        edit(nb, CFG_ANCHOR,
             f"PF_STIFF_SCALE = {stiff}\nPF_STIFF_FRACTION = {frac}\n" + CFG_ANCHOR)

        ast.parse(REPORT)
        nb["cells"].append({
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": REPORT.splitlines(keepends=True),
        })

        for c in nb["cells"]:
            if c.get("cell_type") == "code":
                ast.parse("".join(c.get("source", [])))

        d = OUT / slug
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        (d / f"{slug}.ipynb").write_text(json.dumps(nb))
        m = dict(meta)
        m["id"] = f"taichiiiii/{slug}"
        m["title"] = slug
        m["code_file"] = f"{slug}.ipynb"
        m.pop("id_no", None)
        (d / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
        print(f"built {d}  PF_STIFF_SCALE={stiff}  PF_STIFF_FRACTION={frac}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
