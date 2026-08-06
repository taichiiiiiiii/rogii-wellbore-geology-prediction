"""Weight the variance-reduction blend by where the variance actually is.

The award write-up measures the PF seed spread as the best uncertainty signal on
this task: it correlates +0.48 with error *magnitude*, while nothing predicts the
error's *sign*. Variance reduction is therefore worth most exactly where the spread
is large, and a constant blend weight spends the same correction on points the
filter already agrees on.

`run_pf_lik_ensemble_scales` can already emit the per-point seed spread
(`pf_seed_std`, gated behind SELECTOR_PF_RETURN_STD), so the adaptive weight costs
nothing beyond turning that flag on:

    w_i = W_MAX * std_i / (std_i + S0)

S0 is the spread at which half the maximum weight is applied; the write-up puts the
typical per-well spread near 5 ft, so S0 = 5 centres the response on the observed
scale. Both knobs default to values that reproduce the unmodified pipeline.
"""

from __future__ import annotations

import os
import ast
import json
import shutil
from pathlib import Path

HERE = Path(os.environ.get("ROGII_WORK", "work"))
SRC = HERE / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
OUT = HERE / "adaptpf"
SLUG = "rogii-gs145-adaptpf"

STD_OFF = ("SELECTOR_PF_RETURN_STD = False", "SELECTOR_PF_RETURN_STD = True")
PF_OFF = ("PF_SCALE_PARTNER_WEIGHT = 0.35", "PF_SCALE_PARTNER_WEIGHT = 0.0")
CFG_ANCHOR = "SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'"
CFG_NEW = "PF_ADAPTIVE_MAX_WEIGHT = 0.50\nPF_ADAPTIVE_HALF_SPREAD = 5.0\n" + CFG_ANCHOR

BLEND_OLD = """    _pf_pw = float(globals().get('PF_SCALE_PARTNER_WEIGHT', 0.0))
    if _pf_pw > 0.0:
        _pf_partner = pf_by_scale.get(str(globals().get('PF_SCALE_PARTNER', 'pf_mean')))
        if _pf_partner is not None:
            base = (1.0 - _pf_pw) * base + _pf_pw * np.asarray(_pf_partner, dtype=float)
            globals()['_PF_PARTNER_FIRED'] = int(globals().get('_PF_PARTNER_FIRED', 0)) + 1"""

BLEND_NEW = """    _pf_pw = float(globals().get('PF_SCALE_PARTNER_WEIGHT', 0.0))
    _pf_aw = float(globals().get('PF_ADAPTIVE_MAX_WEIGHT', 0.0))
    if _pf_aw > 0.0:
        # Spend the correction where the seeds disagree: spread predicts error size.
        _pf_partner = pf_by_scale.get(str(globals().get('PF_SCALE_PARTNER', 'pf_mean')))
        _pf_std = pf_by_scale.get('pf_seed_std')
        if _pf_partner is not None and _pf_std is not None:
            _s = np.asarray(_pf_std, dtype=float)
            _s = np.where(np.isfinite(_s), np.maximum(_s, 0.0), 0.0)
            _s0 = max(float(globals().get('PF_ADAPTIVE_HALF_SPREAD', 5.0)), 1e-6)
            _w = _pf_aw * (_s / (_s + _s0))
            base = (1.0 - _w) * base + _w * np.asarray(_pf_partner, dtype=float)
            globals()['_PF_ADAPT_FIRED'] = int(globals().get('_PF_ADAPT_FIRED', 0)) + 1
            globals()['_PF_ADAPT_WBAR'] = float(np.mean(_w))
    elif _pf_pw > 0.0:
        _pf_partner = pf_by_scale.get(str(globals().get('PF_SCALE_PARTNER', 'pf_mean')))
        if _pf_partner is not None:
            base = (1.0 - _pf_pw) * base + _pf_pw * np.asarray(_pf_partner, dtype=float)
            globals()['_PF_PARTNER_FIRED'] = int(globals().get('_PF_PARTNER_FIRED', 0)) + 1"""

# Report the realised weight so firing is provable from the log, not inferred from
# output deltas — those are contaminated by the pipeline's own nondeterminism.
REPORT = """
_pf_adapt_n = int(globals().get('_PF_ADAPT_FIRED', 0))
print(f'adaptive PF blend: fired on {_pf_adapt_n} selector call(s), '
      f'last mean weight={globals().get("_PF_ADAPT_WBAR", float("nan")):.4f}, '
      f'max weight={PF_ADAPTIVE_MAX_WEIGHT}, half-spread={PF_ADAPTIVE_HALF_SPREAD} ft')
"""


def edit(nb: dict, old: str, new: str) -> None:
    hits = [i for i, c in enumerate(nb["cells"]) if old in "".join(c.get("source", []))]
    assert len(hits) == 1, f"expected 1 cell, got {hits} for {old[:60]!r}"
    i = hits[0]
    src = "".join(nb["cells"][i].get("source", []))
    assert src.count(old) == 1, f"cell {i}: {src.count(old)} occurrences"
    src = src.replace(old, new)
    ast.parse(src)
    nb["cells"][i]["source"] = src.splitlines(keepends=True)


def main() -> int:
    nb = json.loads(SRC.read_text())
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())

    edit(nb, *STD_OFF)
    edit(nb, *PF_OFF)
    edit(nb, CFG_ANCHOR, CFG_NEW)
    edit(nb, BLEND_OLD, BLEND_NEW)

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
    print(f"built {OUT / SLUG}  (seed-spread adaptive blend, max w=0.50, S0=5.0 ft)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
