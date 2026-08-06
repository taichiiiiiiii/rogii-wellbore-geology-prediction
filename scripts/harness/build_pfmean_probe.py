"""Build the variance-reduction probes: average the PF seed cloud instead of only selecting from it.

`run_pf_lik_ensemble_scales` already computes, from one 128-seed cloud, every
likelihood-temperature reweighting *and* the plain seed mean:

    for scale in scales:  out[f'pf_scale_{scale:g}'] = (softmax(liks/scale) * preds).sum(0)
    out['pf_mean'] = preds.mean(0)

The selector then picks exactly one of these per bin. The independently
award-winning working note measures the seed spread at ~5 ft per well and reports
the one structural lever that transferred to the real leaderboard (-0.134) as
averaging decorrelated members rather than selecting one of them. `pf_mean` is
the maximal-averaging member of that same cloud, so blending it in is the
faithful translation of that lever into this pipeline — and it costs no extra
compute, which keeps the 9-hour budget untouched.

The knob defaults to 0.0, i.e. a provable no-op, so the diff cannot change the
main line by accident.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

HERE = Path("/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad")
SRC = HERE / "aggr" / "out" / "rogii-gs145-aggr-probe.ipynb"
OUT = HERE / "pfmean"

ANCHOR = """    base = np.asarray(base, dtype=float)
    tvt_beam = np.asarray(tvt_beam, dtype=float)"""

PATCH = """    base = np.asarray(base, dtype=float)
    # Variance reduction: mix in the plain seed mean of the same particle cloud.
    # Selecting one temperature keeps the per-seed luck; averaging cancels it.
    _pf_pw = float(globals().get('PF_SCALE_PARTNER_WEIGHT', 0.0))
    if _pf_pw > 0.0:
        _pf_partner = pf_by_scale.get(str(globals().get('PF_SCALE_PARTNER', 'pf_mean')))
        if _pf_partner is not None:
            base = (1.0 - _pf_pw) * base + _pf_pw * np.asarray(_pf_partner, dtype=float)
            globals()['_PF_PARTNER_FIRED'] = int(globals().get('_PF_PARTNER_FIRED', 0)) + 1
    tvt_beam = np.asarray(tvt_beam, dtype=float)"""

# The aggr probe flipped the profile to 'aggressive'; revert so these probes sit on the main line.
AGGR_OLD = "visible_prefix_profile='aggressive',"
AGGR_NEW = "visible_prefix_profile='balanced',"

VARIANTS = {"rogii-gs145-pfmean035": 0.35, "rogii-gs145-pfmean020": 0.20}


def edit(nb: dict, old: str, new: str) -> None:
    hits = [i for i, c in enumerate(nb["cells"]) if old in "".join(c.get("source", []))]
    assert len(hits) == 1, f"expected exactly 1 cell, got {hits} for {old[:50]!r}"
    i = hits[0]
    src = "".join(nb["cells"][i].get("source", []))
    assert src.count(old) == 1, f"cell {i}: {src.count(old)} occurrences"
    src = src.replace(old, new)
    ast.parse(src)
    nb["cells"][i]["source"] = src.splitlines(keepends=True)


def main() -> int:
    base_nb = json.loads(SRC.read_text())
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())

    for slug, weight in VARIANTS.items():
        nb = json.loads(json.dumps(base_nb))
        edit(nb, ANCHOR, PATCH)
        edit(nb, AGGR_OLD, AGGR_NEW)

        # Declare the knob next to the other selector constants.
        cfg_old = "SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'"
        cfg_new = (
            f"PF_SCALE_PARTNER = 'pf_mean'\nPF_SCALE_PARTNER_WEIGHT = {weight}\n" + cfg_old
        )
        edit(nb, cfg_old, cfg_new)

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
        print(f"built {d}  PF_SCALE_PARTNER_WEIGHT={weight}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
