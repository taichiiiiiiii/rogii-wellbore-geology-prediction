"""Build the acceptance-test harness variants from the leaky harness.

The harness descends from janson's base, which runs at gs x1.0, while our shipped
line is gs x1.45. Matching it costs one line and buys a third calibration point:
gs1.0 vs gs1.45 has a known hidden-set answer too (+0.17), so the gs1.0 run we
already have becomes evidence rather than a throwaway.

Variants (each a single-line diff on top of the gs1.45 baseline, exactly as the
submission probes were built):
    b145      gs x1.45                                     -> reference
    vpcons    visible_prefix_profile   -> 'conservative'   -> truth +0.72
    anchor    final_selection -> 'self_verified_anchor'    -> truth +1.05
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "leaky" / "rogii-harness-leaky.ipynb"

GS_OLD = "gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))"
GS_NEW = GS_OLD + " * 1.45"

PRESET = """    'vp_balanced_modelpkg_005': dict(
        visible_prefix_profile='balanced',
        visible_prefix_final_selection='profile',"""

VARIANTS = {
    "rogii-harness-b145": [],
    "rogii-harness-v145-vpcons": [
        (PRESET, PRESET.replace(
            "visible_prefix_profile='balanced'",
            "visible_prefix_profile='conservative'")),
    ],
    "rogii-harness-v145-anchor": [
        (
            PRESET,
            PRESET.replace(
                "visible_prefix_final_selection='profile'",
                "visible_prefix_final_selection='self_verified_anchor'",
            ),
        ),
    ],
}


def apply(nb: dict, old: str, new: str) -> None:
    """Apply an exact-string edit to the single cell that contains it."""
    hits = [i for i, c in enumerate(nb["cells"]) if old in "".join(c.get("source", []))]
    assert len(hits) == 1, f"expected 1 cell, got {hits} for {old[:60]!r}"
    i = hits[0]
    src = "".join(nb["cells"][i].get("source", []))
    assert src.count(old) == 1, f"cell {i}: {src.count(old)} occurrences"
    src = src.replace(old, new)
    ast.parse(src)
    nb["cells"][i]["source"] = src.splitlines(keepends=True)


def main() -> int:
    base = json.loads(SRC.read_text())
    meta = json.loads((HERE / "leaky" / "kernel-metadata.json").read_text())

    for slug, edits in VARIANTS.items():
        nb = json.loads(json.dumps(base))
        apply(nb, GS_OLD, GS_NEW)
        for old, new in edits:
            apply(nb, old, new)

        for c in nb["cells"]:
            if c.get("cell_type") == "code":
                ast.parse("".join(c.get("source", [])))

        out = HERE / slug
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        (out / f"{slug}.ipynb").write_text(json.dumps(nb))

        m = dict(meta)
        m["id"] = f"taichiiiii/{slug}"
        m["title"] = slug
        m["code_file"] = f"{slug}.ipynb"
        m.pop("id_no", None)
        (out / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
        print(f"built {out}  (+gs1.45, {len(edits)} extra edit(s))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
