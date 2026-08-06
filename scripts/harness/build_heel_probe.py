"""Build the heel-calibration activation probe.

Investigation (Issue-less spike, see analysis notes / session report): the shipped
notebook already contains a per-well GR heel-calibration function
(`_selector_apply_heel_calibration` in the "Ridge/PF and Selector Anchor" cell) that
fits an affine gain/offset (alpha, beta) between the horizontal well's known-prefix GR
and the typewell's GR, guarded by `RUN_HEEL_CALIBRATION = True`. That flag is a red
herring: the function is only ever CALLED from `selector_bimodal_scan` /
`_selector_gr_misfit`, both of which are only reached from `_bimodal_selector_weight`,
which only runs `if RUN_BIMODAL_DETECTOR`. Under the shipped profile
(`vp_balanced_modelpkg_005`), `run_bimodal_detector=False`, so heel calibration is
fully dead code -- it never touches `submission.csv`.

There is no flag-only way to activate it for ALL wells: the only wire from heel
calibration into a live prediction is through the bimodal-hedge branch of
`apply_selector_variant`, which is itself gated by a trigger (median/​p90 GR-diff and
big-diff-fraction thresholds) that most wells will never cross. The one-line change
below (`run_bimodal_detector=False -> True` inside the `vp_balanced_modelpkg_005`
preset, the SAME dict the notebook's `SUBMISSION_PROFILE` already selects) is the
minimal edit that makes heel calibration non-dead-code. It is NOT an isolated test of
heel calibration alone: flipping it also activates the entire bimodal delta_star hedge
(±20ft dz rescan, prefix-trust gate, forced/soft midpoint blending) for whichever
wells cross the trigger, and heel calibration only ever affects the GR array those
wells' hedge math consults -- so any LB delta this probe measures is a bundle effect,
not attributable to heel calibration in isolation.

Local evidence (scripts run standalone against the 773 train wells, see
`heel_local_evidence.py` in the investigation scratch dir) found the calibration
mechanism, as coded, is net HARMFUL by its own reported metric: dividing the observed
GR by the fitted alpha (mean 0.80, i.e. almost always < 1 -- 95% of wells) amplifies
residual noise faster than the affine fit removes bias (classic regression-attenuation
trap). 90.2% of wells get a WORSE in-sample GR-to-typewell RMSE after "calibration"
despite passing the alpha/beta sanity guard (which is nearly non-binding: 772/773
wells pass it), and only 5.4% / 3.1% of wells show an honest-split / true-eval-zone
improvement. This probe is built anyway, per the investigation brief, so it exists as
an artifact for review -- the accompanying report recommends NOT spending an LB slot
on it.

Base: rogii-gs145-pfmean035.ipynb with PF_SCALE_PARTNER_WEIGHT forced back to 0.0,
exactly the "isolate one change" move build_levelling_probe.py and
build_surface_pp_probe.py make on the same base notebook, so the only change vs. the
shipped gs1.45 line is the heel-activation edit.
"""

from __future__ import annotations

import os
import ast
import json
import shutil
from pathlib import Path

HERE = Path(os.environ.get("ROGII_WORK", "work"))
SRC = HERE / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
SLUG = "rogii-gs145-heel"
OUT = HERE / "heel_kernel" / SLUG

# Turn the pf_mean blend back off so this probe isolates one change, exactly as
# build_levelling_probe.py / build_surface_pp_probe.py do on the same base notebook.
PF_OFF = ("PF_SCALE_PARTNER_WEIGHT = 0.35", "PF_SCALE_PARTNER_WEIGHT = 0.0")

# The minimal heel-activation edit identified by the investigation: flip
# run_bimodal_detector inside the EXACT preset the notebook's SUBMISSION_PROFILE
# already selects ('vp_balanced_modelpkg_005'), leaving every other field in that
# preset (and every other preset) untouched. This is the only wire that makes
# RUN_HEEL_CALIBRATION (already True, but currently dead) reachable from a live
# prediction -- see the module docstring for why it is a bundle effect, not an
# isolated heel-calibration test.
HEEL_OLD = """    'vp_balanced_modelpkg_005': dict(
        visible_prefix_profile='balanced',
        visible_prefix_final_selection='profile',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=True,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=True,
        model_package_gated_max_weight=0.00425,
        model_package_gated_scale=6.0,
    ),"""
HEEL_NEW = HEEL_OLD.replace("run_bimodal_detector=False,", "run_bimodal_detector=True,")


def edit(nb: dict, old: str, new: str) -> None:
    """Apply an exact-string edit to the single cell that contains it."""
    hits = [i for i, c in enumerate(nb["cells"]) if old in "".join(c.get("source", []))]
    assert len(hits) == 1, f"expected exactly 1 cell, got {hits} for {old[:60]!r}"
    i = hits[0]
    src = "".join(nb["cells"][i].get("source", []))
    assert src.count(old) == 1, f"cell {i}: {src.count(old)} occurrences"
    src = src.replace(old, new)
    ast.parse(src)
    nb["cells"][i]["source"] = src.splitlines(keepends=True)


def main() -> int:
    nb = json.loads(SRC.read_text())
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())

    edit(nb, *PF_OFF)
    edit(nb, HEEL_OLD, HEEL_NEW)

    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            ast.parse("".join(c.get("source", [])))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb))

    m = dict(meta)
    m["id"] = f"taichiiiii/{SLUG}"
    m["title"] = SLUG
    m["code_file"] = f"{SLUG}.ipynb"
    m.pop("id_no", None)
    (OUT / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
    print(f"built {OUT}  (PF_SCALE_PARTNER_WEIGHT->0.0, run_bimodal_detector False->True "
          f"inside vp_balanced_modelpkg_005)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
