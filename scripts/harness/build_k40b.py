"""Build the "k40b" harness variant: a SECOND, disjoint 40-well held-out set.

Purpose: k40 (K_WELLS=40, WELL_SEED=42, gs x1.45) was iterated against while
developing post-process methods. Any method tuned against k40 is no longer an
honest held-out test for it. k40b holds out a different 40 train wells --
verified to have ZERO overlap with k40's well set -- so a large structural
change can be judged on wells nothing has ever been iterated against.

This is a single-line-per-knob diff on top of the leaky harness, exactly like
build_variants.py's b145 variant, plus two more exact-string edits:
    1. gs x1.45          (same GS_OLD -> GS_NEW as the b145/production config)
    2. K_WELLS  10 -> 40  (the smoke default is K=10; a real CV run needs 40)
    3. WELL_SEED 42 -> 1031

Why seed 1031: the holdout mechanism (see HARNESS CONFIG cell in the leaky
notebook) is `np.random.default_rng(WELL_SEED).choice(eligible_wells, size=
K_WELLS, replace=False)` over the sorted list of eligible train wells (a
train well is eligible iff it has >=20 known rows and >=20 eval rows -- see
MIN_KNOWN_ROWS/MIN_EVAL_ROWS in that cell). That selection was replicated
byte-for-byte locally against the real train data (confirmed to reproduce
k40's exact 40 well ids at WELL_SEED=42, see replicate_selection.py in the
scratchpad), then searched over candidate seeds starting at 1000 until the
first one whose resulting 40-well set has zero intersection with k40's set.
1031 was the first hit. Because eligibility only depends on train data (which
does not change), this local replication is expected to reproduce exactly
what the Kaggle kernel itself will compute for HARNESS_WELL_IDS.

The leaky source notebook (scripts/harness/leaky/rogii-harness-leaky.ipynb)
is not currently present in the repo tree (likely swept by routine scratch
cleanup). This script therefore also accepts a scratch/pulled copy as a
fallback source -- see SRC_CANDIDATES below. Content was diffed cell-by-cell
against a fresh `kaggle kernels pull taichiiiii/rogii-harness-leaky` and is
byte-identical in every cell's source, so either copy is authoritative.
"""

from __future__ import annotations

import os
import ast
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SCRATCH = Path(
    os.environ.get("ROGII_WORK", "work")
)

# Prefer the in-repo copy (matches build_variants.py's own SRC path); fall
# back to scratch copies if the repo copy has been cleaned up.
SRC_CANDIDATES = [
    HERE / "leaky" / "rogii-harness-leaky.ipynb",
    SCRATCH / "k40b_work" / "leaky_pull" / "rogii-harness-leaky.ipynb",
    SCRATCH / "harness_audit" / "leaky" / "rogii-harness-leaky.ipynb",
]
META_CANDIDATES = [
    HERE / "leaky" / "kernel-metadata.json",
    SCRATCH / "k40b_work" / "leaky_pull" / "kernel-metadata.json",
    SCRATCH / "harness_audit" / "leaky" / "kernel-metadata.json",
]

# Output goes to the scratchpad (execution artifacts for this harness live
# there, not in the git-tracked repo tree -- matches how rogii-harness-b145*
# and rogii-rho-r* were built/pushed/downloaded).
OUT_ROOT = SCRATCH / "harness_audit"
SLUG = "rogii-harness-b145-k40b"

GS_OLD = "gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))"
GS_NEW = GS_OLD + " * 1.45"

K_OLD = (
    "K_WELLS = 10                      # number of train wells held out as "
    "pseudo-test (smoke default; raise for a real CV run)."
)
K_NEW = (
    "K_WELLS = 40                      # number of train wells held out as "
    "pseudo-test (smoke default; raise for a real CV run)."
)

SEED_OLD = "WELL_SEED = 42"
SEED_NEW = "WELL_SEED = 1031"

EDITS = [
    (GS_OLD, GS_NEW),
    (K_OLD, K_NEW),
    (SEED_OLD, SEED_NEW),
]


def _first_existing(candidates: list[Path]) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"none of {candidates} exist")


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
    src_path = _first_existing(SRC_CANDIDATES)
    meta_path = _first_existing(META_CANDIDATES)
    print(f"SRC:  {src_path}")
    print(f"META: {meta_path}")

    base = json.loads(src_path.read_text())
    meta = json.loads(meta_path.read_text())

    nb = json.loads(json.dumps(base))
    for old, new in EDITS:
        apply(nb, old, new)

    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            ast.parse("".join(c.get("source", [])))

    out = OUT_ROOT / SLUG
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb))

    m = dict(meta)
    m["id"] = f"taichiiiii/{SLUG}"
    m["title"] = SLUG
    m["code_file"] = f"{SLUG}.ipynb"
    m.pop("id_no", None)
    (out / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
    print(f"built {out}  (+gs1.45, K_WELLS 10->40, WELL_SEED 42->1031, {len(EDITS)} edit(s))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
