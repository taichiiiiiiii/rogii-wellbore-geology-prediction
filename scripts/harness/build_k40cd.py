"""Build the "k40c" and "k40d" harness variants: a THIRD and FOURTH, mutually
disjoint 40-well held-out sets, on top of the existing k40 (WELL_SEED=42) and
k40b (WELL_SEED=1031) from build_variants.py / build_k40b.py.

Purpose: with Kaggle GPU quota exhausted, several judging runs need to be
queued up to fire back-to-back once quota frees (see
scripts/harness/run_holdout_queue.sh). Having 4 mutually disjoint 40-well
holdout sets (160 distinct wells, no well iterated against in more than one
set) means each run is an honest, independent held-out test -- no method can
have been tuned against wells it will be judged on, across ANY of the 4 sets.

This is a single-line-per-knob diff on top of the leaky harness, identical in
structure to build_k40b.py's edits, applied twice (once per variant):
    1. gs x1.45          (same GS_OLD -> GS_NEW as the b145/production config)
    2. K_WELLS  10 -> 40  (the smoke default is K=10; a real CV run needs 40)
    3. WELL_SEED 42 -> {1031, 2083, 3457 depending on target variant}

Why seeds 2083 / 3457: same selection mechanism as k40b (see
scripts/harness/build_k40b.py's docstring and
k40b_work/replicate_selection.py) -- `np.random.default_rng(WELL_SEED).choice
(eligible_wells, size=K_WELLS, replace=False)` over the sorted list of 773
eligible train wells (a train well is eligible iff it has >=20 known rows and
>=20 eval rows). Replicated locally against data/raw/train/ (confirmed to
reproduce k40's exact well set at WELL_SEED=42 and k40b's at WELL_SEED=1031),
then swept forward from seed 2000 (well clear of k40b's own 1000-up search
range, to avoid any accidental collision in how 1031 was found): the first
seed whose 40-well set is disjoint from {k40, k40b} is 2083 (-> k40c); the
sweep then continues and the first seed also disjoint from {k40, k40b, k40c}
is 3457 (-> k40d). See k40cd_work/find_seeds.py in the scratchpad for the
sweep and k40cd_work/seeds_and_wells.json for the full well-id lists and the
6-way pairwise intersection proof (all zero, union size 160).

Source notebook fallback chain and metadata handling are identical to
build_k40b.py -- see that docstring for why the SRC_CANDIDATES list exists.
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
# there, not in the git-tracked repo tree -- matches how rogii-harness-b145*,
# rogii-harness-b145-k40b, and rogii-rho-r* were built/pushed/downloaded).
OUT_ROOT = SCRATCH / "harness_audit"

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

VARIANTS = [
    {"name": "k40c", "seed": 2083},
    {"name": "k40d", "seed": 3457},
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


def build_variant(base: dict, meta: dict, name: str, seed: int) -> Path:
    slug = f"rogii-harness-b145-{name}"
    seed_new = f"WELL_SEED = {seed}"
    edits = [
        (GS_OLD, GS_NEW),
        (K_OLD, K_NEW),
        (SEED_OLD, seed_new),
    ]

    nb = json.loads(json.dumps(base))
    for old, new in edits:
        apply(nb, old, new)

    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            ast.parse("".join(c.get("source", [])))

    out = OUT_ROOT / slug
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb))

    m = dict(meta)
    m["id"] = f"taichiiiii/{slug}"
    m["title"] = slug
    m["code_file"] = f"{slug}.ipynb"
    m.pop("id_no", None)
    (out / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
    print(
        f"built {out}  (+gs1.45, K_WELLS 10->40, WELL_SEED 42->{seed}, "
        f"{len(edits)} edit(s))"
    )
    return out


def main() -> int:
    src_path = _first_existing(SRC_CANDIDATES)
    meta_path = _first_existing(META_CANDIDATES)
    print(f"SRC:  {src_path}")
    print(f"META: {meta_path}")

    base = json.loads(src_path.read_text())
    meta = json.loads(meta_path.read_text())

    for variant in VARIANTS:
        build_variant(base, meta, variant["name"], variant["seed"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
