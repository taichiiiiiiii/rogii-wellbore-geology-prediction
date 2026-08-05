"""Rebuild the two notebooks that were selected as our final submissions.

Both are the upstream public notebook plus a handful of one-line edits. This script
applies those edits to a local copy of the upstream notebook and writes the two
kernels, so the submitted artefacts can be reconstructed exactly without this
repository redistributing anyone else's code.

    # 1. fetch the upstream notebook (Kaggle account required)
    kaggle kernels pull evgendvorkin/rogii-physics-lb-7-872-v48 -p upstream -m

    # 2. rebuild both final kernels
    python build_final_submissions.py upstream/rogii-physics-lb-7-872-v48.ipynb -o build

Every edit is anchored on an exact, unique source string and the script fails loudly
if an anchor is missing or ambiguous — an upstream revision must never be patched
silently into something that only looks right.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

# --- Edits shared by both kernels -------------------------------------------------
#
# 1-2. Kaggle mounts competition and dataset inputs under two different layouts and
#      which one a worker gets is not predictable: `/kaggle/input/competitions/<comp>`
#      and `/kaggle/input/datasets/<owner>/<slug>` (current), or a flat
#      `/kaggle/input/<slug>` (legacy). The upstream notebook hardcodes the current
#      layout in these two constants, so a legacy worker either dies in the first data
#      read or — worse, for the ridge artefacts, which are behind an .exists() guard —
#      silently drops a model component. Resolving at run time is a no-op on a current
#      worker and repairs the legacy one.
#
# 3.   The upstream "Q0522" cell pins the SHA of the reference submission and raises on
#      mismatch. Any knob that changes predictions trips it. Only the raise becomes a
#      print; every structural audit in that cell is left intact.
COMMON_EDITS: list[tuple[str, str]] = [
    (
        "COMPETITION_DATA_ROOT = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'",
        "COMPETITION_DATA_ROOT = next((p for p in ("
        "'/kaggle/input/competitions/rogii-wellbore-geology-prediction', "
        "'/kaggle/input/rogii-wellbore-geology-prediction') "
        "if __import__('os').path.exists(p)), "
        "'/kaggle/input/competitions/rogii-wellbore-geology-prediction')",
    ),
    (
        "RIDGE_ARTIFACT_ROOT = '/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts'",
        "RIDGE_ARTIFACT_ROOT = next((p for p in ("
        "'/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts', "
        "'/kaggle/input/wellbore-geology-prediction-artifacts') "
        "if __import__('os').path.exists(p)), "
        "'/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts')",
    ),
    (
        "if not (_sha_exact or _stats_exact):\n"
        "    raise RuntimeError(f'{_EX_LABEL}: source artifact mismatch: "
        "sha={_base_sha}, stats={_base_stats}')",
        "if not (_sha_exact or _stats_exact):\n"
        "    print(f'{_EX_LABEL}: source artifact differs from reference "
        "(expected: knob probe); continuing. sha={_base_sha}')",
    ),
]

# --- Edits only the seeds kernel needs ---------------------------------------------
#
# Raising the particle-filter seed count changes the posterior enough that the seed
# branch hedge stops firing, which trips two more upstream assertions further down the
# same cell. Both become prints, and the shift is skipped rather than read off an empty
# frame. Consequence worth knowing: this kernel does not apply the upstream's
# hardcoded per-well offset, so its predictions differ from the base config by more
# than the seed count alone.
SEEDS_GUARD_EDITS: list[tuple[str, str]] = [
    (
        "if len(_applied) != 1:\n"
        "    raise RuntimeError(f'{_EX_LABEL}: expected exactly one applied branch, "
        "got {len(_applied)}')",
        "_EX_SKIP_APPLY = (len(_applied) != 1)\n"
        "if _EX_SKIP_APPLY:\n"
        "    print(f'{_EX_LABEL}: no single applied branch (got {len(_applied)}) "
        "-- skipping Q0522 shift (PF-upstream knob probe)')\n"
        "    _applied = _ex_pd.DataFrame([{'well': _EX_EXPECTED_WELL, 'shift': 0.0, "
        "'moved_rows': 0}])",
    ),
    (
        "if _src_well != _EX_EXPECTED_WELL or abs(_src_shift - 2.0) > 1e-9 "
        "or _src_rows != _EX_EXPECTED_ROWS:\n"
        "    raise RuntimeError(f'{_EX_LABEL}: unexpected source branch: "
        "well={_src_well}, shift={_src_shift}, rows={_src_rows}')",
        "if _src_well != _EX_EXPECTED_WELL or abs(_src_shift - 2.0) > 1e-9 "
        "or _src_rows != _EX_EXPECTED_ROWS:\n"
        "    print(f'{_EX_LABEL}: source branch differs from reference "
        "(expected: PF-upstream knob probe); continuing. well={_src_well}, "
        "shift={_src_shift}, rows={_src_rows}')",
    ),
    (
        "_final_v[_mask] += _EX_EXTRA_SHIFT",
        "_final_v[_mask] += (0.0 if _EX_SKIP_APPLY else _EX_EXTRA_SHIFT)",
    ),
    (
        "if not _ex_np.allclose(_delta[_mask], _EX_EXTRA_SHIFT, atol=1e-10, rtol=0.0):",
        "if not _ex_np.allclose(_delta[_mask], "
        "(0.0 if _EX_SKIP_APPLY else _EX_EXTRA_SHIFT), atol=1e-10, rtol=0.0):",
    ),
]

# --- The two kernels ----------------------------------------------------------------
#
# slot1 lowers the degree of the trend fit; slot2 raises the particle-filter seed count.
# One knob each, on top of the shared edits.
KERNELS: dict[str, dict] = {
    "rogii-v2-projdeg2-probe": {
        "public_lb": 6.394,
        "submission_ref": "55252809",
        "edits": COMMON_EDITS
        + [("SP45_PROJECTION_DEGREE = 3", "SP45_PROJECTION_DEGREE = 2")],
    },
    "rogii-v2-seeds192-probe": {
        "public_lb": 6.389,
        "submission_ref": "55252812",
        "edits": COMMON_EDITS
        + SEEDS_GUARD_EDITS
        + [("SP45_SELECTOR_N_SEEDS = 128", "SP45_SELECTOR_N_SEEDS = 192")],
    },
}

KERNEL_METADATA = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [
        "phongnguyn23021656/koolbox-offline",
        "nina2025/rogii-03",
        "pilkwang/rogii-model-package",
        "thbdh5765/rogii-v10-fresh-artifacts",
        "fleongg/rogii-claude-models-pub",
        "needless090/rogii-tabicl-mirror",
        "ravaghi/wellbore-geology-prediction-artifacts",
    ],
    "kernel_sources": [],
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "model_sources": [],
}


def cell_source(cell: dict) -> str:
    return "".join(cell["source"])


def apply_edit(notebook: dict, old: str, new: str) -> None:
    """Apply one edit, insisting that its anchor is present exactly once."""
    hits = [
        i
        for i, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code" and old in cell_source(cell)
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"anchor found in {len(hits)} cells, expected exactly 1: {old[:70]!r}"
        )
    source = cell_source(notebook["cells"][hits[0]])
    if source.count(old) != 1:
        raise SystemExit(
            f"anchor occurs {source.count(old)}x inside one cell: {old[:70]!r}"
        )
    notebook["cells"][hits[0]]["source"] = source.replace(old, new).splitlines(
        keepends=True
    )


def build(upstream: Path, out_dir: Path) -> None:
    base = json.loads(upstream.read_text())
    owner = KERNEL_METADATA.get("owner", "YOUR_KAGGLE_USERNAME")

    for slug, spec in KERNELS.items():
        notebook = copy.deepcopy(base)
        for old, new in spec["edits"]:
            apply_edit(notebook, old, new)

        changed = sum(
            1
            for before, after in zip(base["cells"], notebook["cells"])
            for line_before, line_after in zip(
                cell_source(before).splitlines(), cell_source(after).splitlines()
            )
            if line_before != line_after
        )

        target = out_dir / slug
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{slug}.ipynb").write_text(
            json.dumps(notebook, ensure_ascii=False), encoding="utf-8"
        )
        (target / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": f"{owner}/{slug}",
                    "title": slug,
                    "code_file": f"{slug}.ipynb",
                    **KERNEL_METADATA,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"{slug}: {len(spec['edits'])} edits, {changed} changed lines, "
            f"public LB {spec['public_lb']} (submission {spec['submission_ref']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="upstream .ipynb pulled from Kaggle")
    parser.add_argument("-o", "--out", type=Path, default=Path("build"))
    args = parser.parse_args()
    build(args.upstream, args.out)


if __name__ == "__main__":
    main()
