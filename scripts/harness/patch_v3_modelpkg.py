"""Drop the model-package layer from the harness and stop the prints from lying.

v2 reached the very last stage before dying inside pilkwang's model package: its
own feature builder covers the *real* test ids and returned nothing for our
pseudo-test wells ("Feature builder missing 50306 sample ids"). The package is
tied to the real test set, so it cannot run against a held-out-well pseudo-comp
at all — unlike the ridge artifacts, there is no OOF route around it. Keep it
disabled unconditionally; the notebook already has a graceful skip path
(MODEL_PACKAGE_REQUIRE=False -> model_package_available=False).

The comparison stays sound because the layer is absent from every variant alike,
but the harness now measures the pipeline up to submission_before_model_package.

Also fixes three prints that claimed artifacts were neutralised when the
KEEP_ARTIFACTS guard had in fact left them in place.
"""

# ruff: noqa: E501  -- the replacement blocks are literal notebook source; reflowing them
# would stop them matching the kernel that was actually pushed.

from __future__ import annotations

import ast
import json
from pathlib import Path

NB = Path(__file__).parent / "leaky" / "rogii-harness-leaky.ipynb"

EDITS = [
    # model package: remove the KEEP_ARTIFACTS escape hatch
    (
        """    if not globals().get('HARNESS_KEEP_ARTIFACTS', False):
        MODEL_PACKAGE_ROOTS = ()
    print('HARNESS: MODEL_PACKAGE_ROOTS cleared -> model package correction will skip '""",
        """    # Unconditional, even under HARNESS_KEEP_ARTIFACTS: the package's own feature
    # builder is bound to the real test ids and raises 'Feature builder missing
    # <N> sample ids' on a pseudo-comp. No OOF route exists around it, so the
    # harness measures the pipeline up to submission_before_model_package.csv.
    MODEL_PACKAGE_ROOTS = ()
    print('HARNESS: MODEL_PACKAGE_ROOTS cleared -> model package correction will skip '""",
    ),
    # honest reporting for the two artifacts we now keep
    (
        """    print(f'HARNESS: RIDGE_ARTIFACT_ROOT redirected to {RIDGE_ARTIFACT_ROOT} (absent) -> '
          f'CFG.artifacts_path .exists() checks are False -> from-scratch training on filtered train_df')""",
        """    if globals().get('HARNESS_KEEP_ARTIFACTS', False):
        print(f'HARNESS: RIDGE_ARTIFACT_ROOT KEPT at {RIDGE_ARTIFACT_ROOT} -> pretrained '
              f'boosters load from disk; held-out rows are served from their stored '
              f'GroupKFold out-of-fold predictions, so no retraining and no leak')
    else:
        print(f'HARNESS: RIDGE_ARTIFACT_ROOT redirected to {RIDGE_ARTIFACT_ROOT} (absent) -> '
              f'CFG.artifacts_path .exists() checks are False -> from-scratch training on filtered train_df')""",
    ),
    (
        """    print('HARNESS: LEARNED_MODEL_ROOTS cleared -> _find_models() returns None -> '
          'from-scratch train_stack() path.')""",
        """    if globals().get('HARNESS_KEEP_ARTIFACTS', False):
        print('HARNESS: LEARNED_MODEL_ROOTS KEPT -> pretrained trajectory boosters load; '
              'these are in-sample for the held-out wells (known, accepted leak) but avoid '
              'the from-scratch train_stack() path that reversed variant ordering in R59/R77.')
    else:
        print('HARNESS: LEARNED_MODEL_ROOTS cleared -> _find_models() returns None -> '
              'from-scratch train_stack() path.')""",
    ),
]


def main() -> int:
    nb = json.loads(NB.read_text())
    cells = nb["cells"]
    src = "".join(cells[9].get("source", []))

    for old, new in EDITS:
        n = src.count(old)
        assert n == 1, f"expected 1 hit, got {n} for: {old[:70]!r}"
        src = src.replace(old, new)

    ast.parse(src)
    cells[9]["source"] = src.splitlines(keepends=True)

    bad = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        try:
            ast.parse("".join(c.get("source", [])))
        except SyntaxError as e:
            bad.append((i, e))
    assert not bad, f"syntax errors: {bad}"

    NB.write_text(json.dumps(nb))
    print(f"patched cell 9 ({len(EDITS)} edits), {len(cells)} cells, 0 syntax errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
