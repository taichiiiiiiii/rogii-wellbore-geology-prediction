"""Mutation leak test for the tops-F variant (task step 4d).

Shift the 40 held-out wells' EVAL-zone truth by +777 ft in a copied cache
(levelling_predict_tops.make_shifted_wells builds it in memory: a fresh
Wells object with TVT mutated only at the held-out wells' eval rows, known
zone untouched), rebuild the donor field + network + prediction from that
copy, and assert every sp_* column is bit-identical to the unshifted run.

If the query well's own top columns, or its own eval-zone TVT, ever leaked
into the donor field, the network, or the anchor, this diff would be
nonzero: the donor field only ever contains OTHER (train) wells' rows
(levelling_common.donor_mask excludes the 40 held-out wells entirely), and
the anchor is fit from the KNOWN zone only, so an eval-zone-only shift must
be invisible to every spat column.

Usage:
    uv run python scripts/levelling_tops_mutation_test.py [shift_ft]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling_common import load  # noqa: E402
from levelling_predict import load_pipeline  # noqa: E402
from levelling_predict_tops import load_top_ref, make_shifted_wells, run_variant  # noqa: E402

# sp_oracle is EXCLUDED from the leak assertion by design: it is defined as
# robust_c(r[eval_zone]), i.e. it deliberately reads the eval-zone truth (a
# cheating diagnostic ceiling the original script's own docstring says is
# "never used downstream") -- it is EXPECTED to move by the shift amount, and
# doing so is itself a useful sanity check that the shift landed at all.
SP_COLS = ["sp_tail", "sp_tail100", "sp_tail300", "sp_tail1000", "sp_full", "sp_net"]
DIAGNOSTIC_COLS = ["sp_oracle"]


def main(argv: list[str]) -> int:
    shift = float(argv[1]) if len(argv) > 1 else 777.0
    w = load()
    top_ref = load_top_ref()
    _, held = load_pipeline()

    print(f"mutation leak test: shifting eval-zone truth of {len(held)} held-out wells "
          f"by {shift:+.1f} ft")
    w_shift = make_shifted_wells(w, set(held), shift)

    worst = 0.0
    for lev in ("none", "pair450"):
        base_df, _, _ = run_variant(w, top_ref, lev)
        shift_df, _, _ = run_variant(w_shift, top_ref, lev)
        assert (base_df["well"].to_numpy() == shift_df["well"].to_numpy()).all(), "row order mismatch"
        assert np.array_equal(base_df["pipe"].to_numpy(), shift_df["pipe"].to_numpy()), "pipe changed"
        assert not np.array_equal(base_df["truth"].to_numpy(), shift_df["truth"].to_numpy()), \
            "truth did NOT change -- shift had no effect, test is vacuous"

        max_diff = 0.0
        for c in SP_COLS:
            d = float(np.max(np.abs(base_df[c].to_numpy() - shift_df[c].to_numpy())))
            max_diff = max(max_diff, d)
            print(f"  lev={lev:8s} {c:12s} max|diff| = {d:.3e}")
        for c in DIAGNOSTIC_COLS:
            d = float(np.max(np.abs(base_df[c].to_numpy() - shift_df[c].to_numpy())))
            print(f"  lev={lev:8s} {c:12s} max|diff| = {d:.3e}  (diagnostic-only, "
                  f"expected to equal the shift)")
        worst = max(worst, max_diff)

    print(f"\nOVERALL max|diff| across real prediction sp_* columns, both levelling kinds: "
          f"{worst:.6e}")
    print("PASS: no leak" if worst == 0.0 else "FAIL: nonzero diff -- investigate")
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
