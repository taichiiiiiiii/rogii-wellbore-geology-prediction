"""Check whether local test/ wells duplicate train/ wells (leak investigation).

Forum claim under investigation: "Contact override for visible test wells
(RMSE ~0.01)" — the hypothesis is that some top public kernels special-case
known/visible test well IDs and read TVT straight out of a matching train
file (or an interpolated neighbor) instead of doing real GR<->typewell
registration, which would produce near-zero error on just those wells.

This script verifies, on our own data, whether:

    1. train/ and test/ well lists + CSV schemas match expectations.
    2. each test well's ID / XY trajectory / GR trace overlaps a train well:
       (a) exact well-ID string match
       (b) nearest-neighbor distance of head/tail XY coordinates vs ALL
           train wells (not just the ID match, to catch renamed duplicates)
       (c) GR full-trace match: summary-stat fingerprint (length, mean,
           std, head/tail-value hash) to report candidates cheaply, then a
           full nan-safe zero-lag correlation + exact-equality check
           against every same-length train well (773 wells is cheap enough
           to brute force directly; the fingerprint step is kept anyway as
           a documented/inspectable filter, per the task spec).
    3. counts how many test wells have their eval-zone (TVT_input NaN tail)
       TVT recoverable by reading a duplicate train well's TVT column at
       the same row index.
    4. prints a clear summary: exact dup / near dup / no match counts, plus
       a well-ID correspondence table.

We never read a "ground truth" TVT for a test well from anywhere other than
a train file (test files have no TVT column at all -- confirmed in step 1).
Reading train's own TVT to check whether a *duplicate* train well fully
covers the test eval zone is legitimate investigation, not label leakage
into a predictor.

    uv run python scripts/check_test_train_overlap.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii import data as D  # noqa: E402

EXACT_TOL = 1e-6  # float tolerance for "identical value" after CSV round-trip
HASH_DECIMALS = 6


def _value_hash(values: np.ndarray) -> str:
    """Order-sensitive hash of a rounded 1-D float array."""
    rounded = np.round(values.astype(np.float64), HASH_DECIMALS)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:12]


def summarize_gr(gr: np.ndarray) -> dict:
    """Cheap fingerprint of a GR trace: length, mean/std, head/tail hashes."""
    valid = gr[~np.isnan(gr)]
    head = valid[:5] if valid.size >= 5 else valid
    tail = valid[-5:] if valid.size >= 5 else valid
    return {
        "n": len(gr),
        "n_valid": int(valid.size),
        "mean": float(valid.mean()) if valid.size else float("nan"),
        "std": float(valid.std()) if valid.size else float("nan"),
        "head_hash": _value_hash(head),
        "tail_hash": _value_hash(tail),
    }


def nan_safe_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Pearson correlation at zero lag over positions valid in both arrays.

    Requires equal length (no stretch/warp — this is for detecting literal
    duplicates on the same 1ft MD grid, not registration).
    """
    if len(a) != len(b):
        return float("nan"), 0
    mask = ~(np.isnan(a) | np.isnan(b))
    n = int(mask.sum())
    if n < 10:
        return float("nan"), n
    aa, bb = a[mask], b[mask]
    if aa.std() == 0 or bb.std() == 0:
        return float("nan"), n
    return float(np.corrcoef(aa, bb)[0, 1]), n


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Max |a-b| over positions valid in both (requires equal length)."""
    if len(a) != len(b):
        return float("nan"), 0
    mask = ~(np.isnan(a) | np.isnan(b))
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.max(np.abs(a[mask] - b[mask]))), n


def print_schema() -> None:
    print("=== 1. Well lists & CSV schema ===")
    train_wells = D.list_wells("train")
    test_wells = D.list_wells("test")
    print(f"train wells: {len(train_wells)}")
    print(f"test wells:  {len(test_wells)} -> {test_wells}")

    h_train = D.load_horizontal(train_wells[0], "train")
    h_test = D.load_horizontal(test_wells[0], "test")
    tw_train = D.load_typewell(train_wells[0], "train")
    tw_test = D.load_typewell(test_wells[0], "test")
    print(f"train horizontal_well.csv columns: {list(h_train.columns)}")
    print(f"test  horizontal_well.csv columns: {list(h_test.columns)}")
    print(f"train typewell.csv columns:        {list(tw_train.columns)}")
    print(f"test  typewell.csv columns:        {list(tw_test.columns)}")
    print(
        "-> test horizontal file has NO TVT / formation columns "
        f"({sorted(set(h_train.columns) - set(h_test.columns))} dropped); "
        "test typewell has NO Geology column."
    )


def build_train_index(train_wells: list[str]) -> dict:
    """Load every train well once; cache endpoints, GR array, GR fingerprint."""
    index: dict[str, dict] = {}
    for w in train_wells:
        h = D.load_horizontal(w, "train")
        xy = h[["X", "Y"]].to_numpy(dtype=float)
        gr = h["GR"].to_numpy(dtype=float)
        index[w] = {
            "head_xy": xy[0],
            "tail_xy": xy[-1],
            "gr": gr,
            "md": h["MD"].to_numpy(dtype=float),
            "summary": summarize_gr(gr),
        }
    return index


def nearest_neighbor(point: np.ndarray, train_index: dict, key: str) -> tuple[str, float]:
    best_well, best_dist = None, float("inf")
    for w, rec in train_index.items():
        d = float(np.linalg.norm(rec[key] - point))
        if d < best_dist:
            best_well, best_dist = w, d
    assert best_well is not None
    return best_well, best_dist


def gr_full_match_report(
    test_well: str, gr_test: np.ndarray, train_index: dict
) -> list[dict]:
    """Brute-force zero-lag correlation against every same-length train well."""
    summary_test = summarize_gr(gr_test)
    matches = []
    for w, rec in train_index.items():
        if rec["summary"]["n"] != summary_test["n"]:
            continue  # cheap length prefilter -- most wells excluded here
        corr, n_overlap = nan_safe_corr(gr_test, rec["gr"])
        diff, n_diff = max_abs_diff(gr_test, rec["gr"])
        matches.append(
            {
                "train_well": w,
                "n_overlap_valid": n_overlap,
                "corr": corr,
                "max_abs_diff": diff,
                "n_diff_valid": n_diff,
                "exact_match": bool(n_diff > 0 and diff < EXACT_TOL),
                "same_id": w == test_well,
                "head_hash_match": rec["summary"]["head_hash"] == summary_test["head_hash"],
                "tail_hash_match": rec["summary"]["tail_hash"] == summary_test["tail_hash"],
            }
        )
    matches.sort(key=lambda m: (-1.0 if np.isnan(m["corr"]) else -m["corr"]))
    return matches


def main() -> None:
    print_schema()

    train_wells = D.list_wells("train")
    test_wells = D.list_wells("test")
    train_set = set(train_wells)

    print("\n=== 2a. Exact well-ID overlap ===")
    for w in test_wells:
        status = "EXACT ID MATCH in train" if w in train_set else "no ID match in train"
        print(f"  test/{w}: {status}")

    print("\nLoading + fingerprinting all train wells (one pass)...")
    train_index = build_train_index(train_wells)
    print(f"  cached {len(train_index)} train wells")

    per_well_results = []
    for tw in test_wells:
        h_test = D.load_horizontal(tw, "test")
        xy_test = h_test[["X", "Y"]].to_numpy(dtype=float)
        gr_test = h_test["GR"].to_numpy(dtype=float)

        head_well, head_dist = nearest_neighbor(xy_test[0], train_index, "head_xy")
        tail_well, tail_dist = nearest_neighbor(xy_test[-1], train_index, "tail_xy")
        gr_matches = gr_full_match_report(tw, gr_test, train_index)

        per_well_results.append(
            {
                "test_well": tw,
                "exact_id_match": tw in train_set,
                "head_nn_well": head_well,
                "head_nn_dist": head_dist,
                "tail_nn_well": tail_well,
                "tail_nn_dist": tail_dist,
                "gr_matches": gr_matches,
                "n_eval_rows": int(D.eval_mask(h_test).sum()),
            }
        )

    print("\n=== 2b/2c. Per-test-well coordinate + GR overlap ===")
    for r in per_well_results:
        print(f"\n--- test/{r['test_well']} ---")
        print(f"  exact well-ID match in train: {r['exact_id_match']}")
        print(
            f"  head-XY nearest train well: {r['head_nn_well']} "
            f"(dist={r['head_nn_dist']:.6f})"
        )
        print(
            f"  tail-XY nearest train well: {r['tail_nn_well']} "
            f"(dist={r['tail_nn_dist']:.6f})"
        )
        top = r["gr_matches"][:5]
        n_candidates = len(r["gr_matches"])
        print(f"  GR full-trace matches (top {len(top)} of {n_candidates} same-length candidates):")
        if not top:
            print("    (no train well shares this GR trace's row count)")
        for m in top:
            print(
                f"    train/{m['train_well']}: same_id={m['same_id']} "
                f"corr={m['corr']:.6f} n_overlap={m['n_overlap_valid']} "
                f"max_abs_diff={m['max_abs_diff']:.3g} exact_match={m['exact_match']} "
                f"head_hash={m['head_hash_match']} tail_hash={m['tail_hash_match']}"
            )

    print("\n=== 3. Eval-zone TVT recoverable from a duplicate train well? ===")
    n_full_dup = 0
    n_partial = 0
    n_none = 0
    dup_table = []
    for r in per_well_results:
        tw = r["test_well"]
        n_eval = r["n_eval_rows"]
        exact_dups = [m["train_well"] for m in r["gr_matches"] if m["exact_match"]]

        if not exact_dups:
            print(
                f"  test/{tw}: NO exact-duplicate train well found. "
                f"eval rows={n_eval} -> NOT recoverable."
            )
            n_none += 1
            dup_table.append((tw, None, "no match"))
            continue

        dup_well = exact_dups[0]
        h_test = D.load_horizontal(tw, "test")
        h_dup = D.load_horizontal(dup_well, "train")
        eval_mask_test = D.eval_mask(h_test)

        if len(h_dup) != len(h_test):
            print(
                f"  test/{tw}: exact GR dup {dup_well} but row-count mismatch, "
                "cannot align by index."
            )
            n_partial += 1
            dup_table.append((tw, dup_well, "partial (length mismatch)"))
            continue

        md_test = h_test["MD"].to_numpy(dtype=float)
        md_dup = h_dup["MD"].to_numpy(dtype=float)
        md_diff, _ = max_abs_diff(md_test, md_dup)
        tvt_from_train = h_dup["TVT"].to_numpy(dtype=float)[eval_mask_test]
        n_recoverable = int((~np.isnan(tvt_from_train)).sum())

        status = "FULL" if n_recoverable == n_eval else "PARTIAL"
        print(
            f"  test/{tw}: exact duplicate = train/{dup_well} (MD max_abs_diff={md_diff:.3g}). "
            f"eval rows={n_eval}, TVT recoverable={n_recoverable}/{n_eval} [{status}]"
        )
        if status == "FULL":
            n_full_dup += 1
        else:
            n_partial += 1
        dup_table.append((tw, dup_well, status))

    print("\n=== 4. Summary ===")
    print(f"完全重複 (exact GR/coord match, TVT fully recoverable from train): {n_full_dup}")
    print(f"近傍重複 (GR match found, TVT only partially recoverable):        {n_partial}")
    print(f"該当なし (no exact-duplicate train well found):                    {n_none}")
    print(f"(local test/ well count = {len(test_wells)})")

    print("\n代表例のウェルID対応表 (test -> matched train well -> status):")
    for tw, dup_well, status in dup_table:
        print(f"  test/{tw}  <->  train/{dup_well}  [{status}]")

    print(
        "\n注意: ローカル test/ には Kaggle が配布した例示 3 well しか無い(隠しテスト ~200 well は"
        "含まれない)。この結果は『例示 test well が train の完全複製である』ことの検証であり、"
        "実際に LB 採点される隠しテスト全体が同様に train と重複しているかは本スクリプトでは"
        "検証できない(データがローカルに存在しないため)。"
    )


if __name__ == "__main__":
    main()
