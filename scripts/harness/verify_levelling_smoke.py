"""Smoke-test the levelling post-process cell built by build_levelling_probe.py.

Per this repo's verification discipline, firing can only be demonstrated by a
static code path or an in-kernel print -- never by output similarity or
elapsed time. This script extracts the boot cell and the levelling cell
verbatim from the built notebook and exec()s them in an isolated namespace
against the real local data (773 train wells + the 3 local test wells), so
the printed [LEVELLING] lines and the submission.csv it writes are the actual
in-kernel output, not a re-implementation of the logic under test.

Cannot run the full ~8.3h pipeline locally, so the "base prediction" the
post-process reads is a synthetic carry-last submission.csv built directly
from the 3 local test wells' known TVT_input -- only the post-process cell's
own plumbing is under test here, not the upstream model.

Three checks:
  1. identity  -- LEVELLING_ENABLED=False must not touch submission.csv at all
                  (byte-identical before/after) and must print DISABLED.
  2. full run  -- LEVELLING_ENABLED=True against real local data must execute
                  end to end, print a [LEVELLING] n_donor_wells=... summary
                  line, write a finite, correctly-shaped submission.csv, and
                  its levelling_report.csv must show a per-well anchor row.
  3. time guard -- forcing _LEVELLING_T0 to 9h in the past must print
                  SKIPPED_TIME_GUARD and leave submission.csv byte-identical,
                  proving the runtime guard fires from elapsed time alone.

Usage:
    uv run python scripts/harness/verify_levelling_smoke.py
"""

from __future__ import annotations

import os
import hashlib
import io
import json
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("ROGII_WORK", "work"))
NB_PATH = SCRATCH / "levelling_kernel" / "rogii-gs145-levelling" / "rogii-gs145-levelling.ipynb"
DATA_ROOT = REPO / "data" / "raw"
WORKDIR = SCRATCH / "levelling_kernel" / "smoke_work"

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]


def extract_cells(nb_path: Path) -> tuple[str, str]:
    """Return (boot_cell_source, levelling_cell_source) from the built notebook."""
    nb = json.loads(nb_path.read_text())
    boot, levelling = None, None
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "_LEVELLING_T0 = _levelling_boot_time.time()" in src:
            boot = src
        if "[LEVELLING] params enabled=" in src:
            levelling = src
    if boot is None or levelling is None:
        raise RuntimeError(f"could not locate both cells in {nb_path} "
                            f"(boot={boot is not None} levelling={levelling is not None})")
    return boot, levelling


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_synthetic_submission() -> pd.DataFrame:
    """Carry-last synthetic base predictions for the 3 local test wells'
    eval zones, using only the real known-prefix data (no model needed)."""
    rows = []
    for wid in TEST_WELLS:
        h = pd.read_csv(DATA_ROOT / "test" / f"{wid}__horizontal_well.csv")
        known = h["TVT_input"].dropna()
        last_known = float(known.iloc[-1])
        eval_idx = h.index[h["TVT_input"].isna()]
        for i in eval_idx:
            rows.append({"id": f"{wid}_{i}", "tvt": last_known})
    df = pd.DataFrame(rows)
    sample = pd.read_csv(DATA_ROOT / "sample_submission.csv", dtype={"id": "string"})
    assert set(df["id"]) == set(sample["id"].astype(str)), "id set mismatch vs sample_submission.csv"
    # match sample_submission.csv row order exactly, as the real pipeline would
    df = df.set_index("id").loc[sample["id"].astype(str)].reset_index()
    return df[["id", "tvt"]]


def fresh_workdir(name: str) -> Path:
    d = WORKDIR / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def run_cell(boot_src: str, levelling_src: str, ns: dict) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(boot_src, "<boot_cell>", "exec"), ns)
        exec(compile(levelling_src, "<levelling_cell>", "exec"), ns)
    return buf.getvalue()


def check_identity(boot_src: str, levelling_src: str, base_df: pd.DataFrame) -> bool:
    print("\n=== Check 1: identity when LEVELLING_ENABLED=False ===")
    work = fresh_workdir("identity")
    sub_path = work / "submission.csv"
    base_df.to_csv(sub_path, index=False)
    before_bytes = sub_path.read_bytes()
    before_hash = sha256_bytes(before_bytes)

    ns = {"OUTPUT_DIR": work, "COMPETITION_DATA_ROOT": str(DATA_ROOT),
          "LEVELLING_ENABLED": False}
    out = run_cell(boot_src, levelling_src, ns)
    print(out.strip())

    after_bytes = sub_path.read_bytes()
    after_hash = sha256_bytes(after_bytes)
    ok = (before_hash == after_hash) and ("[LEVELLING] DISABLED" in out)
    print(f"identity byte-for-byte: {before_hash == after_hash}  "
          f"(sha256 before={before_hash[:12]} after={after_hash[:12]})")
    print(f"printed DISABLED line: {'[LEVELLING] DISABLED' in out}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def check_full_run(boot_src: str, levelling_src: str, base_df: pd.DataFrame) -> bool:
    print("\n=== Check 2: full run against real local data (LEVELLING_ENABLED=True) ===")
    work = fresh_workdir("full_run")
    sub_path = work / "submission.csv"
    base_df.to_csv(sub_path, index=False)
    before = pd.read_csv(sub_path)

    ns = {"OUTPUT_DIR": work, "COMPETITION_DATA_ROOT": str(DATA_ROOT),
          "LEVELLING_ENABLED": True}
    t_wall = time.time()
    out = run_cell(boot_src, levelling_src, ns)
    wall_s = time.time() - t_wall
    print(out.strip())
    print(f"(wall clock for this check: {wall_s:.1f}s)")

    fired = "[LEVELLING] n_donor_wells=" in out
    guard_or_disabled = ("SKIPPED_TIME_GUARD" in out) or ("DISABLED" in out)
    if not fired:
        print("RESULT: FAIL (expected a 'donors=' line; "
              f"guard/disabled instead: {guard_or_disabled})")
        return False

    after = pd.read_csv(sub_path)
    same_ids = list(after["id"]) == list(before["id"])
    finite = after["tvt"].notna().all()
    n_changed = int((after["tvt"].to_numpy() != before["tvt"].to_numpy()).sum())
    print(f"submission.csv: rows={len(after)} same_id_order={same_ids} "
          f"all_finite={bool(finite)} rows_changed_vs_before={n_changed}")

    report_path = work / "levelling_report.csv"
    report_ok = report_path.exists()
    if report_ok:
        rep = pd.read_csv(report_path)
        print(f"levelling_report.csv: {len(rep)} wells, "
              f"blended={int(rep['blended'].sum()) if 'blended' in rep else 'n/a'}")
        print(rep.to_string(index=False))

    before_after_path = work / "submission_before_levelling.csv"
    before_after_ok = before_after_path.exists()

    ok = same_ids and finite and report_ok and before_after_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def check_time_guard(boot_src: str, levelling_src: str, base_df: pd.DataFrame) -> bool:
    print("\n=== Check 3: elapsed-time guard trips when t0 is 9h in the past ===")
    work = fresh_workdir("time_guard")
    sub_path = work / "submission.csv"
    base_df.to_csv(sub_path, index=False)
    before_hash = sha256_bytes(sub_path.read_bytes())

    ns = {"OUTPUT_DIR": work, "COMPETITION_DATA_ROOT": str(DATA_ROOT),
          "LEVELLING_ENABLED": True}
    # Run the boot cell, then force it to look like the notebook started 9h ago.
    exec(compile(boot_src, "<boot_cell>", "exec"), ns)
    ns["_LEVELLING_T0"] = time.time() - 9 * 3600.0
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(levelling_src, "<levelling_cell>", "exec"), ns)
    out = buf.getvalue()
    print(out.strip())

    after_hash = sha256_bytes(sub_path.read_bytes())
    tripped = "[LEVELLING] SKIPPED_TIME_GUARD" in out
    untouched = before_hash == after_hash
    print(f"guard line printed: {tripped}  submission.csv untouched: {untouched}")
    ok = tripped and untouched
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    boot_src, levelling_src = extract_cells(NB_PATH)
    print(f"extracted boot cell ({len(boot_src)} chars) and levelling cell "
          f"({len(levelling_src)} chars) from {NB_PATH}")

    base_df = build_synthetic_submission()
    print(f"synthetic base submission: {len(base_df)} rows across {len(TEST_WELLS)} wells "
          f"(carry-last from real known TVT_input, no model)")

    results = {
        "identity": check_identity(boot_src, levelling_src, base_df),
        "full_run": check_full_run(boot_src, levelling_src, base_df),
        "time_guard": check_time_guard(boot_src, levelling_src, base_df),
    }

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
