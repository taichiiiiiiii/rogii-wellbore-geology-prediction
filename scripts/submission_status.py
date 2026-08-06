"""Group every submission by the kernel that produced it, and flag the unscored ones.

Three ways to get this wrong have already bitten, so each is handled explicitly:

  * `competition_submissions` defaults to page_size=20 against a 58-submission history,
    silently dropping the oldest — including our best draw (R105).
  * `status` reads COMPLETE for a run that exceeded the runtime cap; the reliable test
    for "not scored" is an empty `public_score`, and the reason lives on
    `error_description`, not `errorDescription` (R104).
  * Matching configs by description text picks up submissions that merely mention
    another config in prose — it reported GS1.3's best as 6.411 when 6.411 is a GS1.45
    draw whose description says "single-line diff from the iaztec GS1.3 fork". The
    submission's `url` carries the kernel slug, which is unambiguous.

Four more ways found in the 07-30 adversarial review, each guarded below:

  * `page_size=100` was only *incidentally* safe (58 now, <=88 possible by the deadline).
    Truncation is silent and `page_number` is a no-op on this endpoint (pages 1..5 all
    return the same first page), so there is no pagination fallback. Ask for 1000 and
    assert the result came back short of the cap.
  * On a tie at a kernel's best score the ref picked was whichever the API happened to
    list first (newest-first), so a fresh replicate could silently move the recommended
    ref. Ties now resolve to the EARLIEST submission (matches rules 3.7.b) and every
    tied ref is printed.
  * One slug can span several `scriptVersionId`s -- `rogii-stack-v2-blend` really does
    (5 versions). Grouping by slug alone would pool distinct code as one "config", so
    the version count is shown and flagged.
  * Kaggle's automatic selection takes the best PUBLIC scores, which is not what the
    plan wants for slot2. `--final-check` states that divergence explicitly.

Usage:
    uv run python scripts/submission_status.py [--all]
    uv run python scripts/submission_status.py --final-check [REF1 REF2]
"""

from __future__ import annotations

import re
import statistics as st
import sys
from collections import defaultdict

COMPETITION = "rogii-wellbore-geology-prediction"

# Asked-for page size. The endpoint returns everything up to this and never paginates,
# so a result of exactly PAGE_SIZE means we were truncated and must not be trusted.
PAGE_SIZE = 1000

# The planned final two, by immutable ref (a kernel rename changes the slug, not the ref).
PLANNED = {
    "slot1": ("55252809", "rogii-v2-projdeg2-probe", None),
    "slot2": ("55252812", "rogii-v2-seeds192-probe", None),
}


def kernel_of(sub) -> str:
    m = re.search(r"/code/[^/]+/([^?]+)", str(getattr(sub, "url", "") or ""))
    return m.group(1) if m else "<unknown>"


def version_of(sub) -> str:
    m = re.search(r"scriptVersionId=(\d+)", str(getattr(sub, "url", "") or ""))
    return m.group(1) if m else "<unknown>"


def score_of(sub) -> str:
    return str(getattr(sub, "public_score", "") or "").strip()


def fetch(group=None) -> list:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    kw = {"group": group} if group is not None else {}
    subs = list(api.competition_submissions(COMPETITION, page_size=PAGE_SIZE, **kw) or [])
    if group is not None:
        return subs
    if len(subs) >= PAGE_SIZE:
        raise SystemExit(
            f"ABORT: got exactly {len(subs)} submissions == page_size. The list is "
            f"truncated and page_number does not work on this endpoint. Raise PAGE_SIZE."
        )
    if not subs:
        raise SystemExit("ABORT: zero submissions returned -- auth or competition name wrong.")
    return subs


def final_check(subs: list) -> int:
    """Validate the planned final two by ref, and report anything that could change them."""
    by_ref = {str(s.ref): s for s in subs}
    by_kernel: dict[str, list] = defaultdict(list)
    for s in subs:
        if score_of(s):
            by_kernel[kernel_of(s)].append(s)

    argv_refs = [a for a in sys.argv[1:] if a.isdigit()]
    plan = (
        {f"slot{i}": (r, None, None) for i, r in enumerate(argv_refs, 1)}
        if len(argv_refs) == 2
        else PLANNED
    )

    ok = True
    for slot, (ref, want_kernel, want_ver) in plan.items():
        s = by_ref.get(ref)
        print(f"\n=== {slot}: ref {ref} ===")
        if s is None:
            print("  FAIL: ref not present in submission history")
            ok = False
            continue
        sc, kern, ver = score_of(s), kernel_of(s), version_of(s)
        err = str(getattr(s, "error_description", "") or "")
        print(f"  public_score      : {sc or '<EMPTY = NOT SCORED>'}")
        print(f"  private_score     : {str(getattr(s, 'private_score', '') or '') or '<empty>'}")
        print(f"  kernel / version  : {kern} / {ver}")
        print(f"  date (UTC)        : {s.date}")
        print(f"  status            : {s.status}   total_bytes={s.total_bytes}")
        print(f"  error_description : {err or '<none>'}")
        if not sc:
            print("  FAIL: no public score -- this submission CANNOT be selected")
            ok = False
        if int(getattr(s, "total_bytes", 0) or 0) == 0:
            print("  FAIL: total_bytes == 0 -- the rerun produced no submission file")
            ok = False
        if want_kernel and kern != want_kernel:
            print(f"  FAIL: kernel is {kern}, expected {want_kernel}")
            ok = False
        if want_ver and ver != want_ver:
            print(f"  FAIL: scriptVersionId is {ver}, expected {want_ver}")
            ok = False

        peers = by_kernel.get(kern, [])
        vals = sorted(float(p.public_score) for p in peers)
        if vals and sc:
            best = vals[0]
            tied = sorted(
                (p for p in peers if float(p.public_score) == best), key=lambda p: p.date
            )
            print(f"  kernel draws      : n={len(vals)} best={best:.3f} mean={st.mean(vals):.4f}")
            if float(sc) > best:
                print(f"  FAIL: not this kernel's best draw -- {best:.3f} is (ref {tied[0].ref})")
                ok = False
            elif len(tied) > 1:
                print(f"  WARN: tie at {best:.3f} across refs {[str(t.ref) for t in tied]}; "
                      f"earliest is {tied[0].ref}")
        vers = {version_of(p) for p in peers}
        if len(vers) > 1:
            # Pooling across versions only invalidates the check when the slot does not
            # pin a scriptVersionId. With want_ver verified above, the exact code that
            # will be rerun is known and the pooled 'kernel draws' line is merely
            # informational (slot2's fork legitimately has V1 6.442 + V2 6.400).
            print(f"  WARN: slug spans {len(vers)} scriptVersionIds {sorted(vers)} "
                  f"-- 'kernel draws' above pools DIFFERENT code")
            if not want_ver:
                ok = False

    pending = [s for s in subs if not score_of(s) and not str(
        getattr(s, "error_description", "") or "")]
    print(f"\n=== still-pending submissions (could out-score the plan later): {len(pending)} ===")
    for s in sorted(pending, key=lambda s: s.date):
        print(f"  {s.date:%m-%d %H:%M} UTC  {s.ref}  {kernel_of(s)}")

    scored = sorted((s for s in subs if score_of(s)), key=lambda s: float(s.public_score))
    auto = scored[:2]
    chosen = {r for r, _, _ in plan.values()}
    print("\n=== Kaggle auto-selection vs the plan ===")
    # Rule 3.18.c says Kaggle auto-selects if the user does not, but does NOT state the
    # criterion. Observed platform behaviour is best-public-score; either way the plan's
    # slot2 (4th best public) can never be reached automatically.
    print("  Auto-selection criterion is NOT in the rules; observed behaviour is best")
    print("  public score. Top 2 public are:")
    for i, s in enumerate(auto, 1):
        print(f"    #{i} {s.public_score}  ref={s.ref}  {kernel_of(s)}")
    if {str(s.ref) for s in auto} != chosen:
        print("  => AUTO-SELECTION DOES NOT MATCH THE PLAN.")
        print("     Manual selection in the Kaggle UI is MANDATORY, not optional.")
    else:
        print("  => auto-selection happens to match the plan.")

    # Selection cannot be SET from the API, but it CAN be read back: the endpoint
    # accepts group=SUBMISSION_GROUP_SELECTED. This is the only objective proof that
    # the manual UI click actually took effect.
    from kagglesdk.competitions.types.competition_enums import SubmissionGroup

    sel = fetch(group=SubmissionGroup.SUBMISSION_GROUP_SELECTED)
    print(f"\n=== currently SELECTED on Kaggle: {len(sel)} ===")
    for s in sel:
        print(f"  {s.public_score:>7}  ref={s.ref}  {kernel_of(s)}  ({s.date:%m-%d %H:%M} UTC)")
    sel_refs = {str(s.ref) for s in sel}
    if not sel_refs:
        print("  => NOTHING SELECTED YET. The UI click has not been made (or did not save).")
        ok = False
    elif sel_refs != chosen:
        print(f"  => MISMATCH. Selected {sorted(sel_refs)} but the plan is {sorted(chosen)}.")
        ok = False
    else:
        print("  => CONFIRMED: the selection on Kaggle matches the plan exactly.")

    verdict = "PASS -- selection verified" if ok else "FAIL -- selection not yet correct"
    print("\nRESULT:", verdict)
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    subs = fetch()
    if "--final-check" in argv:
        return final_check(subs)

    by_kernel: dict[str, list] = defaultdict(list)
    unscored: list = []
    for s in subs:
        (by_kernel[kernel_of(s)] if score_of(s) else unscored).append(s)

    print(f"{len(subs)} submissions  ({min(s.date for s in subs):%m-%d} .. "
          f"{max(s.date for s in subs):%m-%d})   scored={len(subs) - len(unscored)}")

    rows = []
    for k, ss in by_kernel.items():
        vs = sorted(float(s.public_score) for s in ss)
        best = min(vs)
        # Ties resolve to the EARLIEST submission, so a fresh replicate at the same
        # score cannot silently move the recommended ref.
        tied = sorted((s for s in ss if float(s.public_score) == best), key=lambda s: s.date)
        note = ""
        if len(tied) > 1:
            note += f"  [TIE x{len(tied)}: {','.join(str(t.ref) for t in tied[1:])}]"
        nver = len({version_of(s) for s in ss})
        if nver > 1:
            note += f"  [!! {nver} scriptVersionIds pooled under one slug]"
        rows.append((best, k, len(vs), str(tied[0].ref),
                     st.mean(vs), st.stdev(vs) if len(vs) > 1 else float("nan"), note))

    print(f"\n{'best':>7}  {'n':>2}  {'mean':>7}  {'sd':>6}  {'ref':>9}  kernel")
    for best, k, n, ref, mean, sd, note in sorted(rows):
        sd_s = f"{sd:6.4f}" if sd == sd else "     -"
        print(f"{best:7.3f}  {n:2d}  {mean:7.4f}  {sd_s}  {ref:>9}  {k}{note}")

    print(f"\nunscored ({len(unscored)}):")
    for s in sorted(unscored, key=lambda s: s.date):
        err = str(getattr(s, "error_description", "") or "")
        kind = "TIMEOUT" if "runtime" in err.lower() else ("PENDING" if not err else "ERROR")
        print(f"  {s.date:%m-%d %H:%M}  {s.ref}  {kind:7s}  {kernel_of(s)}")

    if "--all" in argv:
        print("\nall scored, best first:")
        for s in sorted((x for x in subs if str(getattr(x, "public_score", "") or "").strip()),
                        key=lambda x: float(x.public_score)):
            print(f"  {float(s.public_score):7.3f}  {s.ref}  {kernel_of(s)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
