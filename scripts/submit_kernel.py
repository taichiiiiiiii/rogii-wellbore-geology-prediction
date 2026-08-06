"""Submit a saved Kaggle kernel version to the ROGII code competition.

Code competitions cannot take a CSV upload: the submission references a *kernel
version*, and Kaggle re-runs that notebook on the hidden wells. Getting the API
call shape wrong silently wastes one of the five daily slots, so this wrapper
does the quota accounting and the pre-flight checks in one place.

Usage (the project venv already ships `kaggle`, so no `--with` is needed —
using `--with` builds a throwaway environment on every call and inflates the
shared uv cache):
    uv run python scripts/submit_kernel.py \
        --kernel taichiiiii/rogii-gs145-probe --version 1 --message "run7 ..."

    uv run python scripts/submit_kernel.py --dry-run --kernel ...

`--dry-run` performs every check and prints the plan without submitting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

COMPETITION = "rogii-wellbore-geology-prediction"
DAILY_LIMIT = 5


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def used_slots_today(api) -> tuple[int, list[str]]:
    """Submissions made in the current UTC day (the window Kaggle's quota uses)."""
    today = dt.datetime.now(dt.UTC).date()
    refs: list[str] = []
    for sub in api.competition_submissions(COMPETITION):
        date = getattr(sub, "date", None)
        if date is None:
            continue
        if date.date() == today:
            refs.append(str(sub.ref))
    return len(refs), refs


def kernel_state(api, kernel: str) -> str:
    """Kernel run state, e.g. 'KernelWorkerStatus.COMPLETE'.

    The SDK returns a response object (not a dict), so read the attribute.
    """
    status = api.kernels_status(kernel)
    return str(getattr(status, "status", status))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", required=True, help="owner/kernel-slug")
    p.add_argument("--version", type=int, default=1, help="kernel version number")
    p.add_argument("--message", required=True, help="submission description")
    p.add_argument(
        "--output-file",
        default="submission.csv",
        help="name of the output file the kernel writes (Kaggle rejects a wrong name)",
    )
    p.add_argument("--dry-run", action="store_true", help="check only, do not submit")
    args = p.parse_args(argv)

    api = _api()

    status = kernel_state(api, args.kernel)
    used, refs = used_slots_today(api)
    remaining = DAILY_LIMIT - used

    print(f"kernel      : {args.kernel} v{args.version}")
    print(f"kernel state: {status}")
    print(f"slots used  : {used}/{DAILY_LIMIT} today (UTC)  refs={refs}")
    print(f"remaining   : {remaining}")

    if "COMPLETE" not in status.upper():
        print("ABORT: kernel is not COMPLETE — a non-complete kernel cannot be submitted.")
        return 2
    if remaining <= 0:
        print("ABORT: daily submission quota is exhausted (resets at 00:00 UTC).")
        return 3

    if args.dry_run:
        print("dry-run: would submit now.")
        return 0

    # file_name is the *output* file the kernel writes, and kernel_version must be
    # an int — passing None/str here is rejected with an opaque HTTP 400.
    result = api.competition_submit_code(
        file_name=args.output_file,
        message=args.message,
        competition=COMPETITION,
        kernel=args.kernel,
        kernel_version=int(args.version),
    )
    print(f"submitted: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
