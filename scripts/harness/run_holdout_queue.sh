#!/usr/bin/env bash
# Push the queued 40-well holdout harness variants (k40b, k40c, k40d) to
# Kaggle sequentially once GPU quota is available, polling each one to
# completion (or a per-kernel timeout) before moving to the next, and
# downloading each variant's output artifacts as it finishes.
#
# BUILD-ONLY DELIVERABLE: this script is written but NOT executed as part of
# building the harness variants. Quota is exhausted at build time; a separate
# poller process is responsible for actually running this (or an equivalent
# push) once quota frees. Do not run this manually unless you intend to
# consume quota and submit real Kaggle kernel runs.
#
# Usage:
#   scripts/harness/run_holdout_queue.sh
#
# What it does, per variant, in order (k40b -> k40c -> k40d):
#   1. `kaggle kernels push -p <source dir>`         (uploads + starts the run)
#   2. poll `kaggle kernels status <slug>` every 3 minutes
#   3. on KernelWorkerStatus.COMPLETE: `kaggle kernels output <slug> -p
#      harness_audit/<name>/`                        (downloads cv_summary.json etc.)
#   4. on KernelWorkerStatus.ERROR / CANCELLED: log and move on (no download)
#   5. if still running after the per-kernel wall-clock budget (130 min):
#      log a timeout and move on WITHOUT downloading (the remote kernel keeps
#      running server-side; this script just stops waiting on it -- cancel it
#      manually via the Kaggle UI or the MCP `cancel_notebook_session` tool if
#      you want to reclaim quota immediately)
#   6. proceed to the next queued variant regardless of how the previous one
#      finished (best-effort queue -- one bad run should not block the rest)
#
# Source dirs (built by build_k40b.py / build_k40cd.py):
#   harness_audit/rogii-harness-b145-k40b/
#   harness_audit/rogii-harness-b145-k40c/
#   harness_audit/rogii-harness-b145-k40d/
# Output dirs (downloaded results land here):
#   harness_audit/k40b/  harness_audit/k40c/  harness_audit/k40d/

set -u -o pipefail

REPO=/root/competition/Kaggle/rogii-wellbore-geology-prediction
SCRATCH=/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad
AUDIT="$SCRATCH/harness_audit"
KAGGLE="$REPO/.venv/bin/kaggle"

POLL_INTERVAL_SEC=180   # 3 minutes
ABORT_BUDGET_SEC=$((130 * 60))   # 130 minutes per kernel

LOG="$AUDIT/run_holdout_queue.log"
mkdir -p "$AUDIT"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

# name : slug : source push dir
QUEUE=(
  "k40b:taichiiiii/rogii-harness-b145-k40b:$AUDIT/rogii-harness-b145-k40b"
  "k40c:taichiiiii/rogii-harness-b145-k40c:$AUDIT/rogii-harness-b145-k40c"
  "k40d:taichiiiii/rogii-harness-b145-k40d:$AUDIT/rogii-harness-b145-k40d"
)

run_one() {
  local name="$1" slug="$2" src_dir="$3"
  local out_dir="$AUDIT/$name"

  if [ ! -d "$src_dir" ]; then
    log "SKIP $name: source dir $src_dir does not exist (build it first)"
    return 1
  fi

  log "=== $name ($slug): pushing from $src_dir ==="
  if ! "$KAGGLE" kernels push -p "$src_dir" 2>&1 | tee -a "$LOG"; then
    log "FAIL $name: push command failed, skipping to next queued variant"
    return 1
  fi

  local elapsed=0
  local status=""
  while [ "$elapsed" -lt "$ABORT_BUDGET_SEC" ]; do
    sleep "$POLL_INTERVAL_SEC"
    elapsed=$((elapsed + POLL_INTERVAL_SEC))

    status="$("$KAGGLE" kernels status "$slug" 2>&1)"
    log "$name: t+${elapsed}s status: $status"

    case "$status" in
      *KernelWorkerStatus.COMPLETE*)
        log "$name: COMPLETE, downloading output to $out_dir"
        mkdir -p "$out_dir"
        if "$KAGGLE" kernels output "$slug" -p "$out_dir" 2>&1 | tee -a "$LOG"; then
          log "$name: download OK"
        else
          log "$name: download FAILED (kernel completed but output pull errored)"
        fi
        return 0
        ;;
      *KernelWorkerStatus.ERROR*|*KernelWorkerStatus.CANCEL*)
        log "$name: terminal non-success status ($status), skipping download, moving on"
        return 1
        ;;
      *)
        # still running (QUEUED / RUNNING / etc.) -- keep polling
        ;;
    esac
  done

  log "$name: TIMEOUT after ${ABORT_BUDGET_SEC}s (130 min) without completion." \
      "Remote kernel may still be running -- cancel manually via Kaggle UI or" \
      "the MCP cancel_notebook_session tool to reclaim quota. Moving to next" \
      "queued variant without downloading."
  return 1
}

main() {
  log "==== run_holdout_queue.sh starting: ${#QUEUE[@]} variant(s) queued ===="
  local failures=0
  for entry in "${QUEUE[@]}"; do
    IFS=':' read -r name slug src_dir <<< "$entry"
    if ! run_one "$name" "$slug" "$src_dir"; then
      failures=$((failures + 1))
    fi
  done
  log "==== run_holdout_queue.sh done: $((${#QUEUE[@]} - failures))/${#QUEUE[@]} variant(s) downloaded ===="
  [ "$failures" -eq 0 ]
}

main "$@"
