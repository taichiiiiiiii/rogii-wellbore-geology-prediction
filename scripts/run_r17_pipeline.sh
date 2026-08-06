#!/usr/bin/env bash
# R17 full-scale pipeline (issue: R17), strictly serial to respect the 3.8GB
# local RAM budget (see memory MEMORY.md: memory-heavy tasks avoid concurrent
# processes on this box). Phase 1 only: build the two V2 SurfaceBank caches,
# then screen configs a/b/c with seed42 solo and print the decision table.
# Phase 2 (5-seed escalation of a winning config) is a separate follow-up,
# not run automatically here -- it depends on phase-1 results.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[R17] $(date) build_spatial_cache_v2 config b (plane k10) -- full 773 wells"
uv run python scripts/build_spatial_cache_v2.py --config b

echo "[R17] $(date) build_spatial_cache_v2 config c (quadratic k12 power2) -- full 773 wells"
uv run python scripts/build_spatial_cache_v2.py --config c

echo "[R17] $(date) run_stack_v31_cv config a seed42 (control repro, full 773 wells)"
uv run python scripts/run_stack_v31_cv.py --config a --seed-idx 0

echo "[R17] $(date) run_stack_v31_cv config b seed42 (full 773 wells)"
uv run python scripts/run_stack_v31_cv.py --config b --seed-idx 0

echo "[R17] $(date) run_stack_v31_cv config c seed42 (full 773 wells)"
uv run python scripts/run_stack_v31_cv.py --config c --seed-idx 0

echo "[R17] $(date) phase-1 decision table"
uv run python scripts/run_stack_v31_cv.py --compare

echo "[R17] $(date) pipeline done"
