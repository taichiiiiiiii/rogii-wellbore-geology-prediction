"""Field-grouped CV audit for stack v2 all-in (issue: stack v2 fieldcv audit).

v1's field-grouped audit (``scripts/run_fieldcv_audit.py``) re-measured stack(b)
[PF-residual, well-GroupKFold ledger 10.6513] under spatially-isolated folds and
found the gap negligible (field-CV 10.6972, Δ+0.046 -- "空間リーク無視可"). That
conclusion does **not** automatically extend to stack v2: v2's "all-in" config
(``scripts/run_stack_v2_cv.py``, well-GroupKFold OOF **9.2309**, current ledger
best) adds 4 spatial-prior features (``rogii.stack_features.build_spatial_features``)
built from a leave-self-out KNN over *other train wells'* formation-depth
surfaces (``rogii.spatial``). Those features are informative *because* nearby
wells exist in train -- under well-unit GroupKFold a validation well's spatial
neighbors can still sit in the training folds, which v1's audit showed does
happen (well-GroupKFold median nearest-train distance ~552ft). If v2's OOF gain
depends on that neighbor availability, well-GroupKFold could be optimistic
relative to a hidden test set that is spatially disjoint from train.

This script re-measures v2's all-in config under the **same field-grouped split
recipe as v1's audit** (imported, not duplicated): KMeans(k=15, seed=42) on
standardized well centroids -> spatial "fields" -> greedy well-count-balanced
bin-pack (LPT heuristic) of whole fields into folds, so a fold's validation
wells' nearest train well is a *different field*.

Design choices (both deliberate departures from pure duplication, per playbook
02's "read scripts/run_stack_v2_cv.py and reproduce exactly, import functions
where possible, only swap fold assignment"):

  1. **Imports, not duplicates.** ``run_stack_v2_cv``'s feature-matrix builder
     (:func:`run_stack_v2_cv.build_well_matrix_v2`), fold-training loop
     (:func:`run_stack_v2_cv.run_stack_fold_cv`) and feature-column constants are
     imported directly -- guarantees byte-identical features/LGBM config/ES
     protocol without a second hand-maintained copy to drift out of sync.
     ``run_fieldcv_audit``'s field-clustering/fold-assignment/distance-diagnostic
     helpers are imported the same way. Neither v1 file is modified.
  2. **Well-GroupKFold is recomputed here, not just read from the ledger.** The
     saved ``outputs/stack_v2_oof.npz`` (from ``run_stack_v2_cv.py --save-oof``)
     has OOF *predictions* only, not per-config feature importances -- and this
     audit's brief explicitly asks how the spatial features' importance rank
     shifts between well- and field-grouped splits. Recomputing well-GroupKFold
     here (identical wells/features/config to the ledger run) gives an
     apples-to-apples importance comparison and doubles as a regression guard
     (its pooled RMSE must reproduce 9.2309).
  3. **Well centroids are recomputed, not read from ``build_well_matrix_v2``.**
     v2's builder has no clustering use for (X, Y) and doesn't expose them; a
     second pass of ``D.load_horizontal`` (mean X, mean Y over each well's
     *entire* log, matching v1's audit) costs ~0.01s/well, cheap relative to the
     LGBM training time.
  4. **Spatial features are recomputed fold-aware for the field-GroupKFold pass
     -- this is the whole point of the audit, not an optional extra.** ``wm.X``'s
     ``SPATIAL_FEATURE_COLUMNS`` come from ``data/processed/spatial_cache/*.npz``,
     built by ``scripts/build_spatial_cache.py`` from a surface bank over *all*
     773 train wells with only leave-*self*-out exclusion
     (``rogii.spatial.predict_spatial(h, bank, exclude_well=well)``). Reusing
     that cache unmodified for a field-held-out fold's rows would silently keep
     every other well in the same spatial field inside the bank -- exactly the
     neighbor-availability assumption this audit exists to stress-test would
     survive untested. :func:`recompute_field_aware_spatial` rebuilds, once per
     field-fold, a ``rogii.spatial.SurfaceBank`` from every train well *except*
     that fold's held-out field, then reruns ``predict_spatial``/the per-row
     nearest-neighbor-distance diagnostic for exactly that fold's wells against
     the reduced bank, and substitutes the result into a copy of ``wm.X`` used
     only for the field-GroupKFold config. Training-fold rows are left
     untouched (still the original full-bank cache): at submission time every
     real train well already has every *other* real train well available in
     its bank regardless of which field a given CV fold happens to hold out --
     only the held-out fold's own rows need the pessimistic, field-isolated
     recompute to honestly test whether the model's fit *depends* on that
     neighbor availability. The well-GroupKFold regression-guard pass (point 2
     above) intentionally keeps the original full-bank cache unchanged, since
     well-GroupKFold's validation wells' true spatial neighbors are expected to
     remain in the training folds at submission time too (that is exactly the
     asymmetry this audit measures).

Leak boundary: identical to v1 and v2 -- ``TVT`` is read exactly once per well
(inside the imported ``build_well_matrix_v2``) to build ``y_true``/the residual
target; centroids use ``X``/``Y`` only (present in test too, never a leak, same
reasoning as ``run_fieldcv_audit.py``).

Usage::

    uv run python scripts/run_fieldcv_audit_v2.py --n-wells 80   # health check
    uv run python scripts/run_fieldcv_audit_v2.py                # full audit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import build_spatial_cache as BSC  # spatial-cache config: STRIDE/K_NEIGHBORS/METHOD (read-only)
import numpy as np
import pandas as pd
import run_fieldcv_audit as V1AUDIT  # v1 audit: field-assignment/diagnostic helpers (read-only)
import run_stack_v2_cv as V2  # v2 stack: feature matrix / LGBM training (read-only)

from rogii import cv as CV
from rogii import data as D
from rogii import spatial as SP

REPO_ROOT = Path(__file__).resolve().parents[1]
STACK_V2_OOF_PATH = REPO_ROOT / "outputs" / "stack_v2_oof.npz"  # read-only, cross-check only

CARRY_LAST_POOLED_RMSE = 15.9099
PF_BLEND_POOLED_RMSE = 11.0621
V1_STACK_B_POOLED_RMSE = 10.6513
LEDGER_V2_ALLIN_WELL_GROUPKFOLD = 9.2309  # ledger: stack v2 all-in, well-GroupKFold OOF

DELTA_ROBUST_THRESHOLD = 0.3  # < this: v2's 9.2309 is robust to spatial held-out folds
DELTA_NEIGHBOR_DEPENDENT_THRESHOLD = 1.0  # > this: spatial features lean on in-fold neighbors


def compute_centroids(wells: list[str], split: str = "train") -> dict[str, tuple[float, float]]:
    """Mean (X, Y) over each well's *entire* horizontal log (not just the eval zone).

    ``run_stack_v2_cv.build_well_matrix_v2`` has no clustering use for (X, Y) and
    doesn't expose them, so this is a second, cheap (~0.01s/well) read of
    ``D.load_horizontal``. X/Y are known in test too -- never a leak (same
    reasoning as ``run_fieldcv_audit.py``'s centroid capture).
    """
    centroids: dict[str, tuple[float, float]] = {}
    for well in wells:
        h = D.load_horizontal(well, split)
        centroids[well] = (float(h["X"].mean()), float(h["Y"].mean()))
    return centroids


def build_field_folds(
    centroid_map: dict[str, tuple[float, float]],
    wells: list[str],
    n_splits: int,
    k: int,
    seed: int,
) -> tuple[dict[str, int], int, pd.Series, float, bool, list[int]]:
    """Cluster well centroids into spatial fields and assign fields to folds.

    Identical recipe to ``run_fieldcv_audit.py``'s ``main()`` (calls its imported
    helpers, does not duplicate their bodies): KMeans(k, seed) on standardized
    (X, Y) centroids, an imbalance retry (alt k=20/10, kept only if it improves
    balance) if the well-count CV(std/mean) is too high or the smallest field is
    too small, then a greedy well-count-balanced bin-pack (LPT heuristic) of
    fields into folds.
    """
    n_wells = len(wells)
    well_to_field = V1AUDIT.cluster_fields(centroid_map, wells, k, seed)
    counts, cv_score = V1AUDIT.field_balance(well_to_field, k)

    retried = False
    if cv_score > V1AUDIT.IMBALANCE_CV_THRESHOLD or counts.min() < V1AUDIT.IMBALANCE_MIN_FIELD:
        retried = True
        dominant_frac = counts.max() / n_wells
        alt_k = 20 if dominant_frac > 0.25 else 10
        well_to_field_alt = V1AUDIT.cluster_fields(centroid_map, wells, alt_k, seed)
        counts_alt, cv_alt = V1AUDIT.field_balance(well_to_field_alt, alt_k)
        if cv_alt < cv_score:
            k, well_to_field, counts, cv_score = alt_k, well_to_field_alt, counts_alt, cv_alt

    field_sizes = pd.Series(well_to_field).value_counts()
    field_to_fold, fold_totals = V1AUDIT.assign_fields_to_folds(field_sizes, n_splits)
    well_to_fold = {w: field_to_fold[well_to_field[w]] for w in wells}
    return well_to_fold, k, counts, cv_score, retried, fold_totals


def recompute_field_aware_spatial(
    wm: V2._WellMatrix,
    well_to_fold_field: dict[str, int],
    n_splits: int,
) -> pd.DataFrame:
    """Rebuild ``SPATIAL_FEATURE_COLUMNS`` field-fold-aware, for the field-GroupKFold pass.

    Returns a copy of ``wm.X[SPATIAL_FEATURE_COLUMNS]`` where each field-fold's
    held-out wells' rows are recomputed against a ``rogii.spatial.SurfaceBank``
    that excludes *every* well in that fold's held-out field (not merely the
    query well itself, the ``exclude_well`` leave-self-out that
    ``scripts/build_spatial_cache.py``'s cache -- ``wm.X``'s source -- already
    applies). Training-fold rows are left at their original cached values
    (see the module docstring's design-choice 4 for why that's correct, not
    an oversight).

    ``stride``/``k``/``method`` are read from ``build_spatial_cache.py``
    (``BSC.STRIDE``/``BSC.K_NEIGHBORS``/``BSC.METHOD``) so the reduced-bank
    recompute uses the exact same config as the ledger-verified full cache
    (``BSC``'s own regression guard: gateblend theta=2 == 14.8724).
    """
    spatial_cols = list(V2.SPATIAL_FEATURE_COLUMNS)
    recomputed = wm.X[spatial_cols].copy()
    used_well_to_idx = {w: i for i, w in enumerate(wm.used_wells)}
    all_train_wells = D.list_wells("train")

    for fold_i in range(n_splits):
        t0 = time.time()
        held_out = [w for w in wm.used_wells if well_to_fold_field[w] == fold_i]
        held_out_set = set(held_out)
        bank_wells = [w for w in all_train_wells if w not in held_out_set]
        bank = SP.build_surface_bank(bank_wells, split="train", stride=BSC.STRIDE)

        for well in held_out:
            h = D.load_horizontal(well, "train")
            result = SP.predict_spatial(
                h, bank, exclude_well=well, k=BSC.K_NEIGHBORS, method=BSC.METHOD
            )
            nn_dist = BSC._nn_distance_per_row(h, bank, well)
            finite_nn = nn_dist[np.isfinite(nn_dist)] if nn_dist.size else np.array([])
            nn_dist_median = float(np.median(finite_nn)) if finite_nn.size else float("nan")
            anchor = D.last_known_tvt(h)

            feats = V2.build_spatial_features(
                anchor, result.tvt, result.prefix_rmse, nn_dist_median
            )
            w_i = used_well_to_idx[well]
            rows = wm.well_idx == w_i
            if int(rows.sum()) != len(feats):
                raise ValueError(
                    f"row-count mismatch recomputing spatial features for {well}: "
                    f"{int(rows.sum())} rows in wm.X vs {len(feats)} rows from predict_spatial"
                )
            recomputed.loc[rows, spatial_cols] = feats[spatial_cols].to_numpy()

        print(
            f"  field-aware spatial recompute fold {fold_i}: bank={len(bank_wells)} wells, "
            f"held_out={len(held_out)} wells ({time.time() - t0:.1f}s)",
            flush=True,
        )

    return recomputed


def _print_comparison_table(rows: list[tuple[str, float, np.ndarray]]) -> None:
    print(f"{'predictor':<42}{'pooled RMSE':>14}{'median':>10}{'p90':>10}{'max':>10}")
    print("-" * 86)
    for name, pooled, well_arr in rows:
        valid = well_arr[np.isfinite(well_arr)]
        if valid.size == 0:
            print(f"{name:<42}{pooled:>14.4f}{'n/a':>10}{'n/a':>10}{'n/a':>10}")
            continue
        print(
            f"{name:<42}{pooled:>14.4f}{np.median(valid):>10.4f}"
            f"{np.percentile(valid, 90):>10.4f}{np.max(valid):>10.4f}"
        )


def _importance_df(feature_cols: list[str], importances: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_cols, "mean_importance": importances})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


def _spatial_rank_report(label: str, imp_df: pd.DataFrame) -> None:
    ranked = imp_df.reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    spatial_rows = ranked[ranked["feature"].isin(V2.SPATIAL_FEATURE_COLUMNS)]
    print(f"\nspatial feature ranks -- {label} ({len(ranked)} features total):")
    print(spatial_rows[["rank", "feature", "mean_importance"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wells",
        type=int,
        default=None,
        help="Health-check subset size (stratified by tracker-manifest pf_std_mean "
        "decile, same recipe as run_stack_v2_cv.py). Omit for the full 773-well audit.",
    )
    args = parser.parse_args()
    t_start = time.time()

    all_wells = D.list_wells("train")
    if args.n_wells is not None:
        wells = V2._stratified_sample(all_wells, args.n_wells, V2.SEED)
        print(
            f"health-check subset: {len(wells)}/{len(all_wells)} wells "
            "(stratified by pf_std_mean)"
        )
    else:
        wells = all_wells
        print(f"train wells: {len(wells)}")

    # --- 1. feature matrix: imported, byte-identical to run_stack_v2_cv.py ---
    t0 = time.time()
    wm = V2.build_well_matrix_v2(wells, split="train")
    n_wells_used = len(wm.used_wells)
    print(
        f"feature matrix: {wm.X.shape}, rows={len(wm.y_true):,}, "
        f"wells_used={n_wells_used}/{len(wells)} counts={wm.counts} "
        f"({time.time() - t0:.1f}s)"
    )

    # --- 2. well centroids (mean X, mean Y over the whole horizontal log) ---
    t0 = time.time()
    centroid_map = compute_centroids(wm.used_wells, split="train")
    print(f"centroids computed for {len(centroid_map)} wells ({time.time() - t0:.1f}s)")

    # --- 3. field clustering + field->fold assignment (identical recipe to v1 audit) ---
    t0 = time.time()
    well_to_fold_field, k, counts_field, cv_score, retried, fold_totals = build_field_folds(
        centroid_map, wm.used_wells, V2.N_SPLITS, V1AUDIT.DEFAULT_K, V2.SEED
    )
    print(f"\nfield clustering: k={k} seed={V2.SEED}, well-count CV(std/mean)={cv_score:.3f}")
    print(counts_field.rename("n_wells").to_string())
    target_per_fold = n_wells_used / V2.N_SPLITS
    print(
        f"field->fold well-count totals (target ~{target_per_fold:.0f}/fold): "
        f"{fold_totals} (retried={retried}, {time.time() - t0:.1f}s)"
    )

    well_fold_arr_field = np.array(
        [well_to_fold_field[w] for w in wm.used_wells], dtype=np.int32
    )
    row_fold_field = well_fold_arr_field[wm.well_idx]

    # --- 4. well-based folds (identical construction to run_stack_v2_cv.py::main) ---
    folds = CV.well_folds(wells, n_splits=V2.N_SPLITS, seed=V2.SEED)
    well_to_fold_wg = {w: i for i, fold in enumerate(folds) for w in fold}
    well_fold_arr_wg = np.array([well_to_fold_wg[w] for w in wm.used_wells], dtype=np.int32)
    row_fold_wg = well_fold_arr_wg[wm.well_idx]

    # --- 5. baselines on this exact row set ---
    floor_pooled = V2.pooled_rmse(wm.y_true, wm.anchor_arr)
    pf_blend_pooled = V2.pooled_rmse(wm.y_true, wm.pf_blend_arr)
    floor_well_rmse = V2._per_well_rmse(wm.y_true, wm.anchor_arr, wm.well_idx, n_wells_used)
    pf_well_rmse = V2._per_well_rmse(wm.y_true, wm.pf_blend_arr, wm.well_idx, n_wells_used)

    # --- 6. all-in feature columns (identical to run_stack_v2_cv.py's ablation_configs[-1]) ---
    allin_cols = (
        wm.p1_cols
        + list(V2.FEATURE_COLUMNS)
        + list(V2.WELL_FEATURE_COLUMNS)
        + list(V2.SPATIAL_FEATURE_COLUMNS)
    )
    X_allin = wm.X[allin_cols]
    y_r = wm.y_true - wm.pf_blend_arr  # (b) PF-residual target, same as v1/v2
    # 3GB-RAM box: after this point wm.X is only read for SPATIAL_FEATURE_COLUMNS
    # (recompute_field_aware_spatial); shrink it so we never hold two extra
    # full-width copies at once (OOM-killed here on 2026-07-09).
    wm.X = wm.X[list(V2.SPATIAL_FEATURE_COLUMNS)].copy()

    # --- 7. all-in, well-GroupKFold [RECOMPUTE: regression guard + importance source] ---
    print("\n=== all-in, well-GroupKFold [RECOMPUTE, use_es_holdout=True] ===")
    oof_wg, fold_rows_wg, imp_wg = V2.run_stack_fold_cv(
        "all-in:well-GroupKFold",
        X_allin,
        y_r,
        wm.pf_blend_arr,
        wm.y_true,
        row_fold_wg,
        wm.well_idx,
        well_fold_arr_wg,
        wm.used_wells,
        V2.N_SPLITS,
        True,
        V2.ES_FRAC,
        V2.SEED,
    )
    pooled_wg = V2.pooled_rmse(wm.y_true, oof_wg)
    well_rmse_wg = V2._per_well_rmse(wm.y_true, oof_wg, wm.well_idx, n_wells_used)

    if not args.n_wells:
        guard_delta = pooled_wg - LEDGER_V2_ALLIN_WELL_GROUPKFOLD
        print(
            f"\nregression guard: recomputed well-GroupKFold pooled={pooled_wg:.4f} "
            f"vs ledger {LEDGER_V2_ALLIN_WELL_GROUPKFOLD:.4f} (delta={guard_delta:+.4f})"
        )
        if STACK_V2_OOF_PATH.exists():
            with np.load(STACK_V2_OOF_PATH, allow_pickle=True) as npz:
                y_true_npz = npz["y_true"].astype(np.float64)
                oof4_npz = npz["oof_4"].astype(np.float64)
            npz_pooled = V2.pooled_rmse(y_true_npz, oof4_npz)
            print(
                f"regression guard: outputs/stack_v2_oof.npz oof_4 pooled={npz_pooled:.4f} "
                f"(delta vs this recompute={pooled_wg - npz_pooled:+.4f})"
            )

    # --- 8a. field-aware spatial feature recompute [NEW] -- see module docstring
    # design-choice 4: field-GroupKFold's held-out rows must not keep spatial
    # features computed against a bank that still contains their own field's
    # other wells (leave-self-out alone is not field isolation).
    print("\n=== field-aware spatial feature recompute (per field-fold surface bank) ===")
    t0 = time.time()
    spatial_field_df = recompute_field_aware_spatial(wm, well_to_fold_field, V2.N_SPLITS)
    print(f"field-aware spatial recompute total: {time.time() - t0:.1f}s")
    # in-place column swap instead of a third full-width copy (OOM guard).
    # X_allin's pristine form is no longer needed: step 7 already consumed it.
    X_allin[list(V2.SPATIAL_FEATURE_COLUMNS)] = spatial_field_df
    X_allin_field = X_allin

    # --- 8b. all-in, field-GroupKFold [NEW, this run] ---
    print("\n=== all-in, field-GroupKFold [NEW, use_es_holdout=True, field-aware spatial] ===")
    oof_field, fold_rows_field, imp_field = V2.run_stack_fold_cv(
        "all-in:field-GroupKFold",
        X_allin_field,
        y_r,
        wm.pf_blend_arr,
        wm.y_true,
        row_fold_field,
        wm.well_idx,
        well_fold_arr_field,
        wm.used_wells,
        V2.N_SPLITS,
        True,
        V2.ES_FRAC,
        V2.SEED,
    )
    pooled_field = V2.pooled_rmse(wm.y_true, oof_field)
    well_rmse_field = V2._per_well_rmse(wm.y_true, oof_field, wm.well_idx, n_wells_used)

    # --- 9. comparison table ---
    print()
    print("=" * 86)
    rows: list[tuple[str, float, np.ndarray]] = [
        ("carry_last (floor)", floor_pooled, floor_well_rmse),
        ("PF blend w0.7 (split-invariant ref)", pf_blend_pooled, pf_well_rmse),
        ("v1 (b) stack, well-GroupKFold (ledger)", V1_STACK_B_POOLED_RMSE, np.array([])),
        ("v2 all-in, well-GroupKFold [recompute]", pooled_wg, well_rmse_wg),
        ("v2 all-in, field-GroupKFold [NEW]", pooled_field, well_rmse_field),
    ]
    _print_comparison_table(rows)
    print("=" * 86)
    ledger_delta = pooled_field - LEDGER_V2_ALLIN_WELL_GROUPKFOLD
    recompute_delta = pooled_field - pooled_wg
    print(
        f"delta vs ledger well-GroupKFold (9.2309): {ledger_delta:+.4f} | "
        f"delta vs this run's well-GroupKFold recompute: {recompute_delta:+.4f}"
    )
    print(f"total rows scored: {len(wm.y_true):,}")

    # --- 10. per-fold RMSE / best_iteration ---
    print("\nper-fold RMSE / best_iteration -- well-GroupKFold [recompute]:")
    print(pd.DataFrame(fold_rows_wg).to_string(index=False))
    print("\nper-fold RMSE / best_iteration -- field-GroupKFold [NEW]:")
    print(pd.DataFrame(fold_rows_field).to_string(index=False))

    # --- 11. feature importance: spatial rank, well vs field ---
    imp_wg_df = _importance_df(allin_cols, imp_wg)
    imp_field_df = _importance_df(allin_cols, imp_field)
    print("\ntop10 feature importance -- well-GroupKFold [recompute]:")
    print(imp_wg_df.head(10).to_string(index=False))
    print("\ntop10 feature importance -- field-GroupKFold [NEW]:")
    print(imp_field_df.head(10).to_string(index=False))
    _spatial_rank_report("well-GroupKFold [recompute]", imp_wg_df)
    _spatial_rank_report("field-GroupKFold [NEW]", imp_field_df)

    # --- 12. diagnostic: valid-well -> nearest train-well centroid distance ---
    print("\n=== diagnostic: valid-well -> nearest train-well centroid distance ===")
    print("(ft, X/Y projected coords -- consistent with MD/TVT units)")
    dist_field = V1AUDIT.nearest_train_distances(well_to_fold_field, centroid_map, V2.N_SPLITS)
    print("\nfield-GroupKFold:")
    print(V1AUDIT.summarize_distances(dist_field).to_string(index=False))
    dist_wg = V1AUDIT.nearest_train_distances(well_to_fold_wg, centroid_map, V2.N_SPLITS)
    print("\nwell-GroupKFold:")
    print(V1AUDIT.summarize_distances(dist_wg).to_string(index=False))

    # --- 13. verdict ---
    print("\n" + "=" * 86)
    if ledger_delta < DELTA_ROBUST_THRESHOLD:
        verdict = (
            f"ROBUST (delta={ledger_delta:+.4f} < {DELTA_ROBUST_THRESHOLD}): v2 all-in's "
            "9.2309 holds under spatial isolation -- LB transfer expected to track local CV."
        )
    elif ledger_delta <= DELTA_NEIGHBOR_DEPENDENT_THRESHOLD:
        verdict = (
            f"MIXED (0.3 <= delta={ledger_delta:+.4f} <= 1.0): some neighbor-dependence -- "
            "calibrate LB expectation toward the field-CV number, not the well-CV number."
        )
    else:
        verdict = (
            f"NEIGHBOR-DEPENDENT (delta={ledger_delta:+.4f} > "
            f"{DELTA_NEIGHBOR_DEPENDENT_THRESHOLD}): spatial features lean on in-fold neighbor "
            "wells -- 9.2309 is optimistic for hidden test wells outside train's spatial "
            "footprint. Calibrate the submission-time expectation to between the well-CV "
            "(9.2309) and field-CV numbers above."
        )
    print(verdict)
    print("=" * 86)

    print(f"\ntotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
