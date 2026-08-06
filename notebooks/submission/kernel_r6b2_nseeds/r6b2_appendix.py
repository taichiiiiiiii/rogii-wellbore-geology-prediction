# ======================================================================
# R6-b(2) appendix -- PF seed-count sensitivity for the PRODUCTION pf_blend.
#
# Base above = bank_builder.py verbatim (guard stripped at build time).
# Question: does averaging more track_pf seeds (8 -> 16 -> 32) improve the
# pf_blend candidate's pooled RMSE enough to justify bumping N_SEEDS in the
# submission kernel (aggressive-slot candidate (b), analysis/r5_endgame.md)?
#
# Design: run track_pf once per seed (32 seeds), then average nested subsets
# {first 8, first 16, all 32} of the SAME runs -- paired by construction,
# 1x compute instead of 3x. Seeds 0..31 match track_pf_multi's range(n)
# seeding, so the 8-subset equals production pf_blend exactly.
#
# Pre-declared decision rule: adopt-as-aggressive-candidate only if pooled
# RMSE improves by >= 0.05 at 16 or 32 vs 8; otherwise close R6-b(2) for good.
# (Runtime if adopted: 16 seeds ~= +3.1s/well -> hidden rerun ~2.9h, fine;
# 32 seeds ~= +9.3s/well -> ~4.6h, still inside 9h.)
# ======================================================================

R6B2_MAX_SEEDS = 32
R6B2_SUBSETS = (8, 16, 32)


def r6b2_main():
    t0 = time.perf_counter()
    root = find_data_root()
    out_dir = output_dir()
    wells = list_wells("train", root)
    limit = _bank_limit()
    if limit is not None:
        wells = wells[:limit]
    print(f"# R6-b(2) pf seed sensitivity: {len(wells)} wells, "
          f"{R6B2_MAX_SEEDS} seeds, subsets {R6B2_SUBSETS}")

    rows_y = []
    rows_pred = {k: [] for k in R6B2_SUBSETS}
    w_name, w_nrows = [], []
    failures = 0
    for wi, well in enumerate(wells):
        try:
            h = load_horizontal(well, "train", root)
            tw = load_typewell(well, "train", root)
            m = eval_mask(h)
            n_eval = int(m.sum())
            if n_eval == 0:
                continue
            y_true = h["TVT"].to_numpy(dtype=float)[np.flatnonzero(m)]
            fin = np.isfinite(y_true)
            if not fin.any():
                continue
            per_seed = np.stack(
                [track_pf(h, tw, seed=s).tvt for s in range(R6B2_MAX_SEEDS)]
            )  # (32, n_eval)
            rows_y.append(y_true[fin])
            for k in R6B2_SUBSETS:
                rows_pred[k].append(per_seed[:k].mean(axis=0)[fin])
            w_name.append(well)
            w_nrows.append(int(fin.sum()))
        except Exception as e:  # noqa: BLE001 -- sweep must survive odd wells
            failures += 1
            print(f"  [fail] {well}: {type(e).__name__}: {e}")
        if (wi + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {wi + 1}/{len(wells)} wells | {el:.0f}s | "
                  f"{el / (wi + 1):.1f}s/well", flush=True)

    y = np.concatenate(rows_y)
    print(f"\nwells scored={len(w_name)} failures={failures} rows={y.size}")
    print("pooled RMSE of pf_blend by seed count (paired, nested subsets):")
    base = None
    summary = {}
    for k in R6B2_SUBSETS:
        pred = np.concatenate(rows_pred[k])
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        summary[k] = rmse
        if base is None:
            base = rmse
        print(f"  n_seeds={k:2d}: {rmse:8.4f}  (delta vs 8: {rmse - base:+.4f})")

    np.savez_compressed(
        out_dir / "r6b2_nseeds.npz",
        used_wells=np.asarray(w_name), n_rows=np.asarray(w_nrows, dtype=np.int32),
        y_true=y.astype(np.float32),
        **{f"pf_mean_{k}": np.concatenate(rows_pred[k]).astype(np.float32)
           for k in R6B2_SUBSETS},
        subset_keys=np.asarray(list(summary.keys()), dtype=np.int32),
        subset_rmse=np.asarray(list(summary.values())),
    )
    print(f"\nsaved {out_dir / 'r6b2_nseeds.npz'}")
    print(f"# r6b2 total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    r6b2_main()
