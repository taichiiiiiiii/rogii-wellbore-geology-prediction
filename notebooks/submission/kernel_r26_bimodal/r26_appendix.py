# ======================================================================
# R26 appendix -- bimodal-datum census + hedge-vs-commit scoring.
#
# Everything above this marker is bank_builder.py verbatim (minus its
# __main__ guard, stripped at build time by build_kernel.py); this appendix
# reuses its data loaders, affine GR calibration, grid builder and beam
# constants so the DP here is bit-comparable with the production beam
# tracker. Ledger item: R26 (queue #11, 2026-07-11 sweep).
#
# Question this kernel answers with OUR data (not forum claims):
#   1. What fraction of the 773 train wells have a genuinely bimodal DP cost
#      frontier (secondary mode >= MIN_GAP_FT away, within DCOST caps)?
#   2. Is the mode gap concentrated near one stratigraphic bundle (~15 ft)?
#   3. Is the cost margin informative about which mode is correct
#      (souldrive claims r ~= 0.05 = uninformative)?
#   4. Does hedging (midpoint / soft-weighted / conditional) beat committing,
#      pooled over all wells, for the beam candidate alone?
# ======================================================================

R26_MIN_GAP_FT = 6.0
# dcost scale measured in the 8-well smoke: p10-p90 ~= 90..640 cost units,
# so thresholds/temperatures must live in that range to ever fire.
R26_THETA_GRID = (5.0, 20.0, 50.0, 100.0, 200.0, 400.0)
R26_TAU_GRID = (20.0, 50.0, 100.0, 300.0)


def _forward_viterbi_full(gr_eval, grid_gr, init_cost, move_penalty, mismatch_scale, max_move):
    """Same DP as _forward_viterbi but keeps backptr + final frontier.

    Returns (backptr, final_cost, margin). Kept separate so the verbatim
    base implementation stays untouched (bit-comparability with production).
    """
    n_rows = gr_eval.shape[0]
    n_states = grid_gr.shape[0]
    penalty = move_penalty * np.abs(np.arange(-max_move, max_move + 1, dtype=float))
    backptr = np.empty((n_rows, n_states), dtype=np.int32)
    margin = np.zeros(n_rows, dtype=float)
    prev_cost = init_cost
    padded = np.empty(n_states + 2 * max_move, dtype=float)
    state_idx = np.arange(n_states)
    for i in range(n_rows):
        padded[:] = np.inf
        padded[max_move : max_move + n_states] = prev_cost
        window = np.lib.stride_tricks.sliding_window_view(padded, n_states)
        candidates = window + penalty[:, None]
        best_k = np.argmin(candidates, axis=0)
        best_cost = candidates[best_k, state_idx]
        backptr[i] = state_idx - (max_move - best_k)
        gv = gr_eval[i]
        row_mismatch = (gv - grid_gr) ** 2 / mismatch_scale if np.isfinite(gv) else 0.0
        cost_i = best_cost + row_mismatch
        if n_states >= 2:
            part = np.partition(cost_i, 1)
            margin[i] = float(part[1] - part[0])
        prev_cost = cost_i
    return backptr, prev_cost, margin


def _traceback(backptr, end_state):
    n_rows = backptr.shape[0]
    path = np.empty(n_rows, dtype=np.int64)
    path[-1] = int(end_state)
    for i in range(n_rows - 1, 0, -1):
        path[i - 1] = backptr[i, path[i]]
    return path


def _find_modes(final_cost, grid_step_ft, min_gap_ft):
    """Global minimum + best local minimum at least min_gap_ft away.

    Returns (i0, i1, dcost, gap_ft); i1 = -1 when no secondary mode exists.
    """
    n = final_cost.size
    i0 = int(np.argmin(final_cost))
    if n < 3:
        return i0, -1, np.inf, 0.0
    interior = final_cost[1:-1]
    is_min = (interior <= final_cost[:-2]) & (interior <= final_cost[2:])
    cand = np.flatnonzero(is_min) + 1
    min_gap_states = int(round(min_gap_ft / grid_step_ft))
    cand = cand[np.abs(cand - i0) >= max(min_gap_states, 1)]
    if cand.size == 0:
        return i0, -1, np.inf, 0.0
    i1 = int(cand[np.argmin(final_cost[cand])])
    dcost = float(final_cost[i1] - final_cost[i0])
    gap_ft = abs(i1 - i0) * grid_step_ft
    return i0, i1, dcost, gap_ft


def _r26_one_well(h, tw):
    """Commit/alt paths + mode stats for one well. Mirrors _track_beam_impl prep."""
    n_eval = int(eval_mask(h).sum())
    if n_eval == 0:
        raise ValueError("no eval zone")
    anchor = last_known_tvt(h)
    eval_idx = np.flatnonzero(eval_mask(h))
    grid_tvt, lo, hi = _build_grid(tw, anchor, _DEFAULT_GRID_STEP_FT, _DEFAULT_GRID_RANGE_FT)
    n_states = grid_tvt.size
    grid_gr = tw_gr_lookup(tw)(grid_tvt)
    gain, offset = fit_affine_gr(h, tw)
    gr_cal = apply_affine(h["GR"].to_numpy(dtype=float), gain, offset)
    gr_eval = gr_cal[eval_idx]
    anchor_idx = int(round((float(np.clip(anchor, lo, hi)) - lo) / _DEFAULT_GRID_STEP_FT))
    anchor_idx = int(np.clip(anchor_idx, 0, n_states - 1))
    init_cost = _DEFAULT_MOVE_PENALTY * np.abs(np.arange(n_states, dtype=float) - anchor_idx)

    backptr, final_cost, _ = _forward_viterbi_full(
        gr_eval, grid_gr, init_cost, _DEFAULT_MOVE_PENALTY, _DEFAULT_MISMATCH_SCALE,
        _DEFAULT_MAX_MOVE_PER_ROW,
    )
    i0, i1, dcost, gap_ft = _find_modes(final_cost, _DEFAULT_GRID_STEP_FT, R26_MIN_GAP_FT)
    commit = grid_tvt[_traceback(backptr, i0)]
    alt = grid_tvt[_traceback(backptr, i1)] if i1 >= 0 else commit.copy()
    y_true = h["TVT"].to_numpy(dtype=float)[eval_idx]
    return commit, alt, y_true, dcost, gap_ft, (i1 >= 0)


def r26_main():
    t0 = time.perf_counter()
    root = find_data_root()
    out_dir = output_dir()
    wells = list_wells("train", root)
    limit = _bank_limit()
    if limit is not None:
        wells = wells[:limit]
    print(f"# R26 bimodal census: {len(wells)} wells, grid step "
          f"{_DEFAULT_GRID_STEP_FT}ft range {_DEFAULT_GRID_RANGE_FT}ft, "
          f"min_gap {R26_MIN_GAP_FT}ft")

    rows_y, rows_commit, rows_alt, rows_wid = [], [], [], []
    w_name, w_dcost, w_gap, w_bimodal, w_nrows = [], [], [], [], []
    w_err0, w_err1 = [], []  # |datum error| committing to mode0 / mode1
    failures = 0
    for wi, well in enumerate(wells):
        try:
            h = load_horizontal(well, "train", root)
            tw = load_typewell(well, "train", root)
            commit, alt, y_true, dcost, gap_ft, bimodal = _r26_one_well(h, tw)
            fin = np.isfinite(y_true)
            if not fin.any():
                continue
            rows_y.append(y_true[fin]); rows_commit.append(commit[fin])
            rows_alt.append(alt[fin]); rows_wid.append(np.full(fin.sum(), wi, dtype=np.int32))
            w_name.append(well); w_dcost.append(dcost); w_gap.append(gap_ft)
            w_bimodal.append(bimodal); w_nrows.append(int(fin.sum()))
            w_err0.append(abs(float(np.mean(y_true[fin] - commit[fin]))))
            w_err1.append(abs(float(np.mean(y_true[fin] - alt[fin]))))
        except Exception as e:  # noqa: BLE001 -- census must survive odd wells
            failures += 1
            print(f"  [fail] {well}: {type(e).__name__}: {e}")
        if (wi + 1) % 100 == 0:
            print(f"  {wi + 1}/{len(wells)} wells, {time.perf_counter() - t0:.0f}s", flush=True)

    y = np.concatenate(rows_y); commit = np.concatenate(rows_commit)
    alt = np.concatenate(rows_alt); wid = np.concatenate(rows_wid)
    dcost = np.asarray(w_dcost); gap = np.asarray(w_gap)
    bimodal = np.asarray(w_bimodal, dtype=bool)
    err0 = np.asarray(w_err0); err1 = np.asarray(w_err1)

    def pooled(pred):
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    print(f"\nwells scored={len(w_name)} failures={failures} rows={y.size}")
    print(f"bimodal wells (gap>={R26_MIN_GAP_FT}ft local min): "
          f"{bimodal.sum()} ({100 * bimodal.mean():.1f}%)")
    if bimodal.any():
        print("gap_ft percentiles 10/25/50/75/90:",
              np.round(np.percentile(gap[bimodal], [10, 25, 50, 75, 90]), 1))
        print("dcost percentiles 10/25/50/75/90:",
              np.round(np.percentile(dcost[bimodal], [10, 25, 50, 75, 90]), 1))
        alt_better = err1[bimodal] < err0[bimodal]
        print(f"secondary mode closer to true datum: {100 * alt_better.mean():.1f}% "
              f"(coin flip claim: ~51%)")
        fin_d = np.isfinite(dcost[bimodal])
        if fin_d.sum() >= 3:
            r = np.corrcoef(dcost[bimodal][fin_d], alt_better[fin_d].astype(float))[0, 1]
            print(f"corr(dcost, secondary-correct) = {r:+.3f} (claim: ~0.05 = uninformative)")

    results = {"commit": pooled(commit)}
    hedge50 = 0.5 * commit + 0.5 * alt
    results["hedge50_always"] = pooled(hedge50)
    dcost_row = dcost[wid]
    for theta in R26_THETA_GRID:
        use = np.isfinite(dcost_row) & (dcost_row <= theta)
        pred = np.where(use, hedge50, commit)
        results[f"hedge50_if_dcost<={theta:g}"] = pooled(pred)
    for tau in R26_TAU_GRID:
        w1 = np.where(np.isfinite(dcost_row), np.exp(-dcost_row / tau), 0.0)
        w1 = w1 / (1.0 + w1)
        results[f"softmax_tau{tau:g}"] = pooled((1 - w1) * commit + w1 * alt)
    print("\npooled RMSE (beam candidate alone, train truth):")
    for k, v in results.items():
        print(f"  {k:26s} {v:8.4f}  (delta vs commit {v - results['commit']:+.4f})")

    np.savez_compressed(
        out_dir / "r26_bimodal.npz",
        used_wells=np.asarray(w_name), dcost=dcost, gap_ft=gap, bimodal=bimodal,
        err0=err0, err1=err1, n_rows=np.asarray(w_nrows, dtype=np.int32),
        y_true=y.astype(np.float32), commit=commit.astype(np.float32),
        alt=alt.astype(np.float32), well_idx=wid,
        summary_keys=np.asarray(list(results.keys())),
        summary_vals=np.asarray(list(results.values())),
    )
    print(f"\nsaved {out_dir / 'r26_bimodal.npz'}")
    print(f"# r26 total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    r26_main()
