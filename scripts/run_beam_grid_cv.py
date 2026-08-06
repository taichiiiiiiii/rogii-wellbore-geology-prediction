"""Beam-DP multi-configuration candidate-path bank (design.md R2, task: 1仮説1実測).

**Hypothesis under test**: a multi-configuration beam-DP grid (9-12 configs,
vs. the current 3-config ``DEFAULT_BEAM_CONFIGS`` ensemble) produces enough
*genuinely diverse* per-well candidate paths to widen the per-well-oracle
ceiling on the hard-well subset (``scripts/analyze_hard_wells.py``'s 4-way
oracle, currently 7.7732).

**Known structural constraint** (``registration/beam.py`` module docstring,
re-verified for this module's calibration-on/smooth-0 baseline against
``track_beam`` in this script's own dev notes): the DP's cost is
``mismatch/mismatch_scale + move_penalty*|d|`` at every row (including the
anchor prior), so multiplying the whole per-well cost by ``mismatch_scale``
leaves the argmin path bit-for-bit unchanged -- only the *product*
``move_penalty * mismatch_scale`` is an effective stiffness knob. Spending
grid slots on the ratio alone (``mismatch_scale`` a.k.a. "es" varied while
compensating ``move_penalty`` to hold the product fixed) would silently
duplicate paths and pad the config count without adding diversity. This grid
therefore holds ``mismatch_scale`` fixed at 150.0 throughout and spends its
budget on four axes that provably change either the DP's input observation
sequence or its reachable-state set:

1. **product** (``move_penalty`` at fixed ``mismatch_scale=150``), log-ish
   spaced at {2000, 3000, 4500} -- 3000 is the center of the
   already-measured full-773-well optimum (2400-3750, see ``beam.py``); 2000
   and 4500 extend past both ends to probe genuinely untested territory.
2. **GR affine calibration on/off** -- changes the observation values fed to
   the mismatch term directly.
3. **pre-match smoothing radius** {0, 2, 5} ft-index-radius rolling mean on
   the (optionally calibrated) GR series -- also changes the observation
   sequence; radius 5 matches the reference notebook's "sm5" tag (see
   ``registration/beam_grid.py`` module docstring / ``the referenced public notebook
bernubritz__rogii-lb7295-public-rebuild``).
4. **transition width** ``max_move_per_row`` in {1, 2, 3} -- changes which
   states are even reachable from row to row, independent of cost scale.

**Time gate** (task requirement, measured not assumed): a live 20-well timing
pass (phase 1 below) on exactly this 10-config grid measured **412ms/well
summed across all 10 configs** (range 364-436ms per individual config;
``max_move_per_row`` is the main driver, calibration/smoothing add only a
few ms of vectorized numpy overhead). Projected full-773-well cost for all
10 configs: ``773 * 0.4122s ~= 53.1 min`` -- under the 60-minute budget with
~7 min of margin for I/O and the post-hoc oracle/correlation analysis, which
is why the grid below is fixed at K=10 rather than the 12-config upper end
of the "9-12" range suggested by the task (a 12th config would use up nearly
all the remaining margin at ~412ms/well * 2 more configs ~= +10.6 min,
leaving too little slack).

Reuses (does not modify) ``registration/beam.py`` (via
``registration/beam_grid.py``'s new ``track_beam_grid`` wrapper) and
``scripts/analyze_hard_wells.py`` (imported as a module for the exact
4-way-oracle mechanism, extended here to (4+K)-way without touching that
file). Ground-truth ``TVT`` is read here only for scoring (pooled RMSE,
oracle candidate selection) -- never for the beam DP's own inference
(``registration/beam_grid.track_beam_grid`` only ever reads ``TVT_input``
and ``GR``, same leak boundary as ``registration/beam.py``).

Usage::

    uv run python scripts/run_beam_grid_cv.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_hard_wells as ahw  # noqa: E402  (reused, not modified)

from rogii import baseline  # noqa: E402
from rogii import data as D  # noqa: E402
from rogii.registration.beam_grid import BeamGridConfig, track_beam_grid  # noqa: E402

OUTPUT_PATH = _REPO_ROOT / "outputs" / "path_bank_beam.npz"
SPATIAL_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "spatial_cache"

TIME_GATE_N_WELLS = 20
TIME_GATE_BUDGET_MIN = 60.0
PROGRESS_EVERY = 100

CARRY_LAST_RMSE = 15.9099  # ledger floor, sanity anchor
BEAM_ENSEMBLE_BLEND_RMSE = 15.7232  # ledger, DEFAULT_BEAM_CONFIGS blend w0.7, sanity anchor
FOUR_WAY_ORACLE_RMSE = 7.7732  # ledger 2026-07-08, analyze_hard_wells.py oracle_substitution
BLEND_W = 0.7  # matches the ledger's beam blend-w0.7 convention, for the bonus blended table

# --------------------------------------------------------------------------- #
# the 10-config grid -- see module docstring for the axis rationale
# --------------------------------------------------------------------------- #
_MS = 150.0  # mismatch_scale held fixed everywhere (see docstring: varying it
# alone at constant product is a proven no-op on the DP path)

CONFIG_GRID: tuple[BeamGridConfig, ...] = (
    BeamGridConfig("mid_cal_sm0_mm3", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("loose_cal_sm0_mm3", move_penalty=2000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("stiff_cal_sm0_mm3", move_penalty=4500 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("mid_nocal_sm0_mm3", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=False, smooth_radius=0),
    BeamGridConfig("mid_cal_sm2_mm3", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=2),
    BeamGridConfig("mid_cal_sm5_mm3", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=3, gr_calibration=True, smooth_radius=5),
    BeamGridConfig("mid_cal_sm0_mm1", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=1, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("mid_cal_sm0_mm2", move_penalty=3000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=2, gr_calibration=True, smooth_radius=0),
    BeamGridConfig("loose_nocal_sm5_mm2", move_penalty=2000 / _MS, mismatch_scale=_MS,
                    max_move_per_row=2, gr_calibration=False, smooth_radius=5),
    BeamGridConfig("stiff_cal_sm5_mm1", move_penalty=4500 / _MS, mismatch_scale=_MS,
                    max_move_per_row=1, gr_calibration=True, smooth_radius=5),
)


def pooled_rmse(sq_err_sum: float, n: int) -> float:
    return float(np.sqrt(sq_err_sum / n)) if n else float("nan")


# --------------------------------------------------------------------------- #
# phase 1: time gate
# --------------------------------------------------------------------------- #


def time_gate(wells: list[str]) -> None:
    sample = wells[:TIME_GATE_N_WELLS]
    print(f"\n{'=' * 20} PHASE 1: time gate ({len(sample)} wells x {len(CONFIG_GRID)} configs) "
          f"{'=' * 20}\n")

    wdata = [(D.load_horizontal(w, "train"), D.load_typewell(w, "train")) for w in sample]

    per_config_s: dict[str, float] = {}
    for cfg in CONFIG_GRID:
        t0 = time.perf_counter()
        for h, tw in wdata:
            track_beam_grid(h, tw, cfg)
        dt = time.perf_counter() - t0
        per_config_s[cfg.label] = dt / len(wdata)
        print(f"  {cfg.label:24s} per_well={per_config_s[cfg.label] * 1000:7.1f}ms "
              f"product={cfg.product:.0f}")

    total_per_well_s = sum(per_config_s.values())
    projected_min = total_per_well_s * len(wells) / 60.0
    print(f"\n  sum per-well time across {len(CONFIG_GRID)} configs = "
          f"{total_per_well_s * 1000:.1f}ms/well")
    print(f"  projected {len(wells)}-well total = {projected_min:.1f} min "
          f"(budget {TIME_GATE_BUDGET_MIN:.0f} min)")
    if projected_min > TIME_GATE_BUDGET_MIN:
        raise SystemExit(
            f"[time-gate] FAIL: projected {projected_min:.1f} min exceeds the "
            f"{TIME_GATE_BUDGET_MIN:.0f} min budget for the full {len(wells)}-well run. "
            "Reduce CONFIG_GRID before proceeding."
        )
    print("  time-gate PASSED\n")


# --------------------------------------------------------------------------- #
# phase 2: full 773-well bank build
# --------------------------------------------------------------------------- #


def build_bank(wells: list[str]) -> dict[str, np.ndarray]:
    print(f"\n{'=' * 20} PHASE 2: full bank build ({len(wells)} wells) {'=' * 20}\n")

    n_configs = len(CONFIG_GRID)
    tvt_parts: list[list[np.ndarray]] = [[] for _ in range(n_configs)]
    margin_parts: list[list[np.ndarray]] = [[] for _ in range(n_configs)]
    well_idx_parts: list[np.ndarray] = []
    y_true_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    used_wells: list[str] = []
    eval_len_per_well: list[int] = []

    skipped = 0
    failed = 0
    t_start = time.perf_counter()

    for i, well in enumerate(wells, start=1):
        h = D.load_horizontal(well, "train")
        tw = D.load_typewell(well, "train")
        ez = D.eval_mask(h)
        if ez.sum() == 0 or h["TVT_input"].notna().sum() == 0:
            skipped += 1
            continue

        y_true = h["TVT"].to_numpy(dtype=np.float64)[ez]
        anchor_pred = baseline.predict_carry_last(h).astype(np.float64)
        n = y_true.size

        well_i = len(used_wells)
        for c_i, cfg in enumerate(CONFIG_GRID):
            result = track_beam_grid(h, tw, cfg)
            tvt = result.tvt
            margin = result.margin
            if tvt.shape != (n,) or margin.shape != (n,) or not np.all(np.isfinite(tvt)):
                failed += 1
                tvt = anchor_pred.copy()
                margin = np.zeros(n, dtype=float)
            tvt_parts[c_i].append(tvt.astype(np.float32))
            margin_parts[c_i].append(margin.astype(np.float32))

        well_idx_parts.append(np.full(n, well_i, dtype=np.int32))
        y_true_parts.append(y_true)
        anchor_parts.append(anchor_pred)
        used_wells.append(well)
        eval_len_per_well.append(n)

        if i % PROGRESS_EVERY == 0 or i == len(wells):
            elapsed = time.perf_counter() - t_start
            print(f"  [{i}/{len(wells)}] used={len(used_wells)} skipped={skipped} "
                  f"failed_configs={failed} elapsed={elapsed:.1f}s", flush=True)

    print(f"\n# build done: used={len(used_wells)} skipped={skipped} "
          f"failed_configs={failed} total_wells={len(wells)}")

    bank: dict[str, np.ndarray] = {
        "y_true": np.concatenate(y_true_parts),
        "anchor": np.concatenate(anchor_parts),
        "well_idx": np.concatenate(well_idx_parts),
        "used_wells": np.array(used_wells),
        "eval_len_per_well": np.array(eval_len_per_well, dtype=np.int32),
        "config_labels": np.array([c.label for c in CONFIG_GRID]),
        "product_per_config": np.array([c.product for c in CONFIG_GRID], dtype=np.float64),
        "max_move_per_config": np.array(
            [c.max_move_per_row for c in CONFIG_GRID], dtype=np.int32
        ),
        "gr_calibration_per_config": np.array(
            [c.gr_calibration for c in CONFIG_GRID], dtype=np.int8
        ),
        "smooth_radius_per_config": np.array(
            [c.smooth_radius for c in CONFIG_GRID], dtype=np.int32
        ),
    }
    for c_i in range(n_configs):
        bank[f"beam_tvt_{c_i}"] = np.concatenate(tvt_parts[c_i])
        bank[f"beam_margin_{c_i}"] = np.concatenate(margin_parts[c_i])

    return bank


def save_bank(bank: dict[str, np.ndarray]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **bank)
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\n# saved {OUTPUT_PATH} ({size_mb:.1f} MB)")


# --------------------------------------------------------------------------- #
# (a) per-config single pooled RMSE -- diversity confirmation
# --------------------------------------------------------------------------- #


def report_single_config_rmse(bank: dict[str, np.ndarray]) -> None:
    y_true = bank["y_true"]
    anchor = bank["anchor"]
    n_total = y_true.size
    labels = bank["config_labels"]

    print("\n" + "=" * 92)
    print("(a) 構成ごとの単独 pooled RMSE（多様性確認）")
    print("-" * 92)
    print(f"  carry_last (floor)                          pooled RMSE = {CARRY_LAST_RMSE:.4f}")
    print(f"  3-config ensemble blend w0.7 (ledger)        pooled RMSE = "
          f"{BEAM_ENSEMBLE_BLEND_RMSE:.4f}")
    print("-" * 92)
    print(f"{'config':<24}{'product':>9}{'raw pooled RMSE':>18}{'blend w0.7 RMSE':>18}")
    print("-" * 92)
    for c_i, label in enumerate(labels):
        tvt = bank[f"beam_tvt_{c_i}"]
        raw_sse = float(np.sum((y_true - tvt) ** 2))
        raw_rmse = pooled_rmse(raw_sse, n_total)
        blended = anchor + BLEND_W * (tvt - anchor)
        blend_sse = float(np.sum((y_true - blended) ** 2))
        blend_rmse = pooled_rmse(blend_sse, n_total)
        product = bank["product_per_config"][c_i]
        print(f"{label:<24}{product:>9.0f}{raw_rmse:>18.4f}{blend_rmse:>18.4f}")
    print("=" * 92)


# --------------------------------------------------------------------------- #
# (b) extended (4+K)-way oracle -- reuses analyze_hard_wells.py's mechanism
# --------------------------------------------------------------------------- #


def extended_oracle(bank: dict[str, np.ndarray]) -> tuple[float, float]:
    print("\n" + "=" * 92)
    print("(b) (4+K)-way oracle -- analyze_hard_wells.py の機構を流用（未変更）")
    print("-" * 92)

    oof = ahw.load_stack_oof()
    per_well = ahw.per_well_error_decomposition(oof)
    top_wells = ahw.print_hard_well_share(per_well, ahw.TOP_FRAC)

    y_true, pred, boundaries, used_wells_stack = (
        oof["y_true"], oof["pred"], oof["boundaries"], oof["used_wells"]
    )
    well_pos_stack = {str(w): i for i, w in enumerate(used_wells_stack)}
    total_rows = y_true.shape[0]
    baseline_sse = float(np.sum((y_true - pred) ** 2))
    baseline_rmse = pooled_rmse(baseline_sse, total_rows)

    top_set = [str(w) for w in top_wells]
    top_positions_stack = [well_pos_stack[w] for w in top_set]

    pf_blend_pred = oof["pf_blend"]
    anchor_pred = oof["anchor"].copy()

    spatial_pred = np.full(total_rows, np.nan, dtype=np.float64)
    for i in top_positions_stack:
        s, e_ = int(boundaries[i]), int(boundaries[i + 1])
        well = str(used_wells_stack[i])
        with np.load(SPATIAL_CACHE_DIR / f"{well}.npz") as npz:
            spatial_tvt = npz["spatial_tvt"].astype(np.float64)
        assert spatial_tvt.size == (e_ - s), f"{well}: spatial cache length mismatch"
        spatial_pred[s:e_] = spatial_tvt

    base_candidates: dict[str, np.ndarray] = {
        "stack": pred,
        "carry_last": anchor_pred,
        "PF blend": pf_blend_pred,
        "spatial": spatial_pred,
    }

    # ---- bank-side per-well lookup for the same top_wells set ---------------
    well_pos_bank = {str(w): i for i, w in enumerate(bank["used_wells"])}
    bank_boundaries = np.searchsorted(
        bank["well_idx"], np.arange(bank["used_wells"].size + 1)
    )
    labels = list(bank["config_labels"])

    missing_from_bank = [w for w in top_set if w not in well_pos_bank]
    if missing_from_bank:
        raise SystemExit(
            f"[extended-oracle] {len(missing_from_bank)} hard well(s) missing from the beam "
            f"bank (e.g. {missing_from_bank[:5]}) -- bank build must have skipped/failed them."
        )

    beam_pred_by_well: dict[str, dict[str, np.ndarray]] = {}
    for well in top_set:
        bi = well_pos_bank[well]
        bs, be = int(bank_boundaries[bi]), int(bank_boundaries[bi + 1])
        si = well_pos_stack[well]
        ss, se = int(boundaries[si]), int(boundaries[si + 1])
        assert (be - bs) == (se - ss), f"{well}: bank/stack eval-zone length mismatch"
        beam_pred_by_well[well] = {
            label: bank[f"beam_tvt_{c_i}"][bs:be] for c_i, label in enumerate(labels)
        }

    # ---- oracle runner: picks, per top hard well, whichever candidate in
    # ``candidate_names`` minimizes that well's true SSE (ground truth used
    # only to select -- an upper-bound probe, not a deployable predictor,
    # exactly as in ``analyze_hard_wells.oracle_substitution``) ---------------
    def _run_oracle(candidate_names: list[str]) -> tuple[float, dict[str, str]]:
        pred_oracle = pred.copy()
        picks: dict[str, str] = {}
        for i in top_positions_stack:
            s, e_ = int(boundaries[i]), int(boundaries[i + 1])
            well = str(used_wells_stack[i])
            best_name, best_sse = "stack", np.inf
            for name in candidate_names:
                if name.startswith("beam:"):
                    seg = beam_pred_by_well[well][name[len("beam:"):]]
                else:
                    seg = base_candidates[name][s:e_]
                if not np.all(np.isfinite(seg)):
                    continue
                sse = float(np.sum((seg - y_true[s:e_]) ** 2))
                if sse < best_sse:
                    best_sse, best_name = sse, name
            picks[well] = best_name
            if best_name.startswith("beam:"):
                pred_oracle[s:e_] = beam_pred_by_well[well][best_name[len("beam:"):]]
            else:
                pred_oracle[s:e_] = base_candidates[best_name][s:e_]
        rmse = pooled_rmse(float(np.sum((y_true - pred_oracle) ** 2)), total_rows)
        return rmse, picks

    four_way_names = list(base_candidates.keys())
    four_way_rmse, four_way_picks = _run_oracle(four_way_names)
    delta_ledger = four_way_rmse - FOUR_WAY_ORACLE_RMSE
    print(f"\n  baseline (stack v2 all-in, 全 well)      pooled RMSE = {baseline_rmse:.4f}")
    print(
        f"  4-way oracle 再現（regression check）    pooled RMSE = {four_way_rmse:.4f}  "
        f"(ledger 期待値 {FOUR_WAY_ORACLE_RMSE:.4f}, Δ={delta_ledger:+.4f})"
    )

    extended_names = four_way_names + [f"beam:{label}" for label in labels]
    extended_rmse, extended_picks = _run_oracle(extended_names)
    pick_counts_4 = Counter(four_way_picks.values())
    pick_counts_ext = Counter(extended_picks.values())
    delta_4way = extended_rmse - four_way_rmse
    print(
        f"  (4+{len(labels)})-way oracle（+beam×{len(labels)}）        pooled RMSE = "
        f"{extended_rmse:.4f}  (Δ vs 4-way = {delta_4way:+.4f})"
    )
    print(f"\n  4-way 内訳      = {dict(pick_counts_4)}")
    print(f"  (4+K)-way 内訳  = {dict(pick_counts_ext)}")
    n_beam_picks = sum(v for k, v in pick_counts_ext.items() if k.startswith("beam:"))
    print(f"  beam-grid が最良だった hard well 数 = {n_beam_picks}/{len(top_set)}")
    print("=" * 92)

    return four_way_rmse, extended_rmse


# --------------------------------------------------------------------------- #
# (c) error correlation matrix among the K beam configs
# --------------------------------------------------------------------------- #


def report_correlation_matrix(bank: dict[str, np.ndarray]) -> None:
    y_true = bank["y_true"]
    labels = list(bank["config_labels"])
    errs = np.stack([bank[f"beam_tvt_{i}"] - y_true for i in range(len(labels))])

    corr = np.corrcoef(errs)

    print("\n" + "=" * 92)
    print("(c) 構成間の誤差相関行列（多様性の定量, pooled 全773 well）")
    print("-" * 92)
    header = "".join(f"{lab[:10]:>12}" for lab in labels)
    print(f"{'':<24}{header}")
    for i, lab in enumerate(labels):
        row = "".join(f"{corr[i, j]:>12.3f}" for j in range(len(labels)))
        print(f"{lab:<24}{row}")

    off_diag = corr[~np.eye(len(labels), dtype=bool)]
    print("-" * 92)
    print(f"  off-diagonal mean = {off_diag.mean():.4f}  min = {off_diag.min():.4f}  "
          f"max = {off_diag.max():.4f}")
    i_min, j_min = np.unravel_index(np.argmin(corr + np.eye(len(labels)) * 2), corr.shape)
    print(f"  最も無相関なペア: {labels[i_min]} / {labels[j_min]}  (r={corr[i_min, j_min]:.4f})")
    print("=" * 92)


def main() -> None:
    t_start = time.perf_counter()
    wells = D.list_wells("train")
    print(f"# train wells: {len(wells)}")
    print(f"# config grid: {len(CONFIG_GRID)} configs")

    time_gate(wells)

    bank = build_bank(wells)
    save_bank(bank)

    report_single_config_rmse(bank)
    four_way_rmse, extended_rmse = extended_oracle(bank)
    report_correlation_matrix(bank)

    print("\n" + "=" * 92)
    print("## 判定")
    print("-" * 92)
    widened = FOUR_WAY_ORACLE_RMSE - extended_rmse
    print(
        f"  4-way oracle {four_way_rmse:.4f}（ledger {FOUR_WAY_ORACLE_RMSE:.4f}）-> "
        f"(4+{len(CONFIG_GRID)})-way oracle {extended_rmse:.4f}  "
        f"(beam grid が oracle を広げた幅 = {widened:+.4f}ft)"
    )
    print("=" * 92)

    print(f"\ntotal runtime: {(time.perf_counter() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
