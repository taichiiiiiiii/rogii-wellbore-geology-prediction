"""Build a per-row eval-zone feature cache for the information audit (peraudit).

Question under test: how much of the eval-zone drift is predictable from the
well's OWN data -- especially the drilled trajectory, which is fully known at
inference time (X, Y, Z, MD exist for every row of test wells too).

Target definition (the only one used anywhere in this audit)::

    d[i] = TVT_true[i] - TVT_anchor          for i in the eval zone

where ``TVT_anchor`` is the last non-NaN ``TVT_input`` before the eval zone.
Predicting ``d = 0`` is exactly the carry-last (flat-TVT) baseline, i.e. the
"surface carried parallel to the drilled trajectory" model.  In surface terms
``S = TVT + Z`` so ``d = S_true - (Z + TVT_anchor)``: the residual after the
trajectory itself has explained everything it can.

Feature legality rules enforced here:
  * The FULL trajectory (MD, X, Y, Z) may be used at any row, including
    "future" rows inside the eval zone -- it is known at inference.
  * GR may be used everywhere (present in test).
  * TVT / TVT_input may be used ONLY on known-zone rows (index <= anchor).
  * Formation-top columns (ANCC/ASTNU/.../BUDA) are train-only -> never used.
  * The typewell (TVT, GR) may be used (present in test); its Geology column
    is train-only -> never used.

Outputs (into the scratchpad cache dir):
    X.npy      float32 memmap (N_rows, n_features)
    y.npy      float32 (N_rows,)   the drift target d
    well.npy   int32   (N_rows,)   well index -> groups for GroupKFold
    row.npy    int32   (N_rows,)   row index inside the well (for submission ids)
    meta.json  feature names, well list, per-well summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = Path(
    "/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad/peraudit"
)

# ---------------------------------------------------------------- helpers ---


def _roll_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling mean with edge padding (NaN-free input assumed)."""
    if w <= 1:
        return a.astype(np.float64)
    k = np.ones(w) / w
    pad = w // 2
    ap = np.concatenate([np.full(pad, a[0]), a, np.full(w - 1 - pad, a[-1])])
    return np.convolve(ap, k, mode="valid")


def _roll_std(a: np.ndarray, w: int) -> np.ndarray:
    m = _roll_mean(a, w)
    m2 = _roll_mean(a * a, w)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _centred_slope(z: np.ndarray, md: np.ndarray, w: int) -> np.ndarray:
    """dZ/dMD over a centred window of width ~w (uses the full trajectory)."""
    n = len(z)
    h = w // 2
    lo = np.clip(np.arange(n) - h, 0, n - 1)
    hi = np.clip(np.arange(n) + h, 0, n - 1)
    dz = z[hi] - z[lo]
    dm = md[hi] - md[lo]
    return np.where(dm > 0, dz / np.where(dm > 0, dm, 1.0), 0.0)


def _tail_slope(y: np.ndarray, x: np.ndarray, a: int, n: int) -> float:
    """OLS slope of y vs x over the last ``n`` samples ending at index ``a``."""
    lo = max(0, a - n + 1)
    xs, ys = x[lo : a + 1], y[lo : a + 1]
    ok = np.isfinite(ys) & np.isfinite(xs)
    if ok.sum() < 3:
        return 0.0
    xs, ys = xs[ok], ys[ok]
    xm = xs.mean()
    den = np.sum((xs - xm) ** 2)
    if den <= 0:
        return 0.0
    return float(np.sum((xs - xm) * (ys - ys.mean())) / den)


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs; returns (filled, was_nan)."""
    a = a.astype(np.float64)
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    return np.interp(idx, idx[ok], a[ok])


FEATS: list[str] = []


def _f(name: str) -> str:
    if name not in FEATS:
        FEATS.append(name)
    return name


# ------------------------------------------------------------ per-well ------


def build_well(well: str, split: str = "train"):
    """Return (X, y, row_idx, summary) for one well, or None if unusable."""
    from rogii.data import data_root

    p = data_root() / split / f"{well}__horizontal_well.csv"
    h = pd.read_csv(p)
    ti = h["TVT_input"].to_numpy(np.float64)
    ev = np.isnan(ti)
    if not ev.any() or ev.all():
        return None
    known = np.where(~ev)[0]
    a = int(known[-1])
    if a < 50:
        return None
    ei = np.where(ev)[0]
    ei = ei[ei > a]  # eval rows strictly after the anchor
    if len(ei) < 10:
        return None

    md = h["MD"].to_numpy(np.float64)
    x = h["X"].to_numpy(np.float64)
    y_ = h["Y"].to_numpy(np.float64)
    z = h["Z"].to_numpy(np.float64)
    gr_raw = h["GR"].to_numpy(np.float64)
    tvt = h["TVT"].to_numpy(np.float64) if "TVT" in h.columns else None

    n = len(md)
    tvt_a = float(ti[a])
    md_a = float(md[a])
    z_a = float(z[a])

    # ---------------- trajectory geometry (legal everywhere) ----------------
    dmd = md - md_a
    dz = z - z_a
    dx = x - x[a]
    dy = y_ - y_[a]

    slopes = {w: _centred_slope(z, md, w) for w in (11, 51, 151, 401, 1001)}
    # Restrict all tail fits to the LATERAL part of the known zone: the build
    # section has |dZ/dMD| ~ 1 and would swamp any slope estimate.
    lateral = np.abs(slopes[151]) < 0.25
    ti_lat = np.where(lateral, ti, np.nan)
    z_lat = np.where(lateral, z, np.nan)
    # known-zone tail slopes of Z (the trajectory's own trend at the anchor)
    zsl = {L: _tail_slope(z_lat, md, a, L) for L in (200, 500, 1500)}
    # known-zone tail slope of the TRUTH-bearing TVT_input (legal: known zone)
    tsl = {L: _tail_slope(ti_lat, md, a, L) for L in (200, 500, 1500)}
    # surface S = TVT + Z, known only on the known zone
    s_known = np.where(lateral, ti + z, np.nan)
    ssl = {L: _tail_slope(s_known, md, a, L) for L in (200, 500, 1500)}
    for dd in (zsl, tsl, ssl):
        for k_ in dd:
            dd[k_] = float(np.clip(dd[k_], -0.25, 0.25))

    # curvature / steering activity
    sl51 = slopes[51]
    curv = np.gradient(sl51, md)
    abscurv = np.abs(curv)
    cum_abscurv = np.cumsum(abscurv)
    sgn = np.sign(sl51 - zsl[500])
    flips = np.cumsum(np.abs(np.diff(sgn, prepend=sgn[0])) > 0).astype(np.float64)

    # 3D dogleg (uses X, Y too)
    ddx = np.gradient(x, md)
    ddy = np.gradient(y_, md)
    ddz = np.gradient(z, md)
    nrm = np.sqrt(ddx**2 + ddy**2 + ddz**2) + 1e-9
    ux, uy, uz = ddx / nrm, ddy / nrm, ddz / nrm
    dot = np.clip(
        np.r_[1.0, ux[1:] * ux[:-1] + uy[1:] * uy[:-1] + uz[1:] * uz[:-1]], -1, 1
    )
    dogleg = np.degrees(np.arccos(dot)) * 100.0  # deg / 100 ft
    dogleg_r = _roll_mean(dogleg, 101)

    hd = np.sqrt(dx**2 + dy**2)  # horizontal displacement from the anchor

    # look-ahead / look-behind on the trajectory (LEGAL: trajectory is known)
    def shift_dz(k: int) -> np.ndarray:
        i = np.clip(np.arange(n) + k, 0, n - 1)
        return z[i] - z

    md_end = float(md[-1])

    # -------------------------- GR (heel-calibrated) ------------------------
    gr_ok = np.isfinite(gr_raw)
    gr = _interp_nan(gr_raw)
    tw_path = data_root() / split / f"{well}__typewell.csv"
    gr_cal = np.zeros(n)
    tw_pred = np.zeros(n)
    tw_slope = np.zeros(n)
    have_tw = 0.0
    gain, off = 1.0, 0.0
    if tw_path.exists():
        tw = pd.read_csv(tw_path)
        tv = tw["TVT"].to_numpy(np.float64)
        tg = tw["GR"].to_numpy(np.float64)
        m = np.isfinite(tv) & np.isfinite(tg)
        tv, tg = tv[m], tg[m]
        if len(tv) > 20:
            o = np.argsort(tv)
            tv, tg = tv[o], tg[o]
            have_tw = 1.0
            # heel calibration: regress typewell GR (sampled at the KNOWN TVT)
            # onto the well's own GR, using known-zone rows only.
            kk = known[gr_ok[known]]
            if len(kk) > 50:
                tg_at = np.interp(ti[kk], tv, tg)
                A = np.c_[gr[kk], np.ones(len(kk))]
                try:
                    coef, *_ = np.linalg.lstsq(A, tg_at, rcond=None)
                    if np.isfinite(coef).all() and 0.05 < coef[0] < 20:
                        gain, off = float(coef[0]), float(coef[1])
                except np.linalg.LinAlgError:
                    pass
            gr_cal = gain * gr + off
            # typewell GR expected at the CARRY-LAST depth estimate, and its
            # local sensitivity dGR/dTVT there (a 1-D inversion of the residual)
            tw_pred[:] = np.interp(tvt_a, tv, tg)
            eps = 2.0
            tw_slope[:] = (
                np.interp(tvt_a + eps, tv, tg) - np.interp(tvt_a - eps, tv, tg)
            ) / (2 * eps)
    gr_res = gr_cal - tw_pred
    gr_res_r = {w: _roll_mean(gr_res, w) for w in (51, 201, 801)}
    gr_std_r = _roll_std(gr_cal, 201)
    inv = np.where(np.abs(tw_slope) > 1e-3, gr_res_r[201] / np.where(np.abs(tw_slope) > 1e-3, tw_slope, 1.0), 0.0)
    inv = np.clip(inv, -60, 60)

    # ------------------------ known-zone summary scalars --------------------
    kn_len = md_a - float(md[0])
    kn_n = float(len(known))
    ti_k = ti[known]
    lin = np.polyfit(md[known], ti_k, 1) if len(known) > 5 else np.array([0.0, 0.0])
    resid_k = ti_k - np.polyval(lin, md[known])
    kn_resid_std = float(np.std(resid_k))
    kn_tvt_std = float(np.std(ti_k))
    kn_tvt_rng = float(ti_k.max() - ti_k.min())
    slope_stab = float(np.std([tsl[200], tsl[500], tsl[1500]]))
    z_slope_stab = float(np.std([zsl[200], zsl[500], zsl[1500]]))
    # how well did the well hold TVT over its known lateral?
    lat = known[md[known] > md_a - 2000]
    kn_tail_std = float(np.std(ti[lat])) if len(lat) > 20 else kn_tvt_std
    gr_cov_eval = float(np.mean(gr_ok[ei]))

    # ------------------------------ assemble --------------------------------
    cols: list[tuple[str, np.ndarray]] = []

    def add(name: str, arr) -> None:
        _f(name)
        cols.append((name, np.asarray(arr, np.float64)))

    e = ei  # index into the well arrays
    ones = np.ones(len(e))

    # --- group A: trajectory-only (features 0..) ---
    add("dmd", dmd[e])
    add("log_dmd", np.log1p(dmd[e]))
    add("frac_along", dmd[e] / max(md_end - md_a, 1.0))
    add("dz", dz[e])
    add("dz_dev_tail", dz[e] - zsl[500] * dmd[e])
    add("dz_dev_tail200", dz[e] - zsl[200] * dmd[e])
    add("dz_dev_tail1500", dz[e] - zsl[1500] * dmd[e])
    add("hd", hd[e])
    add("dz_over_hd", dz[e] / (hd[e] + 1.0))
    for w in (11, 51, 151, 401, 1001):
        add(f"slope{w}", slopes[w][e])
        add(f"slope{w}_rel", slopes[w][e] - zsl[500])
    add("slope_ratio", slopes[51][e] / (np.abs(zsl[500]) + 1e-4))
    add("curv51", curv[e])
    add("abscurv_r", _roll_mean(abscurv, 201)[e])
    add("cum_abscurv", cum_abscurv[e] - cum_abscurv[a])
    add("cum_abscurv_rate", (cum_abscurv[e] - cum_abscurv[a]) / (dmd[e] + 1.0))
    add("flips", flips[e] - flips[a])
    add("flips_rate", (flips[e] - flips[a]) / (dmd[e] + 1.0))
    add("dogleg", dogleg_r[e])
    add("slopestd201", _roll_std(sl51, 201)[e])
    add("slopestd801", _roll_std(sl51, 801)[e])
    for k in (-800, -300, -100, 100, 300, 800):
        add(f"lookdz{k}", shift_dz(k)[e])
    add("z_minus_evalmean", (z[e] - z[ei].mean()))
    add("z_rank_eval", (np.argsort(np.argsort(z[e])) / max(len(e) - 1, 1)))
    add("z_minus_evalmin", z[e] - z[ei].min())
    add("z_minus_evalmax", z[e] - z[ei].max())
    add("zsl200", zsl[200] * ones)
    add("zsl500", zsl[500] * ones)
    add("zsl1500", zsl[1500] * ones)
    add("z_slope_stab", z_slope_stab * ones)
    add("eval_len", (md_end - md_a) * ones)
    n_traj = len(cols)

    # --- group B: heel-calibrated GR ---
    add("gr_res", gr_res[e])
    add("gr_res51", gr_res_r[51][e])
    add("gr_res201", gr_res_r[201][e])
    add("gr_res801", gr_res_r[801][e])
    add("gr_inv", inv[e])
    add("gr_std201", gr_std_r[e])
    add("gr_cal", gr_cal[e])
    add("gr_ok", gr_ok[e].astype(float))
    add("gr_cov_eval", gr_cov_eval * ones)
    add("tw_slope", tw_slope[e])
    add("tw_pred", tw_pred[e])
    add("gr_gain", gain * ones)
    add("have_tw", have_tw * ones)
    n_gr = len(cols) - n_traj

    # --- group C: known-zone statistics ---
    add("tvt_a", tvt_a * ones)
    add("kn_len", kn_len * ones)
    add("kn_n", kn_n * ones)
    add("kn_resid_std", kn_resid_std * ones)
    add("kn_tvt_std", kn_tvt_std * ones)
    add("kn_tvt_rng", kn_tvt_rng * ones)
    add("kn_tail_std", kn_tail_std * ones)
    add("tsl200", tsl[200] * ones)
    add("tsl500", tsl[500] * ones)
    add("tsl1500", tsl[1500] * ones)
    add("slope_stab", slope_stab * ones)
    add("ssl500", ssl[500] * ones)
    add("tsl500_x_dmd", tsl[500] * dmd[e])
    add("tsl1500_x_dmd", tsl[1500] * dmd[e])

    Xw = np.column_stack([c[1] for c in cols]).astype(np.float32)
    Xw[~np.isfinite(Xw)] = 0.0
    yw = (tvt[e] - tvt_a).astype(np.float32) if tvt is not None else None

    summary = dict(
        well=well,
        n_eval=int(len(e)),
        anchor=a,
        tvt_a=tvt_a,
        md_a=md_a,
        eval_len=float(md_end - md_a),
        tsl500=tsl[500],
        tsl1500=tsl[1500],
        zsl500=zsl[500],
        gr_cov_eval=gr_cov_eval,
        n_traj=n_traj,
        n_gr=n_gr,
    )
    return Xw, yw, e.astype(np.int32), summary


def main() -> None:
    from rogii.data import list_wells

    CACHE.mkdir(parents=True, exist_ok=True)
    wells = list_wells("train")
    Xs, ys, rows, wid, summ = [], [], [], [], []
    for k, w in enumerate(wells):
        try:
            out = build_well(w)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  !! {w}: {exc}")
            continue
        if out is None:
            continue
        Xw, yw, e, s = out
        Xs.append(Xw)
        ys.append(yw)
        rows.append(e)
        wid.append(np.full(len(e), len(summ), np.int32))
        summ.append(s)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(wells)} wells, {sum(len(a) for a in ys):,} rows")

    X = np.concatenate(Xs)
    del Xs
    y = np.concatenate(ys)
    row = np.concatenate(rows)
    g = np.concatenate(wid)
    print(f"X {X.shape} {X.nbytes / 1e6:.0f} MB  wells={len(summ)}")
    np.save(CACHE / "X.npy", X)
    np.save(CACHE / "y.npy", y)
    np.save(CACHE / "row.npy", row)
    np.save(CACHE / "well.npy", g)
    (CACHE / "meta.json").write_text(
        json.dumps({"features": FEATS, "wells": summ}, indent=1)
    )
    print("features:", len(FEATS))


if __name__ == "__main__":
    main()
