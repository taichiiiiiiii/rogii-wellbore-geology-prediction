"""Build the mis-tie network levelling post-process kernel.

The levelling math (a spatial mis-tie network over S = TVT + Z, in the
tradition of survey mis-tie levelling, plus an IDW spatial predictor, a
per-well tail anchor, and a fixed-weight clipped blend) was validated
locally against a 40-well held-out audit (scripts/levelling_{cache,common,
network,predict,rules}.py, FINAL_perwell_a_pair450_clip.csv) and is now
ported into the four functions below (solve_network_offsets,
predict_levelled_surface, fit_query_anchor, gate_and_blend). Fidelity
against that audit is checked by scripts/harness/verify_levelling_fidelity.py
(max per-well RMSE diff must be < 1e-6).

Base: rogii-gs145-pfmean035.ipynb with PF_SCALE_PARTNER_WEIGHT forced back to
0.0 -- the same "isolate one change" move build_surface_pp_probe.py makes, and
for the same reason: pfmean035 is the only base notebook this repo has that
already carries the gs*1.45 tracker-scale diff, so starting from it and
turning its own knob off is cleaner than hand-porting gs*1.45 onto a fresh
base (no w062 base exists anywhere under scratchpad/).

Two edits beyond the PF knob:
  1. A tiny cell PREPENDED at index 0 that stamps _LEVELLING_T0 = time.time()
     before anything else runs, so the post-process can measure true elapsed
     wall-clock time against the 9h budget regardless of where in the elapsed
     ~8.3h the pipeline actually is when it reaches this stage.
  2. The levelling cell APPENDED after every existing cell -- the same
     placement build_surface_pp_probe.py used, because the award write-up
     found corrections placed before the visible-prefix/model-package stages
     confuse those stages' own per-well backtests; last-after-everything is
     the placement that already shipped clean once.

LEVELLING_ENABLED now defaults to True: the algorithm is validated and the
elapsed-time guard (LEVELLING_GUARD_HOURS=8.6) still protects the 9h budget
if this stage is ever reached too late in a run. Set LEVELLING_ENABLED=False
before running the notebook to fall back to the prior identity / no-op
behaviour.
"""

from __future__ import annotations

import os
import ast
import json
import shutil
from pathlib import Path

HERE = Path(os.environ.get("ROGII_WORK", "work"))
SRC = HERE / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
SLUG = "rogii-gs145-levelling"
OUT = HERE / "levelling_kernel" / SLUG

# Turn the pf_mean blend back off so this probe isolates one change, exactly
# as build_surface_pp_probe.py does on the same base notebook.
PF_OFF = ("PF_SCALE_PARTNER_WEIGHT = 0.35", "PF_SCALE_PARTNER_WEIGHT = 0.0")

BOOT_CELL = r'''
# ============================================================================
# Elapsed-time guard boot stamp. Must be the very first cell in the notebook:
# the levelling post-process needs true wall-clock elapsed time since the
# start of the run, not since its own cell started, to know whether it is
# safe to spend time inside the 9h budget.
# ============================================================================
import time as _levelling_boot_time
_LEVELLING_T0 = _levelling_boot_time.time()
'''

CELL = r'''
# ============================================================================
# Mis-tie network levelling post-process.
#
# S = TVT + Z is the regional surface elevation implied by a well's own
# trajectory. Every train well has S known over its FULL lateral (ground
# truth TVT). Every test well has S known only over its KNOWN prefix
# (TVT_input not NaN) -- its eval zone is exactly the S values this stage is
# trying to help predict, so it can never be a donor for itself or anyone
# else.
#
# Algorithm (gate DISABLED -- a fixed-weight clipped blend, not a per-well
# accept/reject gate):
#   c_d   = mis-tie network levelling over donor wells: nearest-point
#           cross-well surface differences within LEVELLING_PAIR_RADIUS_FT
#           (stride LEVELLING_PAIR_STRIDE), Huber IRLS, gauge = centre each
#           connected component.
#   S_hat = IDW over donors: K=LEVELLING_KNN nearest within
#           LEVELLING_RADIUS_FT, power 2, donor stride LEVELLING_DONOR_STRIDE,
#           query well excluded, donor surface value = (TVT + Z) - c_d.
#   c_q   = mean of (S_query - S_hat) over the last LEVELLING_ANCHOR_TAIL_ROWS
#           known rows of the query well (rows with real donor coverage).
#   spat  = (S_hat + c_q) - Z; wherever there is no donor coverage or no
#           anchor, spat falls back to the base prediction row by row.
#   cap   = LEVELLING_CLIP_KAPPA * MAD_ft, where MAD_ft = median(|spat-base|)
#           pooled over ALL judged eval rows this run (fallback rows
#           INCLUDED as a zero contribution -- see note below).
#   final = base + LEVELLING_BLEND_W * clip(spat - base, +-cap)
#
# Donor variant "a" (train wells only): the network solve (c_d) uses every
# mounted well's LEGALLY VISIBLE points -- train wells in full, test wells by
# their known prefix only -- because that is real information available at
# inference and only sharpens the mis-tie network. The spatial IDW field
# (S_hat), by contrast, is restricted to TRAIN WELLS ONLY: this is the
# "donor variant a" validated in scripts/levelling_predict.py (simpler than
# variant "b", measured equivalent there). A query test well is therefore
# excluded from its own field by construction (it was never a train well);
# `exclude_well` is kept on predict_levelled_surface as a defensive no-op.
#
# Blend weight and cap, revision history:
#   - The 40-well k40 audit's in-sample optimum ("clip" family,
#     scripts/levelling_rules.py) is w=0.45 with a FROZEN absolute cap of
#     16.7616 ft (FINAL_perwell_a_pair450_clip.csv); a closely related
#     family ("cliprel", kappa=3.3523 x live pooled MAD) scores almost
#     identically in-sample (6.9270 ft vs 6.9275 ft pooled).
#   - An adversarial robustness review of that in-sample choice found w=0.45
#     is optimal only at the audit's own base-model error level and its
#     benefit sign-flips once the base model is meaningfully better (~32%
#     lower pooled RMSE than the audit); w=0.30 minimax-dominates w=0.45
#     across degraded-transfer scenarios (worst-case regret +0.146 vs
#     +0.690 ft; P(hurts) at a 22% better base drops from 14.7% to 2.2%).
#     w=LEVELLING_BLEND_W=0.30 with a LIVE kappa*MAD_ft cap (kappa=
#     LEVELLING_CLIP_KAPPA=3.3523203402896797, cliprel's own in-sample
#     multiplier) is the rule actually shipped here, since a cap that
#     rescales with the deployment population's own error spread is more
#     robust to distribution shift than one frozen on the audit.
#   - MAD_ft pools |spat-base| over every judged eval row, INCLUDING
#     fallback rows (no donor coverage / no anchor) as an explicit zero
#     contribution -- this is a deliberate choice to match the frozen-review
#     reference (8.1723 -> 7.0903 ft, delta -1.0820, 24/40 wells helped,
#     worst +3.38 ft (well 6ea22614), best -4.44 ft (well c8d9680c) on the
#     k40 audit) rather than excluding those rows from the pool, which shifts
#     MAD_ft only mildly here (no fallback rows occur in the 40-well audit
#     itself, so both conventions give the identical MAD_ft=5.2270 ft there)
#     but matters for a population with real coverage gaps.
# ============================================================================

# ---- parameter block (all constants; LEVELLING_ENABLED=False is identity) --
LEVELLING_ENABLED = bool(globals().get('LEVELLING_ENABLED', True))
LEVELLING_RADIUS_FT = float(globals().get('LEVELLING_RADIUS_FT', 1500.0))
LEVELLING_KNN = int(globals().get('LEVELLING_KNN', 256))
LEVELLING_DONOR_STRIDE = int(globals().get('LEVELLING_DONOR_STRIDE', 8))
LEVELLING_PAIR_RADIUS_FT = float(globals().get('LEVELLING_PAIR_RADIUS_FT', 450.0))
LEVELLING_PAIR_STRIDE = int(globals().get('LEVELLING_PAIR_STRIDE', 16))
LEVELLING_PAIR_MIN_MATCHES = int(globals().get('LEVELLING_PAIR_MIN_MATCHES', 20))
LEVELLING_PAIR_NN_K = int(globals().get('LEVELLING_PAIR_NN_K', 48))
LEVELLING_PAIR_HUBER = float(globals().get('LEVELLING_PAIR_HUBER', 6.0))
LEVELLING_PAIR_IRLS_ITERS = int(globals().get('LEVELLING_PAIR_IRLS_ITERS', 8))
LEVELLING_ANCHOR_TAIL_ROWS = int(globals().get('LEVELLING_ANCHOR_TAIL_ROWS', 300))
LEVELLING_MIN_KNOWN_ROWS = int(globals().get('LEVELLING_MIN_KNOWN_ROWS', 50))
LEVELLING_BLEND_W = float(globals().get('LEVELLING_BLEND_W', 0.30))
LEVELLING_CLIP_KAPPA = float(globals().get('LEVELLING_CLIP_KAPPA', 3.3523203402896797))
LEVELLING_GUARD_HOURS = float(globals().get('LEVELLING_GUARD_HOURS', 8.6))

print(f'[LEVELLING] params enabled={LEVELLING_ENABLED} radius_ft={LEVELLING_RADIUS_FT} '
      f'knn={LEVELLING_KNN} donor_stride={LEVELLING_DONOR_STRIDE} '
      f'pair_radius_ft={LEVELLING_PAIR_RADIUS_FT} pair_stride={LEVELLING_PAIR_STRIDE} '
      f'anchor_tail_rows={LEVELLING_ANCHOR_TAIL_ROWS} min_known_rows={LEVELLING_MIN_KNOWN_ROWS} '
      f'blend_w={LEVELLING_BLEND_W} cap_mult={LEVELLING_CLIP_KAPPA} '
      f'guard_hours={LEVELLING_GUARD_HOURS}')

import numpy as _lv_np
import pandas as _lv_pd
import time as _lv_time
from pathlib import Path as _LvPath


def _lv_data_root():
    """Resolve the competition data root the same way the other guarded
    correction cells in this notebook do (COMPETITION_DATA_ROOT, else CFG)."""
    root = globals().get('COMPETITION_DATA_ROOT', None)
    if root is None:
        _cfg = globals().get('CFG', None)
        if _cfg is not None:
            root = getattr(_cfg, 'DATA', getattr(_cfg, 'dataset_path', None))
    if root is None:
        root = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
    return _LvPath(root)


def build_donor_table(data_root, stride=LEVELLING_DONOR_STRIDE, include_test=True):
    """Enumerate mounted wells and build a strided donor table.

    Train wells: TVT is ground truth over the full lateral, so every strided
    row is a legal donor. Test wells (when include_test=True): only the
    known prefix (TVT_input not NaN) is legal -- the eval zone is exactly
    what this stage predicts and must never leak into the donor pool, for
    itself or anyone else. The stride is applied FIRST (0-indexed from the
    start of that well's own rows), THEN the known-prefix filter -- matching
    the reference's donor_mask() row selection exactly; filtering before
    striding would silently re-index the stride against a different row set.

    Memory: usecols limits what pandas parses per file, stride thins rows
    before concatenation, and X/Y/S stay float64 (Texas State Plane feet are
    ~3e6; float32 would quantise them to ~0.25 ft and destroy the ~300 ft
    well spacing the network/IDW steps resolve -- see
    scripts/levelling_cache.py's docstring for the same warning). No
    well-pair or row-pair matrix is ever materialised here -- this function
    only builds flat per-row arrays.

    Returns dict(well=<U8 array>, x=float64, y=float64, s=float64).
    """
    wells_out, x_out, y_out, s_out = [], [], [], []

    for fp in sorted((data_root / 'train').glob('*__horizontal_well.csv')):
        wid = fp.name.split('__')[0]
        h = _lv_pd.read_csv(fp, usecols=['X', 'Y', 'Z', 'TVT'])
        h = h.iloc[::stride]
        if h.empty:
            continue
        wells_out.append(_lv_np.full(len(h), wid))
        x_out.append(h['X'].to_numpy(dtype='float64'))
        y_out.append(h['Y'].to_numpy(dtype='float64'))
        s_out.append((h['TVT'] + h['Z']).to_numpy(dtype='float64'))

    if include_test:
        for fp in sorted((data_root / 'test').glob('*__horizontal_well.csv')):
            wid = fp.name.split('__')[0]
            h = _lv_pd.read_csv(fp, usecols=['X', 'Y', 'Z', 'TVT_input'])
            h = h.iloc[::stride]
            h = h[h['TVT_input'].notna()]
            if h.empty:
                continue
            wells_out.append(_lv_np.full(len(h), wid))
            x_out.append(h['X'].to_numpy(dtype='float64'))
            y_out.append(h['Y'].to_numpy(dtype='float64'))
            s_out.append((h['TVT_input'] + h['Z']).to_numpy(dtype='float64'))

    if not wells_out:
        return {'well': _lv_np.array([], dtype='<U8'),
                'x': _lv_np.array([], dtype='float64'),
                'y': _lv_np.array([], dtype='float64'),
                's': _lv_np.array([], dtype='float64')}
    return {
        'well': _lv_np.concatenate(wells_out),
        'x': _lv_np.concatenate(x_out),
        'y': _lv_np.concatenate(y_out),
        's': _lv_np.concatenate(s_out),
    }


def solve_network_offsets(donor_table,
                           radius=LEVELLING_PAIR_RADIUS_FT,
                           min_matches=LEVELLING_PAIR_MIN_MATCHES,
                           nn_k=LEVELLING_PAIR_NN_K,
                           huber=LEVELLING_PAIR_HUBER,
                           irls_iters=LEVELLING_PAIR_IRLS_ITERS):
    """Mis-tie network levelling: survey-style datum adjustment.

    Every well w carries an unknown datum constant c_w; levelled surfaces
    S_w - c_w should all sample one true structural surface. Where two
    wells pass near each other their levelled surfaces must agree, giving
    one pairwise observation: c_i - c_j ~= d_ij = robust (median) mis-tie of
    S_i(p) - S_j(nearest q) over nearest-point matches, symmetrised (both
    directions averaged, cancelling most of the surface gradient across the
    well spacing). The network is solved by Huber-IRLS weighted least
    squares (a handful of pairs crossing a fault or targeting different
    zones would otherwise drag a whole component); gauge freedom is fixed
    by centring each connected component independently.

    Port of scripts/levelling_network.py (pair_observations + solve_network).

    Interface: (offsets: dict[well_id: str, c: float], info: dict) -- the
    dict return of the original placeholder is extended with a small info
    dict (n_pairs, n_components, residual_rms_ft) purely for the required
    [LEVELLING] diagnostics; the offsets dict itself is unchanged.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import lsqr

    uniq = _lv_np.unique(donor_table['well'])
    widx = {wid: i for i, wid in enumerate(uniq)}
    n_wells = len(uniq)
    if n_wells == 0 or len(donor_table['s']) == 0:
        return {}, {'n_pairs': 0, 'n_components': 0, 'residual_rms_ft': float('nan')}

    wo = _lv_np.array([widx[w_] for w_ in donor_table['well']], dtype='int64')
    xy = _lv_np.stack([donor_table['x'], donor_table['y']], axis=1)
    s = donor_table['s']
    n_pts = len(s)
    tree = cKDTree(xy)

    k = min(nn_k, n_pts)
    chunk = 20000
    src, dst, diff, dist_l = [], [], [], []
    for a in range(0, n_pts, chunk):
        b = min(a + chunk, n_pts)
        dd, ii = tree.query(xy[a:b], k=k, distance_upper_bound=radius, workers=-1)
        if dd.ndim == 1:
            dd, ii = dd[:, None], ii[:, None]
        valid = _lv_np.isfinite(dd) & (ii < n_pts)
        other = _lv_np.zeros_like(valid)
        other[valid] = wo[ii[valid]] != wo[a:b, None].repeat(k, 1)[valid]
        good = valid & other
        if not good.any():
            continue
        rr, cc = _lv_np.where(good)
        j = ii[rr, cc]
        # keep only the nearest match per (source point, other well)
        key = wo[j].astype('int64') * n_pts + (a + rr)
        order = _lv_np.lexsort((dd[rr, cc], key))
        rr, cc, j, key = rr[order], cc[order], j[order], key[order]
        first = _lv_np.ones(len(key), dtype=bool)
        first[1:] = key[1:] != key[:-1]
        rr, cc, j = rr[first], cc[first], j[first]
        src.append(wo[a + rr])
        dst.append(wo[j])
        diff.append(s[a + rr] - s[j])
        dist_l.append(dd[rr, cc])

    if not src:
        return ({str(w_): 0.0 for w_ in uniq},
                {'n_pairs': 0, 'n_components': int(n_wells), 'residual_rms_ft': float('nan')})

    obs = _lv_pd.DataFrame({
        'i': _lv_np.concatenate(src), 'j': _lv_np.concatenate(dst),
        'd': _lv_np.concatenate(diff), 'dist': _lv_np.concatenate(dist_l),
    })
    # symmetrise: fold (j,i) onto (i,j) with a sign flip so both directions average
    flip = obs['i'] > obs['j']
    lo = _lv_np.where(flip, obs['j'], obs['i'])
    hi = _lv_np.where(flip, obs['i'], obs['j'])
    obs['d'] = _lv_np.where(flip, -obs['d'], obs['d'])
    obs['i'], obs['j'] = lo, hi
    g = obs.groupby(['i', 'j']).agg(
        d_ij=('d', 'median'), n=('d', 'size'), dist=('dist', 'median'),
    ).reset_index()
    pairs = g[g['n'] >= min_matches].reset_index(drop=True)

    if pairs.empty:
        return ({str(w_): 0.0 for w_ in uniq},
                {'n_pairs': 0, 'n_components': int(n_wells), 'residual_rms_ft': float('nan')})

    i = pairs['i'].to_numpy()
    j = pairs['j'].to_numpy()
    d = pairs['d_ij'].to_numpy('float64')
    w0 = _lv_np.sqrt(pairs['n'].to_numpy('float64')) / (1.0 + pairs['dist'].to_numpy('float64') / 300.0)
    w0 = w0 / w0.mean()

    adj = coo_matrix((_lv_np.ones(len(i)), (i, j)), shape=(n_wells, n_wells))
    ncomp, comp = connected_components(adj, directed=False)

    m = len(i)
    rows = _lv_np.concatenate([_lv_np.arange(m), _lv_np.arange(m)])
    cols = _lv_np.concatenate([i, j])
    c = _lv_np.zeros(n_wells)
    weight = w0.copy()
    for _ in range(irls_iters):
        vals = _lv_np.concatenate([weight, -weight])
        A = coo_matrix((vals, (rows, cols)), shape=(m, n_wells)).tocsr()
        c = lsqr(A, d * weight, damp=1e-3, atol=1e-10, btol=1e-10, iter_lim=2000)[0]
        r = (c[i] - c[j]) - d
        scale = max(1.4826 * float(_lv_np.median(_lv_np.abs(r - _lv_np.median(r)))), 1e-3)
        u = _lv_np.abs(r) / (huber * scale)
        weight = w0 * _lv_np.where(u <= 1.0, 1.0, 1.0 / _lv_np.maximum(u, 1e-9))
    for kk in range(ncomp):
        sel = comp == kk
        c[sel] -= c[sel].mean()

    res = (c[i] - c[j]) - d
    offsets = {str(uniq[idx]): float(c[idx]) for idx in range(n_wells)}
    info = {
        'n_pairs': int(len(pairs)), 'n_components': int(ncomp),
        'residual_rms_ft': float(_lv_np.sqrt(_lv_np.mean(res ** 2))) if len(res) else float('nan'),
    }
    return offsets, info


def predict_levelled_surface(x, y, donors, offsets, exclude_well=None,
                              radius=LEVELLING_RADIUS_FT, k=LEVELLING_KNN):
    """The levelled spatial predictor: K-nearest, distance^2-weighted IDW of
    (donor_s - offsets[donor_well]) around each query point.

    `exclude_well` drops one well's own rows from the donor pool before the
    tree is built -- used both for a well's own holdback/anchor fit and for
    its own eval-zone prediction, so a well is never informed by itself.
    For donor variant "a" (train-only field) this is a defensive no-op: a
    test query well was never in `donors` to begin with.

    Memory: builds one cKDTree over the (already strided, float64) donor
    coordinates and queries in chunks of 4096 rows -- no all-pairs distance
    matrix, and peak memory is bounded by chunk_size * k.

    Returns S_hat as float64, NaN wherever no donor was found within
    `radius` of a query point (or the donor pool is empty).
    """
    from scipy.spatial import cKDTree

    dw = donors['well']
    if exclude_well is not None:
        mask = dw != exclude_well
    else:
        mask = _lv_np.ones(len(dw), dtype=bool)
    dx, dy, ds, dw = donors['x'][mask], donors['y'][mask], donors['s'][mask], dw[mask]
    n = len(x)
    out = _lv_np.full(n, _lv_np.nan, dtype='float64')
    if dx.size == 0:
        return out

    off = _lv_np.array([offsets.get(str(w_), 0.0) for w_ in dw], dtype='float64')
    adj_s = ds.astype('float64') - off  # donor surface value = (TVT+Z) - c_d

    tree = cKDTree(_lv_np.column_stack([dx, dy]))
    kk = int(min(k, dx.size))
    x = _lv_np.asarray(x, dtype='float64')
    y = _lv_np.asarray(y, dtype='float64')
    chunk = 4096
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        dist, idx = tree.query(_lv_np.column_stack([x[a:b], y[a:b]]), k=kk, workers=-1)
        if dist.ndim == 1:
            dist, idx = dist[:, None], idx[:, None]
        ok = _lv_np.isfinite(dist) & (dist <= radius)
        idx_safe = _lv_np.where(ok, idx, 0)
        sv = adj_s[idx_safe]
        wgt = _lv_np.where(ok, 1.0 / _lv_np.power(_lv_np.maximum(dist, 1.0), 2.0), 0.0)
        den = wgt.sum(axis=1)
        good = den > 0
        res = _lv_np.full(b - a, _lv_np.nan)
        res[good] = (wgt[good] * sv[good]).sum(axis=1) / den[good]
        out[a:b] = res
    return out


def fit_query_anchor(known_zone_rows, donors, offsets, well_id,
                      tail_rows=LEVELLING_ANCHOR_TAIL_ROWS,
                      min_known_rows=LEVELLING_MIN_KNOWN_ROWS,
                      radius=LEVELLING_RADIUS_FT):
    """Fit the per-query-well anchor constant c_q.

    `known_zone_rows` = dict(x, y, s) for well_id's own known prefix (s is
    its own TVT_input + Z). c_q is the mean of (s - S_hat) over the LAST
    `tail_rows` known rows that have real donor coverage (S_hat finite) --
    the "last 300 known rows" of the spec. If fewer than `min_known_rows`
    known rows have donor coverage at all, no anchor is available.

    Returns c_query: float, or NaN when no anchor is available (too little
    covered known data) -- NaN is the "no anchor" signal gate_and_blend
    (via the caller) falls back to base prediction on, kept explicit rather
    than silently defaulting to 0.0 so an unlevel-able well is visibly
    un-levelled rather than silently mis-levelled.
    """
    x, y, s = known_zone_rows['x'], known_zone_rows['y'], known_zone_rows['s']
    s_hat = predict_levelled_surface(x, y, donors, offsets, exclude_well=well_id, radius=radius)
    ok = _lv_np.isfinite(s_hat)
    if int(ok.sum()) < min_known_rows:
        return float('nan')
    idx_ok = _lv_np.where(ok)[0]
    tail_idx = idx_ok[-tail_rows:]
    resid = s[tail_idx] - s_hat[tail_idx]
    return float(_lv_np.mean(resid))


def gate_and_blend(base_pred, levelled_pred, cap_ft, blend_w=LEVELLING_BLEND_W):
    """Blend base and levelled predictions with a FIXED weight and an
    absolute clip `cap_ft` (in feet) on the spatial correction
    (spat - base) -- the validated final rule (no per-well accept/reject
    gate; "gate DISABLED" per spec). `cap_ft` has no static default: the
    caller computes it once per run as LEVELLING_CLIP_KAPPA * MAD_ft, where
    MAD_ft is pooled over this run's own judged eval rows -- a cap that
    rescales with the deployment population is what makes this rule robust
    to distribution shift (see the module-level revision-history note).
    NaNs in levelled_pred (no donor coverage this row, or no per-well
    anchor at all) fall back to base_pred via a zero delta, so a
    partially-covered eval zone / an unlevel-able well degrades gracefully
    to "do nothing" instead of injecting NaNs into the submission.

    Returns final_pred: float64 array.
    """
    base_pred = _lv_np.asarray(base_pred, dtype='float64')
    diff = _lv_np.asarray(levelled_pred, dtype='float64') - base_pred
    finite = _lv_np.isfinite(diff)
    delta = _lv_np.where(finite, _lv_np.clip(diff, -cap_ft, cap_ft), 0.0)
    return base_pred + blend_w * delta


# ---- drive the pipeline end to end, guarded by the flag and the time guard -
_lv_t0 = globals().get('_LEVELLING_T0', None)
_lv_elapsed_hours = ((_lv_time.time() - _lv_t0) / 3600.0) if _lv_t0 is not None else 0.0

if not LEVELLING_ENABLED:
    print('[LEVELLING] DISABLED donors=0 wells_gated=0 rows_changed=0 max_abs_delta=0.0000')
elif _lv_elapsed_hours > LEVELLING_GUARD_HOURS:
    print(f'[LEVELLING] SKIPPED_TIME_GUARD elapsed_hours={_lv_elapsed_hours:.3f} '
          f'guard_hours={LEVELLING_GUARD_HOURS:.3f}')
else:
    _lv_work = _LvPath(globals().get('OUTPUT_DIR', _LvPath('/kaggle/working')))
    _lv_root = _lv_data_root()
    _lv_sub_path = _lv_work / 'submission.csv'
    if not _lv_sub_path.exists():
        raise RuntimeError('levelling: submission.csv is missing')

    _lv_sub = _lv_pd.read_csv(_lv_sub_path, dtype={'id': 'string'})
    if list(_lv_sub.columns) != ['id', 'tvt']:
        raise RuntimeError(f'levelling: unexpected columns {list(_lv_sub.columns)}')
    _lv_sub.to_csv(_lv_work / 'submission_before_levelling.csv', index=False)

    _lv_before = _lv_sub['tvt'].to_numpy(dtype='float64').copy()
    _lv_well = _lv_sub['id'].astype(str).str.rsplit('_', n=1).str[0]
    _lv_row = _lv_pd.to_numeric(_lv_sub['id'].astype(str).str.rsplit('_', n=1).str[1],
                                 errors='coerce')
    if _lv_row.isna().any():
        raise RuntimeError('levelling: could not parse row index from ids')
    _lv_row = _lv_row.to_numpy(dtype=int)

    # network donors: every mounted well's LEGALLY VISIBLE points (train in
    # full, test by known prefix), stride LEVELLING_PAIR_STRIDE.
    _lv_donors_net = build_donor_table(_lv_root, stride=LEVELLING_PAIR_STRIDE, include_test=True)
    # spatial-field donors: TRAIN WELLS ONLY (donor variant "a"), stride
    # LEVELLING_DONOR_STRIDE.
    _lv_donors_field = build_donor_table(_lv_root, stride=LEVELLING_DONOR_STRIDE, include_test=False)
    _lv_n_donor_wells = int(_lv_np.unique(_lv_donors_field['well']).size)
    _lv_offsets, _lv_net_info = solve_network_offsets(_lv_donors_net, radius=LEVELLING_PAIR_RADIUS_FT,
                                                       min_matches=LEVELLING_PAIR_MIN_MATCHES,
                                                       nn_k=LEVELLING_PAIR_NN_K,
                                                       huber=LEVELLING_PAIR_HUBER,
                                                       irls_iters=LEVELLING_PAIR_IRLS_ITERS)

    # pass 1: per query well, fit the anchor and predict the levelled
    # surface over its own eval zone; collect everything before blending so
    # the MAD_ft diagnostic can be computed pooled across the whole run.
    _lv_out = _lv_before.copy()
    _lv_done, _lv_skipped = 0, 0
    _lv_wells_with_coverage, _lv_wells_total = 0, 0
    _lv_report_rows = []
    _lv_all_diff = []
    _lv_well_slices = []  # (positions, base_slice, spat_slice) for pass 2

    for _wid, _idx in _lv_sub.groupby(_lv_well, sort=False).groups.items():
        _pos = _lv_sub.index.get_indexer(_idx)
        _hw_path = _lv_root / 'test' / f'{_wid}__horizontal_well.csv'
        if not _hw_path.exists():
            _lv_skipped += 1
            continue
        _hw = _lv_pd.read_csv(_hw_path, usecols=['MD', 'X', 'Y', 'Z', 'TVT_input'])
        _rows = _lv_row[_pos]
        if _rows.max() >= len(_hw) or _rows.min() < 0:
            _lv_skipped += 1
            continue

        _lv_wells_total += 1
        _known = _hw[_hw['TVT_input'].notna()]
        if len(_known) == 0:
            _lv_skipped += 1
            continue
        _known_rows = {
            'x': _known['X'].to_numpy(dtype='float64'),
            'y': _known['Y'].to_numpy(dtype='float64'),
            's': (_known['TVT_input'] + _known['Z']).to_numpy(dtype='float64'),
        }
        _c_query = fit_query_anchor(_known_rows, _lv_donors_field, _lv_offsets, _wid,
                                     tail_rows=LEVELLING_ANCHOR_TAIL_ROWS,
                                     min_known_rows=LEVELLING_MIN_KNOWN_ROWS,
                                     radius=LEVELLING_RADIUS_FT)

        _eval_x = _hw['X'].to_numpy(dtype='float64')[_rows]
        _eval_y = _hw['Y'].to_numpy(dtype='float64')[_rows]
        _eval_z = _hw['Z'].to_numpy(dtype='float64')[_rows]
        if not (_lv_np.isfinite(_eval_x).all() and _lv_np.isfinite(_eval_y).all()
                and _lv_np.isfinite(_eval_z).all()):
            _lv_skipped += 1
            continue

        if _lv_np.isfinite(_c_query):
            _s_hat = predict_levelled_surface(_eval_x, _eval_y, _lv_donors_field, _lv_offsets,
                                               exclude_well=_wid, radius=LEVELLING_RADIUS_FT,
                                               k=LEVELLING_KNN)
            _spat = (_s_hat + _c_query) - _eval_z
        else:
            _spat = _lv_np.full(len(_eval_x), _lv_np.nan)

        _diff = _spat - _lv_out[_pos]
        _finite = _lv_np.isfinite(_diff)
        if _finite.any():
            _lv_wells_with_coverage += 1
        # MAD_ft pools |spat-base| over ALL judged eval rows this well
        # contributes, with fallback (no-coverage) rows counted as an
        # explicit zero contribution rather than excluded from the pool.
        _lv_all_diff.append(_lv_np.where(_finite, _diff, 0.0))
        _lv_well_slices.append((_wid, _pos, _spat))
        _lv_done += 1
        _lv_report_rows.append({
            'well': _wid, 'n_known': int(_known_rows['x'].size),
            'c_query': _c_query, 'n_eval': int(len(_eval_x)),
            'coverage': float(_finite.mean()) if len(_finite) else 0.0,
        })

    _lv_mad_ft = (float(_lv_np.median(_lv_np.abs(_lv_np.concatenate(_lv_all_diff))))
                  if _lv_all_diff else float('nan'))
    _lv_cap_ft = LEVELLING_CLIP_KAPPA * _lv_mad_ft if _lv_np.isfinite(_lv_mad_ft) else 0.0

    # pass 2: apply the fixed-weight, live-cap clipped blend everywhere, and
    # report a per-well summary line (well, rows_changed, mean_abs_correction).
    for _wid, _pos, _spat in _lv_well_slices:
        _before_well = _lv_out[_pos].copy()
        _lv_out[_pos] = gate_and_blend(_before_well, _spat, _lv_cap_ft, blend_w=LEVELLING_BLEND_W)
        _well_delta = _lv_np.abs(_lv_out[_pos] - _before_well)
        _well_rows_changed = int((_well_delta > 1e-9).sum())
        _well_mean_abs_correction = float(_well_delta.mean()) if len(_well_delta) else 0.0
        print(f'[LEVELLING] well={_wid} rows_changed={_well_rows_changed} '
              f'mean_abs_correction={_well_mean_abs_correction:.4f}')

    if not _lv_np.isfinite(_lv_out).all():
        raise RuntimeError('levelling: produced non-finite predictions')

    _lv_pd.DataFrame(_lv_report_rows).to_csv(_lv_work / 'levelling_report.csv', index=False)
    _lv_sub['tvt'] = _lv_out
    _lv_sub.to_csv(_lv_sub_path, index=False)

    _lv_delta = _lv_np.abs(_lv_out - _lv_before)
    _lv_rows_changed = int((_lv_delta > 1e-9).sum())
    _lv_frac_rows_changed = _lv_rows_changed / len(_lv_out) if len(_lv_out) else 0.0
    print(f'[LEVELLING] n_donor_wells={_lv_n_donor_wells} n_pairs={_lv_net_info["n_pairs"]} '
          f'n_components={_lv_net_info["n_components"]} MAD_ft={_lv_mad_ft:.4f} '
          f'cap_ft={_lv_cap_ft:.4f} applied_w={LEVELLING_BLEND_W} cap_mult={LEVELLING_CLIP_KAPPA} '
          f'rows_changed={_lv_rows_changed} '
          f'frac_rows_changed={_lv_frac_rows_changed:.4f} max_abs_delta={float(_lv_delta.max()):.4f} '
          f'wells_with_coverage={_lv_wells_with_coverage} wells_total={_lv_wells_total} '
          f'wells_done={_lv_done} wells_skipped={_lv_skipped} '
          f'elapsed_hours={_lv_elapsed_hours:.3f}')
'''


def apply(nb: dict, old: str, new: str) -> None:
    """Apply an exact-string edit to the single cell that contains it."""
    hits = [i for i, c in enumerate(nb["cells"]) if old in "".join(c.get("source", []))]
    assert len(hits) == 1, f"expected 1 cell, got {hits} for {old[:60]!r}"
    i = hits[0]
    src = "".join(nb["cells"][i].get("source", []))
    assert src.count(old) == 1, f"cell {i}: {src.count(old)} occurrences"
    src = src.replace(old, new)
    ast.parse(src)
    nb["cells"][i]["source"] = src.splitlines(keepends=True)


def _code_cell(source: str) -> dict:
    ast.parse(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> int:
    nb = json.loads(SRC.read_text())
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())

    apply(nb, *PF_OFF)

    nb["cells"].insert(0, _code_cell(BOOT_CELL))
    nb["cells"].append(_code_cell(CELL))

    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            ast.parse("".join(c.get("source", [])))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb))

    m = dict(meta)
    m["id"] = f"taichiiiii/{SLUG}"
    m["title"] = SLUG
    m["code_file"] = f"{SLUG}.ipynb"
    m.pop("id_no", None)
    (OUT / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
    print(f"built {OUT}  cells={len(nb['cells'])}  "
          f"(pf_mean off, boot cell prepended, levelling cell appended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
