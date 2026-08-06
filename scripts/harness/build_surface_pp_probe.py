"""Append the surface reparametrisation as a final post-process to the main line.

TVT = surface - Z, and Z is measured exactly in the eval zone, so the implied
surface is the only thing the pipeline actually has to get right. Refitting that
implied surface on a low-order polynomial and subtracting the known Z projects
out whatever shape error the tracker left in it.

Measured paired on held-out wells in the realistic error regime (~7.5 ft) this is
worth -0.13 to -0.16, with deg-4 / anchored / w=0.75 ranking first on two
independent prediction sets. On the three visible wells it hurts, but those score
1.10 because they are train copies the contact override answers outright, so the
surface there is already essentially exact and any smoothing can only damage it.
The transfer is therefore unproven, which is why this ships as a probe with a
pre-declared judgment band rather than a change to the main line.

Placement is last, after the model-package correction: the award write-up found
corrections placed before the gold calibration stage confuse its per-well
backtest, and that moving them after it recovered 0.045.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

HERE = Path("/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad")
SRC = HERE / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
SLUG = "rogii-gs145-surfacepp"
OUT = HERE / "surfacepp" / SLUG

# Turn the pf_mean blend back off so this probe isolates one change.
PF_OFF = ("PF_SCALE_PARTNER_WEIGHT = 0.35", "PF_SCALE_PARTNER_WEIGHT = 0.0")

CELL = r'''
# ============================================================================
# Surface post-process. TVT = surface - Z with Z known in the eval zone, so refit
# the implied surface on a low-order polynomial and subtract Z back off. Runs
# last, after the model-package correction, so it never perturbs the anchor the
# gold visible-prefix stage backtests against.
# ============================================================================
SURFACE_PP_DEGREE = int(globals().get('SURFACE_PP_DEGREE', 4))
SURFACE_PP_WEIGHT = float(globals().get('SURFACE_PP_WEIGHT', 0.75))
SURFACE_PP_ANCHORED = bool(globals().get('SURFACE_PP_ANCHORED', True))

if SURFACE_PP_WEIGHT <= 0.0:
    print('Surface post-process disabled.')
else:
    import numpy as _sp_np
    import pandas as _sp_pd
    from pathlib import Path as _SpPath

    _sp_work = _SpPath(globals().get('OUTPUT_DIR', _SpPath('/kaggle/working')))
    _sp_root = _SpPath(str(globals().get('COMPETITION_DATA_ROOT',
                                         '/kaggle/input/competitions/rogii-wellbore-geology-prediction')))
    _sp_path = _sp_work / 'submission.csv'
    if not _sp_path.exists():
        raise RuntimeError('surface pp: submission.csv is missing')

    _sp_sub = _sp_pd.read_csv(_sp_path, dtype={'id': 'string'})
    if list(_sp_sub.columns) != ['id', 'tvt']:
        raise RuntimeError(f'surface pp: unexpected columns {list(_sp_sub.columns)}')
    _sp_sub.to_csv(_sp_work / 'submission_before_surface_pp.csv', index=False)

    _sp_before = _sp_sub['tvt'].to_numpy(dtype=float).copy()
    _sp_well = _sp_sub['id'].astype(str).str.rsplit('_', n=1).str[0]
    _sp_row = _sp_pd.to_numeric(_sp_sub['id'].astype(str).str.rsplit('_', n=1).str[1],
                                errors='coerce')
    if _sp_row.isna().any():
        raise RuntimeError('surface pp: could not parse row index from ids')
    _sp_row = _sp_row.to_numpy(dtype=int)

    # Skip wells the contact override already answered. Their known-prefix RMSE is
    # ~0.01 ft, i.e. the surface there is not an estimate at all, so refitting it
    # can only do damage. The override reports 0 firings on the hidden wells, so
    # this gate is a no-op exactly where the post-process is meant to act.
    _sp_locked = set()
    _sp_rep = _sp_work / 'gold_contact_override_report.csv'
    if _sp_rep.exists():
        _r = _sp_pd.read_csv(_sp_rep)
        if {'well', 'status'}.issubset(_r.columns):
            _hit = _r['status'].astype(str).str.contains('override', case=False, na=False)
            if 'rows_overridden' in _r.columns:
                _hit &= _sp_pd.to_numeric(_r['rows_overridden'], errors='coerce').fillna(0) > 0
            _sp_locked = set(_r.loc[_hit, 'well'].astype(str))
    print(f'surface pp: {len(_sp_locked)} well(s) locked by the contact override')

    _sp_out = _sp_before.copy()
    _sp_done, _sp_skipped, _sp_locked_n = 0, 0, 0
    for _wid, _idx in _sp_sub.groupby(_sp_well, sort=False).groups.items():
        if str(_wid) in _sp_locked:
            _sp_locked_n += 1
            continue
        _pos = _sp_sub.index.get_indexer(_idx)
        _hw_path = _sp_root / 'test' / f'{_wid}__horizontal_well.csv'
        if not _hw_path.exists():
            _sp_skipped += 1
            continue
        _hw = _sp_pd.read_csv(_hw_path)
        _rows = _sp_row[_pos]
        if _rows.max() >= len(_hw) or _rows.min() < 0:
            _sp_skipped += 1
            continue
        _md = _sp_pd.to_numeric(_hw['MD'], errors='coerce').to_numpy(dtype=float)[_rows]
        _z = _sp_pd.to_numeric(_hw['Z'], errors='coerce').to_numpy(dtype=float)[_rows]
        if not (_sp_np.isfinite(_md).all() and _sp_np.isfinite(_z).all()):
            _sp_skipped += 1
            continue
        # ids are emitted in row order per well, but do not rely on it.
        _order = _sp_np.argsort(_md, kind='stable')
        _inv = _sp_np.empty_like(_order)
        _inv[_order] = _sp_np.arange(len(_order))

        _surf = _sp_out[_pos] + _z
        _x = _md - _md.mean()
        _scale = max(float(_x.std()), 1e-9)
        _x = _x / _scale
        _deg = min(SURFACE_PP_DEGREE, max(len(_md) - 2, 1))
        _fit = _sp_np.polyval(_sp_np.polyfit(_x, _surf, _deg), _x)
        if SURFACE_PP_ANCHORED:
            _first = _order[0]
            _fit = _fit + (_surf[_first] - _fit[_first])
        _new_surf = (1.0 - SURFACE_PP_WEIGHT) * _surf + SURFACE_PP_WEIGHT * _fit
        _cand = _new_surf - _z
        if not _sp_np.isfinite(_cand).all():
            _sp_skipped += 1
            continue
        _sp_out[_pos] = _cand
        _sp_done += 1

    if not _sp_np.isfinite(_sp_out).all():
        raise RuntimeError('surface pp: produced non-finite predictions')
    _sp_sub['tvt'] = _sp_out
    _sp_sub.to_csv(_sp_path, index=False)
    _sp_delta = _sp_np.abs(_sp_out - _sp_before)
    print(f'surface pp: deg={SURFACE_PP_DEGREE} w={SURFACE_PP_WEIGHT} '
          f'anchored={SURFACE_PP_ANCHORED} wells_done={_sp_done} wells_locked={_sp_locked_n} '
          f'wells_skipped={_sp_skipped} rows={len(_sp_sub)} '
          f'mean|delta|={float(_sp_delta.mean()):.4f} max|delta|={float(_sp_delta.max()):.4f}')
'''


def main() -> int:
    nb = json.loads(SRC.read_text())
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())

    hits = [i for i, c in enumerate(nb["cells"]) if PF_OFF[0] in "".join(c.get("source", []))]
    assert len(hits) == 1, f"pf knob: expected 1 cell, got {hits}"
    src = "".join(nb["cells"][hits[0]].get("source", [])).replace(*PF_OFF)
    ast.parse(src)
    nb["cells"][hits[0]]["source"] = src.splitlines(keepends=True)

    ast.parse(CELL)
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CELL.splitlines(keepends=True),
    })

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
    print(f"built {OUT}  cells={len(nb['cells'])}  (pf_mean off, surface pp appended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
