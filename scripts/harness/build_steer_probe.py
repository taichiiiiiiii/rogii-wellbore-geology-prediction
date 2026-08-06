"""Apply the global steering correction as a final post-process.

R94 measured, on held-out wells, that the pipeline's remaining error correlates with
the trajectory residual at r = +0.24 with a sign that holds across wells, and that one
global coefficient is worth -0.27 to -0.29. Unlike R93's per-well coupling, a single
number is identifiable, so it can be estimated once offline and shipped.

For each well the anchor is the last known row; the incoming Z slope comes from the
tail of the known zone; and

    steering residual = Z - [Z_anchor + b_z * (MD - MD_anchor)]
    corrected tvt     = tvt - STEER_BETA * steering residual

Only TVT_input, Z and MD are read, all of which exist for test wells in the known
zone, so nothing here needs a label.

Wells the contact override already answered are skipped, exactly as in the surface
post-process: their known-prefix RMSE is ~0.01 ft, so there is no error to correct and
any nudge is pure damage.

STEER_BETA defaults to 0.0, a provable no-op, and the kernel prints the wells touched
and the realised shift because neither output deltas nor timings can establish firing
in this pipeline.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

HERE = Path("/tmp/claude-0/-root/358631d3-278c-4097-9c59-7f0aee48d95f/scratchpad")
SRC = HERE / "pfmean" / "rogii-gs145-pfmean035" / "rogii-gs145-pfmean035.ipynb"
OUT = HERE / "steer"

PF_OFF = ("PF_SCALE_PARTNER_WEIGHT = 0.35", "PF_SCALE_PARTNER_WEIGHT = 0.0")

CELL = r'''
# ============================================================================
# Global steering correction. The pipeline's residual error co-varies with how far
# the trajectory has drifted from its own incoming slope, in a direction that is
# consistent across wells, so one coefficient corrects it. Runs last so it never
# disturbs the anchor the gold visible-prefix stage backtests against.
# ============================================================================
STEER_BETA = float(globals().get('STEER_BETA', 0.0))
STEER_TAIL = int(globals().get('STEER_TAIL', 200))

if STEER_BETA == 0.0:
    print('steering correction disabled (STEER_BETA=0).')
else:
    import numpy as _st_np
    import pandas as _st_pd
    from pathlib import Path as _StPath

    _st_work = _StPath(globals().get('OUTPUT_DIR', _StPath('/kaggle/working')))
    _st_root = _StPath(str(globals().get('COMPETITION_DATA_ROOT',
                                         '/kaggle/input/competitions/rogii-wellbore-geology-prediction')))
    _st_path = _st_work / 'submission.csv'
    if not _st_path.exists():
        raise RuntimeError('steering: submission.csv is missing')

    _st_sub = _st_pd.read_csv(_st_path, dtype={'id': 'string'})
    if list(_st_sub.columns) != ['id', 'tvt']:
        raise RuntimeError(f'steering: unexpected columns {list(_st_sub.columns)}')
    _st_sub.to_csv(_st_work / 'submission_before_steering.csv', index=False)

    _st_locked = set()
    _st_rep = _st_work / 'gold_contact_override_report.csv'
    if _st_rep.exists():
        _r = _st_pd.read_csv(_st_rep)
        if {'well', 'status'}.issubset(_r.columns):
            _hit = _r['status'].astype(str).str.contains('override', case=False, na=False)
            if 'rows_overridden' in _r.columns:
                _hit &= _st_pd.to_numeric(_r['rows_overridden'], errors='coerce').fillna(0) > 0
            _st_locked = set(_r.loc[_hit, 'well'].astype(str))

    _st_before = _st_sub['tvt'].to_numpy(dtype=float).copy()
    _st_out = _st_before.copy()
    _st_well = _st_sub['id'].astype(str).str.rsplit('_', n=1).str[0]
    _st_row = _st_pd.to_numeric(_st_sub['id'].astype(str).str.rsplit('_', n=1).str[1],
                                errors='coerce')
    if _st_row.isna().any():
        raise RuntimeError('steering: could not parse row index from ids')
    _st_row = _st_row.to_numpy(dtype=int)

    _st_done = _st_locked_n = _st_skipped = 0
    for _wid, _idx in _st_sub.groupby(_st_well, sort=False).groups.items():
        if str(_wid) in _st_locked:
            _st_locked_n += 1
            continue
        _pos = _st_sub.index.get_indexer(_idx)
        _hw_path = _st_root / 'test' / f'{_wid}__horizontal_well.csv'
        if not _hw_path.exists():
            _st_skipped += 1
            continue
        _hw = _st_pd.read_csv(_hw_path)
        _kn = _hw[_hw['TVT_input'].notna()]
        if len(_kn) < 20:
            _st_skipped += 1
            continue
        _tail = _kn.tail(STEER_TAIL)
        _md_t = _st_pd.to_numeric(_tail['MD'], errors='coerce').to_numpy(dtype=float)
        _z_t = _st_pd.to_numeric(_tail['Z'], errors='coerce').to_numpy(dtype=float)
        if not (_st_np.isfinite(_md_t).all() and _st_np.isfinite(_z_t).all()):
            _st_skipped += 1
            continue
        _xc = _md_t - _md_t.mean()
        _den = float(_st_np.sum(_xc * _xc))
        _bz = float(_st_np.sum(_xc * (_z_t - _z_t.mean())) / _den) if _den > 0 else 0.0
        _md_a, _z_a = float(_md_t[-1]), float(_z_t[-1])

        _rows = _st_row[_pos]
        if _rows.max() >= len(_hw) or _rows.min() < 0:
            _st_skipped += 1
            continue
        _md = _st_pd.to_numeric(_hw['MD'], errors='coerce').to_numpy(dtype=float)[_rows]
        _z = _st_pd.to_numeric(_hw['Z'], errors='coerce').to_numpy(dtype=float)[_rows]
        if not (_st_np.isfinite(_md).all() and _st_np.isfinite(_z).all()):
            _st_skipped += 1
            continue

        _steer = _z - (_z_a + _bz * (_md - _md_a))
        _cand = _st_out[_pos] - STEER_BETA * _steer
        if not _st_np.isfinite(_cand).all():
            _st_skipped += 1
            continue
        _st_out[_pos] = _cand
        _st_done += 1

    if not _st_np.isfinite(_st_out).all():
        raise RuntimeError('steering: produced non-finite predictions')
    if _st_done == 0:
        raise RuntimeError('steering: no well was corrected -- the probe is a no-op')
    _st_sub['tvt'] = _st_out
    _st_sub.to_csv(_st_path, index=False)
    _d = _st_np.abs(_st_out - _st_before)
    print(f'steering: beta={STEER_BETA} tail={STEER_TAIL} wells_done={_st_done} '
          f'wells_locked={_st_locked_n} wells_skipped={_st_skipped} rows={len(_st_sub)} '
          f'mean|delta|={float(_d.mean()):.4f} max|delta|={float(_d.max()):.4f}')
'''


def edit(nb: dict, old: str, new: str) -> None:
    hits = [i for i, c in enumerate(nb["cells"]) if old in "".join(c.get("source", []))]
    assert len(hits) == 1, f"expected 1 cell, got {hits}"
    i = hits[0]
    src = "".join(nb["cells"][i].get("source", [])).replace(old, new)
    ast.parse(src)
    nb["cells"][i]["source"] = src.splitlines(keepends=True)


def main(argv: list[str]) -> int:
    beta = float(argv[1]) if len(argv) > 1 else 0.0
    slug = argv[2] if len(argv) > 2 else f"rogii-gs145-steer{str(beta).replace('.', 'p')}"

    nb = json.loads(SRC.read_text())
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
    edit(nb, *PF_OFF)
    edit(nb, "SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'",
         f"STEER_BETA = {beta}\nSTEER_TAIL = 200\nSELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'")

    ast.parse(CELL)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": CELL.splitlines(keepends=True)})

    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            ast.parse("".join(c.get("source", [])))

    d = OUT / slug
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / f"{slug}.ipynb").write_text(json.dumps(nb))
    m = dict(meta)
    m["id"] = f"taichiiiii/{slug}"
    m["title"] = slug
    m["code_file"] = f"{slug}.ipynb"
    m.pop("id_no", None)
    (d / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
    print(f"built {d}  STEER_BETA={beta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
