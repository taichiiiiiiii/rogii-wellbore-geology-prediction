"""Make the leaky harness serve held-out wells from stored out-of-fold predictions.

The pretrained koolbox Trainers in the ridge-artifact dataset were fitted under
GroupKFold(groups=well), so `trainer.oof_preds` holds, for every training row, a
prediction produced by a model that never saw that row's well. That makes the OOF
of a held-out well an honest stand-in for a test prediction — no retraining, no leak.

Two edits:
  cell 19  capture the *unfiltered* id index before the held-out wells are dropped
  cell 21  add the aligner + install it in the lightgbm/catboost loops
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

NB = Path(__file__).parent / "leaky" / "rogii-harness-leaky.ipynb"

CELL19_OLD = """if bool(globals().get('HARNESS_MODE', False)):  # HARNESS PATCH
    _harness_held = set(str(w) for w in globals().get('HARNESS_HELDOUT_WELLS', ()))"""

CELL19_NEW = """if bool(globals().get('HARNESS_MODE', False)):  # HARNESS PATCH
    # The pretrained trainers' oof_preds are indexed by the *unfiltered* train.csv,
    # so keep that id order around to realign them after the held-out wells go.
    _harness_all_ids = train_df['id'].astype(str).to_numpy()  # HARNESS OOF PATCH
    _harness_held = set(str(w) for w in globals().get('HARNESS_HELDOUT_WELLS', ()))"""

CELL21_EXTRA = '''

def _harness_take_oof(trainer, name):
    """Return (oof_for_train, preds_for_test) with no in-sample leakage.

    `trainer.predict(X_test)` would be in-sample here: the pretrained model saw the
    held-out wells. Its stored out-of-fold prediction for those same rows did not,
    so serve the pseudo-test rows from the OOF vector instead and only fall back to
    predict() for ids the artifact does not cover.
    """
    oof = np.asarray(trainer.oof_preds, dtype=float).ravel()
    ids = globals().get('_harness_all_ids', None)
    if not bool(globals().get('HARNESS_MODE', False)) or ids is None or len(oof) != len(ids):
        return trainer.oof_preds, trainer.predict(X_test)

    s = pd.Series(oof, index=pd.Index(ids, dtype=str))
    s = s[~s.index.duplicated(keep='first')]

    tr = s.reindex(train_df['id'].astype(str)).to_numpy()
    if np.isnan(tr).any():
        raise AssertionError(f'HARNESS OOF: {name} missing {int(np.isnan(tr).sum())} train ids')

    te = s.reindex(test_df['id'].astype(str)).to_numpy()
    n_missing = int(np.isnan(te).sum())
    if n_missing:
        fb = np.asarray(trainer.predict(X_test), dtype=float).ravel()
        te = np.where(np.isnan(te), fb, te)
    print(f'HARNESS OOF: {name} train={len(tr)} test={len(te)} '
          f'from-OOF={len(te) - n_missing} fallback-predict={n_missing}', flush=True)
    return tr, te
'''

CELL22_OLD = """    oof_preds[f"lightgbm-{i+1}"] = trainer.oof_preds
    test_preds[f"lightgbm-{i+1}"] = trainer.predict(X_test)"""
CELL22_NEW = """    oof_preds[f"lightgbm-{i+1}"], test_preds[f"lightgbm-{i+1}"] = _harness_take_oof(
        trainer, f"lightgbm-{i+1}")"""

CELL23_OLD = """    oof_preds[f"catboost-{i+1}"] = trainer.oof_preds
    test_preds[f"catboost-{i+1}"] = trainer.predict(X_test)"""
CELL23_NEW = """    oof_preds[f"catboost-{i+1}"], test_preds[f"catboost-{i+1}"] = _harness_take_oof(
        trainer, f"catboost-{i+1}")"""


def patch_cell(cells, idx, old, new):
    src = "".join(cells[idx].get("source", []))
    n = src.count(old)
    assert n == 1, f"cell {idx}: expected 1 hit, got {n}"
    src = src.replace(old, new)
    ast.parse(src)
    cells[idx]["source"] = src.splitlines(keepends=True)
    return src


def main() -> int:
    nb = json.loads(NB.read_text())
    cells = nb["cells"]

    patch_cell(cells, 19, CELL19_OLD, CELL19_NEW)

    src21 = "".join(cells[21].get("source", []))
    assert "_harness_take_oof" not in src21, "cell 21 already patched"
    src21 = src21.rstrip("\n") + "\n" + CELL21_EXTRA
    ast.parse(src21)
    cells[21]["source"] = src21.splitlines(keepends=True)

    patch_cell(cells, 22, CELL22_OLD, CELL22_NEW)
    patch_cell(cells, 23, CELL23_OLD, CELL23_NEW)

    # every code cell must still parse
    bad = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        try:
            ast.parse("".join(c.get("source", [])))
        except SyntaxError as e:
            bad.append((i, e))
    assert not bad, f"syntax errors: {bad}"

    NB.write_text(json.dumps(nb))
    print(f"patched {NB} — cells 19/21/22/23, {len(cells)} cells, 0 syntax errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
