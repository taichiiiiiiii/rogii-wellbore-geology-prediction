"""Full-well seq2seq dataset builder -- analysis/seq2seq_design.md.

Full-well version of ``build_registration_dataset.py`` (R34): instead of
querying only the eval zone, every well's ENTIRE row sequence (known prefix +
eval tail) becomes the query, with two new leak-safe channels marking the
known-zone prefix drift trajectory:

  q_gr          : full-well GR, per-well affine-calibrated to the typewell
                  frame (fit on the KNOWN zone only -- leak-safe, ported
                  from R34).
  q_md, q_z     : full-well MD/Z relative to the well's first row.
  q_prefix_rel  : known rows = TVT_input - anchor; EVAL ROWS = 0.0 (ch6).
  q_is_known    : known rows = 1.0, eval rows = 0.0 (ch7).
  q_target      : TVT - anchor, full well (train-only label; ground truth is
                  read in exactly ONE place below to build this field).
  q_eval_mask   : bool, the scoring/loss mask (True on eval rows).
  q_well        : per-row well index (flat storage with per-well offsets,
                  like ``outputs/stack_v26_oof.npz``).

Leak guards (design.md "leak 厳禁" + task list a-e):
  (a)/(b) asserted inline in ``build_well_query`` -- eval rows never carry
          prefix/is_known signal.
  (c)     the horizontal-well ground-truth TVT column is read in exactly one
          place below (the ``q_target`` line); every other feature comes from
          the known-zone-only input column or plain geometry (present in
          test).
  (d)     the six formation depth columns and the typewell lithology column
          are structurally excluded via the ``usecols`` allow-lists in
          ``_read_horizontal``/``_read_typewell`` -- they are never even
          loaded into memory, let alone read.
  (e)     no cross-well/global statistic feeds any feature -- everything is
          computed per-well (unlike R34, which computed an unused global GR
          mean/std pass; that pass is deliberately dropped here).

Run:  uv run python scripts/build_seq2seq_dataset.py [--smoke N] [--out PATH]
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train"

_MIN_ROWS = 20  # skip threshold, R34-identical (eval<20 or known<20 rows -> skip)

_HORIZONTAL_USECOLS = ["MD", "Z", "GR", "TVT_input", "TVT"]
_TYPEWELL_USECOLS = ["TVT", "GR"]


@dataclass(frozen=True)
class WellQuery:
    """Full-well leak-safe query arrays for one well."""

    q_gr: np.ndarray
    q_md: np.ndarray
    q_z: np.ndarray
    q_prefix_rel: np.ndarray
    q_is_known: np.ndarray
    q_target: np.ndarray
    q_eval_mask: np.ndarray
    tw_gr: np.ndarray
    tw_tvt: np.ndarray
    anchor: float


def _read_horizontal(path: Path) -> pd.DataFrame:
    """Read a horizontal-well CSV restricted to leak-safe columns (guard d)."""
    return pd.read_csv(path, usecols=_HORIZONTAL_USECOLS)


def _read_typewell(path: Path) -> pd.DataFrame:
    """Read a typewell CSV restricted to leak-safe columns (guard d, no Geology)."""
    return pd.read_csv(path, usecols=_TYPEWELL_USECOLS)


def _fit_gr_affine(
    known_gr: np.ndarray,
    known_tvt: np.ndarray,
    tw_tvt_raw: np.ndarray,
    tw_gr_raw: np.ndarray,
) -> tuple[float, float]:
    """Per-well affine GR calibration fit on the KNOWN zone only (leak-safe, ported from R34)."""
    tw_gr_at_known = np.interp(known_tvt, tw_tvt_raw, tw_gr_raw)
    valid = np.isfinite(known_gr) & np.isfinite(tw_gr_at_known)
    if valid.sum() <= 10:
        return 1.0, 0.0
    a, b = np.polyfit(known_gr[valid], tw_gr_at_known[valid], 1)
    return float(a), float(b)


def _prep_typewell(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    twg = tw.dropna(subset=["GR"]).sort_values("TVT")
    return twg["TVT"].to_numpy(), twg["GR"].to_numpy()


def _gr_channel(
    gr_all: np.ndarray, a_gr: float, b_gr: float, tw_mean: float, tw_std: float
) -> np.ndarray:
    calibrated = ((a_gr * gr_all + b_gr) - tw_mean) / tw_std
    filled = pd.Series(calibrated).interpolate(limit_direction="both").fillna(0.0)
    return filled.to_numpy().astype(np.float32)


def build_well_query(h: pd.DataFrame, tw: pd.DataFrame) -> WellQuery | None:
    """Build the full-well leak-safe query arrays for one well.

    Returns ``None`` (skip) when the eval or known zone has fewer than
    ``_MIN_ROWS`` rows -- matches R34's skip contract so the used/skipped
    well-set stays comparable across builders.
    """
    known = h["TVT_input"].notna().to_numpy()
    eval_mask = ~known
    if eval_mask.sum() < _MIN_ROWS or known.sum() < _MIN_ROWS:
        return None

    tvt_input_full = h["TVT_input"].to_numpy()
    known_tvt = tvt_input_full[known]
    anchor = float(known_tvt[-1])

    tw_tvt_raw, tw_gr_raw = _prep_typewell(tw)
    known_gr = h["GR"].to_numpy()[known]
    a_gr, b_gr = _fit_gr_affine(known_gr, known_tvt, tw_tvt_raw, tw_gr_raw)
    tw_mean = float(np.nanmean(tw_gr_raw))
    tw_std = float(np.nanstd(tw_gr_raw)) or 1.0

    q_gr = _gr_channel(h["GR"].to_numpy(), a_gr, b_gr, tw_mean, tw_std)
    md, z = h["MD"].to_numpy(), h["Z"].to_numpy()
    q_md = (md - md[0]).astype(np.float32)
    q_z = (z - z[0]).astype(np.float32)
    q_prefix_rel = np.where(known, tvt_input_full - anchor, 0.0).astype(np.float32)
    q_is_known = known.astype(np.float32)
    q_target = (h["TVT"].to_numpy() - anchor).astype(np.float32)  # guard (c): only TVT read

    assert np.all(q_prefix_rel[eval_mask] == 0.0)  # guard (a)
    assert np.all(q_is_known[eval_mask] == 0.0)  # guard (b)

    return WellQuery(
        q_gr=q_gr,
        q_md=q_md,
        q_z=q_z,
        q_prefix_rel=q_prefix_rel,
        q_is_known=q_is_known,
        q_target=q_target,
        q_eval_mask=eval_mask.astype(bool),
        tw_gr=((tw_gr_raw - tw_mean) / tw_std).astype(np.float32),
        tw_tvt=(tw_tvt_raw - anchor).astype(np.float32),
        anchor=anchor,
    )


def _load_folds(used_wells: list[str]) -> dict[str, int]:
    """Reuse the well-GroupKFold(5, seed=42) from the stack OOF (ported from R34)."""
    oof = ROOT / "outputs" / "stack_v26_oof.npz"
    if oof.exists():
        d = np.load(oof, allow_pickle=True)
        uw = [str(w) for w in d["used_wells"]]
        rf = d["row_fold"]
        wi = d["well_idx"]
        w2f: dict[str, int] = {}
        for w in range(len(uw)):
            idx = np.where(wi == w)[0]
            if len(idx):
                w2f[uw[w]] = int(rf[idx[0]])
        return w2f
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(used_wells))
    return {used_wells[w]: int(i % 5) for i, w in enumerate(perm)}


_FIELDS = ["q_gr", "q_md", "q_z", "q_prefix_rel", "q_is_known", "q_target", "q_eval_mask", "q_well"]


def build(smoke: int | None, out_path: Path) -> None:
    files = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    if smoke:
        files = files[:smoke]

    fields: dict[str, list[np.ndarray]] = {k: [] for k in _FIELDS}
    tw_gr_list, tw_tvt_list, anchors, used_wells = [], [], [], []

    for f in files:
        wid = Path(f).name[:8]
        h = _read_horizontal(Path(f))
        tw = _read_typewell(Path(f.replace("horizontal_well", "typewell")))

        wq = build_well_query(h, tw)
        if wq is None:
            continue

        fields["q_gr"].append(wq.q_gr)
        fields["q_md"].append(wq.q_md)
        fields["q_z"].append(wq.q_z)
        fields["q_prefix_rel"].append(wq.q_prefix_rel)
        fields["q_is_known"].append(wq.q_is_known)
        fields["q_target"].append(wq.q_target)
        fields["q_eval_mask"].append(wq.q_eval_mask)
        fields["q_well"].append(np.full(wq.q_gr.size, len(used_wells), np.int32))
        tw_gr_list.append(wq.tw_gr)
        tw_tvt_list.append(wq.tw_tvt)
        anchors.append(wq.anchor)
        used_wells.append(wid)

    if not used_wells:
        raise SystemExit("no wells passed the skip filter (eval/known < 20 rows for all)")

    w2f = _load_folds(used_wells)
    fold = np.array([w2f.get(w, 0) for w in used_wells], np.int32)

    np.savez_compressed(
        out_path,
        q_gr=np.concatenate(fields["q_gr"]),
        q_md=np.concatenate(fields["q_md"]),
        q_z=np.concatenate(fields["q_z"]),
        q_prefix_rel=np.concatenate(fields["q_prefix_rel"]),
        q_is_known=np.concatenate(fields["q_is_known"]),
        q_target=np.concatenate(fields["q_target"]),
        q_eval_mask=np.concatenate(fields["q_eval_mask"]),
        q_well=np.concatenate(fields["q_well"]),
        tw_gr=np.array(tw_gr_list, dtype=object),
        tw_tvt=np.array(tw_tvt_list, dtype=object),
        anchor=np.array(anchors, np.float32),
        used_wells=np.array(used_wells),
        fold=fold,
    )

    n_rows = sum(x.size for x in fields["q_gr"])
    n_eval = sum(int(m.sum()) for m in fields["q_eval_mask"])
    tw_med = int(np.median([t.size for t in tw_gr_list]))
    print(
        f"[done] wells={len(used_wells)} rows_total={n_rows:,} eval_rows_total={n_eval:,} "
        f"tw_len(med)={tw_med} -> {out_path}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(ROOT / "outputs" / "seq2seq_dataset.npz"))
    a = ap.parse_args()
    build(a.smoke, Path(a.out))
