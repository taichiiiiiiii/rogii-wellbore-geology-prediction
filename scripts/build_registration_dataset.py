"""Phase 0 (medal R&D) — dataset builder for LEARNED neural GR↔typewell registration.

Genuinely-new primitive vs R10 (which regressed TVT−pf_blend from engineered
features, never feeding the typewell as a registration target). Here the model
gets the RAW eval GR sequence + the RAW typewell (GR, TVT) profile and must learn
to align them → TVT. This is the winners' "purely model driven" paradigm and the
only untried lever after R32 (PF/beam primitive oracle-capped at tail 10.84).

Per well we store (flat, with per-well offsets, like outputs/stack_v26_oof.npz):
  eval side (query, length = eval rows):
    e_gr      : eval-zone GR, global z-scored (raw registration signal)
    e_md_rel  : MD − MD[eval_start]           (KNOWN at inference; geometry)
    e_z_rel   : Z  − Z[eval_start]            (KNOWN at inference; wellbore vertical motion)
    e_target  : TVT − anchor                   (LABEL; drift from heel anchor)
  typewell side (key/value, length = typewell rows), stored per well (ragged):
    tw_gr     : typewell GR, global z-scored
    tw_tvt_rel: typewell TVT − anchor          (candidate drift positions)
  scalars:
    anchor    : last_known_tvt
    known GR-TVT context summary (mean/std) for the model to self-calibrate.

Leak-safe: eval-zone uses only GR/MD/Z (all present in test) + typewell (present
in test) + anchor (last known TVT_input). NEVER uses eval TVT/TVT_input/formation
cols as inputs. Target is eval TVT (train-only label).

Run:  uv run python scripts/build_registration_dataset.py [--smoke N] [--out PATH]
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train"


def _load_folds(used_wells: list[str]) -> dict[str, int]:
    """Reuse the exact well-GroupKFold(5, seed=42) from the stack OOF for comparability."""
    oof = ROOT / "outputs" / "stack_v26_oof.npz"
    if oof.exists():
        d = np.load(oof, allow_pickle=True)
        uw = [str(w) for w in d["used_wells"]]
        rf = d["row_fold"]
        wi = d["well_idx"]
        # per-well fold = fold of its first row
        w2f = {}
        for w in range(len(uw)):
            idx = np.where(wi == w)[0]
            if len(idx):
                w2f[uw[w]] = int(rf[idx[0]])
        return w2f
    # fallback: deterministic round-robin
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(used_wells))
    return {used_wells[w]: int(i % 5) for i, w in enumerate(perm)}


def build(smoke: int | None, out_path: Path) -> None:
    files = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    if smoke:
        files = files[:smoke]

    # First pass: global GR normalization stats (eval + typewell GR share the log type)
    gr_sum = gr_sq = gr_n = 0.0
    for f in files:
        gr = pd.read_csv(f, usecols=["GR"])["GR"].to_numpy()
        gr = gr[np.isfinite(gr)]
        gr_sum += gr.sum()
        gr_sq += (gr * gr).sum()
        gr_n += gr.size
    gr_mean = gr_sum / gr_n
    gr_std = float(np.sqrt(gr_sq / gr_n - gr_mean**2))
    print(f"[norm] global GR mean={gr_mean:.2f} std={gr_std:.2f} (n={int(gr_n):,})")

    e_gr, e_md, e_z, e_tgt, e_well = [], [], [], [], []
    tw_gr_list, tw_tvt_list = [], []
    anchors, kn_gr_mean, kn_gr_std, kn_tvt_mean = [], [], [], []
    used_wells = []

    for wi, f in enumerate(files):
        wid = os.path.basename(f)[:8]
        h = pd.read_csv(f, usecols=["MD", "Z", "TVT", "TVT_input", "GR"])
        twf = f.replace("horizontal_well", "typewell")
        tw = pd.read_csv(twf)  # columns: TVT, GR, Geology
        known = h["TVT_input"].notna().to_numpy()
        ev = ~known
        if ev.sum() < 20 or known.sum() < 20:
            continue
        anchor = float(h["TVT_input"].to_numpy()[known][-1])
        eval_start_md = float(h["MD"].to_numpy()[ev][0])
        eval_start_z = float(h["Z"].to_numpy()[ev][0])

        twg = tw.dropna(subset=["GR"]).sort_values("TVT")
        tw_tvt_raw = twg["TVT"].to_numpy()
        tw_gr_raw = twg["GR"].to_numpy()

        # --- per-well GR affine calibration: a*hw_GR + b ~ typewell_GR at known TVT ---
        # leak-safe: uses TVT_input (given in known region, == TVT there) NOT eval TVT.
        known_tvt = h["TVT_input"].to_numpy()[known]           # given known TVT
        known_gr = h["GR"].to_numpy()[known]
        tw_gr_at_known = np.interp(known_tvt, tw_tvt_raw, tw_gr_raw)
        vcal = np.isfinite(known_gr) & np.isfinite(tw_gr_at_known)
        if vcal.sum() > 10:
            a_gr, b_gr = np.polyfit(known_gr[vcal], tw_gr_at_known[vcal], 1)
        else:
            a_gr, b_gr = 1.0, 0.0
        tw_mean = float(np.nanmean(tw_gr_raw)); tw_std = float(np.nanstd(tw_gr_raw)) or 1.0

        eval_gr_raw = h["GR"].to_numpy()[ev]
        egr = ((a_gr * eval_gr_raw + b_gr) - tw_mean) / tw_std   # calibrated to typewell frame
        egr = pd.Series(egr).interpolate(limit_direction="both").fillna(0.0).to_numpy().astype(np.float32)
        emd = (h["MD"].to_numpy()[ev] - eval_start_md).astype(np.float32)
        ez = (h["Z"].to_numpy()[ev] - eval_start_z).astype(np.float32)
        etgt = (h["TVT"].to_numpy()[ev] - anchor).astype(np.float32)

        tgr = ((tw_gr_raw - tw_mean) / tw_std).astype(np.float32)   # same typewell frame
        ttvt = (tw_tvt_raw - anchor).astype(np.float32)

        kgr = ((a_gr * known_gr + b_gr) - tw_mean) / tw_std
        ktvt = known_tvt - anchor

        e_gr.append(egr); e_md.append(emd); e_z.append(ez); e_tgt.append(etgt)
        e_well.append(np.full(egr.size, len(used_wells), np.int32))
        tw_gr_list.append(tgr); tw_tvt_list.append(ttvt)
        anchors.append(anchor)
        kn_gr_mean.append(float(np.nanmean(kgr))); kn_gr_std.append(float(np.nanstd(kgr)))
        kn_tvt_mean.append(float(np.nanmean(ktvt)))
        used_wells.append(wid)

    w2f = _load_folds(used_wells)
    fold = np.array([w2f.get(w, 0) for w in used_wells], np.int32)

    np.savez_compressed(
        out_path,
        e_gr=np.concatenate(e_gr), e_md=np.concatenate(e_md),
        e_z=np.concatenate(e_z), e_target=np.concatenate(e_tgt),
        e_well=np.concatenate(e_well),
        tw_gr=np.array(tw_gr_list, dtype=object), tw_tvt=np.array(tw_tvt_list, dtype=object),
        anchor=np.array(anchors, np.float32),
        kn_gr_mean=np.array(kn_gr_mean, np.float32), kn_gr_std=np.array(kn_gr_std, np.float32),
        kn_tvt_mean=np.array(kn_tvt_mean, np.float32),
        used_wells=np.array(used_wells), fold=fold,
        gr_mean=np.float32(gr_mean), gr_std=np.float32(gr_std),
    )
    n_eval = sum(x.size for x in e_gr)
    print(f"[done] wells={len(used_wells)} eval_rows={n_eval:,} "
          f"tw_len(med)={int(np.median([t.size for t in tw_gr_list]))} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(ROOT / "outputs" / "reg_dataset.npz"))
    a = ap.parse_args()
    build(a.smoke, Path(a.out))
