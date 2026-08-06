"""Stage A/B full-well seq2seq training + eval -- analysis/seq2seq_design.md.

Stage A = R34-exact reproduction sanity check: ch4 (prefix_rel) / ch5
(is_known) are zeroed at the model input and the prefix-holdout auxiliary
loss (C-3) is disabled. If the full-well plumbing (leak-safe query
construction, loss masking, pooled/tail scoring) is wired correctly, Stage A
should land in R34's ballpark (tail_rmse ~14) since the model then sees
(almost) the same information R34's eval-only RegNet saw.

Stage B = the actual ablation under test: ch4/ch5 live, plus the in-model
prefix-holdout auxiliary loss (design C-3) that masks a random contiguous
block of known rows and asks the model to reconstruct their prefix drift from
GR + typewell context alone (never from the masked channels themselves).

Local runs are smoke-only (a handful of wells, CPU, ``--epochs`` small) -- the
real 5-fold Kaggle run is analysis/seq2seq_design.md's "実行" step 3.

Run:  uv run python scripts/run_seq2seq.py --data PATH --stage {A,B} --fold 0
      --epochs N --device {cpu,cuda}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise SystemExit(f"torch required: {e}") from e

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.seq2seq_model import (  # noqa: E402
    AUX_LOSS_WEIGHT,
    HOLDOUT_FRAC,
    SeqRegNet,
    apply_prefix_holdout_mask,
    is_typewell_degenerate,
    select_prefix_holdout_block,
)

K_TW = 256
_SCALE = 100.0  # net-internal ft/100 scale, ported from R34 (grid/target both /100)


def tw_to_grid(
    tw_gr: np.ndarray, tw_tvt: np.ndarray, k: int = K_TW
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a ragged typewell (GR vs TVT) onto k evenly-TVT-spaced keypoints (from R34)."""
    order = np.argsort(tw_tvt)
    t, g = tw_tvt[order], tw_gr[order]
    grid = np.linspace(t[0], t[-1], k).astype(np.float32)
    return np.interp(grid, t, g).astype(np.float32), grid


def _well_arrays(d: np.lib.npyio.NpzFile, w: int) -> dict:
    """Load one well's per-row arrays (normalized to net-internal scale)."""
    m = np.where(d["q_well"] == w)[0]
    tw_gr_raw, tw_tvt_raw = d["tw_gr"][w], d["tw_tvt"][w]
    gg, grid = tw_to_grid(tw_gr_raw, tw_tvt_raw)
    return {
        "gr": d["q_gr"][m].astype(np.float32),
        "md": (d["q_md"][m] / 1000.0).astype(np.float32),
        "z": (d["q_z"][m] / 100.0).astype(np.float32),
        "prefix": (d["q_prefix_rel"][m] / _SCALE).astype(np.float32),
        "isknown": d["q_is_known"][m].astype(np.float32),
        "target": d["q_target"][m].astype(np.float32),
        "eval_mask": d["q_eval_mask"][m].astype(bool),
        "tw_gr": gg,
        "tw_tvt": (grid / _SCALE).astype(np.float32),
        "degenerate": is_typewell_degenerate(tw_gr_raw),
    }


def _to_batch(arr: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.from_numpy(arr)[None].to(device)


def _net_device(net: SeqRegNet) -> torch.device:
    return next(net.parameters()).device


def _forward_well(
    net: SeqRegNet, well: dict, prefix_in: np.ndarray, isknown_in: np.ndarray
) -> torch.Tensor:
    dev = _net_device(net)
    out, _ = net(
        _to_batch(well["gr"], dev),
        _to_batch(well["md"], dev),
        _to_batch(well["z"], dev),
        _to_batch(prefix_in, dev),
        _to_batch(isknown_in, dev),
        _to_batch(well["tw_gr"], dev),
        _to_batch(well["tw_tvt"], dev),
        torch.tensor([well["degenerate"]], device=dev),
    )
    return out


def _stage_inputs(
    well: dict, stage: str, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (prefix_in, isknown_in, holdout_mask) for one training step."""
    if stage == "A":
        zeros = np.zeros_like(well["prefix"])
        return zeros, zeros, np.zeros_like(well["eval_mask"], dtype=bool)
    holdout = select_prefix_holdout_block(well["isknown"], frac=HOLDOUT_FRAC, rng=rng)
    prefix_in, isknown_in, _ = apply_prefix_holdout_mask(well["prefix"], well["isknown"], holdout)
    return prefix_in, isknown_in, holdout


def train_step(
    net: SeqRegNet,
    opt: torch.optim.Optimizer,
    huber: torch.nn.Module,
    well: dict,
    stage: str,
    rng: np.random.RandomState,
) -> float:
    """One well's forward/backward step. Main loss = eval rows; aux loss (Stage B) =
    prefix-holdout reconstruction on the SAME per-row head output (see report:
    q_target == q_prefix_rel on known rows, verified empirically)."""
    dev = _net_device(net)
    prefix_in, isknown_in, holdout = _stage_inputs(well, stage, rng)
    out = _forward_well(net, well, prefix_in, isknown_in)
    y = _to_batch(well["target"] / _SCALE, dev)

    eval_t = _to_batch(well["eval_mask"], dev)
    main_loss = huber(out[eval_t], y[eval_t])

    aux_loss = out.sum() * 0.0
    if stage == "B" and holdout.any():
        hold_t = _to_batch(holdout, dev)
        aux_loss = F.mse_loss(out[hold_t], y[hold_t])

    loss = main_loss + AUX_LOSS_WEIGHT * aux_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())


def eval_well(net: SeqRegNet, well: dict, stage: str) -> tuple[float, float, int, int]:
    """Return (sse_all, sse_tail, n_all, n_tail) for one well's eval zone (no holdout at eval)."""
    prefix_in = well["prefix"] if stage == "B" else np.zeros_like(well["prefix"])
    isknown_in = well["isknown"] if stage == "B" else np.zeros_like(well["isknown"])
    with torch.no_grad():
        out = _forward_well(net, well, prefix_in, isknown_in)
    pred = out.squeeze(0).cpu().numpy() * _SCALE
    eval_idx = np.flatnonzero(well["eval_mask"])
    if eval_idx.size == 0:
        return 0.0, 0.0, 0, 0
    err2 = (pred[eval_idx] - well["target"][eval_idx]) ** 2
    tail_start = eval_idx.size // 2  # tail_rmse = second half of the eval zone (design H-2)
    n_all, n_tail = int(eval_idx.size), int(eval_idx.size - tail_start)
    return float(err2.sum()), float(err2[tail_start:].sum()), n_all, n_tail


def _run_epoch_train(net, opt, huber, d, tr_wells, stage, shuffle_rng, holdout_rng) -> float:
    net.train()
    order = list(tr_wells)
    shuffle_rng.shuffle(order)
    tl = tn = 0.0
    for w in order:
        well = _well_arrays(d, w)
        tl += train_step(net, opt, huber, well, stage, holdout_rng)
        tn += 1
    return tl / max(tn, 1)


def _run_epoch_eval(net, d, va_wells, stage) -> tuple[float, float]:
    net.eval()
    se = se_tail = 0.0
    n = n_tail = 0
    for w in va_wells:
        well = _well_arrays(d, w)
        s, st, cnt, cnt_t = eval_well(net, well, stage)
        se += s
        se_tail += st
        n += cnt
        n_tail += cnt_t
    pooled = float(np.sqrt(se / n)) if n else float("nan")
    tail = float(np.sqrt(se_tail / n_tail)) if n_tail else float("nan")
    return pooled, tail


def run(path: Path, stage: str, epochs: int, fold: int, device: str) -> float:
    d = np.load(path, allow_pickle=True)
    folds = d["fold"]
    nwell = len(d["used_wells"])
    tr_wells = [w for w in range(nwell) if folds[w] != fold]
    va_wells = [w for w in range(nwell) if folds[w] == fold]
    print(f"[stage {stage}] wells train={len(tr_wells)} val={len(va_wells)} device={device}")

    net = SeqRegNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    huber = torch.nn.SmoothL1Loss(beta=5.0)
    shuffle_rng = np.random.RandomState(0)
    holdout_rng = np.random.RandomState(1)

    pooled = float("nan")
    for ep in range(epochs):
        t0 = time.time()
        train_loss = _run_epoch_train(net, opt, huber, d, tr_wells, stage, shuffle_rng, holdout_rng)
        pooled, tail = _run_epoch_eval(net, d, va_wells, stage)
        print(
            f"ep {ep:2d} train_loss={train_loss:.4f} pooled_rmse={pooled:.4f} "
            f"tail_rmse={tail:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return pooled


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT / "outputs" / "seq2seq_dataset.npz"))
    ap.add_argument("--stage", type=str, choices=["A", "B"], default="A")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    run(Path(a.data), a.stage, a.epochs, a.fold, dev)
