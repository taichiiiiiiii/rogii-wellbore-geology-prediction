"""Full-well seq2seq registration model (SeqRegNet) -- analysis/seq2seq_design.md.

Channel layout matches the task spec (not the full design.md list): the
``eval_enc`` input goes from R34's 3ch (gr, md, z) to 5ch by appending
``prefix_rel`` (design.md ch6) and ``is_known`` (design.md ch7) only --
``dGR``/``X_rel``/``Y_rel`` from the design doc's longer channel list are
deliberately NOT added, so the only diff from R34's backbone is isolated to
these two new channels (H-4 variable isolation).

Three design guards live here as pure, independently-testable functions:
  - ``compute_loss_mask`` (M-2): loss mask = eval_mask AND NOT padding, kept
    structurally independent of the ``is_known`` channel.
  - ``select_prefix_holdout_block`` / ``apply_prefix_holdout_mask`` (C-3): the
    in-model prefix-holdout auxiliary loss masks a contiguous block of known
    rows (zeroing ch6/ch7 there) and asks the model to reconstruct the
    original prefix value from GR + typewell context alone. The REJECTED
    earlier design fed ch6 unmasked as its own reconstruction target, which is
    an identity no-op -- these two functions guarantee the mask is applied
    before any target is exposed.
  - ``is_typewell_degenerate`` (M-3): typewell too short/flat to register
    against falls back to a learnable scalar drift (``SeqRegNet.null_reg_tvt``)
    instead of noisy cross-attention.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover -- exercised only when torch is absent
    TORCH_AVAILABLE = False

_MIN_TW_LEN = 50
_MIN_TW_GR_STD = 1e-6
AUX_LOSS_WEIGHT = 0.3
HOLDOUT_FRAC = 0.3


def is_typewell_degenerate(tw_gr: np.ndarray) -> bool:
    """True when the typewell reference is too short or too flat to register against (M-3)."""
    if tw_gr.size < _MIN_TW_LEN:
        return True
    std = float(np.nanstd(tw_gr))
    return bool((not np.isfinite(std)) or std < _MIN_TW_GR_STD)


def select_prefix_holdout_block(
    is_known: np.ndarray,
    frac: float = HOLDOUT_FRAC,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Pick one contiguous block of known rows (~``frac`` of them) for the C-3 aux loss.

    Returns a boolean mask (same length as ``is_known``), True only on known
    rows selected for holdout. All-False when there are no known rows.
    """
    rng = rng if rng is not None else np.random.RandomState()
    known_idx = np.flatnonzero(is_known > 0.5)
    holdout = np.zeros(is_known.shape, dtype=bool)
    if known_idx.size == 0:
        return holdout
    block_len = min(known_idx.size, max(1, round(known_idx.size * frac)))
    start = int(rng.randint(0, known_idx.size - block_len + 1))
    holdout[known_idx[start : start + block_len]] = True
    return holdout


def apply_prefix_holdout_mask(
    q_prefix: np.ndarray, q_is_known: np.ndarray, holdout_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero ch6/ch7 on ``holdout_mask`` rows (never mutates inputs); return the recon target.

    C-3: the masking must happen BEFORE anything is exposed to the model --
    the third return value (the original ``q_prefix`` at holdout rows) is only
    ever used as a loss target, never as an input.
    """
    masked_prefix = q_prefix.copy()
    masked_is_known = q_is_known.copy()
    target = q_prefix[holdout_mask].copy()
    masked_prefix[holdout_mask] = 0.0
    masked_is_known[holdout_mask] = 0.0
    return masked_prefix, masked_is_known, target


if TORCH_AVAILABLE:

    def compute_loss_mask(
        eval_mask: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """loss_mask = eval_mask AND NOT padding (M-2: kept independent of is_known)."""
        if padding_mask is None:
            return eval_mask
        return eval_mask & (~padding_mask)

    class SeqRegNet(nn.Module):
        """R34 ``RegNet`` backbone, byte-identical except a 3->5 channel ``eval_enc`` input.

        See ``scripts.train_registration.RegNet`` for the frozen reference
        backbone this mirrors (typewell cross-attn + bidirectional GRU + head).
        """

        def __init__(self, d: int = 64):
            super().__init__()
            self.eval_enc = nn.Sequential(
                nn.Conv1d(5, d, 7, padding=3),
                nn.GELU(),
                nn.Conv1d(d, d, 7, padding=3, dilation=1),
                nn.GELU(),
                nn.Conv1d(d, d, 7, padding=6, dilation=2),
                nn.GELU(),
            )
            self.tw_enc = nn.Sequential(
                nn.Conv1d(1, d, 7, padding=3),
                nn.GELU(),
                nn.Conv1d(d, d, 7, padding=3),
                nn.GELU(),
            )
            self.q = nn.Linear(d, d)
            self.kk = nn.Linear(d, d)
            self.scale = d**-0.5
            self.gru = nn.GRU(d + 3, d, num_layers=2, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * d, 1)
            # M-3: learnable scalar registration-drift fallback for degenerate typewells.
            self.null_reg_tvt = nn.Parameter(torch.zeros(1))

        def forward(
            self,
            q_gr: torch.Tensor,
            q_md: torch.Tensor,
            q_z: torch.Tensor,
            q_prefix: torch.Tensor,
            q_isknown: torch.Tensor,
            tw_gr: torch.Tensor,
            tw_tvt: torch.Tensor,
            tw_degenerate: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # q_*: (B, L)   tw_gr/tw_tvt: (B, K)   tw_degenerate: (B,) bool or None
            x = torch.stack([q_gr, q_md, q_z, q_prefix, q_isknown], dim=1)  # (B,5,L)
            ef = self.eval_enc(x).transpose(1, 2)  # (B,L,d)
            tf = self.tw_enc(tw_gr.unsqueeze(1)).transpose(1, 2)  # (B,K,d)

            att = torch.softmax((self.q(ef) @ self.kk(tf).transpose(1, 2)) * self.scale, dim=-1)
            reg_tvt = (att @ tw_tvt.unsqueeze(-1)).squeeze(-1)  # (B,L)

            if tw_degenerate is not None:
                b, length = reg_tvt.shape
                null_val = self.null_reg_tvt.view(1, 1).expand(b, length)
                deg = tw_degenerate.view(b, 1).expand(b, length)
                reg_tvt = torch.where(deg, null_val, reg_tvt)

            seq = torch.cat(
                [ef, reg_tvt.unsqueeze(-1), q_md.unsqueeze(-1), q_z.unsqueeze(-1)], dim=-1
            )
            h, _ = self.gru(seq)
            out = self.head(h).squeeze(-1) + reg_tvt
            return out, reg_tvt
