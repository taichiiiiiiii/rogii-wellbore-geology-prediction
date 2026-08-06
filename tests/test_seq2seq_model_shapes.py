"""Shape/behavior tests for src/rogii/seq2seq_model.py (SeqRegNet, Stage A/B primitives).

Covers the task's four required checks:
  - 5ch input (gr, md, z, prefix_rel, is_known) flows through the conv backbone.
  - loss_mask = eval_mask AND NOT padding (compute_loss_mask, design M-2).
  - prefix-holdout masking zeroes ch6/ch7 on the held-out block, and the aux
    target is NOT recoverable from the (masked) input -- design C-3.
  - typewell null-token fallback for a degenerate typewell (design M-3).

If torch is unavailable, this whole module is skipped (import-only checks are
covered separately in the local smoke report; real training happens on Kaggle).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rogii.seq2seq_model import (  # noqa: E402
    SeqRegNet,
    apply_prefix_holdout_mask,
    compute_loss_mask,
    is_typewell_degenerate,
    select_prefix_holdout_block,
)


def _rand_well(length: int = 24, k: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    gr = torch.randn(1, length, generator=g)
    md = torch.randn(1, length, generator=g).abs()
    z = torch.randn(1, length, generator=g)
    prefix = torch.randn(1, length, generator=g)
    isknown = (torch.rand(1, length, generator=g) > 0.5).float()
    tw_gr = torch.randn(1, k, generator=g)
    tw_tvt = torch.linspace(-1.0, 1.0, k).unsqueeze(0)
    return gr, md, z, prefix, isknown, tw_gr, tw_tvt


# --------------------------------------------------------------------------- #
# SeqRegNet forward shapes (5ch)
# --------------------------------------------------------------------------- #


def test_forward_accepts_5_channel_input_and_returns_full_well_shape() -> None:
    net = SeqRegNet(d=8)
    gr, md, z, prefix, isknown, tw_gr, tw_tvt = _rand_well(length=24, k=16)

    out, reg_tvt = net(gr, md, z, prefix, isknown, tw_gr, tw_tvt)

    assert out.shape == (1, 24)
    assert reg_tvt.shape == (1, 24)
    assert torch.isfinite(out).all()


def test_eval_enc_first_conv_layer_has_5_input_channels() -> None:
    net = SeqRegNet(d=8)

    assert net.eval_enc[0].in_channels == 5


def test_forward_batches_of_more_than_one_well() -> None:
    net = SeqRegNet(d=8)
    gr, md, z, prefix, isknown, tw_gr, tw_tvt = _rand_well(length=10, k=6)
    cat2 = lambda t: torch.cat([t, t], dim=0)  # noqa: E731

    out, _ = net(
        cat2(gr), cat2(md), cat2(z), cat2(prefix), cat2(isknown), cat2(tw_gr), cat2(tw_tvt)
    )

    assert out.shape == (2, 10)


# --------------------------------------------------------------------------- #
# compute_loss_mask (design M-2)
# --------------------------------------------------------------------------- #


def test_compute_loss_mask_without_padding_equals_eval_mask() -> None:
    eval_mask = torch.tensor([[True, False, True, False]])

    loss_mask = compute_loss_mask(eval_mask, None)

    assert torch.equal(loss_mask, eval_mask)


def test_compute_loss_mask_excludes_padded_rows_even_if_marked_eval() -> None:
    eval_mask = torch.tensor([[True, True, True, True]])
    padding_mask = torch.tensor([[False, False, True, True]])

    loss_mask = compute_loss_mask(eval_mask, padding_mask)

    assert torch.equal(loss_mask, torch.tensor([[True, True, False, False]]))


def test_compute_loss_mask_does_not_depend_on_is_known() -> None:
    # loss_mask must derive only from eval_mask/padding_mask -- guards against
    # silently conflating it with the is_known channel (M-2 "静かにバグる箇所").
    eval_mask = torch.tensor([[False, True, True]])
    padding_mask = torch.tensor([[True, False, False]])

    loss_mask = compute_loss_mask(eval_mask, padding_mask)

    assert torch.equal(loss_mask, torch.tensor([[False, True, True]]))


# --------------------------------------------------------------------------- #
# prefix-holdout masking (design C-3)
# --------------------------------------------------------------------------- #


def test_select_prefix_holdout_block_is_contiguous_and_within_known_rows() -> None:
    is_known = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.float32)
    rng = np.random.RandomState(0)

    holdout = select_prefix_holdout_block(is_known, frac=0.3, rng=rng)

    assert holdout.dtype == bool
    assert holdout.shape == is_known.shape
    assert np.all(is_known[holdout] == 1.0)  # only known rows selected
    idx = np.flatnonzero(holdout)
    assert idx.size == round(10 * 0.3)
    assert np.array_equal(idx, np.arange(idx.min(), idx.max() + 1))  # contiguous


def test_select_prefix_holdout_block_empty_when_no_known_rows() -> None:
    is_known = np.zeros(5, dtype=np.float32)

    holdout = select_prefix_holdout_block(is_known, frac=0.3, rng=np.random.RandomState(0))

    assert holdout.sum() == 0


def test_apply_prefix_holdout_mask_zeroes_ch6_and_ch7_on_holdout_rows_only() -> None:
    prefix = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    isknown = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    holdout = np.array([False, True, True, False])

    masked_prefix, masked_isknown, target = apply_prefix_holdout_mask(prefix, isknown, holdout)

    np.testing.assert_allclose(masked_prefix, [1.0, 0.0, 0.0, 4.0])
    np.testing.assert_allclose(masked_isknown, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(target, [2.0, 3.0])
    # inputs are never mutated (immutable pattern)
    np.testing.assert_allclose(prefix, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(isknown, [1.0, 1.0, 1.0, 0.0])


def test_apply_prefix_holdout_mask_is_not_a_no_op_identity_copy() -> None:
    # C-3 regression guard: the REJECTED design fed ch6 unmasked as its own aux
    # target (identity no-op). Assert the model-visible input actually changes
    # at holdout positions, and the target is not trivially present in ch6/ch7.
    prefix = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    isknown = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    holdout = np.array([False, True, False])

    masked_prefix, masked_isknown, target = apply_prefix_holdout_mask(prefix, isknown, holdout)

    assert masked_prefix[1] != prefix[1]
    assert masked_isknown[1] != isknown[1]
    assert masked_prefix[1] == 0.0
    assert masked_isknown[1] == 0.0
    assert target[0] == prefix[1]


# --------------------------------------------------------------------------- #
# typewell degeneracy detection (design M-3)
# --------------------------------------------------------------------------- #


def test_is_typewell_degenerate_true_for_short_typewell() -> None:
    tw_gr = np.random.RandomState(0).normal(size=20)
    assert is_typewell_degenerate(tw_gr) is True


def test_is_typewell_degenerate_true_for_constant_typewell() -> None:
    tw_gr = np.full(200, 100.0)
    assert is_typewell_degenerate(tw_gr) is True


def test_is_typewell_degenerate_false_for_normal_typewell() -> None:
    tw_gr = np.random.RandomState(0).normal(loc=100, scale=15, size=200)
    assert is_typewell_degenerate(tw_gr) is False


# --------------------------------------------------------------------------- #
# null-token fallback (design M-3)
# --------------------------------------------------------------------------- #


def test_degenerate_typewell_uses_learnable_null_token_not_real_attention() -> None:
    net = SeqRegNet(d=8)
    gr, md, z, prefix, isknown, tw_gr, tw_tvt = _rand_well(length=10, k=6)
    deg = torch.tensor([True])

    out, reg_tvt = net(gr, md, z, prefix, isknown, tw_gr, tw_tvt, tw_degenerate=deg)

    assert torch.isfinite(out).all()
    expected = net.null_reg_tvt.expand(1, 10)
    assert torch.allclose(reg_tvt, expected)


def test_null_token_is_learnable_and_receives_gradient_when_degenerate() -> None:
    net = SeqRegNet(d=8)
    gr, md, z, prefix, isknown, tw_gr, tw_tvt = _rand_well(length=10, k=6)
    deg = torch.tensor([True])

    out, _ = net(gr, md, z, prefix, isknown, tw_gr, tw_tvt, tw_degenerate=deg)
    out.sum().backward()

    assert net.null_reg_tvt.requires_grad
    assert net.null_reg_tvt.grad is not None
    assert net.null_reg_tvt.grad.abs().sum().item() > 0.0


def test_non_degenerate_wells_still_use_real_cross_attention() -> None:
    net = SeqRegNet(d=8)
    gr, md, z, prefix, isknown, tw_gr, tw_tvt = _rand_well(length=10, k=6)
    deg = torch.tensor([False])

    out, reg_tvt = net(gr, md, z, prefix, isknown, tw_gr, tw_tvt, tw_degenerate=deg)

    null_broadcast = net.null_reg_tvt.expand(1, 10)
    assert not torch.allclose(reg_tvt, null_broadcast)
