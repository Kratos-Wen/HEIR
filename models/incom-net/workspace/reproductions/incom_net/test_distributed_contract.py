import pytest
import torch

from .model import focal_mft_loss


@pytest.mark.parametrize('positives', [0, 1, 3, 8, 16])
def test_eight_rank_loss_matches_global_batch_even_for_sparse_positives(positives):
    torch.manual_seed(42)
    x = torch.randn(16, 24, requires_grad=True)
    target = torch.zeros_like(x)
    target[:positives, 0] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    modes = ('full', 'detector_only', 'vlm_only')
    total, _ = focal_mft_loss({k: {'logits': x} for k in modes}, target, valid, alpha=.5, gamma=.1)
    expected, = torch.autograd.grad(total, x)
    denominator = target.sum().clamp(min=1) / 8
    local_losses = []
    for rank in range(8):
        sl = slice(rank * 2, rank * 2 + 2)
        loss, _ = focal_mft_loss({k: {'logits': x[sl]} for k in modes}, target[sl], valid[sl],
                                alpha=.5, gamma=.1, normalizer=denominator)
        local_losses.append(loss)
    actual, = torch.autograd.grad(sum(local_losses) / 8, x)
    torch.testing.assert_close(actual, expected)


def test_invalid_explicit_denominator_is_rejected():
    x = torch.zeros(1, 24)
    with pytest.raises(ValueError, match='normalizer'):
        focal_mft_loss({k: {'logits': x} for k in ('full', 'detector_only', 'vlm_only')},
                       x, x.bool(), alpha=.5, gamma=.1, normalizer=0)
