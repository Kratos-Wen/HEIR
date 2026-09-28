import pytest
import torch

from .attention import InteractionAwareAttention, PrefixInteractionAdapter


@pytest.mark.parametrize('direction', ['detection', 'generation'])
def test_shared_block_shapes_gradients_and_padding(direction):
    torch.manual_seed(42)
    model = InteractionAwareAttention(8)
    visual = torch.randn(2, 4, 8, requires_grad=True)
    semantic = torch.randn(2, 3, 8, requires_grad=True)
    vm = torch.tensor([[True, True, True, False], [True, True, False, False]])
    sm = torch.tensor([[True, True, False], [True, True, True]])
    output, weight = model(visual, semantic, vm, sm, direction)
    qm, km = (vm, sm) if direction == 'detection' else (sm, vm)
    assert output.shape == (*qm.shape, 8)
    assert not output[~qm].count_nonzero()
    assert not weight.masked_select(~km[:, None].expand_as(weight)).count_nonzero()
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_equation_with_identity_projections():
    model = InteractionAwareAttention(2)
    with torch.no_grad():
        for layer in (model.q, model.k, model.v):
            layer.weight.copy_(torch.eye(2))
        model.modality.weight.zero_()
    q = torch.tensor([[[1., 0.], [0., 1.]]])
    k = torch.tensor([[[2., 3.]]])
    actual, _ = model(q, k, torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool), 'detection')
    torch.testing.assert_close(actual, q+k)


def test_direction_swap_reuses_exact_parameters():
    model = InteractionAwareAttention(4)
    x, y = torch.randn(1, 2, 4), torch.randn(1, 3, 4)
    with torch.no_grad():
        model.modality.weight.zero_()
    xm, ym = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)
    a, _ = model(x, y, xm, ym, 'detection')
    b, _ = model(y, x, ym, xm, 'generation')
    torch.testing.assert_close(a, b)


def test_future_target_never_enters_visual_prefix():
    model = PrefixInteractionAdapter(8)
    x = torch.randn(1, 8, 8, requires_grad=True)
    modality = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0]])
    valid = torch.ones(1, 8, dtype=torch.bool)
    length = torch.tensor([5])
    original = model(x, modality, valid, length, 'detection')
    changed = x.detach().clone()
    changed[:, 5:] += 100
    altered = model(changed, modality, valid, length, 'detection')
    torch.testing.assert_close(original[:, :5], altered[:, :5])
    original[:, :5].sum().backward()
    assert not x.grad[:, 5:].count_nonzero()


def test_reject_label_only_semantic_condition():
    model = PrefixInteractionAdapter(8)
    with pytest.raises(ValueError, match='Both modalities'):
        model(torch.randn(1, 4, 8), torch.tensor([[1, 1, 0, 0]]),
              torch.ones(1, 4, dtype=torch.bool), torch.tensor([2]), 'detection')


def test_all_padding_is_rejected():
    model = InteractionAwareAttention(4)
    with pytest.raises(ValueError, match='conditioning token'):
        model(torch.randn(1, 2, 4), torch.randn(1, 3, 4),
              torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool), 'detection')
