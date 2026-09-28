import math

import torch

from .train_detection import epoch_indices, learning_rate


def test_exact_distributed_coverage_and_resume_order():
    for epoch in (0, 1, 9):
        ranks = [epoch_indices(5400, 8, r, epoch) for r in range(8)]
        flat = sum(ranks, [])
        assert sorted(flat) == list(range(5400))
        assert len(ranks[0]) == 675
        assert ranks[3][400:] == epoch_indices(5400, 8, 3, epoch)[400:]
    assert epoch_indices(5400, 8, 0, 0) != epoch_indices(5400, 8, 0, 1)


def test_fixed_schedule_and_last_partial_batch():
    per_epoch = math.ceil(675 / 4)
    assert per_epoch == 169
    rates = [learning_rate(i, 1690, 169, 5e-5) for i in range(1690)]
    assert math.isclose(rates[168], 5e-5)
    assert math.isclose(rates[-1], 5e-6)
    assert all(0 < r <= 5e-5 for r in rates)
    batches = [len(range(i, min(i + 4, 675))) * 8 for i in range(0, 675, 4)]
    assert sum(batches) == 5400 and batches[-1] == 24


def test_token_normalization_under_rank_average_and_accumulation():
    weights = torch.tensor(0.7, requires_grad=True)
    chunks = [torch.arange(1, n + 1).float() for n in (2, 3, 1, 4)]
    denom = sum(len(x) for x in chunks)
    reference = sum(((weights * x - 1)**2).sum() for x in chunks) / denom
    expected, = torch.autograd.grad(reference, weights)
    gradients = []
    for rank in range(2):
        value = torch.tensor(0.7, requires_grad=True)
        for x in chunks[rank::2]:
            (((value * x - 1)**2).sum() * 2 / denom).backward()
        gradients.append(value.grad)
    torch.testing.assert_close(torch.stack(gradients).mean(), expected)
