import math
import pytest
import torch

from .diffusion import (ForwardDiffusion, check_image, ground_truth, initialize,
                        multinomial_frequency, normalized_multinomial_log_prob, decode)
from .model import SliceDiT


def test_ground_truth_preserves_multilabel_and_channel_order():
    x = ground_truth(torch.tensor([1]), torch.tensor([[1., 0., 1.]]), 3)
    check_image(x)
    assert x[0, 1, :, 0].tolist() == [1, 0, 1]
    assert x[0, 1, :, 1].tolist() == [0, 1, 0]
    assert not x[0, 0].count_nonzero()


def test_forward_exact_steps_stay_on_simplex():
    torch.manual_seed(42)
    prior = initialize(torch.tensor([[.2, .8]]), 3)
    x = ground_truth(torch.tensor([0]), torch.tensor([[1., 1., 0.]]), 2)
    process = ForwardDiffusion(steps=5, trials=100)
    for k in range(1, 6):
        x = process.step(x, prior, k)
        check_image(x)
    assert not torch.equal(x, prior)


def test_effective_trial_count_matches_expanded_equation():
    process = ForwardDiffusion()
    for k in (1, 2, 10, 50):
        weights = torch.stack([process.beta[j] * (1-process.beta[j+1:k+1]).prod() for j in range(1, k+1)])
        expected = round(float(weights.sum().square()/weights.square().sum()*2000))
        assert int(process.effective_trials[k]) == expected
    assert int(process.effective_trials[1]) == 2000


@pytest.mark.parametrize('k', [0, 1, 20, 50])
def test_approximation_simplex(k):
    clean = ground_truth(torch.tensor([0, 1]), torch.tensor([[1., 0.], [0., 1.]]), 2)
    prior = initialize(torch.tensor([[.3, .7], [.8, .2]]), 2)
    x = ForwardDiffusion().sample_approximate(clean, prior, k)
    check_image(x)
    if k == 0:
        torch.testing.assert_close(x, clean)


def test_multinomial_likelihood_not_pixel_probabilities():
    prior = initialize(torch.ones(1, 1), 1)
    x = torch.tensor([[[[.5, .5]]]])
    # Two trials: exactly one presence and one absence has probability .5.
    result = normalized_multinomial_log_prob(x, prior, 2)
    torch.testing.assert_close(result, torch.tensor([[math.log(.5)]], dtype=torch.float64))
    invalid = torch.tensor([[[[.3, .7]]]])
    assert normalized_multinomial_log_prob(invalid, prior, 2).isneginf().all()


def test_joint_marginal_product_is_not_general_inverse():
    joint = torch.tensor([[.5, 0.], [0., .5]])
    product = joint.sum(1)[:, None] * joint.sum(0)[None]
    assert not torch.equal(joint, product)


def test_same_entity_classification_aggregates_across_people():
    x = torch.stack((initialize(torch.tensor([[.9, .1]]), 2)[0],
                     initialize(torch.tensor([[.2, .8]]), 2)[0]))
    nouns, present, _ = decode(x, torch.tensor([9, 9]))
    assert nouns.tolist() == [0, 0]
    assert not present.any()


def test_slice_dit_forward_backward():
    torch.manual_seed(42)
    model = SliceDiT(3, 4, 16, width=24, depth=2, heads=4)
    prior = initialize(torch.tensor([[.2, .3, .5], [.5, .3, .2]]), 4)
    output = model(prior, torch.tensor([1, 50]), torch.randn(2, 16))
    check_image(output)
    output[:, 0, :, 0].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    check_image(model.reverse(prior, torch.randn(2, 16), steps=2))


def test_noninteger_trials_are_not_silently_truncated():
    with pytest.raises(ValueError, match='integer'):
        multinomial_frequency(initialize(torch.ones(1, 1), 2), 1.5)
