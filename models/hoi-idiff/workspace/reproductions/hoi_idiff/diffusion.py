"""HOI-IDiff main-paper equations 4/5 and the multinomial likelihood.

This does not silently replace the unspecified posterior-MSE objective with
DDPM epsilon prediction or clean-image regression. See reconstruction.json.
The last axis is [presence, absence], not [absence, presence].
"""

import torch
from torch import nn
from torch.nn import functional as F


def check_image(image):
    if image.ndim != 4 or image.shape[-1] != 2 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError('Expected B x nouns x actions x 2')
    if not torch.isfinite(image).all() or (image < 0).any():
        raise ValueError('HOI image must be finite and nonnegative')
    torch.testing.assert_close(image.sum((1, 3)), torch.ones_like(image[:, 0, :, 0]), atol=2e-5, rtol=2e-5)


def ground_truth(noun_ids, multi_hot, noun_count):
    if multi_hot.ndim != 2 or noun_ids.shape != multi_hot.shape[:1]:
        raise ValueError('One noun and a multi-hot action vector per pair required')
    if not ((multi_hot == 0) | (multi_hot == 1)).all():
        raise ValueError('Targets must be multi-hot, not mutually exclusive')
    noun = F.one_hot(noun_ids, noun_count).to(multi_hot.dtype)
    return noun[:, :, None, None] * torch.stack((multi_hot, 1 - multi_hot), -1)[:, None]


def initialize(noun_prior, action_count):
    if noun_prior.ndim != 2 or not torch.isfinite(noun_prior).all() or (noun_prior < 0).any():
        raise ValueError('Finite nonnegative noun distribution required')
    torch.testing.assert_close(noun_prior.sum(-1), torch.ones_like(noun_prior[:, 0]), atol=1e-5, rtol=1e-5)
    return noun_prior[:, :, None, None].expand(-1, -1, action_count, 2) / 2


def multinomial_frequency(image, trials):
    """Independent draws per action slice, preserving its entire 2H simplex."""
    check_image(image)
    if int(trials) != trials or trials < 1:
        raise ValueError('Multinomial trials must be a positive integer')
    b, h, w, _ = image.shape
    probability = image.permute(0, 2, 1, 3).reshape(b, w, h * 2)
    sample = torch.distributions.Multinomial(total_count=int(trials), probs=probability).sample() / trials
    return sample.reshape(b, w, h, 2).permute(0, 2, 1, 3)


class ForwardDiffusion(nn.Module):
    def __init__(self, steps=50, trials=2000):
        super().__init__()
        if steps < 1 or trials < 1:
            raise ValueError('Positive steps and trials required')
        self.steps, self.trials = steps, trials
        beta = torch.cat((torch.zeros(1, dtype=torch.float64), torch.linspace(.001, .2, steps, dtype=torch.float64)))
        alpha_bar = torch.cumprod(1 - beta, 0)
        eta = torch.zeros_like(beta)
        for k in range(1, steps + 1):
            eta[k] = (1 - beta[k])**2 * eta[k - 1] + beta[k]**2
        scale = torch.zeros_like(beta)
        scale[1:] = (1 - alpha_bar[1:])**2 / eta[1:]
        self.register_buffer('beta', beta)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('effective_trials', (scale * trials).round().long().clamp_min(1))

    def step(self, previous, prior, k):
        if not 1 <= k <= self.steps or previous.shape != prior.shape:
            raise ValueError('Invalid forward step or incompatible prior')
        check_image(previous)
        beta = self.beta[k].to(previous)
        return (1-beta)*previous + beta*multinomial_frequency(prior, self.trials)

    def sample_approximate(self, clean, prior, k):
        """Eq. 5 moment-matched approximation, NOT an exact Eq. 4 trajectory."""
        if not 0 <= k <= self.steps or clean.shape != prior.shape:
            raise ValueError('Invalid diffusion step')
        check_image(clean)
        check_image(prior)
        if k == 0:
            return clean.clone()
        alpha = self.alpha_bar[k].to(clean)
        noise = multinomial_frequency(prior, int(self.effective_trials[k]))
        return alpha * clean + (1-alpha) * noise


def normalized_multinomial_log_prob(frequency, prior, trials):
    """Supplement Eq. 2 on its integer-count support; never round invalid data.

    Used for auditing posterior likelihoods. A density on a candidate previous
    HOI image is not itself a vector of next-image pixel probabilities.
    """
    check_image(prior)
    if frequency.shape != prior.shape or trials < 1 or int(trials) != trials:
        raise ValueError('Mismatched multinomial arguments')
    counts = frequency.double() * trials
    valid = (torch.isfinite(counts) & (counts >= 0) & ((counts - counts.round()).abs() < 1e-4)).all((1, 3))
    valid &= (counts.sum((1, 3)) - trials).abs() < 1e-4
    safe_counts = torch.where(torch.isfinite(counts), counts, 0).clamp_min(0).round()
    log_prob = torch.lgamma(safe_counts.new_tensor(trials+1)) - torch.lgamma(safe_counts+1).sum((1, 3))
    log_prob += torch.special.xlogy(safe_counts, prior.double()).sum((1, 3))
    return torch.where(valid, log_prob, -torch.inf)


def decode(images, entity_ids):
    """Paper object pooling across pairs, then presence/absence comparison."""
    check_image(images)
    if entity_ids.shape != images.shape[:1]:
        raise ValueError('Entity identity required per pair')
    _, inverse = torch.unique(entity_ids, return_inverse=True)
    pooled = images.new_zeros((int(inverse.max())+1, images.shape[1])) if len(images) else images.new_zeros((0, images.shape[1]))
    pooled.index_add_(0, inverse, images.sum((2, 3)))
    # Mean and sum have identical argmax within an entity group.
    nouns = pooled.argmax(-1)[inverse] if len(images) else entity_ids.clone()
    joint = images[torch.arange(len(images), device=images.device), nouns]
    present = joint[..., 0] > joint[..., 1]
    return nouns, present, joint
