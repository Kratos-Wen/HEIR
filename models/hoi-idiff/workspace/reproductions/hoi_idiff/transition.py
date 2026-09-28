"""User-approved sampled-transition regression, NOT Eq.6 density regression."""

import torch

from .diffusion import check_image


@torch.no_grad()
def exact_transition(clean, prior, steps, process):
    check_image(clean)
    check_image(prior)
    if clean.shape != prior.shape or steps.shape != (len(clean),):
        raise ValueError('Invalid transition request')
    if not ((steps >= 1) & (steps <= process.steps)).all():
        raise ValueError('Timestep out of range')
    b, h, w, _ = clean.shape
    distribution = torch.distributions.Multinomial(total_count=process.trials,
        probs=prior.permute(0, 2, 1, 3).reshape(b, w, 2*h), validate_args=False)
    previous, current = torch.zeros_like(clean), torch.zeros_like(clean)
    state = clean.clone()
    for k in range(1, int(steps.max()) + 1):
        take = (steps == k)[:, None, None, None]
        previous = torch.where(take, state, previous)
        noise = (distribution.sample() / process.trials).reshape(b, w, h, 2).permute(0, 2, 1, 3)
        beta = process.beta[k].to(state)
        state = (1 - beta) * state + beta * noise
        current = torch.where(take, state, current)
    check_image(previous)
    check_image(current)
    return current, previous
