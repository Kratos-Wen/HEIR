"""Disclosed closed-loop sampled-transition repair, not posterior-density MSE.

Retain the paper's Eq.4, M=10 trajectories and K=50 reverse steps. Supervise
each step on the model's own preceding prediction, not a teacher-forced input.
One-step truncated backpropagation is explicit; there is no full-chain BPTT.
"""

import torch

from .diffusion import check_image
from .vcoco import regression_loss


@torch.no_grad()
def trajectory(clean, prior, process):
    check_image(clean)
    check_image(prior)
    if clean.shape != prior.shape:
        raise ValueError('Clean/prior shape mismatch')
    states = [clean]
    for k in range(1, process.steps + 1):
        states.append(process.step(states[-1], prior, k))
    return states


def backward_rollout(model, states, prior, appearance, known, global_count,
                     world_size=1, samples_per_pair=10):
    """Accumulate the exact mean of local step losses; DDP averages ranks.

    One of each M starts uses the deterministic inference prior; the others
    use real sampled endpoints. This boundary coverage is an independent
    reconstruction choice. Targets remain sampled previous states for k>1.
    Empty-label ranks participate with graph-connected zero gradients.
    """
    if samples_per_pair < 1 or len(prior) % samples_per_pair:
        raise ValueError('Require complete groups of M sampled trajectories')
    count = known.sum()
    if global_count < count or world_size < 1:
        raise ValueError('Invalid globally normalized supervision count')
    steps = len(states) - 1
    if steps < 1:
        raise ValueError('Missing forward trajectory')
    current = states[-1].detach().clone()
    current[::samples_per_pair] = prior[::samples_per_pair]
    total = current.new_zeros(())
    for k in range(steps, 0, -1):
        prediction = model(current, torch.full((len(current),), k,
                           device=current.device, dtype=torch.long), appearance)
        summed, _ = regression_loss(prediction, states[k-1], known)
        loss = summed * world_size / global_count.clamp_min(1) / steps
        loss.backward()
        total += summed.detach() / steps
        current = prediction.detach()
    return total, current
