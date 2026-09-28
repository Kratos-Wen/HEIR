"""Batched evaluation of the unchanged exact role-arity recurrence.

Events in a batch have identical candidate and role counts. No probability,
energy, label, or candidate is pruned. The role reduction order is retained.
"""
from functools import lru_cache
import os

import torch


def _safe_add(a, b):
    unreachable = torch.isneginf(a) & torch.isneginf(b)
    return torch.logaddexp(a.masked_fill(unreachable, 0), b.masked_fill(unreachable, 0)).masked_fill(unreachable, -torch.inf)


def _step(coeff, weight, predecessor, reachable, saturated):
    stay = coeff + weight[:, 0, None, None]
    previous = coeff[:, :-1]
    source = previous[:, :, predecessor].permute(2, 0, 1, 3)
    source = torch.where(saturated[:, None, None], _safe_add(source, previous[None]), source)
    source = source.masked_fill(~reachable[:, None, None], -torch.inf)
    value = source+weight[:, 1:].T[:, :, None, None]
    unreachable = torch.isneginf(value).all(dim=0)
    take = torch.logsumexp(value.masked_fill(unreachable[None], 0), dim=0).masked_fill(unreachable, -torch.inf)
    padded = torch.cat((coeff.new_full((coeff.shape[0], 1, coeff.shape[2]), -torch.inf), take), dim=1)
    return _safe_add(stay, padded)


@lru_cache(maxsize=1)
def compiled_step():
    # Variable candidate counts, empty events and forward/backward grad modes
    # exceed Dynamo's small interactive default even with symbolic dimensions.
    torch._dynamo.config.cache_size_limit = 64
    return torch.compile(_step, fullgraph=True, dynamic=True)


@lru_cache(maxsize=32)
def topology(roles, device):
    code = torch.arange(3**roles, device=device)
    powers = 3**torch.arange(roles, device=device)
    digits = (code[None] // powers[:, None]) % 3
    predecessor = torch.where(digits > 0, code[None]-powers[:, None], 0)
    return predecessor, digits > 0, digits == 2


def coefficients(field, weights):
    if weights.ndim != 3 or weights.dtype not in (torch.float32, torch.float64):
        raise ValueError('Exact DP expects FP32/FP64 [events,candidates,1+roles]')
    batch, candidates, channels = weights.shape
    roles = channels-1
    states = 3**roles
    predecessor, reachable, saturated = topology(roles, weights.device)
    coeff = weights.new_full((batch, candidates+1, states), -torch.inf)
    coeff[:, 0, 0] = 0
    if os.environ.get('HEIR_CORISP_COMPILE_DP') == '1':
        step = compiled_step()
        for candidate in range(candidates):
            coeff = step(coeff, weights[:, candidate], predecessor, reachable, saturated)
        return coeff
    for candidate in range(candidates):
        stay = coeff + weights[:, candidate, 0, None, None]
        previous = coeff[:, :-1]
        # [role,event,count,state] keeps the frozen role summation order.
        source = previous[:, :, predecessor].permute(2, 0, 1, 3)
        source = torch.where(saturated[:, None, None],
            field._safe_logaddexp(source, previous[None]), source)
        source = source.masked_fill(~reachable[:, None, None], -torch.inf)
        take = field._safe_logsumexp(source + weights[:, candidate, 1:].T[:, :, None, None], dim=0)
        padded = torch.cat((coeff.new_full((batch, 1, states), -torch.inf), take), dim=1)
        coeff = field._safe_logaddexp(stay, padded)
    return coeff


def partition(field, weights, energy):
    backend = os.environ.get('HEIR_CORISP_DP_BACKEND', 'autograd')
    if backend == 'adjoint':
        from .adjoint_dp import partition as exact_adjoint
        return exact_adjoint(weights, energy)
    if backend != 'autograd':
        raise ValueError(f'Unknown HEIR DP backend: {backend}')
    coeff = coefficients(field, weights)
    if coeff.shape != energy.shape:
        raise ValueError('Composition energy shape mismatch')
    return torch.logsumexp((coeff+energy).flatten(1), dim=1), coeff
