"""Exact batched role-arity DP with a memory-bounded analytic first derivative.

The state is unchanged: exact selected count and a 0/1/2+ digit per role.
At candidate t only counts 0..t are allocated. The reverse recurrence gathers
successor adjoints, avoiding scatter atomics and the generic autograd tape.
No role, candidate or nonzero-probability configuration is removed.
"""
from functools import lru_cache
import os

import torch
from torch.autograd.function import once_differentiable

from .batched_dp import topology, _safe_add


def _forward_step(previous, weight, predecessor, reachable, saturated):
    source = previous[:, :, predecessor].permute(2, 0, 1, 3)
    source = torch.where(saturated[:, None, None],
                         _safe_add(source, previous[None]), source)
    value = source.masked_fill(~reachable[:, None, None], -torch.inf)
    value = value + weight[:, 1:].T[:, :, None, None]
    empty = torch.isneginf(value).all(dim=0)
    take = torch.logsumexp(value.masked_fill(empty[None], 0), dim=0)
    take = take.masked_fill(empty, -torch.inf)
    padding = previous.new_full((previous.shape[0], 1, previous.shape[2]), -torch.inf)
    stay = torch.cat((previous + weight[:, 0, None, None], padding), dim=1)
    return _safe_add(stay, torch.cat((padding, take), dim=1))


def _reverse_step(previous, current, adjoint, weight, successor):
    # Every predecessor has one successor per categorical choice, including
    # saturated 2+ transitions. This is the transpose of the forward DAG.
    stay_log = previous + weight[:, 0, None, None]
    stay_out = current[:, :-1]
    valid = torch.isfinite(stay_log) & torch.isfinite(stay_out)
    stay_prob = (stay_log-stay_out).masked_fill(~valid, 0).exp().masked_fill(~valid, 0)
    stay = adjoint[:, :-1] * stay_prob
    take_log = previous[None] + weight[:, 1:].T[:, :, None, None]
    take_out = current[:, 1:, successor].permute(2, 0, 1, 3)
    valid = torch.isfinite(take_log) & torch.isfinite(take_out)
    take_prob = (take_log-take_out).masked_fill(~valid, 0).exp().masked_fill(~valid, 0)
    take = adjoint[:, 1:, successor].permute(2, 0, 1, 3) * take_prob
    grad_previous = stay + take.sum(dim=0)
    grad_weights = torch.cat((stay.sum(dim=(1, 2))[:, None],
                              take.sum(dim=(2, 3)).T), dim=1)
    return grad_previous, grad_weights


@lru_cache(maxsize=32)
def successors(roles, device):
    code = torch.arange(3**roles, device=device)
    power = 3**torch.arange(roles, device=device)
    digits = (code[None] // power[:, None]) % 3
    return code[None] + (digits < 2)*power[:, None]


@lru_cache(maxsize=1)
def compiled_steps():
    torch._dynamo.config.cache_size_limit = 64
    return (torch.compile(_forward_step, fullgraph=True, dynamic=True),
            torch.compile(_reverse_step, fullgraph=True, dynamic=True))


class _Partition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, energy, compiled):
        batch, candidates, channels = weights.shape
        roles, states = channels-1, 3**(channels-1)
        if energy.shape != (batch, candidates+1, states):
            raise ValueError('Composition energy shape mismatch')
        if weights.dtype not in (torch.float32, torch.float64):
            raise ValueError('Exact adjoint DP requires FP32/FP64 weights')
        if energy.dtype not in (weights.dtype, torch.bfloat16, torch.float16):
            raise ValueError('Composition energy must match weights or use autocast precision')
        # The frozen autocast head may return BF16 energy. Its addition to
        # FP32 coefficients promotes in forward and casts back at this boundary.
        ctx.energy_dtype = energy.dtype
        energy = energy.to(weights.dtype)
        predecessor, reachable, saturated = topology(roles, weights.device)
        step = compiled_steps()[0] if compiled else _forward_step
        current = weights.new_full((batch, 1, states), -torch.inf)
        current[:, 0, 0] = 0
        history = [current]
        for index in range(candidates):
            current = step(current, weights[:, index], predecessor, reachable, saturated)
            history.append(current)
        log_z = torch.logsumexp((current+energy).flatten(1), dim=1)
        ctx.save_for_backward(weights, energy, log_z, *history)
        ctx.compiled = compiled
        ctx.set_materialize_grads(False)
        return log_z, current

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_z, grad_coefficients):
        weights, energy, log_z, *history = ctx.saved_tensors
        current = history[-1]
        adjoint = torch.zeros_like(current)
        grad_energy = torch.zeros_like(energy)
        if grad_z is not None:
            grad_energy = (current+energy-log_z[:, None, None]).exp()*grad_z[:, None, None]
            adjoint = adjoint + grad_energy
        if grad_coefficients is not None:
            adjoint = adjoint + grad_coefficients.masked_fill(~torch.isfinite(current), 0)
        grad_weights = torch.zeros_like(weights)
        successor = successors(weights.shape[-1]-1, weights.device)
        step = compiled_steps()[1] if ctx.compiled else _reverse_step
        for index in range(weights.shape[1]-1, -1, -1):
            adjoint, grad_weights[:, index] = step(
                history[index], history[index+1], adjoint, weights[:, index], successor)
        return grad_weights, grad_energy.to(ctx.energy_dtype), None


def partition(weights, energy, *, compiled=None):
    if weights.ndim != 3 or weights.shape[-1] < 2:
        raise ValueError('Expected [events,candidates,1+roles]')
    if compiled is None:
        compiled = weights.is_cuda and os.environ.get('HEIR_CORISP_COMPILE_DP') == '1'
    return _Partition.apply(weights, energy, compiled)
