"""Batched learned energy and exact DP, with event-local role coordinates.

Mathematically the same conditional distribution; floating-point parity is
checked against the per-event implementation.
"""
from math import sqrt

import torch

from corisp_heir.batched_dp import partition
from corisp_heir.support import state_indices, target_mass
from . import execution


def potentials(field, visible, states, contexts, active_sets, support):
    if not active_sets or len({len(a) for a in active_sets}) != 1:
        raise ValueError('A bucket needs equal nonzero role dimensions')
    v = visible.float()
    null = 1-v.sum(-1)
    if bool((v < 0).any() | (null < -1e-6).any()):
        raise ValueError('Invalid visible role subprobability')
    weights = field._log_probability(torch.cat((null.clamp_min(0)[..., None], v), -1))
    permitted = torch.cat((torch.ones_like(support[..., :1]), support), -1)
    local = torch.tensor([[0]+[r+1 for r in active] for active in active_sets], device=v.device)
    weights = weights.masked_fill(~permitted, -torch.inf).gather(
        -1, local[:, None, :].expand(-1, v.shape[1], -1))
    cardinality = field.cardinality_head(field.cardinality_norm(states.float()))
    cardinality = (cardinality-cardinality[:, :1])[:, :v.shape[1]+1]
    digits = torch.stack([state_indices(tuple(active), field.cfg.num_roles, str(v.device))[0]
                          for active in active_sets])
    if getattr(field, 'heir_no_arity', False):
        return weights, cardinality[..., None].expand(-1, -1, digits.shape[1])
    role_atom = field.arity_role_projection(contexts.float())
    count_atom = field.arity_count_embeddings.float()[digits]
    # Inactive roles still contribute count-zero energy, exactly as in reference.
    bound = field.arity_binding_norm(role_atom[:, None]+count_atom+role_atom[:, None]*count_atom)
    pooled = bound.sum(2)/sqrt(float(field.cfg.num_roles))
    key = field.arity_set_projection(pooled)
    query = field.arity_event_projection(states.float())
    raw = torch.einsum('bd,bsd->bs', query, key)/sqrt(float(field.cfg.arity_rank))
    raw = raw-raw[:, :1]
    scale = field.cfg.max_arity_energy*torch.tanh(field.arity_log_scale.float())
    return weights, cardinality[..., None]+(scale*torch.tanh(raw))[:, None]


def losses(field, visible, states, contexts, desc, role_sets, support, alpha, gamma):
    active_sets = [role_sets[d[1]] for d in desc]
    weights, energy = potentials(field, visible, states, contexts, active_sets, support)
    z, coefficients = partition(field, weights, energy)
    # Empty target numerators are closed form; denominators remain the full DP.
    masses = list((weights[..., 0].sum(-1)+energy[:, 0, 0]).unbind())
    for j, (human, action, indices, groups) in enumerate(desc):
        if groups:
            masses[j] = target_mass(field, human, action, indices, groups, active_sets[j],
                                   weights[j], energy[j], coefficients[j], z[j])
            if masses[j] is None:
                raise ValueError('Unrepresentable target reached the likelihood')
    factors = z.new_tensor([alpha if d[3] else 1-alpha for d in desc])
    return factors*field.focal_transform_nll((z-torch.stack(masses)).clamp_min(0), gamma)


def terms(model, visible, states, context, support, descriptions):
    field, role_sets = model.role_arity_event_field, model.heir_active_roles
    ordered = {}
    counts = dict(events=0, recurrence_batches=0, role_count_states=0)
    for desc in execution.chunks(descriptions, role_sets, getattr(model, 'heir_bucket_size', 32)):
        vv = torch.stack([visible[d[2], d[1]] for d in desc])
        ee = torch.stack([states[d[0], d[1]] for d in desc])
        cc = torch.stack([context[d[0], d[1]] for d in desc])
        ss = torch.stack([support[d[2], d[1]] for d in desc])
        loss = losses(field, vv, ee, cc, desc, role_sets, ss, model.alpha, model.gamma)
        ordered.update({d[1]: value for d, value in zip(desc, loss.unbind())})
        counts['events'] += len(desc)
        counts['recurrence_batches'] += 1
        counts['role_count_states'] += len(desc)*3**len(role_sets[desc[0][1]])
    return ordered, counts


def forward(model, images, targets=None):
    return execution.forward(model, images, targets, term_runner=terms)
