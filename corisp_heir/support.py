"""HEIR-only conditioning on frozen support, without pruning valid states."""
from functools import lru_cache
from math import sqrt

import torch

from heir_protocol.compatibility import Compatibility
from . import environment
from corisp import RoleFillerTargetGroup
from corisp.role_arity_event_field import RoleArityEvent


@lru_cache(maxsize=128)
def state_indices(active, roles, device):
    code = torch.arange(3**len(active), device=device)
    digits = torch.zeros(len(code), roles, device=device, dtype=torch.long)
    for j, r in enumerate(active):
        digits[:, r] = (code//(3**j)) % 3
    full_codes = (digits * (3**torch.arange(roles, device=device))[None]).sum(-1)
    return digits, full_codes


def composition(field, state, context, candidates, active, no_arity=False):
    # Inactive roles retain their count-zero contribution and the original
    # sqrt(R) normalization: this is a submatrix, not a new energy function.
    cardinality = field.cardinality_head(field.cardinality_norm(state.float()))
    cardinality = (cardinality-cardinality[:1])[:candidates+1]
    digits, _ = state_indices(tuple(active), field.cfg.num_roles, state.device)
    if no_arity:
        return cardinality[:, None].expand(-1, len(digits))
    role_atom = field.arity_role_projection(context.float())
    count_atom = field.arity_count_embeddings.float()[digits]
    bound = field.arity_binding_norm(role_atom[None]+count_atom+role_atom[None]*count_atom)
    pooled = bound.sum(1)/sqrt(float(field.cfg.num_roles))
    key = field.arity_set_projection(pooled)
    query = field.arity_event_projection(state.float())
    raw = torch.einsum('d,sd->s', query, key)/sqrt(float(field.cfg.arity_rank))
    raw = raw-raw[:1]
    scale = field.cfg.max_arity_energy*torch.tanh(field.arity_log_scale.float())
    return cardinality[:, None]+(scale*torch.tanh(raw))[None]


def target_mass(field, human, action, indices, groups, active, weights, energy, coefficients, z):
    # Frozen target_log_mass indexes probability columns by role+1. Remap
    # both the target and event locally; cache/export keeps global role IDs.
    local = {r:j for j,r in enumerate(active)}
    mapped = [RoleFillerTargetGroup(local[int(g.role_index)], g.candidate_rows) for g in groups]
    roles = torch.arange(len(active), device=weights.device)
    event = RoleArityEvent(human, action, indices, roles[:0], roles,
                          weights, energy, coefficients, z)
    return field.target_log_mass(event, mapped)


def configure(model, document, vocabulary_sha, axes, ablation='full'):
    if ablation not in ('full', 'no_arity', 'no_relations', 'no_role_feedback'):
        raise ValueError('Unknown controlled ablation')
    model.heir_ablation = ablation
    model.role_arity_event_field.heir_no_arity = 'no_arity' in ablation
    model.role_field.disable_role_feedback = ablation == 'no_role_feedback'
    model.heir_compatibility_document = document
    model.heir_active_roles = None
    if document is None:
        return
    table = Compatibility(document, vocabulary_sha)
    mask, active = aligned_support(table, axes)
    model.register_buffer('heir_support_mask', mask)
    model.heir_active_roles = active
    if any(not roles for roles in model.heir_active_roles):
        raise ValueError('Every observed verb must admit at least one role')


def aligned_support(table, axes):
    """Map published IDs to detector/model indices, including person-first nouns."""
    if set(axes) != set(table.axes) or any(
            len(axes[k]) != len(set(axes[k])) or set(axes[k]) != set(table.axes[k]) for k in axes):
        raise ValueError('Support vocabulary differs from model')
    mask = torch.tensor([[[table.allows(v,n,r) for r in axes['roles']]
                          for v in axes['verbs']] for n in axes['nouns']], dtype=torch.bool)
    active = [tuple(i for i, yes in enumerate(row) if yes) for row in mask.any(0).tolist()]
    return mask, active
