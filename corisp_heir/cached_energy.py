"""Frozen per-event energy arithmetic with cached, non-learned role indices."""
from functools import lru_cache
from math import sqrt

import torch


@lru_cache(maxsize=32)
def indices(roles, device):
    active = torch.arange(roles, device=device)
    code = torch.arange(3**roles, device=device)
    digits = (code[:, None] // (3**active)[None]) % 3
    return active, digits


def composition(field, state, context, candidates):
    # Same operations and individual learned projection shapes as the frozen
    # _event_composition_energy. HEIR admits every role; no nonzero() is needed.
    active, digits = indices(field.cfg.num_roles, state.device)
    cardinality = field.cardinality_head(field.cardinality_norm(state.float()))
    cardinality = (cardinality-cardinality[:1])[:candidates+1]
    role_atom = field.arity_role_projection(context[active].float())
    count_atom = field.arity_count_embeddings.float()[digits]
    bound = field.arity_binding_norm(role_atom[None]+count_atom+role_atom[None]*count_atom)
    pooled = bound.sum(dim=1)/sqrt(float(field.cfg.num_roles))
    key = field.arity_set_projection(pooled)
    query = field.arity_event_projection(state.float())
    raw = torch.einsum('d,sd->s', query, key)/sqrt(float(field.cfg.arity_rank))
    raw = raw-raw[:1]
    scale = field.cfg.max_arity_energy*torch.tanh(field.arity_log_scale.float())
    return cardinality[:, None]+(scale*torch.tanh(raw))[None]
