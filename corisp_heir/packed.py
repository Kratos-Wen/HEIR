"""HEIR training: image-wide exact event batches with no cross-person probability coupling.

Batch only events of equal candidate and role count. Each event keeps its global role IDs for learned
energies and target mapping; only the compact recurrence uses interchangeable local indices. No candidate
or probability mass is pruned. Structured potentials use FP32; visual and contextual features use BF16 autocast.
"""
from math import sqrt

import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment

from .exact_dp import partition
from .support import state_indices, target_mass
from .targets import EventTargets

EVENT_BATCH = 256


def representable(groups, candidates):
    if not groups:
        return True
    if len(groups) > candidates:
        return False
    valid = torch.zeros(len(groups), candidates, dtype=torch.bool)
    for i, group in enumerate(groups):
        valid[i, group.candidate_rows.cpu()] = True
    rows, cols = linear_sum_assignment(valid.numpy(), maximize=True)
    return len(rows) == len(groups) and bool(valid[rows, cols].all())


def chunks(descriptions, role_sets, limit):
    if limit < 1:
        raise ValueError('Event batch limit must be positive')
    batches = {}
    for desc in descriptions:
        active = role_sets[desc[1]]
        if not active or len(set(active)) != len(active):
            raise ValueError('Expected nonempty unique active role IDs')
        batches.setdefault((len(active), len(desc[2])), []).append(desc)
    return [group[start:start + limit] for group in batches.values()
            for start in range(0, len(group), limit)]


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
    z, coefficients = partition(weights, energy)
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
    field, role_sets = model.grounded_role_set, model.heir_active_roles
    ordered = {}
    with torch.autocast(visible.device.type, enabled=False):
        for desc in chunks(descriptions, role_sets, EVENT_BATCH):
            vv = torch.stack([visible[d[2], d[1]] for d in desc])
            ee = torch.stack([states[d[0], d[1]] for d in desc])
            cc = torch.stack([context[d[0], d[1]] for d in desc])
            ss = torch.stack([support[d[2], d[1]] for d in desc])
            loss = losses(field, vv, ee, cc, desc, role_sets, ss, model.alpha, model.gamma)
            ordered.update({(d[0], d[1]): value for d, value in zip(desc, loss.unbind())})
    return ordered


def forward(model, images, targets):
    _, proposals, _, pairs, _, packets = model._extract_packets(images)
    outputs = model._predict_role_packets(images, packets)
    priors = model._role_class_priors(proposals, pairs)
    field = model.grounded_role_set
    output, proposal, paired = outputs[0], proposals[0], pairs[0]
    visible, _, context, values = model._refined_inputs(output, proposal, priors[0], packets[0])
    support = model.heir_support_mask[proposal['labels'][paired[:, 1].long()]]
    if not values['valid_role_mask'].all():
        raise ValueError('HEIR recurrence requires its full role vocabulary')
    agents = output['agent_entity_indices'][0].long()
    states = output['event_states'][0]
    constraints = EventTargets(proposal, agents, paired, targets[0])
    matched_people = set(constraints.assignment.values())
    missing_agents = sum(int((targets[0]['action_observed'][j] == 1).sum())
        for j in range(len(targets[0]['person_ids'])) if j not in matched_people)
    descriptions = []
    count = ignored = unmatched = 0
    for human in range(len(agents)):
        shared_indices = None
        rows = torch.where(values['pair_agent_indices'] == human)[0]
        if bool(values['visible_action_mask'][rows].all()):
            shared_indices = rows
        for action in range(model.role_space.num_actions):
            indices = (shared_indices if shared_indices is not None else torch.where(
                (values['pair_agent_indices'] == human) & values['visible_action_mask'][:, action])[0])
            groups = constraints.groups(human, action, indices)
            if groups is None:
                ignored += 1
                continue
            if not representable(groups, len(indices)):
                unmatched += 1
                continue
            descriptions.append((human, action, indices, groups))
            count += len(groups)
    reference = field.cardinality_head.weight
    denominator = reference.new_tensor(float(count))
    work = dist.all_reduce(denominator, async_op=True) if dist.is_initialized() else None
    ordered = terms(model, visible, states, context, support, descriptions)
    all_losses = [ordered[(d[0], d[1])] for d in descriptions]
    if work is not None:
        work.wait()
        denominator /= dist.get_world_size()
    zero = sum(p.reshape(-1)[0]*0 for p in model.parameters() if p.requires_grad)
    total = torch.stack(all_losses).sum()/denominator.clamp_min(1) if all_losses else zero
    return {'loss': total+zero, 'matched_edges': reference.new_tensor(count),
        'ignored_events': reference.new_tensor(ignored),
        'unrepresentable_events': reference.new_tensor(unmatched),
        'missing_entity_events': reference.new_tensor(constraints.missing_entity_events),
        'missing_agent_events': reference.new_tensor(missing_agents)}
