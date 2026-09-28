"""Batch equal-sized exact recurrences without requiring identical role names.

Each event keeps its global role IDs for learned energies and target mapping.
Only the compact recurrence uses interchangeable local indices. Learned
projections remain individual calls; no candidate or probability mass is pruned.
"""
import os

import torch
import torch.distributed as dist

from corisp_heir import environment, batched_execution
from corisp_heir.batched_dp import partition
from corisp_heir.model import representable
from corisp_heir.support import composition, target_mass
from corisp_heir.targets import EventTargets


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


def losses(field, visible, states, contexts, descriptions, role_sets, support,
           alpha, gamma):
    active_sets = [role_sets[d[1]] for d in descriptions]
    if len({len(active) for active in active_sets}) != 1:
        raise ValueError('Mixed state dimensions in a compact batch')
    weights, energies, invalid = [], [], []
    for j, active in enumerate(active_sets):
        # Global role semantics remain in the reference learned-energy path.
        v = visible[j].float()
        null = 1-v.sum(-1)
        invalid.append((v < 0).any() | (null < -1e-6).any())
        w = field._log_probability(torch.cat((null.clamp_min(0)[:, None], v), -1))
        permitted = torch.cat((torch.ones_like(support[j, :, :1]), support[j]), -1)
        weights.append(w.masked_fill(~permitted, -torch.inf)[:, [0]+[r+1 for r in active]])
        energies.append(composition(field, states[j], contexts[j], len(v), active,
                                    getattr(field, 'heir_no_arity', False)))
    if bool(torch.stack(invalid).any()):
        raise ValueError('Invalid visible role subprobability')
    weights, energies = torch.stack(weights), torch.stack(energies)
    z, coefficients = partition(field, weights, energies)
    terms = []
    for j, (human, action, indices, groups) in enumerate(descriptions):
        mass = target_mass(field, human, action, indices, groups, active_sets[j],
                           weights[j], energies[j], coefficients[j], z[j])
        if mass is None:
            raise ValueError('Unrepresentable target reached the likelihood')
        terms.append((alpha if groups else 1-alpha) * field.focal_transform_nll(
            (z[j]-mass).clamp_min(0), gamma))
    return torch.stack(terms)


def forward(model, images, targets=None, *, term_runner=None, whole_image=False):
    if not model.training or len(images) != 1 or model.heir_active_roles is None:
        return batched_execution.forward(model, images, targets)
    if targets is None:
        raise ValueError('Training requires reviewed HEIR targets')
    _, proposals, _, pairs, _, _, packets = model._extract_packets(images)
    outputs = model._predict_role_packets(images, packets)
    priors = model._role_class_priors(proposals, pairs)
    field = model.role_arity_event_field
    cached = os.environ.get('HEIR_CORISP_METADATA_CACHE') == '1'
    output, proposal, paired = outputs[0], proposals[0], pairs[0]
    visible, _, context, values = model._refined_inputs(output, proposal, priors[0], packets[0])
    role_sets = model.heir_active_roles
    support = model.heir_support_mask[proposal['labels'][paired[:, 1].long()]]
    if not values['valid_role_mask'].all():
        raise ValueError('HEIR recurrence requires its full role vocabulary')
    agents = output['agent_entity_indices'][0].long()
    states = output['event_states'][0]
    constraints = EventTargets(proposal, agents, paired, targets[0], cached=cached)
    matched_people = set(constraints.assignment.values())
    missing_agents = sum(int((targets[0]['action_observed'][j] == 1).sum())
        for j in range(len(targets[0]['person_ids'])) if j not in matched_people)
    plans = []
    count = ignored = unmatched = 0
    for human in range(len(agents)):
        descriptions, shared_indices = [], None
        if cached:
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
        plans.append(descriptions)
    reference = field.cardinality_head.weight
    denominator = reference.new_tensor(float(count))
    work = dist.all_reduce(denominator, async_op=True) if dist.is_initialized() else None
    all_losses = []
    counters = dict(events=0, recurrence_batches=0, role_count_states=0)
    if whole_image:
        if term_runner is None:
            raise ValueError('Whole-image buckets require identity-keyed event terms')
        plans = [[d for plan in plans for d in plan]]
    for descriptions in plans:
        if term_runner is not None:
            ordered, local_counts = term_runner(model, visible, states, context, support, descriptions)
            all_losses.extend(ordered[(d[0], d[1]) if whole_image else d[1]] for d in descriptions)
            for key in counters:
                counters[key] += local_counts[key]
            continue
        ordered = {}
        for desc in chunks(descriptions, role_sets, getattr(model, 'heir_bucket_size', 32)):
            vv = torch.stack([visible[d[2], d[1]] for d in desc])
            ee = torch.stack([states[d[0], d[1]] for d in desc])
            cc = torch.stack([context[d[0], d[1]] for d in desc])
            ss = torch.stack([support[d[2], d[1]] for d in desc])
            term = losses(field, vv, ee, cc, desc, role_sets, ss, model.alpha, model.gamma)
            ordered.update({d[1]: t for d, t in zip(desc, term.unbind())})
            counters['events'] += len(desc)
            counters['recurrence_batches'] += 1
            counters['role_count_states'] += len(desc)*3**len(role_sets[desc[0][1]])
        all_losses.extend(ordered[d[1]] for d in descriptions)
    model.exact_execution_counters = counters
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
