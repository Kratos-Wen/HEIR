"""Memory-bounded execution of the HEIR event likelihood."""
import os
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from corisp.role_arity_event_field import RoleArityEvent
from .batched_dp import partition
from .targets import EventTargets


def potentials(field, visible, states, contexts, active=None, support=None):
    with torch.autocast(device_type=visible.device.type, enabled=False):
        return _potentials(field, visible.float(), states.float(), contexts.float(), active, support)


def _potentials(field, visible, states, contexts, active=None, support=None):
    weights, energy = [], []
    cached = os.environ.get('HEIR_CORISP_METADATA_CACHE') == '1'
    invalid = []
    valid = torch.ones(field.cfg.num_roles, dtype=torch.bool, device=visible.device)
    for j, (v, state, context) in enumerate(zip(visible, states, contexts)):
        # Project individual events before batching their exact recurrences.
        v = v.float()
        null = 1-v.sum(-1)
        if cached:
            invalid.append((v < 0).any() | (null < -1e-6).any())
        elif (v < 0).any() or (null < -1e-6).any():
            raise ValueError('Invalid visible role subprobability')
        full_weights = field._log_probability(torch.cat((null.clamp_min(0)[:, None], v), -1))
        if active is not None:
            permitted = torch.cat((torch.ones_like(support[j, :, :1]), support[j]), -1)
            full_weights = full_weights.masked_fill(~permitted, -torch.inf)
            weights.append(full_weights[:, [0]+[r+1 for r in active]])
            from .support import composition
            energy.append(composition(field, state, context, len(v), active,
                                      getattr(field, 'heir_no_arity', False)))
        elif getattr(field, 'heir_no_arity', False):
            weights.append(full_weights)
            from .support import composition
            energy.append(composition(field, state, context, len(v), tuple(range(field.cfg.num_roles)), True))
        elif cached:
            weights.append(full_weights)
            from .cached_energy import composition
            energy.append(composition(field, state, context, len(v)))
        else:
            weights.append(full_weights)
            energy.append(field._event_composition_energy(event_state=state,
                event_role_context=context, valid_roles=valid, num_candidates=len(v))[0])
    if cached and bool(torch.stack(invalid).any()):
        raise ValueError('Invalid visible role subprobability')
    return torch.stack(weights), torch.stack(energy)


def losses(field, visible, states, contexts, descriptions, alpha, gamma, recompute_potentials=False,
           active_roles=None, support=None):
    if recompute_potentials:
        weights, energy = checkpoint(lambda v, e, c: potentials(field, v, e, c, active_roles, support),
            visible, states, contexts, use_reentrant=False, preserve_rng_state=True)
    else:
        weights, energy = potentials(field, visible, states, contexts, active_roles, support)
    z, coefficients = partition(field, weights, energy)
    active = torch.arange(field.cfg.num_roles, device=visible.device)
    terms = []
    for j, (human, action, indices, groups) in enumerate(descriptions):
        event = RoleArityEvent(human, action, indices, active[:0], active,
                              weights[j], energy[j], coefficients[j], z[j])
        if active_roles is None:
            mass = field.target_log_mass(event, groups)
        else:
            from .support import target_mass
            mass = target_mass(field, human, action, indices, groups, active_roles,
                               weights[j], energy[j], coefficients[j], z[j])
        if mass is None:
            raise ValueError('Unrepresentable target reached the likelihood')
        terms.append((alpha if groups else 1-alpha)*field.focal_transform_nll((z[j]-mass).clamp_min(0), gamma))
    return torch.stack(terms)


def forward(model, images, targets):
    from .model import representable
    if model.training and targets is None:
        raise ValueError('Training requires reviewed HEIR targets')
    sizes, proposals, boxes, pairs, _, _, packets = model._extract_packets(images)
    outputs = model._predict_role_packets(images, packets)
    priors = model._role_class_priors(proposals, pairs)
    field = model.role_arity_event_field
    cached = os.environ.get('HEIR_CORISP_METADATA_CACHE') == '1'
    all_losses, predictions = [], []
    count = ignored = unmatched = missing_entities = missing_agents = 0
    for i, (output, proposal, paired, packet, prior) in enumerate(zip(outputs, proposals, pairs, packets, priors)):
        visible, _, context, values = model._refined_inputs(output, proposal, prior, packet)
        role_sets = getattr(model, 'heir_active_roles', None)
        pair_support = (model.heir_support_mask[proposal['labels'][paired[:, 1].long()]]
                        if role_sets is not None else None)
        if not values['valid_role_mask'].all():
            raise ValueError('HEIR batch recurrence requires its full role vocabulary')
        agents = output['agent_entity_indices'][0].long()
        states = output['event_states'][0]
        constraints = EventTargets(proposal, agents, paired, targets[i], cached=cached) if model.training else None
        if model.training:
            matched_people = set(constraints.assignment.values())
            missing_agents += sum(int((targets[i]['action_observed'][j] == 1).sum())
                for j in range(len(targets[i]['person_ids'])) if j not in matched_people)
        joint = visible.new_zeros(visible.shape)
        saved = []
        # All predicates share the same candidate set in HEIR. Group only by
        # agent to preserve the event order and avoid padded candidate changes.
        for human in range(len(agents)):
            descriptions = []
            shared_indices = None
            if cached:
                rows = torch.where(values['pair_agent_indices'] == human)[0]
                if bool(values['visible_action_mask'][rows].all()):
                    shared_indices = rows
            for action in range(model.role_space.num_actions):
                indices = (shared_indices if shared_indices is not None else
                    torch.where((values['pair_agent_indices'] == human) & values['visible_action_mask'][:, action])[0])
                groups = constraints.groups(human, action, indices) if model.training else None
                if model.training and groups is None:
                    ignored += 1
                    continue
                if model.training and not representable(groups, len(indices)):
                    unmatched += 1
                    continue
                descriptions.append((human, action, indices, groups))
                if model.training:
                    count += len(groups)
            batches = {}
            for d in descriptions:
                key = (role_sets[d[1]] if role_sets is not None else None, len(d[2]))
                batches.setdefault(key, []).append(d)
            chunks = [(active_roles, group[start:start+model.heir_dp_batch])
                      for (active_roles, _), group in batches.items()
                      for start in range(0, len(group), model.heir_dp_batch)]
            ordered_terms = {}
            for active_roles, desc in chunks:
                if len({len(d[2]) for d in desc}) != 1:
                    raise ValueError('Batch must have identical candidate count')
                vv = torch.stack([visible[d[2], d[1]] for d in desc])
                ee = torch.stack([states[d[0], d[1]] for d in desc])
                cc = torch.stack([context[d[0], d[1]] for d in desc])
                ss = (torch.stack([pair_support[d[2], d[1]] for d in desc])
                      if pair_support is not None else None)
                if model.training:
                    def objective(v, e, c, desc=desc, active_roles=active_roles, ss=ss):
                        return losses(field, v, e, c, desc, model.alpha, model.gamma,
                                      active_roles=active_roles, support=ss)
                    if os.environ.get('HEIR_CORISP_SELECTIVE_RECOMPUTE') == '1':
                        # Retain only the compact DP history. Recompute learned
                        # projections, not the exact recurrence, during backward.
                        term = losses(field, vv, ee, cc, desc, model.alpha, model.gamma,
                                      recompute_potentials=True, active_roles=active_roles, support=ss)
                    else:
                        term = checkpoint(objective, vv, ee, cc, use_reentrant=False, preserve_rng_state=True)
                    ordered_terms.update({d[1]:t for d,t in zip(desc,term.unbind())})
                else:
                    weights, energy = potentials(field, vv, ee, cc, active_roles, ss)
                    with torch.enable_grad():
                        leaf = weights.detach().requires_grad_(True)
                        z, _ = partition(field, leaf, energy.detach())
                        marginal = (torch.autograd.grad(z.sum(), leaf)[0].detach().clamp(0, 1)
                                    if leaf.shape[1] else torch.zeros_like(leaf))
                    active = (torch.tensor(active_roles, device=visible.device) if active_roles is not None
                              else torch.arange(field.cfg.num_roles, device=visible.device))
                    for j, (h, a, indices, _) in enumerate(desc):
                        expanded = joint.new_zeros(len(indices), field.cfg.num_roles)
                        expanded[:, active] = marginal[j, :, 1:]
                        joint[indices, a] = expanded
                        saved.append({'agent_index': h, 'action_index': a, 'pair_indices': indices.detach().cpu(),
                            'log_state_weights': weights[j].detach().cpu(), 'composition_energy': energy[j].detach().cpu(),
                            'active_role_indices': active.detach().cpu(), 'log_partition': z[j].detach().cpu()})
            if model.training:
                all_losses.extend(ordered_terms[d[1]] for d in descriptions)
        if model.training:
            missing_entities += constraints.missing_entity_events
        else:
            predictions.append({'boxes': boxes[i], 'labels': proposal['labels'], 'entity_scores': proposal['scores'],
                'pairs': paired, 'joint_scores': joint, 'scores': joint.sum(-1), 'size': sizes[i],
                'agent_entities': agents, 'events': saved, 'entity_identity': 'image_local_detector_proposal_index'})
    if not model.training:
        return predictions
    reference = field.cardinality_head.weight
    denominator = reference.new_tensor(float(count))
    if dist.is_initialized():
        dist.all_reduce(denominator)
        denominator /= dist.get_world_size()
    zero = sum(p.reshape(-1)[0]*0 for p in model.parameters() if p.requires_grad)
    total = torch.stack(all_losses).sum()/denominator.clamp_min(1) if all_losses else zero
    return {'loss': total+zero, 'matched_edges': reference.new_tensor(count),
        'ignored_events': reference.new_tensor(ignored), 'unrepresentable_events': reference.new_tensor(unmatched),
        'missing_entity_events': reference.new_tensor(missing_entities), 'missing_agent_events': reference.new_tensor(missing_agents)}
