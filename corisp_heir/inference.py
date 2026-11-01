"""HEIR inference: exact relation marginals and saved potentials for native set decoding."""
import torch

from .exact_dp import partition
from .support import composition

EVENT_BATCH = 16


def potentials(field, visible, states, contexts, active, support):
    with torch.autocast(device_type=visible.device.type, enabled=False):
        return _potentials(field, visible.float(), states.float(), contexts.float(), active, support)


def _potentials(field, visible, states, contexts, active, support):
    weights, energy, invalid = [], [], []
    for j, (v, state, context) in enumerate(zip(visible, states, contexts)):
        # Project individual events before batching their exact recurrences.
        null = 1-v.sum(-1)
        invalid.append((v < 0).any() | (null < -1e-6).any())
        full_weights = field._log_probability(torch.cat((null.clamp_min(0)[:, None], v), -1))
        permitted = torch.cat((torch.ones_like(support[j, :, :1]), support[j]), -1)
        full_weights = full_weights.masked_fill(~permitted, -torch.inf)
        weights.append(full_weights[:, [0]+[r+1 for r in active]])
        energy.append(composition(field, state, context, len(v), active, getattr(field, 'heir_no_arity', False)))
    if bool(torch.stack(invalid).any()):
        raise ValueError('Invalid visible role subprobability')
    return torch.stack(weights), torch.stack(energy)


def forward(model, images):
    sizes, proposals, boxes, pairs, _, packets = model._extract_packets(images)
    outputs = model._predict_role_packets(images, packets)
    priors = model._role_class_priors(proposals, pairs)
    field = model.grounded_role_set
    role_sets = model.heir_active_roles
    predictions = []
    for i, (output, proposal, paired, packet, prior) in enumerate(zip(outputs, proposals, pairs, packets, priors)):
        visible, _, context, values = model._refined_inputs(output, proposal, prior, packet)
        pair_support = model.heir_support_mask[proposal['labels'][paired[:, 1].long()]]
        if not values['valid_role_mask'].all():
            raise ValueError('HEIR batch recurrence requires its full role vocabulary')
        agents = output['agent_entity_indices'][0].long()
        states = output['event_states'][0]
        joint = visible.new_zeros(visible.shape)
        saved = []
        # All predicates share the same candidate set in HEIR. Group only by
        # agent to preserve the event order and avoid padded candidate changes.
        for human in range(len(agents)):
            shared_indices = None
            rows = torch.where(values['pair_agent_indices'] == human)[0]
            if bool(values['visible_action_mask'][rows].all()):
                shared_indices = rows
            batches = {}
            for action in range(model.role_space.num_actions):
                indices = (shared_indices if shared_indices is not None else
                    torch.where((values['pair_agent_indices'] == human) & values['visible_action_mask'][:, action])[0])
                batches.setdefault((role_sets[action], len(indices)), []).append((human, action, indices))
            chunks = [(active_roles, group[start:start+EVENT_BATCH])
                      for (active_roles, _), group in batches.items()
                      for start in range(0, len(group), EVENT_BATCH)]
            for active_roles, desc in chunks:
                vv = torch.stack([visible[d[2], d[1]] for d in desc])
                ee = torch.stack([states[d[0], d[1]] for d in desc])
                cc = torch.stack([context[d[0], d[1]] for d in desc])
                ss = torch.stack([pair_support[d[2], d[1]] for d in desc])
                weights, energy = potentials(field, vv, ee, cc, active_roles, ss)
                with torch.enable_grad():
                    leaf = weights.detach().requires_grad_(True)
                    z, _ = partition(leaf, energy.detach())
                    marginal = (torch.autograd.grad(z.sum(), leaf)[0].detach().clamp(0, 1)
                                if leaf.shape[1] else torch.zeros_like(leaf))
                active = torch.tensor(active_roles, device=visible.device)
                for j, (h, a, indices) in enumerate(desc):
                    expanded = joint.new_zeros(len(indices), field.cfg.num_roles)
                    expanded[:, active] = marginal[j, :, 1:]
                    joint[indices, a] = expanded
                    saved.append({'agent_index': h, 'action_index': a, 'pair_indices': indices.detach().cpu(),
                        'log_state_weights': weights[j].detach().cpu(), 'composition_energy': energy[j].detach().cpu(),
                        'active_role_indices': active.detach().cpu(), 'log_partition': z[j].detach().cpu()})
        predictions.append({'boxes': boxes[i], 'labels': proposal['labels'], 'entity_scores': proposal['scores'],
            'pairs': paired, 'joint_scores': joint, 'scores': joint.sum(-1), 'size': sizes[i],
            'agent_entities': agents, 'events': saved, 'entity_identity': 'image_local_detector_proposal_index'})
    return predictions
