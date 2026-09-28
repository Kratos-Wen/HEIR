"""CoRISP with frozen visual inputs and HEIR observation scopes."""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from scipy.optimize import linear_sum_assignment

from .environment import ROOT, FROZEN, WEIGHTS, verify_core
from .assets import HEIR_PROTOTYPES_SHA256, require_asset
from .targets import EventTargets
from .proposals import extract
from heir_training.data import sha256
from h_detr.models import build_model as build_detector
from hdetr_corisp_role_arity_event_field_vcoco import HDETRCoRISPRoleArityEventFieldVCOCO
from hdetr_corisp_v6_vcoco import _JointStateClassifierContract
from dinotxt_event_adapter import FrozenDINOtxtVisualBackbone
from corisp import (PreparedProposalAdapterConfig, PreparedProposalPairAdapter,
    CoRISPAgentRoleField, CoRISPAgentRoleFieldConfig, RoleArityEventField,
    RoleArityEventFieldConfig, VCOCORoleSpace)
from corisp.role_arity_event_field import RoleArityEventFieldOutput


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


def event_loss(field, visible, event_state, context, indices, valid, groups,
               agent=0, action=0, recompute=True, alpha=.5, gamma=.1):
    """Complete-set likelihood with optional activation recomputation."""
    def likelihood(v, e, c):
        event, _ = field._build_arity_event(agent_index=agent, action_index=action,
            pair_indices=indices, visible_probability=v,
            typed_probability=v.new_zeros(field.cfg.num_roles), valid_roles=valid,
            typed_null_roles=torch.zeros_like(valid), event_state=e, event_role_context=c)
        mass = field.target_log_mass(event, groups)
        if mass is None:
            raise ValueError('Unrepresentable set passed to exact likelihood')
        nll = (event.log_partition-mass).clamp_min(0)
        return (alpha if groups else 1-alpha)*field.focal_transform_nll(nll, gamma)
    if recompute:
        return checkpoint(likelihood, visible, event_state, context,
                          use_reentrant=False, preserve_rng_state=True)
    return likelihood(visible, event_state, context)


def event_marginals(field, event, humans, actions, shape):
    if event.num_candidates == 0:
        # The only configuration is the empty set, with probability one.
        # There is no leaf whose derivative needs to be evaluated.
        return event.log_state_weights.new_zeros(shape)
    singleton = RoleArityEventFieldOutput(events=(event,), num_agents=humans,
        num_actions=actions, visible_shape=shape)
    return field._with_arity_marginals(singleton).visible_role_marginals


class HEIRCoRISP(HDETRCoRISPRoleArityEventFieldVCOCO):
    def _extract_packets(self, images):
        return extract(self, images)

    def _refined_inputs(self, output, proposal, prior, packet):
        visible, typed = self._detector_composed_unaries(output, proposal, prior)
        dtype, device = output['edge_states'].dtype, output['edge_states'].device
        field = self.role_field
        actions = field.semantic_projection(field.action_prototypes.to(device=device, dtype=dtype))
        roles = field.semantic_projection(field.role_prototypes.to(device=device, dtype=dtype)) + field.role_offset.to(dtype=dtype)
        entity_indices = packet.object_indices[0].long()
        nouns = field.semantic_projection(field.object_prototypes[packet.entity_labels[0, entity_indices]].to(dtype=dtype))
        agents = output['agent_entity_indices'][0].long()
        values = dict(base_visible_role_probs=visible, base_typed_null_role_probs=typed,
            edge_states=output['edge_states'][0], null_edge_states=output['null_edge_states'][0, agents],
            event_states=output['event_states'][0], pair_agent_indices=output['pair_agent_indices'][0].long(),
            pair_entity_indices=entity_indices, noun_anchors=nouns, action_anchors=actions, role_anchors=roles,
            visible_action_mask=output['visible_action_mask'][0].bool(),
            valid_role_mask=self.role_space.valid_role_mask().to(device))
        intervention = ('no_all_relations' if 'no_relations' in getattr(self, 'heir_ablation', 'full') else 'full')
        visible, typed, context, _ = self.role_arity_event_field._relational_unaries(**values, intervention=intervention)
        return visible, typed, context, values

    def forward(self, images, targets=None):
        if getattr(self, 'heir_dp_batch', 0):
            from .batched_execution import forward
            return forward(self, images, targets)
        if self.training and targets is None:
            raise ValueError('Training requires HEIR observed event targets')
        sizes, proposals, boxes, pairs, _, _, packets = self._extract_packets(images)
        outputs = self._predict_role_packets(images, packets)
        priors = self._role_class_priors(proposals, pairs)
        field = self.role_arity_event_field
        losses, predictions, count = [], [], 0
        ignored = 0
        unmatched = 0
        missing_entities = missing_agents = 0
        for i, (output, proposal, paired, packet, prior) in enumerate(zip(outputs, proposals, pairs, packets, priors)):
            visible, typed, context, values = self._refined_inputs(output, proposal, prior, packet)
            agents = output['agent_entity_indices'][0].long()
            events = output['event_states'][0]
            constraints = EventTargets(proposal, agents, paired, targets[i]) if self.training else None
            if self.training:
                matched_people = set(constraints.assignment.values())
                missing_agents += sum(int((targets[i]['action_observed'][j] == 1).sum())
                    for j in range(len(targets[i]['person_ids'])) if j not in matched_people)
            joint = visible.new_zeros(visible.shape)
            saved_events = []
            for human in range(len(agents)):
                for action in range(self.role_space.num_actions):
                    indices = torch.where((values['pair_agent_indices'] == human) & values['visible_action_mask'][:, action])[0]
                    groups = constraints.groups(human, action, indices) if self.training else None
                    if self.training and groups is None:
                        ignored += 1
                        continue
                    if self.training and not representable(groups, len(indices)):
                        unmatched += 1
                        continue
                    valid = values['valid_role_mask'][action]

                    def build_event(v, e, c, human=human, action=action, indices=indices, valid=valid):
                        return field._build_arity_event(agent_index=human, action_index=action,
                            pair_indices=indices, visible_probability=v,
                            typed_probability=v.new_zeros(field.cfg.num_roles), valid_roles=valid,
                            typed_null_roles=torch.zeros_like(valid), event_state=e, event_role_context=c)[0]

                    if self.training:
                        loss = event_loss(field, visible[indices, action], events[human, action],
                            context[human, action], indices, valid, groups, human, action,
                            alpha=self.alpha, gamma=self.gamma)
                        losses.append(loss)
                        count += len(groups)
                    else:
                        event = build_event(visible[indices, action], events[human, action], context[human, action])
                        marginal = event_marginals(field, event, len(agents), self.role_space.num_actions, tuple(visible.shape))
                        joint[indices, action] = marginal[indices, action]
                        # Raw potentials permit exact native set decoding later,
                        # with stable shared proposal IDs, without rerunning a model.
                        saved_events.append({'agent_index': human, 'action_index': action,
                            'pair_indices': indices.detach().cpu(), 'log_state_weights': event.log_state_weights.detach().cpu(),
                            'composition_energy': event.composition_energy.detach().cpu(),
                            'active_role_indices': event.active_role_indices.detach().cpu(),
                            'log_partition': event.log_partition.detach().cpu()})
            if not self.training:
                predictions.append({'boxes': boxes[i], 'labels': proposal['labels'], 'entity_scores': proposal['scores'],
                    'pairs': paired, 'joint_scores': joint, 'scores': joint.sum(-1), 'size': sizes[i],
                    'agent_entities': agents, 'events': saved_events,
                    'entity_identity': 'image_local_detector_proposal_index'})
            else:
                missing_entities += constraints.missing_entity_events
        if not self.training:
            return predictions
        reference = field.cardinality_head.weight
        denominator = reference.new_tensor(float(count))
        if dist.is_initialized():
            dist.all_reduce(denominator)
            denominator /= dist.get_world_size()
        denominator = denominator.clamp_min(1)
        # Keep the same DDP graph even for batches with no observable pair.
        zero = sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        total = torch.stack(losses).sum()/denominator if losses else zero
        return {'loss': total+zero, 'matched_edges': reference.new_tensor(count),
                'ignored_events': reference.new_tensor(ignored),
                'unrepresentable_events': reference.new_tensor(unmatched),
                'missing_entity_events': reference.new_tensor(missing_entities),
                'missing_agent_events': reference.new_tensor(missing_agents)}


def build(data, detector_path, prototypes_path, compatibility=None, ablation='full'):
    verify_core()
    require_asset(prototypes_path, HEIR_PROTOTYPES_SHA256)
    import inspect
    if not Path(inspect.getfile(RoleArityEventField)).is_relative_to(FROZEN):
        raise ValueError('The imported model module is outside this package')
    state = torch.load(detector_path, map_location='cpu', mmap=True, weights_only=False)
    protocol = state['protocol']
    if protocol['schema'] != 'heir_detector_adaptation_v2' or protocol['backbone'] != 'swinl' or protocol['class_order'] != data.nouns:
        raise ValueError('Expected completed HEIR H-DETR/Swin-L in the exact noun order')
    if any(Path(p).name == 'test.json' for p in protocol['inputs']):
        raise ValueError('Detector declares test supervision')
    for filename in ('train.json', 'vocabulary.json'):
        path = data.root/'annotations/train.json' if filename == 'train.json' else data.root/filename
        recorded = [h for p, h in protocol['inputs'].items() if Path(p).name == filename]
        if recorded != [sha256(path)]:
            raise ValueError('Detector release does not match HOI release')
    cfg = SimpleNamespace(**protocol['config'])
    cfg.device = 'cpu'
    cfg.topk = 100  # Detector proposals before person/entity selection.
    detector, _, postprocessors = build_detector(cfg)
    detector.class_embed = nn.ModuleList([nn.Linear(h.in_features, len(data.nouns)) for h in detector.class_embed])
    detector.transformer.decoder.class_embed = detector.class_embed
    detector.load_state_dict(state['model'], strict=True)
    prototypes = torch.load(prototypes_path, weights_only=True, map_location='cpu')
    if prototypes['schema'] != 'corisp_heir_frozen_dinotxt_prototypes_v1':
        raise ValueError('Wrong semantic prototype bank')
    for key in ('verbs', 'nouns', 'roles'):
        if prototypes[key] != getattr(data, key):
            raise ValueError(f'Prototype order mismatch: {key}')
    if prototypes['vocabulary_sha256'] != sha256(data.root/'vocabulary.json'):
        raise ValueError('Prototype ontology changed')
    if prototypes.get('encoder_sha256') != 'a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0':
        raise ValueError('Prototype bank is not from the frozen DINO.txt encoder')
    for key, count in [('action', len(data.verbs)), ('object', len(data.nouns)), ('role', len(data.roles))]:
        value = prototypes[key+'_prototypes']
        if tuple(value.shape) != (count, 2048) or not torch.isfinite(value).all():
            raise ValueError('Invalid semantic prototype dimensions/values')
        torch.testing.assert_close(value.float().norm(dim=-1), torch.ones(count), atol=1e-4, rtol=1e-4)
    # Reuse the lossless action-role indexing class, without importing native
    # V-COCO slot eligibility or a train-derived noun/action prohibition.
    space = VCOCORoleSpace.from_role_classes([f'{v} {r}' for v in data.verbs for r in data.roles], role_names=data.roles)
    d = 384
    adapter = PreparedProposalPairAdapter(PreparedProposalAdapterConfig(detector_dim=cfg.hidden_dim, pair_dim=d,
        dense_dim=detector.backbone.num_channels[-1], d_model=d, human_label=0, max_pairs=1024, include_human_human=True))
    weights = WEIGHTS
    semantic = FrozenDINOtxtVisualBackbone(backbone_weights=weights/'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        dinotxt_weights=weights/'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth', input_size=448)
    role_field = CoRISPAgentRoleField(CoRISPAgentRoleFieldConfig(d_model=d, visual_dim=1024, semantic_dim=2048,
        num_actions=len(data.verbs), roles=tuple(data.roles), num_object_classes=len(data.nouns), inference_steps=2,
        ffn_dim=4*d, dropout=.1, enable_typed_null_fillers=False),
        action_prototypes=prototypes['action_prototypes'], object_prototypes=prototypes['object_prototypes'],
        role_prototypes=prototypes['role_prototypes'], valid_role_mask=space.valid_role_mask(),
        object_action_mask=torch.ones(len(data.nouns), len(data.verbs), dtype=torch.bool), human_label=0)
    field = RoleArityEventField(RoleArityEventFieldConfig(d_model=d, num_roles=len(data.roles),
        max_cardinality=29+len(data.roles), num_heads=8, arity_rank=64, dropout=.1))
    mapping = [list(range(space.num_role_classes)) for _ in data.nouns]
    model = HEIRCoRISP(detector=detector, postprocessor=postprocessors['bbox'], adapter=adapter,
        classifier=_JointStateClassifierContract(d, space.num_role_classes), object_to_target=mapping,
        strong=None, variant='corisp_heir', human_idx=0, box_score_thresh=.05,
        min_instances=3, max_instances=15, raw_lambda=1., alpha=.5, gamma=.1, participation_loss_weight=0.,
        supervision_mode='vcoco_roles', role_space=space, object_to_role_class=mapping, role_field=role_field,
        semantic_backbone=semantic, grounded_role_set=field, null_role_loss_weight=1.)
    model.freeze_detector()
    # HEIR targets replace the inherited native-slot observation model.
    model.supervision_mode = 'heir_observed_role_sets'
    model.heir_dp_batch = 16
    from .support import configure
    configure(model, compatibility, sha256(data.root/'vocabulary.json'),
              {key:getattr(data,key) for key in ('verbs','nouns','roles')}, ablation)
    return model
