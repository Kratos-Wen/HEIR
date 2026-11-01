"""CoRISP with frozen visual inputs and HEIR observation scopes."""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .checkpoint import sha256
from .environment import WEIGHTS
from . import inference, packed
from .proposals import align_detector_query_states
from .support import configure
from h_detr.models import build_model as build_detector
from hdetr_corisp import HDETRCoRISP
from frozen_dinotxt import FrozenDINOtxtVisualBackbone
from corisp import (EventSetField, EventSetFieldConfig, InteractionFieldConfig, ProposalPairAdapter,
    ProposalPairAdapterConfig, RoleConditionedInteractionField, VCOCORoleSpace)


DINOTXT_SHA256 = 'a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0'


class HEIRCoRISP(HDETRCoRISP):
    def _proposal_states(self, detector_output, hidden_states):
        return align_detector_query_states(detector_output['pred_logits'], hidden_states, self.postprocessor.topk)[-1]

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
        disable_relations = self.heir_ablation == 'no_relations'
        visible, typed, context = self.grounded_role_set._aggregate_role_context(**values, disable_relations=disable_relations)
        return visible, typed, context, values

    def forward(self, images, targets=None):
        if not self.training:
            return inference.forward(self, images)
        if targets is None:
            raise ValueError('Training requires HEIR observed event targets')
        if len(images) != 1:
            raise ValueError('Training expects one image per rank')
        return packed.forward(self, images, targets)


def build(data, detector_path, prototypes_path, compatibility, ablation='full'):
    state = torch.load(detector_path, map_location='cpu', mmap=True, weights_only=False)
    protocol = state['protocol']
    if protocol['backbone'] != 'swinl' or protocol['class_order'] != data.nouns:
        raise ValueError('Expected completed HEIR H-DETR/Swin-L in the exact noun order')
    if any(Path(p).name == 'test.json' for p in protocol['inputs']):
        raise ValueError('Detector declares test supervision')
    if [h for p, h in protocol['inputs'].items() if Path(p).name == 'vocabulary.json'] != [sha256(data.root/'vocabulary.json')]:
        raise ValueError('Detector vocabulary does not match this dataset')
    cfg = SimpleNamespace(**protocol['config'])
    cfg.device = 'cpu'
    cfg.topk = 100  # Detector proposals before person/entity selection.
    detector, _, postprocessors = build_detector(cfg)
    detector.class_embed = nn.ModuleList([nn.Linear(h.in_features, len(data.nouns)) for h in detector.class_embed])
    detector.transformer.decoder.class_embed = detector.class_embed
    detector.load_state_dict(state['model'], strict=True)
    prototypes = torch.load(prototypes_path, weights_only=True, map_location='cpu')
    if not {'verbs', 'nouns', 'roles', 'vocabulary_sha256', 'action_prototypes', 'object_prototypes', 'role_prototypes'} <= set(prototypes):
        raise ValueError('Wrong semantic prototype bank')
    for key in ('verbs', 'nouns', 'roles'):
        if prototypes[key] != getattr(data, key):
            raise ValueError(f'Prototype order mismatch: {key}')
    if prototypes['vocabulary_sha256'] != sha256(data.root/'vocabulary.json'):
        raise ValueError('Prototype ontology changed')
    if prototypes.get('encoder_sha256') != DINOTXT_SHA256:
        raise ValueError('Prototype bank is not from the DINO.txt encoder')
    for key, count in [('action', len(data.verbs)), ('object', len(data.nouns)), ('role', len(data.roles))]:
        value = prototypes[key+'_prototypes']
        if tuple(value.shape) != (count, 2048) or not torch.isfinite(value).all():
            raise ValueError('Invalid semantic prototype dimensions/values')
        torch.testing.assert_close(value.float().norm(dim=-1), torch.ones(count), atol=1e-4, rtol=1e-4)
    # Reuse the lossless action-role indexing class, without importing native
    # V-COCO slot eligibility or a train-derived noun/action prohibition.
    space = VCOCORoleSpace.from_role_classes([f'{v} {r}' for v in data.verbs for r in data.roles], role_names=data.roles)
    d = 384
    adapter = ProposalPairAdapter(ProposalPairAdapterConfig(detector_dim=cfg.hidden_dim,
        dense_dim=detector.backbone.num_channels[-1], d_model=d, human_label=0, max_pairs=1024, include_human_human=True))
    weights = WEIGHTS
    semantic = FrozenDINOtxtVisualBackbone(backbone_weights=weights/'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        dinotxt_weights=weights/'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth', input_size=448)
    role_field = RoleConditionedInteractionField(InteractionFieldConfig(d_model=d, visual_dim=1024, semantic_dim=2048,
        num_actions=len(data.verbs), roles=tuple(data.roles), num_object_classes=len(data.nouns), inference_steps=2,
        ffn_dim=4*d, dropout=.1, enable_typed_null_fillers=False),
        action_prototypes=prototypes['action_prototypes'], object_prototypes=prototypes['object_prototypes'],
        role_prototypes=prototypes['role_prototypes'], valid_role_mask=space.valid_role_mask(),
        object_action_mask=torch.ones(len(data.nouns), len(data.verbs), dtype=torch.bool), human_label=0)
    field = EventSetField(EventSetFieldConfig(d_model=d, num_roles=len(data.roles),
        max_cardinality=29+len(data.roles), num_heads=8, arity_rank=64, dropout=.1))
    mapping = [list(range(space.num_role_classes)) for _ in data.nouns]
    model = HEIRCoRISP(detector, postprocessors['bbox'], adapter, role_space=space, object_to_role_class=mapping,
        role_field=role_field, semantic_backbone=semantic, grounded_role_set=field, human_idx=0,
        box_score_thresh=.05, min_instances=3, max_instances=15, alpha=.5, gamma=.1)
    model.freeze_detector()
    configure(model, compatibility, sha256(data.root/'vocabulary.json'),
              {key:getattr(data,key) for key in ('verbs','nouns','roles')}, ablation)
    return model
