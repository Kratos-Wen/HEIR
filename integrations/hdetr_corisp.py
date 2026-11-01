"""Frozen H-DETR proposals and DINO.txt features feeding the CoRISP fields."""
from __future__ import annotations

from typing import List, Sequence

import torch

import torch.distributed as dist

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor

from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list

from ops import compute_prior_scores, prepare_region_proposals

from corisp import EventSetField, ProposalPairAdapter, RoleConditionedInteractionField, VCOCORoleSpace

from frozen_dinotxt import FrozenDINOtxtVisualBackbone


def _inverse_sigmoid(value: Tensor, eps: float = 1e-5) -> Tensor:
    value = value.clamp(min=0.0, max=1.0)
    return torch.log(value.clamp(min=eps) / (1.0 - value).clamp(min=eps))

def _normalize_proposal_precision(
    proposals: list[dict[str, Tensor]],
) -> list[dict[str, Tensor]]:
    """Keep proposal geometry/confidence compatible with FP32 prior operators."""

    return [
        {
            **proposal,
            "boxes": proposal["boxes"].float(),
            "scores": proposal["scores"].float(),
        }
        for proposal in proposals
    ]

def _hdetr_forward_with_tokens(
    detector: nn.Module,
    samples: List[Tensor] | Tensor | NestedTensor,
) -> tuple[dict[str, Tensor], Tensor, list[NestedTensor]]:
    """Run the official H-DETR path while retaining proposal/backbone tokens."""

    if not isinstance(samples, NestedTensor):
        samples = nested_tensor_from_tensor_list(samples)
    features, positional = detector.backbone(samples)

    sources: list[Tensor] = []
    masks: list[Tensor] = []
    for level, feature in enumerate(features):
        source, mask = feature.decompose()
        if mask is None:
            raise RuntimeError("H-DETR returned a feature level without a padding mask.")
        sources.append(detector.input_proj[level](source))
        masks.append(mask)
    if detector.num_feature_levels > len(sources):
        initial_levels = len(sources)
        for level in range(initial_levels, detector.num_feature_levels):
            if level == initial_levels:
                source = detector.input_proj[level](features[-1].tensors)
            else:
                source = detector.input_proj[level](sources[-1])
            mask = F.interpolate(
                samples.mask[None].float(), size=source.shape[-2:]
            ).to(torch.bool)[0]
            pos_level = detector.backbone[1](NestedTensor(source, mask)).to(source.dtype)
            sources.append(source)
            masks.append(mask)
            positional.append(pos_level)

    query_embeddings = None
    if not detector.two_stage or detector.mixed_selection:
        query_embeddings = detector.query_embed.weight[: detector.num_queries]

    attention_mask = torch.zeros(
        detector.num_queries,
        detector.num_queries,
        dtype=torch.bool,
        device=sources[0].device,
    )
    split = detector.num_queries_one2one
    attention_mask[split:, :split] = True
    attention_mask[:split, split:] = True

    (
        hidden_states,
        initial_reference,
        intermediate_references,
        encoder_class,
        encoder_boxes_unactivated,
    ) = detector.transformer(
        sources,
        masks,
        positional,
        query_embeddings,
        attention_mask,
    )

    classes_one2one: list[Tensor] = []
    boxes_one2one: list[Tensor] = []
    classes_one2many: list[Tensor] = []
    boxes_one2many: list[Tensor] = []
    for level in range(hidden_states.shape[0]):
        reference = initial_reference if level == 0 else intermediate_references[level - 1]
        reference = _inverse_sigmoid(reference)
        class_logits = detector.class_embed[level](hidden_states[level])
        box_delta = detector.bbox_embed[level](hidden_states[level])
        if reference.shape[-1] == 4:
            box_delta = box_delta + reference
        else:
            box_delta = box_delta.clone()
            box_delta[..., :2] = box_delta[..., :2] + reference
        boxes = box_delta.sigmoid()
        classes_one2one.append(class_logits[:, :split])
        boxes_one2one.append(boxes[:, :split])
        classes_one2many.append(class_logits[:, split:])
        boxes_one2many.append(boxes[:, split:])

    output = {
        "pred_logits": classes_one2one[-1],
        "pred_boxes": boxes_one2one[-1],
        "pred_logits_one2many": classes_one2many[-1],
        "pred_boxes_one2many": boxes_one2many[-1],
    }
    if detector.two_stage:
        output["enc_outputs"] = {
            "pred_logits": encoder_class,
            "pred_boxes": encoder_boxes_unactivated.sigmoid(),
        }
    return output, hidden_states, features

class HDETRCoRISP(nn.Module):
    """Frozen proposals and DINO.txt features with the role-conditioned field and set field."""

    def __init__(
        self,
        detector: nn.Module,
        postprocessor: nn.Module,
        adapter: ProposalPairAdapter,
        *,
        role_space: VCOCORoleSpace,
        object_to_role_class: Sequence[Sequence[int]],
        role_field: RoleConditionedInteractionField,
        semantic_backbone: FrozenDINOtxtVisualBackbone,
        grounded_role_set: EventSetField,
        human_idx: int = 0,
        box_score_thresh: float = 0.05,
        min_instances: int = 3,
        max_instances: int = 15,
        alpha: float = 0.5,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.adapter = adapter
        self.human_idx = int(human_idx)
        self.box_score_thresh = float(box_score_thresh)
        self.min_instances = int(min_instances)
        self.max_instances = int(max_instances)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.role_space = role_space
        self.object_to_role_class = [list(value) for value in object_to_role_class]
        # Two names for one module: released state dictionaries contain both prefixes.
        self.event_field = role_field
        self.semantic_backbone = semantic_backbone
        self.role_field = role_field
        self.grounded_role_set = grounded_role_set

    def freeze_detector(self) -> None:
        for module in (self.detector, self.semantic_backbone):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "HDETRCoRISP":
        super().train(mode)
        self.detector.eval()
        self.semantic_backbone.eval()
        return self

    def _proposal_states(
        self,
        detector_output: dict[str, Tensor],
        hidden_states: Tensor,
    ) -> Tensor:
        return hidden_states[-1]

    def _extract_packets(
        self,
        images: List[Tensor],
    ) -> tuple[
        Tensor,
        list[dict[str, Tensor]],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list,
    ]:
        image_sizes = torch.as_tensor(
            [image.shape[-2:] for image in images], device=images[0].device
        )
        with torch.no_grad():
            detector_output, hidden_states, features = _hdetr_forward_with_tokens(
                self.detector, images
            )
            states = self._proposal_states(detector_output, hidden_states)
            detections = self.postprocessor(detector_output, image_sizes)
            proposals = prepare_region_proposals(
                detections,
                states,
                image_sizes,
                box_score_thresh=self.box_score_thresh,
                human_idx=self.human_idx,
                min_instances=self.min_instances,
                max_instances=self.max_instances,
            )
            # PViC's prior operator stores detector confidence in FP32. Under an
            # outer BF16 autocast H-DETR emits BF16 scores, and indexed assignment
            # into that tensor is illegal, so normalize once before any use;
            # detector hidden states remain autocast-enabled.
            proposals = _normalize_proposal_precision(proposals)

        packets = self.adapter(
            proposals,
            features[-1].tensors,
            image_sizes,
            dense_valid_masks=~features[-1].mask,
            source="frozen_h_detr_swin_l",
        )
        boxes = [proposal["boxes"] for proposal in proposals]
        pair_indices = [
            torch.stack([packet.subject_indices[0], packet.object_indices[0]], dim=-1)
            for packet in packets
        ]
        object_types = [
            proposal["labels"][pairs[:, 1]]
            for proposal, pairs in zip(proposals, pair_indices)
        ]
        return image_sizes, proposals, boxes, pair_indices, object_types, packets

    def _role_class_priors(
        self,
        proposals: list[dict[str, Tensor]],
        paired_indices: list[Tensor],
    ) -> list[Tensor]:
        priors: list[Tensor] = []
        for proposal, pairs in zip(proposals, paired_indices):
            priors.append(
                compute_prior_scores(
                    pairs[:, 0],
                    pairs[:, 1],
                    proposal["scores"],
                    proposal["labels"],
                    self.role_space.num_role_classes,
                    self.training,
                    self.object_to_role_class,
                )
            )
        return priors

    def _expand_action_roles(self, value: Tensor) -> Tensor:
        return torch.stack(
            [
                value[..., action_index, role_index]
                for action_index, role_index in zip(
                    self.role_space.role_class_to_action,
                    self.role_space.role_class_to_role,
                )
            ],
            dim=-1,
        )

    @staticmethod
    def _distributed_normalizer(value: Tensor) -> Tensor:
        normalizer = value.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(normalizer)
            normalizer = normalizer / dist.get_world_size()
        return normalizer.clamp_min(1.0)

    def _predict_role_packets(
        self,
        images: List[Tensor],
        packets: list,
    ) -> list[dict[str, Tensor]]:
        outputs: list[dict[str, Tensor]] = []
        for image, packet in zip(images, packets):
            agent_queries = self.role_field.build_agent_queries(packet)
            if agent_queries.shape[1]:
                aligned_agents, visual_grid, global_semantic = (
                    self.semantic_backbone.forward_with_event_queries(
                        [image], agent_queries
                    )
                )
            else:
                visual_grid, global_semantic = self.semantic_backbone([image])
                aligned_agents = agent_queries
            outputs.append(
                self.role_field(
                    packet,
                    visual_grid=visual_grid,
                    global_semantic=global_semantic,
                    aligned_agent_tokens=aligned_agents,
                )
            )
        return outputs

    def _detector_composed_unaries(
        self,
        output: dict[str, Tensor],
        proposal: dict[str, Tensor],
        role_class_prior: Tensor,
    ) -> tuple[Tensor, Tensor]:
        class_prior = role_class_prior.prod(dim=1)
        role_prior = class_prior.new_zeros(
            (
                class_prior.shape[0],
                self.role_space.num_actions,
                self.role_space.num_roles,
            )
        )
        for class_index, (action_index, role_index) in enumerate(
            zip(
                self.role_space.role_class_to_action,
                self.role_space.role_class_to_role,
            )
        ):
            role_prior[:, action_index, role_index] = class_prior[:, class_index]

        visible = output["joint_role_probs"].squeeze(0) * role_prior
        agent_entities = output["agent_entity_indices"].squeeze(0).long()
        typed = output["null_role_probs"].squeeze(0)[agent_entities]
        typed = typed * proposal["scores"][agent_entities, None, None]
        return visible, typed
