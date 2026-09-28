from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import torch.nn as nn

from torch import Tensor

from ops import associate_with_ground_truth, binary_focal_loss_with_logits

from dinotxt_event_adapter import FrozenDINOtxtVisualBackbone

from hdetr_corisp_v4_vcoco import HDETRCoRISPEventVCOCO

from corisp import CoRISPAgentRoleField, VCOCORoleSpace

class _JointStateClassifierContract(nn.Module):
    """Retain the host API while forbidding an independent pair head."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    def forward(self, value: Tensor) -> Tensor:
        raise RuntimeError("Predictions must come from the joint role state.")

class HDETRCoRISPAgentRoleVCOCO(HDETRCoRISPEventVCOCO):
    """Frozen proposals and DINO.txt features with a real-person role field."""

    def __init__(
        self,
        *args,
        role_field: CoRISPAgentRoleField,
        role_space: VCOCORoleSpace,
        object_to_role_class: Sequence[Sequence[int]],
        semantic_backbone: FrozenDINOtxtVisualBackbone,
        null_role_loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            role_space=role_space,
            object_to_role_class=object_to_role_class,
            event_field=role_field,
            semantic_backbone=semantic_backbone,
            null_role_loss_weight=null_role_loss_weight,
            image_action_loss_weight=0.0,
            sufficiency_loss_weight=0.0,
            necessity_loss_weight=0.0,
            event_cardinality_weight=0.0,
            event_duplicate_weight=0.0,
            evidence_compactness_weight=0.0,
            **kwargs,
        )
        self.role_field = role_field
        if self.raw_lambda != 1.0:
            raise ValueError("Detector-score composition uses exponent 1.0.")

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

    def _action_priors(self, role_class_priors: list[Tensor]) -> list[Tensor]:
        action_priors: list[Tensor] = []
        for role_prior in role_class_priors:
            legal = role_prior.prod(dim=1)
            action = legal.new_zeros((legal.shape[0], self.role_space.num_actions))
            for class_index, action_index in enumerate(
                self.role_space.role_class_to_action
            ):
                action[:, action_index] = torch.maximum(
                    action[:, action_index], legal[:, class_index]
                )
            action_priors.append(action)
        return action_priors

    def _joint_visible_loss(
        self,
        outputs: list[dict[str, Tensor]],
        role_class_priors: list[Tensor],
        role_class_labels: Tensor,
        pair_counts: list[int],
    ) -> Tensor:
        action_labels, acceptable_roles = self.role_space.collapse_pair_labels(
            role_class_labels
        )
        logits = torch.cat(
            [output["hoi_logits"].squeeze(0) for output in outputs], dim=0
        )
        conditional = torch.cat(
            [
                output["conditional_role_probs"].squeeze(0)
                for output in outputs
            ],
            dim=0,
        )
        action_prior = torch.cat(self._action_priors(role_class_priors), dim=0)
        pair_index, action_index = torch.nonzero(action_prior, as_tuple=True)
        normalizer = self._distributed_normalizer(action_labels.sum())
        if pair_index.numel():
            selected_logits = logits[pair_index, action_index]
            selected_prior = action_prior[pair_index, action_index]
            calibrated = torch.log(
                selected_prior
                / (1.0 + torch.exp(-selected_logits) - selected_prior)
                + 1e-8
            )
            action_loss = binary_focal_loss_with_logits(
                calibrated,
                action_labels[pair_index, action_index],
                reduction="sum",
                alpha=self.alpha,
                gamma=self.gamma,
            ) / normalizer
        else:
            action_loss = logits.sum() * 0.0

        positive = action_labels.bool()
        accepted_probability = (conditional * acceptable_roles).sum(dim=-1)
        if positive.any():
            role_loss = -accepted_probability[positive].clamp_min(1e-8).log().sum()
            role_loss = role_loss / normalizer
        else:
            role_loss = conditional.sum() * 0.0
        if sum(pair_counts) != logits.shape[0]:
            raise RuntimeError("Visible joint-state labels do not align with pair packets.")
        return action_loss + role_loss

    def _typed_null_loss(
        self,
        outputs: list[dict[str, Tensor]],
        proposals: list[dict[str, Tensor]],
        targets: list[dict],
    ) -> Tensor:
        # V-COCO null roles are multilabel slots: one agent-action can have
        # both its object and instrument absent.  The inherited role-class
        # focal loss supervises each typed slot independently and therefore
        # does not collapse simultaneous null fillers into alternatives.
        return self._null_role_loss(
            outputs, proposals, targets, key="null_role_logits"
        )

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None,
    ) -> list[dict[str, Tensor]] | dict[str, Tensor]:
        if self.training and targets is None:
            raise ValueError("V-COCO role-field training requires targets.")
        (
            image_sizes,
            proposals,
            boxes,
            pair_indices,
            _,
            object_types,
            packets,
        ) = self._extract_packets(images)
        outputs = self._predict_role_packets(images, packets)
        role_class_priors = self._role_class_priors(proposals, pair_indices)

        if self.training:
            assert targets is not None
            role_class_labels = associate_with_ground_truth(
                boxes,
                pair_indices,
                targets,
                self.role_space.num_role_classes,
            )
            return {
                "joint_state_loss": self._joint_visible_loss(
                    outputs,
                    role_class_priors,
                    role_class_labels,
                    [len(value) for value in pair_indices],
                ),
                "null_state_loss": self.null_role_loss_weight
                * self._typed_null_loss(outputs, proposals, targets),
            }

        return self._postprocess_event_field(
            boxes,
            proposals,
            pair_indices,
            object_types,
            outputs,
            role_class_priors,
            image_sizes,
        )
