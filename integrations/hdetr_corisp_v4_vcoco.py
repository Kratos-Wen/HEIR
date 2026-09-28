from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import torch.distributed as dist

from torch import Tensor

import torchvision.ops.boxes as box_ops

from ops import associate_with_ground_truth, binary_focal_loss_with_logits, compute_prior_scores, recover_boxes

from dinotxt_event_adapter import FrozenDINOtxtVisualBackbone

from hdetr_corisp_v3 import HDETRCoRISPv3

from corisp import CoRISPEventField, VCOCORoleSpace, event_field_regularization

class HDETRCoRISPEventVCOCO(HDETRCoRISPv3):
    """Frozen H-DETR proposals plus a DINO.txt-aligned latent event field."""

    def __init__(
        self,
        *args,
        role_space: VCOCORoleSpace,
        object_to_role_class: Sequence[Sequence[int]],
        event_field: CoRISPEventField,
        semantic_backbone: FrozenDINOtxtVisualBackbone,
        null_role_loss_weight: float = 1.0,
        image_action_loss_weight: float = 0.25,
        sufficiency_loss_weight: float = 0.20,
        necessity_loss_weight: float = 0.20,
        event_cardinality_weight: float = 0.01,
        event_duplicate_weight: float = 0.05,
        evidence_compactness_weight: float = 0.02,
        intervention_margin: float = 0.20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.role_space = role_space
        self.object_to_role_class = [list(value) for value in object_to_role_class]
        self.event_field = event_field
        self.semantic_backbone = semantic_backbone
        self.null_role_loss_weight = float(null_role_loss_weight)
        self.image_action_loss_weight = float(image_action_loss_weight)
        self.sufficiency_loss_weight = float(sufficiency_loss_weight)
        self.necessity_loss_weight = float(necessity_loss_weight)
        self.event_cardinality_weight = float(event_cardinality_weight)
        self.event_duplicate_weight = float(event_duplicate_weight)
        self.evidence_compactness_weight = float(evidence_compactness_weight)
        self.intervention_margin = float(intervention_margin)
        if self.num_verbs != role_space.num_role_classes:
            raise ValueError("The role space must expose all 24 evaluated role classes.")

    def freeze_detector(self) -> None:
        super().freeze_detector()
        self.semantic_backbone.eval()
        for parameter in self.semantic_backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "HDETRCoRISPEventVCOCO":
        super().train(mode)
        self.semantic_backbone.eval()
        return self

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

    def _predict_event_packets(
        self,
        packets: list,
        visual_grid: Tensor,
        global_semantic: Tensor,
    ) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
        logits: list[Tensor] = []
        diagnostics: list[dict[str, Tensor]] = []
        for index, packet in enumerate(packets):
            output = self.event_field(
                packet,
                visual_grid=visual_grid[index : index + 1],
                global_semantic=global_semantic[index : index + 1],
                return_interventions=self.training,
            )
            logits.append(
                self._expand_action_roles(output["joint_role_logits"]).squeeze(0)
            )
            diagnostics.append(output)
        return logits, diagnostics

    def _null_targets(
        self,
        proposal: dict[str, Tensor],
        target: dict,
    ) -> tuple[Tensor, Tensor]:
        boxes = proposal["boxes"]
        labels = proposal["labels"]
        target_matrix = boxes.new_zeros(
            (boxes.shape[0], self.role_space.num_role_classes)
        )
        valid_human = torch.zeros(
            boxes.shape[0], dtype=torch.bool, device=boxes.device
        )
        gt_visible = recover_boxes(target["boxes_h"], target["size"])
        gt_null = recover_boxes(target["null_agent_boxes"], target["size"])
        all_humans = torch.cat([gt_visible, gt_null], dim=0)
        human = labels == self.human_idx
        if human.any() and all_humans.numel():
            human_indices = torch.nonzero(human, as_tuple=False).flatten()
            overlap = box_ops.box_iou(boxes[human_indices], all_humans)
            valid_human[human_indices[overlap.max(dim=1).values >= 0.5]] = True
        if human.any() and gt_null.numel():
            human_indices = torch.nonzero(human, as_tuple=False).flatten()
            match = box_ops.box_iou(boxes[human_indices], gt_null) >= 0.5
            proposal_index, gt_index = torch.nonzero(match, as_tuple=True)
            if proposal_index.numel():
                target_matrix[
                    human_indices[proposal_index],
                    target["null_role_classes"][gt_index],
                ] = 1.0
        return target_matrix, valid_human

    @staticmethod
    def _distributed_normalizer(value: Tensor) -> Tensor:
        normalizer = value.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(normalizer)
            normalizer = normalizer / dist.get_world_size()
        return normalizer.clamp_min(1.0)

    def _null_role_loss(
        self,
        diagnostics: list[dict[str, Tensor]],
        proposals: list[dict[str, Tensor]],
        targets: list[dict],
        *,
        key: str = "null_role_logits",
    ) -> Tensor:
        selected_logits: list[Tensor] = []
        selected_targets: list[Tensor] = []
        positive_count = diagnostics[0][key].new_zeros(())
        for output, proposal, target in zip(diagnostics, proposals, targets):
            target_matrix, valid_human = self._null_targets(proposal, target)
            class_logits = self._expand_action_roles(output[key].squeeze(0))
            selected_logits.append(class_logits[valid_human])
            selected_targets.append(target_matrix[valid_human])
            positive_count = positive_count + target_matrix[valid_human].sum()
        # Every DDP rank must enter this collective even when its local batch
        # contains no valid/null humans.  A data-dependent early return here
        # otherwise desynchronizes the process group.
        normalizer = self._distributed_normalizer(positive_count)
        if not selected_logits or sum(value.numel() for value in selected_logits) == 0:
            return diagnostics[0][key].sum() * 0.0
        logits = torch.cat(selected_logits, dim=0)
        labels = torch.cat(selected_targets, dim=0)
        return binary_focal_loss_with_logits(
            logits,
            labels,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="sum",
        ) / normalizer

    def _image_action_loss(
        self,
        diagnostics: list[dict[str, Tensor]],
        targets: list[dict],
    ) -> Tensor:
        logits: list[Tensor] = []
        labels: list[Tensor] = []
        for output, target in zip(diagnostics, targets):
            event_probability = (
                output["event_presence_probs"][..., None]
                * output["event_action_probs"]
            )
            image_probability = self.event_field.marginalize_event_contributions(
                event_probability
            )
            logits.append(self.event_field._probability_logit(image_probability))
            action_target = image_probability.new_zeros(
                (1, self.role_space.num_actions)
            )
            role_classes = torch.cat(
                [target["labels"], target["null_role_classes"]], dim=0
            )
            if role_classes.numel():
                action_ids = torch.tensor(
                    [
                        self.role_space.role_class_to_action[int(class_index)]
                        for class_index in role_classes
                    ],
                    device=action_target.device,
                )
                action_target[0, action_ids] = 1.0
            labels.append(action_target)
        joined_logits = torch.cat(logits, dim=0)
        joined_labels = torch.cat(labels, dim=0)
        return binary_focal_loss_with_logits(
            joined_logits,
            joined_labels,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="sum",
        ) / self._distributed_normalizer(joined_labels.sum())

    def _necessity_loss(
        self,
        diagnostics: list[dict[str, Tensor]],
        role_class_labels: Tensor,
        pair_counts: list[int],
        proposals: list[dict[str, Tensor]],
        targets: list[dict],
    ) -> Tensor:
        labels_by_image = role_class_labels.split(pair_counts, dim=0)
        terms: list[Tensor] = []
        positives = role_class_labels.new_zeros(())
        for output, labels, proposal, target in zip(
            diagnostics, labels_by_image, proposals, targets
        ):
            factual = self._expand_action_roles(
                output["joint_role_logits"].squeeze(0)
            )
            erased = self._expand_action_roles(
                output["erased_joint_role_logits"].squeeze(0)
            )
            positive = labels.bool()
            if positive.any():
                terms.append(
                    torch.relu(
                        erased[positive]
                        - factual[positive]
                        + self.intervention_margin
                    ).sum()
                )
                positives = positives + positive.sum()
            null_target, valid_human = self._null_targets(proposal, target)
            null_positive = null_target.bool() & valid_human[:, None]
            if null_positive.any():
                null_factual = self._expand_action_roles(
                    output["null_role_logits"].squeeze(0)
                )
                null_erased = self._expand_action_roles(
                    output["erased_null_role_logits"].squeeze(0)
                )
                terms.append(
                    torch.relu(
                        null_erased[null_positive]
                        - null_factual[null_positive]
                        + self.intervention_margin
                    ).sum()
                )
                positives = positives + null_positive.sum()
        # Keep the collective schedule identical across ranks, including a
        # rank whose local batch has no positive visible or null role.
        normalizer = self._distributed_normalizer(positives)
        if not terms:
            return diagnostics[0]["joint_role_logits"].sum() * 0.0
        return torch.stack(terms).sum() / normalizer

    def _postprocess_event_field(
        self,
        boxes: list[Tensor],
        proposals: list[dict[str, Tensor]],
        paired_indices: list[Tensor],
        object_types: list[Tensor],
        diagnostics: list[dict[str, Tensor]],
        role_class_priors: list[Tensor],
        image_sizes: Tensor,
    ) -> list[dict[str, Tensor]]:
        results: list[dict[str, Tensor]] = []
        for box, proposal, pairs, objects, output, prior, size in zip(
            boxes,
            proposals,
            paired_indices,
            object_types,
            diagnostics,
            role_class_priors,
            image_sizes,
        ):
            visible = self._expand_action_roles(
                output["joint_role_probs"].squeeze(0)
            )
            legal_prior = prior.prod(dim=1)
            pair_index, class_index = torch.nonzero(legal_prior, as_tuple=True)
            visible_scores = (
                visible[pair_index, class_index]
                * legal_prior[pair_index, class_index].pow(self.raw_lambda)
            )

            null = self._expand_action_roles(output["null_role_probs"].squeeze(0))
            human_index = torch.nonzero(
                proposal["labels"] == self.human_idx, as_tuple=False
            ).flatten()
            null_human = human_index[:, None].expand(
                -1, self.role_space.num_role_classes
            ).reshape(-1)
            null_class = torch.arange(
                self.role_space.num_role_classes, device=box.device
            )[None].expand(human_index.numel(), -1).reshape(-1)
            null_scores = null[null_human, null_class]
            null_scores = null_scores * proposal["scores"][null_human].pow(
                self.raw_lambda
            )

            dummy_index = box.shape[0]
            output_boxes = torch.cat([box, box.new_zeros((1, 4))], dim=0)
            null_pairing = torch.stack(
                [null_human, torch.full_like(null_human, dummy_index)], dim=-1
            )
            results.append(
                {
                    "boxes": output_boxes,
                    "pairing": torch.cat(
                        [pairs[pair_index], null_pairing], dim=0
                    ),
                    "scores": torch.cat([visible_scores, null_scores], dim=0),
                    "labels": torch.cat([class_index, null_class], dim=0),
                    "objects": torch.cat(
                        [objects[pair_index], torch.zeros_like(null_human)], dim=0
                    ),
                    "size": size,
                    "x": torch.cat(
                        [pair_index, torch.full_like(null_human, -1)], dim=0
                    ),
                }
            )
        return results

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None,
    ) -> list[dict[str, Tensor]] | dict[str, Tensor]:
        if self.training and targets is None:
            raise ValueError("V-COCO event-field training requires targets.")
        (
            image_sizes,
            proposals,
            boxes,
            pair_indices,
            _,
            object_types,
            packets,
        ) = self._extract_packets(images)
        with torch.no_grad():
            visual_grid, global_semantic = self.semantic_backbone(images)
        logits, diagnostics = self._predict_event_packets(
            packets, visual_grid, global_semantic
        )
        role_class_priors = self._role_class_priors(proposals, pair_indices)

        if self.training:
            assert targets is not None
            role_class_labels = associate_with_ground_truth(
                boxes,
                pair_indices,
                targets,
                self.role_space.num_role_classes,
            )
            joined_logits = torch.cat(logits, dim=0)
            sufficient_logits = torch.cat(
                [
                    self._expand_action_roles(
                        output["sufficient_joint_role_logits"]
                    ).squeeze(0)
                    for output in diagnostics
                ],
                dim=0,
            )
            regularizers = [event_field_regularization(value) for value in diagnostics]
            losses = {
                "cls_loss": self._classification_loss(
                    joined_logits, role_class_priors, role_class_labels
                ),
                "null_role_loss": self.null_role_loss_weight
                * self._null_role_loss(diagnostics, proposals, targets),
                "image_action_loss": self.image_action_loss_weight
                * self._image_action_loss(diagnostics, targets),
                "sufficiency_loss": self.sufficiency_loss_weight
                * self._classification_loss(
                    sufficient_logits, role_class_priors, role_class_labels
                ),
                "necessity_loss": self.necessity_loss_weight
                * self._necessity_loss(
                    diagnostics,
                    role_class_labels,
                    [len(value) for value in pair_indices],
                    proposals,
                    targets,
                ),
                "event_cardinality_loss": self.event_cardinality_weight
                * torch.stack(
                    [value["event_cardinality"] for value in regularizers]
                ).mean(),
                "event_duplicate_loss": self.event_duplicate_weight
                * torch.stack(
                    [value["event_duplicate"] for value in regularizers]
                ).mean(),
                "evidence_compactness_loss": self.evidence_compactness_weight
                * torch.stack(
                    [value["evidence_compactness"] for value in regularizers]
                ).mean(),
            }
            return losses

        return self._postprocess_event_field(
            boxes,
            proposals,
            pair_indices,
            object_types,
            diagnostics,
            role_class_priors,
            image_sizes,
        )
