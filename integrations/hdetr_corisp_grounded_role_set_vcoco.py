from __future__ import annotations

from dataclasses import dataclass

from typing import List, Optional, Sequence

import torch

from torch import Tensor

import torchvision.ops.boxes as box_ops

from scipy.optimize import linear_sum_assignment

from ops import recover_boxes

from hdetr_corisp_v6_vcoco import HDETRCoRISPAgentRoleVCOCO

from corisp import GroundedRoleSetAblation, GroundedRoleSetField, GroundedRoleSetOutput, RoleFillerTargetGroup, VCOCORoleSpace

@dataclass(frozen=True)
class _GroundTruthPersonFrame:
    box: Tensor
    visible_records: tuple[tuple[int, int, Tensor], ...]
    null_records: tuple[tuple[int, int], ...]
    source_identity: int | None = None
    ambiguous_after_transform: bool = False

def _same_box(first: Tensor, second: Tensor) -> bool:
    return bool(torch.allclose(first, second, atol=1e-4, rtol=0.0))

def _ground_truth_person_frames(
    target: dict,
    role_space: VCOCORoleSpace,
) -> list[_GroundTruthPersonFrame]:
    visible_human = recover_boxes(target["boxes_h"], target["size"])
    visible_entity = recover_boxes(target["boxes_o"], target["size"])
    null_human = recover_boxes(target["null_agent_boxes"], target["size"])
    frames: list[dict[str, object]] = []

    def aligned_identities(field: str, length: int) -> list[int | None]:
        values = target.get(field)
        if values is None:
            return [None] * length
        values = values.flatten().long()
        if len(values) != length:
            raise ValueError(f"{field} is not aligned with its role records.")
        return [int(value) for value in values]

    visible_identities = aligned_identities(
        "agent_instance_ids", len(visible_human)
    )
    null_identities = aligned_identities(
        "null_agent_instance_ids", len(null_human)
    )

    def frame_index(box: Tensor, source_identity: int | None) -> int:
        for index, frame in enumerate(frames):
            if source_identity is not None:
                if frame["source_identity"] != source_identity:
                    continue
                if not _same_box(frame["box"], box):
                    raise ValueError(
                        "One source-agent identity acquired inconsistent boxes "
                        "under a shared geometric transform."
                    )
                return index
            if frame["source_identity"] is None and _same_box(frame["box"], box):
                return index
        frames.append(
            {
                "box": box,
                "visible": [],
                "null": [],
                "source_identity": source_identity,
            }
        )
        return len(frames) - 1

    for human_box, entity_box, class_index, source_identity in zip(
        visible_human,
        visible_entity,
        target["labels"],
        visible_identities,
    ):
        class_id = int(class_index)
        index = frame_index(human_box, source_identity)
        frames[index]["visible"].append(
            (
                role_space.role_class_to_action[class_id],
                role_space.role_class_to_role[class_id],
                entity_box,
            )
        )
    for human_box, class_index, source_identity in zip(
        null_human,
        target["null_role_classes"],
        null_identities,
    ):
        class_id = int(class_index)
        index = frame_index(human_box, source_identity)
        frames[index]["null"].append(
            (
                role_space.role_class_to_action[class_id],
                role_space.role_class_to_role[class_id],
            )
        )
    ambiguous: set[int] = set()
    for first in range(len(frames)):
        first_identity = frames[first]["source_identity"]
        if first_identity is None:
            continue
        for second in range(first + 1, len(frames)):
            second_identity = frames[second]["source_identity"]
            if (
                second_identity is not None
                and first_identity != second_identity
                and _same_box(frames[first]["box"], frames[second]["box"])
            ):
                ambiguous.update((first, second))

    return [
        _GroundTruthPersonFrame(
            box=frame["box"],
            visible_records=tuple(frame["visible"]),
            null_records=tuple(frame["null"]),
            source_identity=frame["source_identity"],
            ambiguous_after_transform=index in ambiguous,
        )
        for index, frame in enumerate(frames)
    ]

def _match_detected_agents(
    detected_boxes: Tensor,
    frames: Sequence[_GroundTruthPersonFrame],
    *,
    iou_threshold: float = 0.5,
) -> dict[int, int]:
    """Match real-person identities before supervising their event sets.

    The large valid-edge bonus makes assignment cardinality lexicographically
    primary and total IoU secondary.  Invalid assignments are filtered after
    optimization, so detector duplicates remain reviewed negatives.
    """

    if detected_boxes.numel() == 0 or not frames:
        return {}
    ground_truth = torch.stack([frame.box for frame in frames])
    overlap = box_ops.box_iou(detected_boxes, ground_truth)
    valid = overlap >= iou_threshold
    cardinality_bonus = float(min(overlap.shape) + 1)
    reward = torch.where(
        valid,
        cardinality_bonus + overlap,
        torch.zeros_like(overlap),
    )
    detected_indices, ground_truth_indices = linear_sum_assignment(
        reward.detach().cpu().numpy(), maximize=True
    )
    return {
        int(detected): int(ground_truth_index)
        for detected, ground_truth_index in zip(
            detected_indices.tolist(), ground_truth_indices.tolist()
        )
        if bool(valid[detected, ground_truth_index])
    }

def _match_supervisable_agents(
    detected_boxes: Tensor,
    frames: Sequence[_GroundTruthPersonFrame],
    *,
    iou_threshold: float = 0.5,
) -> tuple[dict[int, int], set[int]]:
    """Match valid identities and ignore detections near crop-collapsed ones."""

    ambiguous_indices = [
        index
        for index, frame in enumerate(frames)
        if frame.ambiguous_after_transform
    ]
    ignored_detected: set[int] = set()
    if detected_boxes.numel() and ambiguous_indices:
        ambiguous_boxes = torch.stack(
            [frames[index].box for index in ambiguous_indices]
        )
        overlap = box_ops.box_iou(detected_boxes, ambiguous_boxes)
        ignored_detected = set(
            torch.nonzero(
                (overlap >= iou_threshold).any(dim=1), as_tuple=False
            ).flatten().tolist()
        )

    frame_indices = [
        index
        for index, frame in enumerate(frames)
        if not frame.ambiguous_after_transform
    ]
    detected_indices = [
        index
        for index in range(len(detected_boxes))
        if index not in ignored_detected
    ]
    if not frame_indices or not detected_indices:
        return {}, ignored_detected

    local_assignment = _match_detected_agents(
        detected_boxes[
            torch.tensor(
                detected_indices, device=detected_boxes.device, dtype=torch.long
            )
        ],
        [frames[index] for index in frame_indices],
        iou_threshold=iou_threshold,
    )
    assignment = {
        detected_indices[detected]: frame_indices[frame]
        for detected, frame in local_assignment.items()
    }
    return assignment, ignored_detected

class HDETRCoRISPGroundedRoleSetVCOCO(HDETRCoRISPAgentRoleVCOCO):
    """One normalized role-filler set for each real person-predicate frame."""

    def __init__(
        self,
        *args,
        grounded_role_set: GroundedRoleSetField,
        ablation: GroundedRoleSetAblation | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.grounded_role_set = grounded_role_set
        self.ablation = ablation or GroundedRoleSetAblation.from_id("full")

    def _detector_composed_distribution(
        self,
        output: dict[str, Tensor],
        proposal: dict[str, Tensor],
        role_class_prior: Tensor,
        *,
        return_marginals: bool,
    ) -> GroundedRoleSetOutput:
        class_prior = role_class_prior.prod(dim=1)
        q = class_prior.shape[0]
        role_prior = class_prior.new_zeros(
            (q, self.role_space.num_actions, self.role_space.num_roles)
        )
        for class_index, (action_index, role_index) in enumerate(
            zip(
                self.role_space.role_class_to_action,
                self.role_space.role_class_to_role,
            )
        ):
            role_prior[:, action_index, role_index] = class_prior[:, class_index]

        visible = output["joint_role_probs"].squeeze(0)
        agent_entities = output["agent_entity_indices"].squeeze(0).long()
        typed = output["null_role_probs"].squeeze(0)[agent_entities]
        if self.ablation.detector_score_mode == "inside":
            visible = visible * role_prior
            typed = typed * proposal["scores"][agent_entities, None, None]
        if not self.ablation.use_typed_null_atoms:
            typed = torch.zeros_like(typed)
        return self.grounded_role_set(
            visible_role_probs=visible,
            typed_null_role_probs=typed,
            pair_agent_indices=output["pair_agent_indices"].squeeze(0).long(),
            event_states=output["event_states"].squeeze(0),
            visible_action_mask=output["visible_action_mask"].squeeze(0).bool(),
            valid_role_mask=self.role_space.valid_role_mask().to(
                device=visible.device
            ),
            return_marginals=return_marginals,
        )

    def _event_target_groups(
        self,
        *,
        distribution: GroundedRoleSetOutput,
        output: dict[str, Tensor],
        proposal: dict[str, Tensor],
        pairs: Tensor,
        target: dict,
    ) -> list[list[RoleFillerTargetGroup] | None]:
        frames = _ground_truth_person_frames(target, self.role_space)
        agent_entities = output["agent_entity_indices"].squeeze(0).long()
        assignment, ignored_agents = _match_supervisable_agents(
            proposal["boxes"][agent_entities], frames
        )
        valid_role_mask = self.role_space.valid_role_mask().to(
            device=proposal["boxes"].device
        )
        targets: list[list[RoleFillerTargetGroup] | None] = []
        for event in distribution.events:
            if event.agent_index in ignored_agents:
                targets.append(None)
                continue
            frame_index = assignment.get(event.agent_index)
            if frame_index is None:
                targets.append([])
                continue
            frame = frames[frame_index]
            visible_by_role: dict[int, list[Tensor]] = {}
            null_roles: set[int] = set()
            for action, role, box in frame.visible_records:
                if action == event.action_index:
                    visible_by_role.setdefault(role, []).append(box)
            for action, role in frame.null_records:
                if action == event.action_index:
                    null_roles.add(role)

            if null_roles and not self.ablation.use_typed_null_atoms:
                targets.append(None)
                continue

            observed_roles = set(visible_by_role).union(null_roles)
            if not observed_roles:
                targets.append([])
                continue
            required_roles = set(
                torch.nonzero(
                    valid_role_mask[event.action_index], as_tuple=False
                ).flatten().tolist()
            )
            # Cropping can remove one member of a multi-role frame.  Such a
            # partial frame is unobserved rather than relabeled as lower
            # cardinality.
            if observed_roles != required_roles:
                targets.append(None)
                continue

            groups: list[RoleFillerTargetGroup] = []
            representable = True
            for role in sorted(required_roles):
                boxes = visible_by_role.get(role, [])
                is_null = role in null_roles
                if is_null and boxes:
                    raise ValueError(
                        "One V-COCO role cannot be both visible and typed-null."
                    )
                if is_null:
                    rows = torch.tensor(
                        [event.typed_candidate_row(role)],
                        device=proposal["boxes"].device,
                        dtype=torch.long,
                    )
                else:
                    if len(boxes) != 1:
                        raise ValueError(
                            "The frozen V-COCO contract requires one Gold filler "
                            "per visible person-action-role slot."
                        )
                    object_indices = pairs[event.pair_indices, 1].long()
                    if object_indices.numel():
                        overlap = box_ops.box_iou(
                            proposal["boxes"][object_indices], boxes[0][None]
                        ).squeeze(-1)
                        rows = torch.nonzero(
                            overlap >= 0.5, as_tuple=False
                        ).flatten()
                        if (
                            rows.numel()
                            and self.ablation.localization_target_mode
                            == "best_iou"
                        ):
                            rows = rows[
                                overlap[rows].argmax().reshape(1)
                            ]
                    else:
                        rows = object_indices
                if rows.numel() == 0:
                    representable = False
                    break
                groups.append(RoleFillerTargetGroup(role, rows))
            targets.append(groups if representable else None)
        return targets

    def _structured_event_loss(
        self,
        distributions: Sequence[GroundedRoleSetOutput],
        constraints: Sequence[Sequence[list[RoleFillerTargetGroup] | None]],
    ) -> Tensor:
        terms: list[Tensor] = []
        reference = self.grounded_role_set.cardinality_head.weight
        positive_count = reference.new_zeros(())
        for distribution, image_constraints in zip(distributions, constraints):
            if len(distribution.events) != len(image_constraints):
                raise RuntimeError("Grounded role-set constraints are misaligned.")
            for event, groups in zip(distribution.events, image_constraints):
                if groups is None:
                    continue
                target_log_mass = self.grounded_role_set.target_log_mass(
                    event, groups
                )
                if target_log_mass is None:
                    continue
                nll = (event.log_partition - target_log_mass).clamp_min(0.0)
                alpha = self.alpha if groups else (1.0 - self.alpha)
                terms.append(
                    alpha
                    * self.grounded_role_set.focal_transform_nll(
                        nll, self.gamma
                    )
                )
                positive_count = positive_count + len(groups)
        normalizer = self._distributed_normalizer(positive_count)
        if not terms:
            return reference.sum() * 0.0
        return torch.stack(terms).sum() / normalizer

    def _postprocess_grounded_role_sets(
        self,
        boxes: list[Tensor],
        proposals: list[dict[str, Tensor]],
        paired_indices: list[Tensor],
        object_types: list[Tensor],
        outputs: list[dict[str, Tensor]],
        distributions: Sequence[GroundedRoleSetOutput],
        role_class_priors: list[Tensor],
        image_sizes: Tensor,
    ) -> list[dict[str, Tensor]]:
        results: list[dict[str, Tensor]] = []
        for box, proposal, pairs, objects, output, distribution, prior, size in zip(
            boxes,
            proposals,
            paired_indices,
            object_types,
            outputs,
            distributions,
            role_class_priors,
            image_sizes,
        ):
            if (
                distribution.visible_role_marginals is None
                or distribution.typed_null_role_marginals is None
            ):
                raise RuntimeError("Inference requires exact role-set marginals.")
            visible = self._expand_action_roles(
                distribution.visible_role_marginals
            )
            supported = prior.prod(dim=1) > 0
            pair_index, class_index = torch.nonzero(supported, as_tuple=True)
            visible_scores = visible[pair_index, class_index]
            if self.ablation.detector_score_mode == "posthoc":
                visible_scores = visible_scores * prior.prod(dim=1)[
                    pair_index, class_index
                ]

            null = self._expand_action_roles(
                distribution.typed_null_role_marginals
            )
            human_index = output["agent_entity_indices"].squeeze(0).long()
            null_human = human_index[:, None].expand(
                -1, self.role_space.num_role_classes
            ).reshape(-1)
            null_class = torch.arange(
                self.role_space.num_role_classes, device=box.device
            )[None].expand(human_index.numel(), -1).reshape(-1)
            if self.ablation.detector_score_mode == "posthoc":
                null = null * proposal["scores"][human_index, None]
            null_scores = null.reshape(-1)

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
            raise ValueError("Grounded role-set training requires targets.")
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
        distributions = [
            self._detector_composed_distribution(
                output,
                proposal,
                prior,
                return_marginals=not self.training,
            )
            for output, proposal, prior in zip(
                outputs, proposals, role_class_priors
            )
        ]

        if self.training:
            assert targets is not None
            constraints = [
                self._event_target_groups(
                    distribution=distribution,
                    output=output,
                    proposal=proposal,
                    pairs=pairs,
                    target=target,
                )
                for distribution, output, proposal, pairs, target in zip(
                    distributions, outputs, proposals, pair_indices, targets
                )
            ]
            return {
                "grounded_role_set_loss": self._structured_event_loss(
                    distributions, constraints
                )
            }

        return self._postprocess_grounded_role_sets(
            boxes,
            proposals,
            pair_indices,
            object_types,
            outputs,
            distributions,
            role_class_priors,
            image_sizes,
        )
