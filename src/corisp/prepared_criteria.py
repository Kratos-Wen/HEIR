from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import torch

from corisp.joint import box_iou_xyxy

from corisp.joint_losses import CoRISPJointTargets

from corisp.schema import DEFAULT_ROLES, normalize_role_name

@dataclass(frozen=True)
class PreparedPairTargetConfig:
    """Contract between a reviewed HEIR edge adapter and a proposal host."""

    roles: tuple[str, ...] = DEFAULT_ROLES
    subject_box_key: str = "boxes_h"
    object_box_key: str = "boxes_o"
    hoi_label_key: str = "labels"
    role_target_key: str = "role_targets"
    role_valid_key: str = "role_valid_mask"
    acceptable_role_key: str = "acceptable_role_mask"
    hoi_valid_key: str = "hoi_valid_mask"
    event_membership_key: str = "event_membership"
    event_valid_key: str = "event_valid_mask"
    event_complete_key: str = "event_annotation_complete"
    hoi_complete_key: str = "hoi_annotation_complete"
    box_format: str = "cxcywh"
    boxes_normalized: bool = True
    assignment_topk: int = 2
    assignment_iou_threshold: float = 0.50
    assignment_score_weight: float = 0.25
    require_role_targets: bool = True
    require_event_targets: bool = True

    def __post_init__(self) -> None:
        normalized = tuple(normalize_role_name(role) for role in self.roles)
        object.__setattr__(self, "roles", normalized)
        if len(normalized) == 0 or len(set(normalized)) != len(normalized):
            raise ValueError("Prepared-pair roles must be non-empty and unique.")
        if self.box_format not in {"cxcywh", "xyxy"}:
            raise ValueError("box_format must be 'cxcywh' or 'xyxy'.")
        if self.assignment_topk <= 0:
            raise ValueError("assignment_topk must be positive.")
        if not 0.0 <= self.assignment_score_weight <= 1.0:
            raise ValueError("assignment_score_weight must lie in [0,1].")

def _tensor(target: dict[str, Any], key: str) -> torch.Tensor | None:
    value = target.get(key)
    return value if isinstance(value, torch.Tensor) else None

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    size = boxes[..., 2:]
    return torch.cat([center - 0.5 * size, center + 0.5 * size], dim=-1)

def _target_boxes(
    target: dict[str, Any],
    key: str,
    cfg: PreparedPairTargetConfig,
    *,
    device: torch.device,
) -> torch.Tensor:
    boxes = _tensor(target, key)
    if boxes is None:
        raise KeyError(f"HEIR full supervision requires target tensor `{key}`.")
    boxes = boxes.to(device=device, dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"`{key}` must be [N,4].")
    if cfg.box_format == "cxcywh":
        boxes = _cxcywh_to_xyxy(boxes)
    if cfg.boxes_normalized:
        size = _tensor(target, "size")
        if size is None or size.numel() != 2:
            raise KeyError("Normalized HEIR boxes require target tensor `size=[H,W]`.")
        height, width = size.to(device=device, dtype=boxes.dtype).flatten()
        scale = torch.stack([width, height, width, height])
        boxes = boxes * scale
    return boxes

def _validate_dense_targets(
    target: dict[str, Any],
    cfg: PreparedPairTargetConfig,
    *,
    num_edges: int,
    num_classes: int,
    num_roles: int,
    device: torch.device,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    roles = _tensor(target, cfg.role_target_key)
    role_valid = _tensor(target, cfg.role_valid_key)
    acceptable = _tensor(target, cfg.acceptable_role_key)
    hoi_valid = _tensor(target, cfg.hoi_valid_key)
    event_membership = _tensor(target, cfg.event_membership_key)
    event_valid = _tensor(target, cfg.event_valid_key)

    if roles is not None:
        roles = roles.to(device=device, dtype=torch.long)
        if roles.shape != (num_edges, num_classes):
            raise ValueError(
                f"`{cfg.role_target_key}` must be [{num_edges},{num_classes}]."
            )
    if role_valid is not None:
        role_valid = role_valid.to(device=device).bool()
        if role_valid.shape != (num_edges, num_classes):
            raise ValueError(
                f"`{cfg.role_valid_key}` must be [{num_edges},{num_classes}]."
            )
    if acceptable is not None:
        acceptable = acceptable.to(device=device).bool()
        if acceptable.shape != (num_edges, num_classes, num_roles):
            raise ValueError(
                f"`{cfg.acceptable_role_key}` must be "
                f"[{num_edges},{num_classes},{num_roles}]."
            )
    if hoi_valid is not None:
        hoi_valid = hoi_valid.to(device=device).bool()
        if hoi_valid.shape != (num_edges, num_classes):
            raise ValueError(
                f"`{cfg.hoi_valid_key}` must be [{num_edges},{num_classes}]."
            )
    if event_membership is not None:
        event_membership = event_membership.to(device=device, dtype=torch.float32)
        if event_membership.ndim != 3 or event_membership.shape[:2] != (
            num_edges,
            num_classes,
        ):
            raise ValueError(
                f"`{cfg.event_membership_key}` must be [{num_edges},{num_classes},E]."
            )
    if event_valid is not None:
        event_valid = event_valid.to(device=device).bool()
        if event_valid.shape != (num_edges, num_classes):
            raise ValueError(
                f"`{cfg.event_valid_key}` must be [{num_edges},{num_classes}]."
            )
    elif event_membership is not None:
        event_valid = event_membership.any(dim=-1)

    return roles, role_valid, acceptable, hoi_valid, event_membership, event_valid

def build_prepared_pair_joint_targets(
    outputs: dict[str, torch.Tensor],
    proposal_boxes: torch.Tensor,
    pair_indices: torch.Tensor,
    target: dict[str, Any],
    cfg: PreparedPairTargetConfig | None = None,
) -> CoRISPJointTargets:
    """Align reviewed roles/events to one image's prepared proposal pairs.

    Role supervision uses up to two quality-ranked proposal pairs per Gold
    edge. Event membership uses only the best pair so one reviewed atomic edge
    is not duplicated in latent-set matching.
    """

    cfg = cfg or PreparedPairTargetConfig()
    role_logits = outputs.get("conditional_role_logits")
    hoi_logits = outputs.get("hoi_logits")
    if role_logits is None or role_logits.ndim != 4 or role_logits.shape[0] != 1:
        raise ValueError("conditional_role_logits must be [1,Q,C,R].")
    if hoi_logits is None or hoi_logits.shape != role_logits.shape[:-1]:
        raise ValueError("hoi_logits must be [1,Q,C] and align with role logits.")
    _, num_pairs, num_classes, num_roles = role_logits.shape
    if num_roles != len(cfg.roles):
        raise ValueError("Configured Function-8 order does not match model outputs.")
    if pair_indices.shape != (num_pairs, 2):
        raise ValueError("pair_indices must be [Q,2] and align with model outputs.")

    device = role_logits.device
    labels = _tensor(target, cfg.hoi_label_key)
    if labels is None:
        raise KeyError(f"HEIR full supervision requires target tensor `{cfg.hoi_label_key}`.")
    labels = labels.to(device=device, dtype=torch.long).flatten()
    num_edges = int(labels.numel())
    if num_edges == 0:
        raise ValueError("HEIR full-supervision images must contain at least one reviewed edge.")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("Gold HOI labels fall outside the configured classifier order.")

    target_subject = _target_boxes(target, cfg.subject_box_key, cfg, device=device)
    target_object = _target_boxes(target, cfg.object_box_key, cfg, device=device)
    if target_subject.shape[0] != num_edges or target_object.shape[0] != num_edges:
        raise ValueError("Gold subject/object boxes must align with edge labels.")
    proposal_boxes = proposal_boxes.to(device=device, dtype=torch.float32)
    pair_indices = pair_indices.to(device=device, dtype=torch.long)
    predicted_subject = proposal_boxes[pair_indices[:, 0]]
    predicted_object = proposal_boxes[pair_indices[:, 1]]
    pair_iou = torch.minimum(
        box_iou_xyxy(predicted_subject[None], target_subject[None])[0],
        box_iou_xyxy(predicted_object[None], target_object[None])[0],
    )

    class_score = hoi_logits[0].detach().float().sigmoid()[:, labels]
    score_weight = float(cfg.assignment_score_weight)
    quality = (1.0 - score_weight) * pair_iou + score_weight * class_score
    k = min(int(cfg.assignment_topk), num_pairs)
    top_pairs = torch.topk(quality, k=k, dim=0).indices
    top_iou = torch.gather(pair_iou, dim=0, index=top_pairs)

    roles, role_valid, acceptable, edge_hoi_valid, event_membership, event_valid = (
        _validate_dense_targets(
            target,
            cfg,
            num_edges=num_edges,
            num_classes=num_classes,
            num_roles=num_roles,
            device=device,
        )
    )
    positive_index = torch.arange(num_edges, device=device)
    if cfg.require_role_targets:
        if not bool(target.get(cfg.hoi_complete_key, False)):
            raise KeyError(
                "HEIR full supervision requires complete reviewed HOI annotation."
            )
        if roles is None:
            raise KeyError(f"HEIR full supervision requires `{cfg.role_target_key}`.")
        valid_gold_roles = roles[positive_index, labels] >= 0
        if role_valid is not None:
            valid_gold_roles &= role_valid[positive_index, labels]
        if not bool(valid_gold_roles.all()):
            raise KeyError("Every reviewed positive HEIR edge requires a valid Function-8 role.")
    if cfg.require_event_targets:
        complete = bool(target.get(cfg.event_complete_key, False))
        if event_membership is None or event_valid is None or not complete:
            raise KeyError(
                "HEIR full supervision requires complete reviewed event membership."
            )
        if not bool(event_valid[positive_index, labels].all()):
            raise KeyError("Every reviewed positive HEIR edge requires valid event membership.")

    aligned_roles = torch.full(
        (1, num_pairs, num_classes), -1, device=device, dtype=torch.long
    )
    aligned_role_valid = torch.zeros_like(aligned_roles, dtype=torch.bool)
    aligned_acceptable = torch.zeros(
        (1, num_pairs, num_classes, num_roles), device=device, dtype=torch.bool
    )
    aligned_hoi_valid = torch.ones_like(aligned_roles, dtype=torch.bool)
    num_events = 0 if event_membership is None else int(event_membership.shape[-1])
    aligned_event = role_logits.new_zeros((num_pairs, num_classes, num_events))
    aligned_event_valid = torch.zeros(
        (num_pairs, num_classes), device=device, dtype=torch.bool
    )

    for edge_idx in range(num_edges):
        class_idx = int(labels[edge_idx])
        valid_candidates = top_iou[:, edge_idx] >= float(cfg.assignment_iou_threshold)
        candidates = top_pairs[:, edge_idx][valid_candidates]
        if candidates.numel() == 0:
            continue
        if event_membership is not None and event_valid is not None:
            best_pair = int(candidates[0])
            aligned_event[best_pair, class_idx] = torch.maximum(
                aligned_event[best_pair, class_idx],
                event_membership[edge_idx, class_idx].to(aligned_event.dtype),
            )
            aligned_event_valid[best_pair, class_idx] |= event_valid[edge_idx, class_idx]

        for pair_idx_tensor in candidates:
            pair_idx = int(pair_idx_tensor)
            if edge_hoi_valid is not None:
                aligned_hoi_valid[0, pair_idx, class_idx] &= edge_hoi_valid[
                    edge_idx, class_idx
                ]
            if roles is None:
                continue
            is_valid = roles[edge_idx, class_idx] >= 0
            if role_valid is not None:
                is_valid &= role_valid[edge_idx, class_idx]
            if not bool(is_valid):
                continue
            role_idx = int(roles[edge_idx, class_idx])
            if role_idx < 0 or role_idx >= num_roles:
                raise ValueError("Gold role index falls outside Function-8.")
            if aligned_roles[0, pair_idx, class_idx] < 0:
                aligned_roles[0, pair_idx, class_idx] = role_idx
            aligned_role_valid[0, pair_idx, class_idx] = True
            if acceptable is None:
                aligned_acceptable[0, pair_idx, class_idx, role_idx] = True
            else:
                aligned_acceptable[0, pair_idx, class_idx] |= acceptable[
                    edge_idx, class_idx
                ]

    event_available = torch.tensor(
        [
            bool(target.get(cfg.event_complete_key, False))
            and event_membership is not None
        ],
        device=device,
        dtype=torch.bool,
    )
    return CoRISPJointTargets(
        role_targets=aligned_roles,
        role_supervision_mask=aligned_role_valid,
        acceptable_role_mask=aligned_acceptable,
        hoi_valid_mask=aligned_hoi_valid,
        event_membership_targets=(aligned_event,),
        event_valid_masks=(aligned_event_valid,),
        event_supervision_available=event_available,
    )
