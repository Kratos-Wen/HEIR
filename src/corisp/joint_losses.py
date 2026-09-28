from __future__ import annotations

from dataclasses import dataclass

import torch

import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

@dataclass(frozen=True)
class CoRISPJointTargets:
    role_targets: torch.Tensor
    role_supervision_mask: torch.Tensor
    acceptable_role_mask: torch.Tensor
    hoi_valid_mask: torch.Tensor
    event_membership_targets: tuple[torch.Tensor, ...] = ()
    event_valid_masks: tuple[torch.Tensor, ...] = ()
    event_supervision_available: torch.Tensor | None = None

@dataclass(frozen=True)
class CoRISPJointLossConfig:
    role_weight: float = 1.0
    intervention_weight: float = 1.0
    sufficiency_weight: float = 0.20
    necessity_weight: float = 0.20
    compactness_weight: float = 0.02
    prior_neutral_weight: float = 0.20
    intervention_margin: float = 0.05
    supporting_temperature: float = 0.7
    event_weight: float = 1.0
    event_membership_weight: float = 1.0
    event_dice_weight: float = 1.0
    event_presence_weight: float = 0.5
    event_no_event_weight: float = 0.1

def _zero_like(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    for value in outputs.values():
        if torch.is_tensor(value):
            return value.sum() * 0.0
    return torch.tensor(0.0)

def _role_nll(
    logits: torch.Tensor,
    targets: CoRISPJointTargets,
) -> tuple[torch.Tensor, torch.Tensor]:
    role_targets = targets.role_targets.to(device=logits.device, dtype=torch.long)
    supervised = targets.role_supervision_mask.to(device=logits.device).bool() & (role_targets >= 0)
    supervised &= targets.hoi_valid_mask.to(device=logits.device).bool()
    if not supervised.any():
        return logits.sum() * 0.0, supervised

    log_probs = F.log_softmax(logits.float(), dim=-1)
    safe_targets = role_targets.clamp(0, logits.shape[-1] - 1)
    nll = -log_probs.gather(-1, safe_targets[..., None]).squeeze(-1)
    acceptable = targets.acceptable_role_mask.to(device=logits.device).bool()
    has_set = acceptable.any(dim=-1)
    set_log_prob = torch.logsumexp(log_probs.masked_fill(~acceptable, float("-inf")), dim=-1)
    nll = torch.where(has_set, -set_log_prob, nll)
    return nll[supervised].mean(), supervised

def _acceptable_roles(
    targets: CoRISPJointTargets,
    num_roles: int,
    device: torch.device,
) -> torch.Tensor:
    role_targets = targets.role_targets.to(device=device, dtype=torch.long)
    acceptable = targets.acceptable_role_mask.to(device=device).bool().clone()
    missing = ~acceptable.any(dim=-1)
    valid_target = (role_targets >= 0) & (role_targets < num_roles) & missing
    target_one_hot = F.one_hot(role_targets.clamp(0, num_roles - 1), num_classes=num_roles).bool()
    return acceptable | (valid_target[..., None] & target_one_hot)

def _joint_state_margin(
    state_logits: torch.Tensor,
    targets: CoRISPJointTargets,
) -> torch.Tensor:
    acceptable = _acceptable_roles(targets, state_logits.shape[-1] - 1, state_logits.device)
    positive_mask = torch.cat([torch.zeros_like(acceptable[..., :1]), acceptable], dim=-1)
    correct = torch.logsumexp(state_logits.float().masked_fill(~positive_mask, float("-inf")), dim=-1)
    competing = torch.logsumexp(state_logits.float().masked_fill(positive_mask, float("-inf")), dim=-1)
    return correct - competing

def _supporting_mask_compactness(
    outputs: dict[str, torch.Tensor],
    targets: CoRISPJointTargets,
    supervised: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = ("basis_masks", "basis_activations", "hoi_factors", "role_factors")
    if not supervised.any() or any(name not in outputs for name in required):
        zero = _zero_like(outputs)
        return zero, zero.detach()
    indices = supervised.nonzero(as_tuple=False)
    batch_idx, query_idx, hoi_idx = indices.unbind(dim=-1)
    role_idx = targets.role_targets.to(device=indices.device, dtype=torch.long)[supervised]
    activations = outputs["basis_activations"][batch_idx, query_idx].float()
    hoi = outputs["hoi_factors"][hoi_idx].float()
    role = outputs["role_factors"][role_idx].float()
    signed_contribution = activations * hoi * role
    weights = F.softmax(signed_contribution / max(float(temperature), 1e-4), dim=-1)
    masks = outputs["basis_masks"][batch_idx, query_idx].float()
    supporting_mask = torch.einsum("nk,nkhw->nhw", weights, masks)
    area = supporting_mask.mean(dim=(-2, -1))
    return area.mean(), area.mean().detach()

def _event_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: CoRISPJointTargets,
    cfg: CoRISPJointLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match latent event slots to reviewed event sets and supervise membership."""

    zero = _zero_like(outputs)
    available = targets.event_supervision_available
    if available is None or not bool(available.any()):
        return zero, zero, zero, zero, zero.detach()
    required = ("event_edge_membership_logits", "event_presence_logits")
    missing = [name for name in required if name not in outputs]
    if missing:
        raise KeyError(f"Event-supervised CoRISP outputs are missing: {missing}")

    membership_logits = outputs["event_edge_membership_logits"]
    presence_logits = outputs["event_presence_logits"]
    if membership_logits.ndim != 4:
        raise ValueError("Event edge membership logits must be [B,Q,C,G].")
    batch_size, num_queries, num_classes, num_slots = membership_logits.shape
    if presence_logits.shape != (batch_size, num_slots):
        raise ValueError("Event presence logits must be [B,G].")
    if len(targets.event_membership_targets) != batch_size or len(targets.event_valid_masks) != batch_size:
        raise ValueError("Event target tuples must contain one tensor per batch item.")

    membership_terms: list[torch.Tensor] = []
    dice_terms: list[torch.Tensor] = []
    presence_terms: list[torch.Tensor] = []
    matched_groups = 0
    supervised_images = 0
    for batch_idx in range(batch_size):
        if not bool(available[batch_idx]):
            continue
        supervised_images += 1
        target_membership = targets.event_membership_targets[batch_idx].to(
            device=membership_logits.device,
            dtype=membership_logits.dtype,
        )
        valid = targets.event_valid_masks[batch_idx].to(device=membership_logits.device).bool()
        if target_membership.ndim != 3 or target_membership.shape[:2] != (
            num_queries,
            num_classes,
        ):
            raise ValueError("Each event membership target must be [Q,C,E].")
        if valid.shape != (num_queries, num_classes):
            raise ValueError("Each event valid mask must be [Q,C].")

        # Flatten only reviewed positive semantic edges. Background classes are
        # handled by the detector loss and must not dominate event assignment.
        pred = membership_logits[batch_idx][valid].transpose(0, 1).float()
        gold = target_membership[valid].transpose(0, 1).float()
        nonempty = gold.sum(dim=-1) > 0
        gold = gold[nonempty]
        num_events = gold.shape[0]
        if num_events > num_slots:
            raise ValueError(
                f"Found {num_events} reviewed events but the model has only {num_slots} event slots."
            )

        selected_slots = torch.empty(0, device=membership_logits.device, dtype=torch.long)
        selected_events = torch.empty(0, device=membership_logits.device, dtype=torch.long)
        if num_events > 0:
            expanded_pred = pred[:, None, :].expand(-1, num_events, -1)
            expanded_gold = gold[None, :, :].expand(num_slots, -1, -1)
            cost_membership = F.binary_cross_entropy_with_logits(
                expanded_pred,
                expanded_gold,
                reduction="none",
            ).mean(dim=-1)
            probability = pred.sigmoid()
            intersection = torch.einsum("gq,eq->ge", probability, gold)
            denominator = probability.sum(dim=-1, keepdim=True) + gold.sum(dim=-1)[None]
            cost_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            cost_presence = -F.logsigmoid(presence_logits[batch_idx].float())[:, None]
            cost = (
                float(cfg.event_membership_weight) * cost_membership
                + float(cfg.event_dice_weight) * cost_dice
                + float(cfg.event_presence_weight) * cost_presence
            )
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            selected_slots = torch.as_tensor(row, device=membership_logits.device, dtype=torch.long)
            selected_events = torch.as_tensor(col, device=membership_logits.device, dtype=torch.long)
            selected_pred = pred[selected_slots]
            selected_gold = gold[selected_events]
            membership_terms.append(
                F.binary_cross_entropy_with_logits(selected_pred, selected_gold)
            )
            selected_probability = selected_pred.sigmoid()
            selected_intersection = (selected_probability * selected_gold).sum(dim=-1)
            selected_denominator = selected_probability.sum(dim=-1) + selected_gold.sum(dim=-1)
            dice_terms.append(
                (1.0 - (2.0 * selected_intersection + 1.0) / (selected_denominator + 1.0)).mean()
            )
            matched_groups += int(num_events)

        presence_target = torch.zeros_like(presence_logits[batch_idx], dtype=torch.float32)
        presence_target[selected_slots] = 1.0
        presence_raw = F.binary_cross_entropy_with_logits(
            presence_logits[batch_idx].float(),
            presence_target,
            reduction="none",
        )
        presence_weight = torch.where(
            presence_target > 0,
            torch.ones_like(presence_raw),
            torch.full_like(presence_raw, float(cfg.event_no_event_weight)),
        )
        presence_terms.append((presence_raw * presence_weight).sum() / presence_weight.sum().clamp_min(1.0))

    loss_membership = torch.stack(membership_terms).mean() if membership_terms else zero
    loss_dice = torch.stack(dice_terms).mean() if dice_terms else zero
    loss_presence = torch.stack(presence_terms).mean() if presence_terms else zero
    loss_event = (
        float(cfg.event_membership_weight) * loss_membership
        + float(cfg.event_dice_weight) * loss_dice
        + float(cfg.event_presence_weight) * loss_presence
    )
    mean_groups = membership_logits.new_tensor(
        float(matched_groups) / max(float(supervised_images), 1.0)
    )
    return loss_event, loss_membership, loss_dice, loss_presence, mean_groups.detach()

def corisp_joint_loss(
    outputs: dict[str, torch.Tensor],
    targets: CoRISPJointTargets,
    cfg: CoRISPJointLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute role and evidence-intervention losses for the joint state."""

    cfg = cfg or CoRISPJointLossConfig()
    required = {
        "conditional_role_logits",
        "joint_state_logits",
        "clean_joint_state_logits",
        "erased_joint_state_logits",
    }
    missing = sorted(required.difference(outputs))
    if missing:
        raise KeyError(f"CoRISP joint outputs are missing: {missing}")

    role_logits = outputs["conditional_role_logits"]
    loss_role, supervised = _role_nll(role_logits, targets)
    zero = role_logits.sum() * 0.0
    if supervised.any():
        full_margin = _joint_state_margin(outputs["joint_state_logits"], targets)
        clean_margin = _joint_state_margin(outputs["clean_joint_state_logits"], targets)
        erased_margin = _joint_state_margin(outputs["erased_joint_state_logits"], targets)
        margin = float(cfg.intervention_margin)
        loss_sufficiency = F.relu(clean_margin + margin - full_margin)[supervised].mean()
        loss_necessity = F.relu(margin - (full_margin - erased_margin))[supervised].mean()
        margin_gain = (full_margin - erased_margin)[supervised].mean().detach()
    else:
        loss_sufficiency = zero
        loss_necessity = zero
        margin_gain = zero.detach()

    neutral_logits = outputs.get("neutral_conditional_role_logits")
    loss_prior_neutral = zero if neutral_logits is None else _role_nll(neutral_logits, targets)[0]
    loss_compact, evidence_area = _supporting_mask_compactness(
        outputs,
        targets,
        supervised,
        cfg.supporting_temperature,
    )
    intervention = (
        cfg.sufficiency_weight * loss_sufficiency
        + cfg.necessity_weight * loss_necessity
        + cfg.compactness_weight * loss_compact
        + cfg.prior_neutral_weight * loss_prior_neutral
    )
    loss_event, loss_event_membership, loss_event_dice, loss_event_presence, event_groups = _event_set_loss(
        outputs,
        targets,
        cfg,
    )
    total = (
        cfg.role_weight * loss_role
        + cfg.intervention_weight * intervention
        + cfg.event_weight * loss_event
    )
    return {
        "loss_corisp_total": total,
        "loss_corisp_joint_role": loss_role.detach(),
        "loss_corisp_intervention": intervention.detach(),
        "loss_corisp_sufficiency": loss_sufficiency.detach(),
        "loss_corisp_necessity": loss_necessity.detach(),
        "loss_corisp_prior_neutral": loss_prior_neutral.detach(),
        "loss_corisp_evidence_compact": loss_compact.detach(),
        "loss_corisp_event": loss_event.detach(),
        "loss_corisp_event_membership": loss_event_membership.detach(),
        "loss_corisp_event_dice": loss_event_dice.detach(),
        "loss_corisp_event_presence": loss_event_presence.detach(),
        "corisp_joint_margin_gain": margin_gain,
        "corisp_evidence_area": evidence_area,
        "corisp_event_groups_per_supervised_image": event_groups,
    }
