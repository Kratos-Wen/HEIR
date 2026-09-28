from __future__ import annotations

from dataclasses import dataclass

from itertools import product

from math import sqrt

from typing import Sequence

import torch

import torch.nn as nn

from torch import Tensor

from corisp.grounded_role_set import RoleFillerTargetGroup

CORISP_ROLE_FILLER_EVENT_FIELD_VERSION = (
    "corisp_joint_v10_role_filler_event_field_v1"
)

ROLE_FILLER_EVENT_FIELD_INTERVENTIONS = (
    "full",
    "unary_only",
    "no_unary_refinement",
    "no_cardinality",
    "no_signature",
    "no_event_relation",
    "no_pair_relation",
    "no_entity_relation",
    "no_all_relations",
    "role_collapsed_routing",
)

ROLE_FILLER_EVENT_FIELD_STRUCTURAL_ABLATIONS = (
    "full",
    "unary_only",
    "local_unary",
    "cardinality_only",
    "local_cardinality",
    "relational_cardinality",
    "local_signature",
    "relational_signature",
    "no_event_relation",
    "no_pair_relation",
    "no_entity_relation",
    "no_all_relations",
    "role_collapsed_routing",
)

_ROLE_FILLER_EVENT_FIELD_EXECUTION_MODES = tuple(
    dict.fromkeys(
        ROLE_FILLER_EVENT_FIELD_INTERVENTIONS
        + ROLE_FILLER_EVENT_FIELD_STRUCTURAL_ABLATIONS
    )
)

@dataclass(frozen=True)
class RoleFillerEventFieldConfig:
    d_model: int = 384
    num_roles: int = 2
    max_cardinality: int = 32
    num_heads: int = 8
    signature_rank: int = 64
    dropout: float = 0.1
    max_log_residual: float = 4.0
    max_signature_energy: float = 4.0
    architecture_version: str = CORISP_ROLE_FILLER_EVENT_FIELD_VERSION

    def __post_init__(self) -> None:
        if self.architecture_version != CORISP_ROLE_FILLER_EVENT_FIELD_VERSION:
            raise ValueError(
                f"Unsupported role-filler event field {self.architecture_version!r}."
            )
        if min(
            self.d_model,
            self.num_roles,
            self.max_cardinality,
            self.num_heads,
            self.signature_rank,
        ) <= 0:
            raise ValueError("Role-filler event-field dimensions must be positive.")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1).")
        if min(self.max_log_residual, self.max_signature_energy) <= 0.0:
            raise ValueError("Residual and signature bounds must be positive.")
        if 2**self.num_roles > 4096:
            raise ValueError("Exact role signatures support at most 12 roles.")

@dataclass(frozen=True)
class RoleFillerEvent:
    """One exact distribution for a detected person-predicate event."""

    agent_index: int
    action_index: int
    pair_indices: Tensor
    typed_role_indices: Tensor
    log_state_weights: Tensor
    composition_energy: Tensor
    log_composition_coefficients: Tensor
    log_partition: Tensor

    @property
    def num_visible_candidates(self) -> int:
        return int(self.pair_indices.numel())

    @property
    def num_candidates(self) -> int:
        return int(self.log_state_weights.shape[0])

    @property
    def num_roles(self) -> int:
        return int(self.log_state_weights.shape[1] - 1)

    def typed_candidate_row(self, role_index: int) -> int:
        match = torch.nonzero(
            self.typed_role_indices == int(role_index), as_tuple=False
        ).flatten()
        if match.numel() != 1:
            raise KeyError(
                f"Event ({self.agent_index},{self.action_index}) has no unique "
                f"typed-null candidate for role {role_index}."
            )
        return self.num_visible_candidates + int(match.item())

@dataclass(frozen=True)
class RoleFillerEventFieldOutput:
    events: tuple[RoleFillerEvent, ...]
    num_agents: int
    num_actions: int
    visible_shape: tuple[int, int, int]
    visible_role_marginals: Tensor | None = None
    typed_null_role_marginals: Tensor | None = None
    event_probabilities: Tensor | None = None
    cardinality_expectations: Tensor | None = None
    role_signature_probabilities: Tensor | None = None
    refinement_scale: Tensor | None = None
    signature_scale: Tensor | None = None

    def event(self, agent_index: int, action_index: int) -> RoleFillerEvent:
        if not 0 <= agent_index < self.num_agents:
            raise IndexError(agent_index)
        if not 0 <= action_index < self.num_actions:
            raise IndexError(action_index)
        return self.events[agent_index * self.num_actions + action_index]

class RoleFillerEventField(nn.Module):
    """A semantic role-filler field followed by one exact set likelihood."""

    _EVENT_RELATION = 0
    _PAIR_RELATION = 1
    _ENTITY_RELATION = 2

    @staticmethod
    def _validate_intervention(intervention: str) -> None:
        if intervention not in ROLE_FILLER_EVENT_FIELD_INTERVENTIONS:
            raise ValueError(
                f"Unsupported role-filler intervention {intervention!r}; expected "
                f"one of {ROLE_FILLER_EVENT_FIELD_INTERVENTIONS}."
            )

    @staticmethod
    def _validate_structural_ablation(structural_ablation: str) -> None:
        if structural_ablation not in ROLE_FILLER_EVENT_FIELD_STRUCTURAL_ABLATIONS:
            raise ValueError(
                f"Unsupported role-filler structural ablation "
                f"{structural_ablation!r}; expected one of "
                f"{ROLE_FILLER_EVENT_FIELD_STRUCTURAL_ABLATIONS}."
            )

    @staticmethod
    def _validate_execution_mode(mode: str) -> None:
        if mode not in _ROLE_FILLER_EVENT_FIELD_EXECUTION_MODES:
            raise ValueError(f"Unsupported role-filler execution mode {mode!r}.")

    def _resolve_execution_mode(
        self,
        *,
        intervention: str,
        structural_ablation: str,
    ) -> str:
        self._validate_intervention(intervention)
        self._validate_structural_ablation(structural_ablation)
        if intervention != "full" and structural_ablation != "full":
            raise ValueError(
                "An inference intervention cannot be combined with a trained "
                "structural ablation."
            )
        if self.training and intervention != "full":
            raise ValueError("Role-filler interventions are inference-only.")
        return (
            structural_ablation
            if structural_ablation != "full"
            else intervention
        )

    @staticmethod
    def _collapse_role_routing(weight: Tensor, support: Tensor) -> Tensor:
        """Preserve positive mass while removing source-role identity."""

        count = support.sum(dim=-1, keepdim=True).clamp_min(1)
        total = weight.sum(dim=-1, keepdim=True)
        collapsed = total / count.to(total.dtype)
        return collapsed.expand_as(weight).masked_fill(~support, 0.0)

    @staticmethod
    def _remove_relation(
        message: Tensor,
        available: Tensor,
        *,
        disabled: bool,
    ) -> tuple[Tensor, Tensor]:
        if not disabled:
            return message, available
        return torch.zeros_like(message), torch.zeros_like(available)

    def __init__(self, cfg: RoleFillerEventFieldConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Noun, predicate, and role anchors remain in one semantic coordinate
        # system.  State binding separates filler content from its role.
        self.anchor_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
        )
        self.filler_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
        )
        self.slot_norm = nn.LayerNorm(d)
        self.state_norm = nn.LayerNorm(d)

        # One attention operator is shared by all semantic roles and all three
        # structural relations.  Relation identity enters as an embedding.
        self.relation_embeddings = nn.Parameter(torch.empty(3, d))
        nn.init.normal_(self.relation_embeddings, std=d**-0.5)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
        )
        self.key_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
        )
        self.value_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
        )
        self.message_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
        )
        self.binding_norm = nn.LayerNorm(d)
        self.score_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # Zero is the exact unary identity.  A single scalar opens the whole
        # relational field, so it cannot become an independent prediction head.
        self.refinement_log_scale = nn.Parameter(torch.zeros(()))

        # Cardinality is retained from v8.  The role-signature factor is a
        # bounded event-conditioned DeepSets energy over the roles that occur.
        self.cardinality_norm = nn.LayerNorm(d)
        self.cardinality_head = nn.Linear(d, cfg.max_cardinality + 1)
        nn.init.zeros_(self.cardinality_head.weight)
        nn.init.zeros_(self.cardinality_head.bias)

        rank = cfg.signature_rank
        self.signature_atom_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, rank, bias=False),
        )
        self.signature_event_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, rank, bias=False),
        )
        self.signature_set_projection = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, rank, bias=False),
        )
        self.signature_log_scale = nn.Parameter(torch.zeros(()))

        signatures = 2**cfg.num_roles
        signature_id = torch.arange(signatures, dtype=torch.long)
        bit = torch.arange(cfg.num_roles, dtype=torch.long)
        membership = ((signature_id[:, None] >> bit[None]) & 1).bool()
        self.register_buffer("signature_membership", membership)
        self.register_buffer("signature_ids", signature_id)

    @staticmethod
    def _safe_logaddexp(first: Tensor, second: Tensor) -> Tensor:
        both_unreachable = torch.isneginf(first) & torch.isneginf(second)
        safe_first = first.masked_fill(both_unreachable, 0.0)
        safe_second = second.masked_fill(both_unreachable, 0.0)
        value = torch.logaddexp(safe_first, safe_second)
        return value.masked_fill(both_unreachable, float("-inf"))

    @staticmethod
    def _safe_logsumexp(value: Tensor, dim: int) -> Tensor:
        all_unreachable = torch.isneginf(value).all(dim=dim)
        safe = value.masked_fill(all_unreachable.unsqueeze(dim), 0.0)
        reduced = torch.logsumexp(safe, dim=dim)
        return reduced.masked_fill(all_unreachable, float("-inf"))

    @staticmethod
    def focal_transform_nll(nll: Tensor, gamma: float) -> Tensor:
        if gamma < 0:
            raise ValueError("Focal gamma must be non-negative.")
        if gamma == 0:
            return nll
        positive = nll > 0
        safe_nll = torch.where(positive, nll, torch.ones_like(nll))
        log_modulation = float(gamma) * torch.log(-torch.expm1(-safe_nll))
        transformed = safe_nll * torch.exp(log_modulation)
        return torch.where(positive, transformed, torch.zeros_like(nll))

    @staticmethod
    def _log_probability(probability: Tensor) -> Tensor:
        safe = probability.clamp_min(torch.finfo(torch.float32).tiny)
        value = safe.log()
        return value.masked_fill(probability <= 0, float("-inf"))

    @staticmethod
    def _safe_mean(numerator: Tensor, denominator: Tensor) -> Tensor:
        value = numerator / denominator[..., None].clamp_min(1e-6).to(
            numerator.dtype
        )
        return torch.where(
            denominator[..., None] > 0,
            value,
            torch.zeros_like(value),
        )

    @classmethod
    def log_composition_coefficients(cls, log_state_weights: Tensor) -> Tensor:
        """Return exact log mass indexed by ``(cardinality, role signature)``."""

        if log_state_weights.ndim != 2 or log_state_weights.shape[1] < 2:
            raise ValueError("log_state_weights must be [M,1+R].")
        candidates = int(log_state_weights.shape[0])
        roles = int(log_state_weights.shape[1] - 1)
        signatures = 2**roles
        coefficient = log_state_weights.new_full(
            (candidates + 1, signatures), float("-inf")
        )
        coefficient[0, 0] = 0.0

        signature_id = torch.arange(
            signatures, device=log_state_weights.device, dtype=torch.long
        )
        role_id = torch.arange(
            roles, device=log_state_weights.device, dtype=torch.long
        )
        contains = ((signature_id[None] >> role_id[:, None]) & 1).bool()
        without = signature_id[None] & ~(1 << role_id[:, None])

        for candidate in range(candidates):
            stay = coefficient + log_state_weights[candidate, 0]
            previous = coefficient[:-1]
            same_signature = previous[None].expand(roles, -1, -1)
            first_occurrence = torch.stack(
                [previous[:, without[role]] for role in range(roles)],
                dim=0,
            )
            source = cls._safe_logaddexp(
                same_signature, first_occurrence
            ).masked_fill(~contains[:, None], float("-inf"))
            take_by_role = source + log_state_weights[
                candidate, 1:
            ][:, None, None]
            take = cls._safe_logsumexp(take_by_role, dim=0)
            padded_take = torch.cat(
                [
                    coefficient.new_full((1, signatures), float("-inf")),
                    take,
                ],
                dim=0,
            )
            coefficient = cls._safe_logaddexp(stay, padded_take)
        return coefficient

    @classmethod
    def log_partition(
        cls,
        log_state_weights: Tensor,
        composition_energy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        candidates = int(log_state_weights.shape[0])
        roles = int(log_state_weights.shape[1] - 1)
        expected = (candidates + 1, 2**roles)
        if composition_energy.shape != expected:
            raise ValueError(f"composition_energy must have shape {expected}.")
        coefficient = cls.log_composition_coefficients(log_state_weights)
        log_z = cls._safe_logsumexp(
            (coefficient + composition_energy).reshape(-1), dim=0
        )
        return log_z, coefficient

    @staticmethod
    def _scatter_role_states(
        state: Tensor,
        weight: Tensor,
        index: Tensor,
        groups: int,
    ) -> tuple[Tensor, Tensor]:
        if state.ndim != 4 or weight.shape != state.shape[:3]:
            raise ValueError("Role-state scatter expects [N,A,R,D] and [N,A,R].")
        output = state.new_zeros(
            (groups, state.shape[1], state.shape[2], state.shape[3])
        )
        mass = weight.new_zeros((groups, weight.shape[1], weight.shape[2]))
        if state.shape[0]:
            output.index_add_(
                0,
                index,
                state * weight[..., None].to(state.dtype),
            )
            mass.index_add_(0, index, weight)
        return output, mass

    def _attend_source_roles(
        self,
        *,
        target_state: Tensor,
        base_context: Tensor,
        diagonal_context: Tensor,
        base_mass: Tensor,
        diagonal_mass: Tensor,
        role_key: Tensor,
        relation_index: int,
    ) -> tuple[Tensor, Tensor]:
        """Attend to source roles while replacing only the target-role diagonal."""

        if target_state.ndim != 4:
            raise ValueError("target_state must be [N,A,R,D].")
        if base_context.shape != target_state.shape:
            raise ValueError("base_context must align with target_state.")
        if diagonal_context.shape != target_state.shape:
            raise ValueError("diagonal_context must align with target_state.")
        if base_mass.shape != target_state.shape[:3]:
            raise ValueError("base_mass must be [N,A,R].")
        if diagonal_mass.shape != target_state.shape[:3]:
            raise ValueError("diagonal_mass must be [N,A,R].")

        n, actions, roles, d = target_state.shape
        relation = self.relation_embeddings[relation_index].to(target_state.dtype)
        source_semantic = role_key[None, None].to(target_state.dtype) + relation
        base_source = base_context + source_semantic
        diagonal_source = diagonal_context + source_semantic

        heads = self.cfg.num_heads
        head_dim = d // heads
        query = self.query_projection(target_state).reshape(
            n, actions, roles, heads, head_dim
        )
        key = self.key_projection(base_source).reshape(
            n, actions, roles, heads, head_dim
        )
        diagonal_key = self.key_projection(diagonal_source).reshape(
            n, actions, roles, heads, head_dim
        )
        score = torch.einsum(
            "narhd,nashd->narhs", query.float(), key.float()
        ) / sqrt(float(head_dim))
        diagonal_score = torch.einsum(
            "narhd,narhd->narh", query.float(), diagonal_key.float()
        )
        diagonal_score = diagonal_score / sqrt(float(head_dim))

        eye = torch.eye(roles, dtype=torch.bool, device=target_state.device)
        score = torch.where(
            eye[None, None, :, None, :],
            diagonal_score[..., None],
            score,
        )
        valid = base_mass[:, :, None, None, :].expand(
            n, actions, roles, heads, roles
        ) > 0
        valid = torch.where(
            eye[None, None, :, None, :],
            (diagonal_mass > 0)[:, :, :, None, None],
            valid,
        )
        all_invalid = ~valid.any(dim=-1)
        masked_score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        attention = torch.softmax(masked_score, dim=-1)
        attention = attention.masked_fill(all_invalid[..., None], 0.0)

        value = self.value_projection(base_source).reshape(
            n, actions, roles, heads, head_dim
        )
        diagonal_value = self.value_projection(diagonal_source).reshape(
            n, actions, roles, heads, head_dim
        )
        message = torch.einsum("narhs,nashd->narhd", attention, value)
        # The base contraction used the non-cavity value on the role diagonal.
        # Replace exactly that one source value without materializing R x R x D.
        for role in range(roles):
            diagonal_weight = attention[:, :, role, :, role]
            message[:, :, role] = message[:, :, role] + diagonal_weight[..., None] * (
                diagonal_value[:, :, role] - value[:, :, role]
            )
        message = message.reshape(n, actions, roles, d).to(target_state.dtype)
        available = ~all_invalid.all(dim=-1)
        return self.message_projection(message), available

    def _combine_relations(
        self,
        state: Tensor,
        messages: Sequence[Tensor],
        available: Sequence[Tensor],
    ) -> Tensor:
        if len(messages) != 3 or len(available) != 3:
            raise ValueError("Visible states require event, pair, and entity relations.")
        message = torch.stack(tuple(messages), dim=-2)
        valid = torch.stack(tuple(available), dim=-1)
        relation = self.relation_embeddings.to(state.dtype)
        query = self.query_projection(state)
        key = self.key_projection(message + relation[None, None, None])
        score = torch.einsum(
            "nard,narjd->narj", query.float(), key.float()
        ) / sqrt(float(self.cfg.d_model))
        all_invalid = ~valid.any(dim=-1)
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1)
        weight = weight.masked_fill(all_invalid[..., None], 0.0)
        return (message * weight[..., None].to(message.dtype)).sum(dim=-2)

    @staticmethod
    def _center_valid_roles(score: Tensor, valid_role_mask: Tensor) -> Tensor:
        valid = valid_role_mask.float()
        count = valid.sum(dim=-1).clamp_min(1.0)
        mean = (score.float() * valid).sum(dim=-1) / count
        return (score.float() - mean[..., None]) * valid

    def _state_delta(
        self,
        bound_state: Tensor,
        action_key: Tensor,
        slot_key: Tensor,
        valid_role_mask: Tensor,
    ) -> Tensor:
        scored = self.score_projection(self.dropout(bound_state))
        role_score = torch.einsum(
            "nard,ard->nar", scored.float(), slot_key.float()
        ) / sqrt(float(self.cfg.d_model))
        action_by_role = torch.einsum(
            "nard,ad->nar", scored.float(), action_key.float()
        ) / sqrt(float(self.cfg.d_model))
        valid = valid_role_mask.float()
        common = (action_by_role * valid[None]).sum(dim=-1)
        common = common / valid.sum(dim=-1).clamp_min(1.0)[None]
        return common[..., None] + self._center_valid_roles(
            role_score, valid_role_mask
        )

    def _refine_visible(
        self,
        base: Tensor,
        delta: Tensor,
        support: Tensor,
        scale: Tensor,
    ) -> Tensor:
        probability = base.float().masked_fill(~support, 0.0)
        positive_sum = probability.sum(dim=-1)
        if (probability < 0).any() or (positive_sum > 1.0 + 1e-5).any():
            raise ValueError("Visible role probabilities are not sub-probabilities.")
        null = (1.0 - positive_sum).clamp_min(0.0)
        factor = torch.exp(scale.float() * torch.tanh(delta.float()))
        positive = probability * factor.masked_fill(~support, 0.0)
        denominator = null + positive.sum(dim=-1)
        return (positive / denominator[..., None].clamp_min(1e-30)).masked_fill(
            ~support, 0.0
        )

    def _refine_typed(
        self,
        base: Tensor,
        delta: Tensor,
        valid_role_mask: Tensor,
        scale: Tensor,
    ) -> Tensor:
        probability = base.float().masked_fill(~valid_role_mask[None], 0.0)
        if (probability < 0).any() or (probability > 1.0 + 1e-5).any():
            raise ValueError("Typed-null probabilities must lie in [0,1].")
        probability = probability.clamp(0.0, 1.0)
        factor = torch.exp(scale.float() * torch.tanh(delta.float()))
        positive = probability * factor
        refined = positive / (1.0 - probability + positive).clamp_min(1e-30)
        return refined.masked_fill(~valid_role_mask[None], 0.0)

    def _relational_unaries(
        self,
        *,
        base_visible_role_probs: Tensor,
        base_typed_null_role_probs: Tensor,
        edge_states: Tensor,
        null_edge_states: Tensor,
        event_states: Tensor,
        pair_agent_indices: Tensor,
        pair_entity_indices: Tensor,
        noun_anchors: Tensor,
        action_anchors: Tensor,
        role_anchors: Tensor,
        visible_action_mask: Tensor,
        valid_role_mask: Tensor,
        intervention: str = "full",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._validate_execution_mode(intervention)
        q, actions, roles = base_visible_role_probs.shape
        humans = base_typed_null_role_probs.shape[0]
        d = self.cfg.d_model
        action_key = self.anchor_projection(action_anchors)
        role_key = self.anchor_projection(role_anchors)
        noun_key = self.anchor_projection(noun_anchors) if q else noun_anchors
        slot_key = self.slot_norm(
            action_key[:, None]
            + role_key[None]
            + action_key[:, None] * role_key[None]
        )

        visible_filler = self.filler_projection(
            edge_states + noun_key[:, None].to(edge_states.dtype)
        )
        visible_state = self.state_norm(
            visible_filler[:, :, None]
            + slot_key[None].to(edge_states.dtype)
            + visible_filler[:, :, None]
            * slot_key[None].to(edge_states.dtype)
        )
        typed_filler = self.filler_projection(null_edge_states)
        typed_state = self.state_norm(
            typed_filler[:, :, None]
            + slot_key[None].to(null_edge_states.dtype)
            + typed_filler[:, :, None]
            * slot_key[None].to(null_edge_states.dtype)
        )

        support = visible_action_mask[..., None] & valid_role_mask[None]
        visible_weight = base_visible_role_probs.detach().float().masked_fill(
            ~support, 0.0
        )
        typed_weight = base_typed_null_role_probs.detach().float().masked_fill(
            ~valid_role_mask[None], 0.0
        )
        if intervention == "role_collapsed_routing":
            visible_weight = self._collapse_role_routing(visible_weight, support)
            typed_support = valid_role_mask[None].expand_as(typed_weight)
            typed_weight = self._collapse_role_routing(
                typed_weight, typed_support
            )

        event_numerator, event_mass = self._scatter_role_states(
            visible_state,
            visible_weight,
            pair_agent_indices,
            humans,
        )
        event_numerator = event_numerator + typed_state * typed_weight[
            ..., None
        ].to(typed_state.dtype)
        event_mass = event_mass + typed_weight

        pair_numerator = (
            visible_state * visible_weight[..., None].to(visible_state.dtype)
        ).sum(dim=1)
        pair_mass = visible_weight.sum(dim=1)

        if q:
            entity_groups = int(pair_entity_indices.max().item()) + 1
            entity_numerator = visible_state.new_zeros(
                (entity_groups, roles, d)
            )
            entity_mass = visible_weight.new_zeros((entity_groups, roles))
            entity_numerator.index_add_(0, pair_entity_indices, pair_numerator)
            entity_mass.index_add_(0, pair_entity_indices, pair_mass)

            own_numerator = visible_state * visible_weight[..., None].to(
                visible_state.dtype
            )

            event_num = event_numerator[pair_agent_indices]
            event_den = event_mass[pair_agent_indices]
            event_base = self._safe_mean(event_num, event_den)
            event_diagonal = self._safe_mean(
                event_num - own_numerator,
                event_den - visible_weight,
            )

            pair_num = pair_numerator[:, None].expand(-1, actions, -1, -1)
            pair_den = pair_mass[:, None].expand(-1, actions, -1)
            pair_base = self._safe_mean(pair_num, pair_den)
            pair_diagonal = self._safe_mean(
                pair_num - own_numerator,
                pair_den - visible_weight,
            )

            entity_num = entity_numerator[pair_entity_indices, None].expand(
                -1, actions, -1, -1
            )
            entity_den = entity_mass[pair_entity_indices, None].expand(
                -1, actions, -1
            )
            entity_base = self._safe_mean(entity_num, entity_den)
            entity_diagonal = self._safe_mean(
                entity_num - own_numerator,
                entity_den - visible_weight,
            )

            event_message, event_available = self._attend_source_roles(
                target_state=visible_state,
                base_context=event_base,
                diagonal_context=event_diagonal,
                base_mass=event_den,
                diagonal_mass=event_den - visible_weight,
                role_key=role_key,
                relation_index=self._EVENT_RELATION,
            )
            pair_message, pair_available = self._attend_source_roles(
                target_state=visible_state,
                base_context=pair_base,
                diagonal_context=pair_diagonal,
                base_mass=pair_den,
                diagonal_mass=pair_den - visible_weight,
                role_key=role_key,
                relation_index=self._PAIR_RELATION,
            )
            entity_message, entity_available = self._attend_source_roles(
                target_state=visible_state,
                base_context=entity_base,
                diagonal_context=entity_diagonal,
                base_mass=entity_den,
                diagonal_mass=entity_den - visible_weight,
                role_key=role_key,
                relation_index=self._ENTITY_RELATION,
            )
            disable_all_relations = intervention in {
                "unary_only",
                "local_unary",
                "cardinality_only",
                "local_cardinality",
                "local_signature",
                "no_all_relations",
            }
            event_message, event_available = self._remove_relation(
                event_message,
                event_available,
                disabled=intervention == "no_event_relation"
                or disable_all_relations,
            )
            pair_message, pair_available = self._remove_relation(
                pair_message,
                pair_available,
                disabled=intervention == "no_pair_relation"
                or disable_all_relations,
            )
            entity_message, entity_available = self._remove_relation(
                entity_message,
                entity_available,
                disabled=intervention == "no_entity_relation"
                or disable_all_relations,
            )
            visible_message = self._combine_relations(
                visible_state,
                (event_message, pair_message, entity_message),
                (event_available, pair_available, entity_available),
            )
        else:
            visible_message = visible_state.new_zeros(visible_state.shape)

        typed_own = typed_state * typed_weight[..., None].to(typed_state.dtype)
        typed_event_base = self._safe_mean(event_numerator, event_mass)
        typed_event_diagonal = self._safe_mean(
            event_numerator - typed_own,
            event_mass - typed_weight,
        )
        typed_message, _ = self._attend_source_roles(
            target_state=typed_state,
            base_context=typed_event_base,
            diagonal_context=typed_event_diagonal,
            base_mass=event_mass,
            diagonal_mass=event_mass - typed_weight,
            role_key=role_key,
            relation_index=self._EVENT_RELATION,
        )
        if intervention == "no_event_relation" or intervention in {
            "unary_only",
            "local_unary",
            "cardinality_only",
            "local_cardinality",
            "local_signature",
            "no_all_relations",
        }:
            typed_message = torch.zeros_like(typed_message)

        visible_bound = self.binding_norm(
            visible_state
            + visible_message
            + visible_state * visible_message
        )
        typed_bound = self.binding_norm(
            typed_state + typed_message + typed_state * typed_message
        )
        visible_delta = self._state_delta(
            visible_bound, action_key, slot_key, valid_role_mask
        )
        typed_delta = self._state_delta(
            typed_bound, action_key, slot_key, valid_role_mask
        )
        scale = self.cfg.max_log_residual * torch.tanh(
            self.refinement_log_scale.float()
        )
        if intervention in {
            "unary_only",
            "cardinality_only",
            "no_unary_refinement",
        }:
            scale = torch.zeros_like(scale)
        refined_visible = self._refine_visible(
            base_visible_role_probs, visible_delta, support, scale
        )
        refined_typed = self._refine_typed(
            base_typed_null_role_probs,
            typed_delta,
            valid_role_mask,
            scale,
        )

        # The role-signature factor reads the same bound states that produced
        # the unaries; it is not an auxiliary branch with separate evidence.
        bound_numerator, bound_mass = self._scatter_role_states(
            visible_bound,
            visible_weight,
            pair_agent_indices,
            humans,
        )
        bound_numerator = bound_numerator + typed_bound * typed_weight[
            ..., None
        ].to(typed_bound.dtype)
        bound_mass = bound_mass + typed_weight
        event_role_context = self._safe_mean(bound_numerator, bound_mass)
        event_role_context = self.state_norm(
            event_role_context
            + event_states[:, :, None]
            + slot_key[None].to(event_role_context.dtype)
        )
        return refined_visible, refined_typed, event_role_context, scale

    def _composition_energy(
        self,
        event_states: Tensor,
        event_role_context: Tensor,
        intervention: str = "full",
    ) -> tuple[Tensor, Tensor]:
        self._validate_execution_mode(intervention)
        humans, actions, roles, _ = event_role_context.shape
        if roles != self.cfg.num_roles:
            raise ValueError("Event role context has the wrong role count.")
        cardinality = self.cardinality_head(
            self.cardinality_norm(event_states.float())
        )
        cardinality = cardinality - cardinality[..., :1]
        if intervention in {
            "unary_only",
            "local_unary",
            "local_signature",
            "relational_signature",
            "no_cardinality",
        }:
            cardinality = torch.zeros_like(cardinality)

        atom = self.signature_atom_projection(event_role_context.float())
        membership = self.signature_membership.to(atom.dtype)
        pooled = torch.einsum("sr,hard->hasd", membership, atom)
        count = membership.sum(dim=-1).clamp_min(1.0).sqrt()
        pooled = pooled / count[None, None, :, None]
        signature_key = self.signature_set_projection(pooled)
        event_query = self.signature_event_projection(event_states.float())
        raw_signature = torch.einsum(
            "had,hasd->has", event_query, signature_key
        ) / sqrt(float(self.cfg.signature_rank))
        signature_scale = self.cfg.max_signature_energy * torch.tanh(
            self.signature_log_scale.float()
        )
        if intervention in {
            "unary_only",
            "local_unary",
            "cardinality_only",
            "local_cardinality",
            "relational_cardinality",
            "no_signature",
        }:
            signature_scale = torch.zeros_like(signature_scale)
        signature = signature_scale * torch.tanh(raw_signature)
        signature = signature.masked_fill(
            ~self.signature_membership.any(dim=-1)[None, None], 0.0
        )
        return cardinality[..., :, None] + signature[..., None, :], signature_scale

    @staticmethod
    def _validate_inputs(
        *,
        base_visible_role_probs: Tensor,
        base_typed_null_role_probs: Tensor,
        edge_states: Tensor,
        null_edge_states: Tensor,
        event_states: Tensor,
        pair_agent_indices: Tensor,
        pair_entity_indices: Tensor,
        noun_anchors: Tensor,
        action_anchors: Tensor,
        role_anchors: Tensor,
        visible_action_mask: Tensor,
        valid_role_mask: Tensor,
        d_model: int,
    ) -> tuple[int, int, int, int]:
        if base_visible_role_probs.ndim != 3:
            raise ValueError("base_visible_role_probs must be [Q,A,R].")
        q, actions, roles = base_visible_role_probs.shape
        if base_typed_null_role_probs.ndim != 3:
            raise ValueError("base_typed_null_role_probs must be [H,A,R].")
        humans = int(base_typed_null_role_probs.shape[0])
        expected = {
            "base_typed_null_role_probs": (humans, actions, roles),
            "edge_states": (q, actions, d_model),
            "null_edge_states": (humans, actions, d_model),
            "event_states": (humans, actions, d_model),
            "pair_agent_indices": (q,),
            "pair_entity_indices": (q,),
            "noun_anchors": (q, d_model),
            "action_anchors": (actions, d_model),
            "role_anchors": (roles, d_model),
            "visible_action_mask": (q, actions),
            "valid_role_mask": (actions, roles),
        }
        values = {
            "base_typed_null_role_probs": base_typed_null_role_probs,
            "edge_states": edge_states,
            "null_edge_states": null_edge_states,
            "event_states": event_states,
            "pair_agent_indices": pair_agent_indices,
            "pair_entity_indices": pair_entity_indices,
            "noun_anchors": noun_anchors,
            "action_anchors": action_anchors,
            "role_anchors": role_anchors,
            "visible_action_mask": visible_action_mask,
            "valid_role_mask": valid_role_mask,
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(values[name].shape)}.")
        if pair_agent_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("pair_agent_indices must be integer.")
        if pair_entity_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("pair_entity_indices must be integer.")
        if q and (
            (pair_agent_indices < 0).any()
            or (pair_agent_indices >= humans).any()
            or (pair_entity_indices < 0).any()
        ):
            raise ValueError("Pair indices reference an invalid agent or entity.")
        if visible_action_mask.dtype != torch.bool or valid_role_mask.dtype != torch.bool:
            raise ValueError("Action and role masks must be boolean.")
        if not valid_role_mask.any(dim=-1).all():
            raise ValueError("Every action must admit at least one role.")
        for name, value in values.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
        return q, humans, actions, roles

    def _build_event(
        self,
        *,
        agent_index: int,
        action_index: int,
        pair_indices: Tensor,
        visible_probability: Tensor,
        typed_probability: Tensor,
        valid_roles: Tensor,
        composition_energy: Tensor,
    ) -> RoleFillerEvent:
        roles = self.cfg.num_roles
        visible_probability = visible_probability.float().masked_fill(
            ~valid_roles[None], 0.0
        )
        visible_null = 1.0 - visible_probability.sum(dim=-1)
        if (visible_probability < 0).any() or (visible_null < -1e-6).any():
            raise ValueError("Visible role probabilities do not form sub-probabilities.")
        visible_states = torch.cat(
            [visible_null.clamp_min(0.0)[:, None], visible_probability], dim=-1
        )

        typed_roles = torch.nonzero(valid_roles, as_tuple=False).flatten()
        typed_states = visible_states.new_zeros((typed_roles.numel(), roles + 1))
        if typed_roles.numel():
            typed_values = typed_probability.float()[typed_roles]
            if (typed_values < 0).any() or (typed_values > 1.0 + 1e-6).any():
                raise ValueError("Typed-null probabilities must lie in [0,1].")
            typed_values = typed_values.clamp(0.0, 1.0)
            typed_states[:, 0] = 1.0 - typed_values
            typed_states[
                torch.arange(typed_roles.numel(), device=typed_roles.device),
                typed_roles + 1,
            ] = typed_values

        probability = torch.cat([visible_states, typed_states], dim=0)
        if probability.shape[0] > self.cfg.max_cardinality:
            raise ValueError(
                f"Event has {probability.shape[0]} candidates, exceeding "
                f"max_cardinality={self.cfg.max_cardinality}."
            )
        log_states = self._log_probability(probability)
        energy = composition_energy[: probability.shape[0] + 1]
        log_z, coefficient = self.log_partition(log_states, energy)
        return RoleFillerEvent(
            agent_index=int(agent_index),
            action_index=int(action_index),
            pair_indices=pair_indices,
            typed_role_indices=typed_roles,
            log_state_weights=log_states,
            composition_energy=energy,
            log_composition_coefficients=coefficient,
            log_partition=log_z,
        )

    def forward(
        self,
        *,
        base_visible_role_probs: Tensor,
        base_typed_null_role_probs: Tensor,
        edge_states: Tensor,
        null_edge_states: Tensor,
        event_states: Tensor,
        pair_agent_indices: Tensor,
        pair_entity_indices: Tensor,
        noun_anchors: Tensor,
        action_anchors: Tensor,
        role_anchors: Tensor,
        visible_action_mask: Tensor,
        valid_role_mask: Tensor,
        return_marginals: bool = False,
        intervention: str = "full",
        structural_ablation: str = "full",
    ) -> RoleFillerEventFieldOutput:
        execution_mode = self._resolve_execution_mode(
            intervention=intervention,
            structural_ablation=structural_ablation,
        )
        q, humans, actions, roles = self._validate_inputs(
            base_visible_role_probs=base_visible_role_probs,
            base_typed_null_role_probs=base_typed_null_role_probs,
            edge_states=edge_states,
            null_edge_states=null_edge_states,
            event_states=event_states,
            pair_agent_indices=pair_agent_indices,
            pair_entity_indices=pair_entity_indices,
            noun_anchors=noun_anchors,
            action_anchors=action_anchors,
            role_anchors=role_anchors,
            visible_action_mask=visible_action_mask,
            valid_role_mask=valid_role_mask,
            d_model=self.cfg.d_model,
        )
        if roles != self.cfg.num_roles:
            raise ValueError("Input role count does not match the field config.")
        visible, typed, event_role_context, refinement_scale = (
            self._relational_unaries(
                base_visible_role_probs=base_visible_role_probs,
                base_typed_null_role_probs=base_typed_null_role_probs,
                edge_states=edge_states,
                null_edge_states=null_edge_states,
                event_states=event_states,
                pair_agent_indices=pair_agent_indices,
                pair_entity_indices=pair_entity_indices,
                noun_anchors=noun_anchors,
                action_anchors=action_anchors,
                role_anchors=role_anchors,
                visible_action_mask=visible_action_mask,
                valid_role_mask=valid_role_mask,
                intervention=execution_mode,
            )
        )
        composition, signature_scale = self._composition_energy(
            event_states, event_role_context, intervention=execution_mode
        )

        events: list[RoleFillerEvent] = []
        for human_index in range(humans):
            human_pairs = pair_agent_indices == human_index
            for action_index in range(actions):
                pair_indices = torch.nonzero(
                    human_pairs & visible_action_mask[:, action_index],
                    as_tuple=False,
                ).flatten()
                events.append(
                    self._build_event(
                        agent_index=human_index,
                        action_index=action_index,
                        pair_indices=pair_indices,
                        visible_probability=visible[pair_indices, action_index],
                        typed_probability=typed[human_index, action_index],
                        valid_roles=valid_role_mask[action_index],
                        composition_energy=composition[human_index, action_index],
                    )
                )

        output = RoleFillerEventFieldOutput(
            events=tuple(events),
            num_agents=humans,
            num_actions=actions,
            visible_shape=(q, actions, roles),
            refinement_scale=refinement_scale,
            signature_scale=signature_scale,
        )
        return self._with_exact_marginals(output) if return_marginals else output

    def _with_exact_marginals(
        self,
        output: RoleFillerEventFieldOutput,
    ) -> RoleFillerEventFieldOutput:
        q, actions, roles = output.visible_shape
        signatures = 2**roles
        reference = self.cardinality_head.weight
        if not output.events:
            return RoleFillerEventFieldOutput(
                events=output.events,
                num_agents=output.num_agents,
                num_actions=output.num_actions,
                visible_shape=output.visible_shape,
                visible_role_marginals=reference.new_zeros((q, actions, roles)),
                typed_null_role_marginals=reference.new_zeros(
                    (output.num_agents, actions, roles)
                ),
                event_probabilities=reference.new_zeros(
                    (output.num_agents, actions)
                ),
                cardinality_expectations=reference.new_zeros(
                    (output.num_agents, actions)
                ),
                role_signature_probabilities=reference.new_zeros(
                    (output.num_agents, actions, signatures)
                ),
                refinement_scale=output.refinement_scale,
                signature_scale=output.signature_scale,
            )

        visible = reference.new_zeros((q, actions, roles))
        typed = reference.new_zeros((output.num_agents, actions, roles))
        event_probability = reference.new_zeros((output.num_agents, actions))
        expectation = reference.new_zeros((output.num_agents, actions))
        signature_probability = reference.new_zeros(
            (output.num_agents, actions, signatures)
        )

        with torch.enable_grad():
            leaves: list[Tensor] = []
            log_partitions: list[Tensor] = []
            detached_energy: list[Tensor] = []
            detached_coefficients: list[Tensor] = []
            for event in output.events:
                leaf = event.log_state_weights.detach().requires_grad_(True)
                energy = event.composition_energy.detach()
                log_z, coefficient = self.log_partition(leaf, energy)
                leaves.append(leaf)
                log_partitions.append(log_z)
                detached_energy.append(energy)
                detached_coefficients.append(coefficient.detach())
            gradients = torch.autograd.grad(
                torch.stack(log_partitions).sum(), leaves
            )

        for event, marginal, log_z, energy, coefficient in zip(
            output.events,
            gradients,
            log_partitions,
            detached_energy,
            detached_coefficients,
        ):
            marginal = marginal.detach().clamp(0.0, 1.0)
            visible_count = event.num_visible_candidates
            visible[event.pair_indices, event.action_index] = marginal[
                :visible_count, 1:
            ]
            for offset, role_index in enumerate(event.typed_role_indices.tolist()):
                typed[event.agent_index, event.action_index, role_index] = marginal[
                    visible_count + offset, role_index + 1
                ]

            log_all_null = (
                event.log_state_weights[:, 0].detach().sum() + energy[0, 0]
            )
            event_probability[event.agent_index, event.action_index] = (
                1.0 - torch.exp(log_all_null - log_z.detach())
            ).clamp(0.0, 1.0)
            posterior = torch.softmax((coefficient + energy).reshape(-1), dim=0)
            posterior = posterior.reshape_as(coefficient)
            count = torch.arange(
                event.num_candidates + 1,
                device=posterior.device,
                dtype=posterior.dtype,
            )
            expectation[event.agent_index, event.action_index] = (
                posterior.sum(dim=-1) * count
            ).sum()
            signature_probability[event.agent_index, event.action_index] = (
                posterior.sum(dim=0)
            )

        return RoleFillerEventFieldOutput(
            events=output.events,
            num_agents=output.num_agents,
            num_actions=output.num_actions,
            visible_shape=output.visible_shape,
            visible_role_marginals=visible,
            typed_null_role_marginals=typed,
            event_probabilities=event_probability,
            cardinality_expectations=expectation,
            role_signature_probabilities=signature_probability,
            refinement_scale=output.refinement_scale,
            signature_scale=output.signature_scale,
        )

    @staticmethod
    def target_log_mass(
        event: RoleFillerEvent,
        groups: Sequence[RoleFillerTargetGroup],
    ) -> Tensor | None:
        if not groups:
            return (
                event.log_state_weights[:, 0].sum()
                + event.composition_energy[0, 0]
            )

        choices: list[list[int]] = []
        role_signature = 0
        for group in groups:
            rows = group.candidate_rows
            if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64):
                raise ValueError("Target candidate rows must be an integer vector.")
            if rows.numel() == 0:
                return None
            if (rows < 0).any() or (rows >= event.num_candidates).any():
                raise ValueError("A target group references an invalid candidate row.")
            if not 0 <= int(group.role_index) < event.num_roles:
                raise ValueError("A target group references an invalid role.")
            choices.append([int(value) for value in rows.tolist()])
            role_signature |= 1 << int(group.role_index)

        assignments: list[Tensor] = []
        row_index = torch.arange(
            event.num_candidates, device=event.log_state_weights.device
        )
        for selected in product(*choices):
            if len(set(selected)) != len(selected):
                continue
            state = torch.zeros(
                event.num_candidates,
                dtype=torch.long,
                device=event.log_state_weights.device,
            )
            for row, group in zip(selected, groups):
                state[row] = int(group.role_index) + 1
            value = event.log_state_weights[row_index, state].sum()
            value = value + event.composition_energy[
                len(groups), role_signature
            ]
            if torch.isfinite(value):
                assignments.append(value)
        if not assignments:
            return None
        return torch.logsumexp(torch.stack(assignments), dim=0)
