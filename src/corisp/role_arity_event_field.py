from __future__ import annotations

from dataclasses import dataclass

from itertools import product

from math import sqrt

from typing import Sequence

import torch

import torch.nn as nn

from torch import Tensor

from corisp.grounded_role_set import RoleFillerTargetGroup

from corisp.role_filler_event_field import RoleFillerEventField, RoleFillerEventFieldConfig

CORISP_ROLE_ARITY_EVENT_FIELD_VERSION = "corisp_joint_v11_role_arity_event_field_v1"

@dataclass(frozen=True)
class RoleArityEventFieldConfig:
    d_model: int = 384
    num_roles: int = 2
    max_cardinality: int = 32
    num_heads: int = 8
    arity_rank: int = 64
    dropout: float = 0.1
    max_log_residual: float = 4.0
    max_arity_energy: float = 4.0
    max_arity_states: int = 8192
    architecture_version: str = CORISP_ROLE_ARITY_EVENT_FIELD_VERSION

    def __post_init__(self) -> None:
        if self.architecture_version != CORISP_ROLE_ARITY_EVENT_FIELD_VERSION:
            raise ValueError(
                f"Unsupported role-arity event field {self.architecture_version!r}."
            )
        if min(
            self.d_model,
            self.num_roles,
            self.max_cardinality,
            self.num_heads,
            self.arity_rank,
            self.max_arity_states,
        ) <= 0:
            raise ValueError("Role-arity event-field dimensions must be positive.")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1).")
        if min(self.max_log_residual, self.max_arity_energy) <= 0.0:
            raise ValueError("Residual and arity bounds must be positive.")
        if 3**self.num_roles > self.max_arity_states:
            raise ValueError(
                "The exact 0/1/2+ role-arity state space exceeds "
                f"max_arity_states={self.max_arity_states}."
            )

@dataclass(frozen=True)
class RoleArityEvent:
    """One normalized role-filler multiset for a person-predicate event."""

    agent_index: int
    action_index: int
    pair_indices: Tensor
    typed_role_indices: Tensor
    active_role_indices: Tensor
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

    @property
    def num_arity_states(self) -> int:
        return int(self.composition_energy.shape[1])

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
class RoleArityEventFieldOutput:
    events: tuple[RoleArityEvent, ...]
    num_agents: int
    num_actions: int
    visible_shape: tuple[int, int, int]
    visible_role_marginals: Tensor | None = None
    typed_null_role_marginals: Tensor | None = None
    event_probabilities: Tensor | None = None
    cardinality_expectations: Tensor | None = None
    role_arity_probabilities: Tensor | None = None
    capped_role_count_expectations: Tensor | None = None
    refinement_scale: Tensor | None = None
    arity_scale: Tensor | None = None

    def event(self, agent_index: int, action_index: int) -> RoleArityEvent:
        if not 0 <= agent_index < self.num_agents:
            raise IndexError(agent_index)
        if not 0 <= action_index < self.num_actions:
            raise IndexError(action_index)
        dense_index = agent_index * self.num_actions + action_index
        if len(self.events) == self.num_agents * self.num_actions:
            return self.events[dense_index]
        for event in self.events:
            if (
                event.agent_index == agent_index
                and event.action_index == action_index
            ):
                return event
        raise KeyError((agent_index, action_index))

class RoleArityEventField(RoleFillerEventField):
    """Role-preserving unaries with exact event-level role arity."""

    _ARITY_BASE = 3

    def __init__(self, cfg: RoleArityEventFieldConfig) -> None:
        # Share role-context operations; use saturated count embeddings for
        # composition instead of the base class's presence-signature head.
        super().__init__(
            RoleFillerEventFieldConfig(
                d_model=cfg.d_model,
                num_roles=cfg.num_roles,
                max_cardinality=cfg.max_cardinality,
                num_heads=cfg.num_heads,
                signature_rank=cfg.arity_rank,
                dropout=cfg.dropout,
                max_log_residual=cfg.max_log_residual,
                max_signature_energy=cfg.max_arity_energy,
            )
        )
        for name in (
            "signature_atom_projection",
            "signature_event_projection",
            "signature_set_projection",
            "signature_log_scale",
            "signature_membership",
            "signature_ids",
        ):
            delattr(self, name)
        self.cfg = cfg

        rank = cfg.arity_rank
        self.arity_role_projection = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, rank, bias=False),
        )
        self.arity_count_embeddings = nn.Parameter(torch.empty(3, rank))
        nn.init.normal_(self.arity_count_embeddings, std=rank**-0.5)
        self.arity_binding_norm = nn.LayerNorm(rank)
        self.arity_set_projection = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, rank, bias=False),
        )
        self.arity_event_projection = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, rank, bias=False),
        )
        self.arity_log_scale = nn.Parameter(torch.zeros(()))

    @classmethod
    def arity_digits(cls, num_active_roles: int, device: torch.device) -> Tensor:
        if num_active_roles <= 0:
            raise ValueError("An event must admit at least one semantic role.")
        states = cls._ARITY_BASE**num_active_roles
        code = torch.arange(states, device=device, dtype=torch.long)
        power = cls._ARITY_BASE ** torch.arange(
            num_active_roles, device=device, dtype=torch.long
        )
        return (code[:, None] // power[None]) % cls._ARITY_BASE

    @classmethod
    def encode_arity_counts(cls, counts: Sequence[int]) -> int:
        code = 0
        power = 1
        for count in counts:
            if int(count) < 0:
                raise ValueError("Role counts cannot be negative.")
            code += min(int(count), 2) * power
            power *= cls._ARITY_BASE
        return code

    @classmethod
    def log_arity_coefficients(
        cls,
        log_state_weights: Tensor,
        active_role_indices: Tensor,
    ) -> Tensor:
        """Return exact log mass indexed by total count and role arity code."""

        if log_state_weights.ndim != 2 or log_state_weights.shape[1] < 2:
            raise ValueError("log_state_weights must be [M,1+R].")
        if active_role_indices.ndim != 1 or active_role_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("active_role_indices must be an integer vector.")
        roles = int(log_state_weights.shape[1] - 1)
        if active_role_indices.numel() == 0:
            raise ValueError("At least one role must be active.")
        if (
            (active_role_indices < 0).any()
            or (active_role_indices >= roles).any()
            or active_role_indices.unique().numel() != active_role_indices.numel()
        ):
            raise ValueError("Active role indices must be unique and in range.")

        candidates = int(log_state_weights.shape[0])
        active = int(active_role_indices.numel())
        states = cls._ARITY_BASE**active
        coefficient = log_state_weights.new_full(
            (candidates + 1, states), float("-inf")
        )
        coefficient[0, 0] = 0.0

        code = torch.arange(states, device=log_state_weights.device)
        powers = cls._ARITY_BASE ** torch.arange(
            active, device=log_state_weights.device, dtype=torch.long
        )
        digits = (code[:, None] // powers[None]) % cls._ARITY_BASE

        for candidate in range(candidates):
            stay = coefficient + log_state_weights[candidate, 0]
            previous = coefficient[:-1]
            take_by_role: list[Tensor] = []
            for local_role, global_role in enumerate(active_role_indices.tolist()):
                digit = digits[:, local_role]
                reachable = digit > 0
                predecessor = torch.where(
                    reachable, code - powers[local_role], torch.zeros_like(code)
                )
                source = previous[:, predecessor]
                # Destination state 2+ receives transitions from both one and
                # the already-saturated 2+ state.
                saturated_source = previous[:, code]
                source = torch.where(
                    (digit == 2)[None],
                    cls._safe_logaddexp(source, saturated_source),
                    source,
                )
                source = source.masked_fill(~reachable[None], float("-inf"))
                take_by_role.append(
                    source + log_state_weights[candidate, int(global_role) + 1]
                )
            take = cls._safe_logsumexp(torch.stack(take_by_role), dim=0)
            padded_take = torch.cat(
                [
                    coefficient.new_full((1, states), float("-inf")),
                    take,
                ],
                dim=0,
            )
            coefficient = cls._safe_logaddexp(stay, padded_take)
        return coefficient

    @classmethod
    def log_arity_partition(
        cls,
        log_state_weights: Tensor,
        active_role_indices: Tensor,
        composition_energy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        coefficient = cls.log_arity_coefficients(
            log_state_weights, active_role_indices
        )
        if composition_energy.shape != coefficient.shape:
            raise ValueError(
                "composition_energy must align with exact arity coefficients; "
                f"got {tuple(composition_energy.shape)} and "
                f"{tuple(coefficient.shape)}."
            )
        return torch.logsumexp(
            (coefficient + composition_energy).reshape(-1), dim=0
        ), coefficient

    def _event_composition_energy(
        self,
        *,
        event_state: Tensor,
        event_role_context: Tensor,
        valid_roles: Tensor,
        num_candidates: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        active_roles = torch.nonzero(valid_roles, as_tuple=False).flatten()
        digits = self.arity_digits(int(active_roles.numel()), event_state.device)

        cardinality = self.cardinality_head(
            self.cardinality_norm(event_state.float())
        )
        cardinality = cardinality - cardinality[:1]
        cardinality = cardinality[: num_candidates + 1]

        role_atom = self.arity_role_projection(
            event_role_context[active_roles].float()
        )
        count_atom = self.arity_count_embeddings.float()[digits]
        bound = self.arity_binding_norm(
            role_atom[None]
            + count_atom
            + role_atom[None] * count_atom
        )
        pooled = bound.sum(dim=1) / sqrt(float(active_roles.numel()))
        key = self.arity_set_projection(pooled)
        query = self.arity_event_projection(event_state.float())
        raw = torch.einsum("d,sd->s", query, key) / sqrt(
            float(self.cfg.arity_rank)
        )
        raw = raw - raw[:1]
        scale = self.cfg.max_arity_energy * torch.tanh(
            self.arity_log_scale.float()
        )
        arity = scale * torch.tanh(raw)
        return cardinality[:, None] + arity[None], active_roles, scale

    def _build_arity_event(
        self,
        *,
        agent_index: int,
        action_index: int,
        pair_indices: Tensor,
        visible_probability: Tensor,
        typed_probability: Tensor,
        valid_roles: Tensor,
        typed_null_roles: Tensor,
        event_state: Tensor,
        event_role_context: Tensor,
    ) -> tuple[RoleArityEvent, Tensor]:
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

        if typed_null_roles.shape != valid_roles.shape:
            raise ValueError("typed_null_roles must match valid_roles.")
        if (typed_null_roles & ~valid_roles).any():
            raise ValueError("Typed-null roles must be a subset of valid roles.")
        typed_roles = torch.nonzero(typed_null_roles, as_tuple=False).flatten()
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
        composition, active_roles, scale = self._event_composition_energy(
            event_state=event_state,
            event_role_context=event_role_context,
            valid_roles=valid_roles,
            num_candidates=int(probability.shape[0]),
        )
        log_states = self._log_probability(probability)
        log_z, coefficient = self.log_arity_partition(
            log_states, active_roles, composition
        )
        return (
            RoleArityEvent(
                agent_index=int(agent_index),
                action_index=int(action_index),
                pair_indices=pair_indices,
                typed_role_indices=typed_roles,
                active_role_indices=active_roles,
                log_state_weights=log_states,
                composition_energy=composition,
                log_composition_coefficients=coefficient,
                log_partition=log_z,
            ),
            scale,
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
        typed_null_role_mask: Tensor | None = None,
        omit_deterministic_null_events: bool = False,
        return_marginals: bool = False,
    ) -> RoleArityEventFieldOutput:
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
        if typed_null_role_mask is None:
            typed_null_role_mask = valid_role_mask
        if typed_null_role_mask.shape != valid_role_mask.shape:
            raise ValueError(
                "typed_null_role_mask must have the same shape as valid_role_mask."
            )
        if typed_null_role_mask.dtype != torch.bool:
            raise ValueError("typed_null_role_mask must be boolean.")
        if (typed_null_role_mask & ~valid_role_mask).any():
            raise ValueError(
                "typed_null_role_mask must be a subset of valid_role_mask."
            )
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
            )
        )

        events: list[RoleArityEvent] = []
        arity_scale = self.arity_log_scale.new_zeros(())
        for human_index in range(humans):
            human_pairs = pair_agent_indices == human_index
            for action_index in range(actions):
                pair_indices = torch.nonzero(
                    human_pairs & visible_action_mask[:, action_index],
                    as_tuple=False,
                ).flatten()
                if (
                    omit_deterministic_null_events
                    and pair_indices.numel() == 0
                    and not bool(typed_null_role_mask[action_index].any())
                ):
                    continue
                event, arity_scale = self._build_arity_event(
                    agent_index=human_index,
                    action_index=action_index,
                    pair_indices=pair_indices,
                    visible_probability=visible[pair_indices, action_index],
                    typed_probability=typed[human_index, action_index],
                    valid_roles=valid_role_mask[action_index],
                    typed_null_roles=typed_null_role_mask[action_index],
                    event_state=event_states[human_index, action_index],
                    event_role_context=event_role_context[
                        human_index, action_index
                    ],
                )
                events.append(event)

        output = RoleArityEventFieldOutput(
            events=tuple(events),
            num_agents=humans,
            num_actions=actions,
            visible_shape=(q, actions, roles),
            refinement_scale=refinement_scale,
            arity_scale=arity_scale,
        )
        return self._with_arity_marginals(output) if return_marginals else output

    def _with_arity_marginals(
        self,
        output: RoleArityEventFieldOutput,
    ) -> RoleArityEventFieldOutput:
        q, actions, roles = output.visible_shape
        reference = self.cardinality_head.weight
        if not output.events:
            arity = reference.new_zeros((output.num_agents, actions, roles, 3))
            arity[..., 0] = 1.0
            return RoleArityEventFieldOutput(
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
                role_arity_probabilities=arity,
                capped_role_count_expectations=reference.new_zeros(
                    (output.num_agents, actions, roles)
                ),
                refinement_scale=output.refinement_scale,
                arity_scale=output.arity_scale,
            )

        visible = reference.new_zeros((q, actions, roles))
        typed = reference.new_zeros((output.num_agents, actions, roles))
        event_probability = reference.new_zeros((output.num_agents, actions))
        expectation = reference.new_zeros((output.num_agents, actions))
        arity_probability = reference.new_zeros(
            (output.num_agents, actions, roles, 3)
        )
        arity_probability[..., 0] = 1.0
        capped_count = reference.new_zeros((output.num_agents, actions, roles))

        with torch.enable_grad():
            leaves: list[Tensor] = []
            log_partitions: list[Tensor] = []
            detached_energy: list[Tensor] = []
            detached_coefficients: list[Tensor] = []
            for event in output.events:
                leaf = event.log_state_weights.detach().requires_grad_(True)
                energy = event.composition_energy.detach()
                log_z, coefficient = self.log_arity_partition(
                    leaf, event.active_role_indices, energy
                )
                leaves.append(leaf)
                log_partitions.append(log_z)
                detached_energy.append(energy)
                detached_coefficients.append(coefficient.detach())
            raw_gradients = torch.autograd.grad(
                torch.stack(log_partitions).sum(),
                leaves,
                allow_unused=True,
            )
            gradients = tuple(
                torch.zeros_like(leaf) if gradient is None else gradient
                for leaf, gradient in zip(leaves, raw_gradients)
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

            code_probability = posterior.sum(dim=0)
            digits = self.arity_digits(
                int(event.active_role_indices.numel()), posterior.device
            )
            for local_role, global_role in enumerate(
                event.active_role_indices.tolist()
            ):
                distribution = torch.stack(
                    [
                        code_probability[digits[:, local_role] == level].sum()
                        for level in range(3)
                    ]
                )
                arity_probability[
                    event.agent_index, event.action_index, global_role
                ] = distribution
                capped_count[
                    event.agent_index, event.action_index, global_role
                ] = torch.dot(
                    distribution,
                    distribution.new_tensor([0.0, 1.0, 2.0]),
                )

        return RoleArityEventFieldOutput(
            events=output.events,
            num_agents=output.num_agents,
            num_actions=output.num_actions,
            visible_shape=output.visible_shape,
            visible_role_marginals=visible,
            typed_null_role_marginals=typed,
            event_probabilities=event_probability,
            cardinality_expectations=expectation,
            role_arity_probabilities=arity_probability,
            capped_role_count_expectations=capped_count,
            refinement_scale=output.refinement_scale,
            arity_scale=output.arity_scale,
        )

    @classmethod
    def target_log_mass(
        cls,
        event: RoleArityEvent,
        groups: Sequence[RoleFillerTargetGroup],
    ) -> Tensor | None:
        if not groups:
            return (
                event.log_state_weights[:, 0].sum()
                + event.composition_energy[0, 0]
            )

        choices: list[list[int]] = []
        active_roles = [int(role) for role in event.active_role_indices.tolist()]
        local_role = {role: index for index, role in enumerate(active_roles)}
        for group in groups:
            rows = group.candidate_rows
            if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64):
                raise ValueError("Target candidate rows must be an integer vector.")
            if rows.numel() == 0:
                return None
            if (rows < 0).any() or (rows >= event.num_candidates).any():
                raise ValueError("A target group references an invalid candidate row.")
            if int(group.role_index) not in local_role:
                raise ValueError("A target group references an inactive role.")
            choices.append([int(value) for value in rows.tolist()])

        assignments: list[Tensor] = []
        seen_states: set[tuple[int, ...]] = set()
        row_index = torch.arange(
            event.num_candidates, device=event.log_state_weights.device
        )
        for selected in product(*choices):
            if len(set(selected)) != len(selected):
                continue
            state_values = [0] * event.num_candidates
            role_counts = [0] * len(active_roles)
            for row, group in zip(selected, groups):
                role = int(group.role_index)
                state_values[int(row)] = role + 1
                role_counts[local_role[role]] += 1
            state_key = tuple(state_values)
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            state = torch.tensor(
                state_values,
                dtype=torch.long,
                device=event.log_state_weights.device,
            )
            arity_code = cls.encode_arity_counts(role_counts)
            value = event.log_state_weights[row_index, state].sum()
            value = value + event.composition_energy[len(groups), arity_code]
            if torch.isfinite(value):
                assignments.append(value)
        if not assignments:
            return None
        return torch.logsumexp(torch.stack(assignments), dim=0)
