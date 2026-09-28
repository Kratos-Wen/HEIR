from __future__ import annotations

from dataclasses import dataclass

from itertools import product

from typing import Sequence

import torch

import torch.nn as nn

from torch import Tensor

CORISP_GROUNDED_ROLE_SET_ARCHITECTURE_VERSION = (
    "corisp_joint_v8_grounded_role_set_field_v1"
)

@dataclass(frozen=True)
class GroundedRoleSetConfig:
    d_model: int = 384
    num_roles: int = 2
    max_cardinality: int = 32
    learn_cardinality: bool = True
    architecture_version: str = CORISP_GROUNDED_ROLE_SET_ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        if self.architecture_version != CORISP_GROUNDED_ROLE_SET_ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported grounded role-set field {self.architecture_version!r}."
            )
        if min(self.d_model, self.num_roles, self.max_cardinality) <= 0:
            raise ValueError("Grounded role-set dimensions must be positive.")

@dataclass(frozen=True)
class GroundedRoleSetEvent:
    """One exact distribution for a real detected person-predicate event."""

    agent_index: int
    action_index: int
    pair_indices: Tensor
    typed_role_indices: Tensor
    log_state_weights: Tensor
    cardinality_energy: Tensor
    log_cardinality_coefficients: Tensor
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
class GroundedRoleSetOutput:
    events: tuple[GroundedRoleSetEvent, ...]
    num_agents: int
    num_actions: int
    visible_shape: tuple[int, int, int]
    visible_role_marginals: Tensor | None = None
    typed_null_role_marginals: Tensor | None = None
    event_probabilities: Tensor | None = None
    cardinality_expectations: Tensor | None = None

    def event(self, agent_index: int, action_index: int) -> GroundedRoleSetEvent:
        if not 0 <= agent_index < self.num_agents:
            raise IndexError(agent_index)
        if not 0 <= action_index < self.num_actions:
            raise IndexError(action_index)
        return self.events[agent_index * self.num_actions + action_index]

@dataclass(frozen=True)
class RoleFillerTargetGroup:
    """Alternative candidate rows that can realize one reviewed role filler."""

    role_index: int
    candidate_rows: Tensor

class GroundedRoleSetField(nn.Module):
    """Cardinality-potential CRF over visible and typed-null role atoms.

    Every visible atom is categorical over ``{null, role_1, ..., role_R}``.
    Every typed-null atom admits only its own role.  A learned event-specific
    cardinality energy couples the atoms while exact dynamic programming keeps
    the partition and all marginals tractable.
    """

    def __init__(self, cfg: GroundedRoleSetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cardinality_norm = nn.LayerNorm(cfg.d_model)
        self.cardinality_head = nn.Linear(
            cfg.d_model, cfg.max_cardinality + 1
        )
        # Zero is an exact factorized distribution, preserving all incoming
        # detector-composed role probabilities at initialization.
        nn.init.zeros_(self.cardinality_head.weight)
        nn.init.zeros_(self.cardinality_head.bias)
        if not cfg.learn_cardinality:
            self.cardinality_norm.requires_grad_(False)
            self.cardinality_head.requires_grad_(False)

    @staticmethod
    def log_cardinality_coefficients(
        log_null: Tensor,
        log_positive: Tensor,
    ) -> Tensor:
        """Return log weights for selecting exactly ``k`` of ``M`` atoms."""

        if log_null.ndim != 1 or log_positive.shape != log_null.shape:
            raise ValueError("log_null and log_positive must be aligned vectors.")
        coefficient = log_null.new_zeros((1,))
        negative_infinity = log_null.new_full((1,), float("-inf"))
        for null_value, positive_value in zip(log_null, log_positive):
            stay = torch.cat([coefficient + null_value, negative_infinity])
            take = torch.cat(
                [negative_infinity, coefficient + positive_value]
            )
            coefficient = GroundedRoleSetField._safe_logaddexp(stay, take)
        return coefficient

    @staticmethod
    def _safe_logaddexp(first: Tensor, second: Tensor) -> Tensor:
        """Exact log-add-exp with a defined gradient for unreachable states."""

        both_unreachable = torch.isneginf(first) & torch.isneginf(second)
        safe_first = first.masked_fill(both_unreachable, 0.0)
        safe_second = second.masked_fill(both_unreachable, 0.0)
        combined = torch.logaddexp(safe_first, safe_second)
        return combined.masked_fill(both_unreachable, float("-inf"))

    @staticmethod
    def focal_transform_nll(nll: Tensor, gamma: float) -> Tensor:
        """Apply ``nll * (1-exp(-nll))**gamma`` at its continuous zero."""

        if gamma < 0:
            raise ValueError("Focal gamma must be non-negative.")
        if gamma == 0:
            return nll
        positive = nll > 0
        safe_nll = torch.where(positive, nll, torch.ones_like(nll))
        log_modulation = float(gamma) * torch.log(-torch.expm1(-safe_nll))
        transformed = safe_nll * torch.exp(log_modulation)
        return torch.where(positive, transformed, torch.zeros_like(nll))

    @classmethod
    def log_partition(
        cls,
        log_state_weights: Tensor,
        cardinality_energy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if log_state_weights.ndim != 2 or log_state_weights.shape[1] < 2:
            raise ValueError("log_state_weights must be [M,1+R].")
        expected = log_state_weights.shape[0] + 1
        if cardinality_energy.shape != (expected,):
            raise ValueError(
                f"cardinality_energy must have shape ({expected},)."
            )
        log_positive = torch.logsumexp(log_state_weights[:, 1:], dim=-1)
        coefficient = cls.log_cardinality_coefficients(
            log_state_weights[:, 0], log_positive
        )
        return torch.logsumexp(coefficient + cardinality_energy, dim=0), coefficient

    @staticmethod
    def _log_probability(probability: Tensor) -> Tensor:
        safe = probability.clamp_min(torch.finfo(torch.float32).tiny)
        value = safe.log()
        return value.masked_fill(probability <= 0, float("-inf"))

    def _build_event(
        self,
        *,
        agent_index: int,
        action_index: int,
        pair_indices: Tensor,
        visible_probability: Tensor,
        typed_probability: Tensor,
        valid_roles: Tensor,
        event_state: Tensor,
    ) -> GroundedRoleSetEvent:
        role_count = self.cfg.num_roles
        if visible_probability.shape != (pair_indices.numel(), role_count):
            raise ValueError("Visible role probabilities do not align with pairs.")
        if typed_probability.shape != (role_count,):
            raise ValueError("Typed-null role probabilities must be [R].")
        if valid_roles.ndim != 1 or valid_roles.dtype != torch.bool:
            raise ValueError("valid_roles must be a bool vector.")
        if valid_roles.shape[0] != role_count:
            raise ValueError("valid_roles does not match the configured role count.")

        visible_probability = visible_probability.float().masked_fill(
            ~valid_roles[None], 0.0
        )
        visible_null = 1.0 - visible_probability.sum(dim=-1)
        if (visible_probability < 0).any() or (visible_null < -1e-6).any():
            raise ValueError("Visible role probabilities do not form sub-probabilities.")
        visible_null = visible_null.clamp_min(0.0)
        visible_states = torch.cat(
            [visible_null[:, None], visible_probability], dim=-1
        )

        typed_roles = torch.nonzero(valid_roles, as_tuple=False).flatten()
        typed_states = visible_states.new_zeros(
            (typed_roles.numel(), role_count + 1)
        )
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

        state_probability = torch.cat([visible_states, typed_states], dim=0)
        if state_probability.shape[0] > self.cfg.max_cardinality:
            raise ValueError(
                f"Event has {state_probability.shape[0]} candidates, exceeding "
                f"max_cardinality={self.cfg.max_cardinality}."
            )
        log_states = self._log_probability(state_probability)
        if self.cfg.learn_cardinality:
            cardinality = self.cardinality_head(
                self.cardinality_norm(event_state.float())
            )[: state_probability.shape[0] + 1]
            cardinality = cardinality - cardinality[0]
        else:
            cardinality = event_state.float().new_zeros(
                state_probability.shape[0] + 1
            )
        log_z, coefficient = self.log_partition(log_states, cardinality)
        return GroundedRoleSetEvent(
            agent_index=int(agent_index),
            action_index=int(action_index),
            pair_indices=pair_indices,
            typed_role_indices=typed_roles,
            log_state_weights=log_states,
            cardinality_energy=cardinality,
            log_cardinality_coefficients=coefficient,
            log_partition=log_z,
        )

    @staticmethod
    def _validate_inputs(
        visible_role_probs: Tensor,
        typed_null_role_probs: Tensor,
        pair_agent_indices: Tensor,
        event_states: Tensor,
        visible_action_mask: Tensor,
        valid_role_mask: Tensor,
    ) -> tuple[int, int, int, int]:
        if visible_role_probs.ndim != 3:
            raise ValueError("visible_role_probs must be [Q,A,R].")
        q, actions, roles = visible_role_probs.shape
        if typed_null_role_probs.ndim != 3:
            raise ValueError("typed_null_role_probs must be [H,A,R].")
        humans = typed_null_role_probs.shape[0]
        if typed_null_role_probs.shape[1:] != (actions, roles):
            raise ValueError("Visible and typed-null role spaces do not align.")
        if pair_agent_indices.shape != (q,) or pair_agent_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("pair_agent_indices must be integer [Q].")
        if q and ((pair_agent_indices < 0).any() or (pair_agent_indices >= humans).any()):
            raise ValueError("pair_agent_indices references an invalid agent.")
        if event_states.ndim != 3 or event_states.shape[:2] != (humans, actions):
            raise ValueError("event_states must be [H,A,D].")
        if visible_action_mask.shape != (q, actions) or visible_action_mask.dtype != torch.bool:
            raise ValueError("visible_action_mask must be bool [Q,A].")
        if valid_role_mask.shape != (actions, roles) or valid_role_mask.dtype != torch.bool:
            raise ValueError("valid_role_mask must be bool [A,R].")
        if not valid_role_mask.any(dim=-1).all():
            raise ValueError("Every action must admit at least one role.")
        for name, value in (
            ("visible_role_probs", visible_role_probs),
            ("typed_null_role_probs", typed_null_role_probs),
            ("event_states", event_states),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
        return q, humans, actions, roles

    def forward(
        self,
        *,
        visible_role_probs: Tensor,
        typed_null_role_probs: Tensor,
        pair_agent_indices: Tensor,
        event_states: Tensor,
        visible_action_mask: Tensor,
        valid_role_mask: Tensor,
        return_marginals: bool = False,
    ) -> GroundedRoleSetOutput:
        q, humans, actions, roles = self._validate_inputs(
            visible_role_probs,
            typed_null_role_probs,
            pair_agent_indices,
            event_states,
            visible_action_mask,
            valid_role_mask,
        )
        events: list[GroundedRoleSetEvent] = []
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
                        visible_probability=visible_role_probs[
                            pair_indices, action_index
                        ],
                        typed_probability=typed_null_role_probs[
                            human_index, action_index
                        ],
                        valid_roles=valid_role_mask[action_index],
                        event_state=event_states[human_index, action_index],
                    )
                )

        output = GroundedRoleSetOutput(
            events=tuple(events),
            num_agents=humans,
            num_actions=actions,
            visible_shape=(q, actions, roles),
        )
        if not return_marginals:
            return output
        return self._with_exact_marginals(output)

    def _with_exact_marginals(
        self,
        output: GroundedRoleSetOutput,
    ) -> GroundedRoleSetOutput:
        if not output.events:
            q, actions, roles = output.visible_shape
            reference = self.cardinality_head.weight
            return GroundedRoleSetOutput(
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
            )

        reference = output.events[0].log_state_weights
        q, actions, roles = output.visible_shape
        visible = reference.new_zeros((q, actions, roles))
        typed = reference.new_zeros((output.num_agents, actions, roles))
        event_probability = reference.new_zeros((output.num_agents, actions))
        expectation = reference.new_zeros((output.num_agents, actions))

        # Reverse-mode differentiation of log Z yields every exact state
        # marginal in one pass through the disconnected event DPs.
        with torch.enable_grad():
            leaves: list[Tensor] = []
            log_partitions: list[Tensor] = []
            detached_cardinality: list[Tensor] = []
            for event in output.events:
                leaf = event.log_state_weights.detach().requires_grad_(True)
                cardinality = event.cardinality_energy.detach()
                log_z, _ = GroundedRoleSetField.log_partition(
                    leaf, cardinality
                )
                leaves.append(leaf)
                log_partitions.append(log_z)
                detached_cardinality.append(cardinality)
            gradients = torch.autograd.grad(
                torch.stack(log_partitions).sum(), leaves
            )

        for event, marginal, log_z, cardinality in zip(
            output.events, gradients, log_partitions, detached_cardinality
        ):
            marginal = marginal.detach().clamp(0.0, 1.0)
            visible_count = event.num_visible_candidates
            visible[event.pair_indices, event.action_index] = marginal[
                :visible_count, 1:
            ]
            for offset, role_index in enumerate(event.typed_role_indices.tolist()):
                typed[
                    event.agent_index, event.action_index, role_index
                ] = marginal[visible_count + offset, role_index + 1]

            log_all_null = (
                event.log_state_weights[:, 0].detach().sum()
                + cardinality[0]
            )
            event_probability[event.agent_index, event.action_index] = (
                1.0 - torch.exp(log_all_null - log_z.detach())
            ).clamp(0.0, 1.0)
            count = torch.arange(
                event.num_candidates + 1,
                device=reference.device,
                dtype=reference.dtype,
            )
            count_probability = torch.softmax(
                event.log_cardinality_coefficients.detach()
                + cardinality,
                dim=0,
            )
            expectation[event.agent_index, event.action_index] = (
                count * count_probability
            ).sum()

        return GroundedRoleSetOutput(
            events=output.events,
            num_agents=output.num_agents,
            num_actions=output.num_actions,
            visible_shape=output.visible_shape,
            visible_role_marginals=visible,
            typed_null_role_marginals=typed,
            event_probabilities=event_probability,
            cardinality_expectations=expectation,
        )

    @staticmethod
    def target_log_mass(
        event: GroundedRoleSetEvent,
        groups: Sequence[RoleFillerTargetGroup],
    ) -> Tensor | None:
        """Sum exact mass over distinct proposal assignments for Gold fillers.

        An empty group list denotes a reviewed negative event and therefore
        selects the exact all-null configuration.  ``None`` is returned only
        when a positive target has no valid distinct assignment.
        """

        if not groups:
            return event.log_state_weights[:, 0].sum() + event.cardinality_energy[0]

        choices: list[list[int]] = []
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
            value = value + event.cardinality_energy[len(groups)]
            if torch.isfinite(value):
                assignments.append(value)
        if not assignments:
            return None
        return torch.logsumexp(torch.stack(assignments), dim=0)
