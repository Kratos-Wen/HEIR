from __future__ import annotations

from dataclasses import dataclass

from math import log

import torch

import torch.nn as nn

import torch.nn.functional as F

from corisp.pair_packet import PairPacket

CORISP_AGENT_ROLE_FIELD_ARCHITECTURE_VERSION = (
    "corisp_joint_v6_agent_role_field_typed_null_v2"
)

@dataclass(frozen=True)
class CoRISPAgentRoleFieldConfig:
    d_model: int = 384
    visual_dim: int = 1024
    semantic_dim: int = 2048
    num_actions: int = 21
    roles: tuple[str, ...] = ("obj", "instr")
    num_object_classes: int = 80
    inference_steps: int = 2
    ffn_dim: int = 1536
    dropout: float = 0.1
    text_temperature: float = 0.07
    enable_typed_null_fillers: bool = False
    architecture_version: str = CORISP_AGENT_ROLE_FIELD_ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        if self.architecture_version != CORISP_AGENT_ROLE_FIELD_ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported agent-role field {self.architecture_version!r}."
            )
        if min(
            self.d_model,
            self.visual_dim,
            self.semantic_dim,
            self.num_actions,
            self.num_object_classes,
            self.inference_steps,
            self.ffn_dim,
        ) <= 0:
            raise ValueError("Agent-role field dimensions and steps must be positive.")
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("roles must be non-empty and unique.")
        if not 0.0 < self.text_temperature < 1.0:
            raise ValueError("text_temperature must lie in (0, 1).")

class CoRISPAgentRoleField(nn.Module):
    """Permutation-equivariant role field over detector-native instances."""

    def __init__(
        self,
        cfg: CoRISPAgentRoleFieldConfig,
        *,
        action_prototypes: torch.Tensor,
        object_prototypes: torch.Tensor,
        valid_role_mask: torch.Tensor | None = None,
        object_action_mask: torch.Tensor | None = None,
        role_prototypes: torch.Tensor | None = None,
        human_label: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.human_label = int(human_label)
        num_roles = len(cfg.roles)
        if action_prototypes.shape != (cfg.num_actions, cfg.semantic_dim):
            raise ValueError("action_prototypes must be [num_actions, semantic_dim].")
        if object_prototypes.shape != (
            cfg.num_object_classes,
            cfg.semantic_dim,
        ):
            raise ValueError(
                "object_prototypes must be [num_object_classes, semantic_dim]."
            )
        if valid_role_mask is None:
            valid_role_mask = torch.ones(
                cfg.num_actions, num_roles, dtype=torch.bool
            )
        if valid_role_mask.shape != (cfg.num_actions, num_roles):
            raise ValueError("valid_role_mask must be [num_actions, num_roles].")
        if not valid_role_mask.any(dim=-1).all():
            raise ValueError("Every action must admit at least one role state.")
        if object_action_mask is None:
            object_action_mask = torch.ones(
                cfg.num_object_classes, cfg.num_actions, dtype=torch.bool
            )
        if object_action_mask.shape != (
            cfg.num_object_classes,
            cfg.num_actions,
        ):
            raise ValueError("object_action_mask must be [num_objects,num_actions].")

        action_prototypes = F.normalize(action_prototypes.float(), dim=-1)
        object_prototypes = F.normalize(object_prototypes.float(), dim=-1)
        if role_prototypes is None:
            role_weights = valid_role_mask.float().transpose(0, 1)
            role_prototypes = role_weights @ action_prototypes
            role_prototypes = role_prototypes / role_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
        if role_prototypes.shape != (num_roles, cfg.semantic_dim):
            raise ValueError("role_prototypes must be [num_roles, semantic_dim].")

        self.register_buffer("action_prototypes", action_prototypes)
        self.register_buffer("object_prototypes", object_prototypes)
        self.register_buffer(
            "role_prototypes", F.normalize(role_prototypes.float(), dim=-1)
        )
        self.register_buffer("valid_role_mask", valid_role_mask.bool())
        self.register_buffer("object_action_mask", object_action_mask.bool())

        d = cfg.d_model
        self.agent_query_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.visual_dim),
        )
        self.aligned_agent_projection = nn.Sequential(
            nn.LayerNorm(cfg.visual_dim),
            nn.Linear(cfg.visual_dim, d),
        )
        # All DINO.txt-aligned concepts share one map. Separate noun, action,
        # role, and global projections would allow four incompatible semantic
        # coordinate systems and add parameters without adding evidence.
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(cfg.semantic_dim),
            nn.Linear(cfg.semantic_dim, d),
        )
        self.entity_visual_projection = nn.Sequential(
            nn.LayerNorm(cfg.visual_dim),
            nn.Linear(cfg.visual_dim, d),
        )
        self.local_visual_projection = nn.Sequential(
            nn.LayerNorm(cfg.visual_dim * 3),
            nn.Linear(cfg.visual_dim * 3, d),
        )
        self.box_projection = nn.Sequential(
            nn.Linear(4, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.entity_norm = nn.LayerNorm(d)
        self.pair_norm = nn.LayerNorm(d)
        self.edge_norm = nn.LayerNorm(d)
        self.event_norm = nn.LayerNorm(d)

        # This is the ordinary-HOI host.  It is semantically aligned to the
        # frozen DINO.txt action bank and remains directly inspectable.
        self.host_semantic_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.semantic_dim),
        )
        self.action_bias = nn.Parameter(torch.zeros(cfg.num_actions))
        self.logit_scale = nn.Parameter(
            torch.tensor(log(1.0 / cfg.text_temperature), dtype=torch.float32)
        )

        # One tied update is reused at every field iteration.  No update is
        # indexed by an event slot or by the number/order of detected people.
        self.edge_evidence = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, d),
        )
        self.role_residual = nn.Linear(d, num_roles)
        self.interaction_residual = nn.Linear(d, 1)
        self.event_update = nn.GRUCell(d, d)
        self.entity_update = nn.GRUCell(d, d)
        self.role_offset = nn.Parameter(torch.zeros(num_roles, d))

        # Zero is the exact host-equivalent point.  The first optimizer step
        # learns whether opening the field is useful before its residual can
        # alter either ordinary HOI or role probabilities.
        self.residual_scale = nn.Parameter(torch.zeros(()))

    @property
    def num_roles(self) -> int:
        return len(self.cfg.roles)

    def _agent_indices(self, packet: PairPacket) -> torch.Tensor:
        if packet.batch_size != 1:
            raise ValueError("The role field consumes compact one-image packets.")
        if packet.entity_labels is None:
            raise ValueError("The role field requires entity labels.")
        valid = packet.entity_labels[0] == self.human_label
        if packet.entity_valid_mask is not None:
            valid = valid & packet.entity_valid_mask[0]
        return torch.nonzero(valid, as_tuple=False).flatten()

    def build_agent_queries(self, packet: PairPacket) -> torch.Tensor:
        """Build real-person query tokens for the frozen DINO.txt head."""

        packet.validate()
        if packet.entity_feats is None:
            raise ValueError("Agent queries require detector entity features.")
        agents = self._agent_indices(packet)
        return self.agent_query_projection(packet.entity_feats[:, agents])

    @staticmethod
    def _pool_boxes(grid: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 4 or boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("grid and boxes must be [B,D,H,W] and [B,N,4].")
        batch, _, height, width = grid.shape
        if boxes.shape[0] != batch:
            raise ValueError("grid and boxes must share a batch dimension.")
        y = (
            torch.arange(height, device=grid.device, dtype=boxes.dtype) + 0.5
        ) / height
        x = (
            torch.arange(width, device=grid.device, dtype=boxes.dtype) + 0.5
        ) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        mask = (
            (xx[None, None] >= x1[..., None, None])
            & (xx[None, None] <= x2[..., None, None])
            & (yy[None, None] >= y1[..., None, None])
            & (yy[None, None] <= y2[..., None, None])
        )
        weights = mask.to(dtype=grid.dtype)
        counts = weights.sum(dim=(-2, -1), keepdim=True)
        pooled = torch.einsum(
            "bnhw,bdhw->bnd", weights / counts.clamp_min(1.0), grid
        )
        empty = counts.flatten(2).squeeze(-1) == 0
        if empty.any():
            center_x = ((x1 + x2) * 0.5 * width).floor().long().clamp(0, width - 1)
            center_y = ((y1 + y2) * 0.5 * height).floor().long().clamp(0, height - 1)
            batch_index = torch.arange(batch, device=grid.device)[:, None].expand_as(
                center_x
            )
            nearest = grid.permute(0, 2, 3, 1)[batch_index, center_y, center_x]
            pooled = torch.where(empty[..., None], nearest, pooled)
        return pooled

    @staticmethod
    def _probability_logit(probability: torch.Tensor) -> torch.Tensor:
        probability = probability.float()
        return torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))

    @staticmethod
    def _weighted_scatter_mean(
        value: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.shape[:-1] != weight.shape[:-1] or weight.shape[-1] != 1:
            raise ValueError("scatter value and weight shapes are inconsistent.")
        output = value.new_zeros((groups, *value.shape[1:]))
        denominator = weight.new_zeros((groups, *weight.shape[1:]))
        if value.shape[0]:
            output.index_add_(0, index, value * weight)
            denominator.index_add_(0, index, weight)
        return output / denominator.clamp_min(1e-6), denominator

    def _semantic_host(self, state: torch.Tensor) -> torch.Tensor:
        semantic = F.normalize(self.host_semantic_projection(state).float(), dim=-1)
        prototypes = self.action_prototypes.to(device=state.device)
        scale = self.logit_scale.float().clamp(max=log(100.0)).exp()
        logits = scale * torch.einsum("...s,as->...a", semantic, prototypes)
        return (logits + self.action_bias.float()).to(dtype=state.dtype)

    def _joint_state(
        self,
        host_logits: torch.Tensor,
        role_residual: torch.Tensor,
        interaction_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return null and positive-role probabilities from one softmax."""

        expected = (*host_logits.shape, self.num_roles)
        if role_residual.shape != expected:
            raise ValueError(
                f"role_residual must be {expected}, got {tuple(role_residual.shape)}."
            )
        if interaction_residual.shape != host_logits.shape:
            raise ValueError("interaction_residual must match host_logits.")
        valid = self.valid_role_mask.to(device=host_logits.device)
        valid_float = valid.float()
        count = valid_float.sum(dim=-1).clamp_min(1.0)
        residual = role_residual.float()
        mean = (residual * valid_float).sum(dim=-1) / count
        centered = (residual - mean[..., None]) * valid_float
        log_uniform = -count.log()
        scale = self.residual_scale.float()
        role_energy = (
            host_logits.float()[..., None]
            + log_uniform[..., None]
            + scale
            * (interaction_residual.float()[..., None] + centered)
        )
        role_energy = role_energy.masked_fill(~valid, float("-inf"))
        null_energy = torch.zeros_like(host_logits.float())[..., None]
        state_probability = torch.softmax(
            torch.cat([null_energy, role_energy], dim=-1), dim=-1
        )
        return state_probability[..., 0], state_probability[..., 1:]

    def _typed_null_state(
        self,
        host_logits: torch.Tensor,
        role_residual: torch.Tensor,
        interaction_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return independent typed-null slots and their noisy-OR marginal.

        Visible edges use :meth:`_joint_state` because one localized edge has
        one primary role.  Null fillers instead represent role slots that may
        be absent together.  At zero residual, equal per-role hazards are
        chosen so their noisy-OR is exactly ``sigmoid(host_logits)``.
        """

        expected = (*host_logits.shape, self.num_roles)
        if role_residual.shape != expected:
            raise ValueError(
                f"role_residual must be {expected}, got {tuple(role_residual.shape)}."
            )
        if interaction_residual.shape != host_logits.shape:
            raise ValueError("interaction_residual must match host_logits.")
        valid = self.valid_role_mask.to(device=host_logits.device)
        valid_float = valid.float()
        count = valid_float.sum(dim=-1).clamp_min(1.0)
        residual = role_residual.float()
        mean = (residual * valid_float).sum(dim=-1) / count
        centered = (residual - mean[..., None]) * valid_float

        event_probability = torch.sigmoid(host_logits.float())
        log_survival = torch.log1p(
            -event_probability.clamp(max=1.0 - 1e-7)
        )
        base_slot_probability = -torch.expm1(log_survival / count)
        base_slot_logit = torch.logit(
            base_slot_probability.clamp(1e-7, 1.0 - 1e-7)
        )
        slot_logit = base_slot_logit[..., None] + self.residual_scale.float() * (
            interaction_residual.float()[..., None] + centered
        )
        role_probability = torch.sigmoid(slot_logit).masked_fill(~valid, 0.0)
        no_filler_probability = torch.prod(1.0 - role_probability, dim=-1)
        any_filler_probability = 1.0 - no_filler_probability
        return no_filler_probability, role_probability, any_filler_probability

    def exact_identity_probabilities(self, host_logits: torch.Tensor) -> torch.Tensor:
        """Expose the initialized exact marginal for conformance tests."""

        zeros = host_logits.new_zeros((*host_logits.shape, self.num_roles))
        _, role = self._joint_state(host_logits, zeros, host_logits.new_zeros(host_logits.shape))
        return role.sum(dim=-1)

    def _build_states(
        self,
        packet: PairPacket,
        visual_grid: torch.Tensor,
        global_semantic: torch.Tensor,
        aligned_agent_tokens: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if packet.entity_feats is None or packet.entity_boxes is None:
            raise ValueError("The role field requires entity boxes and features.")
        if packet.entity_labels is None:
            raise ValueError("The role field requires entity labels.")
        if packet.subject_indices is None or packet.object_indices is None:
            raise ValueError("The role field requires pair-to-entity indices.")
        if visual_grid.shape[:2] != (1, self.cfg.visual_dim):
            raise ValueError(f"visual_grid must be [1,{self.cfg.visual_dim},H,W].")
        if global_semantic.shape != (1, self.cfg.semantic_dim):
            raise ValueError(
                f"global_semantic must be [1,{self.cfg.semantic_dim}]."
            )
        labels = packet.entity_labels[0]
        if (labels < 0).any() or (labels >= self.cfg.num_object_classes).any():
            raise ValueError("entity_labels is outside the object prototype bank.")

        dtype = packet.pair_feats.dtype
        pooled_entity = self._pool_boxes(visual_grid, packet.entity_boxes)
        object_semantic = self.object_prototypes[labels].to(device=labels.device)
        entity = self.entity_norm(
            packet.entity_feats
            + self.entity_visual_projection(pooled_entity.to(dtype=dtype))
            + self.semantic_projection(object_semantic[None].to(dtype=dtype))
            + self.box_projection(packet.entity_boxes.to(dtype=dtype))
        ).squeeze(0)
        global_state = self.semantic_projection(
            global_semantic.to(dtype=dtype)
        ).squeeze(0)

        subject_index = packet.subject_indices[0].long()
        object_index = packet.object_indices[0].long()
        agents = self._agent_indices(packet)
        entity_to_agent = torch.full(
            (entity.shape[0],), -1, dtype=torch.long, device=entity.device
        )
        entity_to_agent[agents] = torch.arange(agents.numel(), device=entity.device)
        pair_agent = entity_to_agent[subject_index]
        if (pair_agent < 0).any():
            raise ValueError("Every role-field subject must reference a human entity.")

        if aligned_agent_tokens is None:
            aligned_agent_tokens = self.build_agent_queries(packet)
        if aligned_agent_tokens.shape != (1, agents.numel(), self.cfg.visual_dim):
            raise ValueError(
                "aligned_agent_tokens must follow the detected-human order and be "
                f"[1,{agents.numel()},{self.cfg.visual_dim}]."
            )
        aligned_agent = self.aligned_agent_projection(
            aligned_agent_tokens.to(dtype=dtype)
        ).squeeze(0)

        union_boxes = torch.cat(
            [
                torch.minimum(packet.subject_boxes[..., :2], packet.object_boxes[..., :2]),
                torch.maximum(packet.subject_boxes[..., 2:], packet.object_boxes[..., 2:]),
            ],
            dim=-1,
        )
        union_visual = self._pool_boxes(visual_grid, union_boxes)
        entity_visual = pooled_entity.squeeze(0)
        local_visual = torch.cat(
            [
                entity_visual[subject_index],
                entity_visual[object_index],
                union_visual.squeeze(0),
            ],
            dim=-1,
        )
        pair_static = self.pair_norm(
            packet.pair_feats.squeeze(0)
            + self.local_visual_projection(local_visual.to(dtype=dtype))
            + 0.5 * (entity[subject_index] + entity[object_index])
            + aligned_agent[pair_agent]
            + global_state
        )
        return entity, pair_static, global_state, agents, pair_agent, object_index

    def _field_residuals(
        self,
        edge_state: torch.Tensor,
        noun_semantic: torch.Tensor | None = None,
        field_context: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence = self.edge_evidence(edge_state)
        role_residual = self.role_residual(evidence)
        interaction_residual = self.interaction_residual(evidence).squeeze(-1)
        return role_residual, interaction_residual, evidence

    def _field_scores(
        self,
        edge_state: torch.Tensor,
        host_logits: torch.Tensor,
        noun_semantic: torch.Tensor | None = None,
        field_context: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        role_residual, interaction_residual, evidence = self._field_residuals(
            edge_state, noun_semantic, field_context
        )
        no_interaction, role_probability = self._joint_state(
            host_logits, role_residual, interaction_residual
        )
        hoi_probability = role_probability.sum(dim=-1)
        return no_interaction, role_probability, hoi_probability, evidence

    def _typed_null_scores(
        self,
        edge_state: torch.Tensor,
        host_logits: torch.Tensor,
        noun_semantic: torch.Tensor | None = None,
        field_context: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        role_residual, interaction_residual, evidence = self._field_residuals(
            edge_state, noun_semantic, field_context
        )
        no_filler, role_probability, any_filler = self._typed_null_state(
            host_logits, role_residual, interaction_residual
        )
        return no_filler, role_probability, any_filler, evidence

    def _prepare_field_context(
        self,
        packet: PairPacket,
        visual_grid: torch.Tensor,
        global_semantic: torch.Tensor,
    ) -> dict[str, object]:
        return {}

    def _finalize_field_output(
        self,
        output: dict[str, torch.Tensor],
        field_context: dict[str, object],
    ) -> dict[str, torch.Tensor]:
        return output

    def forward(
        self,
        packet: PairPacket,
        *,
        visual_grid: torch.Tensor,
        global_semantic: torch.Tensor,
        aligned_agent_tokens: torch.Tensor | None = None,
        host_logits: torch.Tensor | None = None,
        null_host_logits: torch.Tensor | None = None,
        field_context_override: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        packet.validate()
        field_context = self._prepare_field_context(
            packet, visual_grid, global_semantic
        )
        if field_context_override is not None:
            overlap = set(field_context).intersection(field_context_override)
            if overlap:
                raise ValueError(
                    "field_context_override cannot replace prepared keys: "
                    + ", ".join(sorted(overlap))
                )
            field_context.update(field_context_override)
        (
            entity,
            pair_static,
            global_state,
            agents,
            pair_agent,
            object_index,
        ) = self._build_states(
            packet, visual_grid, global_semantic, aligned_agent_tokens
        )
        action_anchor = self.semantic_projection(
            self.action_prototypes.to(device=entity.device, dtype=entity.dtype)
        )
        role_anchor = self.semantic_projection(
            self.role_prototypes.to(device=entity.device, dtype=entity.dtype)
        ) + self.role_offset.to(dtype=entity.dtype)
        visible_noun_semantic = self.object_prototypes[
            packet.entity_labels[0, object_index]
        ].to(device=entity.device)
        visible_action_mask = self.object_action_mask[
            packet.entity_labels[0, object_index]
        ].to(device=entity.device)
        null_noun_semantic = self.object_prototypes[
            packet.entity_labels[0]
        ].to(device=entity.device)

        if host_logits is None:
            visible_host = self._semantic_host(pair_static)
        else:
            if host_logits.shape != (1, packet.num_pairs, self.cfg.num_actions):
                raise ValueError("host_logits must be [1,Q,num_actions].")
            visible_host = host_logits.squeeze(0).to(dtype=pair_static.dtype)
        null_static = self.event_norm(entity + global_state)
        if null_host_logits is None:
            null_host = self._semantic_host(null_static)
        else:
            if null_host_logits.shape != (
                1,
                entity.shape[0],
                self.cfg.num_actions,
            ):
                raise ValueError("null_host_logits must be [1,E,num_actions].")
            null_host = null_host_logits.squeeze(0).to(dtype=entity.dtype)

        event = self.event_norm(
            entity[agents, None, :]
            + action_anchor[None]
            + global_state[None, None]
        )
        if agents.numel():
            aligned = self.aligned_agent_projection(
                (
                    aligned_agent_tokens
                    if aligned_agent_tokens is not None
                    else self.build_agent_queries(packet)
                ).to(dtype=entity.dtype)
            ).squeeze(0)
            event = self.event_norm(event + aligned[:, None])

        visible_no = visible_host.new_ones(visible_host.shape)
        visible_role = visible_host.new_zeros(
            (*visible_host.shape, self.num_roles)
        )
        visible_hoi = visible_host.new_zeros(visible_host.shape)
        edge_state = pair_static[:, None] + action_anchor[None]
        for _ in range(self.cfg.inference_steps):
            edge_state = self.edge_norm(
                pair_static[:, None]
                + action_anchor[None]
                + event[pair_agent]
                + 0.5
                * (entity[packet.subject_indices[0]][:, None] + entity[object_index][:, None])
            )
            if getattr(self, "disable_role_feedback", False):
                evidence = self.edge_evidence(edge_state)
                residual = self.interaction_residual(evidence).squeeze(-1)
                visible_hoi = torch.sigmoid(visible_host.float() + self.residual_scale.float() * residual.float())
                valid = self.valid_role_mask.to(visible_hoi.device)
                uniform = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
                visible_role = visible_hoi[..., None] * uniform[None]
                visible_no = 1 - visible_hoi
            else:
                visible_no, visible_role, visible_hoi, evidence = self._field_scores(
                    edge_state,
                    visible_host,
                    visible_noun_semantic,
                    field_context.get("visible"),
                )
            visible_role = visible_role.masked_fill(
                ~visible_action_mask[..., None], 0.0
            )
            visible_hoi = visible_role.sum(dim=-1)
            visible_no = torch.where(
                visible_action_mask, visible_no, torch.ones_like(visible_no)
            )
            conditional_role = visible_role / visible_hoi[..., None].clamp_min(1e-30)
            role_context = torch.einsum(
                "qar,rd->qad", conditional_role.to(role_anchor.dtype), role_anchor
            )
            if getattr(self, "disable_role_feedback", False):
                role_context = torch.zeros_like(edge_state)
            edge_message = edge_state + evidence + role_context
            message_hoi = visible_hoi.to(edge_message.dtype)
            event_message, _ = self._weighted_scatter_mean(
                edge_message,
                message_hoi[..., None],
                pair_agent,
                agents.numel(),
            )
            event_input = self.event_norm(
                event_message
                + entity[agents, None]
                + action_anchor[None]
                + global_state[None, None]
            )
            if event.numel():
                event = self.event_update(
                    event_input.reshape(-1, self.cfg.d_model),
                    event.reshape(-1, self.cfg.d_model),
                ).reshape_as(event)

            pair_weight = (
                1.0 - torch.prod(1.0 - visible_hoi, dim=-1)
            ).to(edge_message.dtype)
            entity_message = (
                edge_message * message_hoi[..., None]
            ).sum(dim=1) / message_hoi.sum(dim=1, keepdim=True).clamp_min(1e-6)
            incoming, incoming_weight = self._weighted_scatter_mean(
                entity_message,
                pair_weight[:, None],
                object_index,
                entity.shape[0],
            )
            candidate = self.entity_update(incoming, entity)
            entity = torch.where(incoming_weight > 0, candidate, entity)

        # Re-evaluate after the final tied update; this is the only prediction
        # path and therefore the only state consumed by both HOI and role AP.
        edge_state = self.edge_norm(
            pair_static[:, None]
            + action_anchor[None]
            + event[pair_agent]
            + 0.5
            * (entity[packet.subject_indices[0]][:, None] + entity[object_index][:, None])
        )
        visible_no, visible_role, visible_hoi, evidence = self._field_scores(
            edge_state,
            visible_host,
            visible_noun_semantic,
            field_context.get("visible"),
        )
        visible_role = visible_role.masked_fill(
            ~visible_action_mask[..., None], 0.0
        )
        visible_hoi = visible_role.sum(dim=-1)
        visible_no = torch.where(
            visible_action_mask, visible_no, torch.ones_like(visible_no)
        )

        null_edge = self.edge_norm(
            entity[:, None] + action_anchor[None] + global_state[None, None]
        )
        if self.cfg.enable_typed_null_fillers:
            null_no, null_role, null_hoi, null_evidence = self._typed_null_scores(
                null_edge,
                null_host,
                null_noun_semantic,
                field_context.get("null"),
            )
            human_mask = (packet.entity_labels[0] == self.human_label)[:, None]
            null_role = null_role * human_mask[..., None].to(null_role.dtype)
            null_no = torch.prod(1.0 - null_role, dim=-1)
            null_hoi = 1.0 - null_no
        else:
            null_evidence = self.edge_evidence(null_edge)
            null_no = torch.ones_like(null_host, dtype=torch.float32)
            null_role = null_no.new_zeros((*null_host.shape, self.num_roles))
            null_hoi = torch.zeros_like(null_no)

        event_probability = visible_hoi.new_zeros(
            (agents.numel(), self.cfg.num_actions)
        )
        for agent_index in range(agents.numel()):
            participant = visible_hoi[pair_agent == agent_index]
            if participant.numel():
                event_probability[agent_index] = 1.0 - torch.prod(
                    1.0 - participant, dim=0
                )
        # A positive null filler still asserts that the agent-predicate event
        # exists even when its role-bearing entity is absent or invisible.
        if self.cfg.enable_typed_null_fillers and agents.numel():
            null_participant = null_hoi[agents]
            event_probability = 1.0 - (1.0 - event_probability) * (
                1.0 - null_participant
            )

        output = {
            "agent_entity_indices": agents[None],
            "pair_agent_indices": pair_agent[None],
            "visible_action_mask": visible_action_mask[None],
            "host_logits": visible_host[None],
            "no_interaction_probs": visible_no[None],
            "joint_role_probs": visible_role[None],
            "joint_role_logits": self._probability_logit(visible_role)[None],
            "hoi_probs": visible_hoi[None],
            "hoi_logits": self._probability_logit(visible_hoi)[None],
            "conditional_role_probs": (
                visible_role
                / visible_hoi[..., None].clamp_min(1e-30)
            )[None],
            "event_probs": event_probability[None],
            "event_states": event[None],
            "edge_states": edge_state[None],
            "edge_evidence": evidence[None],
            "null_host_logits": null_host[None],
            "null_no_interaction_probs": null_no[None],
            "null_role_probs": null_role[None],
            "null_role_logits": self._probability_logit(null_role)[None],
            "null_hoi_probs": null_hoi[None],
            "null_hoi_logits": self._probability_logit(null_hoi)[None],
            "null_edge_states": null_edge[None],
            "null_edge_evidence": null_evidence[None],
            "residual_scale": self.residual_scale,
        }
        return self._finalize_field_output(output, field_context)
