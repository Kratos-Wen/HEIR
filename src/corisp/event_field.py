from __future__ import annotations

from dataclasses import dataclass

from math import log, sqrt

import torch

import torch.nn as nn

import torch.nn.functional as F

from corisp.pair_packet import PairPacket

CORISP_EVENT_FIELD_ARCHITECTURE_VERSION = "corisp_joint_v4_event_field"

@dataclass(frozen=True)
class CoRISPEventFieldConfig:
    d_model: int = 384
    visual_dim: int = 1024
    semantic_dim: int = 2048
    num_actions: int = 21
    roles: tuple[str, ...] = ("obj", "instr")
    num_object_classes: int = 81
    num_event_slots: int = 16
    decoder_layers: int = 3
    attention_heads: int = 8
    ffn_dim: int = 1536
    evidence_bases: int = 8
    dropout: float = 0.1
    text_temperature: float = 0.07
    mask_floor: float = 1e-4
    architecture_version: str = CORISP_EVENT_FIELD_ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        if self.architecture_version != CORISP_EVENT_FIELD_ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported event-field architecture {self.architecture_version!r}."
            )
        if min(
            self.d_model,
            self.visual_dim,
            self.semantic_dim,
            self.num_actions,
            self.num_object_classes,
            self.num_event_slots,
            self.decoder_layers,
            self.attention_heads,
            self.ffn_dim,
            self.evidence_bases,
        ) <= 0:
            raise ValueError("Event-field dimensions and layer counts must be positive.")
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("roles must be non-empty and unique.")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads.")
        if not 0.0 < self.text_temperature < 1.0:
            raise ValueError("text_temperature must lie in (0, 1).")
        if not 0.0 < self.mask_floor < 0.5:
            raise ValueError("mask_floor must lie in (0, 0.5).")

class _EventDecoderLayer(nn.Module):
    """Decode events against entity structure and dense visual evidence."""

    def __init__(self, cfg: CoRISPEventFieldConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.num_heads = cfg.attention_heads
        self.self_attention = nn.MultiheadAttention(
            d, cfg.attention_heads, dropout=cfg.dropout, batch_first=True
        )
        self.entity_attention = nn.MultiheadAttention(
            d, cfg.attention_heads, dropout=cfg.dropout, batch_first=True
        )
        self.visual_attention = nn.MultiheadAttention(
            d, cfg.attention_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, d),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm_self = nn.LayerNorm(d)
        self.norm_entity = nn.LayerNorm(d)
        self.norm_visual = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)

    def forward(
        self,
        events: torch.Tensor,
        entity_memory: torch.Tensor,
        visual_memory: torch.Tensor,
        *,
        entity_padding_mask: torch.Tensor | None,
        visual_log_gate: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, _ = self.self_attention(events, events, events, need_weights=False)
        events = self.norm_self(events + self.dropout(update))
        update, _ = self.entity_attention(
            events,
            entity_memory,
            entity_memory,
            key_padding_mask=entity_padding_mask,
            need_weights=False,
        )
        events = self.norm_entity(events + self.dropout(update))

        attention_mask = None
        if visual_log_gate is not None:
            if visual_log_gate.shape != (
                events.shape[0],
                events.shape[1],
                visual_memory.shape[1],
            ):
                raise ValueError("visual_log_gate must be [B,G,P].")
            attention_mask = visual_log_gate[:, None].expand(
                -1, self.num_heads, -1, -1
            )
            attention_mask = attention_mask.reshape(
                events.shape[0] * self.num_heads,
                events.shape[1],
                visual_memory.shape[1],
            )
        update, weights = self.visual_attention(
            events,
            visual_memory,
            visual_memory,
            attn_mask=attention_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        events = self.norm_visual(events + self.dropout(update))
        events = self.norm_ffn(events + self.dropout(self.ffn(events)))
        return events, weights.mean(dim=1)

class CoRISPEventField(nn.Module):
    """Role-qualified latent event hypergraph with exact event marginalization."""

    def __init__(
        self,
        cfg: CoRISPEventFieldConfig,
        *,
        action_prototypes: torch.Tensor,
        object_prototypes: torch.Tensor,
        valid_role_mask: torch.Tensor | None = None,
        human_label: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.human_label = int(human_label)
        if action_prototypes.shape != (cfg.num_actions, cfg.semantic_dim):
            raise ValueError(
                "action_prototypes must be [num_actions, semantic_dim]."
            )
        if object_prototypes.shape != (
            cfg.num_object_classes,
            cfg.semantic_dim,
        ):
            raise ValueError(
                "object_prototypes must be [num_object_classes, semantic_dim]."
            )
        if valid_role_mask is None:
            valid_role_mask = torch.ones(
                cfg.num_actions, len(cfg.roles), dtype=torch.bool
            )
        if valid_role_mask.shape != (cfg.num_actions, len(cfg.roles)):
            raise ValueError("valid_role_mask must be [num_actions, num_roles].")
        self.register_buffer(
            "action_prototypes",
            F.normalize(action_prototypes.float(), dim=-1),
        )
        self.register_buffer(
            "object_prototypes",
            F.normalize(object_prototypes.float(), dim=-1),
        )
        self.register_buffer("valid_role_mask", valid_role_mask.bool())

        d = cfg.d_model
        roles = len(cfg.roles)
        self.event_queries = nn.Embedding(cfg.num_event_slots, d)
        self.role_queries = nn.Embedding(roles, d)
        self.visual_projection = nn.Conv2d(cfg.visual_dim, d, kernel_size=1)
        self.detector_dense_scale = nn.Parameter(torch.tensor(0.1))
        self.global_projection = nn.Sequential(
            nn.LayerNorm(cfg.semantic_dim), nn.Linear(cfg.semantic_dim, d)
        )
        self.object_projection = nn.Sequential(
            nn.LayerNorm(cfg.semantic_dim), nn.Linear(cfg.semantic_dim, d)
        )
        self.entity_visual_projection = nn.Sequential(
            nn.LayerNorm(cfg.visual_dim), nn.Linear(cfg.visual_dim, d)
        )
        self.box_projection = nn.Sequential(
            nn.Linear(4, d), nn.GELU(), nn.Linear(d, d)
        )
        self.entity_norm = nn.LayerNorm(d)
        self.visual_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(
            [_EventDecoderLayer(cfg) for _ in range(cfg.decoder_layers)]
        )

        self.evidence_basis = nn.Linear(d, cfg.evidence_bases)
        self.event_basis_mixture = nn.Linear(d, cfg.evidence_bases)
        self.event_presence = nn.Linear(d, 1)
        self.event_action_projection = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, cfg.semantic_dim)
        )
        self.action_bias = nn.Parameter(torch.zeros(cfg.num_actions))
        self.logit_scale = nn.Parameter(
            torch.tensor(log(1.0 / cfg.text_temperature), dtype=torch.float32)
        )
        self.agent_query = nn.Linear(d, d)
        self.role_query = nn.Linear(d, d)
        self.entity_key = nn.Linear(d, d)
        self.role_visibility = nn.Linear(d, roles)
        self.pair_key = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.pair_event_query = nn.Linear(d, d)
        self.null_agent_query = nn.Linear(d, d)

    @property
    def num_roles(self) -> int:
        return len(self.cfg.roles)

    @staticmethod
    def _pool_boxes(grid: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 4 or boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("grid and boxes must be [B,D,H,W] and [B,E,4].")
        batch, channels, height, width = grid.shape
        if boxes.shape[0] != batch:
            raise ValueError("grid and boxes must share the batch dimension.")
        y = (torch.arange(height, device=grid.device, dtype=boxes.dtype) + 0.5) / height
        x = (torch.arange(width, device=grid.device, dtype=boxes.dtype) + 0.5) / width
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
            "behw,bdhw->bed",
            weights / counts.clamp_min(1.0),
            grid,
        )
        empty = counts.flatten(2).squeeze(-1) == 0
        if empty.any():
            center_x = ((x1 + x2) * 0.5 * width).floor().long().clamp(0, width - 1)
            center_y = ((y1 + y2) * 0.5 * height).floor().long().clamp(0, height - 1)
            batch_index = torch.arange(batch, device=grid.device)[:, None].expand_as(center_x)
            nearest = grid.permute(0, 2, 3, 1)[batch_index, center_y, center_x]
            pooled = torch.where(empty[..., None], nearest, pooled)
        return pooled

    @staticmethod
    def marginalize_event_contributions(contribution: torch.Tensor) -> torch.Tensor:
        """Compute an exact noisy-OR marginal over the event dimension."""

        if contribution.ndim < 2:
            raise ValueError("event contribution must contain batch and event axes.")
        eps = torch.finfo(contribution.dtype).eps
        bounded = contribution.clamp(0.0, 1.0 - eps)
        return -torch.expm1(torch.log1p(-bounded).sum(dim=1))

    @staticmethod
    def _probability_logit(probability: torch.Tensor) -> torch.Tensor:
        eps = max(torch.finfo(probability.dtype).eps, 1e-6)
        return torch.logit(probability.clamp(eps, 1.0 - eps))

    def _decode(
        self,
        entity_memory: torch.Tensor,
        visual_memory: torch.Tensor,
        entity_padding_mask: torch.Tensor | None,
        visual_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events = self.event_queries.weight[None].expand(
            entity_memory.shape[0], -1, -1
        )
        log_gate = None
        if visual_gate is not None:
            floor = self.cfg.mask_floor
            log_gate = visual_gate.clamp(floor, 1.0).log()
        attention = visual_memory.new_zeros(
            entity_memory.shape[0], self.cfg.num_event_slots, visual_memory.shape[1]
        )
        for layer in self.layers:
            events, attention = layer(
                events,
                entity_memory,
                visual_memory,
                entity_padding_mask=entity_padding_mask,
                visual_log_gate=log_gate,
            )
        return events, attention

    def _build_memories(
        self,
        packet: PairPacket,
        visual_grid: torch.Tensor,
        global_semantic: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if packet.entity_feats is None or packet.entity_boxes is None:
            raise ValueError("The event field requires entity boxes and features.")
        if packet.entity_labels is None:
            raise ValueError("The event field requires entity class labels.")
        if visual_grid.shape[:2] != (packet.batch_size, self.cfg.visual_dim):
            raise ValueError(
                f"visual_grid must be [B,{self.cfg.visual_dim},H,W]."
            )
        if global_semantic.shape != (packet.batch_size, self.cfg.semantic_dim):
            raise ValueError(
                f"global_semantic must be [B,{self.cfg.semantic_dim}]."
            )
        if (packet.entity_labels < 0).any() or (
            packet.entity_labels >= self.cfg.num_object_classes
        ).any():
            raise ValueError("entity_labels is outside the object prototype bank.")

        visual = self.visual_projection(visual_grid.to(dtype=packet.pair_feats.dtype))
        detector_dense = F.interpolate(
            packet.dense_tokens,
            size=visual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        visual = self.visual_norm(
            (visual + self.detector_dense_scale.to(visual.dtype) * detector_dense)
            .flatten(2)
            .transpose(1, 2)
        )

        pooled_visual = self._pool_boxes(visual_grid, packet.entity_boxes)
        object_semantic = self.object_prototypes[packet.entity_labels]
        entity = packet.entity_feats
        entity = self.entity_norm(
            entity
            + self.entity_visual_projection(pooled_visual.to(dtype=entity.dtype))
            + self.object_projection(object_semantic.to(dtype=entity.dtype))
            + self.box_projection(packet.entity_boxes.to(dtype=entity.dtype))
        )
        global_token = self.global_projection(
            global_semantic.to(dtype=entity.dtype)
        )[:, None]
        entity_memory = torch.cat([global_token, entity], dim=1)
        entity_padding_mask = None
        if packet.entity_valid_mask is not None:
            entity_padding_mask = torch.cat(
                [
                    torch.zeros(
                        packet.batch_size,
                        1,
                        dtype=torch.bool,
                        device=entity.device,
                    ),
                    ~packet.entity_valid_mask,
                ],
                dim=1,
            )
        return entity, visual, entity_padding_mask

    def _support_mask(
        self,
        events: torch.Tensor,
        visual_memory: torch.Tensor,
        visual_attention: torch.Tensor,
    ) -> torch.Tensor:
        basis_logits = self.evidence_basis(visual_memory)
        mixtures = self.event_basis_mixture(events).softmax(dim=-1)
        basis_gate = torch.sigmoid(
            torch.einsum("bgk,bpk->bgp", mixtures, basis_logits)
        )
        attention = visual_attention / visual_attention.mean(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        log_attention = attention.clamp_min(self.cfg.mask_floor).log()
        basis_logit = torch.logit(
            basis_gate.clamp(self.cfg.mask_floor, 1.0 - self.cfg.mask_floor)
        )
        return torch.sigmoid(basis_logit + log_attention).clamp(
            self.cfg.mask_floor, 1.0 - self.cfg.mask_floor
        )

    def _variant_probabilities(
        self,
        events: torch.Tensor,
        entity_state: torch.Tensor,
        packet: PairPacket,
    ) -> dict[str, torch.Tensor]:
        if packet.subject_indices is None or packet.object_indices is None:
            raise ValueError("The event field requires pair-to-entity indices.")
        batch, event_count, d_model = events.shape
        entity_count = entity_state.shape[1]
        pair_count = packet.num_pairs
        action_features = F.normalize(self.event_action_projection(events), dim=-1)
        action_logits = (
            self.logit_scale.clamp(max=log(100.0)).exp()
            * torch.einsum(
                "bgs,as->bga",
                action_features,
                self.action_prototypes.to(dtype=action_features.dtype),
            )
            + self.action_bias
        )
        action_probability = action_logits.sigmoid()
        presence_probability = self.event_presence(events).squeeze(-1).sigmoid()

        entity_key = self.entity_key(entity_state)
        agent_logits = torch.einsum(
            "bgd,bed->bge", self.agent_query(events), entity_key
        ) / sqrt(float(d_model))
        human_mask = packet.entity_labels == self.human_label
        if packet.entity_valid_mask is not None:
            human_mask = human_mask & packet.entity_valid_mask
        agent_probability = agent_logits.sigmoid() * human_mask[:, None].to(
            dtype=agent_logits.dtype
        )

        role_event = self.role_query(events)[:, :, None] + self.role_queries.weight[
            None, None
        ]
        role_entity_logits = torch.einsum(
            "bgrd,bed->bgre", role_event, entity_key
        ) / sqrt(float(d_model))
        role_entity_probability = role_entity_logits.sigmoid()
        if packet.entity_valid_mask is not None:
            role_entity_probability = role_entity_probability * packet.entity_valid_mask[
                :, None, None
            ].to(role_entity_probability.dtype)
        role_visibility_probability = self.role_visibility(events).sigmoid()

        subject_state = torch.gather(
            entity_state,
            1,
            packet.subject_indices[..., None].expand(-1, -1, d_model),
        )
        object_state = torch.gather(
            entity_state,
            1,
            packet.object_indices[..., None].expand(-1, -1, d_model),
        )
        pair_state = self.pair_key(
            packet.pair_feats + 0.5 * (subject_state + object_state)
        )
        pair_logits = torch.einsum(
            "bgd,bqd->bgq", self.pair_event_query(events), pair_state
        ) / sqrt(float(d_model))
        pair_probability = pair_logits.sigmoid()

        subject_agent = torch.gather(
            agent_probability,
            2,
            packet.subject_indices[:, None].expand(-1, event_count, -1),
        )
        object_index = packet.object_indices[:, None, None].expand(
            -1, event_count, self.num_roles, -1
        )
        role_object = torch.gather(role_entity_probability, 3, object_index)
        role_object = role_object.permute(0, 1, 3, 2)
        contribution = (
            presence_probability[:, :, None, None, None]
            * action_probability[:, :, None, :, None]
            * subject_agent[:, :, :, None, None]
            * role_visibility_probability[:, :, None, None, :]
            * role_object[:, :, :, None, :]
            * pair_probability[:, :, :, None, None]
        )
        valid = self.valid_role_mask[None, None, None].to(contribution.dtype)
        contribution = contribution * valid
        joint_probability = self.marginalize_event_contributions(contribution)

        null_compatibility = torch.einsum(
            "bgd,bed->bge", self.null_agent_query(events), entity_key
        ).div(sqrt(float(d_model))).sigmoid()
        null_contribution = (
            presence_probability[:, :, None, None, None]
            * action_probability[:, :, None, :, None]
            * agent_probability[:, :, :, None, None]
            * (1.0 - role_visibility_probability[:, :, None, None, :])
            * null_compatibility[:, :, :, None, None]
        )
        null_contribution = null_contribution * valid
        null_probability = self.marginalize_event_contributions(null_contribution)
        return {
            "events": events,
            "event_presence_probs": presence_probability,
            "event_action_logits": action_logits,
            "event_action_probs": action_probability,
            "event_agent_probs": agent_probability,
            "event_role_visibility_probs": role_visibility_probability,
            "event_role_entity_probs": role_entity_probability,
            "event_pair_probs": pair_probability,
            "event_joint_contribution": contribution,
            "joint_role_probs": joint_probability,
            "joint_role_logits": self._probability_logit(joint_probability),
            "null_role_probs": null_probability,
            "null_role_logits": self._probability_logit(null_probability),
        }

    def forward(
        self,
        packet: PairPacket,
        *,
        visual_grid: torch.Tensor,
        global_semantic: torch.Tensor,
        return_interventions: bool = True,
    ) -> dict[str, torch.Tensor]:
        packet.validate()
        entity_state, visual_memory, entity_padding_mask = self._build_memories(
            packet, visual_grid, global_semantic
        )
        factual_events, attention = self._decode(
            torch.cat(
                [self.global_projection(global_semantic)[:, None], entity_state], dim=1
            ),
            visual_memory,
            entity_padding_mask,
        )
        support_mask = self._support_mask(
            factual_events, visual_memory, attention
        )
        output = self._variant_probabilities(factual_events, entity_state, packet)
        output["support_mask"] = support_mask
        output["visual_attention"] = attention
        if return_interventions:
            sufficient_events, _ = self._decode(
                torch.cat(
                    [self.global_projection(global_semantic)[:, None], entity_state],
                    dim=1,
                ),
                visual_memory,
                entity_padding_mask,
                support_mask,
            )
            erased_events, _ = self._decode(
                torch.cat(
                    [self.global_projection(global_semantic)[:, None], entity_state],
                    dim=1,
                ),
                visual_memory,
                entity_padding_mask,
                1.0 - support_mask,
            )
            sufficient = self._variant_probabilities(
                sufficient_events, entity_state, packet
            )
            erased = self._variant_probabilities(erased_events, entity_state, packet)
            for key in (
                "joint_role_probs",
                "joint_role_logits",
                "null_role_probs",
                "null_role_logits",
            ):
                output[f"sufficient_{key}"] = sufficient[key]
                output[f"erased_{key}"] = erased[key]
        return output

def event_field_regularization(
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Identifiability terms for a sparse, non-duplicated latent event set."""

    events = F.normalize(output["events"], dim=-1)
    presence = output["event_presence_probs"]
    similarity = torch.matmul(events, events.transpose(-1, -2)).square()
    count = events.shape[1]
    off_diagonal = ~torch.eye(
        count, dtype=torch.bool, device=events.device
    )[None]
    active_pair = presence[:, :, None] * presence[:, None, :]
    duplicate = (similarity * active_pair)[off_diagonal.expand_as(similarity)].mean()
    return {
        "event_cardinality": presence.mean(),
        "event_duplicate": duplicate,
        "evidence_compactness": output["support_mask"].mean(),
    }
