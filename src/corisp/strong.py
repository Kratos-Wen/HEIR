from __future__ import annotations

from dataclasses import dataclass

import torch

import torch.nn as nn

from corisp.joint import CoRISPJointConfig, CoRISPJointDecoder, box_iou_xyxy, relation_geometry_features

from corisp.pair_packet import PairPacket

CORISP_STRONG_ARCHITECTURE_VERSION = "corisp_joint_v3_interaction_closure"

_PAIR_RELATION_DIM = 16

@dataclass(frozen=True)
class CoRISPStrongConfig:
    """Configuration for inherited detector-normalized proposal utilities."""

    joint: CoRISPJointConfig
    architecture_version: str = CORISP_STRONG_ARCHITECTURE_VERSION
    context_layers: int = 2
    context_heads: int = 8
    context_ffn_dim: int = 1024
    context_dropout: float = 0.1
    context_residual_init: float = 0.0
    closure_scale_init: float = 1.0
    closure_logit_clip: float = 8.0
    neutral_event_probability: float = 0.5
    joint_refinement_steps: int = 1

    def __post_init__(self) -> None:
        if self.architecture_version != CORISP_STRONG_ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported strong architecture {self.architecture_version!r}; "
                f"expected {CORISP_STRONG_ARCHITECTURE_VERSION!r}."
            )
        if not self.joint.use_event_reasoning:
            raise ValueError("The interaction-closure branch requires latent event reasoning.")
        if self.context_layers <= 0 or self.context_heads <= 0 or self.context_ffn_dim <= 0:
            raise ValueError("Context layer, head, and FFN counts must be positive.")
        if self.joint.d_model % self.context_heads != 0:
            raise ValueError("joint.d_model must be divisible by context_heads.")
        if self.closure_logit_clip <= 0:
            raise ValueError("closure_logit_clip must be positive.")
        if self.closure_scale_init < 0:
            raise ValueError("closure_scale_init must be non-negative.")
        if not 0.0 < self.neutral_event_probability < 1.0:
            raise ValueError("neutral_event_probability must lie in (0,1).")
        if self.joint_refinement_steps not in {1, 2}:
            raise ValueError("joint_refinement_steps must be one or two.")

def _centers(boxes: torch.Tensor) -> torch.Tensor:
    return 0.5 * (boxes[..., :2] + boxes[..., 2:])

def pair_relation_features(packet: PairPacket) -> torch.Tensor:
    """Dense pair-to-pair relation features without a hand-coded HOI rule."""

    sb = packet.subject_boxes
    ob = packet.object_boxes
    ss_iou = box_iou_xyxy(sb, sb)
    oo_iou = box_iou_xyxy(ob, ob)
    so_iou = box_iou_xyxy(sb, ob)
    os_iou = box_iou_xyxy(ob, sb)

    sc = _centers(sb)
    oc = _centers(ob)
    ss_dist = torch.cdist(sc, sc, p=2)
    oo_dist = torch.cdist(oc, oc, p=2)
    so_dist = torch.cdist(sc, oc, p=2)
    os_dist = torch.cdist(oc, sc, p=2)

    b, q = packet.pair_feats.shape[:2]
    zeros = packet.pair_feats.new_zeros((b, q, q))
    shared_ss = shared_oo = shared_so = shared_os = zeros
    if packet.subject_indices is not None and packet.object_indices is not None:
        si = packet.subject_indices
        oi = packet.object_indices
        shared_ss = (si[:, :, None] == si[:, None, :]).to(dtype=packet.pair_feats.dtype)
        shared_oo = (oi[:, :, None] == oi[:, None, :]).to(dtype=packet.pair_feats.dtype)
        shared_so = (si[:, :, None] == oi[:, None, :]).to(dtype=packet.pair_feats.dtype)
        shared_os = (oi[:, :, None] == si[:, None, :]).to(dtype=packet.pair_feats.dtype)

    confidence_product = zeros
    confidence_min = zeros
    if packet.proposal_scores is not None:
        pair_conf = packet.proposal_scores.prod(dim=-1)
        confidence_product = pair_conf[:, :, None] * pair_conf[:, None, :]
        confidence_min = torch.minimum(pair_conf[:, :, None], pair_conf[:, None, :])

    return torch.stack(
        [
            ss_iou,
            oo_iou,
            so_iou,
            os_iou,
            torch.exp(-4.0 * ss_dist),
            torch.exp(-4.0 * oo_dist),
            torch.exp(-4.0 * so_dist),
            torch.exp(-4.0 * os_dist),
            shared_ss,
            shared_oo,
            shared_so,
            shared_os,
            confidence_product,
            confidence_min,
            (ss_iou * oo_iou).sqrt(),
            (so_iou * os_iou).sqrt(),
        ],
        dim=-1,
    )

class InteractionContextBlock(nn.Module):
    """Relation-biased pair attention followed by dense visual retrieval."""

    def __init__(self, cfg: CoRISPStrongConfig) -> None:
        super().__init__()
        d = cfg.joint.d_model
        h = cfg.context_heads
        self.num_heads = h
        self.pair_norm = nn.LayerNorm(d)
        self.dense_query_norm = nn.LayerNorm(d)
        self.dense_memory_norm = nn.LayerNorm(d)
        self.ffn_norm = nn.LayerNorm(d)
        self.pair_attention = nn.MultiheadAttention(d, h, dropout=cfg.context_dropout, batch_first=True)
        self.dense_attention = nn.MultiheadAttention(d, h, dropout=cfg.context_dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.context_ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.context_dropout),
            nn.Linear(cfg.context_ffn_dim, d),
        )
        self.relation_bias = nn.Sequential(
            nn.Linear(_PAIR_RELATION_DIM, d),
            nn.GELU(),
            nn.Linear(d, h),
        )
        init = float(cfg.context_residual_init)
        self.pair_scale = nn.Parameter(torch.tensor(init))
        self.dense_scale = nn.Parameter(torch.tensor(init))
        self.ffn_scale = nn.Parameter(torch.tensor(init))

    def forward(
        self,
        pair_feats: torch.Tensor,
        dense_tokens: torch.Tensor,
        relation_features: torch.Tensor,
        pair_valid_mask: torch.Tensor,
        dense_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, q, _ = pair_feats.shape
        relation_bias = self.relation_bias(relation_features).permute(0, 3, 1, 2)
        key_invalid = ~pair_valid_mask
        relation_bias = relation_bias.masked_fill(key_invalid[:, None, None, :], float("-inf"))
        attn_mask = relation_bias.reshape(b * self.num_heads, q, q)
        pair_input = self.pair_norm(pair_feats)
        pair_context, _ = self.pair_attention(
            pair_input,
            pair_input,
            pair_input,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = pair_feats + self.pair_scale.to(dtype=pair_feats.dtype) * pair_context
        x = x.masked_fill(~pair_valid_mask[..., None], 0.0)

        dense = dense_tokens.flatten(2).transpose(1, 2)
        dense = self.dense_memory_norm(dense)
        dense_key_padding = None
        if dense_valid_mask is not None:
            dense_key_padding = ~dense_valid_mask.flatten(1)
        dense_context, _ = self.dense_attention(
            self.dense_query_norm(x),
            dense,
            dense,
            key_padding_mask=dense_key_padding,
            need_weights=False,
        )
        x = x + self.dense_scale.to(dtype=x.dtype) * dense_context
        x = x + self.ffn_scale.to(dtype=x.dtype) * self.ffn(self.ffn_norm(x))
        x = x.masked_fill(~pair_valid_mask[..., None], 0.0)
        return x, relation_bias, dense_context

class InteractionContextEncoder(nn.Module):
    """Fuse endpoint, geometry, confidence, cross-pair, and image evidence."""

    def __init__(self, cfg: CoRISPStrongConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.joint.d_model
        # Pair + subject + object + product + geometry + two proposal scores.
        self.input_fusion = nn.Sequential(
            nn.Linear(d * 4 + 19, cfg.context_ffn_dim),
            nn.LayerNorm(cfg.context_ffn_dim),
            nn.GELU(),
            nn.Linear(cfg.context_ffn_dim, d),
        )
        nn.init.zeros_(self.input_fusion[-1].weight)
        nn.init.zeros_(self.input_fusion[-1].bias)
        self.blocks = nn.ModuleList([InteractionContextBlock(cfg) for _ in range(cfg.context_layers)])
        # Preserve the host detector's pair representation at initialization;
        # every contextual update is introduced through an explicit residual.
        self.output_norm = nn.Identity()

    def forward(self, packet: PairPacket) -> dict[str, torch.Tensor]:
        packet.validate()
        pair_valid = packet.valid_pairs
        subject = packet.pair_feats if packet.subject_feats is None else packet.subject_feats
        obj = packet.pair_feats if packet.object_feats is None else packet.object_feats
        scores = packet.pair_feats.new_zeros((*packet.pair_feats.shape[:2], 2))
        if packet.proposal_scores is not None:
            scores = packet.proposal_scores.to(dtype=packet.pair_feats.dtype)
        geometry = relation_geometry_features(packet.subject_boxes, packet.object_boxes).to(
            dtype=packet.pair_feats.dtype
        )
        fusion_delta = self.input_fusion(
            torch.cat([packet.pair_feats, subject, obj, subject * obj, geometry, scores], dim=-1)
        )
        x = packet.pair_feats + fusion_delta
        relation_features = pair_relation_features(packet)
        relation_bias = x.new_zeros(
            (packet.batch_size, self.cfg.context_heads, packet.num_pairs, packet.num_pairs)
        )
        dense_context = torch.zeros_like(x)
        for block in self.blocks:
            x, relation_bias, dense_context = block(
                x,
                packet.dense_tokens,
                relation_features,
                pair_valid,
                packet.dense_valid_mask,
            )
        x = self.output_norm(x).masked_fill(~pair_valid[..., None], 0.0)
        return {
            "interaction_pair_feats": x,
            "interaction_input_delta": fusion_delta,
            "interaction_relation_features": relation_features,
            "interaction_relation_bias": relation_bias,
            "interaction_dense_context": dense_context,
        }

class EventRoleProbabilityClosure(nn.Module):
    """Put absolute event participation inside the positive role states."""

    def __init__(self, cfg: CoRISPStrongConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.closure_scale_init == 0:
            self.log_closure_scale = None
            self.register_buffer("fixed_closure_scale", torch.tensor(0.0))
        else:
            self.log_closure_scale = nn.Parameter(torch.tensor(float(cfg.closure_scale_init)).log())
            self.register_buffer("fixed_closure_scale", torch.tensor(float("nan")))

    def effective_scale(self) -> torch.Tensor:
        if self.log_closure_scale is None:
            return self.fixed_closure_scale
        return self.log_closure_scale.exp()

    def event_participation(
        self,
        event_presence_logits: torch.Tensor,
        edge_membership_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if event_presence_logits.ndim != 2:
            raise ValueError("event_presence_logits must be [B,G].")
        if edge_membership_logits.ndim != 4:
            raise ValueError("edge_membership_logits must be [B,Q,C,G].")
        if event_presence_logits.shape[0] != edge_membership_logits.shape[0]:
            raise ValueError("Event presence and membership batch sizes differ.")
        if event_presence_logits.shape[1] != edge_membership_logits.shape[-1]:
            raise ValueError("Event presence and membership slot counts differ.")
        active = event_presence_logits.sigmoid()[:, None, None, :] * edge_membership_logits.sigmoid()
        active = active.clamp(1e-7, 1.0 - 1e-7)
        log_no_event = torch.log1p(-active).sum(dim=-1)
        probability = (-torch.expm1(log_no_event)).clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(probability).clamp(
            -float(self.cfg.closure_logit_clip),
            float(self.cfg.closure_logit_clip),
        )
        return probability, logits

    @staticmethod
    def _close_states(
        state_logits: torch.Tensor,
        participation_logits: torch.Tensor,
        scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state_logits.ndim != 4:
            raise ValueError("joint state logits must be [B,Q,C,1+R].")
        if state_logits.shape[:-1] != participation_logits.shape:
            raise ValueError("Participation logits must match [B,Q,C].")
        null = state_logits[..., :1]
        positive = state_logits[..., 1:] + scale * participation_logits[..., None]
        closed = torch.cat([null, positive], dim=-1)
        hoi_logits = torch.logsumexp(positive, dim=-1) - null.squeeze(-1)
        role_probs = positive.softmax(dim=-1)
        edge_role_prob = hoi_logits.sigmoid()[..., None] * role_probs
        pair_role_prob = edge_role_prob.amax(dim=2).clamp(1e-6, 1.0 - 1e-6)
        return {
            "joint_state_logits": closed,
            "hoi_logits": hoi_logits,
            "closed_conditional_role_probs": role_probs,
            "pair_role_logits": torch.logit(pair_role_prob),
        }

    def forward(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {
            "event_presence_logits",
            "event_edge_membership_logits",
            "joint_state_logits",
            "clean_joint_state_logits",
            "erased_joint_state_logits",
            "neutral_joint_state_logits",
        }
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"Cannot close event-role probability; missing {sorted(missing)}")
        probability, logits = self.event_participation(
            outputs["event_presence_logits"],
            outputs["event_edge_membership_logits"],
        )
        scale = self.effective_scale().to(device=logits.device, dtype=logits.dtype)
        full = self._close_states(outputs["joint_state_logits"], logits, scale)
        clean = self._close_states(outputs["clean_joint_state_logits"], logits, scale)
        erased = self._close_states(outputs["erased_joint_state_logits"], logits, scale)
        neutral = self._close_states(outputs["neutral_joint_state_logits"], logits, scale)
        return {
            **outputs,
            "preclosure_joint_state_logits": outputs["joint_state_logits"],
            "preclosure_hoi_logits": outputs["hoi_logits"],
            "preclosure_clean_hoi_logits": outputs["clean_hoi_logits"],
            "preclosure_erased_hoi_logits": outputs["erased_hoi_logits"],
            "event_participation_probs": probability,
            "event_participation_logits": logits,
            "event_no_event_probs": 1.0 - probability,
            "event_closure_scale": scale,
            "event_closure_nonnegative_penalty": torch.zeros_like(scale),
            **full,
            "clean_joint_state_logits": clean["joint_state_logits"],
            "clean_hoi_logits": clean["hoi_logits"],
            "erased_joint_state_logits": erased["joint_state_logits"],
            "erased_hoi_logits": erased["hoi_logits"],
            "neutral_joint_state_logits": neutral["joint_state_logits"],
            "neutral_hoi_logits": neutral["hoi_logits"],
        }

class CoRISPStrongDecoder(nn.Module):
    """Detector-agnostic joint interaction and role decoder.

    The semantic classifier remains owned by the host detector.  It must score
    ``interaction_pair_feats``, ``enhanced_pair_feats``, and
    ``erased_pair_feats`` with shared weights before calling
    :meth:`finalize_joint`.
    """

    def __init__(self, cfg: CoRISPStrongConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.context_encoder = InteractionContextEncoder(cfg)
        self.joint_decoder = CoRISPJointDecoder(cfg.joint)
        self.probability_closure = EventRoleProbabilityClosure(cfg)
        self._initialize_neutral_event_prior()

    def _initialize_neutral_event_prior(self) -> None:
        """Make noisy-OR participation neutral while keeping gradients live."""

        reasoner = self.joint_decoder.event_reasoner
        if reasoner is None:
            raise RuntimeError("The joint decoder requires an event reasoner.")
        slots = int(self.cfg.joint.num_event_slots)
        target = float(self.cfg.neutral_event_probability)
        active_per_slot = 1.0 - (1.0 - target) ** (1.0 / float(slots))
        # Edge membership starts at 0.5 because its logits are zero.
        presence_probability = min(max(active_per_slot / 0.5, 1e-5), 1.0 - 1e-5)
        presence_bias = torch.logit(torch.tensor(presence_probability))
        nn.init.zeros_(reasoner.pair_membership_proj.weight)
        nn.init.zeros_(reasoner.pair_membership_proj.bias)
        nn.init.zeros_(reasoner.edge_membership_basis[-1].weight)
        nn.init.zeros_(reasoner.edge_membership_basis[-1].bias)
        nn.init.zeros_(reasoner.event_presence.weight)
        nn.init.constant_(reasoner.event_presence.bias, float(presence_bias))

    def forward(self, packet: PairPacket) -> dict[str, torch.Tensor]:
        packet.validate()
        if packet.num_pairs == 0:
            raise ValueError("CoRISPStrongDecoder requires at least one valid pair.")
        if not packet.valid_pairs.all():
            raise ValueError("Pass compact_image(i) packets to CoRISPStrongDecoder.")
        context = self.context_encoder(packet)
        pair_feats = context["interaction_pair_feats"]
        outputs = self.joint_decoder(
            pair_feats=pair_feats,
            dense_tokens=packet.dense_tokens,
            subject_boxes=packet.subject_boxes,
            object_boxes=packet.object_boxes,
            subject_feats=packet.subject_feats,
            object_feats=packet.object_feats,
            entity_boxes=packet.entity_boxes,
            entity_feats=packet.entity_feats,
            erased_pair_feats_input=pair_feats,
        )
        if self.cfg.joint_refinement_steps == 2:
            first_outputs = outputs
            factual_feedback = first_outputs["enhanced_interlayer_feedback"]
            erased_feedback = first_outputs["erased_interlayer_feedback"]
            outputs = self.joint_decoder(
                pair_feats=pair_feats + factual_feedback,
                dense_tokens=packet.dense_tokens,
                subject_boxes=packet.subject_boxes,
                object_boxes=packet.object_boxes,
                subject_feats=packet.subject_feats,
                object_feats=packet.object_feats,
                entity_boxes=packet.entity_boxes,
                entity_feats=packet.entity_feats,
                erased_pair_feats_input=pair_feats + erased_feedback,
            )
            outputs.update(
                {
                    "first_pass_evidence_gate": first_outputs["evidence_gate"],
                    "first_pass_event_participation_inputs": first_outputs[
                        "event_edge_membership_logits"
                    ],
                    "applied_enhanced_interlayer_feedback": factual_feedback,
                    "applied_erased_interlayer_feedback": erased_feedback,
                }
            )
        return {
            **outputs,
            **context,
            "pair_valid_mask": packet.valid_pairs,
            "subject_indices": packet.subject_indices,
            "object_indices": packet.object_indices,
            "proposal_scores": packet.proposal_scores,
            "object_labels": packet.object_labels,
        }

    def forward_images(self, packet: PairPacket) -> list[dict[str, torch.Tensor]]:
        """Compact and process every non-empty image in a padded packet."""

        outputs: list[dict[str, torch.Tensor]] = []
        for i in range(packet.batch_size):
            compact = packet.compact_image(i)
            if compact.num_pairs == 0:
                outputs.append({"pair_valid_mask": compact.valid_pairs})
            else:
                outputs.append(self(compact))
        return outputs

    def finalize_joint(
        self,
        base_logits: torch.Tensor,
        enhanced_logits: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        erased_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        preclosure = self.joint_decoder.finalize_joint(
            base_logits,
            enhanced_logits,
            outputs,
            erased_logits,
        )
        return self.probability_closure(preclosure)
