from __future__ import annotations

from dataclasses import dataclass

import torch

import torch.nn as nn

import torch.nn.functional as F

from corisp.schema import DEFAULT_ROLES, normalize_role_name

CORISP_LEGACY_ARCHITECTURE_VERSION = "corisp_joint_v1"

CORISP_ARCHITECTURE_VERSION = "corisp_joint_v2_eventset"

SUPPORTED_CORISP_ARCHITECTURE_VERSIONS = (
    CORISP_LEGACY_ARCHITECTURE_VERSION,
    CORISP_ARCHITECTURE_VERSION,
)

_RELATION_GEOMETRY_DIM = 17

def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized or absolute xyxy boxes."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((*boxes1.shape[:-1], boxes2.shape[-2]))
    lt = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[..., :, 2] - boxes1[..., :, 0]).clamp_min(0.0) * (
        boxes1[..., :, 3] - boxes1[..., :, 1]
    ).clamp_min(0.0)
    area2 = (boxes2[..., :, 2] - boxes2[..., :, 0]).clamp_min(0.0) * (
        boxes2[..., :, 3] - boxes2[..., :, 1]
    ).clamp_min(0.0)
    return inter / (area1[..., :, None] + area2[..., None, :] - inter).clamp_min(1e-6)

def relation_geometry_features(subject_boxes: torch.Tensor, object_boxes: torch.Tensor) -> torch.Tensor:
    """Geometry descriptor for one human-entity query pair."""

    subject_boxes = subject_boxes.clamp(0.0, 1.0)
    object_boxes = object_boxes.clamp(0.0, 1.0)
    sub_center = 0.5 * (subject_boxes[..., :2] + subject_boxes[..., 2:])
    obj_center = 0.5 * (object_boxes[..., :2] + object_boxes[..., 2:])
    sub_wh = (subject_boxes[..., 2:] - subject_boxes[..., :2]).clamp_min(1e-4)
    obj_wh = (object_boxes[..., 2:] - object_boxes[..., :2]).clamp_min(1e-4)
    delta = obj_center - sub_center
    normalized_delta = delta / sub_wh
    log_size_ratio = torch.log(obj_wh / sub_wh)

    lt = torch.maximum(subject_boxes[..., :2], object_boxes[..., :2])
    rb = torch.minimum(subject_boxes[..., 2:], object_boxes[..., 2:])
    inter_wh = (rb - lt).clamp_min(0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    sub_area = sub_wh[..., 0] * sub_wh[..., 1]
    obj_area = obj_wh[..., 0] * obj_wh[..., 1]
    union = (sub_area + obj_area - inter).clamp_min(1e-6)

    return torch.cat(
        [
            sub_center,
            obj_center,
            sub_wh,
            obj_wh,
            delta,
            normalized_delta,
            log_size_ratio,
            (inter / union)[..., None],
            (obj_area / sub_area.clamp_min(1e-6))[..., None],
            union[..., None],
        ],
        dim=-1,
    )

@dataclass(frozen=True)
class CoRISPJointConfig:
    d_model: int
    architecture_version: str = CORISP_ARCHITECTURE_VERSION
    num_hoi_classes: int = 1
    roles: tuple[str, ...] = DEFAULT_ROLES
    num_bases: int = 8
    hidden_dim: int = 512
    evidence_dim: int = 256
    assignment_temperature: float = 0.05
    mask_temperature: float = 0.7
    decoder_refine_alpha: float = 0.35
    evidence_attention_heads: int = 4
    interlayer_feedback_alpha: float = 0.15
    use_evidence_attention: bool = True
    use_relation_geometry: bool = True
    use_event_reasoning: bool = True
    num_event_slots: int = 16
    event_decoder_layers: int = 2
    event_attention_heads: int = 4
    event_ffn_dim: int = 1024
    event_dropout: float = 0.1
    event_membership_temperature: float = 0.1
    event_refine_alpha: float = 0.20
    joint_temperature: float = 0.7
    residual_scale_init: float = 0.0
    semantic_prior: torch.Tensor | None = None
    semantic_role_valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        normalized = tuple(normalize_role_name(name) for name in self.roles)
        object.__setattr__(self, "roles", normalized)
        if len(normalized) == 0 or len(set(normalized)) != len(normalized):
            raise ValueError("CoRISP roles must be non-empty and unique after normalization.")
        if self.architecture_version not in SUPPORTED_CORISP_ARCHITECTURE_VERSIONS:
            raise ValueError(
                f"Unsupported CoRISP architecture version {self.architecture_version!r}; "
                f"expected one of {SUPPORTED_CORISP_ARCHITECTURE_VERSIONS!r}."
            )
        if self.architecture_version == CORISP_LEGACY_ARCHITECTURE_VERSION and self.use_event_reasoning:
            raise ValueError("corisp_joint_v1 checkpoints cannot enable the v2 latent event set.")
        if self.num_event_slots <= 0:
            raise ValueError("num_event_slots must be positive.")
        if self.event_decoder_layers <= 0:
            raise ValueError("event_decoder_layers must be positive.")
        if self.event_membership_temperature <= 0:
            raise ValueError("event_membership_temperature must be positive.")

    @property
    def num_roles(self) -> int:
        return len(self.roles)

class SceneEntityMemory(nn.Module):
    """Make all subject/object proposals visible to every interaction query."""

    def __init__(self, cfg: CoRISPJointConfig) -> None:
        super().__init__()
        self.entity_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(
        self,
        pair_feats: torch.Tensor,
        subject_boxes: torch.Tensor,
        object_boxes: torch.Tensor,
        subject_feats: torch.Tensor | None = None,
        object_feats: torch.Tensor | None = None,
        entity_boxes: torch.Tensor | None = None,
        entity_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (entity_boxes is None) != (entity_feats is None):
            raise ValueError("entity_boxes and entity_feats must be supplied together.")
        if entity_boxes is not None and entity_feats is not None:
            boxes = entity_boxes
            feats = entity_feats
        else:
            subject_feats = pair_feats if subject_feats is None else subject_feats
            object_feats = pair_feats if object_feats is None else object_feats
            boxes = torch.cat([subject_boxes, object_boxes], dim=1)
            feats = torch.cat([subject_feats, object_feats], dim=1)
        return {
            "entity_boxes": boxes.to(device=pair_feats.device, dtype=pair_feats.dtype).clamp(0.0, 1.0),
            "entity_feats": self.entity_proj(feats.to(device=pair_feats.device, dtype=pair_feats.dtype)),
        }

class LatentEventSetReasoner(nn.Module):
    """Infer a variable-cardinality set of shared events from pair queries.

    A fixed upper bound of event slots is used only as a computational budget.
    Event presence and sigmoid pair memberships allow unused slots, variable
    group sizes, and a pair to participate in more than one concurrent event.
    """

    def __init__(self, cfg: CoRISPJointConfig) -> None:
        super().__init__()
        self.cfg = cfg
        heads = int(cfg.event_attention_heads)
        if heads <= 0 or cfg.d_model % heads != 0:
            heads = 1
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=heads,
            dim_feedforward=int(cfg.event_ffn_dim),
            dropout=float(cfg.event_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_tokens = nn.Embedding(cfg.num_event_slots, cfg.d_model)
        self.pair_memory = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=int(cfg.event_decoder_layers),
            norm=nn.LayerNorm(cfg.d_model),
        )
        self.pair_membership_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.event_membership_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.edge_membership_basis = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_bases),
        )
        self.event_presence = nn.Linear(cfg.d_model, 1)
        self.event_context_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.pair_refiner = nn.Sequential(
            nn.Linear(cfg.d_model * 4, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.d_model),
        )
        self.refine_gate = nn.Sequential(nn.Linear(cfg.d_model * 4, cfg.d_model), nn.Sigmoid())
        nn.init.zeros_(self.edge_membership_basis[-1].weight)
        nn.init.zeros_(self.edge_membership_basis[-1].bias)
        nn.init.zeros_(self.pair_refiner[-1].weight)
        nn.init.zeros_(self.pair_refiner[-1].bias)

    def forward(
        self,
        pair_feats: torch.Tensor,
        relation_context: torch.Tensor,
        hoi_factors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pair_feats.ndim != 3 or relation_context.shape != pair_feats.shape:
            raise ValueError("pair_feats and relation_context must both be [B,Q,D].")
        if hoi_factors.shape != (self.cfg.num_hoi_classes, self.cfg.num_bases):
            raise ValueError("hoi_factors must be [C,K] for edge-conditioned event membership.")
        pair_memory = self.pair_memory(torch.cat([pair_feats, relation_context], dim=-1))
        tokens = self.event_tokens.weight.to(device=pair_feats.device, dtype=pair_feats.dtype)
        tokens = tokens[None].expand(pair_feats.shape[0], -1, -1)
        event_slots = self.decoder(tokens, pair_memory)

        pair_key = F.normalize(self.pair_membership_proj(pair_memory), dim=-1)
        event_key = F.normalize(self.event_membership_proj(event_slots), dim=-1)
        membership_logits = torch.einsum("bqd,bgd->bqg", pair_key, event_key)
        membership_logits = membership_logits / float(self.cfg.event_membership_temperature)
        pair_expanded = pair_memory[:, :, None, :].expand(-1, -1, self.cfg.num_event_slots, -1)
        event_expanded = event_slots[:, None, :, :].expand(-1, pair_feats.shape[1], -1, -1)
        edge_basis = self.edge_membership_basis(
            torch.cat(
                [pair_expanded, event_expanded, pair_expanded * event_expanded],
                dim=-1,
            )
        )
        class_delta = torch.einsum(
            "bqgk,ck->bqcg",
            edge_basis,
            hoi_factors.to(device=pair_feats.device, dtype=pair_feats.dtype),
        ) / max(float(self.cfg.num_bases) ** 0.5, 1.0)
        edge_membership_logits = membership_logits[:, :, None, :] + class_delta
        presence_logits = self.event_presence(event_slots).squeeze(-1)

        membership_prob = membership_logits.sigmoid()
        active_membership = membership_prob * presence_logits.sigmoid()[:, None, :]
        normalized_membership = active_membership / active_membership.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        event_context = torch.einsum("bqg,bgd->bqd", normalized_membership, event_slots)
        event_context = self.event_context_proj(event_context)
        refine_input = torch.cat(
            [pair_feats, event_context, pair_feats * event_context, relation_context],
            dim=-1,
        )
        refine_delta = self.pair_refiner(refine_input)
        refine_gate = self.refine_gate(refine_input)
        refined = pair_feats + float(self.cfg.event_refine_alpha) * refine_gate * refine_delta
        return {
            "event_slot_feats": event_slots,
            "event_presence_logits": presence_logits,
            "event_membership_logits": membership_logits,
            "event_membership_probs": membership_prob,
            "event_edge_membership_logits": edge_membership_logits,
            "event_edge_membership_probs": edge_membership_logits.sigmoid(),
            "event_edge_basis": edge_basis,
            "event_active_membership": active_membership,
            "event_context": event_context,
            "event_refine_gate": refine_gate,
            "event_refine_delta": refine_delta,
            "event_refined_pair_feats": refined,
        }

class LatentEvidenceField(nn.Module):
    """Query-conditioned spatial bases shared by all HOI-class-role states."""

    def __init__(self, cfg: CoRISPJointConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pixel_proj = nn.Conv2d(cfg.d_model, cfg.evidence_dim, kernel_size=1)
        self.basis_embed = nn.Embedding(cfg.num_bases, cfg.d_model)
        self.basis_query = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.evidence_dim),
        )

    @staticmethod
    def _pool(dense_tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        dense = dense_tokens.flatten(2)
        flat_masks = masks.flatten(-2)
        context = torch.einsum("bqkn,bdn->bqkd", flat_masks, dense)
        return context / flat_masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, pair_feats: torch.Tensor, dense_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if pair_feats.ndim != 3:
            raise ValueError("pair_feats must be [B,Q,D].")
        if dense_tokens.ndim != 4:
            raise ValueError("dense_tokens must be [B,D,H,W].")
        pixel = self.pixel_proj(dense_tokens)
        basis = self.basis_embed.weight.to(device=pair_feats.device, dtype=pair_feats.dtype)
        pair = pair_feats[:, :, None, :].expand(-1, -1, self.cfg.num_bases, -1)
        basis = basis[None, None].expand(pair_feats.shape[0], pair_feats.shape[1], -1, -1)
        query = self.basis_query(torch.cat([pair, basis], dim=-1))
        logits = torch.einsum("bdhw,bqkd->bqkhw", pixel, query)
        logits = logits / max(float(self.cfg.evidence_dim) ** 0.5, 1.0)
        masks = torch.sigmoid(logits / max(float(self.cfg.mask_temperature), 1e-4))
        return {
            "basis_mask_logits": logits,
            "basis_masks": masks,
            "basis_contexts": self._pool(dense_tokens, masks),
            "complement_basis_contexts": self._pool(dense_tokens, 1.0 - masks),
        }

class PriorResidualJointField(nn.Module):
    """Low-rank null-augmented HOI-class-role field around a frozen prior."""

    def __init__(self, cfg: CoRISPJointConfig) -> None:
        super().__init__()
        if cfg.num_hoi_classes <= 0:
            raise ValueError("num_hoi_classes must be positive.")
        if cfg.num_bases <= 0:
            raise ValueError("num_bases must be positive.")
        self.cfg = cfg
        self.evidence = LatentEvidenceField(cfg)
        self.hoi_factors = nn.Parameter(torch.empty(cfg.num_hoi_classes, cfg.num_bases))
        self.role_factors = nn.Parameter(torch.empty(cfg.num_roles, cfg.num_bases))
        self.hoi_probe = nn.Linear(cfg.d_model, cfg.num_bases)
        self.scene_proj = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.LayerNorm(cfg.d_model))
        self.activation_proj = nn.Linear(cfg.d_model, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(cfg.residual_scale_init)))

        nn.init.normal_(self.hoi_factors, std=0.02)
        nn.init.normal_(self.role_factors, std=0.02)
        prior = cfg.semantic_prior
        if prior is None:
            prior = torch.ones((cfg.num_hoi_classes, cfg.num_roles), dtype=torch.float32)
        prior = torch.as_tensor(prior, dtype=torch.float32)
        expected = (cfg.num_hoi_classes, cfg.num_roles)
        if tuple(prior.shape) != expected:
            raise ValueError(f"semantic_prior must have shape {expected}, got {tuple(prior.shape)}.")
        valid_mask = cfg.semantic_role_valid_mask
        if valid_mask is None:
            valid_mask = torch.ones(expected, dtype=torch.bool)
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool)
        if tuple(valid_mask.shape) != expected:
            raise ValueError(
                "semantic_role_valid_mask must have shape "
                f"{expected}, got {tuple(valid_mask.shape)}."
            )
        if not valid_mask.any(dim=-1).all():
            raise ValueError("Every semantic class must admit at least one role state.")
        if not torch.isfinite(prior).all() or (prior[valid_mask] <= 0).any():
            raise ValueError("semantic_prior must be finite and positive on valid role states.")
        prior = prior.masked_fill(~valid_mask, 0.0)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        self.register_buffer("semantic_role_prior", prior)
        self.register_buffer("semantic_role_valid_mask", valid_mask)

    def _factors(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        hoi = F.normalize(self.hoi_factors, dim=-1).to(device=device, dtype=dtype)
        role = F.normalize(self.role_factors, dim=-1).to(device=device, dtype=dtype)
        return hoi, role

    def _log_prior(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        prior = self.semantic_role_prior.to(device=device, dtype=dtype)
        valid = self.semantic_role_valid_mask.to(device=device)
        return torch.where(valid, prior.clamp_min(1e-8).log(), -torch.inf)

    def _residual(
        self,
        activations: torch.Tensor,
        hoi_factors: torch.Tensor,
        role_factors: torch.Tensor,
    ) -> torch.Tensor:
        raw = torch.einsum("bqk,ck,rk->bqcr", activations, hoi_factors, role_factors)
        raw = raw / max(float(self.cfg.num_bases) ** 0.5, 1.0)
        prior = self.semantic_role_prior.to(device=raw.device, dtype=raw.dtype)
        raw = raw - (raw * prior[None, None]).sum(dim=-1, keepdim=True)
        return self.residual_scale.to(dtype=raw.dtype) * raw

    def forward(
        self,
        pair_feats: torch.Tensor,
        dense_tokens: torch.Tensor,
        scene_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        evidence = self.evidence(pair_feats, dense_tokens)
        scene = self.scene_proj(scene_context)[:, :, None, :]
        full_contexts = evidence["basis_contexts"] + scene
        complement_contexts = evidence["complement_basis_contexts"] + scene
        activations = torch.tanh(self.activation_proj(full_contexts).squeeze(-1))
        complement_activations = torch.tanh(self.activation_proj(complement_contexts).squeeze(-1))
        hoi_factors, role_factors = self._factors(pair_feats.dtype, pair_feats.device)
        residual = self._residual(activations, hoi_factors, role_factors)
        complement_residual = self._residual(complement_activations, hoi_factors, role_factors)

        probe = torch.tanh(self.hoi_probe(pair_feats))
        provisional_logits = torch.einsum("bqk,ck->bqc", probe, hoi_factors)
        log_prior = self._log_prior(pair_feats.dtype, pair_feats.device)
        role_prob = F.softmax(log_prior[None, None] + residual, dim=-1)
        hoi_prob = provisional_logits.sigmoid()
        hoi_weight = hoi_prob / hoi_prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expected_role = torch.einsum("bqcr,rk->bqck", role_prob, role_factors)
        joint_factor = torch.einsum("bqc,ck,bqck->bqk", hoi_weight, hoi_factors, expected_role)
        signed_contribution = activations * joint_factor
        basis_weight = F.softmax(
            signed_contribution / max(float(self.cfg.joint_temperature), 1e-4),
            dim=-1,
        )
        evidence_gate = torch.einsum("bqk,bqkhw->bqhw", basis_weight, evidence["basis_masks"])
        role_field_context = torch.einsum("bqk,bqkd->bqd", basis_weight, full_contexts)
        complement_field_context = torch.einsum("bqk,bqkd->bqd", basis_weight, complement_contexts)
        return {
            **evidence,
            "basis_activations": activations,
            "complement_basis_activations": complement_activations,
            "hoi_factors": hoi_factors,
            "role_factors": role_factors,
            "joint_residual_logits": residual,
            "complement_joint_residual_logits": complement_residual,
            "provisional_hoi_logits": provisional_logits,
            "provisional_role_probs": role_prob,
            "supporting_basis_weights": basis_weight,
            "evidence_gate": evidence_gate,
            "role_field_context": role_field_context,
            "complement_field_context": complement_field_context,
        }

    @staticmethod
    def _state_outputs(
        base_logits: torch.Tensor,
        hoi_delta: torch.Tensor,
        role_residual: torch.Tensor,
        log_prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        conditional_role_logits = log_prior[None, None] + role_residual
        positive_logits = base_logits[..., None] + hoi_delta[..., None] + conditional_role_logits
        joint_state_logits = torch.cat([torch.zeros_like(positive_logits[..., :1]), positive_logits], dim=-1)
        hoi_logits = torch.logsumexp(positive_logits, dim=-1)
        conditional_role_probs = F.softmax(conditional_role_logits, dim=-1)
        edge_role_prob = hoi_logits.sigmoid()[..., None] * conditional_role_probs
        pair_role_prob = edge_role_prob.amax(dim=2).clamp(1e-6, 1.0 - 1e-6)
        return {
            "joint_state_logits": joint_state_logits,
            "hoi_logits": hoi_logits,
            "conditional_role_logits": conditional_role_logits,
            "conditional_role_probs": conditional_role_probs,
            "pair_role_logits": torch.logit(pair_role_prob),
        }

    def finalize(
        self,
        base_logits: torch.Tensor,
        hoi_delta: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        erased_hoi_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if base_logits.shape[-1] != self.cfg.num_hoi_classes:
            raise ValueError("base HOI class count does not match the CoRISP joint field.")
        log_prior = self._log_prior(base_logits.dtype, base_logits.device)
        residual = outputs["joint_residual_logits"].to(dtype=base_logits.dtype)
        full = self._state_outputs(base_logits, hoi_delta, residual, log_prior)
        zeros_delta = torch.zeros_like(hoi_delta)
        zeros_role = torch.zeros_like(residual)
        clean = self._state_outputs(base_logits, zeros_delta, zeros_role, log_prior)
        if erased_hoi_delta is None:
            erased_hoi_delta = zeros_delta
        erased = self._state_outputs(
            base_logits,
            erased_hoi_delta,
            outputs["complement_joint_residual_logits"].to(dtype=base_logits.dtype),
            log_prior,
        )
        valid = self.semantic_role_valid_mask.to(device=log_prior.device)
        valid_count = valid.sum(dim=-1, keepdim=True).to(dtype=log_prior.dtype)
        uniform_log_prior = torch.where(
            valid,
            -valid_count.log(),
            -torch.inf,
        )
        neutral = self._state_outputs(base_logits, hoi_delta, residual, uniform_log_prior)
        return {
            **outputs,
            **full,
            "clean_joint_state_logits": clean["joint_state_logits"],
            "clean_hoi_logits": clean["hoi_logits"],
            "erased_joint_state_logits": erased["joint_state_logits"],
            "erased_hoi_logits": erased["hoi_logits"],
            "neutral_joint_state_logits": neutral["joint_state_logits"],
            "neutral_conditional_role_logits": neutral["conditional_role_logits"],
            "semantic_role_prior": self.semantic_role_prior,
        }

class CoRISPJointDecoder(nn.Module):
    """Decoder-integrated role field with independent full and erased streams."""

    def __init__(self, cfg: CoRISPJointConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.entity_memory = SceneEntityMemory(cfg)
        self.event_reasoner = LatentEventSetReasoner(cfg) if cfg.use_event_reasoning else None
        self.joint_field = PriorResidualJointField(cfg)
        self.evidence_proj = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.LayerNorm(cfg.d_model))
        heads = int(cfg.evidence_attention_heads)
        if heads <= 0 or cfg.d_model % heads != 0:
            heads = 1
        self.evidence_attention_heads = heads
        self.evidence_attention = nn.MultiheadAttention(cfg.d_model, heads, batch_first=True)
        self.geometry_proj = nn.Sequential(
            nn.Linear(_RELATION_GEOMETRY_DIM, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.d_model),
        )
        self.relation_proj = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.d_model),
        )
        self.query_refiner = nn.Sequential(
            nn.Linear(cfg.d_model * 4, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.d_model),
        )
        self.refine_gate = nn.Sequential(nn.Linear(cfg.d_model * 4, cfg.d_model), nn.Sigmoid())
        self.interlayer_feedback_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        nn.init.zeros_(self.query_refiner[-1].weight)
        nn.init.zeros_(self.query_refiner[-1].bias)

    def _assign(self, boxes: torch.Tensor, entity_boxes: torch.Tensor) -> torch.Tensor:
        iou = box_iou_xyxy(boxes, entity_boxes)
        return F.softmax(iou / max(float(self.cfg.assignment_temperature), 1e-4), dim=-1)

    def _evidence_attention_context(
        self,
        pair_feats: torch.Tensor,
        dense_tokens: torch.Tensor,
        evidence_gate: torch.Tensor,
        *,
        erase: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = (1.0 - evidence_gate) if erase else evidence_gate
        gate = gate.clamp(1e-4, 1.0)
        dense = dense_tokens.flatten(2).transpose(1, 2).to(dtype=pair_feats.dtype)
        attn_bias = torch.log(gate.flatten(2).to(dtype=pair_feats.dtype))
        attn_bias = attn_bias[:, :, None, :].expand(-1, -1, self.evidence_attention_heads, -1)
        attn_bias = attn_bias.permute(0, 2, 1, 3).reshape(
            pair_feats.shape[0] * self.evidence_attention_heads,
            pair_feats.shape[1],
            dense.shape[1],
        )
        context, _ = self.evidence_attention(
            pair_feats,
            dense,
            dense,
            attn_mask=attn_bias,
            need_weights=False,
        )
        return context, gate

    @staticmethod
    def _relation_context(
        subject_assignment: torch.Tensor,
        object_assignment: torch.Tensor,
        entity_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        subject_context = torch.einsum("bqe,bed->bqd", subject_assignment, entity_feats)
        object_context = torch.einsum("bqe,bed->bqd", object_assignment, entity_feats)
        return subject_context, object_context

    def _refine_pair_query(
        self,
        pair_feats: torch.Tensor,
        field_context: torch.Tensor,
        evidence_context: torch.Tensor,
        relation_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        refine_input = torch.cat([pair_feats, field_context, evidence_context, relation_context], dim=-1)
        refine_delta = self.query_refiner(refine_input)
        refine_gate = self.refine_gate(refine_input)
        refined = pair_feats + float(self.cfg.decoder_refine_alpha) * refine_gate * refine_delta
        return refined, refine_gate, refine_delta

    def forward(
        self,
        pair_feats: torch.Tensor,
        dense_tokens: torch.Tensor,
        subject_boxes: torch.Tensor,
        object_boxes: torch.Tensor,
        subject_feats: torch.Tensor | None = None,
        object_feats: torch.Tensor | None = None,
        entity_boxes: torch.Tensor | None = None,
        entity_feats: torch.Tensor | None = None,
        erased_pair_feats_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if erased_pair_feats_input is None:
            erased_pair_feats_input = pair_feats
        if erased_pair_feats_input.shape != pair_feats.shape:
            raise ValueError("erased_pair_feats_input must match pair_feats shape.")

        memory = self.entity_memory(
            pair_feats,
            subject_boxes,
            object_boxes,
            subject_feats,
            object_feats,
            entity_boxes,
            entity_feats,
        )
        subject_assignment = self._assign(subject_boxes, memory["entity_boxes"])
        object_assignment = self._assign(object_boxes, memory["entity_boxes"])
        subject_context, object_context = self._relation_context(
            subject_assignment,
            object_assignment,
            memory["entity_feats"],
        )
        relation_context = self.relation_proj(
            torch.cat([subject_context, object_context, subject_context * object_context], dim=-1)
        )
        if self.cfg.use_relation_geometry:
            geometry_context = self.geometry_proj(
                relation_geometry_features(subject_boxes, object_boxes).to(
                    device=pair_feats.device,
                    dtype=pair_feats.dtype,
                )
            )
            relation_context = relation_context + geometry_context
        else:
            geometry_context = torch.zeros_like(relation_context)

        event_outputs: dict[str, torch.Tensor] = {}
        event_pair_feats = pair_feats
        event_erased_pair_feats = erased_pair_feats_input
        if self.event_reasoner is not None:
            hoi_factors, _ = self.joint_field._factors(pair_feats.dtype, pair_feats.device)
            event_outputs = self.event_reasoner(pair_feats, relation_context, hoi_factors)
            event_pair_feats = event_outputs["event_refined_pair_feats"]
            shared_event_delta = event_pair_feats - pair_feats
            event_erased_pair_feats = erased_pair_feats_input + shared_event_delta

        joint = self.joint_field(event_pair_feats, dense_tokens, relation_context)
        pooled_context = self.evidence_proj(joint["role_field_context"])
        if self.cfg.use_evidence_attention:
            attention_context, evidence_gate = self._evidence_attention_context(
                event_pair_feats,
                dense_tokens,
                joint["evidence_gate"],
                erase=False,
            )
            evidence_context = self.evidence_proj(attention_context + pooled_context)
        else:
            attention_context = pooled_context
            evidence_context = pooled_context
            evidence_gate = joint["evidence_gate"]

        if self.cfg.use_evidence_attention:
            erased_attention_context, erased_evidence_gate = self._evidence_attention_context(
                event_erased_pair_feats,
                dense_tokens,
                evidence_gate,
                erase=True,
            )
            erased_evidence_context = self.evidence_proj(
                erased_attention_context + joint["complement_field_context"]
            )
        else:
            erased_attention_context = joint["complement_field_context"]
            erased_evidence_context = self.evidence_proj(erased_attention_context)
            erased_evidence_gate = 1.0 - evidence_gate
        enhanced, refine_gate, refine_delta = self._refine_pair_query(
            event_pair_feats,
            joint["role_field_context"],
            evidence_context,
            relation_context,
        )
        erased, erased_refine_gate, erased_refine_delta = self._refine_pair_query(
            event_erased_pair_feats,
            joint["complement_field_context"],
            erased_evidence_context,
            relation_context,
        )
        feedback_scale = float(self.cfg.interlayer_feedback_alpha)
        enhanced_feedback = feedback_scale * self.interlayer_feedback_proj(refine_gate * refine_delta)
        erased_feedback = feedback_scale * self.interlayer_feedback_proj(
            erased_refine_gate * erased_refine_delta
        )
        return {
            **joint,
            **memory,
            **event_outputs,
            "subject_assignment": subject_assignment,
            "object_assignment": object_assignment,
            "evidence_context": evidence_context,
            "pooled_evidence_context": pooled_context,
            "evidence_attention_context": attention_context,
            "erased_evidence_context": erased_evidence_context,
            "evidence_gate": evidence_gate,
            "erased_evidence_gate": erased_evidence_gate,
            "subject_context": subject_context,
            "object_context": object_context,
            "relation_context": relation_context,
            "geometry_context": geometry_context,
            "refine_gate": refine_gate,
            "refine_delta": refine_delta,
            "erased_refine_gate": erased_refine_gate,
            "erased_refine_delta": erased_refine_delta,
            "enhanced_pair_feats": enhanced,
            "erased_pair_feats": erased,
            "enhanced_interlayer_feedback": enhanced_feedback,
            "erased_interlayer_feedback": erased_feedback,
        }

    def finalize_joint(
        self,
        base_logits: torch.Tensor,
        enhanced_logits: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        erased_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base_anchor = base_logits.detach()
        hoi_delta = enhanced_logits - base_anchor
        erased_delta = None if erased_logits is None else erased_logits - base_anchor
        return self.joint_field.finalize(base_anchor, hoi_delta, outputs, erased_delta)
