from __future__ import annotations

from dataclasses import dataclass

from typing import Mapping, Sequence

import torch

import torch.nn as nn

from corisp.joint import relation_geometry_features

def _check_boxes(name: str, boxes: torch.Tensor) -> None:
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must be [B,N,4].")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} contains non-finite coordinates.")
    if boxes.numel() == 0:
        return
    if (boxes[..., 2:] < boxes[..., :2]).any():
        raise ValueError(f"{name} must use ordered xyxy coordinates.")
    if (boxes < -1e-6).any() or (boxes > 1.0 + 1e-6).any():
        raise ValueError(f"{name} must be normalized to [0,1].")

@dataclass(frozen=True)
class PairPacket:
    """A batched, padded set of localized human-entity pair hypotheses.

    A ``True`` value means valid for every mask in this class.  Call
    :meth:`compact_image` before passing a padded packet to a reasoner that
    operates on one image at a time.
    """

    pair_feats: torch.Tensor
    dense_tokens: torch.Tensor
    subject_boxes: torch.Tensor
    object_boxes: torch.Tensor
    pair_valid_mask: torch.Tensor | None = None
    subject_feats: torch.Tensor | None = None
    object_feats: torch.Tensor | None = None
    entity_boxes: torch.Tensor | None = None
    entity_feats: torch.Tensor | None = None
    entity_valid_mask: torch.Tensor | None = None
    dense_valid_mask: torch.Tensor | None = None
    subject_indices: torch.Tensor | None = None
    object_indices: torch.Tensor | None = None
    proposal_scores: torch.Tensor | None = None
    object_labels: torch.Tensor | None = None
    entity_labels: torch.Tensor | None = None
    source: str = "detector_agnostic"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.pair_feats.shape[0])

    @property
    def num_pairs(self) -> int:
        return int(self.pair_feats.shape[1])

    @property
    def d_model(self) -> int:
        return int(self.pair_feats.shape[2])

    @property
    def valid_pairs(self) -> torch.Tensor:
        if self.pair_valid_mask is None:
            return torch.ones(
                self.pair_feats.shape[:2],
                dtype=torch.bool,
                device=self.pair_feats.device,
            )
        return self.pair_valid_mask

    def validate(self) -> None:
        if self.pair_feats.ndim != 3:
            raise ValueError("pair_feats must be [B,Q,D].")
        if self.dense_tokens.ndim != 4:
            raise ValueError("dense_tokens must be [B,D,H,W].")
        b, q, d = self.pair_feats.shape
        if self.dense_tokens.shape[:2] != (b, d):
            raise ValueError("dense_tokens must share B and D with pair_feats.")
        if self.subject_boxes.shape != (b, q, 4) or self.object_boxes.shape != (b, q, 4):
            raise ValueError("subject_boxes and object_boxes must both be [B,Q,4].")
        _check_boxes("subject_boxes", self.subject_boxes)
        _check_boxes("object_boxes", self.object_boxes)

        if self.pair_valid_mask is not None:
            if self.pair_valid_mask.shape != (b, q) or self.pair_valid_mask.dtype != torch.bool:
                raise ValueError("pair_valid_mask must be bool [B,Q].")
        for name, value in (("subject_feats", self.subject_feats), ("object_feats", self.object_feats)):
            if value is not None and value.shape != (b, q, d):
                raise ValueError(f"{name} must match pair_feats [B,Q,D].")

        if (self.entity_boxes is None) != (self.entity_feats is None):
            raise ValueError("entity_boxes and entity_feats must be supplied together.")
        if self.entity_boxes is not None and self.entity_feats is not None:
            if self.entity_boxes.ndim != 3 or self.entity_boxes.shape[0] != b:
                raise ValueError("entity_boxes must be [B,E,4].")
            e = self.entity_boxes.shape[1]
            if self.entity_feats.shape != (b, e, d):
                raise ValueError("entity_feats must be [B,E,D].")
            _check_boxes("entity_boxes", self.entity_boxes)
            if self.entity_valid_mask is not None:
                if self.entity_valid_mask.shape != (b, e) or self.entity_valid_mask.dtype != torch.bool:
                    raise ValueError("entity_valid_mask must be bool [B,E].")
        elif self.entity_valid_mask is not None:
            raise ValueError("entity_valid_mask requires entity boxes and features.")

        h, w = self.dense_tokens.shape[-2:]
        if self.dense_valid_mask is not None:
            if self.dense_valid_mask.shape != (b, h, w) or self.dense_valid_mask.dtype != torch.bool:
                raise ValueError("dense_valid_mask must be bool [B,H,W].")
            if not self.dense_valid_mask.flatten(1).any(dim=1).all():
                raise ValueError("Every image must contain at least one valid dense token.")

        for name, value in (("subject_indices", self.subject_indices), ("object_indices", self.object_indices)):
            if value is not None:
                if value.shape != (b, q) or value.dtype not in (torch.int32, torch.int64):
                    raise ValueError(f"{name} must be integer [B,Q].")
        if self.proposal_scores is not None:
            if self.proposal_scores.shape != (b, q, 2):
                raise ValueError("proposal_scores must be [B,Q,2].")
            if not torch.isfinite(self.proposal_scores).all():
                raise ValueError("proposal_scores contains non-finite values.")
        if self.object_labels is not None:
            if self.object_labels.shape != (b, q):
                raise ValueError("object_labels must be [B,Q].")
        if self.entity_labels is not None:
            if self.entity_boxes is None:
                raise ValueError("entity_labels requires entity boxes and features.")
            if self.entity_labels.shape != self.entity_boxes.shape[:2]:
                raise ValueError("entity_labels must be [B,E].")
            if self.entity_labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("entity_labels must contain integer class indices.")

    def compact_image(self, index: int) -> "PairPacket":
        """Return one image with pair/entity padding removed.

        Dense padding is cropped to its valid bounding rectangle and any holes
        inside that rectangle are zeroed.  Standard detector masks are
        rectangular, so this exactly recovers the unpadded feature map.
        """

        if not 0 <= index < self.batch_size:
            raise IndexError(index)
        pair_keep = self.valid_pairs[index]

        dense = self.dense_tokens[index : index + 1]
        dense_mask = None
        if self.dense_valid_mask is not None:
            valid = self.dense_valid_mask[index]
            rows = torch.nonzero(valid.any(dim=1), as_tuple=False).flatten()
            cols = torch.nonzero(valid.any(dim=0), as_tuple=False).flatten()
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            valid = valid[y0:y1, x0:x1]
            dense = dense[..., y0:y1, x0:x1] * valid[None, None].to(dtype=dense.dtype)
            dense_mask = valid[None]

        entity_boxes = self.entity_boxes[index : index + 1] if self.entity_boxes is not None else None
        entity_feats = self.entity_feats[index : index + 1] if self.entity_feats is not None else None
        entity_labels = self.entity_labels[index : index + 1] if self.entity_labels is not None else None
        subject_indices = self._select_pair_tensor(self.subject_indices, index, pair_keep)
        object_indices = self._select_pair_tensor(self.object_indices, index, pair_keep)
        entity_mask = None
        if entity_boxes is not None and entity_feats is not None:
            if self.entity_valid_mask is None:
                keep = torch.ones(entity_boxes.shape[1], dtype=torch.bool, device=entity_boxes.device)
            else:
                keep = self.entity_valid_mask[index]
            if subject_indices is not None and object_indices is not None:
                remap = torch.full((keep.numel(),), -1, dtype=torch.long, device=keep.device)
                remap[keep] = torch.arange(int(keep.sum()), device=keep.device)
                subject_indices = remap[subject_indices]
                object_indices = remap[object_indices]
                if (subject_indices < 0).any() or (object_indices < 0).any():
                    raise ValueError("A valid pair references a padded entity.")
            entity_boxes = entity_boxes[:, keep]
            entity_feats = entity_feats[:, keep]
            if entity_labels is not None:
                entity_labels = entity_labels[:, keep]
            entity_mask = torch.ones((1, int(keep.sum())), dtype=torch.bool, device=keep.device)

        return PairPacket(
            pair_feats=self.pair_feats[index : index + 1, pair_keep],
            dense_tokens=dense,
            subject_boxes=self.subject_boxes[index : index + 1, pair_keep],
            object_boxes=self.object_boxes[index : index + 1, pair_keep],
            pair_valid_mask=torch.ones((1, int(pair_keep.sum())), dtype=torch.bool, device=pair_keep.device),
            subject_feats=self._select_pair_tensor(self.subject_feats, index, pair_keep),
            object_feats=self._select_pair_tensor(self.object_feats, index, pair_keep),
            entity_boxes=entity_boxes,
            entity_feats=entity_feats,
            entity_valid_mask=entity_mask,
            dense_valid_mask=dense_mask,
            subject_indices=subject_indices,
            object_indices=object_indices,
            proposal_scores=self._select_pair_tensor(self.proposal_scores, index, pair_keep),
            object_labels=self._select_pair_tensor(self.object_labels, index, pair_keep),
            entity_labels=entity_labels,
            source=self.source,
        )

    @staticmethod
    def _select_pair_tensor(
        value: torch.Tensor | None,
        index: int,
        keep: torch.Tensor,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        return value[index : index + 1, keep]

@dataclass(frozen=True)
class PreparedProposalAdapterConfig:
    detector_dim: int = 256
    pair_dim: int = 384
    dense_dim: int = 256
    d_model: int = 256
    human_label: int = 0
    max_pairs: int = 225
    include_human_human: bool = True

    def __post_init__(self) -> None:
        if min(self.detector_dim, self.pair_dim, self.dense_dim, self.d_model) <= 0:
            raise ValueError("All adapter dimensions must be positive.")
        if self.max_pairs <= 0:
            raise ValueError("max_pairs must be positive.")

class PreparedProposalPairAdapter(nn.Module):
    """Convert prepared H-DETR/PViC proposals into compact pair packets.

    ``proposals`` must already have passed the detector track's frozen NMS,
    score threshold, and min/max instance policy.  If pair indices or PViC pair
    queries are supplied, their exact order is preserved; otherwise all valid
    human-entity pairs are enumerated deterministically.
    """

    def __init__(self, cfg: PreparedProposalAdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        structural_dim = cfg.detector_dim * 4 + 19
        self.structural_pair_proj = nn.Sequential(
            nn.Linear(structural_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        if cfg.pair_dim == cfg.d_model:
            self.query_pair_proj = nn.Identity()
        else:
            self.query_pair_proj = nn.Sequential(
                nn.Linear(cfg.pair_dim, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
            )
        self.query_gate = nn.Sequential(nn.Linear(cfg.d_model * 2, cfg.d_model), nn.Sigmoid())
        self.query_fusion_scale = nn.Parameter(torch.tensor(0.0))
        self.subject_proj = nn.Sequential(nn.Linear(cfg.detector_dim, cfg.d_model), nn.LayerNorm(cfg.d_model))
        self.object_proj = nn.Sequential(nn.Linear(cfg.detector_dim, cfg.d_model), nn.LayerNorm(cfg.d_model))
        self.entity_proj = nn.Sequential(nn.Linear(cfg.detector_dim, cfg.d_model), nn.LayerNorm(cfg.d_model))
        self.dense_proj = nn.Conv2d(cfg.dense_dim, cfg.d_model, kernel_size=1)

    def _enumerate_pairs(self, labels: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        human = torch.nonzero(labels == self.cfg.human_label, as_tuple=False).flatten()
        if human.numel() == 0 or labels.numel() < 2:
            return torch.empty((0, 2), dtype=torch.long, device=labels.device)
        subjects = human[:, None].expand(-1, labels.numel()).reshape(-1)
        objects = torch.arange(labels.numel(), device=labels.device)[None].expand(human.numel(), -1).reshape(-1)
        keep = subjects != objects
        if not self.cfg.include_human_human:
            keep &= labels[objects] != self.cfg.human_label
        pairs = torch.stack([subjects[keep], objects[keep]], dim=-1)
        if pairs.shape[0] > self.cfg.max_pairs:
            rank = scores[pairs[:, 0]] * scores[pairs[:, 1]]
            order = torch.argsort(rank, descending=True, stable=True)[: self.cfg.max_pairs]
            pairs = pairs[order]
        return pairs

    @staticmethod
    def _normalize_boxes(boxes: torch.Tensor, image_size: torch.Tensor | Sequence[int]) -> torch.Tensor:
        size = torch.as_tensor(image_size, dtype=boxes.dtype, device=boxes.device)
        if size.numel() != 2 or (size <= 0).any():
            raise ValueError("image_size must contain positive (height, width).")
        scale = torch.stack([size[1], size[0], size[1], size[0]])
        return (boxes / scale).clamp(0.0, 1.0)

    def forward(
        self,
        proposals: Sequence[Mapping[str, torch.Tensor]],
        dense_tokens: torch.Tensor | Sequence[torch.Tensor],
        image_sizes: torch.Tensor | Sequence[Sequence[int] | torch.Tensor],
        *,
        paired_indices: Sequence[torch.Tensor] | None = None,
        pair_query_feats: Sequence[torch.Tensor] | None = None,
        dense_valid_masks: torch.Tensor | Sequence[torch.Tensor] | None = None,
        source: str = "h_detr_swin_l",
    ) -> list[PairPacket]:
        if len(proposals) == 0:
            return []
        if len(image_sizes) != len(proposals):
            raise ValueError("image_sizes must have one entry per proposal dictionary.")
        if paired_indices is not None and len(paired_indices) != len(proposals):
            raise ValueError("paired_indices must have one tensor per image.")
        if pair_query_feats is not None and len(pair_query_feats) != len(proposals):
            raise ValueError("pair_query_feats must have one tensor per image.")

        dense_list = self._as_dense_list(dense_tokens, len(proposals))
        mask_list = self._as_mask_list(dense_valid_masks, len(proposals))
        packets: list[PairPacket] = []
        for i, proposal in enumerate(proposals):
            required = {"boxes", "scores", "labels", "hidden_states"}
            missing = required.difference(proposal)
            if missing:
                raise KeyError(f"Proposal {i} is missing keys: {sorted(missing)}")
            boxes = proposal["boxes"]
            scores = proposal["scores"].reshape(-1)
            labels = proposal["labels"].reshape(-1)
            feats = proposal["hidden_states"]
            n = boxes.shape[0]
            if boxes.shape != (n, 4) or scores.shape != (n,) or labels.shape != (n,):
                raise ValueError("Each proposal must contain boxes [N,4], scores [N], labels [N].")
            if feats.shape != (n, self.cfg.detector_dim):
                raise ValueError(
                    f"hidden_states must be [N,{self.cfg.detector_dim}], got {tuple(feats.shape)}."
                )
            pairs = self._enumerate_pairs(labels, scores) if paired_indices is None else paired_indices[i]
            if pairs.ndim != 2 or pairs.shape[-1] != 2:
                raise ValueError("Each paired_indices tensor must be [Q,2].")
            if pairs.shape[0] > self.cfg.max_pairs:
                pairs = pairs[: self.cfg.max_pairs]
            if pairs.numel() and ((pairs < 0).any() or (pairs >= n).any()):
                raise ValueError("paired_indices references an unavailable proposal.")
            q = pairs.shape[0]
            sb = self._normalize_boxes(boxes[pairs[:, 0]], image_sizes[i])
            ob = self._normalize_boxes(boxes[pairs[:, 1]], image_sizes[i])
            sf_raw = feats[pairs[:, 0]]
            of_raw = feats[pairs[:, 1]]
            pair_scores = torch.stack([scores[pairs[:, 0]], scores[pairs[:, 1]]], dim=-1)
            geometry = relation_geometry_features(sb, ob)
            structural = self.structural_pair_proj(
                torch.cat([sf_raw, of_raw, sf_raw * of_raw, sf_raw - of_raw, geometry, pair_scores], dim=-1)
            )
            if pair_query_feats is not None:
                query = pair_query_feats[i]
                if query.shape[0] != q or query.shape[-1] != self.cfg.pair_dim:
                    raise ValueError(f"pair_query_feats[{i}] must be [Q,{self.cfg.pair_dim}].")
                query = self.query_pair_proj(query)
                gate = self.query_gate(torch.cat([query, structural], dim=-1))
                pair_feats = query + self.query_fusion_scale.to(dtype=query.dtype) * gate * structural
            else:
                pair_feats = structural

            dense = dense_list[i]
            if dense.ndim != 3 or dense.shape[0] != self.cfg.dense_dim:
                raise ValueError(f"Each dense feature map must be [{self.cfg.dense_dim},H,W].")
            dense = self.dense_proj(dense[None])
            dense_mask = None if mask_list is None else mask_list[i][None].to(device=dense.device)
            normalized_entities = self._normalize_boxes(boxes, image_sizes[i])
            packet = PairPacket(
                pair_feats=pair_feats[None],
                dense_tokens=dense,
                subject_boxes=sb[None],
                object_boxes=ob[None],
                pair_valid_mask=torch.ones((1, q), dtype=torch.bool, device=boxes.device),
                subject_feats=self.subject_proj(sf_raw)[None],
                object_feats=self.object_proj(of_raw)[None],
                entity_boxes=normalized_entities[None],
                entity_feats=self.entity_proj(feats)[None],
                entity_valid_mask=torch.ones((1, n), dtype=torch.bool, device=boxes.device),
                dense_valid_mask=dense_mask,
                subject_indices=pairs[:, 0][None],
                object_indices=pairs[:, 1][None],
                proposal_scores=pair_scores[None],
                object_labels=labels[pairs[:, 1]][None],
                entity_labels=labels[None],
                source=source,
            )
            packets.append(packet)
        return packets

    @staticmethod
    def _as_dense_list(value: torch.Tensor | Sequence[torch.Tensor], expected: int) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            if value.ndim != 4 or value.shape[0] != expected:
                raise ValueError("dense_tokens tensor must be [B,D,H,W].")
            return list(value.unbind(0))
        if len(value) != expected:
            raise ValueError("dense_tokens must have one feature map per image.")
        return list(value)

    @staticmethod
    def _as_mask_list(
        value: torch.Tensor | Sequence[torch.Tensor] | None,
        expected: int,
    ) -> list[torch.Tensor] | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim != 3 or value.shape[0] != expected:
                raise ValueError("dense_valid_masks tensor must be [B,H,W].")
            return list(value.unbind(0))
        if len(value) != expected:
            raise ValueError("dense_valid_masks must have one mask per image.")
        return list(value)
