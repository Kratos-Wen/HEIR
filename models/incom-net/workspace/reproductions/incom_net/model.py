"""Independent implementation of InCoM-Net Eqs. 1-15 and supplement S1.

Unspecified architectural conventions are listed in reconstruction.json.
Inputs refer to one image, with instance identities aligned across all layers.
No CoRISP component is used.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class InCoMConfig:
    detector_dim: int = 256
    vlm_dim: int = 1024
    cnn_dim: int = 2048
    hidden_dim: int = 384
    levels: int = 3
    decoder_layers: int = 2
    heads: int = 8
    ffn_dim: int = 1536
    dropout: float = 0.1
    num_actions: int = 24
    instance_chunk: int = 4

    def __post_init__(self):
        if self.hidden_dim % self.heads or min(self.levels, self.decoder_layers, self.instance_chunk) < 1:
            raise ValueError("Invalid attention dimensions, depth or chunk size")
        if self.num_actions < 1:
            raise ValueError("num_actions must be positive")


def box_patch_masks(boxes_xyxy: Tensor, grid: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Rasterize normalized boxes by positive patch overlap; do not erase overlaps.

    Surrounding means the union of OTHER instance masks, not the complement
    of this box and not (union minus own pixels). A overlapping neighbor is
    still a neighboring instance (main text Sec. 3.2).
    """
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("Expected [instances,4] normalized xyxy boxes")
    if not torch.isfinite(boxes_xyxy).all() or ((boxes_xyxy < 0) | (boxes_xyxy > 1)).any():
        raise ValueError("Boxes must be finite normalized coordinates")
    if (boxes_xyxy[:, 2:] <= boxes_xyxy[:, :2]).any():
        raise ValueError("Boxes must have positive area")
    height, width = grid
    if height < 1 or width < 1:
        raise ValueError("Invalid patch grid")
    yy, xx = torch.meshgrid(torch.arange(height, device=boxes_xyxy.device),
                            torch.arange(width, device=boxes_xyxy.device), indexing="ij")
    x0, y0 = xx.flatten() / width, yy.flatten() / height
    x1, y1 = (xx.flatten() + 1) / width, (yy.flatten() + 1) / height
    b = boxes_xyxy[:, :, None]
    inside = (x1 > b[:, 0]) & (y1 > b[:, 1]) & (x0 < b[:, 2]) & (y0 < b[:, 3])
    surrounding = (inside.sum(0, keepdim=True) - inside.to(torch.int64)) > 0
    return inside, surrounding


def human_entity_pairs(labels: Tensor, person_label: int = 0) -> Tensor:
    ids = torch.arange(labels.numel(), device=labels.device)
    subjects = ids[labels == person_label]
    h, o = torch.meshgrid(subjects, ids, indexing="ij")
    pairs = torch.stack((h.flatten(), o.flatten()), -1)
    return pairs[pairs[:, 0] != pairs[:, 1]]


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.layers(x)


class ContextAttention(nn.Module):
    """Figure 3: full query features and multiplicatively masked K/V features."""
    def __init__(self, cfg: InCoMConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.heads, cfg.dropout, batch_first=True)
        self.ffn = FFN(cfg.hidden_dim, cfg.ffn_dim, cfg.hidden_dim, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, query: Tensor, mask: Tensor) -> Tensor:
        memory = query * mask[..., None].to(query.dtype)
        attended = self.attn(query, memory, memory, need_weights=False)[0]
        x = self.norm1(query + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        # A scene without neighbors must not create an artificial context from biases.
        return x * mask.any(-1)[:, None, None].to(x.dtype)


class ContextLevel(nn.Module):
    """ICR and ProCA for a single aligned DETR/CLIP depth."""
    def __init__(self, cfg: InCoMConfig):
        super().__init__()
        self.chunk = cfg.instance_chunk
        self.vlm_proj = nn.Linear(cfg.vlm_dim, cfg.hidden_dim)
        self.query_ffn = FFN(cfg.detector_dim, cfg.ffn_dim, cfg.hidden_dim, cfg.dropout)
        self.icr = nn.ModuleList([ContextAttention(cfg) for _ in range(3)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(cfg.hidden_dim, cfg.heads,
                                  cfg.dropout, batch_first=True) for _ in range(3)])
        self.fusion = FFN(3 * cfg.hidden_dim, cfg.ffn_dim, cfg.hidden_dim, cfg.dropout)

    def forward(self, q: Tensor, vlm: Tensor, previous: Tensor,
                region: Tensor, surrounding: Tensor) -> Tensor:
        count = len(q)
        if count == 0:
            return previous
        memory = self.vlm_proj(vlm).unsqueeze(0)
        global_context = self.icr[0](memory, torch.ones(memory.shape[:2], dtype=torch.bool, device=q.device))
        query = previous + self.query_ffn(q)
        chunks = []
        for start in range(0, count, self.chunk):
            end = min(count, start + self.chunk)
            local_query = query[start:end, None]
            expanded = memory.expand(end - start, -1, -1)
            intra = self.icr[1](expanded, region[start:end])
            inter = self.icr[2](expanded, surrounding[start:end])
            g = self.cross[0](local_query, global_context.expand(end - start, -1, -1),
                              global_context.expand(end - start, -1, -1), need_weights=False)[0]
            r = self.cross[1](local_query, intra, intra, need_weights=False)[0]
            c = self.cross[2](local_query, inter, inter, need_weights=False)[0]
            r = r * region[start:end].any(-1)[:, None, None].to(r.dtype)
            c = c * surrounding[start:end].any(-1)[:, None, None].to(c.dtype)
            chunks.append(self.fusion(torch.cat((g, r, c), -1)).squeeze(1))
        return torch.cat(chunks)


class DualSourceDecoderLayer(nn.Module):
    def __init__(self, cfg: InCoMConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.heads, cfg.dropout, batch_first=True)
        self.detector_attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.heads, cfg.dropout, batch_first=True)
        self.vlm_attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.heads, cfg.dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.hidden_dim) for _ in range(3)])
        self.ffn = FFN(cfg.hidden_dim, cfg.ffn_dim, cfg.hidden_dim, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, query, cnn, vlm, use_detector: bool, use_vlm: bool):
        z = self.norms[0](query + self.dropout(self.self_attn(query, query, query, need_weights=False)[0]))
        # Disabled attention is not executed: zero inputs alone would retain projection biases.
        d = self.detector_attn(z, cnn, cnn, need_weights=False)[0] if use_detector else torch.zeros_like(z)
        v = self.vlm_attn(z, vlm, vlm, need_weights=False)[0] if use_vlm else torch.zeros_like(z)
        z = self.norms[1](z + self.dropout(d + v))
        return self.norms[2](z + self.dropout(self.ffn(z)))


class InCoMHead(nn.Module):
    def __init__(self, cfg: InCoMConfig):
        super().__init__()
        self.cfg = cfg
        self.context_levels = nn.ModuleList([ContextLevel(cfg) for _ in range(cfg.levels)])
        self.detector_pair = nn.Sequential(nn.Linear(2 * cfg.detector_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim))
        self.context_pair = nn.Sequential(nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim))
        self.pair_encoder = nn.TransformerEncoderLayer(cfg.hidden_dim, cfg.heads, cfg.ffn_dim,
                                  cfg.dropout, activation="gelu", batch_first=True)
        self.cnn_proj = nn.Linear(cfg.cnn_dim, cfg.hidden_dim)
        self.vlm_proj = nn.Linear(cfg.vlm_dim, cfg.hidden_dim)
        self.decoder = nn.ModuleList([DualSourceDecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_actions)

    def mine_context(self, detector_layers, vlm_layers, boxes, grid):
        cfg = self.cfg
        if detector_layers.shape != (cfg.levels, len(boxes), cfg.detector_dim):
            raise ValueError("Expected aligned [L,K,detector_dim] instance features")
        if vlm_layers.shape != (cfg.levels, grid[0] * grid[1], cfg.vlm_dim):
            raise ValueError("Expected [L,H*W,vlm_dim] patch features, without CLS")
        region, surrounding = box_patch_masks(boxes, grid)
        state = detector_layers.new_zeros((len(boxes), cfg.hidden_dim))
        for layer, q, v in zip(self.context_levels, detector_layers, vlm_layers):
            state = layer(q, v, state, region, surrounding)
        return state

    def reason(self, q, context, cnn, vlm, pairs, mode):
        if mode not in ("full", "detector_only", "vlm_only"):
            raise ValueError(f"Unknown MFT mode: {mode}")
        if pairs.ndim != 2 or pairs.shape[-1] != 2 or pairs.dtype != torch.long:
            raise ValueError("Expected int64 pair indices [P,2]")
        if pairs.numel() and ((pairs < 0).any() or (pairs >= len(q)).any() or (pairs[:, 0] == pairs[:, 1]).any()):
            raise ValueError("Invalid instance pair")
        if not len(pairs):
            features = q.new_zeros((0, self.cfg.hidden_dim))
            return {"logits": self.classifier(features), "pair_features": features}
        use_detector, use_vlm = mode != "vlm_only", mode != "detector_only"
        h, o = pairs.unbind(-1)
        # Mask after affine projection so a disabled pair branch is exactly absent.
        d = self.detector_pair(torch.cat((q[h], q[o]), -1)) if use_detector else context.new_zeros((len(pairs), self.cfg.hidden_dim))
        v = self.context_pair(torch.cat((context[h], context[o]), -1)) if use_vlm else torch.zeros_like(d)
        z = self.pair_encoder((d + v).unsqueeze(0))
        cnn_memory = self.cnn_proj(cnn).unsqueeze(0) if use_detector else None
        vlm_memory = self.vlm_proj(vlm).unsqueeze(0) if use_vlm else None
        for layer in self.decoder:
            z = layer(z, cnn_memory, vlm_memory, use_detector, use_vlm)
        features = z.squeeze(0)
        return {"logits": self.classifier(features), "pair_features": features}

    def forward(self, detector_layers, vlm_layers, boxes, grid, cnn_tokens, pairs, *, mft=None):
        if cnn_tokens.ndim != 2 or cnn_tokens.shape[-1] != self.cfg.cnn_dim or not len(cnn_tokens):
            raise ValueError("CNN memory must contain unpadded [N,cnn_dim] tokens")
        context = self.mine_context(detector_layers, vlm_layers, boxes, grid)
        modes = ("full", "detector_only", "vlm_only") if (self.training if mft is None else mft) else ("full",)
        return {mode: self.reason(detector_layers[-1], context, cnn_tokens, vlm_layers[-1], pairs, mode)
                for mode in modes}


def focal_mft_loss(branches: dict, targets: Tensor, valid_actions: Tensor, *,
                   alpha: float, gamma: float, normalizer=None) -> tuple[Tensor, dict]:
    """Equal-weight SUM over MFT configurations (Eq. 15), not their mean.

    Detector confidence is only used at inference (supplement S1). Train
    supervision is multi-label and masks object-incompatible actions.
    """
    if set(branches) != {"full", "detector_only", "vlm_only"}:
        raise ValueError("Training must provide all three MFT configurations")
    if targets.shape != valid_actions.shape or valid_actions.dtype != torch.bool:
        raise ValueError("Expected a boolean valid-action mask matching targets")
    if not torch.isfinite(targets).all() or ((targets != 0) & (targets != 1)).any():
        raise ValueError("Targets must be binary multi-label annotations")
    if (targets.bool() & ~valid_actions).any():
        raise ValueError("Positive annotation was masked by the object-action table")
    if not (0 <= alpha <= 1 and gamma >= 0):
        raise ValueError("Invalid focal parameters")
    denominator = targets.sum().clamp(min=1) if normalizer is None else torch.as_tensor(normalizer, device=targets.device)
    if denominator.numel() != 1 or not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("Loss normalizer must be a finite positive scalar")
    losses = {}
    for name, branch in branches.items():
        logits = branch["logits"]
        if logits.shape != targets.shape:
            raise ValueError("Logits and labels have different class/pair order")
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = logits.sigmoid()
        pt = targets * prob + (1 - targets) * (1 - prob)
        weight = alpha * targets + (1 - alpha) * (1 - targets)
        losses[name] = (ce * (1 - pt).pow(gamma) * weight)[valid_actions].sum() / denominator
    return sum(losses.values()), losses


def inference_scores(logits, pair_scores, valid_actions, exponent=2.8):
    if pair_scores.shape != (len(logits), 2) or valid_actions.shape != logits.shape:
        raise ValueError("Confidence/mask shape does not match logits")
    if not torch.isfinite(pair_scores).all() or ((pair_scores < 0) | (pair_scores > 1)).any():
        raise ValueError("Detection confidences must be in [0,1]")
    return logits.sigmoid() * pair_scores.prod(-1, keepdim=True).pow(exponent) * valid_actions
