"""UniHOI Eq.4/5 parameter-shared, direction-swapped cross attention.

This is an independent implementation of the specified block, not a released
author HOI model. Its placement and cycle objective are unresolved upstream.
"""

import math
import torch
from torch import nn


class InteractionAwareAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.modality = nn.Embedding(2, width)  # semantic=0, visual=1
        nn.init.normal_(self.modality.weight, std=.02)

    def forward(self, visual, semantic, visual_valid, semantic_valid, direction):
        if direction not in ('detection', 'generation'):
            raise ValueError('Unknown task direction')
        if visual.ndim != 3 or semantic.ndim != 3 or visual.shape[0] != semantic.shape[0]:
            raise ValueError('Batched visual and semantic token sequences required')
        if visual.shape[-1] != self.width or semantic.shape[-1] != self.width:
            raise ValueError('Hidden dimension mismatch')
        if visual_valid.shape != visual.shape[:2] or semantic_valid.shape != semantic.shape[:2]:
            raise ValueError('Padding-mask shape mismatch')
        if visual_valid.dtype != torch.bool or semantic_valid.dtype != torch.bool:
            raise ValueError('Valid-token masks must be boolean')
        if direction == 'detection':
            query, memory, q_valid, k_valid, qt, kt = visual, semantic, visual_valid, semantic_valid, 1, 0
        else:
            query, memory, q_valid, k_valid, qt, kt = semantic, visual, semantic_valid, visual_valid, 0, 1
        if not k_valid.any(-1).all():
            raise ValueError('Every example requires at least one conditioning token')
        q, k, v = self.q(query), self.k(memory), self.v(memory)
        logits = ((q + self.modality.weight[qt]).float() @
                  (k + self.modality.weight[kt]).float().transpose(-1, -2)) / math.sqrt(self.width)
        weights = logits.masked_fill(~k_valid[:, None], -torch.inf).softmax(-1).to(v.dtype)
        output = weights @ v + q
        return output.masked_fill(~q_valid[..., None], 0), weights.masked_fill(~q_valid[..., None], 0)


class PrefixInteractionAdapter(nn.Module):
    """Explicit input-prefix placement hypothesis with no future-target leakage.

    The answer/triplet suffix MUST NOT condition the visual prefix during
    teacher forcing. This guard also applies to any later full-model adapter.
    """
    def __init__(self, width):
        super().__init__()
        self.iaa = InteractionAwareAttention(width)

    def forward(self, embeddings, modality, valid, prefix_lengths, direction):
        if embeddings.ndim != 3 or modality.shape != embeddings.shape[:2] or valid.shape != modality.shape:
            raise ValueError('Invalid token layout')
        if prefix_lengths.shape != (len(embeddings),):
            raise ValueError('One prefix length per example required')
        if not ((modality == 0) | (modality == 1)).all():
            raise ValueError('Only semantic/visual modality IDs are valid')
        if (prefix_lengths < 1).any() or (prefix_lengths > embeddings.shape[1]).any():
            raise ValueError('Invalid prefix boundary')
        output = embeddings.clone()
        for b, boundary in enumerate(prefix_lengths.tolist()):
            prefix_valid = valid[b, :boundary]
            visual_ids = torch.where(prefix_valid & (modality[b, :boundary] == 1))[0]
            semantic_ids = torch.where(prefix_valid & (modality[b, :boundary] == 0))[0]
            if not len(visual_ids) or not len(semantic_ids):
                raise ValueError('Both modalities must exist in the input prefix, not in target labels')
            visual = embeddings[b:b+1, visual_ids]
            semantic = embeddings[b:b+1, semantic_ids]
            result, _ = self.iaa(visual, semantic,
                torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device),
                torch.ones(semantic.shape[:2], dtype=torch.bool, device=semantic.device), direction)
            ids = visual_ids if direction == 'detection' else semantic_ids
            output[b, ids] = result[0]
        return output
