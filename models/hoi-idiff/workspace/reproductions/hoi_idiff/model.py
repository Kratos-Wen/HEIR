"""Slice DiT-S from the HOI-IDiff supplement, reusing official Meta DiT blocks.

Underspecified fusion/normalization conventions are in reconstruction.json.
This is a component implementation, not a qualified end-to-end reproduction.
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn


def official_dit():
    path = Path(__file__).resolve().parents[2] / 'reference_repos/DiT/models.py'
    spec = importlib.util.spec_from_file_location('hoi_idiff_official_dit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SliceDiT(nn.Module):
    def __init__(self, noun_count, action_count, appearance_dim, width=384, depth=12, heads=6):
        super().__init__()
        dit = official_dit()
        self.h, self.w = noun_count, action_count
        self.rows = nn.Linear(action_count*2, width)
        self.columns = nn.Linear(noun_count*2, width)
        self.time = dit.TimestepEmbedder(width)
        self.appearance = nn.Sequential(nn.Linear(appearance_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([dit.DiTBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2*width))
        self.row_out = nn.Linear(width, action_count*2)
        self.column_out = nn.Linear(width, noun_count*2)
        self.fusion = nn.Linear(4, 2)
        position = dit.TimestepEmbedder.timestep_embedding(torch.arange(noun_count+action_count), width)
        self.register_buffer('position', position[None])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, image, steps, appearance):
        b = len(image)
        if image.shape != (b, self.h, self.w, 2) or steps.shape != (b,):
            raise ValueError('Invalid HOI image/timestep layout')
        rows = self.rows(image.reshape(b, self.h, self.w*2))
        cols = self.columns(image.permute(0, 2, 1, 3).reshape(b, self.w, self.h*2))
        tokens = torch.cat((rows, cols), 1) + self.position.to(image)
        condition = self.time(steps) + self.appearance(appearance)
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.modulation(condition).chunk(2, -1)
        tokens = self.norm(tokens) * (1+scale[:, None]) + shift[:, None]
        row_image = self.row_out(tokens[:, :self.h]).reshape(b, self.h, self.w, 2)
        col_image = self.column_out(tokens[:, self.h:]).reshape(b, self.w, self.h, 2).permute(0, 2, 1, 3)
        logits = self.fusion(torch.cat((row_image, col_image), -1))
        # Each vertical slice is a distribution over noun x presence/absence.
        return logits.permute(0, 2, 1, 3).reshape(b, self.w, self.h*2).softmax(-1).reshape(
            b, self.w, self.h, 2).permute(0, 2, 1, 3)

    @torch.no_grad()
    def reverse(self, initialized, appearance, steps=50):
        current = initialized
        for k in range(steps, 0, -1):
            current = self(current, torch.full((len(current),), k, device=current.device), appearance)
        return current
