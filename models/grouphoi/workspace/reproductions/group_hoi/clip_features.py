"""Declared dense-feature reconstruction, NOT the missing author LAVIS fork.

The released consumer requires B x L x 512 visual tokens. We reuse the
official OpenAI ViT forward, then apply its final LN/projection to every token.
Keeping CLS, using the last layer and not L2-normalizing are explicit choices;
pooled parity does not establish equivalence to the unavailable author fork.
"""

import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn


B16_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
OFFICIAL_CLIP = Path(__file__).resolve().parents[2] / 'reference_repos/CLIP/clip'


def official_clip():
    name = 'group_reproduction_openai_clip'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, OFFICIAL_CLIP / '__init__.py', submodule_search_locations=[str(OFFICIAL_CLIP)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


class DenseCLIPFeatures(nn.Module):
    def __init__(self, clip_model, tokenize):
        super().__init__()
        self.clip = clip_model.eval().requires_grad_(False)
        self.tokenize = tokenize
        if not all(hasattr(self.clip.visual, name) for name in ('transformer', 'ln_post', 'proj')):
            raise TypeError('Requires the official CLIP vision transformer')
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def image_features(self, images):
        captured = []
        hook = self.clip.visual.transformer.register_forward_hook(
            lambda module, args, output: captured.append(output))
        try:
            pooled = self.clip.encode_image(images)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].ndim != 3:
            raise ValueError('Unexpected official CLIP token layout')
        tokens = self.clip.visual.ln_post(captured[0].permute(1, 0, 2))
        tokens = tokens @ self.clip.visual.proj
        torch.testing.assert_close(tokens[:, 0], pooled, rtol=1e-5, atol=1e-5)
        return tokens.float(), pooled.float()

    @torch.no_grad()
    def extract_features(self, samples):
        image, text = samples.get('image'), samples.get('text_input')
        if (image is None) == (text is None):
            raise ValueError('Supply exactly one of image or text_input')
        if image is not None:
            return self.image_features(image)[0]
        tokens = self.tokenize(text).to(next(self.clip.parameters()).device)
        # The released consumer discards the second tuple member.
        return self.clip.encode_text(tokens).float(), None


def load_verified_b16(path, device):
    clip = official_clip()
    with open(path, 'rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != B16_SHA256:
            raise ValueError('Not the exact public OpenAI ViT-B/16 checkpoint')
    model, preprocess = clip.load(str(path), device='cpu', jit=False)
    model = model.float().to(device)
    if model.visual.proj.shape != (768, 512):
        raise ValueError('Unexpected ViT-B/16 architecture')
    return DenseCLIPFeatures(model, clip.tokenize).to(device), preprocess
