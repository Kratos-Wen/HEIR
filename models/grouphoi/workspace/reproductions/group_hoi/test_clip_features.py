import pytest
import torch

from .clip_features import DenseCLIPFeatures, official_clip


@pytest.fixture
def adapter():
    clip = official_clip()
    from importlib import import_module
    cls = import_module(clip.__name__ + '.model').CLIP
    model = cls(embed_dim=16, image_resolution=32, vision_layers=1,
                vision_width=64, vision_patch_size=16, context_length=4,
                vocab_size=10, transformer_width=64, transformer_heads=1,
                transformer_layers=1)
    return DenseCLIPFeatures(model, lambda text: torch.tensor([[1, 9, 0, 0]] * len(text)))


def test_tokens_reuse_actual_official_forward_and_keep_pooled_parity(adapter):
    images = torch.randn(2, 3, 32, 32)
    tokens, pooled = adapter.image_features(images)
    assert tokens.shape == (2, 5, 16)
    torch.testing.assert_close(tokens[:, 0], pooled)
    torch.testing.assert_close(pooled, adapter.clip.encode_image(images))
    adapter.train()
    assert not any(m.training for m in adapter.modules())
    assert not any(p.requires_grad for p in adapter.parameters())
    assert not adapter.clip.visual.transformer._forward_hooks


def test_text_contract_and_invalid_joint_request(adapter):
    text, ignored = adapter.extract_features({'text_input': ['one', 'two']})
    assert text.shape == (2, 16) and ignored is None
    with pytest.raises(ValueError, match='exactly one'):
        adapter.extract_features({})
    with pytest.raises(ValueError, match='exactly one'):
        adapter.extract_features({'image': torch.zeros(1), 'text_input': ['one']})


def test_hook_removed_after_failed_forward(adapter):
    with pytest.raises(RuntimeError):
        adapter.image_features(torch.zeros(2, 3, 64, 64))
    assert not adapter.clip.visual.transformer._forward_hooks
