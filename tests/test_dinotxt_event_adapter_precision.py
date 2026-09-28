import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations"))

from dinotxt_event_adapter import FrozenDINOtxtVisualBackbone  # noqa: E402


class _FrozenToyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 8, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens).tanh()


class _ToyReferenceBindingBridge(FrozenDINOtxtVisualBackbone):
    output_dim = 8

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.head = _FrozenContextHead().float()
        self.head_compute_dtype = torch.float32
        self.num_register_tokens = 0
        self.input_size = 32
        self.backbone_calls = 0

    def _backbone_tokens(self, image: torch.Tensor) -> torch.Tensor:
        self.backbone_calls += 1
        return torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 40.0


class _FrozenContextHead(_FrozenToyHead):
    """A deterministic frozen attention head that mixes query and image tokens."""

    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(8))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention = (tokens @ tokens.transpose(-1, -2) / 8**0.5).softmax(-1)
        return self.projection(tokens + attention @ tokens).tanh()


def test_frozen_dinotxt_head_escapes_outer_bf16_autocast() -> None:
    bridge = FrozenDINOtxtVisualBackbone.__new__(FrozenDINOtxtVisualBackbone)
    nn.Module.__init__(bridge)
    bridge.head = _FrozenToyHead().float()
    bridge.head_compute_dtype = torch.float32

    event_tokens = torch.randn(2, 16, 8, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        aligned = bridge._forward_head(event_tokens)
        loss = aligned.square().mean()
    loss.backward()

    assert aligned.dtype == torch.float32
    assert event_tokens.grad is not None
    assert torch.isfinite(event_tokens.grad).all()
    assert all(parameter.grad is None for parameter in bridge.head.parameters())


def test_reference_and_binding_reads_share_backbone_but_not_head_context() -> None:
    bridge = _ToyReferenceBindingBridge()
    reference = torch.linspace(-0.4, 0.7, 16).reshape(1, 2, 8).requires_grad_()
    binding = torch.linspace(-0.9, 0.3, 40).reshape(1, 5, 8).requires_grad_()
    outputs = bridge.forward_with_reference_and_binding_queries(
        [torch.randn(3, 24, 24)], reference, binding
    )
    (
        reference_aligned,
        reference_grid,
        reference_semantic,
        binding_aligned,
        binding_grid,
        binding_semantic,
    ) = outputs
    assert bridge.backbone_calls == 1
    assert reference_aligned.shape == reference.shape
    assert binding_aligned.shape == binding.shape
    assert reference_grid.shape == binding_grid.shape == (1, 8, 2, 2)
    assert reference_semantic.shape == binding_semantic.shape == (1, 16)
    assert not torch.equal(reference_grid, binding_grid)

    # Changing only the binding context must leave the reference read unchanged.
    changed = bridge.forward_with_reference_and_binding_queries(
        [torch.zeros(3, 24, 24)], reference, binding + 0.7
    )
    for original, repeated in zip(outputs[:3], changed[:3]):
        torch.testing.assert_close(original, repeated, rtol=0, atol=0)
    assert not torch.allclose(binding_grid, changed[4])
    reference_gradient, binding_gradient = torch.autograd.grad(
        reference_grid.square().flatten()[0], (reference, binding),
        allow_unused=True, retain_graph=True,
    )
    assert reference_gradient is not None and reference_gradient.abs().sum() > 0
    assert binding_gradient is None

    sum(value.sum() for value in outputs).backward()
    assert reference.grad is not None and torch.isfinite(reference.grad).all()
    assert binding.grad is not None and torch.isfinite(binding.grad).all()
    assert all(parameter.grad is None for parameter in bridge.head.parameters())
