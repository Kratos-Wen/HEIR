import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations"))

from frozen_dinotxt import FrozenDINOtxtVisualBackbone  # noqa: E402


class _FrozenToyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 8, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens).tanh()


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
