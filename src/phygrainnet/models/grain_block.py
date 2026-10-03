from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class GrainBlock(nn.Module):
    """Compact local + dilated depthwise residual block for granular texture."""

    def __init__(self, channels: int, dilation: int = 2) -> None:
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.context = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.mix = nn.Conv2d(channels * 2, channels, 1)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.act = nn.GELU()
        hidden = max(8, channels // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.local(x), self.context(x)], dim=1)
        y = self.act(self.norm(self.mix(y)))
        y = y * self.gate(y)
        return x + y
