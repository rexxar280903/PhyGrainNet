from __future__ import annotations

import torch
from torch import nn

from .distribution_head import ConstrainedDistributionHead
from .grain_block import GrainBlock, _groups


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[GrainBlock(out_channels) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class PhyGrainNetV0(nn.Module):
    """Initial single-view custom architecture trained from scratch."""

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 48, 96, 160, 224),
        blocks: tuple[int, ...] = (1, 2, 2, 3, 2),
        dropout: float = 0.2,
        num_bins: int = 11,
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks):
            raise ValueError("channels and blocks must have equal length")

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 5, stride=2, padding=2),
            nn.GroupNorm(_groups(channels[0]), channels[0]),
            nn.GELU(),
            *[GrainBlock(channels[0]) for _ in range(blocks[0])],
        )

        stages = []
        for i in range(1, len(channels)):
            stages.append(DownsampleStage(channels[i - 1], channels[i], blocks[i]))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.head = ConstrainedDistributionHead(128, num_bins=num_bins)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stages(x)
        embedding = self.embedding(self.pool(x))
        out = self.head(embedding)
        out["embedding"] = embedding
        return out
