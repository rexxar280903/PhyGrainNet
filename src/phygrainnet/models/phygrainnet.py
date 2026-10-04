from __future__ import annotations

import torch
import torch.nn.functional as F
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


class TextureFilters(nn.Module):
    """Fixed (non-learned) Sobel-magnitude and Laplacian channels of luminance.

    These are analytic filters, not pretrained weights, so DEC-001 holds."""

    def __init__(self) -> None:
        super().__init__()
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("kernels", torch.stack([sx, sx.t(), lap]).unsqueeze(1))
        self.register_buffer("rgb2y", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x * self.rgb2y).sum(dim=1, keepdim=True)
        g = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="reflect"), self.kernels)
        mag = torch.sqrt(g[:, :1] ** 2 + g[:, 1:2] ** 2 + 1e-6)
        return torch.cat([x, mag, g[:, 2:3]], dim=1)


class GrainEncoder(nn.Module):
    """Tile encoder trained from scratch: stem + GrainBlock stages + mean/std pooling."""

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 48, 96, 160, 224),
        blocks: tuple[int, ...] = (1, 2, 2, 3, 2),
        texture_stream: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks):
            raise ValueError("channels and blocks must have equal length")
        self.texture = TextureFilters() if texture_stream else None
        in_ch = 5 if texture_stream else 3
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], 5, stride=2, padding=2),
            nn.GroupNorm(_groups(channels[0]), channels[0]),
            nn.GELU(),
            *[GrainBlock(channels[0]) for _ in range(blocks[0])],
        )
        self.stages = nn.Sequential(
            *[DownsampleStage(channels[i - 1], channels[i], blocks[i]) for i in range(1, len(channels))]
        )
        self.out_dim = channels[-1] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.texture is not None:
            x = self.texture(x)
        f = self.stages(self.stem(x))
        mean = f.mean(dim=(2, 3))
        std = f.flatten(2).std(dim=2)
        return torch.cat([mean, std], dim=1)  # texture statistics, not just the average


class AttentionPool(nn.Module):
    """Aggregate a variable number of tile embeddings into one sample embedding."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.out_dim = dim * 3

    def forward(self, e: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # e: (B, T, D); mask: (B, T) 1 = valid
        if mask is None:
            mask = torch.ones(e.shape[:2], device=e.device, dtype=e.dtype)
        mask = mask.to(e.dtype)
        s = self.score(e).squeeze(-1).masked_fill(mask == 0, float("-inf"))
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        attn = (a * e).sum(dim=1)
        n = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (e * mask.unsqueeze(-1)).sum(dim=1) / n
        var = (((e - mean.unsqueeze(1)) ** 2) * mask.unsqueeze(-1)).sum(dim=1) / n
        return torch.cat([mean, var.clamp(min=0).sqrt(), attn], dim=1)


class PhyGrainNetMV(nn.Module):
    """Multi-scale, multi-view PhyGrainNet.

    Input  x: (B, T, S, 3, H, W) - T tiles drawn from all photos of one physical
              sample, S concentric physical fields per tile (e.g. 25.6 mm and 102.4 mm),
              all resampled to a fixed canonical pixels-per-mm before tiling.
    Output dict with 'cumulative' (B, 11), valid by construction (softmax + cumsum).
    """

    def __init__(
        self,
        num_scales: int = 2,
        channels: tuple[int, ...] = (32, 48, 96, 160, 224),
        blocks: tuple[int, ...] = (1, 2, 2, 3, 2),
        texture_stream: bool = True,
        share_scales: bool = True,
        embed_dim: int = 256,
        dropout: float = 0.2,
        num_bins: int = 11,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        n_enc = 1 if share_scales else num_scales
        self.encoders = nn.ModuleList(
            [GrainEncoder(tuple(channels), tuple(blocks), texture_stream) for _ in range(n_enc)]
        )
        enc_dim = self.encoders[0].out_dim
        self.scale_embed = nn.Parameter(torch.zeros(num_scales, enc_dim)) if share_scales and num_scales > 1 else None
        self.tile_proj = nn.Sequential(
            nn.Linear(enc_dim * num_scales, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)
        )
        self.pool = AttentionPool(embed_dim)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.pool.out_dim, embed_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.head = ConstrainedDistributionHead(embed_dim, num_bins=num_bins)

    def encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, c, h, w = x.shape
        feats = []
        for k in range(s):
            enc = self.encoders[0 if len(self.encoders) == 1 else k]
            f = enc(x[:, :, k].reshape(b * t, c, h, w))
            if self.scale_embed is not None:
                f = f + self.scale_embed[k]
            feats.append(f)
        return self.tile_proj(torch.cat(feats, dim=1)).view(b, t, -1)

    def forward(self, x: torch.Tensor, tile_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        e = self.encode_tiles(x)
        z = self.mlp(self.pool(e, tile_mask))
        out = self.head(z)
        out["embedding"] = z
        return out

    def encoder_state(self) -> dict:
        return {"encoders": self.encoders.state_dict()}

    def load_encoder_state(self, state: dict, strict: bool = True) -> None:
        self.encoders.load_state_dict(state["encoders"], strict=strict)


class PhyGrainNetV0(nn.Module):
    """Initial single-view custom architecture trained from scratch (kept for ablation C000)."""

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


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    name = m.get("name", "phygrainnet_mv")
    if name == "phygrainnet_v0":
        return PhyGrainNetV0(tuple(m.get("channels", (32, 48, 96, 160, 224))), tuple(m.get("blocks", (1, 2, 2, 3, 2))), m.get("dropout", 0.2))
    if name == "phygrainnet_mv":
        return PhyGrainNetMV(
            num_scales=len(cfg.get("data", {}).get("fields_mm", [25.6, 102.4])),
            channels=tuple(m.get("channels", (32, 48, 96, 160, 224))),
            blocks=tuple(m.get("blocks", (1, 2, 2, 3, 2))),
            texture_stream=bool(m.get("texture_stream", True)),
            share_scales=bool(m.get("share_scales", True)),
            embed_dim=int(m.get("embed_dim", 256)),
            dropout=float(m.get("dropout", 0.2)),
        )
    raise ValueError(f"unknown model {name!r}")
