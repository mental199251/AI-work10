"""Noise-prediction networks used by the DDPM experiments."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=time.device, dtype=torch.float32) * -scale
        )
        embeddings = time[:, None].float() * frequencies[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return F.pad(embeddings, (0, self.dim - embeddings.shape[-1]))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.condition_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(condition_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.block1(x)
        hidden = hidden + self.condition_projection(condition)[:, :, None, None]
        hidden = self.block2(hidden)
        return hidden + self.residual(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleUNet(nn.Module):
    """A compact 32x32 U-Net supporting optional class conditioning."""

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        num_classes: Optional[int] = None,
        time_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        time_dim = time_dim or base_channels * 4
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.time_dim = time_dim
        self.null_label = num_classes if num_classes is not None else None

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.label_embedding = (
            nn.Embedding(num_classes + 1, time_dim) if num_classes is not None else None
        )

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.init_conv = nn.Conv2d(image_channels, c1, 3, padding=1)
        self.down1 = ResidualBlock(c1, c1, time_dim)
        self.downsample1 = Downsample(c1)
        self.down2 = ResidualBlock(c1, c2, time_dim)
        self.downsample2 = Downsample(c2)
        self.down3 = ResidualBlock(c2, c3, time_dim)
        self.downsample3 = Downsample(c3)
        self.mid1 = ResidualBlock(c3, c3, time_dim)
        self.mid2 = ResidualBlock(c3, c3, time_dim)
        self.upsample3 = Upsample(c3)
        self.up3 = ResidualBlock(c3 + c3, c2, time_dim)
        self.upsample2 = Upsample(c2)
        self.up2 = ResidualBlock(c2 + c2, c1, time_dim)
        self.upsample1 = Upsample(c1)
        self.up1 = ResidualBlock(c1 + c1, c1, time_dim)
        self.final = nn.Sequential(
            nn.GroupNorm(_group_count(c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, image_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition = self.time_mlp(time)
        if self.label_embedding is not None:
            if labels is None:
                labels = torch.full(
                    (x.shape[0],), self.null_label, device=x.device, dtype=torch.long
                )
            condition = condition + self.label_embedding(labels)

        x = self.init_conv(x)
        skip1 = self.down1(x, condition)
        x = self.downsample1(skip1)
        skip2 = self.down2(x, condition)
        x = self.downsample2(skip2)
        skip3 = self.down3(x, condition)
        x = self.downsample3(skip3)
        x = self.mid1(x, condition)
        x = self.mid2(x, condition)
        x = self.upsample3(x)
        x = self.up3(torch.cat((x, skip3), dim=1), condition)
        x = self.upsample2(x)
        x = self.up2(torch.cat((x, skip2), dim=1), condition)
        x = self.upsample1(x)
        x = self.up1(torch.cat((x, skip1), dim=1), condition)
        return self.final(x)

    def model_config(self) -> Dict[str, Any]:
        return {
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "num_classes": self.num_classes,
            "time_dim": self.time_dim,
        }


def build_model(config: Dict[str, Any]) -> SimpleUNet:
    return SimpleUNet(
        image_channels=int(config["image_channels"]),
        base_channels=int(config["base_channels"]),
        num_classes=config.get("num_classes"),
        time_dim=config.get("time_dim"),
    )

