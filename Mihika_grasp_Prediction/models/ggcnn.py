"""A small GG-CNN-inspired baseline model.

This is not a reproduction of the full GG-CNN paper. It keeps the MVP simple:
an encoder-decoder CNN predicts dense grasp quality, angle, and width maps.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GGCNNBaseline(nn.Module):
    """Lightweight fully convolutional grasp predictor."""

    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.decoder1 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.decoder2 = ConvBlock(base_channels * 2, base_channels)

        self.quality_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.angle_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.width_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.encoder1(x)
        e2 = self.encoder2(torch.relu(self.down1(e1)))
        bottleneck = self.bottleneck(torch.relu(self.down2(e2)))

        d1 = self.up1(bottleneck)
        d1 = self.decoder1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.decoder2(torch.cat([d2, e1], dim=1))

        return {
            "quality": self.quality_head(d2),
            "angle": self.angle_head(d2),
            "width": self.width_head(d2),
        }
