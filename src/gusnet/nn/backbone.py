# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""CSP backbone.

Five stages, each halving the spatial size. The last three are handed to the
neck as the feature pyramid::

    input   640 x 640
    stem    320 x 320   stride 2
    stage1  160 x 160   stride 4
    stage2   80 x  80   stride 8    -> P3, small objects
    stage3   40 x  40   stride 16   -> P4, medium objects
    stage4   20 x  20   stride 32   -> P5, large objects (+ SPP)

Width and depth are scaled by two multipliers, which is how one architecture
becomes a family of sizes: width changes how many channels each stage has,
depth changes how many bottlenecks live inside each CSP stage.
"""

from __future__ import annotations

from torch import Tensor, nn

from gusnet.nn.blocks import SPP, ConvNormAct, CSPLayer, make_divisible

__all__ = ["CSPBackbone"]

# Channels and CSP depths at width=depth=1.0, from the stem outwards.
BASE_CHANNELS = (64, 128, 256, 512, 1024)
BASE_DEPTHS = (3, 6, 6, 3)


class CSPBackbone(nn.Module):
    """Feature extractor producing a three-level pyramid.

    Args:
        width: channel multiplier.
        depth: multiplier on the number of bottlenecks per CSP stage.
        in_channels: channels of the input image.
        divisor: channel counts are rounded to a multiple of this.

    Attributes:
        out_channels: channels of ``(P3, P4, P5)``.
        strides: ``(8, 16, 32)``.
    """

    strides: tuple[int, int, int] = (8, 16, 32)

    def __init__(
        self,
        *,
        width: float = 1.0,
        depth: float = 1.0,
        in_channels: int = 3,
        divisor: int = 8,
    ) -> None:
        super().__init__()
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth multipliers must be positive")

        # Kept so a checkpoint can record how to rebuild this exact backbone.
        self.width = float(width)
        self.depth = float(depth)
        self.in_channels = int(in_channels)

        chans = [make_divisible(c * width, divisor) for c in BASE_CHANNELS]
        depths = [max(round(d * depth), 1) for d in BASE_DEPTHS]
        c1, c2, c3, c4, c5 = chans

        self.stem = ConvNormAct(in_channels, c1, kernel_size=3, stride=2)

        self.stage1 = nn.Sequential(
            ConvNormAct(c1, c2, kernel_size=3, stride=2),
            CSPLayer(c2, c2, depth=depths[0]),
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(c2, c3, kernel_size=3, stride=2),
            CSPLayer(c3, c3, depth=depths[1]),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(c3, c4, kernel_size=3, stride=2),
            CSPLayer(c4, c4, depth=depths[2]),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(c4, c5, kernel_size=3, stride=2),
            CSPLayer(c5, c5, depth=depths[3]),
            SPP(c5, c5),
        )

        self.out_channels: tuple[int, int, int] = (c3, c4, c5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(B, 3, H, W)`` -> ``(P3, P4, P5)`` at strides 8, 16 and 32."""
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5
