# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Convolutional building blocks.

Everything in the backbone and the neck is assembled from four pieces:

:class:`ConvNormAct`
    convolution + batch norm + SiLU, the unit every other block is made of;
:class:`Bottleneck`
    two convolutions with an optional residual connection;
:class:`CSPLayer`
    a cross-stage partial stage: half the channels go through the bottlenecks,
    half bypass them, and the two halves are fused at the end (CSPNet,
    Wang et al., arXiv:1911.11929);
:class:`SPP`
    spatial pyramid pooling, which widens the receptive field of the deepest
    feature map without adding stride (He et al., arXiv:1406.4729).

Channel counts are always passed through :func:`make_divisible` so that scaled
variants stay friendly to tensor cores and to quantisation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "Bottleneck",
    "CSPLayer",
    "ConvNormAct",
    "SPP",
    "autopad",
    "make_divisible",
]


def make_divisible(value: float, divisor: int = 8) -> int:
    """Round ``value`` to the nearest multiple of ``divisor``, never below 90%.

    Width multipliers produce fractional channel counts; rounding them to a
    multiple of 8 keeps convolutions on the fast path. The 90% floor stops the
    rounding from silently halving a small layer.
    """
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return int(rounded)


def autopad(kernel_size: int, padding: int | None = None, dilation: int = 1) -> int:
    """Padding that keeps the spatial size unchanged for an odd kernel."""
    if padding is not None:
        return padding
    effective = dilation * (kernel_size - 1) + 1
    return effective // 2


class ConvNormAct(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU.

    The convolution carries no bias because the batch norm immediately after it
    has one; keeping both would be redundant parameters and would make
    conv/bn fusion at export time slightly messier.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        *,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))

    def fuse_forward(self, x: Tensor) -> Tensor:
        """Forward without the norm, for use after conv/bn fusion at export."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Two convolutions, optionally residual.

    The default 1x1 then 3x3 is the classic bottleneck shape: the cheap 1x1
    mixes channels, the 3x3 does the spatial work. Two 3x3 convolutions would
    cost roughly 80% more for the same widths, which matters because the
    backbone is almost entirely made of these.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        shortcut: add the input to the output. Only possible when the channel
            counts match; otherwise it is silently disabled.
        expansion: hidden channels as a fraction of ``out_channels``.
        kernel_sizes: kernels of the two convolutions.
        groups: groups of the second convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        shortcut: bool = True,
        expansion: float = 0.5,
        kernel_sizes: tuple[int, int] = (1, 3),
        groups: int = 1,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvNormAct(in_channels, hidden, kernel_sizes[0], 1)
        self.cv2 = ConvNormAct(hidden, out_channels, kernel_sizes[1], 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class CSPLayer(nn.Module):
    """Cross-stage partial stage.

    The input is projected twice. One branch runs through ``depth``
    bottlenecks, the other skips them entirely, and the concatenation is fused
    by a final 1x1 convolution. Half the gradient path therefore never touches
    the bottlenecks, which is what removes the duplicated gradient flow CSPNet
    was designed to fix -- and it costs less compute than putting every channel
    through the stack.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        depth: number of bottlenecks in the deep branch.
        shortcut: residual connections inside the bottlenecks. Backbones want
            ``True``; necks want ``False``, because there the block is fusing
            two different feature maps rather than deepening one.
        expansion: hidden channels as a fraction of ``out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvNormAct(in_channels, hidden, 1, 1)
        self.cv2 = ConvNormAct(in_channels, hidden, 1, 1)
        self.cv3 = ConvNormAct(2 * hidden, out_channels, 1, 1)
        self.blocks = nn.Sequential(
            *(
                Bottleneck(hidden, hidden, shortcut=shortcut, expansion=1.0)
                for _ in range(max(depth, 0))
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        deep = self.blocks(self.cv1(x))
        bypass = self.cv2(x)
        return self.cv3(torch.cat((deep, bypass), dim=1))


class SPP(nn.Module):
    """Spatial pyramid pooling over a single feature map.

    Concatenates the input with max-pools of three different receptive fields,
    so the deepest stage can see context far wider than its kernels without
    another stride. Two equivalent implementations are available:

    ``fast=False``
        three parallel pools with kernels 5, 9 and 13.
    ``fast=True`` (default)
        the same pools computed as a cascade of three 5x5 pools. Stacking two
        stride-1 max-pools of kernel 5 is exactly a max-pool of kernel 9, and
        three give kernel 13 -- the maximum of maxima over overlapping windows
        is the maximum over the union. The cascade is cheaper because each
        stage pools an already pooled map.

    :func:`gusnet.nn.blocks.SPP.forward` gives bit-identical results either
    way, which the test suite asserts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
        fast: bool = True,
    ) -> None:
        super().__init__()
        if fast and kernel_sizes != (5, 9, 13):
            raise ValueError("the fast cascade only implements kernels (5, 9, 13)")
        hidden = in_channels // 2
        self.cv1 = ConvNormAct(in_channels, hidden, 1, 1)
        self.cv2 = ConvNormAct(hidden * (len(kernel_sizes) + 1), out_channels, 1, 1)
        self.fast = fast
        if fast:
            self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.pools = nn.ModuleList(
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        if self.fast:
            y1 = self.pool(x)
            y2 = self.pool(y1)
            y3 = self.pool(y2)
            features = (x, y1, y2, y3)
        else:
            features = (x, *(pool(x) for pool in self.pools))
        return self.cv2(torch.cat(features, dim=1))
