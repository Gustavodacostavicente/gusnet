# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Path-aggregation feature pyramid.

The backbone gives three maps that are good at different things: P3 has fine
spatial detail but weak semantics, P5 the opposite. The neck makes every level
carry both, in two passes:

* **top-down** (FPN, Lin et al., arXiv:1612.03144): semantics from P5 flow down
  into P4 and P3 through upsampling;
* **bottom-up** (PANet, Liu et al., arXiv:1803.01534): localisation detail from
  P3 flows back up through strided convolutions, shortening the path a small
  object's features have to travel to reach the deep levels.

::

    P5 ──┬─────────────────────────────► [+] ──► N5
         │ 1x1, up                         ▲
    P4 ──┼──► [+] ──► CSP ──┬────────────► [+] ──► N4
         │       ▲          │ 1x1, up        ▲
    P3 ──┴───────┴──────────┴──► [+] ► CSP ──┴──► N3
                                             (3x3 stride 2)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gusnet.nn.blocks import ConvNormAct, CSPLayer

__all__ = ["PAFPN"]


class PAFPN(nn.Module):
    """Fuse a three-level pyramid top-down and then bottom-up.

    Args:
        in_channels: ``(c3, c4, c5)`` from the backbone.
        depth: bottlenecks per fusion block, after scaling.
        out_channels: channels of ``(N3, N4, N5)``. Defaults to ``in_channels``.

    Attributes:
        out_channels: channels of the three outputs.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        *,
        depth: int = 3,
        out_channels: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        c3, c4, c5 = in_channels
        o3, o4, o5 = out_channels or in_channels
        depth = max(int(depth), 1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down: bring P5 down to P4's width, fuse, then again into P3.
        self.lateral_p5 = ConvNormAct(c5, c4, kernel_size=1)
        self.fuse_p4 = CSPLayer(c4 * 2, c4, depth=depth, shortcut=False)
        self.lateral_p4 = ConvNormAct(c4, c3, kernel_size=1)
        self.fuse_p3 = CSPLayer(c3 * 2, o3, depth=depth, shortcut=False)

        # Bottom-up: walk back up with strided convolutions.
        self.down_n3 = ConvNormAct(o3, c3, kernel_size=3, stride=2)
        self.fuse_n4 = CSPLayer(c3 * 2, o4, depth=depth, shortcut=False)
        self.down_n4 = ConvNormAct(o4, c4, kernel_size=3, stride=2)
        self.fuse_n5 = CSPLayer(c4 * 2, o5, depth=depth, shortcut=False)

        self.out_channels: tuple[int, int, int] = (o3, o4, o5)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """``(P3, P4, P5)`` -> ``(N3, N4, N5)`` at the same three strides."""
        p3, p4, p5 = features

        # Top-down.
        p5_reduced = self.lateral_p5(p5)
        p4_merged = self.fuse_p4(torch.cat((self.upsample(p5_reduced), p4), dim=1))

        p4_reduced = self.lateral_p4(p4_merged)
        n3 = self.fuse_p3(torch.cat((self.upsample(p4_reduced), p3), dim=1))

        # Bottom-up.
        n4 = self.fuse_n4(torch.cat((self.down_n3(n3), p4_reduced), dim=1))
        n5 = self.fuse_n5(torch.cat((self.down_n4(n4), p5_reduced), dim=1))

        return n3, n4, n5
