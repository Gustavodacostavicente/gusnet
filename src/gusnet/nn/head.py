# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Anchor-free decoupled detection head.

Two design decisions define this head.

**Decoupled branches.** Classification and localisation want different
features -- one needs to know *what* is there, the other *where* its edges are
-- so each gets its own stack of convolutions instead of sharing one output
tensor. Ge et al. (*YOLOX*, arXiv:2107.08430) showed this both converges faster
and scores better than a coupled head.

**Distances as distributions.** Instead of regressing four numbers, every grid
point predicts four discrete distributions over ``0..reg_max`` (in units of the
level's stride) and the predicted distance is their expectation. A blurry or
ambiguous edge then shows up as a flat distribution rather than as a confident
wrong number, and the shape of that distribution is itself a usable quality
signal. This is the General Focal Loss formulation (Li et al.,
arXiv:2006.04388).

There are no anchor boxes anywhere: a grid point is just a point, and the four
distances take it to the edges of the box it is responsible for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from gusnet.ops.boxes import ltrb_to_xyxy

__all__ = ["DetectHead", "DetectionOutput", "decode_distances", "make_anchor_points"]


def make_anchor_points(
    feature_shapes: Sequence[tuple[int, int]],
    strides: Sequence[int],
    *,
    offset: float = 0.5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Grid points for a feature pyramid, in image pixel coordinates.

    Args:
        feature_shapes: ``(height, width)`` of each level.
        strides: the stride of each level.
        offset: where inside its cell a point sits. ``0.5`` is the centre,
            which is what keeps the four predicted distances symmetric for a
            box that fills the cell.
        device: device of the returned tensors.
        dtype: dtype of the returned points.

    Returns:
        ``(points, strides)`` where ``points`` is ``(A, 2)`` in pixels and
        ``strides`` is ``(A, 1)``, with ``A`` the total number of grid points
        across all levels, concatenated level by level.
    """
    if len(feature_shapes) != len(strides):
        raise ValueError(f"{len(feature_shapes)} feature shapes but {len(strides)} strides")

    all_points: list[Tensor] = []
    all_strides: list[Tensor] = []
    for (height, width), stride in zip(feature_shapes, strides, strict=True):
        xs = (torch.arange(width, device=device, dtype=dtype) + offset) * stride
        ys = (torch.arange(height, device=device, dtype=dtype) + offset) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        all_points.append(torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2))
        all_strides.append(
            torch.full((height * width, 1), float(stride), device=device, dtype=dtype)
        )

    return torch.cat(all_points, dim=0), torch.cat(all_strides, dim=0)


def decode_distances(
    reg_logits: Tensor,
    project: Tensor,
) -> Tensor:
    """Turn ``(..., 4, reg_max + 1)`` logits into ``(..., 4)`` expected distances.

    The softmax makes each of the four rows a probability distribution over the
    integer distances ``0..reg_max``; the dot product with ``project`` takes its
    expectation. The result is in units of the level's stride.
    """
    return torch.einsum("...k,k->...", reg_logits.softmax(dim=-1), project)


class DetectionOutput(dict):
    """Head output, kept as a plain ``dict`` so it survives ``torch.compile``.

    Keys:
        ``cls_logits``: ``(B, A, num_classes)`` raw, pre-sigmoid.
        ``reg_logits``: ``(B, A, 4, reg_max + 1)`` raw distance distributions.
        ``points``: ``(A, 2)`` grid points in image pixels.
        ``strides``: ``(A, 1)`` stride of each point.
        ``boxes``: ``(B, A, 4)`` decoded xyxy in image pixels.
        ``scores``: ``(B, A, num_classes)`` per-class confidence in ``[0, 1]``.
    """


class DetectHead(nn.Module):
    """Decoupled anchor-free head with distributional box regression.

    Args:
        in_channels: channels of each pyramid level coming from the neck.
        num_classes: number of object classes.
        strides: stride of each level, aligned with ``in_channels``.
        reg_max: the largest distance, in stride units, the head can express.
            With ``reg_max=16`` and stride 32 a single point can describe an
            edge up to 512 px away, which covers any box at 640 px input.
        hidden_channels: width of the two branch stems. Defaults to the
            narrowest input level, which keeps the head cheap on the large
            feature maps where it runs most often.
        stem_depth: convolutions per branch stem.

    Attributes:
        num_outputs: ``num_classes + 4 * (reg_max + 1)`` values per grid point.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        *,
        strides: Sequence[int] = (8, 16, 32),
        reg_max: int = 16,
        hidden_channels: int | None = None,
        stem_depth: int = 2,
    ) -> None:
        super().__init__()
        if len(in_channels) != len(strides):
            raise ValueError(f"{len(in_channels)} levels but {len(strides)} strides")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if reg_max < 1:
            raise ValueError("reg_max must be at least 1")

        # Imported here to keep the module import graph acyclic and light.
        from gusnet.nn.blocks import ConvNormAct

        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.strides = tuple(int(s) for s in strides)
        self.num_levels = len(self.strides)
        self.num_outputs = self.num_classes + 4 * (self.reg_max + 1)

        hidden = int(hidden_channels or min(in_channels))

        self.cls_stems = nn.ModuleList()
        self.reg_stems = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for channels in in_channels:
            self.cls_stems.append(_make_stem(ConvNormAct, channels, hidden, stem_depth))
            self.reg_stems.append(_make_stem(ConvNormAct, channels, hidden, stem_depth))
            self.cls_preds.append(nn.Conv2d(hidden, self.num_classes, kernel_size=1))
            self.reg_preds.append(nn.Conv2d(hidden, 4 * (self.reg_max + 1), kernel_size=1))

        # Fixed integers 0..reg_max, the support of the distance distributions.
        self.register_buffer(
            "project", torch.arange(self.reg_max + 1, dtype=torch.float32), persistent=False
        )

        self.reset_parameters()

    def reset_parameters(self, prior_probability: float = 0.01) -> None:
        """Initialise the prediction layers.

        The classification bias is set so that every logit starts at a
        confidence of ``prior_probability``. Without this the head begins by
        claiming an object at each of the thousands of grid points, and the
        resulting loss spike can stall or diverge the first epochs -- the
        standard fix from the focal loss paper (Lin et al., arXiv:1708.02002).
        """
        bias = -math.log((1 - prior_probability) / prior_probability)
        for layer in self.cls_preds:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, bias)
        for layer in self.reg_preds:
            nn.init.normal_(layer.weight, std=0.01)
            # A flat-ish start biased towards short distances: boxes begin about
            # one stride wide instead of at an arbitrary scale.
            nn.init.constant_(layer.bias, 1.0)

    def forward(self, features: Sequence[Tensor]) -> DetectionOutput:
        """Predict from a feature pyramid.

        Args:
            features: one ``(B, C, H, W)`` tensor per level, in the same order
                as ``strides``.

        Returns:
            A :class:`DetectionOutput`. ``boxes`` and ``scores`` are always
            produced; in training the raw ``cls_logits`` and ``reg_logits`` are
            what the loss consumes.
        """
        if len(features) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} feature maps, got {len(features)}")

        cls_outputs: list[Tensor] = []
        reg_outputs: list[Tensor] = []
        shapes: list[tuple[int, int]] = []

        for level, feature in enumerate(features):
            batch = feature.shape[0]
            height, width = feature.shape[2:]
            shapes.append((int(height), int(width)))

            cls = self.cls_preds[level](self.cls_stems[level](feature))
            reg = self.reg_preds[level](self.reg_stems[level](feature))

            # (B, C, H, W) -> (B, H*W, C): grid points become the sequence axis
            # so that every level can be concatenated into one flat prediction.
            cls_outputs.append(cls.permute(0, 2, 3, 1).reshape(batch, height * width, -1))
            reg_outputs.append(reg.permute(0, 2, 3, 1).reshape(batch, height * width, -1))

        cls_logits = torch.cat(cls_outputs, dim=1)
        reg_logits = torch.cat(reg_outputs, dim=1).reshape(
            cls_logits.shape[0], -1, 4, self.reg_max + 1
        )

        points, strides = make_anchor_points(
            shapes, self.strides, device=cls_logits.device, dtype=cls_logits.dtype
        )

        # Distances are predicted in stride units; scale them into pixels.
        distances = decode_distances(reg_logits, self.project.to(reg_logits.dtype))
        boxes = ltrb_to_xyxy(distances * strides, points)

        return DetectionOutput(
            cls_logits=cls_logits,
            reg_logits=reg_logits,
            points=points,
            strides=strides,
            boxes=boxes,
            scores=cls_logits.sigmoid(),
        )


def _make_stem(conv_cls, in_channels: int, hidden: int, depth: int) -> nn.Sequential:
    """``depth`` 3x3 convolutions, the first one changing the channel count."""
    layers = [conv_cls(in_channels, hidden, kernel_size=3)]
    layers += [conv_cls(hidden, hidden, kernel_size=3) for _ in range(max(depth - 1, 0))]
    return nn.Sequential(*layers)
