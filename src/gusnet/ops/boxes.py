# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Bounding-box representations and metrics.

Three formats are used across GUSNet and are always named explicitly:

``xyxy``
    ``(x1, y1, x2, y2)`` absolute corner coordinates. The canonical format for
    datasets, NMS and evaluation.
``cxcywh``
    ``(cx, cy, w, h)`` centre and size. Used by augmentations and by some loss
    formulations.
``ltrb``
    ``(left, top, right, bottom)`` distances from a reference point to the four
    edges of the box. This is the anchor-free regression target: a grid point
    predicts how far each edge is from itself.

Every function accepts a tensor whose last dimension is 4 and preserves the
leading dimensions, so the same code works for ``(4,)``, ``(N, 4)`` and
``(B, N, 4)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "box_area",
    "box_iou",
    "bbox_iou",
    "clip_boxes",
    "cxcywh_to_xyxy",
    "ltrb_to_xyxy",
    "scale_boxes",
    "xyxy_to_cxcywh",
    "xyxy_to_ltrb",
]

EPS = 1e-7


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """``(..., 4)`` corners -> ``(..., 4)`` centre/size."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """``(..., 4)`` centre/size -> ``(..., 4)`` corners."""
    cx, cy, w, h = boxes.unbind(-1)
    hw, hh = w * 0.5, h * 0.5
    return torch.stack((cx - hw, cy - hh, cx + hw, cy + hh), dim=-1)


def xyxy_to_ltrb(boxes: Tensor, points: Tensor) -> Tensor:
    """Distances from ``points`` to the edges of ``boxes``.

    Args:
        boxes: ``(..., 4)`` in xyxy.
        points: ``(..., 2)`` reference points ``(x, y)``, broadcastable to
            ``boxes``.

    Returns:
        ``(..., 4)`` distances ``(l, t, r, b)``. Negative components mean the
        point lies outside the box.
    """
    px, py = points.unbind(-1)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack((px - x1, py - y1, x2 - px, y2 - py), dim=-1)


def ltrb_to_xyxy(dist: Tensor, points: Tensor) -> Tensor:
    """Inverse of :func:`xyxy_to_ltrb`: decode predicted distances into boxes."""
    px, py = points.unbind(-1)
    left, top, right, bottom = dist.unbind(-1)
    return torch.stack((px - left, py - top, px + right, py + bottom), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """Area of ``(..., 4)`` xyxy boxes; degenerate boxes clamp to zero."""
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w * h


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes1: ``(N, 4)``.
        boxes2: ``(M, 4)``.

    Returns:
        ``(N, M)`` IoU matrix.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=EPS)


def bbox_iou(
    boxes1: Tensor,
    boxes2: Tensor,
    *,
    kind: str = "iou",
) -> Tensor:
    """Element-wise IoU between aligned xyxy boxes.

    Unlike :func:`box_iou` this does not build an ``N x M`` matrix: ``boxes1``
    and ``boxes2`` are broadcast against each other, which is what the box loss
    needs once predictions have been matched to targets.

    Args:
        boxes1: ``(..., 4)`` xyxy.
        boxes2: ``(..., 4)`` xyxy, broadcastable to ``boxes1``.
        kind: one of ``"iou"``, ``"giou"``, ``"diou"``, ``"ciou"``.

    Returns:
        ``(...)`` overlap values. Plain IoU is in ``[0, 1]``; the generalized
        variants can go down to ``-1`` and are what a loss should use
        (``loss = 1 - ciou``).

    References:
        GIoU: Rezatofighi et al., arXiv:1902.09630.
        DIoU/CIoU: Zheng et al., arXiv:1911.08287.
    """
    if kind not in {"iou", "giou", "diou", "ciou"}:
        raise ValueError(f"unknown iou kind: {kind!r}")

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)

    w1, h1 = (b1_x2 - b1_x1).clamp(min=0), (b1_y2 - b1_y1).clamp(min=0)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=0), (b2_y2 - b2_y1).clamp(min=0)

    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter
    iou = inter / union.clamp(min=EPS)
    if kind == "iou":
        return iou

    # Smallest enclosing box.
    cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp(min=0)
    ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp(min=0)

    if kind == "giou":
        enclosing = (cw * ch).clamp(min=EPS)
        return iou - (enclosing - union) / enclosing

    # Squared diagonal of the enclosing box and squared centre distance.
    c2 = cw.pow(2) + ch.pow(2) + EPS
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4

    if kind == "diou":
        return iou - rho2 / c2

    # CIoU adds an aspect-ratio consistency term.
    v = (4 / torch.pi**2) * (
        torch.atan(w2 / h2.clamp(min=EPS)) - torch.atan(w1 / h1.clamp(min=EPS))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=EPS)
    return iou - (rho2 / c2 + alpha * v)


def clip_boxes(boxes: Tensor, shape: tuple[int, int]) -> Tensor:
    """Clamp xyxy boxes to an image of ``shape = (height, width)``."""
    height, width = shape
    x1 = boxes[..., 0].clamp(0, width)
    y1 = boxes[..., 1].clamp(0, height)
    x2 = boxes[..., 2].clamp(0, width)
    y2 = boxes[..., 3].clamp(0, height)
    return torch.stack((x1, y1, x2, y2), dim=-1)


def scale_boxes(
    boxes: Tensor,
    ratio: float | tuple[float, float],
    pad: tuple[float, float],
    orig_shape: tuple[int, int],
) -> Tensor:
    """Map boxes from letterboxed image space back to the original image.

    Args:
        boxes: ``(..., 4)`` xyxy in the letterboxed image.
        ratio: the scale factor used by the letterbox, or ``(rx, ry)``.
        pad: ``(pad_x, pad_y)`` padding added on the left and top.
        orig_shape: ``(height, width)`` of the original image, used to clip.

    Returns:
        ``(..., 4)`` xyxy in original-image coordinates.
    """
    rx, ry = (ratio, ratio) if isinstance(ratio, (int, float)) else ratio
    pad_x, pad_y = pad
    out = boxes.clone()
    out[..., 0::2] = (out[..., 0::2] - pad_x) / rx
    out[..., 1::2] = (out[..., 1::2] - pad_y) / ry
    return clip_boxes(out, orig_shape)
