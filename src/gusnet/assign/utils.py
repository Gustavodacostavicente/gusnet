# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Shared machinery for label assignment.

Assignment answers one question: of the thousands of grid points the head
predicts from, which ones are responsible for which object? Everything here is
the bookkeeping the actual strategies (:mod:`gusnet.assign.tal`,
:mod:`gusnet.assign.simota`) build on.

Ground truth arrives from the dataloader as a flat ``(N, 6)`` table because
images hold different numbers of objects. Assigners want rectangular tensors,
so :func:`targets_to_batch` pads that table to ``(B, M, ...)`` and returns a
mask saying which slots are real.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "Assignment",
    "points_in_boxes",
    "points_in_centers",
    "resolve_conflicts",
    "targets_to_batch",
]

EPS = 1e-9


@dataclass
class Assignment:
    """What each grid point is supposed to predict.

    Attributes:
        fg_mask: ``(B, A)`` bool, ``True`` for points assigned to an object.
        target_labels: ``(B, A)`` int64 class index. Meaningless where
            ``fg_mask`` is ``False``.
        target_boxes: ``(B, A, 4)`` xyxy of the assigned object, in image
            pixels. Meaningless where ``fg_mask`` is ``False``.
        target_scores: ``(B, A, C)`` classification target. Not one-hot: the
            value at the assigned class carries how well that point already
            predicts the object, so the classification head is trained to
            output a quality estimate rather than a constant 1.
        target_gt_index: ``(B, A)`` index of the assigned object within the
            padded ground-truth tensors.
    """

    fg_mask: Tensor
    target_labels: Tensor
    target_boxes: Tensor
    target_scores: Tensor
    target_gt_index: Tensor

    @property
    def num_foreground(self) -> int:
        """How many points were assigned to an object across the batch."""
        return int(self.fg_mask.sum())

    def to(self, device: torch.device | str) -> Assignment:
        return Assignment(
            fg_mask=self.fg_mask.to(device),
            target_labels=self.target_labels.to(device),
            target_boxes=self.target_boxes.to(device),
            target_scores=self.target_scores.to(device),
            target_gt_index=self.target_gt_index.to(device),
        )


def targets_to_batch(targets: Tensor, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Pad the flat target table into rectangular per-image tensors.

    Args:
        targets: ``(N, 6)`` rows of ``[batch_index, class, x1, y1, x2, y2]``,
            as produced by :func:`gusnet.data.loader.collate_detection`.
        batch_size: number of images in the batch.

    Returns:
        ``(gt_labels, gt_boxes, gt_mask)`` shaped ``(B, M)``, ``(B, M, 4)`` and
        ``(B, M)``, where ``M`` is the largest object count in the batch (at
        least 1, so the tensors are never degenerate). ``gt_mask`` marks the
        real entries; padded slots hold zeros.
    """
    if targets.ndim != 2 or targets.shape[1] != 6:
        raise ValueError(f"expected targets of shape (N, 6), got {tuple(targets.shape)}")

    device = targets.device
    batch_index = targets[:, 0].long()
    counts = torch.bincount(batch_index, minlength=batch_size)
    max_objects = int(counts.max()) if len(targets) else 0
    max_objects = max(max_objects, 1)

    gt_labels = torch.zeros((batch_size, max_objects), dtype=torch.int64, device=device)
    gt_boxes = torch.zeros((batch_size, max_objects, 4), dtype=targets.dtype, device=device)
    gt_mask = torch.zeros((batch_size, max_objects), dtype=torch.bool, device=device)

    for image in range(batch_size):
        rows = targets[batch_index == image]
        count = len(rows)
        if count == 0:
            continue
        gt_labels[image, :count] = rows[:, 1].long()
        gt_boxes[image, :count] = rows[:, 2:6]
        gt_mask[image, :count] = True

    return gt_labels, gt_boxes, gt_mask


def points_in_boxes(points: Tensor, boxes: Tensor, *, eps: float = 0.01) -> Tensor:
    """Which grid points fall inside which boxes.

    A point is a candidate for an object only if it lies inside it: an
    anchor-free head predicts distances to the four edges, and those distances
    must be positive for the prediction to make sense.

    Args:
        points: ``(A, 2)`` grid points in image pixels.
        boxes: ``(B, M, 4)`` xyxy boxes.
        eps: margin, so points exactly on an edge do not count.

    Returns:
        ``(B, M, A)`` bool.
    """
    px = points[:, 0].view(1, 1, -1)
    py = points[:, 1].view(1, 1, -1)
    x1, y1, x2, y2 = boxes.unsqueeze(-1).unbind(-2)
    return (px > x1 + eps) & (px < x2 - eps) & (py > y1 + eps) & (py < y2 - eps)


def points_in_centers(
    points: Tensor, boxes: Tensor, strides: Tensor, *, radius: float = 2.5
) -> Tensor:
    """Which grid points fall near the centre of which boxes.

    The centre region is a square of side ``2 * radius * stride`` around each
    object's centre. It matters for two reasons: points near the centre see the
    whole object and make better predictions than points near an edge, and for
    very large objects the in-box mask alone would nominate thousands of
    candidates.

    Args:
        points: ``(A, 2)`` grid points in image pixels.
        boxes: ``(B, M, 4)`` xyxy boxes.
        strides: ``(A, 1)`` stride of each point, so the region scales with the
            pyramid level.
        radius: half-side of the region, in strides.

    Returns:
        ``(B, M, A)`` bool.
    """
    px = points[:, 0].view(1, 1, -1)
    py = points[:, 1].view(1, 1, -1)
    half = (radius * strides.view(1, 1, -1)).to(boxes.dtype)

    x1, y1, x2, y2 = boxes.unsqueeze(-1).unbind(-2)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return (px > cx - half) & (px < cx + half) & (py > cy - half) & (py < cy + half)


def resolve_conflicts(mask_pos: Tensor, overlaps: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Give every grid point at most one object.

    Two objects can nominate the same point -- overlapping boxes, or one object
    inside another. Training a single point towards two different boxes would
    average them into a box that matches neither, so the tie goes to the object
    the point already localises best.

    Args:
        mask_pos: ``(B, M, A)`` candidate assignments.
        overlaps: ``(B, M, A)`` IoU between each point's prediction and each
            object, used to break ties.

    Returns:
        ``(mask_pos, fg_mask, target_gt_index)``: the cleaned ``(B, M, A)``
        mask, a ``(B, A)`` bool of assigned points, and a ``(B, A)`` int64
        index of the object each point belongs to.
    """
    mask_pos = mask_pos.bool()
    contested = mask_pos.sum(dim=1) > 1  # (B, A)

    if bool(contested.any()):
        # Rank only the objects that actually nominated the point, so the
        # winner is always one of its candidates.
        candidate_overlaps = overlaps * mask_pos.to(overlaps.dtype)
        best = candidate_overlaps.argmax(dim=1, keepdim=True)  # (B, 1, A)

        winner = torch.zeros_like(mask_pos)
        winner.scatter_(1, best, torch.ones_like(best, dtype=torch.bool))
        mask_pos = torch.where(contested.unsqueeze(1), winner, mask_pos)

    fg_mask = mask_pos.any(dim=1)
    target_gt_index = mask_pos.to(torch.uint8).argmax(dim=1)
    return mask_pos, fg_mask, target_gt_index
