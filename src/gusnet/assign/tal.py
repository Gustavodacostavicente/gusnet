# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Task-aligned label assignment.

The problem this solves is that classification and localisation disagree. A
grid point can be very confident about the class and put the box in the wrong
place, or localise perfectly while scoring the class low. If assignment ignores
that, the two heads drift apart and NMS ends up keeping confident boxes that
are badly placed.

The task-aligned metric scores each candidate with both at once::

    t = s**alpha * u**beta

where ``s`` is the predicted confidence for the object's class and ``u`` is the
IoU between the point's predicted box and the object. Only points that score
high on *both* are selected, and -- this is the part that does the real work --
the classification target is not 1 but that same metric, rescaled. A point that
localises the object poorly is therefore trained towards a *low* confidence, so
the score the model outputs ends up meaning "this box is good", which is
exactly what NMS needs it to mean.

Reimplemented from Feng et al., *TOOD* (arXiv:2108.07755) and the task-aligned
assignment described in the PP-YOLOE report (arXiv:2203.16250).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gusnet.assign.utils import (
    EPS,
    Assignment,
    points_in_boxes,
    resolve_conflicts,
)
from gusnet.ops.boxes import bbox_iou

__all__ = ["TaskAlignedAssigner"]


class TaskAlignedAssigner(nn.Module):
    """Assign grid points to objects by a joint classification/IoU metric.

    Args:
        num_classes: number of object classes.
        topk: candidates kept per object. Larger values speed up early
            training by supplying more positives, at the cost of admitting
            worse ones.
        alpha: exponent on the classification score.
        beta: exponent on the IoU. The usual ``beta`` far above ``alpha``
            makes localisation dominate the ranking, with the score acting as
            a tie-breaker.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        topk: int = 13,
        alpha: float = 0.5,
        beta: float = 6.0,
    ) -> None:
        super().__init__()
        if topk < 1:
            raise ValueError("topk must be at least 1")
        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        points: Tensor,
        gt_labels: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
    ) -> Assignment:
        """Assign a batch.

        Args:
            pred_scores: ``(B, A, C)`` confidences in ``[0, 1]``.
            pred_boxes: ``(B, A, 4)`` predicted xyxy in image pixels.
            points: ``(A, 2)`` grid points in image pixels.
            gt_labels: ``(B, M)`` class indices.
            gt_boxes: ``(B, M, 4)`` xyxy in image pixels.
            gt_mask: ``(B, M)`` bool marking real objects among the padding.

        Returns:
            An :class:`~gusnet.assign.utils.Assignment`.

        Note:
            Runs under ``no_grad``: assignment decides *what* to learn, it is
            not itself learned. Letting gradients flow back through the
            selection would let the model reduce its loss by changing which
            points it is scored on instead of by predicting better.
        """
        batch, anchors, _ = pred_scores.shape
        gt_mask = gt_mask.bool()

        if gt_mask.numel() == 0 or not bool(gt_mask.any()):
            return self._empty(batch, anchors, pred_scores)

        metric, overlaps = self._alignment_metric(
            pred_scores, pred_boxes, gt_labels, gt_boxes, gt_mask
        )

        # A candidate must lie inside the object: the head can only describe a
        # box by positive distances to its four edges.
        inside = points_in_boxes(points, gt_boxes) & gt_mask.unsqueeze(-1)
        candidates = metric * inside

        mask_pos = self._select_topk(candidates) & inside
        mask_pos, fg_mask, target_gt_index = resolve_conflicts(mask_pos, overlaps)

        return self._gather_targets(
            gt_labels, gt_boxes, target_gt_index, fg_mask, mask_pos, metric, overlaps
        )

    # ------------------------------------------------------------------ internals

    def _alignment_metric(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        gt_labels: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """``(B, M, A)`` alignment metric and the IoU matrix it is built from."""
        # IoU of every prediction against every object.
        overlaps = bbox_iou(gt_boxes.unsqueeze(2), pred_boxes.unsqueeze(1), kind="iou").clamp_(
            min=0
        )

        # The predicted confidence for each object's own class.
        index = gt_labels.clamp(min=0).unsqueeze(-1).expand(-1, -1, pred_scores.shape[1])
        scores = pred_scores.permute(0, 2, 1).gather(1, index)  # (B, M, A)

        metric = scores.pow(self.alpha) * overlaps.pow(self.beta)
        metric = metric * gt_mask.unsqueeze(-1)
        return metric, overlaps * gt_mask.unsqueeze(-1)

    def _select_topk(self, candidates: Tensor) -> Tensor:
        """Keep the ``topk`` best-scoring points per object."""
        num_anchors = candidates.shape[-1]
        k = min(self.topk, num_anchors)

        _, indices = candidates.topk(k, dim=-1)
        mask = torch.zeros_like(candidates, dtype=torch.bool)
        mask.scatter_(-1, indices, torch.ones_like(indices, dtype=torch.bool))

        # topk always returns k entries; drop the ones that scored nothing, or
        # an object with three good candidates would collect ten bad ones.
        return mask & (candidates > 0)

    def _gather_targets(
        self,
        gt_labels: Tensor,
        gt_boxes: Tensor,
        target_gt_index: Tensor,
        fg_mask: Tensor,
        mask_pos: Tensor,
        metric: Tensor,
        overlaps: Tensor,
    ) -> Assignment:
        batch = gt_labels.shape[0]
        batch_index = torch.arange(batch, device=gt_labels.device).unsqueeze(-1)

        target_labels = gt_labels[batch_index, target_gt_index]  # (B, A)
        target_boxes = gt_boxes[batch_index, target_gt_index]  # (B, A, 4)

        target_scores = F.one_hot(
            target_labels.clamp(min=0, max=self.num_classes - 1), self.num_classes
        ).to(metric.dtype)
        target_scores = target_scores * fg_mask.unsqueeze(-1)

        # Rescale the metric per object so its best candidate is trained
        # towards that object's best achieved IoU. Without the rescaling the
        # targets would be arbitrarily small numbers whose size depends on how
        # confident the model happens to be, and the classification loss would
        # shrink as the metric shrinks rather than as the predictions improve.
        positive_metric = metric * mask_pos
        best_metric = positive_metric.amax(dim=-1, keepdim=True)
        best_overlap = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        normalised = positive_metric * best_overlap / (best_metric + EPS)
        weight = normalised.amax(dim=1).unsqueeze(-1)  # (B, A, 1)

        return Assignment(
            fg_mask=fg_mask,
            target_labels=target_labels,
            target_boxes=target_boxes,
            target_scores=target_scores * weight,
            target_gt_index=target_gt_index,
        )

    def _empty(self, batch: int, anchors: int, reference: Tensor) -> Assignment:
        """Everything is background -- the batch has no objects at all."""
        device = reference.device
        return Assignment(
            fg_mask=torch.zeros((batch, anchors), dtype=torch.bool, device=device),
            target_labels=torch.zeros((batch, anchors), dtype=torch.int64, device=device),
            target_boxes=torch.zeros((batch, anchors, 4), dtype=reference.dtype, device=device),
            target_scores=torch.zeros(
                (batch, anchors, self.num_classes), dtype=reference.dtype, device=device
            ),
            target_gt_index=torch.zeros((batch, anchors), dtype=torch.int64, device=device),
        )
