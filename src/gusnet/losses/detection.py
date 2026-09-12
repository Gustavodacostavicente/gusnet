# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""The full detection objective.

This is where assignment and the three loss terms meet. Given one forward pass
and the batch's ground truth it:

1. asks the assigner which grid points are responsible for which object;
2. scores every point's classification against the assigner's soft targets;
3. scores the assigned points' boxes, in pixels (IoU) and in bins (DFL).

Two details matter more than the weights.

**The classification term is normalised by the sum of the target scores, not
by the number of positives.** The denominator then reflects how much quality
was assigned, so a batch of hard images with weak targets produces a comparably
scaled loss to a batch of easy ones, and the gradient does not lurch when the
positive count jumps between batches.

**The box terms are weighted averages over the assigned points**, each weighted
by its own target score: a point the assigner considers a mediocre match pulls
the box proportionally less than one that nails it. They are normalised by
their own weights rather than by the global sum, which is what keeps them alive
at cold start -- see :meth:`DetectionLoss._box_terms`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gusnet.assign import Assignment, TaskAlignedAssigner, targets_to_batch
from gusnet.losses.components import DistributionFocalLoss, IoULoss, VarifocalLoss
from gusnet.ops.boxes import xyxy_to_ltrb

__all__ = ["DetectionLoss", "LossBreakdown"]


@dataclass
class LossBreakdown:
    """The scalar terms behind one training step, for logging.

    Attributes:
        total: the value that is backpropagated.
        cls: classification term, already weighted.
        box: IoU term, already weighted.
        dfl: distribution term, already weighted.
        num_foreground: assigned grid points in the batch.
    """

    total: Tensor
    cls: Tensor
    box: Tensor
    dfl: Tensor
    num_foreground: int

    def items(self) -> dict[str, float]:
        """Plain floats, ready to print or log."""
        return {
            "total": float(self.total),
            "cls": float(self.cls),
            "box": float(self.box),
            "dfl": float(self.dfl),
            "fg": float(self.num_foreground),
        }


class DetectionLoss(nn.Module):
    """Assign, then score, one batch of predictions.

    Args:
        num_classes: number of object classes.
        reg_max: bins per box edge, must match the head.
        assigner: label assignment strategy. Defaults to
            :class:`~gusnet.assign.tal.TaskAlignedAssigner`, which supplies a
            fixed number of positives from the first step; SimOTA's dynamic k
            collapses to one positive per object while the model is still
            untrained.
        cls_weight: weight of the classification term.
        box_weight: weight of the IoU term. Much larger than the others because
            ``1 - IoU`` is bounded by ~2 while the classification term sums
            over every class and every grid point.
        dfl_weight: weight of the distribution term.
        iou_kind: which IoU variant the box term uses.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        reg_max: int = 16,
        assigner: nn.Module | None = None,
        cls_weight: float = 0.5,
        box_weight: float = 7.5,
        dfl_weight: float = 1.5,
        iou_kind: str = "ciou",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.assigner = assigner or TaskAlignedAssigner(num_classes)

        self.cls_weight = float(cls_weight)
        self.box_weight = float(box_weight)
        self.dfl_weight = float(dfl_weight)

        self.classification = VarifocalLoss(reduction="none")
        self.box = IoULoss(kind=iou_kind, reduction="none")
        self.distribution = DistributionFocalLoss(self.reg_max, reduction="none")

    def forward(self, output: dict, targets: Tensor) -> tuple[Tensor, LossBreakdown]:
        """Score one forward pass.

        Args:
            output: a :class:`~gusnet.nn.head.DetectionOutput` from the model.
            targets: ``(N, 6)`` rows of ``[batch_index, class, x1, y1, x2, y2]``
                in the coordinates of the image fed to the model.

        Returns:
            ``(total_loss, breakdown)``.
        """
        cls_logits: Tensor = output["cls_logits"]
        reg_logits: Tensor = output["reg_logits"]
        pred_boxes: Tensor = output["boxes"]
        points: Tensor = output["points"]
        strides: Tensor = output["strides"]

        batch = cls_logits.shape[0]
        gt_labels, gt_boxes, gt_mask = targets_to_batch(targets.to(points.device), batch)

        assignment = self._assign(output, points, strides, gt_labels, gt_boxes, gt_mask)

        # One denominator for all three terms, so their relative weights mean
        # the same thing from batch to batch.
        target_scores = assignment.target_scores
        normaliser = target_scores.sum().clamp(min=1.0)

        cls_loss = self.classification(cls_logits, target_scores).sum() / normaliser

        fg_mask = assignment.fg_mask
        if bool(fg_mask.any()):
            box_loss, dfl_loss = self._box_terms(
                reg_logits, pred_boxes, points, strides, assignment
            )
        else:
            box_loss = cls_loss.new_zeros(())
            dfl_loss = cls_loss.new_zeros(())

        cls_term = cls_loss * self.cls_weight
        box_term = box_loss * self.box_weight
        dfl_term = dfl_loss * self.dfl_weight
        total = cls_term + box_term + dfl_term

        return total, LossBreakdown(
            total=total.detach(),
            cls=cls_term.detach(),
            box=box_term.detach(),
            dfl=dfl_term.detach(),
            num_foreground=assignment.num_foreground,
        )

    # ------------------------------------------------------------------ internals

    def _assign(
        self,
        output: dict,
        points: Tensor,
        strides: Tensor,
        gt_labels: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
    ) -> Assignment:
        """Run the assigner on detached predictions.

        Detaching is not an optimisation: assignment must not be something the
        model can influence to lower its loss. It decides what is being asked
        of the model, and that question has to be fixed before the answer is
        scored.
        """
        scores = output["scores"].detach()
        boxes = output["boxes"].detach()

        # SimOTA needs the per-point stride to size its centre region; TAL
        # does not take one.
        if isinstance(self.assigner, TaskAlignedAssigner):
            return self.assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask)
        return self.assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask, strides)

    def _box_terms(
        self,
        reg_logits: Tensor,
        pred_boxes: Tensor,
        points: Tensor,
        strides: Tensor,
        assignment: Assignment,
    ) -> tuple[Tensor, Tensor]:
        """IoU and distribution terms over the assigned points only.

        These are weighted *averages*, normalised by their own weights rather
        than by the global target-score sum. That choice is what makes cold
        start survivable: the task-aligned metric contains ``IoU ** beta``, so
        for an object nothing yet overlaps -- a small object in the first
        epochs -- every target score is around ``1e-8``. Dividing by the global
        sum would leave the box terms at that magnitude, the model would have
        no usable box gradient, and training would stall exactly where it most
        needs to move. A weighted average is scale-free and cannot collapse.
        """
        fg_mask = assignment.fg_mask

        # Each positive is weighted by how good a match the assigner thought it
        # was, so a marginal point does not pull the box as hard as a good one.
        weight = assignment.target_scores.sum(dim=-1)[fg_mask]
        weight_sum = weight.sum()
        if float(weight_sum) < 1e-6:
            # Nothing overlaps anything yet and every quality estimate is zero.
            # Weight the positives equally instead, so the assigner's choice of
            # points still produces a gradient.
            weight = torch.ones_like(weight)
            weight_sum = weight.sum()

        box_loss = self.box(pred_boxes[fg_mask], assignment.target_boxes[fg_mask])
        box_loss = (box_loss * weight).sum() / weight_sum

        # The head predicts distances in bin units, one bin per stride, so the
        # pixel targets have to be divided by the stride of their own level.
        target_ltrb = xyxy_to_ltrb(assignment.target_boxes, points.unsqueeze(0))
        target_ltrb = target_ltrb / strides.unsqueeze(0)
        # A distance beyond reg_max cannot be expressed; clamping just short of
        # the last bin keeps the upper interpolation neighbour in range.
        target_ltrb = target_ltrb.clamp(0, self.reg_max - 0.01)

        dfl_loss = self.distribution(reg_logits[fg_mask], target_ltrb[fg_mask])
        dfl_loss = (dfl_loss * weight).sum() / weight_sum

        return box_loss, dfl_loss

    @torch.no_grad()
    def assign_only(self, output: dict, targets: Tensor) -> Assignment:
        """Run just the assignment step, for diagnostics."""
        batch = output["cls_logits"].shape[0]
        gt_labels, gt_boxes, gt_mask = targets_to_batch(targets.to(output["points"].device), batch)
        return self._assign(
            output, output["points"], output["strides"], gt_labels, gt_boxes, gt_mask
        )
