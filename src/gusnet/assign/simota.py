# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""SimOTA label assignment.

Assignment can be read as a transport problem: each object has a supply of
labels to hand out, each grid point has a demand for one, and every pairing has
a cost. Solving it exactly (Sinkhorn, as in OTA) is expensive, so SimOTA keeps
the two ideas that carry the benefit and drops the solver:

* **dynamic k** -- an object's supply is not a constant. It is estimated from
  the IoUs of its best candidates, so a large, clearly visible object gets many
  positives and a small or occluded one gets few. A fixed k has to be wrong for
  one of those two cases.
* **a cost that combines both tasks** -- classification cost plus IoU cost, with
  candidates outside the object's centre region priced out entirely.

Reimplemented from Ge et al., *OTA* (arXiv:2103.14259) and the SimOTA
simplification described in *YOLOX* (arXiv:2107.08430).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gusnet.assign.utils import Assignment, points_in_boxes, points_in_centers
from gusnet.ops.boxes import box_iou

__all__ = ["SimOTAAssigner"]

INFEASIBLE = 1e5  # cost that removes a candidate without special-casing it


class SimOTAAssigner(nn.Module):
    """Assign grid points to objects by solving a cheap matching per image.

    Args:
        num_classes: number of object classes.
        center_radius: half-side of the centre region, in strides.
        iou_weight: weight of the IoU term relative to the classification term.
        candidate_topk: how many IoUs are summed to estimate an object's
            dynamic k.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        center_radius: float = 2.5,
        iou_weight: float = 3.0,
        candidate_topk: int = 10,
    ) -> None:
        super().__init__()
        if candidate_topk < 1:
            raise ValueError("candidate_topk must be at least 1")
        self.num_classes = int(num_classes)
        self.center_radius = float(center_radius)
        self.iou_weight = float(iou_weight)
        self.candidate_topk = int(candidate_topk)

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        points: Tensor,
        gt_labels: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
        strides: Tensor,
    ) -> Assignment:
        """Assign a batch, one image at a time.

        Args:
            pred_scores: ``(B, A, C)`` confidences in ``[0, 1]``.
            pred_boxes: ``(B, A, 4)`` predicted xyxy in image pixels.
            points: ``(A, 2)`` grid points in image pixels.
            gt_labels: ``(B, M)`` class indices.
            gt_boxes: ``(B, M, 4)`` xyxy in image pixels.
            gt_mask: ``(B, M)`` bool marking real objects among the padding.
            strides: ``(A, 1)`` stride of each grid point.

        Returns:
            An :class:`~gusnet.assign.utils.Assignment`.
        """
        batch, anchors, _ = pred_scores.shape
        device = pred_scores.device
        dtype = pred_scores.dtype
        gt_mask = gt_mask.bool()

        fg_mask = torch.zeros((batch, anchors), dtype=torch.bool, device=device)
        target_labels = torch.zeros((batch, anchors), dtype=torch.int64, device=device)
        target_boxes = torch.zeros((batch, anchors, 4), dtype=dtype, device=device)
        target_scores = torch.zeros((batch, anchors, self.num_classes), dtype=dtype, device=device)
        target_gt_index = torch.zeros((batch, anchors), dtype=torch.int64, device=device)

        for image in range(batch):
            count = int(gt_mask[image].sum())
            if count == 0:
                continue

            matched, gt_index, ious = self._assign_one(
                pred_scores[image],
                pred_boxes[image],
                points,
                gt_labels[image, :count],
                gt_boxes[image, :count],
                strides,
            )
            if matched.numel() == 0:
                continue

            labels = gt_labels[image, :count][gt_index]
            fg_mask[image, matched] = True
            target_labels[image, matched] = labels
            target_boxes[image, matched] = gt_boxes[image, :count][gt_index]
            target_gt_index[image, matched] = gt_index
            # The classification target carries the achieved IoU, so confidence
            # and localisation quality stay tied together.
            target_scores[image, matched, labels] = ious.to(dtype)

        return Assignment(
            fg_mask=fg_mask,
            target_labels=target_labels,
            target_boxes=target_boxes,
            target_scores=target_scores,
            target_gt_index=target_gt_index,
        )

    # ------------------------------------------------------------------ internals

    def _assign_one(
        self,
        scores: Tensor,
        boxes: Tensor,
        points: Tensor,
        gt_labels: Tensor,
        gt_boxes: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Match one image.

        Returns:
            ``(anchor_index, gt_index, iou)``: the indices of the assigned grid
            points, which object each got, and the IoU of that pairing.
        """
        empty = torch.zeros(0, dtype=torch.int64, device=scores.device)
        gt_boxes = gt_boxes.unsqueeze(0)

        inside_box = points_in_boxes(points, gt_boxes)[0]  # (M, A)
        inside_center = points_in_centers(points, gt_boxes, strides, radius=self.center_radius)[0]

        # A point is worth scoring if it is inside the object or near its
        # centre; the intersection of the two is what the cost prefers.
        candidate = (inside_box | inside_center).any(dim=0)  # (A,)
        if not bool(candidate.any()):
            return empty, empty, scores.new_zeros(0)

        candidate_index = candidate.nonzero(as_tuple=False).squeeze(-1)
        preferred = (inside_box & inside_center)[:, candidate_index]  # (M, Ac)

        iou = box_iou(gt_boxes[0], boxes[candidate_index]).clamp(min=0)  # (M, Ac)
        cost = self._cost(scores[candidate_index], gt_labels, iou, preferred)

        matched_local, gt_index, matched_iou = self._dynamic_k_matching(cost, iou)
        if matched_local.numel() == 0:
            return empty, empty, scores.new_zeros(0)

        return candidate_index[matched_local], gt_index, matched_iou

    def _cost(self, scores: Tensor, gt_labels: Tensor, iou: Tensor, preferred: Tensor) -> Tensor:
        """``(M, Ac)`` cost of pairing each object with each candidate."""
        num_gt = gt_labels.shape[0]
        num_candidates = scores.shape[0]

        # -log(iou) grows steeply as the overlap approaches zero, which is what
        # makes a badly localised candidate genuinely expensive rather than
        # merely slightly worse.
        iou_cost = -torch.log(iou + 1e-8)

        targets = F.one_hot(gt_labels.clamp(min=0), self.num_classes).to(scores.dtype)
        targets = targets.unsqueeze(1).expand(num_gt, num_candidates, self.num_classes)

        # The square root softens the classification term: early in training
        # every score is near zero and an unsoftened BCE would swamp the IoU
        # term with noise from a head that has not learned anything yet.
        probabilities = scores.unsqueeze(0).expand_as(targets).sqrt().clamp(1e-6, 1 - 1e-6)
        cls_cost = F.binary_cross_entropy(probabilities, targets, reduction="none").sum(-1)

        return cls_cost + self.iou_weight * iou_cost + INFEASIBLE * (~preferred)

    def _dynamic_k_matching(self, cost: Tensor, iou: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Pick each object's k cheapest candidates, with k estimated from IoU."""
        num_gt, num_candidates = cost.shape
        matching = torch.zeros_like(cost, dtype=torch.bool)

        # k = how much total overlap the object's best candidates already
        # achieve. Ten candidates at IoU 0.9 means the object is easy and can
        # support nine positives; ten at IoU 0.1 means it can support one.
        topk = min(self.candidate_topk, num_candidates)
        top_ious, _ = iou.topk(topk, dim=1)
        dynamic_k = top_ious.sum(dim=1).int().clamp(min=1, max=num_candidates)

        for gt_index in range(num_gt):
            k = int(dynamic_k[gt_index])
            _, cheapest = cost[gt_index].topk(k, largest=False)
            matching[gt_index, cheapest] = True

        # One point, one object: contested points go to the cheaper pairing.
        contested = matching.sum(dim=0) > 1
        if bool(contested.any()):
            winner = cost[:, contested].argmin(dim=0)
            matching[:, contested] = False
            matching[winner, contested.nonzero(as_tuple=False).squeeze(-1)] = True

        matched = matching.any(dim=0)
        matched_index = matched.nonzero(as_tuple=False).squeeze(-1)
        if matched_index.numel() == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=cost.device)
            return empty, empty, cost.new_zeros(0)

        gt_index = matching[:, matched_index].to(torch.uint8).argmax(dim=0)
        matched_iou = iou[gt_index, matched_index]
        return matched_index, gt_index, matched_iou
