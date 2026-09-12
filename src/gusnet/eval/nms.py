# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Turning 8400 dense predictions into a list of objects.

The head answers the same question at every grid point, so a single object is
described by every point the assigner would have made responsible for it --
typically a dozen overlapping boxes with similar scores. Non-maximum
suppression is what collapses that back into one detection per object: sort by
confidence, keep the best box, discard everything that overlaps it too much,
repeat.

Three thresholds control the outcome, and they are not interchangeable:

``conf_threshold``
    what counts as a detection at all. Low (``0.001``) when measuring mAP,
    because the metric integrates the whole precision/recall curve and needs
    the low-confidence tail; high (``0.25``) when showing results to a person,
    who does not want to look at 300 boxes.
``iou_threshold``
    how much overlap means "the same object". Too low and two genuinely
    adjacent objects collapse into one; too high and every object keeps a
    cluster of duplicates.
``max_det``
    a hard cap, mostly a guard against a pathological image producing tens of
    thousands of boxes.

NMS is per class by default: a car overlapping a person is two objects, not a
duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.ops import batched_nms, nms

from gusnet.ops.boxes import scale_boxes

__all__ = ["Detections", "non_max_suppression"]


@dataclass
class Detections:
    """What a model found in one image.

    Attributes:
        boxes: ``(N, 4)`` xyxy in pixels, sorted by descending score.
        scores: ``(N,)`` confidences in ``[0, 1]``.
        labels: ``(N,)`` int64 class indices.
    """

    boxes: Tensor
    scores: Tensor
    labels: Tensor

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def to(self, device: torch.device | str) -> Detections:
        return Detections(
            boxes=self.boxes.to(device),
            scores=self.scores.to(device),
            labels=self.labels.to(device),
        )

    def scale_to_original(
        self,
        ratio: float | tuple[float, float],
        pad: tuple[float, float],
        orig_shape: tuple[int, int],
    ) -> Detections:
        """Map the boxes out of letterboxed space, back to the source image."""
        return Detections(
            boxes=scale_boxes(self.boxes, ratio, pad, orig_shape),
            scores=self.scores,
            labels=self.labels,
        )

    def filter(self, min_score: float) -> Detections:
        """Keep only detections above ``min_score``."""
        keep = self.scores >= min_score
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep])


def non_max_suppression(
    output: dict,
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    max_nms: int = 30000,
    class_agnostic: bool = False,
    multi_label: bool = False,
) -> list[Detections]:
    """Reduce a batch of dense predictions to per-image detections.

    Args:
        output: a :class:`~gusnet.nn.head.DetectionOutput`, or any mapping with
            ``boxes`` ``(B, A, 4)`` and ``scores`` ``(B, A, C)``.
        conf_threshold: minimum confidence to consider a prediction at all.
        iou_threshold: overlap above which a lower-scoring box is suppressed.
        max_det: maximum detections returned per image.
        max_nms: cap on how many candidates enter the suppression itself.
            Sorting and suppressing every one of 8400 × C candidates on a
            pathological image is slow and never changes the top ``max_det``.
        class_agnostic: suppress across classes as well. Useful when the
            classes are mutually exclusive views of one object and a duplicate
            under a different label is still a duplicate.
        multi_label: let one grid point emit several classes. Off by default,
            which keeps one box per point at its best class -- the right choice
            unless the dataset genuinely has overlapping labels.

    Returns:
        One :class:`Detections` per image, sorted by descending score.
    """
    boxes: Tensor = output["boxes"]
    scores: Tensor = output["scores"]
    if boxes.ndim != 3 or scores.ndim != 3:
        raise ValueError(
            f"expected boxes (B, A, 4) and scores (B, A, C), got "
            f"{tuple(boxes.shape)} and {tuple(scores.shape)}"
        )

    results: list[Detections] = []
    for image_boxes, image_scores in zip(boxes, scores, strict=True):
        results.append(
            _suppress_one(
                image_boxes,
                image_scores,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_det=max_det,
                max_nms=max_nms,
                class_agnostic=class_agnostic,
                multi_label=multi_label,
            )
        )
    return results


def _suppress_one(
    boxes: Tensor,
    scores: Tensor,
    *,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    max_nms: int,
    class_agnostic: bool,
    multi_label: bool,
) -> Detections:
    """NMS for a single image. ``boxes`` is ``(A, 4)``, ``scores`` is ``(A, C)``."""
    if multi_label:
        # Every (point, class) pair above the threshold becomes a candidate.
        point_index, class_index = (scores > conf_threshold).nonzero(as_tuple=True)
        candidate_boxes = boxes[point_index]
        candidate_scores = scores[point_index, class_index]
        candidate_labels = class_index
    else:
        best_scores, best_labels = scores.max(dim=-1)
        keep = best_scores > conf_threshold
        candidate_boxes = boxes[keep]
        candidate_scores = best_scores[keep]
        candidate_labels = best_labels[keep]

    if candidate_boxes.numel() == 0:
        return _empty(boxes)

    if candidate_scores.shape[0] > max_nms:
        top = candidate_scores.topk(max_nms).indices
        candidate_boxes = candidate_boxes[top]
        candidate_scores = candidate_scores[top]
        candidate_labels = candidate_labels[top]

    if class_agnostic:
        kept = nms(candidate_boxes, candidate_scores, iou_threshold)
    else:
        # batched_nms offsets each class into its own coordinate band, so boxes
        # of different classes can never suppress one another.
        kept = batched_nms(candidate_boxes, candidate_scores, candidate_labels, iou_threshold)

    kept = kept[:max_det]
    return Detections(
        boxes=candidate_boxes[kept],
        scores=candidate_scores[kept],
        labels=candidate_labels[kept].to(torch.int64),
    )


def _empty(reference: Tensor) -> Detections:
    device = reference.device
    return Detections(
        boxes=torch.zeros((0, 4), dtype=reference.dtype, device=device),
        scores=torch.zeros((0,), dtype=reference.dtype, device=device),
        labels=torch.zeros((0,), dtype=torch.int64, device=device),
    )
