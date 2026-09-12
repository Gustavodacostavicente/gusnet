# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Mean average precision, implemented from the definition.

mAP is the standard detection metric and it is worth understanding rather than
importing, because almost every surprising number a detector produces is
explained by one of its details.

**What it measures.** For one class at one IoU threshold: sort every prediction
in the dataset by confidence, walk down the list, and mark each as a true
positive if it matches a ground-truth box that nothing better has already
claimed. That gives a precision/recall curve — as you accept lower-confidence
predictions, recall rises and precision falls. Average precision is the area
under it.

**Why the curve is flattened first.** The raw curve is jagged: one lucky
detection can bump precision back up. AP uses the *maximum precision at or
beyond each recall level*, which answers the question that actually matters —
"if I need this much recall, what is the best precision I can have?" — and
makes the metric stable.

**Why ten IoU thresholds.** AP at IoU 0.5 barely rewards good localisation: a
box can be visibly wrong and still count. COCO-style mAP averages AP over
IoU 0.50 to 0.95 in steps of 0.05, so a model that puts boxes in almost the
right place scores well below one that puts them exactly right. `map50` is
reported too, because it is what older literature quotes.

Matching rules follow the COCO convention: each ground-truth box can be claimed
by at most one prediction (the highest-confidence one that overlaps it enough),
and unmatched predictions are false positives. A class with no ground truth
anywhere in the dataset contributes no AP at all rather than a zero, which would
otherwise drag the mean down with a class the data never tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from gusnet.eval.nms import Detections
from gusnet.ops.boxes import box_iou

__all__ = ["GroundTruth", "MeanAveragePrecision", "MetricResult", "average_precision"]

#: COCO's ten thresholds: 0.50, 0.55, ..., 0.95.
DEFAULT_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))

#: COCO interpolates the precision/recall curve at 101 evenly spaced recalls.
RECALL_POINTS = 101


@dataclass
class GroundTruth:
    """The objects actually present in one image.

    Attributes:
        boxes: ``(N, 4)`` xyxy in the same coordinate space as the predictions.
        labels: ``(N,)`` int64 class indices.
    """

    boxes: Tensor
    labels: Tensor

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


@dataclass
class MetricResult:
    """The outcome of an evaluation.

    Attributes:
        map50_95: mAP averaged over IoU 0.50:0.95 — the headline number.
        map50: mAP at IoU 0.50, the looser, older convention.
        map75: mAP at IoU 0.75, a strict-localisation view.
        precision: precision at the best-F1 point of the curve, IoU 0.50.
        recall: recall at that same point.
        ap_per_class: ``{class index: AP averaged over thresholds}``.
        num_images: images evaluated.
        num_objects: ground-truth objects seen.
        num_detections: detections considered.
    """

    map50_95: float = 0.0
    map50: float = 0.0
    map75: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    ap_per_class: dict[int, float] = field(default_factory=dict)
    num_images: int = 0
    num_objects: int = 0
    num_detections: int = 0

    def items(self) -> dict[str, float]:
        """The scalar metrics, ready to log."""
        return {
            "mAP50-95": self.map50_95,
            "mAP50": self.map50,
            "mAP75": self.map75,
            "precision": self.precision,
            "recall": self.recall,
        }

    def format(self, class_names: list[str] | None = None) -> str:
        """A human-readable summary table."""
        lines = [
            f"images {self.num_images}  objects {self.num_objects}  "
            f"detections {self.num_detections}",
            f"mAP50-95 {self.map50_95:.4f}   mAP50 {self.map50:.4f}   mAP75 {self.map75:.4f}",
            f"precision {self.precision:.4f}   recall {self.recall:.4f}",
        ]
        if self.ap_per_class:
            lines.append("")
            lines.append(f"{'class':<24} {'AP50-95':>8}")
            for index, value in sorted(self.ap_per_class.items()):
                name = (
                    class_names[index]
                    if class_names is not None and index < len(class_names)
                    else str(index)
                )
                lines.append(f"{name:<24} {value:>8.4f}")
        return "\n".join(lines)


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Area under a precision/recall curve, COCO style.

    The curve is first made monotonically non-increasing in precision — each
    point takes the maximum precision achieved at that recall or beyond — and
    then sampled at 101 evenly spaced recall levels. Sampling rather than
    integrating exactly is what makes the number comparable across datasets of
    different sizes.

    Args:
        recalls: ``(N,)`` non-decreasing recall values.
        precisions: ``(N,)`` precision at each of those recalls.

    Returns:
        The average precision in ``[0, 1]``.
    """
    if len(recalls) == 0:
        return 0.0

    # Sentinels so the curve starts at recall 0 and ends past the last point.
    recalls = np.concatenate(([0.0], recalls, [recalls[-1] + 1e-3]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically non-increasing, right to left.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    points = np.linspace(0.0, 1.0, RECALL_POINTS)
    interpolated = np.interp(points, recalls, precisions, left=precisions[0], right=0.0)
    return float(interpolated.mean())


class MeanAveragePrecision:
    """Accumulate detections and ground truth, then compute mAP.

    Usage is the usual accumulate-then-compute::

        metric = MeanAveragePrecision(num_classes=80)
        for images, truth in loader:
            metric.update(model_detections, truth)
        result = metric.compute()

    Args:
        num_classes: how many classes exist.
        iou_thresholds: the thresholds to average over. Defaults to COCO's ten.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if not iou_thresholds:
            raise ValueError("at least one IoU threshold is required")

        self.num_classes = int(num_classes)
        self.iou_thresholds = tuple(float(t) for t in iou_thresholds)
        self.reset()

    def reset(self) -> None:
        """Forget everything accumulated so far."""
        # One row per detection: [class, score, matched at threshold 0, 1, ...]
        self._matches: list[np.ndarray] = []
        self._scores: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._object_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.num_images = 0
        self.num_objects = 0
        self.num_detections = 0

    @torch.no_grad()
    def update(self, detections: Detections, truth: GroundTruth) -> None:
        """Add one image's predictions and ground truth.

        Args:
            detections: what the model reported, in any coordinate space.
            truth: the objects present, **in the same space**.
        """
        self.num_images += 1
        self.num_objects += len(truth)
        self.num_detections += len(detections)

        for label in truth.labels.tolist():
            if 0 <= int(label) < self.num_classes:
                self._object_counts[int(label)] += 1

        if len(detections) == 0:
            return

        matched = self._match(detections, truth)
        self._matches.append(matched)
        self._scores.append(detections.scores.detach().cpu().numpy())
        self._labels.append(detections.labels.detach().cpu().numpy())

    def compute(self) -> MetricResult:
        """Reduce everything accumulated into a :class:`MetricResult`."""
        result = MetricResult(
            num_images=self.num_images,
            num_objects=self.num_objects,
            num_detections=self.num_detections,
        )
        if not self._matches:
            return result

        matches = np.concatenate(self._matches, axis=0)  # (D, T) bool
        scores = np.concatenate(self._scores, axis=0)  # (D,)
        labels = np.concatenate(self._labels, axis=0)  # (D,)

        order = np.argsort(-scores)
        matches, scores, labels = matches[order], scores[order], labels[order]

        num_thresholds = len(self.iou_thresholds)
        ap = np.zeros((self.num_classes, num_thresholds))
        # Only classes the dataset actually contains are scored; a class with no
        # ground truth is untested, not failed.
        present = self._object_counts > 0

        best_precision = 0.0
        best_recall = 0.0
        best_f1 = -1.0

        for class_index in np.nonzero(present)[0]:
            selected = labels == class_index
            total = int(self._object_counts[class_index])
            if not selected.any():
                continue  # the class exists but nothing was detected for it

            class_matches = matches[selected]
            for threshold_index in range(num_thresholds):
                true_positives = np.cumsum(class_matches[:, threshold_index])
                false_positives = np.cumsum(~class_matches[:, threshold_index])

                recalls = true_positives / total
                precisions = true_positives / np.maximum(
                    true_positives + false_positives, np.finfo(np.float64).eps
                )
                ap[class_index, threshold_index] = average_precision(recalls, precisions)

                if threshold_index == 0 and len(recalls):
                    f1 = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
                    peak = int(np.argmax(f1))
                    if f1[peak] > best_f1:
                        best_f1 = float(f1[peak])
                        best_precision = float(precisions[peak])
                        best_recall = float(recalls[peak])

        scored = ap[present]
        result.map50_95 = float(scored.mean()) if scored.size else 0.0
        result.map50 = float(scored[:, 0].mean()) if scored.size else 0.0
        result.map75 = float(self._at_threshold(scored, 0.75))
        result.precision = best_precision
        result.recall = best_recall
        result.ap_per_class = {
            int(index): float(ap[index].mean()) for index in np.nonzero(present)[0]
        }
        return result

    # ------------------------------------------------------------------ internals

    def _at_threshold(self, scored: np.ndarray, threshold: float) -> float:
        """Mean AP at one specific IoU threshold, if it is being tracked."""
        if not scored.size or threshold not in self.iou_thresholds:
            return 0.0
        return float(scored[:, self.iou_thresholds.index(threshold)].mean())

    def _match(self, detections: Detections, truth: GroundTruth) -> np.ndarray:
        """Decide, per detection and per threshold, whether it is a true positive.

        Returns:
            ``(D, T)`` bool: detection ``d`` matched something at threshold
            ``t``. Detections arrive already sorted by descending score, which
            is what makes the greedy assignment correct: the most confident
            prediction gets first claim on each object.
        """
        num_detections = len(detections)
        num_thresholds = len(self.iou_thresholds)
        matched = np.zeros((num_detections, num_thresholds), dtype=bool)

        if len(truth) == 0:
            return matched  # every detection is a false positive

        overlaps = box_iou(detections.boxes.float(), truth.boxes.float()).cpu().numpy()
        det_labels = detections.labels.cpu().numpy()
        gt_labels = truth.labels.cpu().numpy()

        # A detection can only claim an object of its own class.
        same_class = det_labels[:, None] == gt_labels[None, :]
        overlaps = np.where(same_class, overlaps, -1.0)

        for threshold_index, threshold in enumerate(self.iou_thresholds):
            claimed = np.zeros(len(truth), dtype=bool)
            for detection_index in range(num_detections):
                candidates = overlaps[detection_index].copy()
                candidates[claimed] = -1.0
                best = int(np.argmax(candidates))
                if candidates[best] >= threshold:
                    matched[detection_index, threshold_index] = True
                    claimed[best] = True

        return matched
