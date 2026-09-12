# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for suppression, mean average precision and the evaluation loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from gusnet.data import DetectionDataset
from gusnet.eval import (
    Detections,
    EvalConfig,
    GroundTruth,
    MeanAveragePrecision,
    average_precision,
    detections_to_rows,
    evaluate,
    non_max_suppression,
)

# ------------------------------------------------------------------------- NMS


def _output(boxes: list[list[float]], scores: list[list[float]]) -> dict:
    return {
        "boxes": torch.tensor([boxes], dtype=torch.float32),
        "scores": torch.tensor([scores], dtype=torch.float32),
    }


def test_nms_collapses_duplicates_of_the_same_object():
    """The head describes one object from a dozen points; NMS is what undoes that."""
    output = _output(
        [[10, 10, 50, 50], [12, 12, 52, 52], [11, 9, 49, 51], [200, 200, 240, 240]],
        [[0.9], [0.8], [0.7], [0.95]],
    )
    found = non_max_suppression(output, conf_threshold=0.1, iou_threshold=0.5)[0]

    assert len(found) == 2
    # Sorted by score: the distant object first, then the best of the cluster.
    assert found.scores.tolist() == pytest.approx([0.95, 0.9])


def test_nms_keeps_overlapping_objects_of_different_classes():
    """A car overlapping a person is two objects, not a duplicate."""
    output = _output([[10, 10, 50, 50], [11, 11, 51, 51]], [[0.9, 0.0], [0.0, 0.8]])
    assert len(non_max_suppression(output, conf_threshold=0.1, iou_threshold=0.5)[0]) == 2


def test_class_agnostic_nms_suppresses_across_classes():
    output = _output([[10, 10, 50, 50], [11, 11, 51, 51]], [[0.9, 0.0], [0.0, 0.8]])
    found = non_max_suppression(output, conf_threshold=0.1, iou_threshold=0.5, class_agnostic=True)[
        0
    ]
    assert len(found) == 1
    assert found.labels.tolist() == [0]


def test_nms_applies_the_confidence_threshold():
    output = _output([[10, 10, 50, 50], [200, 200, 240, 240]], [[0.9], [0.2]])
    assert len(non_max_suppression(output, conf_threshold=0.5)[0]) == 1


def test_nms_caps_the_detection_count():
    boxes = [[i * 100, 0, i * 100 + 40, 40] for i in range(10)]
    scores = [[0.9] for _ in range(10)]
    found = non_max_suppression(_output(boxes, scores), conf_threshold=0.1, max_det=3)[0]
    assert len(found) == 3


def test_nms_returns_empty_detections_when_nothing_passes():
    found = non_max_suppression(_output([[10, 10, 50, 50]], [[0.01]]), conf_threshold=0.5)[0]
    assert len(found) == 0
    assert found.boxes.shape == (0, 4)
    assert found.labels.dtype == torch.int64


def test_nms_without_multi_label_keeps_one_class_per_point():
    output = _output([[10, 10, 50, 50]], [[0.9, 0.8]])
    found = non_max_suppression(output, conf_threshold=0.1, iou_threshold=0.9)[0]
    assert len(found) == 1
    assert found.labels.tolist() == [0]


def test_nms_with_multi_label_emits_every_class_above_the_threshold():
    output = _output([[10, 10, 50, 50]], [[0.9, 0.8]])
    found = non_max_suppression(output, conf_threshold=0.1, iou_threshold=0.9, multi_label=True)[0]
    assert len(found) == 2
    assert sorted(found.labels.tolist()) == [0, 1]


def test_nms_handles_a_batch():
    output = {
        "boxes": torch.rand(3, 20, 4) * 10 + torch.tensor([0.0, 0.0, 50.0, 50.0]),
        "scores": torch.rand(3, 20, 2),
    }
    results = non_max_suppression(output, conf_threshold=0.5)
    assert len(results) == 3


def test_nms_rejects_the_wrong_shapes():
    with pytest.raises(ValueError, match=r"expected boxes \(B, A, 4\)"):
        non_max_suppression({"boxes": torch.zeros(4), "scores": torch.zeros(4)})


def test_detections_scale_back_to_the_original_image():
    detections = Detections(
        boxes=torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
    )
    scaled = detections.scale_to_original(0.5, (10.0, 0.0), (200, 100))
    assert scaled.boxes.tolist() == [[20.0, 40.0, 100.0, 120.0]]
    assert scaled.scores.tolist() == pytest.approx([0.9])


def test_detections_filter_by_score():
    detections = Detections(
        boxes=torch.zeros(3, 4),
        scores=torch.tensor([0.9, 0.4, 0.1]),
        labels=torch.zeros(3, dtype=torch.int64),
    )
    assert len(detections.filter(0.5)) == 1


# --------------------------------------------------------------- average precision


def test_average_precision_of_a_perfect_curve_is_one():
    recalls = np.linspace(0.1, 1.0, 10)
    precisions = np.ones(10)
    assert average_precision(recalls, precisions) == pytest.approx(1.0, abs=1e-6)


def test_average_precision_of_an_empty_curve_is_zero():
    assert average_precision(np.array([]), np.array([])) == 0.0


def test_average_precision_uses_the_maximum_precision_to_the_right():
    """The jagged raw curve is flattened, which is what makes AP stable."""
    recalls = np.array([0.5, 1.0])
    jagged = average_precision(recalls, np.array([0.5, 1.0]))
    smooth = average_precision(recalls, np.array([1.0, 1.0]))
    # A dip that later recovers is filled in, so both reach the same value.
    assert jagged == pytest.approx(smooth)


def test_average_precision_falls_when_precision_falls():
    recalls = np.linspace(0.1, 1.0, 10)
    assert average_precision(recalls, np.full(10, 0.5)) < average_precision(recalls, np.ones(10))


# ----------------------------------------------------------------------- mAP


def _detections(boxes, scores, labels) -> Detections:
    return Detections(
        boxes=torch.tensor(boxes, dtype=torch.float32),
        scores=torch.tensor(scores, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.int64),
    )


def _truth(boxes, labels) -> GroundTruth:
    return GroundTruth(
        boxes=torch.tensor(boxes, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.int64),
    )


def test_perfect_detections_score_one():
    metric = MeanAveragePrecision(num_classes=2)
    boxes = [[10, 10, 50, 50], [100, 100, 160, 160]]
    metric.update(_detections(boxes, [0.9, 0.8], [0, 1]), _truth(boxes, [0, 1]))

    result = metric.compute()
    assert result.map50_95 == pytest.approx(1.0, abs=1e-3)
    assert result.map50 == pytest.approx(1.0, abs=1e-3)
    assert result.precision == pytest.approx(1.0, abs=1e-3)
    assert result.recall == pytest.approx(1.0, abs=1e-3)


def test_detecting_nothing_scores_zero():
    metric = MeanAveragePrecision(num_classes=1)
    metric.update(_detections([], [], []), _truth([[10, 10, 50, 50]], [0]))
    result = metric.compute()
    assert result.map50_95 == 0.0
    assert result.num_objects == 1
    assert result.num_detections == 0


def test_the_right_box_with_the_wrong_class_is_a_miss():
    metric = MeanAveragePrecision(num_classes=2)
    box = [[10, 10, 50, 50]]
    metric.update(_detections(box, [0.9], [1]), _truth(box, [0]))
    assert metric.compute().map50 == 0.0


def test_a_loose_box_passes_at_iou_50_and_fails_at_iou_75():
    """This is precisely why mAP averages over ten thresholds."""
    metric = MeanAveragePrecision(num_classes=1)
    truth = [[0, 0, 100, 100]]
    # IoU here is 80*80 / (100*100 + 80*80 - 80*80) = 0.64
    metric.update(_detections([[0, 0, 80, 80]], [0.9], [0]), _truth(truth, [0]))

    result = metric.compute()
    assert result.map50 == pytest.approx(1.0, abs=1e-3)
    assert result.map75 == 0.0
    assert 0.0 < result.map50_95 < 1.0


def test_a_duplicate_detection_costs_precision():
    """Each object can be claimed once; the second box is a false positive."""
    metric = MeanAveragePrecision(num_classes=1)
    truth = [[10, 10, 50, 50]]
    metric.update(
        _detections([[10, 10, 50, 50], [11, 11, 51, 51]], [0.9, 0.8], [0, 0]),
        _truth(truth, [0]),
    )
    assert metric.compute().map50 < 1.0


def test_a_class_absent_from_the_data_is_not_scored():
    """An untested class must not drag the mean down with a zero."""
    metric = MeanAveragePrecision(num_classes=10)
    box = [[10, 10, 50, 50]]
    metric.update(_detections(box, [0.9], [3]), _truth(box, [3]))

    result = metric.compute()
    assert result.map50_95 == pytest.approx(1.0, abs=1e-3)
    assert set(result.ap_per_class) == {3}


def test_low_confidence_extras_hurt_less_than_high_confidence_ones():
    """Ranking matters: a confident mistake is worse than a hesitant one."""

    def score_with(extra_confidence: float) -> float:
        metric = MeanAveragePrecision(num_classes=1)
        detections = _detections(
            [[10, 10, 50, 50], [300, 300, 340, 340]], [0.9, extra_confidence], [0, 0]
        )
        metric.update(detections, _truth([[10, 10, 50, 50]], [0]))
        return metric.compute().map50

    assert score_with(0.1) > score_with(0.95)


def test_metric_accumulates_across_images():
    metric = MeanAveragePrecision(num_classes=1)
    box = [[10, 10, 50, 50]]
    for _ in range(4):
        metric.update(_detections(box, [0.9], [0]), _truth(box, [0]))

    result = metric.compute()
    assert result.num_images == 4
    assert result.num_objects == 4
    assert result.map50 == pytest.approx(1.0, abs=1e-3)


def test_metric_reset_clears_everything():
    metric = MeanAveragePrecision(num_classes=1)
    box = [[10, 10, 50, 50]]
    metric.update(_detections(box, [0.9], [0]), _truth(box, [0]))
    metric.reset()
    assert metric.compute().num_images == 0


def test_metric_result_formats_a_table():
    metric = MeanAveragePrecision(num_classes=2)
    box = [[10, 10, 50, 50]]
    metric.update(_detections(box, [0.9], [1]), _truth(box, [1]))
    text = metric.compute().format(["cat", "dog"])
    assert "mAP50-95" in text
    assert "dog" in text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_classes": 0}, "num_classes must be at least 1"),
        ({"num_classes": 2, "iou_thresholds": ()}, "at least one IoU threshold"),
    ],
)
def test_metric_validates_its_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MeanAveragePrecision(**kwargs)


def test_detections_to_rows_uses_coco_xywh():
    rows = detections_to_rows(_detections([[10, 20, 40, 60]], [0.8], [2]), image_id=7)
    assert len(rows) == 1
    assert rows[0]["image_id"] == 7
    assert rows[0]["category_id"] == 2
    assert rows[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]  # corners -> x, y, w, h
    assert rows[0]["score"] == pytest.approx(0.8)


# ------------------------------------------------------------------- the loop


class _OracleModel(nn.Module):
    """A model that reports the ground truth exactly.

    It exists to test the evaluator itself: if the coordinate bookkeeping
    between letterboxed predictions and original-image ground truth is wrong
    anywhere, a perfect detector will not score 1.0 and this will say so.
    """

    def __init__(self, dataset: DetectionDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.cursor = 0

    def forward(self, images: torch.Tensor) -> dict:
        batch, _, height, width = images.shape
        max_objects = max(
            (len(self.dataset[self.cursor + i]["boxes"]) for i in range(batch)), default=1
        )
        anchors = max(max_objects, 1)

        boxes = torch.zeros(batch, anchors, 4)
        scores = torch.zeros(batch, anchors, self.num_classes)

        for offset in range(batch):
            item = self.dataset[self.cursor + offset]
            for index, (box, label) in enumerate(zip(item["boxes"], item["classes"], strict=True)):
                boxes[offset, index] = box
                scores[offset, index, int(label)] = 0.99

        self.cursor += batch
        return {"boxes": boxes, "scores": scores}


def test_evaluate_scores_a_perfect_detector_at_one(folder_dataset: Path):
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=128, augment=False)
    config = EvalConfig(batch_size=2, img_size=128, device="cpu", verbose=False)

    result = evaluate(_OracleModel(dataset), dataset, config)

    assert result.num_images == len(dataset)
    assert result.num_objects == sum(len(a.boxes) for a in dataset.annotations)
    assert result.map50_95 == pytest.approx(1.0, abs=1e-3)


def test_evaluate_runs_an_untrained_model_without_error(folder_dataset: Path):
    from gusnet.models import GUSNet

    torch.manual_seed(0)
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=64, augment=False)
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)
    config = EvalConfig(batch_size=2, img_size=64, device="cpu", verbose=False, max_det=50)

    result = evaluate(model, dataset, config)

    assert result.num_images == len(dataset)
    assert 0.0 <= result.map50_95 <= 1.0


def test_evaluate_turns_augmentation_off(folder_dataset: Path):
    """Measuring a model on mosaics measures something never asked of it."""
    from gusnet.models import GUSNet

    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=64, augment=True)
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)

    evaluate(model, dataset, EvalConfig(batch_size=2, device="cpu", verbose=False, max_det=10))
    assert dataset.augment is False
