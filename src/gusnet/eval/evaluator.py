# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Running a model over a dataset and scoring what it finds.

Three details in here are easy to get wrong and expensive to get wrong.

**The confidence threshold must be tiny.** mAP integrates the whole
precision/recall curve, so cutting predictions at 0.25 truncates the curve and
under-reports the metric — often by several points. Evaluation uses 0.001; the
0.25 you would show a person belongs to inference, not to measurement.

**The NMS threshold is looser than at inference.** Suppressing aggressively
removes duplicates but also removes recall, and recall is half of what is being
measured. 0.7 is the usual evaluation setting against 0.45 for display.

**Predictions and ground truth must live in the same coordinate space.** Both
are mapped out of letterboxed space back to the original image before matching.
IoU happens to be invariant under the shared scale-and-pad, so this changes no
number — but it makes every box that is written out or drawn meaningful, and it
removes a whole class of silent mismatches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from gusnet.data import DetectionDataset, build_dataloader
from gusnet.eval.metrics import GroundTruth, MeanAveragePrecision, MetricResult
from gusnet.eval.nms import Detections, non_max_suppression
from gusnet.ops.boxes import scale_boxes

__all__ = ["EvalConfig", "evaluate"]


@dataclass
class EvalConfig:
    """How to run an evaluation.

    Attributes:
        batch_size: images per forward pass.
        img_size: evaluation resolution. Should match training.
        conf_threshold: minimum confidence. Keep it near zero for mAP.
        iou_threshold: NMS overlap threshold.
        max_det: cap on detections per image.
        workers: dataloader workers.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ...
        half: run in float16 on CUDA. Roughly doubles throughput and moves mAP
            by less than 0.1 point.
        verbose: print a summary when finished.
    """

    batch_size: int = 16
    img_size: int = 640
    conf_threshold: float = 0.001
    iou_threshold: float = 0.7
    max_det: int = 300
    workers: int = 0
    device: str = "auto"
    half: bool = False
    verbose: bool = True

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: DetectionDataset,
    config: EvalConfig | None = None,
) -> MetricResult:
    """Score a model on a dataset.

    Args:
        model: the detector. Put in eval mode automatically; for a trained run
            this should be the EMA copy.
        dataset: the evaluation set. Augmentation is forced off — measuring a
            model on mosaics measures something that will never be asked of it.
        config: evaluation settings.

    Returns:
        A :class:`~gusnet.eval.metrics.MetricResult`.
    """
    config = config or EvalConfig()
    device = config.resolved_device()

    # Evaluating with augmentation on is a silent, serious mistake, so it is
    # switched off here rather than left to the caller to remember.
    dataset.augment = False

    model = model.to(device).eval()
    use_half = bool(config.half and device.type == "cuda")
    if use_half:
        model = model.half()

    loader = build_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        drop_last=False,
    )

    metric = MeanAveragePrecision(dataset.num_classes)
    started = time.perf_counter()

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        if use_half:
            images = images.half()

        output = model(images)
        detections = non_max_suppression(
            output,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            max_det=config.max_det,
        )

        for index, found in enumerate(detections):
            ratio = batch["ratios"][index]
            pad = batch["pads"][index]
            orig_shape = batch["orig_shapes"][index]

            metric.update(
                found.to("cpu").scale_to_original(ratio, pad, orig_shape),
                _truth_in_original_space(batch, index, ratio, pad, orig_shape),
            )

    result = metric.compute()
    elapsed = time.perf_counter() - started

    if config.verbose:
        print(result.format(dataset.class_names))
        per_image = elapsed / max(result.num_images, 1) * 1000
        print(f"\n{elapsed:.1f}s total, {per_image:.1f} ms/image on {device}")

    return result


def _truth_in_original_space(
    batch: dict,
    index: int,
    ratio: float,
    pad: tuple[float, float],
    orig_shape: tuple[int, int],
) -> GroundTruth:
    """Undo the letterbox on one image's ground truth."""
    boxes = batch["boxes"][index].float()
    if len(boxes):
        boxes = scale_boxes(boxes, ratio, pad, orig_shape)
    return GroundTruth(boxes=boxes, labels=batch["classes"][index])


def detections_to_rows(detections: Detections, image_id: int) -> list[dict]:
    """Serialise detections into COCO-style result records.

    COCO expects ``[x, y, width, height]``, not corners — a conversion worth
    doing in one place rather than at every call site that writes results out.
    """
    rows: list[dict] = []
    for box, score, label in zip(
        detections.boxes.tolist(),
        detections.scores.tolist(),
        detections.labels.tolist(),
        strict=True,
    ):
        x1, y1, x2, y2 = box
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return rows
