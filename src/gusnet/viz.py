# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Drawing utilities.

Visual inspection is the cheapest bug detector in a detection pipeline: a box
that is off by a factor of two, a flip that did not move the labels with the
image, a mosaic tile pasted at the wrong offset — all of these are invisible in
a loss curve and obvious in a rendered image.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch

__all__ = [
    "class_color",
    "draw_boxes",
    "draw_points",
    "save_batch_preview",
    "tensor_to_image",
]


def class_color(index: int) -> tuple[int, int, int]:
    """A stable, well-spread RGB colour for a class index.

    The golden-ratio hue step keeps consecutive classes far apart in hue, so
    neighbouring ids never come out as near-identical colours.
    """
    hue = (index * 0.61803398875) % 1.0
    hsv = np.uint8([[[int(hue * 179), 220, 240]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    """``(3, H, W)`` float in ``[0, 1]`` -> ``(H, W, 3)`` uint8 RGB."""
    array = image.detach().cpu().float().clamp(0, 1).mul(255).byte()
    return array.permute(1, 2, 0).numpy()


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray | torch.Tensor,
    classes: np.ndarray | torch.Tensor | None = None,
    *,
    class_names: Sequence[str] | None = None,
    scores: np.ndarray | torch.Tensor | None = None,
    thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """Draw xyxy boxes onto a copy of ``image``.

    Args:
        image: ``(H, W, 3)`` uint8 RGB.
        boxes: ``(N, 4)`` xyxy in pixels.
        classes: ``(N,)`` class indices, used for colour and label.
        class_names: names indexed by class id.
        scores: ``(N,)`` confidences appended to the label.
        thickness: rectangle line width.
        font_scale: label text size.

    Returns:
        A new ``(H, W, 3)`` uint8 RGB array.
    """
    canvas = np.ascontiguousarray(image.copy())
    boxes = _to_numpy(boxes).reshape(-1, 4)
    if len(boxes) == 0:
        return canvas

    classes = _to_numpy(classes).astype(int) if classes is not None else np.zeros(len(boxes), int)
    scores = _to_numpy(scores) if scores is not None else None

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        cls = int(classes[i]) if i < len(classes) else 0
        color = class_color(cls)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = class_names[cls] if class_names is not None and cls < len(class_names) else str(cls)
        if scores is not None and i < len(scores):
            label = f"{label} {float(scores[i]):.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Put the caption above the box, or inside it when there is no room.
        top = y1 - text_h - baseline
        if top < 0:
            top = y1
        cv2.rectangle(
            canvas, (x1, top), (x1 + text_w + 2, top + text_h + baseline), color, thickness=-1
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 1, top + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def draw_points(
    image: np.ndarray,
    points: np.ndarray | torch.Tensor,
    classes: np.ndarray | torch.Tensor | None = None,
    *,
    radius: int = 3,
) -> np.ndarray:
    """Mark grid points on a copy of ``image``.

    Used to look at label assignment: which of the thousands of grid points a
    strategy made responsible for each object. A wrong assigner is obvious here
    -- positives scattered outside the objects, clustered on one edge, or far
    too few -- and invisible in any scalar metric.

    Args:
        image: ``(H, W, 3)`` uint8 RGB.
        points: ``(N, 2)`` ``(x, y)`` in pixels.
        classes: ``(N,)`` used to colour each point.
        radius: dot radius in pixels.

    Returns:
        A new ``(H, W, 3)`` uint8 RGB array.
    """
    canvas = np.ascontiguousarray(image.copy())
    coords = _to_numpy(points).reshape(-1, 2)
    if len(coords) == 0:
        return canvas

    labels = _to_numpy(classes).astype(int) if classes is not None else np.zeros(len(coords), int)
    for i, (x, y) in enumerate(coords):
        color = class_color(int(labels[i]) if i < len(labels) else 0)
        centre = (int(round(float(x))), int(round(float(y))))
        cv2.circle(canvas, centre, radius, color, thickness=-1)
        cv2.circle(canvas, centre, radius, (255, 255, 255), thickness=1)
    return canvas


def save_batch_preview(
    batch: dict,
    path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    max_images: int = 16,
) -> Path:
    """Render a collated batch as one image grid and write it to ``path``.

    Intended as a smoke test for the data pipeline: run it once after changing
    any augmentation and look at the result.
    """
    images = batch["images"][:max_images]
    per_image_boxes = batch["boxes"][:max_images]
    per_image_classes = batch["classes"][:max_images]

    tiles = [
        draw_boxes(tensor_to_image(img), box, cls, class_names=class_names)
        for img, box, cls in zip(images, per_image_boxes, per_image_classes, strict=True)
    ]

    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    height, width = tiles[0].shape[:2]
    grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * height : (r + 1) * height, c * width : (c + 1) * width] = tile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return path


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
