# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Image and box augmentations.

All functions operate on plain numpy arrays and share one convention:

* ``image`` is ``(H, W, 3)`` uint8 RGB,
* ``boxes`` is ``(N, 4)`` float32 **xyxy in absolute pixels**,
* ``classes`` is ``(N,)`` int64.

Boxes are always transformed together with the image, and degenerate boxes are
dropped by :func:`filter_boxes` after any geometric operation.

The augmentations implemented here (mosaic, mixup, HSV jitter, random affine)
are described in the YOLOv4 paper (Bochkovskiy et al., arXiv:2004.10934) and in
the Bag-of-Freebies literature; they are reimplemented from those descriptions.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

__all__ = [
    "filter_boxes",
    "hsv_augment",
    "mixup",
    "mosaic4",
    "random_affine",
    "random_hflip",
]


def filter_boxes(
    boxes: np.ndarray,
    classes: np.ndarray,
    *,
    min_size: float = 2.0,
    min_area_ratio: float = 0.1,
    original: np.ndarray | None = None,
    max_aspect: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop boxes that a geometric transform has destroyed.

    A box survives if it is still at least ``min_size`` pixels wide and tall,
    its aspect ratio is sane, and -- when ``original`` is given -- it kept at
    least ``min_area_ratio`` of its area (i.e. it was not mostly cropped away).

    Args:
        boxes: ``(N, 4)`` transformed boxes.
        classes: ``(N,)`` labels aligned with ``boxes``.
        min_size: minimum width and height in pixels.
        min_area_ratio: minimum surviving fraction of the original area.
        original: ``(N, 4)`` boxes before the transform.
        max_aspect: maximum allowed width/height ratio (either direction).

    Returns:
        The filtered ``(boxes, classes)``.
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 4).astype(np.float32), classes.reshape(0).astype(np.int64)

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w > min_size) & (h > min_size)
    keep &= np.maximum(w / (h + 1e-9), h / (w + 1e-9)) < max_aspect

    if original is not None and len(original) == len(boxes):
        ow = original[:, 2] - original[:, 0]
        oh = original[:, 3] - original[:, 1]
        keep &= (w * h) / (ow * oh + 1e-9) > min_area_ratio

    return boxes[keep], classes[keep]


def hsv_augment(
    image: np.ndarray,
    *,
    hgain: float = 0.015,
    sgain: float = 0.7,
    vgain: float = 0.4,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Random hue/saturation/value jitter.

    Colour jitter in HSV space keeps objects recognisable while making the model
    robust to lighting and white balance. Gains are relative: ``0.4`` means the
    channel is multiplied by a factor in ``[0.6, 1.4]``.
    """
    if hgain == sgain == vgain == 0:
        return image
    rnd = rng or random
    gains = np.array(
        [
            rnd.uniform(-1, 1) * hgain + 1,
            rnd.uniform(-1, 1) * sgain + 1,
            rnd.uniform(-1, 1) * vgain + 1,
        ],
        dtype=np.float32,
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    dtype = image.dtype
    x = np.arange(256, dtype=np.int16)
    lut_h = ((x * gains[0]) % 180).astype(dtype)  # hue is circular, 0..179
    lut_s = np.clip(x * gains[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * gains[2], 0, 255).astype(dtype)

    merged = cv2.merge(
        (cv2.LUT(hsv[..., 0], lut_h), cv2.LUT(hsv[..., 1], lut_s), cv2.LUT(hsv[..., 2], lut_v))
    )
    return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)


def random_hflip(
    image: np.ndarray, boxes: np.ndarray, *, p: float = 0.5, rng: random.Random | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal flip with probability ``p``, mirroring the boxes with it."""
    rnd = rng or random
    if rnd.random() >= p:
        return image, boxes
    width = image.shape[1]
    image = np.ascontiguousarray(image[:, ::-1])
    if len(boxes):
        boxes = boxes.copy()
        x1 = boxes[:, 0].copy()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - x1
    return image, boxes


def random_affine(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    *,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    border: tuple[int, int] = (0, 0),
    fill: int = 114,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random rotation / scale / shear / translation, applied to image and boxes.

    ``border`` is the (negative) padding used to crop a mosaic canvas back to
    the training size: pass ``(-img_size // 2, -img_size // 2)`` after
    :func:`mosaic4` and the output is ``img_size x img_size``.

    Args:
        image: ``(H, W, 3)`` uint8.
        boxes: ``(N, 4)`` xyxy.
        classes: ``(N,)``.
        degrees: max absolute rotation, in degrees.
        translate: max translation as a fraction of the output size.
        scale: scale is sampled from ``[1 - scale, 1 + scale]``.
        shear: max absolute shear, in degrees.
        border: ``(border_y, border_x)`` added to the output size.
        fill: fill value for revealed pixels.
        rng: optional RNG for reproducible augmentation.

    Returns:
        ``(image, boxes, classes)`` after the transform and after filtering.
    """
    rnd = rng or random
    height = image.shape[0] + border[0] * 2
    width = image.shape[1] + border[1] * 2

    # Centre the image on the origin so rotation/shear happen about its middle.
    center = np.eye(3, dtype=np.float32)
    center[0, 2] = -image.shape[1] / 2
    center[1, 2] = -image.shape[0] / 2

    rotate_scale = np.eye(3, dtype=np.float32)
    angle = rnd.uniform(-degrees, degrees)
    gain = rnd.uniform(1 - scale, 1 + scale)
    rotate_scale[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=gain)

    shear_m = np.eye(3, dtype=np.float32)
    shear_m[0, 1] = np.tan(np.radians(rnd.uniform(-shear, shear)))
    shear_m[1, 0] = np.tan(np.radians(rnd.uniform(-shear, shear)))

    translate_m = np.eye(3, dtype=np.float32)
    translate_m[0, 2] = (0.5 + rnd.uniform(-translate, translate)) * width
    translate_m[1, 2] = (0.5 + rnd.uniform(-translate, translate)) * height

    matrix = translate_m @ shear_m @ rotate_scale @ center
    image = cv2.warpAffine(image, matrix[:2], dsize=(width, height), borderValue=(fill, fill, fill))

    if len(boxes) == 0:
        return image, boxes.reshape(0, 4).astype(np.float32), classes.reshape(0).astype(np.int64)

    # Transform the four corners of each box, then take the enclosing box.
    n = len(boxes)
    corners = np.ones((n * 4, 3), dtype=np.float32)
    corners[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    corners = corners @ matrix.T
    corners = corners[:, :2].reshape(n, 8)

    xs = corners[:, [0, 2, 4, 6]]
    ys = corners[:, [1, 3, 5, 7]]
    new = np.stack((xs.min(1), ys.min(1), xs.max(1), ys.max(1)), axis=1)

    new[:, 0::2] = new[:, 0::2].clip(0, width)
    new[:, 1::2] = new[:, 1::2].clip(0, height)

    new, new_classes = filter_boxes(new, classes, original=boxes)
    return image, new, new_classes


def mosaic4(
    tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    img_size: int,
    *,
    fill: int = 114,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch four images into one canvas of side ``2 * img_size``.

    Mosaic puts objects in unusual positions and scales and multiplies the
    number of objects per batch, which is why it helps small-object accuracy so
    much. The canvas is twice the training size on purpose: a following
    :func:`random_affine` with ``border=(-img_size // 2, -img_size // 2)`` crops
    it back down, and that crop is part of the augmentation.

    Args:
        tiles: exactly four ``(image, boxes, classes)`` tuples.
        img_size: the training size.
        fill: background value for uncovered canvas.
        rng: optional RNG.

    Returns:
        ``(canvas, boxes, classes)`` with boxes in canvas coordinates.
    """
    if len(tiles) != 4:
        raise ValueError(f"mosaic4 needs exactly 4 tiles, got {len(tiles)}")
    rnd = rng or random

    s = img_size
    canvas = np.full((s * 2, s * 2, 3), fill, dtype=np.uint8)
    # Mosaic centre, kept away from the canvas edges so every tile stays visible.
    cx = int(rnd.uniform(s * 0.5, s * 1.5))
    cy = int(rnd.uniform(s * 0.5, s * 1.5))

    all_boxes: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []

    for i, (image, boxes, classes) in enumerate(tiles):
        h, w = image.shape[:2]
        # Scale each tile so its longest side matches img_size.
        ratio = s / max(h, w)
        if ratio != 1:
            interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=interp)
        h, w = image.shape[:2]

        if i == 0:  # top-left tile, anchored at the centre and growing up-left
            x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, s * 2), cy
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(s * 2, cy + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom-right
            x1a, y1a, x2a, y2a = cx, cy, min(cx + w, s * 2), min(s * 2, cy + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        canvas[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

        if len(boxes):
            # Offset from tile coordinates to canvas coordinates.
            off_x = x1a - x1b
            off_y = y1a - y1b
            moved = boxes.astype(np.float32) * ratio
            moved[:, 0::2] += off_x
            moved[:, 1::2] += off_y
            all_boxes.append(moved)
            all_classes.append(classes)

    if all_boxes:
        merged = np.concatenate(all_boxes, axis=0)
        merged_cls = np.concatenate(all_classes, axis=0)
        merged[:, 0::2] = merged[:, 0::2].clip(0, s * 2)
        merged[:, 1::2] = merged[:, 1::2].clip(0, s * 2)
        merged, merged_cls = filter_boxes(merged, merged_cls)
    else:
        merged = np.zeros((0, 4), dtype=np.float32)
        merged_cls = np.zeros((0,), dtype=np.int64)

    return canvas, merged, merged_cls


def mixup(
    a: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    alpha: float = 32.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend two samples and concatenate their labels.

    The blending weight is drawn from ``Beta(alpha, alpha)``; with the usual
    ``alpha=32`` the weight concentrates near ``0.5``, producing a genuine
    double exposure rather than a barely perceptible ghost.

    Both samples must already have the same shape (mixup runs after mosaic).
    """
    image_a, boxes_a, cls_a = a
    image_b, boxes_b, cls_b = b
    if image_a.shape != image_b.shape:
        raise ValueError(f"mixup needs equal shapes, got {image_a.shape} and {image_b.shape}")

    generator = rng or np.random.default_rng()
    weight = float(generator.beta(alpha, alpha))
    blended = image_a.astype(np.float32) * weight + image_b.astype(np.float32) * (1 - weight)

    boxes = np.concatenate((boxes_a, boxes_b), axis=0).astype(np.float32)
    classes = np.concatenate((cls_a, cls_b), axis=0).astype(np.int64)
    return blended.astype(np.uint8), boxes, classes
