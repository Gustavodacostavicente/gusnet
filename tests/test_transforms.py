# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the augmentations.

The property that matters in every one of these is the same: whatever happens
to the pixels must happen to the boxes.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from gusnet.data.transforms import (
    filter_boxes,
    hsv_augment,
    mixup,
    mosaic4,
    random_affine,
    random_hflip,
)


def _tile(size: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    boxes = np.array([[20.0, 30.0, 80.0, 120.0], [100.0, 100.0, 180.0, 190.0]], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    return image, boxes, classes


# --------------------------------------------------------------------- flipping


def test_hflip_mirrors_boxes():
    image, boxes, _ = _tile()
    flipped, new_boxes = random_hflip(image, boxes, p=1.0)
    width = image.shape[1]
    assert new_boxes[0, 0] == pytest.approx(width - boxes[0, 2])
    assert new_boxes[0, 2] == pytest.approx(width - boxes[0, 0])
    # Vertical coordinates and widths are untouched.
    assert np.allclose(new_boxes[:, 1], boxes[:, 1])
    assert np.allclose(new_boxes[:, 2] - new_boxes[:, 0], boxes[:, 2] - boxes[:, 0])
    assert flipped.shape == image.shape


def test_double_hflip_is_the_identity():
    image, boxes, _ = _tile()
    once = random_hflip(image, boxes, p=1.0)
    twice_image, twice_boxes = random_hflip(*once, p=1.0)
    assert np.array_equal(twice_image, image)
    assert np.allclose(twice_boxes, boxes)


def test_hflip_at_probability_zero_does_nothing():
    image, boxes, _ = _tile()
    same_image, same_boxes = random_hflip(image, boxes, p=0.0)
    assert np.array_equal(same_image, image)
    assert np.allclose(same_boxes, boxes)


# ------------------------------------------------------------------ colour jitter


def test_hsv_augment_preserves_shape_and_dtype():
    image, _, _ = _tile()
    out = hsv_augment(image, rng=random.Random(0))
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_hsv_augment_with_zero_gains_is_a_no_op():
    image, _, _ = _tile()
    assert np.array_equal(hsv_augment(image, hgain=0, sgain=0, vgain=0), image)


# ------------------------------------------------------------------ random affine


def test_affine_with_no_randomness_keeps_boxes_where_they_were():
    image, boxes, classes = _tile()
    out_image, out_boxes, out_classes = random_affine(
        image, boxes, classes, degrees=0, translate=0, scale=0, shear=0
    )
    assert out_image.shape == image.shape
    assert np.allclose(out_boxes, boxes, atol=1e-3)
    assert np.array_equal(out_classes, classes)


def test_affine_border_crops_a_mosaic_back_to_size():
    size = 320
    canvas = np.zeros((size * 2, size * 2, 3), dtype=np.uint8)
    boxes = np.array([[300.0, 300.0, 400.0, 400.0]], dtype=np.float32)
    classes = np.array([0], dtype=np.int64)

    out_image, _, _ = random_affine(
        canvas, boxes, classes, border=(-size // 2, -size // 2), rng=random.Random(0)
    )
    assert out_image.shape == (size, size, 3)


def test_affine_keeps_boxes_and_classes_aligned():
    image, boxes, classes = _tile()
    _, out_boxes, out_classes = random_affine(
        image, boxes, classes, degrees=10, translate=0.2, scale=0.4, rng=random.Random(3)
    )
    assert len(out_boxes) == len(out_classes)


def test_affine_on_an_empty_label_set():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    _, boxes, classes = random_affine(image, np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)


# ------------------------------------------------------------------------ mosaic


def test_mosaic4_canvas_size_and_box_bounds():
    size = 160
    tiles = [_tile(200, seed=i) for i in range(4)]
    canvas, boxes, classes = mosaic4(tiles, size, rng=random.Random(0))

    assert canvas.shape == (size * 2, size * 2, 3)
    assert len(boxes) == len(classes)
    assert boxes.min() >= 0
    assert boxes.max() <= size * 2


def test_mosaic4_gathers_objects_from_several_tiles():
    size = 320  # big canvas relative to the tiles: nothing gets cropped away
    tiles = [_tile(200, seed=i) for i in range(4)]
    _, boxes, _ = mosaic4(tiles, size, rng=random.Random(1))
    # Four tiles of two objects each; the centre crop may lose some, but a
    # mosaic that kept only one tile's worth would mean the offsets are wrong.
    assert len(boxes) > 2


def test_mosaic4_requires_four_tiles():
    with pytest.raises(ValueError, match="exactly 4 tiles"):
        mosaic4([_tile()], 160)


# ------------------------------------------------------------------------- mixup


def test_mixup_blends_images_and_concatenates_labels():
    a = _tile(128, seed=0)
    b = _tile(128, seed=1)
    image, boxes, classes = mixup(a, b, rng=np.random.default_rng(0))

    assert image.shape == a[0].shape
    assert image.dtype == np.uint8
    assert len(boxes) == len(a[1]) + len(b[1])
    assert len(classes) == len(boxes)


def test_mixup_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal shapes"):
        mixup(_tile(128), _tile(64))


# ------------------------------------------------------------------- box filter


def test_filter_boxes_drops_tiny_and_extreme_boxes():
    boxes = np.array(
        [
            [0.0, 0.0, 50.0, 50.0],  # fine
            [0.0, 0.0, 1.0, 50.0],  # too narrow
            [0.0, 0.0, 100.0, 3.0],  # extreme aspect ratio
        ],
        dtype=np.float32,
    )
    classes = np.array([0, 1, 2], dtype=np.int64)
    kept_boxes, kept_classes = filter_boxes(boxes, classes)
    assert len(kept_boxes) == 1
    assert kept_classes.tolist() == [0]


def test_filter_boxes_drops_mostly_cropped_boxes():
    original = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
    cropped = np.array([[0.0, 0.0, 100.0, 5.0]], dtype=np.float32)  # 5% of the area left
    kept, _ = filter_boxes(cropped, np.array([0]), original=original)
    assert len(kept) == 0


def test_filter_boxes_on_empty_input():
    boxes, classes = filter_boxes(np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)
