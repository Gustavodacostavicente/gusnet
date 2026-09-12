# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the drawing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gusnet.data import DetectionDataset, collate_detection
from gusnet.viz import class_color, draw_boxes, save_batch_preview, tensor_to_image


def test_class_color_is_stable_and_distinct():
    assert class_color(3) == class_color(3)
    assert class_color(0) != class_color(1)


def test_tensor_to_image_roundtrip():
    tensor = torch.zeros(3, 8, 12)
    tensor[0] = 1.0
    image = tensor_to_image(tensor)
    assert image.shape == (8, 12, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [255, 0, 0]


def test_draw_boxes_marks_pixels_without_touching_the_input():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[8.0, 8.0, 40.0, 40.0]], dtype=np.float32)
    drawn = draw_boxes(image, boxes, np.array([0]), class_names=["thing"])

    assert drawn.shape == image.shape
    assert drawn.any(), "nothing was drawn"
    assert not image.any(), "draw_boxes modified its input"


def test_draw_boxes_with_no_boxes():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert not draw_boxes(image, np.zeros((0, 4), dtype=np.float32)).any()


def test_save_batch_preview_writes_a_grid(folder_dataset: Path, tmp_path: Path):
    ds = DetectionDataset.from_folder(folder_dataset, "train", img_size=128, augment=False)
    batch = collate_detection([ds[i] for i in range(4)])

    out = save_batch_preview(batch, tmp_path / "preview" / "batch.jpg", class_names=ds.class_names)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_draw_points_marks_pixels_without_touching_the_input():
    from gusnet.viz import draw_points

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    drawn = draw_points(image, np.array([[16.0, 16.0]]), np.array([1]))
    assert drawn.any(), "nothing was drawn"
    assert not image.any(), "draw_points modified its input"


def test_draw_points_with_no_points():
    from gusnet.viz import draw_points

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert not draw_points(image, np.zeros((0, 2), dtype=np.float32)).any()
