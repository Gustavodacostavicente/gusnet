# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the letterbox resize."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gusnet.data.letterbox import letterbox, letterbox_boxes
from gusnet.ops.boxes import scale_boxes


def _image(height: int, width: int) -> np.ndarray:
    return np.full((height, width, 3), 200, dtype=np.uint8)


def test_output_is_square_and_ratio_is_the_limiting_one():
    result = letterbox(_image(480, 640), 640)
    assert result.image.shape == (640, 640, 3)
    assert result.ratio == pytest.approx(1.0)  # width is the limiting side
    assert result.pad[0] == pytest.approx(0.0)
    assert result.pad[1] == pytest.approx(80.0)  # (640 - 480) / 2
    assert result.orig_shape == (480, 640)


def test_aspect_ratio_is_preserved():
    result = letterbox(_image(200, 400), 640)
    content_h = 640 - 2 * result.pad[1]
    content_w = 640 - 2 * result.pad[0]
    assert content_w / content_h == pytest.approx(400 / 200, rel=0.02)


def test_scaleup_false_never_enlarges():
    result = letterbox(_image(100, 120), 640, scaleup=False)
    assert result.ratio == pytest.approx(1.0)
    assert result.image.shape == (640, 640, 3)


def test_non_centred_padding_puts_the_image_top_left():
    result = letterbox(_image(200, 400), 640, center=False)
    assert result.pad == (0.0, 0.0)
    assert result.image.shape == (640, 640, 3)


def test_stride_mode_pads_only_to_the_next_multiple():
    result = letterbox(_image(480, 640), 640, stride=32)
    height, width = result.image.shape[:2]
    assert height % 32 == 0 and width % 32 == 0
    assert height < 640  # no full padding to a square


def test_boxes_survive_a_letterbox_roundtrip():
    image = _image(480, 640)
    boxes = np.array([[10.0, 20.0, 300.0, 400.0], [500.0, 100.0, 620.0, 300.0]], dtype=np.float32)

    result = letterbox(image, 416)
    moved = letterbox_boxes(boxes, result)
    recovered = scale_boxes(
        torch.from_numpy(moved), result.ratio, result.pad, result.orig_shape
    ).numpy()

    assert np.allclose(recovered, boxes, atol=1e-3)


def test_empty_boxes_keep_their_shape():
    result = letterbox(_image(100, 100), 128)
    assert letterbox_boxes(np.zeros((0, 4), dtype=np.float32), result).shape == (0, 4)


def test_rejects_non_rgb_input():
    with pytest.raises(ValueError, match="expected an"):
        letterbox(np.zeros((10, 10), dtype=np.uint8), 64)
