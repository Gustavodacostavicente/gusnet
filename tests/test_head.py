# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for anchor points, the DFL decode and the detection head."""

from __future__ import annotations

import math

import pytest
import torch

from gusnet.nn.head import DetectHead, decode_distances, make_anchor_points
from gusnet.ops.boxes import ltrb_to_xyxy

# ------------------------------------------------------------------ anchor points


def test_anchor_points_sit_at_cell_centres():
    points, strides = make_anchor_points([(2, 2)], [8])
    assert points.tolist() == [[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0]]
    assert strides.tolist() == [[8.0], [8.0], [8.0], [8.0]]


def test_anchor_points_are_ordered_row_major_per_level():
    points, _ = make_anchor_points([(2, 3)], [1], offset=0.0)
    # x varies fastest, matching the (B, H*W, C) flattening in the head.
    assert points.tolist() == [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ]


def test_anchor_points_concatenate_levels_in_order():
    shapes = [(8, 8), (4, 4), (2, 2)]
    points, strides = make_anchor_points(shapes, [8, 16, 32])
    assert points.shape == (64 + 16 + 4, 2)
    assert strides[:64].unique().tolist() == [8.0]
    assert strides[64:80].unique().tolist() == [16.0]
    assert strides[80:].unique().tolist() == [32.0]


def test_anchor_points_reject_mismatched_strides():
    with pytest.raises(ValueError, match="feature shapes but"):
        make_anchor_points([(2, 2)], [8, 16])


# -------------------------------------------------------------------- DFL decode


def test_a_one_hot_distribution_decodes_to_that_integer():
    reg_max = 16
    project = torch.arange(reg_max + 1, dtype=torch.float32)
    logits = torch.full((1, 1, 4, reg_max + 1), -50.0)
    for edge, bin_index in enumerate([0, 3, 7, 16]):
        logits[0, 0, edge, bin_index] = 50.0

    distances = decode_distances(logits, project)
    assert torch.allclose(distances[0, 0], torch.tensor([0.0, 3.0, 7.0, 16.0]), atol=1e-4)


def test_a_uniform_distribution_decodes_to_the_mean():
    reg_max = 8
    project = torch.arange(reg_max + 1, dtype=torch.float32)
    distances = decode_distances(torch.zeros(1, 1, 4, reg_max + 1), project)
    assert torch.allclose(distances, torch.full((1, 1, 4), reg_max / 2))


def test_an_ambiguous_edge_lands_between_two_bins():
    # Two equally likely bins, 4 and 6: the expectation is 5. A direct
    # regression could only have committed to one of them.
    project = torch.arange(9, dtype=torch.float32)
    logits = torch.full((1, 1, 4, 9), -50.0)
    logits[..., 4] = 0.0
    logits[..., 6] = 0.0
    assert torch.allclose(decode_distances(logits, project), torch.full((1, 1, 4), 5.0))


def test_decoded_distances_become_boxes_in_pixels():
    points = torch.tensor([[100.0, 100.0]])
    strides = torch.tensor([[8.0]])
    distances = torch.tensor([[2.0, 3.0, 4.0, 5.0]])  # in stride units
    boxes = ltrb_to_xyxy(distances * strides, points)
    assert boxes.tolist() == [[100 - 16, 100 - 24, 100 + 32, 100 + 40]]


# --------------------------------------------------------------------- the head


def _features(channels=(16, 32, 64), size=32, batch=2):
    return [
        torch.randn(batch, c, size // s, size // s)
        for c, s in zip(channels, (1, 2, 4), strict=True)
    ]


def test_head_output_shapes():
    head = DetectHead((16, 32, 64), num_classes=5, reg_max=8).eval()
    with torch.no_grad():
        out = head(_features())

    anchors = 32 * 32 + 16 * 16 + 8 * 8
    assert out["cls_logits"].shape == (2, anchors, 5)
    assert out["reg_logits"].shape == (2, anchors, 4, 9)
    assert out["points"].shape == (anchors, 2)
    assert out["strides"].shape == (anchors, 1)
    assert out["boxes"].shape == (2, anchors, 4)
    assert out["scores"].shape == (2, anchors, 5)


def test_head_scores_are_probabilities_and_logits_are_not():
    head = DetectHead((16, 32, 64), num_classes=3).eval()
    with torch.no_grad():
        out = head(_features())
    assert float(out["scores"].min()) >= 0.0
    assert float(out["scores"].max()) <= 1.0
    assert torch.allclose(out["scores"], out["cls_logits"].sigmoid())


def test_head_starts_predicting_almost_no_objects():
    """The classification bias prior must hold at initialisation.

    Thousands of grid points all claiming an object on step one produces a loss
    spike large enough to stall training; the prior is what prevents it.
    """
    head = DetectHead((16, 32, 64), num_classes=80).eval()
    expected_bias = -math.log((1 - 0.01) / 0.01)
    for layer in head.cls_preds:
        assert torch.allclose(layer.bias, torch.full_like(layer.bias, expected_bias))

    with torch.no_grad():
        scores = head(_features())["scores"]
    assert float(scores.mean()) < 0.05


def test_head_boxes_are_consistent_with_its_own_raw_outputs():
    head = DetectHead((16, 32, 64), num_classes=4, reg_max=7).eval()
    with torch.no_grad():
        out = head(_features())
        recomputed = ltrb_to_xyxy(
            decode_distances(out["reg_logits"], head.project) * out["strides"], out["points"]
        )
    assert torch.allclose(out["boxes"], recomputed, atol=1e-5)


def test_head_is_differentiable():
    head = DetectHead((16, 32, 64), num_classes=4)
    out = head(_features(batch=1))
    (out["cls_logits"].sum() + out["boxes"].sum()).backward()
    assert all(p.grad is not None for p in head.parameters() if p.requires_grad)


def test_head_rejects_the_wrong_number_of_levels():
    head = DetectHead((16, 32, 64), num_classes=4)
    with pytest.raises(ValueError, match="expected 3 feature maps"):
        head([torch.randn(1, 16, 8, 8)])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_classes": 0}, "num_classes must be at least 1"),
        ({"num_classes": 4, "reg_max": 0}, "reg_max must be at least 1"),
        ({"num_classes": 4, "strides": (8, 16)}, "3 levels but 2 strides"),
    ],
)
def test_head_validates_its_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DetectHead((16, 32, 64), **kwargs)
