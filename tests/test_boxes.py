# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for box conversions and overlap metrics."""

from __future__ import annotations

import pytest
import torch

from gusnet.ops import boxes as B


def test_xyxy_cxcywh_roundtrip():
    xyxy = torch.tensor([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 1.0, 1.0]])
    assert torch.allclose(B.cxcywh_to_xyxy(B.xyxy_to_cxcywh(xyxy)), xyxy)


def test_cxcywh_values():
    cxcywh = B.xyxy_to_cxcywh(torch.tensor([10.0, 20.0, 30.0, 60.0]))
    assert torch.allclose(cxcywh, torch.tensor([20.0, 40.0, 20.0, 40.0]))


def test_ltrb_roundtrip_with_broadcast():
    boxes = torch.tensor([[10.0, 10.0, 30.0, 50.0], [0.0, 5.0, 8.0, 9.0]])
    points = torch.tensor([[15.0, 20.0], [4.0, 6.0]])
    dist = B.xyxy_to_ltrb(boxes, points)
    assert torch.allclose(B.ltrb_to_xyxy(dist, points), boxes)


def test_ltrb_is_negative_outside_the_box():
    box = torch.tensor([10.0, 10.0, 20.0, 20.0])
    outside = torch.tensor([5.0, 15.0])
    dist = B.xyxy_to_ltrb(box, outside)
    assert dist[0] < 0  # the point is to the left of the left edge


def test_box_area_clamps_degenerate_boxes():
    degenerate = torch.tensor([[10.0, 10.0, 5.0, 20.0]])
    assert B.box_area(degenerate).item() == 0.0


def test_box_iou_matrix():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # identical
            [5.0, 0.0, 15.0, 10.0],  # half overlap -> 50/150
            [20.0, 20.0, 30.0, 30.0],  # disjoint
        ]
    )
    iou = B.box_iou(a, b)
    assert iou.shape == (1, 3)
    assert iou[0, 0].item() == pytest.approx(1.0)
    assert iou[0, 1].item() == pytest.approx(1 / 3)
    assert iou[0, 2].item() == pytest.approx(0.0)


@pytest.mark.parametrize("kind", ["iou", "giou", "diou", "ciou"])
def test_bbox_iou_is_one_for_identical_boxes(kind):
    box = torch.tensor([[3.0, 4.0, 13.0, 24.0]])
    assert B.bbox_iou(box, box, kind=kind).item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("kind", ["giou", "diou", "ciou"])
def test_generalized_iou_penalises_distance(kind):
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    near = torch.tensor([[20.0, 0.0, 30.0, 10.0]])
    far = torch.tensor([[60.0, 0.0, 70.0, 10.0]])
    # Plain IoU cannot tell these apart: both are zero.
    assert B.bbox_iou(a, near, kind="iou").item() == pytest.approx(0.0)
    assert B.bbox_iou(a, far, kind="iou").item() == pytest.approx(0.0)
    # The generalized variants must, which is the whole point of using them.
    assert B.bbox_iou(a, near, kind=kind).item() > B.bbox_iou(a, far, kind=kind).item()


def test_ciou_adds_an_aspect_ratio_penalty_on_top_of_diou():
    # Two boxes that are concentric, so the DIoU centre term is zero for both,
    # and that have the same area and the same IoU with the target -- only the
    # aspect ratio differs. Only CIoU can separate them.
    target = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    square = torch.tensor([[2.0, 2.0, 18.0, 18.0]])  # 16x16, aspect 1.0
    oblong = torch.tensor([[2.0, 6.0, 18.0, 14.0]])  # 16x8, aspect 2.0

    diou_square = B.bbox_iou(target, square, kind="diou")
    ciou_square = B.bbox_iou(target, square, kind="ciou")
    ciou_oblong = B.bbox_iou(target, oblong, kind="ciou")

    # The aspect term costs nothing when the shapes already agree...
    assert ciou_square.item() == pytest.approx(diou_square.item(), abs=1e-6)
    # ...and costs something when they do not.
    assert ciou_oblong.item() < B.bbox_iou(target, oblong, kind="diou").item()
    assert ciou_oblong.item() < ciou_square.item()


def test_bbox_iou_broadcasts():
    a = torch.rand(2, 5, 4)
    a[..., 2:] += a[..., :2] + 1  # make valid boxes
    b = a[:, :1]
    assert B.bbox_iou(a, b, kind="ciou").shape == (2, 5)


def test_bbox_iou_rejects_unknown_kind():
    box = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="unknown iou kind"):
        B.bbox_iou(box, box, kind="nope")


def test_clip_boxes():
    boxes = torch.tensor([[-5.0, -5.0, 100.0, 200.0]])
    clipped = B.clip_boxes(boxes, (50, 80))
    assert torch.allclose(clipped, torch.tensor([[0.0, 0.0, 80.0, 50.0]]))


def test_scale_boxes_undoes_a_letterbox():
    # A 100x200 image scaled by 0.5 and padded by 20 px on the left.
    original = torch.tensor([[10.0, 20.0, 60.0, 120.0]])
    ratio, pad = 0.5, (20.0, 0.0)
    letterboxed = original * ratio
    letterboxed[:, 0::2] += pad[0]
    letterboxed[:, 1::2] += pad[1]

    recovered = B.scale_boxes(letterboxed, ratio, pad, (200, 100))
    assert torch.allclose(recovered, original, atol=1e-4)
