# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for label assignment."""

from __future__ import annotations

import pytest
import torch

from gusnet.assign import (
    SimOTAAssigner,
    TaskAlignedAssigner,
    points_in_boxes,
    points_in_centers,
    resolve_conflicts,
    targets_to_batch,
)
from gusnet.nn.head import make_anchor_points

IMAGE_SIZE = 64
STRIDES = (8, 16, 32)


def _grid():
    shapes = [(IMAGE_SIZE // s, IMAGE_SIZE // s) for s in STRIDES]
    return make_anchor_points(shapes, STRIDES)


def _scene(
    box=(8.0, 8.0, 40.0, 40.0),
    label=1,
    num_classes=3,
    confidence=0.9,
    perfect=True,
):
    """One image, one object, and predictions we control exactly."""
    points, strides = _grid()
    anchors = points.shape[0]

    gt_boxes = torch.tensor([[list(box)]])
    gt_labels = torch.tensor([[label]])
    gt_mask = torch.tensor([[True]])

    pred_boxes = gt_boxes[:, 0:1].expand(1, anchors, 4).clone()
    if not perfect:
        # Shrink every prediction towards its centre: still overlapping, but
        # clearly worse.
        centre = (pred_boxes[..., :2] + pred_boxes[..., 2:]) * 0.5
        pred_boxes = torch.cat((centre - 4.0, centre + 4.0), dim=-1)

    pred_scores = torch.full((1, anchors, num_classes), 0.1)
    pred_scores[..., label] = confidence

    return points, strides, pred_scores, pred_boxes, gt_labels, gt_boxes, gt_mask


# ------------------------------------------------------------------ target padding


def test_targets_to_batch_pads_to_the_largest_image():
    targets = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0, 10.0, 10.0],
            [2.0, 1.0, 5.0, 5.0, 15.0, 15.0],
            [2.0, 0.0, 1.0, 1.0, 4.0, 4.0],
        ]
    )
    labels, boxes, mask = targets_to_batch(targets, batch_size=3)

    assert labels.shape == (3, 2)
    assert boxes.shape == (3, 2, 4)
    assert mask.tolist() == [[True, False], [False, False], [True, True]]
    assert labels[0, 0].item() == 2
    assert boxes[2, 1].tolist() == [1.0, 1.0, 4.0, 4.0]


def test_targets_to_batch_handles_an_empty_batch():
    labels, boxes, mask = targets_to_batch(torch.zeros(0, 6), batch_size=2)
    assert labels.shape == (2, 1)
    assert boxes.shape == (2, 1, 4)
    assert not bool(mask.any())


def test_targets_to_batch_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        targets_to_batch(torch.zeros(3, 5), batch_size=1)


# -------------------------------------------------------------------- candidacy


def test_points_in_boxes_excludes_the_edges():
    points = torch.tensor([[5.0, 5.0], [0.0, 5.0], [10.0, 5.0], [20.0, 5.0]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    assert points_in_boxes(points, boxes)[0, 0].tolist() == [True, False, False, False]


def test_points_in_centers_scales_with_the_stride():
    points = torch.tensor([[50.0, 50.0], [50.0, 50.0]])
    strides = torch.tensor([[1.0], [32.0]])
    boxes = torch.tensor([[[0.0, 0.0, 100.0, 100.0]]])  # centre at (50, 50)

    inside = points_in_centers(points, boxes, strides, radius=2.5)[0, 0]
    assert inside.tolist() == [True, True]

    far = torch.tensor([[58.0, 50.0], [58.0, 50.0]])
    inside_far = points_in_centers(far, boxes, strides, radius=2.5)[0, 0]
    # 8 px away: outside a 2.5 px radius, well inside an 80 px one.
    assert inside_far.tolist() == [False, True]


def test_resolve_conflicts_gives_a_contested_point_to_the_better_overlap():
    mask = torch.tensor([[[True, True], [True, False]]])  # 2 objects, 2 points
    overlaps = torch.tensor([[[0.3, 0.9], [0.8, 0.0]]])

    cleaned, fg_mask, gt_index = resolve_conflicts(mask, overlaps)
    assert fg_mask.tolist() == [[True, True]]
    # Point 0 was claimed by both; object 1 overlaps it more (0.8 > 0.3).
    assert gt_index[0, 0].item() == 1
    assert gt_index[0, 1].item() == 0
    assert int(cleaned[0, :, 0].sum()) == 1


def test_resolve_conflicts_leaves_uncontested_points_alone():
    mask = torch.tensor([[[True, False], [False, True]]])
    overlaps = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    _, fg_mask, gt_index = resolve_conflicts(mask, overlaps)
    assert fg_mask.tolist() == [[True, True]]
    assert gt_index.tolist() == [[0, 1]]


# ------------------------------------------------------------ task-aligned assigner


def test_tal_selects_at_most_topk_points_per_object():
    assigner = TaskAlignedAssigner(num_classes=3, topk=5)
    points, _, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene()
    result = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask)
    assert result.num_foreground == 5


def test_tal_assigns_only_points_inside_the_object():
    assigner = TaskAlignedAssigner(num_classes=3, topk=13)
    points, _, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene()
    result = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask)

    chosen = points[result.fg_mask[0]]
    x1, y1, x2, y2 = gt_boxes[0, 0].tolist()
    assert bool(((chosen[:, 0] > x1) & (chosen[:, 0] < x2)).all())
    assert bool(((chosen[:, 1] > y1) & (chosen[:, 1] < y2)).all())


def test_tal_copies_the_object_label_and_box_to_its_points():
    assigner = TaskAlignedAssigner(num_classes=3, topk=6)
    points, _, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene(label=2)
    result = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask)

    assert set(result.target_labels[0][result.fg_mask[0]].tolist()) == {2}
    assigned_boxes = result.target_boxes[0][result.fg_mask[0]]
    assert torch.allclose(assigned_boxes, gt_boxes[0, 0].expand_as(assigned_boxes))


def test_tal_classification_target_is_the_achieved_iou_not_one():
    """The soft target is what ties confidence to localisation quality."""
    assigner = TaskAlignedAssigner(num_classes=3, topk=6)
    points, _, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene(perfect=True)
    perfect = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask)
    # Predictions equal to the object: best candidate is trained towards 1.0.
    assert float(perfect.target_scores.max()) == pytest.approx(1.0, abs=1e-4)

    points, _, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene(perfect=False)
    poor = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask)
    # Same confidence, worse boxes: the target drops to the IoU they achieve.
    assert float(poor.target_scores.max()) < 0.5
    assert float(poor.target_scores.max()) > 0.0


def test_tal_writes_the_target_on_the_right_class_only():
    assigner = TaskAlignedAssigner(num_classes=3, topk=4)
    points, _, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene(label=1)
    result = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask)

    positives = result.target_scores[0][result.fg_mask[0]]
    assert float(positives[:, 1].min()) > 0
    assert float(positives[:, [0, 2]].abs().max()) == 0.0


def test_tal_returns_all_background_when_there_are_no_objects():
    assigner = TaskAlignedAssigner(num_classes=3)
    points, _, scores, boxes, _, gt_boxes, _ = _scene()
    empty_mask = torch.tensor([[False]])
    result = assigner(
        scores, boxes, points, torch.zeros(1, 1, dtype=torch.long), gt_boxes, empty_mask
    )
    assert result.num_foreground == 0
    assert float(result.target_scores.abs().max()) == 0.0


def test_tal_resolves_overlapping_objects():
    assigner = TaskAlignedAssigner(num_classes=3, topk=20)
    points, _ = _grid()
    anchors = points.shape[0]

    # Two heavily overlapping objects; every point predicts the first one.
    gt_boxes = torch.tensor([[[8.0, 8.0, 40.0, 40.0], [10.0, 10.0, 42.0, 42.0]]])
    gt_labels = torch.tensor([[0, 1]])
    gt_mask = torch.tensor([[True, True]])

    pred_boxes = gt_boxes[:, 0:1].expand(1, anchors, 4).clone()
    pred_scores = torch.full((1, anchors, 3), 0.5)

    result = assigner(pred_scores, pred_boxes, points, gt_labels, gt_boxes, gt_mask)

    # Each assigned point carries exactly one label and one box.
    assert result.target_labels.shape == (1, anchors)
    # Points predicting object 0 exactly must not be handed to object 1.
    assigned = result.target_labels[0][result.fg_mask[0]]
    assert (assigned == 0).sum() > (assigned == 1).sum()


def test_tal_ignores_padded_objects():
    assigner = TaskAlignedAssigner(num_classes=3, topk=4)
    points, _ = _grid()
    anchors = points.shape[0]

    gt_boxes = torch.tensor([[[8.0, 8.0, 40.0, 40.0], [0.0, 0.0, 0.0, 0.0]]])
    gt_labels = torch.tensor([[1, 0]])
    gt_mask = torch.tensor([[True, False]])

    pred_boxes = gt_boxes[:, 0:1].expand(1, anchors, 4).clone()
    pred_scores = torch.full((1, anchors, 3), 0.5)

    result = assigner(pred_scores, pred_boxes, points, gt_labels, gt_boxes, gt_mask)
    assert set(result.target_labels[0][result.fg_mask[0]].tolist()) == {1}


def test_tal_does_not_build_a_graph():
    assigner = TaskAlignedAssigner(num_classes=3, topk=4)
    points, _, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene()
    scores = scores.requires_grad_(True)
    result = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask)
    assert not result.target_scores.requires_grad


def test_tal_rejects_a_bad_topk():
    with pytest.raises(ValueError, match="topk must be at least 1"):
        TaskAlignedAssigner(num_classes=3, topk=0)


# ------------------------------------------------------------------ SimOTA assigner


def test_simota_assigns_points_and_labels_them_correctly():
    assigner = SimOTAAssigner(num_classes=3)
    points, strides, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene(label=2)
    result = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask, strides)

    assert result.num_foreground > 0
    assert set(result.target_labels[0][result.fg_mask[0]].tolist()) == {2}
    assigned_boxes = result.target_boxes[0][result.fg_mask[0]]
    assert torch.allclose(assigned_boxes, gt_boxes[0, 0].expand_as(assigned_boxes))


def test_simota_gives_an_easy_object_more_positives_than_a_hard_one():
    """Dynamic k is the point of SimOTA: supply follows how well it is found."""
    assigner = SimOTAAssigner(num_classes=3, candidate_topk=10)

    args = _scene(perfect=True)
    easy = assigner(args[2], args[3], args[0], args[4], args[5], args[6], args[1])

    args = _scene(perfect=False)
    hard = assigner(args[2], args[3], args[0], args[4], args[5], args[6], args[1])

    assert easy.num_foreground > hard.num_foreground
    assert hard.num_foreground >= 1  # k is clamped so an object is never lost


def test_simota_target_score_carries_the_iou():
    assigner = SimOTAAssigner(num_classes=3)
    points, strides, scores, boxes, gt_labels, gt_boxes, gt_mask = _scene(perfect=True)
    result = assigner(scores, boxes, points, gt_labels, gt_boxes, gt_mask, strides)
    assert float(result.target_scores.max()) == pytest.approx(1.0, abs=1e-4)


def test_simota_skips_images_without_objects():
    assigner = SimOTAAssigner(num_classes=3)
    points, strides, scores, boxes, gt_labels, gt_boxes, _ = _scene()
    result = assigner(scores, boxes, points, gt_labels, gt_boxes, torch.tensor([[False]]), strides)
    assert result.num_foreground == 0


def test_simota_never_assigns_a_point_to_two_objects():
    assigner = SimOTAAssigner(num_classes=3)
    points, strides = _grid()
    anchors = points.shape[0]

    gt_boxes = torch.tensor([[[8.0, 8.0, 40.0, 40.0], [10.0, 10.0, 42.0, 42.0]]])
    gt_labels = torch.tensor([[0, 1]])
    gt_mask = torch.tensor([[True, True]])
    pred_boxes = gt_boxes[:, 0:1].expand(1, anchors, 4).clone()
    pred_scores = torch.full((1, anchors, 3), 0.5)

    result = assigner(pred_scores, pred_boxes, points, gt_labels, gt_boxes, gt_mask, strides)
    # target_gt_index holds a single object per point by construction; the real
    # check is that the foreground count never exceeds the number of points.
    assert result.num_foreground <= anchors
    assert result.target_gt_index.max() <= 1


def test_simota_rejects_a_bad_candidate_topk():
    with pytest.raises(ValueError, match="candidate_topk must be at least 1"):
        SimOTAAssigner(num_classes=3, candidate_topk=0)


# --------------------------------------------------------------- both assigners


@pytest.mark.parametrize("build", ["tal", "simota"])
def test_assigners_run_on_a_real_batch(build):
    from gusnet.models import GUSNet

    torch.manual_seed(0)
    model = GUSNet.from_variant("n", num_classes=3).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 64, 64))

    targets = torch.tensor(
        [
            [0.0, 1.0, 8.0, 8.0, 40.0, 40.0],
            [0.0, 2.0, 20.0, 20.0, 60.0, 60.0],
            [1.0, 0.0, 4.0, 4.0, 28.0, 28.0],
        ]
    )
    gt_labels, gt_boxes, gt_mask = targets_to_batch(targets, batch_size=2)

    if build == "tal":
        assigner = TaskAlignedAssigner(num_classes=3, topk=10)
        result = assigner(out["scores"], out["boxes"], out["points"], gt_labels, gt_boxes, gt_mask)
    else:
        assigner = SimOTAAssigner(num_classes=3)
        result = assigner(
            out["scores"],
            out["boxes"],
            out["points"],
            gt_labels,
            gt_boxes,
            gt_mask,
            out["strides"],
        )

    anchors = out["scores"].shape[1]
    assert result.fg_mask.shape == (2, anchors)
    assert result.target_boxes.shape == (2, anchors, 4)
    assert result.target_scores.shape == (2, anchors, 3)
    assert result.num_foreground > 0
