# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the loss terms and the full detection objective."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from gusnet.losses import (
    DetectionLoss,
    DistributionFocalLoss,
    IoULoss,
    VarifocalLoss,
)

# ------------------------------------------------------------------- varifocal


def test_varifocal_is_near_zero_for_a_perfect_prediction():
    targets = torch.tensor([[[1.0, 0.0, 0.0]]])
    logits = torch.tensor([[[12.0, -12.0, -12.0]]])
    assert float(VarifocalLoss()(logits, targets)) < 1e-4


def test_varifocal_weights_positives_by_their_target_quality():
    """A half-quality positive must pull half as hard as a perfect one."""
    loss = VarifocalLoss(reduction="none")
    logits = torch.zeros(1, 1, 1)  # p = 0.5, so BCE is identical in both cases

    strong = loss(logits, torch.tensor([[[1.0]]]))
    weak = loss(logits, torch.tensor([[[0.5]]]))
    # BCE(0.5, 1.0) = BCE(0.5, 0.5) is not true, so compare the weighting
    # directly: the weak target's loss is scaled by q = 0.5.
    expected_ratio = (
        0.5
        * F.binary_cross_entropy_with_logits(logits, torch.tensor([[[0.5]]]))
        / F.binary_cross_entropy_with_logits(logits, torch.tensor([[[1.0]]]))
    )
    assert float(weak) / float(strong) == pytest.approx(float(expected_ratio), rel=1e-4)


def test_varifocal_suppresses_easy_negatives():
    """The point of the focal term: confident background stops contributing."""
    loss = VarifocalLoss(reduction="none")
    target = torch.zeros(1, 1, 1)

    easy = loss(torch.tensor([[[-8.0]]]), target)  # p is tiny, clearly background
    hard = loss(torch.tensor([[[0.0]]]), target)  # p = 0.5, still looks like an object
    assert float(easy) < float(hard) / 1000


def test_varifocal_reduces_to_bce_with_neutral_parameters():
    # With alpha=1 and gamma=0 the negative weight is 1, and on hard 0/1 targets
    # the positive weight q is 1 too, so nothing is reweighted at all.
    loss = VarifocalLoss(alpha=1.0, gamma=0.0, reduction="sum")
    logits = torch.randn(2, 5, 3)
    targets = (torch.rand(2, 5, 3) > 0.5).float()
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
    assert float(loss(logits, targets)) == pytest.approx(float(expected), rel=1e-5)


def test_varifocal_reductions():
    logits = torch.randn(2, 4, 3)
    targets = torch.rand(2, 4, 3)
    none = VarifocalLoss(reduction="none")(logits, targets)
    assert none.shape == (2, 4, 3)
    assert float(VarifocalLoss(reduction="sum")(logits, targets)) == pytest.approx(
        float(none.sum()), rel=1e-5
    )
    assert float(VarifocalLoss(reduction="mean")(logits, targets)) == pytest.approx(
        float(none.mean()), rel=1e-5
    )


def test_varifocal_rejects_an_unknown_reduction():
    with pytest.raises(ValueError, match="unknown reduction"):
        VarifocalLoss(reduction="median")


# ------------------------------------------------------------------- iou loss


def test_iou_loss_is_zero_for_identical_boxes():
    box = torch.tensor([[10.0, 10.0, 30.0, 40.0]])
    assert float(IoULoss(kind="ciou")(box, box)) == pytest.approx(0.0, abs=1e-5)


def test_ciou_loss_still_has_a_direction_when_boxes_do_not_overlap():
    """Plain IoU gives the same flat 1.0 for every miss; CIoU does not."""
    target = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    near = torch.tensor([[15.0, 0.0, 25.0, 10.0]])
    far = torch.tensor([[80.0, 0.0, 90.0, 10.0]])

    plain = IoULoss(kind="iou")
    assert float(plain(near, target)) == pytest.approx(float(plain(far, target)))

    ciou = IoULoss(kind="ciou")
    assert float(ciou(near, target)) < float(ciou(far, target))


def test_iou_loss_shapes_and_reductions():
    pred = torch.rand(6, 4)
    pred[:, 2:] += pred[:, :2] + 1
    target = pred + 0.5
    assert IoULoss(reduction="none")(pred, target).shape == (6,)
    assert IoULoss(reduction="sum")(pred, target).ndim == 0


# --------------------------------------------------------- distribution focal


def _peaked(bins: int, index: int, batch: int = 1) -> torch.Tensor:
    logits = torch.full((batch, 4, bins), -20.0)
    logits[..., index] = 20.0
    return logits


def test_dfl_is_near_zero_when_the_target_sits_on_the_predicted_bin():
    loss = DistributionFocalLoss(reg_max=8)
    target = torch.full((1, 4), 3.0)
    assert float(loss(_peaked(9, 3), target)) < 1e-4


def test_dfl_prefers_mass_on_both_neighbours_for_a_target_between_bins():
    """This is what a plain regression on the expectation cannot enforce."""
    loss = DistributionFocalLoss(reg_max=8)
    target = torch.full((1, 4), 3.5)

    committed = _peaked(9, 3)  # all mass on bin 3
    split = torch.full((1, 4, 9), -20.0)
    split[..., 3] = 0.0
    split[..., 4] = 0.0  # mass shared between 3 and 4, expectation 3.5

    assert float(loss(split, target)) < float(loss(committed, target))


def test_dfl_penalises_a_distribution_with_the_right_mean_but_wrong_shape():
    loss = DistributionFocalLoss(reg_max=8)
    target = torch.full((1, 4), 4.0)

    exact = _peaked(9, 4)
    spread = torch.full((1, 4, 9), -20.0)
    spread[..., 0] = 0.0
    spread[..., 8] = 0.0  # expectation is also 4.0, but the shape is nonsense

    assert float(loss(exact, target)) < float(loss(spread, target))


def test_dfl_handles_a_target_on_the_last_bin():
    loss = DistributionFocalLoss(reg_max=8)
    value = float(loss(_peaked(9, 8), torch.full((1, 4), 8.0)))
    assert math.isfinite(value)
    assert value < 1e-4


def test_dfl_clamps_targets_outside_the_supported_range():
    loss = DistributionFocalLoss(reg_max=4)
    value = float(loss(_peaked(5, 4), torch.full((1, 4), 99.0)))
    assert math.isfinite(value)


def test_dfl_rejects_a_mismatched_bin_count():
    with pytest.raises(ValueError, match="expected 9 bins"):
        DistributionFocalLoss(reg_max=8)(torch.zeros(1, 4, 5), torch.zeros(1, 4))


def test_dfl_returns_one_value_per_sample():
    loss = DistributionFocalLoss(reg_max=8, reduction="none")
    assert loss(torch.zeros(7, 4, 9), torch.full((7, 4), 2.0)).shape == (7,)


# --------------------------------------------------------------- full objective


def _batch(num_classes: int = 3, size: int = 64):
    targets = torch.tensor(
        [
            [0.0, 1.0, 8.0, 8.0, 40.0, 40.0],
            [0.0, 2.0, 20.0, 24.0, 60.0, 56.0],
            [1.0, 0.0, 4.0, 4.0, 28.0, 28.0],
        ]
    )
    images = torch.rand(2, 3, size, size)
    return images, targets, num_classes


def test_detection_loss_returns_a_scalar_and_a_breakdown():
    from gusnet.models import GUSNet

    torch.manual_seed(0)
    images, targets, num_classes = _batch()
    model = GUSNet.from_variant("n", num_classes=num_classes)
    criterion = DetectionLoss(num_classes)

    total, breakdown = criterion(model(images), targets)

    assert total.ndim == 0
    assert total.requires_grad
    assert breakdown.num_foreground > 0
    assert set(breakdown.items()) == {"total", "cls", "box", "dfl", "fg"}
    assert all(math.isfinite(v) for v in breakdown.items().values())
    assert float(breakdown.total) == pytest.approx(
        float(breakdown.cls + breakdown.box + breakdown.dfl), rel=1e-5
    )


def test_detection_loss_backpropagates_to_every_parameter():
    from gusnet.models import GUSNet

    torch.manual_seed(0)
    images, targets, num_classes = _batch()
    model = GUSNet.from_variant("n", num_classes=num_classes)

    total, _ = DetectionLoss(num_classes)(model(images), targets)
    total.backward()

    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or not bool(param.grad.abs().sum()))
    ]
    assert not missing, f"no gradient reached: {missing[:5]}"


def test_detection_loss_handles_a_batch_with_no_objects():
    from gusnet.models import GUSNet

    torch.manual_seed(0)
    model = GUSNet.from_variant("n", num_classes=3)
    total, breakdown = DetectionLoss(3)(model(torch.rand(2, 3, 64, 64)), torch.zeros(0, 6))

    assert math.isfinite(float(total.detach()))
    assert float(breakdown.box) == 0.0
    assert float(breakdown.dfl) == 0.0
    assert breakdown.num_foreground == 0
    total.backward()  # background-only batches must still train the classifier


def test_detection_loss_prefers_correct_predictions():
    """A model that already gets it right must score better than a random one."""
    from gusnet.models import GUSNet

    torch.manual_seed(0)
    images, targets, num_classes = _batch()
    model = GUSNet.from_variant("n", num_classes=num_classes).eval()
    criterion = DetectionLoss(num_classes)

    with torch.no_grad():
        output = model(images)
        random_loss, _ = criterion(output, targets)

        # Replace the predictions with the ground truth of image 0's first box.
        cheating = dict(output)
        cheating["boxes"] = targets[0, 2:6].expand_as(output["boxes"]).clone()
        cheating["scores"] = torch.zeros_like(output["scores"])
        cheating["scores"][..., 1] = 0.99
        cheating_loss, _ = criterion(cheating, targets)

    assert float(cheating_loss) < float(random_loss)


def test_detection_loss_can_use_simota():
    from gusnet.assign import SimOTAAssigner
    from gusnet.models import GUSNet

    torch.manual_seed(0)
    images, targets, num_classes = _batch()
    model = GUSNet.from_variant("n", num_classes=num_classes)
    criterion = DetectionLoss(num_classes, assigner=SimOTAAssigner(num_classes))

    total, breakdown = criterion(model(images), targets)
    assert math.isfinite(float(total.detach()))
    assert breakdown.num_foreground > 0


def test_the_model_can_overfit_a_single_image():
    """The end-to-end check: data, network, assignment and losses must close.

    Memorising one image is the weakest possible learning task. If the loss does
    not collapse here, something in the chain is disconnected -- and every
    scalar metric downstream would be meaningless.
    """
    from gusnet.models import GUSNet

    torch.manual_seed(0)
    image = torch.rand(1, 3, 64, 64)
    targets = torch.tensor(
        [
            [0.0, 0.0, 8.0, 8.0, 32.0, 40.0],
            [0.0, 1.0, 36.0, 20.0, 60.0, 56.0],
        ]
    )

    model = GUSNet.from_variant("n", num_classes=2)
    criterion = DetectionLoss(2)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3)

    first = None
    for step in range(120):
        optimiser.zero_grad()
        total, breakdown = criterion(model(image), targets)
        total.backward()
        optimiser.step()
        if step == 0:
            first = float(breakdown.total)
        last = float(breakdown.total)

    assert last < first * 0.35, f"loss barely moved: {first:.3f} -> {last:.3f}"


def test_overfitting_makes_the_predicted_boxes_approach_the_targets():
    """Not just a falling number: the boxes have to end up in the right place.

    Checked in eval mode, which is the mode that will ever matter. That costs
    extra steps: BatchNorm's ``momentum=0.03`` means the running statistics need
    a couple of hundred updates to catch up with the batch statistics the
    network is actually training on, so a model that already predicts perfectly
    in train mode can still be useless in eval mode for a while. On a single
    repeated image that lag is the whole gap; in real training it is invisible.
    """
    from gusnet.models import GUSNet
    from gusnet.ops.boxes import box_iou

    torch.manual_seed(0)
    image = torch.rand(1, 3, 64, 64)
    gt = torch.tensor([[12.0, 12.0, 44.0, 44.0]])
    targets = torch.tensor([[0.0, 0.0, 12.0, 12.0, 44.0, 44.0]])

    model = GUSNet.from_variant("n", num_classes=1)
    criterion = DetectionLoss(1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3)

    for _ in range(250):
        optimiser.zero_grad()
        total, _ = criterion(model(image), targets)
        total.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        out = model(image)
        best = out["scores"][0, :, 0].argmax()
        iou = float(box_iou(out["boxes"][0, best : best + 1], gt))
        confidence = float(out["scores"][0, best, 0])

    assert iou > 0.7, f"the most confident box only reaches IoU {iou:.2f}"
    # The soft targets should also have taught it to be confident about a box
    # this good -- that coupling is the point of the task-aligned assignment.
    assert confidence > 0.5, f"a perfect box is only scored {confidence:.2f}"
