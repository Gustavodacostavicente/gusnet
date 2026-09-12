# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""End-to-end tests for the assembled detector."""

from __future__ import annotations

import pytest
import torch

from gusnet.data import DetectionDataset, build_dataloader
from gusnet.models import VARIANTS, GUSNet


@pytest.fixture(scope="module")
def tiny_model() -> GUSNet:
    return GUSNet.from_variant("n", num_classes=4).eval()


def test_forward_produces_one_prediction_per_grid_point(tiny_model: GUSNet):
    size = 128
    with torch.no_grad():
        out = tiny_model(torch.rand(2, 3, size, size))

    expected = sum((size // s) ** 2 for s in tiny_model.strides)
    assert out["cls_logits"].shape == (2, expected, 4)
    assert out["boxes"].shape == (2, expected, 4)
    assert out["points"].shape == (expected, 2)


def test_anchor_count_at_the_usual_training_size():
    # 640 px with strides 8/16/32 gives the familiar 8400 predictions.
    assert sum((640 // s) ** 2 for s in (8, 16, 32)) == 8400


def test_boxes_come_out_in_image_pixel_coordinates(tiny_model: GUSNet):
    with torch.no_grad():
        out = tiny_model(torch.rand(1, 3, 128, 128))
    boxes = out["boxes"]
    # Boxes are xyxy, so the second corner is never before the first.
    assert bool((boxes[..., 2] >= boxes[..., 0]).all())
    assert bool((boxes[..., 3] >= boxes[..., 1]).all())
    # At initialisation every box is small and near its own grid point, so
    # nothing should be wildly outside the image.
    assert float(boxes.abs().max()) < 4 * 128


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_builds_and_runs(variant):
    model = GUSNet.from_variant(variant, num_classes=2).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 64))
    assert out["cls_logits"].shape == (1, 64 + 16 + 4, 2)


def test_variants_are_ordered_by_size():
    counts = [GUSNet.from_variant(v, num_classes=2).num_parameters() for v in "nsmlx"]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1] / 5, "the family should span a real range of sizes"


def test_variant_name_accepts_the_prefixed_spelling():
    assert GUSNet.from_variant("GUSNet-n", num_classes=2).num_parameters() == (
        GUSNet.from_variant("n", num_classes=2).num_parameters()
    )


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        GUSNet.from_variant("xxl")


def test_input_must_be_divisible_by_the_largest_stride(tiny_model: GUSNet):
    with pytest.raises(ValueError, match="not divisible by the largest stride"):
        tiny_model(torch.rand(1, 3, 100, 128))


def test_input_must_be_four_dimensional(tiny_model: GUSNet):
    with pytest.raises(ValueError, match=r"expected a \(B, C, H, W\) tensor"):
        tiny_model(torch.rand(3, 64, 64))


def test_class_names_must_match_the_class_count():
    with pytest.raises(ValueError, match="class names for"):
        GUSNet(num_classes=3, class_names=["a", "b"])


def test_class_names_are_carried_with_the_model():
    model = GUSNet(num_classes=2, width=0.25, depth=0.34, class_names=["cat", "dog"])
    assert model.class_names == ["cat", "dog"]


def test_gradients_reach_every_part_of_the_network():
    model = GUSNet.from_variant("n", num_classes=3)
    out = model(torch.rand(1, 3, 64, 64))
    (out["cls_logits"].sum() + out["boxes"].sum()).backward()

    without_grad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or not bool(param.grad.abs().sum()))
    ]
    assert not without_grad, f"no gradient reached: {without_grad[:5]}"


def test_eval_mode_is_deterministic(tiny_model: GUSNet):
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(tiny_model(x)["boxes"], tiny_model(x)["boxes"])


def test_state_dict_roundtrip(tiny_model: GUSNet):
    clone = GUSNet.from_variant("n", num_classes=4).eval()
    clone.load_state_dict(tiny_model.state_dict())

    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(clone(x)["boxes"], tiny_model(x)["boxes"], atol=1e-6)


def test_the_dataloader_feeds_the_model(folder_dataset):
    """Phase 1 and phases 2-3 have to meet: a real batch must go straight in."""
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=128, augment=False)
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = GUSNet.from_variant("n", num_classes=dataset.num_classes).eval()
    with torch.no_grad():
        out = model(batch["images"])

    assert out["cls_logits"].shape[0] == batch["images"].shape[0]
    assert out["cls_logits"].shape[2] == dataset.num_classes
