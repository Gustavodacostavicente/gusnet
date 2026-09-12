# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the dataset readers, the item pipeline and batching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gusnet.data import AugmentConfig, DetectionDataset, build_dataloader, collate_detection
from gusnet.ops.boxes import scale_boxes

# ------------------------------------------------------------------- readers


def test_from_folder_reads_images_labels_and_classes(folder_dataset: Path):
    ds = DetectionDataset.from_folder(folder_dataset, "train", img_size=320)
    assert len(ds) == 6
    assert ds.class_names == ["square", "circle"]
    assert ds.num_classes == 2

    ann = ds.annotations[0]
    assert ann.boxes.shape == (2, 4)
    assert ann.classes.tolist() == [0, 1]
    # Stored normalised, so every coordinate is in [0, 1].
    assert ann.boxes.min() >= 0.0
    assert ann.boxes.max() <= 1.0


def test_from_folder_needs_class_names(folder_dataset: Path):
    (folder_dataset / "classes.txt").unlink()
    with pytest.raises(FileNotFoundError, match="no class_names given"):
        DetectionDataset.from_folder(folder_dataset, "train")


def test_from_folder_rejects_a_missing_split(folder_dataset: Path):
    with pytest.raises(FileNotFoundError, match="no image directory"):
        DetectionDataset.from_folder(folder_dataset, "nope")


def test_missing_label_file_is_a_negative_sample(folder_dataset: Path):
    (folder_dataset / "labels" / "train" / "img000.txt").unlink()
    ds = DetectionDataset.from_folder(folder_dataset, "train")
    assert len(ds.annotations[0].boxes) == 0
    item = ds[0]
    assert item["boxes"].shape == (0, 4)


def test_malformed_label_line_is_reported(folder_dataset: Path):
    bad = folder_dataset / "labels" / "train" / "img001.txt"
    bad.write_text("0 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 5 fields"):
        DetectionDataset.from_folder(folder_dataset, "train")


def test_from_coco_remaps_category_ids(coco_dataset: tuple[Path, Path]):
    images_dir, ann_file = coco_dataset
    ds = DetectionDataset.from_coco(images_dir, ann_file, img_size=320)
    assert len(ds) == 4
    # COCO ids 10 and 25 become contiguous 0 and 1.
    assert ds.class_names == ["square", "circle"]
    assert sorted(set(ds.annotations[0].classes.tolist())) == [0, 1]
    assert ds.annotations[0].image_id == 1


def test_coco_boxes_convert_from_xywh_to_normalised_xyxy(coco_dataset: tuple[Path, Path]):
    images_dir, ann_file = coco_dataset
    ds = DetectionDataset.from_coco(images_dir, ann_file)
    boxes = ds.annotations[0].boxes
    assert np.all(boxes[:, 2] > boxes[:, 0])
    assert np.all(boxes[:, 3] > boxes[:, 1])
    assert boxes.max() <= 1.0


# -------------------------------------------------------------- item pipeline


def test_item_without_augmentation_is_letterboxed(folder_dataset: Path):
    ds = DetectionDataset.from_folder(folder_dataset, "train", img_size=320, augment=False)
    item = ds[0]

    assert item["image"].shape == (3, 320, 320)
    assert item["image"].dtype == torch.float32
    assert float(item["image"].min()) >= 0.0 and float(item["image"].max()) <= 1.0
    assert item["boxes"].shape == (2, 4)
    assert item["classes"].dtype == torch.int64
    assert item["orig_shape"] == (240, 320)


def test_boxes_map_back_to_the_original_image(folder_dataset: Path):
    ds = DetectionDataset.from_folder(folder_dataset, "train", img_size=416, augment=False)
    item = ds[0]
    ann = ds.annotations[0]
    height, width = item["orig_shape"]

    expected = ann.boxes.copy()
    expected[:, 0::2] *= width
    expected[:, 1::2] *= height

    recovered = scale_boxes(item["boxes"], item["ratio"], item["pad"], item["orig_shape"])
    assert np.allclose(recovered.numpy(), expected, atol=1.0)


def test_augmented_item_has_the_training_size(folder_dataset: Path):
    ds = DetectionDataset.from_folder(
        folder_dataset,
        "train",
        img_size=256,
        augment=True,
        config=AugmentConfig(mosaic=1.0, mixup=1.0),
        seed=0,
    )
    for index in range(len(ds)):
        item = ds[index]
        assert item["image"].shape == (3, 256, 256)
        assert item["boxes"].shape[0] == item["classes"].shape[0]
        if len(item["boxes"]):
            assert float(item["boxes"].min()) >= 0.0
            assert float(item["boxes"].max()) <= 256.0


def test_mosaic_disabled_falls_back_to_letterbox(folder_dataset: Path):
    ds = DetectionDataset.from_folder(
        folder_dataset,
        "train",
        img_size=256,
        augment=True,
        config=AugmentConfig(mosaic=0.0, mixup=0.0),
        seed=0,
    )
    item = ds[0]
    assert item["image"].shape == (3, 256, 256)
    assert item["orig_shape"] == (240, 320)


def test_seeded_dataset_is_reproducible(folder_dataset: Path):
    def first_item():
        ds = DetectionDataset.from_folder(
            folder_dataset, "train", img_size=256, augment=True, seed=42
        )
        return ds[0]

    a, b = first_item(), first_item()
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["boxes"], b["boxes"])


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="dataset is empty"):
        DetectionDataset([], ["a"])


# -------------------------------------------------------------------- batching


def test_collate_builds_flat_targets(folder_dataset: Path):
    ds = DetectionDataset.from_folder(folder_dataset, "train", img_size=320, augment=False)
    batch = collate_detection([ds[i] for i in range(4)])

    assert batch["images"].shape == (4, 3, 320, 320)
    assert len(batch["boxes"]) == 4
    assert batch["targets"].shape[1] == 6

    total = sum(len(b) for b in batch["boxes"])
    assert batch["targets"].shape[0] == total
    # Column 0 is the index of the image inside the batch.
    assert set(batch["targets"][:, 0].tolist()) <= {0.0, 1.0, 2.0, 3.0}
    # Column 1 holds class ids.
    assert set(batch["targets"][:, 1].tolist()) <= {0.0, 1.0}


def test_collate_handles_a_batch_with_no_objects(folder_dataset: Path):
    for path in (folder_dataset / "labels" / "train").glob("*.txt"):
        path.write_text("", encoding="utf-8")
    ds = DetectionDataset.from_folder(folder_dataset, "train", img_size=128, augment=False)
    batch = collate_detection([ds[0], ds[1]])
    assert batch["targets"].shape == (0, 6)


def test_dataloader_yields_usable_batches(folder_dataset: Path):
    ds = DetectionDataset.from_folder(folder_dataset, "train", img_size=192, augment=True, seed=1)
    loader = build_dataloader(ds, batch_size=3, shuffle=False, num_workers=0)

    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        assert batch["images"].shape == (3, 3, 192, 192)
        assert len(batch["image_ids"]) == 3
