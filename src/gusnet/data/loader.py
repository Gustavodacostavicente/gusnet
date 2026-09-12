# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Batching for detection.

Detection batches are awkward because every image has a different number of
objects. Two representations are produced and both are kept in the batch:

``boxes`` / ``classes``
    lists of per-image tensors, natural for evaluation and visualisation;
``targets``
    a single ``(M, 6)`` tensor ``[batch_index, class, x1, y1, x2, y2]`` holding
    every object in the batch, which is what the label assigner and the loss
    actually want.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["build_dataloader", "collate_detection"]


def collate_detection(samples: list[dict]) -> dict:
    """Collate variable-length detection samples into one batch."""
    images = torch.stack([s["image"] for s in samples], dim=0)
    boxes = [s["boxes"] for s in samples]
    classes = [s["classes"] for s in samples]

    rows = []
    for index, (box, cls) in enumerate(zip(boxes, classes, strict=True)):
        if len(box) == 0:
            continue
        batch_index = torch.full((len(box), 1), float(index))
        rows.append(torch.cat((batch_index, cls.unsqueeze(1).float(), box), dim=1))

    targets = torch.cat(rows, dim=0) if rows else torch.zeros((0, 6), dtype=torch.float32)

    return {
        "images": images,
        "boxes": boxes,
        "classes": classes,
        "targets": targets,
        "image_ids": [s["image_id"] for s in samples],
        "ratios": [s["ratio"] for s in samples],
        "pads": [s["pad"] for s in samples],
        "orig_shapes": [s["orig_shape"] for s in samples],
    }


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    persistent_workers: bool | None = None,
) -> DataLoader:
    """Create a ``DataLoader`` wired up for detection batches.

    Args:
        dataset: a :class:`~gusnet.data.dataset.DetectionDataset`.
        batch_size: images per batch.
        shuffle: shuffle each epoch. Turn off for validation.
        num_workers: worker processes. ``0`` keeps decoding in the main process,
            which is the right default on Windows, where each worker pays the
            cost of re-importing the package.
        pin_memory: defaults to ``True`` when CUDA is available.
        drop_last: drop a trailing partial batch (keeps BatchNorm statistics
            stable during training).
        persistent_workers: defaults to ``True`` whenever workers are used.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_detection,
    )
