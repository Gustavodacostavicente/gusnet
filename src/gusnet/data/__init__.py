# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Datasets, augmentations and batching."""

from __future__ import annotations

from gusnet.data.dataset import Annotation, AugmentConfig, DetectionDataset
from gusnet.data.letterbox import LetterboxResult, letterbox, letterbox_boxes
from gusnet.data.loader import build_dataloader, collate_detection
from gusnet.data.transforms import (
    filter_boxes,
    hsv_augment,
    mixup,
    mosaic4,
    random_affine,
    random_hflip,
)

__all__ = [
    "Annotation",
    "AugmentConfig",
    "DetectionDataset",
    "LetterboxResult",
    "build_dataloader",
    "collate_detection",
    "filter_boxes",
    "hsv_augment",
    "letterbox",
    "letterbox_boxes",
    "mixup",
    "mosaic4",
    "random_affine",
    "random_hflip",
]
