# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Synthetic datasets so the tests never need a download."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

CLASS_NAMES = ["square", "circle"]


def _draw_sample(
    width: int, height: int, seed: int
) -> tuple[np.ndarray, list[tuple[int, list[float]]]]:
    """An image with two coloured shapes and their ground-truth boxes."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    objects: list[tuple[int, list[float]]] = []

    for cls in range(2):
        w = int(rng.integers(width // 8, width // 4))
        h = int(rng.integers(height // 8, height // 4))
        x1 = int(rng.integers(0, width - w))
        y1 = int(rng.integers(0, height - h))
        x2, y2 = x1 + w, y1 + h
        color = (220, 60, 60) if cls == 0 else (60, 220, 60)
        if cls == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.ellipse(
                image, ((x1 + x2) // 2, (y1 + y2) // 2), (w // 2, h // 2), 0, 0, 360, color, -1
            )
        objects.append((cls, [float(x1), float(y1), float(x2), float(y2)]))

    return image, objects


@pytest.fixture
def folder_dataset(tmp_path: Path) -> Path:
    """A dataset in the ``images/`` + ``labels/`` layout with 6 training images."""
    root = tmp_path / "synthetic"
    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(6):
        width, height = (320 + 40 * i, 240 + 20 * i)
        image, objects = _draw_sample(width, height, seed=i)
        cv2.imwrite(str(images_dir / f"img{i:03d}.jpg"), image)

        lines = []
        for cls, (x1, y1, x2, y2) in objects:
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (labels_dir / f"img{i:03d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def coco_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """The same content, expressed as a COCO instances JSON file."""
    root = tmp_path / "coco"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    for i in range(4):
        width, height = (320, 240)
        image, objects = _draw_sample(width, height, seed=100 + i)
        name = f"coco{i:03d}.jpg"
        cv2.imwrite(str(images_dir / name), image)
        images.append({"id": i + 1, "file_name": name, "width": width, "height": height})

        for cls, (x1, y1, x2, y2) in objects:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i + 1,
                    # COCO category ids are 1-based and may have gaps.
                    "category_id": 10 if cls == 0 else 25,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 10, "name": "square"},
            {"id": 25, "name": "circle"},
        ],
    }
    ann_file = root / "instances_train.json"
    ann_file.write_text(json.dumps(payload), encoding="utf-8")
    return images_dir, ann_file
