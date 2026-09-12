# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Detection dataset.

The dataset is deliberately split in two layers:

:class:`Annotation`
    A pure description of one labelled image: a path, normalised boxes and
    class ids. Building this list is cheap and format-specific, so there is one
    small reader per format (:meth:`DetectionDataset.from_folder`,
    :meth:`DetectionDataset.from_coco`).
:class:`DetectionDataset`
    Turns annotations into training tensors: decoding, letterboxing, mosaic,
    affine, colour jitter and the conversion to ``torch``.

Boxes are stored **normalised xyxy** (all four values in ``[0, 1]``) because
that is the only representation that survives not knowing the image size until
it is decoded. They become absolute pixels as soon as the image is loaded.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gusnet.data.letterbox import letterbox, letterbox_boxes
from gusnet.data.transforms import (
    hsv_augment,
    mixup,
    mosaic4,
    random_affine,
    random_hflip,
)

__all__ = ["Annotation", "AugmentConfig", "DetectionDataset", "IMAGE_SUFFIXES"]

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


@dataclass
class Annotation:
    """One labelled image.

    Attributes:
        image_path: path to the image file.
        boxes: ``(N, 4)`` float32, **normalised xyxy** in ``[0, 1]``.
        classes: ``(N,)`` int64 class indices, zero-based.
        image_id: stable id used when writing evaluation results.
    """

    image_path: Path
    boxes: np.ndarray
    classes: np.ndarray
    image_id: int = 0

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        if len(self.boxes) != len(self.classes):
            raise ValueError(
                f"{self.image_path}: {len(self.boxes)} boxes but {len(self.classes)} classes"
            )


@dataclass
class AugmentConfig:
    """Augmentation strengths.

    Defaults follow what is standard for one-stage detectors at 640 px. The two
    probabilities at the top are the ones worth tuning first: mosaic is the
    single most effective augmentation here, and it is also the one that must be
    switched off for the last epochs of training so the model finishes on
    realistic images.
    """

    mosaic: float = 1.0
    mixup: float = 0.1
    hflip: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    mixup_alpha: float = 32.0


@dataclass
class _RawSample:
    image: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    orig_shape: tuple[int, int] = field(default=(0, 0))


class DetectionDataset(Dataset):
    """A detection dataset that yields letterboxed, augmented tensors.

    Each item is a ``dict`` with:

    ``image``
        ``(3, H, W)`` float32 in ``[0, 1]``, RGB.
    ``boxes``
        ``(N, 4)`` float32 xyxy in the coordinates of ``image``.
    ``classes``
        ``(N,)`` int64.
    ``image_id``
        int, for evaluation.
    ``ratio``, ``pad``, ``orig_shape``
        the letterbox parameters needed to map predictions back to the original
        image. Meaningless (``1.0`` / ``(0, 0)``) when mosaic produced the item,
        which is fine because mosaic only ever runs during training.
    """

    def __init__(
        self,
        annotations: Sequence[Annotation],
        class_names: Sequence[str],
        *,
        img_size: int = 640,
        augment: bool = False,
        config: AugmentConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if not annotations:
            raise ValueError("dataset is empty")
        self.annotations = list(annotations)
        self.class_names = list(class_names)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.config = config or AugmentConfig()
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ readers

    @classmethod
    def from_folder(
        cls,
        root: str | Path,
        split: str = "train",
        *,
        class_names: Sequence[str] | None = None,
        **kwargs,
    ) -> DetectionDataset:
        """Read the plain ``images/`` + ``labels/`` layout.

        ::

            root/
              images/<split>/frame001.jpg
              labels/<split>/frame001.txt
              classes.txt            # one class name per line (optional)

        Each label file holds one object per line, in normalised centre form::

            <class_index> <cx> <cy> <w> <h>

        all four coordinates in ``[0, 1]``. An empty or missing label file means
        a legitimate negative sample and is kept.

        Args:
            root: dataset root.
            split: subdirectory name under ``images/`` and ``labels/``.
            class_names: overrides ``classes.txt``.
            **kwargs: forwarded to :class:`DetectionDataset`.
        """
        root = Path(root)
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        if not images_dir.is_dir():
            raise FileNotFoundError(f"no image directory at {images_dir}")

        if class_names is None:
            names_file = root / "classes.txt"
            if not names_file.is_file():
                raise FileNotFoundError(
                    f"no class_names given and no {names_file}; one name per line"
                )
            class_names = [
                line.strip() for line in names_file.read_text("utf-8").splitlines() if line.strip()
            ]

        annotations: list[Annotation] = []
        paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for image_id, image_path in enumerate(paths):
            label_path = labels_dir / f"{image_path.stem}.txt"
            boxes, classes = _read_label_file(label_path)
            annotations.append(Annotation(image_path, boxes, classes, image_id))

        if not annotations:
            raise ValueError(f"no images found in {images_dir}")
        return cls(annotations, class_names, **kwargs)

    @classmethod
    def from_coco(
        cls,
        images_dir: str | Path,
        annotation_file: str | Path,
        *,
        drop_crowd: bool = True,
        **kwargs,
    ) -> DetectionDataset:
        """Read a COCO-style instances JSON file.

        COCO stores boxes as ``[x, y, width, height]`` in absolute pixels and
        numbers its categories with gaps, so category ids are remapped to a
        contiguous zero-based range in the order they appear in the file.

        Args:
            images_dir: directory holding the image files.
            annotation_file: path to ``instances_*.json``.
            drop_crowd: skip annotations flagged ``iscrowd`` (they have no usable
                box and are ignored by the COCO metric anyway).
            **kwargs: forwarded to :class:`DetectionDataset`.
        """
        images_dir = Path(images_dir)
        with Path(annotation_file).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        categories = sorted(raw["categories"], key=lambda c: c["id"])
        class_names = [c["name"] for c in categories]
        category_index = {c["id"]: i for i, c in enumerate(categories)}

        by_image: dict[int, list[dict]] = {}
        for ann in raw.get("annotations", []):
            if drop_crowd and ann.get("iscrowd", 0):
                continue
            by_image.setdefault(ann["image_id"], []).append(ann)

        annotations: list[Annotation] = []
        for image in raw["images"]:
            width = float(image["width"])
            height = float(image["height"])
            boxes: list[list[float]] = []
            classes: list[int] = []
            for ann in by_image.get(image["id"], []):
                x, y, w, h = (float(v) for v in ann["bbox"])
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x / width, y / height, (x + w) / width, (y + h) / height])
                classes.append(category_index[ann["category_id"]])
            annotations.append(
                Annotation(
                    image_path=images_dir / image["file_name"],
                    boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                    classes=np.array(classes, dtype=np.int64),
                    image_id=int(image["id"]),
                )
            )

        if not annotations:
            raise ValueError(f"no images listed in {annotation_file}")
        return cls(annotations, class_names, **kwargs)

    # ------------------------------------------------------------------ dataset

    def __len__(self) -> int:
        return len(self.annotations)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def load_raw(self, index: int) -> _RawSample:
        """Decode one image and return its boxes in absolute pixel coordinates."""
        ann = self.annotations[index]
        image = cv2.imread(str(ann.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read image {ann.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes = ann.boxes.copy()
        if len(boxes):
            boxes[:, 0::2] *= width
            boxes[:, 1::2] *= height
        return _RawSample(image, boxes, ann.classes.copy(), (height, width))

    def __getitem__(self, index: int) -> dict:
        cfg = self.config
        use_mosaic = self.augment and cfg.mosaic > 0 and self._rng.random() < cfg.mosaic

        if use_mosaic:
            image, boxes, classes = self._mosaic_sample(index)
            ratio, pad, orig_shape = 1.0, (0.0, 0.0), (self.img_size, self.img_size)

            if cfg.mixup > 0 and self._rng.random() < cfg.mixup:
                other = self._mosaic_sample(self._rng.randrange(len(self)))
                image, boxes, classes = mixup(
                    (image, boxes, classes), other, alpha=cfg.mixup_alpha, rng=self._np_rng
                )
        else:
            raw = self.load_raw(index)
            result = letterbox(raw.image, self.img_size, scaleup=self.augment)
            image = result.image
            boxes = letterbox_boxes(raw.boxes, result)
            classes = raw.classes
            ratio, pad, orig_shape = result.ratio, result.pad, result.orig_shape

        if self.augment:
            image = hsv_augment(
                image, hgain=cfg.hsv_h, sgain=cfg.hsv_s, vgain=cfg.hsv_v, rng=self._rng
            )
            image, boxes = random_hflip(image, boxes, p=cfg.hflip, rng=self._rng)

        return {
            "image": _to_tensor(image),
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)),
            "classes": torch.from_numpy(np.ascontiguousarray(classes, dtype=np.int64)),
            "image_id": self.annotations[index].image_id,
            "ratio": ratio,
            "pad": pad,
            "orig_shape": orig_shape,
        }

    def _mosaic_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build one mosaic and crop it back to ``img_size`` with a random affine."""
        indices = [index] + [self._rng.randrange(len(self)) for _ in range(3)]
        self._rng.shuffle(indices)
        tiles = []
        for i in indices:
            raw = self.load_raw(i)
            tiles.append((raw.image, raw.boxes, raw.classes))

        canvas, boxes, classes = mosaic4(tiles, self.img_size, rng=self._rng)
        cfg = self.config
        border = (-self.img_size // 2, -self.img_size // 2)
        return random_affine(
            canvas,
            boxes,
            classes,
            degrees=cfg.degrees,
            translate=cfg.translate,
            scale=cfg.scale,
            shear=cfg.shear,
            border=border,
            rng=self._rng,
        )


def _read_label_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``<class> <cx> <cy> <w> <h>`` lines into normalised xyxy."""
    if not path.is_file():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes: list[list[float]] = []
    classes: list[int] = []
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
        cls, cx, cy, w, h = parts
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        classes.append(int(cls))

    return (
        np.array(boxes, dtype=np.float32).reshape(-1, 4).clip(0.0, 1.0),
        np.array(classes, dtype=np.int64),
    )


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """``(H, W, 3)`` uint8 RGB -> ``(3, H, W)`` float32 in ``[0, 1]``."""
    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).float().div_(255.0)
