# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Running a trained model on real inputs.

Inference is training's pipeline with almost everything removed, and the few
things that stay differ in ways that matter:

* **no augmentation**, obviously, but also **no upscaling**. The letterbox runs
  with ``scaleup=False``: enlarging a small image adds no information and costs
  accuracy.
* **a high confidence threshold**. Evaluation uses 0.001 to trace the whole
  precision/recall curve; a person looking at results wants 0.25, not 300 boxes
  sorted by hope.
* **boxes mapped back to the source image**. Everything the model says is in
  letterboxed coordinates; nothing outside this module should ever see those.

A folder of images is batched even though the images have different shapes,
because the letterbox makes them all the same size before they are stacked --
each one keeps its own ratio and padding for the trip back.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from gusnet.data.dataset import IMAGE_SUFFIXES
from gusnet.data.letterbox import letterbox
from gusnet.eval.nms import Detections, non_max_suppression
from gusnet.viz import draw_boxes

__all__ = ["PredictConfig", "Predictor", "VIDEO_SUFFIXES", "iter_source"]

VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


@dataclass
class PredictConfig:
    """How to run inference.

    Attributes:
        img_size: the size images are letterboxed to. Should match training.
        conf_threshold: minimum confidence to report a detection.
        iou_threshold: NMS overlap threshold.
        max_det: cap on detections per image.
        batch_size: images per forward pass when a folder is given.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ...
        half: run in float16 on CUDA.
    """

    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    batch_size: int = 8
    device: str = "auto"
    half: bool = False

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_source(source: str | Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(name, image)`` from a file, a directory or a video.

    Images come out as ``(H, W, 3)`` uint8 **RGB**, which is the convention
    everywhere in GUSNet — OpenCV's BGR is converted at the boundary, once,
    rather than being allowed to leak inward.

    Args:
        source: an image file, a directory of images, or a video file.

    Yields:
        ``(name, image)``. For a video the name is the zero-padded frame index.
    """
    source = Path(source)

    if source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"no images found under {source}")
        for path in paths:
            yield path.stem, _read_image(path)
        return

    if not source.is_file():
        raise FileNotFoundError(f"no such file or directory: {source}")

    if source.suffix.lower() in VIDEO_SUFFIXES:
        yield from _iter_video(source)
        return

    yield source.stem, _read_image(source)


class Predictor:
    """A trained model, ready to run on arbitrary inputs.

    Args:
        model: the detector, already loaded with weights.
        class_names: names for the labels. Falls back to the model's own.
        config: inference settings.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        class_names: list[str] | None = None,
        config: PredictConfig | None = None,
    ) -> None:
        self.config = config or PredictConfig()
        self.device = self.config.resolved_device()

        self.model = model.to(self.device).eval()
        self.half = bool(self.config.half and self.device.type == "cuda")
        if self.half:
            self.model = self.model.half()

        self.class_names = class_names or getattr(model, "class_names", None)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        config: PredictConfig | None = None,
        prefer_ema: bool = True,
    ) -> Predictor:
        """Load a checkpoint and wrap it in a predictor.

        The averaged weights are preferred, since those are the ones that were
        evaluated and the ones meant to be deployed.
        """
        from gusnet.train import model_from_checkpoint

        model, _ = model_from_checkpoint(path, prefer_ema=prefer_ema)
        return cls(model, config=config)

    @torch.no_grad()
    def predict(self, images: list[np.ndarray]) -> list[Detections]:
        """Detect objects in a list of RGB images.

        Args:
            images: ``(H, W, 3)`` uint8 RGB arrays, any sizes.

        Returns:
            One :class:`~gusnet.eval.nms.Detections` per image, in that image's
            **own** pixel coordinates.
        """
        if not images:
            return []

        prepared = [letterbox(image, self.config.img_size, scaleup=False) for image in images]
        batch = torch.stack([_to_tensor(result.image) for result in prepared], dim=0).to(
            self.device
        )
        if self.half:
            batch = batch.half()

        output = self.model(batch)
        found = non_max_suppression(
            output,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            max_det=self.config.max_det,
        )

        return [
            detections.to("cpu").scale_to_original(result.ratio, result.pad, result.orig_shape)
            for detections, result in zip(found, prepared, strict=True)
        ]

    def predict_one(self, image: np.ndarray) -> Detections:
        """Detect objects in a single RGB image."""
        return self.predict([image])[0]

    def run(self, source: str | Path) -> Iterator[tuple[str, np.ndarray, Detections]]:
        """Detect over a file, a directory or a video, in batches.

        Yields:
            ``(name, image, detections)`` for each input, with the image in RGB
            and the detections in its own coordinates.
        """
        names: list[str] = []
        images: list[np.ndarray] = []

        for name, image in iter_source(source):
            names.append(name)
            images.append(image)
            if len(images) == self.config.batch_size:
                yield from zip(names, images, self.predict(images), strict=True)
                names, images = [], []

        if images:
            yield from zip(names, images, self.predict(images), strict=True)

    def annotate(self, image: np.ndarray, detections: Detections) -> np.ndarray:
        """Draw detections onto a copy of the image."""
        return draw_boxes(
            image,
            detections.boxes,
            detections.labels,
            class_names=self.class_names,
            scores=detections.scores,
        )


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _iter_video(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video {path}")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield f"frame_{index:06d}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).float().div_(255.0)
