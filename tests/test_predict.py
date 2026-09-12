# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for inference on real inputs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from gusnet.models import GUSNet
from gusnet.predict import PredictConfig, Predictor, iter_source


def _image(height: int = 90, width: int = 160) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (60, 70), (200, 60, 60), -1)
    return image


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "images"
    folder.mkdir()
    for index in range(5):
        cv2.imwrite(str(folder / f"shot{index}.jpg"), _image(90 + index * 10, 160))
    return folder


# ------------------------------------------------------------------ the source


def test_iter_source_reads_a_single_image(image_folder: Path):
    name, image = next(iter(iter_source(image_folder / "shot0.jpg")))
    assert name == "shot0"
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8


def test_iter_source_reads_a_folder_in_order(image_folder: Path):
    names = [name for name, _ in iter_source(image_folder)]
    assert names == ["shot0", "shot1", "shot2", "shot3", "shot4"]


def test_iter_source_converts_to_rgb(tmp_path: Path):
    """OpenCV reads BGR; everything inside GUSNet is RGB."""
    path = tmp_path / "blue.png"
    cv2.imwrite(str(path), np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8))  # BGR blue
    _, image = next(iter(iter_source(path)))
    assert image[0, 0].tolist() == [0, 0, 255]  # RGB blue


def test_iter_source_rejects_a_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no such file or directory"):
        next(iter(iter_source(tmp_path / "nope.jpg")))


def test_iter_source_rejects_an_empty_folder(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no images found"):
        next(iter(iter_source(tmp_path / "empty")))


# --------------------------------------------------------------- the predictor


class _FixedBoxModel(nn.Module):
    """Reports one box at a known place in letterboxed coordinates."""

    num_classes = 2
    class_names = ["alpha", "beta"]

    def __init__(self, box: list[float]) -> None:
        super().__init__()
        self.box = box

    def forward(self, images: torch.Tensor) -> dict:
        batch = images.shape[0]
        boxes = torch.tensor([self.box]).expand(batch, 1, 4).clone()
        scores = torch.zeros(batch, 1, self.num_classes)
        scores[:, 0, 1] = 0.9
        return {"boxes": boxes, "scores": scores}


def test_predictor_maps_boxes_back_to_the_source_image():
    """Nothing outside the predictor should ever see letterboxed coordinates."""
    image = _image(90, 160)  # letterboxed to 128 with ratio 0.8, pad y = 28
    model = _FixedBoxModel([16.0, 44.0, 48.0, 84.0])

    predictor = Predictor(
        model, config=PredictConfig(img_size=128, conf_threshold=0.5, device="cpu")
    )
    found = predictor.predict_one(image)

    assert len(found) == 1
    # (16 - 0) / 0.8 = 20 ; (44 - 28) / 0.8 = 20 ; (48) / 0.8 = 60 ; (84 - 28) / 0.8 = 70
    assert found.boxes[0].tolist() == pytest.approx([20.0, 20.0, 60.0, 70.0], abs=1.0)
    assert found.labels.tolist() == [1]


def test_predictor_never_upscales():
    """Enlarging a small image adds no information and costs accuracy.

    A 40x40 image letterboxed to 320 keeps ratio 1.0 and is padded by 140 px on
    every side, so the content sits at [140, 180]. A box drawn exactly around
    that content must come back as the whole image. Had the letterbox scaled
    the image up 8x instead, the same box would map back to an eighth of it.
    """
    small = _image(40, 40)
    model = _FixedBoxModel([140.0, 140.0, 180.0, 180.0])
    predictor = Predictor(model, config=PredictConfig(img_size=320, device="cpu"))

    found = predictor.predict_one(small)
    assert found.boxes[0].tolist() == pytest.approx([0.0, 0.0, 40.0, 40.0], abs=1.0)


def test_predictor_handles_a_batch_of_different_sizes():
    model = _FixedBoxModel([10.0, 10.0, 50.0, 50.0])
    predictor = Predictor(model, config=PredictConfig(img_size=128, device="cpu"))

    results = predictor.predict([_image(90, 160), _image(200, 100), _image(64, 64)])
    assert len(results) == 3
    assert all(len(found) == 1 for found in results)
    # Each image keeps its own ratio and padding, so the boxes differ.
    assert results[0].boxes[0].tolist() != results[1].boxes[0].tolist()


def test_predictor_on_an_empty_list():
    model = _FixedBoxModel([0.0, 0.0, 1.0, 1.0])
    assert Predictor(model, config=PredictConfig(device="cpu")).predict([]) == []


def test_predictor_run_covers_every_input(image_folder: Path):
    model = _FixedBoxModel([10.0, 10.0, 50.0, 50.0])
    predictor = Predictor(model, config=PredictConfig(img_size=128, batch_size=2, device="cpu"))
    results = list(predictor.run(image_folder))
    assert len(results) == 5
    assert [name for name, _, _ in results] == [f"shot{i}" for i in range(5)]


def test_predictor_takes_class_names_from_the_model():
    predictor = Predictor(_FixedBoxModel([0.0, 0.0, 1.0, 1.0]), config=PredictConfig(device="cpu"))
    assert predictor.class_names == ["alpha", "beta"]


def test_predictor_annotate_does_not_touch_the_input():
    image = _image()
    model = _FixedBoxModel([10.0, 10.0, 50.0, 50.0])
    predictor = Predictor(model, config=PredictConfig(img_size=128, device="cpu"))

    before = image.copy()
    annotated = predictor.annotate(image, predictor.predict_one(image))
    assert np.array_equal(image, before)
    assert not np.array_equal(annotated, image)


def test_predictor_runs_a_real_model(image_folder: Path):
    torch.manual_seed(0)
    model = GUSNet.from_variant("n", num_classes=2)
    predictor = Predictor(
        model, config=PredictConfig(img_size=64, conf_threshold=0.3, device="cpu")
    )

    for _, image, found in predictor.run(image_folder):
        height, width = image.shape[:2]
        if len(found):
            # Boxes are clipped to the source image, never beyond it.
            assert float(found.boxes[:, 0].min()) >= 0.0
            assert float(found.boxes[:, 2].max()) <= width
            assert float(found.boxes[:, 3].max()) <= height


def test_predictor_loads_from_a_checkpoint(tmp_path: Path, image_folder: Path):
    from gusnet.train import save_checkpoint

    model = GUSNet.from_variant("n", num_classes=2, class_names=["alpha", "beta"])
    path = save_checkpoint(tmp_path / "ckpt.pt", model)

    predictor = Predictor.from_checkpoint(path, config=PredictConfig(img_size=64, device="cpu"))
    assert predictor.class_names == ["alpha", "beta"]
    assert len(list(predictor.run(image_folder))) == 5
