# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for export and the latency benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gusnet.export import (
    ExportWrapper,
    SuppressedModel,
    benchmark,
    export_onnx,
    export_torchscript,
)
from gusnet.models import GUSNet

SIZE = 64


@pytest.fixture(scope="module")
def model() -> GUSNet:
    torch.manual_seed(0)
    return GUSNet.from_variant("n", num_classes=3).eval()


# ------------------------------------------------------------------ the wrapper


def test_export_wrapper_returns_tensors_not_a_dict(model: GUSNet):
    """A dict output is convenient in Python and inexpressible in ONNX."""
    with torch.no_grad():
        output = ExportWrapper(model).eval()(torch.rand(1, 3, SIZE, SIZE))

    assert isinstance(output, tuple)
    boxes, scores = output
    anchors = sum((SIZE // stride) ** 2 for stride in model.strides)
    assert boxes.shape == (1, anchors, 4)
    assert scores.shape == (1, anchors, 3)


def test_suppressed_model_returns_one_row_per_detection(model: GUSNet):
    suppressed = SuppressedModel(ExportWrapper(model).eval(), conf_threshold=0.0).eval()
    with torch.no_grad():
        detections = suppressed(torch.rand(1, 3, SIZE, SIZE))

    assert detections.ndim == 2
    assert detections.shape[1] == 6  # x1, y1, x2, y2, score, class
    assert float(detections[:, 4].max()) <= 1.0


# -------------------------------------------------------------------- TorchScript


def test_torchscript_export_matches_pytorch_exactly(model: GUSNet, tmp_path: Path):
    path = export_torchscript(model, tmp_path / "m.torchscript", img_size=SIZE)
    assert path.is_file()

    images = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        reference = ExportWrapper(model).eval()(images)
        loaded = torch.jit.load(str(path))(images)

    assert torch.equal(loaded[0], reference[0])
    assert torch.equal(loaded[1], reference[1])


def test_torchscript_with_nms_keeps_the_count_dynamic(model: GUSNet, tmp_path: Path):
    """The reason suppression is scripted rather than traced.

    A trace records whatever detection count the example input produced and
    bakes it in, so every other image gets the wrong number of boxes back --
    silently. Scripted code keeps the real dynamic shape.
    """
    path = export_torchscript(
        model, tmp_path / "nms.torchscript", img_size=SIZE, nms=True, conf_threshold=0.0
    )
    loaded = torch.jit.load(str(path))

    counts = []
    for seed in range(4):
        torch.manual_seed(seed)
        with torch.no_grad():
            detections = loaded(torch.rand(1, 3, SIZE, SIZE))
        assert detections.shape[1] == 6
        counts.append(int(detections.shape[0]))

    assert len(set(counts)) > 1, f"the count was baked in: {counts}"


def test_torchscript_nms_agrees_with_the_python_path(model: GUSNet, tmp_path: Path):
    from gusnet.eval import non_max_suppression

    torch.manual_seed(7)
    images = torch.rand(1, 3, SIZE, SIZE)
    path = export_torchscript(
        model, tmp_path / "nms.torchscript", img_size=SIZE, nms=True, conf_threshold=0.02
    )

    with torch.no_grad():
        exported = torch.jit.load(str(path))(images)
        expected = non_max_suppression(model(images), conf_threshold=0.02)[0]

    assert exported.shape[0] == len(expected)
    assert torch.allclose(exported[:, :4], expected.boxes, atol=1e-4)


# --------------------------------------------------------------------------- ONNX


@pytest.fixture
def onnxruntime():
    return pytest.importorskip("onnxruntime", reason="install the 'export' extra")


def test_onnx_export_matches_pytorch(model: GUSNet, tmp_path: Path, onnxruntime):
    import numpy as np

    path = export_onnx(model, tmp_path / "m.onnx", img_size=SIZE)
    assert path.is_file()

    images = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        reference = ExportWrapper(model).eval()(images)

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    boxes, scores = session.run(None, {"images": images.numpy()})

    # Not bit-exact: the ONNX graph optimiser reassociates float arithmetic.
    assert np.abs(boxes - reference[0].numpy()).max() < 1e-3
    assert np.abs(scores - reference[1].numpy()).max() < 1e-4


def test_onnx_with_nms_keeps_the_count_dynamic(model: GUSNet, tmp_path: Path, onnxruntime):
    import numpy as np

    path = export_onnx(model, tmp_path / "nms.onnx", img_size=SIZE, nms=True, conf_threshold=0.0)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    counts = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        images = rng.random((1, 3, SIZE, SIZE), dtype=np.float32)
        detections = session.run(None, {"images": images})[0]
        assert detections.shape[1] == 6
        counts.append(int(detections.shape[0]))

    assert len(set(counts)) > 1, f"the count was baked in: {counts}"


def test_onnx_dynamic_batch(model: GUSNet, tmp_path: Path, onnxruntime):
    import numpy as np

    path = export_onnx(model, tmp_path / "dyn.onnx", img_size=SIZE, dynamic_batch=True)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for batch in (1, 3):
        images = np.zeros((batch, 3, SIZE, SIZE), dtype=np.float32)
        boxes, _ = session.run(None, {"images": images})
        assert boxes.shape[0] == batch


def test_onnx_rejects_dynamic_batch_with_nms(model: GUSNet, tmp_path: Path):
    with pytest.raises(ValueError, match="cannot be combined"):
        export_onnx(model, tmp_path / "bad.onnx", img_size=SIZE, dynamic_batch=True, nms=True)


# ---------------------------------------------------------------------- benchmark


def test_benchmark_reports_consistent_numbers(model: GUSNet):
    result = benchmark(model, img_size=SIZE, batch_size=2, iterations=3, warmup=1, device="cpu")

    assert set(result) == {"ms_per_image", "ms_per_batch", "fps", "batch_size", "img_size"}
    assert result["ms_per_image"] > 0
    assert result["ms_per_batch"] == pytest.approx(result["ms_per_image"] * 2, rel=1e-6)
    assert result["fps"] == pytest.approx(1000.0 / result["ms_per_image"], rel=1e-6)


def test_benchmark_leaves_the_model_usable(model: GUSNet):
    benchmark(model, img_size=SIZE, iterations=2, warmup=0, device="cpu")
    with torch.no_grad():
        assert model(torch.rand(1, 3, SIZE, SIZE))["boxes"].shape[-1] == 4
