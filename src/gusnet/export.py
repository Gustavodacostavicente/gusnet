# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Taking a trained model out of PyTorch.

A deployment runtime wants a graph of tensor operations, not a Python object.
Two things stand between GUSNet and that graph.

**The output is a dict.** Convenient in Python, not expressible in ONNX and not
stable under tracing. :class:`ExportWrapper` flattens it to a tuple of tensors
and drops what inference does not need — the raw logits and the anchor points
exist for the loss, not for a consumer.

**Suppression has a data-dependent output size**, and that is the hard part.
The number of detections depends on the pixels, so a plain trace records
whatever count the example input happened to produce and bakes it in — an
exported model that silently returns the wrong number of boxes for every other
image. Both export paths here avoid that, in different ways:

* **TorchScript** traces the network and then *scripts* a small suppression
  module around it. Scripted code keeps real control flow and real dynamic
  shapes; traced code does not.
* **ONNX** uses PyTorch's legacy tracing exporter for this one case, because it
  lowers suppression to the ONNX ``NonMaxSuppression`` operator, which is
  dynamic by construction. The current dynamo-based exporter cannot represent a
  data-dependent output size at all and fails outright.

Folding NMS in is off by default regardless: it freezes the thresholds into the
file, and a threshold that can only be changed by re-exporting is a threshold
nobody tunes.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.ops import batched_nms

__all__ = [
    "ExportWrapper",
    "SuppressedModel",
    "benchmark",
    "export_onnx",
    "export_torchscript",
]


class ExportWrapper(nn.Module):
    """A GUSNet that returns tensors instead of a dict.

    Output is ``(boxes, scores)`` of ``(B, A, 4)`` and ``(B, A, C)``, with
    boxes as xyxy in the coordinates of the letterboxed input. Suppression and
    the mapping back to original image coordinates are the consumer's job.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        output = self.model(images)
        return output["boxes"], output["scores"]


class SuppressedModel(nn.Module):
    """``ExportWrapper`` plus non-maximum suppression, written to be scriptable.

    Output is one ``(N, 6)`` tensor of ``[x1, y1, x2, y2, score, class]``, with
    ``N`` genuinely varying per input. Batch size 1 only: the number of
    detections differs per image, and a ragged batch has no tensor
    representation.

    Every construct in ``forward`` is deliberately one TorchScript accepts —
    explicit annotations, no Python containers, no branching on tensor values.
    """

    def __init__(
        self,
        core: nn.Module,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ) -> None:
        super().__init__()
        self.core = core
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)

    def forward(self, images: Tensor) -> Tensor:
        boxes, scores = self.core(images)

        best_scores, labels = scores[0].max(dim=-1)
        keep = best_scores > self.conf_threshold
        candidates = boxes[0][keep]
        candidate_scores = best_scores[keep]
        candidate_labels = labels[keep]

        selected = batched_nms(candidates, candidate_scores, candidate_labels, self.iou_threshold)
        selected = selected[: self.max_det]

        return torch.cat(
            (
                candidates[selected],
                candidate_scores[selected].unsqueeze(1),
                candidate_labels[selected].unsqueeze(1).to(candidates.dtype),
            ),
            dim=1,
        )


def export_torchscript(
    model: nn.Module,
    path: str | Path,
    *,
    img_size: int = 640,
    nms: bool = False,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
) -> Path:
    """Export to TorchScript, runnable from C++ with no Python at all.

    The network itself is traced: its forward pass has no data-dependent
    control flow, and a trace is simpler and faster than a script. When ``nms``
    is on, the suppression module wrapped around it is *scripted* instead, so
    the detection count stays dynamic.

    Args:
        model: the detector.
        path: destination ``.torchscript`` file.
        img_size: the size the trace is taken at.
        nms: fold suppression into the graph (batch size 1 only).
        conf_threshold: baked-in confidence floor, with ``nms``.
        iou_threshold: baked-in NMS threshold, with ``nms``.
        max_det: baked-in detection cap, with ``nms``.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    core = ExportWrapper(model).eval()
    example = torch.zeros(1, 3, img_size, img_size)

    with torch.no_grad():
        traced = torch.jit.trace(core, example, strict=False)
        if nms:
            exported = torch.jit.script(
                SuppressedModel(traced, conf_threshold, iou_threshold, max_det).eval()
            )
        else:
            exported = traced

    exported.save(str(path))
    return path


def export_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    img_size: int = 640,
    opset: int = 17,
    dynamic_batch: bool = False,
    nms: bool = False,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
) -> Path:
    """Export to ONNX.

    Args:
        model: the detector.
        path: destination ``.onnx`` file.
        img_size: the size the graph is built for.
        opset: ONNX opset version.
        dynamic_batch: make the batch dimension dynamic. Incompatible with
            ``nms``, which is inherently per image.
        nms: fold suppression into the graph (batch size 1 only).
        conf_threshold: baked-in confidence floor, with ``nms``.
        iou_threshold: baked-in NMS threshold, with ``nms``.
        max_det: baked-in detection cap, with ``nms``.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if nms and dynamic_batch:
        raise ValueError("dynamic_batch and nms cannot be combined: NMS is per image")

    core = ExportWrapper(model).eval()
    graph: nn.Module = core
    output_names = ["boxes", "scores"]
    if nms:
        graph = SuppressedModel(core, conf_threshold, iou_threshold, max_det).eval()
        output_names = ["detections"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}}
        dynamic_axes.update({name: {0: "batch"} for name in output_names})

    example = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        torch.onnx.export(
            graph,
            example,
            str(path),
            input_names=["images"],
            output_names=output_names,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            # The dynamo exporter cannot express a data-dependent output size,
            # so the suppression path goes through the legacy tracer, which
            # lowers it to the ONNX NonMaxSuppression operator.
            dynamo=not nms,
        )
    return path


@torch.no_grad()
def benchmark(
    model: nn.Module,
    *,
    img_size: int = 640,
    batch_size: int = 1,
    iterations: int = 50,
    warmup: int = 10,
    device: str = "auto",
    half: bool = False,
) -> dict[str, float]:
    """Measure forward-pass latency.

    The warmup passes are not optional noise reduction: the first CUDA call
    initialises the context, and cuDNN picks its convolution algorithms over
    the first few passes. Timing those measures the setup, not the model.

    Args:
        model: the detector.
        img_size: input resolution.
        batch_size: images per forward pass.
        iterations: timed passes.
        warmup: untimed passes first.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ...
        half: run in float16 on CUDA.

    Returns:
        ``{"ms_per_image", "ms_per_batch", "fps", "batch_size", "img_size"}``.
    """
    resolved = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved).eval()

    use_half = bool(half and resolved.type == "cuda")
    if use_half:
        model = model.half()

    images = torch.zeros(batch_size, 3, img_size, img_size, device=resolved)
    if use_half:
        images = images.half()

    for _ in range(max(warmup, 0)):
        model(images)
    _synchronise(resolved)

    started = time.perf_counter()
    for _ in range(max(iterations, 1)):
        model(images)
    _synchronise(resolved)
    elapsed = time.perf_counter() - started

    per_batch = elapsed / max(iterations, 1)
    per_image = per_batch / max(batch_size, 1)
    return {
        "ms_per_image": per_image * 1000,
        "ms_per_batch": per_batch * 1000,
        "fps": 1.0 / per_image,
        "batch_size": float(batch_size),
        "img_size": float(img_size),
    }


def _synchronise(device: torch.device) -> None:
    """CUDA is asynchronous; without this the timer measures queueing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
