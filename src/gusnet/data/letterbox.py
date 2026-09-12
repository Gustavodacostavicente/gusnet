# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Aspect-ratio preserving resize with padding ("letterbox").

Detectors take a fixed-size square input, but images come in arbitrary shapes.
Squashing them distorts objects, so the image is scaled by a single factor and
the leftover area is filled with a neutral colour. The scale factor and the
padding are returned so that predictions can be mapped back to the original
image (see :func:`gusnet.ops.boxes.scale_boxes`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["LetterboxResult", "letterbox", "letterbox_boxes"]

PAD_VALUE = 114  # mid grey, the usual neutral fill for detection inputs


@dataclass(frozen=True)
class LetterboxResult:
    """Everything needed to undo a letterbox.

    Attributes:
        image: ``(H, W, 3)`` uint8 letterboxed image.
        ratio: the single scale factor applied to the original image.
        pad: ``(pad_x, pad_y)`` pixels added on the left and on the top.
        orig_shape: ``(height, width)`` of the input image.
    """

    image: np.ndarray
    ratio: float
    pad: tuple[float, float]
    orig_shape: tuple[int, int]


def letterbox(
    image: np.ndarray,
    new_shape: int | tuple[int, int] = 640,
    *,
    color: int | tuple[int, int, int] = PAD_VALUE,
    scaleup: bool = True,
    center: bool = True,
    stride: int | None = None,
) -> LetterboxResult:
    """Resize ``image`` into ``new_shape`` without changing its aspect ratio.

    Args:
        image: ``(H, W, 3)`` uint8 array.
        new_shape: target ``size`` or ``(height, width)``.
        color: fill value for the padded region.
        scaleup: if ``False`` the image is never enlarged, only shrunk. Use
            ``False`` at inference time: upscaling a small image adds no
            information and costs accuracy.
        center: pad both sides equally. When ``False`` the image is placed in
            the top-left corner, which is what mosaic tiles want.
        stride: if given, pad only up to the next multiple of ``stride``
            instead of filling ``new_shape`` completely ("rectangular"
            inference, cheaper for non-square images).

    Returns:
        A :class:`LetterboxResult`.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {image.shape}")

    orig_h, orig_w = image.shape[:2]
    target_h, target_w = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape

    ratio = min(target_h / orig_h, target_w / orig_w)
    if not scaleup:
        ratio = min(ratio, 1.0)

    # Size of the image content once scaled.
    content_w = int(round(orig_w * ratio))
    content_h = int(round(orig_h * ratio))

    pad_w = target_w - content_w
    pad_h = target_h - content_h
    if stride is not None:
        pad_w = int(np.mod(pad_w, stride))
        pad_h = int(np.mod(pad_h, stride))

    if center:
        pad_left, pad_top = pad_w / 2, pad_h / 2
    else:
        pad_left, pad_top = 0.0, 0.0

    if (orig_w, orig_h) != (content_w, content_h):
        # INTER_AREA is the right filter when shrinking; it avoids aliasing.
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (content_w, content_h), interpolation=interp)

    top = int(round(pad_top - 0.1))
    left = int(round(pad_left - 0.1))
    bottom = pad_h - top
    right = pad_w - left

    fill = (color, color, color) if isinstance(color, int) else color
    out = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
    return LetterboxResult(
        image=out,
        ratio=ratio,
        pad=(float(left), float(top)),
        orig_shape=(orig_h, orig_w),
    )


def letterbox_boxes(boxes: np.ndarray, result: LetterboxResult) -> np.ndarray:
    """Apply the same transform to ``(N, 4)`` xyxy boxes in original coordinates."""
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    pad_x, pad_y = result.pad
    out = boxes.astype(np.float32).copy()
    out[:, 0::2] = out[:, 0::2] * result.ratio + pad_x
    out[:, 1::2] = out[:, 1::2] * result.ratio + pad_y
    return out
