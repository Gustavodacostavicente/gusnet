# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""The GUSNet detector: backbone + neck + head.

One architecture, five sizes. The width multiplier scales channels and the
depth multiplier scales how many bottlenecks each CSP stage holds, so the same
code produces anything from a model that runs on a laptop CPU to one that needs
a real GPU.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from gusnet.nn.backbone import CSPBackbone
from gusnet.nn.head import DetectHead, DetectionOutput
from gusnet.nn.neck import PAFPN

__all__ = ["GUSNet", "VARIANTS"]

#: ``name -> (width multiplier, depth multiplier)``.
VARIANTS: dict[str, tuple[float, float]] = {
    "n": (0.25, 0.34),
    "s": (0.50, 0.34),
    "m": (0.75, 0.67),
    "l": (1.00, 1.00),
    "x": (1.25, 1.34),
}


class GUSNet(nn.Module):
    """One-stage anchor-free detector.

    Args:
        num_classes: number of object classes.
        width: channel multiplier.
        depth: multiplier on bottlenecks per stage.
        reg_max: largest edge distance, in stride units, the head can express.
        in_channels: channels of the input image.
        class_names: optional names, carried along so a checkpoint knows what
            its class indices mean.

    Attributes:
        strides: ``(8, 16, 32)``.
    """

    def __init__(
        self,
        num_classes: int = 80,
        *,
        width: float = 1.0,
        depth: float = 1.0,
        reg_max: int = 16,
        in_channels: int = 3,
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(f"{len(class_names)} class names for {num_classes} classes")

        self.num_classes = int(num_classes)
        self.class_names = list(class_names) if class_names is not None else None

        self.backbone = CSPBackbone(width=width, depth=depth, in_channels=in_channels)
        neck_depth = max(round(3 * depth), 1)
        self.neck = PAFPN(self.backbone.out_channels, depth=neck_depth)
        self.head = DetectHead(
            self.neck.out_channels,
            num_classes=self.num_classes,
            strides=self.backbone.strides,
            reg_max=reg_max,
        )

        self._init_weights()

    @classmethod
    def from_variant(cls, variant: str, num_classes: int = 80, **kwargs) -> GUSNet:
        """Build a named size: ``"n"``, ``"s"``, ``"m"``, ``"l"`` or ``"x"``."""
        key = variant.lower().removeprefix("gusnet-").removeprefix("gusnet")
        if key not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        width, depth = VARIANTS[key]
        return cls(num_classes, width=width, depth=depth, **kwargs)

    @property
    def strides(self) -> tuple[int, ...]:
        return self.backbone.strides

    @property
    def stride_max(self) -> int:
        return max(self.strides)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(self, images: Tensor) -> DetectionOutput:
        """Run the detector.

        Args:
            images: ``(B, 3, H, W)`` float tensor in ``[0, 1]``. ``H`` and ``W``
                must be multiples of the largest stride (32).

        Returns:
            A :class:`~gusnet.nn.head.DetectionOutput` with one prediction per
            grid point, all levels concatenated.
        """
        if images.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) tensor, got shape {tuple(images.shape)}")
        height, width = images.shape[2:]
        if height % self.stride_max or width % self.stride_max:
            raise ValueError(
                f"input {height}x{width} is not divisible by the largest stride "
                f"{self.stride_max}; pad or resize it first"
            )

        features = self.backbone(images)
        features = self.neck(features)
        return self.head(features)

    def _init_weights(self) -> None:
        """Kaiming init for convolutions, plus detector-friendly BatchNorm.

        The default BatchNorm ``eps`` and ``momentum`` are tuned for
        classification with large batches. Detection trains with far fewer
        images per batch, so the running statistics are noisier; a smaller
        momentum averages them over a longer window and a smaller eps keeps the
        normalisation from being dominated by that noise.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # The head sets its own prediction-layer priors; re-apply them so the
        # generic Kaiming pass above does not undo the classification bias.
        self.head.reset_parameters()

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, strides={self.strides}"
