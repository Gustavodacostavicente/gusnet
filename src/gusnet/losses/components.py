# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""The three loss terms a GUSNet prediction is scored with.

Each one exists because a simpler choice fails in a specific way:

:class:`VarifocalLoss`
    plain cross-entropy would drown in the tens of thousands of background
    points and would train confidence towards a constant 1, disconnecting the
    score from how good the box actually is.
:class:`IoULoss`
    an L1 loss on the four coordinates optimises something that is not the
    metric: two boxes with the same coordinate error can have very different
    overlap, and a box that misses entirely gives no gradient direction.
:class:`DistributionFocalLoss`
    the head does not output a distance, it outputs a distribution over
    distances; this is what supervises that distribution instead of only its
    expectation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gusnet.ops.boxes import bbox_iou

__all__ = ["DistributionFocalLoss", "IoULoss", "VarifocalLoss"]


class VarifocalLoss(nn.Module):
    """Asymmetric focal loss over IoU-aware soft targets.

    The two sides of the problem are not symmetric, so they are not weighted
    the same way:

    * **Positives** carry a target ``q`` in ``(0, 1]`` -- the assigner's quality
      estimate -- and are weighted by ``q`` itself. A point that localises its
      object well matters more than one that barely does, and the confidence it
      learns is that same ``q``, so the score comes to mean "this box is good".
    * **Negatives** have ``q = 0`` and are weighted by ``alpha * p**gamma``.
      Once the model is confident a point is background, ``p`` is near zero and
      the weight vanishes; the loss then concentrates on the few background
      points that still look like objects, which is what keeps the thousands of
      easy negatives from dominating the gradient.

    Setting ``alpha=1`` and ``gamma=0`` reduces this to plain BCE.

    Reimplemented from Zhang et al., *VarifocalNet* (arXiv:2008.13367).

    Args:
        alpha: weight of the negative term.
        gamma: how sharply easy negatives are suppressed.
        reduction: ``"sum"``, ``"mean"`` or ``"none"``.
    """

    def __init__(self, *, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "sum") -> None:
        super().__init__()
        if reduction not in {"sum", "mean", "none"}:
            raise ValueError(f"unknown reduction {reduction!r}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Score raw classification logits against soft targets.

        Args:
            logits: ``(..., C)`` pre-sigmoid.
            targets: ``(..., C)`` values in ``[0, 1]``; zero means background.

        Returns:
            A scalar, or ``(..., C)`` when ``reduction="none"``.
        """
        probabilities = logits.sigmoid()
        weight = torch.where(targets > 0, targets, self.alpha * probabilities.pow(self.gamma))
        # The logits form is used deliberately: computing BCE from the sigmoid
        # loses precision and can produce infinities once a logit saturates.
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weight

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss


class IoULoss(nn.Module):
    """``1 - IoU`` between matched boxes, in one of the generalized forms.

    Args:
        kind: ``"iou"``, ``"giou"``, ``"diou"`` or ``"ciou"``. The default
            CIoU adds a centre-distance and an aspect-ratio term, so a box that
            does not overlap at all still receives a gradient pointing towards
            the target instead of a flat zero.
        reduction: ``"sum"``, ``"mean"`` or ``"none"``.
    """

    def __init__(self, *, kind: str = "ciou", reduction: str = "none") -> None:
        super().__init__()
        if reduction not in {"sum", "mean", "none"}:
            raise ValueError(f"unknown reduction {reduction!r}")
        self.kind = kind
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Score ``(..., 4)`` predicted xyxy boxes against ``(..., 4)`` targets."""
        loss = 1.0 - bbox_iou(pred, target, kind=self.kind)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss


class DistributionFocalLoss(nn.Module):
    """Supervise a discrete distribution towards a continuous distance.

    The head predicts ``reg_max + 1`` logits per box edge, and the distance it
    reports is their expectation. A target like 4.3 falls between two bins, so
    the loss pushes probability onto bins 4 and 5 in proportion 0.7 / 0.3 --
    cross-entropy against the two neighbours, linearly interpolated.

    Supervising only the expectation would leave the distribution's shape
    unconstrained: the model could place mass on bins 0 and 9 and still report
    4.5. Here the mass has to sit where the edge actually is, which is what
    makes the distribution readable as a confidence.

    Reimplemented from Li et al., *Generalized Focal Loss* (arXiv:2006.04388).

    Args:
        reg_max: the largest bin index, so the support is ``0..reg_max``.
        reduction: ``"sum"``, ``"mean"`` or ``"none"``.
    """

    def __init__(self, reg_max: int = 16, *, reduction: str = "none") -> None:
        super().__init__()
        if reduction not in {"sum", "mean", "none"}:
            raise ValueError(f"unknown reduction {reduction!r}")
        if reg_max < 1:
            raise ValueError("reg_max must be at least 1")
        self.reg_max = int(reg_max)
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Score distance distributions against continuous distances.

        Args:
            logits: ``(N, 4, reg_max + 1)`` raw, one distribution per edge.
            target: ``(N, 4)`` distances in bin units, within ``[0, reg_max]``.

        Returns:
            ``(N,)`` averaged over the four edges, or a scalar if reduced.
        """
        if logits.shape[-1] != self.reg_max + 1:
            raise ValueError(f"expected {self.reg_max + 1} bins, got {logits.shape[-1]}")

        target = target.clamp(0, self.reg_max)
        lower = target.floor()
        upper = lower + 1
        weight_lower = upper - target
        weight_upper = 1.0 - weight_lower

        flat = logits.reshape(-1, self.reg_max + 1)
        lower_index = lower.reshape(-1).long()
        # An exact hit on the last bin would index past the end; its weight is
        # zero there, so clamping changes nothing except the lookup.
        upper_index = upper.reshape(-1).long().clamp(max=self.reg_max)

        loss = F.cross_entropy(flat, lower_index, reduction="none") * weight_lower.reshape(-1)
        loss = loss + F.cross_entropy(flat, upper_index, reduction="none") * weight_upper.reshape(
            -1
        )
        loss = loss.reshape(target.shape).mean(dim=-1)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss
