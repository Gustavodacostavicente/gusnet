# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Optimiser construction and learning-rate scheduling."""

from __future__ import annotations

import math
from collections.abc import Callable

from torch import nn, optim

__all__ = ["build_optimizer", "cosine_schedule", "warmup_factor"]


def build_optimizer(
    model: nn.Module,
    *,
    name: str = "sgd",
    lr: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 5e-4,
) -> optim.Optimizer:
    """Build an optimiser with weight decay applied only where it belongs.

    Parameters are split into three groups:

    * **convolution and linear weights** get weight decay -- shrinking them is
      a genuine capacity constraint;
    * **normalisation weights** do not. Decaying a BatchNorm scale towards zero
      does not regularise anything, it just squashes the activations it was
      there to rescale;
    * **biases** do not either, for the same reason: a bias is an offset, and
      pulling every offset towards zero is a bias, not a regulariser.

    Lumping them together is the single most common silent mistake in a
    training script -- it costs accuracy and nothing reports it.

    Args:
        model: the model to optimise.
        name: ``"sgd"`` or ``"adamw"``. SGD with Nesterov momentum is the
            default because it generalises slightly better on detection at long
            schedules; AdamW converges faster on short ones and on small data.
        lr: initial learning rate.
        momentum: SGD momentum, or AdamW's ``beta1``.
        weight_decay: decay for the decayed group only.

    Returns:
        A configured optimiser whose first parameter group is the decayed one.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param_name == "bias" or isinstance(module, nn.modules.batchnorm._NormBase):
                no_decay.append(param)
            else:
                decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    key = name.lower()
    if key == "sgd":
        return optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    if key == "adamw":
        return optim.AdamW(groups, lr=lr, betas=(momentum, 0.999))
    raise ValueError(f"unknown optimizer {name!r}; use 'sgd' or 'adamw'")


def cosine_schedule(epochs: int, final_factor: float = 0.01) -> Callable[[int], float]:
    """A cosine learning-rate curve, as a multiplier on the initial rate.

    Cosine decay spends most of the schedule at a high rate and then anneals
    smoothly to ``final_factor``. The long tail at a small rate is where the
    model settles into a minimum instead of bouncing around it, and a smooth
    curve avoids the loss spikes that step decays produce.

    Args:
        epochs: total epochs the schedule spans.
        final_factor: multiplier at the last epoch.

    Returns:
        A function mapping epoch index to a multiplier in
        ``[final_factor, 1.0]``.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    def factor(epoch: int) -> float:
        progress = min(max(epoch / epochs, 0.0), 1.0)
        cosine = (1.0 - math.cos(progress * math.pi)) / 2.0
        return (1.0 - cosine) * (1.0 - final_factor) + final_factor

    return factor


def warmup_factor(iteration: int, total_warmup: int) -> float:
    """Linear ramp from 0 to 1 over the warmup iterations.

    A detection head starts with random weights and an assigner that will hand
    it thousands of targets on the first batch. Stepping at the full learning
    rate into that produces a gradient large enough to leave the model in a
    state it never recovers from; the ramp lets the first few hundred steps
    move the weights gently instead.
    """
    if total_warmup <= 0:
        return 1.0
    return min(max(iteration / total_warmup, 0.0), 1.0)
