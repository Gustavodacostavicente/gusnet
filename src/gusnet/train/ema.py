# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Exponential moving average of the weights.

SGD does not converge to a point, it orbits one. The weights at the end of any
given step are a sample from that orbit, and a different sample would score
differently on the same data. Keeping a running average of them lands nearer
the centre of the orbit than any individual sample, which is worth a real
accuracy gain for almost no cost -- so the EMA copy, not the live model, is
what gets evaluated and shipped.

The average covers buffers as well as parameters, which matters here: BatchNorm
running statistics lag behind the weights by hundreds of steps early in
training, and evaluating raw weights against stale statistics is a common way
to conclude a perfectly healthy model is broken.
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn

__all__ = ["ModelEMA"]


class ModelEMA:
    """A shadow copy of a model, updated as a moving average.

    Args:
        model: the model being trained. It is deep-copied once; the original is
            never modified.
        decay: the asymptotic decay. ``0.9999`` averages over roughly the last
            ten thousand steps.
        tau: how quickly the decay ramps up to ``decay``. The ramp exists
            because a fixed high decay at step one would hold on to the random
            initialisation for thousands of steps; starting near zero lets the
            average track the model closely while it is still changing fast.
        updates: number of updates already applied, for resuming.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.9999,
        tau: float = 2000.0,
        updates: int = 0,
    ) -> None:
        self.ema = deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = int(updates)

    def current_decay(self) -> float:
        """The decay in effect at the current step."""
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold one training step into the average."""
        self.updates += 1
        decay = self.current_decay()

        live = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if not value.dtype.is_floating_point:
                # Integer buffers such as ``num_batches_tracked`` are counters,
                # not quantities; averaging them is meaningless.
                value.copy_(live[key])
                continue
            value.mul_(decay).add_(live[key].detach().to(value.dtype), alpha=1.0 - decay)

    def state_dict(self) -> dict:
        return {"ema": self.ema.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.ema.load_state_dict(state["ema"])
        self.updates = int(state.get("updates", 0))
