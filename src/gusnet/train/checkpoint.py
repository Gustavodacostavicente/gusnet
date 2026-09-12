# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Saving and restoring training state.

A checkpoint here carries enough to rebuild the model from nothing -- the
constructor arguments, not only the weights. A file of tensors whose shapes
imply an architecture you have to guess is a checkpoint you will eventually be
unable to load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from gusnet import __version__

__all__ = ["load_checkpoint", "model_from_checkpoint", "save_checkpoint"]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    ema: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint.

    Args:
        path: destination file.
        model: the live model. Its constructor arguments are recorded alongside
            the weights.
        ema: the averaged model, if one is kept. This is the one to deploy.
        optimizer: saved only so training can resume exactly; drop it for a
            release checkpoint and the file shrinks by two thirds.
        epoch: the epoch just completed.
        metrics: whatever was measured, for the record.
        config: the training configuration, for reproducibility.
        training_state: the loop's own bookkeeping -- iteration count, the EMA's
            update count, the AMP scaler, the best score so far. Weights and
            optimizer alone do not make a run resumable: restart with the
            iteration count at zero and the warmup schedule runs again over an
            already-trained model, which is worse than not resuming at all.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format": 1,
        "gusnet_version": __version__,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "model_args": _model_args(model),
        "metrics": metrics or {},
        "config": config or {},
        "training_state": training_state or {},
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Read a checkpoint written by :func:`save_checkpoint`.

    Note:
        Loaded with ``weights_only=True``. A checkpoint is a pickle, and an
        unrestricted pickle from an untrusted source executes arbitrary code on
        load; there is no reason for a weights file to need that.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return torch.load(path, map_location=map_location, weights_only=True)


def model_from_checkpoint(
    path: str | Path,
    *,
    prefer_ema: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    """Rebuild a model and load its weights.

    Args:
        path: the checkpoint file.
        prefer_ema: load the averaged weights when the checkpoint has them.
            That is almost always what you want -- the EMA copy is the one that
            was evaluated.
        map_location: device to load onto.

    Returns:
        ``(model, checkpoint)``, the model in eval mode.
    """
    from gusnet.models import GUSNet

    checkpoint = load_checkpoint(path, map_location=map_location)
    model = GUSNet(**checkpoint["model_args"])

    weights = checkpoint["ema"] if prefer_ema and "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(weights)
    return model.eval(), checkpoint


def _model_args(model: nn.Module) -> dict[str, Any]:
    """Recover the constructor arguments of a :class:`~gusnet.models.GUSNet`."""
    backbone = getattr(model, "backbone", None)
    head = getattr(model, "head", None)
    if backbone is None or head is None:
        raise TypeError("save_checkpoint expects a GUSNet model")

    return {
        "num_classes": model.num_classes,
        "width": backbone.width,
        "depth": backbone.depth,
        "in_channels": backbone.in_channels,
        "reg_max": head.reg_max,
        "class_names": model.class_names,
    }
