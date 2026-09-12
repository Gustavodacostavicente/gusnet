# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""The training loop.

Everything the previous phases built is inert until something drives it. This
is that: batches in, gradients out, for as many epochs as asked, with the
handful of mechanisms that separate a loop that trains from a loop that merely
runs.

* **Warmup** -- the first few hundred steps ramp the learning rate, the SGD
  momentum and the bias learning rate up from near zero. A randomly initialised
  detector handed thousands of targets at full learning rate takes a step it
  does not recover from.
* **Cosine decay** -- most of the schedule at a high rate, a long smooth anneal
  at the end.
* **EMA** -- the averaged weights, not the live ones, are what gets saved and
  evaluated.
* **AMP** -- half precision forward and backward with a loss scaler, roughly
  doubling throughput on any recent NVIDIA GPU.
* **Closing mosaic** -- mosaic is switched off for the last epochs so the model
  finishes on images that look like the ones it will see at inference.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from gusnet.data import DetectionDataset, build_dataloader
from gusnet.eval import EvalConfig, evaluate
from gusnet.losses import DetectionLoss
from gusnet.train.checkpoint import save_checkpoint
from gusnet.train.ema import ModelEMA
from gusnet.train.optim import build_optimizer, cosine_schedule, warmup_factor

__all__ = ["TrainConfig", "Trainer", "seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch in one call.

    GUSNet's own randomness runs through explicit generators
    (:class:`random.Random` and :func:`numpy.random.default_rng`), which this
    does not touch. The global seeds are set anyway so that anything else in
    the process -- a library's internal shuffling, an initialisation helper --
    is reproducible too.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - deliberately seeds the legacy global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    """Everything that controls a run.

    Attributes:
        epochs: how many passes over the dataset.
        batch_size: images per step.
        img_size: training resolution; must be a multiple of 32.
        optimizer: ``"sgd"`` or ``"adamw"``.
        lr0: initial learning rate.
        lr_final_factor: the cosine schedule ends at ``lr0 * this``.
        momentum: SGD momentum, or AdamW's beta1.
        weight_decay: applied to convolution weights only.
        warmup_epochs: length of the ramp, in epochs.
        warmup_momentum: momentum at the start of the ramp.
        warmup_bias_lr: the bias group starts at this rate and comes *down* to
            ``lr0``. Biases are the fastest thing a detector can usefully learn
            early -- especially the classification prior -- so they are allowed
            to move while the weights are still ramping up.
        close_mosaic: epochs at the end with mosaic and mixup disabled.
        val_interval: epochs between validations, when a validation set is
            given. Validation costs a full pass over that set, so on a large
            one it is worth doing every few epochs rather than every epoch.
        eval_conf_threshold: confidence floor during validation. Deliberately
            near zero: mAP integrates the whole precision/recall curve, and a
            high threshold truncates it and under-reports the metric.
        eval_iou_threshold: NMS overlap threshold during validation.
        grad_clip: maximum gradient norm, or ``None``.
        amp: use mixed precision when CUDA is available.
        ema_decay: asymptotic decay of the weight average.
        ema_tau: how fast that decay ramps in.
        workers: dataloader worker processes.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ...
        seed: RNG seed.
        save_dir: where checkpoints and logs go.
        log_interval: steps between progress lines.
    """

    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640

    optimizer: str = "sgd"
    lr0: float = 0.01
    lr_final_factor: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4

    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    close_mosaic: int = 15
    grad_clip: float | None = 10.0
    amp: bool = True

    ema_decay: float = 0.9999
    ema_tau: float = 2000.0

    val_interval: int = 1
    eval_conf_threshold: float = 0.001
    eval_iou_threshold: float = 0.7

    workers: int = 0
    device: str = "auto"
    seed: int = 0
    save_dir: Path = field(default_factory=lambda: Path("runs/train"))
    log_interval: int = 10

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def as_dict(self) -> dict:
        data = asdict(self)
        data["save_dir"] = str(self.save_dir)
        return data


class Trainer:
    """Drive a model through a training run.

    Args:
        model: the detector to train.
        dataset: the training dataset.
        config: the run configuration.
        criterion: the objective. Defaults to
            :class:`~gusnet.losses.detection.DetectionLoss` for the model's
            class count.
        val_dataset: optional validation set. When given, ``best.pt`` is
            selected by mAP instead of by training loss -- which is the only
            honest way to choose a checkpoint, since training loss keeps
            falling long after generalisation has stopped improving.

    Attributes:
        history: one dict of per-epoch metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: DetectionDataset,
        config: TrainConfig | None = None,
        *,
        criterion: nn.Module | None = None,
        val_dataset: DetectionDataset | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.val_dataset = val_dataset
        self.device = self.config.resolved_device()

        seed_everything(self.config.seed)

        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion or DetectionLoss(model.num_classes, reg_max=model.head.reg_max)

        self.optimizer = build_optimizer(
            self.model,
            name=self.config.optimizer,
            lr=self.config.lr0,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        self.lr_factor = cosine_schedule(self.config.epochs, self.config.lr_final_factor)
        self.ema = ModelEMA(self.model, decay=self.config.ema_decay, tau=self.config.ema_tau)

        self.amp_enabled = bool(self.config.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)

        self.loader = self._build_loader(mosaic=True)
        self.history: list[dict[str, float]] = []
        self._mosaic_open = True
        self._step = 0

    # ---------------------------------------------------------------- the loop

    def train(self) -> list[dict[str, float]]:
        """Run the whole schedule and return the per-epoch history."""
        config = self.config
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        steps_per_epoch = max(len(self.loader), 1)
        warmup_iterations = int(config.warmup_epochs * steps_per_epoch)
        best = float("inf")

        print(
            f"training on {self.device} | {len(self.dataset)} images | "
            f"{config.epochs} epochs | batch {config.batch_size} | "
            f"amp {'on' if self.amp_enabled else 'off'}"
        )

        # Without a validation set the only signal available is training loss,
        # which keeps falling long after the model has stopped generalising.
        # It is a placeholder, and `score` says which of the two is in use.
        maximise = self.val_dataset is not None
        best = -float("inf") if maximise else float("inf")

        for epoch in range(config.epochs):
            self._maybe_close_mosaic(epoch)
            metrics = self._train_one_epoch(epoch, warmup_iterations)
            metrics.update(self._maybe_validate(epoch))
            self.history.append(metrics)

            save_checkpoint(
                save_dir / "last.pt",
                self.model,
                ema=self.ema.ema,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config.as_dict(),
            )

            # When validating, only epochs that actually produced a mAP can
            # compete for `best`. Falling back to the training loss on the
            # others would compare two quantities on different scales and let
            # a large early loss win a maximisation.
            score = metrics.get("mAP50-95") if maximise else metrics["total"]
            improved = score is not None and (score > best if maximise else score < best)
            if improved:
                best = score
                save_checkpoint(
                    save_dir / "best.pt",
                    self.model,
                    ema=self.ema.ema,
                    epoch=epoch,
                    metrics=metrics,
                    config=config.as_dict(),
                )

        label = "mAP50-95" if maximise else "training loss"
        print(f"done. best {label} {best:.4f}. checkpoints in {save_dir}")
        return self.history

    def _maybe_validate(self, epoch: int) -> dict[str, float]:
        """Score the averaged weights on the validation set, if there is one.

        The EMA copy is evaluated rather than the live model: it is the one
        that will be deployed, and early in training its BatchNorm statistics
        are the more settled of the two.
        """
        config = self.config
        if self.val_dataset is None:
            return {}
        last_epoch = epoch == config.epochs - 1
        if not last_epoch and (epoch + 1) % max(config.val_interval, 1):
            return {}

        result = evaluate(
            self.ema.ema,
            self.val_dataset,
            EvalConfig(
                batch_size=config.batch_size,
                img_size=config.img_size,
                conf_threshold=config.eval_conf_threshold,
                iou_threshold=config.eval_iou_threshold,
                # Deliberately single-process, and not inherited from the
                # training config. Spawning dataloader workers while the
                # training loader's persistent workers are still alive
                # deadlocks on Windows -- observed hanging indefinitely at the
                # first validation, with the GPU idle and no error. Validation
                # does not need them anyway: there is no mosaic, so each sample
                # decodes one image instead of four.
                workers=0,
                device=str(self.device),
                verbose=False,
            ),
        )
        print(
            f"  val  mAP50-95 {result.map50_95:.4f}  mAP50 {result.map50:.4f}  "
            f"P {result.precision:.3f}  R {result.recall:.3f}"
        )
        return result.items()

    def _train_one_epoch(self, epoch: int, warmup_iterations: int) -> dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "cls": 0.0, "box": 0.0, "dfl": 0.0, "fg": 0.0}
        batches = 0
        started = time.perf_counter()

        for batch in self.loader:
            self._apply_schedule(epoch, warmup_iterations)

            images = batch["images"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                output = self.model(images)
                loss, breakdown = self.criterion(output, targets)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.config.grad_clip:
                # Unscale first: clipping a scaled gradient clips the wrong
                # thing, since the scale factor changes between steps.
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)

            for key, value in breakdown.items().items():
                totals[key] += value
            batches += 1
            self._step += 1

            if self.config.log_interval and batches % self.config.log_interval == 0:
                self._log_step(epoch, batches, breakdown)

        elapsed = time.perf_counter() - started
        metrics = {key: value / max(batches, 1) for key, value in totals.items()}
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["seconds"] = elapsed

        print(
            f"epoch {epoch + 1}/{self.config.epochs}  "
            f"loss {metrics['total']:.4f}  cls {metrics['cls']:.4f}  "
            f"box {metrics['box']:.4f}  dfl {metrics['dfl']:.4f}  "
            f"fg {metrics['fg']:.0f}  lr {metrics['lr']:.5f}  {elapsed:.1f}s"
        )
        return metrics

    # ---------------------------------------------------------------- schedule

    def _apply_schedule(self, epoch: int, warmup_iterations: int) -> None:
        """Set the learning rate and momentum for the step about to be taken."""
        config = self.config
        epoch_factor = self.lr_factor(epoch)

        if self._step < warmup_iterations:
            ramp = warmup_factor(self._step, warmup_iterations)
            for index, group in enumerate(self.optimizer.param_groups):
                # Group 1 holds biases and norm weights; it starts high and
                # comes down, while the decayed group ramps up from zero.
                start = config.warmup_bias_lr if index == 1 else 0.0
                group["lr"] = start + ramp * (config.lr0 * epoch_factor - start)
                if "momentum" in group:
                    group["momentum"] = config.warmup_momentum + ramp * (
                        config.momentum - config.warmup_momentum
                    )
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = config.lr0 * epoch_factor
                if "momentum" in group:
                    group["momentum"] = config.momentum

    def _maybe_close_mosaic(self, epoch: int) -> None:
        """Disable mosaic and mixup for the final epochs.

        Mosaic is worth a lot of accuracy but every image it produces is a
        collage that cannot occur at inference. Training to the last step on
        them leaves a gap between what the model has seen and what it will see,
        so the tail of the schedule runs on plain images.
        """
        remaining = self.config.epochs - epoch
        if not self._mosaic_open or remaining > self.config.close_mosaic:
            return

        self.dataset.config.mosaic = 0.0
        self.dataset.config.mixup = 0.0
        self._mosaic_open = False
        # Workers hold their own copy of the dataset, so the loader has to be
        # rebuilt for the change to reach them.
        self.loader = self._build_loader(mosaic=False)
        print(f"epoch {epoch + 1}: mosaic and mixup disabled for the final epochs")

    def _build_loader(self, *, mosaic: bool) -> DataLoader:
        del mosaic  # the dataset already carries the augmentation config
        return build_dataloader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            drop_last=len(self.dataset) > self.config.batch_size,
        )

    def _log_step(self, epoch: int, batch_index: int, breakdown) -> None:
        values = breakdown.items()
        print(
            f"  e{epoch + 1} b{batch_index}/{len(self.loader)}  "
            f"loss {values['total']:.4f}  fg {values['fg']:.0f}  "
            f"lr {self.optimizer.param_groups[0]['lr']:.5f}"
        )
