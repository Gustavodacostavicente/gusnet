# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the training loop and everything it is built from."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from gusnet.data import DetectionDataset
from gusnet.models import GUSNet
from gusnet.train import (
    ModelEMA,
    TrainConfig,
    Trainer,
    build_optimizer,
    cosine_schedule,
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
    warmup_factor,
)

# --------------------------------------------------------------------- the EMA


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))


def test_ema_decay_ramps_up_from_zero():
    """A fixed high decay would cling to the random initialisation."""
    ema = ModelEMA(_tiny_model(), decay=0.999, tau=100.0)
    assert ema.current_decay() == 0.0

    ema.updates = 1
    early = ema.current_decay()
    ema.updates = 1000
    late = ema.current_decay()

    assert early < 0.02
    assert late == pytest.approx(0.999, abs=1e-3)


def test_ema_moves_towards_the_live_model():
    model = _tiny_model()
    ema = ModelEMA(model, decay=0.9, tau=1.0)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(10.0)

    before = ema.ema[0].weight.clone()
    ema.update(model)
    after = ema.ema[0].weight

    moved = (after - before).abs().sum()
    assert float(moved) > 0
    # It follows, but it does not jump all the way there.
    assert float((after - model[0].weight.detach()).abs().sum()) > 0


def test_ema_tracks_batchnorm_buffers():
    """The buffers are the reason a raw model can look broken in eval mode."""
    model = _tiny_model()
    ema = ModelEMA(model, decay=0.5, tau=1.0)

    model.train()
    model(torch.randn(4, 3, 8, 8))  # updates running_mean / running_var
    assert not torch.equal(model[1].running_var, ema.ema[1].running_var)

    for _ in range(50):
        ema.update(model)
    assert torch.allclose(model[1].running_var, ema.ema[1].running_var, atol=1e-3)


def test_ema_does_not_average_integer_counters():
    model = _tiny_model()
    ema = ModelEMA(model, decay=0.9, tau=1.0)
    model.train()
    for _ in range(3):
        model(torch.randn(2, 3, 8, 8))
    ema.update(model)
    assert int(ema.ema[1].num_batches_tracked) == int(model[1].num_batches_tracked)


def test_ema_never_requires_gradients():
    ema = ModelEMA(_tiny_model())
    assert not any(p.requires_grad for p in ema.ema.parameters())


def test_ema_state_dict_roundtrip():
    model = _tiny_model()
    ema = ModelEMA(model, decay=0.9, tau=1.0)
    ema.update(model)

    restored = ModelEMA(_tiny_model(), decay=0.9, tau=1.0)
    restored.load_state_dict(ema.state_dict())

    assert restored.updates == ema.updates
    assert torch.equal(restored.ema[0].weight, ema.ema[0].weight)


# --------------------------------------------------------------- the optimiser


def test_optimizer_excludes_norms_and_biases_from_weight_decay():
    """Decaying a BatchNorm scale is not regularisation, it is damage."""
    model = GUSNet.from_variant("n", num_classes=2)
    optimizer = build_optimizer(model, name="sgd", weight_decay=5e-4)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 5e-4
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    decayed = {id(p) for p in optimizer.param_groups[0]["params"]}
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert id(module.weight) not in decayed
            assert id(module.bias) not in decayed
        if isinstance(module, nn.Conv2d):
            assert id(module.weight) in decayed


def test_optimizer_covers_every_trainable_parameter():
    model = GUSNet.from_variant("n", num_classes=2)
    optimizer = build_optimizer(model)
    grouped = sum(len(group["params"]) for group in optimizer.param_groups)
    assert grouped == len([p for p in model.parameters() if p.requires_grad])


@pytest.mark.parametrize("name", ["sgd", "adamw"])
def test_optimizer_kinds(name):
    optimizer = build_optimizer(_tiny_model(), name=name)
    assert optimizer.__class__.__name__.lower() == name


def test_optimizer_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(_tiny_model(), name="rmsprop")


# ---------------------------------------------------------------- the schedule


def test_cosine_schedule_starts_at_one_and_ends_at_the_final_factor():
    schedule = cosine_schedule(100, final_factor=0.01)
    assert schedule(0) == pytest.approx(1.0)
    assert schedule(100) == pytest.approx(0.01)


def test_cosine_schedule_decreases_monotonically():
    schedule = cosine_schedule(50)
    values = [schedule(e) for e in range(51)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_cosine_schedule_spends_the_early_epochs_high():
    """A cosine holds the rate up at first, unlike a linear decay."""
    schedule = cosine_schedule(100)
    assert schedule(10) > 0.9
    assert schedule(50) == pytest.approx(0.5, abs=0.02)


def test_cosine_schedule_rejects_zero_epochs():
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        cosine_schedule(0)


def test_warmup_factor_ramps_and_clamps():
    assert warmup_factor(0, 100) == 0.0
    assert warmup_factor(50, 100) == pytest.approx(0.5)
    assert warmup_factor(200, 100) == 1.0
    assert warmup_factor(5, 0) == 1.0


# --------------------------------------------------------------- checkpointing


def test_checkpoint_roundtrip_rebuilds_the_model(tmp_path: Path):
    model = GUSNet.from_variant("n", num_classes=3, class_names=["a", "b", "c"])
    ema = ModelEMA(model)
    path = save_checkpoint(
        tmp_path / "ckpt.pt", model, ema=ema.ema, epoch=7, metrics={"total": 1.5}
    )

    restored, checkpoint = model_from_checkpoint(path, prefer_ema=False)
    assert checkpoint["epoch"] == 7
    assert checkpoint["metrics"]["total"] == 1.5
    assert restored.class_names == ["a", "b", "c"]
    assert restored.num_parameters() == model.num_parameters()

    x = torch.rand(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(restored(x)["boxes"], model(x)["boxes"], atol=1e-6)


def test_checkpoint_records_how_to_rebuild_the_architecture(tmp_path: Path):
    """Weights alone are not a checkpoint; the shape of the model must travel."""
    model = GUSNet(num_classes=5, width=0.25, depth=0.34, reg_max=8)
    path = save_checkpoint(tmp_path / "ckpt.pt", model)

    args = load_checkpoint(path)["model_args"]
    assert args == {
        "num_classes": 5,
        "width": 0.25,
        "depth": 0.34,
        "in_channels": 3,
        "reg_max": 8,
        "class_names": None,
    }


def test_checkpoint_prefers_the_averaged_weights(tmp_path: Path):
    model = GUSNet.from_variant("n", num_classes=2)
    ema = ModelEMA(model, decay=0.5, tau=1.0)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)  # make the live weights differ from the average

    path = save_checkpoint(tmp_path / "ckpt.pt", model, ema=ema.ema)
    averaged, _ = model_from_checkpoint(path, prefer_ema=True)
    live, _ = model_from_checkpoint(path, prefer_ema=False)

    averaged_weight = averaged.backbone.stem.conv.weight
    live_weight = live.backbone.stem.conv.weight
    assert not torch.allclose(averaged_weight, live_weight)


def test_loading_a_missing_checkpoint_is_a_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        load_checkpoint(tmp_path / "nope.pt")


def test_save_checkpoint_rejects_a_foreign_model(tmp_path: Path):
    with pytest.raises(TypeError, match="expects a GUSNet model"):
        save_checkpoint(tmp_path / "ckpt.pt", _tiny_model())


# ----------------------------------------------------------------- the trainer


def _short_config(tmp_path: Path, **overrides) -> TrainConfig:
    base = {
        "epochs": 4,
        "batch_size": 2,
        "img_size": 64,
        "optimizer": "adamw",
        "lr0": 2e-3,
        "warmup_epochs": 1.0,
        "close_mosaic": 2,
        "workers": 0,
        "device": "cpu",
        "save_dir": tmp_path / "run",
        "log_interval": 0,
        "amp": False,
    }
    base.update(overrides)
    return TrainConfig(**base)


def test_trainer_runs_and_reduces_the_loss(folder_dataset: Path, tmp_path: Path):
    dataset = DetectionDataset.from_folder(
        folder_dataset, "train", img_size=64, augment=True, seed=0
    )
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)
    config = _short_config(tmp_path, epochs=6)

    history = Trainer(model, dataset, config).train()

    assert len(history) == 6
    assert all(key in history[0] for key in ("total", "cls", "box", "dfl", "fg", "lr"))
    assert history[-1]["total"] < history[0]["total"]


def test_trainer_writes_last_and_best_checkpoints(folder_dataset: Path, tmp_path: Path):
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=64, augment=True)
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)
    config = _short_config(tmp_path)

    Trainer(model, dataset, config).train()

    assert (config.save_dir / "last.pt").is_file()
    assert (config.save_dir / "best.pt").is_file()
    # last.pt can resume; best.pt is for deployment and skips the optimiser.
    assert "optimizer" in load_checkpoint(config.save_dir / "last.pt")
    assert "optimizer" not in load_checkpoint(config.save_dir / "best.pt")


def test_trainer_closes_mosaic_for_the_final_epochs(folder_dataset: Path, tmp_path: Path):
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=64, augment=True)
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)
    config = _short_config(tmp_path, epochs=4, close_mosaic=2)

    assert dataset.config.mosaic == 1.0
    Trainer(model, dataset, config).train()
    assert dataset.config.mosaic == 0.0
    assert dataset.config.mixup == 0.0


def test_trainer_warms_the_learning_rate_up_from_zero(folder_dataset: Path, tmp_path: Path):
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=64, augment=False)
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)
    trainer = Trainer(model, dataset, _short_config(tmp_path, epochs=10, warmup_epochs=4.0))

    steps_per_epoch = len(trainer.loader)
    warmup = int(4.0 * steps_per_epoch)

    trainer._step = 0
    trainer._apply_schedule(0, warmup)
    start = trainer.optimizer.param_groups[0]["lr"]

    trainer._step = warmup // 2
    trainer._apply_schedule(0, warmup)
    middle = trainer.optimizer.param_groups[0]["lr"]

    trainer._step = warmup
    trainer._apply_schedule(0, warmup)
    after = trainer.optimizer.param_groups[0]["lr"]

    assert start == pytest.approx(0.0)
    assert 0 < middle < after
    assert after == pytest.approx(trainer.config.lr0)


def test_trainer_keeps_an_ema_that_differs_from_the_live_weights(
    folder_dataset: Path, tmp_path: Path
):
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=64, augment=False)
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)
    trainer = Trainer(model, dataset, _short_config(tmp_path, epochs=3))
    trainer.train()

    assert trainer.ema.updates > 0
    live = trainer.model.backbone.stem.conv.weight
    averaged = trainer.ema.ema.backbone.stem.conv.weight
    assert not torch.allclose(live, averaged)


def test_trainer_selects_best_by_map_when_validating(folder_dataset: Path, tmp_path: Path):
    """With a validation set, `best.pt` must be chosen by mAP, not by loss.

    The two live in the same metrics dict on different scales, so an epoch
    without a validation result must not compete: otherwise a large early
    training loss wins a maximisation and `best.pt` is the worst checkpoint.
    """
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=64, augment=True)
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)
    config = _short_config(tmp_path, epochs=4, val_interval=3)

    trainer = Trainer(model, dataset, config, val_dataset=dataset)
    history = trainer.train()

    validated = [entry for entry in history if "mAP50-95" in entry]
    assert validated, "no epoch was validated"
    assert all(0.0 <= entry["mAP50-95"] <= 1.0 for entry in validated)

    best = load_checkpoint(config.save_dir / "best.pt")
    assert "mAP50-95" in best["metrics"]
    assert best["metrics"]["mAP50-95"] == max(e["mAP50-95"] for e in validated)


def test_trainer_validates_on_the_last_epoch_regardless_of_interval(
    folder_dataset: Path, tmp_path: Path
):
    dataset = DetectionDataset.from_folder(folder_dataset, "train", img_size=64, augment=False)
    model = GUSNet.from_variant("n", num_classes=dataset.num_classes)
    config = _short_config(tmp_path, epochs=3, val_interval=10)

    history = Trainer(model, dataset, config, val_dataset=dataset).train()
    assert "mAP50-95" in history[-1]
    assert "mAP50-95" not in history[0]


# ------------------------------------------------------------------- resuming


def _dataset(folder_dataset: Path, augment: bool = False) -> DetectionDataset:
    return DetectionDataset.from_folder(
        folder_dataset, "train", img_size=64, augment=augment, seed=0
    )


def test_resume_continues_at_the_next_epoch(folder_dataset: Path, tmp_path: Path):
    dataset = _dataset(folder_dataset)
    config = _short_config(tmp_path, epochs=3)
    first = Trainer(GUSNet.from_variant("n", num_classes=dataset.num_classes), dataset, config)
    first.train()

    config = _short_config(tmp_path, epochs=6)
    second = Trainer(
        GUSNet.from_variant("n", num_classes=dataset.num_classes),
        dataset,
        config,
        resume=config.save_dir / "last.pt",
    )
    history = second.train()

    assert len(history) == 3, "should run only the remaining epochs"
    assert load_checkpoint(config.save_dir / "last.pt")["epoch"] == 5


def test_resume_restores_the_iteration_counter(folder_dataset: Path, tmp_path: Path):
    """Otherwise warmup runs again, dropping an already-trained model to lr 0.

    This is the failure that makes a naive resume worse than no resume: the
    weights come back, the schedule does not, and the first epochs after
    restarting undo progress instead of continuing it.
    """
    dataset = _dataset(folder_dataset)
    config = _short_config(tmp_path, epochs=4, warmup_epochs=10.0)
    first = Trainer(GUSNet.from_variant("n", num_classes=dataset.num_classes), dataset, config)
    first.train()
    assert first._step > 0

    config = _short_config(tmp_path, epochs=8, warmup_epochs=10.0)
    second = Trainer(
        GUSNet.from_variant("n", num_classes=dataset.num_classes),
        dataset,
        config,
        resume=config.save_dir / "last.pt",
    )
    assert second._step == first._step

    # The learning rate picks up along the ramp rather than at its start.
    warmup = int(10.0 * len(second.loader))
    second._apply_schedule(4, warmup)
    assert second.optimizer.param_groups[0]["lr"] > 0.0


def test_resume_restores_the_ema_and_optimizer_state(folder_dataset: Path, tmp_path: Path):
    dataset = _dataset(folder_dataset)
    config = _short_config(tmp_path, epochs=3)
    first = Trainer(GUSNet.from_variant("n", num_classes=dataset.num_classes), dataset, config)
    first.train()

    config = _short_config(tmp_path, epochs=6)
    second = Trainer(
        GUSNet.from_variant("n", num_classes=dataset.num_classes),
        dataset,
        config,
        resume=config.save_dir / "last.pt",
    )

    assert second.ema.updates == first.ema.updates
    assert torch.allclose(
        second.ema.ema.backbone.stem.conv.weight, first.ema.ema.backbone.stem.conv.weight
    )
    # AdamW's moment estimates survive: restarting them costs real progress.
    assert second.optimizer.state_dict()["state"], "optimizer state is empty"


def test_resume_keeps_the_best_score(folder_dataset: Path, tmp_path: Path):
    """A mediocre first epoch after resuming must not overwrite a good best.pt."""
    dataset = _dataset(folder_dataset)
    config = _short_config(tmp_path, epochs=3)
    first = Trainer(GUSNet.from_variant("n", num_classes=dataset.num_classes), dataset, config)
    first.train()

    config = _short_config(tmp_path, epochs=6)
    second = Trainer(
        GUSNet.from_variant("n", num_classes=dataset.num_classes),
        dataset,
        config,
        resume=config.save_dir / "last.pt",
    )
    assert second._best == pytest.approx(first._best)


def test_resume_refuses_a_finished_run(folder_dataset: Path, tmp_path: Path):
    dataset = _dataset(folder_dataset)
    config = _short_config(tmp_path, epochs=3)
    Trainer(GUSNet.from_variant("n", num_classes=dataset.num_classes), dataset, config).train()

    with pytest.raises(SystemExit, match="already finished epoch"):
        Trainer(
            GUSNet.from_variant("n", num_classes=dataset.num_classes),
            dataset,
            _short_config(tmp_path, epochs=3),
            resume=config.save_dir / "last.pt",
        )


def test_resume_from_a_missing_checkpoint_is_a_clear_error(folder_dataset: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        Trainer(
            GUSNet.from_variant("n", num_classes=2),
            _dataset(folder_dataset),
            _short_config(tmp_path),
            resume=tmp_path / "nope.pt",
        )


def test_a_resumed_run_matches_an_uninterrupted_one(folder_dataset: Path, tmp_path: Path):
    """Six epochs straight through, against three plus three.

    Not bit-exact -- the dataloader's shuffling restarts -- but the loss after
    resuming must be in the same region, not back where training began.
    """
    straight = Trainer(
        GUSNet.from_variant("n", num_classes=2),
        _dataset(folder_dataset, augment=True),
        _short_config(tmp_path / "straight", epochs=6),
    ).train()

    split_config = _short_config(tmp_path / "split", epochs=3)
    Trainer(
        GUSNet.from_variant("n", num_classes=2),
        _dataset(folder_dataset, augment=True),
        split_config,
    ).train()

    resumed = Trainer(
        GUSNet.from_variant("n", num_classes=2),
        _dataset(folder_dataset, augment=True),
        _short_config(tmp_path / "split", epochs=6),
        resume=split_config.save_dir / "last.pt",
    ).train()

    assert resumed[-1]["total"] < straight[0]["total"], "resuming threw away the progress"


def test_resume_estimates_the_counters_for_an_old_checkpoint(folder_dataset: Path, tmp_path: Path):
    """Checkpoints written before the counters existed must not restart warmup.

    Starting at step zero would run the warmup ramp again over an already
    trained model and restart the EMA's decay ramp. The epoch number is enough
    to estimate both, and estimating beats resetting.
    """
    dataset = _dataset(folder_dataset)
    config = _short_config(tmp_path, epochs=3)
    Trainer(GUSNet.from_variant("n", num_classes=dataset.num_classes), dataset, config).train()

    # Strip the bookkeeping, as an older checkpoint would have it.
    path = config.save_dir / "last.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["training_state"] = {}
    torch.save(payload, path)

    config = _short_config(tmp_path, epochs=6)
    resumed = Trainer(
        GUSNet.from_variant("n", num_classes=dataset.num_classes), dataset, config, resume=path
    )

    assert resumed._step == 3 * len(resumed.loader)
    assert resumed.ema.updates == resumed._step
